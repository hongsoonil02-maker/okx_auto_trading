#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_last5d.py — 최근 N일 실거래 성과 분석 (OKX bills 원장 + trades.jsonl 교차검증)
- 헤지모드 (symbol, posSide) 단위로 라운드트립 재구성 (DCA/분할익절/거래소SL/강제청산 포함)
- 지표: 거래수, 승률, 총손익, PF, 기대값, MDD, 평균/중앙 보유시간, 방향/섹터/시간대/레짐별 분해
- 레짐: BTC 1h ADX14 + EMA50 → trend_up / trend_down / chop
"""
import argparse, csv, glob, json, os, sys, time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import ccxt
from dotenv import load_dotenv

BASE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE, ".env"))
KST = ZoneInfo("Asia/Seoul")

MAJORS = {'BTC','ETH','SOL','XRP','ADA','AVAX','LINK','DOT','BNB','TRX'}
MEMES = {'DOGE','SHIB','PEPE','BOME','WIF','BONK','FLOKI','FARTCOIN','PUMP','PEOPLE','MOODENG',
         'PNUT','ACT','NEIRO','TURBO','MEW','POPCAT','GIGA','BRETT','TRUMP','MELANIA','PENGU','AI16Z'}
# OKX bills subType
OPEN_LONG, OPEN_SHORT = {3}, {4}
CLOSE_LONG = {5, 100, 104, 112, 125}
CLOSE_SHORT = {6, 101, 105, 113, 126}
LIQ = {100, 101, 104, 105, 125, 126}

LEV_MAJOR = int(os.getenv("OKX_MAJOR_LEVERAGE", os.getenv("OKX_LEVERAGE", "10")))
LEV_VENTURE = int(os.getenv("OKX_VENTURE_LEVERAGE", "5"))
LEV_SHORT = int(os.getenv("OKX_SHORT_LEVERAGE", "5"))


def sector(sym):
    base = sym.split('/')[0].split('-')[0]
    return 'major' if base in MAJORS else ('meme' if base in MEMES else 'alt')


def lev_for(sym, side):
    if side == 'short':
        return LEV_SHORT
    return LEV_MAJOR if sector(sym) == 'major' else LEV_VENTURE


def to_ccxt(inst):  # BTC-USDT-SWAP -> BTC/USDT:USDT
    p = inst.split('-')
    return f"{p[0]}/{p[1]}:{p[1]}"


# ── 소스 1: OKX bills (진짜 원장) ──
def fetch_bills(ex, begin_ms, end_ms):
    out, after = [], None
    while True:
        params = {"instType": "SWAP", "type": "2", "begin": str(begin_ms), "end": str(end_ms), "limit": "100"}
        if after:
            params["after"] = after
        try:
            res = ex.privateGetAccountBills(params)
            data = res.get("data", [])
            out.extend(data)
            if len(data) < 100:
                break
            after = data[-1]["billId"]
            time.sleep(0.25)
        except Exception as e:
            print(f"⚠️ bills fetch warning: {e}")
            break
    return out


def roundtrips_from_bills(bills, markets):
    """(instId, posSide)별 진입/청산 레그를 누적해 qty가 0으로 돌아올 때 라운드트립 확정."""
    legs = defaultdict(list)
    for b in bills:
        st = int(b.get("subType", 0))
        if st in OPEN_LONG | CLOSE_LONG:
            ps = 'long'
        elif st in OPEN_SHORT | CLOSE_SHORT:
            ps = 'short'
        else:
            continue
        legs[(b["instId"], ps)].append(b)

    rts, open_now = [], []
    for (inst, ps), rows in legs.items():
        rows.sort(key=lambda r: int(r["ts"]))
        sym = to_ccxt(inst)
        cur = None
        for r in rows:
            st, ts = int(r["subType"]), int(r["ts"]) / 1000
            sz, px = float(r.get("sz") or 0), float(r.get("px") or r.get("fillPx") or 0)
            fee, pnl = float(r.get("fee") or 0), float(r.get("pnl") or 0)
            is_open = st in OPEN_LONG | OPEN_SHORT
            if cur is None:
                if not is_open:
                    continue  # 윈도우 이전에 열린 포지션의 청산 → 진입정보 없음, 스킵
                cur = dict(symbol=sym, side=ps, sector=sector(sym), entry_ts=ts, qty=0.0, cost=0.0,
                           total_entry_sz=0.0, n_entries=0, n_exits=0, pnl=0.0, fee=0.0, liq=False, taker_legs=0, legs=0)
            cur["legs"] += 1
            cur["taker_legs"] += 1 if r.get("execType") == "T" else 0
            cur["fee"] += fee
            if is_open:
                cur["qty"] += sz
                cur["cost"] += sz * px
                cur["total_entry_sz"] += sz
                cur["n_entries"] += 1
                cur["last_entry_ts"] = ts
            else:
                cur["qty"] -= sz
                cur["pnl"] += pnl
                cur["n_exits"] += 1
                cur["liq"] |= (st in LIQ)
                if cur["qty"] <= 1e-9:
                    cur["exit_ts"] = ts
                    cur["avg_entry"] = cur["cost"] / max(1e-9, cur["total_entry_sz"])
                    rts.append(cur)
                    cur = None
        if cur is not None:
            open_now.append(dict(symbol=sym, side=ps, qty=cur["qty"], since=cur["entry_ts"]))
    return rts, open_now


def finalize(rts, markets):
    rows = []
    for r in rts:
        sym = r["symbol"]
        cs = float((markets.get(sym) or {}).get("contractSize") or 1)
        notional = r["cost"] * cs  # 진입 노셔널(USDT)
        lev = lev_for(sym, r["side"])
        margin = notional / lev if notional else 0.0
        net = r["pnl"] + r["fee"]  # fee는 음수
        rows.append(dict(
            symbol=sym, side=r["side"], sector=r["sector"], lev=lev,
            entry_kst=datetime.fromtimestamp(r["entry_ts"], KST).strftime("%m-%d %H:%M"),
            exit_kst=datetime.fromtimestamp(r["exit_ts"], KST).strftime("%m-%d %H:%M"),
            entry_ts=r["entry_ts"], exit_ts=r["exit_ts"],
            hold_h=(r["exit_ts"] - r["entry_ts"]) / 3600,
            entry_hour_kst=datetime.fromtimestamp(r["entry_ts"], KST).hour,
            n_entries=r["n_entries"], n_exits=r["n_exits"], taker_ratio=r["taker_legs"] / max(1, r["legs"]),
            notional=notional, margin=margin, gross_pnl=r["pnl"], fee=r["fee"], net_pnl=net,
            roe=(net / margin) if margin else 0.0, liquidation=r["liq"],
        ))
    return pd.DataFrame(rows)


# ── 소스 2: trades.jsonl (봇이 직접 보낸 주문만; 교차검증용) ──
def roundtrips_from_jsonl(path, markets):
    if not os.path.exists(path):
        return pd.DataFrame(), 0
    pos, rts, orphan = {}, [], 0
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        ts = r.get("ts", 0); ts = ts / 1000 if ts > 1e10 else ts
        side, px, amt = r.get("side"), float(r.get("price") or 0), float(r.get("amount") or 0)
        if not px:
            continue
        ps = 'short' if side in ("SELL", "CLOSE_SHORT") else 'long'
        key = (r["symbol"], ps)
        cs = float((markets.get(r["symbol"]) or {}).get("contractSize") or 1)
        if side in ("BUY", "SELL"):
            p = pos.setdefault(key, dict(qty=0.0, cost=0.0, entry_ts=ts, n_entries=0, n_exits=0, pnl=0.0))
            p["qty"] += amt; p["cost"] += amt * px; p["n_entries"] += 1
        else:
            p = pos.get(key)
            if not p or p["qty"] <= 0:
                orphan += 1; continue
            cq = p["qty"] if amt == 0 else min(amt, p["qty"])
            avg = p["cost"] / p["qty"]
            sgn = 1 if ps == 'long' else -1
            p["pnl"] += (px - avg) * cq * cs * sgn; p["n_exits"] += 1
            p["qty"] -= cq; p["cost"] = avg * p["qty"]
            if p["qty"] <= 1e-9:
                lev = lev_for(r["symbol"], ps); notional = avg * (p["cost"] / max(avg, 1e-9) + cq) * cs
                rts.append(dict(symbol=r["symbol"], side=ps, sector=sector(r["symbol"]), entry_ts=p["entry_ts"], exit_ts=ts,
                                hold_h=(ts - p["entry_ts"]) / 3600, n_entries=p["n_entries"], n_exits=p["n_exits"],
                                gross_pnl=p["pnl"], fee=-notional * 0.001, net_pnl=p["pnl"] - notional * 0.001,
                                entry_hour_kst=datetime.fromtimestamp(p["entry_ts"], KST).hour, lev=lev,
                                margin=notional / lev, roe=(p["pnl"] - notional * 0.001) / max(notional / lev, 1e-9),
                                liquidation=False, taker_ratio=1.0, notional=notional))
                del pos[key]
    return pd.DataFrame(rts), orphan


# ── 레짐 태깅 ──
def btc_regime_series(begin_ms):
    ex = ccxt.okx()
    since = begin_ms - 250 * 3600 * 1000
    try:
        o = ex.fetch_ohlcv("BTC/USDT:USDT", "1h", since=since, limit=300)
    except Exception:
        o = ex.fetch_ohlcv("BTC-USDT-SWAP", "1h", since=since, limit=300)
    df = pd.DataFrame(o, columns=['t', 'o', 'h', 'l', 'c', 'v'])
    h, l, c = df.h, df.l, df.c
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    up, dn = h.diff(), -l.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0); mdm = dn.where((dn > up) & (dn > 0), 0.0)
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    mdi = 100 * mdm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, float('nan'))
    df['adx'] = dx.ewm(alpha=1 / 14, adjust=False).mean().fillna(0)
    df['ema50'] = c.ewm(span=50, adjust=False).mean()
    df['atr_pct'] = atr / c * 100
    df['regime'] = ['chop' if a < 20 else ('trend_up' if cc > e else 'trend_down')
                    for a, cc, e in zip(df.adx, df.c, df.ema50)]
    df['t'] = df.t // 1000
    return df.set_index('t')


def tag_regime(df, reg):
    if df.empty or reg.empty:
        return df
    idx = reg.index.values
    def at(ts):
        i = idx.searchsorted(ts, side='right') - 1
        return reg.iloc[max(0, min(i, len(reg) - 1))]
    df['regime'] = [at(t)['regime'] for t in df.entry_ts]
    df['btc_adx'] = [round(at(t)['adx'], 1) for t in df.entry_ts]
    df['btc_atr_pct'] = [round(at(t)['atr_pct'], 2) for t in df.entry_ts]
    df['btc_move_pct'] = [round((at(e)['c'] / max(1e-9, at(s)['c']) - 1) * 100, 2) for s, e in zip(df.entry_ts, df.exit_ts)]
    return df


# ── 지표 ──
def metrics(df, equity_start=None):
    if df.empty:
        return {"trades": 0}
    w, l = df[df.net_pnl > 0], df[df.net_pnl <= 0]
    gp, gl = w.net_pnl.sum(), -l.net_pnl.sum()
    curve = df.sort_values('exit_ts').net_pnl.cumsum()
    dd = (curve - curve.cummax())
    m = dict(
        trades=len(df), wins=len(w), losses=len(l), win_rate=round(len(w) / len(df) * 100, 1),
        total_net_pnl=round(df.net_pnl.sum(), 2), total_fee=round(df.fee.sum(), 2),
        profit_factor=round(gp / gl, 2) if gl else float('inf'),
        avg_win=round(w.net_pnl.mean(), 2) if len(w) else 0, avg_loss=round(l.net_pnl.mean(), 2) if len(l) else 0,
        payoff=round(w.net_pnl.mean() / -l.net_pnl.mean(), 2) if len(w) and len(l) else None,
        expectancy=round(df.net_pnl.mean(), 2), avg_roe_pct=round(df.roe.mean() * 100, 1),
        mdd_usdt=round(dd.min(), 2),
        mdd_pct_of_start=round(dd.min() / equity_start * 100, 1) if equity_start else None,
        avg_hold_h=round(df.hold_h.mean(), 1), median_hold_h=round(df.hold_h.median(), 1),
        avg_win_hold_h=round(w.hold_h.mean(), 1) if len(w) else 0, avg_loss_hold_h=round(l.hold_h.mean(), 1) if len(l) else 0,
        avg_dca_entries=round(df.n_entries.mean(), 2), liquidations=int(df.liquidation.sum()),
        taker_ratio=round(df.taker_ratio.mean(), 2),
    )
    return m


def breakdown(df, col):
    g = df.groupby(col)
    out = pd.DataFrame({
        'n': g.size(), 'win_rate': (g.net_pnl.apply(lambda s: (s > 0).mean() * 100)).round(1),
        'net_pnl': g.net_pnl.sum().round(2), 'avg_pnl': g.net_pnl.mean().round(2),
        'pf': g.net_pnl.apply(lambda s: round(s[s > 0].sum() / max(1e-9, -s[s <= 0].sum()), 2)),
        'avg_hold_h': g.hold_h.mean().round(1),
    }).sort_values('net_pnl')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=(datetime.now(KST) - timedelta(days=5)).strftime("%Y-%m-%d"))
    ap.add_argument("--until", default=None)
    ap.add_argument("--equity-start", type=float, default=None)
    ap.add_argument("--no-bills", action="store_true", help="OKX bills 조회 생략 (jsonl만)")
    a = ap.parse_args()
    begin = datetime.strptime(a.since, "%Y-%m-%d").replace(tzinfo=KST)
    end = datetime.strptime(a.until, "%Y-%m-%d").replace(tzinfo=KST) + timedelta(days=1) if a.until else datetime.now(KST)
    begin_ms, end_ms = int(begin.timestamp() * 1000), int(end.timestamp() * 1000)

    pub = ccxt.okx(); markets = pub.load_markets()
    reg = btc_regime_series(begin_ms)

    md = [f"# 실거래 성과 분석 {a.since} ~ {end.strftime('%Y-%m-%d')} (KST)\n\n"]
    sources = {}

    if not a.no_bills:
        api_key = os.getenv("OKX_API_KEY")
        secret = os.getenv("OKX_SECRET") or os.getenv("OKX_API_SECRET")
        pw = os.getenv("OKX_PASSPHRASE") or os.getenv("OKX_PASSWORD")
        if api_key and secret and pw:
            try:
                ex = ccxt.okx({"apiKey": api_key, "secret": secret, "password": pw, "enableRateLimit": True})
                bills = fetch_bills(ex, begin_ms - 3 * 86400000, end_ms)
                rts, open_now = roundtrips_from_bills(bills, markets)
                dfb = finalize(rts, markets)
                if not dfb.empty:
                    dfb = dfb[(dfb.exit_ts >= begin.timestamp()) & (dfb.exit_ts < end.timestamp())].reset_index(drop=True)
                    sources['bills'] = tag_regime(dfb, reg)
                md.append(f"- OKX bills 레그 {len(bills)}건 → 윈도우 내 청산 완료 라운드트립 **{len(dfb) if not dfb.empty else 0}**건, 미청산 포지션 {len(open_now)}개\n")
            except Exception as e:
                md.append(f"- ⚠️ OKX bills 조회 실패: {e} (jsonl 데이터로 대체 분석 진행)\n")
        else:
            md.append("- ⚠️ OKX API 키 미설정 또는 불완전 → jsonl 데이터로 대체 분석 진행\n")

    dfj, orphan = roundtrips_from_jsonl(os.path.join(BASE, "state", "trades.jsonl"), markets)
    if not dfj.empty:
        dfj = dfj[(dfj.exit_ts >= begin.timestamp()) & (dfj.exit_ts < end.timestamp())].reset_index(drop=True)
        sources['jsonl'] = tag_regime(dfj, reg) if not dfj.empty else dfj
    md.append(f"- trades.jsonl 라운드트립 {len(dfj)}건 (진입기록 없는 청산 {orphan}건 스킵). bills 대비 부족분 = 거래소 SL/강제청산/수동 청산 추정\n")

    main_df = sources.get('bills') if ('bills' in sources and not sources['bills'].empty) else sources.get('jsonl', pd.DataFrame())
    result = {"window": [a.since, end.strftime('%Y-%m-%d')], "summary": {k: metrics(v, a.equity_start) for k, v in sources.items()}}

    if main_df is not None and not main_df.empty:
        m = result["summary"]['bills' if ('bills' in sources and not sources['bills'].empty) else 'jsonl']
        md.append("\n## 1. 핵심 지표\n\n")
        md += [f"- **{k}**: {v}\n" for k, v in m.items()]
        for col, title in [('side', '방향'), ('sector', '섹터'), ('regime', 'BTC 레짐(진입시점)'),
                           ('entry_hour_kst', '진입 시간대 KST'), ('symbol', '종목')]:
            b = breakdown(main_df, col); result[f"by_{col}"] = json.loads(b.to_json(orient='index'))
            md.append(f"\n## {title}별\n\n{b.to_markdown()}\n")
        main_df['day'] = [datetime.fromtimestamp(t, KST).strftime('%m-%d') for t in main_df.exit_ts]
        md.append(f"\n## 일자별(청산일)\n\n{breakdown(main_df, 'day').to_markdown()}\n")
        cols = ['symbol', 'side', 'sector', 'entry_kst', 'exit_kst', 'hold_h', 'n_entries', 'n_exits', 'lev',
                'margin', 'net_pnl', 'roe', 'regime', 'btc_adx', 'btc_move_pct', 'liquidation']
        cols = [c for c in cols if c in main_df.columns]
        md.append(f"\n## 최대 손실 10건\n\n{main_df.nsmallest(10, 'net_pnl')[cols].round(3).to_markdown(index=False)}\n")
        md.append(f"\n## 최대 수익 10건\n\n{main_df.nlargest(10, 'net_pnl')[cols].round(3).to_markdown(index=False)}\n")
        losses = main_df[main_df.net_pnl < 0].net_pnl
        if len(losses):
            top5 = losses.nsmallest(5).sum()
            result["loss_concentration_top5_pct"] = round(top5 / losses.sum() * 100, 1)
            md.append(f"\n- 손실 집중도: 최악 5건이 총손실의 **{result['loss_concentration_top5_pct']}%**\n")
        churn = main_df.groupby('symbol').size().sort_values(ascending=False).head(10)
        md.append(f"\n## 재진입 상위 종목 (왕복 의심)\n\n{churn.to_frame('roundtrips').to_markdown()}\n")
        os.makedirs(os.path.join(BASE, "reports"), exist_ok=True)
        main_df.to_csv(os.path.join(BASE, "reports", "roundtrips_last5d.csv"), index=False)

    files = sorted(glob.glob(os.path.join(BASE, "reports", "daily_report_*.txt")))[-5:]
    if files:
        md.append("\n## 참고: 기존 daily_report\n\n")
        for f in files:
            line = next((l for l in open(f, encoding='utf-8') if l.startswith('OKX')), '').strip()
            md.append(f"- {os.path.basename(f)}: {line}\n")

    out_md = os.path.join(BASE, "reports", "analysis_last5d.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("".join(md))
    json.dump(result, open(os.path.join(BASE, "reports", "analysis_last5d.json"), "w"), indent=2, ensure_ascii=False, default=str)
    print("".join(md))


if __name__ == "__main__":
    main()
