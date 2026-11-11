#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/research_alpaca_daily_swing.py
=============================================================================
Alpaca 신규 우상향 전략 탐색 — 일봉 스윙/로테이션 전략군 (2026-11-10 23:55 UTC)

배경: 장중 모멘텀(Golden v3), 1H RSI2 딥(V4 pods), 베어랠리 롱, MA200 단독은 모두 기각.
      아직 미검증인 '일봉 기반·저빈도·학계 검증 이상현상' 전략군을 20년 데이터로 검증.
      PDT($25k) 제약과 무관하도록 전부 오버나잇 보유(스윙) 구조.

■ 사전 등록 채택 기준 (실행 전 확정, 결과 보고 변경 금지)
  기간: IS 2006-01-01 ~ 2018-12-31 / OOS 2019-01-01 ~ 최신
  AC1: IS, OOS 모두 CAGR >= 8%
  AC2: OOS Sharpe >= 1.0  그리고  OOS Sharpe >= SPY 보유 OOS Sharpe
  AC3: IS, OOS 모두 MDD >= -20%
  AC4: OOS 진입 횟수 >= 20 (월간 리밸런싱 전략은 리밸런스 횟수로 대체)
  AC5: 최근 365일 수익 > 0
  AC6: 파라미터 강건성 — 해당 전략의 모든 이웃 변형(그리드) 중 >= 2/3가 AC1·AC2 충족
  → 전부 충족 시에만 '후보' 채택 → 이후 모의계좌 포워드 테스트 단계로 진행.

■ 체결 모델 (룩어헤드 차단): 종가 d에 신호 산출 → 익일 d+1 시가 체결.
  일수익 = W(d-2)·갭수익(d) + W(d-1)·장중수익(d) − 5bps × 회전율.  현금 수익률 0%.
  (S6 오버나이트만 예외: 종가 매수 → 익일 시가 매도, 왕복 10bps)

■ 전략군
  S1 GEM 듀얼 모멘텀 (Antonacci): 월말 12M 수익 비교, 위험자산 vs SHY, 패배 시 IEF
  S2 섹터 로테이션: 9개 SPDR 섹터 6M 모멘텀 상위 N, SPY>MA200일 때만, 아니면 IEF
  S3 RSI(2) 평균회귀 (Connors): SPY/QQQ 각 50%, 종가>MA200 & RSI2<임계 → 종가>MA5 청산
  S4 GTAA (Faber): SPY/EFA/IEF/GLD/VNQ 각 1/N, 월말 종가>10개월 SMA인 자산만 보유
  S5 월말·월초 효과: 월 마지막 2거래일~첫 3거래일 SPY 보유
  S6 오버나이트 QQQ: QQQ>MA200일 때 종가 매수·익일 시가 매도
  C1 (사전 등록 조합): S1·S3·S4 기본형 1/3씩
"""
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf

DATA = "/home/hongsoonil02/quant_system/data/daily_swing_20y.pkl"
TICKERS = ["SPY", "QQQ", "IWM", "EFA", "TLT", "IEF", "SHY", "GLD", "VNQ",
           "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]
FEE = 0.0005
IS_END = "2018-12-31"
OOS_START = "2019-01-01"


def load():
    if os.path.exists(DATA):
        return pd.read_pickle(DATA)
    raw = yf.download(TICKERS, start="2004-06-01", auto_adjust=True, progress=False, group_by="ticker")
    o = pd.DataFrame({t: raw[t]["Open"] for t in TICKERS})
    c = pd.DataFrame({t: raw[t]["Close"] for t in TICKERS})
    o, c = o.dropna(how="all"), c.dropna(how="all")
    pd.to_pickle((o, c), DATA)
    return o, c


def run_weights(T, o, c):
    """T: target weights decided at close d (DataFrame aligned to c). Executed next open."""
    T = T.reindex(c.index).fillna(0.0)
    gap = (o / c.shift(1) - 1).fillna(0.0)
    intra = (c / o - 1).fillna(0.0)
    W1, W2 = T.shift(1).fillna(0.0), T.shift(2).fillna(0.0)
    ret = (W2 * gap).sum(1) + (W1 * intra).sum(1) - FEE * (W1 - W2).abs().sum(1)
    entries = ((W1.sum(1) > 0) & (W2.sum(1) == 0)).astype(int)
    rebal = ((W1 - W2).abs().sum(1) > 1e-9).astype(int)
    return ret, entries, rebal


def month_end_mask(idx):
    s = pd.Series(idx, index=idx)
    return s.groupby([idx.year, idx.month]).transform("max") == s


def rsi(series, n=2):
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def hold_monthly(T_me, idx):
    """Forward-fill month-end decisions to all days."""
    return T_me.reindex(idx).ffill().fillna(0.0)


# ── Strategies ───────────────────────────────────────────────────────────────
def s1_gem(o, c, lb=252, risk=("SPY", "EFA")):
    me = month_end_mask(c.index)
    r = c / c.shift(lb) - 1
    T = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    for d in c.index[me]:
        if pd.isna(r.loc[d, "SHY"]) or r.loc[d, list(risk)].isna().any():
            continue
        best = r.loc[d, list(risk)].idxmax()
        T.loc[d, best if r.loc[d, best] > r.loc[d, "SHY"] else "IEF"] = 1.0
    T[~me] = np.nan
    return hold_monthly(T, c.index)


SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]


def s2_sector(o, c, lb=126, top=3):
    me = month_end_mask(c.index)
    r = c[SECTORS] / c[SECTORS].shift(lb) - 1
    ma = c["SPY"].rolling(200).mean()
    T = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    for d in c.index[me]:
        if pd.isna(ma.loc[d]) or r.loc[d].isna().any():
            continue
        T.loc[d] = 0.0
        if c.loc[d, "SPY"] > ma.loc[d]:
            T.loc[d, r.loc[d].nlargest(top).index] = 1.0 / top
        else:
            T.loc[d, "IEF"] = 1.0
    return hold_monthly(T, c.index)


def s3_rsi2(o, c, th=10.0, exit_mode="ma5", syms=("SPY", "QQQ")):
    T = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    w = 1.0 / len(syms)
    for s in syms:
        px = c[s]
        ma200, ma5, r2 = px.rolling(200).mean(), px.rolling(5).mean(), rsi(px, 2)
        pos, held = np.zeros(len(px)), False
        for i in range(len(px)):
            if not held:
                if px.iloc[i] > ma200.iloc[i] and r2.iloc[i] < th:
                    held = True
            else:
                if (exit_mode == "ma5" and px.iloc[i] > ma5.iloc[i]) or \
                   (exit_mode == "rsi70" and r2.iloc[i] > 70):
                    held = False
            pos[i] = w if held else 0.0
        T[s] = pos
    return T


def s4_gtaa(o, c, assets=("SPY", "EFA", "IEF", "GLD", "VNQ"), sma_m=10, invvol=False):
    me = month_end_mask(c.index)
    sma = c.rolling(21 * sma_m).mean()
    vol = c.pct_change().rolling(63).std()
    T = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    for d in c.index[me]:
        if sma.loc[d, list(assets)].isna().any():
            continue
        T.loc[d] = 0.0
        on = [a for a in assets if c.loc[d, a] > sma.loc[d, a]]
        if not on:
            continue
        if invvol:
            iv = 1.0 / vol.loc[d, on]
            for a in on:
                T.loc[d, a] = iv[a] / iv.sum() * len(on) / len(assets)
        else:
            for a in on:
                T.loc[d, a] = 1.0 / len(assets)
    return hold_monthly(T, c.index)


def s5_tom(o, c, pre=2, post=3):
    idx = c.index
    s = pd.Series(range(len(idx)), index=idx)
    g = s.groupby([idx.year, idx.month])
    from_end = g.transform("max") - s          # 0 = last trading day
    from_start = s - g.transform("min")        # 0 = first trading day
    # we want to HOLD on days: last `pre` days and first `post` days.
    hold = (from_end < pre) | (from_start < post)
    # decision at close d targets holding on d+1 → shift hold back by 1
    T = pd.DataFrame(0.0, index=idx, columns=c.columns)
    T["SPY"] = hold.shift(-1).fillna(False).astype(float)
    return T


def s6_overnight(o, c, sym="QQQ", ma_n=200):
    ma = c[sym].rolling(ma_n).mean()
    sig = (c[sym] > ma).astype(float)      # decided & bought at close d
    gap = (o[sym] / c[sym].shift(1) - 1).fillna(0.0)
    held = sig.shift(1).fillna(0.0)
    ret = held * gap - held * 2 * FEE
    entries = held.astype(int)
    return ret, entries, entries


# ── Metrics ─────────────────────────────────────────────────────────────────
def stats(ret, start=None, end=None):
    r = ret.loc[start:end].dropna()
    if len(r) < 20:
        return dict(cagr=np.nan, sharpe=np.nan, mdd=np.nan)
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    mdd = (eq / eq.cummax() - 1).min()
    return dict(cagr=cagr, sharpe=sharpe, mdd=mdd)


def window_ret(ret, days):
    r = ret.iloc[-days:]
    return (1 + r).prod() - 1


def evaluate(name, ret, entries, rebal, monthly, spy_oos_sharpe):
    i, oo = stats(ret, None, IS_END), stats(ret, OOS_START, None)
    n_oos = int((rebal if monthly else entries).loc[OOS_START:].sum())
    y1, d60 = window_ret(ret, 252), window_ret(ret, 42)
    ac = {
        "AC1": i["cagr"] >= 0.08 and oo["cagr"] >= 0.08,
        "AC2": oo["sharpe"] >= 1.0 and oo["sharpe"] >= spy_oos_sharpe,
        "AC3": i["mdd"] >= -0.20 and oo["mdd"] >= -0.20,
        "AC4": n_oos >= 20,
        "AC5": y1 > 0,
    }
    return dict(name=name, is_cagr=i["cagr"], is_mdd=i["mdd"], is_sh=i["sharpe"],
                oos_cagr=oo["cagr"], oos_mdd=oo["mdd"], oos_sh=oo["sharpe"],
                n_oos=n_oos, y1=y1, d60=d60, **ac)


def main():
    o, c = load()
    c = c.loc["2005-01-01":]
    o = o.loc[c.index]
    print(f"데이터: {c.index[0].date()} ~ {c.index[-1].date()} ({len(c)}일)\n")

    spy_ret = c["SPY"].pct_change().fillna(0.0)
    qqq_ret = c["QQQ"].pct_change().fillna(0.0)
    spy_oos_sh = stats(spy_ret, OOS_START)["sharpe"]

    grids = {
        "S1_GEM": [("SPY/EFA lb12m", dict(lb=252)), ("SPY/EFA lb9m", dict(lb=189)),
                   ("SPY/EFA lb6m", dict(lb=126)), ("QQQ/EFA lb12m", dict(lb=252, risk=("QQQ", "EFA")))],
        "S2_SECTOR": [(f"lb{lb}d top{t}", dict(lb=lb, top=t)) for lb in (63, 126, 252) for t in (2, 3, 4)],
        "S3_RSI2": [(f"th{th} {ex}", dict(th=th, exit_mode=ex)) for th in (5, 10, 15) for ex in ("ma5", "rsi70")],
        "S4_GTAA": [(f"sma{m}m {'iv' if iv else 'eq'}", dict(sma_m=m, invvol=iv)) for m in (8, 10, 12) for iv in (False, True)],
        "S5_TOM": [(f"pre{p} post{q}", dict(pre=p, post=q)) for p in (1, 2, 3) for q in (2, 3, 4)],
        "S6_OVN": [(f"{s} ma{n}", dict(sym=s, ma_n=n)) for s in ("QQQ", "SPY") for n in (100, 200)],
    }
    fns = {"S1_GEM": s1_gem, "S2_SECTOR": s2_sector, "S3_RSI2": s3_rsi2,
           "S4_GTAA": s4_gtaa, "S5_TOM": s5_tom, "S6_OVN": s6_overnight}
    monthly = {"S1_GEM", "S2_SECTOR", "S4_GTAA"}
    base = {"S1_GEM": "SPY/EFA lb12m", "S2_SECTOR": "lb126d top3", "S3_RSI2": "th10 ma5",
            "S4_GTAA": "sma10m eq", "S5_TOM": "pre2 post3", "S6_OVN": "QQQ ma200"}

    rows, base_rets = [], {}
    b = stats(spy_ret, None, IS_END), stats(spy_ret, OOS_START), stats(qqq_ret, OOS_START)
    print(f"[벤치] SPY 보유  IS CAGR {b[0]['cagr']*100:5.1f}% MDD {b[0]['mdd']*100:6.1f}% | "
          f"OOS CAGR {b[1]['cagr']*100:5.1f}% Sharpe {b[1]['sharpe']:.2f} MDD {b[1]['mdd']*100:6.1f}%")
    print(f"[벤치] QQQ 보유  OOS CAGR {b[2]['cagr']*100:5.1f}% Sharpe {b[2]['sharpe']:.2f} MDD {b[2]['mdd']*100:6.1f}%\n")

    for fam, grid in grids.items():
        fam_rows = []
        for label, kw in grid:
            if fam == "S6_OVN":
                ret, ent, reb = s6_overnight(o, c, **kw)
            else:
                ret, ent, reb = run_weights(fns[fam](o, c, **kw), o, c)
            r = evaluate(f"{fam} {label}", ret, ent, reb, fam in monthly, spy_oos_sh)
            r["fam"], r["is_base"] = fam, (label == base[fam])
            if r["is_base"]:
                base_rets[fam] = ret
            fam_rows.append(r)
        ok = sum(1 for r in fam_rows if r["AC1"] and r["AC2"])
        for r in fam_rows:
            r["AC6"] = ok >= len(fam_rows) * 2 / 3
        rows += fam_rows

    # C1 pre-registered combo
    comb = (base_rets["S1_GEM"] + base_rets["S3_RSI2"] + base_rets["S4_GTAA"]) / 3
    rc = evaluate("C1 S1+S3+S4 (1/3씩)", comb, comb * 0 + 1, comb * 0 + 1, True, spy_oos_sh)
    rc["AC4"], rc["AC6"], rc["fam"], rc["is_base"] = True, True, "C1", True
    rows.append(rc)

    hdr = f"{'전략':<28}|{'IS CAGR':>8}{'IS MDD':>8}|{'OOS CAGR':>9}{'OOS Sh':>7}{'OOS MDD':>8}|{'N':>5}{'1Y':>7}{'60d':>7}| AC1-6"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        flags = "".join("O" if r[k] else "x" for k in ("AC1", "AC2", "AC3", "AC4", "AC5", "AC6"))
        star = " ★" if all(r[k] for k in ("AC1", "AC2", "AC3", "AC4", "AC5", "AC6")) else ""
        mark = "*" if r["is_base"] else " "
        print(f"{mark}{r['name']:<27}|{r['is_cagr']*100:7.1f}%{r['is_mdd']*100:7.1f}%|"
              f"{r['oos_cagr']*100:8.1f}%{r['oos_sh']:7.2f}{r['oos_mdd']*100:7.1f}%|"
              f"{r['n_oos']:5d}{r['y1']*100:6.1f}%{r['d60']*100:6.1f}%| {flags}{star}")
    pd.to_pickle(rows, "/home/hongsoonil02/quant_system/data/daily_swing_results.pkl")


if __name__ == "__main__":
    main()

