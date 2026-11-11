#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/research_alpaca_swing_v2.py
=============================================================================
Alpaca 신규 우상향 전략 2차 탐색 (2026-11-10 23:55 UTC)
1차(S1~S6)는 전부 AC1~AC6 미달. 원인:
  - 단순 월간 모멘텀(GEM)은 2022 주식+채권 동반 하락장에서 IEF도 함께 깨지며 MDD -29~34%
  - RSI(2) 2종목은 자본 가동률 ~15%에 그쳐 CAGR 3~7% (Sharpe 0.84, MDD -8%대로 양호하나 수익률 부족)
  - 일간 단타(S5/S6)는 수수료 마찰로 탈락

2차 전략군 (동일한 사전 등록 기준 AC1~AC6 유지):
  AC1: IS(2006~2018) & OOS(2019~현재) 모두 CAGR >= 8%
  AC2: OOS Sharpe >= 1.0 AND >= SPY OOS Sharpe (0.90)
  AC3: IS & OOS 모두 MDD >= -20%
  AC4: OOS 진입/리밸런스 >= 20회
  AC5: 최근 365일(252거래일) 수익 > 0
  AC6: 그리드 내 이웃 변형 중 >= 2/3가 AC1 & AC2 충족

후보 전략군:
  S7_VAA   : Vigilant Asset Allocation (Keller & Keuning) - 13612W 모멘텀, 카나리아 음수 시 즉시 SHY/IEF/GLD 대피
  S8_DAA   : Defensive Asset Allocation - 카나리아(EEM/AGG/TLT) 체크 후 공격/방어 전환
  S9_MULTI : Multi-Asset Momentum + Gold/Cash (SPY, QQQ, EFA, TLT, GLD 중 절대>0 & 최고모멘텀/분산, 아니면 BIL/SHY)
  S10_MEGA_MOM : 대형주+ETF 횡단면 모멘텀 + MA200 절대추세 필터 (상장일 이후 동적 편입)
  S11_MEGA_RSI2: 다종목 RSI(2) 스윙 (최대 K슬롯, 유휴자금 BIL/SHY 파킹 옵션 포함)
  S12_TREND_VOL: QQQ/SPY/GLD 변동성 역가중 + 개별 MA 추세필터 (미충족 비중은 BIL/SHY)
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf

DATA = "/home/hongsoonil02/quant_system/data/daily_swing_v2_20y.pkl"
BASE_ETFS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "SHY", "BIL", "GLD", "VNQ", "AGG", "LQD", "DBC"]
MEGAS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "COST", "NFLX", "AMD", "ORCL", "QCOM", "XOM", "JPM"]
TICKERS = sorted(set(BASE_ETFS + MEGAS))

FEE = 0.0005
IS_START = "2006-01-01"
IS_END = "2018-12-31"
OOS_START = "2019-01-01"


def load():
    if os.path.exists(DATA):
        return pd.read_pickle(DATA)
    raw = yf.download(TICKERS, start="2004-06-01", auto_adjust=True, progress=False, group_by="ticker")
    o = pd.DataFrame({t: raw[t]["Open"] for t in TICKERS})
    c = pd.DataFrame({t: raw[t]["Close"] for t in TICKERS})
    o = o.dropna(how="all")
    c = c.dropna(how="all")
    pd.to_pickle((o, c), DATA)
    return o, c


def month_end_mask(idx):
    s = pd.Series(idx, index=idx)
    return s.groupby([idx.year, idx.month]).transform("max") == s


def week_end_mask(idx):
    s = pd.Series(idx, index=idx)
    iso = idx.isocalendar()
    return s.groupby([iso.year, iso.week]).transform("max") == s


def hold_periodic(T_per, idx):
    return T_per.reindex(idx).ffill().fillna(0.0)


def run_weights(T, o, c):
    T = T.reindex(c.index).fillna(0.0)
    gap = (o / c.shift(1) - 1).fillna(0.0)
    intra = (c / o - 1).fillna(0.0)
    W1, W2 = T.shift(1).fillna(0.0), T.shift(2).fillna(0.0)
    ret = (W2 * gap).sum(1) + (W1 * intra).sum(1) - FEE * (W1 - W2).abs().sum(1)
    entries = ((W1.sum(1) > 0) & (W2.sum(1) == 0)).astype(int)
    rebal = ((W1 - W2).abs().sum(1) > 1e-6).astype(int)
    return ret, entries, rebal


def rsi(series, n=2):
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def mom_13612w(c):
    r1 = c / c.shift(21) - 1
    r3 = c / c.shift(63) - 1
    r6 = c / c.shift(126) - 1
    r12 = c / c.shift(252) - 1
    return (12 * r1 + 4 * r3 + 2 * r6 + 1 * r12) / 19.0


# ── S7: VAA (Vigilant Asset Allocation) ──────────────────────────────────────
def s7_vaa(o, c, top_n=1, use_gold=False, def_choice="best"):
    me = month_end_mask(c.index)
    m = mom_13612w(c)
    canary = ["SPY", "EFA", "EEM", "AGG"] if "AGG" in c and not c["AGG"].isna().all() else ["SPY", "EFA", "EEM", "IEF"]
    off_univ = ["QQQ", "SPY", "EFA", "EEM", "TLT"] + (["GLD"] if use_gold else [])
    def_univ = ["SHY", "IEF", "LQD"] + (["GLD"] if use_gold else [])

    T = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    for d in c.index[me]:
        if m.loc[d, canary].isna().any() or m.loc[d, off_univ].isna().any() or m.loc[d, def_univ].isna().any():
            continue
        T.loc[d] = 0.0
        if (m.loc[d, canary] < 0).any():
            if def_choice == "best":
                best_def = m.loc[d, def_univ].idxmax()
                T.loc[d, best_def] = 1.0
            else:
                T.loc[d, "SHY"] = 1.0
        else:
            picks = m.loc[d, off_univ].nlargest(top_n).index
            T.loc[d, picks] = 1.0 / len(picks)
    return hold_periodic(T, c.index)


# ── S8: DAA / Canary Momentum ────────────────────────────────────────────────
def s8_daa(o, c, canary_bad=1, top_n=2, def_pool=("SHY", "IEF", "GLD")):
    me = month_end_mask(c.index)
    m = mom_13612w(c)
    canary = ["EEM", "AGG"]
    off_univ = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "TLT", "GLD", "DBC"]
    T = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    for d in c.index[me]:
        if m.loc[d, canary].isna().any() or m.loc[d, list(def_pool)].isna().any():
            continue
        T.loc[d] = 0.0
        bad = int((m.loc[d, canary] < 0).sum())
        if bad >= canary_bad:
            best_def = m.loc[d, list(def_pool)].idxmax()
            T.loc[d, best_def] = 1.0
        else:
            avail = m.loc[d, off_univ].dropna()
            picks = avail.nlargest(top_n).index
            T.loc[d, picks] = 1.0 / len(picks)
    return hold_periodic(T, c.index)


# ── S9: Multi-Asset Dual Momentum + Gold/Trend ──────────────────────────────
def s9_multi(o, c, lb=126, top_n=2, cash_sym="SHY"):
    me = month_end_mask(c.index)
    univ = ["QQQ", "SPY", "EFA", "GLD", "TLT", "DBC"]
    r = c / c.shift(lb) - 1
    ma200 = c.rolling(200).mean()
    T = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    for d in c.index[me]:
        if pd.isna(r.loc[d, cash_sym]):
            continue
        T.loc[d] = 0.0
        # Eligible: momentum > cash_sym AND above MA200
        elig = [u for u in univ if pd.notna(r.loc[d, u]) and pd.notna(ma200.loc[d, u])
                and r.loc[d, u] > r.loc[d, cash_sym] and c.loc[d, u] > ma200.loc[d, u]]
        if not elig:
            T.loc[d, cash_sym] = 1.0
        else:
            picks = r.loc[d, elig].nlargest(top_n).index
            for p in picks:
                T.loc[d, p] = 1.0 / top_n
            rem = 1.0 - len(picks) / top_n
            if rem > 1e-6:
                T.loc[d, cash_sym] = rem
    return hold_periodic(T, c.index)


# ── S10: Mega-Cap Cross-Sectional Momentum + MA200 Filter ────────────────────
def s10_mega_mom(o, c, lb=126, top_n=4, freq="M", cash_sym="SHY"):
    mask = month_end_mask(c.index) if freq == "M" else week_end_mask(c.index)
    univ = MEGAS + ["QQQ", "GLD"]
    r = c[univ] / c[univ].shift(lb) - 1
    r_spy = c["SPY"] / c["SPY"].shift(lb) - 1
    ma200 = c.rolling(200).mean()
    T = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    for d in c.index[mask]:
        if pd.isna(ma200.loc[d, "SPY"]):
            continue
        T.loc[d] = 0.0
        if c.loc[d, "SPY"] <= ma200.loc[d, "SPY"]:
            T.loc[d, cash_sym] = 1.0
            continue
        elig = [u for u in univ if pd.notna(r.loc[d, u]) and pd.notna(ma200.loc[d, u])
                and c.loc[d, u] > ma200.loc[d, u] and r.loc[d, u] > 0]
        if not elig:
            T.loc[d, cash_sym] = 1.0
        else:
            picks = r.loc[d, elig].nlargest(top_n).index
            for p in picks:
                T.loc[d, p] = 1.0 / top_n
            rem = 1.0 - len(picks) / top_n
            if rem > 1e-6:
                T.loc[d, cash_sym] = rem
    return hold_periodic(T, c.index)


# ── S11: Multi-Asset / Mega-Cap RSI(2) Swing (Max K Slots + Cash Park) ──────
def s11_mega_rsi2(o, c, th=10.0, max_slots=4, park_cash=True, use_megas=True):
    univ = (["QQQ", "SPY", "IWM", "GLD"] + MEGAS) if use_megas else ["QQQ", "SPY", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "VNQ"]
    univ = [u for u in univ if u in c.columns]
    ma200 = c[univ].rolling(200).mean()
    ma5 = c[univ].rolling(5).mean()
    r2 = c[univ].apply(lambda s: rsi(s, 2))
    T = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    held = set()
    w = 1.0 / max_slots
    for i, d in enumerate(c.index):
        # Exits first
        for s in list(held):
            px = c.iloc[i][s]
            m5 = ma5.iloc[i][s]
            if pd.isna(px) or pd.isna(m5) or px > m5:
                held.remove(s)
        # Entries ranked by lowest RSI2
        if len(held) < max_slots:
            cands = []
            for s in univ:
                if s in held:
                    continue
                px, m200, rv = c.iloc[i][s], ma200.iloc[i][s], r2.iloc[i][s]
                if pd.notna(px) and pd.notna(m200) and pd.notna(rv) and px > m200 and rv < th:
                    cands.append((rv, s))
            cands.sort()
            for _, s in cands[: max_slots - len(held)]:
                held.add(s)
        for s in held:
            T.iloc[i, T.columns.get_loc(s)] = w
        if park_cash:
            rem = 1.0 - len(held) * w
            if rem > 1e-6 and pd.notna(c.iloc[i]["SHY"]):
                T.iloc[i, T.columns.get_loc("SHY")] = rem
    return T


# ── S12: Risk-Parity + Trend Filter (QQQ, SPY, GLD, TLT, SHY) ───────────────
def s12_trend_rp(o, c, ma_n=150, park="SHY", freq="W", lev=1.0):
    mask = week_end_mask(c.index) if freq == "W" else month_end_mask(c.index)
    univ = ["SPY", "QQQ", "GLD", "TLT"]
    ma = c[univ].rolling(ma_n).mean()
    vol = c[univ].pct_change().rolling(63).std()
    T = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    for d in c.index[mask]:
        if ma.loc[d].isna().any() or vol.loc[d].isna().any():
            continue
        T.loc[d] = 0.0
        iv = 1.0 / vol.loc[d].clip(lower=1e-4)
        base_w = iv / iv.sum()
        # Only hold assets above MA(ma_n), excess goes to park (SHY)
        park_w = 0.0
        for u in univ:
            if c.loc[d, u] > ma.loc[d, u]:
                T.loc[d, u] = base_w[u] * lev
            else:
                park_w += base_w[u]
        if park and pd.notna(c.loc[d, park]):
            T.loc[d, park] = park_w
    return hold_periodic(T, c.index)


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


def evaluate(name, ret, entries, rebal, periodic, spy_oos_sh):
    i = stats(ret, IS_START, IS_END)
    oo = stats(ret, OOS_START, None)
    n_oos = int((rebal if periodic else entries).loc[OOS_START:].sum())
    y1 = (1 + ret.iloc[-252:]).prod() - 1
    d60 = (1 + ret.iloc[-42:]).prod() - 1
    ac = {
        "AC1": i["cagr"] >= 0.08 and oo["cagr"] >= 0.08,
        "AC2": oo["sharpe"] >= 1.0 and oo["sharpe"] >= spy_oos_sh,
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
    spy_oos_sh = stats(spy_ret, OOS_START)["sharpe"]
    b_is, b_oo = stats(spy_ret, IS_START, IS_END), stats(spy_ret, OOS_START)
    print(f"[벤치] SPY 보유  IS CAGR {b_is['cagr']*100:5.1f}% MDD {b_is['mdd']*100:6.1f}% | "
          f"OOS CAGR {b_oo['cagr']*100:5.1f}% Sharpe {b_oo['sharpe']:.2f} MDD {b_oo['mdd']*100:6.1f}%\n")

    grids = {
        "S7_VAA": [
            ("top1 cash_best", dict(top_n=1, use_gold=False, def_choice="best")),
            ("top1 gold_best", dict(top_n=1, use_gold=True, def_choice="best")),
            ("top2 cash_best", dict(top_n=2, use_gold=False, def_choice="best")),
            ("top2 gold_best", dict(top_n=2, use_gold=True, def_choice="best")),
        ],
        "S8_DAA": [
            ("bad1 top2", dict(canary_bad=1, top_n=2)),
            ("bad1 top3", dict(canary_bad=1, top_n=3)),
            ("bad2 top2", dict(canary_bad=2, top_n=2)),
            ("bad2 top3", dict(canary_bad=2, top_n=3)),
        ],
        "S9_MULTI": [
            (f"lb{lb} top{tn}", dict(lb=lb, top_n=tn)) for lb in (63, 126, 189) for tn in (1, 2, 3)
        ],
        "S10_MEGA_MOM": [
            (f"{fr} lb{lb} top{tn}", dict(lb=lb, top_n=tn, freq=fr)) for fr in ("M", "W") for lb in (63, 126) for tn in (3, 5)
        ],
        "S11_MEGA_RSI2": [
            (f"{'MG' if mg else 'ET'} th{th} k{k} {'pk' if pk else 'no'}", dict(th=th, max_slots=k, park_cash=pk, use_megas=mg))
            for mg in (True, False) for th in (10, 15) for k in (3, 5) for pk in (True,)
        ],
        "S12_TREND_RP": [
            (f"{fr} ma{ma} {'pk' if pk else 'no'}", dict(ma_n=ma, park=pk, freq=fr))
            for fr in ("W", "M") for ma in (100, 150, 200) for pk in ("SHY", None)
        ],
    }
    fns = {
        "S7_VAA": s7_vaa, "S8_DAA": s8_daa, "S9_MULTI": s9_multi,
        "S10_MEGA_MOM": s10_mega_mom, "S11_MEGA_RSI2": s11_mega_rsi2, "S12_TREND_RP": s12_trend_rp
    }
    periodic = {"S7_VAA", "S8_DAA", "S9_MULTI", "S10_MEGA_MOM", "S12_TREND_RP"}

    rows, saved = [], {}
    for fam, grid in grids.items():
        fam_rows = []
        for label, kw in grid:
            T = fns[fam](o, c, **kw)
            ret, ent, reb = run_weights(T, o, c)
            r = evaluate(f"{fam} {label}", ret, ent, reb, fam in periodic, spy_oos_sh)
            r["fam"] = fam
            fam_rows.append(r)
            saved[r["name"]] = ret
        ok = sum(1 for r in fam_rows if r["AC1"] and r["AC2"])
        for r in fam_rows:
            r["AC6"] = ok >= len(fam_rows) * 2 / 3
        rows += fam_rows

    hdr = f"{'전략':<30}|{'IS CAGR':>8}{'IS MDD':>8}|{'OOS CAGR':>9}{'OOS Sh':>7}{'OOS MDD':>8}|{'N':>5}{'1Y':>7}{'60d':>7}| AC1-6"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        flags = "".join("O" if r[k] else "x" for k in ("AC1", "AC2", "AC3", "AC4", "AC5", "AC6"))
        star = " ★PASS★" if all(r[k] for k in ("AC1", "AC2", "AC3", "AC4", "AC5", "AC6")) else ""
        print(f"{r['name']:<30}|{r['is_cagr']*100:7.1f}%{r['is_mdd']*100:7.1f}%|"
              f"{r['oos_cagr']*100:8.1f}%{r['oos_sh']:7.2f}{r['oos_mdd']*100:7.1f}%|"
              f"{r['n_oos']:5d}{r['y1']*100:6.1f}%{r['d60']*100:6.1f}%| {flags}{star}")


if __name__ == "__main__":
    main()

