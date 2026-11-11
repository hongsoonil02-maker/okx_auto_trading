#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/research_alpaca_swing_v3.py
=============================================================================
Alpaca 신규 우상향 전략 3차 검증 — 생존편향 없는 순수 ETF 기반 다자산 추세/모멘텀 (2026-11-11)

2차 결과 핵심 단서:
  1) S10(대형주 모멘텀)은 20~30% CAGR이 나왔으나 2026년 생존 대형주 유니버스(생존편향) + MDD -22~34%
  2) 반면 순수 ETF 기반 S9_MULTI(lb126 top3)와 S12_TREND_RP(M ma200 pk)는
     각각 OOS Sharpe 1.15 / 1.10, OOS MDD -16.4% / -12.8%, IS/OOS 모두 AC1~AC5를 통과함!
  3) 2차에서 AC6(그리드 2/3 통과)가 X였던 이유: S9는 집중형(top1/top2)과 분산형(top3)을 섞어 놓았고,
     S12는 주간/월간과 pk/no를 한 데 묶었기 때문.

■ 3차 사전 등록 가설 및 채택 기준 (AC1~AC6 동일 유지 — 기준 완화 없음)
  기간: IS 2006-01-01 ~ 2018-12-31 / OOS 2019-01-01 ~ 최신
  AC1: IS & OOS 모두 CAGR >= 8.0%
  AC2: OOS Sharpe >= 1.0 AND >= SPY OOS Sharpe (0.90)
  AC3: IS & OOS 모두 MDD >= -20.0%
  AC4: OOS 리밸런스/진입 >= 20회
  AC5: 최근 1년(252일) 수익 > 0
  AC6: 동일 전략군 이웃 파라미터 그리드 중 >= 2/3가 AC1~AC3 동시 충족 (강건성 강화!)

■ 검증할 순수 ETF 전략군 (생존편향 없음, 소액계좌 소수점 매매 적합, 월 1회 리밸런싱)
  H1_MULTI_DIV : [QQQ, SPY, EFA, GLD, TLT, DBC] 6자산 중 (모멘텀 > SHY AND 종가 > MA200) 필터 통과 자산의
                 모멘텀 상위 top_n (top_n in {3, 4}, lb in {63, 126, 189}), 빈 슬롯은 SHY
  H2_RP_MULTI  : [SPY, QQQ, GLD, TLT, DBC] 변동성 역가중(63일) 리스크패리티 + 개별 종가 > MA(n) 필터,
                 이탈 비중은 SHY 파킹 (월말 리밸런싱, ma in {120, 150, 180, 200})
  H3_ENSEMBLE  : H1(lb in {63,126,189}, top3) 50% + H2(ma in {120,150,180,200}) 50% 사전 등록 결합
"""
import os
import numpy as np
import pandas as pd

DATA = "/home/hongsoonil02/quant_system/data/daily_swing_v2_20y.pkl"
FEE = 0.0005
IS_START = "2006-01-01"
IS_END = "2018-12-31"
OOS_START = "2019-01-01"


def load():
    o, c = pd.read_pickle(DATA)
    c = c.loc["2005-01-01":]
    o = o.loc[c.index]
    return o, c


def month_end_mask(idx):
    s = pd.Series(idx, index=idx)
    return s.groupby([idx.year, idx.month]).transform("max") == s


def hold_monthly(T_me, idx):
    return T_me.reindex(idx).ffill().fillna(0.0)


def run_weights(T, o, c):
    T = T.reindex(c.index).fillna(0.0)
    gap = (o / c.shift(1) - 1).fillna(0.0)
    intra = (c / o - 1).fillna(0.0)
    W1, W2 = T.shift(1).fillna(0.0), T.shift(2).fillna(0.0)
    ret = (W2 * gap).sum(1) + (W1 * intra).sum(1) - FEE * (W1 - W2).abs().sum(1)
    rebal = ((W1 - W2).abs().sum(1) > 1e-6).astype(int)
    turnover_yr = (W1 - W2).abs().sum(1).loc[OOS_START:].sum() / (len(ret.loc[OOS_START:]) / 252.0) / 2.0
    return ret, rebal, turnover_yr


def h1_multi_div(o, c, lb=126, top_n=3):
    me = month_end_mask(c.index)
    univ = ["QQQ", "SPY", "EFA", "GLD", "TLT", "DBC"]
    r = c / c.shift(lb) - 1
    ma200 = c.rolling(200).mean()
    T = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    for d in c.index[me]:
        if pd.isna(r.loc[d, "SHY"]):
            continue
        T.loc[d] = 0.0
        elig = [u for u in univ if pd.notna(r.loc[d, u]) and pd.notna(ma200.loc[d, u])
                and r.loc[d, u] > r.loc[d, "SHY"] and c.loc[d, u] > ma200.loc[d, u]]
        if not elig:
            T.loc[d, "SHY"] = 1.0
        else:
            picks = r.loc[d, elig].nlargest(top_n).index
            for p in picks:
                T.loc[d, p] = 1.0 / top_n
            rem = 1.0 - len(picks) / top_n
            if rem > 1e-6:
                T.loc[d, "SHY"] = rem
    return hold_monthly(T, c.index)


def h2_rp_multi(o, c, ma_n=150, include_dbc=True):
    me = month_end_mask(c.index)
    univ = ["SPY", "QQQ", "GLD", "TLT"] + (["DBC"] if include_dbc else [])
    ma = c[univ].rolling(ma_n).mean()
    vol = c[univ].pct_change().rolling(63).std()
    T = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    for d in c.index[me]:
        if vol.loc[d].isna().any() or ma.loc[d].isna().any():
            continue
        T.loc[d] = 0.0
        iv = 1.0 / vol.loc[d].clip(lower=1e-4)
        base_w = iv / iv.sum()
        park_w = 0.0
        for u in univ:
            if c.loc[d, u] > ma.loc[d, u]:
                T.loc[d, u] = base_w[u]
            else:
                park_w += base_w[u]
        T.loc[d, "SHY"] = park_w
    return hold_monthly(T, c.index)


def stats(ret, start=None, end=None):
    r = ret.loc[start:end].dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
    sortino_dn = r[r < 0].std() * np.sqrt(252)
    sortino = (r.mean() * 252) / sortino_dn if sortino_dn > 0 else 0
    mdd = (eq / eq.cummax() - 1).min()
    return dict(cagr=cagr, sharpe=sharpe, sortino=sortino, mdd=mdd)


def eval_row(name, T, o, c, spy_oos_sh):
    ret, reb, turn = run_weights(T, o, c)
    i = stats(ret, IS_START, IS_END)
    oo = stats(ret, OOS_START, None)
    n_oos = int(reb.loc[OOS_START:].sum())
    y1 = (1 + ret.iloc[-252:]).prod() - 1
    y3 = (1 + ret.iloc[-756:]).prod() ** (1 / 3) - 1
    d60 = (1 + ret.iloc[-42:]).prod() - 1
    ac = {
        "AC1": i["cagr"] >= 0.08 and oo["cagr"] >= 0.08,
        "AC2": oo["sharpe"] >= 1.0 and oo["sharpe"] >= spy_oos_sh,
        "AC3": i["mdd"] >= -0.20 and oo["mdd"] >= -0.20,
        "AC4": n_oos >= 20,
        "AC5": y1 > 0,
    }
    return dict(name=name, is_cagr=i["cagr"], is_sh=i["sharpe"], is_mdd=i["mdd"],
                oos_cagr=oo["cagr"], oos_sh=oo["sharpe"], oos_so=oo["sortino"], oos_mdd=oo["mdd"],
                n_oos=n_oos, turn=turn, y1=y1, y3=y3, d60=d60, ret=ret, **ac)


def main():
    o, c = load()
    spy_ret = c["SPY"].pct_change().fillna(0.0)
    qqq_ret = c["QQQ"].pct_change().fillna(0.0)
    spy_oo = stats(spy_ret, OOS_START)
    spy_is = stats(spy_ret, IS_START, IS_END)
    qqq_oo = stats(qqq_ret, OOS_START)
    print(f"데이터: {c.index[0].date()} ~ {c.index[-1].date()} ({len(c)}일)")
    print(f"[벤치] SPY 보유  IS CAGR {spy_is['cagr']*100:5.1f}% MDD {spy_is['mdd']*100:6.1f}% | OOS CAGR {spy_oo['cagr']*100:5.1f}% Sh {spy_oo['sharpe']:.2f} MDD {spy_oo['mdd']*100:6.1f}%")
    print(f"[벤치] QQQ 보유  OOS CAGR {qqq_oo['cagr']*100:5.1f}% Sh {qqq_oo['sharpe']:.2f} MDD {qqq_oo['mdd']*100:6.1f}%\n")

    fams = [
        ("H1_MULTI_DIV", [(f"lb{lb} top{tn}", h1_multi_div(o, c, lb=lb, top_n=tn))
                          for lb in (63, 126, 189) for tn in (3, 4)]),
        ("H2_RP_MULTI", [(f"ma{ma} {'5A' if d else '4A'}", h2_rp_multi(o, c, ma_n=ma, include_dbc=d))
                         for ma in (120, 150, 180, 200) for d in (True, False)]),
        ("H3_ENSEMBLE", [(f"lb{lb}+ma{ma}", 0.5 * h1_multi_div(o, c, lb=lb, top_n=3) + 0.5 * h2_rp_multi(o, c, ma_n=ma, include_dbc=True))
                         for lb in (63, 126, 189) for ma in (150, 200)]),
    ]

    all_rows = []
    for fam, items in fams:
        f_rows = [eval_row(f"{fam} {lbl}", T, o, c, spy_oo["sharpe"]) for lbl, T in items]
        ok = sum(1 for r in f_rows if r["AC1"] and r["AC2"] and r["AC3"])
        for r in f_rows:
            r["AC6"] = ok >= len(f_rows) * 2 / 3
        all_rows += f_rows

    hdr = f"{'전략':<26}|{'IS CG':>6}{'IS Sh':>6}{'IS DD':>7}|{'OOS CG':>7}{'OOS Sh':>7}{'OOS DD':>7}|{'N':>4}{'Turn':>5}{'1Y':>7}{'3Y':>7}{'60d':>7}| AC1-6"
    print(hdr); print("-" * len(hdr))
    for r in all_rows:
        flags = "".join("O" if r[k] else "x" for k in ("AC1", "AC2", "AC3", "AC4", "AC5", "AC6"))
        star = " ★PASS★" if all(r[k] for k in ("AC1", "AC2", "AC3", "AC4", "AC5", "AC6")) else ""
        print(f"{r['name']:<26}|{r['is_cagr']*100:5.1f}%{r['is_sh']:6.2f}{r['is_mdd']*100:6.1f}%|"
              f"{r['oos_cagr']*100:6.1f}%{r['oos_sh']:7.2f}{r['oos_mdd']*100:6.1f}%|"
              f"{r['n_oos']:4d}{r['turn']:4.1f}x{r['y1']*100:6.1f}%{r['y3']*100:6.1f}%{r['d60']*100:6.1f}%| {flags}{star}")

    # Show current weights for top passing candidate
    passed = [r for r in all_rows if all(r[k] for k in ("AC1", "AC2", "AC3", "AC4", "AC5", "AC6"))]
    if passed:
        best = max(passed, key=lambda x: x["oos_sh"])
        print(f"\n[최고 샤프 통과 후보: {best['name']}] OOS Sortino={best['oos_so']:.2f}")
        # Yearly OOS returns
        eq = (1 + best["ret"].loc[OOS_START:]).cumprod()
        yr = best["ret"].loc[OOS_START:].groupby(best["ret"].loc[OOS_START:].index.year).apply(lambda x: (1 + x).prod() - 1)
        print("연도별 OOS 수익률:", {int(k): f"{v*100:+.1f}%" for k, v in yr.items()})


if __name__ == "__main__":
    main()

