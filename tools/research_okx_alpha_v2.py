#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/research_okx_alpha_v2.py
=============================================================================
OKX 무기한 선물 2차 알파 탐색 — 2022~2026 실데이터 (1차 캐시 재사용)

1차 발견:
  - 2024-07 이후 OOS에서 BTC 자체가 CAGR +6.5%, Sharpe 0.36, MDD -53.3%로 저조했고,
    8종목 알트 분산 로테이션은 알트 약세에 휩쓸려 OOS 마이너스 기록.
  - 반면 4H Donchian(K4_4H_LO 36/18 vt25)는 OOS CAGR +11.0%, Sharpe 1.05, WkPF 1.56x, MDD -11.2%로
    BTC를 압도하며 높은 위험조정수익을 보임 (유일하게 OOS Sharpe >= 1.0 근접).
  - 단기 1D 양방향(K3_EMA_LS 5/25)은 최근 1년 하락장에서 +38.6% 방어력을 보임.

2차 전략군 (1차와 동일한 사전 등록 기준 AC1~AC6 적용 — 기준 완화 없음):
  M1_CORE_4H  : 코어 3메이저(BTC, ETH, SOL) 집중 4H Donchian + VolTarget(35~45%)
  M2_ASYM_4H  : 비대칭 양방향 — 4H Donchian 롱 + 엄격 하락장 1D/4H 숏(0.6x 비중)
  M3_VB_1D    : 일봉 래리 윌리엄스 변동성 돌파 + 노이즈 필터 + MA 추세 필터 (왕복 20bps 차감)
  M4_CS_REV   : 8메이저 시장중립 단기 평균회귀 롱숏 (단기 급락 2코인 롱 + 단기 급등 2코인 숏)
  M4_CS_MOM   : 8메이저 시장중립 모멘텀 롱숏 (강세 2코인 롱 + 약세 2코인 숏)
  M5_CORE_1D  : 코어 3메이저 1D 추세 + 변동성 돌파 결합
"""
import numpy as np
import pandas as pd

from tools.research_okx_alpha import (
    load_all, run_weights, stats, eval_daily,
    FEE, IS_START, IS_END, OOS_START, SYMS,
)

CORE3 = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]


def m1_core_4h(o4, h4, l4, c4, c1, syms=CORE3, entry_b=36, exit_b=18, vol_t=0.40, use_btc_filter=True, allow_short=False):
    hi_n = h4.shift(1).rolling(entry_b).max()
    lo_n = l4.shift(1).rolling(entry_b).min()
    lo_ex = l4.shift(1).rolling(exit_b).min()
    hi_ex = h4.shift(1).rolling(exit_b).max()
    vol = c4.pct_change().rolling(180).std() * np.sqrt(365 * 6)

    btc_d = c1["BTC-USDT-SWAP"]
    btc_ema = btc_d.ewm(span=50, adjust=False).mean().shift(1)
    btc_ok_l = (btc_d.shift(1) > btc_ema).reindex(c4.index, method="ffill").fillna(False) if use_btc_filter else pd.Series(True, index=c4.index)
    btc_ok_s = (btc_d.shift(1) < btc_ema).reindex(c4.index, method="ffill").fillna(False) if use_btc_filter else pd.Series(True, index=c4.index)

    T4 = pd.DataFrame(0.0, index=c4.index, columns=c4.columns)
    n = len(syms)
    for s in syms:
        arr = np.zeros(len(c4))
        st = 0
        for i in range(max(entry_b, 190), len(c4)):
            px = c4[s].iloc[i]
            v = max(0.20, float(vol[s].iloc[i]))
            w_abs = min(2.0 / n, (vol_t / n) / v)
            if st == 1 and px < lo_ex[s].iloc[i]:
                st = 0
            elif st == -1 and px > hi_ex[s].iloc[i]:
                st = 0
            if st == 0:
                if px > hi_n[s].iloc[i] and btc_ok_l.iloc[i]:
                    st = 1
                elif allow_short and px < lo_n[s].iloc[i] and btc_ok_s.iloc[i]:
                    st = -1
            arr[i] = st * (w_abs if st == 1 else w_abs * 0.6)
        T4[s] = arr

    ret4, ent4, _ = run_weights(T4, o4, c4, fee=FEE)
    return (1.0 + ret4).resample("1D").prod() - 1.0, ent4.resample("1D").sum()


def m2_asym_4h(o4, h4, l4, c4, c1, entry_b=36, exit_b=18, vol_t=0.40, short_scale=0.6):
    # Strict bear regime for shorts: BTC < EMA50 AND EMA15 < EMA50 AND 14d return < -3%
    btc_d = c1["BTC-USDT-SWAP"]
    e15 = btc_d.ewm(span=15, adjust=False).mean().shift(1)
    e50 = btc_d.ewm(span=50, adjust=False).mean().shift(1)
    r14 = (btc_d / btc_d.shift(14) - 1.0).shift(1)
    bear_strict = ((btc_d.shift(1) < e50) & (e15 < e50) & (r14 < -0.03)).reindex(c4.index, method="ffill").fillna(False)
    # Long permitted whenever BTC > EMA50 or 14d return > 0
    bull_ok = ((btc_d.shift(1) > e50) | (r14 > 0.0)).reindex(c4.index, method="ffill").fillna(False)

    hi_n = h4.shift(1).rolling(entry_b).max()
    lo_n = l4.shift(1).rolling(entry_b).min()
    lo_ex = l4.shift(1).rolling(exit_b).min()
    hi_ex = h4.shift(1).rolling(exit_b).max()
    vol = c4.pct_change().rolling(180).std() * np.sqrt(365 * 6)

    T4 = pd.DataFrame(0.0, index=c4.index, columns=c4.columns)
    n = len(CORE3)
    for s in CORE3:
        arr = np.zeros(len(c4))
        st = 0
        for i in range(max(entry_b, 190), len(c4)):
            px = c4[s].iloc[i]
            v = max(0.20, float(vol[s].iloc[i]))
            w_abs = min(2.0 / n, (vol_t / n) / v)
            if st == 1 and px < lo_ex[s].iloc[i]:
                st = 0
            elif st == -1 and px > hi_ex[s].iloc[i]:
                st = 0
            if st == 0:
                if px > hi_n[s].iloc[i] and bull_ok.iloc[i]:
                    st = 1
                elif px < lo_n[s].iloc[i] and bear_strict.iloc[i]:
                    st = -1
            arr[i] = st * (w_abs if st == 1 else w_abs * short_scale)
        T4[s] = arr

    ret4, ent4, _ = run_weights(T4, o4, c4, fee=FEE)
    return (1.0 + ret4).resample("1D").prod() - 1.0, ent4.resample("1D").sum()


def m3_vol_breakout(o, h, l, c, syms=CORE3, k=0.5, use_noise=True, ma_filter=20, vol_t=0.35):
    """Daily Larry Williams Volatility Breakout: no lookahead.
    At day t: target = O[t] + k * (H[t-1] - L[t-1]).
    Entered intraday only if prior day close > MA(ma_filter) and H[t] >= target.
    Exited at close[t] (or next open approximated by close[t] with full 20bps round-trip fee).
    """
    prev_rng = (h - l).shift(1)
    noise = (1.0 - (c - o).abs() / (h - l).replace(0, np.nan)).rolling(20).mean().shift(1).clip(0.2, 0.8)
    k_eff = noise if use_noise else pd.DataFrame(k, index=c.index, columns=c.columns)
    ma = c.rolling(ma_filter).mean().shift(1) if ma_filter > 0 else None
    vol = c.pct_change().rolling(20).std().shift(1) * np.sqrt(365)

    daily_ret = pd.Series(0.0, index=c.index)
    daily_ent = pd.Series(0, index=c.index)
    n = len(syms)
    for s in syms:
        trig = o[s] + k_eff[s] * prev_rng[s]
        cond_trend = True if ma is None else (c[s].shift(1) > ma[s])
        hit = (h[s] >= trig) & (o[s] < trig) & cond_trend & prev_rng[s].notna() & vol[s].notna()
        w = ((vol_t / n) / vol[s].clip(lower=0.20)).clip(upper=1.5 / n)
        trade_ret = (c[s] / trig - 1.0) - 2.0 * FEE  # 20 bps round-trip
        daily_ret += hit.astype(float) * w * trade_ret.fillna(0.0)
        daily_ent += hit.astype(int)
    return daily_ret.fillna(0.0), daily_ent


def m4_cross_sectional(o, c, lb=5, top_n=2, mode="rev", rebal_days=1, vol_t=0.20):
    """Dollar-neutral cross-sectional strategy across 8 majors."""
    r = c / c.shift(lb) - 1.0
    vol = c.pct_change().rolling(30).std() * np.sqrt(365)
    T = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    for i in range(35, len(c)):
        if i % rebal_days != 0:
            continue
        if r.iloc[i].isna().any() or vol.iloc[i].isna().any():
            continue
        row = pd.Series(0.0, index=c.columns)
        ranked = r.iloc[i].sort_values()
        if mode == "rev":
            longs = ranked.index[:top_n]
            shorts = ranked.index[-top_n:]
        else:
            longs = ranked.index[-top_n:]
            shorts = ranked.index[:top_n]
        for s in longs:
            row[s] = min(0.5 / top_n, (vol_t / (2 * top_n)) / max(0.20, float(vol.iloc[i][s])))
        for s in shorts:
            row[s] = -min(0.5 / top_n, (vol_t / (2 * top_n)) / max(0.20, float(vol.iloc[i][s])))
        T.iloc[i] = row
    return T.ffill().fillna(0.0)


def main():
    (o1, h1, l1, c1), (o4, h4, l4, c4) = load_all()
    print(f"데이터: {c1.index[0].date()} ~ {c1.index[-1].date()} (1D {len(c1)}일 / 4H {len(c4)}봉)")
    btc_ret = c1["BTC-USDT-SWAP"].pct_change().fillna(0.0)
    b_oo = stats(btc_ret, OOS_START)
    print(f"[벤치] BTC OOS CAGR {b_oo['cagr']*100:5.1f}% Sh {b_oo['sharpe']:.2f} MDD {b_oo['mdd']*100:6.1f}%\n")

    fams = []

    # M1: Core 3 4H Donchian (Long-only)
    m1_list = []
    for en, ex in ((24, 12), (36, 18), (48, 24), (60, 30)):
        r, e = m1_core_4h(o4, h4, l4, c4, c1, syms=CORE3, entry_b=en, exit_b=ex, vol_t=0.45, use_btc_filter=True)
        m1_list.append(eval_daily(f"M1_CORE3_4H {en}/{ex} vt45", r, e))
    fams.append(("M1_CORE3_4H", m1_list))

    # M1b: Core 3 4H Donchian without BTC EMA filter (asset's own trend only)
    m1b_list = []
    for en, ex in ((24, 12), (36, 18), (48, 24), (60, 30)):
        r, e = m1_core_4h(o4, h4, l4, c4, c1, syms=CORE3, entry_b=en, exit_b=ex, vol_t=0.45, use_btc_filter=False)
        m1b_list.append(eval_daily(f"M1b_4H_NOFLT {en}/{ex}", r, e))
    fams.append(("M1b_4H_NOFLT", m1b_list))

    # M2: Asymmetric 4H Long + Strict Bear Short
    m2_list = []
    for en, ex in ((24, 12), (36, 18), (48, 24), (60, 30)):
        r, e = m2_asym_4h(o4, h4, l4, c4, c1, entry_b=en, exit_b=ex, vol_t=0.45, short_scale=0.6)
        m2_list.append(eval_daily(f"M2_ASYM_4H {en}/{ex}", r, e))
    fams.append(("M2_ASYM_4H", m2_list))

    # M3: Volatility Breakout 1D
    m3_list = []
    for use_n, k in ((True, 0.0), (False, 0.4), (False, 0.5), (False, 0.6)):
        r, e = m3_vol_breakout(o1, h1, l1, c1, syms=CORE3, k=k, use_noise=use_n, ma_filter=15, vol_t=0.40)
        lbl = "noise" if use_n else f"k{k}"
        m3_list.append(eval_daily(f"M3_VB_CORE3 {lbl}", r, e))
    fams.append(("M3_VB_CORE3", m3_list))

    # M4: Dollar-Neutral Reversal & Momentum
    m4_rev, m4_mom = [], []
    for lb, rd in ((1, 1), (3, 1), (5, 3), (7, 7)):
        ret, ent, reb = run_weights(m4_cross_sectional(o1, c1, lb=lb, top_n=2, mode="rev", rebal_days=rd, vol_t=0.30), o1, c1)
        m4_rev.append(eval_daily(f"M4_CS_REV lb{lb} r{rd}", ret, reb))
    for lb, rd in ((7, 3), (14, 7), (21, 7), (30, 7)):
        ret, ent, reb = run_weights(m4_cross_sectional(o1, c1, lb=lb, top_n=2, mode="mom", rebal_days=rd, vol_t=0.30), o1, c1)
        m4_mom.append(eval_daily(f"M4_CS_MOM lb{lb} r{rd}", ret, reb))
    fams += [("M4_CS_REV", m4_rev), ("M4_CS_MOM", m4_mom)]

    # M5: Ensemble (M1b 4H Trend + M3 VB)
    m5_list = []
    r_vb, e_vb = m3_vol_breakout(o1, h1, l1, c1, syms=CORE3, use_noise=True, ma_filter=15, vol_t=0.50)
    for en, ex in ((24, 12), (36, 18), (48, 24)):
        r_4h, e_4h = m1_core_4h(o4, h4, l4, c4, c1, syms=CORE3, entry_b=en, exit_b=ex, vol_t=0.50, use_btc_filter=True)
        r_4h_a = r_4h.reindex(r_vb.index).fillna(0.0)
        e_4h_a = e_4h.reindex(e_vb.index).fillna(0)
        m5_list.append(eval_daily(f"M5_ENS 4H{en}/{ex}+VB", 0.5 * r_4h_a + 0.5 * r_vb, e_4h_a + e_vb))
    fams.append(("M5_ENS", m5_list))

    rows = []
    for fam, items in fams:
        ok = sum(1 for r in items if r["AC1"] and r["AC2"] and r["AC3"])
        for r in items:
            r["AC6"] = ok >= len(items) * 2 / 3
        rows += items

    hdr = f"{'전략':<24}|{'IS CG':>6}{'IS Sh':>6}{'IS DD':>7}|{'OOS CG':>7}{'OOS Sh':>7}{'OOS DD':>7}|{'N':>5}{'WkPF':>6}{'1Y':>7}{'60d':>7}| AC1-6"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        flags = "".join("O" if r[k] else "x" for k in ("AC1", "AC2", "AC3", "AC4", "AC5", "AC6"))
        star = " ★PASS★" if all(r[k] for k in ("AC1", "AC2", "AC3", "AC4", "AC5", "AC6")) else ""
        print(f"{r['name']:<24}|{r['is_cagr']*100:5.1f}%{r['is_sh']:6.2f}{r['is_mdd']*100:6.1f}%|"
              f"{r['oos_cagr']*100:6.1f}%{r['oos_sh']:7.2f}{r['oos_mdd']*100:6.1f}%|"
              f"{r['n_oos']:5d}{r['wk_pf']:5.2f}x{r['y1']*100:6.1f}%{r['d60']*100:6.1f}%| {flags}{star}")


if __name__ == "__main__":
    main()

