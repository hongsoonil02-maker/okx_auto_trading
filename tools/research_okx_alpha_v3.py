#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/research_okx_alpha_v3.py
=============================================================================
OKX 3차 정밀 알파 탐색:
  1) M4_CS_REV(lb1 r1)가 샤프 -3.0 나온 이유 분해 -> 초단기 크립토 모멘텀(lb1~3)이
     0bps / Maker 2bps / Taker 10bps 에서 실제로 알파가 있는지?
  2) 1차 최우수 베이스라인 K4_4H_LO 36/18 (8메이저, IS Sh 0.95 / OOS Sh 1.05, DD -11.2%)
     에 엄격 하락장 숏(bear_strict) 또는 다중 타임프레임(24/12 + 36/18 + 60/30) 결합 검증
  3) BTC 단독 1D/4H 추세추종 + 엄격 하락장 숏 + 1~3일 알트 모멘텀 결합 가능성
  4) 실제 OKX Funding Rate 히스토리 수집 가능성 체크 및 Carry/Extreme 팩터 검증
"""
import numpy as np
import pandas as pd

from tools.research_okx_alpha import (
    load_all, run_weights, stats, eval_daily,
    FEE, IS_START, IS_END, OOS_START, SYMS,
)
from tools.research_okx_alpha_v2 import m4_cross_sectional


def donchian_8_4h(o4, h4, l4, c4, c1, entry_b=36, exit_b=18, vol_t=0.35, asym_short=0.0):
    hi_n = h4.shift(1).rolling(entry_b).max()
    lo_n = l4.shift(1).rolling(entry_b).min()
    lo_ex = l4.shift(1).rolling(exit_b).min()
    hi_ex = h4.shift(1).rolling(exit_b).max()
    vol = c4.pct_change().rolling(180).std() * np.sqrt(365 * 6)

    btc_d = c1["BTC-USDT-SWAP"]
    e15 = btc_d.ewm(span=15, adjust=False).mean().shift(1)
    e50 = btc_d.ewm(span=50, adjust=False).mean().shift(1)
    r14 = (btc_d / btc_d.shift(14) - 1.0).shift(1)

    btc_ok_l = (btc_d.shift(1) > e50).reindex(c4.index, method="ffill").fillna(False)
    bear_strict = ((btc_d.shift(1) < e50) & (e15 < e50) & (r14 < -0.03)).reindex(c4.index, method="ffill").fillna(False)

    T4 = pd.DataFrame(0.0, index=c4.index, columns=c4.columns)
    n = len(SYMS)
    for s in SYMS:
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
                elif asym_short > 0 and px < lo_n[s].iloc[i] and bear_strict.iloc[i]:
                    st = -1
            arr[i] = st * (w_abs if st == 1 else w_abs * asym_short)
        T4[s] = arr
    ret4, ent4, _ = run_weights(T4, o4, c4, fee=FEE)
    return (1.0 + ret4).resample("1D").prod() - 1.0, ent4.resample("1D").sum()


def multi_tf_4h(o4, h4, l4, c4, c1, vol_t=0.40, asym_short=0.5):
    """Equal blend of 3 Donchian speeds (24/12, 36/18, 48/24) across 8 majors."""
    r1, e1 = donchian_8_4h(o4, h4, l4, c4, c1, 24, 12, vol_t=vol_t, asym_short=asym_short)
    r2, e2 = donchian_8_4h(o4, h4, l4, c4, c1, 36, 18, vol_t=vol_t, asym_short=asym_short)
    r3, e3 = donchian_8_4h(o4, h4, l4, c4, c1, 48, 24, vol_t=vol_t, asym_short=asym_short)
    return (r1 + r2 + r3) / 3.0, e2


def main():
    (o1, h1, l1, c1), (o4, h4, l4, c4) = load_all()
    print("=== 1) Short-Horizon Cross-Sectional Momentum (lb1~5) Fee Decomposition ===")
    for lb, rd in ((1, 1), (2, 1), (3, 1), (3, 3), (5, 3)):
        T = m4_cross_sectional(o1, c1, lb=lb, top_n=2, mode="mom", rebal_days=rd, vol_t=0.30)
        for lbl, fee in (("0bps", 0.0), ("Maker2bp", 0.0002), ("Taker10bp", 0.0010)):
            ret, _, reb = run_weights(T, o1, c1, fee=fee)
            i, oo = stats(ret, IS_START, IS_END), stats(ret, OOS_START)
            print(f"CS_MOM lb{lb} r{rd} [{lbl:<9}] | IS {i['cagr']*100:+5.1f}% Sh {i['sharpe']:+5.2f} | OOS {oo['cagr']*100:+5.1f}% Sh {oo['sharpe']:+5.2f} DD {oo['mdd']*100:6.1f}%")

    print("\n=== 2) 8-Majors 4H Donchian (Long-Only vs Asymmetric Bear Short + Multi-Speed) ===")
    fams = []
    f_lo, f_as = [], []
    for en, ex in ((24, 12), (36, 18), (48, 24), (60, 30)):
        r_lo, e_lo = donchian_8_4h(o4, h4, l4, c4, c1, en, ex, vol_t=0.40, asym_short=0.0)
        f_lo.append(eval_daily(f"D8_4H_LO {en}/{ex} vt40", r_lo, e_lo))
        r_as, e_as = donchian_8_4h(o4, h4, l4, c4, c1, en, ex, vol_t=0.40, asym_short=0.6)
        f_as.append(eval_daily(f"D8_4H_ASYM {en}/{ex} vt40", r_as, e_as))
    r_m_lo, e_m_lo = multi_tf_4h(o4, h4, l4, c4, c1, vol_t=0.45, asym_short=0.0)
    r_m_as, e_m_as = multi_tf_4h(o4, h4, l4, c4, c1, vol_t=0.45, asym_short=0.6)
    f_lo.append(eval_daily("D8_MULTI_LO vt45", r_m_lo, e_m_lo))
    f_as.append(eval_daily("D8_MULTI_ASYM vt45", r_m_as, e_m_as))
    fams += [("D8_4H_LO", f_lo), ("D8_4H_ASYM", f_as)]

    # 3) Test 4H Reversal / Pullback inside macro uptrend (buy 4H RSI2/drop when BTC > EMA50)
    print("\n=== 3) Checking 4H & 1D Mean-Reversion Pullback in Uptrend (RSI2 < th, exit MA5) ===")
    f_mr = []
    for syms_name, syms in (("CORE3", ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]), ("MAJ8", SYMS)):
        for th in (10.0, 20.0):
            # 1D RSI(2) inside uptrend (close > MA100)
            T = pd.DataFrame(0.0, index=c1.index, columns=c1.columns)
            for s in syms:
                px = c1[s]
                d = px.diff()
                up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
                dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
                r2 = 100.0 - 100.0 / (1.0 + up / dn.replace(0, np.nan))
                ma100 = px.rolling(100).mean()
                ma5 = px.rolling(5).mean()
                vol = px.pct_change().rolling(30).std() * np.sqrt(365)
                pos = np.zeros(len(px)); held = False
                for i in range(105, len(px)):
                    if not held and px.iloc[i] > ma100.iloc[i] and r2.iloc[i] < th:
                        held = True
                    elif held and px.iloc[i] > ma5.iloc[i]:
                        held = False
                    pos[i] = min(2.0 / len(syms), (0.40 / len(syms)) / max(0.20, float(vol.iloc[i]))) if held else 0.0
                T[s] = pos
            ret, ent, _ = run_weights(T, o1, c1, fee=FEE)
            f_mr.append(eval_daily(f"MR_1D {syms_name} th{int(th)}", ret, ent))
    fams.append(("MR_1D", f_mr))

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

