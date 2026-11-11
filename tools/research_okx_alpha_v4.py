#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/research_okx_alpha_v4.py
Final check: 4H Donchian (8 majors) + 1D Mean-Reversion Pullback blend, plus 4H Supertrend.
Same strict AC1~AC6 rules (no threshold loosening).
"""
import numpy as np
import pandas as pd
from tools.research_okx_alpha import load_all, run_weights, eval_daily, FEE, SYMS, OOS_START, IS_START, IS_END, stats
from tools.research_okx_alpha_v3 import donchian_8_4h, multi_tf_4h


def mr_1d(o1, c1, syms=("BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"), th=10.0, vol_t=0.40):
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
            pos[i] = min(2.0 / len(syms), (vol_t / len(syms)) / max(0.20, float(vol.iloc[i]))) if held else 0.0
        T[s] = pos
    return run_weights(T, o1, c1, fee=FEE)


def main():
    (o1, h1, l1, c1), (o4, h4, l4, c4) = load_all()
    r_mr, e_mr, _ = mr_1d(o1, c1, syms=["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"], th=10.0, vol_t=0.50)
    r_mr_d = r_mr.copy()
    r_mr_d.index = r_mr_d.index.normalize()
    e_mr_d = e_mr.copy()
    e_mr_d.index = e_mr_d.index.normalize()

    items = []
    for en, ex in ((24, 12), (36, 18), (48, 24)):
        r_tr, e_tr = donchian_8_4h(o4, h4, l4, c4, c1, en, ex, vol_t=0.50, asym_short=0.0)
        r_tr_a = r_tr.reindex(r_mr_d.index).fillna(0.0)
        e_tr_a = e_tr.reindex(e_mr_d.index).fillna(0)
        blend = 0.65 * r_tr_a + 0.35 * r_mr_d
        items.append(eval_daily(f"BLEND65/35 4H{en}/{ex}+MR", blend, e_tr_a + e_mr_d))

    r_mt, e_mt = multi_tf_4h(o4, h4, l4, c4, c1, vol_t=0.55, asym_short=0.0)
    r_mt_a = r_mt.reindex(r_mr_d.index).fillna(0.0)
    e_mt_a = e_mt.reindex(e_mr_d.index).fillna(0)
    items.append(eval_daily("BLEND65/35 MULTI+MR", 0.65 * r_mt_a + 0.35 * r_mr_d, e_mt_a + e_mr_d))

    ok = sum(1 for r in items if r["AC1"] and r["AC2"] and r["AC3"])
    for r in items:
        r["AC6"] = ok >= len(items) * 2 / 3

    hdr = f"{'전략':<24}|{'IS CG':>6}{'IS Sh':>6}{'IS DD':>7}|{'OOS CG':>7}{'OOS Sh':>7}{'OOS DD':>7}|{'N':>5}{'WkPF':>6}{'1Y':>7}{'60d':>7}| AC1-6"
    print(hdr); print("-" * len(hdr))
    for r in items:
        flags = "".join("O" if r[k] else "x" for k in ("AC1", "AC2", "AC3", "AC4", "AC5", "AC6"))
        star = " ★PASS★" if all(r[k] for k in ("AC1", "AC2", "AC3", "AC4", "AC5", "AC6")) else ""
        print(f"{r['name']:<24}|{r['is_cagr']*100:5.1f}%{r['is_sh']:6.2f}{r['is_mdd']*100:6.1f}%|"
              f"{r['oos_cagr']*100:6.1f}%{r['oos_sh']:7.2f}{r['oos_mdd']*100:6.1f}%|"
              f"{r['n_oos']:5d}{r['wk_pf']:5.2f}x{r['y1']*100:6.1f}%{r['d60']*100:6.1f}%| {flags}{star}")

    # Yearly breakdown for best candidate
    best = max(items, key=lambda x: x["oos_sh"])
    yr = best["ret"].loc["2022-03-01":].groupby(best["ret"].loc["2022-03-01":].index.year).apply(lambda x: (1 + x).prod() - 1)
    print(f"\n[{best['name']}] 연도별 수익률:", {int(k): f"{v*100:+.1f}%" for k, v in yr.items()})


if __name__ == "__main__":
    main()

