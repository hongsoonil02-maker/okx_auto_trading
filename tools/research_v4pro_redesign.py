# -*- coding: utf-8 -*-
"""Exploratory redesign matrix. Development periods are already observed; the
untouched 2024-07~2025-07 holdout is for one final check after selection."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest_v4pro_scenarios as bt


def load_dataset(prefix):
    if prefix == "current":
        raw_1h = {sym: bt.fetch(sym, "1H", 2999) for sym in bt.SYMS}
        raw_15m = {sym: bt.fetch(sym, "15m", 11599) for sym in bt.SYMS}
    else:
        raw_1h = {sym: pd.read_pickle(os.path.join(bt.DATA_DIR, f"{sym}_1H_{prefix}.pkl")) for sym in bt.SYMS}
        raw_15m = {sym: pd.read_pickle(os.path.join(bt.DATA_DIR, f"{sym}_15m_{prefix}.pkl")) for sym in bt.SYMS}
    data_1h = {sym: bt.prep_1h(frame) for sym, frame in raw_1h.items()}
    data_15m_raw = {sym: bt.prep_15m(frame) for sym, frame in raw_15m.items()}
    data_1h = bt.trim_1h_to_15m_coverage(data_1h, data_15m_raw)
    data_15m = bt.align_15m_to_confirmed_1h(data_15m_raw, data_1h["BTC-USDT-SWAP"].ts.values)
    return data_1h, data_15m


BASE = dict(bt.SCEN["S8_현재_10x_3of4"], risk_sizing=True)
VARIANTS = {
    "base": {},
    "no_s1": {"s1_arm": 9.99},
    "no_s1s2": {"s1_arm": 9.99, "s2_arm": 9.99},
    "hold_ma20": {"s1_arm": 9.99, "s2_arm": 9.99, "s3_arm": 9.99, "tp": 9.99},
    "hold_ma20_tp4": {"s1_arm": 9.99, "s2_arm": 9.99, "s3_arm": 9.99, "tp": 0.04},
    "no_s1_trail": {"s1_arm": 9.99, "s3_drop": 0.015},
    "early_trail": {"s1_arm": 9.99, "s2_arm": 9.99, "s3_arm": 0.01, "s3_drop": 0.012, "s3_min": 0.002},
    "early_trail2": {"s1_arm": 9.99, "s2_arm": 9.99, "s3_arm": 0.015, "s3_drop": 0.015, "s3_min": 0.003},
    "wide_trail": {"s1_arm": 9.99, "s2_arm": 9.99, "s3_arm": 0.02, "s3_drop": 0.02, "s3_min": 0.005},
    "no_ma20": {"ma20_exit": False},
    "no_s1_no_ma20": {"s1_arm": 9.99, "ma20_exit": False},
    "no_s1s2_no_ma20": {"s1_arm": 9.99, "s2_arm": 9.99, "ma20_exit": False},
    "tp3": {"tp": 0.03},
    "tp4": {"tp": 0.04},
    "no_tp": {"tp": 9.99},
    "no_s1_tp3": {"s1_arm": 9.99, "tp": 0.03},
    "breadth4": {"breadth": 4},
    "adx22": {"min_adx1h": 22.0, "min_adx15": 22.0},
    "adx25": {"min_adx1h": 25.0, "min_adx15": 25.0},
    "slope30": {"slope3": 0.30},
    "spread50": {"min_spread": 0.50},
    "maxdist25": {"max_dist": 2.5},
    "btcwave6": {"btc_wave_max": 6},
    "symwave4": {"sym_wave_max": 4},
    "long_only": {"allow_short": False},
    "short_only": {"allow_long": False},
    "adx22_breadth4": {"breadth": 4, "min_adx1h": 22.0, "min_adx15": 22.0},
    "sl_tight": {"base_sl_core": 0.020, "base_sl_alt": 0.020, "max_dyn_sl": 0.035, "dyn_mult": 2.5},
    "sl_wide": {"base_sl_core": 0.030, "base_sl_alt": 0.030, "max_dyn_sl": 0.045, "dyn_mult": 3.5},
    "dyn_tight": {"max_dyn_sl": 0.030, "dyn_mult": 2.2},
    "breakeven_only": {"s1_arm": 0.01, "s1_floor": 0.0, "s2_arm": 9.99, "s3_arm": 9.99, "tp": 9.99},
}


def metrics(result):
    tr = result["tr"]
    half_ts = pd.to_datetime(tr["xtime"])
    mid = half_ts.min() + (half_ts.max() - half_ts.min()) / 2
    h1 = tr[half_ts <= mid].pnl.sum()
    h2 = tr[half_ts > mid].pnl.sum()
    wins = tr[tr.pnl > 0].pnl.sum()
    losses = abs(tr[tr.pnl <= 0].pnl.sum())
    pf = wins / losses if losses else float("inf")
    xrp = tr[tr.sym == "XRP-USDT-SWAP"].pnl.sum()
    return {
        "n": len(tr),
        "wr": (tr.pnl > 0).mean() * 100.0 if len(tr) else 0.0,
        "pf": pf,
        "ret": result["ret_pct"],
        "mdd": result["mdd"],
        "h1": h1,
        "h2": h2,
        "xrp": xrp,
    }


def main():
    datasets = {name: load_dataset(name) for name in ("current", "oos")}
    rows = []
    for name, override in VARIANTS.items():
        cfg = dict(BASE, **override)
        row = {"variant": name}
        for dataset_name, (data_1h, data_15m) in datasets.items():
            result = bt.run(data_1h, data_15m, cfg)
            for key, value in metrics(result).items():
                row[f"{dataset_name}_{key}"] = value
        rows.append(row)
        print(
            f"{name:20s} | CUR n={row['current_n']:3d} WR={row['current_wr']:4.1f} "
            f"PF={row['current_pf']:4.2f} ret={row['current_ret']:+6.1f}% MDD={row['current_mdd']:5.1f}% "
            f"h1={row['current_h1']:+5.0f} h2={row['current_h2']:+5.0f} | "
            f"OOS n={row['oos_n']:3d} WR={row['oos_wr']:4.1f} PF={row['oos_pf']:4.2f} "
            f"ret={row['oos_ret']:+6.1f}% MDD={row['oos_mdd']:5.1f}% "
            f"h1={row['oos_h1']:+6.0f} h2={row['oos_h2']:+6.0f}",
            flush=True,
        )
    pd.to_pickle(rows, os.path.join(bt.DATA_DIR, "v4pro_redesign_matrix.pkl"))


if __name__ == "__main__":
    main()
