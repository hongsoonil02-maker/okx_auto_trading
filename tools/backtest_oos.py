# -*- coding: utf-8 -*-
"""Out-of-sample / untouched-holdout checks for pre-registered S8 variants."""
import argparse
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest_v4pro_scenarios as bt
from backtest_v4pro_scenarios import trim_1h_to_15m_coverage

OOS_START = "2025-07-08"
OOS_END = "2026-07-08"


def ms(text):
    return int(datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch_older(inst, bar, start_ms, end_ms, tag):
    path = os.path.join(bt.DATA_DIR, f"{inst}_{bar}_{tag}.pkl")
    if os.path.exists(path):
        df = pd.read_pickle(path)
        if len(df) and df.ts.iloc[0] <= start_ms + 86400000 and df.ts.iloc[-1] >= end_ms - 86400000:
            return df
    out, after = [], str(end_ms)
    print(
        f"fetch {tag} {inst} {bar} "
        f"{pd.to_datetime(start_ms, unit='ms').date()}~{pd.to_datetime(end_ms, unit='ms').date()}"
    )
    while True:
        params = {"instId": inst, "bar": bar, "limit": "100", "after": after}
        try:
            data = requests.get(
                "https://www.okx.com/api/v5/market/history-candles",
                params=params, timeout=10,
            ).json().get("data", [])
        except Exception as exc:
            print("  fetch error", exc)
            break
        if not data:
            break
        out.extend(data)
        oldest = min(int(row[0]) for row in data)
        after = str(oldest)
        if oldest <= start_ms:
            break
        time.sleep(0.12)
    if not out:
        raise RuntimeError(f"no OOS data {inst} {bar}")
    out = sorted(out, key=lambda row: int(row[0]))
    df = pd.DataFrame(out, columns=["ts", "open", "high", "low", "close", "vol", "vc", "vcq", "confirm"])
    df = df[df["confirm"] == "1"].reset_index(drop=True)
    df["ts"] = df["ts"].astype("int64")
    for col in ("open", "high", "low", "close", "vol"):
        df[col] = df[col].astype(float)
    df = df[(df.ts >= start_ms) & (df.ts < end_ms)].reset_index(drop=True)
    df.to_pickle(path)
    print(
        f"  cached {len(df)}: {pd.to_datetime(df.ts.iloc[0], unit='ms')} ~ "
        f"{pd.to_datetime(df.ts.iloc[-1], unit='ms')}"
    )
    return df


def print_result(name, result):
    print(
        f"[{name}] {result['trades']}건 승률 {result['winrate']:.1f}% PF {result['pf']:.2f} "
        f"net {result['net']:+.0f}U ({result['ret_pct']:+.1f}%) MDD {result['mdd']:.1f}% "
        f"기대 {result['expect']:+.1f}U TP {result['tp_hits']}건 청산 {result['liq']}건"
    )
    print(f"   사유: {result['reasons']}")
    print(f"   심볼: {result['by_sym']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=OOS_START)
    parser.add_argument("--end", default=OOS_END)
    parser.add_argument("--tag", default="oos")
    parser.add_argument("--preset", choices=("oos", "holdout", "final"), default="oos")
    args = parser.parse_args()
    start_ms, end_ms = ms(args.start), ms(args.end)
    raw_1h = {sym: fetch_older(sym, "1H", start_ms, end_ms, args.tag) for sym in bt.SYMS}
    raw_15m = {sym: fetch_older(sym, "15m", start_ms, end_ms, args.tag) for sym in bt.SYMS}
    data_1h = {sym: bt.prep_1h(frame) for sym, frame in raw_1h.items()}
    data_15m_raw = {sym: bt.prep_15m(frame) for sym, frame in raw_15m.items()}
    data_1h = bt.trim_1h_to_15m_coverage(data_1h, data_15m_raw)
    data_15m = bt.align_15m_to_confirmed_1h(
        data_15m_raw,
        data_1h["BTC-USDT-SWAP"].ts.values,
    )
    span0 = pd.to_datetime(data_1h["BTC-USDT-SWAP"].ts.iloc[60], unit="ms")
    span1 = pd.to_datetime(data_1h["BTC-USDT-SWAP"].ts.iloc[-1], unit="ms")
    print(f"\n{args.preset} 구간: {span0} ~ {span1} ({(span1-span0).total_seconds()/86400:.1f}일)\n")
    candidate = dict(
        bt.SCEN["S8_현재_10x_3of4"],
        risk_sizing=True,
        s1_arm=9.99, s2_arm=9.99, s3_arm=9.99, tp=9.99,
        breadth=4, min_adx1h=22.0, min_adx15=22.0, min_spread=0.50,
    )
    candidate_daily = dict(candidate, daily_filter=True, daily_slope_min=0.0)
    scenarios = {
        "oos": [
            ("S8_현재_10x_3of4", bt.SCEN["S8_현재_10x_3of4"]),
            ("R1_S8_리스크0.5%", bt.SCEN["R1_S8_리스크0.5%"]),
            ("R2_S8_리스크+일손실2%", bt.SCEN["R2_S8_리스크+일손실2%"]),
        ],
        "holdout": [
            ("S8_현재_10x_3of4", bt.SCEN["S8_현재_10x_3of4"]),
            ("R1_S8_리스크0.5%", bt.SCEN["R1_S8_리스크0.5%"]),
            ("H_MA20_ADX22_SPREAD50", candidate),
        ],
        "final": [
            ("H_MA20_ADX22_SPREAD50_DAILY", candidate_daily),
        ],
    }
    results = []
    for name, cfg in scenarios[args.preset]:
        result = bt.run(data_1h, data_15m, cfg)
        print_result(name, result)
        result["name"] = name
        results.append(result)
    pd.to_pickle(results, os.path.join(bt.DATA_DIR, f"v4pro_{args.tag}_results.pkl"))


if __name__ == "__main__":
    main()
