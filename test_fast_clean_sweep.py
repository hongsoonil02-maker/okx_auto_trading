# -*- coding: utf-8 -*-
"""
test_fast_clean_sweep.py — High-Speed Vectorized 1H Backtest
Compares:
1) Close-based Lookahead (The original flawed method)
2) Open-based Clean (The lookahead-free reality)
Runs in < 5 seconds via pre-indexed date maps.
"""

import os
import glob
import pickle
import time
import numpy as np
import pandas as pd

CACHE = "state/bt_cache"
daily_path = sorted(glob.glob(os.path.join(CACHE, "alpaca_daily_*.pkl")))[-1]
with open(daily_path, "rb") as f:
    daily = pickle.load(f)

t0 = time.time()

# 1. Pre-build daily date map: date -> list of (sym, open, close, prev_close)
# This takes only ~2 seconds for all 8,621 symbols!
date_map = {}
for sym, df in daily.items():
    if len(df) < 5:
        continue
    times = df["time"].dt.date.values
    opens = df["open"].values
    closes = df["close"].values
    n = len(df)
    for i in range(1, n):
        d = times[i]
        pc = closes[i - 1]
        op = opens[i]
        cl = closes[i]
        if pc > 0 and not np.isnan(pc):
            if d not in date_map:
                date_map[d] = []
            date_map[d].append((sym, op, cl, pc))

print(f"⚡ Pre-indexed {len(date_map)} trading days in {time.time()-t0:.2f}s", flush=True)

start_date = pd.to_datetime("2026-04-13").date()
calendar = sorted([d for d in date_map.keys() if d >= start_date])

# 2. Run simulation for both modes
for mode in ["close_lookahead", "open_clean"]:
    t_mode = time.time()
    trades = []
    
    for day in calendar:
        entries = date_map.get(day, [])
        movers = []
        for sym, op, cl, pc in entries:
            # Lookahead uses cl (daily close at 16:00), Clean uses op (market open at 09:30)
            px = cl if mode == "close_lookahead" else op
            if 2.0 <= px <= 50.0:
                pct = (px / pc - 1) * 100
                if pct >= 3.0:
                    movers.append((pct, sym))
        
        movers.sort(reverse=True)
        watch = [s for _, s in movers[:5]]
        if not watch:
            continue
            
        hcache = os.path.join(CACHE, f"alpaca_1Hour_{day}.pkl")
        if not os.path.exists(hcache):
            continue
        with open(hcache, "rb") as f:
            intraday = pickle.load(f)
            
        for sym in watch:
            df = intraday.get(sym)
            if df is None or len(df) < 16:
                continue
            
            day_idx = df.index[df["time"].dt.date == day].tolist()
            for j in day_idx:
                if j < 15:
                    continue
                prev, curr = df.loc[j - 1], df.loc[j]
                if not (prev["close"] < prev["open"]):
                    continue
                if prev["close"] < prev["ema9"] * 0.99 if "ema9" in df else False:
                    continue
                
                trig, sl = prev["high"], prev["low"]
                risk = trig - sl
                if risk <= 0:
                    continue
                rpct = risk / trig
                if rpct > 0.05 or rpct < 0.01:
                    continue
                if curr["high"] < trig:
                    continue
                
                tp_price = trig + (risk * 3.0)
                fut = [df.loc[k] for k in day_idx if k > j]
                exit_p = None
                for b in fut:
                    if b["low"] <= sl:
                        exit_p = sl
                        break
                    if b["high"] >= tp_price:
                        exit_p = tp_price
                        break
                if exit_p is None:
                    exit_p = df.loc[day_idx[-1], "close"]
                
                ret = exit_p / trig - 1
                trades.append(ret)
                break
                
    arr = np.array(trades)
    n = len(arr)
    wr = (arr > 0).mean() * 100 if n else 0
    gw = arr[arr > 0].sum()
    gl = -arr[arr <= 0].sum()
    pf = gw / gl if gl > 0 else 0
    avg_ret = arr.mean() * 100 if n else 0
    print(f"📊 [{mode:16s}] 완료 ({time.time()-t_mode:.2f}s) | 거래수: {n:3d}건 | 승률: {wr:5.1f}% | PF: {pf:5.2f} | 평균수익률: {avg_ret:+.2f}%", flush=True)
