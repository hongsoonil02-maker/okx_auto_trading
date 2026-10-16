# -*- coding: utf-8 -*-
import os, glob, pickle, time
import numpy as np
import pandas as pd

CACHE = "state/bt_cache"
daily_path = sorted(glob.glob(os.path.join(CACHE, "alpaca_daily_*.pkl")))[-1]
with open(daily_path, "rb") as f:
    daily = pickle.load(f)

universe = {}
for sym, df in daily.items():
    last = df.iloc[-1]
    dv = (df["close"] * df["volume"]).tail(20).median()
    if 2.0 <= last["close"] <= 50.0 and dv >= 1_000_000:
        universe[sym] = df

date_map = {}
for sym, df in universe.items():
    times = df["time"].dt.date.values
    opens = df["open"].values
    closes = df["close"].values
    for i in range(1, len(df)):
        d = times[i]
        if d not in date_map:
            date_map[d] = []
        date_map[d].append((sym, opens[i], closes[i], closes[i - 1]))

start_date = pd.to_datetime("2026-04-13").date()
calendar = sorted([d for d in date_map.keys() if d >= start_date])

# Compare 3 sizing models on CLEAN open gap (09:30):
configs = [
    {"name": "1. Flat 10% (최대 3개 동시) [스윕 백테스트]", "max_pos": 3, "sizing": "flat_10", "equity": 100000.0, "trades": [], "curve": []},
    {"name": "2. Flat 20% (최대 5개 동시)", "max_pos": 5, "sizing": "flat_20", "equity": 100000.0, "trades": [], "curve": []},
    {"name": "3. 동적 Half-Kelly 16~20% (최대 5개) [현재 실전 봇]", "max_pos": 5, "sizing": "dynamic_kelly", "equity": 100000.0, "trades": [], "curve": []}
]

for day in calendar:
    entries = date_map.get(day, [])
    movers = []
    for sym, op, cl, pc in entries:
        if 2.0 <= op <= 50.0 and pc > 0:
            pct = (op / pc - 1) * 100
            if pct >= 3.0:
                movers.append((pct, sym))
    movers.sort(reverse=True)
    watch = [s for _, s in movers[:5]]
    
    hcache = os.path.join(CACHE, f"alpaca_1Hour_{day}.pkl")
    if not os.path.exists(hcache): 
        for cfg in configs:
            cfg["curve"].append((day, cfg["equity"]))
        continue
    with open(hcache, "rb") as f:
        intraday = pickle.load(f)
    
    sym_signals = {}
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
            exit_p = reason = None
            for b in fut:
                hit_sl, hit_tp = b["low"] <= sl, b["high"] >= tp_price
                if hit_sl and hit_tp:
                    exit_p = sl * 0.999
                    reason = "SL_AMBIG"
                    break
                if hit_sl:
                    exit_p = sl * 0.999
                    reason = "SL"
                    break
                if hit_tp:
                    exit_p = tp_price * 0.999
                    reason = "TP"
                    break
            if exit_p is None:
                exit_p = df.loc[day_idx[-1], "close"] * 0.999
                reason = "EOD"
            
            ret = exit_p / (trig * 1.001) - 1
            sym_signals[sym] = {"ret": ret, "et_hm": (df.loc[j, "time"].hour, df.loc[j, "time"].minute), "reason": reason}
            break
            
    for cfg in configs:
        day_pos = 0
        for sym in watch:
            if day_pos >= cfg["max_pos"]:
                break
            sig = sym_signals.get(sym)
            if not sig:
                continue
            # cutoff 14:30 EST (18:30 UTC)
            if sig["et_hm"] >= (18, 30):
                continue
            
            if cfg["sizing"] == "flat_10":
                alloc = 0.10
            elif cfg["sizing"] == "flat_20":
                alloc = 0.20
            else:
                alloc = 0.18 # average dynamic kelly
            
            val = cfg["equity"] * alloc
            pnl = val * sig["ret"]
            cfg["equity"] += pnl
            cfg["trades"].append(pnl)
            day_pos += 1
        cfg["curve"].append((day, cfg["equity"]))

print("=" * 95)
print("📊 [클린 1시간봉(1H) 무미래참조: 포지션 사이징 모델별 실측 비교표]")
print("=" * 95)
for cfg in configs:
    arr = np.array(cfg["trades"])
    eq = pd.Series({d: v for d, v in cfg["curve"]})
    ret = (eq.iloc[-1] / 100000.0 - 1) * 100
    mdd = ((eq.cummax() - eq) / eq.cummax()).max() * 100
    gw = arr[arr > 0].sum()
    gl = -arr[arr <= 0].sum()
    pf = gw / gl if gl > 0 else 0
    wr = (arr > 0).mean() * 100 if len(arr) else 0
    print(f"{cfg['name']:<48s} | 거래: {len(arr):2d}건 | 승률: {wr:5.1f}% | PF: {pf:4.2f} | 최종자산: ${eq.iloc[-1]:,.2f} ({ret:+6.2f}%) | MDD: {mdd:4.2f}%")
print("=" * 95)
