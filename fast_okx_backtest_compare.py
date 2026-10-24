# -*- coding: utf-8 -*-
"""
fast_okx_backtest_compare.py — Compare Timeframes (15m, 1h, 4h) with Ross Cameron Pullback Strategy on OKX
"""

import os
import sys
import math
import asyncio
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt_async
from dotenv import load_dotenv

CURRENT_DIR = "/home/hongsoonil02/quant_system"
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

SYMBOLS = ["SOL/USDT:USDT", "SUI/USDT:USDT", "DOGE/USDT:USDT", "ETH/USDT:USDT", "BTC/USDT:USDT"]
INITIAL_CAPITAL = 960.0
MARGIN_PCT = 0.15
LEVERAGE = 3
FEE_PCT = 0.0005     # 0.05% taker
SLIPPAGE_PCT = 0.0005

CONFIGS = {
    "15m": {"min_surge": 0.015, "max_sl": 0.035, "min_sl": 0.005, "rr": 2.0, "timeout": 32, "bars": 1000},
    "1h":  {"min_surge": 0.025, "max_sl": 0.055, "min_sl": 0.010, "rr": 2.0, "timeout": 24, "bars": 500},
    "4h":  {"min_surge": 0.035, "max_sl": 0.080, "min_sl": 0.015, "rr": 2.5, "timeout": 18, "bars": 300},
}

async def fetch_bars(ex, symbol, tf, limit):
    try:
        bars = await ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        if not bars or len(bars) < 30:
            return None
        df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        tp = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, 1)
        df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
        return df
    except Exception as e:
        return None

def backtest_tf(data_by_symbol, tf):
    cfg = CONFIGS[tf]
    capital = INITIAL_CAPITAL
    trades = []
    
    for symbol, df in data_by_symbol.items():
        if df is None or len(df) < 25:
            continue
        o = df["open"].values
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        v = df["volume"].values
        ema9 = df["ema9"].values
        vwap = df["vwap"].values
        
        in_pos = False
        entry_p = 0.0
        sl_p = 0.0
        tp_p = 0.0
        pos_usd = 0.0
        entry_idx = 0
        
        for i in range(15, len(df)):
            curr_o, curr_h, curr_l, curr_c, curr_v = o[i], h[i], l[i], c[i], v[i]
            prev_o, prev_h, prev_l, prev_c, prev_v = o[i-1], h[i-1], l[i-1], c[i-1], v[i-1]
            
            if in_pos:
                bars_held = i - entry_idx
                # 1. Stop loss hit
                if curr_l <= sl_p:
                    loss_pct = (sl_p / entry_p) - 1.0 - (FEE_PCT * 2)
                    pnl = pos_usd * loss_pct
                    capital += pnl
                    trades.append({"symbol": symbol, "pnl": pnl, "win": False, "ret": loss_pct, "reason": "SL"})
                    in_pos = False
                # 2. Take profit hit
                elif curr_h >= tp_p:
                    win_pct = (tp_p / entry_p) - 1.0 - (FEE_PCT * 2)
                    pnl = pos_usd * win_pct
                    capital += pnl
                    trades.append({"symbol": symbol, "pnl": pnl, "win": True, "ret": win_pct, "reason": "TP"})
                    in_pos = False
                # 3. Timeout exit
                elif bars_held >= cfg["timeout"]:
                    exit_pct = (curr_c / entry_p) - 1.0 - (FEE_PCT * 2)
                    pnl = pos_usd * exit_pct
                    capital += pnl
                    trades.append({"symbol": symbol, "pnl": pnl, "win": exit_pct > 0, "ret": exit_pct, "reason": "Timeout"})
                    in_pos = False
                    
            if not in_pos:
                prior_h = np.max(h[max(0, i-6):i-1])
                prior_l = np.min(l[max(0, i-6):i-1])
                prior_v_mean = np.mean(v[max(0, i-6):i-1])
                
                # Ross Cameron First Pullback Condition
                if prev_c < prev_o and prev_c >= ema9[i-1] * 0.995 and prev_l >= vwap[i-1] * 0.990:
                    surge = prior_h - prior_l
                    if surge / prior_l >= cfg["min_surge"]:
                        retrace = (prior_h - prev_l) / surge
                        if retrace <= 0.55 and prev_v <= prior_v_mean * 1.30:
                            if curr_h >= prev_h: # Breakout trigger
                                risk = prev_h - prev_l
                                risk_pct = risk / prev_h
                                if cfg["min_sl"] <= risk_pct <= cfg["max_sl"]:
                                    in_pos = True
                                    entry_p = prev_h * (1.0 + SLIPPAGE_PCT)
                                    sl_p = prev_l
                                    tp_p = prev_h + (risk * cfg["rr"])
                                    pos_usd = (capital * MARGIN_PCT) * LEVERAGE
                                    entry_idx = i

    df_trades = pd.DataFrame(trades)
    if len(df_trades) == 0:
        return {"tf": tf, "trades": 0, "win_rate": 0, "pf": 0, "pnl": 0, "ret": 0, "days": 0}
        
    wins = df_trades[df_trades["win"] == True]
    losses = df_trades[df_trades["win"] == False]
    gross_win = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 1e-6
    pf = gross_win / gross_loss
    
    return {
        "tf": tf,
        "trades": len(df_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(df_trades) * 100,
        "pf": pf,
        "pnl": df_trades["pnl"].sum(),
        "ret": (df_trades["pnl"].sum() / INITIAL_CAPITAL) * 100,
        "final_equity": capital
    }

async def main():
    ex = ccxt_async.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    await ex.load_markets()
    
    results = []
    for tf, cfg in CONFIGS.items():
        data_by_symbol = {}
        for s in SYMBOLS:
            df = await fetch_bars(ex, s, tf, cfg["bars"])
            data_by_symbol[s] = df
        res = backtest_tf(data_by_symbol, tf)
        results.append(res)
        
    await ex.close()
    
    print("BACKTEST_RESULTS_READY")
    for r in results:
        print(f"RESULT|{r['tf']}|{r['trades']}|{r['win_rate']:.1f}|{r['pf']:.2f}|{r['pnl']:+.2f}|{r['ret']:+.2f}%")

if __name__ == "__main__":
    asyncio.run(main())
