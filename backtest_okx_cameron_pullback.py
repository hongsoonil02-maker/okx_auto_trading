# -*- coding: utf-8 -*-
"""
backtest_okx_cameron_pullback.py — Rigorous Multi-Day Backtest for OKX Crypto First Pullback Strategy
Simulates across major liquid volatile swaps (SOL, SUI, DOGE, ETH, BTC)
Includes realistic taker fee (0.05%), slippage (0.05%), and 2x leverage.
Tests Take-Profit Multipliers: 1:2.0, 1:2.5, 1:3.0
"""

import os
import sys
import math
import asyncio
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt_async
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

SYMBOLS = ["SOL/USDT:USDT", "SUI/USDT:USDT", "DOGE/USDT:USDT", "ETH/USDT:USDT", "BTC/USDT:USDT"]
INITIAL_CAPITAL = 960.0        # Matching real account equity
ALLOCATION_PCT = 0.15          # 15% (~$144)
LEVERAGE = 2                   # 2x leverage
FEE_PCT = 0.0005               # 0.05% taker fee
SLIPPAGE_PCT = 0.0005          # 0.05% crypto slippage
MIN_SURGE_PCT = 0.008          # 0.8% impulse surge
MAX_SL_PCT = 0.035             # 3.5%
MIN_SL_PCT = 0.005             # 0.5%

async def fetch_historical_bars(symbols, days=7):
    ex = ccxt_async.okx({
        "apiKey": os.getenv("OKX_API_KEY"),
        "secret": os.getenv("OKX_SECRET"),
        "password": os.getenv("OKX_PASSPHRASE"),
        "enableRateLimit": True,
        "options": {"defaultType": "swap"}
    })
    await ex.load_markets()
    bars_dict = {}

    start_ts = ex.parse8601("2026-10-03T00:00:00Z")
    
    for s in symbols:
        try:
            all_bars = []
            curr_ts = start_ts
            # Fetch in 300-bar chunks
            for _ in range(6):
                bars = await ex.fetch_ohlcv(s, timeframe="5m", since=curr_ts, limit=300)
                if not bars:
                    break
                all_bars.extend(bars)
                curr_ts = bars[-1][0] + (5 * 60 * 1000)
                if len(bars) < 300:
                    break
            
            if len(all_bars) >= 50:
                df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
                # Deduplicate
                df.drop_duplicates(subset=["timestamp"], inplace=True)
                df.sort_values("timestamp", inplace=True)
                df.reset_index(drop=True, inplace=True)
                
                tp = (df["high"] + df["low"] + df["close"]) / 3
                df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, 1)
                df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
                bars_dict[s] = df
                print(f"✅ Loaded {s}: {len(df)} 5m bars ({df['time'].iloc[0]} ~ {df['time'].iloc[-1]})")
        except Exception as e:
            print(f"❌ Failed to fetch {s}: {e}")

    await ex.close()
    return bars_dict

def simulate_okx_cameron(bars_data, rr_ratio: float):
    capital = INITIAL_CAPITAL
    trades = []

    for sym, df in bars_data.items():
        in_pos = False
        entry_idx = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        pos_size_usd = 0.0

        for i in range(10, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            t = curr["time"]

            # Manage open trade
            if in_pos:
                # 1. Take Profit hit
                if curr["high"] >= take_profit:
                    exit_p = take_profit * (1.0 - SLIPPAGE_PCT)
                    raw_ret = (exit_p / entry_price - 1)
                    pnl_dollar = (raw_ret * pos_size_usd) - (pos_size_usd * (FEE_PCT * 2))
                    pnl_pct = (pnl_dollar / (pos_size_usd / LEVERAGE)) * 100
                    capital += pnl_dollar
                    trades.append({"symbol": sym, "pnl_dollar": pnl_dollar, "pnl_pct": pnl_pct, "reason": "TP", "hold": i - entry_idx})
                    in_pos = False
                    continue

                # 2. Stop Loss hit
                elif curr["low"] <= stop_loss:
                    exit_p = stop_loss * (1.0 - SLIPPAGE_PCT)
                    raw_ret = (exit_p / entry_price - 1)
                    pnl_dollar = (raw_ret * pos_size_usd) - (pos_size_usd * (FEE_PCT * 2))
                    pnl_pct = (pnl_dollar / (pos_size_usd / LEVERAGE)) * 100
                    capital += pnl_dollar
                    trades.append({"symbol": sym, "pnl_dollar": pnl_dollar, "pnl_pct": pnl_pct, "reason": "SL", "hold": i - entry_idx})
                    in_pos = False
                    continue

                # 3. Timeout exit (after 24 bars / 2 hours)
                elif (i - entry_idx) >= 24:
                    exit_p = curr["close"] * (1.0 - SLIPPAGE_PCT)
                    raw_ret = (exit_p / entry_price - 1)
                    pnl_dollar = (raw_ret * pos_size_usd) - (pos_size_usd * (FEE_PCT * 2))
                    pnl_pct = (pnl_dollar / (pos_size_usd / LEVERAGE)) * 100
                    capital += pnl_dollar
                    trades.append({"symbol": sym, "pnl_dollar": pnl_dollar, "pnl_pct": pnl_pct, "reason": "TIME", "hold": i - entry_idx})
                    in_pos = False
                    continue

            # Look for setup
            if not in_pos:
                # Pullback candle must be red
                if prev["close"] >= prev["open"]:
                    continue

                # Support above 9 EMA & VWAP
                if prev["close"] < prev["ema9"] * 0.998 or prev["low"] < prev["vwap"] * 0.995:
                    continue

                # Prior impulse wave
                prior = df.iloc[max(0, i-6):i-1]
                surge_low = prior["low"].min()
                surge_high = prior["high"].max()
                surge = surge_high - surge_low
                if (surge / surge_low) < MIN_SURGE_PCT:
                    continue

                # Retracement <= 50%
                pullback_depth = surge_high - prev["low"]
                if (pullback_depth / surge) > 0.50:
                    continue

                # Volume contraction
                if prev["volume"] > prior["volume"].mean() * 1.10:
                    continue

                # Breakout trigger
                trigger_price = prev["high"]
                if curr["high"] >= trigger_price:
                    sl = prev["low"]
                    risk = trigger_price - sl
                    risk_pct = risk / trigger_price

                    if risk_pct > MAX_SL_PCT or risk_pct < MIN_SL_PCT:
                        continue

                    entry_price = trigger_price * (1.0 + SLIPPAGE_PCT)
                    stop_loss = sl
                    take_profit = trigger_price + (risk * rr_ratio)
                    
                    pos_margin = capital * ALLOCATION_PCT
                    pos_size_usd = pos_margin * LEVERAGE

                    in_pos = True
                    entry_idx = i

    df_res = pd.DataFrame(trades)
    if len(df_res) == 0:
        return {"net_pnl": 0, "return_pct": 0, "win_rate": 0, "pf": 0, "trades": 0}

    wins = df_res[df_res["pnl_dollar"] > 0]
    losses = df_res[df_res["pnl_dollar"] <= 0]
    gross_win = wins["pnl_dollar"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl_dollar"].sum()) if len(losses) > 0 else 1e-6
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    total_pnl = df_res["pnl_dollar"].sum()

    return {
        "final_equity": capital,
        "net_pnl": total_pnl,
        "return_pct": (total_pnl / INITIAL_CAPITAL) * 100,
        "total_trades": len(df_res),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / len(df_res)) * 100,
        "pf": pf,
        "avg_win_pct": wins["pnl_pct"].mean() if len(wins) > 0 else 0,
        "avg_loss_pct": losses["pnl_pct"].mean() if len(losses) > 0 else 0,
        "reasons": df_res["reason"].value_counts().to_dict(),
        "trades_df": df_res
    }

async def main():
    print("=" * 80)
    print("🔬 [OKX 선물 전용] 로스 카메론 1차 눌림목(First Pullback) 퀀트 백테스트")
    print(f"   초기 시드: {INITIAL_CAPITAL:.2f} USDT | 레버리지: {LEVERAGE}x | 1회 투입: {ALLOCATION_PCT*100}%")
    print(f"   거래 수수료: {FEE_PCT*100}% (Taker) | 슬리피지: {SLIPPAGE_PCT*100}%")
    print("=" * 80)

    bars_data = await fetch_historical_bars(SYMBOLS)
    print("\n📊 [R:R 익절 비율별 시뮬레이션 결과]")
    print("-" * 80)
    print(f"{'시나리오 (R:R)':<25} | {'최종 자산(USDT)':<15} | {'순손익':<12} | {'수익률':<9} | {'승률':<8} | {'PF':<6} | {'거래수':<6}")
    print("-" * 80)

    for rr in [2.0, 2.5, 3.0, 3.5]:
        res = simulate_okx_cameron(bars_data, rr_ratio=rr)
        pnl_str = f"{res['net_pnl']:+,.2f} USDT"
        eq_str = f"{res['final_equity']:,.2f} USDT"
        ret_str = f"{res['return_pct']:+.2f}%"
        win_str = f"{res['win_rate']:.1f}%"
        pf_str = f"{res['pf']:.2f}"
        tr_str = f"{res['total_trades']}회"
        print(f"R:R 1:{rr:<20} | {eq_str:<15} | {pnl_str:<12} | {ret_str:<9} | {win_str:<8} | {pf_str:<6} | {tr_str:<6}")

    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
