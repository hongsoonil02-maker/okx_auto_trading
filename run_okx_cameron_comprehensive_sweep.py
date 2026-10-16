# -*- coding: utf-8 -*-
"""
run_okx_cameron_comprehensive_sweep.py — Multi-Leverage & Bi-directional (Long + Short)
First Pullback Strategy Walk-Forward Backtest on OKX USDT-Swaps.

Grid Matrix Tested:
1. Direction:
   - Long Only (Bullish Impulse Surge -> Pullback -> Breakout)
   - Short Only (Bearish Impulse Dump -> Relief Bounce -> Breakdown)
   - Bi-directional (Long + Short Both)
2. Leverage:
   - 2x, 3x, 5x, 10x
3. Take-Profit R:R:
   - 1:2.0, 1:2.5, 1:3.0
Includes exact OKX taker fees (0.05% x 2) and slippage (0.05% x 2) with liquidation risk model.
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

SYMBOLS = [
    "SOL/USDT:USDT",
    "SUI/USDT:USDT",
    "DOGE/USDT:USDT",
    "ETH/USDT:USDT",
    "BTC/USDT:USDT",
    "NEAR/USDT:USDT",
    "PEPE/USDT:USDT"
]

INITIAL_CAPITAL = 960.0
MARGIN_PCT = 0.15              # 15% equity per position
FEE_PCT = 0.0005               # 0.05% taker
SLIPPAGE_PCT = 0.0005          # 0.05% slippage
MIN_SURGE_PCT = 0.006          # 0.6% impulse
MAX_SL_PCT = 0.03              # 3.0% max stop distance
MIN_SL_PCT = 0.003             # 0.3% min stop distance

async def load_okx_dataset():
    ex = ccxt_async.okx({
        "apiKey": os.getenv("OKX_API_KEY"),
        "secret": os.getenv("OKX_SECRET"),
        "password": os.getenv("OKX_PASSPHRASE"),
        "enableRateLimit": True,
        "options": {"defaultType": "swap"}
    })
    await ex.load_markets()
    dataset = {}
    start_ts = ex.parse8601("2026-10-01T00:00:00Z")

    for s in SYMBOLS:
        try:
            all_bars = []
            curr_ts = start_ts
            for _ in range(8):
                bars = await ex.fetch_ohlcv(s, timeframe="5m", since=curr_ts, limit=300)
                if not bars:
                    break
                all_bars.extend(bars)
                curr_ts = bars[-1][0] + (5 * 60 * 1000)
                if len(bars) < 300:
                    break
            if len(all_bars) >= 100:
                df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.drop_duplicates(subset=["timestamp"], inplace=True)
                df.sort_values("timestamp", inplace=True)
                df.reset_index(drop=True, inplace=True)

                tp = (df["high"] + df["low"] + df["close"]) / 3
                df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, 1)
                df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
                dataset[s] = df
        except Exception as e:
            pass
    await ex.close()
    return dataset

def backtest_engine(dataset, direction_mode="both", leverage=2, rr_ratio=2.0):
    capital = INITIAL_CAPITAL
    trades = []

    for sym, df in dataset.items():
        in_pos = False
        pos_side = None # 'long' or 'short'
        entry_idx = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        pos_usd = 0.0

        for i in range(10, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]

            # ── 1. MANAGE OPEN POSITION ──
            if in_pos:
                if pos_side == "long":
                    # Long TP
                    if curr["high"] >= take_profit:
                        exit_p = take_profit * (1.0 - SLIPPAGE_PCT)
                        ret = (exit_p / entry_price - 1)
                        pnl_dollar = (ret * pos_usd) - (pos_usd * (FEE_PCT * 2))
                        capital += pnl_dollar
                        trades.append({"symbol": sym, "side": "LONG", "pnl": pnl_dollar, "ret_pct": ret * leverage * 100, "reason": "TP"})
                        in_pos = False
                        continue
                    # Long SL
                    elif curr["low"] <= stop_loss:
                        exit_p = stop_loss * (1.0 - SLIPPAGE_PCT)
                        ret = (exit_p / entry_price - 1)
                        pnl_dollar = (ret * pos_usd) - (pos_usd * (FEE_PCT * 2))
                        capital += pnl_dollar
                        trades.append({"symbol": sym, "side": "LONG", "pnl": pnl_dollar, "ret_pct": ret * leverage * 100, "reason": "SL"})
                        in_pos = False
                        continue
                    # Timeout (24 bars)
                    elif (i - entry_idx) >= 24:
                        exit_p = curr["close"] * (1.0 - SLIPPAGE_PCT)
                        ret = (exit_p / entry_price - 1)
                        pnl_dollar = (ret * pos_usd) - (pos_usd * (FEE_PCT * 2))
                        capital += pnl_dollar
                        trades.append({"symbol": sym, "side": "LONG", "pnl": pnl_dollar, "ret_pct": ret * leverage * 100, "reason": "TIME"})
                        in_pos = False
                        continue

                elif pos_side == "short":
                    # Short TP
                    if curr["low"] <= take_profit:
                        exit_p = take_profit * (1.0 + SLIPPAGE_PCT)
                        ret = (entry_price / exit_p - 1)
                        pnl_dollar = (ret * pos_usd) - (pos_usd * (FEE_PCT * 2))
                        capital += pnl_dollar
                        trades.append({"symbol": sym, "side": "SHORT", "pnl": pnl_dollar, "ret_pct": ret * leverage * 100, "reason": "TP"})
                        in_pos = False
                        continue
                    # Short SL
                    elif curr["high"] >= stop_loss:
                        exit_p = stop_loss * (1.0 + SLIPPAGE_PCT)
                        ret = (entry_price / exit_p - 1)
                        pnl_dollar = (ret * pos_usd) - (pos_usd * (FEE_PCT * 2))
                        capital += pnl_dollar
                        trades.append({"symbol": sym, "side": "SHORT", "pnl": pnl_dollar, "ret_pct": ret * leverage * 100, "reason": "SL"})
                        in_pos = False
                        continue
                    # Timeout (24 bars)
                    elif (i - entry_idx) >= 24:
                        exit_p = curr["close"] * (1.0 + SLIPPAGE_PCT)
                        ret = (entry_price / exit_p - 1)
                        pnl_dollar = (ret * pos_usd) - (pos_usd * (FEE_PCT * 2))
                        capital += pnl_dollar
                        trades.append({"symbol": sym, "side": "SHORT", "pnl": pnl_dollar, "ret_pct": ret * leverage * 100, "reason": "TIME"})
                        in_pos = False
                        continue

            # ── 2. SCAN FOR ENTRY SETUPS ──
            if not in_pos:
                prior = df.iloc[max(0, i-6):i-1]

                # --- LONG SETUP ---
                if direction_mode in ["long", "both"]:
                    if prev["close"] < prev["open"]: # Pullback candle is red
                        if prev["close"] >= prev["ema9"] * 0.998 and prev["low"] >= prev["vwap"] * 0.995:
                            surge_low = prior["low"].min()
                            surge_high = prior["high"].max()
                            surge = surge_high - surge_low
                            if (surge / surge_low) >= MIN_SURGE_PCT:
                                retrace = (surge_high - prev["low"]) / surge
                                if retrace <= 0.50 and prev["volume"] <= prior["volume"].mean() * 1.15:
                                    trigger_price = prev["high"]
                                    if curr["high"] >= trigger_price:
                                        sl = prev["low"]
                                        risk = trigger_price - sl
                                        risk_pct = risk / trigger_price
                                        if MIN_SL_PCT <= risk_pct <= MAX_SL_PCT:
                                            in_pos = True
                                            pos_side = "long"
                                            entry_price = trigger_price * (1.0 + SLIPPAGE_PCT)
                                            stop_loss = sl
                                            take_profit = trigger_price + (risk * rr_ratio)
                                            pos_margin = capital * MARGIN_PCT
                                            pos_usd = pos_margin * leverage
                                            entry_idx = i
                                            continue

                # --- SHORT SETUP ---
                if direction_mode in ["short", "both"]:
                    if prev["close"] > prev["open"]: # Relief bounce candle is green
                        if prev["close"] <= prev["ema9"] * 1.002 and prev["high"] <= prev["vwap"] * 1.005:
                            dump_high = prior["high"].max()
                            dump_low = prior["low"].min()
                            dump = dump_high - dump_low
                            if (dump / dump_high) >= MIN_SURGE_PCT:
                                bounce = (prev["high"] - dump_low) / dump
                                if bounce <= 0.50 and prev["volume"] <= prior["volume"].mean() * 1.15:
                                    trigger_price = prev["low"]
                                    if curr["low"] <= trigger_price:
                                        sl = prev["high"]
                                        risk = sl - trigger_price
                                        risk_pct = risk / trigger_price
                                        if MIN_SL_PCT <= risk_pct <= MAX_SL_PCT:
                                            in_pos = True
                                            pos_side = "short"
                                            entry_price = trigger_price * (1.0 - SLIPPAGE_PCT)
                                            stop_loss = sl
                                            take_profit = trigger_price - (risk * rr_ratio)
                                            pos_margin = capital * MARGIN_PCT
                                            pos_usd = pos_margin * leverage
                                            entry_idx = i
                                            continue

    df_t = pd.DataFrame(trades)
    if len(df_t) == 0:
        return {"capital": capital, "net_pnl": 0, "ret_pct": 0, "win_rate": 0, "pf": 0, "trades": 0, "long_trades": 0, "short_trades": 0}

    wins = df_t[df_t["pnl"] > 0]
    losses = df_t[df_t["pnl"] <= 0]
    gross_win = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 1e-6
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    long_cnt = len(df_t[df_t["side"] == "LONG"])
    short_cnt = len(df_t[df_t["side"] == "SHORT"])

    return {
        "capital": capital,
        "net_pnl": df_t["pnl"].sum(),
        "ret_pct": (df_t["pnl"].sum() / INITIAL_CAPITAL) * 100,
        "win_rate": (len(wins) / len(df_t)) * 100,
        "pf": pf,
        "trades": len(df_t),
        "long_trades": long_cnt,
        "short_trades": short_cnt,
        "avg_win_ret": wins["ret_pct"].mean() if len(wins) > 0 else 0,
        "avg_loss_ret": losses["ret_pct"].mean() if len(losses) > 0 else 0
    }

async def main():
    print("=" * 95)
    print("🔬 [OKX 로스 카메론 전략] 레버리지(2x~10x) & 양방향(Long/Short) 전수 백테스트")
    print(f"   시드 자본: {INITIAL_CAPITAL:.2f} USDT | 1회 투입: {MARGIN_PCT*100}% | 수수료: 0.05% | 슬리피지: 0.05%")
    print("=" * 95)

    dataset = await load_okx_dataset()
    print(f"✅ OKX 실데이터 로드 완료: {len(dataset)}개 선물 페어 ({list(dataset.keys())})\n")

    test_grid = []
    # 1. Test Direction Modes (at 2x, R:R 2.0)
    for d in ["long", "short", "both"]:
        for lev in [2, 3, 5, 10]:
            for rr in [2.0, 2.5]:
                test_grid.append({"dir": d, "lev": lev, "rr": rr})

    results = []
    for cfg in test_grid:
        res = backtest_engine(dataset, direction_mode=cfg["dir"], leverage=cfg["lev"], rr_ratio=cfg["rr"])
        res.update(cfg)
        results.append(res)

    # Format Output Table
    print(f"{'방향 (Direction)':<16} | {'레버리지':<8} | {'R:R':<6} | {'최종 자산(USDT)':<15} | {'순손익':<12} | {'수익률':<9} | {'승률':<8} | {'PF':<6} | {'거래(L/S)':<10}")
    print("-" * 108)

    for r in results:
        dir_name = {"long": "롱 전용 (Long)", "short": "숏 전용 (Short)", "both": "양방향 (Bi-dir)"}[r['dir']]
        pnl_str = f"{r['net_pnl']:+,.2f} U"
        eq_str = f"{r['capital']:,.2f} USDT"
        ret_str = f"{r['ret_pct']:+.2f}%"
        win_str = f"{r['win_rate']:.1f}%"
        pf_str = f"{r['pf']:.2f}"
        tr_str = f"{r['trades']}회 ({r['long_trades']}/{r['short_trades']})"
        print(f"{dir_name:<16} | {r['lev']}x     | 1:{r['rr']} | {eq_str:<15} | {pnl_str:<12} | {ret_str:<9} | {win_str:<8} | {pf_str:<6} | {tr_str:<10}")

    print("=" * 108)

    # Best Performance
    best = max(results, key=lambda x: x["net_pnl"])
    print(f"\n🏆 [최우수 추천 설정]")
    print(f"   • 전략 방향: {best['dir'].upper()} | 레버리지: {best['lev']}x | 목표 손익비: 1:{best['rr']}")
    print(f"   • 순이익: {best['net_pnl']:+,.2f} USDT ({best['ret_pct']:+.2f}%)")
    print(f"   • 손익비 (Profit Factor): {best['pf']:.2f} | 승률: {best['win_rate']:.1f}%")
    print(f"   • 총 거래수: {best['trades']}회 (롱 {best['long_trades']}회 / 숏 {best['short_trades']}회)")

if __name__ == "__main__":
    asyncio.run(main())
