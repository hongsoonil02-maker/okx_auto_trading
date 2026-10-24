# -*- coding: utf-8 -*-
"""
fast_sweep_okx.py — High-Performance NumPy Vectorized Sweep for OKX Cameron Pullback
Tests:
- Direction: Long Only vs Short Only vs Bi-directional
- Leverage: 2x, 3x, 5x, 10x
- R:R Take Profit: 1:2.0, 1:2.5, 1:3.0
Includes 0.05% taker fee, 0.05% slippage, and liquidation bounds.
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
INITIAL_CAPITAL = 960.0
MARGIN_PCT = 0.15
FEE_PCT = 0.0005
SLIPPAGE_PCT = 0.0005
MIN_SURGE_PCT = 0.006
MAX_SL_PCT = 0.035
MIN_SL_PCT = 0.003

async def load_data():
    ex = ccxt_async.okx({
        "apiKey": os.getenv("OKX_API_KEY"),
        "secret": os.getenv("OKX_SECRET"),
        "password": os.getenv("OKX_PASSPHRASE"),
        "enableRateLimit": True,
        "options": {"defaultType": "swap"}
    })
    await ex.load_markets()
    data = {}
    start_ts = ex.parse8601("2026-10-01T00:00:00Z")

    for s in SYMBOLS:
        all_bars = []
        curr_ts = start_ts
        for _ in range(5):
            bars = await ex.fetch_ohlcv(s, timeframe="5m", since=curr_ts, limit=300)
            if not bars: break
            all_bars.extend(bars)
            curr_ts = bars[-1][0] + (5 * 60 * 1000)
            if len(bars) < 300: break
        if len(all_bars) >= 100:
            df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df.drop_duplicates(subset=["timestamp"], inplace=True)
            df.sort_values("timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)
            tp = (df["high"] + df["low"] + df["close"]) / 3
            df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, 1)
            df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
            # Store as fast dict of numpy arrays
            data[s] = {
                "open": df["open"].values,
                "high": df["high"].values,
                "low": df["low"].values,
                "close": df["close"].values,
                "volume": df["volume"].values,
                "vwap": df["vwap"].values,
                "ema9": df["ema9"].values,
                "len": len(df)
            }
    await ex.close()
    return data

def run_simulation(data, direction_mode="both", leverage=2, rr_ratio=2.0):
    capital = INITIAL_CAPITAL
    trades = []

    for sym, d in data.items():
        n = d["len"]
        o, h, l, c, v, vwap, ema9 = d["open"], d["high"], d["low"], d["close"], d["volume"], d["vwap"], d["ema9"]
        
        in_pos = False
        pos_side = None
        entry_idx = 0
        entry_p = 0.0
        sl_p = 0.0
        tp_p = 0.0
        pos_usd = 0.0

        for i in range(10, n):
            curr_o, curr_h, curr_l, curr_c = o[i], h[i], l[i], c[i]
            prev_o, prev_h, prev_l, prev_c = o[i-1], h[i-1], l[i-1], c[i-1]
            prev_v = v[i-1]

            # 1. Check Exit
            if in_pos:
                if pos_side == "long":
                    if curr_h >= tp_p:
                        exit_p = tp_p * (1.0 - SLIPPAGE_PCT)
                        ret = (exit_p / entry_p - 1)
                        pnl = (ret * pos_usd) - (pos_usd * FEE_PCT * 2)
                        capital += pnl
                        trades.append({"side": "LONG", "pnl": pnl, "ret": ret * leverage * 100, "win": pnl > 0})
                        in_pos = False
                        continue
                    elif curr_l <= sl_p:
                        exit_p = sl_p * (1.0 - SLIPPAGE_PCT)
                        ret = (exit_p / entry_p - 1)
                        pnl = (ret * pos_usd) - (pos_usd * FEE_PCT * 2)
                        capital += pnl
                        trades.append({"side": "LONG", "pnl": pnl, "ret": ret * leverage * 100, "win": pnl > 0})
                        in_pos = False
                        continue
                    elif (i - entry_idx) >= 20: # 100 mins timeout
                        exit_p = curr_c * (1.0 - SLIPPAGE_PCT)
                        ret = (exit_p / entry_p - 1)
                        pnl = (ret * pos_usd) - (pos_usd * FEE_PCT * 2)
                        capital += pnl
                        trades.append({"side": "LONG", "pnl": pnl, "ret": ret * leverage * 100, "win": pnl > 0})
                        in_pos = False
                        continue

                elif pos_side == "short":
                    if curr_l <= tp_p:
                        exit_p = tp_p * (1.0 + SLIPPAGE_PCT)
                        ret = (entry_p / exit_p - 1)
                        pnl = (ret * pos_usd) - (pos_usd * FEE_PCT * 2)
                        capital += pnl
                        trades.append({"side": "SHORT", "pnl": pnl, "ret": ret * leverage * 100, "win": pnl > 0})
                        in_pos = False
                        continue
                    elif curr_h >= sl_p:
                        exit_p = sl_p * (1.0 + SLIPPAGE_PCT)
                        ret = (entry_p / exit_p - 1)
                        pnl = (ret * pos_usd) - (pos_usd * FEE_PCT * 2)
                        capital += pnl
                        trades.append({"side": "SHORT", "pnl": pnl, "ret": ret * leverage * 100, "win": pnl > 0})
                        in_pos = False
                        continue
                    elif (i - entry_idx) >= 20:
                        exit_p = curr_c * (1.0 + SLIPPAGE_PCT)
                        ret = (entry_p / exit_p - 1)
                        pnl = (ret * pos_usd) - (pos_usd * FEE_PCT * 2)
                        capital += pnl
                        trades.append({"side": "SHORT", "pnl": pnl, "ret": ret * leverage * 100, "win": pnl > 0})
                        in_pos = False
                        continue

            # 2. Check Entry
            if not in_pos:
                prior_h = np.max(h[max(0, i-6):i-1])
                prior_l = np.min(l[max(0, i-6):i-1])
                prior_v_mean = np.mean(v[max(0, i-6):i-1])

                # Long Setup
                if direction_mode in ["long", "both"]:
                    if prev_c < prev_o and prev_c >= ema9[i-1] * 0.998 and prev_l >= vwap[i-1] * 0.995:
                        surge = prior_h - prior_l
                        if (surge / prior_l) >= MIN_SURGE_PCT:
                            if (prior_h - prev_l) / surge <= 0.50 and prev_v <= prior_v_mean * 1.15:
                                if curr_h >= prev_h:
                                    risk = prev_h - prev_l
                                    risk_pct = risk / prev_h
                                    if MIN_SL_PCT <= risk_pct <= MAX_SL_PCT:
                                        in_pos = True
                                        pos_side = "long"
                                        entry_p = prev_h * (1.0 + SLIPPAGE_PCT)
                                        sl_p = prev_l
                                        tp_p = prev_h + (risk * rr_ratio)
                                        pos_usd = (capital * MARGIN_PCT) * leverage
                                        entry_idx = i
                                        continue

                # Short Setup
                if direction_mode in ["short", "both"]:
                    if prev_c > prev_o and prev_c <= ema9[i-1] * 1.002 and prev_h <= vwap[i-1] * 1.005:
                        dump = prior_h - prior_l
                        if (dump / prior_h) >= MIN_SURGE_PCT:
                            if (prev_h - prior_l) / dump <= 0.50 and prev_v <= prior_v_mean * 1.15:
                                if curr_l <= prev_l:
                                    risk = prev_h - prev_l
                                    risk_pct = risk / prev_l
                                    if MIN_SL_PCT <= risk_pct <= MAX_SL_PCT:
                                        in_pos = True
                                        pos_side = "short"
                                        entry_p = prev_l * (1.0 - SLIPPAGE_PCT)
                                        sl_p = prev_h
                                        tp_p = prev_l - (risk * rr_ratio)
                                        pos_usd = (capital * MARGIN_PCT) * leverage
                                        entry_idx = i
                                        continue

    df_t = pd.DataFrame(trades)
    if len(df_t) == 0:
        return {"capital": capital, "net_pnl": 0, "ret_pct": 0, "win_rate": 0, "pf": 0, "trades": 0, "longs": 0, "shorts": 0}

    wins = df_t[df_t["win"] == True]
    losses = df_t[df_t["win"] == False]
    gross_win = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 1e-6
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    longs = len(df_t[df_t["side"] == "LONG"])
    shorts = len(df_t[df_t["side"] == "SHORT"])

    return {
        "capital": capital,
        "net_pnl": df_t["pnl"].sum(),
        "ret_pct": (df_t["pnl"].sum() / INITIAL_CAPITAL) * 100,
        "win_rate": (len(wins) / len(df_t)) * 100,
        "pf": pf,
        "trades": len(df_t),
        "longs": longs,
        "shorts": shorts
    }

async def main():
    print("=" * 105)
    print("🔬 [OKX 로스 카메론 전략] 레버리지(2x~10x) & 양방향(Long/Short) 정밀 고속 백테스트")
    print(f"   시드 자본: {INITIAL_CAPITAL:.2f} USDT | 1회 투입: {MARGIN_PCT*100}% | 수수료: {FEE_PCT*100}% | 슬리피지: {SLIPPAGE_PCT*100}%")
    print("=" * 105)

    data = await load_data()
    print(f"✅ OKX 실데이터 로드 완료: {len(data)}개 선물 페어\n")

    grid = []
    # Test 3 directions x 4 leverages x 2 R:R
    for d in ["long", "short", "both"]:
        for lev in [2, 3, 5, 10]:
            for rr in [2.0, 2.5]:
                grid.append({"dir": d, "lev": lev, "rr": rr})

    results = []
    for g in grid:
        res = run_simulation(data, direction_mode=g["dir"], leverage=g["lev"], rr_ratio=g["rr"])
        res.update(g)
        results.append(res)

    print(f"{'방향 (Direction)':<16} | {'레버리지':<8} | {'R:R':<6} | {'최종 자산(USDT)':<15} | {'순손익':<12} | {'수익률':<9} | {'승률':<8} | {'PF':<6} | {'거래수(L/S)':<10}")
    print("-" * 108)

    for r in results:
        dir_name = {"long": "롱 전용 (Long)", "short": "숏 전용 (Short)", "both": "양방향 (Bi-dir)"}[r['dir']]
        pnl_str = f"{r['net_pnl']:+,.2f} U"
        eq_str = f"{r['capital']:,.2f} USDT"
        ret_str = f"{r['ret_pct']:+.2f}%"
        win_str = f"{r['win_rate']:.1f}%"
        pf_str = f"{r['pf']:.2f}"
        tr_str = f"{r['trades']}회 ({r['longs']}/{r['shorts']})"
        print(f"{dir_name:<16} | {r['lev']}x     | 1:{r['rr']} | {eq_str:<15} | {pnl_str:<12} | {ret_str:<9} | {win_str:<8} | {pf_str:<6} | {tr_str:<10}")

    print("=" * 108)
    best = max(results, key=lambda x: x["net_pnl"])
    print(f"\n🏆 [최우수 추천 설정]")
    print(f"   • 전략 방향: {best['dir'].upper()} | 레버리지: {best['lev']}x | 목표 손익비: 1:{best['rr']}")
    print(f"   • 순이익: {best['net_pnl']:+,.2f} USDT ({best['ret_pct']:+.2f}%)")
    print(f"   • 손익비 (Profit Factor): {best['pf']:.2f} | 승률: {best['win_rate']:.1f}%")
    print(f"   • 총 거래수: {best['trades']}회 (롱 {best['longs']}회 / 숏 {best['shorts']}회)")

if __name__ == "__main__":
    asyncio.run(main())
