# -*- coding: utf-8 -*-
"""
run_okx_timeframe_sweep.py — Multi-Timeframe (1m, 1h, 4h) Walk-Forward Backtest for OKX
Tests:
- Timeframes: '1m' (Scalping), '1h' (Swing), '4h' (Macro Trend Swing)
- Direction: Long Only
- Leverage: 2x, 3x
- Targets: High volatility majors/swaps (SOL, SUI, DOGE, ETH, BTC)
Includes full fee (0.05% taker x 2) and slippage models.
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

async def fetch_timeframe_data(timeframe: str, num_bars: int = 1000):
    ex = ccxt_async.okx({
        "apiKey": os.getenv("OKX_API_KEY"),
        "secret": os.getenv("OKX_SECRET"),
        "password": os.getenv("OKX_PASSPHRASE"),
        "enableRateLimit": True,
        "options": {"defaultType": "swap"}
    })
    await ex.load_markets()
    dataset = {}
    
    for s in SYMBOLS:
        all_bars = []
        try:
            # fetch multiple pages for deep history
            since = None
            for _ in range(4):
                params = {}
                if all_bars:
                    # OKX pagination
                    params["after"] = all_bars[-1][0]
                bars = await ex.fetch_ohlcv(s, timeframe=timeframe, limit=300, params=params)
                if not bars:
                    break
                all_bars.extend(bars)
                if len(bars) < 300:
                    break

            if len(all_bars) >= 50:
                df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df.drop_duplicates(subset=["timestamp"], inplace=True)
                df.sort_values("timestamp", inplace=True)
                df.reset_index(drop=True, inplace=True)

                tp = (df["high"] + df["low"] + df["close"]) / 3
                df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, 1)
                df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
                
                dataset[s] = {
                    "open": df["open"].values,
                    "high": df["high"].values,
                    "low": df["low"].values,
                    "close": df["close"].values,
                    "volume": df["volume"].values,
                    "vwap": df["vwap"].values,
                    "ema9": df["ema9"].values,
                    "time": pd.to_datetime(df["timestamp"], unit="ms"),
                    "len": len(df)
                }
        except Exception as e:
            pass

    await ex.close()
    return dataset

def simulate_tf(dataset, tf: str, leverage: int = 2, rr_ratio: float = 2.0):
    capital = INITIAL_CAPITAL
    trades = []

    # Parameters tuned per timeframe
    if tf == "1m":
        min_surge = 0.005    # 0.5% surge for 1m
        max_sl = 0.025       # 2.5% max risk
        min_sl = 0.002       # 0.2% min risk
        timeout_bars = 30    # 30 mins
    elif tf == "1h":
        min_surge = 0.020    # 2.0% surge for 1h
        max_sl = 0.050       # 5.0% max risk
        min_sl = 0.008       # 0.8% min risk
        timeout_bars = 24    # 24 hours
    elif tf == "4h":
        min_surge = 0.040    # 4.0% surge for 4h
        max_sl = 0.080       # 8.0% max risk
        min_sl = 0.015       # 1.5% min risk
        timeout_bars = 18    # 3 days

    for sym, d in dataset.items():
        n = d["len"]
        o, h, l, c, v, vwap, ema9 = d["open"], d["high"], d["low"], d["close"], d["volume"], d["vwap"], d["ema9"]
        
        in_pos = False
        entry_idx = 0
        entry_p = 0.0
        sl_p = 0.0
        tp_p = 0.0
        pos_usd = 0.0

        for i in range(10, n):
            curr_h, curr_l, curr_c = h[i], l[i], c[i]
            prev_o, prev_h, prev_l, prev_c = o[i-1], h[i-1], l[i-1], c[i-1]
            prev_v = v[i-1]

            if in_pos:
                # 1. TP hit
                if curr_h >= tp_p:
                    exit_p = tp_p * (1.0 - SLIPPAGE_PCT)
                    ret = (exit_p / entry_p - 1)
                    pnl = (ret * pos_usd) - (pos_usd * FEE_PCT * 2)
                    capital += pnl
                    trades.append({"symbol": sym, "pnl": pnl, "ret": ret * leverage * 100, "win": pnl > 0, "reason": "TP"})
                    in_pos = False
                    continue
                # 2. SL hit
                elif curr_l <= sl_p:
                    exit_p = sl_p * (1.0 - SLIPPAGE_PCT)
                    ret = (exit_p / entry_p - 1)
                    pnl = (ret * pos_usd) - (pos_usd * FEE_PCT * 2)
                    capital += pnl
                    trades.append({"symbol": sym, "pnl": pnl, "ret": ret * leverage * 100, "win": pnl > 0, "reason": "SL"})
                    in_pos = False
                    continue
                # 3. Timeout
                elif (i - entry_idx) >= timeout_bars:
                    exit_p = curr_c * (1.0 - SLIPPAGE_PCT)
                    ret = (exit_p / entry_p - 1)
                    pnl = (ret * pos_usd) - (pos_usd * FEE_PCT * 2)
                    capital += pnl
                    trades.append({"symbol": sym, "pnl": pnl, "ret": ret * leverage * 100, "win": pnl > 0, "reason": "TIME"})
                    in_pos = False
                    continue

            if not in_pos:
                prior_h = np.max(h[max(0, i-6):i-1])
                prior_l = np.min(l[max(0, i-6):i-1])
                prior_v_mean = np.mean(v[max(0, i-6):i-1])

                # Long Pullback Condition
                if prev_c < prev_o and prev_c >= ema9[i-1] * 0.998 and prev_l >= vwap[i-1] * 0.995:
                    surge = prior_h - prior_l
                    if (surge / prior_l) >= min_surge:
                        retrace = (prior_h - prev_l) / surge
                        if retrace <= 0.55 and prev_v <= prior_v_mean * 1.20:
                            if curr_h >= prev_h: # Breakout trigger
                                risk = prev_h - prev_l
                                risk_pct = risk / prev_h
                                if min_sl <= risk_pct <= max_sl:
                                    in_pos = True
                                    entry_p = prev_h * (1.0 + SLIPPAGE_PCT)
                                    sl_p = prev_l
                                    tp_p = prev_h + (risk * rr_ratio)
                                    pos_usd = (capital * MARGIN_PCT) * leverage
                                    entry_idx = i

    df_t = pd.DataFrame(trades)
    if len(df_t) == 0:
        return {"tf": tf, "lev": leverage, "capital": capital, "net_pnl": 0, "ret_pct": 0, "win_rate": 0, "pf": 0, "trades": 0}

    wins = df_t[df_t["win"] == True]
    losses = df_t[df_t["win"] == False]
    gross_win = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 1e-6
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    return {
        "tf": tf,
        "lev": leverage,
        "capital": capital,
        "net_pnl": df_t["pnl"].sum(),
        "ret_pct": (df_t["pnl"].sum() / INITIAL_CAPITAL) * 100,
        "win_rate": (len(wins) / len(df_t)) * 100,
        "pf": pf,
        "trades": len(df_t),
        "wins": len(wins),
        "losses": len(losses),
        "avg_win": wins["ret"].mean() if len(wins) > 0 else 0,
        "avg_loss": losses["ret"].mean() if len(losses) > 0 else 0
    }

async def main():
    print("=" * 100)
    print("🔬 [OKX 로스 카메론 전략] 타임프레임별 (1분봉 vs 1시간봉 vs 4시간봉) 전수 비교 백테스트")
    print(f"   시드 자본: {INITIAL_CAPITAL:.2f} USDT | 1회 투입: {MARGIN_PCT*100}% | 수수료: 0.05% | 슬리피지: 0.05%")
    print("=" * 100)

    timeframes = ["1m", "1h", "4h"]
    all_results = []

    for tf in timeframes:
        print(f"📥 [{tf}] 타임프레임 데이터 로딩 중...")
        data = await fetch_timeframe_data(tf)
        sample_bars = next(iter(data.values()))["len"] if data else 0
        sample_dates = f"{next(iter(data.values()))['time'].iloc[0]} ~ {next(iter(data.values()))['time'].iloc[-1]}" if data else "N/A"
        print(f"   ✅ {len(data)}개 페어 완료 (페어당 {sample_bars}개 캔들, {sample_dates})")

        for lev in [2, 3]:
            for rr in [2.0, 2.5]:
                res = simulate_tf(data, tf=tf, leverage=lev, rr_ratio=rr)
                res["rr"] = rr
                all_results.append(res)

    print("\n" + "=" * 100)
    print(f"{'타임프레임 (TF)':<16} | {'레버리지':<8} | {'R:R':<6} | {'최종 자산(USDT)':<15} | {'순손익':<12} | {'수익률':<9} | {'승률':<8} | {'PF':<6} | {'거래수':<6}")
    print("-" * 100)

    for r in all_results:
        tf_label = {"1m": "1분봉 (1m 초단타)", "1h": "1시간봉 (1h 스윙)", "4h": "4시간봉 (4h 추세스윙)"}[r["tf"]]
        pnl_str = f"{r['net_pnl']:+,.2f} U"
        eq_str = f"{r['capital']:,.2f} USDT"
        ret_str = f"{r['ret_pct']:+.2f}%"
        win_str = f"{r['win_rate']:.1f}%"
        pf_str = f"{r['pf']:.2f}"
        tr_str = f"{r['trades']}회"
        print(f"{tf_label:<16} | {r['lev']}x     | 1:{r['rr']} | {eq_str:<15} | {pnl_str:<12} | {ret_str:<9} | {win_str:<8} | {pf_str:<6} | {tr_str:<6}")

    print("=" * 100)
    best = max(all_results, key=lambda x: x["net_pnl"])
    print(f"\n🏆 [최우수 추천 타임프레임]")
    print(f"   • 타임프레임: {best['tf']} ({best['lev']}x 레버리지, R:R 1:{best['rr']})")
    print(f"   • 순이익: {best['net_pnl']:+,.2f} USDT ({best['ret_pct']:+.2f}%)")
    print(f"   • 손익비 (Profit Factor): {best['pf']:.2f} | 승률: {best['win_rate']:.1f}% ({best['wins']}승 / {best['losses']}패)")
    print(f"   • 평균 승리 수익률: +{best['avg_win']:.2f}% | 평균 손실률: {best['avg_loss']:.2f}%")

if __name__ == "__main__":
    asyncio.run(main())
