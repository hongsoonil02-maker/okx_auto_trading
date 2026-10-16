# -*- coding: utf-8 -*-
"""
backtest_cameron_penny_jev.py — OKX Cameron Penny/Meme Pullback with Jev LOB Gate Comparison
Evaluates:
1. Pure Cameron Pullback (1h) on 10 Meme/Penny OKX Swaps (DOGE, SHIB, PEPE, BONK, WIF, FLOKI, NOT, GALA, CHZ, SUI)
2. Cameron Pullback + 4H Trend Filter
3. Cameron Pullback + Jev LOB Microstructure Gate (Simulated Buyer Pressure / Gate Threshold 0.60)
4. Cameron Pullback + 4H Trend + Jev LOB Gate (Full Hybrid)
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

SYMBOLS = [
    "DOGE/USDT:USDT",
    "SHIB/USDT:USDT",
    "PEPE/USDT:USDT",
    "BONK/USDT:USDT",
    "WIF/USDT:USDT",
    "FLOKI/USDT:USDT",
    "SUI/USDT:USDT",
    "GALA/USDT:USDT"
]

INITIAL_CAPITAL = 960.0
MARGIN_PCT = 0.15
LEVERAGE = 3
FEE_PCT = 0.0005     # 0.05% taker
SLIPPAGE_PCT = 0.0005

async def load_all_data():
    ex = ccxt_async.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    await ex.load_markets()
    
    data_1h = {}
    data_4h = {}
    
    for s in SYMBOLS:
        try:
            # 1h bars
            b1 = await ex.fetch_ohlcv(s, timeframe="1h", limit=400)
            if b1 and len(b1) > 50:
                df1 = pd.DataFrame(b1, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df1.sort_values("timestamp", inplace=True)
                df1.reset_index(drop=True, inplace=True)
                tp = (df1["high"] + df1["low"] + df1["close"]) / 3
                df1["vwap"] = (tp * df1["volume"]).cumsum() / df1["volume"].cumsum().replace(0, 1)
                df1["ema9"] = df1["close"].ewm(span=9, adjust=False).mean()
                
                # JEV LOB micro-momentum proxy: evaluates close position within high-low range + volume expansion
                rng = np.maximum(df1["high"] - df1["low"], 1e-6)
                close_pos = (df1["close"] - df1["low"]) / rng
                vol_ratio = df1["volume"] / df1["volume"].rolling(10).mean().replace(0, 1)
                df1["jev_score"] = np.clip(0.35 + 0.40 * close_pos + 0.15 * np.clip(vol_ratio - 1.0, 0, 1.5), 0.1, 0.95)
                
                data_1h[s] = df1

            # 4h bars
            b4 = await ex.fetch_ohlcv(s, timeframe="4h", limit=150)
            if b4 and len(b4) > 20:
                df4 = pd.DataFrame(b4, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df4["ema20"] = df4["close"].ewm(span=20, adjust=False).mean()
                data_4h[s] = df4
        except Exception as e:
            pass

    await ex.close()
    return data_1h, data_4h

def simulate_pipeline(data_1h, data_4h, use_4h_trend=False, use_jev_gate=False, jev_thresh=0.60, rr_ratio=2.0):
    capital = INITIAL_CAPITAL
    trades = []
    
    for s, df in data_1h.items():
        if df is None or len(df) < 25:
            continue
            
        df4 = data_4h.get(s)
        
        o = df["open"].values
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        v = df["volume"].values
        ema9 = df["ema9"].values
        vwap = df["vwap"].values
        jev = df["jev_score"].values
        ts = df["timestamp"].values
        
        in_pos = False
        entry_idx = 0
        entry_p = 0.0
        sl_p = 0.0
        tp_p = 0.0
        pos_usd = 0.0
        
        for i in range(12, len(df)):
            curr_h, curr_l, curr_c = h[i], l[i], c[i]
            prev_o, prev_h, prev_l, prev_c = o[i-1], h[i-1], l[i-1], c[i-1]
            
            # 1. Manage position
            if in_pos:
                bars_held = i - entry_idx
                # Stop loss
                if curr_l <= sl_p:
                    loss = (sl_p / entry_p) - 1.0 - (FEE_PCT * 2)
                    pnl = pos_usd * loss
                    capital += pnl
                    trades.append({"symbol": s, "win": False, "pnl": pnl, "ret": loss})
                    in_pos = False
                # Take profit
                elif curr_h >= tp_p:
                    gain = (tp_p / entry_p) - 1.0 - (FEE_PCT * 2)
                    pnl = pos_usd * gain
                    capital += pnl
                    trades.append({"symbol": s, "win": True, "pnl": pnl, "ret": gain})
                    in_pos = False
                # Timeout after 24 bars (24 hours)
                elif bars_held >= 24:
                    exit_p = curr_c
                    ret = (exit_p / entry_p) - 1.0 - (FEE_PCT * 2)
                    pnl = pos_usd * ret
                    capital += pnl
                    trades.append({"symbol": s, "win": ret > 0, "pnl": pnl, "ret": ret})
                    in_pos = False
                    
            # 2. Evaluate Setup
            if not in_pos:
                # 4H Trend Filter (Optional)
                if use_4h_trend and df4 is not None:
                    # check if current 1h is above 4h EMA20
                    sub4 = df4[df4["timestamp"] <= ts[i]]
                    if len(sub4) > 0 and sub4["close"].iloc[-1] < sub4["ema20"].iloc[-1]:
                        continue
                        
                # Ross Cameron Pullback setup:
                # 1) Prev candle red
                # 2) Above 9 EMA & VWAP
                if prev_c < prev_o and prev_c >= ema9[i-1] * 0.995 and prev_l >= vwap[i-1] * 0.990:
                    prior_h = np.max(h[max(0, i-6):i-1])
                    prior_l = np.min(l[max(0, i-6):i-1])
                    surge = prior_h - prior_l
                    if surge / prior_l >= 0.025: # 2.5% surge
                        retrace = (prior_h - prev_l) / surge
                        if retrace <= 0.55: # Healthy pullback
                            # Trigger check
                            if curr_h >= prev_h:
                                # JEV AI Gate check (Optional)
                                if use_jev_gate and jev[i-1] < jev_thresh:
                                    continue # Vetoed by JEV!
                                    
                                risk = prev_h - prev_l
                                risk_pct = risk / prev_h
                                if 0.008 <= risk_pct <= 0.065:
                                    in_pos = True
                                    entry_p = prev_h * (1.0 + SLIPPAGE_PCT)
                                    sl_p = prev_l
                                    tp_p = prev_h + (risk * rr_ratio)
                                    pos_usd = (capital * MARGIN_PCT) * LEVERAGE
                                    entry_idx = i

    df_t = pd.DataFrame(trades)
    if len(df_t) == 0:
        return {"trades": 0, "win_rate": 0, "pf": 0, "pnl": 0, "ret": 0}
        
    wins = df_t[df_t["win"] == True]
    losses = df_t[df_t["win"] == False]
    gross_w = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_l = abs(losses["pnl"].sum()) if len(losses) > 0 else 1e-6
    pf = gross_w / gross_l
    
    return {
        "trades": len(df_t),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / len(df_t)) * 100,
        "pf": pf,
        "pnl": df_t["pnl"].sum(),
        "ret": (df_t["pnl"].sum() / INITIAL_CAPITAL) * 100
    }

async def main():
    print("📥 OKX 선물 데이터 로딩 중 (페니/밈 8대 종목: DOGE, SHIB, PEPE, BONK, WIF, FLOKI, SUI, GALA)...")
    data_1h, data_4h = await load_all_data()
    print("✅ 데이터 로딩 완료. 4개 시나리오 백테스트 실행 중...\n")
    
    scenarios = [
        {"name": "1. 순수 로스 카메룬 1시간봉 (No Filter)", "4h": False, "jev": False, "th": 0},
        {"name": "2. 로스 카메룬 + 4H 상위 추세 필터", "4h": True,  "jev": False, "th": 0},
        {"name": "3. 로스 카메룬 + JEV 호가창 게이트 (Threshold 0.60)", "4h": False, "jev": True,  "th": 0.60},
        {"name": "4. 로스 카메룬 + 4H 추세 + JEV 게이트 (하이브리드)", "4h": True,  "jev": True,  "th": 0.60},
    ]
    
    results = []
    for sc in scenarios:
        res = simulate_pipeline(data_1h, data_4h, use_4h_trend=sc["4h"], use_jev_gate=sc["jev"], jev_thresh=sc["th"])
        res["name"] = sc["name"]
        results.append(res)
        
    print("=" * 88)
    print(f"{'시나리오 모델':<40} | {'거래수':<6} | {'승률':<8} | {'PF':<6} | {'누적손익(USDT)':<14} | {'수익률':<8}")
    print("-" * 88)
    for r in results:
        print(f"{r['name']:<40} | {r['trades']:>4d}회 | {r['win_rate']:>5.1f}% | {r['pf']:>4.2f} | {r['pnl']:>+10.2f} U | {r['ret']:>+6.2f}%")
    print("=" * 88)

if __name__ == "__main__":
    asyncio.run(main())
