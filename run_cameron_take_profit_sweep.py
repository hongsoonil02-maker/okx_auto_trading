# -*- coding: utf-8 -*-
"""
run_cameron_take_profit_sweep.py — Multi-Scenario Take-Profit & Split Exit Backtest
Compares:
1. R:R = 1:2.0 (Baseline)
2. R:R = 1:2.5
3. R:R = 1:3.0
4. R:R = 1:3.5
5. R:R = 1:4.0
6. Split Exit (50% at 1:1.5 with Stop moved to Break-even, 50% at 1:3.0)
7. Split Exit (50% at 1:2.0 with Stop to Break-even, 50% at 1:4.0)

All scenarios include realistic slippage (0.5%), commission ($0.005/share), and exact 1-minute bar mechanics.
"""

import os
import sys
import math
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "PKVGU7IXDJMAF6JBWWVMP2I6R3")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "CCnTinES4eWwkQqifr9879ASCcXSpm9kjuvbZnCKpHM3")
ALPACA_DATA_URL = "https://data.alpaca.markets/v2"

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
}

START_DATE = "2026-09-25T13:30:00Z"
END_DATE = "2026-10-09T20:00:00Z"
INITIAL_CAPITAL = 100000.0
POSITION_SIZE_PCT = 0.20
SLIPPAGE_PCT = 0.005           # 0.5%
COMMISSION_PER_SHARE = 0.005   # $0.005/share
MAX_SL_PCT = 0.06
MIN_SL_PCT = 0.008

def fetch_universe_data():
    r_movers = requests.get("https://data.alpaca.markets/v1beta1/screener/stocks/movers?top=30", headers=HEADERS).json().get("gainers", [])
    r_actives = requests.get("https://data.alpaca.markets/v1beta1/screener/stocks/most-actives?by=volume&top=30", headers=HEADERS).json().get("most_actives", [])
    
    symbols = []
    for g in r_movers:
        sym = g.get("symbol", "")
        if 2 <= len(sym) <= 5 and not sym.endswith("W") and sym.isalpha():
            symbols.append(sym)
    for a in r_actives:
        sym = a.get("symbol", "")
        if 2 <= len(sym) <= 5 and sym not in symbols and sym.isalpha():
            symbols.append(sym)

    bars_dict = {}
    for i in range(0, len(symbols), 5):
        chunk = symbols[i:i+5]
        sym_str = ",".join(chunk)
        url = f"{ALPACA_DATA_URL}/stocks/bars"
        params = {
            "symbols": sym_str,
            "timeframe": "1Min",
            "start": START_DATE,
            "end": END_DATE,
            "limit": 10000,
            "feed": "iex"
        }
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json().get("bars", {})
                for sym, b_list in data.items():
                    if b_list and len(b_list) >= 20:
                        df = pd.DataFrame(b_list)
                        df.rename(columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap"}, inplace=True)
                        if "vwap" not in df.columns or df["vwap"].isna().all():
                            tp = (df["high"] + df["low"] + df["close"]) / 3
                            df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, 1)
                        df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
                        df["time"] = pd.to_datetime(df["time"])
                        if 1.5 <= df["close"].mean() <= 25.0:
                            bars_dict[sym] = df
        except Exception as e:
            pass
    return bars_dict

def simulate_strategy(bars_data, rr_ratio: float, split_exit: bool = False, split_tp1_mult: float = 1.5, split_tp2_mult: float = 3.0):
    capital = INITIAL_CAPITAL
    trades = []
    
    for sym, df in bars_data.items():
        in_position = False
        entry_idx = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        qty = 0
        # Split exit state
        qty_rem = 0
        tp1_hit = False
        tp1_price = 0.0
        tp2_price = 0.0

        for i in range(10, len(df)):
            curr_bar = df.iloc[i]
            prev_bar = df.iloc[i-1]
            t = curr_bar["time"]

            # Manage position
            if in_position:
                if not split_exit:
                    # Single target mode
                    if curr_bar["high"] >= take_profit:
                        exit_price = take_profit * (1.0 - SLIPPAGE_PCT)
                        pnl_dollar = (exit_price - entry_price) * qty - (qty * COMMISSION_PER_SHARE * 2)
                        pnl_pct = (exit_price / entry_price - 1) * 100
                        capital += pnl_dollar
                        trades.append({"symbol": sym, "pnl_dollar": pnl_dollar, "pnl_pct": pnl_pct, "reason": "TP", "hold": i - entry_idx})
                        in_position = False
                        continue
                    elif curr_bar["low"] <= stop_loss:
                        exit_price = stop_loss * (1.0 - SLIPPAGE_PCT)
                        pnl_dollar = (exit_price - entry_price) * qty - (qty * COMMISSION_PER_SHARE * 2)
                        pnl_pct = (exit_price / entry_price - 1) * 100
                        capital += pnl_dollar
                        trades.append({"symbol": sym, "pnl_dollar": pnl_dollar, "pnl_pct": pnl_pct, "reason": "SL", "hold": i - entry_idx})
                        in_position = False
                        continue
                    elif (i - entry_idx) >= 30:
                        exit_price = curr_bar["close"] * (1.0 - SLIPPAGE_PCT)
                        pnl_dollar = (exit_price - entry_price) * qty - (qty * COMMISSION_PER_SHARE * 2)
                        pnl_pct = (exit_price / entry_price - 1) * 100
                        capital += pnl_dollar
                        trades.append({"symbol": sym, "pnl_dollar": pnl_dollar, "pnl_pct": pnl_pct, "reason": "TIME", "hold": i - entry_idx})
                        in_position = False
                        continue
                else:
                    # Split exit mode
                    # Check TP1
                    if not tp1_hit and curr_bar["high"] >= tp1_price:
                        tp1_hit = True
                        half_qty = math.floor(qty / 2)
                        qty_rem = qty - half_qty
                        exit_price = tp1_price * (1.0 - SLIPPAGE_PCT)
                        pnl_p1 = (exit_price - entry_price) * half_qty - (half_qty * COMMISSION_PER_SHARE * 2)
                        capital += pnl_p1
                        # Move stop loss to breakeven + slippage buffer
                        stop_loss = entry_price * 1.002

                    # Check TP2 for remaining
                    if tp1_hit:
                        if curr_bar["high"] >= tp2_price:
                            exit_price = tp2_price * (1.0 - SLIPPAGE_PCT)
                            pnl_p2 = (exit_price - entry_price) * qty_rem - (qty_rem * COMMISSION_PER_SHARE * 2)
                            capital += pnl_p2
                            total_pnl = pnl_p1 + pnl_p2
                            trades.append({"symbol": sym, "pnl_dollar": total_pnl, "pnl_pct": (total_pnl / (qty * entry_price)) * 100, "reason": "TP1+TP2", "hold": i - entry_idx})
                            in_position = False
                            continue
                        elif curr_bar["low"] <= stop_loss:
                            exit_price = stop_loss * (1.0 - SLIPPAGE_PCT)
                            pnl_p2 = (exit_price - entry_price) * qty_rem - (qty_rem * COMMISSION_PER_SHARE * 2)
                            capital += pnl_p2
                            total_pnl = pnl_p1 + pnl_p2
                            trades.append({"symbol": sym, "pnl_dollar": total_pnl, "pnl_pct": (total_pnl / (qty * entry_price)) * 100, "reason": "TP1+BE", "hold": i - entry_idx})
                            in_position = False
                            continue
                    else:
                        # TP1 not hit, check original stop loss
                        if curr_bar["low"] <= stop_loss:
                            exit_price = stop_loss * (1.0 - SLIPPAGE_PCT)
                            pnl_dollar = (exit_price - entry_price) * qty - (qty * COMMISSION_PER_SHARE * 2)
                            capital += pnl_dollar
                            trades.append({"symbol": sym, "pnl_dollar": pnl_dollar, "pnl_pct": (exit_price / entry_price - 1) * 100, "reason": "SL", "hold": i - entry_idx})
                            in_position = False
                            continue

                    # Time stop
                    if (i - entry_idx) >= 30:
                        exit_price = curr_bar["close"] * (1.0 - SLIPPAGE_PCT)
                        cur_qty = qty_rem if tp1_hit else qty
                        pnl_time = (exit_price - entry_price) * cur_qty - (cur_qty * COMMISSION_PER_SHARE * 2)
                        capital += pnl_time
                        total_pnl = (pnl_p1 + pnl_time) if tp1_hit else pnl_time
                        trades.append({"symbol": sym, "pnl_dollar": total_pnl, "pnl_pct": (total_pnl / (qty * entry_price)) * 100, "reason": "TIME", "hold": i - entry_idx})
                        in_position = False
                        continue

            # Check setup
            if not in_position:
                if not (13 <= t.hour <= 15):
                    continue

                if prev_bar["close"] > prev_bar["open"] * 1.002:
                    continue

                if prev_bar["close"] < prev_bar["ema9"] * 0.985:
                    continue

                prior_bars = df.iloc[max(0, i-6):i-1]
                surge_low = prior_bars["low"].min()
                surge_high = prior_bars["high"].max()
                surge = surge_high - surge_low

                if surge <= 0 or (surge / surge_low) < 0.015:
                    continue

                pullback_retrace = surge_high - prev_bar["low"]
                if (pullback_retrace / surge) > 0.60:
                    continue

                trigger_price = prev_bar["high"]
                if curr_bar["high"] >= trigger_price:
                    sl = prev_bar["low"]
                    risk = trigger_price - sl
                    risk_pct = risk / trigger_price

                    if risk_pct > MAX_SL_PCT or risk_pct < MIN_SL_PCT:
                        continue

                    entry_price = trigger_price * (1.0 + SLIPPAGE_PCT)
                    stop_loss = sl
                    
                    if not split_exit:
                        take_profit = trigger_price + (risk * rr_ratio)
                    else:
                        tp1_price = trigger_price + (risk * split_tp1_mult)
                        tp2_price = trigger_price + (risk * split_tp2_mult)
                        tp1_hit = False

                    alloc = capital * POSITION_SIZE_PCT
                    qty = math.floor(alloc / entry_price)

                    if qty > 0:
                        in_position = True
                        entry_idx = i

    df_res = pd.DataFrame(trades)
    if len(df_res) == 0:
        return {"return_pct": 0, "win_rate": 0, "profit_factor": 0, "trades": 0, "net_pnl": 0}

    wins = df_res[df_res["pnl_dollar"] > 0]
    losses = df_res[df_res["pnl_dollar"] <= 0]
    win_rate = (len(wins) / len(df_res)) * 100
    total_pnl = df_res["pnl_dollar"].sum()
    gross_win = wins["pnl_dollar"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl_dollar"].sum()) if len(losses) > 0 else 1e-6
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    
    return {
        "final_equity": capital,
        "net_pnl": total_pnl,
        "return_pct": (total_pnl / INITIAL_CAPITAL) * 100,
        "total_trades": len(df_res),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "profit_factor": pf,
        "avg_win_pct": wins["pnl_pct"].mean() if len(wins) > 0 else 0,
        "avg_loss_pct": losses["pnl_pct"].mean() if len(losses) > 0 else 0,
        "reasons": df_res["reason"].value_counts().to_dict()
    }

def main():
    print("=" * 85)
    print("🔬 [로스 카메론 전략] 익절 비율 & 분할 익절 시나리오 비교 백테스트")
    print(f"   자본: ${INITIAL_CAPITAL:,.2f} | 슬리피지: {SLIPPAGE_PCT*100}% | 수수료: ${COMMISSION_PER_SHARE}/주")
    print("=" * 85)

    bars_data = fetch_universe_data()
    print(f"✅ 수신 완료: {len(bars_data)}개 종목 데이터로 전체 시나리오 일괄 시뮬레이션 시작...\n")

    scenarios = [
        {"name": "시나리오 1: 단일 R:R 1:2.0 (기본형)", "rr": 2.0, "split": False},
        {"name": "시나리오 2: 단일 R:R 1:2.5", "rr": 2.5, "split": False},
        {"name": "시나리오 3: 단일 R:R 1:3.0", "rr": 3.0, "split": False},
        {"name": "시나리오 4: 단일 R:R 1:3.5", "rr": 3.5, "split": False},
        {"name": "시나리오 5: 단일 R:R 1:4.0", "rr": 4.0, "split": False},
        {"name": "시나리오 6: 분할 익절 A (50% @ 1:1.5 + 스탑본절, 50% @ 1:3.0)", "rr": 0, "split": True, "tp1": 1.5, "tp2": 3.0},
        {"name": "시나리오 7: 분할 익절 B (50% @ 1:2.0 + 스탑본절, 50% @ 1:4.0)", "rr": 0, "split": True, "tp1": 2.0, "tp2": 4.0},
    ]

    results = []
    for s in scenarios:
        if not s["split"]:
            res = simulate_strategy(bars_data, rr_ratio=s["rr"])
        else:
            res = simulate_strategy(bars_data, rr_ratio=0, split_exit=True, split_tp1_mult=s["tp1"], split_tp2_mult=s["tp2"])
        res["scenario"] = s["name"]
        results.append(res)

    # Display comparison table
    print(f"{'시나리오':<48} | {'순손익($)':<12} | {'수익률':<8} | {'승률':<8} | {'PF':<6} | {'거래수':<6}")
    print("-" * 96)
    for r in results:
        pnl_str = f"${r['net_pnl']:+,.1f}"
        ret_str = f"{r['return_pct']:+.2f}%"
        win_str = f"{r['win_rate']:.1f}%"
        pf_str = f"{r['profit_factor']:.2f}"
        trade_str = f"{r['total_trades']}회"
        print(f"{r['scenario']:<48} | {pnl_str:<12} | {ret_str:<8} | {win_str:<8} | {pf_str:<6} | {trade_str:<6}")

    print("=" * 96)
    # Print detail of top performing scenario
    best = max(results, key=lambda x: x["profit_factor"])
    print(f"\n🏆 최고 성과 모델: [{best['scenario']}]")
    print(f"   • 순이익: ${best['net_pnl']:+,.2f} ({best['return_pct']:+.2f}%)")
    print(f"   • 손익비 (Profit Factor): {best['profit_factor']:.2f}")
    print(f"   • 승률: {best['win_rate']:.1f}% ({best['wins']}승 / {best['losses']}패)")
    print(f"   • 평균 승리 수익: +{best['avg_win_pct']:.2f}% | 평균 손실: {best['avg_loss_pct']:.2f}%")
    print(f"   • 청산 세부: {best['reasons']}")

if __name__ == "__main__":
    main()
