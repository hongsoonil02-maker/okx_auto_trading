# -*- coding: utf-8 -*-
"""
backtest_cameron_pullback.py — Full Walk-Forward Backtest for Ross Cameron Small-Cap Pullback Strategy
Tests across multi-day Alpaca 1-minute historical data for volatile penny runners.
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

# ── STRATEGY CONFIG ──
START_DATE = "2026-09-25T13:30:00Z"
END_DATE = "2026-10-09T20:00:00Z"
INITIAL_CAPITAL = 100000.0
POSITION_SIZE_PCT = 0.20       # 20% equity per trade ($20k)
SLIPPAGE_PCT = 0.005           # 0.5% realistic penny slippage
COMMISSION_PER_SHARE = 0.005   # $0.005/share
RISK_REWARD = 2.0              # 1:2 R:R
MAX_SL_PCT = 0.06              # 6% max risk
MIN_SL_PCT = 0.008             # 0.8% min risk

def get_historical_bars(symbols, start, end):
    bars_dict = {}
    for i in range(0, len(symbols), 5):
        chunk = symbols[i:i+5]
        sym_str = ",".join(chunk)
        url = f"{ALPACA_DATA_URL}/stocks/bars"
        params = {
            "symbols": sym_str,
            "timeframe": "1Min",
            "start": start,
            "end": end,
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
                        bars_dict[sym] = df
        except Exception as e:
            print(f"Error fetching {chunk}: {e}")
    return bars_dict

def run_backtest():
    print("=" * 80)
    print("🚀 [Ross Cameron 1-Min First Pullback Strategy] 정밀 백테스트")
    print(f"   초기 자본: ${INITIAL_CAPITAL:,.2f} | R:R 목표: 1:{RISK_REWARD} | 슬리피지: {SLIPPAGE_PCT*100}%")
    print(f"   테스트 구간: {START_DATE[:10]} ~ {END_DATE[:10]} | 1회 투입: {POSITION_SIZE_PCT*100}%")
    print("=" * 80)

    # Scan volatile penny runners
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

    print(f"🔍 대상 소형주 후보군: {len(symbols)}개 종목 데이터 수신 중...")
    bars_data = get_historical_bars(symbols, START_DATE, END_DATE)
    
    # Filter tickers with trade activity and price between $1.5 and $25
    valid_bars = {}
    for sym, df in bars_data.items():
        if len(df) >= 30 and 1.5 <= df["close"].mean() <= 25.0:
            valid_bars[sym] = df

    print(f"✅ 백테스트 유효 종목: {len(valid_bars)}개 ({list(valid_bars.keys())})")
    print("-" * 80)

    capital = INITIAL_CAPITAL
    trades = []
    
    for sym, df in valid_bars.items():
        in_position = False
        entry_idx = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0
        qty = 0

        for i in range(10, len(df)):
            curr_bar = df.iloc[i]
            prev_bar = df.iloc[i-1]
            t = curr_bar["time"]

            # Manage open trade
            if in_position:
                # 1. Take Profit hit
                if curr_bar["high"] >= take_profit:
                    exit_price = take_profit * (1.0 - SLIPPAGE_PCT)
                    pnl_dollar = (exit_price - entry_price) * qty - (qty * COMMISSION_PER_SHARE * 2)
                    pnl_pct = (exit_price / entry_price - 1) * 100
                    capital += pnl_dollar
                    trades.append({
                        "symbol": sym,
                        "entry_time": df.iloc[entry_idx]["time"],
                        "exit_time": t,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "qty": qty,
                        "pnl_dollar": pnl_dollar,
                        "pnl_pct": pnl_pct,
                        "hold_bars": i - entry_idx,
                        "reason": "TAKE_PROFIT (1:2 R:R)"
                    })
                    in_position = False
                    continue

                # 2. Stop Loss hit
                elif curr_bar["low"] <= stop_loss:
                    exit_price = stop_loss * (1.0 - SLIPPAGE_PCT)
                    pnl_dollar = (exit_price - entry_price) * qty - (qty * COMMISSION_PER_SHARE * 2)
                    pnl_pct = (exit_price / entry_price - 1) * 100
                    capital += pnl_dollar
                    trades.append({
                        "symbol": sym,
                        "entry_time": df.iloc[entry_idx]["time"],
                        "exit_time": t,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "qty": qty,
                        "pnl_dollar": pnl_dollar,
                        "pnl_pct": pnl_pct,
                        "hold_bars": i - entry_idx,
                        "reason": "STOP_LOSS (Swing Low)"
                    })
                    in_position = False
                    continue

                # 3. Time Stop (after 30 bars / 30 mins)
                elif (i - entry_idx) >= 30:
                    exit_price = curr_bar["close"] * (1.0 - SLIPPAGE_PCT)
                    pnl_dollar = (exit_price - entry_price) * qty - (qty * COMMISSION_PER_SHARE * 2)
                    pnl_pct = (exit_price / entry_price - 1) * 100
                    capital += pnl_dollar
                    trades.append({
                        "symbol": sym,
                        "entry_time": df.iloc[entry_idx]["time"],
                        "exit_time": t,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "qty": qty,
                        "pnl_dollar": pnl_dollar,
                        "pnl_pct": pnl_pct,
                        "hold_bars": i - entry_idx,
                        "reason": "TIME_EXIT (30m Cap)"
                    })
                    in_position = False
                    continue

            # Check new entry setup
            if not in_position:
                # Regular morning hours (09:30 to 11:30 EST -> 13:30 to 15:30 UTC)
                if not (13 <= t.hour <= 15):
                    continue

                # Setup: Previous bar must be a pullback (red candle or narrow doji)
                is_pullback = prev_bar["close"] <= prev_bar["open"] * 1.002
                if not is_pullback:
                    continue

                # Trend Support: Must not collapse below 9 EMA by more than 1.5%
                if prev_bar["close"] < prev_bar["ema9"] * 0.985:
                    continue

                # Impulse wave in last 3~7 bars
                prior_bars = df.iloc[max(0, i-6):i-1]
                surge_low = prior_bars["low"].min()
                surge_high = prior_bars["high"].max()
                surge = surge_high - surge_low

                # At least 1.5% surge
                if surge <= 0 or (surge / surge_low) < 0.015:
                    continue

                # Retracement <= 60%
                pullback_retrace = surge_high - prev_bar["low"]
                retrace_ratio = pullback_retrace / surge
                if retrace_ratio > 0.60:
                    continue

                # Breakout Trigger: Current high exceeds previous bar high
                trigger_price = prev_bar["high"]
                if curr_bar["high"] >= trigger_price:
                    sl = prev_bar["low"]
                    risk = trigger_price - sl
                    risk_pct = risk / trigger_price

                    if risk_pct > MAX_SL_PCT or risk_pct < MIN_SL_PCT:
                        continue

                    # Fill with slippage
                    entry_price = trigger_price * (1.0 + SLIPPAGE_PCT)
                    stop_loss = sl
                    take_profit = trigger_price + (risk * RISK_REWARD)
                    
                    alloc = capital * POSITION_SIZE_PCT
                    qty = math.floor(alloc / entry_price)

                    if qty > 0:
                        in_position = True
                        entry_idx = i

    trades_df = pd.DataFrame(trades)
    print("\n📊 [로스 카메론 1분봉 눌림목 백테스트 성과 분석표]")
    print("=" * 80)
    if len(trades_df) == 0:
        print("⚠️ 조건에 맞는 거래가 검출되지 않았습니다.")
        return

    wins = trades_df[trades_df["pnl_dollar"] > 0]
    losses = trades_df[trades_df["pnl_dollar"] <= 0]
    win_rate = (len(wins) / len(trades_df)) * 100
    total_pnl = trades_df["pnl_dollar"].sum()
    final_equity = INITIAL_CAPITAL + total_pnl
    return_pct = (total_pnl / INITIAL_CAPITAL) * 100
    
    gross_profit = wins["pnl_dollar"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl_dollar"].sum()) if len(losses) > 0 else 1e-6
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    print(f"• 초기 자본           : ${INITIAL_CAPITAL:,.2f}")
    print(f"• 최종 자산 (Equity)   : ${final_equity:,.2f}")
    print(f"• 총 순수익 (Return)   : ${total_pnl:+,.2f} ({return_pct:+.2f}%)")
    print(f"• 총 거래 횟수        : {len(trades_df)}회 (승: {len(wins)}회 / 패: {len(losses)}회)")
    print(f"• 승률 (Win Rate)     : {win_rate:.1f}%")
    print(f"• 손익비 (Profit Factor) : {profit_factor:.2f}")
    print(f"• 평균 승리 수익률     : +{wins['pnl_pct'].mean():.2f}%" if len(wins) > 0 else "• 평균 승리 수익률 : N/A")
    print(f"• 평균 손실률         : {losses['pnl_pct'].mean():.2f}%" if len(losses) > 0 else "• 평균 손실률 : N/A")
    print(f"• 평균 보유 시간       : {trades_df['hold_bars'].mean():.1f} 분")
    print("-" * 80)
    print("📌 [청산 사유별 비중]")
    print(trades_df["reason"].value_counts().to_string())
    print("-" * 80)
    print("📝 [최근 매매 상세 로그 (Top 10)]")
    cols = ["symbol", "entry_time", "exit_time", "entry_price", "exit_price", "pnl_pct", "pnl_dollar", "reason"]
    print(trades_df[cols].tail(10).to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    run_backtest()
