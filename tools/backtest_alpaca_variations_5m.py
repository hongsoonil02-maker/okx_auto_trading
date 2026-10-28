#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/backtest_alpaca_variations_5m.py
=============================================================================
High-Resolution 5-Minute Bar Comparative Matrix Backtest for Alpaca Golden Trader:
Testing All 8 Combinations of 3 Alpha Enhancements:
[1] Overnight Runner Holding (오버나잇 분할 스윙)
[2] Early Momentum Entry (진입 트리거 완화 / 랠리 초입 장악)
[3] Dynamic ATR Trailing Stop (추세 추종 트레일링 스탑)
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

SYMBOLS = ["TQQQ", "TSLA", "NVDA", "PLTR", "MSTR"]
BENCHMARK_SYM = "QQQ"
ALL_TICKERS = [BENCHMARK_SYM] + SYMBOLS

INITIAL_EQUITY = 100_000.0
MAX_POSITIONS = 3
ALLOC_PER_POS = 0.30       # 30% per trade (Max 90% capital utilization)
FEE_SLIPPAGE = 0.0005      # 5 bps per trade
BASE_SL_PCT = 0.024        # 2.4% initial stop loss
BASE_TP_PCT = 0.080        # 8.0% fixed take profit

def load_data(period="60d", interval="5m"):
    print(f"📥 [Data Loader] Downloading 5-Min bars for {ALL_TICKERS} (period={period})...")
    data = yf.download(ALL_TICKERS, period=period, interval=interval, progress=False, group_by="ticker")
    clean = {}
    for sym in ALL_TICKERS:
        try:
            df = data[sym].dropna(subset=["Close"]).copy()
            if len(df) < 200:
                continue
            c = df["Close"]
            df["ema9"] = c.ewm(span=9, adjust=False).mean()
            df["ema21"] = c.ewm(span=21, adjust=False).mean()
            df["ema50"] = c.ewm(span=50, adjust=False).mean()
            
            delta = c.diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / (loss + 1e-9)
            df["rsi"] = 100 - (100 / (1 + rs))
            
            h, l, c_prev = df["High"], df["Low"], c.shift(1)
            tr = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
            df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
            
            clean[sym] = df
        except Exception as e:
            print(f"⚠️ Error {sym}: {e}")

    ref_idx = clean[BENCHMARK_SYM].index
    aligned = {}
    for sym in clean:
        aligned[sym] = clean[sym].reindex(ref_idx).ffill().bfill()
        
    print(f"✅ Data aligned: {len(ref_idx)} 5-Min bars ({ref_idx[0].strftime('%Y-%m-%d')} ~ {ref_idx[-1].strftime('%Y-%m-%d')})\n")
    return aligned, ref_idx

class GoldenBacktest:
    def __init__(self, data, timeline, overnight: bool, early_entry: bool, trailing_stop: bool):
        self.data = data
        self.timeline = timeline
        self.overnight = overnight
        self.early_entry = early_entry
        self.trailing_stop = trailing_stop

    def run(self):
        cash = INITIAL_EQUITY
        positions = {}
        equity_curve = []
        trades = []

        qqq = self.data[BENCHMARK_SYM]

        for i in range(50, len(self.timeline)):
            t = self.timeline[i]
            is_eod = False
            # Check if next bar is on next date or last bar
            if i < len(self.timeline) - 1:
                next_t = self.timeline[i + 1]
                if next_t.date() != t.date():
                    is_eod = True
            else:
                is_eod = True

            # 1. Update valuation and manage open positions
            open_val = 0.0
            to_remove = []

            for sym, pos in list(positions.items()):
                df = self.data[sym]
                high = float(df["High"].iloc[i])
                low = float(df["Low"].iloc[i])
                close = float(df["Close"].iloc[i])
                atr = float(df["atr"].iloc[i])

                if high > pos["peak"]:
                    pos["peak"] = high

                # Dynamic Trailing Stop update
                if self.trailing_stop:
                    gain_pct = (pos["peak"] - pos["entry"]) / pos["entry"]
                    if gain_pct >= 0.020:  # If gained >= 2.0%, activate ATR trail
                        trail_sl = pos["peak"] - (2.0 * atr)
                        if trail_sl > pos["sl"]:
                            pos["sl"] = trail_sl

                # Check Stop Loss
                if low <= pos["sl"]:
                    exit_px = pos["sl"] * (1 - FEE_SLIPPAGE)
                    pnl = (exit_px - pos["entry"]) * pos["shares"]
                    pnl_pct = (exit_px - pos["entry"]) / pos["entry"]
                    cash += pos["shares"] * exit_px
                    trades.append({"sym": sym, "type": "SL", "pnl": pnl, "pct": pnl_pct})
                    to_remove.append(sym)
                    continue

                # Check Fixed Take Profit (if trailing stop not active)
                if not self.trailing_stop and high >= pos["tp"]:
                    exit_px = pos["tp"] * (1 - FEE_SLIPPAGE)
                    pnl = (exit_px - pos["entry"]) * pos["shares"]
                    pnl_pct = (exit_px - pos["entry"]) / pos["entry"]
                    cash += pos["shares"] * exit_px
                    trades.append({"sym": sym, "type": "TP", "pnl": pnl, "pct": pnl_pct})
                    to_remove.append(sym)
                    continue

                # EOD Logic (Market Close)
                if is_eod:
                    if not self.overnight:
                        # Baseline: Flatten everything
                        exit_px = close * (1 - FEE_SLIPPAGE)
                        pnl = (exit_px - pos["entry"]) * pos["shares"]
                        pnl_pct = (exit_px - pos["entry"]) / pos["entry"]
                        cash += pos["shares"] * exit_px
                        trades.append({"sym": sym, "type": "EOD_FLAT", "pnl": pnl, "pct": pnl_pct})
                        to_remove.append(sym)
                        continue
                    else:
                        # Feature 1: Overnight Runner
                        cur_gain = (close - pos["entry"]) / pos["entry"]
                        if cur_gain >= 0.015 and not pos["half_taken"]:
                            # Lock 50% profit, hold remaining 50% overnight with Breakeven SL
                            half = int(pos["shares"] * 0.5)
                            if half > 0:
                                exit_px = close * (1 - FEE_SLIPPAGE)
                                pnl = (exit_px - pos["entry"]) * half
                                pnl_pct = (exit_px - pos["entry"]) / pos["entry"]
                                cash += half * exit_px
                                pos["shares"] -= half
                                pos["half_taken"] = True
                                pos["sl"] = max(pos["sl"], pos["entry"] * 1.002)
                                trades.append({"sym": sym, "type": "HALF_TP", "pnl": pnl, "pct": pnl_pct})
                        elif cur_gain < 0.005:
                            # If not clearly in profit, close before overnight risk
                            exit_px = close * (1 - FEE_SLIPPAGE)
                            pnl = (exit_px - pos["entry"]) * pos["shares"]
                            pnl_pct = (exit_px - pos["entry"]) / pos["entry"]
                            cash += pos["shares"] * exit_px
                            trades.append({"sym": sym, "type": "EOD_CUT", "pnl": pnl, "pct": pnl_pct})
                            to_remove.append(sym)
                            continue

                open_val += pos["shares"] * close

            for sym in to_remove:
                positions.pop(sym, None)

            # 2. Check Entries
            if len(positions) < MAX_POSITIONS and not is_eod:
                qqq_c = float(qqq["Close"].iloc[i])
                qqq_e50 = float(qqq["ema50"].iloc[i])
                qqq_e9 = float(qqq["ema9"].iloc[i])
                qqq_e21 = float(qqq["ema21"].iloc[i])
                macro_bull = (qqq_c > qqq_e50) and (qqq_e9 > qqq_e21)

                slots = MAX_POSITIONS - len(positions)
                for sym in SYMBOLS:
                    if sym in positions or slots <= 0:
                        continue
                    df = self.data[sym]
                    c = float(df["Close"].iloc[i])
                    e9 = float(df["ema9"].iloc[i])
                    e21 = float(df["ema21"].iloc[i])
                    rsi = float(df["rsi"].iloc[i])
                    atr = float(df["atr"].iloc[i])

                    entry = False
                    if not self.early_entry:
                        # Baseline: Rigid entry condition (Macro Bullish + EMA9 > EMA21 + close > EMA9 + RSI > 52)
                        if macro_bull and (e9 > e21) and (c > e9) and (rsi > 52.0):
                            entry = True
                    else:
                        # Feature 2: Early Momentum Entry (Captures early rally wave, RSI > 48, close > EMA21)
                        if (e9 > e21) and (c > e21) and (48.0 <= rsi <= 75.0):
                            entry = True

                    if entry:
                        cur_eq = cash + open_val
                        alloc = cur_eq * ALLOC_PER_POS
                        buy_px = c * (1 + FEE_SLIPPAGE)
                        shares = int(alloc / buy_px)
                        cost = shares * buy_px
                        if shares > 0 and cash >= cost:
                            cash -= cost
                            sl_px = buy_px * (1 - BASE_SL_PCT)
                            tp_px = buy_px * (1 + BASE_TP_PCT)
                            positions[sym] = {
                                "shares": shares, "entry": buy_px, "peak": buy_px,
                                "sl": sl_px, "tp": tp_px, "half_taken": False
                            }
                            open_val += shares * c
                            slots -= 1

            total_eq = cash + open_val
            equity_curve.append(total_eq)

        # Final close
        for sym, pos in list(positions.items()):
            c = float(self.data[sym]["Close"].iloc[-1])
            exit_px = c * (1 - FEE_SLIPPAGE)
            pnl = (exit_px - pos["entry"]) * pos["shares"]
            cash += pos["shares"] * exit_px
            trades.append({"sym": sym, "type": "FINAL", "pnl": pnl, "pct": (exit_px - pos["entry"]) / pos["entry"]})
        positions.clear()

        # Stats
        s = pd.Series(equity_curve)
        peak = s.cummax()
        dd = (s - peak) / peak
        mdd_pct = abs(float(dd.min())) * 100

        total_ret_pct = ((cash - INITIAL_EQUITY) / INITIAL_EQUITY) * 100
        n_trades = len(trades)
        if n_trades > 0:
            tdf = pd.DataFrame(trades)
            wins = tdf[tdf["pnl"] > 0]
            losses = tdf[tdf["pnl"] <= 0]
            win_rate = (len(wins) / n_trades) * 100
            gross_win = wins["pnl"].sum() if len(wins) > 0 else 0.0
            gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 1e-9
            pf = gross_win / gross_loss if gross_loss > 0 else 99.0
            avg_pnl_pct = tdf["pct"].mean() * 100
        else:
            win_rate, pf, avg_pnl_pct = 0.0, 0.0, 0.0

        return {
            "total_return_pct": total_ret_pct,
            "mdd_pct": mdd_pct,
            "win_rate": win_rate,
            "profit_factor": pf,
            "trades": n_trades,
            "avg_trade_ret_pct": avg_pnl_pct,
            "final_equity": cash
        }

def run_suite():
    data, timeline = load_data(period="60d", interval="5m")

    qqq_start = float(data[BENCHMARK_SYM]["Close"].iloc[50])
    qqq_end = float(data[BENCHMARK_SYM]["Close"].iloc[-1])
    qqq_ret = ((qqq_end - qqq_start) / qqq_start) * 100

    scenarios = [
        # (Name, Overnight, EarlyEntry, TrailingStop)
        ("1. [Baseline] 현재 당일청산 원본", False, False, False),
        
        # 1개씩 적용
        ("2. [단일] A. 오버나잇 스윙만", True, False, False),
        ("3. [단일] B. 조기 모멘텀 진입만", False, True, False),
        ("4. [단일] C. ATR 트레일링 스탑만", False, False, True),
        
        # 2개 조합
        ("5. [2개조합] A+B (오버나잇 + 조기진입)", True, True, False),
        ("6. [2개조합] A+C (오버나잇 + 트레일링)", True, False, True),
        ("7. [2개조합] B+C (조기진입 + 트레일링)", False, True, True),
        
        # 3개 전부 적용
        ("8. [3개전체] A+B+C (트리플 풀부스터)", True, True, True),
    ]

    print("=" * 95)
    print(f"📊 [알파카 골든트레이더 5분봉] 3대 개선변수 전수 백테스트 (최근 60일, {len(timeline)}개 봉)")
    print(f"📌 벤치마크 나스닥 (QQQ Buy & Hold) 수익률: {qqq_ret:+.2f}%")
    print("=" * 95)

    results = []
    for name, r1, r2, r3 in scenarios:
        engine = GoldenBacktest(data, timeline, overnight=r1, early_entry=r2, trailing_stop=r3)
        res = engine.run()
        alpha = res["total_return_pct"] - qqq_ret
        res["name"] = name
        res["alpha"] = alpha
        results.append(res)
        print(f"{name:<38} | 수익률: {res['total_return_pct']:>+6.2f}% (알파: {alpha:>+6.2f}%) | MDD: {res['mdd_pct']:>5.2f}% | 승률: {res['win_rate']:>5.1f}% | PF: {res['profit_factor']:>4.2f} | 매매: {res['trades']:>3}회")

    print("=" * 95)
    return results, qqq_ret

if __name__ == "__main__":
    run_suite()
