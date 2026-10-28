#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/backtest_alpaca_3enhancements.py
=============================================================================
Comparative Matrix Backtest for Alpaca Nasdaq Trading:
Evaluating 3 Core Enhancements across All Combinations (2^3 = 8 scenarios)
1. Overnight Runner (오버나잇 분할 스윙 홀딩)
2. Relaxed Trend Entry (1H 추세 진입 완화 / 랠리 초입 포착)
3. ATR Dynamic Trailing Stop (추세 추종 트레일링 스탑)
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

SYMBOLS = ["TQQQ", "TSLA", "NVDA", "PLTR", "MSTR"]
BENCHMARK_SYM = "QQQ"
ALL_TICKERS = [BENCHMARK_SYM] + SYMBOLS

INITIAL_EQUITY = 100_000.0
MAX_POSITIONS = 3
ALLOC_PER_POS = 0.30       # 30% per trade (Max 90% gross)
SLIPPAGE_FEE_PCT = 0.0005  # 5 bps per trade
BASE_SL_PCT = 0.024        # 2.4% initial stop loss
BASE_TP_PCT = 0.080        # 8.0% fixed target (Baseline)

def calc_hma(series: pd.Series, length: int = 100) -> pd.Series:
    def wma(s, l):
        weights = np.arange(1, l + 1)
        return s.rolling(l).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    half_l = max(1, int(length / 2))
    sqrt_l = max(1, int(np.sqrt(length)))
    diff = 2 * wma(series, half_l) - wma(series, length)
    return wma(diff, sqrt_l)

def download_data(period="90d", interval="1h"):
    print(f"📥 [Data Loader] Downloading 1H bars for {ALL_TICKERS} (Period: {period})...")
    data = yf.download(ALL_TICKERS, period=period, interval=interval, progress=False, group_by="ticker")
    
    clean = {}
    for sym in ALL_TICKERS:
        try:
            df = data[sym].dropna(subset=["Close"]).copy()
            if len(df) < 60:
                continue
            df["datetime"] = df.index
            c = df["Close"]
            df["ema20"] = c.ewm(span=20, adjust=False).mean()
            df["ema50"] = c.ewm(span=50, adjust=False).mean()
            df["hma100"] = calc_hma(c, 100)
            df["hma50"] = calc_hma(c, 50)
            
            # MACD
            fast_ema = c.ewm(span=12, adjust=False).mean()
            slow_ema = c.ewm(span=26, adjust=False).mean()
            df["macd"] = fast_ema - slow_ema
            df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
            
            # RSI
            delta = c.diff()
            gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / (loss + 1e-9)
            df["rsi"] = 100 - (100 / (1 + rs))
            
            # ATR 14
            h, l, c_prev = df["High"], df["Low"], c.shift(1)
            tr = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
            df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
            
            # Volume MA20
            df["vol_ma20"] = df["Volume"].rolling(20).mean()
            
            clean[sym] = df
        except Exception as e:
            print(f"⚠️ Error preparing {sym}: {e}")
            
    # Align on benchmark timeline
    ref_idx = clean[BENCHMARK_SYM].index
    aligned = {}
    for sym in clean:
        aligned[sym] = clean[sym].reindex(ref_idx).ffill().bfill()
        
    print(f"✅ Data ready: {len(ref_idx)} 1H bars ({ref_idx[0].strftime('%Y-%m-%d')} ~ {ref_idx[-1].strftime('%Y-%m-%d')})\n")
    return aligned, ref_idx

class BacktestEngine:
    def __init__(self, data, ref_idx, overnight_runner: bool, relaxed_entry: bool, trailing_stop: bool):
        self.data = data
        self.ref_idx = ref_idx
        self.overnight_runner = overnight_runner
        self.relaxed_entry = relaxed_entry
        self.trailing_stop = trailing_stop

    def run(self):
        equity = INITIAL_EQUITY
        cash = INITIAL_EQUITY
        positions = {}  # sym -> {qty, entry_price, peak_price, sl_price, tp_price, half_taken, is_runner}
        equity_curve = []
        trade_log = []

        for idx, t in enumerate(self.ref_idx):
            is_eod = False
            # Check if this is the last bar of the trading session (typically 15:30 or 19:00 UTC)
            if idx < len(self.ref_idx) - 1:
                next_t = self.ref_idx[idx + 1]
                if next_t.date() != t.date():
                    is_eod = True
            else:
                is_eod = True

            # 1. Update current valuation and manage open positions
            open_pos_value = 0.0
            closed_syms = []

            for sym, pos in list(positions.items()):
                df = self.data[sym]
                bar = df.loc[t]
                high = float(bar["High"])
                low = float(bar["Low"])
                close = float(bar["Close"])
                atr = float(bar["atr"])

                # Update peak price for trailing stop
                if high > pos["peak_price"]:
                    pos["peak_price"] = high

                # Update trailing stop if enabled
                if self.trailing_stop:
                    # Activate trailing stop when gain >= 2.5%
                    gain_pct = (pos["peak_price"] - pos["entry_price"]) / pos["entry_price"]
                    if gain_pct >= 0.025:
                        trail_sl = pos["peak_price"] - (1.8 * atr)
                        if trail_sl > pos["sl_price"]:
                            pos["sl_price"] = trail_sl

                # Check Stop-Loss
                if low <= pos["sl_price"]:
                    exit_px = pos["sl_price"] * (1 - SLIPPAGE_FEE_PCT)
                    pnl = (exit_px - pos["entry_price"]) * pos["qty"]
                    pnl_pct = (exit_px - pos["entry_price"]) / pos["entry_price"]
                    cash += pos["qty"] * exit_px
                    trade_log.append({
                        "sym": sym, "type": "SL", "entry": pos["entry_price"],
                        "exit": exit_px, "pnl": pnl, "pnl_pct": pnl_pct, "date": t
                    })
                    closed_syms.append(sym)
                    continue

                # Check Baseline Take-Profit (if trailing stop not active)
                if not self.trailing_stop and high >= pos["tp_price"]:
                    exit_px = pos["tp_price"] * (1 - SLIPPAGE_FEE_PCT)
                    pnl = (exit_px - pos["entry_price"]) * pos["qty"]
                    pnl_pct = (exit_px - pos["entry_price"]) / pos["entry_price"]
                    cash += pos["qty"] * exit_px
                    trade_log.append({
                        "sym": sym, "type": "TP", "entry": pos["entry_price"],
                        "exit": exit_px, "pnl": pnl, "pnl_pct": pnl_pct, "date": t
                    })
                    closed_syms.append(sym)
                    continue

                # EOD Logic (Market Close)
                if is_eod:
                    if not self.overnight_runner:
                        # Baseline: Flatten everything at EOD
                        exit_px = close * (1 - SLIPPAGE_FEE_PCT)
                        pnl = (exit_px - pos["entry_price"]) * pos["qty"]
                        pnl_pct = (exit_px - pos["entry_price"]) / pos["entry_price"]
                        cash += pos["qty"] * exit_px
                        trade_log.append({
                            "sym": sym, "type": "EOD_CLOSE", "entry": pos["entry_price"],
                            "exit": exit_px, "pnl": pnl, "pnl_pct": pnl_pct, "date": t
                        })
                        closed_syms.append(sym)
                        continue
                    else:
                        # Feature 1: Overnight Runner
                        # If in profit >= +2.0% and haven't locked half yet -> lock 50%, hold runner
                        unrealized_pct = (close - pos["entry_price"]) / pos["entry_price"]
                        if unrealized_pct >= 0.020 and not pos["half_taken"]:
                            half_qty = int(pos["qty"] * 0.5)
                            if half_qty > 0:
                                exit_px = close * (1 - SLIPPAGE_FEE_PCT)
                                pnl = (exit_px - pos["entry_price"]) * half_qty
                                pnl_pct = (exit_px - pos["entry_price"]) / pos["entry_price"]
                                cash += half_qty * exit_px
                                pos["qty"] -= half_qty
                                pos["half_taken"] = True
                                # Move stop-loss to Breakeven (+0.2% to cover fees)
                                pos["sl_price"] = max(pos["sl_price"], pos["entry_price"] * 1.002)
                                pos["is_runner"] = True
                                trade_log.append({
                                    "sym": sym, "type": "HALF_TP", "entry": pos["entry_price"],
                                    "exit": exit_px, "pnl": pnl, "pnl_pct": pnl_pct, "date": t
                                })
                        elif unrealized_pct < 0.010:
                            # Not profitable enough to risk overnight gap down -> close position
                            exit_px = close * (1 - SLIPPAGE_FEE_PCT)
                            pnl = (exit_px - pos["entry_price"]) * pos["qty"]
                            pnl_pct = (exit_px - pos["entry_price"]) / pos["entry_price"]
                            cash += pos["qty"] * exit_px
                            trade_log.append({
                                "sym": sym, "type": "EOD_FLAT", "entry": pos["entry_price"],
                                "exit": exit_px, "pnl": pnl, "pnl_pct": pnl_pct, "date": t
                            })
                            closed_syms.append(sym)
                            continue

                open_pos_value += pos["qty"] * close

            for sym in closed_syms:
                positions.pop(sym, None)

            # 2. Check for New Entries
            if len(positions) < MAX_POSITIONS and not is_eod:
                available_slots = MAX_POSITIONS - len(positions)
                for sym in SYMBOLS:
                    if sym in positions or available_slots <= 0:
                        continue
                    df = self.data[sym]
                    if idx < 3:
                        continue
                    row = df.loc[t]
                    prev = df.iloc[idx - 1]

                    close = float(row["Close"])
                    hma100 = float(row["hma100"])
                    hma50 = float(row["hma50"])
                    prev_hma100 = float(prev["hma100"])
                    prev_hma50 = float(prev["hma50"])
                    ema20 = float(row["ema20"])
                    macd = float(row["macd"])
                    macd_sig = float(row["macd_sig"])
                    prev_macd = float(prev["macd"])
                    prev_macd_sig = float(prev["macd_sig"])
                    rsi = float(row["rsi"])
                    vol = float(row["Volume"])
                    vol_ma20 = float(row["vol_ma20"])

                    entry_signal = False
                    if not self.relaxed_entry:
                        # Baseline: Rigid 3-Indicator simultaneous condition
                        hma_rising = (hma100 > prev_hma100) and (close > hma100)
                        vol_ok = vol > vol_ma20
                        macd_cross = (prev_macd <= prev_macd_sig) and (macd > macd_sig)
                        rsi_ok = (50 <= rsi < 70)
                        if hma_rising and vol_ok and macd_cross and rsi_ok:
                            entry_signal = True
                    else:
                        # Feature 2: Relaxed Trend Momentum Entry (Early Rally Catch)
                        # HMA50 or HMA100 sloping up, price above EMA20, MACD positive or crossing, RSI > 50
                        hma_rising = (hma50 > prev_hma50) or (hma100 > prev_hma100)
                        above_ema20 = close > ema20
                        macd_bull = (macd > macd_sig) or ((prev_macd <= prev_macd_sig) and (macd > macd_sig))
                        rsi_bull = 50 <= rsi <= 76
                        if hma_rising and above_ema20 and macd_bull and rsi_bull:
                            entry_signal = True

                    if entry_signal:
                        curr_equity = cash + open_pos_value
                        pos_size_dollars = curr_equity * ALLOC_PER_POS
                        buy_price = close * (1 + SLIPPAGE_FEE_PCT)
                        qty = int(pos_size_dollars / buy_price)
                        cost = qty * buy_price

                        if qty > 0 and cash >= cost:
                            cash -= cost
                            sl_px = buy_price * (1 - BASE_SL_PCT)
                            tp_px = buy_price * (1 + BASE_TP_PCT)
                            positions[sym] = {
                                "qty": qty, "entry_price": buy_price, "peak_price": buy_price,
                                "sl_price": sl_px, "tp_price": tp_px, "half_taken": False, "is_runner": False
                            }
                            open_pos_value += qty * close
                            available_slots -= 1

            total_equity = cash + open_pos_value
            equity_curve.append(total_equity)

        # Final Close of any remaining open positions
        last_t = self.ref_idx[-1]
        for sym, pos in list(positions.items()):
            close = float(self.data[sym].loc[last_t]["Close"])
            exit_px = close * (1 - SLIPPAGE_FEE_PCT)
            pnl = (exit_px - pos["entry_price"]) * pos["qty"]
            pnl_pct = (exit_px - pos["entry_price"]) / pos["entry_price"]
            cash += pos["qty"] * exit_px
            trade_log.append({
                "sym": sym, "type": "FINAL_CLOSE", "entry": pos["entry_price"],
                "exit": exit_px, "pnl": pnl, "pnl_pct": pnl_pct, "date": last_t
            })
        positions.clear()
        total_equity = cash

        # Calculate Statistics
        s = pd.Series(equity_curve)
        peak = s.cummax()
        dd = (s - peak) / peak
        mdd_pct = abs(float(dd.min())) * 100

        total_ret_pct = ((total_equity - INITIAL_EQUITY) / INITIAL_EQUITY) * 100
        n_trades = len(trade_log)
        if n_trades > 0:
            tdf = pd.DataFrame(trade_log)
            wins = tdf[tdf["pnl"] > 0]
            losses = tdf[tdf["pnl"] <= 0]
            win_rate = (len(wins) / n_trades) * 100
            gross_win = wins["pnl"].sum() if len(wins) > 0 else 0.0
            gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 1e-9
            profit_factor = gross_win / gross_loss if gross_loss > 0 else 99.0
            avg_ret_pct = tdf["pnl_pct"].mean() * 100
        else:
            win_rate = 0.0
            profit_factor = 0.0
            avg_ret_pct = 0.0

        return {
            "total_return_pct": total_ret_pct,
            "mdd_pct": mdd_pct,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "trades": n_trades,
            "avg_trade_ret_pct": avg_ret_pct,
            "final_equity": total_equity
        }

def run_all_scenarios():
    data, ref_idx = download_data(period="90d", interval="1h")
    
    # Calculate Benchmark (QQQ Buy & Hold) Return
    qqq_start = float(data[BENCHMARK_SYM].iloc[0]["Close"])
    qqq_end = float(data[BENCHMARK_SYM].iloc[-1]["Close"])
    qqq_return_pct = ((qqq_end - qqq_start) / qqq_start) * 100

    scenarios = [
        # (Name, Overnight_Runner, Relaxed_Entry, Trailing_Stop)
        ("Baseline (현재 원본)", False, False, False),
        
        # 1개만 적용 (3가지)
        ("1개 적용: [A] 오버나잇 스윙만", True, False, False),
        ("1개 적용: [B] 진입조건 완화만", False, True, False),
        ("1개 적용: [C] ATR 트레일링만", False, False, True),
        
        # 2개 조합 (3가지)
        ("2개 조합: [A+B] 오버나잇 + 진입완화", True, True, False),
        ("2개 조합: [A+C] 오버나잇 + 트레일링", True, False, True),
        ("2개 조합: [B+C] 진입완화 + 트레일링", False, True, True),
        
        # 3개 모두 적용
        ("★ 3개 전부 적용: [A+B+C] 트리플 풀부스터", True, True, True)
    ]

    results = []
    print("=" * 85)
    print(f"📊 [알파카 나스닥] 3대 개선변수 매트릭스 백테스트 (최근 90일, 대상: {', '.join(SYMBOLS)})")
    print(f"📌 벤치마크 (QQQ Buy & Hold) 수익률: {qqq_return_pct:+.2f}%")
    print("=" * 85)

    for name, r1, r2, r3 in scenarios:
        engine = BacktestEngine(data, ref_idx, overnight_runner=r1, relaxed_entry=r2, trailing_stop=r3)
        res = engine.run()
        alpha = res["total_return_pct"] - qqq_return_pct
        res["name"] = name
        res["alpha"] = alpha
        results.append(res)
        print(f"⚙️  {name:<35} | 수익률: {res['total_return_pct']:>+6.2f}% (알파: {alpha:>+6.2f}%) | MDD: {res['mdd_pct']:>5.2f}% | 승률: {res['win_rate']:>5.1f}% | PF: {res['profit_factor']:>4.2f} | 매매: {res['trades']:>3}회")

    print("=" * 85)
    return results, qqq_return_pct

if __name__ == "__main__":
    run_all_scenarios()
