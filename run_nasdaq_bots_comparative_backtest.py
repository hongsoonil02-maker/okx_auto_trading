#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_nasdaq_bots_comparative_backtest.py
=============================================================================
Comparative Backtest: Google Alpaca Bot (Chandelier) vs Coinbot KIS Nasdaq Bot (Multi-Tier)
Simulates identical market conditions on high-resolution 5-Minute bar data.
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# ── 1. CONFIGURATION & CONSTANTS ──
INITIAL_EQUITY = 100000.0  # $100,000 base capital
FEE_SLIPPAGE_BPS = 5.0     # 5 bps (0.05%) round-trip fee & slippage per trade

UNIVERSE_ALL = [
    "QQQ", "TQQQ", "SQQQ", "NVDA", "TSLA", "AAPL", 
    "MSTR", "PLTR", "AMD", "COIN", "AVGO", "META"
]

LEVERAGED_3X = {"TQQQ", "SQQQ"}
HIGH_BETA = {"MSTR", "PLTR", "COIN", "TSLA"}
TECH_LEADERS = {"NVDA", "AMD", "AVGO", "META"}

def get_symbol_risk_kis(sym: str):
    """(SL_pct, TP_pct, Trail_Arm_pct, Trail_Drop_pct)"""
    if sym in LEVERAGED_3X:
        return 0.024, 0.080, 0.025, 0.010
    if sym in HIGH_BETA:
        return 0.025, 0.090, 0.028, 0.012
    if sym in TECH_LEADERS:
        return 0.015, 0.060, 0.020, 0.008
    return 0.012, 0.045, 0.015, 0.008

# ── 2. DATA DOWNLOAD & PREPARATION ──
def load_market_data(period="60d", interval="5m"):
    print(f"📥 [Data Loader] Downloading 5-Min bars for {len(UNIVERSE_ALL)} symbols (period={period})...")
    data = yf.download(UNIVERSE_ALL, period=period, interval=interval, progress=False, group_by="ticker")
    
    clean_dfs = {}
    for sym in UNIVERSE_ALL:
        try:
            df = data[sym].dropna(subset=["Close"]).copy()
            if len(df) < 100:
                continue
            
            c = df["Close"]
            df["ema9"] = c.ewm(span=9, adjust=False).mean()
            df["ema21"] = c.ewm(span=21, adjust=False).mean()
            df["ema50"] = c.ewm(span=50, adjust=False).mean()
            
            delta = c.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / (loss + 1e-9)
            df["rsi"] = 100 - (100 / (1 + rs))
            
            # ATR (14 period)
            h, l, c_prev = df["High"], df["Low"], c.shift(1)
            tr = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
            df["atr"] = tr.ewm(alpha=1.0/14, adjust=False).mean()
            
            clean_dfs[sym] = df
        except Exception as e:
            print(f"⚠️ Failed to process {sym}: {e}")
            
    # Align to QQQ index
    ref_idx = clean_dfs["QQQ"].index
    aligned = {}
    for sym, df in clean_dfs.items():
        reindexed = df.reindex(ref_idx).ffill().bfill()
        aligned[sym] = reindexed
        
    print(f"✅ Data aligned: {len(ref_idx)} bars from {ref_idx[0].strftime('%Y-%m-%d')} to {ref_idx[-1].strftime('%Y-%m-%d')}")
    return aligned, ref_idx


# ── 3. STRATEGY A: GOOGLE ALPACA JEV TRADER (ATR CHANDELIER) ──
def run_alpaca_chandelier_sim(aligned, timeline):
    """
    Model A: Alpaca JEV Trader v2.1
    - Universe: Mega Tech (NVDA, TSLA, AAPL)
    - Max Positions: 3 (30% alloc each = 90% capital)
    - Exit: ATR Chandelier (stop = peak - 6 * ATR(14)), No TP!
    - Macro Gate: QQQ EMA50 (Long only in Bullish/Neutral)
    """
    universe = ["NVDA", "AAPL", "TSLA"]
    max_positions = 3
    alloc_pct = 0.30
    atr_mult = 6.0
    
    equity = INITIAL_EQUITY
    cash = INITIAL_EQUITY
    positions = {}  # sym -> dict
    trades = []
    equity_curve = []
    
    qqq = aligned["QQQ"]
    
    for i in range(50, len(timeline)):
        t = timeline[i]
        
        # 1. Update positions & Check Exits
        to_close = []
        for sym, pos in positions.items():
            df = aligned[sym]
            cur_p = df["Close"].iloc[i]
            high_p = df["High"].iloc[i]
            low_p = df["Low"].iloc[i]
            atr = df["atr"].iloc[i]
            
            # Update peak
            if high_p > pos["peak_price"]:
                pos["peak_price"] = high_p
                
            # ATR Chandelier Stop: stop = peak - atr_mult * atr
            target_stop = pos["peak_price"] - (atr_mult * atr)
            pos["stop_price"] = max(pos["stop_price"], target_stop)  # Never lower stop
            
            # Exit Check: low penetrates stop
            if low_p <= pos["stop_price"]:
                exit_price = min(cur_p, pos["stop_price"])
                pnl_dollar = pos["shares"] * (exit_price - pos["entry_price"])
                pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
                
                # Deduct fee
                fee = (pos["shares"] * exit_price) * (FEE_SLIPPAGE_BPS / 10000.0)
                net_pnl = pnl_dollar - fee
                
                cash += (pos["shares"] * exit_price) - fee
                trades.append({
                    "symbol": sym,
                    "entry_time": pos["entry_time"],
                    "exit_time": t,
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "pnl_dollar": net_pnl,
                    "holding_bars": i - pos["entry_bar"],
                    "reason": "ATR_CHANDELIER_STOP"
                })
                to_close.append(sym)
                
        for sym in to_close:
            del positions[sym]
            
        # 2. Check Entries
        # Macro Regime
        qqq_c = qqq["Close"].iloc[i]
        qqq_e50 = qqq["ema50"].iloc[i]
        qqq_e9 = qqq["ema9"].iloc[i]
        qqq_e21 = qqq["ema21"].iloc[i]
        
        is_bullish = (qqq_c > qqq_e50 * 1.0015) and (qqq_e9 > qqq_e21)
        is_bearish = (qqq_c < qqq_e50 * 0.9985) and (qqq_e9 < qqq_e21)
        
        # Alpaca Mega-Tech only enters when not bearish
        if not is_bearish and len(positions) < max_positions:
            for sym in universe:
                if sym in positions or len(positions) >= max_positions:
                    continue
                df = aligned[sym]
                c = df["Close"].iloc[i]
                e9 = df["ema9"].iloc[i]
                e21 = df["ema21"].iloc[i]
                rsi = df["rsi"].iloc[i]
                atr = df["atr"].iloc[i]
                
                # Signal: EMA9 > EMA21 cross & close > EMA9 & RSI > 48
                if (e9 > e21) and (c > e9) and (rsi > 48.0):
                    pos_size = equity * alloc_pct
                    shares = int(pos_size / c)
                    if shares > 0 and cash >= (shares * c):
                        fee = (shares * c) * (FEE_SLIPPAGE_BPS / 10000.0)
                        cash -= (shares * c + fee)
                        init_stop = c - (atr_mult * atr)
                        positions[sym] = {
                            "shares": shares,
                            "entry_price": c,
                            "peak_price": c,
                            "stop_price": init_stop,
                            "entry_time": t,
                            "entry_bar": i
                        }
                        
        # Calc Total Equity
        cur_pos_val = sum(pos["shares"] * aligned[sym]["Close"].iloc[i] for sym, pos in positions.items())
        total_equity = cash + cur_pos_val
        equity = total_equity
        equity_curve.append({"time": t, "equity": total_equity})
        
    return {
        "name": "Google Alpaca Bot (ATR Chandelier)",
        "final_equity": equity,
        "total_return_pct": (equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100.0,
        "trades": trades,
        "equity_curve": pd.DataFrame(equity_curve)
    }


# ── 4. STRATEGY B: COINBOT KIS NASDAQ TRADER (MULTI-TIER BRACKET) ──
def run_kis_multitier_sim(aligned, timeline):
    """
    Model B: KIS Nasdaq Trader v2.5
    - Universe: 11 symbols (TQQQ, SQQQ, MSTR, PLTR, NVDA, TSLA, AMD, COIN, AVGO, META, AAPL)
    - Max Positions: 3 (35% alloc each = ~100% capital)
    - Exit: Multi-Tier Fixed % (TP +4.5~9%, SL -1.2~2.5%, Trailing Arm +1.5~2.8%, Breakeven Lock +0.2%)
    - Macro Gate: QQQ EMA50 (Bullish -> TQQQ+Leaders, Bearish -> SQQQ Inverse Short)
    - Regime Shift Unwind: Automatically liquidates opposing ETFs on trend reversal
    """
    universe_all = [
        "TQQQ", "SQQQ", "NVDA", "TSLA", "MSTR", 
        "PLTR", "AMD", "COIN", "AVGO", "META", "AAPL"
    ]
    max_positions = 3
    alloc_pct = 0.35
    
    equity = INITIAL_EQUITY
    cash = INITIAL_EQUITY
    positions = {}  # sym -> dict
    trades = []
    equity_curve = []
    
    qqq = aligned["QQQ"]
    
    for i in range(50, len(timeline)):
        t = timeline[i]
        
        # 1. Macro Regime Check & Regime Shift Unwind
        qqq_c = qqq["Close"].iloc[i]
        qqq_e50 = qqq["ema50"].iloc[i]
        qqq_e9 = qqq["ema9"].iloc[i]
        qqq_e21 = qqq["ema21"].iloc[i]
        
        is_bullish = (qqq_c > qqq_e50 * 1.0015) and (qqq_e9 > qqq_e21)
        is_bearish = (qqq_c < qqq_e50 * 0.9985) and (qqq_e9 < qqq_e21)
        
        # Regime Shift Unwind: liquidates opposing ETFs
        to_close = []
        if is_bullish and "SQQQ" in positions:
            pos = positions["SQQQ"]
            c = aligned["SQQQ"]["Close"].iloc[i]
            pnl = pos["shares"] * (c - pos["entry_price"])
            pnl_pct = (c - pos["entry_price"]) / pos["entry_price"]
            fee = (pos["shares"] * c) * (FEE_SLIPPAGE_BPS / 10000.0)
            cash += (pos["shares"] * c - fee)
            trades.append({
                "symbol": "SQQQ", "entry_time": pos["entry_time"], "exit_time": t,
                "entry_price": pos["entry_price"], "exit_price": c,
                "pnl_pct": pnl_pct, "pnl_dollar": pnl - fee,
                "holding_bars": i - pos["entry_bar"], "reason": "REGIME_SHIFT_UNWIND"
            })
            to_close.append("SQQQ")
        elif is_bearish and "TQQQ" in positions:
            pos = positions["TQQQ"]
            c = aligned["TQQQ"]["Close"].iloc[i]
            pnl = pos["shares"] * (c - pos["entry_price"])
            pnl_pct = (c - pos["entry_price"]) / pos["entry_price"]
            fee = (pos["shares"] * c) * (FEE_SLIPPAGE_BPS / 10000.0)
            cash += (pos["shares"] * c - fee)
            trades.append({
                "symbol": "TQQQ", "entry_time": pos["entry_time"], "exit_time": t,
                "entry_price": pos["entry_price"], "exit_price": c,
                "pnl_pct": pnl_pct, "pnl_dollar": pnl - fee,
                "holding_bars": i - pos["entry_bar"], "reason": "REGIME_SHIFT_UNWIND"
            })
            to_close.append("TQQQ")
            
        for sym in to_close:
            del positions[sym]
            
        # 2. Check Exits (Trailing, Breakeven, TP, Hard SL)
        to_close = []
        for sym, pos in positions.items():
            df = aligned[sym]
            cur_p = df["Close"].iloc[i]
            high_p = df["High"].iloc[i]
            low_p = df["Low"].iloc[i]
            
            entry_p = pos["entry_price"]
            pos["peak_price"] = max(pos["peak_price"], high_p)
            peak_p = pos["peak_price"]
            
            unrealized_pct = (cur_p - entry_p) / entry_p
            max_ret = (peak_p - entry_p) / entry_p
            trail_drop = (peak_p - cur_p) / peak_p if peak_p > 0 else 0.0
            
            sl_pct, tp_pct, arm_pct, drop_pct = get_symbol_risk_kis(sym)
            
            reason = None
            exit_price = cur_p
            
            # (1) Trailing Stop
            if max_ret >= arm_pct and trail_drop >= drop_pct:
                reason = "TRAILING_STOP"
            # (2) Breakeven Defense (+1.5% peak and drop back to +0.2%)
            elif max_ret >= 0.015 and unrealized_pct <= 0.002:
                reason = "BREAKEVEN_LOCK"
            # (3) Hard Take Profit
            elif unrealized_pct >= tp_pct:
                reason = "TAKE_PROFIT"
            # (4) Hard Stop Loss
            elif unrealized_pct <= -sl_pct:
                reason = "STOP_LOSS"
                
            if reason:
                pnl = pos["shares"] * (exit_price - entry_p)
                fee = (pos["shares"] * exit_price) * (FEE_SLIPPAGE_BPS / 10000.0)
                net_pnl = pnl - fee
                cash += (pos["shares"] * exit_price - fee)
                trades.append({
                    "symbol": sym, "entry_time": pos["entry_time"], "exit_time": t,
                    "entry_price": entry_p, "exit_price": exit_price,
                    "pnl_pct": (exit_price - entry_p) / entry_p,
                    "pnl_dollar": net_pnl,
                    "holding_bars": i - pos["entry_bar"], "reason": reason
                })
                to_close.append(sym)
                
        for sym in to_close:
            del positions[sym]
            
        # 3. Check Entries with Priority Universe
        if len(positions) < max_positions:
            candidates = []
            if is_bullish:
                candidates = ["TQQQ", "MSTR", "PLTR", "NVDA", "TSLA", "AMD", "COIN", "AVGO", "META"]
            elif is_bearish:
                candidates = ["SQQQ"]
            else:
                candidates = ["PLTR", "NVDA", "TSLA", "MSTR", "META", "AAPL"]
                
            has_bull_etf = "TQQQ" in positions
            has_bear_etf = "SQQQ" in positions
            
            for sym in candidates:
                if sym in positions or len(positions) >= max_positions:
                    continue
                # Mutual Exclusion
                if sym == "SQQQ" and has_bull_etf:
                    continue
                if sym == "TQQQ" and has_bear_etf:
                    continue
                if sym not in aligned:
                    continue
                    
                df = aligned[sym]
                c = df["Close"].iloc[i]
                e9 = df["ema9"].iloc[i]
                e21 = df["ema21"].iloc[i]
                rsi = df["rsi"].iloc[i]
                
                # Signal
                if (e9 > e21) and (c > e9) and (rsi > 48.0):
                    pos_size = equity * alloc_pct
                    shares = int(pos_size / c)
                    if shares > 0 and cash >= (shares * c):
                        fee = (shares * c) * (FEE_SLIPPAGE_BPS / 10000.0)
                        cash -= (shares * c + fee)
                        positions[sym] = {
                            "shares": shares, "entry_price": c,
                            "peak_price": c, "entry_time": t, "entry_bar": i
                        }
                        if sym == "TQQQ": has_bull_etf = True
                        if sym == "SQQQ": has_bear_etf = True
                        
        cur_pos_val = sum(pos["shares"] * aligned[sym]["Close"].iloc[i] for sym, pos in positions.items())
        total_equity = cash + cur_pos_val
        equity = total_equity
        equity_curve.append({"time": t, "equity": total_equity})
        
    return {
        "name": "Coinbot KIS Nasdaq Bot (Multi-Tier Bracket)",
        "final_equity": equity,
        "total_return_pct": (equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100.0,
        "trades": trades,
        "equity_curve": pd.DataFrame(equity_curve)
    }


# ── 5. PERFORMANCE METRICS CALCULATOR ──
def compute_metrics(res):
    trades = res["trades"]
    eq_df = res["equity_curve"]
    
    if not trades or eq_df.empty:
        return {}
        
    df_t = pd.DataFrame(trades)
    n_trades = len(df_t)
    wins = df_t[df_t["pnl_dollar"] > 0]
    losses = df_t[df_t["pnl_dollar"] <= 0]
    
    win_rate = (len(wins) / n_trades) * 100.0 if n_trades > 0 else 0.0
    tot_profit = wins["pnl_dollar"].sum() if len(wins) > 0 else 0.0
    tot_loss = abs(losses["pnl_dollar"].sum()) if len(losses) > 0 else 1e-9
    profit_factor = tot_profit / tot_loss
    
    avg_win = wins["pnl_pct"].mean() * 100.0 if len(wins) > 0 else 0.0
    avg_loss = losses["pnl_pct"].mean() * 100.0 if len(losses) > 0 else 0.0
    
    # Max Drawdown
    eq_series = eq_df["equity"]
    roll_max = eq_series.cummax()
    drawdown = (eq_series - roll_max) / roll_max
    max_dd = abs(drawdown.min()) * 100.0
    
    # Average holding bars (5 min per bar)
    avg_hold_mins = df_t["holding_bars"].mean() * 5.0
    
    return {
        "final_equity": res["final_equity"],
        "return_pct": res["total_return_pct"],
        "max_drawdown": max_dd,
        "total_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "avg_hold_hours": avg_hold_mins / 60.0,
        "trades_df": df_t
    }


# ── MAIN EXECUTION ──
if __name__ == "__main__":
    aligned, timeline = load_market_data(period="60d", interval="5m")
    
    print("\n" + "=" * 75)
    print("🚀 [Backtest Launch] Running Head-to-Head Comparative Simulation...")
    print("=" * 75)
    
    res_alpaca = run_alpaca_chandelier_sim(aligned, timeline)
    res_kis = run_kis_multitier_sim(aligned, timeline)
    
    m_alpaca = compute_metrics(res_alpaca)
    m_kis = compute_metrics(res_kis)
    
    print("\n" + "=" * 80)
    print("🏆 [HEAD-TO-HEAD BACKTEST RESULTS: LAST 60 DAYS (5-MIN BARS)]")
    print("=" * 80)
    print(f"{'Performance Metric':<28} | {'Google Alpaca (Chandelier)':<23} | {'Coinbot KIS (Multi-Tier)':<23}")
    print("-" * 80)
    print(f"{'Initial Capital':<28} | ${INITIAL_EQUITY:,.2f}              | ${INITIAL_EQUITY:,.2f}")
    print(f"{'Final Equity':<28} | ${m_alpaca['final_equity']:,.2f}              | ${m_kis['final_equity']:,.2f}")
    print(f"{'Total Net Return (%)':<28} | {m_alpaca['return_pct']:+.2f}%                  | {m_kis['return_pct']:+.2f}%")
    print(f"{'Max Drawdown (MDD)':<28} | {m_alpaca['max_drawdown']:.2f}%                   | {m_kis['max_drawdown']:.2f}%")
    print(f"{'Profit Factor (PF)':<28} | {m_alpaca['profit_factor']:.2f}                    | {m_kis['profit_factor']:.2f}")
    print(f"{'Win Rate (%)':<28} | {m_alpaca['win_rate']:.1f}% ({len(m_alpaca['trades_df'][m_alpaca['trades_df']['pnl_dollar']>0])}/{m_alpaca['total_trades']})           | {m_kis['win_rate']:.1f}% ({len(m_kis['trades_df'][m_kis['trades_df']['pnl_dollar']>0])}/{m_kis['total_trades']})")
    print(f"{'Total Trades':<28} | {m_alpaca['total_trades']} trades                 | {m_kis['total_trades']} trades")
    print(f"{'Avg Win / Avg Loss':<28} | +{m_alpaca['avg_win_pct']:.2f}% / {m_alpaca['avg_loss_pct']:.2f}%          | +{m_kis['avg_win_pct']:.2f}% / {m_kis['avg_loss_pct']:.2f}%")
    print(f"{'Avg Holding Time':<28} | {m_alpaca['avg_hold_hours']:.1f} hours                 | {m_kis['avg_hold_hours']:.1f} hours")
    print("=" * 80)
    
    # Exit Reason Breakdown
    print("\n🔍 [Exit Reason Breakdown]")
    print("--- Google Alpaca Bot ---")
    print(m_alpaca["trades_df"]["reason"].value_counts().to_string())
    print("\n--- Coinbot KIS Nasdaq Bot ---")
    print(m_kis["trades_df"]["reason"].value_counts().to_string())
    print("\n✅ Simulation Completed.")
