# -*- coding: utf-8 -*-
"""
backtest_okx_cameron_4h_swing.py — Backtest for okx_cameron_4h_swing_trader.py
Replicates the live bot logic exactly:
- 4H first-pullback setup (red pullback, EMA9/VWAP support, surge, <=55% retrace, vol contraction)
- Entry stop at prev-bar high, SL at prev-bar low, TP at 1.86R
- Trailing SL to break-even (+0.2%) once +1.0R reached
- Sizing: 8% equity x 3 leverage, max 3 concurrent positions, one per symbol
- Taker fee 0.05% + slippage 0.05% per side
NOTE: The JEV AI gate is an LLM call and cannot be backtested -> pass-through.
"""

import os
import asyncio
from datetime import datetime, timezone
import pandas as pd
import ccxt.async_support as ccxt_async
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

SYMBOLS = [
    "SOL/USDT:USDT", "SUI/USDT:USDT", "DOGE/USDT:USDT", "ETH/USDT:USDT", "BTC/USDT:USDT",
    "XRP/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT", "NEAR/USDT:USDT",
    "APT/USDT:USDT", "PEPE/USDT:USDT", "WIF/USDT:USDT", "SHIB/USDT:USDT", "FET/USDT:USDT",
]

TIMEFRAME = "4h"
DAYS = int(os.getenv("BT_DAYS", "120"))
NO_TRAILING = os.getenv("BT_NO_TRAILING", "0") == "1"
INITIAL_CAPITAL = 960.0
ALLOCATION_PCT = 0.08
LEVERAGE = 3
MAX_CONCURRENT_POSITIONS = 3
RISK_REWARD_RATIO = 1.86
MIN_SURGE_PCT = 0.025
MAX_SL_PCT = 0.060
MIN_SL_PCT = 0.015
TRAILING_TRIGGER_RR = 1.0
FEE_PCT = 0.0005
SLIPPAGE_PCT = 0.0005
WINDOW = 35  # live bot fetches limit=35 bars per scan


async def fetch_bars(ex, symbol):
    end_ts = ex.milliseconds()
    start_ts = end_ts - DAYS * 24 * 3600 * 1000
    all_bars, since = [], start_ts
    while True:
        bars = await ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since, limit=100)
        if not bars:
            break
        all_bars.extend(bars)
        since = bars[-1][0] + 1
        if len(bars) < 100:
            break
        await asyncio.sleep(ex.rateLimit / 1000)
    if len(all_bars) < 60:
        return None
    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def add_indicators(df):
    df = df.copy()
    tp = (df["high"] + df["low"] + df["close"]) / 3
    # Rolling 35-bar cumulative VWAP to mirror the live bot's windowed cumsum
    df["vwap"] = (tp * df["volume"]).rolling(WINDOW).sum() / df["volume"].rolling(WINDOW).sum().replace(0, 1)
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    return df


def setup_at(df, i):
    """Evaluate setup with bar i as 'prev_bar' (last closed). Returns setup dict or None."""
    if i < 14:
        return None
    pb = df.iloc[i]
    if pb["close"] >= pb["open"]:
        return None
    if pb["close"] < pb["ema9"] * 0.995 or pb["low"] < pb["vwap"] * 0.990:
        return None
    prior = df.iloc[i - 5:i]  # bars i-5..i-1 == live df.iloc[-7:-2] with curr=i+1
    surge_low, surge_high = prior["low"].min(), prior["high"].max()
    surge = surge_high - surge_low
    surge_pct = surge / surge_low if surge_low > 0 else 0
    if surge_pct < MIN_SURGE_PCT:
        return None
    retrace_ratio = (surge_high - pb["low"]) / surge if surge > 0 else 1.0
    if retrace_ratio > 0.55:
        return None
    prior_vol_mean = prior["volume"].mean()
    if prior["volume"].mean() > 0 and pb["volume"] > prior_vol_mean * 1.20:
        return None
    trigger_price, stop_loss = pb["high"], pb["low"]
    risk = trigger_price - stop_loss
    risk_pct = risk / trigger_price if trigger_price > 0 else 0
    if risk_pct > MAX_SL_PCT or risk_pct < MIN_SL_PCT:
        return None
    return {
        "trigger_price": trigger_price,
        "stop_loss": stop_loss,
        "take_profit": trigger_price + risk * RISK_REWARD_RATIO,
        "risk": risk,
    }


def run_backtest(data):
    equity = INITIAL_CAPITAL
    open_pos = {}   # symbol -> dict
    trades = []
    equity_curve = []

    max_len = max(len(df) for df in data.values())

    for i in range(max_len):
        # 1) Manage open positions on bar i
        for sym in list(open_pos.keys()):
            df = data[sym]
            if i >= len(df):
                continue
            bar = df.iloc[i]
            p = open_pos[sym]

            if i == p["entry_i"]:
                # entry bar: entry filled intra-bar; check same-bar stop-out (pessimistic)
                if bar["low"] <= p["sl"]:
                    exit_p = p["sl"] * (1 - SLIPPAGE_PCT)
                    trades.append(close_trade(sym, p, exit_p, bar, "SL_SAME_BAR"))
                    equity += trades[-1]["pnl"]
                    del open_pos[sym]
                continue

            # trailing: +1R reached -> SL to break-even *1.002
            if not NO_TRAILING and p["sl"] < p["entry"] and bar["high"] >= p["entry"] + p["risk"]:
                p["sl"] = p["entry"] * 1.002

            exit_p, reason = None, None
            hit_sl = bar["low"] <= p["sl"]
            hit_tp = bar["high"] >= p["tp"]
            if hit_sl and hit_tp:
                exit_p, reason = p["sl"] * (1 - SLIPPAGE_PCT), "SL_AMBIG"  # pessimistic
            elif hit_sl:
                exit_p, reason = p["sl"] * (1 - SLIPPAGE_PCT), "SL"
            elif hit_tp:
                exit_p, reason = p["tp"] * (1 - SLIPPAGE_PCT), "TP"
            if exit_p is not None:
                trades.append(close_trade(sym, p, exit_p, bar, reason))
                equity += trades[-1]["pnl"]
                del open_pos[sym]

        # mark-to-market equity
        eq = equity
        for sym, p in open_pos.items():
            df = data[sym]
            if i < len(df):
                eq += p["notional"] * (df.iloc[i]["close"] / p["entry"] - 1)
        equity_curve.append((i, eq))

        # 2) New entries: bar i is 'prev' (closed), bar i+1 is the live 'curr' bar
        if len(open_pos) >= MAX_CONCURRENT_POSITIONS or i + 1 >= max_len:
            continue
        for sym in SYMBOLS:
            if sym in open_pos or sym not in data or i + 1 >= len(data[sym]):
                continue
            setup = setup_at(data[sym], i)
            if not setup:
                continue
            nxt = data[sym].iloc[i + 1]
            if nxt["high"] < setup["trigger_price"]:
                continue
            entry = setup["trigger_price"] * (1 + SLIPPAGE_PCT)
            notional = equity * ALLOCATION_PCT * LEVERAGE
            open_pos[sym] = {
                "entry": entry, "sl": setup["stop_loss"], "tp": setup["take_profit"],
                "risk": setup["risk"], "notional": notional, "entry_i": i + 1,
                "entry_time": nxt["timestamp"],
            }
            if len(open_pos) >= MAX_CONCURRENT_POSITIONS:
                break

    return trades, equity_curve


def close_trade(sym, p, exit_p, bar, reason):
    notional = p["notional"]
    ret = exit_p / p["entry"] - 1
    fees = notional * FEE_PCT * 2
    pnl = notional * ret - fees
    return {
        "symbol": sym, "entry_time": pd.to_datetime(p["entry_time"], unit="ms"),
        "exit_time": pd.to_datetime(bar["timestamp"], unit="ms"),
        "entry": p["entry"], "exit": exit_p, "reason": reason,
        "ret_pct": ret * 100, "pnl": pnl, "notional": notional,
    }


def report(trades):
    if not trades:
        print("No trades generated.")
        return
    equity = INITIAL_CAPITAL
    peak = equity
    max_dd = 0.0
    wins = losses = be = 0
    gross_win = gross_loss = 0.0
    for t in sorted(trades, key=lambda x: x["exit_time"]):
        equity += t["pnl"]
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)
        if t["pnl"] > 0.5:
            wins += 1
            gross_win += t["pnl"]
        elif t["ret_pct"] > -0.1:
            be += 1
        else:
            losses += 1
            gross_loss += abs(t["pnl"])
    n = len(trades)
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    print("=" * 72)
    print(f"OKX Cameron 4H Swing Backtest — last {DAYS} days | {len(SYMBOLS)} symbols")
    print("=" * 72)
    print(f"Trades: {n} | Wins: {wins} | Break-even: {be} | Losses: {losses}")
    print(f"Win Rate: {wins/n*100:.1f}% (excl. BE: {wins/(wins+losses)*100:.1f}%)")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Total PnL: ${sum(t['pnl'] for t in trades):+.2f} on ${INITIAL_CAPITAL:.0f} ({(equity/INITIAL_CAPITAL-1)*100:+.1f}%)")
    print(f"Final Equity: ${equity:.2f} | Max Drawdown: {max_dd*100:.1f}%")
    print(f"Avg trade: ${sum(t['pnl'] for t in trades)/n:+.2f} | Avg ret: {sum(t['ret_pct'] for t in trades)/n:+.2f}%")
    reasons = {}
    for t in trades:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    print(f"Exit reasons: {reasons}")
    print("-" * 72)
    print(f"{'Symbol':<18}{'Entry':<18}{'Exit':<18}{'Reason':<12}{'Ret%':>8}{'PnL$':>9}")
    for t in sorted(trades, key=lambda x: x["exit_time"]):
        print(f"{t['symbol']:<18}{str(t['entry_time'])[:16]:<18}{str(t['exit_time'])[:16]:<18}"
              f"{t['reason']:<12}{t['ret_pct']:>7.2f}%{t['pnl']:>9.2f}")


async def main():
    ex = ccxt_async.okx({
        "apiKey": os.getenv("OKX_API_KEY"),
        "secret": os.getenv("OKX_SECRET"),
        "password": os.getenv("OKX_PASSPHRASE"),
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
    await ex.load_markets()
    data = {}
    for s in SYMBOLS:
        try:
            df = await fetch_bars(ex, s)
            if df is not None:
                data[s] = add_indicators(df)
                print(f"  {s}: {len(df)} bars")
            else:
                print(f"  {s}: insufficient data, skipped")
        except Exception as e:
            print(f"  {s}: fetch failed: {e}")
    await ex.close()
    print("-" * 72)
    trades, _ = run_backtest(data)
    report(trades)


if __name__ == "__main__":
    asyncio.run(main())
