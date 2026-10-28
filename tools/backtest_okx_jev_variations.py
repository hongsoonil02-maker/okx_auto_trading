# -*- coding: utf-8 -*-
"""
backtest_okx_jev_variations.py
Comprehensive quantitative backtest comparing OKX Jev Trader variations:
  - Var 0: Baseline (Current logic: 5m BTC EMA50 macro, 5x lev, fixed 1.8% SL, 6% TP, 25% margin)
  - Var 1: Multi-Timeframe (MTF) Trend (BTC 5m + 15m EMA50 agreement)
  - Var 2: Dynamic ATR SL/TP (SL = 2.2 * ATR, TP = 2.5 * SL, Trail = 1.2 * SL)
  - Var 3: Fixed Dollar Risk Sizing ($40 USDT risk per trade)
  - Var 4: Combined Optimal v3.0 (MTF + Dynamic ATR SL/TP + Fixed Dollar Risk)
"""

import os
import sys
import time
import math
import requests
import datetime
import pandas as pd
import numpy as np

DATA_DIR = "/home/hongsoonil02/quant_system/data"
os.makedirs(DATA_DIR, exist_ok=True)

TARGET_SYMBOLS = ["BTC-USDT-SWAP", "SOL-USDT-SWAP", "DOGE-USDT-SWAP"]
TRADE_SYMBOLS = ["SOL-USDT-SWAP", "DOGE-USDT-SWAP"]
TOTAL_BARS = 6000  # ~21 days of 5m candles

def fetch_and_cache(inst_id: str, bar: str = "5m", target_bars: int = TOTAL_BARS) -> pd.DataFrame:
    cache_file = os.path.join(DATA_DIR, f"{inst_id}_{bar}.pkl")
    if os.path.exists(cache_file):
        df = pd.read_pickle(cache_file)
        if len(df) >= target_bars * 0.9:
            return df

    url = "https://www.okx.com/api/v5/market/history-candles"
    all_candles = []
    after = ""
    print(f"Fetching {inst_id} {bar} ({target_bars} bars)...")
    while len(all_candles) < target_bars:
        params = {"instId": inst_id, "bar": bar, "limit": "100"}
        if after:
            params["after"] = after
        try:
            r = requests.get(url, params=params, timeout=6).json()
            data = r.get("data", [])
            if not data:
                break
            all_candles.extend(data)
            after = data[-1][0]
            time.sleep(0.04)
        except Exception as e:
            print(f"Error fetching: {e}")
            break

    if not all_candles:
        raise RuntimeError(f"Failed to fetch data for {inst_id} {bar}")

    all_candles = sorted(all_candles, key=lambda x: int(x[0]))
    df = pd.DataFrame(all_candles, columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"])
    df = df[df["confirm"] == "1"].reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
    for col in ("open", "high", "low", "close", "vol"):
        df[col] = df[col].astype(float)

    df.to_pickle(cache_file)
    print(f"Cached {len(df)} candles for {inst_id} {bar}: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]

    df["ema9"] = c.ewm(span=9, adjust=False).mean()
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()

    tr = np.maximum(h - l, np.maximum(abs(h - c.shift()), abs(l - c.shift())))
    df["atr"] = tr.rolling(14).mean()
    df["atr_ratio"] = df["atr"] / c
    df["dist_ema20_pct"] = (c / df["ema20"] - 1.0) * 100.0

    df["bullish"] = (c > df["ema9"]) & (df["ema9"] > df["ema20"]) & (c > df["ema50"])
    df["bearish"] = (c < df["ema9"]) & (df["ema9"] < df["ema20"]) & (c < df["ema50"])
    return df

def run_backtest(
    btc_5m: pd.DataFrame,
    btc_15m: pd.DataFrame,
    asset_dfs: dict,
    asset_15m_dfs: dict,
    var_name: str,
    use_mtf: bool = False,
    dynamic_atr_sl: bool = False,
    fixed_dollar_risk: bool = False,
    risk_usd_per_trade: float = 40.0,
    base_sl_pct: float = 0.018,
    base_tp_pct: float = 0.060,
    base_trail_arm: float = 0.025,
    base_trail_drop: float = 0.008,
    leverage: float = 5.0,
    initial_capital: float = 10000.0
):
    capital = initial_capital
    max_capital = capital
    max_drawdown_pct = 0.0

    # Ensure ts is int
    btc_5m["ts"] = btc_5m["ts"].astype(int)
    btc_15m["ts"] = btc_15m["ts"].astype(int)
    for sym in TRADE_SYMBOLS:
        asset_dfs[sym]["ts"] = asset_dfs[sym]["ts"].astype(int)
        asset_15m_dfs[sym]["ts"] = asset_15m_dfs[sym]["ts"].astype(int)

    # Align on common timestamps of 5m
    common_ts = btc_5m["ts"].values
    btc_dict = btc_5m.set_index("ts").to_dict(orient="index")
    btc_15m_dict = btc_15m.set_index("ts").to_dict(orient="index")

    # Mapping 5m timestamp to latest confirmed 15m timestamp
    # 15m timestamps are multiples of 15 * 60 * 1000 = 900000 ms
    def get_15m_ts(ts_5m):
        return (int(ts_5m) // 900000) * 900000

    assets = {}
    for sym in TRADE_SYMBOLS:
        assets[sym] = {
            "5m": asset_dfs[sym].set_index("ts").to_dict(orient="index"),
            "15m": asset_15m_dfs[sym].set_index("ts").to_dict(orient="index"),
        }

    positions = {}  # sym -> dict(side, entry_px, size_contracts, notional, sl_px, tp_px, peak_px, entry_time, sl_pct, tp_pct, trail_arm, trail_drop)
    cooldowns = {sym: 0 for sym in TRADE_SYMBOLS}
    closed_trades = []

    for ts in common_ts:
        if ts not in btc_dict:
            continue
        btc_bar = btc_dict[ts]
        ts_15m = get_15m_ts(ts)
        btc_15m_bar = btc_15m_dict.get(ts_15m)

        # 1. Manage Open Positions
        for sym in list(positions.keys()):
            pos = positions[sym]
            sym_bar = assets[sym]["5m"].get(ts)
            if not sym_bar:
                continue

            curr_px = sym_bar["close"]
            high_px = sym_bar["high"]
            low_px = sym_bar["low"]
            is_long = pos["side"] == "LONG"

            # Check peak
            if is_long:
                if high_px > pos["peak_px"]:
                    pos["peak_px"] = high_px
            else:
                if low_px < pos["peak_px"]:
                    pos["peak_px"] = low_px

            # Check SL
            hit_sl = False
            exit_px = curr_px
            reason = ""

            if is_long:
                if low_px <= pos["sl_px"]:
                    hit_sl = True
                    exit_px = pos["sl_px"]
                    reason = "STOP_LOSS"
            else:
                if high_px >= pos["sl_px"]:
                    hit_sl = True
                    exit_px = pos["sl_px"]
                    reason = "STOP_LOSS"

            # Check TP
            hit_tp = False
            if not hit_sl:
                if is_long and high_px >= pos["tp_px"]:
                    hit_tp = True
                    exit_px = pos["tp_px"]
                    reason = "TAKE_PROFIT"
                elif not is_long and low_px <= pos["tp_px"]:
                    hit_tp = True
                    exit_px = pos["tp_px"]
                    reason = "TAKE_PROFIT"

            # Check Trailing Stop
            hit_trail = False
            if not hit_sl and not hit_tp:
                if is_long:
                    gain_from_entry = (pos["peak_px"] - pos["entry_px"]) / pos["entry_px"]
                    if gain_from_entry >= pos["trail_arm"]:
                        pullback = (pos["peak_px"] - curr_px) / pos["peak_px"]
                        if pullback >= pos["trail_drop"]:
                            hit_trail = True
                            exit_px = curr_px
                            reason = "TRAILING_EXIT"
                else:
                    gain_from_entry = (pos["entry_px"] - pos["peak_px"]) / pos["entry_px"]
                    if gain_from_entry >= pos["trail_arm"]:
                        pullback = (curr_px - pos["peak_px"]) / pos["peak_px"]
                        if pullback >= pos["trail_drop"]:
                            hit_trail = True
                            exit_px = curr_px
                            reason = "TRAILING_EXIT"

            if hit_sl or hit_tp or hit_trail:
                pnl_pct = (exit_px - pos["entry_px"]) / pos["entry_px"] if is_long else (pos["entry_px"] - exit_px) / pos["entry_px"]
                pnl_usd = pos["notional"] * pnl_pct
                # Commission (0.05% taker fee round-trip = 0.001)
                fee = pos["notional"] * 0.001
                net_pnl = pnl_usd - fee

                capital += net_pnl
                max_capital = max(max_capital, capital)
                dd = (max_capital - capital) / max_capital * 100.0
                max_drawdown_pct = max(max_drawdown_pct, dd)

                closed_trades.append({
                    "symbol": sym,
                    "side": pos["side"],
                    "entry_ts": pos["entry_time"],
                    "exit_ts": ts,
                    "entry_px": pos["entry_px"],
                    "exit_px": exit_px,
                    "net_pnl": net_pnl,
                    "pnl_pct": pnl_pct * 100.0,
                    "reason": reason,
                    "notional": pos["notional"]
                })

                # Cooldown 30m = 6 bars
                cooldowns[sym] = ts + 30 * 60 * 1000
                del positions[sym]

        # 2. Evaluate Entry Opportunities
        if len(positions) >= 2:
            continue

        btc_c = btc_bar["close"]
        btc_e50 = btc_bar["ema50"]
        macro_bullish = btc_c > (btc_e50 * 1.001)
        macro_bearish = btc_c < (btc_e50 * 0.999)

        if use_mtf and btc_15m_bar:
            btc_15m_c = btc_15m_bar["close"]
            btc_15m_e50 = btc_15m_bar["ema50"]
            # Both 5m and 15m must agree
            if macro_bullish and not (btc_15m_c > btc_15m_e50):
                macro_bullish = False
            if macro_bearish and not (btc_15m_c < btc_15m_e50):
                macro_bearish = False

        if not (macro_bullish or macro_bearish):
            continue

        for sym in TRADE_SYMBOLS:
            if len(positions) >= 2:
                break
            if sym in positions:
                continue
            if ts < cooldowns[sym]:
                continue

            bar_5m = assets[sym]["5m"].get(ts)
            if not bar_5m:
                continue

            # Chop filter
            atr_ratio = bar_5m["atr_ratio"]
            if math.isnan(atr_ratio) or atr_ratio < 0.0030:
                continue

            # Overextension guard
            dist_e20 = bar_5m["dist_ema20_pct"]
            if bar_5m["bearish"] and dist_e20 < -1.2:
                continue
            if bar_5m["bullish"] and dist_e20 > 1.2:
                continue

            # MTF asset confirmation
            if use_mtf:
                bar_15m = assets[sym]["15m"].get(ts_15m)
                if bar_15m:
                    if macro_bullish and not (bar_15m["close"] > bar_15m["ema20"]):
                        continue
                    if macro_bearish and not (bar_15m["close"] < bar_15m["ema20"]):
                        continue

            target_side = None
            if macro_bullish and bar_5m["bullish"]:
                target_side = "LONG"
            elif macro_bearish and bar_5m["bearish"]:
                target_side = "SHORT"

            if not target_side:
                continue

            # Calculate SL / TP
            px = bar_5m["close"]
            if dynamic_atr_sl:
                # Dynamic ATR: SL = max(1.8%, min(3.5%, 2.2 * ATR))
                sl_pct = max(0.018, min(0.035, 2.2 * atr_ratio))
                tp_pct = max(0.040, min(0.080, 2.5 * sl_pct))
                trail_arm = max(0.018, sl_pct * 1.1)
                trail_drop = max(0.006, sl_pct * 0.35)
            else:
                sl_pct = base_sl_pct
                tp_pct = base_tp_pct
                trail_arm = base_trail_arm
                trail_drop = base_trail_drop

            # Sizing
            if fixed_dollar_risk:
                # Target max loss = $40 USDT
                notional = risk_usd_per_trade / sl_pct
                # Cap notional at $4,000 to protect capital
                notional = min(notional, 4000.0)
            else:
                # Fixed 25% margin @ 5x leverage, max $2,500 margin = $12,500 notional
                margin = min(capital * 0.25, 1250.0)
                notional = margin * leverage

            sl_px = px * (1.0 - sl_pct) if target_side == "LONG" else px * (1.0 + sl_pct)
            tp_px = px * (1.0 + tp_pct) if target_side == "LONG" else px * (1.0 - tp_pct)

            positions[sym] = {
                "side": target_side,
                "entry_px": px,
                "notional": notional,
                "sl_px": sl_px,
                "tp_px": tp_px,
                "peak_px": px,
                "entry_time": ts,
                "sl_pct": sl_pct,
                "tp_pct": tp_pct,
                "trail_arm": trail_arm,
                "trail_drop": trail_drop
            }

    # Summary
    df_trades = pd.DataFrame(closed_trades)
    if len(df_trades) == 0:
        return {"var_name": var_name, "trades": 0, "net_pnl": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "max_dd": 0.0}

    wins = df_trades[df_trades["net_pnl"] > 0]
    losses = df_trades[df_trades["net_pnl"] <= 0]
    win_rate = len(wins) / len(df_trades) * 100.0
    tot_pnl = df_trades["net_pnl"].sum()
    gross_win = wins["net_pnl"].sum() if len(wins) else 0.0
    gross_loss = abs(losses["net_pnl"].sum()) if len(losses) else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else 99.0

    return {
        "var_name": var_name,
        "trades": len(df_trades),
        "win_rate": win_rate,
        "net_pnl": tot_pnl,
        "return_pct": (capital / initial_capital - 1.0) * 100.0,
        "avg_win": wins["net_pnl"].mean() if len(wins) else 0.0,
        "avg_loss": losses["net_pnl"].mean() if len(losses) else 0.0,
        "profit_factor": pf,
        "max_dd": max_drawdown_pct
    }

def main():
    print("=== Starting Historical Data Download ===")
    btc_5m = compute_indicators(fetch_and_cache("BTC-USDT-SWAP", bar="5m", target_bars=TOTAL_BARS))
    btc_15m = compute_indicators(fetch_and_cache("BTC-USDT-SWAP", bar="15m", target_bars=TOTAL_BARS // 3))

    asset_dfs = {}
    asset_15m_dfs = {}
    for sym in TRADE_SYMBOLS:
        asset_dfs[sym] = compute_indicators(fetch_and_cache(sym, bar="5m", target_bars=TOTAL_BARS))
        asset_15m_dfs[sym] = compute_indicators(fetch_and_cache(sym, bar="15m", target_bars=TOTAL_BARS // 3))

    print("\n=== Running Backtest Variations ===")
    variations = [
        ("Var 0: Baseline (Current Live Engine)", False, False, False),
        ("Var 1: + MTF Trend Gate (15m Confirmation)", True, False, False),
        ("Var 2: + Dynamic ATR SL/TP (Adaptive Breathing)", False, True, False),
        ("Var 3: + Fixed $40 Dollar Risk Sizing", False, False, True),
        ("Var 4: Combined Optimal v3.0 (MTF + Dynamic ATR + Fixed $40 Risk)", True, True, True),
    ]

    results = []
    for name, mtf, dyn_atr, fix_risk in variations:
        res = run_backtest(
            btc_5m, btc_15m, asset_dfs, asset_15m_dfs,
            var_name=name,
            use_mtf=mtf,
            dynamic_atr_sl=dyn_atr,
            fixed_dollar_risk=fix_risk,
            risk_usd_per_trade=40.0,
            base_sl_pct=0.018,
            base_tp_pct=0.060,
            initial_capital=10000.0
        )
        results.append(res)

    print("\n" + "=" * 90)
    print(f"{'Variation':<45} | {'Trades':<6} | {'WinRate':<7} | {'Net PnL ($)':<11} | {'Return':<7} | {'ProfitFac':<9} | {'MaxDD':<6}")
    print("-" * 90)
    for r in results:
        print(f"{r['var_name']:<45} | {r['trades']:<6} | {r['win_rate']:5.1f}% | ${r['net_pnl']:+9.2f} | {r['return_pct']:+5.1f}% | {r['profit_factor']:7.2f}x | {r['max_dd']:4.1f}%")
    print("=" * 90)

if __name__ == "__main__":
    main()
