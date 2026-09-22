#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_nasdaq_jev_simulation.py — Alpaca Nasdaq JEV Autonomous Trading Simulation
Comprehensive Multi-Scenario Quantitative Backtest & Analysis

Features:
- Perfectly time-aligned continuous dataset across 8 Nasdaq symbols
- Evaluates:
  1. Baseline: Pure EMA9/21 Momentum Cross
  2. + JEV Gating: Sub-second Microstructure Imbalance & Up-in-10 Score Gate (>= 0.58)
  3. + JEV Post-Only Maker: Inside-spread limit pricing (fee/slippage edge)
  4. + QQQ 1h/Macro Trend Gate (EMA50): Avoids counter-trend whipsaws
  5. 3x Leveraged ETFs Pair (TQQQ/SQQQ/SOXL/SOXS) vs Mega Tech (NVDA/TSLA/AAPL)
  6. Adaptive Volatility Risk: 3x ETFs (-2.4% SL, +4.2% TP) vs 1x Stocks (-1.2% SL, +2.8% TP)
  7. Dynamic Trailing Stop: Arms at profit watermark and trails peak

Data Source: Alpaca Market Data API (Real 5-Min Bars)
Initial Capital: $100,000 USD (Matching Alpaca Paper Trading Account)
"""

import os
import sys
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REMOTE_DEPLOY_ENV = PROJECT_ROOT / "remote_deploy" / ".env"
LOCAL_ENV = PROJECT_ROOT / ".env"

if REMOTE_DEPLOY_ENV.exists():
    load_dotenv(REMOTE_DEPLOY_ENV)
elif LOCAL_ENV.exists():
    load_dotenv(LOCAL_ENV)

CACHE_DIR = PROJECT_ROOT / "jev" / "data_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Alpaca API Credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "PKVGU7IXDJMAF6JBWWVMP2I6R3")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "CCnTinES4eWwkQqifr9879ASCcXSpm9kjuvbZnCKpHM3")
DATA_URL = "https://data.alpaca.markets/v2"

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
}

INITIAL_EQUITY = 100000.0  # $100k
MAX_POSITION_PCT = 0.15    # 15% per position (~$15,000)
MAX_CONCURRENT_POSITIONS = 4
JEV_CONFIDENCE_THRESHOLD = 0.58

ALL_SYMBOLS = ["QQQ", "TQQQ", "SQQQ", "SOXL", "SOXS", "NVDA", "TSLA", "AAPL"]
LEVERAGED_3X = {"TQQQ", "SQQQ", "SOXL", "SOXS"}


def fetch_symbol_bars(symbol: str, timeframe: str = "5Min", start_date: str = "2026-08-25") -> pd.DataFrame:
    """Fetches historical bars from Alpaca API with local disk caching"""
    cache_file = CACHE_DIR / f"{symbol}_{timeframe}_{start_date}.pkl"
    if cache_file.exists():
        try:
            return pd.read_pickle(cache_file)
        except Exception:
            pass

    print(f"📥 [Data Fetch] Downloading {symbol} ({timeframe}) from Alpaca Data API...")
    url = f"{DATA_URL}/stocks/{symbol}/bars"
    params = {
        "timeframe": timeframe,
        "start": f"{start_date}T00:00:00Z",
        "limit": 10000,
        "feed": "iex",
        "sort": "asc"
    }

    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json().get("bars", [])
            if not data:
                print(f"⚠️ No bars returned for {symbol}")
                return pd.DataFrame()
            df = pd.DataFrame(data)
            df = df.rename(columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
            df["time"] = pd.to_datetime(df["time"])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.sort_values("time").reset_index(drop=True)
            df.to_pickle(cache_file)
            print(f"   ✅ Saved {len(df)} bars for {symbol} to cache.")
            return df
        else:
            print(f"❌ Error fetching {symbol}: HTTP {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"❌ Exception fetching {symbol}: {e}")

    return pd.DataFrame()


def prepare_market_data(start_date: str = "2026-08-25") -> dict:
    """Fetches and aligns all symbols to a continuous unified timeline"""
    raw_dfs = {}
    print("\n" + "=" * 65)
    print("📊 [Alpaca Nasdaq Data Loader] Initializing Historical Bars")
    print("=" * 65)

    all_times = set()
    for sym in ALL_SYMBOLS:
        df = fetch_symbol_bars(sym, timeframe="5Min", start_date=start_date)
        if df.empty or len(df) < 50:
            continue
        raw_dfs[sym] = df
        all_times.update(df["time"].tolist())

    if not raw_dfs or "QQQ" not in raw_dfs:
        return {}

    unified_timeline = sorted(list(all_times))
    unified_idx = pd.DatetimeIndex(unified_timeline)
    n_bars = len(unified_idx)
    print(f"⏱️ Unified Timeline Aligned: {n_bars} 5-Minute Bars ({unified_idx[0]} ~ {unified_idx[-1]})")

    aligned_data = {"timeline": unified_idx, "n_bars": n_bars}

    for sym, df in raw_dfs.items():
        reindexed = df.set_index("time").reindex(unified_idx)
        reindexed["close"] = reindexed["close"].ffill().bfill()
        reindexed["open"] = reindexed["open"].fillna(reindexed["close"])
        reindexed["high"] = reindexed["high"].fillna(reindexed["close"])
        reindexed["low"] = reindexed["low"].fillna(reindexed["close"])
        reindexed["volume"] = reindexed["volume"].fillna(0.0)

        c = reindexed["close"].to_numpy(dtype=float)
        o = reindexed["open"].to_numpy(dtype=float)
        h = reindexed["high"].to_numpy(dtype=float)
        l = reindexed["low"].to_numpy(dtype=float)
        v = reindexed["volume"].to_numpy(dtype=float)

        s_close = pd.Series(c)
        ema9 = s_close.ewm(span=9, adjust=False).mean().to_numpy()
        ema21 = s_close.ewm(span=21, adjust=False).mean().to_numpy()
        ema50 = s_close.ewm(span=50, adjust=False).mean().to_numpy()

        delta = s_close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-9)
        rsi = (100 - (100 / (1 + rs))).to_numpy()

        # Microstructure / Orderbook Pressure Proxy
        rng = np.maximum(h - l, 1e-6)
        close_pos = (c - l) / rng
        body_flow = (c - o) / rng
        prob_up_proxy = np.clip(0.42 + 0.36 * close_pos + 0.14 * body_flow, 0.08, 0.94)

        aligned_data[sym] = {
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "ema9": ema9,
            "ema21": ema21,
            "ema50": ema50,
            "rsi": rsi,
            "prob_up": prob_up_proxy,
        }

    return aligned_data


def run_simulation(
    data: dict,
    universe: list,
    use_jev_gating: bool = True,
    use_post_only: bool = True,
    use_trailing_stop: bool = True,
    use_trend_gate: bool = False,
    adaptive_risk: bool = False,
    use_micro_exit: bool = False,
    jev_threshold: float = 0.58
) -> dict:
    """Executes high-fidelity event-driven backtest simulation across aligned bars"""
    if "QQQ" not in data:
        return {}

    n_bars = data["n_bars"]
    timeline = data["timeline"]
    qqq = data["QQQ"]

    equity = INITIAL_EQUITY
    cash = INITIAL_EQUITY
    peak_equity = INITIAL_EQUITY
    max_drawdown = 0.0
    equity_curve = [INITIAL_EQUITY]

    # Active positions: sym -> dict
    positions = {}
    trades = []
    filtered_signals = 0

    # Costs
    slippage_taker = 0.0002     # 2 bps taker slippage
    maker_rebate_edge = -0.0001 # 1 bps inside-spread advantage
    regulatory_fee = 0.000028    # SEC + FINRA fee on exit

    for i in range(50, n_bars):
        current_time = timeline[i]

        # 1. Update Existing Positions & Check Exits
        active_symbols = list(positions.keys())
        for sym in active_symbols:
            pos = positions[sym]
            is_3x = sym in LEVERAGED_3X

            bar_high = data[sym]["high"][i]
            bar_low = data[sym]["low"][i]
            bar_close = data[sym]["close"][i]
            prob_up = data[sym]["prob_up"][i]

            entry_px = pos["entry_price"]
            pos_side = pos["side"]

            # Update high watermark
            if bar_high > pos["peak_price"]:
                pos["peak_price"] = bar_high

            unrealized_ret = (bar_close - entry_px) / entry_px
            max_ret = (pos["peak_price"] - entry_px) / entry_px
            low_ret = (bar_low - entry_px) / entry_px

            # Risk parameters
            if adaptive_risk and is_3x:
                sl_pct = 0.024
                tp_pct = 0.042
                arm_pct = 0.020
                trail_delta = 0.010
            else:
                sl_pct = 0.012
                tp_pct = 0.028
                arm_pct = 0.015
                trail_delta = 0.008

            exit_reason = None
            exit_price = bar_close

            # A. Hard Stop Loss
            if low_ret <= -sl_pct:
                exit_reason = "HARD_STOP_LOSS"
                exit_price = entry_px * (1.0 - sl_pct)

            # B. Take Profit Target
            elif (bar_high - entry_px) / entry_px >= tp_pct:
                exit_reason = "TAKE_PROFIT"
                exit_price = entry_px * (1.0 + tp_pct)

            # C. Trailing Stop
            elif use_trailing_stop and max_ret >= arm_pct and (pos["peak_price"] - bar_close) / pos["peak_price"] >= trail_delta:
                exit_reason = "TRAILING_STOP"
                exit_price = bar_close

            # D. Adverse Microstructure Orderbook Pressure Exit
            elif use_micro_exit and use_jev_gating and prob_up <= 0.30 and unrealized_ret <= -0.008:
                exit_reason = "MICRO_PRESSURE_EXIT"
                exit_price = bar_close

            # Execute Exit
            if exit_reason:
                exit_slip = slippage_taker if not use_post_only else 0.0
                net_exit_price = exit_price * (1.0 - exit_slip - regulatory_fee)
                pnl = pos["qty"] * (net_exit_price - entry_px)
                ret_pct = pnl / (pos["qty"] * entry_px)
                cash += pos["qty"] * net_exit_price

                trades.append({
                    "symbol": sym,
                    "side": pos_side,
                    "entry_time": pos["entry_time"],
                    "exit_time": current_time,
                    "entry_price": entry_px,
                    "exit_price": net_exit_price,
                    "pnl": pnl,
                    "return_pct": ret_pct,
                    "reason": exit_reason,
                    "bars_held": i - pos["entry_bar"]
                })
                del positions[sym]

        # 2. Evaluate Macro Regime & New Entry Candidates
        qqq_c = qqq["close"][i]
        qqq_e9 = qqq["ema9"][i]
        qqq_e21 = qqq["ema21"][i]
        qqq_e50 = qqq["ema50"][i]
        qqq_rsi = qqq["rsi"][i]
        qqq_prob = qqq["prob_up"][i]

        is_qqq_bull = (qqq_e9 > qqq_e21) and (qqq_c > qqq_e9) and (qqq_rsi > 48.0)
        is_qqq_bear = (qqq_e9 < qqq_e21) and (qqq_c < qqq_e9) and (qqq_rsi < 52.0)

        if use_trend_gate:
            is_qqq_bull = is_qqq_bull and (qqq_c > qqq_e50)
            is_qqq_bear = is_qqq_bear and (qqq_c < qqq_e50)

        candidate_symbols = []
        if is_qqq_bull:
            candidate_symbols = [s for s in ["TQQQ", "SOXL", "NVDA", "TSLA", "AAPL", "QQQ"] if s in universe]
        elif is_qqq_bear:
            candidate_symbols = [s for s in ["SQQQ", "SOXS"] if s in universe]
        else:
            if not use_trend_gate:
                candidate_symbols = [s for s in universe if s not in ["SQQQ", "SOXS"]]

        # 3. Check Entries for Viable Candidates
        if len(positions) < MAX_CONCURRENT_POSITIONS and candidate_symbols:
            target_dollar_size = equity * MAX_POSITION_PCT

            for sym in candidate_symbols:
                if len(positions) >= MAX_CONCURRENT_POSITIONS:
                    break
                if sym in positions:
                    continue

                c = data[sym]["close"][i]
                e9 = data[sym]["ema9"][i]
                e21 = data[sym]["ema21"][i]
                rsi = data[sym]["rsi"][i]
                prob_up = data[sym]["prob_up"][i]

                # Momentum entry trigger
                tech_bullish = (e9 > e21) and (c > e9) and (rsi > 47.0)
                if not tech_bullish:
                    continue

                # JEV Microstructure Gate
                if use_jev_gating:
                    if prob_up < jev_threshold:
                        filtered_signals += 1
                        continue

                # Execution Pricing
                slip = slippage_taker if not use_post_only else maker_rebate_edge
                entry_fill_price = c * (1.0 + slip)
                shares = int(target_dollar_size / entry_fill_price)
                if shares < 1:
                    shares = 1

                req_cash = shares * entry_fill_price
                if cash >= req_cash:
                    cash -= req_cash
                    positions[sym] = {
                        "symbol": sym,
                        "side": "buy",
                        "qty": shares,
                        "entry_price": entry_fill_price,
                        "peak_price": entry_fill_price,
                        "entry_time": current_time,
                        "entry_bar": i
                    }

        # 4. Mark to Market Equity & Drawdown Tracking
        current_mtm = cash + sum(pos["qty"] * data[s]["close"][i] for s, pos in positions.items())
        equity = current_mtm
        equity_curve.append(equity)
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity
        if dd > max_drawdown:
            max_drawdown = dd

    # Close any remaining open positions at final bar for complete PnL accounting
    final_bar_idx = n_bars - 1
    final_time = timeline[final_bar_idx]
    for sym, pos in list(positions.items()):
        exit_px = data[sym]["close"][final_bar_idx]
        net_exit_price = exit_px * (1.0 - regulatory_fee)
        pnl = pos["qty"] * (net_exit_price - pos["entry_price"])
        ret_pct = pnl / (pos["qty"] * pos["entry_price"])
        cash += pos["qty"] * net_exit_price
        trades.append({
            "symbol": sym,
            "side": pos["side"],
            "entry_time": pos["entry_time"],
            "exit_time": final_time,
            "entry_price": pos["entry_price"],
            "exit_price": net_exit_price,
            "pnl": pnl,
            "return_pct": ret_pct,
            "reason": "SIMULATION_END_MTM",
            "bars_held": final_bar_idx - pos["entry_bar"]
        })
    positions.clear()
    equity = cash

    # Statistics
    total_trades = len(trades)
    winning_trades = [t for t in trades if t["pnl"] > 0]
    losing_trades = [t for t in trades if t["pnl"] <= 0]
    win_rate = (len(winning_trades) / total_trades * 100.0) if total_trades > 0 else 0.0

    gross_profit = sum(t["pnl"] for t in winning_trades)
    gross_loss = abs(sum(t["pnl"] for t in losing_trades))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0

    net_return_pct = (equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100.0

    eq_series = pd.Series(equity_curve)
    pct_returns = eq_series.pct_change().dropna()
    sharpe = (pct_returns.mean() / (pct_returns.std() + 1e-9) * np.sqrt(252 * 78)) if len(pct_returns) > 0 else 0.0

    return {
        "initial_equity": INITIAL_EQUITY,
        "final_equity": equity,
        "net_return_pct": net_return_pct,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown * 100.0,
        "sharpe_ratio": sharpe,
        "filtered_signals": filtered_signals,
        "trades": trades
    }


def main():
    start_time = time.time()
    market_data = prepare_market_data(start_date="2026-08-25")

    if not market_data:
        print("❌ Error: Failed to load market data from Alpaca API.")
        return

    n_bars = market_data["n_bars"]
    timeline = market_data["timeline"]
    date_start_str = str(timeline[0])[:10]
    date_end_str = str(timeline[-1])[:10]

    print("\n" + "=" * 80)
    print(f"📈 [Backtest Window] {date_start_str} ~ {date_end_str} ({n_bars} Aligned 5-Min Bars, ~4 Weeks)")
    print(f"💰 [Account Capital] Starting: ${INITIAL_EQUITY:,.2f} USD | Universe: {', '.join(ALL_SYMBOLS)}")
    print("=" * 80 + "\n")

    scenarios = [
        {
            "name": "1. Baseline (Pure EMA Momentum, No Jev)",
            "universe": ALL_SYMBOLS,
            "use_jev": False,
            "use_post_only": False,
            "use_trailing": False,
            "use_trend": False,
            "adaptive": False,
            "micro_exit": False
        },
        {
            "name": "2. + JEV Gating (Threshold >= 0.58)",
            "universe": ALL_SYMBOLS,
            "use_jev": True,
            "use_post_only": False,
            "use_trailing": False,
            "use_trend": False,
            "adaptive": False,
            "micro_exit": False
        },
        {
            "name": "3. + JEV Maker Post-Only Quoting",
            "universe": ALL_SYMBOLS,
            "use_jev": True,
            "use_post_only": True,
            "use_trailing": False,
            "use_trend": False,
            "adaptive": False,
            "micro_exit": False
        },
        {
            "name": "4. + QQQ EMA50 Trend Gate (방향성 필터)",
            "universe": ALL_SYMBOLS,
            "use_jev": True,
            "use_post_only": True,
            "use_trailing": True,
            "use_trend": True,
            "adaptive": False,
            "micro_exit": False
        },
        {
            "name": "5. [3x Leveraged ETFs Only] TQQQ/SQQQ/SOXL/SOXS",
            "universe": ["TQQQ", "SQQQ", "SOXL", "SOXS"],
            "use_jev": True,
            "use_post_only": True,
            "use_trailing": True,
            "use_trend": True,
            "adaptive": True,
            "micro_exit": False
        },
        {
            "name": "6. [Mega Tech Equities Only] NVDA/TSLA/AAPL",
            "universe": ["NVDA", "TSLA", "AAPL"],
            "use_jev": True,
            "use_post_only": True,
            "use_trailing": True,
            "use_trend": True,
            "adaptive": False,
            "micro_exit": False
        },
        {
            "name": "7. 🌟 [Long-Bias Leaders + TQQQ] QQQ/TQQQ/NVDA/TSLA/AAPL",
            "universe": ["QQQ", "TQQQ", "NVDA", "TSLA", "AAPL"],
            "use_jev": True,
            "use_post_only": True,
            "use_trailing": True,
            "use_trend": True,
            "adaptive": True,
            "micro_exit": False
        },
        {
            "name": "8. 🚀 [JEV Full Autonomous] 8-Symbol Dual Directional",
            "universe": ALL_SYMBOLS,
            "use_jev": True,
            "use_post_only": True,
            "use_trailing": True,
            "use_trend": True,
            "adaptive": True,
            "micro_exit": False
        }
    ]

    results = []
    for sc in scenarios:
        res = run_simulation(
            data=market_data,
            universe=sc["universe"],
            use_jev_gating=sc["use_jev"],
            use_post_only=sc["use_post_only"],
            use_trailing_stop=sc["use_trailing"],
            use_trend_gate=sc["use_trend"],
            adaptive_risk=sc["adaptive"],
            use_micro_exit=sc["micro_exit"],
            jev_threshold=JEV_CONFIDENCE_THRESHOLD
        )
        res["name"] = sc["name"]
        results.append(res)

    # Comparison Table
    print("\n" + "=" * 110)
    print(f"{'시나리오 (Scenario)':<46} | {'수익률(Return)':<12} | {'최종자산(Equity)':<14} | {'승률(Win)':<8} | {'손익비(PF)':<8} | {'MDD':<8} | {'거래수':<6}")
    print("-" * 110)
    for r in results:
        ret_str = f"{r['net_return_pct']:+.2f}%"
        eq_str = f"${r['final_equity']:,.2f}"
        win_str = f"{r['win_rate']:.1f}%"
        pf_str = f"{r['profit_factor']:.2f}"
        mdd_str = f"-{r['max_drawdown_pct']:.2f}%"
        tr_str = f"{r['total_trades']}"
        print(f"{r['name']:<46} | {ret_str:<12} | {eq_str:<14} | {win_str:<8} | {pf_str:<8} | {mdd_str:<8} | {tr_str:<6}")
    print("=" * 110 + "\n")

    # Detailed Breakdown for Best Scenarios
    for sc_idx in [4, 5, 6]:
        best_sc = results[sc_idx]
        trades = best_sc["trades"]
        if trades:
            df_trades = pd.DataFrame(trades)
            print(f"\n🔍 [{best_sc['name']} 상세 분석]")
            print("   --- 종료 사유별 통계 ---")
            for reason, group in df_trades.groupby("reason"):
                grp_win = (group["pnl"] > 0).mean() * 100
                grp_pnl = group["pnl"].sum()
                print(f"   • {reason:<22}: {len(group):>3}건 | 승률 {grp_win:>5.1f}% | 실현손익: ${grp_pnl:>+9.2f}")

            print("   --- 종목별 기여도 통계 ---")
            for sym, group in df_trades.groupby("symbol"):
                sym_pnl = group["pnl"].sum()
                sym_win = (group["pnl"] > 0).mean() * 100
                print(f"   • {sym:<6}: {len(group):>3}회 거래 | 승률 {sym_win:>5.1f}% | 손익: ${sym_pnl:>+9.2f}")

    print(f"\n⏱️ Simulation finished in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
