#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_pre_aug21_14d.py — Faithful backtest of OKX Pre-Aug-21 strategy logic
(commits 9a13bd9 [Aug 10] & a29f116 [Aug 18] / agrolib/quant_system_20260818)
across:
  - Period A (Baseline): 2026-08-07 00:00 ~ 2026-08-21 00:00 UTC (14 days before Aug 21)
  - Period B (Current):  2026-10-28 00:00 ~ 2026-11-11 10:00 UTC (last 14 days)
And direct comparison with the currently deployed BLEND 65/35 strategy on Period B.
"""
import asyncio
import json
import os
import pickle
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import ccxt.async_support as ccxt_async

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from okx_blend_strategy import compute_blend_target_weights, SYMS_8

UNIVERSE_8 = [s.split("-")[0] for s in SYMS_8]
COIN_TO_INST = {s.split("-")[0]: s for s in SYMS_8}

CACHE_PATH = "/home/hongsoonil02/quant_system/data/okx_pre_aug21_compare_cache.pkl"

MAJOR_COINS = ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX", "LINK", "DOT", "BNB", "TRX"]
MAJOR_SYMS = [f"{c}/USDT:USDT" for c in MAJOR_COINS]

EXCLUDE_BASE = set(MAJOR_COINS) | {
    "TSLA", "NVDA", "AAPL", "AMZN", "MSFT", "META", "GOOG", "GOOGL", "COIN",
    "SPCX", "OPENAI", "ANTHROPIC", "RDDT", "MU", "SNDK", "SOXL", "XAU", "CL",
    "SKHY", "KORU", "CBRS", "AEON", "PLTR", "AMD", "INTC", "QCOM", "BABA",
    "UBER", "ABNB", "SNAP", "MSTR", "HOOD", "RIVN", "NIO", "PYPL", "SQ", "SHOP",
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "XAG",
    "KR200", "SKHYNIX", "SHAZ", "ISRG", "ROBO", "RAM", "DRAM", "GME",
}

AUG_WARMUP_MS = int(datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
AUG_START_MS = int(datetime(2026, 8, 7, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
AUG_END_MS = int(datetime(2026, 8, 21, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

NOW_WARMUP_MS = int(datetime(2026, 10, 22, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
NOW_START_MS = int(datetime(2026, 10, 28, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
NOW_END_MS = int(datetime(2026, 11, 11, 10, 0, tzinfo=timezone.utc).timestamp() * 1000)

FEE_RATE = 0.0005  # OKX taker 0.05% per leg
LEVERAGE = 10


def calc_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    hl2 = (df["h"] + df["l"]) / 2.0
    atr = (df["h"].combine(df["c"].shift(), max) - df["l"].combine(df["c"].shift(), min)).rolling(period).mean()
    fu = (hl2 + multiplier * atr).to_numpy(dtype=float, copy=True)
    fl = (hl2 - multiplier * atr).to_numpy(dtype=float, copy=True)
    c = df["c"].to_numpy(dtype=float)
    n = len(df)
    sd = np.ones(n, dtype=int)
    sv = np.zeros(n, dtype=float)
    for i in range(period, n):
        if c[i] > fu[i - 1]:
            sd[i] = 1
        elif c[i] < fl[i - 1]:
            sd[i] = -1
        else:
            sd[i] = sd[i - 1]
            if sd[i] == 1 and fl[i] < fl[i - 1]:
                fl[i] = fl[i - 1]
            if sd[i] == -1 and fu[i] > fu[i - 1]:
                fu[i] = fu[i - 1]
        sv[i] = fl[i] if sd[i] == 1 else fu[i]
    return pd.Series(sd, index=df.index), pd.Series(sv, index=df.index)


def calc_stoch_rsi(series: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rmin = rsi.rolling(period).min()
    rmax = rsi.rolling(period).max()
    stoch = (rsi - rmin) / (rmax - rmin).replace(0.0, np.nan)
    k = stoch.rolling(smooth_k).mean() * 100.0
    d = k.rolling(smooth_d).mean()
    return k, d


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("t").reset_index(drop=True)
    df["st_d_l"], df["st_v_l"] = calc_supertrend(df, 10, 3.0)
    df["st_d_t"], df["st_v_t"] = calc_supertrend(df, 10, 2.0)
    df["vol_ma"] = df["v"].rolling(20).mean()
    df["stoch_k"], df["stoch_d"] = calc_stoch_rsi(df["c"], 14, 3, 3)
    df["ema50"] = df["c"].ewm(span=50, adjust=False).mean()
    return df


async def fetch_range(ex, sym: str, tf: str, since_ms: int, end_ms: int) -> pd.DataFrame:
    rows = []
    cur = since_ms
    while cur < end_ms:
        try:
            batch = await ex.fetch_ohlcv(sym, tf, since=cur, limit=300)
        except Exception:
            break
        if not batch:
            break
        rows.extend([b for b in batch if b[0] <= end_ms])
        last_t = batch[-1][0]
        if last_t <= cur or last_t >= end_ms or len(batch) < 300:
            break
        cur = last_t + 1
        await asyncio.sleep(0.04)
    if not rows:
        return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"])
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"]).drop_duplicates("t")
    return df.sort_values("t").reset_index(drop=True)


async def load_or_fetch_data():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    ex = ccxt_async.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    try:
        markets = await ex.load_markets()
        tickers = await ex.fetch_tickers()
        venture_cands = []
        for s, t in tickers.items():
            if s in markets and markets[s].get("swap") and "USDT" in s:
                base = s.split("-")[0].split("/")[0]
                if base in EXCLUDE_BASE:
                    continue
                vol = t.get("quoteVolume")
                if vol is None:
                    try:
                        vc = float(t.get("info", {}).get("volCcy24h") or 0)
                        last = float(t.get("last") or 0)
                        vol = vc * last
                    except Exception:
                        vol = 0
                vol = float(vol or 0)
                if vol >= 10_000_000:
                    venture_cands.append((s, vol))
        venture_cands.sort(key=lambda x: -x[1])
        # Top 25 liquid venture swaps + historical Aug winners if listed (OKB, TRB, LTC, BOME, HYPE)
        top_venture = [s for s, _ in venture_cands[:25]]
        for extra in ["OKB/USDT:USDT", "LTC/USDT:USDT", "BOME/USDT:USDT", "TRB/USDT:USDT", "FIL/USDT:USDT"]:
            if extra in markets and extra not in top_venture:
                top_venture.append(extra)

        print(f"Fetching Major ({len(MAJOR_SYMS)}) + Venture ({len(top_venture)}) for Aug & Now...")
        data = {
            "aug": {"major": {}, "venture": {}, "htf_1h": {}},
            "now": {"major": {}, "venture": {}, "htf_1h": {}},
        }
        for period_key, warmup_ms, end_ms in [
            ("aug", AUG_WARMUP_MS, AUG_END_MS),
            ("now", NOW_WARMUP_MS, NOW_END_MS),
        ]:
            # BTC 1h regime (need 220h before start -> fetch extra 10d before start)
            btc_start = warmup_ms - 10 * 86400 * 1000
            data[period_key]["htf_1h"]["BTC/USDT:USDT"] = await fetch_range(ex, "BTC/USDT:USDT", "1h", btc_start, end_ms)

            for sym in MAJOR_SYMS:
                df30 = await fetch_range(ex, sym, "30m", warmup_ms, end_ms)
                if len(df30) >= 250:
                    data[period_key]["major"][sym] = df30
                df1h = await fetch_range(ex, sym, "1h", warmup_ms, end_ms)
                if len(df1h) >= 60:
                    data[period_key]["htf_1h"][sym] = df1h

            for sym in top_venture:
                df15 = await fetch_range(ex, sym, "15m", warmup_ms, end_ms)
                if len(df15) >= 250:
                    data[period_key]["venture"][sym] = df15
                df1h = await fetch_range(ex, sym, "1h", warmup_ms, end_ms)
                if len(df1h) >= 60:
                    data[period_key]["htf_1h"][sym] = df1h

        # Also fetch 8-majors 1D and 4H for BLEND 65/35 Period B comparison (with 120d warmup before Oct 28)
        blend_warmup_ms = NOW_START_MS - 120 * 86400 * 1000
        blend_aug_warmup_ms = AUG_START_MS - 120 * 86400 * 1000
        data["blend_1d"] = {}
        data["blend_4h"] = {}
        for coin in UNIVERSE_8:
            sym = f"{coin}/USDT:USDT"
            data["blend_1d"][coin] = await fetch_range(ex, sym, "1d", blend_aug_warmup_ms, NOW_END_MS)
            data["blend_4h"][coin] = await fetch_range(ex, sym, "4h", blend_aug_warmup_ms, NOW_END_MS)

        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(data, f)
        return data
    finally:
        await ex.close()


def build_regime_maps(btc_1h: pd.DataFrame):
    df = btc_1h.copy().sort_values("t").reset_index(drop=True)
    df["ema200"] = df["c"].ewm(span=200, adjust=False).mean()
    df["ema50"] = df["c"].ewm(span=50, adjust=False).mean()
    df["long_ok"] = df["c"] >= df["ema200"]
    df["short_ok"] = ~(df["long_ok"] & (df["ema50"] > df["ema50"].shift(5)))
    ts = df["t"].to_numpy()
    lok = df["long_ok"].to_numpy()
    sok = df["short_ok"].to_numpy()
    return ts, lok, sok


def build_htf_slope_map(df_1h: pd.DataFrame):
    df = df_1h.copy().sort_values("t").reset_index(drop=True)
    ema = df["c"].ewm(span=50, adjust=False).mean()
    slope = (ema - ema.shift(5)) / ema.shift(5)
    return df["t"].to_numpy(), slope.fillna(0.0).to_numpy()


def simulate_pre_aug21(
    period_data: dict,
    start_ms: int,
    end_ms: int,
    version: str = "aug18",          # "aug18" (a29f116) or "aug10" (9a13bd9)
    initial_equity: float = 3981.91,
    min_pos_margin: float = 100.0,
    max_open_positions: int = 15,
):
    """
    Reproduce BaseStrategyBrain (OKXMajorStrategyBrain 30m + OKXVentureStrategyBrain 15m).
    version="aug18":
      - Major weight=0.5, Venture weight=1.5
      - Major Hard SL=-30%, Venture Hard SL=-15%
      - HTF 1h EMA50 slope > 0.5% blocks short
      - Short regime filter active
      - Winner pyramiding active (0.35 Major, 0.50 Venture)
      - Equity/15 * weight * ratio sizing (min_pos_margin floor)
    version="aug10":
      - Uniform weight=1.0, Hard SL=-30%
      - No HTF short filter, no short regime filter, no pyramiding
      - max_open_positions=10 (or 5 per bot), proportional 5%..20% sizing (or equity/10)
    """
    btc_ts, btc_lok, btc_sok = build_regime_maps(period_data["htf_1h"]["BTC/USDT:USDT"])

    # Prepare indicators per bot symbol
    bots = []
    for sym, raw in period_data["major"].items():
        df = prepare_df(raw)
        htf = period_data["htf_1h"].get(sym)
        htf_ts, htf_sl = build_htf_slope_map(htf) if htf is not None else (np.array([]), np.array([]))
        bots.append(("major", sym, df, 30, htf_ts, htf_sl))
    for sym, raw in period_data["venture"].items():
        df = prepare_df(raw)
        htf = period_data["htf_1h"].get(sym)
        htf_ts, htf_sl = build_htf_slope_map(htf) if htf is not None else (np.array([]), np.array([]))
        bots.append(("venture", sym, df, 15, htf_ts, htf_sl))

    # Build lookup by timestamp for unified timeline simulation
    events_by_t = {}
    for bot_type, sym, df, tf_min, htf_ts, htf_sl in bots:
        # map timestamp -> row index (skip first 210 warmup rows, and filter start_ms..end_ms)
        t_arr = df["t"].to_numpy()
        for i in range(205, len(df)):
            t = int(t_arr[i])
            if t < start_ms or t > end_ms:
                continue
            events_by_t.setdefault(t, []).append((bot_type, sym, df, i, tf_min, htf_ts, htf_sl))

    all_ts = sorted(events_by_t.keys())
    cash = initial_equity
    positions = {}  # (sym, side) -> {'margin', 'qty_notional', 'avg', 'bot_type', 'max_pnl'}
    dca_state = {}  # sym -> dict
    last_px = {}
    closed_trades = []
    equity_curve = []
    total_fees = 0.0

    def get_equity():
        eq = cash
        for (sym, side), pos in positions.items():
            px = last_px.get(sym, pos["avg"])
            sgn = 1.0 if side == "long" else -1.0
            pnl_pct = ((px - pos["avg"]) / pos["avg"]) * LEVERAGE * sgn
            eq += pos["margin"] * (1.0 + pnl_pct)
        return eq

    def lookup_regime(t):
        idx = np.searchsorted(btc_ts, t, side="right") - 1
        if idx < 0:
            return True, True
        return bool(btc_lok[idx]), bool(btc_sok[idx])

    def lookup_htf_up(htf_ts, htf_sl, t):
        if len(htf_ts) == 0:
            return False
        idx = np.searchsorted(htf_ts, t, side="right") - 1
        if idx < 0:
            return False
        return float(htf_sl[idx]) > 0.005

    def open_or_add(sym, side, px, margin_amt, bot_type, tag):
        nonlocal cash, total_fees
        if margin_amt <= 0 or margin_amt * (1.0 + LEVERAGE * FEE_RATE) > cash:
            return False
        fee = margin_amt * LEVERAGE * FEE_RATE
        cash -= (margin_amt + fee)
        total_fees += fee
        key = (sym, side)
        add_notional = margin_amt * LEVERAGE
        add_coin = add_notional / px
        if key in positions:
            p = positions[key]
            tot_coin = p["coin"] + add_coin
            p["avg"] = (p["avg"] * p["coin"] + px * add_coin) / tot_coin
            p["coin"] = tot_coin
            p["margin"] += margin_amt
        else:
            positions[key] = {
                "margin": margin_amt,
                "coin": add_coin,
                "avg": px,
                "bot_type": bot_type,
                "tag": tag,
            }
        return True

    def close_fraction(sym, side, px, frac, reason):
        nonlocal cash, total_fees
        key = (sym, side)
        p = positions.get(key)
        if not p:
            return 0.0
        frac = min(1.0, max(0.0, frac))
        margin_rel = p["margin"] * frac
        sgn = 1.0 if side == "long" else -1.0
        pnl_pct = ((px - p["avg"]) / p["avg"]) * LEVERAGE * sgn
        gross_pnl = margin_rel * pnl_pct
        fee = margin_rel * LEVERAGE * FEE_RATE
        cash += margin_rel + gross_pnl - fee
        total_fees += fee
        net_pnl = gross_pnl - fee - (margin_rel * LEVERAGE * FEE_RATE)  # net of both legs for trade report
        # Track realized equity delta cleanly via cash changes; store close event
        closed_trades.append({
            "sym": sym,
            "bot": p["bot_type"],
            "side": side,
            "frac": frac,
            "margin": margin_rel,
            "pnl_net": gross_pnl - fee,  # exit fee deducted here; entry fee already deducted from cash
            "pnl_pct_lev": pnl_pct * 100.0,
            "reason": reason,
        })
        if frac >= 0.9999:
            del positions[key]
        else:
            p["margin"] -= margin_rel
            p["coin"] *= (1.0 - frac)
        return gross_pnl - fee

    for t in all_ts:
        long_reg_ok, short_reg_ok = lookup_regime(t)
        if version == "aug10":
            short_reg_ok = True

        for bot_type, sym, df, i, tf_min, htf_ts, htf_sl in events_by_t[t]:
            prev = df.iloc[i - 1]
            curr = df.iloc[i]
            prev2 = df.iloc[i - 2]
            px = float(curr["c"])
            last_px[sym] = px

            vol_cond = (curr["v"] > prev["vol_ma"] * 1.0) or (prev["v"] > prev2["vol_ma"] * 1.0)
            is_long_bo = prev["st_d_l"] == -1 and curr["st_d_l"] == 1
            is_short_bo = prev["st_d_l"] == 1 and curr["st_d_l"] == -1
            is_long_pb = curr["st_d_l"] == 1 and prev["stoch_k"] < 20 and curr["stoch_k"] >= 20
            is_short_pb = curr["st_d_l"] == -1 and prev["stoch_k"] > 80 and curr["stoch_k"] <= 80

            is_ema_up = curr["c"] > curr["ema50"]
            is_ema_dn = curr["c"] < curr["ema50"]
            is_long_tc = curr["st_d_l"] == 1 and prev["st_d_l"] == 1
            is_short_tc = curr["st_d_l"] == -1 and prev["st_d_l"] == -1
            is_long_mom = curr["stoch_k"] > curr["stoch_d"] and curr["stoch_k"] < 80
            is_short_mom = curr["stoch_k"] < curr["stoch_d"] and curr["stoch_k"] > 20

            long_score = (50 if is_ema_up else 0) + (40 if is_long_bo else 0) + (30 if is_long_pb else 0) + (20 if is_long_tc else 0) + (20 if is_long_mom else 0)
            short_score = (50 if is_ema_dn else 0) + (40 if is_short_bo else 0) + (30 if is_short_pb else 0) + (20 if is_short_tc else 0) + (20 if is_short_mom else 0)

            is_long_sig = (long_score >= 70) and vol_cond and long_reg_ok
            is_short_sig = (short_score >= 90) and vol_cond and short_reg_ok

            dca = dca_state.setdefault(sym, {
                "entry_count": 0, "exit_count": 0, "pyramid_count": 0,
                "last_entry_t": 0, "last_exit_t": 0, "first_entry_t": 0,
                "max_pnl_pct": 0.0, "last_close_t": 0, "last_close_side": None,
            })

            def dca_ready(t_now):
                return (t_now - dca.get("last_entry_t", 0)) >= 4 * tf_min * 60 * 1000

            def calc_margin(entry_type):
                eq = max(cash, get_equity())
                free = max(0.0, cash)
                if version == "aug10":
                    # Pre-Aug-18: uniform 5% prior Kelly * ratio or equity/10
                    base_m = eq / max(1, max_open_positions)
                    ratios = {"new": 0.50, "dca": 0.25, "pyramid": 0.35, "flip": 0.30, "reentry": 0.30}
                    m = min(base_m * ratios.get(entry_type, 0.5), free * 0.95)
                    return m
                weight = 0.5 if bot_type == "major" else 1.5
                base_m = (eq / max(1, max_open_positions)) * weight
                pyr_r = 0.35 if bot_type == "major" else 0.50
                ratios = {"new": 0.50, "dca": 0.25, "pyramid": pyr_r, "flip": 0.30, "reentry": 0.30}
                m = min(base_m * ratios.get(entry_type, 0.50), free * 0.95)
                if m < min_pos_margin:
                    if free >= min_pos_margin * 1.1:
                        m = min_pos_margin
                    else:
                        return 0.0
                return m

            hard_sl = -0.30 if (version == "aug10" or bot_type == "major") else -0.15
            flip_enabled = (bot_type == "major")
            reentry_enabled = (bot_type == "venture")

            pos_l = positions.get((sym, "long"))
            pos_s = positions.get((sym, "short"))

            # Long management
            if pos_l is not None:
                avg_l = pos_l["avg"]
                pnl_l = ((px - avg_l) / avg_l) * LEVERAGE
                dca["max_pnl_pct"] = max(dca["max_pnl_pct"], pnl_l)
                is_profit = px > avg_l * 1.03
                st_v_l = curr["st_v_t"] if is_profit else curr["st_v_l"]
                st_d_l = curr["st_d_t"] if is_profit else curr["st_d_l"]
                close_l_sig = (st_d_l == -1) or (px < st_v_l)

                force_l = False
                hard_l = False
                tp_l = False
                if pnl_l <= hard_sl:
                    force_l = True
                    hard_l = True
                elif dca["max_pnl_pct"] >= 1.00 and pnl_l <= dca["max_pnl_pct"] - 0.30:
                    force_l = True
                elif dca["max_pnl_pct"] >= 0.50 and pnl_l <= dca["max_pnl_pct"] * 0.60:
                    force_l = True
                elif dca["max_pnl_pct"] >= 0.40 and pnl_l <= 0.30:
                    force_l = True
                elif dca["max_pnl_pct"] >= 0.20 and pnl_l <= 0.05:
                    force_l = True
                elif pnl_l >= 0.15 and dca["exit_count"] == 0:
                    tp_l = True
                elif pnl_l >= 0.30 and dca["exit_count"] == 1:
                    tp_l = True
                elif pnl_l >= 0.50 and dca["exit_count"] == 2:
                    tp_l = True

                # Winner pyramiding (Aug 18+)
                if version == "aug18":
                    if pnl_l >= 0.40 and dca["pyramid_count"] == 0 and dca_ready(t) and is_ema_up and curr["st_d_l"] == 1:
                        if open_or_add(sym, "long", px, calc_margin("pyramid"), bot_type, "pyr1"):
                            dca["pyramid_count"] = 1
                            dca["last_entry_t"] = t
                    elif pnl_l >= 1.00 and dca["pyramid_count"] == 1 and dca_ready(t) and is_ema_up and is_long_mom:
                        if open_or_add(sym, "long", px, calc_margin("pyramid"), bot_type, "pyr2"):
                            dca["pyramid_count"] = 2
                            dca["last_entry_t"] = t
                    if dca["pyramid_count"] >= 1 and pnl_l < 0.20:
                        force_l = True

                if dca["exit_count"] > 0 and px < pos_l["avg"]:
                    candles_held = (t - dca.get("first_entry_t", t)) / (tf_min * 60 * 1000)
                    if candles_held >= 3:
                        force_l = True

                if force_l and dca["last_exit_t"] != t:
                    close_fraction(sym, "long", px, 1.0, "HARD_SL" if hard_l else "TRAIL_BE")
                    if flip_enabled and not hard_l:
                        open_or_add(sym, "short", px, calc_margin("flip"), bot_type, "flip")
                    dca["exit_count"] = 8
                    dca["entry_count"] = 0
                    dca["pyramid_count"] = 0
                    dca["last_exit_t"] = t
                    dca["max_pnl_pct"] = 0.0
                    dca["last_close_t"] = t
                    dca["last_close_side"] = "long"
                elif (close_l_sig or tp_l) and dca["exit_count"] < 8 and dca["last_exit_t"] != t:
                    frac = 1.0 / max(1, (3 - dca["exit_count"]))
                    close_fraction(sym, "long", px, frac, "TP_SCALE" if tp_l else "ST_SCALE")
                    dca["exit_count"] += 1
                    dca["last_exit_t"] = t
                    if dca["exit_count"] >= 3 or (sym, "long") not in positions:
                        dca["entry_count"] = 0
                        dca["exit_count"] = 0
                        dca["pyramid_count"] = 0
                        dca["max_pnl_pct"] = 0.0
                elif not (close_l_sig or tp_l) and not force_l:
                    if dca["entry_count"] < 8 and dca_ready(t):
                        if open_or_add(sym, "long", px, calc_margin("dca"), bot_type, "dca"):
                            dca["entry_count"] += 1
                            dca["last_entry_t"] = t

            # Short management
            if pos_s is not None:
                avg_s = pos_s["avg"]
                pnl_s = ((avg_s - px) / avg_s) * LEVERAGE
                dca["max_pnl_pct"] = max(dca["max_pnl_pct"], pnl_s)
                is_profit = px < avg_s * (2.0 - 1.03)
                st_v_s = curr["st_v_t"] if is_profit else curr["st_v_l"]
                st_d_s = curr["st_d_t"] if is_profit else curr["st_d_l"]
                close_s_sig = (st_d_s == 1) or (px > st_v_s)

                force_s = False
                hard_s = False
                tp_s = False
                if pnl_s <= hard_sl:
                    force_s = True
                    hard_s = True
                elif dca["max_pnl_pct"] >= 1.00 and pnl_s <= dca["max_pnl_pct"] - 0.30:
                    force_s = True
                elif dca["max_pnl_pct"] >= 0.50 and pnl_s <= dca["max_pnl_pct"] * 0.60:
                    force_s = True
                elif dca["max_pnl_pct"] >= 0.40 and pnl_s <= 0.30:
                    force_s = True
                elif dca["max_pnl_pct"] >= 0.20 and pnl_s <= 0.05:
                    force_s = True
                elif pnl_s >= 0.15 and dca["exit_count"] == 0:
                    tp_s = True
                elif pnl_s >= 0.30 and dca["exit_count"] == 1:
                    tp_s = True
                elif pnl_s >= 0.50 and dca["exit_count"] == 2:
                    tp_s = True

                if version == "aug18":
                    if pnl_s >= 0.40 and dca["pyramid_count"] == 0 and dca_ready(t) and is_ema_dn and curr["st_d_l"] == -1:
                        if open_or_add(sym, "short", px, calc_margin("pyramid"), bot_type, "pyr1"):
                            dca["pyramid_count"] = 1
                            dca["last_entry_t"] = t
                    elif pnl_s >= 1.00 and dca["pyramid_count"] == 1 and dca_ready(t) and is_ema_dn and is_short_mom:
                        if open_or_add(sym, "short", px, calc_margin("pyramid"), bot_type, "pyr2"):
                            dca["pyramid_count"] = 2
                            dca["last_entry_t"] = t
                    if dca["pyramid_count"] >= 1 and pnl_s < 0.20:
                        force_s = True

                if dca["exit_count"] > 0 and px > pos_s["avg"]:
                    candles_held = (t - dca.get("first_entry_t", t)) / (tf_min * 60 * 1000)
                    if candles_held >= 3:
                        force_s = True

                if force_s and dca["last_exit_t"] != t:
                    close_fraction(sym, "short", px, 1.0, "HARD_SL" if hard_s else "TRAIL_BE")
                    if flip_enabled and not hard_s and long_reg_ok:
                        open_or_add(sym, "long", px, calc_margin("flip"), bot_type, "flip")
                    dca["exit_count"] = 8
                    dca["entry_count"] = 0
                    dca["pyramid_count"] = 0
                    dca["last_exit_t"] = t
                    dca["max_pnl_pct"] = 0.0
                    dca["last_close_t"] = t
                    dca["last_close_side"] = "short"
                elif (close_s_sig or tp_s) and dca["exit_count"] < 8 and dca["last_exit_t"] != t:
                    frac = 1.0 / max(1, (3 - dca["exit_count"]))
                    close_fraction(sym, "short", px, frac, "TP_SCALE" if tp_s else "ST_SCALE")
                    dca["exit_count"] += 1
                    dca["last_exit_t"] = t
                    if dca["exit_count"] >= 3 or (sym, "short") not in positions:
                        dca["entry_count"] = 0
                        dca["exit_count"] = 0
                        dca["pyramid_count"] = 0
                        dca["max_pnl_pct"] = 0.0
                elif is_short_pb and not force_s and dca["entry_count"] < 8 and dca_ready(t):
                    if open_or_add(sym, "short", px, calc_margin("dca"), bot_type, "dca"):
                        dca["entry_count"] += 1
                        dca["last_entry_t"] = t

            # Flat entry checks (only if neither long nor short open on this sym)
            if (sym, "long") not in positions and (sym, "short") not in positions:
                active_syms = {s for s, _ in positions.keys()}
                if len(active_syms) >= max_open_positions:
                    continue

                # Re-entry
                if reentry_enabled and dca.get("last_close_t") and dca.get("last_entry_t") != t:
                    cooldown_ms = 4 * tf_min * 60 * 1000
                    if (t - dca["last_close_t"]) >= cooldown_ms:
                        sc = dca.get("last_close_side")
                        re_l = sc == "long" and curr["st_d_l"] == 1 and is_ema_up and long_reg_ok
                        re_s = sc == "short" and curr["st_d_l"] == -1 and is_ema_dn and short_reg_ok
                        if re_l or re_s:
                            side = "long" if re_l else "short"
                            if open_or_add(sym, side, px, calc_margin("reentry"), bot_type, "reentry"):
                                dca["entry_count"] = 1
                                dca["exit_count"] = 0
                                dca["pyramid_count"] = 0
                                dca["max_pnl_pct"] = 0.0
                                dca["last_entry_t"] = t
                                dca["first_entry_t"] = t
                                dca["last_close_t"] = 0
                            continue

                if is_long_sig and dca.get("last_entry_t") != t:
                    if open_or_add(sym, "long", px, calc_margin("new"), bot_type, "new"):
                        dca["entry_count"] = 1
                        dca["exit_count"] = 0
                        dca["pyramid_count"] = 0
                        dca["max_pnl_pct"] = 0.0
                        dca["last_entry_t"] = t
                        dca["first_entry_t"] = t
                elif is_short_sig and dca.get("last_entry_t") != t:
                    htf_blocked = (version == "aug18") and lookup_htf_up(htf_ts, htf_sl, t)
                    if not htf_blocked:
                        if open_or_add(sym, "short", px, calc_margin("new"), bot_type, "new"):
                            dca["entry_count"] = 1
                            dca["exit_count"] = 0
                            dca["pyramid_count"] = 0
                            dca["max_pnl_pct"] = 0.0
                            dca["last_entry_t"] = t
                            dca["first_entry_t"] = t

        equity_curve.append((t, get_equity()))

    # Mark-to-market settle open positions at end
    open_mtm = 0.0
    for (sym, side), p in list(positions.items()):
        open_mtm += close_fraction(sym, side, last_px.get(sym, p["avg"]), 1.0, "EOP_MTM")

    final_eq = cash
    eq_vals = np.array([e for _, e in equity_curve], dtype=float)
    mdd = 0.0
    if len(eq_vals) > 0:
        peak = np.maximum.accumulate(eq_vals)
        dd = (eq_vals - peak) / peak * 100.0
        mdd = float(dd.min())

    # Note: closed_trades pnl_net didn't subtract entry fee; actual total PnL = final_eq - initial_equity
    wins = [tr for tr in closed_trades if tr["pnl_net"] > 0]
    losses = [tr for tr in closed_trades if tr["pnl_net"] <= 0]
    gross_win = sum(tr["pnl_net"] for tr in wins)
    gross_loss = abs(sum(tr["pnl_net"] for tr in losses))
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    by_bot = {}
    for tr in closed_trades:
        b = tr["bot"]
        by_bot.setdefault(b, {"n": 0, "wins": 0, "pnl": 0.0})
        by_bot[b]["n"] += 1
        if tr["pnl_net"] > 0:
            by_bot[b]["wins"] += 1
        by_bot[b]["pnl"] += tr["pnl_net"]

    return {
        "initial_eq": initial_equity,
        "final_eq": final_eq,
        "ret_pct": (final_eq - initial_equity) / initial_equity * 100.0,
        "net_pnl": final_eq - initial_equity,
        "mdd_pct": mdd,
        "n_closes": len(closed_trades),
        "wr_pct": len(wins) / len(closed_trades) * 100.0 if closed_trades else 0.0,
        "pf": pf,
        "fees": total_fees,
        "by_bot": by_bot,
        "trades": closed_trades,
    }


def simulate_blend_window(data: dict, start_ms: int, end_ms: int, initial_equity: float = 3981.91):
    """Simulate current live BLEND 65/35 engine on [start_ms, end_ms] using 1D & 4H data."""
    def build_panel(raw_map):
        h_cols, l_cols, c_cols = {}, {}, {}
        for coin, inst in COIN_TO_INST.items():
            df = raw_map[coin].copy().drop_duplicates("t").sort_values("t")
            idx = pd.to_datetime(df["t"], unit="ms", utc=True)
            h_cols[inst] = pd.Series(df["h"].to_numpy(float), index=idx)
            l_cols[inst] = pd.Series(df["l"].to_numpy(float), index=idx)
            c_cols[inst] = pd.Series(df["c"].to_numpy(float), index=idx)
        return pd.DataFrame(h_cols).ffill(), pd.DataFrame(l_cols).ffill(), pd.DataFrame(c_cols).ffill()

    h4_all, l4_all, c4_all = build_panel(data["blend_4h"])
    _, _, c1_all = build_panel(data["blend_1d"])

    start_dt = pd.to_datetime(start_ms, unit="ms", utc=True)
    end_dt = pd.to_datetime(end_ms, unit="ms", utc=True)
    eval_dts = [dt for dt in c4_all.index if start_dt <= dt <= end_dt]

    eq = initial_equity
    prev_w = {s: 0.0 for s in SYMS_8}
    prev_px = {}
    curve = []
    trades = 0
    fees = 0.0

    for dt in eval_dts:
        sub_h4 = h4_all.loc[:dt].tail(240)
        sub_l4 = l4_all.loc[:dt].tail(240)
        sub_c4 = c4_all.loc[:dt].tail(240)
        # Confirmed daily bars only: bar open + 24h <= current 4h bar close (~dt + 4h)
        dt_close = dt + pd.Timedelta(hours=4)
        sub_c1 = c1_all.loc[c1_all.index + pd.Timedelta(hours=24) <= dt_close].tail(160)
        cur_px = {s: float(sub_c4.iloc[-1][s]) for s in SYMS_8}

        bar_pnl = 0.0
        for s, w in prev_w.items():
            if w != 0.0 and s in prev_px:
                r = (cur_px[s] - prev_px[s]) / prev_px[s]
                bar_pnl += eq * w * r
        eq += bar_pnl

        tgt_w, _ = compute_blend_target_weights(sub_h4, sub_l4, sub_c4, sub_c1)
        turnover = sum(abs(tgt_w[s] - prev_w[s]) for s in SYMS_8)
        if turnover > 0.02:
            fee = eq * turnover * FEE_RATE
            eq -= fee
            fees += fee
            trades += sum(1 for s in SYMS_8 if abs(tgt_w[s] - prev_w[s]) > 0.01)
            prev_w = tgt_w
        prev_px = cur_px
        curve.append(eq)

    arr = np.array(curve, dtype=float)
    mdd = float(((arr - np.maximum.accumulate(arr)) / np.maximum.accumulate(arr) * 100.0).min()) if len(arr) else 0.0
    return {
        "initial_eq": initial_equity,
        "final_eq": eq,
        "ret_pct": (eq - initial_equity) / initial_equity * 100.0,
        "net_pnl": eq - initial_equity,
        "mdd_pct": mdd,
        "rebal_orders": trades,
        "fees": fees,
    }


def main():
    data = asyncio.run(load_or_fetch_data())
    print("\nData loaded:")
    for pk in ("aug", "now"):
        print(f"  {pk}: Major={len(data[pk]['major'])}, Venture={len(data[pk]['venture'])}, 1h={len(data[pk]['htf_1h'])}")

    scenarios = [
        ("Period A: 08/07~08/21 (Aug18 로직, $3,982, min$100)", "aug", AUG_START_MS, AUG_END_MS, "aug18", 3981.91, 100.0, 15),
        ("Period A: 08/07~08/21 (Aug18 로직, $3,982, 비례사이징)", "aug", AUG_START_MS, AUG_END_MS, "aug18", 3981.91, 10.0, 15),
        ("Period A: 08/07~08/21 (Aug18 로직, $19,000, min$100)", "aug", AUG_START_MS, AUG_END_MS, "aug18", 19000.0, 100.0, 15),
        ("Period A: 08/07~08/21 (Aug10 로직, $3,982, 비례)", "aug", AUG_START_MS, AUG_END_MS, "aug10", 3981.91, 10.0, 10),

        ("Period B: 10/28~11/11 (Aug18 로직, $3,982, min$100)", "now", NOW_START_MS, NOW_END_MS, "aug18", 3981.91, 100.0, 15),
        ("Period B: 10/28~11/11 (Aug18 로직, $3,982, 비례사이징)", "now", NOW_START_MS, NOW_END_MS, "aug18", 3981.91, 10.0, 15),
        ("Period B: 10/28~11/11 (Aug18 로직, $1,100, min$100)", "now", NOW_START_MS, NOW_END_MS, "aug18", 1100.0, 100.0, 15),
        ("Period B: 10/28~11/11 (Aug18 로직, $19,000, min$100)", "now", NOW_START_MS, NOW_END_MS, "aug18", 19000.0, 100.0, 15),
        ("Period B: 10/28~11/11 (Aug10 로직, $3,982, 비례)", "now", NOW_START_MS, NOW_END_MS, "aug10", 3981.91, 10.0, 10),
    ]

    print("\n" + "=" * 105)
    print(f"{'Scenario':<54} | {'Ret%':>7} | {'NetPnL':>9} | {'MDD%':>6} | {'Closes':>6} | {'WR%':>5} | {'PF':>5} | {'Fees':>7}")
    print("-" * 105)
    for label, pkey, s_ms, e_ms, ver, eq0, mmin, mpos in scenarios:
        res = simulate_pre_aug21(data[pkey], s_ms, e_ms, version=ver, initial_equity=eq0, min_pos_margin=mmin, max_open_positions=mpos)
        print(f"{label:<54} | {res['ret_pct']:+6.2f}% | ${res['net_pnl']:+8.2f} | {res['mdd_pct']:6.2f}% | {res['n_closes']:6d} | {res['wr_pct']:5.1f}% | {res['pf']:5.2f} | ${res['fees']:6.1f}")
        for b, bd in sorted(res["by_bot"].items()):
            bwr = bd["wins"] / bd["n"] * 100 if bd["n"] else 0
            print(f"    -> {b:<8}: {bd['n']:4d} closes, WR {bwr:5.1f}%, gross-exit PnL ${bd['pnl']:+8.2f}")

        # Top/Bottom symbols for key runs
        if "Aug18 로직, $3,982, min$100" in label or "Aug18 로직, $3,982, 비례사이징" in label:
            sym_pnl = {}
            for tr in res["trades"]:
                sym_pnl[tr["sym"]] = sym_pnl.get(tr["sym"], 0.0) + tr["pnl_net"]
            ss = sorted(sym_pnl.items(), key=lambda x: -x[1])
            print("    -> Top3:", [(s.split("/")[0], round(v, 1)) for s, v in ss[:3]], "Bot3:", [(s.split("/")[0], round(v, 1)) for s, v in ss[-3:]])

    print("\n" + "=" * 105)
    print("BLEND 65/35 (Current Live Engine) Comparison ($3,982 USDT):")
    for lbl, s_ms, e_ms in [("Period A: 08/07~08/21", AUG_START_MS, AUG_END_MS), ("Period B: 10/28~11/11", NOW_START_MS, NOW_END_MS)]:
        bres = simulate_blend_window(data, s_ms, e_ms, initial_equity=3981.91)
        print(f"  {lbl:<25} | Ret {bres['ret_pct']:+6.2f}% | Net ${bres['net_pnl']:+8.2f} | MDD {bres['mdd_pct']:6.2f}% | Orders {bres['rebal_orders']:3d} | Fees ${bres['fees']:5.2f}")


if __name__ == "__main__":
    main()
