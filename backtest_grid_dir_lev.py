# -*- coding: utf-8 -*-
"""
backtest_grid_dir_lev.py — Grid validation for ideas #2 (partial TP) and #3 (tighter MAX_SL)
crossed with direction (long / short / both) and leverage (1x / 2x / 3x / 5x) on 2h & 4h.

Variants:
  BASE   : no trailing (current live config after fix)
  PARTIAL: 50% take-profit at +1R, runner keeps original SL and TP
  SL35   : MAX_SL 6% -> 3.5% (avoid wide-stop tail losses)
  P+SL35 : both

Short setup = mirror of long (green pullback below EMA9/VWAP after a dump,
trigger = break of prev-bar low, SL = prev-bar high, TP = 1.86R down).
Uses cached CSV data from backtest_tf_compare.py (state/bt_cache/).
"""

import os
import asyncio
import numpy as np
import pandas as pd
import ccxt.async_support as ccxt_async
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))
CACHE_DIR = os.path.join(CURRENT_DIR, "state", "bt_cache")

SYMBOLS = [
    "SOL/USDT:USDT", "SUI/USDT:USDT", "DOGE/USDT:USDT", "ETH/USDT:USDT", "BTC/USDT:USDT",
    "XRP/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT", "NEAR/USDT:USDT",
    "APT/USDT:USDT", "PEPE/USDT:USDT", "WIF/USDT:USDT", "SHIB/USDT:USDT", "FET/USDT:USDT",
]

TF_DAYS = {"2h": 120, "4h": 120}
RR = 1.86
ALLOC = 0.08
MAX_POS = 3
FEE = 0.0005
SLIP = 0.0005
WINDOW = 35
INITIAL_CAPITAL = 960.0

BASE_BANDS = dict(min_surge=0.025, min_sl=0.015, max_sl=0.060)
SL35_BANDS = dict(min_surge=0.025, min_sl=0.015, max_sl=0.035)

CONFIGS = [
    ("BASE",    dict(partial=False, bands=BASE_BANDS)),
    ("PARTIAL", dict(partial=True,  bands=BASE_BANDS)),
    ("SL35",    dict(partial=False, bands=SL35_BANDS)),
    ("P+SL35",  dict(partial=True,  bands=SL35_BANDS)),
]
DIRECTIONS = ["long", "short", "both"]
LEVERAGES = [1, 2, 3, 5]


async def get_df(ex, symbol, tf):
    cache = os.path.join(CACHE_DIR, f"{symbol.replace('/', '').replace(':', '_')}_{tf}_{TF_DAYS[tf]}d.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    since = ex.milliseconds() - TF_DAYS[tf] * 24 * 3600 * 1000
    tf_ms = (120 if tf == "2h" else 240) * 60 * 1000
    all_bars = []
    while True:
        bars = await ex.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=300)
        if not bars:
            break
        all_bars.extend(bars)
        since = bars[-1][0] + tf_ms
        if len(bars) < 300:
            break
    if len(all_bars) < 100:
        return None
    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df


def build_arrays(df, bands):
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    o = df["open"].values.astype(float)
    v = df["volume"].values.astype(float)
    ts = df["timestamp"].values

    tp = (h + l + c) / 3
    vwap = (pd.Series(tp * v).rolling(WINDOW).sum() / pd.Series(v).rolling(WINDOW).sum().replace(0, 1)).values
    ema9 = pd.Series(c).ewm(span=9, adjust=False).mean().values

    prior_high = pd.Series(h).rolling(5).max().shift(1).values
    prior_low = pd.Series(l).rolling(5).min().shift(1).values
    prior_vol = pd.Series(v).rolling(5).mean().shift(1).values

    with np.errstate(divide="ignore", invalid="ignore"):
        surge = prior_high - prior_low
        surge_pct_lo = surge / prior_low
        surge_pct_hi = surge / prior_high
        retrace_dn = np.where(surge > 0, (prior_high - l) / surge, 1.0)   # pullback down
        retrace_up = np.where(surge > 0, (h - prior_low) / surge, 1.0)    # pullback up
        risk_pct = np.where(h > 0, (h - l) / h, 0.0)

    vol_ok = v <= prior_vol * 1.20
    risk_ok = (risk_pct >= bands["min_sl"]) & (risk_pct <= bands["max_sl"])

    long_mask = (
        (c < o) & (c >= ema9 * 0.995) & (l >= vwap * 0.990)
        & (surge_pct_lo >= bands["min_surge"]) & (retrace_dn <= 0.55) & vol_ok & risk_ok
    )
    short_mask = (
        (c > o) & (c <= ema9 * 1.005) & (h <= vwap * 1.010)
        & (surge_pct_hi >= bands["min_surge"]) & (retrace_up <= 0.55) & vol_ok & risk_ok
    )
    long_mask[:14] = False
    short_mask[:14] = False
    long_idx = np.where(long_mask[:-1] & (h[1:] >= h[:-1]))[0] + 1
    short_idx = np.where(short_mask[:-1] & (l[1:] <= l[:-1]))[0] + 1
    return dict(ts=ts, o=o, h=h, l=l, c=c, n=len(c), long_idx=long_idx, short_idx=short_idx)


def simulate(data, direction, leverage, partial):
    equity = INITIAL_CAPITAL
    peak = equity
    max_dd = 0.0
    open_pos = {}
    trades = []

    entry_events = {}
    max_len = 0
    for sym, d in data.items():
        max_len = max(max_len, d["n"])
        if direction in ("long", "both"):
            for j in d["long_idx"]:
                entry_events.setdefault(j, []).append((sym, "long"))
        if direction in ("short", "both"):
            for j in d["short_idx"]:
                entry_events.setdefault(j, []).append((sym, "short"))

    def book(sym, exit_p, ts, reason, frac=1.0):
        nonlocal equity, peak, max_dd
        p = open_pos[sym]
        notional = p["notional"] * frac
        ret = (exit_p / p["entry"] - 1) if p["side"] == "long" else (p["entry"] / exit_p - 1)
        pnl = notional * ret - notional * FEE * 2
        trades.append(dict(symbol=sym, side=p["side"], reason=reason, ret=ret, pnl=pnl))
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)

    for i in range(max_len):
        for sym in list(open_pos.keys()):
            d = data[sym]
            if i >= d["n"]:
                continue
            p = open_pos[sym]
            hi, lo = d["h"][i], d["l"][i]
            is_long = p["side"] == "long"
            r1 = p["entry_trig"] + p["risk"] if is_long else p["entry_trig"] - p["risk"]

            hit_sl = (lo <= p["sl"]) if is_long else (hi >= p["sl"])
            hit_tp = (hi >= p["tp"]) if is_long else (lo <= p["tp"])
            hit_r1 = (hi >= r1) if is_long else (lo <= r1)

            if i == p["entry_i"]:
                if hit_sl:
                    book(sym, p["sl"] * (1 - SLIP) if is_long else p["sl"] * (1 + SLIP), d["ts"][i], "SL_SAME_BAR")
                    del open_pos[sym]
                elif hit_tp:
                    if partial and hit_r1:
                        book(sym, r1 * (1 - SLIP) if is_long else r1 * (1 + SLIP), d["ts"][i], "PARTIAL_1R", 0.5)
                    book(sym, p["tp"] * (1 - SLIP) if is_long else p["tp"] * (1 + SLIP), d["ts"][i], "TP", 0.5 if partial else 1.0)
                    del open_pos[sym]
                continue

            if partial and not p["half"] and hit_r1 and not hit_sl:
                book(sym, r1 * (1 - SLIP) if is_long else r1 * (1 + SLIP), d["ts"][i], "PARTIAL_1R", 0.5)
                p["half"] = True
                if hit_tp:
                    book(sym, p["tp"] * (1 - SLIP) if is_long else p["tp"] * (1 + SLIP), d["ts"][i], "TP", 0.5)
                    del open_pos[sym]
                continue

            if hit_sl:
                book(sym, p["sl"] * (1 - SLIP) if is_long else p["sl"] * (1 + SLIP), d["ts"][i], "SL", 0.5 if p["half"] else 1.0)
                del open_pos[sym]
            elif hit_tp:
                book(sym, p["tp"] * (1 - SLIP) if is_long else p["tp"] * (1 + SLIP), d["ts"][i], "TP", 0.5 if p["half"] else 1.0)
                del open_pos[sym]

        if len(open_pos) < MAX_POS:
            for sym, side in entry_events.get(i, []):
                if len(open_pos) >= MAX_POS or sym in open_pos:
                    continue
                d = data[sym]
                i0 = i - 1
                if side == "long":
                    trig, sl = d["h"][i0], d["l"][i0]
                    tp = trig + (trig - sl) * RR
                else:
                    trig, sl = d["l"][i0], d["h"][i0]
                    tp = trig - (sl - trig) * RR
                entry = trig * (1 + SLIP) if side == "long" else trig * (1 - SLIP)
                open_pos[sym] = dict(
                    side=side, entry=entry, entry_trig=trig, sl=sl, tp=tp,
                    risk=abs(trig - sl), notional=equity * ALLOC * leverage,
                    entry_i=i, half=False,
                )
    return trades, equity, max_dd


def stats(trades, equity, max_dd):
    if not trades:
        return dict(n=0, wr=0.0, pf=0.0, ret=0.0, dd=0.0)
    wins = [t for t in trades if t["pnl"] > 0.5]
    losses = [t for t in trades if t["pnl"] <= 0.5 and t["ret"] <= -0.001]
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    return dict(n=len(trades), wr=len(wins) / len(trades) * 100,
                pf=gw / gl if gl > 0 else float("inf"),
                ret=(equity / INITIAL_CAPITAL - 1) * 100, dd=max_dd * 100)


async def main():
    ex = ccxt_async.okx({
        "apiKey": os.getenv("OKX_API_KEY"), "secret": os.getenv("OKX_SECRET"),
        "password": os.getenv("OKX_PASSPHRASE"),
        "enableRateLimit": True, "options": {"defaultType": "swap"},
    })
    await ex.load_markets()

    rows = []
    for tf in TF_DAYS:
        raw = {}
        for s in SYMBOLS:
            df = await get_df(ex, s, tf)
            if df is not None:
                raw[s] = df
        arrays_cache = {}
        for name, cfg in CONFIGS:
            key = id(cfg["bands"])
            if key not in arrays_cache:
                arrays_cache[key] = {s: build_arrays(df, cfg["bands"]) for s, df in raw.items()}
            data = arrays_cache[key]
            for direction in DIRECTIONS:
                for lev in LEVERAGES:
                    trades, equity, max_dd = simulate(data, direction, lev, cfg["partial"])
                    st = stats(trades, equity, max_dd)
                    rows.append(dict(tf=tf, config=name, dir=direction, lev=lev, **st))
                    print(f"{tf} {name:<8} {direction:<6} {lev}x  n={st['n']:>3}  WR={st['wr']:>5.1f}%  "
                          f"PF={st['pf']:>5.2f}  Ret={st['ret']:>+7.1f}%  DD={st['dd']:>5.1f}%", flush=True)
    await ex.close()

    df = pd.DataFrame(rows)
    out = os.path.join(CACHE_DIR, "grid_dir_lev_results.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    asyncio.run(main())
