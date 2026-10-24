# -*- coding: utf-8 -*-
"""
backtest_tf_compare.py — Multi-timeframe comparison of the OKX Cameron pullback strategy
For each timeframe, runs 5 configs:
  A: original bands + BE trailing        (exact live-bot replica)
  B: original bands, no trailing
  C: original bands, no trailing + trend filter (close>EMA20 & BTC>EMA50)
  D: sqrt-time scaled bands, no trailing (bands * sqrt(tf/4h))
  E: scaled bands, no trailing + trend filter
Data cached to CSV under state/bt_cache/.
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
os.makedirs(CACHE_DIR, exist_ok=True)

SYMBOLS = [
    "SOL/USDT:USDT", "SUI/USDT:USDT", "DOGE/USDT:USDT", "ETH/USDT:USDT", "BTC/USDT:USDT",
    "XRP/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT", "NEAR/USDT:USDT",
    "APT/USDT:USDT", "PEPE/USDT:USDT", "WIF/USDT:USDT", "SHIB/USDT:USDT", "FET/USDT:USDT",
]

TF_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "1h": 60, "2h": 120, "4h": 240}
TF_DAYS = {"1m": 14, "3m": 21, "5m": 30, "15m": 60, "1h": 90, "2h": 120, "4h": 120}

# base 4h-tuned params (live bot)
BASE = dict(min_surge=0.025, min_sl=0.015, max_sl=0.060)
RR = 1.86
ALLOC = 0.08
LEVERAGE = 3
MAX_POS = 3
FEE = 0.0005
SLIP = 0.0005
WINDOW = 35
INITIAL_CAPITAL = 960.0

CONFIGS = [
    ("A:orig+trail",  dict(scaled=False, trailing=True,  trend=False)),
    ("B:orig",        dict(scaled=False, trailing=False, trend=False)),
    ("C:orig+trend",  dict(scaled=False, trailing=False, trend=True)),
    ("D:scaled",      dict(scaled=True,  trailing=False, trend=False)),
    ("E:scaled+trend",dict(scaled=True,  trailing=False, trend=True)),
]


async def fetch_bars(ex, symbol, tf):
    cache = os.path.join(CACHE_DIR, f"{symbol.replace('/', '').replace(':', '_')}_{tf}_{TF_DAYS[tf]}d.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    end_ts = ex.milliseconds()
    since = end_ts - TF_DAYS[tf] * 24 * 3600 * 1000
    tf_ms = TF_MINUTES[tf] * 60 * 1000
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
    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().values
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().values

    prior_high = pd.Series(h).rolling(5).max().shift(1).values
    prior_low = pd.Series(l).rolling(5).min().shift(1).values
    prior_vol = pd.Series(v).rolling(5).mean().shift(1).values

    with np.errstate(divide="ignore", invalid="ignore"):
        surge = prior_high - prior_low
        surge_pct = surge / prior_low
        retrace = np.where(surge > 0, (prior_high - l) / surge, 1.0)
        risk_pct = np.where(h > 0, (h - l) / h, 0.0)

    mask = (
        (c < o)
        & (c >= ema9 * 0.995)
        & (l >= vwap * 0.990)
        & (surge_pct >= bands["min_surge"])
        & (retrace <= 0.55)
        & (v <= prior_vol * 1.20)
        & (risk_pct >= bands["min_sl"])
        & (risk_pct <= bands["max_sl"])
    )
    mask[:14] = False
    n = len(c)
    # entry at bar i+1 if its high breaks bar i's high
    entry_idx = np.where(mask[:-1] & (h[1:] >= h[:-1]))[0] + 1
    return dict(ts=ts, o=o, h=h, l=l, c=c, ema20=ema20, ema50=ema50,
                entry_idx=entry_idx, n=n)


def simulate(data, btc_regime, cfg):
    """data: symbol -> arrays dict. Returns list of trades."""
    equity = INITIAL_CAPITAL
    peak = equity
    max_dd = 0.0
    open_pos = {}
    trades = []

    entry_events = {}  # bar_index -> list of (symbol, setup_i)
    max_len = 0
    for sym, d in data.items():
        max_len = max(max_len, d["n"])
        for j in d["entry_idx"]:
            i = j - 1
            if cfg["trend"]:
                if not (d["c"][i] > d["ema20"][i]):
                    continue
                if not btc_regime.get(int(d["ts"][i]), False):
                    continue
            entry_events.setdefault(j, []).append(sym)

    for i in range(max_len):
        # manage
        for sym in list(open_pos.keys()):
            d = data[sym]
            if i >= d["n"]:
                continue
            p = open_pos[sym]
            hi, lo = d["h"][i], d["l"][i]
            if i == p["entry_i"]:
                if lo <= p["sl"]:
                    exit_p = p["sl"] * (1 - SLIP)
                    trades.append(close_trade(p, exit_p, d["ts"][i], "SL_SAME_BAR"))
                    equity += trades[-1]["pnl"]
                    peak = max(peak, equity)
                    max_dd = max(max_dd, (peak - equity) / peak)
                    del open_pos[sym]
                continue
            if cfg["trailing"] and p["sl"] < p["entry"] and hi >= p["entry"] + p["risk"]:
                p["sl"] = p["entry"] * 1.002
            hit_sl, hit_tp = lo <= p["sl"], hi >= p["tp"]
            exit_p = reason = None
            if hit_sl and hit_tp:
                exit_p, reason = p["sl"] * (1 - SLIP), "SL_AMBIG"
            elif hit_sl:
                exit_p, reason = p["sl"] * (1 - SLIP), "SL"
            elif hit_tp:
                exit_p, reason = p["tp"] * (1 - SLIP), "TP"
            if exit_p is not None:
                trades.append(close_trade(p, exit_p, d["ts"][i], reason))
                equity += trades[-1]["pnl"]
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / peak)
                del open_pos[sym]

        # entries at this bar
        if len(open_pos) < MAX_POS:
            for sym in entry_events.get(i, []):
                if len(open_pos) >= MAX_POS or sym in open_pos:
                    continue
                d = data[sym]
                i0 = i - 1
                trig, sl = d["h"][i0], d["l"][i0]
                entry = trig * (1 + SLIP)
                open_pos[sym] = dict(
                    sym=sym, entry=entry, sl=sl, tp=trig + (trig - sl) * RR,
                    risk=trig - sl, notional=equity * ALLOC * LEVERAGE,
                    entry_i=i, entry_ts=d["ts"][i],
                )
    return trades, equity, max_dd


def close_trade(p, exit_p, exit_ts, reason):
    ret = exit_p / p["entry"] - 1
    pnl = p["notional"] * ret - p["notional"] * FEE * 2
    return dict(symbol=p["sym"], reason=reason, ret=ret, pnl=pnl,
                hold_bars=0, entry_ts=p["entry_ts"], exit_ts=exit_ts)


def stats(trades, equity, max_dd):
    if not trades:
        return dict(n=0, wr=0.0, pf=0.0, ret=0.0, dd=0.0, avg=0.0)
    wins = [t for t in trades if t["pnl"] > 0.5]
    losses = [t for t in trades if t["pnl"] <= 0.5 and t["ret"] <= -0.001]
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    return dict(
        n=len(trades),
        wr=len(wins) / len(trades) * 100,
        pf=gw / gl if gl > 0 else float("inf"),
        ret=(equity / INITIAL_CAPITAL - 1) * 100,
        dd=max_dd * 100,
        avg=sum(t["pnl"] for t in trades) / len(trades),
    )


async def main():
    ex = ccxt_async.okx({
        "apiKey": os.getenv("OKX_API_KEY"),
        "secret": os.getenv("OKX_SECRET"),
        "password": os.getenv("OKX_PASSPHRASE"),
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
    await ex.load_markets()

    rows = []
    for tf in TF_MINUTES:
        days = TF_DAYS[tf]
        scale = (TF_MINUTES[tf] / 240) ** 0.5
        print(f"\n=== {tf} ({days}d, scale {scale:.3f}) — fetching... ===", flush=True)
        raw = {}
        for s in SYMBOLS:
            try:
                df = await fetch_bars(ex, s, tf)
                if df is not None:
                    raw[s] = df
            except Exception as e:
                print(f"  {s}: fetch failed: {e}")
        if "BTC/USDT:USDT" not in raw:
            print(f"  BTC missing, skipping {tf}")
            continue

        for name, cfg in CONFIGS:
            bands = dict(BASE) if not cfg["scaled"] else {
                k: v * scale for k, v in BASE.items()
            }
            data = {s: build_arrays(df, bands) for s, df in raw.items()}
            btc = data["BTC/USDT:USDT"]
            btc_regime = {int(t): bool(c > e) for t, c, e in zip(btc["ts"], btc["c"], btc["ema50"])}
            trades, equity, max_dd = simulate(data, btc_regime, cfg)
            st = stats(trades, equity, max_dd)
            rows.append(dict(tf=tf, days=days, config=name, **st))
            print(f"  {name:<16} n={st['n']:>3}  WR={st['wr']:>5.1f}%  PF={st['pf']:>5.2f}  "
                  f"Ret={st['ret']:>+6.1f}%  DD={st['dd']:>5.1f}%  Avg=${st['avg']:>+6.2f}", flush=True)

    await ex.close()
    print("\n" + "=" * 86)
    print(f"{'TF':<5}{'Days':>5}{'Config':<17}{'Trades':>7}{'WR%':>7}{'PF':>7}{'Return%':>9}{'MaxDD%':>8}{'Avg$':>8}")
    print("-" * 86)
    for r in rows:
        print(f"{r['tf']:<5}{r['days']:>5}{r['config']:<17}{r['n']:>7}{r['wr']:>6.1f}%{r['pf']:>7.2f}"
              f"{r['ret']:>+8.1f}%{r['dd']:>7.1f}%{r['avg']:>+8.2f}")
    pd.DataFrame(rows).to_csv(os.path.join(CACHE_DIR, "tf_compare_results.csv"), index=False)


if __name__ == "__main__":
    asyncio.run(main())
