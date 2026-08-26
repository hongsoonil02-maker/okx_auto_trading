#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fetch_lg_data.py — 백테스트용 OHLCV 수집 → /tmp/lg_data.pkl 저장 (1회 실행)"""
import asyncio, pickle, os
from datetime import datetime, timezone
import pandas as pd
import ccxt.async_support as ccxt_async

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
    "BNB/USDT:USDT", "DOGE/USDT:USDT", "ZEC/USDT:USDT", "CRV/USDT:USDT",
    "NEAR/USDT:USDT", "LTC/USDT:USDT", "HYPE/USDT:USDT", "AAVE/USDT:USDT",
    "UNI/USDT:USDT", "TAO/USDT:USDT", "PEPE/USDT:USDT", "LIT/USDT:USDT",
]
OUT = "/tmp/lg_data.pkl"


def calc_supertrend(df, period=10, multiplier=3.0):
    hl2 = (df['h'] + df['l']) / 2
    atr = (df['h'].combine(df['c'].shift(), max) - df['l'].combine(df['c'].shift(), min)).rolling(period).mean()
    fu = hl2 + multiplier * atr
    fl = hl2 - multiplier * atr
    sd = pd.Series(1, index=df.index, dtype='int')
    for i in range(period, len(df)):
        if df['c'].iloc[i] > fu.iloc[i - 1]:
            sd.iloc[i] = 1
        elif df['c'].iloc[i] < fl.iloc[i - 1]:
            sd.iloc[i] = -1
        else:
            sd.iloc[i] = sd.iloc[i - 1]
    return sd


def calc_stoch_k(series, period=14, smooth=3):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    stoch = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    return stoch.rolling(smooth).mean() * 100


def calc_adx(df, period=14):
    h, l, c = df['h'], df['l'], df['c']
    up, dn = h.diff(), -l.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).where((pdi + mdi) != 0)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)


async def fetch_all(ex, sym, tf, since_ms):
    out = []
    since = since_ms
    for _ in range(40):
        batch = await ex.fetch_ohlcv(sym, tf, since=since, limit=300)
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 300:
            break
        since = batch[-1][0] + 1
        await asyncio.sleep(0.04)
    df = pd.DataFrame(out, columns=['t', 'o', 'h', 'l', 'c', 'v']).drop_duplicates('t')
    return df.sort_values('t').reset_index(drop=True)


async def main():
    ex = ccxt_async.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    since = int(datetime(2026, 6, 20, tzinfo=timezone.utc).timestamp() * 1000)
    try:
        print("BTC 1h 수집...")
        btc = await fetch_all(ex, "BTC/USDT:USDT", "1h", since - 400 * 3600 * 1000)
        btc['adx'] = calc_adx(btc, 14)
        btc['ema50'] = btc['c'].ewm(span=50, adjust=False).mean()
        btc_map = {}
        for i in range(1, len(btc) - 1):
            btc_map[int(btc['t'].iloc[i])] = (
                float(btc['adx'].iloc[i]),
                bool(btc['c'].iloc[i] > btc['ema50'].iloc[i]),
            )

        data = {}
        sym_map = {}
        for sym in SYMBOLS:
            try:
                h1 = await fetch_all(ex, sym, "1h", since - 400 * 3600 * 1000)
                h1['ema50'] = h1['c'].ewm(span=50, adjust=False).mean()
                sm = {}
                for i in range(1, len(h1) - 1):
                    sm[int(h1['t'].iloc[i])] = bool(h1['c'].iloc[i] > h1['ema50'].iloc[i])
                sym_map[sym] = sm

                df = await fetch_all(ex, sym, "15m", since)
                if len(df) < 3000:
                    print(f"  {sym}: 부족({len(df)}) 제외")
                    continue
                df['st_dir'] = calc_supertrend(df)
                df['stoch_k'] = calc_stoch_k(df['c'])
                df = df.dropna().reset_index(drop=True)
                data[sym] = df
                print(f"  {sym}: {len(df)}캔들")
            except Exception as e:
                print(f"  {sym}: 실패 {str(e)[:60]}")

        with open(OUT, "wb") as f:
            pickle.dump((btc_map, data, sym_map), f)
        print(f"저장 완료: {OUT} ({len(data)}심볼)")
    finally:
        await ex.close()


if __name__ == "__main__":
    asyncio.run(main())
