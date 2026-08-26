import asyncio
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt_async

FEE_RATE = 0.0005
MAX_POS = 15
WEIGHT = 1.5
NEW_RATIO = 0.50
INITIAL_EQUITY = 19000.0
CHOP_START = datetime(2026, 8, 21, 14, 0, tzinfo=timezone.utc).timestamp() * 1000
TRAILING_PCT = 0.06

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "LINK/USDT:USDT", "BNB/USDT:USDT", # Majors
    "ZRO/USDT:USDT", "CAP/USDT:USDT", "VVV/USDT:USDT", "LTC/USDT:USDT",
    "DOGE/USDT:USDT", "AXTI/USDT:USDT", "CHIP/USDT:USDT", "CRCL/USDT:USDT",
    "AAVE/USDT:USDT", "LIT/USDT:USDT", "FIL/USDT:USDT", "HYPE/USDT:USDT",
    "ZEC/USDT:USDT", "TRB/USDT:USDT", "BOME/USDT:USDT", "ENA/USDT:USDT", "CRV/USDT:USDT",
]

def calc_supertrend(df, period=10, multiplier=3.0):
    hl2 = (df['h'] + df['l']) / 2
    atr = (df['h'].combine(df['c'].shift(), max) - df['l'].combine(df['c'].shift(), min)).rolling(period).mean()
    fu = hl2 + multiplier * atr
    fl = hl2 - multiplier * atr
    sd = pd.Series(1, index=df.index, dtype='int')
    sv = pd.Series(0.0, index=df.index, dtype='float64')
    for i in range(period, len(df)):
        if df['c'].iloc[i] > fu.iloc[i - 1]:
            sd.iloc[i] = 1
        elif df['c'].iloc[i] < fl.iloc[i - 1]:
            sd.iloc[i] = -1
        else:
            sd.iloc[i] = sd.iloc[i - 1]
            if sd.iloc[i] == 1 and fl.iloc[i] < fl.iloc[i - 1]:
                fl.iloc[i] = fl.iloc[i - 1]
            if sd.iloc[i] == -1 and fu.iloc[i] > fu.iloc[i - 1]:
                fu.iloc[i] = fu.iloc[i - 1]
        sv.iloc[i] = fl.iloc[i] if sd.iloc[i] == 1 else fu.iloc[i]
    return sd, sv

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
    for _ in range(20):
        batch = await ex.fetch_ohlcv(sym, tf, since=since, limit=300)
        if not batch: break
        out.extend(batch)
        if len(batch) < 300: break
        since = batch[-1][0] + 1
        await asyncio.sleep(0.05)
    df = pd.DataFrame(out, columns=['t', 'o', 'h', 'l', 'c', 'v']).drop_duplicates('t')
    return df.sort_values('t').reset_index(drop=True)

def simulate(data, btc_map, flags, lev_major=10, lev_venture=10):
    cash = INITIAL_EQUITY
    positions = {}
    trades, fees = [], 0.0
    all_ts = sorted(set().union(*[set(df['t']) for df in data.values()]))
    warmup = 60
    
    majors = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'AVAX', 'LINK', 'DOT', 'BNB', 'TRX']

    def get_lev(sym):
        base = sym.split('/')[0]
        return lev_major if base in majors else lev_venture

    for idx, t in enumerate(all_ts):
        if idx < warmup: continue
        hour_floor = (t // 3600000) * 3600000
        last_h = hour_floor - 3600000
        adx_now, btc_bull, above200 = btc_map.get(last_h, (30.0, True, True))

        for sym, df in data.items():
            lev = get_lev(sym)
            rows = df.index[df['t'] == t]
            if len(rows) == 0: continue
            i = rows[0]
            if i < 1: continue
            curr, prev = df.iloc[i], df.iloc[i - 1]
            px = float(curr['c'])

            pos = positions.get(sym)
            if pos:
                pos['last_px'] = px
                pnl_pct = (px - pos['entry']) / pos['entry'] * lev * pos['dir']
                pos['extreme'] = max(pos['extreme'], pnl_pct) if pos['dir'] == 1 else min(pos['extreme'], pnl_pct)
                pos['highest'] = max(pos['highest'], px) if pos['dir'] == 1 else min(pos['highest'], px)
                best = pos['extreme']

                exit_now = pnl_pct <= -0.15  # 하드스탑 15%
                exit_now = exit_now or (best > 0.4 and pnl_pct < best * 0.5)
                exit_now = exit_now or (best > 0.2 and pnl_pct < 0.05)
                exit_now = exit_now or pnl_pct >= 0.50

                if flags['trailing']:
                    if pos['dir'] == 1 and px < pos['highest'] * (1 - TRAILING_PCT): exit_now = True
                    elif pos['dir'] == -1 and px > pos['highest'] * (1 + TRAILING_PCT): exit_now = True

                if exit_now:
                    gross = pos['margin'] * pnl_pct
                    fee = pos['margin'] * lev * FEE_RATE
                    cash += pos['margin'] + gross - fee
                    fees += fee
                    trades.append({'t': t, 'pnl': gross - fee, 'dir': pos['dir'], 'sym': sym})
                    del positions[sym]
                continue

            long_sig = curr['st_dir'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
            short_sig = curr['st_dir'] == -1 and prev['stoch_k'] > 80 and curr['stoch_k'] <= 80

            if flags['bear_short'] and above200 and short_sig: short_sig = False
            if not (long_sig or short_sig): continue

            score = 70
            if long_sig: score = 50 + 40 + 30 + (20 if curr['st_dir'] == prev['st_dir'] else 0) + (20 if curr['stoch_k'] > curr['stoch_d'] else 0)
            d = 1 if long_sig else -1

            equity = sum(p['margin'] * (1 + (p['last_px'] - p['entry']) / p['entry'] * get_lev(s) * p['dir']) for s, p in positions.items()) + cash
            margin = (equity / MAX_POS) * WEIGHT * NEW_RATIO

            if flags['conviction']:
                conv_mult = min(2.0, max(0.5, score / 70.0))
                margin *= conv_mult

            if margin < 100 or margin > cash: continue
            fee = margin * lev * FEE_RATE
            cash -= margin + fee
            fees += fee
            positions[sym] = {'entry': px, 'margin': margin, 'dir': d, 'extreme': 0.0, 'highest': px, 'last_px': px}

    equity = cash
    for sym, pos in positions.items():
        lev = get_lev(sym)
        pnl_pct = (pos['last_px'] - pos['entry']) / pos['entry'] * lev * pos['dir']
        fee = pos['margin'] * lev * FEE_RATE
        equity += pos['margin'] * (1 + pnl_pct) - fee
        fees += fee
        trades.append({'pnl': pos['margin'] * pnl_pct - fee, 'dir': pos['dir'], 'sym': sym})

    pnl_total = equity - INITIAL_EQUITY
    pnl_chop = sum(tr['pnl'] for tr in trades if 't' in tr and tr['t'] >= CHOP_START)
    return {'total': pnl_total, 'chop': pnl_chop, 'n': len(trades), 'equity': equity}

async def main():
    ex = ccxt_async.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    since = int(datetime(2026, 8, 18, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    try:
        btc = await fetch_all(ex, "BTC/USDT:USDT", "1h", since - 260 * 3600 * 1000)
        btc['adx'] = calc_adx(btc, 14)
        btc['ema200'] = btc['c'].ewm(span=200, adjust=False).mean()
        btc['ema50'] = btc['c'].ewm(span=50, adjust=False).mean()
        btc_map = {}
        for i in range(1, len(btc) - 1):
            r = btc.iloc[i]
            above200 = bool(r['c'] >= r['ema200'])
            ema50_up = bool(btc['ema50'].iloc[i] > btc['ema50'].iloc[i - 4])
            btc_map[int(r['t'])] = (float(r['adx']), above200 and ema50_up, above200)

        data = {}
        for sym in SYMBOLS:
            try:
                df = await fetch_all(ex, sym, "15m", since)
                if len(df) >= 200:
                    df['st_dir'], _ = calc_supertrend(df)
                    df['stoch_k'] = calc_stoch_k(df['c'])
                    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
                    df = df.dropna().reset_index(drop=True)
                    data[sym] = df
            except Exception: pass
        
        F = {'conviction': True, 'trailing': True, 'adx_pyramid': False, 'bear_short': True}
        print(f"{'조합 (메이저/벤처)':20s} {'전체수익':>10s} {'피크이후':>10s} {'거래수':>6s}")
        
        combos = [
            ("단일 10x / 10x", 10, 10),
            ("단일 5x / 5x", 5, 5),
            ("듀얼 10x / 5x", 10, 5),
            ("듀얼 15x / 5x", 15, 5)
        ]
        
        for name, lev_maj, lev_ven in combos:
            r = simulate(data, btc_map, F, lev_maj, lev_ven)
            print(f"  {name:18s} {r['total']:+10.2f} {r['chop']:+10.2f} {r['n']:6d}")
    finally:
        await ex.close()

if __name__ == "__main__":
    asyncio.run(main())
