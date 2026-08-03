#!/usr/bin/env python3
import asyncio, os, sys
import pandas as pd, numpy as np
import ccxt.async_support as ccxt_async

def calc_supertrend(df, period=10, multiplier=3.0):
    hl2 = (df['h'] + df['l']) / 2
    atr = (df['h'].combine(df['c'].shift(), max) - df['l'].combine(df['c'].shift(), min)).rolling(period).mean()
    fu = hl2 + (multiplier * atr); fl = hl2 - (multiplier * atr)
    sd = pd.Series(1, index=df.index, dtype='int'); sv = pd.Series(0.0, index=df.index, dtype='float64')
    for i in range(period, len(df)):
        if df['c'].iloc[i] > fu.iloc[i-1]: sd.iloc[i] = 1
        elif df['c'].iloc[i] < fl.iloc[i-1]: sd.iloc[i] = -1
        else:
            sd.iloc[i] = sd.iloc[i-1]
            if sd.iloc[i] == 1 and fl.iloc[i] < fl.iloc[i-1]: fl.iloc[i] = fl.iloc[i-1]
            if sd.iloc[i] == -1 and fu.iloc[i] > fu.iloc[i-1]: fu.iloc[i] = fu.iloc[i-1]
        sv.iloc[i] = fl.iloc[i] if sd.iloc[i] == 1 else fu.iloc[i]
    return sd, sv

def calc_stoch_rsi(series, period=14, sk=3, sd=3):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    sr = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    k = sr.rolling(sk).mean() * 100; d = k.rolling(sd).mean()
    return k, d

def simulate(df, mode, tight_mult=2.5, profit_thr=1.02, vol_mult=1.0, min_hold_candles=3, max_dca=3):
    trades = []
    long_pos = None; short_pos = None
    df['st_d_t'], df['st_v_t'] = calc_supertrend(df, 10, tight_mult)

    for i in range(200, len(df)):
        prev = df.iloc[i-1]; curr = df.iloc[i]
        vol_cond = curr['v'] > prev['vol_ma'] * vol_mult

        is_long_bo = prev['st_d_l'] == -1 and curr['st_d_l'] == 1
        is_short_bo = prev['st_d_l'] == 1 and curr['st_d_l'] == -1

        is_long_pb = curr['st_d_l'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
        is_short_pb = curr['st_d_l'] == -1 and prev['stoch_k'] > 80 and curr['stoch_k'] <= 80

        is_long_sig = False
        is_short_sig = False

        if mode == 'yesterday':
            is_long_sig = (is_long_bo or is_long_pb) and vol_cond
            is_short_sig = (is_short_bo or is_short_pb) and vol_cond
        elif mode == 'strict_cross':
            is_ema_cross_up = prev['c'] <= prev['ema200'] and curr['c'] > curr['ema200']
            is_ema_cross_down = prev['c'] >= prev['ema200'] and curr['c'] < curr['ema200']
            ls = 60 if is_ema_cross_up else 0
            if is_long_bo: ls += 40
            if is_long_pb: ls += 40
            ss = 60 if is_ema_cross_down else 0
            if is_short_bo: ss += 40
            if is_short_pb: ss += 40
            is_long_sig = (ls >= 100) and vol_cond
            is_short_sig = (ss >= 100) and vol_cond
        elif mode == 'trend_60':
            is_ema_up = curr['c'] > curr['ema200']
            is_ema_down = curr['c'] < curr['ema200']
            ls = 60 if is_ema_up else 0
            if is_long_bo: ls += 40
            if is_long_pb: ls += 40
            ss = 60 if is_ema_down else 0
            if is_short_bo: ss += 40
            if is_short_pb: ss += 40
            is_long_sig = (ls >= 100) and vol_cond
            is_short_sig = (ss >= 100) and vol_cond
        elif mode == 'trend_40':
            # EMA 40, BO 40, PB 40. Threshold 80.
            is_ema_up = curr['c'] > curr['ema200']
            is_ema_down = curr['c'] < curr['ema200']
            ls = 40 if is_ema_up else 0
            if is_long_bo: ls += 40
            if is_long_pb: ls += 40
            ss = 40 if is_ema_down else 0
            if is_short_bo: ss += 40
            if is_short_pb: ss += 40
            is_long_sig = (ls >= 80) and vol_cond
            is_short_sig = (ss >= 80) and vol_cond

        # ── Long ──
        if long_pos:
            ep = long_pos['entry']
            is_pft = curr['c'] > ep * profit_thr
            stv = curr['st_v_t'] if is_pft else curr['st_v_l']
            std = curr['st_d_t'] if is_pft else curr['st_d_l']
            close_l = std == -1 or curr['c'] < stv
            force_l = False
            if long_pos['exit_count'] > 0 and curr['c'] < ep:
                if (i - long_pos.get('first_i', i)) >= min_hold_candles: force_l = True

            if force_l:
                trades.append({'side': 'long', 'pnl_pct': (curr['c'] - ep) / ep * long_pos['size'] * 10.0, 'type': 'be_stop'})
                long_pos = None
            elif close_l and long_pos['exit_count'] < max_dca:
                rem = max_dca - long_pos['exit_count']
                sf = long_pos['size'] / rem if rem > 0 else long_pos['size']
                trades.append({'side': 'long', 'pnl_pct': (curr['c'] - ep) / ep * sf * 10.0, 'type': 'dca_exit'})
                long_pos['size'] -= sf; long_pos['exit_count'] += 1
                if long_pos['exit_count'] >= max_dca or long_pos['size'] < 0.001: long_pos = None
            elif not close_l and long_pos['entry_count'] < max_dca:
                long_pos['entry_count'] += 1
                add = 1.0 / max_dca
                long_pos['entry'] = (ep * long_pos['size'] + curr['c'] * add) / (long_pos['size'] + add)
                long_pos['size'] += add

        # ── Short ──
        if short_pos:
            ep = short_pos['entry']
            is_pft = curr['c'] < ep * (2.0 - profit_thr)
            stv = curr['st_v_t'] if is_pft else curr['st_v_l']
            std = curr['st_d_t'] if is_pft else curr['st_d_l']
            close_s = std == 1 or curr['c'] > stv
            force_s = False
            if short_pos['exit_count'] > 0 and curr['c'] > ep:
                if (i - short_pos.get('first_i', i)) >= min_hold_candles: force_s = True

            if force_s:
                trades.append({'side': 'short', 'pnl_pct': (ep - curr['c']) / ep * short_pos['size'] * 10.0, 'type': 'be_stop'})
                short_pos = None
            elif close_s and short_pos['exit_count'] < max_dca:
                rem = max_dca - short_pos['exit_count']
                sf = short_pos['size'] / rem if rem > 0 else short_pos['size']
                trades.append({'side': 'short', 'pnl_pct': (ep - curr['c']) / ep * sf * 10.0, 'type': 'dca_exit'})
                short_pos['size'] -= sf; short_pos['exit_count'] += 1
                if short_pos['exit_count'] >= max_dca or short_pos['size'] < 0.001: short_pos = None
            elif not close_s and short_pos['entry_count'] < max_dca:
                short_pos['entry_count'] += 1
                add = 1.0 / max_dca
                short_pos['entry'] = (ep * short_pos['size'] + curr['c'] * add) / (short_pos['size'] + add)
                short_pos['size'] += add

        active = (1 if long_pos else 0) + (1 if short_pos else 0)
        if active >= 3: continue
        
        alloc_mult = 1.0 if mode == 'yesterday' else 1.25
            
        if not long_pos and is_long_sig:
            long_pos = {'entry': curr['c'], 'size': alloc_mult/max_dca, 'entry_count': 1, 'exit_count': 0, 'first_i': i}
        if not short_pos and is_short_sig:
            short_pos = {'entry': curr['c'], 'size': alloc_mult/max_dca, 'entry_count': 1, 'exit_count': 0, 'first_i': i}

    lc = df.iloc[-1]['c']
    if long_pos: trades.append({'side': 'long', 'pnl_pct': (lc - long_pos['entry'])/long_pos['entry']*long_pos['size'] * 10.0, 'type': 'eod'})
    if short_pos: trades.append({'side': 'short', 'pnl_pct': (short_pos['entry'] - lc)/short_pos['entry']*short_pos['size'] * 10.0, 'type': 'eod'})
    return trades

def stats(trades, bh_ret):
    if not trades: return {'ret': 0, 'wr': 0, 'n': 0}
    pnls = [t['pnl_pct'] for t in trades]
    ret = sum(pnls) * 100
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / len(pnls) * 100
    return {'ret': round(ret, 3), 'wr': round(wr, 1), 'n': len(trades)}

async def main():
    print("=" * 110)
    print(" 📊 4 Bots Backtest Report (EMA Loosening Comparison)")
    print("=" * 110)

    configs = [
        {"bot": "Major Crypto", "tf": "1h", "syms": ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]},
        {"bot": "Stock Majors", "tf": "1h", "syms": ["TSLA/USDT:USDT", "AAPL/USDT:USDT", "NVDA/USDT:USDT"]},
        {"bot": "Venture Alts", "tf": "15m", "syms": ["DOGE/USDT:USDT", "PEPE/USDT:USDT", "WIF/USDT:USDT"]},
        {"bot": "Stock Ventures", "tf": "15m", "syms": ["CRM/USDT:USDT", "OKTA/USDT:USDT", "TMF/USDT:USDT"]},
    ]

    ex = ccxt_async.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    try:
        for cfg in configs:
            print(f"\n[{cfg['bot']}] - TF: {cfg['tf']}")
            print(f"{'Symbol':<15} │ {'Yesterday':<20} │ {'Strict (Cross)':<20} │ {'Trend (60pt)':<20} │ {'Trend (40pt)':<20}")
            print("-" * 110)
            for sym in cfg['syms']:
                try:
                    ohlcv = await ex.fetch_ohlcv(sym, cfg['tf'], limit=1500)
                    if not ohlcv or len(ohlcv) < 200:
                        continue
                    df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
                    df['st_d_l'], df['st_v_l'] = calc_supertrend(df, 10, 4.0)
                    df['vol_ma'] = df['v'].rolling(20).mean()
                    k, d = calc_stoch_rsi(df['c'], 14, 3, 3)
                    df['stoch_k'] = k; df['stoch_d'] = d
                    df['ema200'] = df['c'].ewm(span=200, adjust=False).mean()

                    bh = 0
                    tr_yest = simulate(df.copy(), mode='yesterday')
                    tr_strict = simulate(df.copy(), mode='strict_cross')
                    tr_t60 = simulate(df.copy(), mode='trend_60')
                    tr_t40 = simulate(df.copy(), mode='trend_40')

                    sy = stats(tr_yest, bh); ss = stats(tr_strict, bh)
                    st60 = stats(tr_t60, bh); st40 = stats(tr_t40, bh)
                    
                    fs = lambda x: f"{x['ret']:>5.1f}%({x['n']:>2d})"
                    print(f"{sym:<15} │ {fs(sy):<20} │ {fs(ss):<20} │ {fs(st60):<20} │ {fs(st40):<20}")
                except Exception as e:
                    print(f"{sym:<15} │ ❌ Error: {str(e)}")
            await asyncio.sleep(0.5)
    finally:
        await ex.close()

if __name__ == "__main__":
    asyncio.run(main())
