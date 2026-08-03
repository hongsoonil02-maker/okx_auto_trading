#!/usr/bin/env python3
import asyncio, pandas as pd, numpy as np
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

def simulate(df, ema_period, tight_mult=2.5, loose_mult=4.0, vol_mult=1.2, min_hold=3, max_dca=3, tp_thr=1.01):
    trades = []
    long_pos = None; short_pos = None
    df['st_d_t'], df['st_v_t'] = calc_supertrend(df, 10, tight_mult)
    df['st_d_l'], df['st_v_l'] = calc_supertrend(df, 10, loose_mult)
    df['ema_target'] = df['c'].ewm(span=ema_period, adjust=False).mean()

    for i in range(250, len(df)):
        prev = df.iloc[i-1]; curr = df.iloc[i]
        vol_cond = curr['v'] > prev['vol_ma'] * vol_mult

        is_long_bo = prev['st_d_l'] == -1 and curr['st_d_l'] == 1
        is_short_bo = prev['st_d_l'] == 1 and curr['st_d_l'] == -1

        is_long_pb = curr['st_d_l'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
        is_short_pb = curr['st_d_l'] == -1 and prev['stoch_k'] > 80 and curr['stoch_k'] <= 80

        ls = 60 if curr['c'] > curr['ema_target'] else 0
        if is_long_bo: ls += 40
        if is_long_pb: ls += 40
        
        ss = 60 if curr['c'] < curr['ema_target'] else 0
        if is_short_bo: ss += 40
        if is_short_pb: ss += 40
        
        is_long_sig = (ls >= 100) and vol_cond
        is_short_sig = (ss >= 100) and vol_cond

        # ── Long ──
        if long_pos:
            ep = long_pos['entry']
            is_pft = curr['c'] > ep * tp_thr
            stv = curr['st_v_t'] if is_pft else curr['st_v_l']
            std = curr['st_d_t'] if is_pft else curr['st_d_l']
            close_l = std == -1 or curr['c'] < stv
            force_l = False
            if long_pos['exit_count'] > 0 and curr['c'] < ep:
                if (i - long_pos.get('first_i', i)) >= min_hold: force_l = True

            if force_l:
                trades.append({'pnl': (curr['c'] - ep) / ep * long_pos['size'] * 10.0})
                long_pos = None
            elif close_l and long_pos['exit_count'] < max_dca:
                rem = max_dca - long_pos['exit_count']
                sf = long_pos['size'] / rem if rem > 0 else long_pos['size']
                trades.append({'pnl': (curr['c'] - ep) / ep * sf * 10.0})
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
            is_pft = curr['c'] < ep * (2.0 - tp_thr)
            stv = curr['st_v_t'] if is_pft else curr['st_v_l']
            std = curr['st_d_t'] if is_pft else curr['st_d_l']
            close_s = std == 1 or curr['c'] > stv
            force_s = False
            if short_pos['exit_count'] > 0 and curr['c'] > ep:
                if (i - short_pos.get('first_i', i)) >= min_hold: force_s = True

            if force_s:
                trades.append({'pnl': (ep - curr['c']) / ep * short_pos['size'] * 10.0})
                short_pos = None
            elif close_s and short_pos['exit_count'] < max_dca:
                rem = max_dca - short_pos['exit_count']
                sf = short_pos['size'] / rem if rem > 0 else short_pos['size']
                trades.append({'pnl': (ep - curr['c']) / ep * sf * 10.0})
                short_pos['size'] -= sf; short_pos['exit_count'] += 1
                if short_pos['exit_count'] >= max_dca or short_pos['size'] < 0.001: short_pos = None
            elif not close_s and short_pos['entry_count'] < max_dca:
                short_pos['entry_count'] += 1
                add = 1.0 / max_dca
                short_pos['entry'] = (ep * short_pos['size'] + curr['c'] * add) / (short_pos['size'] + add)
                short_pos['size'] += add

        active = (1 if long_pos else 0) + (1 if short_pos else 0)
        if active >= 3: continue
            
        if not long_pos and is_long_sig:
            long_pos = {'entry': curr['c'], 'size': 1.0/max_dca, 'entry_count': 1, 'exit_count': 0, 'first_i': i}
        if not short_pos and is_short_sig:
            short_pos = {'entry': curr['c'], 'size': 1.0/max_dca, 'entry_count': 1, 'exit_count': 0, 'first_i': i}

    lc = df.iloc[-1]['c']
    if long_pos: trades.append({'pnl': (lc - long_pos['entry'])/long_pos['entry']*long_pos['size'] * 10.0})
    if short_pos: trades.append({'pnl': (short_pos['entry'] - lc)/short_pos['entry']*short_pos['size'] * 10.0})
    return trades

def stats(trades):
    if not trades: return {'ret': 0, 'n': 0}
    pnls = [t['pnl'] for t in trades]
    ret = sum(pnls) * 100
    return {'ret': round(ret, 2), 'n': len(trades)}

async def main():
    ex = ccxt_async.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    
    print("="*80)
    print(" 🚀 DCA Entries Comparison Backtest (3 vs 5 vs 8)")
    print("="*80)
    
    print("\n[Major Crypto] - 1h TF, EMA 50, TP 1.01")
    print(f"{'Symbol':<15} | {'DCA 3':<20} | {'DCA 5':<20} | {'DCA 8':<20}")
    crypto_syms = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    for sym in crypto_syms:
        ohlcv = await ex.fetch_ohlcv(sym, '1h', limit=2000)
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
        df['vol_ma'] = df['v'].rolling(20).mean()
        delta = df['c'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan); rsi = 100 - (100 / (1 + rs))
        df['stoch_k'] = ((rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())).rolling(3).mean() * 100

        tr_3 = simulate(df.copy(), ema_period=50, tp_thr=1.01, max_dca=3)
        tr_5 = simulate(df.copy(), ema_period=50, tp_thr=1.01, max_dca=5)
        tr_8 = simulate(df.copy(), ema_period=50, tp_thr=1.01, max_dca=8)
        
        s_3 = stats(tr_3); s_5 = stats(tr_5); s_8 = stats(tr_8)
        f = lambda s: f"{s['ret']:>6.2f}% ({s['n']:>2d})"
        print(f"{sym:<15} | {f(s_3):<20} | {f(s_5):<20} | {f(s_8):<20}")
        await asyncio.sleep(0.5)
        
    print("\n[Stock Majors] - 15m TF, EMA 200, TP 1.01")
    print(f"{'Symbol':<15} | {'DCA 3':<20} | {'DCA 5':<20} | {'DCA 8':<20}")
    stock_syms = ["TSLA/USDT:USDT", "NVDA/USDT:USDT"]
    for sym in stock_syms:
        ohlcv = await ex.fetch_ohlcv(sym, '15m', limit=2000)
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
        df['vol_ma'] = df['v'].rolling(20).mean()
        delta = df['c'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan); rsi = 100 - (100 / (1 + rs))
        df['stoch_k'] = ((rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())).rolling(3).mean() * 100

        tr_3 = simulate(df.copy(), ema_period=200, tp_thr=1.01, max_dca=3)
        tr_5 = simulate(df.copy(), ema_period=200, tp_thr=1.01, max_dca=5)
        tr_8 = simulate(df.copy(), ema_period=200, tp_thr=1.01, max_dca=8)
        
        s_3 = stats(tr_3); s_5 = stats(tr_5); s_8 = stats(tr_8)
        f = lambda s: f"{s['ret']:>6.2f}% ({s['n']:>2d})"
        print(f"{sym:<15} | {f(s_3):<20} | {f(s_5):<20} | {f(s_8):<20}")
        await asyncio.sleep(0.5)
        
    await ex.close()

if __name__ == "__main__":
    asyncio.run(main())
