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

def calc_stoch_rsi(series, period=14, sk=3, sd=3):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    sr = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    k = sr.rolling(sk).mean() * 100; d = k.rolling(sd).mean()
    return k, d

def sim_yesterday(df, tight_mult, profit_thr, vol_mult):
    trades = []
    long_pos = None; short_pos = None
    df['st_d_t'], df['st_v_t'] = calc_supertrend(df, 10, tight_mult)

    for i in range(20, len(df)):
        prev, curr = df.iloc[i-1], df.iloc[i]
        vol_cond = curr['v'] > prev['vol_ma'] * vol_mult

        is_l_bo = prev['st_d_l'] == -1 and curr['st_d_l'] == 1 and vol_cond
        is_l_pb = curr['st_d_l'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
        is_s_bo = prev['st_d_l'] == 1 and curr['st_d_l'] == -1 and vol_cond
        is_s_pb = curr['st_d_l'] == -1 and prev['stoch_k'] > 80 and curr['stoch_k'] <= 80
        
        # Long
        if long_pos:
            ep = long_pos['entry']
            long_pos['max_p'] = max(long_pos['max_p'], curr['c'])
            is_pft = curr['c'] > ep * profit_thr
            std = curr['st_d_t'] if is_pft else curr['st_d_l']
            stv = curr['st_v_t'] if is_pft else curr['st_v_l']
            
            close_l = std == -1 or curr['c'] < stv
            if long_pos['max_p'] > ep * 1.02 and curr['c'] < ep:
                close_l = True # Yesterday's hardcoded breakeven
            if close_l:
                trades.append({'side': 'long', 'pnl_pct': (curr['c'] - ep)/ep, 'type': 'exit'})
                long_pos = None
                
        # Short
        if short_pos:
            ep = short_pos['entry']
            short_pos['min_p'] = min(short_pos['min_p'], curr['c'])
            is_pft = curr['c'] < ep * (2.0 - profit_thr)
            std = curr['st_d_t'] if is_pft else curr['st_d_l']
            stv = curr['st_v_t'] if is_pft else curr['st_v_l']
            
            close_s = std == 1 or curr['c'] > stv
            if short_pos['min_p'] < ep * 0.98 and curr['c'] > ep:
                close_s = True
            if close_s:
                trades.append({'side': 'short', 'pnl_pct': (ep - curr['c'])/ep, 'type': 'exit'})
                short_pos = None

        if not long_pos and (is_l_bo or is_l_pb):
            long_pos = {'entry': curr['c'], 'max_p': curr['c']}
        if not short_pos and (is_s_bo or is_s_pb):
            short_pos = {'entry': curr['c'], 'min_p': curr['c']}
            
    return trades

def sim_today(df, tight_mult, profit_thr, vol_mult, min_hold, max_dca=3):
    trades = []
    long_pos = None; short_pos = None
    df['st_d_t'], df['st_v_t'] = calc_supertrend(df, 10, tight_mult)

    for i in range(20, len(df)):
        prev, curr = df.iloc[i-1], df.iloc[i]
        vol_cond = curr['v'] > prev['vol_ma'] * vol_mult
        
        is_l_bo = prev['st_d_l'] == -1 and curr['st_d_l'] == 1 and vol_cond
        is_l_pb = curr['st_d_l'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
        is_s_bo = prev['st_d_l'] == 1 and curr['st_d_l'] == -1 and vol_cond
        is_s_pb = curr['st_d_l'] == -1 and prev['stoch_k'] > 80 and curr['stoch_k'] <= 80

        # Long
        if long_pos:
            ep = long_pos['entry']
            is_pft = curr['c'] > ep * profit_thr
            std = curr['st_d_t'] if is_pft else curr['st_d_l']
            stv = curr['st_v_t'] if is_pft else curr['st_v_l']
            
            close_l = std == -1 or curr['c'] < stv
            force_l = False
            if long_pos['exit_c'] > 0 and curr['c'] < ep:
                if i - long_pos['first_i'] >= min_hold:
                    force_l = True
                    
            if force_l:
                trades.append({'side': 'long', 'pnl_pct': (curr['c'] - ep)/ep * long_pos['size'], 'type': 'be_stop'})
                long_pos = None
            elif close_l and long_pos['exit_c'] < max_dca:
                sf = long_pos['size'] / (max_dca - long_pos['exit_c'])
                trades.append({'side': 'long', 'pnl_pct': (curr['c'] - ep)/ep * sf, 'type': 'dca_exit'})
                long_pos['size'] -= sf; long_pos['exit_c'] += 1
                if long_pos['exit_c'] >= max_dca or long_pos['size'] < 0.001: long_pos = None
            elif not close_l and long_pos['entry_c'] < max_dca:
                add = 1.0 / max_dca
                long_pos['entry'] = (ep * long_pos['size'] + curr['c'] * add) / (long_pos['size'] + add)
                long_pos['size'] += add; long_pos['entry_c'] += 1

        # Short
        if short_pos:
            ep = short_pos['entry']
            is_pft = curr['c'] < ep * (2.0 - profit_thr)
            std = curr['st_d_t'] if is_pft else curr['st_d_l']
            stv = curr['st_v_t'] if is_pft else curr['st_v_l']
            
            close_s = std == 1 or curr['c'] > stv
            force_s = False
            if short_pos['exit_c'] > 0 and curr['c'] > ep:
                if i - short_pos['first_i'] >= min_hold:
                    force_s = True
                    
            if force_s:
                trades.append({'side': 'short', 'pnl_pct': (ep - curr['c'])/ep * short_pos['size'], 'type': 'be_stop'})
                short_pos = None
            elif close_s and short_pos['exit_c'] < max_dca:
                sf = short_pos['size'] / (max_dca - short_pos['exit_c'])
                trades.append({'side': 'short', 'pnl_pct': (ep - curr['c'])/ep * sf, 'type': 'dca_exit'})
                short_pos['size'] -= sf; short_pos['exit_c'] += 1
                if short_pos['exit_c'] >= max_dca or short_pos['size'] < 0.001: short_pos = None
            elif not close_s and short_pos['entry_c'] < max_dca:
                add = 1.0 / max_dca
                short_pos['entry'] = (ep * short_pos['size'] + curr['c'] * add) / (short_pos['size'] + add)
                short_pos['size'] += add; short_pos['entry_c'] += 1

        if not long_pos and (is_l_bo or is_l_pb):
            long_pos = {'entry': curr['c'], 'size': 1.0/max_dca, 'entry_c': 1, 'exit_c': 0, 'first_i': i}
        if not short_pos and (is_s_bo or is_s_pb):
            short_pos = {'entry': curr['c'], 'size': 1.0/max_dca, 'entry_c': 1, 'exit_c': 0, 'first_i': i}
            
    return trades

def stats(trades, leverage=1.0):
    if not trades: return {'ret': 0, 'wr': 0, 'n': 0, 'dd': 0}
    pnls = []
    for t in trades:
        l_pnl = t['pnl_pct'] * leverage
        if l_pnl <= -1.0: l_pnl = -1.0 # Liquidation
        pnls.append(l_pnl * 100)
    ret = sum(pnls); wins = sum(1 for p in pnls if p > 0)
    dd = min((np.cumsum(pnls) - np.maximum.accumulate(np.cumsum(pnls))).min(), 0) if len(pnls)>0 else 0
    return {'ret': round(ret, 2), 'wr': round(wins/len(pnls)*100, 1), 'n': len(trades), 'dd': round(dd, 2)}

async def main():
    print("=" * 70)
    print(" 🤖 10x vs 20x vs 30x LEVERAGE Backtest Comparison")
    print("=" * 70)
    
    cfgs = [
        {"name": "Venture (Meme/Alt)", "syms": ["DOGE/USDT:USDT", "PEPE/USDT:USDT", "WIF/USDT:USDT"], 
         "params": (2.0, 1.025, 1.0, 3, 3)},
        {"name": "Venture (New Crypto)", "syms": ["GRVT/USDT:USDT", "AEON/USDT:USDT", "FLY/USDT:USDT"], 
         "params": (2.0, 1.025, 1.0, 3, 3)},
        {"name": "Stock Venture (New Stocks)", "syms": ["CRM/USDT:USDT", "OKTA/USDT:USDT", "TMF/USDT:USDT"], 
         "params": (2.5, 1.02, 1.1, 3, 3)},
    ]
    
    ex = ccxt_async.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    
    for c in cfgs:
        print(f"\n[{c['name']}]")
        print(f" Symbol          | 10x (Ret/DD) | 20x (Ret/DD) | 30x (Ret/DD) ")
        print("-" * 70)
        t_y = []; t_t = []
        for sym in c['syms']:
            try:
                ohlcv = await ex.fetch_ohlcv(sym, '15m', limit=500)
                df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
                df['st_d_l'], df['st_v_l'] = calc_supertrend(df, 10, 4.0)
                df['vol_ma'] = df['v'].rolling(20).mean()
                k, d = calc_stoch_rsi(df['c'], 14, 3, 3); df['stoch_k'] = k; df['stoch_d'] = d
                
                tr = sim_today(df.copy(), *c['params'])
                t_y.extend(tr); t_t.extend(tr)
                
                s10 = stats(tr, 10.0); s20 = stats(tr, 20.0); s30 = stats(tr, 30.0)
                print(f" {sym:<15} | {s10['ret']:>6.1f}% {s10['dd']:>6.1f}% | {s20['ret']:>6.1f}% {s20['dd']:>6.1f}% | {s30['ret']:>6.1f}% {s30['dd']:>6.1f}%")
            except Exception as e: pass
        
        s10_tot = stats(t_y, 10.0); s20_tot = stats(t_y, 20.0); s30_tot = stats(t_y, 30.0)
        print("-" * 70)
        print(f" TOTAL           | {s10_tot['ret']:>6.1f}% {s10_tot['dd']:>6.1f}% | {s20_tot['ret']:>6.1f}% {s20_tot['dd']:>6.1f}% | {s30_tot['ret']:>6.1f}% {s30_tot['dd']:>6.1f}%")
        
    await ex.close()

if __name__ == "__main__":
    asyncio.run(main())
