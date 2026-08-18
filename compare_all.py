import asyncio, pandas as pd, numpy as np
import traceback
import ccxt.async_support as ccxt_async
from tournament_common import calc_supertrend

def stats(trades):
    if not trades: return {'ret': 0, 'n': 0, 'wins': 0, 'losses': 0, 'wr': 0.0}
    pnls = [t['pnl'] for t in trades]
    ret = sum(pnls) * 100
    wins = len([p for p in pnls if p > 0])
    return {'ret': ret, 'n': len(trades), 'wins': wins, 'losses': len(trades)-wins, 'wr': wins/len(trades)*100}

def simulate_old(df, ema_period=50, max_dca=8, tp_thr=1.01):
    trades = []
    long_pos = None; short_pos = None
    df['st_d_t'], df['st_v_t'] = calc_supertrend(df, 10, 2.5)
    df['st_d_l'], df['st_v_l'] = calc_supertrend(df, 10, 4.0)
    df['ema_target'] = df['c'].ewm(span=ema_period, adjust=False).mean()

    for i in range(250, len(df)):
        prev = df.iloc[i-1]; curr = df.iloc[i]
        vol_cond = curr['v'] > prev['vol_ma'] * 1.2
        is_long_sig = ((prev['st_d_l'] == -1 and curr['st_d_l'] == 1) or (curr['st_d_l'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20)) and curr['c'] > curr['ema_target'] and vol_cond
        
        if long_pos:
            ep = long_pos['entry']
            is_pft = curr['c'] > ep * tp_thr
            std = curr['st_d_t'] if is_pft else curr['st_d_l']
            close_l = std == -1 or curr['c'] < (curr['st_v_t'] if is_pft else curr['st_v_l'])
            if close_l:
                trades.append({'pnl': (curr['c'] - ep) / ep * long_pos['size'] * 10.0})
                long_pos = None
            elif not close_l and long_pos['entry_count'] < max_dca:
                long_pos['entry_count'] += 1
                add = 1.0 / max_dca
                long_pos['entry'] = (ep * long_pos['size'] + curr['c'] * add) / (long_pos['size'] + add)
                long_pos['size'] += add
        if not long_pos and is_long_sig:
            long_pos = {'entry': curr['c'], 'size': 1.0/max_dca, 'entry_count': 1}
    return trades

def simulate_new(df, ema_period=50, max_dca=12, tp_thr=1.01):
    trades = []
    long_pos = None
    df['st_d_t'], df['st_v_t'] = calc_supertrend(df, 10, 2.5)
    df['st_d_l'], df['st_v_l'] = calc_supertrend(df, 10, 4.0)
    df['ema_target'] = df['c'].ewm(span=ema_period, adjust=False).mean()
    df['ema_slope'] = df['ema_target'].diff()

    for i in range(250, len(df)):
        prev = df.iloc[i-1]; curr = df.iloc[i]
        vol_cond = curr['v'] > prev['vol_ma'] * 1.2
        is_long_sig = ((prev['st_d_l'] == -1 and curr['st_d_l'] == 1) or (curr['st_d_l'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20)) and curr['c'] > curr['ema_target'] and curr['ema_slope'] > 0 and vol_cond
        
        if long_pos:
            ep = long_pos['entry']
            is_pft = curr['c'] > ep * tp_thr
            std = curr['st_d_t'] if is_pft else curr['st_d_l']
            
            held_candles = i - long_pos['first_i']
            if held_candles >= 96 and abs(curr['c'] - ep) / ep < 0.005:
                trades.append({'pnl': (curr['c'] - ep) / ep * long_pos['size'] * 10.0})
                long_pos = None
                continue

            st_close_l = std == -1 or curr['c'] < (curr['st_v_t'] if is_pft else curr['st_v_l'])
            close_l = st_close_l and curr['ema_slope'] < 0 
            
            if close_l:
                trades.append({'pnl': (curr['c'] - ep) / ep * long_pos['size'] * 10.0})
                long_pos = None
            elif not close_l and long_pos['entry_count'] < max_dca:
                if abs(curr['c'] - ep) / ep >= 0.005:
                    long_pos['entry_count'] += 1
                    add = 1.0 / max_dca
                    long_pos['entry'] = (ep * long_pos['size'] + curr['c'] * add) / (long_pos['size'] + add)
                    long_pos['size'] += add
        
        if not long_pos and is_long_sig:
            long_pos = {'entry': curr['c'], 'size': 1.0/max_dca, 'entry_count': 1, 'first_i': i}
    return trades

async def main():
    ex = ccxt_async.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    syms = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "DOGE/USDT:USDT", "PEPE/USDT:USDT"]
    print(f"{'Symbol':<15} | {'OLD Ret':>8} | {'OLD WR':>6} | {'OLD N':>5} || {'NEW Ret':>8} | {'NEW WR':>6} | {'NEW N':>5}")
    print("-" * 80)
    
    t_old_ret, t_old_wins, t_old_n = 0, 0, 0
    t_new_ret, t_new_wins, t_new_n = 0, 0, 0
    
    for sym in syms:
        try:
            ohlcv = await ex.fetch_ohlcv(sym, '15m', limit=3000)
            df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
            df['vol_ma'] = df['v'].rolling(20).mean()
            delta = df['c'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan); rsi = 100 - (100 / (1 + rs))
            df['stoch_k'] = ((rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())).rolling(3).mean() * 100
            
            s_old = stats(simulate_old(df.copy()))
            s_new = stats(simulate_new(df.copy()))
            
            t_old_ret += s_old['ret']; t_old_wins += s_old['wins']; t_old_n += s_old['n']
            t_new_ret += s_new['ret']; t_new_wins += s_new['wins']; t_new_n += s_new['n']
            
            print(f"{sym:<15} | {s_old['ret']:>7.2f}% | {s_old['wr']:>5.1f}% | {s_old['n']:>5} || {s_new['ret']:>7.2f}% | {s_new['wr']:>5.1f}% | {s_new['n']:>5}")
        except Exception as e:
            print("ERROR", e)
            traceback.print_exc()
    print("-" * 80)
    o_wr = t_old_wins/t_old_n*100 if t_old_n else 0
    n_wr = t_new_wins/t_new_n*100 if t_new_n else 0
    print(f"{'TOTAL':<15} | {t_old_ret:>7.2f}% | {o_wr:>5.1f}% | {t_old_n:>5} || {t_new_ret:>7.2f}% | {n_wr:>5.1f}% | {t_new_n:>5}")
    await ex.close()

if __name__ == '__main__':
    asyncio.run(main())
