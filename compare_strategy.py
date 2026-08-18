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
        
        # 1. EMA 기울기 필터 (진입 시)
        is_long_sig = ((prev['st_d_l'] == -1 and curr['st_d_l'] == 1) or (curr['st_d_l'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20)) and curr['c'] > curr['ema_target'] and curr['ema_slope'] > 0 and vol_cond
        
        if long_pos:
            ep = long_pos['entry']
            is_pft = curr['c'] > ep * tp_thr
            std = curr['st_d_t'] if is_pft else curr['st_d_l']
            
            # 2. 24시간(96캔들) 보유 & 0.5% 미만 변동 시 강제 청산
            held_candles = i - long_pos['first_i']
            if held_candles >= 96 and abs(curr['c'] - ep) / ep < 0.005:
                trades.append({'pnl': (curr['c'] - ep) / ep * long_pos['size'] * 10.0})
                long_pos = None
                continue

            # 3. Supertrend가 꺾여도 EMA 기울기가 급격히 꺾이지 않았으면 유지 (우상향 중이면 홀딩)
            st_close_l = std == -1 or curr['c'] < (curr['st_v_t'] if is_pft else curr['st_v_l'])
            close_l = st_close_l and curr['ema_slope'] < 0 # Supertrend 꺾임 + EMA 기울기 꺾임
            
            if close_l:
                trades.append({'pnl': (curr['c'] - ep) / ep * long_pos['size'] * 10.0})
                long_pos = None
            elif not close_l and long_pos['entry_count'] < max_dca:
                # 4. 불타기/물타기 스텝 세분화 (12단계, 가격이 평단에서 0.5% 이상 벗어날 때만 추가 진입)
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
    syms = ["LTC/USDT:USDT", "SOL/USDT:USDT"]
    print("Fetching data...")
    for sym in syms:
        ohlcv = await ex.fetch_ohlcv(sym, '15m', limit=2000)
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
        df['vol_ma'] = df['v'].rolling(20).mean()
        delta = df['c'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan); rsi = 100 - (100 / (1 + rs))
        df['stoch_k'] = ((rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())).rolling(3).mean() * 100
        
        tr_old = simulate_old(df.copy())
        tr_new = simulate_new(df.copy())
        
        pnl_old = sum(t['pnl'] for t in tr_old)*100
        pnl_new = sum(t['pnl'] for t in tr_new)*100
        print(f"[{sym}] OLD Return: {pnl_old:.2f}% | NEW Return: {pnl_new:.2f}% (Trades: Old {len(tr_old)}, New {len(tr_new)})")
    await ex.close()

if __name__ == '__main__':
    asyncio.run(main())
