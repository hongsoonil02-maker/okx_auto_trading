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

async def check():
    ex = ccxt_async.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    ohlcv = await ex.fetch_ohlcv("MEGA/USDT:USDT", '15m', limit=200)
    df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
    df['st_d_loose'], df['st_v_loose'] = calc_supertrend(df, 10, 4.0)
    df['ema_target'] = df['c'].ewm(span=200, adjust=False).mean()
    delta = df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    df['stoch_k'] = ((rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())).rolling(3).mean() * 100
    df['vol_ma'] = df['v'].rolling(20).mean()

    prev = df.iloc[-2]
    curr = df.iloc[-1]
    
    is_ema_trend_up = curr['c'] > curr['ema_target']
    is_long_breakout = prev['st_d_loose'] == -1 and curr['st_d_loose'] == 1
    is_long_pullback = curr['st_d_loose'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
    vol_cond = curr['v'] > prev['vol_ma'] * 1.0

    long_score = 0
    if is_ema_trend_up: long_score += 60
    if is_long_breakout: long_score += 40
    if is_long_pullback: long_score += 40
    
    print(f"EMA Up: {is_ema_trend_up}")
    print(f"Long Breakout: {is_long_breakout}")
    print(f"Long Pullback: {is_long_pullback}")
    print(f"Vol Cond: {vol_cond}")
    print(f"Long Score: {long_score}")
    
    is_ema_trend_down = curr['c'] < curr['ema_target']
    is_short_breakout = prev['st_d_loose'] == 1 and curr['st_d_loose'] == -1
    is_short_pullback = curr['st_d_loose'] == -1 and prev['stoch_k'] > 80 and curr['stoch_k'] <= 80

    short_score = 0
    if is_ema_trend_down: short_score += 60
    if is_short_breakout: short_score += 40
    if is_short_pullback: short_score += 40
    
    print(f"EMA Down: {is_ema_trend_down}")
    print(f"Short Breakout: {is_short_breakout}")
    print(f"Short Pullback: {is_short_pullback}")
    print(f"Short Score: {short_score}")
    
    await ex.close()

if __name__ == "__main__":
    asyncio.run(check())
