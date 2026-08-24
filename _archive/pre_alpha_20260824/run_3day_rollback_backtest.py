import asyncio
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt_async
from datetime import datetime, timedelta, timezone

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

def simulate(df, initial_capital=10000, leverage=10):
    capital = initial_capital
    base_margin = initial_capital * 0.1 # 10% of capital as initial margin
    
    long_pos = None
    short_pos = None
    trades = []
    
    # OLD LOGIC PARAMETERS (from Aug 18)
    HARD_STOP_LOSS = -0.30  # -30%
    MAX_PYRAMID = 2
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Supertrend & StochRSI logic
        is_uptrend = row['st_dir'] == 1
        is_downtrend = row['st_dir'] == -1
        
        stoch_oversold = prev['stoch_k'] < 20 and row['stoch_k'] >= 20
        stoch_overbought = prev['stoch_k'] > 80 and row['stoch_k'] <= 80
        
        # Long Logic
        if long_pos is None:
            if is_uptrend and stoch_oversold:
                long_pos = {
                    'entry': row['c'], 'margin': base_margin, 'size': (base_margin * leverage) / row['c'],
                    'pyramid_count': 0, 'highest': row['c']
                }
        else:
            pnl_pct = (row['c'] - long_pos['entry']) / long_pos['entry'] * leverage
            long_pos['highest'] = max(long_pos['highest'], row['c'])
            highest_pnl = (long_pos['highest'] - long_pos['entry']) / long_pos['entry'] * leverage
            
            # Stop Loss (Aug 18 logic had a fixed hard stop at -30%)
            if pnl_pct <= HARD_STOP_LOSS:
                capital += long_pos['margin'] * pnl_pct
                trades.append({'type': 'SL_LONG', 'pnl': long_pos['margin'] * pnl_pct, 'time': row.name})
                long_pos = None
                continue
                
            # Take Profit / Trailing Stop (Aug 18 logic)
            if highest_pnl > 0.4 and pnl_pct < highest_pnl * 0.5:
                capital += long_pos['margin'] * pnl_pct
                trades.append({'type': 'TS_LONG', 'pnl': long_pos['margin'] * pnl_pct, 'time': row.name})
                long_pos = None
                continue
            elif highest_pnl > 0.2 and pnl_pct < 0.05:
                capital += long_pos['margin'] * pnl_pct
                trades.append({'type': 'TS_LONG', 'pnl': long_pos['margin'] * pnl_pct, 'time': row.name})
                long_pos = None
                continue
            elif pnl_pct >= 0.5:
                capital += long_pos['margin'] * pnl_pct
                trades.append({'type': 'TP_LONG', 'pnl': long_pos['margin'] * pnl_pct, 'time': row.name})
                long_pos = None
                continue

        # Short Logic
        if short_pos is None:
            if is_downtrend and stoch_overbought:
                short_pos = {
                    'entry': row['c'], 'margin': base_margin, 'size': (base_margin * leverage) / row['c'],
                    'pyramid_count': 0, 'lowest': row['c']
                }
        else:
            pnl_pct = (short_pos['entry'] - row['c']) / short_pos['entry'] * leverage
            short_pos['lowest'] = min(short_pos['lowest'], row['c'])
            highest_pnl = (short_pos['entry'] - short_pos['lowest']) / short_pos['entry'] * leverage
            
            if pnl_pct <= HARD_STOP_LOSS:
                capital += short_pos['margin'] * pnl_pct
                trades.append({'type': 'SL_SHORT', 'pnl': short_pos['margin'] * pnl_pct, 'time': row.name})
                short_pos = None
                continue
                
            if highest_pnl > 0.4 and pnl_pct < highest_pnl * 0.5:
                capital += short_pos['margin'] * pnl_pct
                trades.append({'type': 'TS_SHORT', 'pnl': short_pos['margin'] * pnl_pct, 'time': row.name})
                short_pos = None
                continue
            elif highest_pnl > 0.2 and pnl_pct < 0.05:
                capital += short_pos['margin'] * pnl_pct
                trades.append({'type': 'TS_SHORT', 'pnl': short_pos['margin'] * pnl_pct, 'time': row.name})
                short_pos = None
                continue
            elif pnl_pct >= 0.5:
                capital += short_pos['margin'] * pnl_pct
                trades.append({'type': 'TP_SHORT', 'pnl': short_pos['margin'] * pnl_pct, 'time': row.name})
                short_pos = None
                continue

    # Close open pos at end
    if long_pos:
        pnl_pct = (df.iloc[-1]['c'] - long_pos['entry']) / long_pos['entry'] * leverage
        capital += long_pos['margin'] * pnl_pct
        trades.append({'type': 'END_LONG', 'pnl': long_pos['margin'] * pnl_pct})
    if short_pos:
        pnl_pct = (short_pos['entry'] - df.iloc[-1]['c']) / short_pos['entry'] * leverage
        capital += short_pos['margin'] * pnl_pct
        trades.append({'type': 'END_SHORT', 'pnl': short_pos['margin'] * pnl_pct})

    return capital, trades

async def run_backtest():
    ex = ccxt_async.okx({'enableRateLimit': True})
    symbols = ['ZRO/USDT', 'CHIP/USDT', 'TRUMP/USDT', 'PUMP/USDT', 'DOS/USDT', 'LAB/USDT', 'KAITO/USDT', 'RE/USDT']
    
    since = ex.milliseconds() - (3 * 24 * 60 * 60 * 1000)
    
    results = {}
    total_pnl = 0
    total_trades = 0
    
    for sym in symbols:
        try:
            ohlcv = await ex.fetch_ohlcv(sym, '15m', since=since)
            if not ohlcv: continue
            df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
            df['t'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('t', inplace=True)
            
            df['st_dir'], df['st_val'] = calc_supertrend(df)
            df['stoch_k'], df['stoch_d'] = calc_stoch_rsi(df['c'])
            df.dropna(inplace=True)
            
            # Start with 2000 per coin as simulated allocation
            final_cap, trades = simulate(df, initial_capital=2000, leverage=10)
            
            pnl = final_cap - 2000
            total_pnl += pnl
            total_trades += len(trades)
            
            results[sym] = {
                'pnl': pnl,
                'trades': len(trades)
            }
            print(f"[{sym}] PnL: {pnl:.2f} USDT | Trades: {len(trades)}")
        except Exception as e:
            print(f"Error {sym}: {e}")
            
    print(f"\n=== BACKTEST RESULTS (Last 3 Days) ===")
    print(f"Total Theoretical PnL (Old Logic): {total_pnl:.2f} USDT")
    print(f"Total Trades: {total_trades}")
    print(f"Actual Loss Recorded (Buggy Logic): -15,000.00 USDT")
    print(f"Difference: {total_pnl - (-15000):.2f} USDT saved with old logic")

    await ex.close()

asyncio.run(run_backtest())
