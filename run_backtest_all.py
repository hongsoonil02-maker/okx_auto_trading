import asyncio
import ccxt.async_support as ccxt_async
import pandas as pd
import numpy as np

def calc_supertrend(df, period=10, multiplier=3.0):
    hl2 = (df['h'] + df['l']) / 2
    atr = (df['h'].combine(df['c'].shift(), max) - df['l'].combine(df['c'].shift(), min)).rolling(period).mean()
    final_upperband = hl2 + (multiplier * atr)
    final_lowerband = hl2 - (multiplier * atr)
    st_dir = pd.Series(1, index=df.index, dtype='int')
    st_val = pd.Series(0.0, index=df.index, dtype='float64')
    for i in range(period, len(df)):
        if df['c'].iloc[i] > final_upperband.iloc[i-1]:
            st_dir.iloc[i] = 1
        elif df['c'].iloc[i] < final_lowerband.iloc[i-1]:
            st_dir.iloc[i] = -1
        else:
            st_dir.iloc[i] = st_dir.iloc[i-1]
            if st_dir.iloc[i] == 1 and final_lowerband.iloc[i] < final_lowerband.iloc[i-1]:
                final_lowerband.iloc[i] = final_lowerband.iloc[i-1]
            if st_dir.iloc[i] == -1 and final_upperband.iloc[i] > final_upperband.iloc[i-1]:
                final_upperband.iloc[i] = final_upperband.iloc[i-1]
        if st_dir.iloc[i] == 1:
            st_val.iloc[i] = final_lowerband.iloc[i]
        else:
            st_val.iloc[i] = final_upperband.iloc[i]
    return st_dir, st_val

def calc_stoch_rsi(series, period=14, smooth_k=3, smooth_d=3):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    k = stoch_rsi.rolling(smooth_k).mean() * 100
    d = k.rolling(smooth_d).mean()
    return k, d

def simulate(df, bot_type, mode='before'):
    trades = []
    in_position = False
    entry_price = 0.0
    pos_size = 0.0
    exit_count = 0
    pnl = 0.0
    
    for i in range(20, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]
        
        is_long_sig = False
        close_long_sig = False
        
        # --- BEFORE LOGIC (Current bot specifics) ---
        if mode == 'before':
            if bot_type == 'Major':
                # ST 3.0
                is_long_breakout = prev['st_d_30'] == -1 and curr['st_d_30'] == 1
                is_long_pullback = curr['st_d_30'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
                is_long_sig = is_long_breakout or is_long_pullback
                close_long_sig = curr['st_d_30'] == -1 or curr['c'] < curr['st_v_30']
                
            elif bot_type == 'StockLinked':
                # ST 3.5
                is_long_breakout = prev['st_d_35'] == -1 and curr['st_d_35'] == 1
                is_long_pullback = curr['st_d_35'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
                is_long_sig = is_long_breakout or is_long_pullback
                close_long_sig = curr['st_d_35'] == -1 or curr['c'] < curr['st_v_35']
                
            elif bot_type == 'MemeAlt':
                # ST 4.0 Loose / 2.5 Tight
                is_long_breakout = prev['st_d_40'] == -1 and curr['st_d_40'] == 1
                is_long_pullback = curr['st_d_40'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
                is_long_sig = is_long_breakout or is_long_pullback
                if in_position:
                    st_v_long = curr['st_v_25'] if curr['c'] > entry_price * 1.01 else curr['st_v_40']
                    st_d_long = curr['st_d_25'] if curr['c'] > entry_price * 1.01 else curr['st_d_40']
                    close_long_sig = st_d_long == -1 or curr['c'] < st_v_long
                    
            elif bot_type == 'NewListing':
                # ST 4.0 / 2.0 (High volatility)
                is_long_breakout = prev['st_d_40'] == -1 and curr['st_d_40'] == 1
                is_long_pullback = curr['st_d_40'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
                is_long_sig = is_long_breakout or is_long_pullback
                if in_position:
                    st_v_long = curr['st_v_20'] if curr['c'] > entry_price * 1.015 else curr['st_v_40']
                    st_d_long = curr['st_d_20'] if curr['c'] > entry_price * 1.015 else curr['st_d_40']
                    close_long_sig = st_d_long == -1 or curr['c'] < st_v_long
                    
        # --- AFTER LOGIC (Improved Unified Strategy) ---
        else:
            # Volume filter applied, and smart trailing stop
            vol_cond = curr['v'] > prev['vol_ma'] * 1.2
            
            # Base entry uses 4.0 for all to avoid whipsaws, plus volume confirmation
            is_long_breakout = prev['st_d_40'] == -1 and curr['st_d_40'] == 1 and vol_cond
            is_long_pullback = curr['st_d_40'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
            is_long_sig = is_long_breakout or is_long_pullback
            
            if in_position:
                # Dynamic tight trailing based on bot volatility
                tgt_pct = 1.015 if bot_type in ['Major', 'StockLinked'] else 1.025
                is_profit = curr['c'] > entry_price * tgt_pct
                
                tight_v = curr['st_v_20'] if bot_type in ['MemeAlt', 'NewListing'] else curr['st_v_25']
                tight_d = curr['st_d_20'] if bot_type in ['MemeAlt', 'NewListing'] else curr['st_d_25']
                
                st_v_long = tight_v if is_profit else curr['st_v_40']
                st_d_long = tight_d if is_profit else curr['st_d_40']
                
                close_long_sig = st_d_long == -1 or curr['c'] < st_v_long
                if exit_count > 0 and curr['c'] < entry_price:
                    close_long_sig = True # Breakeven
        
        # Execution
        if not in_position:
            if is_long_sig:
                in_position = True
                entry_price = curr['c']
                pos_size = 1.0
                exit_count = 0
        else:
            # Exit logic
            if close_long_sig:
                if mode == 'after' and exit_count > 0 and curr['c'] < entry_price:
                    sell_size = pos_size
                    trade_pnl = (curr['c'] - entry_price) / entry_price * sell_size
                    pnl += trade_pnl
                    pos_size = 0
                    in_position = False
                    trades.append(trade_pnl)
                else:
                    # Partial exit (split into 8 chunks for DCA out) - assuming standard DCA out in both modes
                    sell_size = pos_size / (8 - exit_count)
                    trade_pnl = (curr['c'] - entry_price) / entry_price * sell_size
                    pnl += trade_pnl
                    pos_size -= sell_size
                    exit_count += 1
                    trades.append(trade_pnl)
                    if exit_count >= 8 or pos_size < 0.001:
                        in_position = False

    win_trades = [t for t in trades if t > 0]
    win_rate = len(win_trades) / len(trades) * 100 if trades else 0
    return {
        "Total Return (%)": pnl * 100,
        "Total Trades": len(trades),
        "Win Rate (%)": win_rate,
        "Max Loss (%)": min(trades)*100 if trades else 0,
    }

async def run_all():
    exchange = ccxt_async.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    
    bots = [
        {"name": "Major", "symbol": "BTC/USDT:USDT"},
        {"name": "StockLinked", "symbol": "WDC/USDT:USDT"},
        {"name": "MemeAlt", "symbol": "DOGE/USDT:USDT"},
        {"name": "NewListing", "symbol": "WIF/USDT:USDT"}
    ]
    
    try:
        results_str = ""
        for b in bots:
            symbol = b['symbol']
            bot_type = b['name']
            print(f"Fetching data for {symbol} ({bot_type})...")
            ohlcv = await exchange.fetch_ohlcv(symbol, '15m', limit=1440)
            df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
            
            # Precalculate indicators
            df['st_d_20'], df['st_v_20'] = calc_supertrend(df, 10, 2.0)
            df['st_d_25'], df['st_v_25'] = calc_supertrend(df, 10, 2.5)
            df['st_d_30'], df['st_v_30'] = calc_supertrend(df, 10, 3.0)
            df['st_d_35'], df['st_v_35'] = calc_supertrend(df, 10, 3.5)
            df['st_d_40'], df['st_v_40'] = calc_supertrend(df, 10, 4.0)
            
            df['vol_ma'] = df['v'].rolling(20).mean()
            k, d = calc_stoch_rsi(df['c'], 14, 3, 3)
            df['stoch_k'] = k
            df['stoch_d'] = d
            
            res_before = simulate(df, bot_type, 'before')
            res_after = simulate(df, bot_type, 'after')
            
            results_str += f"=== {bot_type} Bot ({symbol}) ===\n"
            for key in res_before.keys():
                results_str += f"{key} | Before: {res_before[key]:.2f} | After: {res_after[key]:.2f}\n"
            results_str += "\n"
                
        print("\n\n" + results_str)
            
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(run_all())
