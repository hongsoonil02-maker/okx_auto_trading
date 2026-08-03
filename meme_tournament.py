import asyncio
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt_async
import json

from final_backtest_all import simulate
from vibe_trading_simulator import simulate_vibe
from detailed_tournament import detailed_stats

MEME_SYMBOLS = ['PEPE/USDT:USDT', 'DOGE/USDT:USDT', 'WIF/USDT:USDT', 'ORDI/USDT:USDT', 'SHIB/USDT:USDT', 'MEME/USDT:USDT']
TIMEFRAME = "15m"
LIMIT = 2000

async def fetch_data(ex, symbol):
    try:
        ohlcv = await ex.fetch_ohlcv(symbol, TIMEFRAME, limit=LIMIT)
        if not ohlcv or len(ohlcv) < 200:
            return None
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
        # Current Base requires some extra columns
        df['vol_ma'] = df['v'].rolling(20).mean()
        delta = df['c'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        df['stoch_k'] = ((rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())).rolling(3).mean() * 100
        return df
    except Exception as e:
        return None

async def main():
    ex = ccxt_async.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    data_map = {}
    
    print("📡 Fetching data for Meme/Alt Crypto Tokens on OKX...")
    for sym in MEME_SYMBOLS:
        df = await fetch_data(ex, sym)
        if df is not None:
            data_map[sym] = df
        await asyncio.sleep(0.5)
    await ex.close()
    
    if not data_map:
        print(json.dumps({"error": "No data found for any meme symbols."}))
        return

    report = {}
    
    # 1. Current Venture Base
    agent_name = "Current_Venture_Base"
    params = {"ema_period": 200, "tight_mult": 2.0, "loose_mult": 4.0, "vol_mult": 1.0, "min_hold": 3, "max_dca": 3, "tp_thr": 1.025, "scale_out": False}
    
    total_ret = 0.0
    total_trades = 0
    total_wins = 0
    sym_stats = {}
    
    for sym, df in data_map.items():
        tr = simulate(df.copy(), **params)
        s = detailed_stats(tr)
        sym_stats[sym] = s
        total_ret += s['ret']
        total_trades += s['n']
        total_wins += s['wins']
        
    overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
    report[agent_name] = {
        'overall_ret': round(total_ret, 2),
        'total_trades': total_trades,
        'overall_win_rate': round(overall_win_rate, 2),
        'symbols': sym_stats
    }

    # 2. Geumgang Logic (Vibe)
    agent_name = "Geumgang_Logic"
    
    total_ret = 0.0
    total_trades = 0
    total_wins = 0
    sym_stats = {}
    
    for sym, df in data_map.items():
        tr = simulate_vibe(df.copy())
        s = detailed_stats(tr)
        sym_stats[sym] = s
        total_ret += s['ret']
        total_trades += s['n']
        total_wins += s['wins']
        
    overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
    report[agent_name] = {
        'overall_ret': round(total_ret, 2),
        'total_trades': total_trades,
        'overall_win_rate': round(overall_win_rate, 2),
        'symbols': sym_stats
    }
        
    print(json.dumps(report, indent=4))

if __name__ == "__main__":
    asyncio.run(main())
