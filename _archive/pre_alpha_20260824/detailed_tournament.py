import asyncio
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt_async
import json
import time

from final_backtest_all import simulate

def detailed_stats(trades):
    if not trades:
        return {'ret': 0.0, 'n': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0}
    pnls = [t['pnl'] for t in trades]
    ret = sum(pnls) * 100
    wins = len([p for p in pnls if p > 0])
    losses = len(trades) - wins
    win_rate = (wins / len(trades)) * 100
    return {'ret': round(ret, 2), 'n': len(trades), 'wins': wins, 'losses': losses, 'win_rate': round(win_rate, 2)}

AGENTS = {
    "Current_Base": {"ema_period": 200, "tight_mult": 2.0, "loose_mult": 4.0, "vol_mult": 1.0, "min_hold": 3, "max_dca": 3, "tp_thr": 1.025, "scale_out": True},
    "Alpha_Trend":  {"ema_period": 50,  "tight_mult": 2.0, "loose_mult": 3.0, "vol_mult": 1.2, "min_hold": 3, "max_dca": 8, "tp_thr": 1.03,  "scale_out": True}
}

TEST_SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "DOGE/USDT:USDT", "PEPE/USDT:USDT"]
TIMEFRAME = "15m"
LIMIT = 2000

async def fetch_data(ex, symbol):
    try:
        ohlcv = await ex.fetch_ohlcv(symbol, TIMEFRAME, limit=LIMIT)
        if not ohlcv or len(ohlcv) < 200:
            return None
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
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
    for sym in TEST_SYMBOLS:
        df = await fetch_data(ex, sym)
        if df is not None:
            data_map[sym] = df
        await asyncio.sleep(0.5)
    await ex.close()
    
    report = {}
    for agent_name, params in AGENTS.items():
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
        
    print(json.dumps(report, indent=4))

if __name__ == "__main__":
    asyncio.run(main())
