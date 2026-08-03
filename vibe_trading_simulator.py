import asyncio
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt_async
import time

def simulate_vibe(df, bb_period=20, bb_std=2.5, rsi_period=14, tp_pct=0.005, dca_pct=0.005, max_dca=8):
    # Calculate Bollinger Bands
    df['sma20'] = df['c'].rolling(bb_period).mean()
    df['std'] = df['c'].rolling(bb_period).std()
    df['upper_bb'] = df['sma20'] + (df['std'] * bb_std)
    df['lower_bb'] = df['sma20'] - (df['std'] * bb_std)
    
    # Calculate RSI
    delta = df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    trades = []
    pos = None
    
    for i in range(50, len(df)):
        curr = df.iloc[i]
        
        is_long_sig = curr['c'] < curr['lower_bb'] and curr['rsi'] < 30
        is_short_sig = curr['c'] > curr['upper_bb'] and curr['rsi'] > 70
        
        if pos:
            ep = pos['entry']
            if pos['side'] == 'long':
                is_pft = curr['c'] >= ep * (1 + tp_pct)
                is_mean_rev = curr['c'] >= curr['sma20']
                if is_pft or is_mean_rev:
                    trades.append({'side': 'long', 'pnl': (curr['c'] - ep) / ep * pos['size'] * 50.0})
                    pos = None
                elif curr['c'] <= ep * (1 - dca_pct) and pos['dca_count'] < max_dca:
                    pos['dca_count'] += 1
                    add = 1.0 / max_dca
                    pos['entry'] = (ep * pos['size'] + curr['c'] * add) / (pos['size'] + add)
                    pos['size'] += add
            else:
                is_pft = curr['c'] <= ep * (1 - tp_pct)
                is_mean_rev = curr['c'] <= curr['sma20']
                if is_pft or is_mean_rev:
                    trades.append({'side': 'short', 'pnl': (ep - curr['c']) / ep * pos['size'] * 50.0})
                    pos = None
                elif curr['c'] >= ep * (1 + dca_pct) and pos['dca_count'] < max_dca:
                    pos['dca_count'] += 1
                    add = 1.0 / max_dca
                    pos['entry'] = (ep * pos['size'] + curr['c'] * add) / (pos['size'] + add)
                    pos['size'] += add
        else:
            if is_long_sig:
                pos = {'side': 'long', 'entry': curr['c'], 'size': 1.0/max_dca, 'dca_count': 1}
            elif is_short_sig:
                pos = {'side': 'short', 'entry': curr['c'], 'size': 1.0/max_dca, 'dca_count': 1}
                
    if pos:
        ep = pos['entry']
        lc = df.iloc[-1]['c']
        if pos['side'] == 'long':
            trades.append({'side': 'long', 'pnl': (lc - ep) / ep * pos['size'] * 50.0})
        else:
            trades.append({'side': 'short', 'pnl': (ep - lc) / ep * pos['size'] * 50.0})
            
    return trades

async def main():
    ex = ccxt_async.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    symbols = ["XRP/USDT:USDT", "ETH/USDT:USDT", "BTC/USDT:USDT"]
    
    print("🚀 [Vibe-Trading AI] 금강대제 로직 백테스트 가동")
    print(f"테스트 기간: 15m 봉 기준 최근 2000캔들 (~7월 중순~8월초 데이터 포함)")
    print("="*60)
    
    total_trades = 0
    total_wins = 0
    total_ret = 0.0
    
    for sym in symbols:
        try:
            ohlcv = await ex.fetch_ohlcv(sym, "15m", limit=2000)
            df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
            trades = simulate_vibe(df)
            
            n = len(trades)
            if n == 0:
                continue
            pnls = [t['pnl'] for t in trades]
            ret = sum(pnls) * 100
            wins = len([p for p in pnls if p > 0])
            
            total_trades += n
            total_wins += wins
            total_ret += ret
            
            win_rate = (wins / n) * 100
            print(f"[{sym}] 매매 횟수: {n}회 | 승률: {win_rate:.1f}% | 누적 수익률(ROE): {ret:+.2f}%")
            
        except Exception as e:
            print(f"Error on {sym}: {e}")
            
    if total_trades > 0:
        overall_win_rate = (total_wins / total_trades) * 100
        print("="*60)
        print(f"🏆 전체 종합 승률: {overall_win_rate:.1f}% ({total_wins}승 / {total_trades-total_wins}패)")
        print(f"💰 전체 누적 수익률: {total_ret:+.2f}%")
        
    await ex.close()

if __name__ == "__main__":
    asyncio.run(main())
