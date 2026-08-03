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

async def check_chart(symbol):
    ex = ccxt_async.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    try:
        ohlcv = await ex.fetch_ohlcv(symbol, '15m', limit=250)
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
        df['t_str'] = pd.to_datetime(df['t'], unit='ms')
        
        # Calculate indicators exactly like the bot
        df['st_d_l'], df['st_v_l'] = calc_supertrend(df, 10, 4.0)
        df['ema_200'] = df['c'].ewm(span=200, adjust=False).mean()
        
        delta = df['c'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        stoch_k = ((rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())).rolling(3).mean() * 100
        df['stoch_k'] = stoch_k
        df['vol_ma'] = df['v'].rolling(20).mean()

        print(f"\\n📊 [{symbol}] 15분봉 최신 차트 분석 (봇의 시선)")
        print("="*80)
        print(f"{'시간 (UTC)':<20} | {'종가':<10} | {'EMA200':<10} | {'Supertrend':<12} | {'StochRSI':<10} | {'Volume 배수'}")
        print("-" * 80)
        
        for i in range(len(df)-5, len(df)):
            row = df.iloc[i]
            st_color = "🟢 BUY" if row['st_d_l'] == 1 else "🔴 SELL"
            vol_mult = row['v'] / row['vol_ma'] if row['vol_ma'] > 0 else 0
            is_above_ema = "✅ 위" if row['c'] > row['ema_200'] else "❌ 아래"
            
            print(f"{str(row['t_str']):<20} | {row['c']:<10.4f} | {row['ema_200']:<10.4f} ({is_above_ema}) | {st_color:<12} | {row['stoch_k']:<10.1f} | {vol_mult:.1f}x")
            
        print("="*80)
        print("💡 봇이 왜 진입했을까?")
        print("- 종가가 EMA200 위에 있음 (상승장 기본 조건 만족)")
        print("- Supertrend가 🟢 BUY (상승 추세) 유지 중")
        print("- 어딘가에서 StochRSI 돌파나 거래량 급증 신호가 감지되어 점수가 100점을 넘겼음")
    finally:
        await ex.close()

if __name__ == "__main__":
    asyncio.run(check_chart("MEGA/USDT:USDT"))
