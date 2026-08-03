import asyncio
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt_async
import json
import time

# final_backtest_all.py에서 시뮬레이터 재사용
from final_backtest_all import simulate, stats

AGENTS = {
    "Current_Base": {"ema_period": 200, "tight_mult": 2.0, "loose_mult": 4.0, "vol_mult": 1.0, "min_hold": 3, "max_dca": 8, "tp_thr": 1.025, "scale_out": True},
    "Alpha_Trend":  {"ema_period": 50,  "tight_mult": 2.0, "loose_mult": 3.0, "vol_mult": 1.2, "min_hold": 3, "max_dca": 8, "tp_thr": 1.03,  "scale_out": True},
    "Beta_Safe":    {"ema_period": 200, "tight_mult": 1.5, "loose_mult": 2.5, "vol_mult": 1.5, "min_hold": 5, "max_dca": 5, "tp_thr": 1.01,  "scale_out": False},
    "Gamma_Sniper": {"ema_period": 100, "tight_mult": 2.5, "loose_mult": 4.0, "vol_mult": 1.5, "min_hold": 1, "max_dca": 8, "tp_thr": 1.015, "scale_out": True},
    "Delta_DCA":    {"ema_period": 200, "tight_mult": 2.5, "loose_mult": 4.0, "vol_mult": 1.0, "min_hold": 3, "max_dca": 12, "tp_thr": 1.02,  "scale_out": True},
    "Epsilon_High": {"ema_period": 200, "tight_mult": 3.0, "loose_mult": 5.0, "vol_mult": 1.2, "min_hold": 3, "max_dca": 8, "tp_thr": 1.05,  "scale_out": False},
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
        print(f"Error fetching {symbol}: {e}")
        return None

async def main():
    print("🚀 [ORCA AI Tournament Engine] 시작...")
    start_t = time.time()
    ex = ccxt_async.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    
    data_map = {}
    print("1. OHLCV 과거 데이터 수집 중...")
    for sym in TEST_SYMBOLS:
        df = await fetch_data(ex, sym)
        if df is not None:
            data_map[sym] = df
        await asyncio.sleep(0.5)
    await ex.close()
    
    print("2. 5 AI 에이전트 병렬 시뮬레이션 가동...")
    results = {agent: 0.0 for agent in AGENTS}
    details = {agent: {} for agent in AGENTS}
    
    for agent_name, params in AGENTS.items():
        total_ret = 0.0
        for sym, df in data_map.items():
            tr = simulate(df.copy(), **params)
            s = stats(tr)
            total_ret += s['ret']
            details[agent_name][sym] = s['ret']
        results[agent_name] = total_ret
        print(f"✅ {agent_name} 완료 -> 총 누적 수익률: {total_ret:.2f}%")
        
    print("\n🏆 [토너먼트 결과 발표]")
    sorted_agents = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for i, (agent, ret) in enumerate(sorted_agents):
        print(f"{i+1}위: {agent} ({ret:.2f}%)")
        
    # JSON 파일로 리포트 덤프
    with open('tournament_results.json', 'w') as f:
        json.dump({'ranking': sorted_agents, 'details': details, 'params': AGENTS}, f, indent=4)
        
    print(f"소요 시간: {time.time()-start_t:.1f}초")

if __name__ == "__main__":
    asyncio.run(main())
