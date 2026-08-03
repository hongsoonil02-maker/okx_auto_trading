import os
import asyncio
import ccxt.async_support as ccxt_async
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

async def close_failed_positions():
    print("🚨 긴급 청산 스크립트 가동...")
    exchange = ccxt_async.okx({
        "apiKey": os.getenv("OKX_API_KEY", ""),
        "secret": os.getenv("OKX_SECRET", "") or os.getenv("OKX_API_SECRET", ""),
        "password": os.getenv("OKX_PASSPHRASE", "") or os.getenv("OKX_PASSWORD", ""),
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
    
    try:
        await exchange.load_markets()
        positions = await exchange.fetch_positions()
        
        failed_bases = ['MU', 'SHAZ', 'ISRG', 'ROBO', 'RAM', 'DRAM', 'GME']
        
        closed_count = 0
        for pos in positions:
            symbol = pos['symbol']
            base = symbol.split('/')[0].split('-')[0]
            if base in failed_bases and float(pos['contracts']) > 0:
                print(f"🔪 청산 대상 발견: {symbol}, 수량: {pos['contracts']}, PnL: {pos['percentage']}%")
                side = 'sell' if pos['side'] == 'long' else 'buy'
                
                try:
                    await exchange.create_order(
                        symbol=symbol,
                        type='market',
                        side=side,
                        amount=pos['contracts'],
                        params={'reduceOnly': True, 'posSide': pos['side']}
                    )
                    print(f"✅ {symbol} 청산 완료!")
                    closed_count += 1
                except Exception as ex:
                    print(f"청산 실패 ({symbol}): {ex}")
                    
        if closed_count == 0:
            print("대응되는 포지션을 찾을 수 없거나 이미 모두 청산되었습니다.")
            
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(close_failed_positions())
