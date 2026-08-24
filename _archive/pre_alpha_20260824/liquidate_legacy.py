import asyncio
import ccxt.async_support as ccxt_async
import os
from dotenv import load_dotenv

load_dotenv()

async def liquidate_all_positions():
    print("🚨 모든 활성 포지션을 시장가로 긴급 청산합니다...")
    exchange = ccxt_async.okx({
        'apiKey': os.getenv('OKX_API_KEY'),
        'secret': os.getenv('OKX_SECRET'),
        'password': os.getenv('OKX_PASSPHRASE'),
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })
    
    try:
        positions = await exchange.fetch_positions()
        for p in positions:
            symbol = p['symbol']
            side = p['side']
            contracts = float(p['contracts'])
            
            if contracts > 0:
                print(f"🧹 청산 대상 발견: {symbol} | {side} | {contracts} 계약")
                close_side = "sell" if side == "long" else "buy"
                params = {"reduceOnly": True}
                if side == "long":
                    params['posSide'] = "long"
                else:
                    params['posSide'] = "short"
                    
                await exchange.create_order(
                    symbol=symbol,
                    type="market",
                    side=close_side,
                    amount=contracts,
                    params=params
                )
                print(f"✅ {symbol} 청산 주문 완료!")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        await exchange.close()
        print("🏁 청산 스크립트 완료.")

if __name__ == "__main__":
    asyncio.run(liquidate_all_positions())
