#!/usr/bin/env python3
import asyncio, os, ccxt.async_support as ccxt_async
from dotenv import load_dotenv

async def check_balance():
    load_dotenv()
    exchange = ccxt_async.okx({
        'apiKey': os.getenv('OKX_API_KEY'),
        'secret': os.getenv('OKX_SECRET'),
        'password': os.getenv('OKX_PASSPHRASE'),
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })
    try:
        balance = await exchange.fetch_balance()
        free_usdt = balance.get('USDT', {}).get('free', 0)
        total_usdt = balance.get('USDT', {}).get('total', 0)
        print("=" * 40)
        print(f"💰 현재 OKX 계좌 자산 현황")
        print("=" * 40)
        print(f"총 보유 USDT (Total)   : {total_usdt:,.2f} USDT")
        print(f"주문 가능 USDT (Free)  : {free_usdt:,.2f} USDT")
        print("=" * 40)
    except Exception as e:
        print(f"❌ 잔고 조회 실패: {e}")
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(check_balance())
