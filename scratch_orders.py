import os
import asyncio
import ccxt.async_support as ccxt_async
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

async def check():
    exchange = ccxt_async.okx({
        "apiKey": os.getenv("OKX_API_KEY", ""),
        "secret": os.getenv("OKX_SECRET", "") or os.getenv("OKX_API_SECRET", ""),
        "password": os.getenv("OKX_PASSPHRASE", "") or os.getenv("OKX_PASSWORD", ""),
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })
    try:
        orders = await exchange.fetch_open_orders()
        if not orders:
            print("현재 미체결 주문(Open Orders)이 없습니다.")
        else:
            for o in orders:
                print(f"미체결 주문: {o['symbol']}, 방향: {o['side']}, 수량: {o['amount']}, 가격: {o['price']}")
    finally:
        await exchange.close()

asyncio.run(check())
