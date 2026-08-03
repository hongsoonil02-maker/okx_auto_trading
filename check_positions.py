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
        positions = await exchange.fetch_positions()
        active = [p for p in positions if float(p['contracts']) > 0]
        if not active:
            print("현재 보유 중인 포지션이 없습니다.")
        else:
            for p in active:
                print(f"포지션 보유: {p['symbol']}, 수량: {p['contracts']}, PnL: {p['percentage']}%")
    finally:
        await exchange.close()

asyncio.run(check())
