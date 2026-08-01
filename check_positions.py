import asyncio
import ccxt.async_support as ccxt_async
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OKX_API_KEY", "")
API_SECRET = os.getenv("OKX_SECRET", "") or os.getenv("OKX_API_SECRET", "")
PASSPHRASE = os.getenv("OKX_PASSPHRASE", "") or os.getenv("OKX_PASSWORD", "")

async def main():
    exchange = ccxt_async.okx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "password": PASSPHRASE,
        "enableRateLimit": True,
        "options": {"defaultType": "swap"}
    })
    try:
        positions = await exchange.fetch_positions()
        for p in positions:
            if float(p.get('contracts', 0)) > 0:
                print(f"{p['symbol']} | Side: {p['side']} | Contracts: {p['contracts']}")
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())
