import asyncio
import ccxt.async_support as ccxt_async
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    exchange = ccxt_async.okx({
        "apiKey": os.getenv("OKX_API_KEY", ""),
        "secret": os.getenv("OKX_SECRET", "") or os.getenv("OKX_API_SECRET", ""),
        "password": os.getenv("OKX_PASSPHRASE", "") or os.getenv("OKX_PASSWORD", ""),
        "enableRateLimit": True,
        "options": {"defaultType": "swap"}
    })
    try:
        # Cancel all open orders first
        print("Canceling all open orders for LITE...")
        await exchange.cancel_all_orders('LITE/USDT:USDT')
        print("Orders canceled. Retrying close_position...")
        order = await exchange.close_position('LITE/USDT:USDT', side='short')
        print("Success!", order)
    except Exception as e:
        print("Failed:", e)
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())
