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
        symbol = 'LITE/USDT:USDT'
        print(f"Trying close_position for {symbol}...")
        if exchange.has.get('closePosition'):
            order = await exchange.close_position(symbol, side='short')
            print("Success!", order)
        else:
            print("closePosition not supported in ccxt for okx.")
            
    except Exception as e:
        print("Failed:", e)
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())
