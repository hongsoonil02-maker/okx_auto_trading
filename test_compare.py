import asyncio, pandas as pd, numpy as np
import ccxt.async_support as ccxt_async
async def main():
    ex = ccxt_async.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    try:
        ohlcv = await ex.fetch_ohlcv("BTC/USDT:USDT", '15m', limit=3000)
        print(f"Data length: {len(ohlcv)}")
    except Exception as e:
        print(e)
    await ex.close()
asyncio.run(main())
