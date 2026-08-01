import ccxt
import asyncio

async def run():
    ex = ccxt.async_support.okx()
    markets = await ex.load_markets()
    m = markets.get('OPENAI/USDT:USDT')
    if m:
        print(f"Contract Size: {m['contractSize']}")
    else:
        print("Market not found")
    await ex.close()

asyncio.run(run())
