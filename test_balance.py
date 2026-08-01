import asyncio, os
from bot_a_kr_stock import BotAKRStock, KISAuth
from dotenv import load_dotenv

async def main():
    load_dotenv()
    kis_auth = KISAuth(
        api_key=os.getenv("KIS_API_KEY") or os.getenv("KIS_APP_KEY", ""),
        api_secret=(os.getenv("KIS_API_SECRET") or os.getenv("KIS_APP_SECRET") or os.getenv("KIS_SECRET", "")),
        account_no=(os.getenv("KIS_ACCOUNT_NO") or os.getenv("KIS_CANO", "")),
    )
    bot = BotAKRStock(kis_auth)
    await bot.init_session()
    b = await bot.fetch_balance()
    if b and 'output2' in b:
        print(b['output2'])
    await bot.close_session()

asyncio.run(main())
