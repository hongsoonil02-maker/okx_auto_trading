import ccxt
import pandas as pd
import time

exchange = ccxt.okx({'enableRateLimit': True})
ohlcv = exchange.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=220)
closes = pd.Series([c[4] for c in ohlcv], dtype='float64')
ema200 = closes.ewm(span=200, adjust=False).mean().iloc[-1]
ema50 = closes.ewm(span=50, adjust=False).mean().iloc[-1]
current_price = closes.iloc[-1]

print(f"BTC Current 1h Price: {current_price:.2f}")
print(f"BTC 1h EMA50: {ema50:.2f}")
print(f"BTC 1h EMA200: {ema200:.2f}")
print(f"Long Regime OK (Price >= EMA200): {current_price >= ema200}")
print(f"Dual Gate OK (Price > EMA50): {current_price > ema50}")
