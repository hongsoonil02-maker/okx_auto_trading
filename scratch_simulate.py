import yfinance as yf
import pandas as pd

tickers = ['001790.KS', '002820.KS', '102260.KS']
capital_per_ticker = 3000000  # Assume 3,000,000 KRW per ticker
capital_per_entry = capital_per_ticker / 3

print(f"{'Ticker':<10} | {'09:00 Price':<12} | {'09:10 Price':<12} | {'09:20 Price':<12} | {'09:30 Price':<12} | {'10:30 Price':<12} | {'Old PnL':<12} | {'New PnL':<12}")
print("-" * 95)

for ticker in tickers:
    df = yf.download(ticker, period='1d', interval='5m', progress=False)
    if df.empty:
        continue
    
    # yfinance uses MultiIndex for columns now, flatten it if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    # Get times (00:00 is 09:00 KST, 00:10 is 09:10 KST, etc.)
    # We use Open prices for the entries at that exact minute, or Close of the previous if we enter at the end of the minute
    def get_price(time_str):
        # time_str in UTC, e.g., '2026-07-27 00:10:00+00:00'
        # find the exact time or nearest
        for idx in df.index:
            if str(idx).startswith(f"2026-07-27 {time_str}"):
                return float(df.loc[idx, 'Open'])
        return None

    p_0900 = get_price("00:00:00")
    p_0910 = get_price("00:10:00")
    p_0920 = get_price("00:20:00")
    p_0930 = get_price("00:30:00")
    p_1030 = get_price("01:30:00") # 10:30 KST liquidation
    
    if None in [p_0900, p_0910, p_0920, p_0930, p_1030]:
        continue

    # Old Strategy: Buy all at 09:00
    shares_old = capital_per_ticker / p_0900
    pnl_old = shares_old * (p_1030 - p_0900)
    
    # New Strategy: Buy 1/3 at 09:10, 09:20, 09:30
    shares_1 = capital_per_entry / p_0910
    shares_2 = capital_per_entry / p_0920
    shares_3 = capital_per_entry / p_0930
    total_shares = shares_1 + shares_2 + shares_3
    avg_price = capital_per_ticker / total_shares
    pnl_new = total_shares * (p_1030 - avg_price)
    
    print(f"{ticker:<10} | {p_0900:<12.1f} | {p_0910:<12.1f} | {p_0920:<12.1f} | {p_0930:<12.1f} | {p_1030:<12.1f} | {pnl_old:<12.1f} | {pnl_new:<12.1f}")
