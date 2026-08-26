import ccxt
import os
from dotenv import load_dotenv

load_dotenv()
try:
    ex = ccxt.okx({
        'apiKey': os.getenv('OKX_API_KEY'),
        'secret': os.getenv('OKX_SECRET'),
        'password': os.getenv('OKX_PASSPHRASE'),
        'enableRateLimit': True,
    })
    
    bal = ex.fetch_balance()
    usdt_bal = bal['info']['data'][0]['details']
    
    usdt_eq = 0
    for d in usdt_bal:
        if d['ccy'] == 'USDT':
            usdt_eq = float(d['eq'])
            break
            
    positions = ex.fetch_positions()
    total_upl = sum(float(p['info'].get('upl', 0)) for p in positions)
    
    print(f"총 자산(Equity): {usdt_eq:.2f} USDT")
    print(f"현재 미실현 손익(Floating PnL): {total_upl:.2f} USDT")
    print(f"활성 포지션 수: {len(positions)}개")
except Exception as e:
    print(f"Error fetching data: {e}")
