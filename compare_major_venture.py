#!/usr/bin/env python3
import os
import json
from collections import defaultdict
from daily_analyzer import _load_trades_jsonl

# Major 봇 대상 코인 추정 리스트 (okx_major_strategy에 주로 들어가는 10개 코인)
MAJOR_COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOT", "TRX", "LINK"]

def main():
    lookback = 30
    trades = _load_trades_jsonl(lookback)
    
    if not trades:
        print("No trades found.")
        return
        
    major_stats = {"wins": 0, "losses": 0, "pnl_sum": 0.0, "trades": 0}
    venture_stats = {"wins": 0, "losses": 0, "pnl_sum": 0.0, "trades": 0}
    
    for t in trades:
        sym = t['symbol'].split('-')[0].split('/')[0].upper()
        pnl = t['pnl']
        
        if sym in MAJOR_COINS:
            major_stats['trades'] += 1
            major_stats['pnl_sum'] += pnl
            if pnl > 0:
                major_stats['wins'] += 1
            else:
                major_stats['losses'] += 1
        else:
            venture_stats['trades'] += 1
            venture_stats['pnl_sum'] += pnl
            if pnl > 0:
                venture_stats['wins'] += 1
            else:
                venture_stats['losses'] += 1
                
    print("="*60)
    print("📊 [Major vs Venture 봇 최근 30일 성과 비교]")
    print("="*60)
    
    for name, stats in [("Major 봇 (Top 10 코인)", major_stats), ("Venture 봇 (기타 알트코인)", venture_stats)]:
        t = stats['trades']
        w = stats['wins']
        l = stats['losses']
        pnl = stats['pnl_sum']
        wr = (w / t * 100) if t > 0 else 0
        avg_pnl = (pnl / t) if t > 0 else 0
        
        print(f"📌 {name}")
        print(f"   - 총 거래 수: {t}회")
        print(f"   - 승률: {wr:.1f}% (승 {w} / 패 {l})")
        print(f"   - 누적 PnL(마진 대비 %): {pnl:+.2f}%")
        print(f"   - 평균 PnL: {avg_pnl:+.2f}%")
        print("-" * 60)

if __name__ == "__main__":
    main()
