#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
okx_stock_strategy.py — OKX Stock Token Auto Trading Engine (MAJORS)
- Trades major US stock tokens (TSLA, AAPL, etc.)
- Geumgang Mirror Logic (Mean Reversion BB + DCA) applied
"""
import asyncio
import sys
from strategy_common import BaseStrategyBrain

class OKXStockStrategyBrain(BaseStrategyBrain):
    STRATEGY_NAME = "OKX_Stock_Majors"
    LOG_FILE = "okx_stock_strategy.log"
    LOGGER_NAME = "OKX_Stock_Strategy"
    
    STOCK_KEYWORDS = ['TSLA', 'NVDA', 'AAPL', 'AMZN', 'MSFT', 'META', 'GOOG']
    BLACKLIST = ['KR200', 'SKHYNIX']
    # 주식 토큰은 일 거래대금이 $0.3M~3M 수준이라 기본 10M 기준 적용 시 영구 차단됨
    MIN_QUOTE_VOLUME = 100000

    def _symbol_matches(self, symbol: str, ticker_data: dict, markets: dict) -> bool:
        if symbol not in markets or not markets[symbol].get('swap'):
            return False
        if 'USDT' not in symbol:
            return False
        if not any(k in symbol for k in self.STOCK_KEYWORDS):
            return False
        if any(b in symbol for b in self.BLACKLIST):
            return False
        return True



if __name__ == "__main__":
    brain = OKXStockStrategyBrain()
    try:
        asyncio.run(brain.run_all())
    except KeyboardInterrupt:
        brain.logger.info("🛑 프로그램 종료")
