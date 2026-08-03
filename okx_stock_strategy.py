#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
okx_stock_strategy.py — OKX Stock Token Auto Trading Engine (MAJORS)
- Trades major US stock tokens (TSLA, AAPL, etc.)
"""
import asyncio
import sys
import pytz
from datetime import datetime
from strategy_common import BaseStrategyBrain

class OKXStockStrategyBrain(BaseStrategyBrain):
    STRATEGY_NAME = "OKX_Stock_Majors"
    LOG_FILE = "okx_stock_strategy.log"
    LOGGER_NAME = "OKX_Stock_Strategy"
    SUPERTREND_MULT_TIGHT = 2.5
    PROFIT_THRESHOLD = 1.01
    VOL_CONFIRM_MULT = 1.2
    MIN_HOLD_CANDLES = 3
    EMA_PERIOD = 200
    TIMEFRAME = "15m"
    TIMEFRAME_MINUTES = 15
    SCALE_OUT_EXITS = False
    STOCK_KEYWORDS = ['TSLA', 'NVDA', 'AAPL', 'AMZN', 'MSFT', 'META', 'GOOG']
    BLACKLIST = ['KR200', 'SKHYNIX']


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
