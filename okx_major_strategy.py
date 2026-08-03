#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
okx_major_strategy.py — OKX Major Crypto Auto Trading Engine
- Trades Top 10 Major Cryptocurrencies
- Supertrend + StochRSI logic (1h timeframe)
- Replaces the old shadow mirror copy trading bot
"""
import asyncio
import sys
from strategy_common import BaseStrategyBrain

class OKXMajorStrategyBrain(BaseStrategyBrain):
    STRATEGY_NAME = "OKX_Major_Strategy"
    LOG_FILE = "okx_major_strategy.log"
    LOGGER_NAME = "OKX_Major_Strategy"
    
    # Use 1h timeframe as requested
    TIMEFRAME = "1h"
    TIMEFRAME_MINUTES = 60
    SCALE_OUT_EXITS = False
    
    # Use tighter settings for major coins, similar to stock majors
    SUPERTREND_MULT_TIGHT = 2.5
    PROFIT_THRESHOLD = 1.01
    VOL_CONFIRM_MULT = 1.2
    MIN_HOLD_CANDLES = 3
    EMA_PERIOD = 50
    
    # Top 10 Major Coins
    MAJOR_COINS = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'AVAX', 'LINK', 'DOT', 'BNB', 'TRX']

    def _symbol_matches(self, symbol: str, ticker_data: dict, markets: dict) -> bool:
        if symbol not in markets or not markets[symbol].get('swap'):
            return False
        if 'USDT' not in symbol:
            return False
        
        # Check if base coin is in MAJOR_COINS (e.g. BTC-USDT-SWAP -> BTC)
        base_coin = symbol.split('-')[0]
        if base_coin not in self.MAJOR_COINS:
            return False
            
        return True

if __name__ == "__main__":
    brain = OKXMajorStrategyBrain()
    try:
        asyncio.run(brain.run_all())
    except KeyboardInterrupt:
        brain.logger.info("🛑 프로그램 종료")
