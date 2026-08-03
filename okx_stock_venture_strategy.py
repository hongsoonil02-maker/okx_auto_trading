#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
okx_stock_venture_strategy.py — OKX Stock Token Dedicated Auto Trading Engine (VENTURE/MEME)
- Trades OKX US Stock Tokens (SPACEX, OPENAI, RDDT, etc.)
- Strict US Market time filtering
- Supertrend + StochRSI logic (15m timeframe)
(Refactored to inherit from strategy_common.BaseStrategyBrain)
"""
import asyncio
import sys
from strategy_common import BaseStrategyBrain


class OKXStockVentureStrategyBrain(BaseStrategyBrain):
    STRATEGY_NAME = "OKX_Stock_Venture"
    LOG_FILE = "okx_stock_venture_strategy.log"
    LOGGER_NAME = "OKX_Stock_Venture_Strategy"
    SUPERTREND_MULT_TIGHT = 2.5
    PROFIT_THRESHOLD = 1.02
    VOL_CONFIRM_MULT = 1.1        # [개선안 #1] 1.2→1.1 (약간 완화)
    MIN_HOLD_CANDLES = 3          # [개선안 #3] Breakeven 최소 3캔들 보유
    STOCK_KEYWORDS = ['SPACEX', 'OPENAI', 'ANTHROPIC', 'RDDT', 'XSTOCKS']
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
    brain = OKXStockVentureStrategyBrain()
    try:
        asyncio.run(brain.run_all())
    except KeyboardInterrupt:
        brain.logger.info("🛑 프로그램 종료")
