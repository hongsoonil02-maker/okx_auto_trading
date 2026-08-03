#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
okx_venture_strategy.py — OKX Crypto Auto Trading Engine (VENTURE/MEME)
- Trades highly volatile Meme/Alt coins (15m timeframe)
- Supertrend + StochRSI logic
(Refactored to inherit from strategy_common.BaseStrategyBrain)
"""
import asyncio
import sys
from strategy_common import BaseStrategyBrain


class OKXVentureStrategyBrain(BaseStrategyBrain):
    STRATEGY_NAME = "OKX_Venture"
    LOG_FILE = "okx_venture_strategy.log"
    LOGGER_NAME = "OKX_Venture_Strategy"
    SUPERTREND_MULT_TIGHT = 2.0   # 롤백: 2.5→2.0 (백테스트 결과 WIF 수익 악화 확인)
    PROFIT_THRESHOLD = 1.025
    VOL_CONFIRM_MULT = 1.0        # [개선안 #1] 1.2→1.0 (급등장 진입 지연 방지)
    MIN_HOLD_CANDLES = 3          # [개선안 #3] Breakeven 최소 3캔들 보유
    STOCK_KEYWORDS = []
    BLACKLIST = ['KR200', 'SKHYNIX']
    MAJORS_AND_STOCKS = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'AVAX', 'LINK', 'DOT', 'BNB', 'TRX', 'TSLA', 'NVDA', 'AAPL', 'AMZN', 'MSFT', 'META', 'GOOG', 'COIN']

    def _symbol_matches(self, symbol: str, ticker_data: dict, markets: dict) -> bool:
        if symbol not in markets or not markets[symbol].get('swap'):
            return False
        if 'USDT' not in symbol:
            return False
        base_coin = symbol.split('-')[0]
        if base_coin in self.MAJORS_AND_STOCKS:
            return False
        if any(b in symbol for b in self.BLACKLIST):
            return False
        return True


if __name__ == "__main__":
    brain = OKXVentureStrategyBrain()
    try:
        asyncio.run(brain.run_all())
    except KeyboardInterrupt:
        brain.logger.info("🛑 프로그램 종료")
