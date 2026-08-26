#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
okx_major_strategy.py — OKX Major Crypto Auto Trading Engine
- Trades Top 10 Major Cryptocurrencies
- Supertrend + StochRSI logic (inherited from BaseStrategyBrain)
- 1h timeframe, leverage controlled by OKX_LEVERAGE env var
"""
import asyncio
import sys
import os
from strategy_common import BaseStrategyBrain


class OKXMajorStrategyBrain(BaseStrategyBrain):
    STRATEGY_NAME = "OKX_Major_Strategy"
    LOG_FILE = "okx_major_strategy.log"
    LOGGER_NAME = "OKX_Major_Strategy"
    STRATEGY_LEVERAGE = int(os.getenv("OKX_MAJOR_LEVERAGE", "10"))
    
    # [포트폴리오 & 리스크 튜닝]
    PORTFOLIO_WEIGHT = 0.5       # 자본 분산 배분 (보수적)


    TIMEFRAME = "30m"       # 신호 빈도 확대: 1h → 30m
    TIMEFRAME_MINUTES = 30

    MAJOR_COINS = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'AVAX', 'LINK', 'DOT', 'BNB', 'TRX']

    # [백테스트 검증] Major 30m 트레일링청산 플립: PF 0.76→1.22 (-2075%→+1559%)
    FLIP_ON_TRAILING_CLOSE = True

    def _symbol_matches(self, symbol: str, ticker_data: dict, markets: dict) -> bool:
        if symbol not in markets or not markets[symbol].get('swap'):
            return False
        if 'USDT' not in symbol:
            return False
        base_coin = symbol.split('-')[0].split('/')[0]
        if base_coin not in self.MAJOR_COINS:
            return False
        return True


if __name__ == "__main__":
    brain = OKXMajorStrategyBrain()
    try:
        asyncio.run(brain.run_all())
    except KeyboardInterrupt:
        brain.logger.info("🛑 프로그램 종료")