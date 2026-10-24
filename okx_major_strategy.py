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
    
    # [포트폴리오 & 리스크 튜닝] BTC, ETH, SOL 3대 메이저 정예화
    PORTFOLIO_WEIGHT = 0.50      # 메이저 3종에 자금 50% 배정
    HARD_STOP_LOSS_PCT = float(os.getenv("OKX_HARD_STOP_LOSS", "-0.30"))

    # [토너먼트 1위 반영] Chandelier_Max_Runner: ARM=0.35, K=3.5 (백테스트 1위: +35.3%, PF 3.92)
    ATR_TRAILING_ENABLED = True
    ATR_TRAIL_ARM_PNL = 0.35

    TIMEFRAME = "1h"
    TIMEFRAME_MINUTES = 60
    CHOP_ADX_BLOCK_THRESHOLD = 20.0  # ADX < 20 횡보장 휩쏘 진입 차단

    # [사용자 지침] 메이저 3대 대장주 엄선 (BTC, ETH, SOL)
    MAJOR_COINS = ['BTC', 'ETH', 'SOL']

    # [수익성 개선] 트레일링청산 플립 활성화: Major 30m에서 PF 0.76→1.22 (흑자 전환)
    # Venture에서는 악화(1.23→1.13)하므로 Major에만 활성화
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