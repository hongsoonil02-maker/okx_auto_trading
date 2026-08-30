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
    
    # [포트폴리오 & 리스크 튜닝 - alv*** 벤치마킹]
    PORTFOLIO_WEIGHT = 0.25      # 자본 분산 배분 (실질 레버리지 2.5배 이하로 엄격 통제)
    # [alv*** 칼손절] 마진 기준 -8% 즉시 손절 (손익비 2.3:1 극대화, 하락장 계좌 반토막 원천 차단)
    HARD_STOP_LOSS_PCT = -0.08   # 마진 기준 -8% 하드 스탑
    SOFT_STOP_LOSS_PCT = -0.06   # 마진 기준 -6% 조기 손절

    # [alv*** 빅스윙 트레일링] 조기 익절 상한을 없애고 샹들리에 ATR 트레일링으로 대세 추세 홀딩
    ATR_TRAILING_ENABLED = True
    ATR_TRAIL_K = 2.5            # 최고점 - 2.5 * ATR 이탈 시에만 청산
    ATR_TRAIL_ARM_PNL = 0.20     # +20% 수익 도달 시 트레일링 자동 가동


    # [옵션 A 검증 적용] 메이저 15m + ADX>=20 횡보 필터
    # 백테스트 성과: 수익률 +373.9%, 승률 63.0%, PF 2.68, MDD 30.1%
    TIMEFRAME = "15m"
    TIMEFRAME_MINUTES = 15
    CHOP_ADX_BLOCK_THRESHOLD = 20.0  # ADX < 20 횡보장 휩쏘 진입 차단

    MAJOR_COINS = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'AVAX', 'LINK', 'DOT', 'BNB', 'TRX']

    # [백테스트 검증] Major 트레일링청산 플립
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