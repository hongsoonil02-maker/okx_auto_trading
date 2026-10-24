#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
okx_venture_strategy.py — OKX Crypto Auto Trading Engine (VENTURE/MEME)
- Trades highly volatile Meme/Alt coins (15m timeframe)
- Updated to Vibe-Trading AI (Geumgang Mirror Logic - Mean Reversion BB + DCA)
"""
import asyncio
import sys
import os
from strategy_common import BaseStrategyBrain

class OKXVentureStrategyBrain(BaseStrategyBrain):
    STRATEGY_NAME = "OKX_Venture"
    LOG_FILE = "okx_venture_strategy.log"
    LOGGER_NAME = "OKX_Venture_Strategy"
    STRATEGY_LEVERAGE = int(os.getenv("OKX_VENTURE_LEVERAGE", "5"))
    
    # [포트폴리오 & 리스크 튜닝 - alv*** 타이트 손절 원칙]
    PORTFOLIO_WEIGHT = 1.0       # 자본 분산 균형 배분
    # [Fix] 마진 -10%/-7% 고정 스탑 제거 (3x에서 가격 -2.3% = 알트 30m 1 ATR 안 → 실거래 알트 28건 승률 7%).
    # 스탑은 BaseStrategyBrain ATR 스탑(OKX_ATR_STOP_K) 사용, HARD_STOP은 env 최후 방어선만 유지.
    HARD_STOP_LOSS_PCT = float(os.getenv("OKX_HARD_STOP_LOSS", "-0.30"))
    # [Venture V3 업데이트 - User Directive] Jev AI의 초고속 판단력을 활용한 양방향 초단타 스캘핑 모드
    ATR_TRAIL_ARM_PNL = 0.10   # 10% 수익 발생 시 트레일링 즉시 발동 (스캘핑 특화)
    ATR_TRAIL_K = 1.5          # 고점 대비 ATR 1.5배 하락 시 칼익절 (매우 타이트하게)
    # [개선] 15m 알트 잦은 턴오버 및 양방향 휩소 방지를 위해 플립 비활성화
    FLIP_ON_TRAILING_CLOSE = False
    
    # [Venture V3 업데이트] 15m 초단타
    TIMEFRAME = "15m"
    TIMEFRAME_MINUTES = 15
    # [개선] 횡보장(ADX<22) 노이즈 진입 및 수수료 누수 차단 (기존 15.0 -> 22.0 상향)
    CHOP_ADX_BLOCK_THRESHOLD = 22.0
    
    # [사용자 지침] 유동성 최상위 엄선 5대 알트코인 정예 유니버스 (잡코인 배제)
    TARGET_ALTS = ['DOGE', 'SUI', 'PEPE', 'NEAR', 'AVAX']
    REENTRY_ENABLED = True

    def _symbol_matches(self, symbol: str, ticker_data: dict, markets: dict) -> bool:
        if symbol not in markets or not markets[symbol].get('swap'):
            return False
        if 'USDT' not in symbol:
            return False
        base_coin = symbol.split('-')[0].split('/')[0]
        return base_coin in self.TARGET_ALTS



if __name__ == "__main__":
    brain = OKXVentureStrategyBrain()
    try:
        asyncio.run(brain.run_all())
    except KeyboardInterrupt:
        brain.logger.info("🛑 프로그램 종료")
