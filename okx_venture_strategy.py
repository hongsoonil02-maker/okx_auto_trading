#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
okx_venture_strategy.py — OKX Crypto Auto Trading Engine (VENTURE/MEME)
- Trades highly volatile Meme/Alt coins (15m timeframe)
- Updated to Vibe-Trading AI (Geumgang Mirror Logic - Mean Reversion BB + DCA)
"""
import asyncio
import sys
from strategy_common import BaseStrategyBrain

class OKXVentureStrategyBrain(BaseStrategyBrain):
    STRATEGY_NAME = "OKX_Venture"
    LOG_FILE = "okx_venture_strategy.log"
    LOGGER_NAME = "OKX_Venture_Strategy"
    
    # [포트폴리오 & 리스크 튜닝]
    PORTFOLIO_WEIGHT = 1.5       # 자본 집중 배분
    HARD_STOP_LOSS_PCT = -0.15   # 꼬리 위험 방어 (15% 컷)
    PYRAMID_RATIO = 0.50         # 추세 승자 적극 불타기
    
    STOCK_KEYWORDS = []
    BLACKLIST = ['KR200', 'SKHYNIX', 'MU', 'SHAZ', 'ISRG', 'ROBO', 'RAM', 'DRAM', 'GME']
    # 주식/상품 토큰 전부 제외 (주식 전용 봇과 중복 거래 방지)
    MAJORS_AND_STOCKS = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'AVAX', 'LINK', 'DOT', 'BNB', 'TRX',
                         'TSLA', 'NVDA', 'AAPL', 'AMZN', 'MSFT', 'META', 'GOOG', 'GOOGL', 'COIN',
                         'SPCX', 'OPENAI', 'ANTHROPIC', 'RDDT', 'MU', 'SNDK', 'SOXL', 'XAU', 'CL',
                         # [Fix] 주식/ETF 토큰 추가 — 크립토와 다른 가격 패턴, 숏 구조적 불리
                         'SKHY', 'KORU', 'CBRS', 'AEON',
                         'PLTR', 'AMD', 'INTC', 'QCOM', 'BABA', 'UBER', 'ABNB', 'SNAP',
                         'MSTR', 'HOOD', 'RIVN', 'NIO', 'PYPL', 'SQ', 'SHOP',
                         'SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'SLV', 'XAG',
                         ]
    # [백테스트 검증] Venture 15m에서 재진입 PF 1.15→1.20 (180일, N=231)
    REENTRY_ENABLED = True

    def _symbol_matches(self, symbol: str, ticker_data: dict, markets: dict) -> bool:
        if symbol not in markets or not markets[symbol].get('swap'):
            return False
        if 'USDT' not in symbol:
            return False
        # [Fix] ccxt 심볼은 'BTC/USDT:USDT' 형태라 '-' split만으론 base 추출 불가
        # → '/' split 추가 (기존엔 제외 목록이 작동하지 않아 ETH/주식토큰도 거래했음)
        base_coin = symbol.split('-')[0].split('/')[0]
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
