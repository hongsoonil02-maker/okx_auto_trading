#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
okx_stock_venture_strategy.py — OKX Stock Token Dedicated Auto Trading Engine (VENTURE/MEME)
- Trades OKX US Stock Tokens (SPACEX, OPENAI, RDDT, etc.)
- Geumgang Mirror Logic (Mean Reversion BB + DCA) applied
"""
import asyncio
import sys
from strategy_common import BaseStrategyBrain

class OKXStockVentureStrategyBrain(BaseStrategyBrain):
    STRATEGY_NAME = "OKX_Stock_Venture"
    LOG_FILE = "okx_stock_venture_strategy.log"
    LOGGER_NAME = "OKX_Stock_Venture_Strategy"
    
    # OKX 실제 상장 심볼 기준 (SPACEX→SPCX, XSTOCKS→개별 종목 코드)
    STOCK_KEYWORDS = ['SPCX', 'OPENAI', 'ANTHROPIC', 'RDDT', 'MU', 'SNDK', 'SOXL', 'XAU', 'CL']
    BLACKLIST = ['KR200', 'SKHYNIX', 'SKHY']
    # 벤처 주식 토큰은 거래대금이 낮으므로 완화 (RDDT 포함 기준)
    MIN_QUOTE_VOLUME = 50000

    def _symbol_matches(self, symbol: str, ticker_data: dict, markets: dict) -> bool:
        if symbol not in markets or not markets[symbol].get('swap'):
            return False
        if 'USDT' not in symbol:
            return False
        # [Fix] 짧은 키워드(MU, CL, RE 등)는 부분문자열 매칭 시 MUBARAK 같은
        # 밈코인이 섞이므로 기준통화 정확 일치로 비교
        base_coin = symbol.split('-')[0].split('/')[0]
        if base_coin not in self.STOCK_KEYWORDS:
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
