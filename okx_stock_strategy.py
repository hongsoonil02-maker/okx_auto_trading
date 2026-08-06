#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
okx_stock_strategy.py — OKX Stock Token Auto Trading Engine (MAJORS)
- Trades major US stock tokens (TSLA, AAPL, etc.)
- Geumgang Mirror Logic (Mean Reversion BB + DCA) applied
"""
import asyncio
import sys
import pandas as pd
import numpy as np
from strategy_common import BaseStrategyBrain

class OKXStockStrategyBrain(BaseStrategyBrain):
    STRATEGY_NAME = "OKX_Stock_Majors"
    LOG_FILE = "okx_stock_strategy.log"
    LOGGER_NAME = "OKX_Stock_Strategy"
    
    # [Vibe-Trading AI] 금강대제 로직 세팅
    TIMEFRAME = "15m"
    TIMEFRAME_MINUTES = 15
    SCALE_OUT_EXITS = False
    STRATEGY_LEVERAGE = 50  # 50x 레버리지 강제 오버라이드
    
    # 8회 DCA 물타기 허용 (Vibe-Trading AI)
    MAX_DCA_ENTRIES = 8
    
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

    # 부모 클래스의 check_auto_logic을 완전히 오버라이드하여 독자적인 Vibe-Trading 로직 적용
    async def check_auto_logic(self, symbol):
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, self.TIMEFRAME, limit=300)
            if not ohlcv or len(ohlcv) < 200:
                return
            df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])

            # 지표 계산: 볼린저 밴드(20, 2.5)
            bb_period = 20
            bb_std = 2.5
            df['sma20'] = df['c'].rolling(bb_period).mean()
            df['std'] = df['c'].rolling(bb_period).std()
            df['upper_bb'] = df['sma20'] + (df['std'] * bb_std)
            df['lower_bb'] = df['sma20'] - (df['std'] * bb_std)
            
            # 지표 계산: RSI(14)
            rsi_period = 14
            delta = df['c'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
            rs = gain / loss.replace(0, np.nan)
            df['rsi'] = 100 - (100 / (1 + rs))

            curr = df.iloc[-1]
            
            dca = self.dca_state.setdefault(symbol, {'entry_count': 0, 'exit_count': 0, 'last_entry_t': 0, 'last_exit_t': 0, 'first_entry_t': 0, 'max_pnl_pct': 0.0})
            
            pos_long = self.auto_active_pos.get((symbol, 'long'))
            has_long = pos_long is not None
            avg_price_long = pos_long['avgPrice'] if has_long else 0
            
            pos_short = self.auto_active_pos.get((symbol, 'short'))
            has_short = pos_short is not None
            avg_price_short = pos_short['avgPrice'] if has_short else 0

            # 시그널 판단 (역추세: BB 상단돌파+RSI과매수 -> 숏 / BB 하단이탈+RSI과매도 -> 롱)
            is_long_sig = curr['c'] < curr['lower_bb'] and curr['rsi'] < 30
            is_short_sig = curr['c'] > curr['upper_bb'] and curr['rsi'] > 70
            
            tp_pct = 0.005 # 0.5% 가격이동
            dca_pct = 0.005 # 0.5% 가격이동 시 물타기

            if has_long:
                ep = avg_price_long
                is_pft = curr['c'] >= ep * (1 + tp_pct)
                is_mean_rev = curr['c'] >= curr['sma20']
                
                if is_pft or is_mean_rev:
                    self.logger.info(f"[{symbol}] [금강대제 주식 로직] 롱 익절/본절 조건 충족 (Mean Reversion)")
                    await self._send_signal(symbol, "close_long", size_multiplier=1.0, leverage=self.STRATEGY_LEVERAGE)
                elif curr['c'] <= ep * (1 - dca_pct) and dca['entry_count'] < self.MAX_DCA_ENTRIES:
                    self.logger.info(f"[{symbol}] [금강대제 주식 로직] 롱 DCA 물타기 조건 충족")
                    await self._send_signal(symbol, "buy_long", size_multiplier=1.0/self.MAX_DCA_ENTRIES, leverage=self.STRATEGY_LEVERAGE)
            else:
                if is_long_sig:
                    self.logger.info(f"[{symbol}] [금강대제 주식 로직] 신규 롱 진입 (RSI 과매도 + BB 하단 이탈)")
                    await self._send_signal(symbol, "buy_long", size_multiplier=1.0/self.MAX_DCA_ENTRIES, leverage=self.STRATEGY_LEVERAGE)

            if has_short:
                ep = avg_price_short
                is_pft = curr['c'] <= ep * (1 - tp_pct)
                is_mean_rev = curr['c'] <= curr['sma20']
                
                if is_pft or is_mean_rev:
                    self.logger.info(f"[{symbol}] [금강대제 주식 로직] 숏 익절/본절 조건 충족 (Mean Reversion)")
                    await self._send_signal(symbol, "close_short", size_multiplier=1.0, leverage=self.STRATEGY_LEVERAGE)
                elif curr['c'] >= ep * (1 + dca_pct) and dca['entry_count'] < self.MAX_DCA_ENTRIES:
                    self.logger.info(f"[{symbol}] [금강대제 주식 로직] 숏 DCA 물타기 조건 충족")
                    await self._send_signal(symbol, "sell_short", size_multiplier=1.0/self.MAX_DCA_ENTRIES, leverage=self.STRATEGY_LEVERAGE)
            else:
                if is_short_sig:
                    self.logger.info(f"[{symbol}] [금강대제 주식 로직] 신규 숏 진입 (RSI 과매수 + BB 상단 돌파)")
                    await self._send_signal(symbol, "sell_short", size_multiplier=1.0/self.MAX_DCA_ENTRIES, leverage=self.STRATEGY_LEVERAGE)

        except Exception as e:
            self.logger.error(f"⚠️ [{symbol}] 자동 매매 검사 실패: {e}")

if __name__ == "__main__":
    brain = OKXStockStrategyBrain()
    try:
        asyncio.run(brain.run_all())
    except KeyboardInterrupt:
        brain.logger.info("🛑 프로그램 종료")
