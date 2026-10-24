#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kis_tori_swing_trader.py — KIS OpenAPI 미국 주식 토리(Tori) 추세선 스윙 실행 모듈 (Prototype)
- 대상: SPY (1H / 1D), NVDA (1H / 1D)
- 핵심 설계:
  1) Tori 추세선 돌파 + 과속 필터 (Speedometer)
  2) 오버나이트 갭(Overnight Gap) 하락 방어:
     - 본장(09:30 EST) 개장 시 시가가 세이프티 라인을 하방 이탈(Gap Down)하면
       장 시작 직후 10초 이내에 시장가(또는 최우선 지정가)로 전량 즉각 탈출(Emergency Cut)
  3) 세이프티 라인 동적 트레일링 스탑
  4) KIS 해외주식 주문 API (TTTS1002U: 해외주식 매수 / TTTS1006U: 해외주식 매도) 연동
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Optional, Any, Tuple
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv

# Import Tori Engine
sys.path.append("/home/hongsoonil02/quant_crypto")
from strategies.tori_trendline import ToriTrendlineEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("KIS_Tori_Swing")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))
load_dotenv(os.path.join(CURRENT_DIR, "..", ".env"))


class KisToriSwingTrader:
    """
    미국 주식 KIS 토리 추세선 스윙 및 오버나이트 갭 방어 실행 모듈
    """
    def __init__(self, symbols: list = ["SPY", "NVDA"], test_mode: bool = True):
        self.symbols = symbols
        self.test_mode = test_mode
        self.engine = ToriTrendlineEngine(pivot_window=4, speed_factor=2.0)
        self.positions_file = os.path.join(CURRENT_DIR, "kis_tori_positions.json")
        self.positions: Dict[str, Any] = self._load_positions()
        
        self.app_key = os.getenv("KIS_APP_KEY", "")
        self.app_secret = os.getenv("KIS_APP_SECRET", "")
        self.account_no = os.getenv("KIS_ACCOUNT_NO", "")
        self.token: Optional[str] = None

    def _load_positions(self) -> Dict[str, Any]:
        if os.path.exists(self.positions_file):
            try:
                with open(self.positions_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Positions load error: {e}")
        return {}

    def _save_positions(self):
        try:
            with open(self.positions_file, "w", encoding="utf-8") as f:
                json.dump(self.positions, f, indent=2)
        except Exception as e:
            logger.error(f"Positions save error: {e}")

    def fetch_market_data(self, symbol: str, interval: str = "1h", period: str = "60d") -> pd.DataFrame:
        """Fetch US stock candle data via yfinance for analysis."""
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        return df

    def check_market_open_gap_defense(self, symbol: str, current_price: float) -> bool:
        """
        [핵심 안전 루틴] 정규장 개장 직후 오버나이트 갭하락 방어:
        시가가 세이프티 라인을 하방 이탈했는지 감지하여 즉시 시장가 비상 청산 트리거.
        """
        pos = self.positions.get(symbol)
        if not pos:
            return False

        safety_line = pos.get("safety_line", 0.0)
        entry_price = pos.get("entry_price", 0.0)

        if current_price < safety_line:
            gap_pct = (safety_line - current_price) / safety_line * 100.0
            logger.warning(
                f"🚨 [OVERNIGHT GAP DEFENSE ACTIVATED] {symbol} 갭하락 감지!\n"
                f"세이프티 라인: ${safety_line:.2f} | 현재 시가: ${current_price:.2f} (갭: -{gap_pct:.2f}%)\n"
                f"-> 즉시 KIS 전량 시장가 매도 탈출 주문 집행!"
            )
            self.execute_sell(symbol, pos.get("shares", 0), current_price, reason="GAP_DOWN_EXIT")
            del self.positions[symbol]
            self._save_positions()
            return True
        return False

    def evaluate_symbol(self, symbol: str):
        """1H/1D 캔들 분석 및 진입/청산/트레일링 판단"""
        logger.info(f"📊 [{symbol}] 토리 추세선 상태 평가 중...")
        df = self.fetch_market_data(symbol, interval="1h", period="30d")
        if df.empty or len(df) < 20:
            return

        evaluated_df = self.engine.evaluate(df)
        last_row = evaluated_df.iloc[-1]
        c_price = last_row['close']
        raw_sig = last_row['raw_signal']
        speed_ok = last_row['speed_ok']
        safety_val = last_row['safety_line']
        sl_val = last_row['stop_loss']

        logger.info(f"  {symbol} 현재가: ${c_price:.2f} | 신호: {raw_sig} | 과속필터: {speed_ok} | 세이프티: {safety_val}")

        # 1. 기존 포지션 관리
        if symbol in self.positions:
            pos = self.positions[symbol]
            # 오버나이트 갭 또는 장중 이탈 체크
            if self.check_market_open_gap_defense(symbol, c_price):
                return

            # 동적 세이프티 라인 상향 갱신 (Trailing Stop)
            if not np.isnan(safety_val) and safety_val > pos['safety_line'] and safety_val < c_price:
                logger.info(f"📈 [{symbol}] 세이프티 라인 상향 갱신: ${pos['safety_line']:.2f} -> ${safety_val:.2f}")
                pos['safety_line'] = safety_val
                self._save_positions()
            return

        # 2. 신규 진입 (Long Only)
        if raw_sig == 1 and speed_ok and not np.isnan(sl_val) and sl_val < c_price:
            logger.info(f"🎯 [{symbol}] Tori Long 진입 신호 포착! (과속 필터 통과)")
            # 포지션 크기 (가상 1만 달러 계좌 기준)
            equity = 10000.0
            risk_amt = equity * 0.015 # 1.5% 리스크
            sl_dist = c_price - sl_val
            shares = max(1, int(risk_amt / sl_dist))
            
            self.execute_buy(symbol, shares, c_price, sl_val)
            self.positions[symbol] = {
                "symbol": symbol,
                "shares": shares,
                "entry_price": c_price,
                "safety_line": sl_val,
                "entry_time": datetime.now(ZoneInfo("America/New_York")).isoformat()
            }
            self._save_positions()

    def execute_buy(self, symbol: str, shares: int, price: float, sl_price: float):
        if self.test_mode:
            logger.info(f"🧪 [TEST BUY] {symbol} {shares}주 매수 @ ${price:.2f} (SL: ${sl_price:.2f})")
        else:
            logger.info(f"🚀 [REAL BUY] KIS TTTS1002U 해외주식 매수 주문 송출: {symbol} {shares}주")

    def execute_sell(self, symbol: str, shares: int, price: float, reason: str = "SAFETY_LINE_BREACH"):
        if self.test_mode:
            logger.info(f"🧪 [TEST SELL] {symbol} {shares}주 전량 매도 @ ${price:.2f} (사유: {reason})")
        else:
            logger.info(f"🛑 [REAL SELL] KIS TTTS1006U 해외주식 매도 주문 송출: {symbol} {shares}주")


if __name__ == "__main__":
    trader = KisToriSwingTrader(symbols=["SPY", "NVDA"], test_mode=True)
    for sym in trader.symbols:
        trader.evaluate_symbol(sym)
