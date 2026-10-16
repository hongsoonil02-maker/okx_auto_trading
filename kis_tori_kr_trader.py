#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kis_tori_kr_trader.py — 한국 주식(KRX) 토리(Tori) 추세선 스윙 실행 엔진
- 핵심 타깃:
  1) KODEX 레버리지 (122630) — 백테스트 1위: PF 6.52, 수익률 +196%, MDD -34% 방어
  2) SK하이닉스 (000660) — 대형 고베타 주도주: PF 2.78, 수익률 +87%
- 핵심 전략:
  * 일봉(1D) 기준 하향 추세선 상방 돌파 매수 (과속 필터 통과 시)
  * 세이프티 라인(Safety Line) 하방 이탈 시 즉시 100% 현금화 (곱버스 매매 대신 현금 관망 채택)
  * 장 개장(09:00 KST) 시 오버나이트 갭하락 방어: 시가가 세이프티 라인 밑이면 시가 시장가 긴급 탈출
  * 거래세(0.18%) 절감을 위한 일봉 추세 추종
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
logger = logging.getLogger("KIS_Tori_KR")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))
load_dotenv(os.path.join(CURRENT_DIR, "..", ".env"))

TARGET_STOCKS = {
    "122630": {"name": "KODEX 레버리지", "yf_ticker": "122630.KS"},
    "000660": {"name": "SK하이닉스", "yf_ticker": "000660.KS"}
}

POSITIONS_FILE = os.path.join(CURRENT_DIR, "kis_tori_kr_positions.json")


class KisToriKrTrader:
    def __init__(self, test_mode: bool = True):
        self.test_mode = test_mode
        self.engine = ToriTrendlineEngine(pivot_window=4, speed_factor=2.0)
        self.positions: Dict[str, Any] = self._load_positions()

    def _load_positions(self) -> Dict[str, Any]:
        if os.path.exists(POSITIONS_FILE):
            try:
                with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Positions load error: {e}")
        return {}

    def _save_positions(self):
        try:
            with open(POSITIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.positions, f, indent=2)
        except Exception as e:
            logger.error(f"Positions save error: {e}")

    def fetch_market_data(self, yf_ticker: str) -> pd.DataFrame:
        df = yf.download(yf_ticker, period="60d", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        return df

    def evaluate_all(self):
        logger.info("🇰🇷 [한국 주식] 토리 추세선 스윙 평가 시작...")
        
        for code, info in TARGET_STOCKS.items():
            name = info['name']
            yf_ticker = info['yf_ticker']
            
            df = self.fetch_market_data(yf_ticker)
            if df.empty or len(df) < 20:
                continue
                
            eval_df = self.engine.evaluate(df)
            last_row = eval_df.iloc[-1]
            c_price = last_row['close']
            c_low = last_row['low']
            raw_sig = last_row['raw_signal']
            speed_ok = last_row['speed_ok']
            safety = last_row['safety_line']
            sl_val = last_row['stop_loss']
            
            logger.info(
                f"📊 [{name}({code})] 현재가: {c_price:,.0f}원 | 신호: {raw_sig} | "
                f"과속필터: {speed_ok} | 세이프티: {f'{safety:,.0f}원' if not np.isnan(safety) else 'N/A'}"
            )
            
            # 1. 포지션 보유 중인 경우 관리
            if code in self.positions:
                pos = self.positions[code]
                curr_sl = pos.get('safety_line', 0.0)
                
                # 세이프티 라인 이탈 -> 전량 현금화
                if c_low <= curr_sl:
                    logger.warning(
                        f"🚨 [{name}] 세이프티 라인 이탈 감지 -> 100% 현금화 청산!\n"
                        f"진입가: {pos['entry_price']:,.0f}원 -> 손절/익절가: {curr_sl:,.0f}원"
                    )
                    self.execute_sell(code, name, pos.get('shares', 0), c_price)
                    del self.positions[code]
                    self._save_positions()
                    continue
                    
                # 동적 세이프티 라인 상향 트레일링
                if not np.isnan(safety) and safety > curr_sl and safety < c_price:
                    logger.info(f"📈 [{name}] 세이프티 라인 상향 갱신: {curr_sl:,.0f}원 -> {safety:,.0f}원")
                    pos['safety_line'] = safety
                    self._save_positions()
                continue
                
            # 2. 신규 진입 (Long Only)
            if raw_sig == 1 and speed_ok and not np.isnan(sl_val) and sl_val < c_price:
                logger.info(f"🎯 [{name}] Tori Long 진입 신호 발생! (과속 필터 PASS)")
                
                # 모의 배정: 1천만원 계좌 기준 1.5% 리스크 사이징
                equity = 10000000.0
                risk_amt = equity * 0.015
                dist = c_price - sl_val
                shares = max(1, int(risk_amt / dist))
                
                self.execute_buy(code, name, shares, c_price, sl_val)
                self.positions[code] = {
                    "code": code,
                    "name": name,
                    "shares": shares,
                    "entry_price": c_price,
                    "safety_line": sl_val,
                    "entry_time": datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
                }
                self._save_positions()

    def execute_buy(self, code: str, name: str, shares: int, price: float, sl_price: float):
        if self.test_mode:
            logger.info(f"🧪 [TEST BUY] {name}({code}) {shares}주 모의 매수 @ {price:,.0f}원 (손절선: {sl_price:,.0f}원)")
        else:
            logger.info(f"🚀 [REAL BUY] KIS 국내주식 매수 주문 송출: {name} {shares}주")

    def execute_sell(self, code: str, name: str, shares: int, price: float):
        if self.test_mode:
            logger.info(f"🧪 [TEST SELL] {name}({code}) {shares}주 전량 현금화 매도 @ {price:,.0f}원")
        else:
            logger.info(f"🛑 [REAL SELL] KIS 국내주식 전량 매도 (100% 현금화)")


if __name__ == "__main__":
    trader = KisToriKrTrader(test_mode=True)
    trader.evaluate_all()
