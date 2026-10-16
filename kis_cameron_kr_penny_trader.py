# -*- coding: utf-8 -*-
"""
kis_cameron_kr_penny_trader.py — 한국 주식(KRX) 동전주/소형 테마주 30분봉 1차 눌림목(First Pullback) 실행 엔진
Alpaca / KIS 미국주식 개선 로직 100% 동일 동기화:
- Timeframe: 30분봉 (30m)
- Setup:
  1) 1차 급등 파동 (+3.0% 이상 서지)
  2) 30분봉 눌림목 (음봉 또는 몸통 45% 이하 도지/팽이 캔들)
  3) VWAP & 9 EMA 2~3% 유연 버퍼 지지 확인
  4) 되돌림 폭 60% 이내 & 거래량 축소 (평균 거래량 1.35배 이하)
  5) 직전봉 고점 돌파 시 매수 발주
- Position Sizing & Risk Management:
  * 동적 Half-Kelly 공식 적용 (16% ~ 20%)
  * 최대 동시 보유 5종목 (가동률 100% 완전 활용)
  * 통합증거금 기준 1,487만 원 구매력 완전 연동
  * 청산 후 15분 재진입 쿨다운 (COOLDOWN_SEC = 900)
"""

import os
import sys
import time
import math
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple, Any
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

STATE_DIR = os.path.join(CURRENT_DIR, "state")
os.makedirs(STATE_DIR, exist_ok=True)
POSITIONS_FILE = os.path.join(STATE_DIR, "kis_cameron_penny_positions.json")
LOG_FILE = os.path.join(CURRENT_DIR, "kis_cameron_penny.log")

# ── LOGGING SETUP ──
logger = logging.getLogger("KIS_Cameron_Penny")
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

# ── WATCHLIST: 동전주 & 급등 테마주 ──
PENNY_TARGETS = {
    "004830": {"name": "덕성", "yf_ticker": "004830.KS"},
    "250060": {"name": "모비스", "yf_ticker": "250060.KQ"},
    "002780": {"name": "진흥기업", "yf_ticker": "002780.KS"},
    "104040": {"name": "대성파인텍", "yf_ticker": "104040.KQ"},
    "051780": {"name": "큐로홀딩스", "yf_ticker": "051780.KQ"},
    "900110": {"name": "이스트아시아", "yf_ticker": "900110.KQ"}
}

TIMEFRAME = "30m"             # 30분봉 고회전 모멘텀
RISK_REWARD_RATIO = 2.5       # 1:2.5 R:R
MAX_PORTFOLIO_PCT = 0.20      # 20% per trade (동적 Half-Kelly: 16% ~ 20%)
MAX_CONCURRENT_POSITIONS = 5  # 최대 동시 보유 5종목 (가동률 100%)
MIN_SURGE_PCT = 0.030         # 3.0% 이상 서지
MAX_SL_PCT = 0.070            # 최대 7% 손절폭
MIN_SL_PCT = 0.015            # 최소 1.5% 손절폭
COOLDOWN_SEC = 900            # 15분 재진입 쿨다운

KIS_APP_KEY = os.getenv("KIS_API_KEY", "")
KIS_APP_SECRET = os.getenv("KIS_SECRET", "")
_acc = os.getenv("KIS_ACCOUNT_NO", "")
KIS_CANO, KIS_PRDT_ABRV = _acc.split("-") if "-" in _acc else (_acc[:8], _acc[8:10])
KIS_URL = os.getenv("KIS_URL", "https://openapi.koreainvestment.com:9443")


class KisCameronPennyTrader:
    def __init__(self, test_mode: bool = True):
        self.test_mode = test_mode
        self.positions: Dict[str, Any] = self._load_positions()
        self.cooldowns: Dict[str, float] = {}
        self.account_equity = 14872230.0  # 기본 통합증거금 기준 1,487만 원
        logger.info(f"🇰🇷 [KIS 동전주 30분봉 스윙 엔진] 초기화 완료 (Test Mode: {self.test_mode})")

    def _load_positions(self) -> Dict[str, Any]:
        if os.path.exists(POSITIONS_FILE):
            try:
                with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"❌ 포지션 파일 로드 실패: {e}")
        return {}

    def _save_positions(self):
        try:
            with open(POSITIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.positions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ 포지션 파일 저장 실패: {e}")

    def fetch_market_data(self, yf_ticker: str) -> Optional[pd.DataFrame]:
        try:
            df = yf.download(yf_ticker, period="1mo", interval=TIMEFRAME, progress=False)
            if df.empty or len(df) < 20:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            tp = (df["high"] + df["low"] + df["close"]) / 3
            df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, 1)
            df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
            return df
        except Exception as e:
            logger.error(f"❌ 데이터 수신 오류 ({yf_ticker}): {e}")
            return None

    def calculate_dynamic_position_size(self, vol_ratio: float, retrace_ratio: float, trigger_price: float) -> int:
        """동적 Half-Kelly 포지션 사이징 (16% ~ 20%)"""
        conviction = 1.0
        if vol_ratio > 1.1:
            conviction -= 0.10
        if retrace_ratio > 0.45:
            conviction -= 0.10

        kelly_pct = max(0.16, min(MAX_PORTFOLIO_PCT, MAX_PORTFOLIO_PCT * conviction))
        target_allocation = self.account_equity * kelly_pct
        qty = math.floor(target_allocation / trigger_price)
        logger.info(f"   📊 [Dynamic Kelly Sizing] Target: {kelly_pct*100:.1f}% (~₩{target_allocation:,.0f}) -> {qty}주")
        return qty

    def evaluate_cameron_setup(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """30분봉 로스 카메론 풀백 셋업 판정 (완화된 가드레일 적용)"""
        curr_bar = df.iloc[-1]
        prev_bar = df.iloc[-2]

        # 1. 눌림목 캔들: 음봉 또는 실몸통 45% 이하 도지/팽이 캔들 허용
        prev_range = prev_bar["high"] - prev_bar["low"]
        is_bearish = prev_bar["close"] < prev_bar["open"]
        is_small_body = prev_range > 0 and (abs(prev_bar["close"] - prev_bar["open"]) / prev_range) <= 0.45
        if not (is_bearish or is_small_body):
            return False, {}

        # 2. 9 EMA 및 VWAP 지지 확인 (2~3% 유연 버퍼)
        if prev_bar["close"] < prev_bar["ema9"] * 0.98 or prev_bar["low"] < prev_bar["vwap"] * 0.97:
            return False, {}

        # 3. 직전 3~6봉 급등 파동 (Impulse)
        prior = df.iloc[-7:-2]
        surge_low = prior["low"].min()
        surge_high = prior["high"].max()
        surge = surge_high - surge_low
        if surge <= 0 or (surge / surge_low) < MIN_SURGE_PCT:
            return False, {}

        # 4. 되돌림 60% 이내 & 거래량 축소
        pullback_depth = surge_high - prev_bar["low"]
        retrace_ratio = pullback_depth / surge
        if retrace_ratio > 0.60:
            return False, {}

        prior_avg_vol = prior["volume"].mean()
        if prior_avg_vol > 0 and prev_bar["volume"] > prior_avg_vol * 1.35:
            return False, {}

        # 5. 돌파 트리거
        trigger_price = prev_bar["high"]
        stop_loss = prev_bar["low"]
        risk = trigger_price - stop_loss
        if risk <= 0:
            return False, {}

        risk_pct = risk / trigger_price
        if not (MIN_SL_PCT <= risk_pct <= MAX_SL_PCT):
            return False, {}

        take_profit = trigger_price + (risk * RISK_REWARD_RATIO)
        is_triggered = curr_bar["high"] >= trigger_price

        vol_ratio = prev_bar["volume"] / prior_avg_vol if prior_avg_vol > 0 else 1.0

        return is_triggered, {
            "trigger_price": trigger_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_pct": risk_pct * 100,
            "target_pct": (take_profit / trigger_price - 1) * 100,
            "vol_ratio": vol_ratio,
            "retrace_ratio": retrace_ratio
        }

    def run_cycle(self):
        now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
        now_t = time.time()
        logger.info(f"⏰ [평가 주기] KST {now_kst.strftime('%H:%M:%S')} | 보유 포지션: {list(self.positions.keys())}")

        # 1. 보유 포지션 손절/익절 감시
        for code in list(self.positions.keys()):
            pos = self.positions[code]
            info = PENNY_TARGETS.get(code, {})
            df = self.fetch_market_data(info.get("yf_ticker", ""))
            if df is None:
                continue

            curr_price = df.iloc[-1]["close"]
            sl = pos["stop_loss"]
            tp = pos["take_profit"]

            if curr_price <= sl:
                logger.warning(f"🛑 [손절 발동] {pos['name']}({code}) 현재가: {curr_price:,.0f}원 <= 손절선: {sl:,.0f}원 (쿨다운 15분 등록)")
                del self.positions[code]
                self.cooldowns[code] = now_t
                self._save_positions()
            elif curr_price >= tp:
                logger.info(f"🎉 [목표 익절] {pos['name']}({code}) 현재가: {curr_price:,.0f}원 >= 목표가: {tp:,.0f}원 (쿨다운 15분 등록)")
                del self.positions[code]
                self.cooldowns[code] = now_t
                self._save_positions()

        if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
            return

        # 2. 신규 동전주 1차 눌림목 감시
        for code, info in PENNY_TARGETS.items():
            if code in self.positions or (now_t - self.cooldowns.get(code, 0) < COOLDOWN_SEC):
                continue

            df = self.fetch_market_data(info["yf_ticker"])
            if df is None:
                continue

            triggered, setup = self.evaluate_cameron_setup(df)
            if triggered:
                qty = self.calculate_dynamic_position_size(setup["vol_ratio"], setup["retrace_ratio"], setup["trigger_price"])
                logger.info(
                    f"🎯 [동전주 매수 신호] {info['name']}({code}) {qty}주 돌파가: {setup['trigger_price']:,.0f}원 | "
                    f"손절: {setup['stop_loss']:,.0f}원 (-{setup['risk_pct']:.1f}%) | "
                    f"목표: {setup['take_profit']:,.0f}원 (+{setup['target_pct']:.1f}%)"
                )
                self.positions[code] = {
                    "code": code,
                    "name": info["name"],
                    "qty": qty,
                    "entry_price": setup["trigger_price"],
                    "stop_loss": setup["stop_loss"],
                    "take_profit": setup["take_profit"],
                    "entry_time": now_kst.isoformat()
                }
                self._save_positions()
                if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
                    break


if __name__ == "__main__":
    trader = KisCameronPennyTrader(test_mode=True)
    trader.run_cycle()
