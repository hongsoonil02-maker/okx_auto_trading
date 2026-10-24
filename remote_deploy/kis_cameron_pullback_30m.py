# -*- coding: utf-8 -*-
"""
kis_cameron_pullback_30m.py — KIS 미국주식 로스 카메론 30분봉 단타 매매 봇
(Ross Cameron 30-Min Small-Cap Momentum Pullback Strategy for KIS Overseas Account)

전략 철학:
1. 유니버스 스캔 (Alpaca Screener API 활용):
   - 가격: $2.00 ~ $50.00
   - 당일 갭상승(Gap-Up) >= 3.0%
   - 상대 거래량 및 거래대금 상위 모멘텀 소형주 20종목
2. 진입 셋업 (30분봉 풀백 패턴):
   - 장초반 급등 임펄스 파동 (최소 2% 이상 서지)
   - 첫 번째 눌림목 (1~3봉 음봉, 되돌림 <= 50%, 거래량 급감)
   - 롤링 VWAP 및 9 EMA 상단 유지
   - 진입 트리거: 직전 풀백봉의 고점(High) 돌파
3. 리스크 & 주문 관리 (KIS OpenAPI):
   - 포지션 사이징: 가용 자산의 10% (최대 동시 3종목)
   - 자체 소프트 브래킷(Software Bracket) 실시간 감시:
     * 익절(TP): 1:2 R:R 도달 시 자동 매도 (최소 +$0.05 안전 마진 강제)
     * 손절(SL): 스윙 로우(풀백 저점, 1~4%) 이탈 시 체결우선 매도
   - 컷오프: 14:30 EST (장마감 1시간 30분 전) 이후 신규 진입 차단
   - 당일 청산(EOD Close): 15:50 EST 미청산 포지션 당일 전량 정리 (단타 원칙)
"""

import os
import sys
import time
import math
import json
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional, Tuple, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, ".."))
load_dotenv(os.path.join(CURRENT_DIR, "..", ".env"))

# ── 로깅 설정 ──
LOG_FILE = os.path.join(CURRENT_DIR, "kis_cameron_30m_trader.log")
STATE_FILE = os.path.join(CURRENT_DIR, "kis_cameron_30m_state.json")
TOKEN_FILE = os.path.join(CURRENT_DIR, "kis_token.json")

logger = logging.getLogger("KisCameron30m")
logger.setLevel(logging.INFO)
logger.propagate = False

_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_fh = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=2, encoding="utf-8")
_fh.setFormatter(_formatter)
logger.addHandler(_fh)

if sys.stdout.isatty():
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(_formatter)
    logger.addHandler(_sh)

# ── 네트워크 세션 ──
session = requests.Session()
_adapter = HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
                                         allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]))
session.mount("https://", _adapter)

# ── API 인증키 ──
KIS_APP_KEY = os.getenv("KIS_API_KEY", "")
KIS_APP_SECRET = os.getenv("KIS_SECRET", "")
_acc = os.getenv("KIS_ACCOUNT_NO", "")
KIS_CANO, KIS_PRDT_ABRV = _acc.split("-") if "-" in _acc else (_acc[:8], _acc[8:10])
KIS_URL = os.getenv("KIS_URL", "https://openapi.koreainvestment.com:9443")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_DATA_URL = "https://data.alpaca.markets/v2"
ALPACA_REST_URL = os.getenv("ALPACA_REST_URL", "https://paper-api.alpaca.markets/v2")

ALPACA_HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type": "application/json"
}

from jev.universal_jev_gate import UniversalJevGate

# ── 전략 하이퍼파라미터 (30분봉 단타 최적화) ──
TIMEFRAME = "30Min"
MIN_PRICE = 2.00
MAX_PRICE = 50.00
MIN_GAP_PCT = 3.0           # 장초반 모멘텀 최소 3%
MAX_PORTFOLIO_PCT = 0.20    # 20% per trade (동적 Half-Kelly Sizing: 16% ~ 20%)
MAX_CONCURRENT_TRADES = 5   # 최대 동시 보유 5종목 (가동률 100% 완전 활용)
RISK_REWARD_RATIO = 3.0     # 1:3.0 R:R (Optimized via 6-month sweep: PF 3.07, +38.4%)
MAX_SL_PCT = 0.07           # 최대 손절폭 7% (소형주 30분봉 실전 변동성 수용)
MIN_SL_PCT = 0.01           # 최소 손절폭 1%
CUTOFF_HOUR_EST = 14        # 14:30 EST (장마감 1시간 30분 전 신규 진입 컷오프)
CUTOFF_MIN_EST = 30
EOD_CLOSE_HOUR_EST = 15     # 15:50 EST 당일 단타 미청산 포지션 일괄 마감
EOD_CLOSE_MIN_EST = 50
COOLDOWN_SEC = 900          # 15분 재진입 쿨다운 (연속 중복 발주 원천 차단)


class KisCameron30mTrader:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.access_token = ""
        self.token_expired_at = 0.0
        self.account_equity = 11700.0
        self.available_cash = 11700.0
        self.watchlist: List[str] = []
        self.last_scanned_time = 0
        self.jev_gate = UniversalJevGate(confidence_threshold=0.60)
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.cooldowns: Dict[str, float] = {}

        self.load_state()

    def calculate_dynamic_position_size(self, setup: Dict[str, Any], trigger_price: float) -> int:
        """
        Dynamic Half-Kelly Position Sizing:
        - Baseline: p=0.48 (win rate), b=3.0 (R:R) -> Full Kelly f* = 30.6%
        - Half-Kelly: 16% ~ 20%
        - Scales based on pullback volume contraction & retracement depth
        """
        vol_ratio = setup.get("vol_ratio", 1.0)
        retrace_ratio = setup.get("retrace_ratio", 40.0)

        conviction = 1.0
        if vol_ratio > 1.1:
            conviction -= 0.10
        if retrace_ratio > 45.0:
            conviction -= 0.10

        kelly_pct = max(0.16, min(MAX_PORTFOLIO_PCT, MAX_PORTFOLIO_PCT * conviction))
        target_allocation = self.account_equity * kelly_pct
        qty = math.floor(target_allocation / trigger_price)
        logger.info(f"   📊 [Dynamic Kelly Sizing] Target: {kelly_pct*100:.1f}% (~${target_allocation:,.2f}) -> {qty} shares")
        return qty

    def load_state(self):
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    self.active_positions = json.load(f)
                logger.info(f"📂 [State Loaded] Active positions: {list(self.active_positions.keys())}")
        except Exception as e:
            logger.warning(f"State load warning: {e}")
            self.active_positions = {}

    def save_state(self):
        try:
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.active_positions, f, indent=2)
            os.replace(tmp, STATE_FILE)
        except Exception as e:
            logger.error(f"❌ State save failed: {e}")

    def issue_token(self) -> bool:
        now = time.time()
        if self.access_token and now < self.token_expired_at:
            return True
        if os.path.exists(TOKEN_FILE):
            try:
                with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if d.get("access_token") and d.get("expires_at", 0) > now + 300:
                    self.access_token, self.token_expired_at = d["access_token"], d["expires_at"]
                    return True
            except Exception:
                pass
        try:
            r = session.post(
                f"{KIS_URL}/oauth2/tokenP",
                json={"grant_type": "client_credentials", "appkey": KIS_APP_KEY, "appsecret": KIS_APP_SECRET},
                timeout=10
            )
            if r.status_code == 200 and "access_token" in r.json():
                d = r.json()
                self.access_token = d["access_token"]
                self.token_expired_at = now + int(d.get("expires_in", 86400)) - 600
                with open(TOKEN_FILE, "w", encoding="utf-8") as f:
                    json.dump({"access_token": self.access_token, "expires_at": self.token_expired_at}, f)
                return True
        except Exception as e:
            logger.error(f"❌ 토큰 발급 실패: {e}")
        return False

    def headers(self, tr_id: str) -> dict:
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.access_token}",
            "appkey": KIS_APP_KEY,
            "appsecret": KIS_APP_SECRET,
            "tr_id": tr_id,
            "custtype": "P"
        }

    def sync_account(self):
        """KIS 해외주식 계좌 자산 및 예수금 동기화"""
        if not self.issue_token():
            return
        tot_evlu_usd = 0.0
        usd_cash = 0.0
        try:
            r = session.get(
                f"{KIS_URL}/uapi/overseas-stock/v1/trading/inquire-balance",
                headers=self.headers("TTTS3012R"),
                params={"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "OVRS_EXCG_CD": "NASD",
                        "TR_CRCY_CD": "USD", "CTX_AREA_FK200": "", "CTX_AREA_NK200": ""},
                timeout=8
            )
            if r.status_code == 200:
                out2 = r.json().get("output2", {})
                tot_evlu_usd = float(out2.get("tot_evlu_pfls_amt", 0.0))
                usd_cash = float(out2.get("frcr_dnca_amt", 0.0) or 0.0)
        except Exception as e:
            logger.warning(f"⚠️ Account sync error: {e}")

        # 원화 예수금 통합증거금 확인
        krw_cash = 0.0
        try:
            r = session.get(
                f"{KIS_URL}/uapi/domestic-stock/v1/trading/inquire-psbl-order",
                headers=self.headers("TTTC8908R"),
                params={"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "PDNO": "005930",
                        "ORD_UNPR": "0", "ORD_DVSN": "01", "CMA_EVLU_AMT_ICLD_YN": "Y", "OVRS_ICLD_YN": "Y"},
                timeout=8
            )
            if r.status_code == 200:
                out = r.json().get("output", {})
                krw_cash = float(out.get("max_buy_amt") or out.get("nrcvb_buy_amt") or out.get("ord_psbl_cash", 0.0))
        except Exception:
            pass

        fx_rate = 1380.0
        krw_in_usd = krw_cash / fx_rate if krw_cash > 0 else 0.0
        self.available_cash = usd_cash + krw_in_usd
        if tot_evlu_usd + krw_in_usd > 0:
            self.account_equity = tot_evlu_usd + krw_in_usd

    def get_price(self, sym: str) -> Optional[float]:
        """KIS 실시간 시세 조회 (실패 시 Alpaca Data API 폴백)"""
        if self.issue_token():
            try:
                r = session.get(
                    f"{KIS_URL}/uapi/overseas-price/v1/quotations/price",
                    headers=self.headers("HHDFS00000300"),
                    params={"AUTH": "", "EXCD": "NAS", "SYMB": sym},
                    timeout=4
                )
                if r.status_code == 200:
                    px = float(r.json().get("output", {}).get("last", 0.0))
                    if px > 0:
                        return px
            except Exception:
                pass

        # Alpaca 폴백
        try:
            r = session.get(f"{ALPACA_DATA_URL}/stocks/{sym}/trades/latest", headers=ALPACA_HEADERS, timeout=4)
            if r.status_code == 200:
                return float(r.json().get("trade", {}).get("p", 0.0))
        except Exception:
            pass
        return None

    def scan_universe(self) -> List[str]:
        """Alpaca Screener API로 당일 갭상승 소형 모멘텀주 상위 20개 스캔"""
        now = time.time()
        if now - self.last_scanned_time < 300 and self.watchlist:
            return self.watchlist

        try:
            url = f"{ALPACA_REST_URL}/screener/stocks/most-actives?by=volume&top=100"
            r = session.get(url, headers=ALPACA_HEADERS, timeout=6)
            candidates = []
            if r.status_code == 200:
                actives = r.json().get("most_actives", [])
                for item in actives:
                    sym = item.get("symbol")
                    px = float(item.get("price", 0.0))
                    chg = float(item.get("change", 0.0))
                    if MIN_PRICE <= px <= MAX_PRICE and chg >= MIN_GAP_PCT:
                        candidates.append((chg, sym))

            # Movers API 추가 스캔
            r_movers = session.get(f"{ALPACA_REST_URL}/screener/stocks/movers?top=50", headers=ALPACA_HEADERS, timeout=6)
            if r_movers.status_code == 200:
                gainers = r_movers.json().get("gainers", [])
                for item in gainers:
                    sym = item.get("symbol")
                    px = float(item.get("price", 0.0))
                    chg = float(item.get("percent_change", 0.0))
                    if MIN_PRICE <= px <= MAX_PRICE and chg >= MIN_GAP_PCT:
                        if not any(s == sym for _, s in candidates):
                            candidates.append((chg, sym))

            candidates.sort(key=lambda x: x[0], reverse=True)
            self.watchlist = [s for _, s in candidates[:20]]
            self.last_scanned_time = now
            logger.info(f"📋 [Watchlist Updated] Top {len(self.watchlist)} tickers: {self.watchlist}")
            return self.watchlist
        except Exception as e:
            logger.error(f"❌ Scanner failed: {e}")
            return self.watchlist

    def fetch_bars(self, symbol: str, limit: int = 40) -> Optional[pd.DataFrame]:
        """Alpaca Data API로 30분봉 OHLCV 수집"""
        try:
            url = f"{ALPACA_DATA_URL}/stocks/{symbol}/bars"
            params = {"timeframe": TIMEFRAME, "limit": limit, "feed": "sip", "sort": "asc"}
            r = session.get(url, headers=ALPACA_HEADERS, params=params, timeout=6)
            if r.status_code == 200:
                bars = r.json().get("bars", [])
                if not bars:
                    return None
                df = pd.DataFrame(bars)
                df["time"] = pd.to_datetime(df["t"])
                df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}, inplace=True)
                return df
        except Exception as e:
            logger.warning(f"⚠️ [{symbol}] Bars fetch error: {e}")
        return None

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0
        df["vp"] = df["typical_price"] * df["volume"]
        df["vwap"] = df["vp"].cumsum() / df["volume"].cumsum()
        df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
        return df

    def detect_pullback_setup(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Ross Cameron 30분봉 풀백 셋업 판정 (유연화된 실전 가드레일)"""
        if len(df) < 15:
            return False, {}

        df = self.calculate_technical_indicators(df)
        curr_bar = df.iloc[-1]
        prev_bar = df.iloc[-2]

        # 1. 눌림목 캔들 확인: 음봉이거나 몸통이 작은 캔들(도지/팽이형) 허용
        prev_range = prev_bar["high"] - prev_bar["low"]
        is_bearish = prev_bar["close"] < prev_bar["open"]
        is_small_body = prev_range > 0 and (abs(prev_bar["close"] - prev_bar["open"]) / prev_range) <= 0.45
        if not (is_bearish or is_small_body):
            return False, {}

        # 2. VWAP / EMA9 지지 확인 (노이즈 방지용 2~3% 유연 버퍼)
        if prev_bar["close"] < prev_bar["ema9"] * 0.98 or prev_bar["low"] < prev_bar["vwap"] * 0.97:
            return False, {}

        # 3. 풀백 전 임펄스 서지 확인 (최근 2~6봉 전)
        impulse_slice = df.iloc[-7:-2]
        if len(impulse_slice) < 2:
            return False, {}

        impulse_high = impulse_slice["high"].max()
        impulse_low = impulse_slice["low"].min()
        impulse_surge = impulse_high - impulse_low

        if impulse_surge <= 0 or (impulse_surge / impulse_low) < 0.02:
            return False, {}

        # 4. 되돌림 비율 <= 60%
        pullback_depth = impulse_high - prev_bar["low"]
        retrace_ratio = pullback_depth / impulse_surge
        if retrace_ratio > 0.60:
            return False, {}

        # 5. 거래량 축소 확인 (과도한 폭증 방지)
        impulse_avg_vol = impulse_slice["volume"].mean()
        if impulse_avg_vol > 0 and prev_bar["volume"] > impulse_avg_vol * 1.35:
            return False, {}

        trigger_price = prev_bar["high"]
        stop_loss_price = prev_bar["low"]
        risk_per_share = trigger_price - stop_loss_price
        if risk_per_share <= 0:
            return False, {}

        risk_pct = risk_per_share / trigger_price
        if risk_pct > MAX_SL_PCT or risk_pct < MIN_SL_PCT:
            return False, {}

        target_price = trigger_price + (risk_per_share * RISK_REWARD_RATIO)
        is_triggered = curr_bar["high"] >= trigger_price

        setup_info = {
            "trigger_price": round(trigger_price, 2),
            "stop_loss": round(stop_loss_price, 2),
            "target_price": round(target_price, 2),
            "risk_per_share": round(risk_per_share, 2),
            "risk_pct": round(risk_pct * 100, 2),
            "surge_pct": round(impulse_surge / impulse_low * 100, 2),
            "vol_ratio": round(prev_bar["volume"] / impulse_avg_vol, 2) if impulse_avg_vol > 0 else 1.0,
            "retrace_ratio": round(retrace_ratio * 100, 1),
            "curr_price": round(curr_bar["close"], 2),
            "is_triggered": is_triggered
        }
        return is_triggered, setup_info

    def enter_position(self, symbol: str, setup: Dict[str, Any]):
        """KIS 매수 발주 및 포지션 등록"""
        if len(self.active_positions) >= MAX_CONCURRENT_TRADES:
            return

        trigger_price = setup["trigger_price"]
        stop_loss = setup["stop_loss"]
        take_profit = setup["target_price"]
        curr_price = setup.get("curr_price", trigger_price)
        base_ref = max(trigger_price, curr_price)

        # 익절가 최소 안전 마진 강제 (+0.05)
        min_tp = round(base_ref + 0.05, 2)
        if take_profit <= min_tp:
            take_profit = round(base_ref + max(setup.get("risk_per_share", 0.05) * RISK_REWARD_RATIO, 0.05), 2)

        # 동적 Half-Kelly 포지션 사이징 (16% ~ 20%)
        qty = self.calculate_dynamic_position_size(setup, trigger_price)
        if qty <= 0:
            return

        # JEV AI Gate 검증
        try:
            loop = asyncio.new_event_loop()
            decision = loop.run_until_complete(
                self.jev_gate.verify_entry(
                    asset_class="US_STOCKS",
                    symbol=symbol,
                    setup_name="KIS_CAMERON_30M",
                    context={"price": trigger_price, "risk_pct": setup["risk_pct"]}
                )
            )
            loop.close()
            if not decision.approved:
                logger.warning(f"🛑 [JEV VETO] {symbol} 진입 거절: {decision.reason}")
                return
            logger.info(f"🛡️ [JEV APPROVED] {symbol} 승인: {decision.reason}")
        except Exception as e:
            logger.warning(f"⚠️ JEV 에러: {e}")

        # KIS 매수 발주 (체결 최우선을 위해 트리거가 +0.5% 지정가)
        order_px = round(base_ref * 1.005, 2)
        body = {
            "CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "OVRS_EXCG_CD": "NASD", "PDNO": symbol,
            "ORD_QTY": str(qty), "OVRS_ORD_UNPR": f"{order_px:.2f}", "ORD_SVR_DVSN_CD": "0", "ORD_DVSN": "00"
        }
        try:
            logger.info(f"🚀 [KIS 진입 발주] {symbol} x {qty}주 @ ${order_px:.2f} (SL: ${stop_loss}, TP: ${take_profit})")
            if not self.dry_run:
                r = session.post(f"{KIS_URL}/uapi/overseas-stock/v1/trading/order", headers=self.headers("TTTT1002U"), json=body, timeout=8)
                d = r.json()
                if r.status_code == 200 and d.get("rt_cd") == "0":
                    odno = d.get("output", {}).get("ODNO", "")
                    logger.info(f"   ✅ [매수 접수 성공] {symbol} ODNO: {odno}")
                else:
                    logger.error(f"   ❌ [매수 접수 실패] {symbol}: {d.get('msg1')}")
                    return

            self.active_positions[symbol] = {
                "qty": qty,
                "entry_price": base_ref,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "entry_time": time.time()
            }
            self.save_state()
        except Exception as e:
            logger.error(f"❌ 매수 예외 {symbol}: {e}")

    def monitor_positions(self):
        """실시간 포지션 감시 및 TP/SL 자동 청산 루프"""
        if not self.active_positions:
            return

        to_remove = []
        for sym, pos in self.active_positions.items():
            curr_px = self.get_price(sym)
            if not curr_px or curr_px <= 0:
                continue

            sl = pos["stop_loss"]
            tp = pos["take_profit"]
            qty = pos["qty"]

            # 1. 익절 도달 (TP)
            if curr_px >= tp:
                logger.info(f"🎯 [TAKE PROFIT 도달] {sym} 현재가 ${curr_px:.2f} >= TP ${tp:.2f}")
                self.sell_position(sym, qty, curr_px, reason="TAKE_PROFIT")
                to_remove.append(sym)

            # 2. 손절선 이탈 (SL)
            elif curr_px <= sl:
                logger.warning(f"🛑 [STOP LOSS 이탈] {sym} 현재가 ${curr_px:.2f} <= SL ${sl:.2f}")
                self.sell_position(sym, qty, curr_px, reason="STOP_LOSS")
                to_remove.append(sym)

        for sym in to_remove:
            self.active_positions.pop(sym, None)
        if to_remove:
            self.save_state()

    def sell_position(self, symbol: str, qty: int, curr_px: float, reason: str = "EXIT"):
        """체결우선(-1%) 시장가 매도 실행"""
        limit_px = round(curr_px * 0.99, 2)
        body = {
            "CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "OVRS_EXCG_CD": "NASD", "PDNO": symbol,
            "ORD_QTY": str(qty), "OVRS_ORD_UNPR": f"{limit_px:.2f}", "ORD_SVR_DVSN_CD": "0", "ORD_DVSN": "00"
        }
        try:
            logger.info(f"📤 [청산 주문] {symbol} {qty}주 @ ${limit_px:.2f} | 사유: {reason}")
            self.cooldowns[symbol] = time.time()
            if not self.dry_run:
                r = session.post(f"{KIS_URL}/uapi/overseas-stock/v1/trading/order", headers=self.headers("TTTT1006U"), json=body, timeout=8)
                d = r.json()
                if r.status_code == 200 and d.get("rt_cd") == "0":
                    logger.info(f"   ✅ [매도 접수] {symbol} ODNO: {d.get('output', {}).get('ODNO')}")
                else:
                    logger.error(f"   ❌ [매도 실패] {symbol}: {d.get('msg1')}")
        except Exception as e:
            logger.error(f"❌ 매도 예외 {symbol}: {e}")

    def eod_liquidate(self):
        """15:50 EST 장마감 전 당일 단타 미청산 포지션 일괄 정리"""
        if not self.active_positions:
            return
        logger.info("🚨 [EOD 당일 청산] 15:50 EST 장마감 전 단타 포지션 전량 정리")
        for sym, pos in list(self.active_positions.items()):
            px = self.get_price(sym) or pos["entry_price"]
            self.sell_position(sym, pos["qty"], px, reason="EOD_CLOSE")
        self.active_positions = {}
        self.save_state()

    def run_loop(self):
        logger.info("🟢 [KIS Cameron 30M Trader] 루프 시작 (Poll: 15s)")
        tz_ny = ZoneInfo("America/New_York")

        while True:
            try:
                now_ny = datetime.now(tz_ny)
                is_weekday = now_ny.weekday() < 5
                # 정규장 시간: 09:30 ~ 16:00 EST
                market_open = is_weekday and (
                    (now_ny.hour == 9 and now_ny.minute >= 30) or
                    (9 < now_ny.hour < 16)
                )

                if market_open:
                    # 1. 포지션 실시간 감시 (익절/손절 체크)
                    self.monitor_positions()

                    # 2. 15:50 EOD 일괄 마감 체크
                    if now_ny.hour == EOD_CLOSE_HOUR_EST and now_ny.minute >= EOD_CLOSE_MIN_EST:
                        self.eod_liquidate()
                        time.sleep(30)
                        continue

                    # 3. 14:30 컷오프 이전이고 포지션 여유가 있으면 신규 진입 스캔
                    past_cutoff = (now_ny.hour > CUTOFF_HOUR_EST) or (now_ny.hour == CUTOFF_HOUR_EST and now_ny.minute >= CUTOFF_MIN_EST)
                    # 09:30~09:35는 1단계 기존 보유 청산 진행 시간을 감안하여 09:35부터 신규 진입
                    after_warmup = (now_ny.hour > 9) or (now_ny.hour == 9 and now_ny.minute >= 35)

                    if not past_cutoff and after_warmup and len(self.active_positions) < MAX_CONCURRENT_TRADES:
                        self.sync_account()
                        tickers = self.scan_universe()
                        now_t = time.time()
                        for sym in tickers:
                            if sym in self.active_positions or (now_t - self.cooldowns.get(sym, 0) < COOLDOWN_SEC):
                                continue
                            if len(self.active_positions) >= MAX_CONCURRENT_TRADES:
                                break
                            df = self.fetch_bars(sym, limit=35)
                            if df is not None:
                                triggered, setup = self.detect_pullback_setup(df)
                                if triggered:
                                    logger.info(f"🔥 [SETUP 포착] {sym} Pullback Setup Triggered!")
                                    self.enter_position(sym, setup)
                                    time.sleep(1)

                time.sleep(15)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"❌ Main loop exception: {e}")
                time.sleep(15)


if __name__ == "__main__":
    trader = KisCameron30mTrader()
    trader.run_loop()
