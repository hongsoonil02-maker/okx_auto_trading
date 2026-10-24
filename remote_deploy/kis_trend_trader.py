# -*- coding: utf-8 -*-
"""
kis_trend_trader.py — Autonomous KOSPI/KOSDAQ Trend Following Execution Engine
Replaces the deprecated microstructure scalper with a validated Quant Trend Following Model:
- Strategy: Donchian Channel 20-Day Breakout + 60-Day SMA Trend Filter
- Exit: ATR 2.0x Dynamic Trailing Stop & 10-Day Low Breakdown
- Risk Management: Max 3 Concurrent Positions, 30% Allocation, Strict Cash Guard via inquire-psbl-order
- Realistic Tax/Fee Defense: Long Holding (~18 days avg) eliminating 0.18% transaction tax churn
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import yfinance as yf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, "..", ".env"))

# ── LOGGING SETUP ──
LOG_FILE = os.path.join(CURRENT_DIR, "kis_trend_trader.log")
logger = logging.getLogger("KisTrendTrader")
logger.setLevel(logging.INFO)
logger.propagate = False  # 중복 로깅 방지

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=3, encoding="utf-8")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
# systemd가 stdout을 같은 로그 파일로 append하므로 콘솔 핸들러는 대화형 실행 시에만 추가 (중복 방지)
if sys.stdout.isatty():
    logger.addHandler(stream_handler)

# ── ROBUST HTTP SESSION ──
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

# ── KIS API CREDENTIALS ──
KIS_APP_KEY = os.getenv("KIS_API_KEY", "")
KIS_APP_SECRET = os.getenv("KIS_SECRET", "")
account_no = os.getenv("KIS_ACCOUNT_NO", "")
if "-" in account_no:
    KIS_CANO, KIS_PRDT_ABRV = account_no.split("-")
elif len(account_no) >= 10:
    KIS_CANO, KIS_PRDT_ABRV = account_no[:8], account_no[8:10]
else:
    KIS_CANO, KIS_PRDT_ABRV = "", ""

KIS_URL = os.getenv("KIS_URL", "https://openapi.koreainvestment.com:9443")

# ── TRACKED UNIVERSE ──
# KOSPI/KOSDAQ 주도 섹터 톱픽 22종목 (반도체, 방산, 전력, 바이오, K-푸드, 조선, 자동차, 밸류업)
UNIVERSE = {
    # 기존 대형 우량주
    "005930": {"name": "삼성전자", "yf": "005930.KS"},
    "000660": {"name": "SK하이닉스", "yf": "000660.KS"},
    "373220": {"name": "LG에너지솔루션", "yf": "373220.KS"},
    "207940": {"name": "삼성바이오로직스", "yf": "207940.KS"},
    "005380": {"name": "현대차", "yf": "005380.KS"},
    "000270": {"name": "기아", "yf": "000270.KS"},
    "068270": {"name": "셀트리온", "yf": "068270.KS"},
    "035420": {"name": "NAVER", "yf": "035420.KS"},
    "035720": {"name": "카카오", "yf": "035720.KS"},
    "042700": {"name": "한미반도체", "yf": "042700.KS"},
    # [Step 2] 시장 주도 섹터 핵심 대장주 12종 추가 편입
    "012450": {"name": "한화에어로스페이스", "yf": "012450.KS"}, # K-방산/우주항공 1위
    "064350": {"name": "현대로템", "yf": "064350.KS"},          # 방산/철도 랠리 대장주
    "267260": {"name": "HD현대일렉트릭", "yf": "267260.KS"},    # AI 변압기/전력인프라 대장주
    "298040": {"name": "효성중공업", "yf": "298040.KS"},        # 초고압 변압기 대장주
    "196170": {"name": "알테오젠", "yf": "196170.KQ"},          # 코스닥 바이오 1위 기술수출 대장주
    "003230": {"name": "삼양식품", "yf": "003230.KS"},          # K-푸드/불닭 글로벌 수출 대장주
    "009540": {"name": "HD한국조선해양", "yf": "009540.KS"},    # 조선 슈퍼사이클 대장주
    "034020": {"name": "두산에너빌리티", "yf": "034020.KS"},    # 원전/SMR 수주 대장주
    "141080": {"name": "리가켐바이오", "yf": "141080.KQ"},      # ADC 차세대 항암 플랫폼 대장주
    "259960": {"name": "크래프톤", "yf": "259960.KS"},          # 실적 서프라이즈 게임 대장주
    "000810": {"name": "삼성화재", "yf": "000810.KS"},          # 밸류업 금융/보험 최우량주
    "105560": {"name": "KB금융", "yf": "105560.KS"},            # 밸류업 은행 1위 대장주
}

EXCLUDED_SYMBOLS = {"007630", "259290"}  # 상폐/거래불가 종목

# ── STRATEGY HYPERPARAMETERS ──
MAX_CONCURRENT_POSITIONS = 3
MAX_POSITION_PCT = 0.30       # 종목당 총자산의 최대 30%
ATR_TRAILING_MULTIPLIER = 2.0 # 고점 대비 ATR 2.0배 하락 시 트레일링 스탑
HARD_STOP_LOSS_PCT = 0.07     # 진입가 대비 최대 -7% 절대 손절선
POLL_INTERVAL_SEC = 30        # 정규장 중 30초 주기 모니터링
POSITIONS_FILE = os.path.join(CURRENT_DIR, "kis_positions.json")
TOKEN_FILE = os.path.join(CURRENT_DIR, "kis_token.json")

# KRX 정규장 휴장일 (2026-2027 공휴일 및 연말 휴장일)
KRX_HOLIDAYS = {
    # 2026
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-02-18", "2026-03-01", "2026-03-02",
    "2026-05-01", "2026-05-05", "2026-05-24", "2026-05-25", "2026-06-06", "2026-08-15",
    "2026-08-17", "2026-09-24", "2026-09-25", "2026-09-26", "2026-10-03", "2026-10-05",
    "2026-10-09", "2026-12-25", "2026-12-31",
    # 2027
    "2027-01-01", "2027-02-05", "2027-02-08", "2027-02-09", "2027-03-01", "2027-05-01",
    "2027-05-05", "2027-05-13", "2027-06-06", "2027-08-15", "2027-08-16", "2027-09-14",
    "2027-09-15", "2027-09-16", "2027-10-03", "2027-10-09", "2027-10-11", "2027-12-25",
    "2027-12-31",
}


class KisTrendTrader:
    def __init__(self):
        self.access_token = ""
        self.token_expired_at = 0
        self.account_info = {"equity": 0.0, "cash": 0.0}
        self.positions: Dict[str, dict] = {}
        self.daily_indicators: Dict[str, dict] = {}
        self.market_regime: Dict[str, Any] = {
            "is_bullish": True,
            "kospi_close": 0.0,
            "kospi_sma60": 0.0,
            "kospi_ret60": 0.0
        }
        self.last_indicator_update: str = ""
        self.last_market_status_log: float = 0.0

        self.load_positions()
        
        logger.info("=" * 75)
        logger.info("🚀 [KIS Trend Trader] Initializing KRX Trend Following Engine v3.0")
        logger.info(f"   REST Endpoint: {KIS_URL}")
        logger.info(f"   Account: {KIS_CANO}-{KIS_PRDT_ABRV}")
        univ_str = ", ".join([f"{s}({v['name']})" for s, v in UNIVERSE.items()])
        logger.info(f"   Universe: {univ_str}")
        logger.info(f"   Max Positions: {MAX_CONCURRENT_POSITIONS} | Alloc per Pos: {MAX_POSITION_PCT*100:.0f}%")
        logger.info(f"   Exit Rules: ATR {ATR_TRAILING_MULTIPLIER}x Trailing Stop | Hard Stop: -{HARD_STOP_LOSS_PCT*100:.0f}%")
        if self.positions:
            logger.info(f"   📂 Restored {len(self.positions)} active positions: {list(self.positions.keys())}")
        logger.info("=" * 75)

        # 초기 기동 시 시장 국면 및 계좌 상태 즉시 사전 분석
        try:
            self.fetch_daily_indicators()
            self.fetch_account_balance()
        except Exception as e:
            logger.warning(f"⚠️ Initial warmup warning: {e}")

    def save_positions(self):
        """포지션 정보를 JSON 파일로 원자적(Atomic) 영속화"""
        try:
            tmp_pos = f"{POSITIONS_FILE}.tmp.{os.getpid()}"
            with open(tmp_pos, "w", encoding="utf-8") as f:
                json.dump(self.positions, f, ensure_ascii=False, indent=2)
            os.replace(tmp_pos, POSITIONS_FILE)
        except Exception as e:
            logger.error(f"❌ Failed to save positions: {e}")

    def load_positions(self):
        """JSON 파일에서 포지션 복원"""
        try:
            if os.path.exists(POSITIONS_FILE):
                with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
                    self.positions = json.load(f)
                for ex in EXCLUDED_SYMBOLS:
                    self.positions.pop(ex, None)
                logger.info(f"📂 [Positions Restored] {len(self.positions)} loaded from {POSITIONS_FILE}")
            else:
                self.positions = {}
        except Exception as e:
            logger.error(f"❌ Failed to load positions: {e}")
            self.positions = {}

    def issue_token(self) -> bool:
        """KIS OAuth2 Access Token 관리 (파일 캐싱 및 자동 갱신)"""
        now = time.time()
        if self.access_token and now < self.token_expired_at:
            return True

        # 1. 파일 캐시 확인
        if not self.access_token and os.path.exists(TOKEN_FILE):
            try:
                with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                token = cache.get("access_token")
                exp = cache.get("expires_at", 0)
                if token and now < exp:
                    self.access_token = token
                    self.token_expired_at = exp
                    logger.info("🔐 [KIS API] Reused valid token from cache.")
                    return True
            except Exception as e:
                logger.warning(f"⚠️ Cache read error: {e}")

        # 2. 신규 발급 요청
        url = f"{KIS_URL}/oauth2/tokenP"
        payload = {
            "grant_type": "client_credentials",
            "appkey": KIS_APP_KEY,
            "appsecret": KIS_APP_SECRET
        }
        try:
            resp = session.post(url, json=payload, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                self.access_token = data.get("access_token")
                expires_in = int(data.get("expires_in", 86400))
                self.token_expired_at = now + expires_in - 600  # 10분 마진
                
                tmp_token = f"{TOKEN_FILE}.tmp.{os.getpid()}"
                with open(tmp_token, "w", encoding="utf-8") as f:
                    json.dump({"access_token": self.access_token, "expires_at": self.token_expired_at}, f)
                os.replace(tmp_token, TOKEN_FILE)
                logger.info("🔐 [KIS API] Successfully issued new Access Token.")
                return True
            else:
                logger.error(f"❌ Token issue failed: {resp.text}")
                self.token_expired_at = now + 60
                return False
        except Exception as e:
            logger.error(f"❌ Token request exception: {e}")
            self.token_expired_at = now + 30
            return False

    def get_headers(self, tr_id: str) -> dict:
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.access_token}",
            "appkey": KIS_APP_KEY,
            "appsecret": KIS_APP_SECRET,
            "tr_id": tr_id
        }

    def check_market_clock(self) -> Tuple[bool, str]:
        """KST 기준 정규장 시간 (09:20 ~ 15:15) 및 KRX 휴장일 확인.
        개장 직후 20분간(09:00~09:20)은 호가 스프레드 및 시초가 갭 변동성 안정화 대기"""
        kst_now = datetime.now(timezone(timedelta(hours=9)))
        
        # 1. 주말 체크
        if kst_now.weekday() >= 5:
            return False, f"Weekend ({kst_now.strftime('%A')})"

        # 2. KRX 공휴일/휴장일 체크 (한글날, 대체공휴일 등)
        today_str = kst_now.strftime("%Y-%m-%d")
        if today_str in KRX_HOLIDAYS:
            return False, f"KRX Holiday ({today_str})"
            
        current_time = kst_now.time()
        open_time = datetime.strptime("09:05", "%H:%M").time()  # 09:05 시작으로 개장 5분 변동성 슬리피지 누수 차단
        close_time = datetime.strptime("15:15", "%H:%M").time()
        
        if open_time <= current_time <= close_time:
            return True, "Market Open"
        elif current_time < open_time:
            return False, f"Pre-market (09:05 개장 대기)"
        else:
            return False, f"Market Closed (15:15 종료)"

    def fetch_daily_indicators(self):
        """일 1회 KOSPI 시장 국면 판별 및 유니버스 전 종목의 지표/RS 상대강도 산출"""
        today_str = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
        if self.last_indicator_update == today_str and self.daily_indicators:
            return

        logger.info(f"📊 [Indicator Update] Updating Market Regime & Universe Indicators ({today_str})...")

        # ── [Step 1] KOSPI Market Regime Check (^KS11) ──
        try:
            df_kospi = yf.download("^KS11", period="4mo", interval="1d", progress=False)
            if not df_kospi.empty and len(df_kospi) >= 60:
                if isinstance(df_kospi.columns, pd.MultiIndex):
                    df_kospi.columns = [c[0].lower() for c in df_kospi.columns]
                else:
                    df_kospi.columns = [c.lower() for c in df_kospi.columns]
                df_kospi = df_kospi.dropna()
                now_kst = datetime.now(timezone(timedelta(hours=9)))
                if df_kospi.index[-1].date() >= now_kst.date():
                    df_kospi = df_kospi.iloc[:-1]
                k_close = float(df_kospi["close"].iloc[-1])
                k_sma60 = float(df_kospi["close"].rolling(60).mean().iloc[-1])
                k_ret60 = float((k_close / df_kospi["close"].iloc[-60] - 1.0) * 100.0)
                is_bullish = bool(k_close >= k_sma60)
                self.market_regime = {
                    "is_bullish": is_bullish,
                    "kospi_close": k_close,
                    "kospi_sma60": k_sma60,
                    "kospi_ret60": k_ret60
                }
                regime_badge = "🟢 BULL MARKET (상승 추세 - 신규 매수 허용)" if is_bullish else "🔴 BEAR MARKET (하락 추세 - 신규 매수 전면 차단 / 현금 보존)"
                logger.info(
                    f"🏛️ [KOSPI Market Regime] {regime_badge} | "
                    f"지수: {k_close:,.2f} | 60일선: {k_sma60:,.2f} | 60일수익률: {k_ret60:+.2f}%"
                )
        except Exception as e:
            logger.error(f"❌ Failed to fetch KOSPI regime: {e}")

        kospi_ret60 = self.market_regime.get("kospi_ret60", 0.0)

        for sym, meta in UNIVERSE.items():
            yf_ticker = meta["yf"]
            try:
                # 최근 80 영업일 일봉 다운로드
                df = yf.download(yf_ticker, period="4mo", interval="1d", progress=False)
                if df.empty or len(df) < 30:
                    logger.warning(f"⚠️ Not enough data for {meta['name']} ({yf_ticker})")
                    continue
                    
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0].lower() for c in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]

                df = df.dropna()
                
                # 완성봉만 사용: 마지막 행이 오늘(KST) 날짜면 장중 미완성 봉이므로 제외, 장전이면 어제 완성봉 유지
                now_kst = datetime.now(timezone(timedelta(hours=9)))
                last_bar_date = df.index[-1].date()
                if last_bar_date >= now_kst.date():
                    df = df.iloc[:-1]

                if len(df) < 20:
                    logger.warning(f"⚠️ [{sym}] insufficient completed bars: {len(df)}")
                    continue

                # TR & ATR14
                tr = np.maximum(
                    df["high"] - df["low"],
                    np.maximum(
                        abs(df["high"] - df["close"].shift(1)),
                        abs(df["low"] - df["close"].shift(1))
                    )
                )
                atr14 = float(tr.rolling(14).mean().iloc[-1])
                
                # 이동평균
                sma20 = float(df["close"].rolling(20).mean().iloc[-1])
                sma60 = float(df["close"].rolling(60).mean().iloc[-1]) if len(df) >= 60 else sma20
                
                # 직전 완성봉 기준 20일 최고가 & 10일 최저가
                donchian_high20 = float(df["high"].tail(20).max())
                donchian_low10 = float(df["low"].tail(10).min())

                # [Step 2] 60일 수익률 및 Relative Strength (RS vs KOSPI) 산출
                ret60 = float((df["close"].iloc[-1] / df["close"].iloc[-60] - 1.0) * 100.0) if len(df) >= 60 else 0.0
                rs_score = ret60 - kospi_ret60
                
                self.daily_indicators[sym] = {
                    "donchian_high20": donchian_high20,
                    "donchian_low10": donchian_low10,
                    "sma20": sma20,
                    "sma60": sma60,
                    "atr14": atr14,
                    "ret60": ret60,
                    "rs_score": rs_score,
                    "last_close": float(df["close"].iloc[-1])
                }
                logger.info(
                    f"   ✓ {meta['name']}({sym}) | 20일고가: ₩{donchian_high20:,.0f} | "
                    f"60선: ₩{sma60:,.0f} | ATR: ₩{atr14:,.0f} | 60일수익률: {ret60:+.1f}% | RS: {rs_score:+.1f}%"
                )
            except Exception as e:
                logger.error(f"❌ Failed to calculate indicators for {sym}: {e}")

        self.last_indicator_update = today_str
        logger.info(f"✅ Daily Indicators updated successfully for {len(self.daily_indicators)} stocks.")

        # [동적 ATR 보완] 보유 포지션의 ATR을 최신 변동성으로 동기화 (진입 시점 고정 결함 해소)
        pos_atr_updated = False
        for sym, pos in self.positions.items():
            if sym in self.daily_indicators:
                latest_atr = self.daily_indicators[sym].get("atr14", 0.0)
                old_atr = pos.get("atr", 0.0)
                if latest_atr > 0 and abs(old_atr - latest_atr) > 1.0:
                    pos["atr"] = latest_atr
                    pos_atr_updated = True
                    logger.info(f"🔄 [ATR Dynamically Updated] {UNIVERSE.get(sym, {}).get('name', sym)}({sym}): ₩{old_atr:,.0f} ➜ ₩{latest_atr:,.0f}")
        if pos_atr_updated:
            self.save_positions()

    def fetch_account_balance(self):
        """계좌 총평가액(Equity) 및 보유 종목 실시간 동기화"""
        if not self.issue_token():
            return

        url = f"{KIS_URL}/uapi/domestic-stock/v1/trading/inquire-balance"
        headers = self.get_headers("TTTC8434R")
        params = {
            "CANO": KIS_CANO,
            "ACNT_PRDT_CD": KIS_PRDT_ABRV,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }
        try:
            resp = session.get(url, headers=headers, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("rt_cd") != "0":
                    logger.warning(f"⚠️ Balance fetch failed: {data.get('msg1')}")
                    return

                output2 = data.get("output2", [])
                if output2:
                    equity = float(output2[0].get("tot_evlu_amt", 0))
                    true_cash = self.fetch_available_cash()
                    self.account_info["equity"] = max(equity, true_cash)

                output1 = data.get("output1", [])
                active_pdnos = set()
                updated = False

                for item in output1:
                    pdno = item.get("pdno", "")
                    if pdno in EXCLUDED_SYMBOLS:
                        continue
                    try:
                        hldg_qty = int(item.get("hldg_qty", 0))
                        pchs_avg = float(item.get("pchs_avg_pric", 0))
                    except (ValueError, TypeError):
                        continue

                    if pdno in UNIVERSE and hldg_qty > 0:
                        active_pdnos.add(pdno)
                        if pdno not in self.positions:
                            # 새 포지션 등록
                            atr = self.daily_indicators.get(pdno, {}).get("atr14", pchs_avg * 0.03)
                            self.positions[pdno] = {
                                "qty": hldg_qty,
                                "entry_price": pchs_avg,
                                "entry_date": datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d"),
                                "peak_price": pchs_avg,
                                "atr": atr
                            }
                            updated = True
                            logger.info(f"📥 [Position Registered] {UNIVERSE[pdno]['name']}({pdno}) | {hldg_qty}주 @ ₩{pchs_avg:,.0f}")
                        else:
                            pos = self.positions[pdno]
                            if pos.get("qty") != hldg_qty:
                                pos["qty"] = hldg_qty
                                pos["entry_price"] = pchs_avg
                                updated = True

                # 계좌에서 청산된 종목 동기화
                for sym in list(self.positions.keys()):
                    if sym not in active_pdnos:
                        logger.info(f"📤 [Position Cleared] {UNIVERSE.get(sym, {}).get('name', sym)}({sym}) no longer in account.")
                        self.positions.pop(sym, None)
                        updated = True

                if updated:
                    self.save_positions()
                    
                logger.info(f"💼 [Account Status] Equity: ₩{self.account_info['equity']:,.0f} | Active Positions: {len(self.positions)}/{MAX_CONCURRENT_POSITIONS}")
        except Exception as e:
            logger.error(f"❌ Balance fetch exception: {e}")

    def fetch_available_cash(self, symbol: str = "005930") -> float:
        """inquire-psbl-order (TTTC8908R) 호출을 통한 실제 주문가능현금 확인
        (매도대금 재사용이 포함된 미수 없는 순수 매수가능금액 nrcvb_buy_amt 우선 적용)"""
        if not self.issue_token():
            return 0.0

        url = f"{KIS_URL}/uapi/domestic-stock/v1/trading/inquire-psbl-order"
        headers = self.get_headers("TTTC8908R")
        params = {
            "CANO": KIS_CANO,
            "ACNT_PRDT_CD": KIS_PRDT_ABRV,
            "PDNO": symbol,
            "ORD_UNPR": "0",
            "ORD_DVSN": "01",
            "CMA_EVLU_AMT_ICLD_YN": "Y",
            "OVRS_ICLD_YN": "Y"
        }
        try:
            resp = session.get(url, headers=headers, params=params, timeout=4)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("rt_cd") == "0" and "output" in data:
                    out = data["output"]
                    # [개선] 통합증거금/외화대용 주문가능금액(max_buy_amt/nrcvb_buy_amt) 우선 활용
                    cash = float(out.get("max_buy_amt") or out.get("nrcvb_buy_amt") or out.get("ord_psbl_cash", 0))
                    self.account_info["cash"] = cash
                    self.account_info["equity"] = max(self.account_info.get("equity", 0.0), cash)
                    logger.info(f"💵 [Buying Power Checked] 주문가능총액: ₩{cash:,.0f} (순수현금: ₩{float(out.get('ord_psbl_cash', 0)):,.0f})")
                    return cash
        except Exception as e:
            logger.error(f"❌ Failed to fetch true buying cash: {e}")
        return 0.0

    def get_current_price(self, symbol: str) -> float:
        """KIS 실시간 현재가 조회"""
        if not self.issue_token():
            return 0.0
            
        url = f"{KIS_URL}/uapi/domestic-stock/v1/quotations/inquire-price"
        headers = self.get_headers("FHKST01010100")
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol
        }
        try:
            resp = session.get(url, headers=headers, params=params, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if "output" in data:
                    return float(data["output"].get("stck_prpr", 0))
        except Exception as e:
            logger.error(f"❌ Price fetch error for {symbol}: {e}")
        return 0.0

    def submit_kis_order(self, symbol: str, side: str, qty: int, price: float = 0.0) -> bool:
        """주문 발주 (시장가 01 또는 지정가 00)"""
        if not self.issue_token():
            return False

        url = f"{KIS_URL}/uapi/domestic-stock/v1/trading/order-cash"
        tr_id = "TTTC0802U" if side.lower() == "buy" else "TTTC0801U"
        headers = self.get_headers(tr_id)
        
        ord_dvsn = "01" if price == 0.0 else "00"
        payload = {
            "CANO": KIS_CANO,
            "ACNT_PRDT_CD": KIS_PRDT_ABRV,
            "PDNO": symbol,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(qty),
            "ORD_UNPR": str(int(price))
        }
        try:
            resp = session.post(url, headers=headers, json=payload, timeout=8)
            data = resp.json()
            if data.get("rt_cd") == "0":
                logger.info(f"✅ [Order Sent] {side.upper()} {qty}주 {UNIVERSE.get(symbol, {}).get('name', symbol)}({symbol}) | {data.get('msg1')}")
                return True
            else:
                logger.error(f"❌ [Order Failed] {side.upper()} {symbol}: {data.get('msg1')} ({data.get('msg_cd')})")
                return False
        except Exception as e:
            logger.error(f"❌ Order exception {symbol}: {e}")
            return False

    def manage_positions(self):
        """보유 포지션 출구 전략 (ATR 트레일링 스탑 & 하드 손절)"""
        for sym, pos in list(self.positions.items()):
            if sym in EXCLUDED_SYMBOLS:
                self.positions.pop(sym, None)
                continue

            current_price = self.get_current_price(sym)
            if current_price <= 0:
                continue

            entry_price = float(pos.get("entry_price") or pos.get("avg_price", 0.0))
            if entry_price <= 0:
                continue
            pos["entry_price"] = entry_price
            peak_price = max(float(pos.get("peak_price", entry_price)), current_price)
            pos["peak_price"] = peak_price
            
            atr = float(pos.get("atr", self.daily_indicators.get(sym, {}).get("atr14", entry_price * 0.03)))
            
            # 1. ATR Trailing Stop (고점 대비 2.0x ATR 하락)
            trail_stop_level = peak_price - (ATR_TRAILING_MULTIPLIER * atr)
            # 2. Hard Stop Loss (진입가 대비 -7% 또는 2.5x ATR)
            hard_stop_level = max(entry_price * (1.0 - HARD_STOP_LOSS_PCT), entry_price - (2.5 * atr))
            # 3. 10일 최저가 이탈
            donchian_low10 = self.daily_indicators.get(sym, {}).get("donchian_low10", 0.0)

            effective_stop = max(trail_stop_level, donchian_low10)
            
            pnl_pct = (current_price / entry_price - 1.0) * 100.0
            
            should_exit = False
            reason = ""

            if current_price < hard_stop_level:
                should_exit = True
                reason = f"HARD_STOP ({pnl_pct:+.2f}%)"
            elif current_price < effective_stop and current_price < peak_price:
                should_exit = True
                reason = f"ATR_TRAIL_STOP ({pnl_pct:+.2f}%, 고점 ₩{peak_price:,.0f})"

            if should_exit:
                qty = int(pos["qty"])
                logger.info(f"🔄 [Exit Signal Triggered] {UNIVERSE.get(sym, {}).get('name', sym)}({sym}) | Reason: {reason} | Qty: {qty}")
                success = self.submit_kis_order(sym, "sell", qty, price=0.0)
                if success:
                    self.positions.pop(sym, None)
                    self.save_positions()

    def rebalance_underallocated_positions(self):
        """[Step 1] 비중 미달 보유 포지션 리밸런싱 / 비중 보충 (Top-up)
        예: 한미반도체(042700) 등 과거 예수금 계산 오류로 1주만 매수된 포지션을
        목표 비중(30%, 약 145만 원)까지 D+2 가용현금으로 추가 매수"""
        if not self.market_regime.get("is_bullish", True):
            return

        equity = float(self.account_info.get("equity", 0.0))
        if equity <= 0:
            return

        target_size = equity * MAX_POSITION_PCT  # 포지션당 목표 금액 (약 30%)

        for sym, pos in list(self.positions.items()):
            if sym in EXCLUDED_SYMBOLS:
                continue

            current_price = self.get_current_price(sym)
            if current_price <= 0:
                continue

            current_qty = int(pos.get("qty", 0))
            current_val = current_qty * current_price

            # 목표 비중의 65% 미만으로 크게 미달된 포지션인 경우
            if current_val < (target_size * 0.65):
                old_entry = float(pos.get("entry_price", current_price))
                ind = self.daily_indicators.get(sym, {})
                sma60 = ind.get("sma60", 0.0)

                # [피라미딩 안전판 1] 하락/손실 중인 종목에 대한 역추세 물타기 전면 차단 (오직 수익 중인 승자에만 추가)
                if current_price < old_entry:
                    logger.info(
                        f"🛡️ [Top-Up Guard] {UNIVERSE.get(sym, {}).get('name', sym)}({sym}): "
                        f"현재가(₩{current_price:,.0f})가 평단가(₩{old_entry:,.0f}) 이하로 손실 중이므로 비중 보충 스킵 (역추세 물타기 방지)"
                    )
                    continue

                # [피라미딩 안전판 2] 60일선 하회 종목 추가 매수 차단
                if sma60 > 0 and current_price < sma60:
                    logger.info(
                        f"🛡️ [Top-Up Guard] {UNIVERSE.get(sym, {}).get('name', sym)}({sym}): "
                        f"현재가(₩{current_price:,.0f})가 60일선(₩{sma60:,.0f}) 아래에 있어 비중 보충 스킵"
                    )
                    continue

                available_cash = self.fetch_available_cash(sym)
                if available_cash < 500_000:
                    logger.info(
                        f"ℹ️ [Top-Up Skip] {UNIVERSE.get(sym, {}).get('name', sym)}({sym}): "
                        f"가용현금 부족 (₩{available_cash:,.0f} < ₩500,000)"
                    )
                    continue

                deficit_amt = min(target_size - current_val, available_cash * 0.85)
                add_qty = int(deficit_amt / current_price)

                if add_qty >= 1:
                    logger.info(
                        f"⚖️ [Position Top-Up] {UNIVERSE.get(sym, {}).get('name', sym)}({sym}) 비중 보충 매수: "
                        f"현재 {current_qty}주(₩{current_val:,.0f}, {current_val/equity*100:.1f}%) -> "
                        f"+{add_qty}주 추가 주문 (목표 ₩{target_size:,.0f})"
                    )
                    success = self.submit_kis_order(sym, "buy", add_qty, price=0.0)
                    if success:
                        old_qty = current_qty
                        old_entry = float(pos.get("entry_price", current_price))
                        new_qty = old_qty + add_qty
                        new_avg = ((old_qty * old_entry) + (add_qty * current_price)) / new_qty

                        pos["qty"] = new_qty
                        pos["entry_price"] = new_avg
                        self.save_positions()
                        logger.info(
                            f"✅ [Top-Up Complete] {UNIVERSE.get(sym, {}).get('name', sym)}({sym}) "
                            f"보유수량: {new_qty}주 | 갱신 평단: ₩{new_avg:,.0f}"
                        )
                        time.sleep(1)

    def evaluate_entry_opportunities(self):
        """[Step 1 & Step 2] KOSPI 장세 필터 + RS 모멘텀 순위 기반 돈키언 20일 신고가 돌파 진입"""
        # 1. KOSPI 시장 필터 (60일선 하회 시 전면 매수 금지, 100% 현금 방어)
        if not self.market_regime.get("is_bullish", True):
            logger.info("🛡️ [Regime Guard] KOSPI 지수가 60일선 아래(하락장)에 있습니다. 신규 진입을 전면 차단하고 100% 현금을 보존합니다.")
            return

        # 2. 비중 미달 포지션 보충 우선 실행 (Step 1 Top-up)
        self.rebalance_underallocated_positions()

        active_count = len(self.positions)
        if active_count >= MAX_CONCURRENT_POSITIONS:
            return

        # 주문가능현금 실시간 확인
        available_cash = self.fetch_available_cash()
        if available_cash < 500_000:  # 최소 50만 원 미만이면 신규 진입 불가
            return

        equity = float(self.account_info.get("equity", 10_000_000))
        target_size = min(equity * MAX_POSITION_PCT, available_cash * 0.85) # 상한가 증거금 완충 85%

        candidates = []
        for sym, meta in UNIVERSE.items():
            if sym in self.positions or sym in EXCLUDED_SYMBOLS:
                continue

            ind = self.daily_indicators.get(sym)
            if not ind:
                continue

            current_price = self.get_current_price(sym)
            if current_price <= 0:
                continue

            donchian_high = ind["donchian_high20"]
            sma60 = ind["sma60"]
            sma20 = ind["sma20"]

            # 진입 조건: 20일 신고가 돌파 + 60일선 위 + 20일선 >= 60일선 정배열
            is_breakout = (current_price > donchian_high)
            is_trend_aligned = (current_price > sma60) and (sma20 >= sma60)

            if is_breakout and is_trend_aligned:
                rs_score = ind.get("rs_score", 0.0)
                momentum_dist = (current_price - sma60) / sma60
                # 종합 주도주 랭킹 점수: RS 상대강도(70%) + 60선 이격도(30%)
                total_score = (rs_score * 0.7) + (momentum_dist * 100.0 * 0.3)
                candidates.append((sym, meta["name"], current_price, total_score, rs_score, ind["atr14"]))

        # [Step 2] 상대강도(RS) & 모멘텀 최상위 주도주 우선 정렬
        candidates.sort(key=lambda x: x[3], reverse=True)

        for sym, name, price, total_score, rs_score, atr in candidates:
            if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
                break

            qty = int(target_size / price)
            if qty < 1:
                continue

            logger.info(
                f"🎯 [Leader Breakout Detected] {name}({sym}) | 돌파가: ₩{price:,.0f} | "
                f"RS(시장초과): {rs_score:+.1f}% | 종합점수: {total_score:.1f} | 진입수량: {qty}주"
            )
            success = self.submit_kis_order(sym, "buy", qty, price=0.0)
            if success:
                self.positions[sym] = {
                    "qty": qty,
                    "entry_price": price,
                    "entry_date": datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d"),
                    "peak_price": price,
                    "atr": atr
                }
                self.save_positions()
                # 1개 매수 후 재계산
                available_cash = self.fetch_available_cash()
                target_size = min(equity * MAX_POSITION_PCT, available_cash * 0.85)

    def run_cycle(self):
        is_open, status_msg = self.check_market_clock()
        if not is_open:
            now = time.time()
            if now - self.last_market_status_log > 600:
                kst = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"🌙 [{status_msg}] Current KST: {kst} | 정규장(09:20~15:15) 대기 중...")
                self.last_market_status_log = now
            time.sleep(60)
            return

        # 1. 일별 지표 갱신 (일 1회)
        self.fetch_daily_indicators()
        # 2. 계좌 잔고 동기화
        self.fetch_account_balance()
        # 3. 보유 종목 추세 관리 (ATR 트레일링 스탑)
        self.manage_positions()
        # 4. 신규 돌파 종목 탐색 및 진입
        self.evaluate_entry_opportunities()

    def start(self):
        logger.info("🟢 [Daemon Started] Entering autonomous trend following loop...")
        while True:
            try:
                self.run_cycle()
                time.sleep(POLL_INTERVAL_SEC)
            except KeyboardInterrupt:
                logger.info("🛑 [Stop Signal] Stopping KIS Trend Trader gracefully.")
                break
            except Exception as e:
                logger.error(f"❌ [Main Loop Exception] {e}")
                time.sleep(15)


if __name__ == "__main__":
    trader = KisTrendTrader()
    trader.start()
