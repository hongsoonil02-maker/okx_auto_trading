# -*- coding: utf-8 -*-
"""
kis_nasdaq_regime_trader.py — KIS 미국 지수 ETF 200일선 레짐 퀀트 엔진
전략 (메비우스 파베르 Ivy Portfolio / Trend Following 정통 모델):
- 유니버스: QQQ (나스닥 100 지수 ETF) — 생존자 편향 0%, 개별주 어닝쇼크 갭다운 리스크 0%
- 레짐 게이트: QQQ 전일 완성봉 종가 >= QQQ 200일 이동평균선 (SMA200)
  * 🟢 BULL: QQQ 매수 및 100% 보유 (장기 우상향 복리 향유)
  * 🔴 BEAR: QQQ 전량 매도 및 달러 현금 100% 보존 (2008년 금융위기, 2022년 인플레이션 폭락장 완전 회피)
- 레거시 포지션(AAPL, NVDA, TSLA 등) 자동 리밸런싱:
  * 봇 기동 후 첫 액션 윈도우(10:00 ET)에서 레거시 개별주를 순차 매도하고 QQQ로 안전하게 일원화
- 실행 인프라:
  * KIS REST API (원장 대사 TTTS3012R, 실시간 시세 HHDFS00000300, 주문 TTTT1002U/1006U)
  * 개장 30분 변동성 회피 (10:00~15:30 ET 액션 윈도우)
  * 미국 서머타임(DST) 자동 추적 (ZoneInfo) + NYSE 연간 휴장일 캘린더
  * 원자적 파일 I/O 및 회전식 로깅
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, "..", ".env"))

# ── LOGGING ──
LOG_FILE = os.path.join(CURRENT_DIR, "kis_nasdaq_regime_trader.log")
logger = logging.getLogger("KisNasdaqRegimeTrader")
logger.setLevel(logging.INFO)
logger.propagate = False
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=3, encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ── HTTP SESSION ──
session = requests.Session()
retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
                       allowed_methods=["HEAD", "GET", "OPTIONS", "POST"])
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

# ── CREDENTIALS ──
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
TOKEN_FILE = os.path.join(CURRENT_DIR, "kis_token.json")
STATE_FILE = os.path.join(CURRENT_DIR, "kis_nasdaq_regime_state.json")

# ── STRATEGY CONFIG ──
TARGET_TICKER = "QQQ"             # 주력 지수 ETF
ALLOCATION_PCT = 0.98             # 가용 자산의 98% 투입 (여유 버퍼 2%)
POLL_INTERVAL_SEC = 60            # 감시 주기 60초
PENDING_TIMEOUT_SEC = 60.0        # 미체결 타임아웃

# 액션 허용 윈도우 (개장 30분 변동성 회피)
ACTION_START_ET = (10, 0)         # 10:00 ET
ACTION_END_ET = (15, 30)          # 15:30 ET

# NYSE 휴장일 (2026-2027)
US_MARKET_HOLIDAYS = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}


class KisNasdaqRegimeTrader:
    def __init__(self):
        self.access_token = ""
        self.token_expired_at = 0
        self.account_info = {"usd_cash": 0.0, "krw_cash": 0.0, "fx_rate": 0.0, "total_buying_power_usd": 0.0}
        self.holdings: Dict[str, dict] = {}       # {symbol: {"qty": int, "avg_price": float}}
        self.pending_orders: Dict[str, dict] = {} # {symbol: {"side": str, "odno": str, "ts": float, "qty": int}}
        
        self.regime_bullish = False
        self.qqq_last_close = 0.0
        self.qqq_sma200 = 0.0
        self.last_indicator_date = ""
        self.last_status_log = 0.0
        self.last_reconcile_time = 0.0

        self.load_state()

        logger.info("=" * 75)
        logger.info(f"🚀 [KIS Nasdaq Regime Trader v1.0] QQQ 200-Day SMA Regime Engine")
        logger.info(f"   Target: {TARGET_TICKER} (Nasdaq 100 ETF) | Alloc: {ALLOCATION_PCT*100:.0f}%")
        logger.info(f"   Strategy: Close >= SMA200 ➜ 🟢 Bull (100% Invested) | Close < SMA200 ➜ 🔴 Bear (100% Cash)")
        logger.info(f"   Legacy Stocks: Auto-rebalanced to {TARGET_TICKER} during action window")
        logger.info(f"   Action Window: {ACTION_START_ET[0]:02d}:{ACTION_START_ET[1]:02d}~{ACTION_END_ET[0]:02d}:{ACTION_END_ET[1]:02d} ET")
        logger.info("=" * 75)

    # ── 상태 영속화 (atomic) ──
    def load_state(self):
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.pending_orders = data.get("pending_orders", {})
                logger.info(f"📂 [State Restored] Pending orders: {list(self.pending_orders.keys())}")
        except Exception as e:
            logger.warning(f"⚠️ State load error: {e}")

    def save_state(self):
        try:
            tmp = f"{STATE_FILE}.tmp.{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"pending_orders": self.pending_orders}, f, ensure_ascii=False, indent=2)
            os.replace(tmp, STATE_FILE)
        except Exception as e:
            logger.error(f"❌ Failed to save state: {e}")

    # ── 토큰 관리 ──
    def issue_token(self) -> bool:
        now = time.time()
        if self.access_token and now < self.token_expired_at:
            return True
        if not self.access_token and os.path.exists(TOKEN_FILE):
            try:
                with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("expires_at", 0) > now + 300:
                    self.access_token = data["access_token"]
                    self.token_expired_at = data["expires_at"]
                    return True
            except Exception as e:
                logger.warning(f"⚠️ Token cache read error: {e}")
        try:
            resp = session.post(f"{KIS_URL}/oauth2/tokenP",
                                json={"grant_type": "client_credentials", "appkey": KIS_APP_KEY, "appsecret": KIS_APP_SECRET},
                                timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                self.access_token = data.get("access_token", "")
                self.token_expired_at = now + int(data.get("expires_in", 86400))
                tmp = f"{TOKEN_FILE}.tmp.{os.getpid()}"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump({"access_token": self.access_token, "expires_at": self.token_expired_at}, f)
                os.replace(tmp, TOKEN_FILE)
                logger.info("🔑 [Auth] KIS Access Token Issued & Cached (atomic).")
                return True
            logger.error(f"❌ Token failed: HTTP {resp.status_code} - {resp.text}")
            return False
        except Exception as e:
            logger.error(f"❌ Token exception: {e}")
            return False

    def get_auth_headers(self, tr_id: str) -> dict:
        self.issue_token()
        return {"content-type": "application/json", "authorization": f"Bearer {self.access_token}",
                "appkey": KIS_APP_KEY, "appsecret": KIS_APP_SECRET, "tr_id": tr_id, "custtype": "P"}

    # ── 시장 시계 ──
    def get_market_state(self) -> str:
        """Returns 'CLOSED' | 'WATCH' | 'ACTION'"""
        now_ny = datetime.now(ZoneInfo("America/New_York"))
        if now_ny.weekday() >= 5:
            return "CLOSED"
        if now_ny.strftime("%Y-%m-%d") in US_MARKET_HOLIDAYS:
            return "CLOSED"
        minutes = now_ny.hour * 60 + now_ny.minute
        open_m, close_m = 9 * 60 + 30, 16 * 60
        if not (open_m <= minutes <= close_m):
            return "CLOSED"
        act_start = ACTION_START_ET[0] * 60 + ACTION_START_ET[1]
        act_end = ACTION_END_ET[0] * 60 + ACTION_END_ET[1]
        return "ACTION" if act_start <= minutes <= act_end else "WATCH"

    # ── QQQ 200일선 레짐 판별 (하루 1회) ──
    def update_regime_indicator(self):
        now_ny = datetime.now(ZoneInfo("America/New_York"))
        today_et = now_ny.strftime("%Y-%m-%d")
        if self.last_indicator_date == today_et and self.qqq_sma200 > 0:
            return

        logger.info(f"📊 [Regime Check] Fetching daily bars for {TARGET_TICKER}...")
        try:
            df = yf.Ticker(TARGET_TICKER).history(period="2y", interval="1d")
            if df is None or len(df) < 205:
                logger.error("❌ Insufficient historical bars for QQQ SMA200")
                return

            # 완성봉 기준: 마지막 행이 오늘 날짜이면 장중 미완성 봉이므로 제외
            if df.index[-1].date() >= now_ny.date():
                df = df.iloc[:-1]

            close = float(df["Close"].iloc[-1])
            sma200 = float(df["Close"].rolling(200).mean().iloc[-1])
            is_bull = close >= sma200

            self.qqq_last_close = close
            self.qqq_sma200 = sma200
            self.regime_bullish = is_bull
            self.last_indicator_date = today_et

            status_str = "🟢 BULL (상승장 - QQQ 보유)" if is_bull else "🔴 BEAR (하락장 - 현금 100% 보존)"
            logger.info(f"🏛️ [Macro Regime] QQQ 전일종가 ${close:.2f} vs SMA200 ${sma200:.2f} ➜ {status_str}")
        except Exception as e:
            logger.error(f"❌ Failed to update regime indicator: {e}")

    # ── 환율 ──
    def get_usd_krw_rate(self) -> float:
        if not hasattr(self, "_cached_fx"):
            self._cached_fx = (0.0, 1380.0)
        now = time.time()
        if now - self._cached_fx[0] < 3600.0 and self._cached_fx[1] > 0:
            return self._cached_fx[1]
        try:
            df = yf.Ticker("USDKRW=X").history(period="1d", interval="5m")
            if df is not None and not df.empty:
                rate = float(df["Close"].iloc[-1])
                if rate > 900.0:
                    self._cached_fx = (now, rate)
                    return rate
        except Exception:
            pass
        self._cached_fx = (now, 1380.0)
        return 1380.0

    # ── 원장 대사 및 계좌 현황 ──
    def reconcile_account(self):
        # 1. USD 예수금 조회 (TTTS3007R)
        usd_cash = 0.0
        try:
            res_u = session.get(f"{KIS_URL}/uapi/overseas-stock/v1/trading/inquire-present-balance",
                                headers=self.get_auth_headers("TTTS3007R"),
                                params={"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "WCRC_FRCR_DVS_CD": "02",
                                        "NATN_CD": "840", "TR_MKET_CD": "00", "INQR_DVS_CD": "00"},
                                timeout=10)
            if res_u.status_code == 200:
                for row in res_u.json().get("output1", []):
                    if row.get("crcy_cd") == "USD":
                        usd_cash = float(row.get("frcr_dnca_amt", 0.0))
        except Exception as e:
            logger.warning(f"⚠️ USD cash fetch error: {e}")

        # 2. KRW 통합증거금 조회 (TTTC8908R)
        krw_orderable = 0.0
        try:
            res = session.get(f"{KIS_URL}/uapi/domestic-stock/v1/trading/inquire-psbl-order",
                              headers=self.get_auth_headers("TTTC8908R"),
                              params={"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "PDNO": "005930",
                                      "ORD_UNPR": "0", "ORD_DVSN": "01", "CMA_EVLU_AMT_ICLD_YN": "Y", "OVRS_ICLD_YN": "Y"},
                              timeout=10)
            if res.status_code == 200:
                out = res.json().get("output", {})
                krw_orderable = float(out.get("max_buy_amt", out.get("ord_psbl_cash", 0.0)))
        except Exception as e:
            logger.warning(f"⚠️ KRW orderable fetch error: {e}")

        fx = self.get_usd_krw_rate()
        total_bp = usd_cash + (krw_orderable / fx if krw_orderable > 0 else 0.0)
        self.account_info.update({"usd_cash": usd_cash, "krw_cash": krw_orderable, "fx_rate": fx,
                                  "total_buying_power_usd": total_bp})

        # 3. 해외주식 실제 보유 잔고 대사 (TTTS3012R)
        try:
            res_b = session.get(f"{KIS_URL}/uapi/overseas-stock/v1/trading/inquire-balance",
                                headers=self.get_auth_headers("TTTS3012R"),
                                params={"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "OVRS_EXCG_CD": "NASD",
                                        "TR_CRCY_CD": "USD", "CTX_AREA_FK200": "", "CTX_AREA_NK200": ""},
                                timeout=10)
            if res_b.status_code == 200 and res_b.json().get("rt_cd") == "0":
                real_holdings = {}
                for row in res_b.json().get("output1", []):
                    sym = row.get("ovrs_pdno", "").strip().upper()
                    try:
                        qty = int(float(row.get("ovrs_cblc_qty", 0)))
                        avg_px = float(row.get("pchs_avg_pric", 0.0))
                    except (ValueError, TypeError):
                        continue
                    if sym and qty > 0:
                        real_holdings[sym] = {"qty": qty, "avg_price": avg_px}

                # Pending 주문 체결 확인
                for sym in list(self.pending_orders.keys()):
                    p_info = self.pending_orders[sym]
                    side = p_info.get("side")
                    if side == "sell" and sym not in real_holdings:
                        logger.info(f"✅ [Exit Confirmed] {sym} 전량 매도 체결 확인")
                        self.pending_orders.pop(sym, None)
                        self.save_state()
                    elif side == "buy" and sym in real_holdings:
                        logger.info(f"✅ [Entry Confirmed] {sym} 매수 체결 확인 ({real_holdings[sym]['qty']}주)")
                        self.pending_orders.pop(sym, None)
                        self.save_state()

                self.holdings = real_holdings
        except Exception as e:
            logger.error(f"❌ Balance reconciliation error: {e}")

    # ── 실시간 시세 ──
    def get_realtime_price(self, symbol: str) -> Optional[float]:
        try:
            res = session.get(f"{KIS_URL}/uapi/overseas-price/v1/quotations/price",
                              headers=self.get_auth_headers("HHDFS00000300"),
                              params={"AUTH": "", "EXCD": "NAS", "SYMB": symbol}, timeout=5)
            if res.status_code == 200:
                last = float(res.json().get("output", {}).get("last", 0.0))
                if last > 0:
                    return last
        except Exception as e:
            logger.error(f"❌ [{symbol}] Quote error: {e}")
        return None

    # ── 주문 집행 ──
    def submit_order(self, symbol: str, side: str, qty: int, price: float) -> Optional[str]:
        if qty < 1:
            return None
        tr_id = "TTTT1002U" if side.lower() == "buy" else "TTTT1006U"
        body = {"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "OVRS_EXCG_CD": "NASD", "PDNO": symbol,
                "ORD_QTY": str(qty), "OVRS_ORD_UNPR": f"{price:.2f}", "ORD_SVR_DVSN_CD": "0", "ORD_DVSN": "00"}
        try:
            logger.info(f"📤 [Order Dispatch] {side.upper()} {symbol} x {qty} @ ${price:.2f} (TR: {tr_id})")
            res = session.post(f"{KIS_URL}/uapi/overseas-stock/v1/trading/order",
                                headers=self.get_auth_headers(tr_id), json=body, timeout=10)
            data = res.json()
            if res.status_code == 200 and data.get("rt_cd") == "0":
                odno = data.get("output", {}).get("ODNO", "")
                logger.info(f"✅ [Order Accepted] {side.upper()} {symbol} x {qty} | ODNO: {odno}")
                return odno if odno else "UNKNOWN"
            logger.error(f"❌ [Order Rejected] {side.upper()} {symbol} | {data.get('msg_cd')}: {data.get('msg1')}")
            return None
        except Exception as e:
            logger.error(f"❌ [Order Exception] {symbol}: {e}")
            return None

    def cancel_order(self, symbol: str, odno: str, qty: int) -> bool:
        if not odno or odno == "UNKNOWN":
            return False
        body = {"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "OVRS_EXCG_CD": "NASD", "PDNO": symbol,
                "ORG_ODNO": str(odno), "RVSE_CNCL_DVSN_CD": "02", "ORD_QTY": str(qty),
                "OVRS_ORD_UNPR": "0", "ORD_SVR_DVSN_CD": "0"}
        try:
            res = session.post(f"{KIS_URL}/uapi/overseas-stock/v1/trading/order-rvsecncl",
                               headers=self.get_auth_headers("TTTT1004U"), json=body, timeout=10)
            data = res.json()
            if res.status_code == 200 and data.get("rt_cd") == "0":
                logger.info(f"🚫 [Order Cancelled] {symbol} (ODNO: {odno})")
                return True
            logger.warning(f"⚠️ [Cancel Failed] {symbol} | {data.get('msg_cd')}: {data.get('msg1')}")
            return False
        except Exception as e:
            logger.error(f"❌ [Cancel Exception] {symbol}: {e}")
            return False

    # ── 미체결 관리 ──
    def manage_pending_orders(self):
        now = time.time()
        for sym, p in list(self.pending_orders.items()):
            if now - p.get("ts", 0.0) >= PENDING_TIMEOUT_SEC:
                odno = p.get("odno", "")
                qty = p.get("qty", 1)
                logger.warning(f"⏱️ [Order Timeout] {sym} ({p.get('side')}) 미체결 타임아웃 — 취소 접수")
                self.cancel_order(sym, odno, qty)
                self.pending_orders.pop(sym, None)
                self.save_state()

    # ── 핵심 리밸런싱 실행 로직 ──
    def execute_rebalance(self):
        self.manage_pending_orders()

        # Step 1: 레거시 개별주 (AAPL, NVDA, TSLA 등) 정리
        # QQQ 지수 ETF 체제로 일원화하기 위해 레거시 보유 종목 순차 매도
        legacy_symbols = [s for s in self.holdings.keys() if s != TARGET_TICKER]
        if legacy_symbols:
            for sym in legacy_symbols:
                if sym in self.pending_orders:
                    continue
                qty = self.holdings[sym]["qty"]
                px = self.get_realtime_price(sym)
                if not px:
                    continue
                logger.info(f"🔄 [Legacy Rebalance] {sym} {qty}주 매도 접수 (지수 ETF 체제 전환)")
                odno = self.submit_order(sym, "sell", qty, px)
                if odno:
                    self.pending_orders[sym] = {"side": "sell", "odno": odno, "ts": time.time(), "qty": qty}
                    self.save_state()
            # 레거시 매도가 진행 중이면 매도 체결 후 현금이 입금될 때까지 QQQ 매수는 다음 주기로 대기
            return

        # Step 2: 레짐에 따른 QQQ 포지션 관리
        qqq_held = TARGET_TICKER in self.holdings
        qqq_qty = self.holdings.get(TARGET_TICKER, {}).get("qty", 0)

        if not self.regime_bullish:
            # 🔴 BEAR MARKET: QQQ 전량 매도 ➜ 100% 현금 대피
            if qqq_held and qqq_qty > 0 and TARGET_TICKER not in self.pending_orders:
                px = self.get_realtime_price(TARGET_TICKER)
                if px:
                    logger.warning(f"🚨 [Bear Regime Exit] QQQ 200일선 하회 — 전량 매도 접수 ({qqq_qty}주 @ ${px:.2f})")
                    odno = self.submit_order(TARGET_TICKER, "sell", qqq_qty, px)
                    if odno:
                        self.pending_orders[TARGET_TICKER] = {"side": "sell", "odno": odno, "ts": time.time(), "qty": qqq_qty}
                        self.save_state()
            return

        # 🟢 BULL MARKET: QQQ 매수 및 유지
        if self.regime_bullish:
            if TARGET_TICKER in self.pending_orders:
                return
            bp = self.account_info.get("total_buying_power_usd", 0.0)
            px = self.get_realtime_price(TARGET_TICKER)
            if not px or px <= 0:
                return

            target_funds = bp * ALLOCATION_PCT
            buy_qty = int(target_funds / px)

            # 가용 자금으로 1주 이상 매수 가능할 때 신규/추가 매수
            if buy_qty >= 1:
                logger.info(f"🎯 [Bull Regime Entry/Top-up] QQQ {buy_qty}주 매수 접수 @ ${px:.2f} (가용구매력: ${bp:.2f})")
                odno = self.submit_order(TARGET_TICKER, "buy", buy_qty, px)
                if odno:
                    self.pending_orders[TARGET_TICKER] = {"side": "buy", "odno": odno, "ts": time.time(), "qty": buy_qty}
                    self.save_state()

    # ── 메인 루프 ──
    def run_cycle(self):
        state = self.get_market_state()
        now = time.time()
        self.update_regime_indicator()

        # 5분 주기 원장 대사
        if now - self.last_reconcile_time >= 300.0:
            self.reconcile_account()
            self.last_reconcile_time = now

        if state == "CLOSED":
            if now - self.last_status_log > 1800:
                bp = self.account_info.get("total_buying_power_usd", 0.0)
                reg_str = "🟢Bull" if self.regime_bullish else "🔴Bear"
                held_str = ", ".join(f"{s}({info['qty']}주)" for s, info in self.holdings.items()) or "None"
                logger.info(f"🌙 [Market Closed] Regime: {reg_str} | Buying Power: ${bp:.2f} | Holdings: [{held_str}]")
                self.last_status_log = now
            return

        # 액션 윈도우 (10:00 ~ 15:30 ET) 내에서만 리밸런싱 실행
        if state == "ACTION":
            self.execute_rebalance()

        if now - self.last_status_log > 300:
            bp = self.account_info.get("total_buying_power_usd", 0.0)
            reg_str = "🟢Bull" if self.regime_bullish else "🔴Bear"
            held_str = ", ".join(f"{s}({info['qty']}주)" for s, info in self.holdings.items()) or "Cash 100%"
            logger.info(f"💓 [{state}] Regime: {reg_str} (QQQ ${self.qqq_last_close:.2f} vs 200선 ${self.qqq_sma200:.2f}) | "
                        f"Holdings: [{held_str}] | Power: ${bp:.2f}")
            self.last_status_log = now

    def start(self):
        logger.info("🟢 [Daemon Started] Entering autonomous Nasdaq regime tracking loop...")
        self.update_regime_indicator()
        self.reconcile_account()
        while True:
            try:
                self.run_cycle()
            except Exception as e:
                logger.error(f"❌ [Cycle Exception] {e}")
            time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    trader = KisNasdaqRegimeTrader()
    trader.start()
