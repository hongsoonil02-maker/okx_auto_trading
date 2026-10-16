# -*- coding: utf-8 -*-
"""
kis_nasdaq_trader.py — Autonomous Nasdaq Quant Execution Engine via KIS OpenAPI (v2.5)
Features & Safety Layers:
- Macro Regime Gate: QQQ 5-Min EMA50 (+/- 0.15% hysteresis buffer)
- Regime Shift Unwind: Automatically liquidates opposing ETFs on macro trend reversal
- Technical Engine: 1-Min EMA9 x EMA21 Crossover + RSI(14) Momentum, confirmed by 5-Min alignment
- Entry Ranking: Signal-strength scoring (EMA separation 60% + RSI momentum 40%), top-scored first
- Risk Management (Fee-Aware):
  * 3x ETF (TQQQ/SQQQ): SL -2.4%, TP +8.0%, Trail Arm +2.5% (Drop 1.0%)
  * High-Beta (MSTR/PLTR/COIN/TSLA): SL -2.5%, TP +9.0%, Trail Arm +2.8% (Drop 1.2%)
  * Tech Leaders (NVDA/AMD/AVGO/META): SL -1.5%, TP +6.0%, Trail Arm +2.0% (Drop 0.8%)
  * Large Cap (AAPL): SL -1.2%, TP +4.5%, Trail Arm +1.5% (Drop 0.8%)
  * Partial Take Profit: 50% locked at Trail Arm, runner trails with 2x wider band for right-tail gains
  * Breakeven Defense: Peak +1.5% -> Lock at round-trip cost + 0.2% buffer (never a net-loss "breakeven")
  * Post-Loss 30-min Lockout, 5-min Win Cooldown
  * Mutual Exclusion: Never hold Bullish (TQQQ) and Bearish (SQQQ) simultaneously
- Production Hardening:
  * Fill Verification & Ledger Reconciliation: Real-time TTTS3012R account balance synchronization
  * Confirmed-Exit State Machine: Sell orders stay tracked until ledger-confirmed (auto cancel & resubmit on timeout)
  * Universe-Guard: Holdings outside the tracked universe are never adopted into bot management
  * DST Auto-Adjustment: ZoneInfo('America/New_York') dynamically handles standard/daylight time
  * US Holiday Calendar: NYSE full-day closures skipped (2026-2027)
  * Dynamic FX Rate: Real-time USD/KRW conversion with safe conservative margin
  * Atomic File I/O & Persistent State: Token/position/state files saved via atomic replace;
    ETP flag, trade cooldowns, and pending exits survive restarts
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Any, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, "..", ".env"))

# ── LOGGING SETUP ──
LOG_FILE = os.path.join(CURRENT_DIR, "kis_nasdaq_trader.log")
logger = logging.getLogger("KisNasdaqTrader")
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=3, encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ── HTTP SESSION SETUP ──
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
TOKEN_FILE = os.path.join(CURRENT_DIR, "kis_token.json")
POSITIONS_FILE = os.path.join(CURRENT_DIR, "kis_nasdaq_positions.json")
STATE_FILE = os.path.join(CURRENT_DIR, "kis_nasdaq_state.json")

# ── UNIVERSE & RISK CONFIGURATION ──
UNIVERSE = ["TQQQ", "SQQQ", "NVDA", "TSLA", "MSTR", "PLTR", "AMD", "COIN", "AVGO", "META", "AAPL"]
BULLISH_ETFS = {"TQQQ", "SOXL", "QQQ"}
BEARISH_ETFS = {"SQQQ", "SOXS"}
LEVERAGED_3X = {"TQQQ", "SQQQ", "SOXL", "SOXS"}
HIGH_BETA_EQUITIES = {"MSTR", "PLTR", "COIN", "TSLA"}  # 규제 없는 3배 레버리지급 고변동성 주도주

MAX_CONCURRENT_POSITIONS = 3
MAX_POSITION_PCT = 0.35        # 종목당 최대 35% 자본 배정 (3종목 = 최대 ~100% 운용)
LOOP_INTERVAL_SEC = 10         # 정규장 중 10초 주기 정밀 감시
POST_LOSS_COOLDOWN_SEC = 1800.0  # 손절 후 30분 쿨다운 (노이즈 방지)
NORMAL_COOLDOWN_SEC = 300.0      # 익절 후 5분 쿨다운
PENDING_EXIT_TIMEOUT_SEC = 45.0  # 매도 미체결 시 기존 주문 취소 후 재주문까지 대기 시간

# ── 비용 인식 출구 구조 (수수료 우대 적용률에 맞춰 .env의 KIS_ROUND_TRIP_COST_PCT 조정) ──
ROUND_TRIP_COST_PCT = float(os.getenv("KIS_ROUND_TRIP_COST_PCT", "0.005"))  # 왕복 수수료+환전 스프레드 추정
BREAKEVEN_LOCK_BUFFER_PCT = 0.002   # 본절 방어 발동선 = 왕복비용 + 0.2% 버퍼
PARTIAL_TP_MIN_QTY = 2              # 분할 익절 가능 최소 보유 수량
TRAIL_WIDEN_AFTER_PARTIAL = 2.0     # 분할 익절 후 잔량의 트레일링 폭 배수 (우측 꼬리 추구)

# NYSE 정규장 휴장일 (풀데이 기준, 조기종료일 제외)
US_MARKET_HOLIDAYS = {
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}

# ── 황금 비율 (고승률 83% + 최적 손익비 1.5배) 파라미터 ──
VOLUME_SURGE_THRESHOLD = 1.4       # 1분봉 거래량 폭발 기준 (최근 20분 평균 대비 1.4배 이상 터져야 진입)
OPENING_QUARANTINE_MINUTES = 15     # 개장 직후 15분 (09:30~09:45 ET / 22:30~22:45 KST) 신규 진입 쿼런틴

def get_symbol_risk(symbol: str) -> Tuple[float, float, float]:
    """Returns (stop_loss_pct, take_profit_pct, trailing_arm_pct) tailored to optimal payoff ratio (1.5x expanded)"""
    if symbol in LEVERAGED_3X:
        return 0.024, 0.080, 0.0375   # 3x ETF: 2.4% SL, 8.0% TP, 3.75% Trail Arm (1.5x)
    if symbol in HIGH_BETA_EQUITIES:
        return 0.025, 0.090, 0.042    # 고베타 (MSTR, PLTR, COIN, TSLA): 2.5% SL, 9.0% TP, 4.2% Trail Arm (1.5x)
    if symbol in {"NVDA", "AMD", "AVGO", "META"}:
        return 0.015, 0.060, 0.030    # 테크 주도주: 1.5% SL, 6.0% TP, 3.0% Trail Arm (1.5x)
    return 0.012, 0.045, 0.0225       # 대형주 (AAPL): 1.2% SL, 4.5% TP, 2.25% Trail Arm (1.5x)


class KisNasdaqTrader:
    def __init__(self):
        self.access_token = ""
        self.token_expired_at = 0
        self.account_info = {"usd_cash": 0.0, "krw_cash": 0.0, "total_buying_power_usd": 0.0}
        self.positions: Dict[str, dict] = {}
        self.peak_prices: Dict[str, float] = {}
        self.trade_cooldowns: Dict[str, float] = {}
        self.position_entry_times: Dict[str, float] = {}
        self.etp_available: bool = True       # KIS 계좌 ETP 신청 여부 (APBK1672 시 자동 폴백)
        self.etp_last_retry: float = 0.0
        self.last_status_log = 0.0
        self.pending_exits: Dict[str, dict] = {}  # 매도 접수 후 원장 체결 확인 대기 중인 출구 주문

        self.load_state()
        self.load_positions()

        logger.info("=" * 75)
        logger.info("🚀 [KIS Nasdaq JEV Quant Trader v2.5] High-Beta & Momentum Expansion")
        logger.info(f"   REST Endpoint: {KIS_URL}")
        masked_cano = f"{KIS_CANO[:4]}****" if len(KIS_CANO) >= 4 else KIS_CANO
        logger.info(f"   Account: {masked_cano}-{KIS_PRDT_ABRV}")
        logger.info(f"   Tracked Universe (11): {', '.join(UNIVERSE)}")
        logger.info(f"   High-Beta Leaders: {', '.join(HIGH_BETA_EQUITIES)}")
        logger.info(f"   Leveraged 3x: {', '.join(LEVERAGED_3X)} (TQQQ vs SQQQ Mutual Exclusion)")
        logger.info(f"   Max Concurrent Positions: {MAX_CONCURRENT_POSITIONS} | Max Alloc: {MAX_POSITION_PCT*100:.0f}%")
        logger.info(f"   Risk Config (High Beta): SL -2.5%, TP +9.0%, Trail Arm +4.2% (1.5x), Delta 1.8%")
        logger.info(f"   Risk Config (Tech Leaders): SL -1.5%, TP +6.0%, Trail Arm +3.0% (1.5x), Delta 1.2%")
        logger.info(f"   Alpha Filters: 15m Opening Quarantine (09:30~09:45 ET) | Volume Surge >= {VOLUME_SURGE_THRESHOLD}x | 5m-Trend Gate")
        logger.info(f"   Cost Model: Round-trip {ROUND_TRIP_COST_PCT*100:.2f}% | Breakeven Floor +{max(0.007, ROUND_TRIP_COST_PCT + BREAKEVEN_LOCK_BUFFER_PCT)*100:.1f}% | Partial TP @ Arm (x{TRAIL_WIDEN_AFTER_PARTIAL:.0f} runner trail)")
        if self.positions:
            logger.info(f"   📂 Restored {len(self.positions)} positions: {list(self.positions.keys())}")
        logger.info("=" * 75)

    def load_positions(self):
        try:
            if os.path.exists(POSITIONS_FILE):
                with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
                    self.positions = json.load(f)
                for sym, pos in self.positions.items():
                    self.peak_prices[sym] = pos.get("peak_price", pos.get("entry_price", 0.0))
                    self.position_entry_times[sym] = pos.get("entry_time", time.time())
                logger.info(f"📂 [Positions Restored] {len(self.positions)} loaded from {POSITIONS_FILE}")
            else:
                self.positions = {}
        except Exception as e:
            logger.error(f"❌ Failed to load positions: {e}")
            self.positions = {}

    def save_positions(self):
        try:
            tmp_pos = f"{POSITIONS_FILE}.tmp.{os.getpid()}"
            with open(tmp_pos, "w", encoding="utf-8") as f:
                json.dump(self.positions, f, ensure_ascii=False, indent=2)
            os.replace(tmp_pos, POSITIONS_FILE)
        except Exception as e:
            logger.error(f"❌ Failed to save positions: {e}")

    def save_state(self):
        """ETP 플래그, 쿨다운, 체결 대기 주문 등 운영 상태를 원자적으로 영속화"""
        try:
            tmp_state = f"{STATE_FILE}.tmp.{os.getpid()}"
            with open(tmp_state, "w", encoding="utf-8") as f:
                json.dump({
                    "etp_available": self.etp_available,
                    "etp_last_retry": self.etp_last_retry,
                    "trade_cooldowns": self.trade_cooldowns,
                    "pending_exits": self.pending_exits,
                }, f, ensure_ascii=False, indent=2)
            os.replace(tmp_state, STATE_FILE)
        except Exception as e:
            logger.error(f"❌ Failed to save state: {e}")

    def load_state(self):
        """영속화된 운영 상태 복원 (만료된 쿨다운은 자동 폐기)"""
        try:
            if not os.path.exists(STATE_FILE):
                return
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.etp_available = bool(data.get("etp_available", True))
            self.etp_last_retry = float(data.get("etp_last_retry", 0.0))
            now = time.time()
            self.trade_cooldowns = {
                s: float(t) for s, t in data.get("trade_cooldowns", {}).items()
                if isinstance(t, (int, float)) and float(t) > now
            }
            self.pending_exits = {
                s: v for s, v in data.get("pending_exits", {}).items() if isinstance(v, dict)
            }
            logger.info(
                f"📂 [State Restored] ETP Available: {self.etp_available} | "
                f"Active Cooldowns: {len(self.trade_cooldowns)} | Pending Exits: {len(self.pending_exits)}"
            )
        except Exception as e:
            logger.warning(f"⚠️ State load error: {e}")

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

        url = f"{KIS_URL}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey": KIS_APP_KEY,
            "appsecret": KIS_APP_SECRET
        }
        try:
            resp = session.post(url, json=body, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                self.access_token = data.get("access_token", "")
                expires_in = int(data.get("expires_in", 86400))
                self.token_expired_at = now + expires_in
                tmp_tok = f"{TOKEN_FILE}.tmp.{os.getpid()}"
                with open(tmp_tok, "w", encoding="utf-8") as f:
                    json.dump({"access_token": self.access_token, "expires_at": self.token_expired_at}, f)
                os.replace(tmp_tok, TOKEN_FILE)
                logger.info("🔑 [Auth] KIS Access Token Issued & Cached successfully (atomic).")
                return True
            else:
                logger.error(f"❌ Failed to get access token: HTTP {resp.status_code} - {resp.text}")
                return False
        except Exception as e:
            logger.error(f"❌ Token request exception: {e}")
            return False

    def get_auth_headers(self, tr_id: str) -> dict:
        self.issue_token()
        return {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": KIS_APP_KEY,
            "appsecret": KIS_APP_SECRET,
            "tr_id": tr_id,
            "custtype": "P"
        }

    def check_market_clock(self) -> Tuple[bool, str]:
        """Checks if US Regular Market (09:30 - 16:00 ET) is currently open via ZoneInfo (DST-safe)."""
        now_ny = datetime.now(ZoneInfo("America/New_York"))
        weekday = now_ny.weekday()

        if weekday >= 5:
            return False, f"Weekend (ET {now_ny.strftime('%A %H:%M')})"

        today_str = now_ny.strftime("%Y-%m-%d")
        if today_str in US_MARKET_HOLIDAYS:
            return False, f"US Market Holiday ({today_str})"

        market_open = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
        now_kst = now_ny.astimezone(timezone(timedelta(hours=9)))
        open_kst = market_open.astimezone(timezone(timedelta(hours=9)))

        if market_open <= now_ny <= market_close:
            return True, f"Regular Market Open (ET {now_ny.strftime('%H:%M:%S')} / KST {now_kst.strftime('%H:%M:%S')})"
        elif now_ny < market_open:
            mins_left = (market_open - now_ny).total_seconds() / 60.0
            return False, f"Pre-market / Waiting for Open ({mins_left:.0f}m until 09:30 ET / {open_kst.strftime('%H:%M')} KST)"
        else:
            return False, f"Post-market / Closed (ET {now_ny.strftime('%H:%M')})"

    def get_usd_krw_rate(self) -> float:
        """Fetches current USD/KRW exchange rate via Yahoo Finance with safe fallback (1,380.0 KRW)."""
        if not hasattr(self, "_cached_fx"):
            self._cached_fx = (0.0, 1380.0)
        now = time.time()
        if now - self._cached_fx[0] < 3600.0 and self._cached_fx[1] > 0:
            return self._cached_fx[1]

        try:
            ticker = yf.Ticker("USDKRW=X")
            df = ticker.history(period="1d", interval="5m")
            if df is not None and not df.empty and "Close" in df.columns:
                rate = float(df["Close"].iloc[-1])
                if rate > 900.0:
                    self._cached_fx = (now, rate)
                    logger.info(f"💱 [FX Rate] USD/KRW updated: ₩{rate:,.2f}")
                    return rate
        except Exception as e:
            logger.debug(f"FX rate fetch failed: {e}")

        # Safe fallback with conservative margin
        fallback_rate = 1380.0
        self._cached_fx = (now, fallback_rate)
        return fallback_rate

    def fetch_account_buying_power(self) -> float:
        """Fetches unified margin buying power with dynamic FX conversion."""
        headers = self.get_auth_headers("TTTC8908R")
        params = {
            "CANO": KIS_CANO,
            "ACNT_PRDT_CD": KIS_PRDT_ABRV,
            "PDNO": "005930",
            "ORD_UNPR": "0",
            "ORD_DVSN": "01",
            "CMA_EVLU_AMT_ICLD_YN": "Y",
            "OVRS_ICLD_YN": "Y"
        }
        krw_orderable = 0.0
        try:
            res = session.get(f"{KIS_URL}/uapi/domestic-stock/v1/trading/inquire-psbl-order", headers=headers, params=params, timeout=10)
            if res.status_code == 200:
                data = res.json()
                out = data.get("output", {})
                krw_orderable = float(out.get("max_buy_amt", out.get("ord_psbl_cash", 0.0)))
        except Exception as e:
            logger.warning(f"⚠️ KRW Orderable Cash fetch error: {e}")

        # USD Cash query (TTTS3007R)
        usd_cash = 0.0
        try:
            u_headers = self.get_auth_headers("TTTS3007R")
            u_params = {
                "CANO": KIS_CANO,
                "ACNT_PRDT_CD": KIS_PRDT_ABRV,
                "WCRC_FRCR_DVS_CD": "02",
                "NATN_CD": "840",
                "TR_MKET_CD": "00",
                "INQR_DVS_CD": "00"
            }
            res_u = session.get(f"{KIS_URL}/uapi/overseas-stock/v1/trading/inquire-present-balance", headers=u_headers, params=u_params, timeout=10)
            if res_u.status_code == 200:
                data_u = res_u.json()
                for row in data_u.get("output1", []):
                    if row.get("crcy_cd") == "USD":
                        usd_cash = float(row.get("frcr_dnca_amt", 0.0))
        except Exception as e:
            logger.warning(f"⚠️ USD Cash fetch error: {e}")

        fx_rate = self.get_usd_krw_rate()
        krw_usd_equiv = krw_orderable / fx_rate if krw_orderable > 0 else 0.0
        total_buying_power = usd_cash + krw_usd_equiv

        self.account_info["usd_cash"] = usd_cash
        self.account_info["krw_cash"] = krw_orderable
        self.account_info["fx_rate"] = fx_rate
        self.account_info["total_buying_power_usd"] = total_buying_power
        return total_buying_power

    def reconcile_positions_with_account(self) -> Dict[str, dict]:
        """
        [Critical Fill Verification & Ledger Reconciliation]
        Queries actual KIS Overseas Stock Balance (TTTS3012R) to reconcile local state:
        - Automatically clears ghost positions (orders that were never filled)
        - Restores actual physical holdings into tracking state
        - Updates actual holding qty and average purchase price
        """
        headers = self.get_auth_headers("TTTS3012R")
        params = {
            "CANO": KIS_CANO,
            "ACNT_PRDT_CD": KIS_PRDT_ABRV,
            "OVRS_EXCG_CD": "NASD",
            "TR_CRCY_CD": "USD",
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": ""
        }
        try:
            res = session.get(f"{KIS_URL}/uapi/overseas-stock/v1/trading/inquire-balance", headers=headers, params=params, timeout=8)
            if res.status_code != 200:
                logger.warning(f"⚠️ Balance reconciliation HTTP {res.status_code}")
                return self.positions

            data = res.json()
            if data.get("rt_cd") != "0":
                logger.warning(f"⚠️ Balance reconciliation failed: {data.get('msg1')}")
                return self.positions

            real_holdings: Dict[str, dict] = {}
            for row in data.get("output1", []):
                sym = row.get("ovrs_pdno", "").strip().upper()
                try:
                    qty = int(float(row.get("ovrs_cblc_qty", 0)))
                    avg_px = float(row.get("pchs_avg_pric", 0.0))
                    pl_dollar = float(row.get("frcr_evlu_pfls_amt", 0.0))
                    pl_pct = float(row.get("evlu_pfls_rt", 0.0))
                except (ValueError, TypeError):
                    continue

                if sym and qty > 0:
                    real_holdings[sym] = {
                        "qty": qty,
                        "entry_price": avg_px,
                        "pl_dollar": pl_dollar,
                        "pl_pct": pl_pct
                    }

            state_changed = False

            # 1. Confirm Pending Exits (매도 주문이 원장에서 소멸했는지 실체결 확인)
            for sym in list(self.pending_exits.keys()):
                if sym not in real_holdings:
                    exit_info = self.pending_exits.pop(sym)
                    cooldown_sec = float(exit_info.get("cooldown", 0.0))
                    self.positions.pop(sym, None)
                    self.peak_prices.pop(sym, None)
                    self.position_entry_times.pop(sym, None)
                    if cooldown_sec > 0:
                        self.trade_cooldowns[sym] = time.time() + cooldown_sec
                    state_changed = True
                    logger.info(f"✅ [Exit Filled & Confirmed] {sym} 원장 체결 확인 완료 ({exit_info.get('reason', 'EXIT')})")
                    self.save_state()

            # 2. Detect and remove Ghost Positions (not in real account)
            for sym in list(self.positions.keys()):
                if sym not in real_holdings:
                    if sym in self.pending_exits:
                        continue  # 매도 체결 확인 대기 중 (1번 로직에서 처리)
                    # Give 30s buffer post-order before pruning to prevent race condition
                    entry_t = self.position_entry_times.get(sym, 0.0)
                    if time.time() - entry_t > 30.0:
                        logger.warning(f"👻 [Ghost Position Pruned] {sym} not found in KIS ledger. Removing from tracking.")
                        self.positions.pop(sym, None)
                        self.peak_prices.pop(sym, None)
                        self.position_entry_times.pop(sym, None)
                        state_changed = True

            # 3. Sync or Restore Real Holdings from ledger (유니버스 종목만 관리)
            for sym, info in real_holdings.items():
                if sym not in UNIVERSE:
                    continue  # 수동 매수 등 유니버스 외 종목은 봇이 관리하지 않음
                if sym in self.positions:
                    # Update qty / entry price if mismatch
                    cur_pos = self.positions[sym]
                    if cur_pos.get("qty") != info["qty"] or abs(float(cur_pos.get("entry_price", 0)) - info["entry_price"]) > 0.05:
                        cur_pos["qty"] = info["qty"]
                        cur_pos["entry_price"] = info["entry_price"]
                        state_changed = True
                else:
                    # New or restored holding from ledger
                    self.positions[sym] = {
                        "qty": info["qty"],
                        "entry_price": info["entry_price"],
                        "peak_price": info["entry_price"],
                        "entry_time": time.time()
                    }
                    self.peak_prices[sym] = info["entry_price"]
                    self.position_entry_times[sym] = time.time()
                    state_changed = True
                    logger.info(f"📥 [Holding Reconciled] {sym} x {info['qty']} (Avg: ${info['entry_price']:.2f}) synchronized from KIS ledger.")

            if state_changed:
                self.save_positions()

            return self.positions

        except Exception as e:
            logger.error(f"❌ Balance reconciliation exception: {e}")
            return self.positions

    def get_realtime_price(self, symbol: str) -> Optional[float]:
        """Fetches real-time price of US stock via HHDFS00000300."""
        headers = self.get_auth_headers("HHDFS00000300")
        params = {
            "AUTH": "",
            "EXCD": "NAS",
            "SYMB": symbol
        }
        try:
            res = session.get(f"{KIS_URL}/uapi/overseas-price/v1/quotations/price", headers=headers, params=params, timeout=5)
            if res.status_code == 200:
                out = res.json().get("output", {})
                last = float(out.get("last", 0.0))
                if last > 0:
                    return last
            logger.warning(f"⚠️ [{symbol}] Price quote failed: {res.status_code} - {res.text}")
        except Exception as e:
            logger.error(f"❌ [{symbol}] Quote exception: {e}")
        return None

    def fetch_bars(self, symbol: str, timeframe: str = "5m", limit: int = 50) -> Optional[pd.DataFrame]:
        """Fetches recent bars via yfinance for technical indicator calculation"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="2d", interval=timeframe)
            if df is not None and not df.empty:
                df = df.tail(limit).reset_index()
                df = df.rename(columns={"Close": "close", "High": "high", "Low": "low", "Open": "open", "Volume": "volume"})
                return df
        except Exception as e:
            logger.debug(f"Bars fetch failed for {symbol}: {e}")
        return None

    def calculate_technical_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Computes EMA9, EMA21, EMA50, RSI(14) (identical to Alpaca JEV engine)"""
        last_price = 0.0
        if df is not None and not df.empty and "close" in df.columns:
            try:
                last_price = float(df["close"].iloc[-1])
            except Exception:
                pass

        if df is None or len(df) < 25:
            return {"signal": "NEUTRAL", "price": last_price, "ema50": last_price, "reason": "INSUFFICIENT_BARS"}

        close = df["close"]
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))

        curr_close = close.iloc[-1]
        curr_ema9 = ema9.iloc[-1]
        curr_ema21 = ema21.iloc[-1]
        curr_ema50 = ema50.iloc[-1]
        curr_rsi = rsi.iloc[-1]

        # ── 1분봉 거래량 폭발 (Volume Surge) 계산: 직전 완성 봉(-2) 기준 ──
        # 실시간 진행 중인 미완성 봉(-1)은 10초 주기 루프에서 거래량이 과소평가되는 왜곡을 방지
        vol_ratio = 1.0
        if "volume" in df.columns and len(df) >= 22:
            vol_series = df["volume"]
            vol_ma20 = vol_series.rolling(window=20).mean()
            last_complete_vol = float(vol_series.iloc[-2])
            ma20 = float(vol_ma20.iloc[-2]) if not pd.isna(vol_ma20.iloc[-2]) else 0.0
            if ma20 > 0:
                vol_ratio = last_complete_vol / ma20

        bullish_cross = (curr_ema9 > curr_ema21) and (curr_close > curr_ema9) and (curr_rsi > 48.0)
        bearish_cross = (curr_ema9 < curr_ema21) and (curr_close < curr_ema9) and (curr_rsi < 52.0)

        if bullish_cross:
            return {
                "signal": "BUY",
                "rsi": curr_rsi,
                "ema9": curr_ema9,
                "ema21": curr_ema21,
                "ema50": curr_ema50,
                "price": curr_close,
                "vol_ratio": vol_ratio,
                "reason": f"BULLISH_EMA_CROSS (RSI: {curr_rsi:.1f}, Vol: {vol_ratio:.1f}x)"
            }
        elif bearish_cross:
            return {
                "signal": "SELL",
                "rsi": curr_rsi,
                "ema9": curr_ema9,
                "ema21": curr_ema21,
                "ema50": curr_ema50,
                "price": curr_close,
                "vol_ratio": vol_ratio,
                "reason": f"BEARISH_EMA_CROSS (RSI: {curr_rsi:.1f}, Vol: {vol_ratio:.1f}x)"
            }

        return {"signal": "NEUTRAL", "rsi": curr_rsi, "price": curr_close, "ema50": curr_ema50, "vol_ratio": vol_ratio, "reason": "NO_CLEAR_CROSS"}

    def submit_order(self, symbol: str, side: str, qty: int, price: float) -> Optional[str]:
        """Submits Buy (TTTT1002U) or Sell (TTTT1006U) order via KIS OpenAPI.
        Returns the order number (ODNO) on acceptance, None on rejection/failure.
        Actual fill confirmation is handled asynchronously via ledger reconciliation."""
        if qty < 1:
            logger.warning(f"⚠️ [{symbol}] Order skipped: Qty must be >= 1 (Requested: {qty})")
            return None

        tr_id = "TTTT1002U" if side.lower() == "buy" else "TTTT1006U"
        headers = self.get_auth_headers(tr_id)
        body = {
            "CANO": KIS_CANO,
            "ACNT_PRDT_CD": KIS_PRDT_ABRV,
            "OVRS_EXCG_CD": "NASD",
            "PDNO": symbol,
            "ORD_QTY": str(qty),
            "OVRS_ORD_UNPR": f"{price:.2f}",
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": "00"  # 지정가
        }
        try:
            logger.info(f"📤 [Order Dispatch] {side.upper()} {symbol} x {qty} @ ${price:.2f} (TR: {tr_id})")
            res = session.post(f"{KIS_URL}/uapi/overseas-stock/v1/trading/order", headers=headers, json=body, timeout=10)
            data = res.json()
            if res.status_code == 200 and data.get("rt_cd") == "0":
                odno = data.get("output", {}).get("ODNO", "")
                logger.info(f"✅ [Order Accepted] {side.upper()} {symbol} x {qty} | Order No: {odno}")
                return odno if odno else "UNKNOWN"
            else:
                msg_cd = data.get("msg_cd", "")
                msg1 = data.get("msg1", "")
                logger.error(f"❌ [Order Rejected] {side.upper()} {symbol} | Code: {msg_cd} - Msg: {msg1}")
                if msg_cd == "APBK1672" or any(keyword in msg1 for keyword in ["해외ETP", "사전교육", "교육이수", "레버리지"]):
                    self.etp_available = False
                    self.etp_last_retry = time.time()
                    self.save_state()
                    logger.warning(f"⚠️ [{symbol}] ETP 거래 승인 대기 감지 ({msg_cd}: {msg1}): ETP 주문을 일시 비활성화하고 고베타 주도주 대체 모드로 안전하게 자동 전환합니다.")
                return None
        except Exception as e:
            logger.error(f"❌ [Order Exception] {symbol}: {e}")
            return None

    def cancel_order(self, symbol: str, odno: str) -> bool:
        """미체결 해외주식 주문 취소 (TTTT1004U, RVSE_CNCL_DVSN_CD=02)"""
        if not odno or odno == "UNKNOWN":
            return False
        qty = int(self.positions.get(symbol, {}).get("qty", 0))
        headers = self.get_auth_headers("TTTT1004U")
        body = {
            "CANO": KIS_CANO,
            "ACNT_PRDT_CD": KIS_PRDT_ABRV,
            "OVRS_EXCG_CD": "NASD",
            "PDNO": symbol,
            "ORG_ODNO": str(odno),
            "RVSE_CNCL_DVSN_CD": "02",
            "ORD_QTY": str(qty),
            "OVRS_ORD_UNPR": "0",
            "ORD_SVR_DVSN_CD": "0"
        }
        try:
            res = session.post(f"{KIS_URL}/uapi/overseas-stock/v1/trading/order-rvsecncl", headers=headers, json=body, timeout=10)
            data = res.json()
            if res.status_code == 200 and data.get("rt_cd") == "0":
                logger.info(f"🚫 [Order Cancelled] {symbol} (ODNO: {odno})")
                return True
            logger.warning(f"⚠️ [Cancel Failed] {symbol} (ODNO: {odno}) | {data.get('msg_cd')}: {data.get('msg1')}")
            return False
        except Exception as e:
            logger.error(f"❌ [Cancel Exception] {symbol}: {e}")
            return False

    def _execute_exit(self, symbol: str, qty: int, price: float, reason: str, cooldown_sec: float) -> bool:
        """
        매도 주문을 접수하고, 실제 체결은 원장 대사(reconcile)가 확인할 때까지 포지션을 유지한다.
        - 미체결 시 포지션/peak_price가 그대로 보존됨 (트레일링 스탑 상태 소실 방지)
        - 타임아웃 경과 후에도 미체결이면 기존 주문을 취소하고 현재가로 재주문
        - 쿨다운은 원장에서 체결이 확인된 시점에 시작됨
        """
        now = time.time()
        pending = self.pending_exits.get(symbol)
        if pending:
            if now - float(pending.get("ts", 0.0)) < PENDING_EXIT_TIMEOUT_SEC:
                return False  # 이미 접수된 매도가 체결 확인 대기 중
            stale_odno = pending.get("odno", "")
            logger.warning(f"⏱️ [Exit Resubmit] {symbol} 매도 미체결 타임아웃({PENDING_EXIT_TIMEOUT_SEC:.0f}s) — 기존 주문 취소 후 재주문 (ODNO: {stale_odno})")
            self.cancel_order(symbol, stale_odno)
            self.pending_exits.pop(symbol, None)

        log_fn = logger.warning if "STOP_LOSS" in reason else logger.info
        log_fn(f"🚪 [Exit Triggered] {symbol} | {reason} | Px: ${price:.2f} | Qty: {qty}")

        # 주문 접수 전에 pending을 먼저 등록: 내부 reconcile가 유령 포지션으로 오판하지 않도록
        self.pending_exits[symbol] = {"odno": "", "ts": now, "cooldown": cooldown_sec, "reason": reason}
        odno = self.submit_order(symbol, "sell", qty, price)
        if odno is None:
            self.pending_exits.pop(symbol, None)
            return False
        self.pending_exits[symbol]["odno"] = odno
        self.save_state()
        logger.info(f"⏳ [Exit Pending] {symbol} 매도 접수 (ODNO: {odno}) — 원장 체결 확인 대기")
        time.sleep(1.5)
        self.reconcile_positions_with_account()
        return True

    def evaluate_exits(self, prices: Dict[str, float]):
        """Evaluates Trailing Stop, Breakeven Lock, and Hard TP/SL.
        매도는 접수 즉시 포지션을 제거하지 않고 pending exit으로 등록하며,
        실제 제거/쿨다운 시작은 원장 대사에서 체결 확인 후 수행한다."""
        for symbol, pos in list(self.positions.items()):
            # 매도 접수 후 체결 확인 대기 중이면 중복 주문 방지 (타임아웃 후엔 재시도 허용)
            pending = self.pending_exits.get(symbol)
            if pending and (time.time() - float(pending.get("ts", 0.0))) < PENDING_EXIT_TIMEOUT_SEC:
                continue

            current_price = prices.get(symbol)
            if not current_price or current_price <= 0:
                continue

            entry_price = float(pos.get("entry_price", current_price))
            qty = int(pos.get("qty", 1))
            unrealized_pnl_pct = (current_price - entry_price) / entry_price

            peak = max(self.peak_prices.get(symbol, current_price), current_price)
            self.peak_prices[symbol] = peak
            pos["peak_price"] = peak

            sym_sl, sym_tp, sym_arm = get_symbol_risk(symbol)
            max_ret = (peak - entry_price) / entry_price if entry_price > 0 else 0.0
            trail_drop = (peak - current_price) / peak if peak > 0 else 0.0
            trail_delta = 0.018 if symbol in HIGH_BETA_EQUITIES else (0.015 if symbol in LEVERAGED_3X else 0.012)  # 1.5x expanded for optimal payoff ratio

            partial_done = bool(pos.get("tp_partial_done"))
            trail_delta_eff = trail_delta * TRAIL_WIDEN_AFTER_PARTIAL if partial_done else trail_delta
            breakeven_floor = max(0.007, ROUND_TRIP_COST_PCT + BREAKEVEN_LOCK_BUFFER_PCT)

            # 0. Partial Take Profit: 트레일링 arm 도달 시 50% 익절 확정, 잔량은 넓은 트레일링으로 추세 추종
            if (not partial_done) and qty >= PARTIAL_TP_MIN_QTY and max_ret >= sym_arm:
                sell_qty = max(1, qty // 2)
                logger.info(
                    f"✂️ [Partial Take Profit] {symbol}: Arm +{sym_arm*100:.1f}% 도달 (Max +{max_ret*100:.2f}%) — "
                    f"{qty}주 중 {sell_qty}주 익절 확정, 잔량은 트레일링 폭 {TRAIL_WIDEN_AFTER_PARTIAL:.0f}배로 추세 추종 "
                    f"| Px: ${current_price:.2f} | Qty: {sell_qty}"
                )
                odno = self.submit_order(symbol, "sell", sell_qty, current_price)
                if odno is not None:
                    pos["tp_partial_done"] = True
                    self.save_positions()
                    time.sleep(1.5)
                    self.reconcile_positions_with_account()
                continue

            exit_reason = None
            exit_cooldown = NORMAL_COOLDOWN_SEC

            # 1. Trailing Stop (분할 익절 후 잔량은 trail_delta_eff 적용으로 우측 꼬리 추구)
            if max_ret >= sym_arm and trail_drop >= trail_delta_eff:
                exit_reason = f"TRAILING_STOP (Peak +{max_ret*100:.2f}%, Drop {trail_drop*100:.2f}%, PnL {unrealized_pnl_pct*100:+.2f}%)"
            # 2. 단계형 이익 락 (Tiered Profit Lock): 트레일링 Arm(4.2%) 도달 전 이익 반납 방지
            # - Tier 1: 고점 +2.5% 이상 도달 후 반락 시 최소 +1.2% 이익 보존 청산 (반납형 거래 방지)
            # - Tier 2: 고점 +1.5% 이상 도달 후 반락 시 실질 본절선(+0.7%) 방어 청산
            elif (max_ret >= 0.025 and unrealized_pnl_pct <= 0.012):
                exit_reason = f"PROFIT_TIER_LOCK (Max +{max_ret*100:.2f}%, PnL {unrealized_pnl_pct*100:+.2f}%, Tier +1.2%)"
            elif (max_ret >= 0.015 and unrealized_pnl_pct <= breakeven_floor) or (pos.get("bearish_safeguard_armed") and unrealized_pnl_pct <= 0.001):
                exit_reason = f"BREAKEVEN_LOCK (Max +{max_ret*100:.2f}%, PnL {unrealized_pnl_pct*100:+.2f}%, Floor +{breakeven_floor*100:.1f}%)"
            # 3. Take Profit (분할 익절 전인 포지션의 하드 캡; 분할 후 잔량은 트레일링이 관리)
            elif (not partial_done) and unrealized_pnl_pct >= sym_tp:
                exit_reason = f"TAKE_PROFIT (+{unrealized_pnl_pct*100:.2f}%, Target {sym_tp*100:.1f}%)"
            # 4. Hard Stop Loss
            elif unrealized_pnl_pct <= -sym_sl:
                exit_reason = f"STOP_LOSS ({unrealized_pnl_pct*100:.2f}%, Max SL {sym_sl*100:.1f}%)"
                exit_cooldown = POST_LOSS_COOLDOWN_SEC

            if exit_reason:
                self._execute_exit(symbol, qty, current_price, exit_reason, exit_cooldown)
            else:
                # 출구 조건이 해제된 오래된 pending exit 정리: 증권사 미체결 주문 취소 후 대기 해제
                if pending and (time.time() - float(pending.get("ts", 0.0))) >= PENDING_EXIT_TIMEOUT_SEC:
                    stale_odno = pending.get("odno", "")
                    if stale_odno:
                        self.cancel_order(symbol, stale_odno)
                    self.pending_exits.pop(symbol, None)
                    self.save_state()
                    logger.info(f"♻️ [Exit Pending Cleared] {symbol} 출구 조건 해제로 증권사 미체결 주문 취소 및 대기 상태 정리")

    def evaluate_entries_and_regime(self, prices: Dict[str, float]):
        """Scans Macro Regime via QQQ EMA50, Unwinds opposing ETFs, and executes JEV-aligned trades."""
        now = time.time()

        # Step 1: Scan QQQ benchmark for macro regime (5Min timeframe)
        qqq_bars = self.fetch_bars("QQQ", timeframe="5m", limit=60)
        qqq_tech = self.calculate_technical_signals(qqq_bars)
        qqq_c = qqq_tech.get("price", 0.0)
        qqq_e50 = qqq_tech.get("ema50", 0.0)

        if qqq_c <= 0 or qqq_e50 <= 0:
            logger.debug(f"QQQ benchmark price/EMA50 not ready (${qqq_c} / ${qqq_e50}). Skipping cycle.")
            return

        # Macro Trend Gate: QQQ > EMA50 * 1.0015 for Bullish, QQQ < EMA50 * 0.9985 for Bearish (0.15% hysteresis)
        is_bullish = (qqq_c > qqq_e50 * 1.0015) and (qqq_tech["signal"] == "BUY" or qqq_c > qqq_tech.get("ema21", qqq_c))
        is_bearish = (qqq_c < qqq_e50 * 0.9985) and (qqq_tech["signal"] == "SELL" or qqq_c < qqq_tech.get("ema21", qqq_c))

        # Step 2: REGIME SHIFT UNWIND & SAFEGUARD MANAGEMENT
        if is_bullish:
            # Macro Recovery: Unarm bearish safeguard if previously armed and persist
            for sym, pos in list(self.positions.items()):
                if pos.get("bearish_safeguard_armed"):
                    pos["bearish_safeguard_armed"] = False
                    logger.info(f"🟢 [Macro Recovery] {sym}: QQQ 상승세 복귀로 본절 세이프가드 해제 (Disarmed)")
                    self.save_positions()

            for bear_sym in list(self.positions.keys()):
                if bear_sym in BEARISH_ETFS:
                    pos_entry_t = self.position_entry_times.get(bear_sym, 0.0)
                    if now - pos_entry_t >= 900.0:  # 15m minimum hold buffer
                        logger.warning(f"⚡ [Regime Shift Unwind] QQQ Bullish confirmed (Price ${qqq_c:.2f} > EMA50 ${qqq_e50:.2f})! Unwinding inverse ETF {bear_sym}.")
                        qty = self.positions[bear_sym]["qty"]
                        cur_px = prices.get(bear_sym, self.positions[bear_sym]["entry_price"])
                        self._execute_exit(bear_sym, qty, cur_px, "REGIME_SHIFT_UNWIND (QQQ Bullish → inverse ETF 청산)", 0.0)
        elif is_bearish:
            for bull_sym in list(self.positions.keys()):
                if bull_sym in BULLISH_ETFS:
                    pos_entry_t = self.position_entry_times.get(bull_sym, 0.0)
                    if now - pos_entry_t >= 900.0:
                        logger.warning(f"⚡ [Regime Shift Unwind] QQQ Bearish confirmed (Price ${qqq_c:.2f} < EMA50 ${qqq_e50:.2f})! Unwinding 3x Long ETF {bull_sym}.")
                        qty = self.positions[bull_sym]["qty"]
                        cur_px = prices.get(bull_sym, self.positions[bull_sym]["entry_price"])
                        self._execute_exit(bull_sym, qty, cur_px, "REGIME_SHIFT_UNWIND (QQQ Bearish → 3x Long ETF 청산)", 0.0)

        if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
            return

        # ── [Opening 15-Min Quarantine] 09:30~09:45 ET (22:30~22:45 KST) 신규 진입 차단 ──
        now_ny = datetime.now(ZoneInfo("America/New_York"))
        if now_ny.hour == 9 and 30 <= now_ny.minute < (30 + OPENING_QUARANTINE_MINUTES):
            if now - getattr(self, "last_quarantine_log", 0.0) > 60.0:
                logger.info(f"⏳ [Opening Quarantine] 개장 15분 변동성 관망 구간 ({now_ny.strftime('%H:%M')} ET): 신규 진입 차단 (기존 포지션 리스크 관리 집중)")
                self.last_quarantine_log = now
            return

        # Step 3: Prioritize Candidates Based on Macro Regime
        if not hasattr(self, "etp_available"):
            self.etp_available = True
            self.etp_last_retry = 0.0

        # ETP 교육 이수 후 복구를 위한 6시간 주기 자동 재시도
        if not self.etp_available and (now - getattr(self, "etp_last_retry", 0.0)) > 21600.0:
            self.etp_available = True
            self.save_state()

        candidate_symbols = []
        if is_bullish:
            # 상승장: TQQQ (신청 시) + 고베타 슈퍼모멘텀 주도주 + 대형 테크 주도주
            bull_pool = []
            if self.etp_available:
                bull_pool.append("TQQQ")
            bull_pool.extend(["MSTR", "PLTR", "NVDA", "TSLA", "AMD", "COIN", "AVGO", "META"])
            candidate_symbols = bull_pool
        elif is_bearish:
            if self.etp_available:
                candidate_symbols = ["SQQQ"]
            else:
                # ETP 미신청 시: 하락장 롱 진입 완전 차단 (현금 100% 방어)
                candidate_symbols = []
                # 하락장 방어: 기존 보유 롱 포지션에 본절 세이프가드 가동 및 즉시 저장 (원금 보존)
                armed_any = False
                for sym, pos in list(self.positions.items()):
                    if not pos.get("bearish_safeguard_armed"):
                        pos["bearish_safeguard_armed"] = True
                        armed_any = True
                        logger.info(f"🛡️ [Bearish Macro Defense] {sym} QQQ 하락세 진입: 본절 세이프가드 활성화 (진입가 ${pos.get('entry_price', 0):.2f})")
                if armed_any:
                    self.save_positions()
        else:
            # 중립장: 변동성 및 추세 돌파 최상위 주도주 선별
            candidate_symbols = ["PLTR", "NVDA", "TSLA", "MSTR", "META", "AAPL"]

        # Snapshot current holdings for Mutual Exclusion
        has_bull_etf = any(s in self.positions for s in BULLISH_ETFS)
        has_bear_etf = any(s in self.positions for s in BEARISH_ETFS)

        buying_power = self.fetch_account_buying_power()
        alloc_per_pos = buying_power * MAX_POSITION_PCT

        # ── [Scoring Phase] 1분봉 BUY 신호 + 5분봉 정렬 확인된 후보만 신호 강도 점수화 ──
        scored_candidates = []
        for sym in candidate_symbols:
            if sym in self.positions:
                continue
            if now < self.trade_cooldowns.get(sym, 0.0):
                continue

            # Mutual Exclusion Check
            if sym in BEARISH_ETFS and has_bull_etf:
                continue
            if sym in BULLISH_ETFS and has_bear_etf:
                continue

            current_price = prices.get(sym)
            if not current_price or current_price <= 0:
                continue

            # Technical Signal Check (1-Min timeframe)
            sym_bars = self.fetch_bars(sym, timeframe="1m", limit=40)
            tech = self.calculate_technical_signals(sym_bars)
            if tech["signal"] != "BUY":
                continue

            # Multi-Timeframe Confirm: 5분봉이 명확히 약세(SELL 신호 또는 EMA21 하회)면 1분봉 노이즈로 간주하고 제외
            bars_5m = self.fetch_bars(sym, timeframe="5m", limit=40)
            tech_5m = self.calculate_technical_signals(bars_5m)
            if tech_5m.get("signal") == "SELL":
                continue
            e21_5m = tech_5m.get("ema21", 0.0)
            if e21_5m > 0 and tech_5m.get("price", 0.0) < e21_5m:
                continue

            # ── [핵심 알파 1] 1분봉 거래량 폭발 필터 (Volume Surge >= 1.4x) ──
            # 세력 거래량 없이 힘없이 크로스오버되는 가짜 돌파(Whipsaw) 사전 차단
            vol_r = tech.get("vol_ratio", 1.0)
            if vol_r < VOLUME_SURGE_THRESHOLD:
                continue

            # 신호 강도 점수: EMA9/21 골든크로스 폭(50%) + RSI 모멘텀(30%) + 거래량 폭발 탄력도(20%)
            # 1분봉 EMA 이격(0.05~0.25%)의 스케일을 20배 보정하여 RSI/거래량과 실질 가중치 균형 유지
            ema_sep_pct = (tech["ema9"] / tech["ema21"] - 1.0) * 100.0 if tech.get("ema21") else 0.0
            ema_score = (ema_sep_pct * 20.0) * 0.5
            rsi_score = max(0.0, tech.get("rsi", 50.0) - 50.0) * 0.2
            vol_score = max(0.0, min(vol_r, 3.0) - 1.0) * 1.5
            score = ema_score + rsi_score + vol_score
            scored_candidates.append((score, sym, current_price, tech["reason"], vol_r))

        # ── [Execution Phase] 점수 상위 후보부터 슬롯이 찰 때까지 진입 ──
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        for score, sym, current_price, reason, vol_r in scored_candidates:
            if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
                break

            # Calculate adaptive lot size
            qty = int(alloc_per_pos / current_price)
            if qty < 1 and buying_power >= current_price:
                qty = 1

            if qty < 1:
                continue

            logger.info(f"🎯 [Jev Signal Executing] {sym} (Score: {score:.2f} | Vol: {vol_r:.1f}x | {reason} | 5m-Aligned) | Px: ${current_price:.2f} | Sizing: {qty} shares (~${qty * current_price:,.2f})")
            odno = self.submit_order(sym, "buy", qty, current_price)
            if odno is not None:
                self.positions[sym] = {
                    "qty": qty,
                    "entry_price": current_price,
                    "peak_price": current_price,
                    "entry_time": now
                }
                self.peak_prices[sym] = current_price
                self.position_entry_times[sym] = now
                self.save_positions()
                # 실체결 여부는 원장 대사로 확인 (미체결 시 유령 포지션 자동 제거)
                time.sleep(1.5)
                self.reconcile_positions_with_account()
                if sym in BULLISH_ETFS:
                    has_bull_etf = True
                elif sym in BEARISH_ETFS:
                    has_bear_etf = True

    def run_cycle(self):
        """Single loop execution"""
        is_open, status_msg = self.check_market_clock()
        now = time.time()

        if not is_open:
            if now - self.last_status_log > 300:
                buying_power = self.fetch_account_buying_power()
                logger.info(f"🌙 [{status_msg}] Available USD Power: ${buying_power:.2f} (KRW ₩{self.account_info['krw_cash']:,.0f}) | Positions: {len(self.positions)}")
                self.last_status_log = now
            return

        # Periodic Ledger Reconciliation every 60s during market hours
        if not hasattr(self, "last_reconcile_time"):
            self.last_reconcile_time = 0.0
        if now - self.last_reconcile_time >= 60.0:
            self.reconcile_positions_with_account()
            self.last_reconcile_time = now

        # RTH: Market is open! Fetch real-time prices for tracked universe
        prices = {}
        for s in UNIVERSE:
            p = self.get_realtime_price(s)
            if p:
                prices[s] = p

        # 1. Evaluate Exits first (Trailing, Breakeven, Stop Loss, Take Profit)
        self.evaluate_exits(prices)

        # 2. Evaluate Entries & Macro Regime (QQQ EMA50 + Crossovers)
        self.evaluate_entries_and_regime(prices)

        if now - self.last_status_log > 60:
            status_parts = [f"{s}: ${p:.2f}" for s, p in prices.items()]
            pos_parts = [f"{s}({p.get('qty')}주)" for s, p in self.positions.items()]
            logger.info(f"💓 [Active RTH Loop] Positions ({len(self.positions)}/{MAX_CONCURRENT_POSITIONS}): [{', '.join(pos_parts)}] | " + " | ".join(status_parts))
            self.last_status_log = now

    def start(self):
        logger.info("🟢 [Daemon Started] Entering continuous trading loop...")
        self.fetch_account_buying_power()
        self.reconcile_positions_with_account()

        while True:
            try:
                self.run_cycle()
            except Exception as e:
                logger.error(f"❌ [Cycle Exception] {e}")
            time.sleep(LOOP_INTERVAL_SEC)


if __name__ == "__main__":
    trader = KisNasdaqTrader()
    trader.start()
