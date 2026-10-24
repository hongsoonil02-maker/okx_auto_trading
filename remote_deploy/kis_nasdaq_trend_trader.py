# -*- coding: utf-8 -*-
"""
kis_nasdaq_trend_trader.py — KIS 미국주식 일봉 추세추종 엔진 (KRX kis_trend_trader.py 이식)
전략 (kis_us_trend_backtest.py로 9.7년 검증: CAGR +23.8%, MDD 28%, Sharpe 1.06):
- 진입: 20일 돈치안 신고가 돌파 + 종가 > SMA60 + SMA20 >= SMA60
- 레짐 게이트: QQQ 종가 >= QQQ SMA60 일 때만 신규 진입 (하락장 신규 진입 차단, 현금 방어)
- 랭킹: RS(60일 초과수익 vs QQQ) 70% + 60일선 이격도 30%
- 청산: ATR(진입 시점 고정) 2.0x 트레일링(고점 대비) / 하드스탑 max(-7%, -2.5xATR) / 10일 최저가 이탈
- 규모: 최대 3종목, 종목당 30%, 롱온리 (인버스/레버리지 숏 없음)
실행 인프라 (kis_nasdaq_trader.py에서 검증된 계층 이식):
- 체결 확인 & 원장 대사 (TTTS3012R), pending exit 상태기계 (타임아웃 시 취소 후 재주문)
- ZoneInfo(DST 안전) + NYSE 휴장일 캘린더 + 동적 환율 + atomic 파일 I/O
- 개장 30분 변동성 회피: 주문 액션은 10:00~15:30 ET에만 수행
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

# ── LOGGING ──
LOG_FILE = os.path.join(CURRENT_DIR, "kis_nasdaq_trend_trader.log")
logger = logging.getLogger("KisNasdaqTrendTrader")
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
POSITIONS_FILE = os.path.join(CURRENT_DIR, "kis_nasdaq_trend_positions.json")
STATE_FILE = os.path.join(CURRENT_DIR, "kis_nasdaq_trend_state.json")

# ── UNIVERSE & STRATEGY CONFIG (롱온리 추세 유니버스 — 인버스 제외) ──
UNIVERSE = ["TQQQ", "NVDA", "TSLA", "MSTR", "PLTR", "AMD", "COIN", "AVGO", "META", "AAPL"]
MAX_CONCURRENT_POSITIONS = 3
MAX_POSITION_PCT = 0.30           # 종목당 30% (백테스트와 동일)
CASH_GUARD_PCT = 0.85             # 가용 구매력의 85%까지만 사용
POLL_INTERVAL_SEC = 60            # 일봉 전략: 60초 감시로 충분
PENDING_EXIT_TIMEOUT_SEC = 60.0   # 매도 미체결 시 취소 후 재주문까지 대기

# 주문 액션 허용 창 (개장/마감 변동성 회피)
ACTION_START_ET = (10, 0)         # 10:00 ET
ACTION_END_ET = (15, 30)          # 15:30 ET

# NYSE 휴장일 (풀데이)
US_MARKET_HOLIDAYS = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}


class KisUsTrendTrader:
    def __init__(self):
        self.access_token = ""
        self.token_expired_at = 0
        self.account_info = {"usd_cash": 0.0, "krw_cash": 0.0, "fx_rate": 0.0, "total_buying_power_usd": 0.0}
        self.positions: Dict[str, dict] = {}
        self.pending_exits: Dict[str, dict] = {}
        self.daily_indicators: Dict[str, dict] = {}
        self.qqq_regime_bullish = False
        self.qqq_ret60 = 0.0
        self.last_indicator_date = ""
        self.last_status_log = 0.0

        self.load_state()
        self.load_positions()

        logger.info("=" * 75)
        logger.info("🚀 [KIS US Trend Trader v1.0] Daily Donchian Trend Following (KRX 엔진 이식)")
        logger.info(f"   Universe (10): {', '.join(UNIVERSE)}")
        logger.info(f"   Entry: 20일 돈치안 돌파 + SMA60 상회 + 정배열 | Regime: QQQ >= SMA60")
        logger.info(f"   Exit: ATR 2.0x Trail / Hard max(-7%, -2.5xATR) / 10일 최저가")
        logger.info(f"   Sizing: 최대 {MAX_CONCURRENT_POSITIONS}종목 x {MAX_POSITION_PCT*100:.0f}% | 롱온리")
        logger.info(f"   Action Window: {ACTION_START_ET[0]:02d}:{ACTION_START_ET[1]:02d}~{ACTION_END_ET[0]:02d}:{ACTION_END_ET[1]:02d} ET")
        if self.positions:
            logger.info(f"   📂 Restored {len(self.positions)} positions: {list(self.positions.keys())}")
        logger.info("=" * 75)

    # ── 상태/포지션 영속화 (atomic) ──
    def load_positions(self):
        try:
            if os.path.exists(POSITIONS_FILE):
                with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
                    self.positions = json.load(f)
                logger.info(f"📂 [Positions Restored] {len(self.positions)} loaded")
            else:
                self.positions = {}
        except Exception as e:
            logger.error(f"❌ Failed to load positions: {e}")
            self.positions = {}

    def save_positions(self):
        try:
            tmp = f"{POSITIONS_FILE}.tmp.{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.positions, f, ensure_ascii=False, indent=2)
            os.replace(tmp, POSITIONS_FILE)
        except Exception as e:
            logger.error(f"❌ Failed to save positions: {e}")

    def save_state(self):
        try:
            tmp = f"{STATE_FILE}.tmp.{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"pending_exits": self.pending_exits}, f, ensure_ascii=False, indent=2)
            os.replace(tmp, STATE_FILE)
        except Exception as e:
            logger.error(f"❌ Failed to save state: {e}")

    def load_state(self):
        try:
            if not os.path.exists(STATE_FILE):
                return
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.pending_exits = {s: v for s, v in data.get("pending_exits", {}).items() if isinstance(v, dict)}
            if self.pending_exits:
                logger.info(f"📂 [State Restored] Pending exits: {list(self.pending_exits.keys())}")
        except Exception as e:
            logger.warning(f"⚠️ State load error: {e}")

    # ── 토큰 (공유 캐시, atomic 쓰기) ──
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
        """Returns 'CLOSED' | 'WATCH' (정규장이나 액션 창 밖) | 'ACTION' (주문 허용 창)"""
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

    # ── 일봉 지표 (yfinance, 하루 1회) ──
    def fetch_daily_indicators(self):
        now_ny = datetime.now(ZoneInfo("America/New_York"))
        today_et = now_ny.strftime("%Y-%m-%d")
        if self.last_indicator_date == today_et and self.daily_indicators:
            return
        logger.info("📊 [Indicators] Fetching daily bars for universe + QQQ ...")
        inds = {}
        for sym in UNIVERSE + ["QQQ"]:
            try:
                df = yf.Ticker(sym).history(period="1y", interval="1d")
                if df is None or len(df) < 70:
                    logger.warning(f"⚠️ [{sym}] insufficient daily bars")
                    continue
                # 완성봉만 사용: 마지막 행이 오늘(ET) 날짜면 장중 미완성 봉이므로 제외.
                # 장전 조회 시 마지막 행은 어제 완성봉이므로 유지 (무조건 iloc[:-1] 하면 윈도우가 하루 밀림)
                last_bar_date = df.index[-1].date()
                if last_bar_date >= now_ny.date():
                    df = df.iloc[:-1]
                c, h, l = df["Close"], df["High"], df["Low"]
                tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
                inds[sym] = {
                    "sma20": float(c.rolling(20).mean().iloc[-1]),
                    "sma60": float(c.rolling(60).mean().iloc[-1]),
                    "atr14": float(tr.rolling(14).mean().iloc[-1]),
                    "donchian_high20": float(h.tail(20).max()),
                    "donchian_low10": float(l.tail(10).min()),
                    "ret60": float(c.iloc[-1] / c.iloc[-61] - 1.0),
                    "close": float(c.iloc[-1]),
                    "last_bar": df.index[-1].strftime("%Y-%m-%d"),
                }
            except Exception as e:
                logger.error(f"❌ [{sym}] daily indicator error: {e}")

        if "QQQ" in inds:
            q = inds["QQQ"]
            self.qqq_regime_bullish = q["close"] >= q["sma60"]
            self.qqq_ret60 = q["ret60"]
            logger.info(f"📈 [Macro Regime] QQQ ${q['close']:.2f} vs SMA60 ${q['sma60']:.2f} → "
                        f"{'🟢 Bullish (진입 허용)' if self.qqq_regime_bullish else '🔴 Bearish (신규 진입 차단)'}")
        self.daily_indicators = inds
        self.last_indicator_date = today_et

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

    # ── 구매력/평가 ──
    def fetch_account_buying_power(self) -> float:
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
        fx = self.get_usd_krw_rate()
        total = usd_cash + (krw_orderable / fx if krw_orderable > 0 else 0.0)
        self.account_info.update({"usd_cash": usd_cash, "krw_cash": krw_orderable, "fx_rate": fx,
                                  "total_buying_power_usd": total})
        return total

    # ── 원장 대사 (체결 확인 + 유령 제거 + 유니버스 가드 채택) ──
    def reconcile_positions_with_account(self) -> Dict[str, dict]:
        try:
            res = session.get(f"{KIS_URL}/uapi/overseas-stock/v1/trading/inquire-balance",
                              headers=self.get_auth_headers("TTTS3012R"),
                              params={"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "OVRS_EXCG_CD": "NASD",
                                      "TR_CRCY_CD": "USD", "CTX_AREA_FK200": "", "CTX_AREA_NK200": ""},
                              timeout=8)
            if res.status_code != 200 or res.json().get("rt_cd") != "0":
                logger.warning(f"⚠️ Balance reconciliation failed: HTTP {res.status_code}")
                return self.positions
            real_holdings = {}
            for row in res.json().get("output1", []):
                sym = row.get("ovrs_pdno", "").strip().upper()
                try:
                    qty = int(float(row.get("ovrs_cblc_qty", 0)))
                    avg_px = float(row.get("pchs_avg_pric", 0.0))
                except (ValueError, TypeError):
                    continue
                if sym and qty > 0:
                    real_holdings[sym] = {"qty": qty, "entry_price": avg_px}

            state_changed = False
            # 1. Pending exit 체결 확인
            for sym in list(self.pending_exits.keys()):
                if sym not in real_holdings:
                    info = self.pending_exits.pop(sym)
                    self.positions.pop(sym, None)
                    state_changed = True
                    logger.info(f"✅ [Exit Filled & Confirmed] {sym} 원장 체결 확인 ({info.get('reason', 'EXIT')})")
                    self.save_state()
            # 2. 유령 포지션 제거 (pending 제외)
            for sym in list(self.positions.keys()):
                if sym not in real_holdings and sym not in self.pending_exits:
                    logger.warning(f"👻 [Ghost Position Pruned] {sym} not in ledger. Removing.")
                    self.positions.pop(sym, None)
                    state_changed = True
            # 3. 실보유 동기화/채택 (유니버스 한정)
            for sym, info in real_holdings.items():
                if sym not in UNIVERSE:
                    continue
                if sym in self.positions:
                    cur = self.positions[sym]
                    if cur.get("qty") != info["qty"]:
                        cur["qty"] = info["qty"]
                        state_changed = True
                else:
                    atr = self.daily_indicators.get(sym, {}).get("atr14", info["entry_price"] * 0.03)
                    self.positions[sym] = {
                        "qty": info["qty"], "entry_price": info["entry_price"],
                        "peak_price": info["entry_price"], "atr": atr,
                        "entry_date": datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d"),
                    }
                    state_changed = True
                    logger.info(f"📥 [Holding Adopted] {sym} x {info['qty']} (Avg ${info['entry_price']:.2f}, ATR ${atr:.2f})")
            if state_changed:
                self.save_positions()
            return self.positions
        except Exception as e:
            logger.error(f"❌ Balance reconciliation exception: {e}")
            return self.positions

    # ── 시세 ──
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
            logger.error(f"❌ [{symbol}] Quote exception: {e}")
        return None

    # ── 주문 ──
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

    def cancel_order(self, symbol: str, odno: str) -> bool:
        if not odno or odno == "UNKNOWN":
            return False
        qty = int(self.positions.get(symbol, {}).get("qty", 0))
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

    def _execute_exit(self, symbol: str, qty: int, price: float, reason: str) -> bool:
        """매도 접수 후 원장 체결 확인까지 포지션 유지. 미체결 타임아웃 시 취소 후 재주문."""
        now = time.time()
        pending = self.pending_exits.get(symbol)
        if pending:
            if now - float(pending.get("ts", 0.0)) < PENDING_EXIT_TIMEOUT_SEC:
                return False
            stale = pending.get("odno", "")
            logger.warning(f"⏱️ [Exit Resubmit] {symbol} 미체결 타임아웃 — 취소 후 재주문 (ODNO: {stale})")
            self.cancel_order(symbol, stale)
            self.pending_exits.pop(symbol, None)

        log_fn = logger.warning if "STOP_LOSS" in reason else logger.info
        log_fn(f"🚪 [Exit Triggered] {symbol} | {reason} | Px: ${price:.2f} | Qty: {qty}")
        self.pending_exits[symbol] = {"odno": "", "ts": now, "reason": reason}
        odno = self.submit_order(symbol, "sell", qty, price)
        if odno is None:
            self.pending_exits.pop(symbol, None)
            return False
        self.pending_exits[symbol]["odno"] = odno
        self.save_state()
        logger.info(f"⏳ [Exit Pending] {symbol} 매도 접수 (ODNO: {odno}) — 원장 확인 대기")
        time.sleep(1.5)
        self.reconcile_positions_with_account()
        return True

    # ── 전략: 청산 ──
    def manage_exits(self, prices: Dict[str, float]):
        for symbol, pos in list(self.positions.items()):
            pending = self.pending_exits.get(symbol)
            if pending and (time.time() - float(pending.get("ts", 0.0))) < PENDING_EXIT_TIMEOUT_SEC:
                continue
            price = prices.get(symbol)
            if not price or price <= 0:
                continue
            entry = float(pos.get("entry_price", price))
            qty = int(pos.get("qty", 1))
            atr = float(pos.get("atr", entry * 0.03))
            peak = max(float(pos.get("peak_price", price)), price)
            pos["peak_price"] = peak

            ind = self.daily_indicators.get(symbol, {})
            donchian_low10 = ind.get("donchian_low10", 0.0)
            trail_stop = peak - 2.0 * atr
            hard_stop = max(entry * 0.93, entry - 2.5 * atr)
            effective_stop = max(trail_stop, donchian_low10)

            if price < hard_stop:
                self._execute_exit(symbol, qty, price,
                                   f"STOP_LOSS (hard stop, entry ${entry:.2f}, ATR ${atr:.2f})")
            elif price < effective_stop and price < peak:
                self._execute_exit(symbol, qty, price,
                                   f"ATR_TRAIL_STOP (peak ${peak:.2f}, stop ${effective_stop:.2f})")

    # ── 전략: 진입 ──
    def evaluate_entries(self, prices: Dict[str, float]):
        if not self.qqq_regime_bullish:
            return
        if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
            return

        buying_power = self.fetch_account_buying_power()
        equity = buying_power + sum(int(p.get("qty", 0)) * prices.get(s, 0.0) for s, p in self.positions.items())
        if equity <= 0:
            return

        candidates = []
        for sym in UNIVERSE:
            if sym in self.positions or sym in self.pending_exits:
                continue
            ind = self.daily_indicators.get(sym)
            price = prices.get(sym)
            if not ind or not price:
                continue
            if price > ind["donchian_high20"] and price > ind["sma60"] and ind["sma20"] >= ind["sma60"]:
                rs = (ind["ret60"] - self.qqq_ret60) * 100.0
                dist = (price / ind["sma60"] - 1.0) * 100.0
                score = rs * 0.7 + dist * 0.3
                candidates.append((score, sym, price, ind))

        candidates.sort(key=lambda x: x[0], reverse=True)
        for score, sym, price, ind in candidates:
            if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
                break
            target = min(equity * MAX_POSITION_PCT, buying_power * CASH_GUARD_PCT)
            qty = int(target / price)
            if qty < 1:
                continue
            logger.info(f"🎯 [Trend Entry] {sym}: 20일 신고가 ${ind['donchian_high20']:.2f} 돌파 "
                        f"(RS {score:.1f} | 60일선 ${ind['sma60']:.2f} | ATR ${ind['atr14']:.2f}) | {qty}주 @ ${price:.2f}")
            odno = self.submit_order(sym, "buy", qty, price)
            if odno is not None:
                self.positions[sym] = {
                    "qty": qty, "entry_price": price, "peak_price": price,
                    "atr": ind["atr14"],
                    "entry_date": datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d"),
                }
                self.save_positions()
                time.sleep(1.5)
                self.reconcile_positions_with_account()

    # ── 메인 루프 ──
    def run_cycle(self):
        state = self.get_market_state()
        now = time.time()
        self.fetch_daily_indicators()

        if state == "CLOSED":
            if now - self.last_status_log > 1800:
                bp = self.fetch_account_buying_power()
                regime = "🟢Bull" if self.qqq_regime_bullish else "🔴Bear"
                logger.info(f"🌙 [Market Closed] Regime: {regime} | Power: ${bp:.2f} | Positions: {len(self.positions)}")
                self.last_status_log = now
            return

        # 주기적 원장 대사 (5분): 수동 거래·부분체결·유령 포지션을 주문 이벤트 없이도 동기화
        if now - getattr(self, "last_reconcile_time", 0.0) >= 300.0:
            self.reconcile_positions_with_account()
            self.last_reconcile_time = now

        # 실시간 가격 조회 (보유 종목 + 돌파 후보 전체)
        prices = {}
        for s in UNIVERSE:
            p = self.get_realtime_price(s)
            if p:
                prices[s] = p

        # 청산은 액션 창에서만 (개장/마감 변동성에서의 오발동 방지)
        if state == "ACTION":
            self.manage_exits(prices)
            self.evaluate_entries(prices)

        if now - self.last_status_log > 300:
            pos_str = ", ".join(f"{s}({p.get('qty')}주, 고점 ${p.get('peak_price', 0):.2f})" for s, p in self.positions.items())
            regime = "🟢Bull" if self.qqq_regime_bullish else "🔴Bear"
            logger.info(f"💓 [{state}] Regime: {regime} | Positions ({len(self.positions)}/{MAX_CONCURRENT_POSITIONS}): [{pos_str}]")
            self.last_status_log = now

    def start(self):
        logger.info("🟢 [Daemon Started] Entering continuous trend-following loop...")
        self.fetch_daily_indicators()
        self.fetch_account_buying_power()
        self.reconcile_positions_with_account()
        while True:
            try:
                self.run_cycle()
            except Exception as e:
                logger.error(f"❌ [Cycle Exception] {e}")
            time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    trader = KisUsTrendTrader()
    trader.start()
