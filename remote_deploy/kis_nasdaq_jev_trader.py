# -*- coding: utf-8 -*-
"""
kis_nasdaq_jev_trader.py — Autonomous Nasdaq Jev Alpha Trader via KIS OpenAPI
Based on proven Alpaca Golden Era (Sep 24 ~ Oct 07) settings:
- Universe: NVDA, AAPL, TSLA (Focused 3 Mega-Caps)
- Sub-Second Jev Microstructure LOB Gating (Local Bun HTTP Daemon /predict)
- Golden Risk Profile:
    * 1x Stock (NVDA, AAPL): SL -1.2%, TP +2.8%, Trail Arm +1.5%
    * High-Beta (TSLA):      SL -2.2%, TP +4.5%, Trail Arm +2.5%
- Cooldown: 30-min lockout after stop loss, 5-min cooldown after win
- Capital Allocation: Operates strictly within designated 50% capital pool (default: $2,500)
- Execution: Limit orders with Maker bias, active position tracking, local trailing & stop management
"""

import os
import sys
import time
import json
import logging
import math
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Any, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, ".."))
load_dotenv(os.path.join(CURRENT_DIR, "..", ".env"))

try:
    from jev.universal_jev_gate import UniversalJevGate
except ImportError:
    UniversalJevGate = None

# ── LOGGING SETUP ──
LOG_FILE = os.path.join(CURRENT_DIR, "kis_nasdaq_jev_trader.log")
logger = logging.getLogger("KisNasdaqJevTrader")
logger.setLevel(logging.INFO)
logger.propagate = False
_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=3, encoding="utf-8")
_file_handler.setFormatter(_formatter)
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(_formatter)
logger.addHandler(_file_handler)
logger.addHandler(_stream_handler)

# ── HTTP SESSION SETUP ──
session = requests.Session()
_adapter = HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
                                         allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]))
session.mount("https://", _adapter)
session.mount("http://", _adapter)

# ── CREDENTIALS & CONSTANTS ──
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
JEV_URL = os.getenv("JEV_URL", "http://10.142.0.2:8020") # quant-nasdaq-only Bun Daemon
TOKEN_FILE = os.path.join(CURRENT_DIR, "kis_token.json")
POSITIONS_FILE = os.path.join(CURRENT_DIR, "kis_nasdaq_jev_positions.json")
STATE_FILE = os.path.join(CURRENT_DIR, "kis_nasdaq_jev_state.json")

# ── GOLDEN ERA TRADING PARAMETERS ──
SYMBOLS_UNIVERSE = ["NVDA", "AAPL", "TSLA"]
MAX_CONCURRENT_POSITIONS = 3
ALLOCATION_RATIO = float(os.getenv("KIS_JEV_ALLOCATION", "0.50")) # 50% capital pool
MAX_POSITION_PCT = 0.30            # 30% per position out of the JEV pool (~$750 per pos)
LOOP_INTERVAL_SEC = 10             # 10s poll cycle during regular market hours
POST_LOSS_COOLDOWN_SEC = 1800.0    # 30m lockout post loss
NORMAL_COOLDOWN_SEC = 300.0        # 5m cooldown post win
PENDING_TIMEOUT_SEC = 90           # 미체결 주문 취소·재시도까지 대기
RECONCILE_INTERVAL_SEC = 30        # 원장 대조 주기
SL_LIMIT_OFFSET = 0.995            # 손절 매도: 현재가 -0.5% (체결 우선 제한가)
TP_LIMIT_OFFSET = 0.998            # 익절/트레일링 매도: 현재가 -0.2%

NY = ZoneInfo("America/New_York")
US_MARKET_HOLIDAYS = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}

def get_symbol_risk(symbol: str) -> Tuple[float, float, float]:
    """Returns (stop_loss_pct, take_profit_pct, trailing_arm_pct) from the proven Golden Era"""
    if symbol == "TSLA":
        return 0.022, 0.045, 0.025   # TSLA: SL -2.2%, TP +4.5%, Trail +2.5%
    return 0.012, 0.028, 0.015       # NVDA, AAPL: SL -1.2%, TP +2.8%, Trail +1.5%

class KisNasdaqJevTrader:
    def __init__(self):
        self.access_token = ""
        self.token_expired_at = 0.0
        self.positions: Dict[str, dict] = {} # sym -> {qty, avg_entry, peak_price, entry_time}
        self.trade_cooldowns: Dict[str, float] = {}
        self.jev_gate = UniversalJevGate(timeout_ms=2500) if UniversalJevGate else None
        self.total_account_equity: float = 5000.0
        self.jev_pool_size: float = 2500.0
        self.load_positions()
        
        logger.info("=" * 75)
        logger.info("🚀 [KIS Nasdaq Jev Trader v1.0] Autonomous Alpha Engine")
        logger.info(f"   Universe (Proven 3): {', '.join(SYMBOLS_UNIVERSE)}")
        logger.info(f"   Capital Allocation: {ALLOCATION_RATIO*100:.0f}% of total account")
        logger.info("   Risk Profile (NVDA/AAPL): SL -1.2% | TP +2.8% | Trail +1.5%")
        logger.info("   Risk Profile (TSLA):      SL -2.2% | TP +4.5% | Trail +2.5%")
        logger.info(f"   JEV Gating Engine: {'Local UniversalJevGate (coinbot)' if self.jev_gate else JEV_URL}")
        logger.info("=" * 75)

    def load_positions(self):
        if os.path.exists(POSITIONS_FILE):
            try:
                with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.positions = data.get("positions", {})
                    self.trade_cooldowns = data.get("cooldowns", {})
            except Exception as e:
                logger.warning(f"⚠️ Failed to load positions: {e}")

    def save_positions(self):
        try:
            tmp = f"{POSITIONS_FILE}.tmp.{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"positions": self.positions, "cooldowns": self.trade_cooldowns}, f, indent=2)
            os.replace(tmp, POSITIONS_FILE)
        except Exception as e:
            logger.error(f"❌ Failed to save positions: {e}")

    def issue_token(self) -> bool:
        now = time.time()
        if self.access_token and now < self.token_expired_at:
            return True
        if os.path.exists(TOKEN_FILE):
            try:
                with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                    c = json.load(f)
                    if c.get("access_token") and now < c.get("expires_at", 0):
                        self.access_token = c["access_token"]
                        self.token_expired_at = c["expires_at"]
                        return True
            except Exception:
                pass
        try:
            r = session.post(f"{KIS_URL}/oauth2/tokenP", json={
                "grant_type": "client_credentials",
                "appkey": KIS_APP_KEY,
                "appsecret": KIS_APP_SECRET
            }, timeout=8)
            d = r.json()
            if r.status_code == 200 and "access_token" in d:
                self.access_token = d["access_token"]
                self.token_expired_at = now + int(d.get("expires_in", 86400)) - 600
                with open(TOKEN_FILE, "w", encoding="utf-8") as f:
                    json.dump({"access_token": self.access_token, "expires_at": self.token_expired_at}, f)
                return True
        except Exception as e:
            logger.error(f"❌ Token error: {e}")
        return False

    def headers(self, tr_id: str) -> dict:
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.access_token}",
            "appkey": KIS_APP_KEY,
            "appsecret": KIS_APP_SECRET,
            "tr_id": tr_id,
        }

    def is_market_open(self) -> bool:
        now_ny = datetime.now(NY)
        d_str = now_ny.strftime("%Y-%m-%d")
        if d_str in US_MARKET_HOLIDAYS or now_ny.weekday() >= 5:
            return False
        # Regular trading hours: 09:30 to 16:00 ET
        open_time = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
        close_time = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
        return open_time <= now_ny < close_time

    def get_price(self, sym: str) -> Optional[float]:
        if not self.issue_token():
            return None
        try:
            r = session.get(f"{KIS_URL}/uapi/overseas-price/v1/quotations/price",
                            headers=self.headers("HHDFS00000300"),
                            params={"AUTH": "", "EXCD": "NAS", "SYMB": sym}, timeout=5)
            if r.status_code == 200:
                last = float(r.json().get("output", {}).get("last", 0.0))
                if last > 0:
                    return last
        except Exception as e:
            logger.warning(f"⚠️ [{sym}] quote failed: {e}")
        return None

    def sync_account(self):
        if not self.issue_token():
            return
        # 1. 해외주식 주식 평가액 조회 (TTTS3012R)
        tot_evlu_usd = 0.0
        usd_cash = 0.0
        try:
            r = session.get(f"{KIS_URL}/uapi/overseas-stock/v1/trading/inquire-balance",
                            headers=self.headers("TTTS3012R"),
                            params={"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "OVRS_EXCG_CD": "NASD",
                                    "TR_CRCY_CD": "USD", "CTX_AREA_FK200": "", "CTX_AREA_NK200": ""}, timeout=8)
            if r.status_code == 200:
                out2 = r.json().get("output2", {})
                tot_evlu_usd = float(out2.get("tot_evlu_pfls_amt", 0.0))
                usd_cash = float(out2.get("frcr_dnca_amt", 0.0) or 0.0)
        except Exception as e:
            logger.warning(f"⚠️ Account sync error (Overseas): {e}")

        # 2. 국내 원화 예수금 / 주문가능금액 조회 (TTTC8908R 통합증거금 대응)
        krw_cash = 0.0
        try:
            r = session.get(f"{KIS_URL}/uapi/domestic-stock/v1/trading/inquire-psbl-order",
                            headers=self.headers("TTTC8908R"),
                            params={"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "PDNO": "005930",
                                    "ORD_UNPR": "0", "ORD_DVSN": "01", "CMA_EVLU_AMT_ICLD_YN": "Y", "OVRS_ICLD_YN": "Y"}, timeout=8)
            if r.status_code == 200:
                out = r.json().get("output", {})
                krw_cash = float(out.get("ord_psbl_cash", 0.0))
        except Exception as e:
            logger.warning(f"⚠️ Account sync error (KRW Cash): {e}")

        # 3. 실시간 환율 반영 (기본 1380원)
        fx_rate = 1380.0
        try:
            import yfinance as yf
            df_fx = yf.Ticker("USDKRW=X").history(period="1d", interval="5m")
            if not df_fx.empty and float(df_fx["Close"].iloc[-1]) > 900:
                fx_rate = float(df_fx["Close"].iloc[-1])
        except Exception:
            pass

        krw_in_usd = krw_cash / fx_rate if krw_cash > 0 else 0.0
        self.available_cash_usd = usd_cash + krw_in_usd
        total_combined_usd = tot_evlu_usd + krw_in_usd
        if total_combined_usd > 0:
            self.total_account_equity = total_combined_usd
            self.jev_pool_size = self.total_account_equity * ALLOCATION_RATIO
            logger.info(f"💼 [Account Sync] Total Equity: ${self.total_account_equity:,.2f} "
                        f"(주식 ${tot_evlu_usd:,.2f} + 가용현금 ${self.available_cash_usd:,.2f} [₩{krw_cash:,.0f}]) "
                        f"| JEV Pool (50%): ${self.jev_pool_size:,.2f}")

    def query_jev_decision(self, sym: str, side: str = "buy") -> Tuple[bool, float, str]:
        """Queries local UniversalJevGate (standalone) or remote Bun Sidecar"""
        if self.jev_gate:
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                dec = loop.run_until_complete(
                    self.jev_gate.verify_entry(
                        asset_class="US_STOCKS",
                        symbol=sym,
                        setup_name="KIS_JEV_LOB",
                        context={"side": side}
                    )
                )
                loop.close()
                return dec.approved, dec.confidence, dec.reason
            except Exception as e:
                logger.warning(f"⚠️ UniversalJevGate error for {sym}: {e}")

        try:
            r = session.post(f"{JEV_URL}/predict", json={"symbol": sym, "side": side}, timeout=2)
            if r.status_code == 200:
                d = r.json()
                return d.get("approved", False), float(d.get("score", 0.5)), d.get("reason", "")
        except Exception:
            pass
        return False, 0.5, "CONNECTION_ERROR"

    def order(self, sym: str, side: str, qty: int, price: float) -> Optional[str]:
        if qty < 1 or not self.issue_token():
            return None
        tr = "TTTT1002U" if side == "buy" else "TTTT1006U"
        body = {
            "CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "OVRS_EXCG_CD": "NASD", "PDNO": sym,
            "ORD_QTY": str(qty), "OVRS_ORD_UNPR": f"{price:.2f}", "ORD_SVR_DVSN_CD": "0", "ORD_DVSN": "00"
        }
        try:
            logger.info(f"📤 [KIS Order] {side.upper()} {sym} x {qty} @ ${price:.2f}")
            r = session.post(f"{KIS_URL}/uapi/overseas-stock/v1/trading/order", headers=self.headers(tr), json=body, timeout=10)
            d = r.json()
            if r.status_code == 200 and d.get("rt_cd") == "0":
                odno = d.get("output", {}).get("ODNO", "")
                logger.info(f"✅ [Accepted] {side.upper()} {sym} x {qty} | ODNO: {odno}")
                return odno
            logger.error(f"❌ [Rejected] {side.upper()} {sym} | {d.get('msg_cd')}: {d.get('msg1')}")
        except Exception as e:
            logger.error(f"❌ [Order Exception] {sym}: {e}")
        return None

    def cancel_order(self, sym: str, odno: str, qty: int) -> bool:
        if not odno or not self.issue_token():
            return False
        body = {"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "OVRS_EXCG_CD": "NASD", "PDNO": sym,
                "ORG_ODNO": str(odno), "RVSE_CNCL_DVSN_CD": "02", "ORD_QTY": str(qty),
                "OVRS_ORD_UNPR": "0", "ORD_SVR_DVSN_CD": "0"}
        try:
            r = session.post(f"{KIS_URL}/uapi/overseas-stock/v1/trading/order-rvsecncl",
                             headers=self.headers("TTTT1004U"), json=body, timeout=10)
            d = r.json()
            if d.get("rt_cd") == "0":
                logger.info(f"🚫 [Cancelled] {sym} ODNO {odno}")
                return True
            logger.info(f"ℹ️ [Cancel skipped] {sym} ODNO {odno} ({d.get('msg1')}) — 이미 체결됐을 가능성")
        except Exception as e:
            logger.warning(f"⚠️ cancel exception {sym}: {e}")
        return False

    def fetch_ledger(self) -> Optional[Dict[str, dict]]:
        """원장(TTTS3012R) 기준 보유 종목. 조회 실패 시 None (빈 dict와 구분)."""
        if not self.issue_token():
            return None
        try:
            r = session.get(f"{KIS_URL}/uapi/overseas-stock/v1/trading/inquire-balance",
                            headers=self.headers("TTTS3012R"),
                            params={"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "OVRS_EXCG_CD": "NASD",
                                    "TR_CRCY_CD": "USD", "CTX_AREA_FK200": "", "CTX_AREA_NK200": ""}, timeout=8)
            if r.status_code == 200 and r.json().get("rt_cd") == "0":
                held: Dict[str, dict] = {}
                for row in r.json().get("output1", []):
                    sym = (row.get("ovrs_pdno") or "").strip().upper()
                    try:
                        qty = int(float(row.get("ovrs_cblc_qty", 0)))
                        avg = float(row.get("pchs_avg_pric", 0.0))
                    except (ValueError, TypeError):
                        continue
                    if sym and qty > 0:
                        held[sym] = {"qty": qty, "avg": avg}
                return held
        except Exception as e:
            logger.warning(f"⚠️ ledger fetch error: {e}")
        return None

    def reconcile_positions(self):
        """원장 대조: 미체결 확인/취소, 수량·평단 동기화, 유령 포지션 정리, 외부 보유분 편입"""
        held = self.fetch_ledger()
        if held is None:
            return
        now = time.time()
        changed = False
        for sym, pos in list(self.positions.items()):
            # 1) 진입 체결 대기 중
            if pos.get("pending_entry"):
                h = held.get(sym)
                if h:
                    logger.info(f"✅ [Fill Confirmed] {sym} {h['qty']}주 @ ${h['avg']:.2f} (실체결가 반영)")
                    pos.update(qty=h["qty"], avg_entry=h["avg"],
                               peak_price=max(pos.get("peak_price", h["avg"]), h["avg"]))
                    pos.pop("pending_entry", None)
                    changed = True
                elif now - pos.get("entry_time", 0) > PENDING_TIMEOUT_SEC:
                    self.cancel_order(sym, pos.get("odno", ""), pos["qty"])
                    logger.warning(f"⏱️ [Entry Timeout] {sym} 미체결 {PENDING_TIMEOUT_SEC}s 초과 — 취소 후 추적 해제")
                    del self.positions[sym]
                    changed = True
                continue
            # 2) 청산 체결 대기 중
            if pos.get("pending_exit"):
                if sym not in held:
                    logger.info(f"✅ [Exit Filled] {sym} 원장 확인 — 포지션 정리")
                    cd = POST_LOSS_COOLDOWN_SEC if pos.get("last_pnl_pct", 0) < 0 else NORMAL_COOLDOWN_SEC
                    self.trade_cooldowns[sym] = now + cd
                    del self.positions[sym]
                    changed = True
                elif now - pos.get("exit_order_ts", 0) > PENDING_TIMEOUT_SEC:
                    self.cancel_order(sym, pos.get("exit_odno", ""), pos["qty"])
                    pos.pop("pending_exit", None)
                    logger.warning(f"⏱️ [Exit Timeout] {sym} 미체결 — 취소, 다음 사이클 새 가격으로 재주문")
                    changed = True
                continue
            # 3) 일반 포지션: 원장과 동기화
            h = held.get(sym)
            if not h:
                logger.warning(f"⚠️ [Reconcile] {sym} 원장에 없음(외부 매도/수동 청산) — 추적 해제")
                del self.positions[sym]
                changed = True
            elif h["qty"] != pos["qty"] or abs(h["avg"] - pos["avg_entry"]) > 0.01:
                logger.info(f"🔄 [Reconcile] {sym} 원장 동기화: {pos['qty']}주@${pos['avg_entry']:.2f} → {h['qty']}주@${h['avg']:.2f}")
                pos["qty"] = h["qty"]
                pos["avg_entry"] = h["avg"]
                changed = True
        # 4) 유니버스 내 미추적 보유분 편입
        for sym in SYMBOLS_UNIVERSE:
            if sym in held and sym not in self.positions:
                logger.info(f"📥 [Reconcile] {sym} 원장 보유 편입: {held[sym]['qty']}주 @ ${held[sym]['avg']:.2f}")
                self.positions[sym] = {"qty": held[sym]["qty"], "avg_entry": held[sym]["avg"],
                                       "peak_price": held[sym]["avg"], "entry_time": now}
                changed = True
        if changed:
            self.save_positions()

    def manage_exits(self):
        """Monitors positions for Take Profit, Stop Loss, and Trailing Stops"""
        for sym, pos in list(self.positions.items()):
            if pos.get("pending_entry") or pos.get("pending_exit"):
                continue  # 체결 대기 중 — reconcile이 처리
            qty = pos["qty"]
            avg_entry = pos["avg_entry"]
            peak = pos["peak_price"]
            curr_px = self.get_price(sym)
            if not curr_px:
                continue

            # Update peak price
            if curr_px > peak:
                peak = curr_px
                pos["peak_price"] = peak
                self.save_positions()

            pnl_pct = (curr_px - avg_entry) / avg_entry
            peak_pct = (peak - avg_entry) / avg_entry
            sl_pct, tp_pct, trail_arm = get_symbol_risk(sym)

            exit_reason = None
            # 1. Hard Stop Loss
            if pnl_pct <= -sl_pct:
                exit_reason = f"STOP_LOSS_{pnl_pct*100:.2f}%"
            # 2. Hard Take Profit
            elif pnl_pct >= tp_pct:
                exit_reason = f"TAKE_PROFIT_{pnl_pct*100:.2f}%"
            # 3. Trailing Stop
            elif peak_pct >= trail_arm and (peak - curr_px) / peak >= 0.010: # 1% drop from peak
                exit_reason = f"TRAILING_STOP_{pnl_pct*100:.2f}%"

            if exit_reason:
                # 손절은 체결 우선(-0.5% 제한가), 익절/트레일링은 -0.2% 제한가
                offset = SL_LIMIT_OFFSET if pnl_pct < 0 else TP_LIMIT_OFFSET
                limit_px = round(curr_px * offset, 2)
                logger.info(f"🚨 [Exit Triggered] {sym} | Reason: {exit_reason} | Current: ${curr_px:.2f} → 제한가 ${limit_px:.2f} (Entry: ${avg_entry:.2f})")
                odno = self.order(sym, "sell", qty, limit_px)
                if odno:
                    # 체결 확인은 reconcile이 수행 (미체결 시 90초 후 취소·재주문)
                    pos["pending_exit"] = True
                    pos["exit_odno"] = odno
                    pos["exit_order_ts"] = time.time()
                    pos["last_pnl_pct"] = pnl_pct
                    self.save_positions()

    def scan_and_enter(self):
        """Evaluates entry signals and executes approved Jev trades within capital pool"""
        now_ny = datetime.now(NY)
        # 15:30 ET 이후 신규 진입 차단 (장마감 전 EOD 청산 대비)
        if (now_ny.hour == 15 and now_ny.minute >= 30) or (now_ny.hour >= 16):
            return

        if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
            return

        now = time.time()
        for sym in SYMBOLS_UNIVERSE:
            if sym in self.positions:
                continue
            if now < self.trade_cooldowns.get(sym, 0.0):
                continue

            curr_px = self.get_price(sym)
            if not curr_px or curr_px <= 0:
                continue

            # Step 1: Query Jev AI Microstructure Decision
            approved, score, reason = self.query_jev_decision(sym, "buy")
            logger.info(f"🔎 [Jev Scan] {sym} | Approved: {approved} | Score: {score:.3f} | Reason: {reason}")

            if approved:
                # Sizing: 30% of Jev Pool, capped by actual available cash (98% buffer for commissions)
                max_cash_alloc = getattr(self, "available_cash_usd", 0.0) * 0.98
                target_alloc = min(self.jev_pool_size * MAX_POSITION_PCT, max_cash_alloc)
                qty = int(target_alloc / curr_px)
                if qty < 1:
                    logger.info(f"ℹ️ [{sym}] Qty is 0 (Price ${curr_px:.2f} > Alloc ${target_alloc:.2f} / Cash ${max_cash_alloc:.2f}); 5m cooldown.")
                    self.trade_cooldowns[sym] = now + 300.0
                    self.save_positions()
                    continue

                logger.info(f"🎯 [Jev Execution] Approved! Sizing {qty} shares of {sym} (~${qty * curr_px:.2f})")
                odno = self.order(sym, "buy", qty, curr_px)
                if odno:
                    # 체결 확정 전 펜딩 상태 — 실수량/실체결가는 reconcile이 원장에서 반영
                    self.positions[sym] = {
                        "qty": qty,
                        "avg_entry": curr_px,
                        "peak_price": curr_px,
                        "entry_time": time.time(),
                        "odno": odno,
                        "pending_entry": True
                    }
                    self.save_positions()
                    if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
                        break
                else:
                    logger.warning(f"⚠️ [{sym}] Order rejected by broker — 5m cooldown applied to avoid API spam.")
                    self.trade_cooldowns[sym] = now + 300.0
                    self.save_positions()

    def run_cycle(self):
        if not self.is_market_open():
            logger.info(f"🌙 [Market Closed] Regular US hours are 09:30~16:00 ET. Next open waiting...")
            time.sleep(60)
            return

        self.sync_account()
        now = time.time()
        if now - getattr(self, "_last_reconcile", 0.0) >= RECONCILE_INTERVAL_SEC:
            self._last_reconcile = now
            self.reconcile_positions()
        self.manage_exits()
        self.scan_and_enter()
        time.sleep(LOOP_INTERVAL_SEC)

    def start(self):
        logger.info("🟢 Starting KIS Nasdaq Jev Trader daemon...")
        while True:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                logger.info("🛑 Stopped by user.")
                break
            except Exception as e:
                logger.error(f"❌ Main Loop Exception: {e}", exc_info=True)
                time.sleep(10)

if __name__ == "__main__":
    trader = KisNasdaqJevTrader()
    trader.start()
