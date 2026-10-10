# -*- coding: utf-8 -*-
"""
kis_qqq_regime_trader.py — 나스닥 지수 레짐 트레이더 (QQQ/TQQQ + 200일선)
전략 (9.7년 검증, 생존자 편향 없음 — 지수 자체를 매매):
- 신호: QQQ 현재가 > QQQ 200일 이동평균  → 상승 레짐 / 아니면 하락 레짐
- 상승 레짐: QQQ (1-w) + TQQQ (w) 보유, 목표 대비 ±20% 이탈 시에만 리밸런싱
- 하락 레짐: 전량 현금
- 하루 1회, 15:30~15:50 ET 결정 창에서만 주문 (장 마감 직전 = 일봉 종가 근사)
백테스트 (2017.01~2026.10, 수수료 0.25%/편도):
  w=0.0: CAGR +14.6% MDD 25%  | w=0.3: +21.4% MDD 37% | w=0.5: +25.5% MDD 43% | w=1.0: +33.6% MDD 58%
전환: 기존 추세봇 보유 종목(AAPL/NVDA/TSLA)은 첫 결정 창에서 청산 후 지수 ETF로 전환
"""

import os
import json
import time
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import yfinance as yf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, "..", ".env"))

LOG_FILE = os.path.join(CURRENT_DIR, "kis_qqq_regime_trader.log")
logger = logging.getLogger("KisQqqRegimeTrader")
logger.setLevel(logging.INFO)
logger.propagate = False
_fh = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=3, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_fh)

session = requests.Session()
_adapter = HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
                                         allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]))
session.mount("https://", _adapter)

KIS_APP_KEY = os.getenv("KIS_API_KEY", "")
KIS_APP_SECRET = os.getenv("KIS_SECRET", "")
_acc = os.getenv("KIS_ACCOUNT_NO", "")
if "-" in _acc:
    KIS_CANO, KIS_PRDT_ABRV = _acc.split("-")
elif len(_acc) >= 10:
    KIS_CANO, KIS_PRDT_ABRV = _acc[:8], _acc[8:10]
else:
    KIS_CANO, KIS_PRDT_ABRV = "", ""
KIS_URL = os.getenv("KIS_URL", "https://openapi.koreainvestment.com:9443")
TOKEN_FILE = os.path.join(CURRENT_DIR, "kis_token.json")
STATE_FILE = os.path.join(CURRENT_DIR, "kis_qqq_regime_state.json")
LEGACY_POSITIONS_FILE = os.path.join(CURRENT_DIR, "kis_nasdaq_trend_positions.json")

# ── 전략 설정 (손익 프로필을 바꾸려면 TQQQ_WEIGHT 하나만 조정) ──
SIGNAL_SYMBOL = "QQQ"
SMA_PERIOD = 200
TQQQ_WEIGHT = float(os.getenv("KIS_TQQQ_WEIGHT", "0.5"))          # 0.0 ~ 1.0
TARGET_WEIGHTS = {"QQQ": 1.0 - TQQQ_WEIGHT, "TQQQ": TQQQ_WEIGHT}
REBALANCE_TOLERANCE = 0.20          # 목표 대비 ±20% 벗어날 때만 매매 (불필요한 수수료 억제)
# 50% QQQ Regime ETF + 50% JEV Alpha 개별주 황금비 분할
ALLOCATION_RATIO = float(os.getenv("KIS_REGIME_ALLOCATION", "0.50")) # 전체 계좌 자산 중 ETF 봇 할당 비중 (기본 50%)
INVEST_BUFFER = 0.97 * ALLOCATION_RATIO                              # 할당 비중의 97%만 투자 (수수료/여유금)
DECISION_START_ET = (15, 30)
DECISION_END_ET = (15, 50)
RETRY_INTERVAL_SEC = 300            # 결정 창 내 미체결 시 5분 후 취소·재주문
POLL_INTERVAL_SEC = 60

US_MARKET_HOLIDAYS = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}
NY = ZoneInfo("America/New_York")


class KisQqqRegimeTrader:
    def __init__(self):
        self.access_token = ""
        self.token_expired_at = 0.0
        self.state = {"legacy_liquidate": [], "legacy_seeded": False, "open_orders": {},
                      "last_decision_date": "", "last_attempt_ts": 0.0, "last_regime": ""}
        self.holdings: Dict[str, dict] = {}       # 원장 기준 보유 (sym -> qty, avg)
        self.account_info = {"total_buying_power_usd": 0.0, "usd_cash": 0.0, "krw_cash": 0.0, "fx_rate": 0.0}
        self.sma200 = 0.0
        self.sma_date = ""
        self.last_status_log = 0.0
        self.load_state()
        self.seed_legacy()

        logger.info("=" * 75)
        logger.info("🚀 [KIS QQQ Regime Trader v1.0] 지수 ETF + 200일선 레짐")
        logger.info(f"   Signal: {SIGNAL_SYMBOL} > SMA{SMA_PERIOD} | Targets (bull): " +
                    ", ".join(f"{s} {w*100:.0f}%" for s, w in TARGET_WEIGHTS.items()) + " | Bear: 100% cash")
        logger.info(f"   Rebalance tolerance ±{REBALANCE_TOLERANCE*100:.0f}% | Decision window "
                    f"{DECISION_START_ET[0]:02d}:{DECISION_START_ET[1]:02d}~{DECISION_END_ET[0]:02d}:{DECISION_END_ET[1]:02d} ET")
        if self.state["legacy_liquidate"]:
            logger.info(f"   🔁 Legacy liquidation pending: {self.state['legacy_liquidate']}")
        logger.info("=" * 75)

    # ── 상태 ──
    def load_state(self):
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    self.state.update(json.load(f))
        except Exception as e:
            logger.warning(f"⚠️ State load error: {e}")

    def save_state(self):
        try:
            tmp = f"{STATE_FILE}.tmp.{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
            os.replace(tmp, STATE_FILE)
        except Exception as e:
            logger.error(f"❌ State save error: {e}")

    def seed_legacy(self):
        """이전 추세봇이 들고 있던 종목을 1회 청산 목록으로 등록"""
        if self.state.get("legacy_seeded"):
            return
        legacy = []
        try:
            if os.path.exists(LEGACY_POSITIONS_FILE):
                with open(LEGACY_POSITIONS_FILE, "r", encoding="utf-8") as f:
                    legacy = [s for s in json.load(f).keys() if s not in TARGET_WEIGHTS]
        except Exception as e:
            logger.warning(f"⚠️ Legacy positions read error: {e}")
        self.state["legacy_liquidate"] = legacy
        self.state["legacy_seeded"] = True
        self.save_state()

    # ── 토큰 ──
    def issue_token(self) -> bool:
        now = time.time()
        if self.access_token and now < self.token_expired_at:
            return True
        if not self.access_token and os.path.exists(TOKEN_FILE):
            try:
                with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if d.get("expires_at", 0) > now + 300:
                    self.access_token, self.token_expired_at = d["access_token"], d["expires_at"]
                    return True
            except Exception:
                pass
        try:
            r = session.post(f"{KIS_URL}/oauth2/tokenP", json={"grant_type": "client_credentials",
                             "appkey": KIS_APP_KEY, "appsecret": KIS_APP_SECRET}, timeout=10)
            if r.status_code == 200:
                d = r.json()
                self.access_token = d.get("access_token", "")
                self.token_expired_at = now + int(d.get("expires_in", 86400))
                tmp = f"{TOKEN_FILE}.tmp.{os.getpid()}"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump({"access_token": self.access_token, "expires_at": self.token_expired_at}, f)
                os.replace(tmp, TOKEN_FILE)
                logger.info("🔑 [Auth] Token issued & cached (atomic)")
                return True
            logger.error(f"❌ Token failed: {r.status_code} {r.text}")
        except Exception as e:
            logger.error(f"❌ Token exception: {e}")
        return False

    def headers(self, tr_id: str) -> dict:
        self.issue_token()
        return {"content-type": "application/json", "authorization": f"Bearer {self.access_token}",
                "appkey": KIS_APP_KEY, "appsecret": KIS_APP_SECRET, "tr_id": tr_id, "custtype": "P"}

    # ── 시계 ──
    def market_state(self) -> str:
        now = datetime.now(NY)
        if now.weekday() >= 5 or now.strftime("%Y-%m-%d") in US_MARKET_HOLIDAYS:
            return "CLOSED"
        m = now.hour * 60 + now.minute
        if not (9 * 60 + 30 <= m <= 16 * 60):
            return "CLOSED"
        s, e = DECISION_START_ET[0] * 60 + DECISION_START_ET[1], DECISION_END_ET[0] * 60 + DECISION_END_ET[1]
        return "DECISION" if s <= m <= e else "OPEN"

    # ── 신호 ──
    def refresh_sma(self):
        """완성봉 기준 200일 이동평균 (오늘 미완성 봉 제외)"""
        today = datetime.now(NY).strftime("%Y-%m-%d")
        if self.sma_date == today and self.sma200 > 0:
            return
        try:
            df = yf.Ticker(SIGNAL_SYMBOL).history(period="2y", interval="1d")
            if df is None or len(df) < SMA_PERIOD + 5:
                logger.warning("⚠️ insufficient QQQ daily bars")
                return
            if df.index[-1].date() >= datetime.now(NY).date():
                df = df.iloc[:-1]
            self.sma200 = float(df["Close"].tail(SMA_PERIOD).mean())
            self.sma_date = today
            logger.info(f"📊 [SMA{SMA_PERIOD}] {SIGNAL_SYMBOL} = ${self.sma200:.2f} (완성봉 {df.index[-1].date()} 기준)")
        except Exception as e:
            logger.error(f"❌ SMA fetch error: {e}")

    def get_price(self, sym: str) -> Optional[float]:
        try:
            r = session.get(f"{KIS_URL}/uapi/overseas-price/v1/quotations/price", headers=self.headers("HHDFS00000300"),
                            params={"AUTH": "", "EXCD": "NAS", "SYMB": sym}, timeout=5)
            if r.status_code == 200:
                last = float(r.json().get("output", {}).get("last", 0.0))
                if last > 0:
                    return last
        except Exception as e:
            logger.error(f"❌ [{sym}] quote exception: {e}")
        return None

    # ── 계좌 ──
    def fx_rate(self) -> float:
        if not hasattr(self, "_fx"):
            self._fx = (0.0, 1380.0)
        if time.time() - self._fx[0] < 3600 and self._fx[1] > 0:
            return self._fx[1]
        try:
            df = yf.Ticker("USDKRW=X").history(period="1d", interval="5m")
            rate = float(df["Close"].iloc[-1])
            if rate > 900:
                self._fx = (time.time(), rate)
                return rate
        except Exception:
            pass
        self._fx = (time.time(), 1380.0)
        return 1380.0

    def buying_power(self) -> float:
        krw = 0.0
        try:
            r = session.get(f"{KIS_URL}/uapi/domestic-stock/v1/trading/inquire-psbl-order", headers=self.headers("TTTC8908R"),
                            params={"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "PDNO": "005930", "ORD_UNPR": "0",
                                    "ORD_DVSN": "01", "CMA_EVLU_AMT_ICLD_YN": "Y", "OVRS_ICLD_YN": "Y"}, timeout=10)
            if r.status_code == 200:
                o = r.json().get("output", {})
                krw = float(o.get("max_buy_amt", o.get("ord_psbl_cash", 0.0)))
        except Exception as e:
            logger.warning(f"⚠️ KRW power error: {e}")
        usd = 0.0
        try:
            r = session.get(f"{KIS_URL}/uapi/overseas-stock/v1/trading/inquire-present-balance", headers=self.headers("TTTS3007R"),
                            params={"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "WCRC_FRCR_DVS_CD": "02",
                                    "NATN_CD": "840", "TR_MKET_CD": "00", "INQR_DVS_CD": "00"}, timeout=10)
            if r.status_code == 200:
                for row in r.json().get("output1", []):
                    if row.get("crcy_cd") == "USD":
                        usd = float(row.get("frcr_dnca_amt", 0.0))
        except Exception as e:
            logger.warning(f"⚠️ USD cash error: {e}")
        fx = self.fx_rate()
        total = usd + (krw / fx if krw > 0 else 0.0)
        self.account_info.update({"usd_cash": usd, "krw_cash": krw, "fx_rate": fx, "total_buying_power_usd": total})
        return total

    def reconcile(self):
        """원장(TTTS3012R) 기준 보유 동기화 — 전략 종목 + 레거시 청산 대상만 추적"""
        try:
            r = session.get(f"{KIS_URL}/uapi/overseas-stock/v1/trading/inquire-balance", headers=self.headers("TTTS3012R"),
                            params={"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "OVRS_EXCG_CD": "NASD",
                                    "TR_CRCY_CD": "USD", "CTX_AREA_FK200": "", "CTX_AREA_NK200": ""}, timeout=8)
            if r.status_code != 200 or r.json().get("rt_cd") != "0":
                logger.warning(f"⚠️ reconcile failed HTTP {r.status_code}")
                return
            held = {}
            for row in r.json().get("output1", []):
                sym = row.get("ovrs_pdno", "").strip().upper()
                try:
                    qty = int(float(row.get("ovrs_cblc_qty", 0)))
                    avg = float(row.get("pchs_avg_pric", 0.0))
                except (ValueError, TypeError):
                    continue
                if sym and qty > 0 and (sym in TARGET_WEIGHTS or sym in self.state["legacy_liquidate"]):
                    held[sym] = {"qty": qty, "avg": avg}
            self.holdings = held
            # 레거시 종목이 원장에서 사라졌으면 청산 완료 처리
            remaining = [s for s in self.state["legacy_liquidate"] if s in held]
            if remaining != self.state["legacy_liquidate"]:
                done = set(self.state["legacy_liquidate"]) - set(remaining)
                logger.info(f"✅ [Legacy Liquidated] {sorted(done)} 원장 확인 완료")
                self.state["legacy_liquidate"] = remaining
                self.save_state()
        except Exception as e:
            logger.error(f"❌ reconcile exception: {e}")

    # ── 주문 ──
    def order(self, sym: str, side: str, qty: int, price: float) -> Optional[str]:
        if qty < 1:
            return None
        tr = "TTTT1002U" if side == "buy" else "TTTT1006U"
        body = {"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "OVRS_EXCG_CD": "NASD", "PDNO": sym,
                "ORD_QTY": str(qty), "OVRS_ORD_UNPR": f"{price:.2f}", "ORD_SVR_DVSN_CD": "0", "ORD_DVSN": "00"}
        try:
            logger.info(f"📤 [Order] {side.upper()} {sym} x {qty} @ ${price:.2f}")
            r = session.post(f"{KIS_URL}/uapi/overseas-stock/v1/trading/order", headers=self.headers(tr), json=body, timeout=10)
            d = r.json()
            if r.status_code == 200 and d.get("rt_cd") == "0":
                odno = d.get("output", {}).get("ODNO", "") or "UNKNOWN"
                logger.info(f"✅ [Accepted] {side.upper()} {sym} x {qty} | ODNO {odno}")
                self.state["open_orders"][sym] = {"odno": odno, "qty": qty, "side": side, "ts": time.time()}
                self.save_state()
                return odno
            logger.error(f"❌ [Rejected] {side.upper()} {sym} | {d.get('msg_cd')}: {d.get('msg1')}")
        except Exception as e:
            logger.error(f"❌ [Order exception] {sym}: {e}")
        return None

    def cancel_open_orders(self):
        for sym, o in list(self.state["open_orders"].items()):
            odno = o.get("odno", "")
            if odno and odno != "UNKNOWN":
                body = {"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "OVRS_EXCG_CD": "NASD", "PDNO": sym,
                        "ORG_ODNO": str(odno), "RVSE_CNCL_DVSN_CD": "02", "ORD_QTY": str(o.get("qty", 0)),
                        "OVRS_ORD_UNPR": "0", "ORD_SVR_DVSN_CD": "0"}
                try:
                    r = session.post(f"{KIS_URL}/uapi/overseas-stock/v1/trading/order-rvsecncl",
                                     headers=self.headers("TTTT1004U"), json=body, timeout=10)
                    d = r.json()
                    if d.get("rt_cd") == "0":
                        logger.info(f"🚫 [Cancelled] {sym} ODNO {odno}")
                    else:
                        logger.info(f"ℹ️ [Cancel skipped] {sym} ODNO {odno} ({d.get('msg1')}) — 이미 체결됐을 가능성")
                except Exception as e:
                    logger.warning(f"⚠️ cancel exception {sym}: {e}")
            self.state["open_orders"].pop(sym, None)
        self.save_state()

    # ── 결정 ──
    def decide_and_trade(self):
        self.refresh_sma()
        qqq_px = self.get_price(SIGNAL_SYMBOL)
        if not qqq_px or self.sma200 <= 0:
            logger.warning("⚠️ 신호 데이터 부족 — 이번 결정 건너뜀")
            return
        bullish = qqq_px > self.sma200
        regime = "BULL" if bullish else "BEAR"
        if regime != self.state.get("last_regime"):
            logger.warning(f"⚡ [Regime] {self.state.get('last_regime') or '—'} → {regime} | QQQ ${qqq_px:.2f} vs SMA200 ${self.sma200:.2f}")
            self.state["last_regime"] = regime

        self.cancel_open_orders()
        self.reconcile()
        prices = {s: self.get_price(s) for s in set(list(TARGET_WEIGHTS) + self.state["legacy_liquidate"])}
        power = self.buying_power()
        holdings_value = sum(h["qty"] * (prices.get(s) or h["avg"]) for s, h in self.holdings.items())
        investable = (power + holdings_value) * INVEST_BUFFER

        # 1. 레거시 종목 청산 (레짐 무관)
        for sym in list(self.state["legacy_liquidate"]):
            h = self.holdings.get(sym)
            px = prices.get(sym)
            if h and px:
                logger.info(f"🔁 [Legacy Exit] {sym} x {h['qty']} — 지수 ETF 전략으로 전환 청산")
                self.order(sym, "sell", h["qty"], px)

        # 2. 하락 레짐: 전량 현금
        if not bullish:
            for sym in TARGET_WEIGHTS:
                h = self.holdings.get(sym)
                px = prices.get(sym)
                if h and px:
                    logger.warning(f"🛡️ [Bear Exit] {sym} x {h['qty']} @ ${px:.2f} — 200일선 하회, 현금 전환")
                    self.order(sym, "sell", h["qty"], px)
            self._mark_attempt()
            logger.info(f"🌙 [Decision] BEAR — 현금 100% 유지 | QQQ ${qqq_px:.2f} < SMA200 ${self.sma200:.2f}")
            return

        # 3. 상승 레짐: 목표 비중 대비 ±tolerance 벗어난 종목만 리밸런싱 (매도 먼저, 매수 나중)
        plan = []
        for sym, w in TARGET_WEIGHTS.items():
            px = prices.get(sym)
            if not px or w <= 0:
                continue
            target_val = investable * w
            cur_qty = self.holdings.get(sym, {}).get("qty", 0)
            cur_val = cur_qty * px
            if abs(cur_val - target_val) <= target_val * REBALANCE_TOLERANCE:
                continue
            diff_qty = int(target_val / px) - cur_qty
            if diff_qty != 0:
                plan.append((sym, diff_qty, px, cur_val, target_val))
        plan.sort(key=lambda x: x[1])  # 매도(음수)부터
        available = power
        for sym, dq, px, cur_val, target_val in plan:
            if dq < 0:
                logger.info(f"⚖️ [Rebalance] {sym} 매도 {-dq}주 (${cur_val:,.0f} → 목표 ${target_val:,.0f})")
                self.order(sym, "sell", -dq, px)
                available += -dq * px * 0.99
            else:
                cost = dq * px * 1.003
                if cost > available:
                    dq = int(available / (px * 1.003))
                if dq < 1:
                    logger.info(f"ℹ️ [Rebalance] {sym} 매수 보류 — 가용 자금 부족 (${available:,.0f})")
                    continue
                logger.info(f"⚖️ [Rebalance] {sym} 매수 {dq}주 (${cur_val:,.0f} → 목표 ${target_val:,.0f})")
                self.order(sym, "buy", dq, px)
                available -= dq * px * 1.003
        if not plan:
            logger.info(f"✅ [Decision] BULL — 목표 비중 유지 중 (리밸런싱 불필요) | QQQ ${qqq_px:.2f} > SMA200 ${self.sma200:.2f}")
        self._mark_attempt()

    def _mark_attempt(self):
        self.state["last_attempt_ts"] = time.time()
        self.state["last_decision_date"] = datetime.now(NY).strftime("%Y-%m-%d")
        self.save_state()

    # ── 루프 ──
    def run_cycle(self):
        st = self.market_state()
        now = time.time()
        if st == "DECISION":
            today = datetime.now(NY).strftime("%Y-%m-%d")
            first_today = self.state.get("last_decision_date") != today
            retry_due = (now - float(self.state.get("last_attempt_ts", 0.0))) >= RETRY_INTERVAL_SEC
            if first_today or (self.state["open_orders"] and retry_due):
                self.decide_and_trade()
                time.sleep(2.0)
                self.reconcile()
        interval = 1800 if st == "CLOSED" else 600
        if now - self.last_status_log > interval:
            self.refresh_sma()
            self.reconcile()
            power = self.buying_power()
            pos = ", ".join(f"{s} {h['qty']}주" for s, h in self.holdings.items()) or "현금 100%"
            logger.info(f"💓 [{st}] Regime {self.state.get('last_regime') or '—'} | SMA200 ${self.sma200:.2f} | "
                        f"Power ${power:,.0f} | Holdings: {pos}" +
                        (f" | Legacy pending: {self.state['legacy_liquidate']}" if self.state["legacy_liquidate"] else ""))
            self.last_status_log = now

    def start(self):
        logger.info("🟢 [Daemon Started] 지수 레짐 루프 진입")
        self.refresh_sma()
        self.reconcile()
        self.buying_power()
        while True:
            try:
                self.run_cycle()
            except Exception as e:
                logger.error(f"❌ [Cycle Exception] {e}")
            time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    KisQqqRegimeTrader().start()
