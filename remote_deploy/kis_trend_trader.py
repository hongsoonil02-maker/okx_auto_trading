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
from typing import Dict, List, Optional, Any
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

logger.addHandler(stream_handler)
logger.addHandler(file_handler)

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
# KOSPI/KOSDAQ 대표 주도 대형주 10종목
UNIVERSE = {
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


class KisTrendTrader:
    def __init__(self):
        self.access_token = ""
        self.token_expired_at = 0
        self.account_info = {"equity": 0.0, "cash": 0.0}
        self.positions: Dict[str, dict] = {}
        self.daily_indicators: Dict[str, dict] = {}
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

    def save_positions(self):
        """포지션 정보를 JSON 파일로 영속화"""
        try:
            with open(POSITIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.positions, f, ensure_ascii=False, indent=2)
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
                
                with open(TOKEN_FILE, "w", encoding="utf-8") as f:
                    json.dump({"access_token": self.access_token, "expires_at": self.token_expired_at}, f)
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

    def check_market_clock(self) -> bool:
        """KST 기준 정규장 시간 (09:05 ~ 15:15) 확인. 개장 직후 5분 갭 변동성 안정화 대기"""
        kst_now = datetime.now(timezone(timedelta(hours=9)))
        
        # 주말 체크
        if kst_now.weekday() >= 5:
            return False
            
        current_time = kst_now.time()
        open_time = datetime.strptime("09:05", "%H:%M").time()
        close_time = datetime.strptime("15:15", "%H:%M").time()
        
        return (open_time <= current_time <= close_time)

    def fetch_daily_indicators(self):
        """일 1회 유니버스 전 종목의 60일 일봉을 수집하여 지표 산출"""
        today_str = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
        if self.last_indicator_update == today_str and self.daily_indicators:
            return

        logger.info(f"📊 [Indicator Update] Updating Daily Indicators for Universe ({today_str})...")
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
                
                # 전일 기준 20일 최고가 & 10일 최저가 (오늘 봉 제외)
                donchian_high20 = float(df["high"].iloc[:-1].tail(20).max())
                donchian_low10 = float(df["low"].iloc[:-1].tail(10).min())
                
                self.daily_indicators[sym] = {
                    "donchian_high20": donchian_high20,
                    "donchian_low10": donchian_low10,
                    "sma20": sma20,
                    "sma60": sma60,
                    "atr14": atr14,
                    "last_close": float(df["close"].iloc[-1])
                }
                logger.info(
                    f"   ✓ {meta['name']}({sym}) | 20일고가: ₩{donchian_high20:,.0f} | "
                    f"20선: ₩{sma20:,.0f} | 60선: ₩{sma60:,.0f} | ATR: ₩{atr14:,.0f}"
                )
            except Exception as e:
                logger.error(f"❌ Failed to calculate indicators for {sym}: {e}")

        self.last_indicator_update = today_str
        logger.info(f"✅ Daily Indicators updated successfully for {len(self.daily_indicators)} stocks.")

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
                    self.account_info["equity"] = equity

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
        """inquire-psbl-order (TTTC8908R) 호출을 통한 실제 주문가능현금(ord_psbl_cash) 엄격 확인"""
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
            "CMA_EVLU_AMT_ICLD_YN": "N",
            "OVRS_ICLD_YN": "N"
        }
        try:
            resp = session.get(url, headers=headers, params=params, timeout=4)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("rt_cd") == "0" and "output" in data:
                    cash = float(data["output"].get("ord_psbl_cash", 0))
                    self.account_info["cash"] = cash
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

            entry_price = float(pos["entry_price"])
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

    def evaluate_entry_opportunities(self):
        """돈키언 채널 20일 신고가 돌파 + 60일선 정배열 추세 스캔"""
        active_count = len(self.positions)
        if active_count >= MAX_CONCURRENT_POSITIONS:
            return

        # 주문가능현금 실시간 확인
        available_cash = self.fetch_available_cash()
        if available_cash < 500_000: # 최소 50만 원 미만이면 신규 진입 불가
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

            # 진입 조건: 20일 신고가 돌파 + 60일선 위 + 20일선 > 60일선 정배열
            is_breakout = (current_price > donchian_high)
            is_trend_aligned = (current_price > sma60) and (sma20 >= sma60)

            if is_breakout and is_trend_aligned:
                momentum_score = (current_price - sma60) / sma60
                candidates.append((sym, meta["name"], current_price, momentum_score, ind["atr14"]))

        # 모멘텀 순 정렬
        candidates.sort(key=lambda x: x[3], reverse=True)

        for sym, name, price, score, atr in candidates:
            if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
                break

            qty = int(target_size / price)
            if qty < 1:
                continue

            logger.info(f"🎯 [Trend Breakout Detected] {name}({sym}) | 돌파가: ₩{price:,.0f} | 60선괴리: +{score*100:.1f}% | 수량: {qty}주")
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
        if not self.check_market_clock():
            now = time.time()
            if now - self.last_market_status_log > 600:
                kst = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"🌙 [Market Closed] Current KST: {kst} | 정규장(09:05~15:15) 대기 중...")
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
