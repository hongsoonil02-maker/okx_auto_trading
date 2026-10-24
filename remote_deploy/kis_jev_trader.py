# -*- coding: utf-8 -*-
"""
kis_jev_trader.py — Autonomous KOSPI/KOSDAQ Quant Execution Engine
Integrated with Jev AI Microstructure Gating & KIS OpenAPI (Korea Investment & Securities)

Constraints Applied:
- Long-Only Strategy (롱 전용)
- No ETFs (ETF 거래 불가)
- Stocks: 005930 (Samsung Elec), 000660 (SK Hynix), 035420 (Naver), 042700 (Hanmi Semi)
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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Read from the quant_system environment file which holds the keys
load_dotenv(os.path.join(CURRENT_DIR, "..", ".env"))

# Import local Jev client (Assuming Jev AI can process Korean tickers if fed properly)
try:
    from jev_nasdaq_client import query_jev_nasdaq, get_quotes_cache
except ImportError:
    sys.path.append(CURRENT_DIR)
    from jev_nasdaq_client import query_jev_nasdaq, get_quotes_cache

# ── LOGGING SETUP ──
LOG_FILE = os.path.join(CURRENT_DIR, "kis_jev_trader.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
    ]
)
logger = logging.getLogger("KisJevTrader")

# ── ROBUST API SESSION SETUP ──
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "DELETE"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

# ── KIS API CONFIGURATION ──
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

# Long Only, Non-ETF Universe (Top 10 Stocks)
# 삼성전자, SK하이닉스, LG에너지솔루션, 삼성바이오로직스, 현대차, 기아, 셀트리온, NAVER, 카카오, 한미반도체
SYMBOLS_UNIVERSE = [
    "005930", "000660", "373220", "207940", "005380", 
    "000270", "068270", "035420", "035720", "042700"
]

# ── 상장폐지/정리매매/거래불가 종목 영구 제외 블랙리스트 ──
# 007630: 폴루스바이오팜, 259290: 폴루스 등 상폐 종목은 보유주식 및 자동매매에서 완전히 없는 것으로 취급
EXCLUDED_SYMBOLS = {"007630", "259290"}

MAX_CONCURRENT_POSITIONS = 3
MAX_POSITION_PCT = 0.25          # 25% of equity per position
LOOP_INTERVAL_SEC = 10           # Poll market every 10 seconds
SIMULATION_MODE = False          # ✅ 실거래 모드 활성화 (가상 주문 X)
POSITIONS_FILE = os.path.join(CURRENT_DIR, "kis_positions.json")  # 포지션 영속화 파일

class KisJevTrader:
    def __init__(self):
        self.access_token = ""
        self.token_expired_at = 0
        self.account_info = {}
        self.positions = {}
        self.trade_cooldowns = {}
        self.is_market_open = False
        
        # 포지션 파일에서 복원
        self.load_positions()
        
        logger.info("=" * 70)
        logger.info("🚀 [KIS Jev Trader] Initializing Autonomous KOSPI/KOSDAQ Engine")
        logger.info(f"   REST Endpoint: {KIS_URL}")
        logger.info(f"   Account Number: {KIS_CANO}-{KIS_PRDT_ABRV}")
        logger.info(f"   Tracked Universe (Long-Only, No ETF): {', '.join(SYMBOLS_UNIVERSE)}")
        if self.positions:
            logger.info(f"   📂 Restored {len(self.positions)} positions from file: {list(self.positions.keys())}")
        logger.info("=" * 70)

    def save_positions(self):
        """포지션 정보를 JSON 파일로 저장 (프로세스 재시작 시 복원용)"""
        try:
            with open(POSITIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.positions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to save positions: {e}")

    def load_positions(self):
        """JSON 파일에서 포지션 정보 복원"""
        try:
            if os.path.exists(POSITIONS_FILE):
                with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
                    self.positions = json.load(f)
                # 상장폐지 종목 원천 배제
                for ex_sym in EXCLUDED_SYMBOLS:
                    if ex_sym in self.positions:
                        self.positions.pop(ex_sym, None)
                logger.info(f"📂 [Positions Restored] {len(self.positions)} positions loaded from {POSITIONS_FILE}")
            else:
                self.positions = {}
        except Exception as e:
            logger.error(f"❌ Failed to load positions: {e}")
            self.positions = {}

    def issue_token(self):
        """Issue or Renew KIS OAuth2 Access Token with file caching to prevent rate-limit (EGW00133)"""
        now = time.time()
        if self.access_token and now < self.token_expired_at:
            return True

        token_file = os.path.join(CURRENT_DIR, "kis_token.json")

        # 1. 파일 캐시에서 유효한 토큰 읽기 시도
        if not self.access_token and os.path.exists(token_file):
            try:
                with open(token_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                cached_token = cache_data.get("access_token")
                cached_expires_at = cache_data.get("expires_at", 0)
                if cached_token and now < cached_expires_at:
                    self.access_token = cached_token
                    self.token_expired_at = cached_expires_at
                    logger.info("🔐 [KIS API] Reused cached Access Token from kis_token.json.")
                    return True
            except Exception as e:
                logger.warning(f"⚠️ Failed to read cached token: {e}")

        # 2. 신규 토큰 발급
        url = f"{KIS_URL}/oauth2/tokenP"
        payload = {
            "grant_type": "client_credentials",
            "appkey": KIS_APP_KEY,
            "appsecret": KIS_APP_SECRET
        }
        try:
            resp = session.post(url, json=payload, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                self.access_token = data.get("access_token")
                expires_in = data.get("expires_in", 86400)
                self.token_expired_at = now + int(expires_in) - 600 # 10분 버퍼
                logger.info("🔐 [KIS API] Successfully issued new Access Token.")

                # 파일에 저장
                try:
                    with open(token_file, "w", encoding="utf-8") as f:
                        json.dump({
                            "access_token": self.access_token,
                            "expires_at": self.token_expired_at
                        }, f)
                except Exception as fe:
                    logger.warning(f"⚠️ Failed to cache token to file: {fe}")

                return True
            else:
                logger.error(f"❌ Failed to issue token: {resp.text}")
                # 1분당 1회 제한 등이 걸렸을 때 매 10초마다 계속 찌르지 않도록 60초 쿨다운
                self.token_expired_at = now + 60
                return False
        except Exception as e:
            logger.error(f"❌ Token issuance exception: {e}")
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
        """Check KST time for Korean Market (09:00 ~ 15:20)"""
        kst_now = datetime.now(timezone(timedelta(hours=9)))
        current_time = kst_now.time()
        
        # 주말 체크
        if kst_now.weekday() >= 5:
            self.is_market_open = False
            return False
            
        # 09:00 ~ 15:20 정규장
        open_time = datetime.strptime("09:00", "%H:%M").time()
        close_time = datetime.strptime("15:20", "%H:%M").time()
        
        self.is_market_open = (open_time <= current_time <= close_time)
        return self.is_market_open

    def fetch_account_balance(self):
        """Fetch KIS account balance"""
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
                output2 = data.get("output2", [])
                if output2:
                    summary = output2[0]
                    equity = float(summary.get("tot_evlu_amt", 0))
                    cash = float(summary.get("dnca_tot_amt", 0))
                    self.account_info = {"equity": equity, "cash": cash}
                    logger.info(f"💼 [Account] Equity: ₩{equity:,.0f} | Cash: ₩{cash:,.0f}")

                # ── 실제 KIS 계좌 보유 종목(output1)과 self.positions 자동 동기화 ──
                output1 = data.get("output1", [])
                positions_updated = False
                current_time = time.time()
                active_pdnos = set()

                for item in output1:
                    pdno = item.get("pdno", "")
                    # 상장폐지/거래불가 종목은 계좌 잔고에 있더라도 완전히 없는 것으로 취급
                    if pdno in EXCLUDED_SYMBOLS:
                        continue

                    try:
                        hldg_qty = int(item.get("hldg_qty", 0))
                        pchs_avg_pric = float(item.get("pchs_avg_pric", 0))
                    except (ValueError, TypeError):
                        continue

                    if pdno in SYMBOLS_UNIVERSE and hldg_qty > 0:
                        active_pdnos.add(pdno)
                        if pdno not in self.positions:
                            self.positions[pdno] = {
                                "qty": hldg_qty,
                                "avg_price": pchs_avg_pric,
                                "entry_time": current_time,
                                "peak_price": pchs_avg_pric
                            }
                            positions_updated = True
                            logger.info(f"📥 [Position Synced from KIS] {pdno} | Qty: {hldg_qty} | Avg: ₩{pchs_avg_pric:,.0f}")
                        else:
                            # 수량이나 매입가 변동이 있으면 실계좌 기준으로 업데이트
                            pos = self.positions[pdno]
                            if pos.get("qty") != hldg_qty:
                                pos["qty"] = hldg_qty
                                pos["avg_price"] = pchs_avg_pric
                                positions_updated = True

                # 계좌에서 청산/매도 완료되어 잔고에 없는 종목은 positions에서 제거
                for sym in list(self.positions.keys()):
                    if sym not in active_pdnos:
                        logger.info(f"📤 [Position Cleared] {sym} no longer in KIS balance, removing from tracker.")
                        self.positions.pop(sym, None)
                        positions_updated = True

                if positions_updated:
                    self.save_positions()
        except Exception as e:
            logger.error(f"❌ Balance fetch error: {e}")

    def get_current_price(self, symbol: str) -> float:
        """Fetch current price of a Korean stock"""
        if not self.issue_token():
            return 0.0
            
        url = f"{KIS_URL}/uapi/domestic-stock/v1/quotations/inquire-price"
        headers = self.get_headers("FHKST01010100")
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol
        }
        try:
            resp = session.get(url, headers=headers, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if "output" in data:
                    return float(data["output"].get("stck_prpr", 0))
        except Exception as e:
            logger.error(f"❌ Error fetching price for {symbol}: {e}")
        return 0.0

    def get_jev_score_from_kis_orderbook(self, symbol: str) -> dict:
        """Fetch real orderbook from KIS and compute Jev AI Score (Imbalance-based)"""
        if not self.issue_token():
            return {"approved": False, "score": 0.0, "reason": "TOKEN_ERROR"}
            
        url = f"{KIS_URL}/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn"
        headers = self.get_headers("FHKST01010200")
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol
        }
        try:
            resp = session.get(url, headers=headers, params=params, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                out1 = data.get("output1", {})
                if not out1:
                    return {"approved": False, "score": 0.0, "reason": "NO_DATA"}
                    
                total_ask = float(out1.get("total_askp_rsqn", 0))
                total_bid = float(out1.get("total_bidp_rsqn", 0))
                
                if (total_ask + total_bid) == 0:
                    return {"approved": False, "score": 0.0, "reason": "EMPTY_BOOK"}
                    
                # Orderbook Imbalance Calculation (-1.0 to 1.0)
                # 매수 잔량이 많으면 양수(Bullish), 매도 잔량이 많으면 음수(Bearish)
                imbalance = (total_bid - total_ask) / (total_ask + total_bid)
                
                # Jev Heuristic Model Mapping
                prob_up = min(max(0.50 + imbalance * 0.35, 0.10), 0.90)
                approved = (prob_up >= 0.60) # 60% 이상 확신일 때만 승인
                
                return {
                    "approved": approved,
                    "score": round(prob_up, 3),
                    "imbalance": round(imbalance, 3),
                    "reason": f"KIS_ORDERBOOK_IMB ({imbalance:+.2f})"
                }
        except Exception as e:
            logger.error(f"❌ KIS Orderbook fetch error: {e}")
            
        return {"approved": False, "score": 0.0, "reason": "EXCEPTION"}

    def submit_kis_order(self, symbol: str, side: str, qty: int, price: float = 0.0) -> bool:
        """Submit Cash Order (Market order if price=0.0)"""
        if SIMULATION_MODE:
            logger.info(f"🧪 [SIMULATION MODE] Simulated {side.upper()} order for {qty}x {symbol} at ₩{price:,.0f} (or Market)")
            return True

        if not self.issue_token():
            return False
            
        url = f"{KIS_URL}/uapi/domestic-stock/v1/trading/order-cash"
        tr_id = "TTTC0802U" if side == "buy" else "TTTC0801U"
        headers = self.get_headers(tr_id)
        
        ord_dvsn = "01" if price == 0.0 else "00" # 01: 시장가, 00: 지정가
        
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
                logger.info(f"✅ [Order Success] {side.upper()} {qty}x {symbol} | MSG: {data.get('msg1')}")
                return True
            else:
                logger.error(f"❌ [Order Failed] {side.upper()} {symbol}: {data.get('msg1')} - {data.get('msg_cd')}")
                return False
        except Exception as e:
            logger.error(f"❌ Order exception for {symbol}: {e}")
            return False

    def manage_open_positions(self):
        """Manage Trailing Stop, Stop Loss, Take Profit"""
        for sym, pos in list(self.positions.items()):
            if sym in EXCLUDED_SYMBOLS:
                self.positions.pop(sym, None)
                continue

            qty = pos["qty"]
            avg_price = pos["avg_price"]
            
            current_price = self.get_current_price(sym)
            if current_price <= 0:
                continue
                
            # Update peak price for trailing stop
            peak = max(pos.get("peak_price", avg_price), current_price)
            pos["peak_price"] = peak
            
            unrealized_plpc = (current_price - avg_price) / avg_price
            max_ret = (peak - avg_price) / avg_price
            trail_drop = (peak - current_price) / peak
            
            sl_pct = 0.02
            tp_pct = 0.05
            trail_arm = 0.03
            trail_delta = 0.015
            
            reason = ""
            if max_ret >= trail_arm and trail_drop >= trail_delta:
                reason = f"TRAILING_STOP (+{unrealized_plpc*100:.2f}%)"
            elif unrealized_plpc >= tp_pct:
                reason = f"TAKE_PROFIT (+{unrealized_plpc*100:.2f}%)"
            elif unrealized_plpc <= -sl_pct:
                reason = f"STOP_LOSS ({unrealized_plpc*100:.2f}%)"
                
            if reason:
                logger.info(f"🔄 [Closing Position] {sym} | Reason: {reason}")
                success = self.submit_kis_order(sym, "sell", qty, price=0.0)
                if success:
                    self.positions.pop(sym, None)
                    self.save_positions()
                    # 10 minute cooldown after exit
                    self.trade_cooldowns[sym] = time.time() + 600

    def evaluate_trading_opportunities(self):
        """Scan Universe and execute Long-only trades"""
        active_count = len(self.positions)
        if active_count >= MAX_CONCURRENT_POSITIONS:
            return

        equity = float(self.account_info.get("equity", 10000000))
        target_size = equity * MAX_POSITION_PCT

        for sym in SYMBOLS_UNIVERSE:
            if sym in EXCLUDED_SYMBOLS or sym in self.positions:
                continue
            if time.time() < self.trade_cooldowns.get(sym, 0.0):
                continue
                
            current_price = self.get_current_price(sym)
            if current_price <= 0:
                continue

            # KIS 호가창 기반 Jev AI 분석 (Orderbook Imbalance)
            jev_resp = self.get_jev_score_from_kis_orderbook(sym)
            is_approved = jev_resp.get("approved", False)
            score = jev_resp.get("score", 0.0)
            reason = jev_resp.get("reason", "")
            
            logger.info(f"🔎 [Jev AI Scan] {sym} | Score: {score:.3f} | Approved: {is_approved} | {reason}")
            
            if is_approved:
                qty = int(target_size / current_price)
                if qty < 1:
                    continue

                logger.info(f"🎯 [Executing Trade] Approved by Jev AI! Symbol: {sym} | Price: ₩{current_price:,.0f} | Qty: {qty}")
                success = self.submit_kis_order(sym, "buy", qty, price=0.0) # Market order
                if success:
                    self.positions[sym] = {
                        "qty": qty,
                        "avg_price": current_price,
                        "entry_time": time.time(),
                        "peak_price": current_price
                    }
                    self.save_positions()
                self.trade_cooldowns[sym] = time.time() + 300

    def run_cycle(self):
        if not self.check_market_clock():
            # logger.info("🌙 [Market Closed] Waiting for KST 09:00 정규장...")
            time.sleep(60)
            return

        self.fetch_account_balance()
        self.manage_open_positions()
        self.evaluate_trading_opportunities()

    def start(self):
        logger.info("🟢 [Daemon Started] Entering continuous trading loop...")
        while True:
            try:
                self.run_cycle()
                time.sleep(LOOP_INTERVAL_SEC)
            except KeyboardInterrupt:
                logger.info("🛑 [Stop Signal] Stopping KIS Jev Trader gracefully.")
                break
            except Exception as e:
                logger.error(f"❌ [Main Loop Exception] {e}")
                time.sleep(10)

if __name__ == "__main__":
    trader = KisJevTrader()
    trader.start()
