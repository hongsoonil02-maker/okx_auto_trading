# -*- coding: utf-8 -*-
"""
alpaca_cameron_pullback_trader.py — Ross Cameron Small-Cap Momentum Pullback Strategy
Designed for Alpaca Paper Trading ($100k Account)

Strategy Core Philosophy (Ross Cameron 1-Min First Pullback):
1. Pre-Market / Market Open Scanner:
   - Price: $2.00 ~ $20.00
   - Gap-up >= 10%
   - High relative volume & top gainers / active penny tickers
2. Entry Setup (1-Minute Candle Pattern):
   - Clear morning surge (Impulse Wave: at least 2 strong bullish candles)
   - First Pullback: 1~3 bearish candles
   - Pullback retracement <= 50% of the impulse leg
   - Pullback volume significantly lower than impulse volume
   - Price remains above VWAP and 9 EMA
   - Entry Trigger: Next candle breaks above the High of the previous pullback candle
3. Risk & Order Management:
   - Position Sizing: 20% of account equity per trade (Cap at $20,000)
   - Native Alpaca Bracket Order:
       * Stop-Loss: Swing Low (Low of the pullback candle) -> Market Stop
       * Take-Profit: 1:2 Risk-to-Reward minimum -> Limit Order
   - Hard Cutoff: No new entries after 10:30 AM EST.
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
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

# ── LOGGING CONFIG ──
LOG_FILE = os.path.join(CURRENT_DIR, "alpaca_cameron_30m_trader.log")
STATE_FILE = os.path.join(CURRENT_DIR, "state", "alpaca_cameron_30m_state.json")
ORDER_TAG = "cam30m"
logger = logging.getLogger("AlpacaCameronTrader")
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
# 로테이션 없는 FileHandler는 디스크를 잠식함 (하루 5MB+) → Rotating 적용
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=2, encoding="utf-8")
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)
    # systemd가 stdout을 같은 로그로 append하므로 콘솔 출력은 대화형 실행 시에만 (중복 방지)
    if sys.stdout.isatty():
        logger.addHandler(stream_handler)

# ── CREDENTIALS & SESSIONS ──
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    sys.exit("❌ ALPACA_API_KEY / ALPACA_SECRET_KEY missing in quant_system/.env")
ALPACA_REST_URL = os.getenv("ALPACA_REST_URL", "https://paper-api.alpaca.markets/v2")
ALPACA_DATA_URL = "https://data.alpaca.markets/v2"

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type": "application/json"
}

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

from jev.universal_jev_gate import UniversalJevGate

# ── STRATEGY HYPERPARAMETERS (1-Hour Swing Optimized) ──
TIMEFRAME = "30Min"          # 30M 병행 인스턴스 (백테스트: 6개월 +31.6%, N=246, PF 2.67 — 수익 극대화 TF)
MIN_PRICE = 2.00
MAX_PRICE = 50.00
MIN_GAP_PCT = 3.0           # Minimum 3% momentum surge on 1H
MAX_PORTFOLIO_PCT = 0.20    # 20% per trade (Dynamic Half-Kelly Sizing: 16% ~ 20%)
MAX_CONCURRENT_TRADES = 5   # Max 5 concurrent positions (가동률 100% 완전 활용)
RISK_REWARD_RATIO = 3.0     # 1:3.0 R:R (Optimized via 6-month sweep: PF 3.07, +38.4%)
MAX_SL_PCT = 0.07           # Max 7% stop loss distance (소형주 30분봉 실전 변동성 5~7% 완벽 수용)
MIN_SL_PCT = 0.01           # Min 1% stop loss distance
CUTOFF_HOUR_EST = 14        # Allow entries until 2:30 PM EST (장마감 1시간 30분 전 컷오프)
CUTOFF_MIN_EST = 30
COOLDOWN_SEC = 900          # 15분 재진입 쿨다운 (연속 중복 발주 원천 차단)
ENTRY_GRACE_SEC = 90        # 90초 체결 대기 보호 유예시간 (Stale 오판 방지)


class AlpacaCameronPullbackTrader:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.account_equity = 100000.0
        self.watchlist: List[str] = []
        self.last_scanned_time = 0
        self.jev_gate = UniversalJevGate(confidence_threshold=0.60)
        # 자기 진입 종목만 추적 (1H 인스턴스와 같은 계좌 병행 — 서로 포지션 오인 방지)
        self.own_entries: Dict[str, float] = {}
        self.cooldowns: Dict[str, float] = {}
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    self.own_entries = json.load(f)
        except Exception:
            self.own_entries = {}
        logger.info(f"🚀 [Ross Cameron 30M Swing Engine + JEV Gate] Initializing... Dry Run: {self.dry_run}")
        logger.info(f"   REST Endpoint: {ALPACA_REST_URL}")

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Fetches active pending orders from Alpaca"""
        try:
            resp = session.get(f"{ALPACA_REST_URL}/orders?status=open", headers=HEADERS, timeout=6)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"❌ Error fetching open orders: {e}")
        return []

    def calculate_dynamic_position_size(self, setup: Dict[str, Any], trigger_price: float) -> int:
        """
        Dynamic Half-Kelly Position Sizing:
        - Baseline: p=0.48 (win rate), b=3.0 (R:R) -> Full Kelly f* = 30.6%
        - Half-Kelly (Capital Preservation): 16% ~ 20%
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

    def get_account_summary(self) -> Dict[str, Any]:
        """Fetches equity and buying power from Alpaca"""
        try:
            resp = session.get(f"{ALPACA_REST_URL}/account", headers=HEADERS, timeout=6)
            if resp.status_code == 200:
                acc = resp.json()
                self.account_equity = float(acc.get("equity", 100000.0))
                return {
                    "equity": self.account_equity,
                    "buying_power": float(acc.get("buying_power", 0.0)),
                    "status": acc.get("status")
                }
        except Exception as e:
            logger.error(f"❌ Failed to get account summary: {e}")
        return {"equity": self.account_equity, "buying_power": 0.0, "status": "UNKNOWN"}

    def get_market_clock(self) -> Dict[str, Any]:
        """Checks if US market is open"""
        try:
            resp = session.get(f"{ALPACA_REST_URL}/clock", headers=HEADERS, timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"❌ Failed to get market clock: {e}")
        return {"is_open": False}

    def scan_top_movers(self, limit: int = 25) -> List[str]:
        """
        Scans for Ross Cameron momentum runners:
        - $2.00 <= Price <= $50.00
        - Percent Change >= 3%
        """
        logger.info("🔍 [Scanner] Scanning top gainers from Alpaca Screener API...")
        candidates = []  # Strictly small-cap / penny momentum runners
        try:
            # 1. Alpaca Movers API (top=50 max supported by Alpaca)
            resp = session.get("https://data.alpaca.markets/v1beta1/screener/stocks/movers?top=50", headers=HEADERS, timeout=6)
            if resp.status_code == 200:
                gainers = resp.json().get("gainers", [])
                for g in gainers:
                    sym = g.get("symbol", "")
                    price = float(g.get("price", 0.0))
                    pct = float(g.get("percent_change", 0.0))
                    if MIN_PRICE <= price <= MAX_PRICE and pct >= MIN_GAP_PCT:
                        if len(sym) <= 5 and not sym.endswith("W") and sym not in candidates:
                            candidates.append(sym)
                            logger.info(f"   🎯 Scanner Match: {sym} | Price: ${price:.2f} | Change: +{pct:.1f}%")
        except Exception as e:
            logger.error(f"❌ Error during scanner: {e}")

        final_targets = candidates[:limit]
        logger.info(f"📋 [Watchlist Updated] Target Universe ({len(final_targets)}): {final_targets}")
        self.watchlist = final_targets
        return final_targets

    def fetch_1min_bars(self, symbol: str, limit: int = 60) -> Optional[pd.DataFrame]:
        """Fetches recent 30-Min bars with VWAP and 9 EMA calculation"""
        try:
            url = f"{ALPACA_DATA_URL}/stocks/bars"
            # 14일 전 시점부터 조회하여 주말/연휴/장외시간에도 캔들 충분히 확보 (1H/30M 누락 방지)
            start_iso = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
            params = {
                "symbols": symbol,
                "timeframe": TIMEFRAME,
                "start": start_iso,
                "limit": 100,
                "sort": "desc",
                "feed": "iex"
            }
            resp = session.get(url, headers=HEADERS, params=params, timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                raw_bars = data.get("bars", {}).get(symbol, [])
                if not raw_bars or len(raw_bars) < 15:
                    return None

                raw_bars.reverse()
                df = pd.DataFrame(raw_bars)
                df.rename(columns={
                    "t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap"
                }, inplace=True)
                
                # If vwap not provided by API, calculate standard cumulative VWAP
                if "vwap" not in df.columns or df["vwap"].isna().all():
                    typical_price = (df["high"] + df["low"] + df["close"]) / 3
                    df["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, 1)

                # Exponential Moving Average 9
                df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()

                return df
        except Exception as e:
            logger.error(f"❌ Error fetching bars for {symbol}: {e}")
        return None

    def detect_first_pullback_setup(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluates Ross Cameron First Pullback Setup for 30-Minute Timeframe:
        - 1. Impulse Wave: Lookback 3~8 bars for a strong rally (>= 2% impulse surge).
        - 2. Pullback Zone: Consolidation or retrace (red bar, doji, or small-bodied candle with body <= 45% of range).
        - 3. Retracement: <= 60% of the impulse surge (tolerates normal small-cap volatility).
        - 4. Volume Contraction: Pullback volume <= impulse avg volume * 1.35.
        - 5. Trend Support: Pullback low above VWAP (0.97 buffer) and close near/above 9 EMA (0.98 buffer).
        - 6. Trigger: Current bar price crosses above the High of the pullback zone.
        - 7. Risk Control: 1% <= SL distance <= MAX_SL_PCT (7%).
        """
        if len(df) < 15:
            return False, {}

        curr_bar = df.iloc[-1]
        prev_bar = df.iloc[-2]

        # 1. Pullback candle check: 음봉이거나 몸통이 작은 캔들(도지/팽이형)이어야 함
        prev_range = prev_bar["high"] - prev_bar["low"]
        is_bearish = prev_bar["close"] < prev_bar["open"]
        is_small_body = prev_range > 0 and (abs(prev_bar["close"] - prev_bar["open"]) / prev_range) <= 0.45
        is_pullback_candle = is_bearish or is_small_body

        if not is_pullback_candle:
            return False, {}

        # 2. Check support above 9 EMA & VWAP (노이즈 방지용 2~3% 유연 버퍼)
        if prev_bar["close"] < prev_bar["ema9"] * 0.98 or prev_bar["low"] < prev_bar["vwap"] * 0.97:
            return False, {}

        # 3. Scan for impulse leg prior to pullback (2~6 bars before)
        prior_bars = df.iloc[-7:-2]
        impulse_low = prior_bars["low"].min()
        impulse_high = prior_bars["high"].max()
        impulse_surge = impulse_high - impulse_low

        if impulse_surge <= 0 or (impulse_surge / impulse_low) < 0.02:
            return False, {}  # 2% 미만은 충격파 부재

        # 4. Retracement check: 최대 60% 되돌림까지 수용
        pullback_retrace = impulse_high - prev_bar["low"]
        retrace_ratio = pullback_retrace / impulse_surge
        if retrace_ratio > 0.60:
            return False, {}

        # 5. Volume contraction: 눌림목 거래량이 충격파 평균 거래량 대비 과도하게 폭증하지 않음
        impulse_avg_vol = prior_bars["volume"].mean()
        if impulse_avg_vol > 0 and prev_bar["volume"] > impulse_avg_vol * 1.35:
            return False, {}

        # 6. Breakout Trigger & Stop Loss
        trigger_price = prev_bar["high"]
        stop_loss_price = prev_bar["low"]

        risk_per_share = trigger_price - stop_loss_price
        if risk_per_share <= 0:
            return False, {}

        risk_pct = risk_per_share / trigger_price
        if risk_pct > MAX_SL_PCT or risk_pct < MIN_SL_PCT:
            return False, {}

        target_price = trigger_price + (risk_per_share * RISK_REWARD_RATIO)

        # Trigger condition: current high >= trigger_price
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

    def save_state(self):
        try:
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.own_entries, f)
            os.replace(tmp, STATE_FILE)
        except Exception as e:
            logger.error(f"❌ state save failed: {e}")

    def execute_bracket_order(self, symbol: str, setup: Dict[str, Any]) -> bool:
        """
        Executes an Alpaca Native Bracket Order:
        - Market/Marketable Limit Buy
        - Auto Take-Profit Limit
        - Auto Stop-Loss Market
        """
        trigger_price = setup["trigger_price"]
        stop_loss = setup["stop_loss"]
        take_profit = setup["target_price"]
        curr_price = setup.get("curr_price", trigger_price)
        base_ref = max(trigger_price, curr_price)

        # Alpaca Bracket Order Validation Guard:
        # take_profit.limit_price must be >= base_price + 0.01.
        # 체결 시점 시장가 급등에 의한 422 주문 거절 방지를 위해 최소 안전 마진 강제
        min_tp = round(base_ref + 0.05, 2)
        if take_profit <= min_tp:
            take_profit = round(base_ref + max(setup.get("risk_per_share", 0.05) * RISK_REWARD_RATIO, 0.05), 2)

        # Calculate position size via Dynamic Half-Kelly Sizing (16% ~ 20%)
        qty = self.calculate_dynamic_position_size(setup, trigger_price)

        if qty <= 0:
            logger.warning(f"⚠️ [Order Skipped] Calculated quantity for {symbol} is 0.")
            return False

        order_payload = {
            "symbol": symbol,
            "qty": qty,
            "side": "buy",
            "type": "market",
            "time_in_force": "gtc",
            "order_class": "bracket",
            "client_order_id": f"{ORDER_TAG}-{symbol}-{int(time.time())}",
            "take_profit": {
                "limit_price": str(take_profit)
            },
            "stop_loss": {
                "stop_price": str(stop_loss)
            }
        }

        # Jev Pre-Trade Verification (Microstructure / Offering Gate)
        loop = asyncio.new_event_loop()
        try:
            decision = loop.run_until_complete(
                self.jev_gate.verify_entry(
                    asset_class="US_STOCKS",
                    symbol=symbol,
                    setup_name="CAMERON_30M_PULLBACK",
                    context={
                        "surge_pct": setup.get("surge_pct", 0),
                        "vol_ratio": setup.get("vol_ratio", 1.0),
                        "retrace_pct": setup.get("retrace_ratio", 40),
                        "price": trigger_price
                    }
                )
            )
            if not decision.approved:
                logger.warning(f"🛑 [JEV VETO] {symbol} Order blocked by AI Gate! Reason: {decision.reason} (Conf: {decision.confidence:.2f})")
                return False
            logger.info(f"🛡️ [JEV APPROVED] {symbol} Entry validated by AI Gate! Reason: {decision.reason} (Conf: {decision.confidence:.2f})")
        except Exception as e:
            logger.warning(f"⚠️ Jev verification error: {e}. Proceeding with caution.")
        finally:
            loop.close()

        logger.info(f"⚡ [BRACKET ORDER TRIGGERED] {symbol}")
        logger.info(f"   Quantity: {qty} shares (~${qty * trigger_price:,.2f})")
        logger.info(f"   Trigger: ${trigger_price:.2f} | Stop: ${stop_loss:.2f} (-{setup['risk_pct']}%) | Target: ${take_profit:.2f} (+{setup['risk_pct']*2:.1f}%)")

        if self.dry_run:
            logger.info("   [Dry Run Mode] Bracket order logged without transmission.")
            self.own_entries[symbol] = time.time()
            self.save_state()
            return True

        try:
            resp = session.post(f"{ALPACA_REST_URL}/orders", headers=HEADERS, json=order_payload, timeout=8)
            if resp.status_code in [200, 201]:
                order_res = resp.json()
                logger.info(f"   ✅ [Order Accepted] Order ID: {order_res.get('id')} | Status: {order_res.get('status')}")
                self.own_entries[symbol] = time.time()
                self.save_state()
                return True
            else:
                logger.error(f"   ❌ [Order Failed] Status: {resp.status_code} | Msg: {resp.text}")
        except Exception as e:
            logger.error(f"   ❌ Exception sending bracket order for {symbol}: {e}")

        return False

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Fetches active positions from Alpaca"""
        try:
            resp = session.get(f"{ALPACA_REST_URL}/positions", headers=HEADERS, timeout=6)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"❌ Error fetching open positions: {e}")
        return []

    def close_position(self, symbol: str) -> bool:
        """Closes position at market and cancels associated open orders"""
        try:
            logger.info(f"🚨 [EOD Close] Closing position for {symbol}...")
            # cancel_orders=true: 기존 브래킷(TP/SL) 주문을 먼저 취소하여 수량 묶임(403) 방지
            resp = session.delete(f"{ALPACA_REST_URL}/positions/{symbol}?cancel_orders=true", headers=HEADERS, timeout=8)
            if resp.status_code in [200, 204]:
                logger.info(f"   ✅ [Position Liquidated] {symbol} EOD Market Close Success")
                return True
            else:
                logger.error(f"   ❌ [Position Close Failed] {symbol} Status: {resp.status_code} | Msg: {resp.text}")
        except Exception as e:
            logger.error(f"   ❌ Exception closing position for {symbol}: {e}")
        return False

    def run_iteration(self):
        """Single evaluation cycle"""
        acc = self.get_account_summary()
        open_positions = self.get_open_positions()
        pos_symbols = {p["symbol"] for p in open_positions}
        open_orders = self.get_open_orders()
        pending_symbols = {o["symbol"] for o in open_orders}

        # Check market clock
        clock = self.get_market_clock()
        is_open = clock.get("is_open", False)

        now_est = datetime.now(ZoneInfo("America/New_York"))
        now_t = time.time()
        
        logger.info(f"⏰ [Heartbeat] EST: {now_est.strftime('%H:%M:%S')} | Market Open: {is_open} | Open Positions: {list(pos_symbols)} | Pending Orders: {list(pending_symbols)} | Equity: ${acc['equity']:,.2f}")

        # Scan for movers if watchlist empty or every 10 minutes
        if not self.watchlist or (now_t - self.last_scanned_time > 600):
            self.scan_top_movers(limit=25)
            self.last_scanned_time = now_t

        # 자기 진입 종목만 카운트 (1H 병행 인스턴스/레거시 메가캡 무시)
        # 중요: 주문 후 Alpaca 체결 지연/대기 중인 상태(pending_symbols)이거나
        # 발주 후 ENTRY_GRACE_SEC(90초) 이내인 경우 stale로 삭제하지 않음!
        stale = []
        for s, entry_t in list(self.own_entries.items()):
            if s not in pos_symbols and s not in pending_symbols:
                if now_t - entry_t > ENTRY_GRACE_SEC:
                    stale.append(s)

        for s in stale:
            logger.info(f"📤 [Position Closed] {s} 브래킷 청산 확인 — 추적 해제 및 {COOLDOWN_SEC//60}분 쿨다운 등록")
            del self.own_entries[s]
            self.cooldowns[s] = now_t
        if stale:
            self.save_state()

        own_positions = [s for s in self.own_entries if s in pos_symbols]

        # EOD Market Close (Option 7): 15:50 ~ 15:59 EST 당일 보유분 일괄 종가 청산
        if now_est.hour == 15 and now_est.minute >= 50:
            if own_positions:
                logger.info(f"🔔 [EOD Window] 15:50 EST 도달 - 당사 보유 종목 {own_positions} 일괄 종가 청산 진행")
                for s in list(own_positions):
                    if self.close_position(s):
                        self.own_entries.pop(s, None)
                        self.cooldowns[s] = now_t
                self.save_state()
            return

        # Check market open (정규장 미개장 시 신규 진입 차단 — 프리마켓/장후 주문 큐잉 및 시초가 갭 슬리피지 방지)
        if not is_open:
            return

        # 동시 진입 종목 수 체크: 포지션 보유 종목 + 현재 접수/체결 대기 중인 종목 합산
        total_active_entries = set(own_positions).union({s for s in self.own_entries if s in pending_symbols})
        if len(total_active_entries) >= MAX_CONCURRENT_TRADES:
            logger.info(f"⏸️ Max concurrent positions ({len(total_active_entries)}/{MAX_CONCURRENT_TRADES}) reached. Monitoring exits...")
            return

        # Check time cutoff (14:30 EST)
        if now_est.hour > CUTOFF_HOUR_EST or (now_est.hour == CUTOFF_HOUR_EST and now_est.minute >= CUTOFF_MIN_EST):
            logger.info(f"⏳ Past cutoff time ({CUTOFF_HOUR_EST}:{CUTOFF_MIN_EST:02d} EST). No new small-cap entries permitted.")
            return

        # Scan watchlist for First Pullback Setups
        for symbol in self.watchlist:
            # 1. 이미 보유 중
            # 2. 접수 대기 중
            # 3. 내부 추적 중
            # 4. 최근 청산 후 쿨다운 기간(15분) 내
            if (symbol in self.own_entries or 
                symbol in pos_symbols or 
                symbol in pending_symbols or 
                (now_t - self.cooldowns.get(symbol, 0) < COOLDOWN_SEC)):
                continue

            df = self.fetch_1min_bars(symbol, limit=40)
            if df is None:
                continue

            triggered, setup = self.detect_first_pullback_setup(df)
            if triggered:
                logger.info(f"🔥 [SETUP CONFIRMED] {symbol} | Retrace: {setup['retrace_ratio']}% | Risk: {setup['risk_pct']}%")
                success = self.execute_bracket_order(symbol, setup)
                if success:
                    break  # Enter one at a time

    def start_loop(self, poll_interval_sec: int = 15):
        """Continuous execution loop"""
        logger.info("🟢 Starting Alpaca Cameron Trader Execution Loop...")
        try:
            while True:
                self.run_iteration()
                time.sleep(poll_interval_sec)
        except KeyboardInterrupt:
            logger.info("🛑 Trader stopped manually by user.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Alpaca Ross Cameron Pullback Trader")
    parser.add_argument("--dry-run", action="store_true", help="Run in simulation mode without placing live orders")
    parser.add_argument("--once", action="store_true", help="Run a single scan iteration and exit")
    args = parser.parse_args()

    trader = AlpacaCameronPullbackTrader(dry_run=args.dry_run)
    if args.once:
        trader.run_iteration()
    else:
        trader.start_loop(poll_interval_sec=15)
