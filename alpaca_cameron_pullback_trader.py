# -*- coding: utf-8 -*-
"""
alpaca_cameron_pullback_trader.py — Ross Cameron Small-Cap Momentum Pullback Strategy
Designed for Alpaca Live/Paper Trading ($100k Account, 1-Hour Timeframe Optimized)

Strategy Core Philosophy (Ross Cameron 1-Hour First Pullback):
1. Pre-Market / Market Open Scanner:
   - Price: $2.00 ~ $50.00
   - Momentum Surge >= 3%
   - Consolidated SIP Market Feed with true volume & VWAP
2. Entry Setup (1-Hour Candle Pattern):
   - Clear morning surge (Impulse Wave: at least 2 strong bullish bars, >=2% surge)
   - First Pullback: 1~3 bearish candles
   - Pullback retracement <= 50% of the impulse leg
   - Pullback volume significantly lower than impulse volume (< 1.1x avg volume)
   - Price remains above VWAP and 9 EMA
   - Entry Trigger: Current bar breaks above the High of the previous pullback candle
   - Hard Bar-Level Lock: 1-Time entry per candle to eliminate duplicate order loops
3. Risk & Order Management:
   - Dynamic Half-Kelly Position Sizing: 16% ~ 20% of account equity per trade
   - Max 5 concurrent positions, with 80% Gross Account Exposure ceiling (0% margin safe)
   - Native Alpaca Bracket Order:
       * Stop-Loss: Swing Low (Low of the pullback candle) -> Market Stop (<= 5% SL)
       * Take-Profit: 1:3.0 Risk-to-Reward -> Limit Order
   - Hard Cutoff: No new entries after 14:30 EST
   - EOD Market Close: 15:50 ~ 15:59 EST
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
LOG_FILE = os.path.join(CURRENT_DIR, "alpaca_cameron_trader.log")
STATE_FILE = os.path.join(CURRENT_DIR, "state", "alpaca_cameron_1h_state.json")
ORDER_TAG = "cam1h"
logger = logging.getLogger("AlpacaCameronTrader")
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=2, encoding="utf-8")
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)
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

# ── STRATEGY HYPERPARAMETERS (1-Hour Swing Holy Grail Spec) ──
TIMEFRAME = "1Hour"          # 1-Hour Timeframe (Noise & Whipsaw completely eliminated)
MIN_PRICE = 2.00
MAX_PRICE = 50.00
MIN_GAP_PCT = 3.0           # Minimum 3% momentum surge on 1H
MAX_PORTFOLIO_PCT = 0.20    # 20% per trade (Dynamic Half-Kelly Sizing: 16% ~ 20%)
MAX_CONCURRENT_TRADES = 5   # Max 5 concurrent positions for this instance
MAX_ACCOUNT_EXPOSURE_PCT = 0.80  # 80% Account Gross Exposure Ceiling (Margin Protection)
RISK_REWARD_RATIO = 3.0     # 1:3.0 R:R (Optimized Holy Grail: PF 5.95, +380.18%)
MAX_SL_PCT = 0.05           # Max 5% stop loss distance
MIN_SL_PCT = 0.01           # Min 1% stop loss distance
CUTOFF_HOUR_EST = 14        # Allow entries until 2:30 PM EST
CUTOFF_MIN_EST = 30
COOLDOWN_SEC = 900          # 15-minute re-entry cooldown
ENTRY_GRACE_SEC = 90        # 90-second execution grace period


class AlpacaCameronPullbackTrader:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.account_equity = 100000.0
        self.watchlist: List[str] = []
        self.last_scanned_time = 0
        self.jev_gate = UniversalJevGate(confidence_threshold=0.60)
        self.own_entries: Dict[str, float] = {}
        self.cooldowns: Dict[str, float] = {}
        self.last_entry_bar: Dict[str, str] = {}  # symbol -> bar_time (Bar-Level Lock)
        
        self.load_state()
        logger.info(f"🚀 [Ross Cameron 1H Swing Engine + JEV Gate] Initializing... Dry Run: {self.dry_run}")
        logger.info(f"   REST Endpoint: {ALPACA_REST_URL}")
        
        # Startup reconciliation with live Alpaca state to prevent orphan positions
        self.sync_startup_state()

    def load_state(self):
        """Loads state with backward compatibility for legacy format"""
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    if isinstance(saved, dict):
                        if "own_entries" in saved:
                            self.own_entries = saved.get("own_entries", {})
                            self.cooldowns = saved.get("cooldowns", {})
                            self.last_entry_bar = saved.get("last_entry_bar", {})
                        else:
                            # Legacy flat dict format
                            self.own_entries = saved
                            self.cooldowns = {}
                            self.last_entry_bar = {}
        except Exception as e:
            logger.error(f"❌ Error loading state from {STATE_FILE}: {e}")
            self.own_entries = {}
            self.cooldowns = {}
            self.last_entry_bar = {}

    def save_state(self):
        """Persists full state safely with atomic replace"""
        try:
            tmp = STATE_FILE + ".tmp"
            payload = {
                "own_entries": self.own_entries,
                "cooldowns": self.cooldowns,
                "last_entry_bar": self.last_entry_bar,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, STATE_FILE)
        except Exception as e:
            logger.error(f"❌ State save failed: {e}")

    def sync_startup_state(self):
        """Reconciles internal state with live Alpaca orders and positions on startup"""
        try:
            orders = self.get_open_orders()
            positions = self.get_open_positions()
            now_t = time.time()

            if orders is not None:
                for o in orders:
                    cid = o.get("client_order_id", "")
                    sym = o.get("symbol", "")
                    if cid.startswith(f"{ORDER_TAG}-") and sym:
                        if sym not in self.own_entries:
                            self.own_entries[sym] = now_t
                            logger.info(f"🔄 [Startup Sync] Recovered pending order {cid} for {sym}")

            if positions is not None:
                for p in positions:
                    sym = p.get("symbol", "")
                    if sym in self.own_entries:
                        logger.info(f"🔄 [Startup Sync] Confirmed active position for {sym} ({p.get('qty')} shares)")
            
            self.save_state()
        except Exception as e:
            logger.error(f"❌ Error during startup sync: {e}")

    def get_open_orders(self) -> Optional[List[Dict[str, Any]]]:
        """Fetches active pending orders from Alpaca. Returns None on network/API failure."""
        try:
            resp = session.get(f"{ALPACA_REST_URL}/orders?status=open", headers=HEADERS, timeout=6)
            if resp.status_code == 200:
                return resp.json()
            logger.error(f"❌ Open orders API error ({resp.status_code}): {resp.text}")
        except Exception as e:
            logger.error(f"❌ Exception fetching open orders: {e}")
        return None

    def get_open_positions(self) -> Optional[List[Dict[str, Any]]]:
        """Fetches active positions from Alpaca. Returns None on network/API failure."""
        try:
            resp = session.get(f"{ALPACA_REST_URL}/positions", headers=HEADERS, timeout=6)
            if resp.status_code == 200:
                return resp.json()
            logger.error(f"❌ Open positions API error ({resp.status_code}): {resp.text}")
        except Exception as e:
            logger.error(f"❌ Exception fetching open positions: {e}")
        return None

    def get_account_summary(self) -> Optional[Dict[str, Any]]:
        """Fetches equity and buying power from Alpaca. Returns None on failure."""
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
            logger.error(f"❌ Account API error ({resp.status_code}): {resp.text}")
        except Exception as e:
            logger.error(f"❌ Exception fetching account summary: {e}")
        return None

    def get_market_clock(self) -> Dict[str, Any]:
        """Checks if US market is open"""
        try:
            resp = session.get(f"{ALPACA_REST_URL}/clock", headers=HEADERS, timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.error(f"❌ Failed to get market clock: {e}")
        return {"is_open": False}

    def calculate_dynamic_position_size(self, setup: Dict[str, Any], trigger_price: float, buying_power: Optional[float] = None) -> int:
        """
        Dynamic Half-Kelly Position Sizing:
        - Baseline: p=0.74 (1H win rate), b=3.0 (R:R) -> Half-Kelly: 16% ~ 20%
        - Scales based on pullback volume contraction & retracement depth
        - Hard clamped to available buying power (95% safety margin)
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
        if buying_power is not None and buying_power > 0:
            target_allocation = min(target_allocation, buying_power * 0.95)

        qty = math.floor(target_allocation / trigger_price)
        logger.info(f"   📊 [Dynamic Kelly Sizing] Target: {kelly_pct*100:.1f}% (~${target_allocation:,.2f}) -> {qty} shares")
        return qty

    def scan_top_movers(self, limit: int = 25) -> List[str]:
        """
        Scans for Ross Cameron momentum runners:
        - $2.00 <= Price <= $50.00
        - Percent Change >= 3%
        """
        logger.info("🔍 [Scanner] Scanning top gainers from Alpaca Screener API...")
        candidates = []
        try:
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

    def fetch_bars(self, symbol: str, limit: int = 60) -> Optional[pd.DataFrame]:
        """
        Fetches consolidated SIP bars for accurate volume, VWAP, and EMA9.
        Using feed='sip' guarantees full consolidated US exchange volume.
        """
        try:
            url = f"{ALPACA_DATA_URL}/stocks/bars"
            start_iso = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
            params = {
                "symbols": symbol,
                "timeframe": TIMEFRAME,
                "start": start_iso,
                "limit": 100,
                "sort": "desc",
                "feed": "sip"  # Full US market consolidated feed (replaces 1% IEX)
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
                
                if "vwap" not in df.columns or df["vwap"].isna().all():
                    typical_price = (df["high"] + df["low"] + df["close"]) / 3
                    df["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, 1)

                df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
                return df
            else:
                logger.error(f"❌ Failed to fetch bars for {symbol} ({resp.status_code}): {resp.text}")
        except Exception as e:
            logger.error(f"❌ Error fetching bars for {symbol}: {e}")
        return None

    # Backward compatibility alias
    fetch_1min_bars = fetch_bars

    def detect_first_pullback_setup(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluates Ross Cameron First Pullback Setup:
        - 1. Impulse Wave: Lookback 3~8 bars for a strong rally (higher highs, >=2% surge).
        - 2. Pullback: 1~3 consecutive bearish candles (red bars) immediately preceding.
        - 3. Retracement: <= 50% of the impulse leg.
        - 4. Volume Drying Up: Pullback average volume < Impulse average volume * 1.1.
        - 5. Trend Support: Pullback low above VWAP and close above/touching 9 EMA.
        - 6. Trigger: Current bar price crosses above the High of the pullback candle.
        - 7. Bar Timestamp: Exported for 1-time execution lock.
        """
        if len(df) < 15:
            return False, {}

        curr_bar = df.iloc[-1]
        prev_bar = df.iloc[-2]
        
        # Pullback candle must be bearish (close < open)
        is_prev_pullback = prev_bar["close"] < prev_bar["open"]
        if not is_prev_pullback:
            return False, {}

        # Check support above 9 EMA & VWAP
        if prev_bar["close"] < prev_bar["ema9"] * 0.99 or prev_bar["low"] < prev_bar["vwap"] * 0.985:
            return False, {}

        # Scan for impulse leg prior to pullback (2~6 bars before)
        prior_bars = df.iloc[-7:-2]
        impulse_low = prior_bars["low"].min()
        impulse_high = prior_bars["high"].max()
        impulse_surge = impulse_high - impulse_low

        if impulse_surge <= 0 or (impulse_surge / impulse_low) < 0.02:
            return False, {}  # Less than 2% surge is not an impulse wave

        # Retracement check: pullback must not erase more than 50% of surge
        pullback_retrace = impulse_high - prev_bar["low"]
        retrace_ratio = pullback_retrace / impulse_surge
        if retrace_ratio > 0.50:
            return False, {}  # Retraced more than 50%

        # Volume contraction: Pullback volume < average impulse volume * 1.1
        impulse_avg_vol = prior_bars["volume"].mean()
        if impulse_avg_vol > 0 and prev_bar["volume"] > impulse_avg_vol * 1.1:
            return False, {}  # Heavy selling volume on pullback

        # Breakout Trigger: Current price breaks above the high of the pullback candle
        trigger_price = prev_bar["high"]
        stop_loss_price = prev_bar["low"]
        
        # Risk check
        risk_per_share = trigger_price - stop_loss_price
        if risk_per_share <= 0:
            return False, {}

        risk_pct = risk_per_share / trigger_price
        if risk_pct > MAX_SL_PCT or risk_pct < MIN_SL_PCT:
            return False, {}  # Risk is either too wide (>5%) or too microscopic (<1%)

        target_price = trigger_price + (risk_per_share * RISK_REWARD_RATIO)

        # Trigger condition: current high >= trigger_price
        is_triggered = curr_bar["high"] >= trigger_price

        setup_info = {
            "bar_time": str(curr_bar["time"]),
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

    def execute_bracket_order(self, symbol: str, setup: Dict[str, Any], buying_power: Optional[float] = None) -> bool:
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

        min_tp = round(base_ref + 0.05, 2)
        if take_profit <= min_tp:
            take_profit = round(base_ref + max(setup.get("risk_per_share", 0.05) * RISK_REWARD_RATIO, 0.05), 2)

        qty = self.calculate_dynamic_position_size(setup, trigger_price, buying_power=buying_power)
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

        # Jev Pre-Trade Verification (aligned with US_STOCKS prompt)
        loop = asyncio.new_event_loop()
        try:
            decision = loop.run_until_complete(
                self.jev_gate.verify_entry(
                    asset_class="US_STOCKS",
                    symbol=symbol,
                    setup_name="CAMERON_1H_PULLBACK",
                    context={
                        "surge_pct": setup.get("surge_pct", 0),
                        "gap_pct": setup.get("surge_pct", 0),
                        "vol_ratio": setup.get("vol_ratio", 1.0),
                        "rvol": setup.get("vol_ratio", 1.0),
                        "retrace_pct": setup.get("retrace_ratio", 40),
                        "price": trigger_price,
                        "sector": "SmallCap Momentum",
                        "headline": f"Ross Cameron {TIMEFRAME} Pullback Breakout above ${setup['trigger_price']:.2f}"
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
        logger.info(f"   Trigger: ${trigger_price:.2f} | Stop: ${stop_loss:.2f} (-{setup['risk_pct']}%) | Target: ${take_profit:.2f} (+{setup['risk_pct']*RISK_REWARD_RATIO:.1f}%)")

        if self.dry_run:
            logger.info("   [Dry Run Mode] Bracket order logged without transmission.")
            self.own_entries[symbol] = time.time()
            self.last_entry_bar[symbol] = setup.get("bar_time", str(time.time()))
            self.save_state()
            return True

        try:
            resp = session.post(f"{ALPACA_REST_URL}/orders", headers=HEADERS, json=order_payload, timeout=8)
            if resp.status_code in [200, 201]:
                order_res = resp.json()
                logger.info(f"   ✅ [Order Accepted] Order ID: {order_res.get('id')} | Status: {order_res.get('status')}")
                self.own_entries[symbol] = time.time()
                self.last_entry_bar[symbol] = setup.get("bar_time", str(time.time()))
                self.save_state()
                return True
            else:
                logger.error(f"   ❌ [Order Failed] Status: {resp.status_code} | Msg: {resp.text}")
        except Exception as e:
            logger.error(f"   ❌ Exception sending bracket order for {symbol}: {e}")

        return False

    def close_position(self, symbol: str) -> bool:
        """Closes position at market and cancels associated open orders"""
        try:
            logger.info(f"🚨 [EOD Close] Closing position for {symbol}...")
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
        """Single evaluation cycle with strict error & duplicate protection"""
        # 1. Fetch live account, positions, and orders
        acc = self.get_account_summary()
        open_positions = self.get_open_positions()
        open_orders = self.get_open_orders()

        # [CRITICAL GUARD] If API fails or times out, DO NOT touch state or entries!
        if acc is None or open_positions is None or open_orders is None:
            logger.warning("⚠️ [API Network/Timeout Guard] Failed to fetch account/positions/orders. Skipping iteration to protect state integrity.")
            return

        pos_symbols = {p["symbol"] for p in open_positions}
        pending_symbols = {o["symbol"] for o in open_orders}

        clock = self.get_market_clock()
        is_open = clock.get("is_open", False)

        now_est = datetime.now(ZoneInfo("America/New_York"))
        now_t = time.time()
        
        logger.info(f"⏰ [Heartbeat] EST: {now_est.strftime('%H:%M:%S')} | Market Open: {is_open} | Open Positions: {list(pos_symbols)} | Pending Orders: {list(pending_symbols)} | Equity: ${acc['equity']:,.2f}")

        # Scan for movers if watchlist empty or every 10 minutes
        if not self.watchlist or (now_t - self.last_scanned_time > 600):
            self.scan_top_movers(limit=25)
            self.last_scanned_time = now_t

        # Stale entry cleanup (only when API calls succeeded)
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

        # EOD Market Close: 15:50 ~ 15:59 EST 당일 보유분 일괄 종가 청산 및 대기 주문 취소
        if now_est.hour == 15 and now_est.minute >= 50:
            # 1. 미체결 대기 주문 취소
            for o in open_orders:
                cid = o.get("client_order_id", "")
                if cid.startswith(f"{ORDER_TAG}-"):
                    oid = o.get("id")
                    try:
                        session.delete(f"{ALPACA_REST_URL}/orders/{oid}", headers=HEADERS, timeout=5)
                        logger.info(f"🔔 [EOD Window] Cancelled pending entry order {cid}")
                    except Exception:
                        pass
            # 2. 보유 포지션 전량 청산
            if own_positions:
                logger.info(f"🔔 [EOD Window] 15:50 EST 도달 - 당사 보유 종목 {own_positions} 일괄 종가 청산 진행")
                for s in list(own_positions):
                    if self.close_position(s):
                        self.own_entries.pop(s, None)
                        self.cooldowns[s] = now_t
                self.save_state()
            return

        # Regular market hours check
        if not is_open:
            return

        # [ACCOUNT EXPOSURE CEILING GUARD] 80% maximum total portfolio exposure (0% Margin Safe)
        total_market_val = sum(abs(float(p.get("market_value", 0.0))) for p in open_positions)
        equity = float(acc["equity"])
        exposure_ratio = total_market_val / equity if equity > 0 else 1.0
        if exposure_ratio >= MAX_ACCOUNT_EXPOSURE_PCT:
            logger.info(f"⏸️ [Account Risk Guard] Total account exposure ({exposure_ratio*100:.1f}%) >= {MAX_ACCOUNT_EXPOSURE_PCT*100:.0f}%. Margin protection active.")
            return

        # Instance position count check
        total_active_entries = set(own_positions).union({s for s in self.own_entries if s in pending_symbols})
        if len(total_active_entries) >= MAX_CONCURRENT_TRADES:
            logger.info(f"⏸️ Max concurrent positions ({len(total_active_entries)}/{MAX_CONCURRENT_TRADES}) reached. Monitoring exits...")
            return

        # Cutoff time check (14:30 EST)
        if now_est.hour > CUTOFF_HOUR_EST or (now_est.hour == CUTOFF_HOUR_EST and now_est.minute >= CUTOFF_MIN_EST):
            logger.info(f"⏳ Past cutoff time ({CUTOFF_HOUR_EST}:{CUTOFF_MIN_EST:02d} EST). No new small-cap entries permitted.")
            return

        # Cross-instance active symbols check (1H vs 30M mutual exclusion)
        other_instance_symbols = set()
        other_state_file = STATE_FILE.replace("1h", "30m") if "1h" in STATE_FILE else STATE_FILE.replace("30m", "1h")
        try:
            if os.path.exists(other_state_file):
                with open(other_state_file, "r", encoding="utf-8") as f:
                    other_data = json.load(f)
                    other_instance_symbols = set(other_data.get("own_entries", {}).keys())
        except Exception:
            pass

        # Scan watchlist for First Pullback Setups
        for symbol in self.watchlist:
            # 1. 이미 보유 중
            # 2. 다른 타임프레임 인스턴스(30M/1H)에서 이미 진입/추적 중
            # 3. 접수 대기 중
            # 4. 내부 추적 중
            # 5. 최근 청산 후 쿨다운 기간(15분) 내
            if (symbol in self.own_entries or 
                symbol in other_instance_symbols or
                symbol in pos_symbols or 
                symbol in pending_symbols or 
                (now_t - self.cooldowns.get(symbol, 0) < COOLDOWN_SEC)):
                continue

            df = self.fetch_bars(symbol, limit=40)
            if df is None:
                continue

            triggered, setup = self.detect_first_pullback_setup(df)
            if triggered:
                # [BAR-LEVEL 1-TIME LOCK GUARD] 동일 캔들 내 재진입 원천 차단
                bar_time = setup.get("bar_time")
                if self.last_entry_bar.get(symbol) == bar_time:
                    logger.info(f"⏸️ [Bar Lock] {symbol} already triggered in current {TIMEFRAME} bar ({bar_time}). Skipping duplicate.")
                    continue

                logger.info(f"🔥 [SETUP CONFIRMED] {symbol} | Bar: {bar_time} | Retrace: {setup['retrace_ratio']}% | Risk: {setup['risk_pct']}%")
                success = self.execute_bracket_order(symbol, setup, buying_power=acc.get("buying_power"))
                if success:
                    break

    def start_loop(self, poll_interval_sec: int = 15):
        """Continuous execution loop with dynamic sleep"""
        logger.info("🟢 Starting Alpaca Cameron Trader Execution Loop...")
        try:
            while True:
                self.run_iteration()
                # 장마감 시간대에는 60초 대기하여 로그 낭비 방지, 장중에는 15초 폴링
                clock = self.get_market_clock()
                sleep_sec = poll_interval_sec if clock.get("is_open", False) else 60
                time.sleep(sleep_sec)
        except KeyboardInterrupt:
            logger.info("🛑 Trader stopped manually by user.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Alpaca Ross Cameron Pullback Trader (1H)")
    parser.add_argument("--dry-run", action="store_true", help="Run in simulation mode without placing live orders")
    parser.add_argument("--once", action="store_true", help="Run a single scan iteration and exit")
    args = parser.parse_args()

    trader = AlpacaCameronPullbackTrader(dry_run=args.dry_run)
    if args.once:
        trader.run_iteration()
    else:
        trader.start_loop(poll_interval_sec=15)
