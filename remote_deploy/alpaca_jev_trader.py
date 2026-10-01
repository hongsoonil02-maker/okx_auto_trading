# -*- coding: utf-8 -*-
"""
alpaca_jev_trader.py — Autonomous Nasdaq Quant Execution Engine v2.1
Integrated with Jev AI Microstructure Gating & Alpaca Paper Trading

Features:
- Live BBO & Imbalance stream from local Bun Sidecar (port 8020)
- Jev Sub-Second Gating (/predict) for trade confirmation
- Dual-Directional Alpha with MUTUAL EXCLUSION LOCK:
  * Bullish: TQQQ, SOXL, NVDA, TSLA, AAPL, QQQ
  * Bearish: SQQQ, SOXS
  * Mutual Exclusion: Never hold Bullish & Bearish leveraged ETFs simultaneously
  * Regime Shift Unwind: Automatically liquidates opposing ETFs on macro trend reversal
- Strict Concurrency Limit: Enforces MAX_CONCURRENT_POSITIONS (4) including pending buy orders
- Post-Loss Lockout: 30-minute cooling period after stop-loss to eliminate churn
- GTC Bracket Protection: "time_in_force": "gtc" prevents overnight stop-loss expiration
- Automatic Reconciliation: Resolves legacy conflicting positions at market open
- Market Session Awareness (/v2/clock) with 24/7 graceful supervision
"""

import os
import sys
import time
import math
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Set
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler

# Ensure local dir modules can be imported
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

# Import local Jev client
try:
    from jev_nasdaq_client import query_jev_nasdaq, get_quotes_cache
except ImportError:
    sys.path.append(CURRENT_DIR)
    from jev_nasdaq_client import query_jev_nasdaq, get_quotes_cache

# ── LOGGING SETUP ──
LOG_FILE = os.path.join(CURRENT_DIR, "alpaca_jev_trader.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
    ]
)
logger = logging.getLogger("AlpacaJevTrader")

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

# ── CONFIGURATION ──
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "PKVGU7IXDJMAF6JBWWVMP2I6R3")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "CCnTinES4eWwkQqifr9879ASCcXSpm9kjuvbZnCKpHM3")
ALPACA_REST_URL = os.getenv("ALPACA_REST_URL", "https://paper-api.alpaca.markets/v2")
ALPACA_DATA_URL = "https://data.alpaca.markets/v2"

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type": "application/json"
}

# Universe & Pairing - Optimized for Scenario 6 (Mega Tech Only)
TSLA_ENABLED = os.getenv("ALPACA_TSLA_ENABLED", "true").lower() == "true"
SYMBOLS_UNIVERSE = ["NVDA", "AAPL"]
if TSLA_ENABLED:
    SYMBOLS_UNIVERSE.append("TSLA")

BULLISH_ETFS = {"TQQQ", "SOXL", "QQQ"}  # Kept for regime-shift unwind compatibility
BEARISH_ETFS = {"SQQQ", "SOXS"}         # Kept for regime-shift unwind compatibility
LEVERAGED_3X = {"TQQQ", "SQQQ", "SOXL", "SOXS"}

MAX_CONCURRENT_POSITIONS = 3
MAX_POSITION_PCT = 0.30          # Increased to 30% per position (3 symbols * 30% = 90% usage)
LOOP_INTERVAL_SEC = 10           # Poll market every 10 seconds during open hours
POST_LOSS_COOLDOWN_SEC = 1800.0  # 30-minute lockout after stop-loss to eliminate churn
NORMAL_COOLDOWN_SEC = 300.0      # 5-minute normal exit cooldown

def get_symbol_risk(symbol: str) -> Tuple[float, float, float]:
    """Returns (stop_loss_pct, take_profit_pct, trailing_arm_pct) tailored to asset volatility"""
    if symbol in LEVERAGED_3X:
        return 0.024, 0.042, 0.020   # 3x ETF: 2.4% SL, 4.2% TP, 2.0% Trail Arm
    if symbol == "TSLA":
        return 0.022, 0.045, 0.025   # TSLA High Vol: 2.2% SL, 4.5% TP, 2.5% Trail Arm
    return 0.012, 0.028, 0.015       # 1x Equities: 1.2% SL, 2.8% TP, 1.5% Trail Arm

class AlpacaJevTrader:
    def __init__(self, simulation_mode: Optional[bool] = None):
        self.simulation_mode: bool = (
            simulation_mode if simulation_mode is not None
            else (os.getenv("JEV_SIMULATION_MODE", "false").lower() == "true" or "--simulation" in sys.argv)
        )
        self.account_info: Dict[str, Any] = {}
        self.positions: Dict[str, Any] = {}
        self.peak_prices: Dict[str, float] = {}
        self.last_clock_check: float = 0.0
        self.is_market_open: bool = False
        self.next_open_str: str = ""
        self.next_close_str: str = ""
        self.trade_cooldowns: Dict[str, float] = {}  # Symbol -> cooldown timestamp
        self.position_entry_times: Dict[str, float] = {}  # Symbol -> entry timestamp (buffer protection)
        
        logger.info("=" * 70)
        logger.info(f"🚀 [Alpaca Jev Trader v2.1] Initializing Autonomous Nasdaq Quant Engine")
        logger.info(f"   Mode: {'SIMULATION' if self.simulation_mode else 'LIVE PAPER'}")
        logger.info(f"   REST Endpoint: {ALPACA_REST_URL}")
        logger.info(f"   Tracked Universe: {', '.join(SYMBOLS_UNIVERSE)}")
        logger.info(f"   Max Concurrent Positions: {MAX_CONCURRENT_POSITIONS} (Strict Enforced)")
        logger.info(f"   Position Sizing: {MAX_POSITION_PCT*100:.1f}% of Equity")
        logger.info(f"   Risk Config (3x ETF): SL -2.4%, TP +4.2%, Trail Arm +2.0%")
        logger.info(f"   Risk Config (1x Stock): SL -1.2%, TP +2.8%, Trail Arm +1.5%")
        logger.info(f"   Lockout Guard: {POST_LOSS_COOLDOWN_SEC/60:.0f}m cooldown post stop-loss")
        logger.info(f"   Mutual Exclusion: Long ETF vs Inverse ETF conflict strictly prohibited")
        logger.info("=" * 70)

    def sync_account(self) -> bool:
        """Fetches account details & cash / buying power"""
        try:
            resp = session.get(f"{ALPACA_REST_URL}/account", headers=HEADERS, timeout=5)
            if resp.status_code == 200:
                self.account_info = resp.json()
                equity = float(self.account_info.get("equity", 0.0))
                cash = float(self.account_info.get("cash", 0.0))
                bp = float(self.account_info.get("buying_power", 0.0))
                status = self.account_info.get("status", "UNKNOWN")
                logger.info(f"💼 [Account Sync] Status: {status} | Equity: ${equity:,.2f} | Cash: ${cash:,.2f} | Buying Power: ${bp:,.2f}")
                return True
            else:
                logger.error(f"❌ Account sync failed: HTTP {resp.status_code} - {resp.text}")
                return False
        except Exception as e:
            logger.error(f"❌ Account sync exception: {e}")
            return False

    def sync_positions(self) -> Dict[str, Any]:
        """Fetches current open positions from Alpaca"""
        if self.simulation_mode:
            return self.positions
        try:
            resp = session.get(f"{ALPACA_REST_URL}/positions", headers=HEADERS, timeout=5)
            if resp.status_code == 200:
                pos_list = resp.json()
                self.positions = {p["symbol"]: p for p in pos_list}
                return self.positions
            else:
                logger.error(f"❌ Positions sync failed: HTTP {resp.status_code}")
                return {}
        except Exception as e:
            logger.error(f"❌ Positions sync exception: {e}")
            return {}

    def get_pending_orders(self) -> List[Dict[str, Any]]:
        """Fetches currently open/pending orders from Alpaca"""
        if self.simulation_mode:
            return []
        try:
            resp = session.get(f"{ALPACA_REST_URL}/orders?status=open", headers=HEADERS, timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.debug(f"Open orders fetch exception: {e}")
        return []

    def get_active_count(self) -> int:
        """Returns total count of active positions plus pending buy entry orders"""
        open_orders = self.get_pending_orders()
        pending_buys = {o["symbol"] for o in open_orders if o.get("side") == "buy"}
        total_active = set(self.positions.keys()).union(pending_buys)
        return len(total_active)

    def check_market_clock(self) -> bool:
        """Checks if US stock market is open"""
        now = time.time()
        if now - self.last_clock_check < 30.0 and self.last_clock_check > 0:
            return self.is_market_open

        try:
            resp = session.get(f"{ALPACA_REST_URL}/clock", headers=HEADERS, timeout=5)
            if resp.status_code == 200:
                clock = resp.json()
                self.is_market_open = clock.get("is_open", False)
                self.next_open_str = clock.get("next_open", "")
                self.next_close_str = clock.get("next_close", "")
                self.last_clock_check = now
                return self.is_market_open
        except Exception as e:
            logger.warning(f"⚠️ Clock check error: {e}")
        return self.is_market_open

    def fetch_bars(self, symbol: str, timeframe: str = "1Min", limit: int = 40) -> Optional[pd.DataFrame]:
        """Fetches recent bars from Alpaca Market Data API"""
        try:
            url = f"{ALPACA_DATA_URL}/stocks/bars"
            params = {
                "symbols": symbol,
                "timeframe": timeframe,
                "limit": limit,
                "feed": "iex",
                "sort": "desc"
            }
            resp = session.get(url, headers=HEADERS, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                bars = data.get("bars", {}).get(symbol, [])
                if not bars:
                    return None
                df = pd.DataFrame(bars)
                df = df.rename(columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
                df["close"] = pd.to_numeric(df["close"])
                df["high"] = pd.to_numeric(df["high"])
                df["low"] = pd.to_numeric(df["low"])
                df = df.iloc[::-1].reset_index(drop=True)  # Ascending order
                return df
        except Exception as e:
            logger.debug(f"Bars fetch failed for {symbol}: {e}")
        return None

    def calculate_technical_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Computes EMA9, EMA21, EMA50, RSI(14) on candle dataframe"""
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

        # Momentum Crossover Logic
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
                "reason": f"BULLISH_EMA_CROSS (RSI: {curr_rsi:.1f})"
            }
        elif bearish_cross:
            return {
                "signal": "SELL",
                "rsi": curr_rsi,
                "ema9": curr_ema9,
                "ema21": curr_ema21,
                "ema50": curr_ema50,
                "price": curr_close,
                "reason": f"BEARISH_EMA_CROSS (RSI: {curr_rsi:.1f})"
            }

        return {"signal": "NEUTRAL", "rsi": curr_rsi, "price": curr_close, "ema50": curr_ema50, "reason": "NO_CLEAR_CROSS"}

    def submit_alpaca_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        stop_loss_pct: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Submits GTC bracket order to Alpaca Paper Trading REST API"""
        if qty <= 0:
            logger.warning(f"⚠️ Invalid order qty for {symbol}: {qty}")
            return None

        # Fetch latest price to compute bracket SL/TP
        quotes = get_quotes_cache()
        q = quotes.get(symbol, {})
        base_price = limit_price or q.get("askPrice") or q.get("bidPrice") or 0.0

        if base_price <= 0:
            bars = self.fetch_bars(symbol, timeframe="1Min", limit=5)
            if bars is not None and not bars.empty:
                base_price = float(bars["close"].iloc[-1])

        # Enforce GTC time-in-force so bracket orders protect overnight
        order_data: Dict[str, Any] = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side.lower(),
            "type": order_type.lower(),
            "time_in_force": "gtc"
        }

        if order_type.lower() == "limit" and limit_price:
            order_data["limit_price"] = str(round(limit_price, 2))

        # Use adaptive risk per asset class
        sym_sl, sym_tp, sym_trail = get_symbol_risk(symbol)
        sl_pct = stop_loss_pct if stop_loss_pct is not None else sym_sl
        tp_pct = take_profit_pct if take_profit_pct is not None else sym_tp

        # Add bracket stop-loss and take-profit if price is known
        if base_price > 0:
            if side.lower() == "buy":
                tp_price = round(base_price * (1.0 + tp_pct), 2)
                sl_price = round(base_price * (1.0 - sl_pct), 2)
            else:
                tp_price = round(base_price * (1.0 - tp_pct), 2)
                sl_price = round(base_price * (1.0 + sl_pct), 2)

            order_data["order_class"] = "bracket"
            order_data["take_profit"] = {"limit_price": str(tp_price)}
            order_data["stop_loss"] = {"stop_price": str(sl_price)}

        logger.info(f"📤 [Order Submit] {side.upper()} {qty}x {symbol} | Type: {order_type} (GTC) | Bracket TP: ${tp_price if base_price > 0 else 'N/A'} / SL: ${sl_price if base_price > 0 else 'N/A'}")

        if self.simulation_mode:
            logger.info(f"🧪 [SIMULATION FILL] {side.upper()} {qty}x {symbol} @ ${base_price:.2f} | Bracket TP: ${tp_price} / SL: ${sl_price}")
            sim_pos = {
                "symbol": symbol,
                "qty": str(qty),
                "side": side.lower(),
                "avg_entry_price": str(base_price),
                "current_price": str(base_price),
                "unrealized_plpc": 0.0,
                "unrealized_pl": 0.0
            }
            self.positions[symbol] = sim_pos
            self.trade_cooldowns[symbol] = time.time() + 60
            return {"id": f"sim_{int(time.time())}", "status": "simulated", "symbol": symbol}

        try:
            resp = session.post(f"{ALPACA_REST_URL}/orders", headers=HEADERS, json=order_data, timeout=8)
            if resp.status_code in [200, 201]:
                res_json = resp.json()
                logger.info(f"✅ [Order Confirmed] ID: {res_json.get('id')} | Status: {res_json.get('status')} | {symbol}")
                self.trade_cooldowns[symbol] = time.time() + 180  # 3 min cooldown
                self.position_entry_times[symbol] = time.time()
                return res_json
            else:
                logger.error(f"❌ Order submission rejected: HTTP {resp.status_code} - {resp.text}")
                return None
        except Exception as e:
            logger.error(f"❌ Order submission exception: {e}")
            return None

    def manage_open_positions(self):
        """Monitors unrealized PnL and executes dynamic trailing stop exits"""
        if not self.positions:
            return

        quotes = get_quotes_cache()
        now = time.time()

        # Log portfolio holding overview every 60 seconds
        if not hasattr(self, "last_portfolio_log"):
            self.last_portfolio_log = 0.0

        if now - self.last_portfolio_log >= 60.0:
            status_parts = []
            for s, p in self.positions.items():
                pnl = float(p.get("unrealized_plpc", 0.0)) * 100.0
                status_parts.append(f"{s}: {pnl:+.2f}%")
            logger.info(f"📊 [Active Positions {len(self.positions)}/{MAX_CONCURRENT_POSITIONS}] " + " | ".join(status_parts))
            self.last_portfolio_log = now

        for symbol, pos in list(self.positions.items()):
            qty = abs(float(pos.get("qty", 0)))
            side = pos.get("side", "long")
            avg_entry = float(pos.get("avg_entry_price", 0.0))
            unrealized_plpc = float(pos.get("unrealized_plpc", 0.0))  # Profit percentage
            current_price = float(pos.get("current_price", 0.0))

            if current_price <= 0 and avg_entry > 0:
                current_price = avg_entry * (1.0 + unrealized_plpc)

            # Update high watermark for trailing stop
            peak = max(self.peak_prices.get(symbol, current_price), current_price)
            self.peak_prices[symbol] = peak

            sym_sl, sym_tp, sym_arm = get_symbol_risk(symbol)
            q = quotes.get(symbol, {})
            imb = q.get("imbalance", 0.0)

            trail_delta = 0.010 if symbol in LEVERAGED_3X else 0.008
            max_ret = (peak - avg_entry) / avg_entry if avg_entry > 0 else 0.0
            trail_drop = (peak - current_price) / peak if peak > 0 else 0.0

            # 1. Trailing Stop Exit Trigger
            if max_ret >= sym_arm and trail_drop >= trail_delta:
                logger.info(f"🎯 [Trailing Stop Trigger] {symbol} {side.upper()} reached max +{max_ret*100:.2f}%, dropped {trail_drop*100:.2f}% from peak ${peak:.2f}")
                self.close_position(symbol, f"TRAILING_STOP_{unrealized_plpc*100:.1f}PCT")

            # 1b. Breakeven Defense: Lock gains after +1.5% peak if price pulls back near entry
            elif max_ret >= 0.015 and unrealized_plpc <= 0.002:
                logger.info(f"🛡️ [Breakeven Defense] {symbol} {side.upper()} reached max +{max_ret*100:.2f}%, locking breakeven exit (PnL: {unrealized_plpc*100:+.2f}%)")
                self.close_position(symbol, f"BREAKEVEN_{unrealized_plpc*100:.2f}PCT")

            # 2. Hard Take Profit Exit Trigger
            elif unrealized_plpc >= sym_tp:
                logger.info(f"🎯 [Take Profit Trigger] {symbol} {side.upper()} reached +{unrealized_plpc*100:.2f}% (Target: {sym_tp*100:.1f}%)")
                self.close_position(symbol, f"TAKE_PROFIT_{unrealized_plpc*100:.1f}PCT")

            # 3. Hard Stop Loss Exit Trigger
            elif unrealized_plpc <= -sym_sl:
                logger.warning(f"🛑 [Stop Loss Trigger] {symbol} {side.upper()} hit loss -{abs(unrealized_plpc)*100:.2f}% (Max SL: {sym_sl*100:.1f}%)")
                self.close_position(symbol, f"STOP_LOSS_{unrealized_plpc*100:.1f}PCT")

            # 4. Severe Orderbook Pressure Exit Trigger
            # 3x ETF requires deeper drawdown and more severe imbalance to avoid IEX thin-book noise
            min_loss_for_micro = -0.018 if symbol in LEVERAGED_3X else -0.010
            imb_threshold = -0.85 if symbol in LEVERAGED_3X else -0.75
            if (side == "long" and imb <= imb_threshold and unrealized_plpc < min_loss_for_micro):
                logger.warning(f"⚠️ [Microstructure Exit] {symbol} Long facing heavy ask wall (Imbalance: {imb:.2f}, PnL: {unrealized_plpc*100:.2f}%). Exiting to preserve capital.")
                self.close_position(symbol, "HEAVY_ASK_WALL_EXIT")

    def close_position(self, symbol: str, reason: str = "MANUAL"):
        """Closes an open position via Alpaca REST API with Post-Loss Lockout"""
        logger.info(f"🔄 [Closing Position] {symbol} | Reason: {reason}")
        pos = self.positions.get(symbol, {})
        unrealized_pl = float(pos.get("unrealized_pl", 0.0))
        unrealized_plpc = float(pos.get("unrealized_plpc", 0.0))

        self.peak_prices.pop(symbol, None)
        self.position_entry_times.pop(symbol, None)

        # ── POST-LOSS LOCKOUT LOGIC ──
        is_loss = (unrealized_pl < 0) or ("STOP_LOSS" in reason) or ("HEAVY_ASK" in reason)
        if is_loss:
            cooldown_sec = POST_LOSS_COOLDOWN_SEC
            logger.info(f"⏳ [Post-Loss Lockout] {symbol} closed with loss ({unrealized_plpc*100:+.2f}%). Enforcing {POST_LOSS_COOLDOWN_SEC/60:.0f}m cooling period.")
        else:
            cooldown_sec = NORMAL_COOLDOWN_SEC

        self.trade_cooldowns[symbol] = time.time() + cooldown_sec

        if self.simulation_mode:
            logger.info(f"🧪 [SIMULATION EXIT] {symbol} closed successfully | Reason: {reason}")
            self.positions.pop(symbol, None)
            return

        try:
            # 1. Cancel ALL open orders for this symbol to release held quantities
            open_orders = self.get_pending_orders()
            canceled_any = False
            for o in open_orders:
                if o.get("symbol") == symbol:
                    try:
                        oid = o.get("id", "")
                        resp_cancel = session.delete(f"{ALPACA_REST_URL}/orders/{oid}", headers=HEADERS, timeout=3)
                        logger.info(f"🧹 [Cancel Order] {symbol} order {oid[:8]}... → HTTP {resp_cancel.status_code}")
                        canceled_any = True
                    except Exception:
                        pass
            if canceled_any:
                time.sleep(1.5)  # Allow Alpaca backend to fully release held_for_orders

            # 2. Close position with retry on held_for_orders (403) errors
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                resp = session.delete(f"{ALPACA_REST_URL}/positions/{symbol}?cancel_orders=true", headers=HEADERS, timeout=8)
                if resp.status_code in [200, 204]:
                    logger.info(f"✅ [Position Closed] {symbol} closed successfully.")
                    self.positions.pop(symbol, None)
                    break
                elif resp.status_code == 404:
                    logger.info(f"ℹ️ [Position Closed] {symbol} was already closed or position not found (HTTP 404).")
                    self.positions.pop(symbol, None)
                    break
                elif resp.status_code == 403 and "held_for_orders" in resp.text:
                    logger.warning(
                        f"⚠️ [Retry {attempt}/{max_retries}] {symbol} held_for_orders — "
                        f"cancelling ALL orders and retrying in {attempt * 2}s..."
                    )
                    # Nuclear option: cancel ALL open orders for this symbol
                    try:
                        session.delete(f"{ALPACA_REST_URL}/orders", headers=HEADERS, timeout=5,
                                       params={"symbols": symbol})
                    except Exception:
                        pass
                    time.sleep(attempt * 2)  # Progressive backoff: 2s, 4s, 6s
                else:
                    logger.error(f"❌ Failed to close {symbol}: HTTP {resp.status_code} - {resp.text}")
                    break
            else:
                # All retries exhausted — last resort: cancel ALL orders then close
                logger.error(f"🚨 [Last Resort] {symbol} — all {max_retries} retries failed. Cancelling ALL account orders...")
                try:
                    session.delete(f"{ALPACA_REST_URL}/orders", headers=HEADERS, timeout=5)
                    time.sleep(3)
                    resp = session.delete(f"{ALPACA_REST_URL}/positions/{symbol}", headers=HEADERS, timeout=8)
                    if resp.status_code in [200, 204]:
                        logger.info(f"✅ [Position Closed] {symbol} closed after last-resort cancel-all.")
                        self.positions.pop(symbol, None)
                    else:
                        logger.error(f"❌ [Last Resort Failed] {symbol}: HTTP {resp.status_code} - {resp.text}")
                except Exception as e2:
                    logger.error(f"❌ [Last Resort Exception] {symbol}: {e2}")

        except Exception as e:
            logger.error(f"❌ Exception closing {symbol}: {e}")

    def reconcile_conflicting_positions(self):
        """Resolves opposing leveraged positions during active trading hours"""
        if not self.is_market_open:
            return

        has_bull = any(s in self.positions for s in BULLISH_ETFS)
        has_bear = any(s in self.positions for s in BEARISH_ETFS)

        if has_bull and has_bear:
            logger.warning(f"⚠️ [Conflict Reconciliation] Holding both Bullish {[s for s in BULLISH_ETFS if s in self.positions]} and Bearish {[s for s in BEARISH_ETFS if s in self.positions]} ETFs! Reconciling against QQQ macro trend...")
            qqq_bars = self.fetch_bars("QQQ", timeframe="1Min", limit=60)
            qqq_tech = self.calculate_technical_signals(qqq_bars)
            qqq_c = qqq_tech.get("price", 0.0)
            qqq_e50 = qqq_tech.get("ema50", 0.0)

            # Guard against invalid / uninitialized price
            if qqq_c <= 0 or qqq_e50 <= 0:
                logger.warning(f"⚠️ [Reconciliation Guard] QQQ price (${qqq_c:.2f}) or EMA50 (${qqq_e50:.2f}) not ready. Skipping reconciliation.")
                return

            if qqq_c >= qqq_e50:
                # Macro Bullish: Liquidate Bearish ETFs
                for s in list(self.positions.keys()):
                    if s in BEARISH_ETFS:
                        logger.info(f"🛡️ [Reconciliation Unwind] Macro Bullish confirmed (QQQ ${qqq_c:.2f} >= EMA50 ${qqq_e50:.2f}). Unwinding {s}.")
                        self.close_position(s, "CONFLICT_RECONCILIATION_CLOSE_BEAR")
            else:
                # Macro Bearish: Liquidate 3x Bullish ETFs
                for s in list(self.positions.keys()):
                    if s in {"TQQQ", "SOXL"}:
                        logger.info(f"🛡️ [Reconciliation Unwind] Macro Bearish confirmed (QQQ ${qqq_c:.2f} < EMA50 ${qqq_e50:.2f}). Unwinding 3x Long {s}.")
                        self.close_position(s, "CONFLICT_RECONCILIATION_CLOSE_BULL")

    def evaluate_trading_opportunities(self):
        """Scans Universe, checks Signals, consults Jev AI, and executes trades with Mutual Exclusion"""
        active_count = self.get_active_count()
        if active_count >= MAX_CONCURRENT_POSITIONS:
            now = time.time()
            if not hasattr(self, "last_max_pos_log"):
                self.last_max_pos_log = 0.0
            if now - self.last_max_pos_log >= 60.0:
                logger.info(f"💼 [Capacity Full] Active positions at limit ({active_count}/{MAX_CONCURRENT_POSITIONS}): {list(self.positions.keys())}. Monitoring open trades...")
                self.last_max_pos_log = now
            return

        equity = float(self.account_info.get("equity", 100000.0))
        target_dollar_size = equity * MAX_POSITION_PCT

        quotes = get_quotes_cache()

        # Step 1: Scan QQQ benchmark for macro regime (5Min timeframe for multi-hour trend stability)
        qqq_bars = self.fetch_bars("QQQ", timeframe="5Min", limit=60)
        if qqq_bars is None or len(qqq_bars) < 20:
            qqq_bars = self.fetch_bars("QQQ", timeframe="1Min", limit=100)
        qqq_tech = self.calculate_technical_signals(qqq_bars)
        qqq_quote = quotes.get("QQQ", {})
        qqq_imb = qqq_quote.get("imbalance", 0.0)
        qqq_c = qqq_tech.get("price", 0.0)
        qqq_e50 = qqq_tech.get("ema50", 0.0)

        # Guard against zero or uninitialized macro price
        if qqq_c <= 0 or qqq_e50 <= 0:
            logger.debug(f"QQQ benchmark price/EMA50 not ready (${qqq_c} / ${qqq_e50}). Skipping scan cycle.")
            return

        # Macro Trend Gate: QQQ > EMA50 * 1.0015 for Bullish, QQQ < EMA50 * 0.9985 for Bearish (0.15% hysteresis buffer)
        is_bullish = (qqq_c > qqq_e50 * 1.0015) and (qqq_tech["signal"] == "BUY" or qqq_imb >= 0.15)
        is_bearish = (qqq_c < qqq_e50 * 0.9985) and (qqq_tech["signal"] == "SELL" or qqq_imb <= -0.15)

        # ── REGIME SHIFT UNWIND: AUTOMATICALLY CLOSE OPPOSING LEVERAGED POSITIONS ──
        # Protect recently opened positions from instant noise liquidations (min 15m hold unless hard stop hit)
        now_ts = time.time()
        if is_bullish:
            for bear_sym in list(self.positions.keys()):
                if bear_sym in BEARISH_ETFS:
                    pos_entry_t = self.position_entry_times.get(bear_sym, 0.0)
                    if now_ts - pos_entry_t >= 900.0:  # 15m minimum hold buffer
                        logger.warning(f"⚡ [Regime Shift Unwind] QQQ Bullish confirmed (Price ${qqq_c:.2f} > EMA50 ${qqq_e50:.2f})! Unwinding inverse ETF {bear_sym} to prevent decay.")
                        self.close_position(bear_sym, "MACRO_REGIME_BULLISH_UNWIND")
        elif is_bearish:
            for bull_sym in list(self.positions.keys()):
                if bull_sym in {"TQQQ", "SOXL"}:
                    pos_entry_t = self.position_entry_times.get(bull_sym, 0.0)
                    if now_ts - pos_entry_t >= 900.0:  # 15m minimum hold buffer
                        logger.warning(f"⚡ [Regime Shift Unwind] QQQ Bearish confirmed (Price ${qqq_c:.2f} < EMA50 ${qqq_e50:.2f})! Unwinding 3x Long ETF {bull_sym} to prevent decay.")
                        self.close_position(bull_sym, "MACRO_REGIME_BEARISH_UNWIND")

        # Prioritize candidates according to macro regime (Mega Tech Only)
        candidate_symbols = []
        if is_bullish:
            candidate_symbols = [("NVDA", "buy"), ("AAPL", "buy")]
            if TSLA_ENABLED:
                candidate_symbols.append(("TSLA", "buy"))
        elif is_bearish:
            # Skip buying in bearish macro for Mega Tech Long-Only strategy
            candidate_symbols = []
        else:
            # Neutral macro: selectively scan top resilient tech equities
            candidate_symbols = [("AAPL", "buy"), ("NVDA", "buy")]
            if TSLA_ENABLED:
                candidate_symbols.append(("TSLA", "buy"))

        # Snapshot current holdings for Mutual Exclusion
        has_bull_etf = any(s in self.positions for s in BULLISH_ETFS)
        has_bear_etf = any(s in self.positions for s in BEARISH_ETFS)

        for sym, side in candidate_symbols:
            if self.get_active_count() >= MAX_CONCURRENT_POSITIONS:
                break
            if sym in self.positions:
                continue
            if time.time() < self.trade_cooldowns.get(sym, 0.0):
                remaining_cd = int(self.trade_cooldowns[sym] - time.time())
                logger.debug(f"⏳ Cooldown active for {sym}: {remaining_cd}s remaining. Skipping.")
                continue

            # ── MUTUAL EXCLUSION CHECK ──
            if sym in BEARISH_ETFS and has_bull_etf:
                logger.debug(f"🚫 [Mutual Exclusion] Skip {sym}: Currently holding Bullish ETF(s) ({[s for s in BULLISH_ETFS if s in self.positions]})")
                continue
            if sym in BULLISH_ETFS and has_bear_etf:
                logger.debug(f"🚫 [Mutual Exclusion] Skip {sym}: Currently holding Bearish/Inverse ETF(s) ({[s for s in BEARISH_ETFS if s in self.positions]})")
                continue

            # Fetch symbol bars
            df = self.fetch_bars(sym, timeframe="1Min", limit=100)
            tech = self.calculate_technical_signals(df)

            # Require technical alignment or strong LOB push
            sym_q = quotes.get(sym, {})
            sym_imb = sym_q.get("imbalance", 0.0)
            sym_price = sym_q.get("midPrice") or sym_q.get("askPrice") or 0.0

            if sym_price <= 0 and df is not None and not df.empty:
                sym_price = float(df["close"].iloc[-1])

            if sym_price <= 0:
                continue

            # Symbol-specific minimum Jev score and imbalance threshold (TSLA strictly guarded)
            min_score = 0.72 if sym == "TSLA" else 0.55
            min_imb = 0.40 if sym == "TSLA" else 0.25

            signal_viable = (tech["signal"] == "BUY") or (sym_imb >= min_imb)
            if not signal_viable:
                continue

            # Step 2: Query Jev AI Sub-Second Orderbook Decision
            logger.info(f"🔎 [Signal Detected] {sym} {side.upper()} | Tech: {tech['reason']} | Imbalance: {sym_imb:+.2f}")
            jev_resp = query_jev_nasdaq(sym, side)

            is_approved = jev_resp.get("approved", False)
            score = jev_resp.get("score", 0.50)
            latency = jev_resp.get("latency_ms", 0.0)
            reason = jev_resp.get("reason", "N/A")
            target_px = jev_resp.get("target_price")

            logger.info(f"⚡ [Jev Decision] {sym} | Approved: {is_approved} | Score: {score:.3f} | Latency: {latency:.1f}ms | Reason: {reason}")

            if is_approved and score >= min_score:
                # Calculate share quantity
                shares = int(target_dollar_size / sym_price)
                if shares < 1:
                    shares = 1

                logger.info(f"🎯 [Executing Trade] Approved by Jev AI! Sizing: {shares} shares (~${shares * sym_price:,.2f}) of {sym}")
                order_res = self.submit_alpaca_order(
                    symbol=sym,
                    side=side,
                    qty=shares,
                    order_type="market" if not target_px else "limit",
                    limit_price=target_px
                )
                if order_res:
                    # Update local state cache
                    if sym in BULLISH_ETFS:
                        has_bull_etf = True
                    elif sym in BEARISH_ETFS:
                        has_bear_etf = True

    def run_cycle(self):
        """Single loop execution cycle"""
        # 1. Check Clock
        is_open = self.check_market_clock()
        if not is_open:
            if self.simulation_mode:
                logger.info(f"🧪 [Simulation Mode Active] Off-market hours (Next Open: {self.next_open_str}). Executing offline simulation cycle...")
            else:
                logger.info(f"🌙 [Market Closed] Next Open: {self.next_open_str} | Waiting for regular trading hours...")
                time.sleep(60)
                return

        # 2. Sync State
        self.sync_account()
        self.sync_positions()

        # 3. Reconcile Any Conflicting Opposing Positions
        self.reconcile_conflicting_positions()

        # 4. Manage Open Trades (Trailing Stop, TP, SL, LOB Wall Exit)
        self.manage_open_positions()

        # 5. Search & Execute Opportunities
        self.evaluate_trading_opportunities()

    def start(self):
        """Main 24/7 autonomous loop"""
        logger.info("🟢 [Daemon Started] Entering continuous trading loop...")
        self.sync_account()
        self.sync_positions()

        while True:
            try:
                self.run_cycle()
                time.sleep(LOOP_INTERVAL_SEC)
            except KeyboardInterrupt:
                logger.info("🛑 [Stop Signal] Stopping Alpaca Jev Trader gracefully.")
                break
            except Exception as e:
                logger.error(f"❌ [Main Loop Exception] {e}", exc_info=True)
                time.sleep(10)

if __name__ == "__main__":
    trader = AlpacaJevTrader()
    trader.start()
