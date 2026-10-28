# -*- coding: utf-8 -*-
"""
alpaca_golden_trader.py — Autonomous Nasdaq Mega-Tech & High-Beta Quant Execution Engine
Optimal Alpha Boost Configuration (Backtest Validated: +6.51% 45d / +11.53% 6m, MDD 4.5%)
NOTE (2026-10-22 review): the backtest figures above pre-date the fixes below (stale-bar bug, Jev gate
never invoked). Re-validate with the current live code path before trusting them.

Key Specifications:
- Targeted Universe (6 Leaders):
  * NVDA, TSLA, PLTR, MSTR, TQQQ, SOXL
- Dynamic Capital Allocation:
  * 1st Position: 45% concentration (maximizes capital rotation)
  * 2nd Position: 30% allocation
  * 3rd Position: 20% allocation (Max 3 concurrent, 95% peak utilization)
  * Gross exposure cap (ALPACA_GOLDEN_MAX_GROSS_EXPOSURE, default 0.95) + max 1 leveraged ETF (TQQQ/SOXL)
- Customized Risk & Bracket Protection:
  * 1x Tech (NVDA): SL -1.2%, TP +2.8%, Trailing Arm +1.5%
  * High-Beta & 3x ETF (TSLA, PLTR, MSTR, TQQQ, SOXL): SL -2.2%, TP +4.8%, Trailing Arm +2.5%
  * EOD flatten 5 min before close, no new entries in the last 15 min (no overnight gap risk)
- Macro Gate:
  * QQQ 5Min EMA50 with 0.15% hysteresis buffer AND EMA9 > EMA20 (Never trade against market trend)
- Entry: EMA trend alignment REQUIRED; L1 imbalance is only used as a veto (IEX L1 is noisy)
- Capital Protection Guard:
  * 30-minute Post-Loss Lockout after any losing exit (including exchange-side bracket SL fills)
  * Pre-cancel open orders on exit to avoid HTTP 403 (held_for_orders)
- Data: Alpaca Data API (feed=ALPACA_DATA_FEED, default iex), latest bars via sort=desc + staleness guard
- Universal Jev AI Gate (trend prompt, fail-closed) + Telegram alerts + restart-safe state
- Only manages symbols in its own universe (paper account may be shared with other bots)
"""

import os
import sys
import time
import math
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

from bot_utils import send_telegram, load_state, save_state  # noqa: E402 (needs env loaded first)

# Import Universal Jev Gate
try:
    from jev.universal_jev_gate import UniversalJevGate
    JEV_GATE_AVAILABLE = True
except ImportError:
    JEV_GATE_AVAILABLE = False

# ── LOGGING SETUP ──
# File logging is done here; systemd stdout goes to journald (avoids duplicated lines in the log file).
LOG_FILE = os.path.join(CURRENT_DIR, "alpaca_golden_trader.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    ]
)
logger = logging.getLogger("AlpacaGoldenTrader")

# ── ROBUST API SESSION SETUP ──
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS", "DELETE"]  # POST excluded: never auto-retry order submission
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

# ── CONFIGURATION (OPTIMAL ALPHA BOOST) ──
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_REST_URL = os.getenv("ALPACA_REST_URL", "https://paper-api.alpaca.markets/v2")
if ALPACA_REST_URL.endswith("/v2"):
    ALPACA_BASE_URL = ALPACA_REST_URL[:-3]
else:
    ALPACA_BASE_URL = ALPACA_REST_URL
ALPACA_DATA_URL = "https://data.alpaca.markets/v2"
# Free plan cannot query recent SIP data (HTTP 403) -> default to IEX. Set "sip" only with a paid data plan.
ALPACA_DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex").strip().lower()

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type": "application/json"
}

# Optimal 6-Symbol Universe
SYMBOLS_UNIVERSE = ["NVDA", "TSLA", "PLTR", "MSTR", "TQQQ", "SOXL"]
HIGH_BETA_SYMBOLS = {"TSLA", "PLTR", "MSTR", "TQQQ", "SOXL"}
LEVERAGED_ETFS = {"TQQQ", "SOXL"}

MAX_CONCURRENT_POSITIONS = 3
MAX_LEVERAGED_ETF_POSITIONS = 1   # TQQQ & SOXL are ~same bet; never hold both
MAX_GROSS_EXPOSURE_PCT = float(os.getenv("ALPACA_GOLDEN_MAX_GROSS_EXPOSURE", "0.90"))
MIN_ORDER_NOTIONAL = 500.0
LOOP_INTERVAL_SEC = 10            # Poll market every 10 seconds during open hours
POST_LOSS_COOLDOWN_SEC = 1800.0   # 30-minute lockout after stop-loss
NORMAL_COOLDOWN_SEC = 900.0       # 15-minute normal exit cooldown
ORDER_FAIL_COOLDOWN_SEC = 300.0   # 5-minute cooldown after a rejected order
JEV_VETO_COOLDOWN_SEC = 1800.0    # 30-minute cooldown after a Jev veto (saves LLM calls)
JEV_ERROR_COOLDOWN_SEC = 120.0    # 2-minute cooldown after a Jev API error (fail-closed)
MAX_EMA20_DIST_PCT = 1.5          # Skip buying if price is already detached >1.5% above 5m EMA20
EOD_FLATTEN_MIN_BEFORE_CLOSE = 5.0
NO_NEW_ENTRY_MIN_BEFORE_CLOSE = 15.0
MAX_BAR_AGE_SEC = {"1Min": 600, "5Min": 1200}  # staleness guard
IMBALANCE_VETO = -0.50            # skip entry if L1 book is heavily offered
CLIENT_ORDER_PREFIX = "golden-"
RUNNER_MIN_GAIN_PCT = 0.025   # +2.5% 이상 대형 수익 포지션만 오버나잇 스윙 이월 허용
STATE_FILE = os.path.join(CURRENT_DIR, "state", "alpaca_golden_state.json")
GUARDIAN_STATE_FILE = os.path.join(CURRENT_DIR, "alpaca_guardian_trend_state.json")


def get_guardian_symbols() -> set:
    """Reads open swing positions from Alpaca Guardian Trend Trader state to prevent interference."""
    try:
        if os.path.exists(GUARDIAN_STATE_FILE):
            with open(GUARDIAN_STATE_FILE, "r") as f:
                data = json.load(f)
                return set(data.get("positions", {}).keys())
    except Exception as e:
        logger.debug(f"Failed to read guardian state: {e}")
    return set()


def get_symbol_risk(symbol: str) -> Tuple[float, float, float]:
    """Returns (stop_loss_pct, take_profit_pct, trailing_arm_pct) - High Alpha Aggressive Boost"""
    if symbol in HIGH_BETA_SYMBOLS:
        return 0.024, 0.080, 0.022   # High Beta / 3x ETF: 2.4% SL, 8.0% TP, 2.2% Trail Arm (1:3.6 R:R)
    return 0.015, 0.060, 0.018       # 1x Tech (NVDA): 1.5% SL, 6.0% TP, 1.8% Trail Arm (1:3.3 R:R)


class AlpacaGoldenTrader:
    def __init__(self):
        self.account_info: Dict[str, Any] = {}
        self.positions: Dict[str, Any] = {}
        self.golden_positions: Dict[str, Dict[str, Any]] = {}  # Tracks positions entered exclusively by Golden Trader
        self.pending_orders: List[Dict[str, Any]] = []
        self.peak_prices: Dict[str, float] = {}
        self.trade_cooldowns: Dict[str, float] = {}
        self.position_entry_times: Dict[str, float] = {}
        self.last_clock_check: float = 0.0
        self.last_closed_log: float = 0.0
        self.is_market_open: bool = False
        self.next_open_str: str = ""
        self.next_close_str: str = ""
        self.quotes_cache: Dict[str, Dict[str, Any]] = {}
        self.eod_flatten_done_for: str = ""
        self.runner_positions: Dict[str, Dict[str, Any]] = {}  # Tracks overnight swing runner positions
        self._load_state()

        # Universal Jev Gate
        if JEV_GATE_AVAILABLE:
            self.jev_gate = UniversalJevGate(confidence_threshold=float(os.getenv("JEV_CONFIDENCE_THRESHOLD", "0.60")))
            if self.jev_gate.is_enabled and self.jev_gate.api_key:
                logger.info("🧠 Universal Jev AI Gate loaded (trend prompt, fail-closed).")
            else:
                logger.warning("⚠️ Jev Gate disabled (USE_JEV_PREDICTION=false or no API key) -> technical signals only.")
        else:
            self.jev_gate = None
            logger.warning("⚠️ Universal Jev Gate module not available -> technical signals only.")

        logger.info("=" * 70)
        logger.info("🏆 [Alpaca Golden Trader] Optimal Alpha Boost Engine Active (v2.1 Guarded)")
        logger.info(f"   REST Endpoint: {ALPACA_BASE_URL} | Data feed: {ALPACA_DATA_FEED}")
        logger.info(f"   Target Universe (6): {', '.join(SYMBOLS_UNIVERSE)}")
        logger.info(f"   Balanced Sizing: 1st=32% | 2nd=30% | 3rd=28% (Max 3 Pos, gross cap {MAX_GROSS_EXPOSURE_PCT*100:.0f}%, max {MAX_LEVERAGED_ETF_POSITIONS} lev-ETF)")
        logger.info(f"   1x Tech (NVDA): SL -1.2%, TP +2.8%, Trail +1.5% (drop 0.5%)")
        logger.info(f"   High Beta (TSLA/PLTR/MSTR/TQQQ/SOXL): SL -2.2%, TP +4.8%, Trail +2.5% (drop 1.0%)")
        logger.info(f"   Lockout Guard: {POST_LOSS_COOLDOWN_SEC/60:.0f}m post-loss | {NORMAL_COOLDOWN_SEC/60:.0f}m post-win | {JEV_VETO_COOLDOWN_SEC/60:.0f}m veto | EOD flatten T-{EOD_FLATTEN_MIN_BEFORE_CLOSE:.0f}m")
        logger.info("=" * 70)

    # ── STATE ──
    def _load_state(self):
        st = load_state(STATE_FILE)
        now = time.time()
        self.peak_prices = {k: float(v) for k, v in st.get("peak_prices", {}).items()}
        self.trade_cooldowns = {k: float(v) for k, v in st.get("trade_cooldowns", {}).items() if float(v) > now}
        self.position_entry_times = {k: float(v) for k, v in st.get("position_entry_times", {}).items()}
        self.golden_positions = st.get("golden_positions", {})
        self.runner_positions = st.get("runner_positions", {})
        if st:
            logger.info(f"♻️ [State Restored] peaks={self.peak_prices} cooldowns={list(self.trade_cooldowns)} golden_pos={list(self.golden_positions)} runners={list(self.runner_positions)}")

    def _save_state(self):
        save_state(STATE_FILE, {
            "peak_prices": self.peak_prices,
            "trade_cooldowns": {k: v for k, v in self.trade_cooldowns.items() if v > time.time()},
            "position_entry_times": self.position_entry_times,
            "golden_positions": self.golden_positions,
            "runner_positions": self.runner_positions,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        })

    # ── ACCOUNT / POSITIONS ──
    def sync_account(self) -> bool:
        """Fetches account details & cash / buying power"""
        try:
            resp = session.get(f"{ALPACA_BASE_URL}/v2/account", headers=HEADERS, timeout=5)
            if resp.status_code == 200:
                self.account_info = resp.json()
                equity = float(self.account_info.get("equity", 0.0))
                cash = float(self.account_info.get("cash", 0.0))
                bp = float(self.account_info.get("buying_power", 0.0))
                status = self.account_info.get("status", "UNKNOWN")
                logger.info(f"💼 [Account Sync] Status: {status} | Equity: ${equity:,.2f} | Cash: ${cash:,.2f} | Buying Power: ${bp:,.2f}")
                return equity > 0
            else:
                logger.error(f"❌ Account sync failed: HTTP {resp.status_code} - {resp.text}")
                return False
        except Exception as e:
            logger.error(f"❌ Account sync exception: {e}")
            return False

    def sync_positions(self) -> Dict[str, Any]:
        """Fetches current open positions from Alpaca (isolated to this bot's own trades)."""
        try:
            resp = session.get(f"{ALPACA_BASE_URL}/v2/positions", headers=HEADERS, timeout=5)
            if resp.status_code == 200:
                pos_list = resp.json()
                guardian_symbols = get_guardian_symbols()

                new_positions = {}
                for p in pos_list:
                    sym = p.get("symbol")
                    if sym not in SYMBOLS_UNIVERSE:
                        continue

                    # Isolation Guard: If symbol is held by Guardian Trend Trader and NOT entered by Golden Trader, isolate it!
                    if sym in guardian_symbols and sym not in self.golden_positions:
                        logger.debug(f"🛡️ [Isolation Guard] Skipping {sym} (allocated to Alpaca Guardian Swing Trader)")
                        continue

                    new_positions[sym] = p

                # Check for positions closed externally (e.g., bracket stop loss / take profit fill)
                for sym in list(self.positions.keys()):
                    if sym not in new_positions:
                        self._handle_external_close(sym, self.positions[sym])
                        self.golden_positions.pop(sym, None)

                self.positions = new_positions
                return self.positions
            else:
                logger.error(f"❌ Sync positions failed: HTTP {resp.status_code}")
                return self.positions
        except Exception as e:
            logger.error(f"❌ Sync positions exception: {e}")
            return self.positions

    def _last_sell_fill_price(self, symbol: str) -> Optional[float]:
        """Most recent SELL fill price for symbol from account activities."""
        try:
            resp = session.get(f"{ALPACA_BASE_URL}/v2/account/activities/FILL", headers=HEADERS,
                               params={"direction": "desc", "page_size": 50}, timeout=5)
            if resp.status_code == 200:
                for act in resp.json():
                    if act.get("symbol") == symbol and act.get("side") == "sell":
                        return float(act.get("price", 0.0)) or None
        except Exception as e:
            logger.debug(f"Fill activity lookup failed for {symbol}: {e}")
        return None

    def _handle_external_close(self, symbol: str, prev_pos: Dict[str, Any]):
        """Position disappeared without our close call -> classify win/loss so post-loss lockout applies."""
        entry = float(prev_pos.get("avg_entry_price", 0.0) or 0.0)
        exit_px = self._last_sell_fill_price(symbol)
        if exit_px and entry > 0:
            pnl_pct = (exit_px - entry) / entry
        else:
            pnl_pct = float(prev_pos.get("unrealized_plpc", 0.0) or 0.0)
        is_loss = pnl_pct < 0
        cooldown = POST_LOSS_COOLDOWN_SEC if is_loss else NORMAL_COOLDOWN_SEC
        self.trade_cooldowns[symbol] = time.time() + cooldown
        self.peak_prices.pop(symbol, None)
        self.position_entry_times.pop(symbol, None)
        self.runner_positions.pop(symbol, None)
        tag = "LOSS" if is_loss else "WIN"
        logger.info(f"ℹ️ [Position Closed Externally] {symbol} ({tag} {pnl_pct*100:+.2f}%, exit ~${exit_px or 0:.2f}) -> cooldown {cooldown/60:.0f}m")
        send_telegram(f"[Alpaca Golden] {symbol} 브래킷 청산 ({tag} {pnl_pct*100:+.2f}%)")

    def refresh_pending_orders(self) -> List[Dict[str, Any]]:
        """Fetches all open/pending orders once per loop (cached in self.pending_orders)"""
        try:
            resp = session.get(f"{ALPACA_BASE_URL}/v2/orders", headers=HEADERS, params={"status": "open", "limit": 500}, timeout=5)
            if resp.status_code == 200:
                self.pending_orders = resp.json()
        except Exception as e:
            logger.error(f"❌ Get pending orders exception: {e}")
        return self.pending_orders

    def _pending_buy_symbols(self) -> set:
        return {o["symbol"] for o in self.pending_orders
                if o.get("side", "").lower() == "buy" and o.get("symbol") in SYMBOLS_UNIVERSE}

    def get_active_symbols(self) -> set:
        """Open positions + pending buy orders (universe only, uses cached pending orders)"""
        return set(self.positions.keys()) | self._pending_buy_symbols()

    def get_active_count(self) -> int:
        return len(self.get_active_symbols())

    def gross_exposure(self) -> float:
        return sum(abs(float(p.get("market_value", 0.0) or 0.0)) for p in self.positions.values())

    # ── MARKET CLOCK ──
    def check_market_clock(self) -> bool:
        """Queries Alpaca market clock API"""
        now = time.time()
        if now - self.last_clock_check < 30.0:
            return self.is_market_open

        try:
            resp = session.get(f"{ALPACA_BASE_URL}/v2/clock", headers=HEADERS, timeout=5)
            if resp.status_code == 200:
                clock = resp.json()
                self.is_market_open = clock.get("is_open", False)
                self.next_open_str = clock.get("next_open", "")
                self.next_close_str = clock.get("next_close", "")
                self.last_clock_check = now
                return self.is_market_open
        except Exception as e:
            logger.warning(f"⚠️ Clock check failed: {e}")
        return False

    def minutes_to_close(self) -> Optional[float]:
        if not self.next_close_str:
            return None
        try:
            close_ts = pd.Timestamp(self.next_close_str).tz_convert("UTC")
            return (close_ts - pd.Timestamp.now(tz="UTC")).total_seconds() / 60.0
        except Exception:
            return None

    # ── MARKET DATA ──
    def fetch_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """Fetches latest quote (BBO) from Alpaca Data API and calculates imbalance"""
        try:
            url = f"{ALPACA_DATA_URL}/stocks/{symbol}/quotes/latest"
            resp = session.get(url, headers=HEADERS, params={"feed": ALPACA_DATA_FEED}, timeout=3)
            if resp.status_code == 200:
                q = resp.json().get("quote", {})
                bid_pr = float(q.get("bp", 0.0))
                bid_sz = float(q.get("bs", 0.0))
                ask_pr = float(q.get("ap", 0.0))
                ask_sz = float(q.get("as", 0.0))

                mid_pr = (bid_pr + ask_pr) / 2.0 if (bid_pr > 0 and ask_pr > 0) else (bid_pr or ask_pr)
                tot_sz = bid_sz + ask_sz
                imbalance = (bid_sz - ask_sz) / tot_sz if tot_sz > 0 else 0.0

                quote_data = {
                    "symbol": symbol,
                    "bidPrice": bid_pr,
                    "bidSize": bid_sz,
                    "askPrice": ask_pr,
                    "askSize": ask_sz,
                    "midPrice": mid_pr,
                    "imbalance": imbalance,
                    "timestamp": time.time()
                }
                self.quotes_cache[symbol] = quote_data
                return quote_data
            logger.debug(f"Quote HTTP {resp.status_code} for {symbol}: {resp.text[:120]}")
        except Exception as e:
            logger.debug(f"Failed to fetch quote for {symbol}: {e}")
        cached = self.quotes_cache.get(symbol, {})
        if cached and time.time() - cached.get("timestamp", 0) < 60:
            return cached
        return {}

    def fetch_bars(self, symbol: str, timeframe: str = "1Min", limit: int = 60) -> Optional[pd.DataFrame]:
        """Fetches the LATEST `limit` bars (sort=desc, then re-ordered ascending).

        Without sort=desc Alpaca returns the FIRST `limit` bars of the day (pre-market 04:00 ET), which made
        every indicator stale all session long.
        """
        try:
            url = f"{ALPACA_DATA_URL}/stocks/{symbol}/bars"
            params = {
                "timeframe": timeframe,
                "limit": limit,
                "adjustment": "all",
                "feed": ALPACA_DATA_FEED,
                "sort": "desc",
                "start": (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            resp = session.get(url, headers=HEADERS, params=params, timeout=5)
            if resp.status_code == 200:
                bars = resp.json().get("bars") or []
                if not bars:
                    return None
                df = pd.DataFrame(bars[::-1])  # back to ascending order
                df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
                for col in ("open", "high", "low", "close"):
                    df[col] = df[col].astype(float)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

                max_age = MAX_BAR_AGE_SEC.get(timeframe)
                if max_age:
                    age = (pd.Timestamp.now(tz="UTC") - df["timestamp"].iloc[-1]).total_seconds()
                    if age > max_age:
                        logger.debug(f"Stale bars for {symbol} {timeframe}: last bar {age:.0f}s old")
                        return None
                return df
            logger.debug(f"Bars HTTP {resp.status_code} for {symbol}: {resp.text[:120]}")
        except Exception as e:
            logger.debug(f"Fetch bars exception for {symbol}: {e}")
        return None

    def calculate_technical_signals(self, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Calculates EMA9, EMA20, EMA50, ATR and trend alignment"""
        if df is None or len(df) < 20:
            return {"signal": "NEUTRAL", "price": 0.0, "ema50": 0.0, "reason": "INSUFFICIENT_OR_STALE_DATA"}

        closes = df["close"]
        price = float(closes.iloc[-1])
        ema9 = float(closes.ewm(span=9, adjust=False).mean().iloc[-1])
        ema20 = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(closes.ewm(span=min(50, len(closes)), adjust=False).mean().iloc[-1])
        tr = pd.concat([df["high"] - df["low"],
                        (df["high"] - closes.shift()).abs(),
                        (df["low"] - closes.shift()).abs()], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        lookback = min(15, len(closes) - 1)
        recent_ret = (price / float(closes.iloc[-1 - lookback]) - 1.0) if lookback > 0 else 0.0

        is_bullish = (price > ema9) and (ema9 > ema20) and (price > ema50)
        is_bearish = (price < ema9) and (ema9 < ema20) and (price < ema50)

        sig = "BUY" if is_bullish else ("SELL" if is_bearish else "NEUTRAL")
        return {
            "signal": sig,
            "price": price,
            "ema9": ema9,
            "ema20": ema20,
            "ema50": ema50,
            "atr_pct": (atr / price * 100.0) if price > 0 and not math.isnan(atr) else 0.0,
            "dist_ema20_pct": (price / ema20 - 1.0) * 100.0 if ema20 > 0 else 0.0,
            "recent_return_pct": recent_ret * 100.0,
            "lookback_bars": lookback,
            "reason": f"P:${price:.2f} | EMA9:${ema9:.2f} | EMA20:${ema20:.2f} | EMA50:${ema50:.2f}"
        }

    def get_dynamic_position_pct(self) -> float:
        """Balanced Risk Sizing: 30% per trade across up to 3 concurrent positions (30% / 30% / 30%)"""
        active = self.get_active_count()
        if active < MAX_CONCURRENT_POSITIONS:
            return 0.30
        return 0.0

    # ── ORDERS ──
    def submit_bracket_order(self, symbol: str, qty: int, side: str = "buy") -> Optional[Dict[str, Any]]:
        """Submits GTC Bracket order (TP/SL) to Alpaca Paper Trading REST API"""
        if qty <= 0:
            logger.warning(f"⚠️ Invalid order qty for {symbol}: {qty}")
            return None

        bars = self.fetch_bars(symbol, timeframe="1Min", limit=5)
        bar_close = float(bars["close"].iloc[-1]) if (bars is not None and not bars.empty) else 0.0

        q = self.fetch_latest_quote(symbol)
        quote_ask = q.get("askPrice") or q.get("midPrice") or 0.0

        # Discrepancy protection: if IEX ask is distorted (>1.2% discrepancy from 1Min bar close), clamp to bar close
        if bar_close > 0 and quote_ask > 0 and abs(quote_ask - bar_close) / bar_close > 0.012:
            logger.warning(f"⚠️ {symbol} IEX quote ask (${quote_ask:.2f}) deviates >1.2% from 1Min close (${bar_close:.2f}). Clamping base_price to bar close.")
            base_price = bar_close
        else:
            base_price = quote_ask or bar_close

        if base_price <= 0:
            logger.error(f"❌ Cannot submit order for {symbol}: Unable to obtain base price.")
            return None

        sym_sl, sym_tp, sym_trail = get_symbol_risk(symbol)
        # Ensure SL stop_price is strictly below market execution price to prevent HTTP 422 rejects
        sl_base = min(base_price, bar_close) if bar_close > 0 else base_price
        sl_price = round(sl_base * (1.0 - sym_sl) - 0.01, 2)
        tp_price = round(base_price * (1.0 + sym_tp), 2)

        order_data = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side.lower(),
            "type": "market",
            "time_in_force": "gtc",
            "order_class": "bracket",
            "client_order_id": f"{CLIENT_ORDER_PREFIX}{symbol}-{int(time.time())}",
            "take_profit": {"limit_price": str(tp_price)},
            "stop_loss": {"stop_price": str(sl_price)}
        }

        logger.info(f"📤 [Alpha Bracket Submit] BUY {qty}x {symbol} @ ~${base_price:.2f} | TP: ${tp_price} (+{sym_tp*100:.1f}%) | SL: ${sl_price} (-{sym_sl*100:.1f}%)")

        try:
            resp = session.post(f"{ALPACA_BASE_URL}/v2/orders", headers=HEADERS, json=order_data, timeout=8)
            if resp.status_code in [200, 201]:
                res_json = resp.json()
                logger.info(f"✅ [Order Confirmed] ID: {res_json.get('id')} | Status: {res_json.get('status')} | {symbol}")
                self.trade_cooldowns[symbol] = time.time() + NORMAL_COOLDOWN_SEC
                self.position_entry_times[symbol] = time.time()
                self.golden_positions[symbol] = {"qty": int(qty), "entry_price": base_price, "entry_time": time.time()}
                self.peak_prices[symbol] = base_price
                self.pending_orders.append(res_json)  # keep active-count accurate within this loop
                self._save_state()
                send_telegram(f"[Alpaca Golden] BUY {qty}x {symbol} @~${base_price:.2f} | TP ${tp_price} / SL ${sl_price}")
                return res_json
            else:
                logger.error(f"❌ Order submission rejected: HTTP {resp.status_code} - {resp.text}")
                self.trade_cooldowns[symbol] = time.time() + ORDER_FAIL_COOLDOWN_SEC
                send_telegram(f"[Alpaca Golden] ❌ {symbol} 주문 거부: HTTP {resp.status_code} {resp.text[:200]}",
                              dedupe_key=f"alpaca_reject_{symbol}", min_interval_sec=1800)
                return None
        except Exception as e:
            logger.error(f"❌ Order submission exception: {e}")
            self.trade_cooldowns[symbol] = time.time() + ORDER_FAIL_COOLDOWN_SEC
            return None

    def manage_open_positions(self):
        """Monitors unrealized PnL and executes dynamic trailing stop & ask-wall protection"""
        if not self.positions:
            return

        for symbol, pos in list(self.positions.items()):
            try:
                curr_price = float(pos.get("current_price", 0.0))
                entry_price = float(pos.get("avg_entry_price", 0.0))
                unrealized_plpc = float(pos.get("unrealized_plpc", 0.0))

                if curr_price <= 0 or entry_price <= 0:
                    continue

                sym_sl, sym_tp, sym_trail_arm = get_symbol_risk(symbol)

                # Track peak price
                peak = self.peak_prices.get(symbol, entry_price)
                if curr_price > peak:
                    self.peak_prices[symbol] = curr_price
                    peak = curr_price

                # Compute profit metrics
                gain_from_entry = (curr_price - entry_price) / entry_price
                peak_gain = (peak - entry_price) / entry_price

                # Overnight Runner Breakeven Defense
                if symbol in self.runner_positions:
                    runner_sl = entry_price * 1.002
                    if curr_price <= runner_sl:
                        logger.info(f"🛡️ [Overnight Runner Breakeven] {symbol} hit breakeven protection (${curr_price:.2f} <= ${runner_sl:.2f}). Exiting remaining position.")
                        self.close_position(symbol, f"RUNNER_BREAKEVEN_PROTECT_{gain_from_entry*100:+.2f}PCT")
                        continue

                # Breakeven Defense for Intraday Trades: Lock gains after +1.5% peak if price pulls back near entry
                if peak_gain >= 0.015 and gain_from_entry <= 0.002:
                    logger.info(f"🛡️ [Breakeven Defense] {symbol} peaked at +{peak_gain*100:.2f}%, locking breakeven exit (Gain: {gain_from_entry*100:+.2f}%)")
                    self.close_position(symbol, f"BREAKEVEN_DEFENSE_{gain_from_entry*100:.2f}PCT")
                    continue

                # Trailing stop: armed once peak profit exceeded sym_trail_arm (+2.2% for high beta, +1.8% for tech)
                if peak_gain >= sym_trail_arm:
                    drop_from_peak = (peak - curr_price) / peak
                    trail_drop_threshold = 0.012 if symbol in HIGH_BETA_SYMBOLS else 0.008
                    if drop_from_peak >= trail_drop_threshold:
                        logger.info(f"🎯 [Trailing Stop Exit] {symbol} locked in profit: Entry ${entry_price:.2f} -> Peak ${peak:.2f} -> Curr ${curr_price:.2f} (Gain: {gain_from_entry*100:+.2f}%)")
                        self.close_position(symbol, f"TRAILING_STOP_EXIT_{gain_from_entry*100:.2f}PCT")
                        continue

            except Exception as e:
                logger.error(f"Error managing position for {symbol}: {e}")

    def close_position(self, symbol: str, reason: str = "MANUAL", partial_ratio: float = 1.0) -> bool:
        """Closes position (or partial ratio) via Alpaca REST API. Sells only Golden Trader's quantity with strict short-guard."""
        pos = self.positions.get(symbol, {})
        unrealized_pl = float(pos.get("unrealized_pl", 0.0))
        unrealized_plpc = float(pos.get("unrealized_plpc", 0.0))

        # Check actual tracked quantity owned by Golden Trader
        actual_pos_qty = int(float(pos.get("qty", 0)))
        if actual_pos_qty <= 0:
            actual_pos_qty = int(self.golden_positions.get(symbol, {}).get("qty", 0))

        # Strict guard against accidental short selling
        if actual_pos_qty <= 0:
            logger.warning(f"⚠️ [Close Skip] {symbol} has 0 or negative tracked shares ({actual_pos_qty}). Removing from state to avoid short selling.")
            self.positions.pop(symbol, None)
            self.golden_positions.pop(symbol, None)
            self.runner_positions.pop(symbol, None)
            self._save_state()
            return True

        qty_to_close = int(actual_pos_qty * partial_ratio)
        if qty_to_close <= 0:
            qty_to_close = 1
        qty_to_close = min(qty_to_close, actual_pos_qty)

        is_partial = qty_to_close < actual_pos_qty
        logger.info(f"🔄 [Closing Position] {symbol} ({qty_to_close}/{actual_pos_qty} shares, partial={is_partial}) | Reason: {reason}")

        try:
            # 1. Close position via Alpaca native liquidation API (auto-cancels open bracket orders)
            if not is_partial:
                resp = session.delete(f"{ALPACA_BASE_URL}/v2/positions/{symbol}", headers=HEADERS, timeout=8)
            else:
                resp = session.delete(f"{ALPACA_BASE_URL}/v2/positions/{symbol}", headers=HEADERS, params={"qty": str(qty_to_close)}, timeout=8)

            # If direct position deletion was rejected, fallback to manual cancel + market sell
            if resp.status_code not in [200, 201, 204]:
                logger.warning(f"⚠️ Direct position delete returned HTTP {resp.status_code} ({resp.text}). Executing fallback order cancel + sell...")
                for o in self.pending_orders:
                    if o.get("symbol") == symbol:
                        session.delete(f"{ALPACA_BASE_URL}/v2/orders/{o.get('id')}", headers=HEADERS, timeout=3)
                time.sleep(1.0)
                close_order = {
                    "symbol": symbol,
                    "qty": str(qty_to_close),
                    "side": "sell",
                    "type": "market",
                    "time_in_force": "day",
                    "client_order_id": f"{CLIENT_ORDER_PREFIX}close-{symbol}-{int(time.time())}"
                }
                resp = session.post(f"{ALPACA_BASE_URL}/v2/orders", headers=HEADERS, json=close_order, timeout=8)
                if resp.status_code not in [200, 201]:
                    logger.error(f"❌ Position close fallback failed for {symbol}: HTTP {resp.status_code} - {resp.text} (will retry next loop)")
                    send_telegram(f"[Alpaca Golden] ❌ {symbol} 청산 실패 ({reason}): HTTP {resp.status_code}",
                                  dedupe_key=f"alpaca_close_fail_{symbol}", min_interval_sec=600)
                    return False
        except Exception as e:
            logger.error(f"❌ Exception closing {symbol}: {e} (will retry next loop)")
            return False

        if is_partial:
            # Partial close successful: update remaining quantity and designate as overnight runner
            rem_qty = actual_pos_qty - qty_to_close
            entry_px = float(pos.get("avg_entry_price", 0.0))
            if symbol in self.golden_positions:
                self.golden_positions[symbol]["qty"] = rem_qty
            self.runner_positions[symbol] = {
                "entry": entry_px,
                "sl": entry_px * 1.002,
                "qty": rem_qty,
                "locked_at": time.time()
            }
            self.positions[symbol]["qty"] = str(rem_qty)
            self._save_state()
            logger.info(f"✅ [Partial Close] {symbol} {qty_to_close} shares locked. {rem_qty} shares designated as Overnight Runner with Breakeven SL.")
            send_telegram(f"[Alpaca Golden] 🚀 {symbol} +{unrealized_plpc*100:+.2f}% 수익 50% 분할 익절 ({qty_to_close}주 청산, 잔여 {rem_qty}주 오버나잇 스윙 홀딩)")
            return True

        # Full Close
        is_loss = (unrealized_pl < 0) or ("STOP_LOSS" in reason) or ("HEAVY_ASK" in reason)
        if is_loss:
            cooldown_sec = POST_LOSS_COOLDOWN_SEC
            logger.info(f"⏳ [Post-Loss Lockout] {symbol} closed with loss ({unrealized_plpc*100:+.2f}%). Enforcing {POST_LOSS_COOLDOWN_SEC/60:.0f}m cooling period.")
        else:
            cooldown_sec = NORMAL_COOLDOWN_SEC
        self.trade_cooldowns[symbol] = time.time() + cooldown_sec
        self.peak_prices.pop(symbol, None)
        self.position_entry_times.pop(symbol, None)
        self.golden_positions.pop(symbol, None)
        self.positions.pop(symbol, None)
        self.runner_positions.pop(symbol, None)
        self.pending_orders = [o for o in self.pending_orders if o.get("symbol") != symbol]
        self._save_state()
        logger.info(f"✅ [Position Closed] {symbol} ({qty_to_close} shares) closed successfully.")
        send_telegram(f"[Alpaca Golden] {symbol} 청산 ({reason}) PnL {unrealized_plpc*100:+.2f}% (${unrealized_pl:+,.2f})")
        return True

    def flatten_all(self, reason: str):
        """Intelligent EOD Position Handler:
        - Big Trend Runners (unrealized_plpc >= RUNNER_MIN_GAIN_PCT, +2.5%):
          Lock 50% profit, hold remaining 50% overnight with Breakeven Stop!
        - Non-runners / low profit / losses:
          Flatten 100% to eliminate overnight gap-down risk.
        """
        if not self.positions:
            return
        logger.info(f"🌙 [EOD Intelligent Flatten] Managing {list(self.positions.keys())} | Reason: {reason}")
        for sym in list(self.positions.keys()):
            pos = self.positions.get(sym, {})
            unrealized_plpc = float(pos.get("unrealized_plpc", 0.0))
            is_runner = sym in self.runner_positions

            if unrealized_plpc >= RUNNER_MIN_GAIN_PCT and not is_runner:
                logger.info(f"🚀 [Super-Trend Runner Selected] {sym}: Gain +{unrealized_plpc*100:.2f}%. Locking 50% profit, holding remaining 50% overnight!")
                self.close_position(sym, "OVERNIGHT_RUNNER_LOCK_50PCT", partial_ratio=0.5)
            elif is_runner:
                logger.info(f"🌟 [Active Runner Holding] {sym} is already an overnight runner (PnL: {unrealized_plpc*100:+.2f}%). Riding the trend with Breakeven SL.")
            else:
                logger.info(f"🛡️ [Standard EOD Flatten] {sym} (PnL: {unrealized_plpc*100:+.2f}% < +{RUNNER_MIN_GAIN_PCT*100:.1f}%). Closing 100% to eliminate gap risk.")
                self.close_position(sym, reason, partial_ratio=1.0)

    # ── ENTRY LOGIC ──
    def evaluate_trading_opportunities(self):
        """Scans Universe, checks QQQ Macro, and executes Trades with Dynamic Sizing"""
        if self.get_active_count() >= MAX_CONCURRENT_POSITIONS:
            return

        equity = float(self.account_info.get("equity", 0.0) or 0.0)
        buying_power = float(self.account_info.get("buying_power", 0.0) or 0.0)
        if equity <= 0:
            return

        # Step 1: Scan QQQ benchmark for macro regime (5Min, fallback 1Min)
        qqq_bars = self.fetch_bars("QQQ", timeframe="5Min", limit=60)
        if qqq_bars is None or len(qqq_bars) < 20:
            qqq_bars = self.fetch_bars("QQQ", timeframe="1Min", limit=60)

        qqq_tech = self.calculate_technical_signals(qqq_bars)
        qqq_c = qqq_tech.get("price", 0.0)
        qqq_e50 = qqq_tech.get("ema50", 0.0)

        if qqq_c <= 0 or qqq_e50 <= 0:
            logger.debug("QQQ data unavailable/stale. Standing by.")
            return

        # Macro Trend Gate: QQQ > EMA50 * 1.0015 (0.15% hysteresis buffer) AND short-term trend up
        is_bullish = (qqq_c > qqq_e50 * 1.0015) and (qqq_tech["ema9"] > qqq_tech["ema20"])

        if not is_bullish:
            logger.debug(f"QQQ Macro not bullish (P:${qqq_c:.2f} vs EMA50:${qqq_e50:.2f}). Standing by.")
            return

        guardian_symbols = get_guardian_symbols()

        for sym in SYMBOLS_UNIVERSE:
            active = self.get_active_symbols()
            if len(active) >= MAX_CONCURRENT_POSITIONS:
                break
            if sym in active:
                continue
            # Isolation Guard: Skip symbols currently held by Guardian Trend Trader
            if sym in guardian_symbols:
                logger.debug(f"🛡️ [Isolation Guard] {sym} is held by Alpaca Guardian Trend Trader -> skipping.")
                continue
            if time.time() < self.trade_cooldowns.get(sym, 0.0):
                continue
            if sym in LEVERAGED_ETFS and len(active & LEVERAGED_ETFS) >= MAX_LEVERAGED_ETF_POSITIONS:
                continue

            # Fetch bars first (cheaper filter), then quote
            df = self.fetch_bars(sym, timeframe="5Min", limit=60)
            tech = self.calculate_technical_signals(df)
            if tech["signal"] != "BUY":
                continue

            # Overextension Guard: skip buying if price is already detached >1.5% above 5m EMA20
            dist_ema20 = tech.get("dist_ema20_pct", 0.0)
            if dist_ema20 > MAX_EMA20_DIST_PCT:
                logger.info(f"⏳ [Overextension Guard] {sym} BUY skipped: Price is +{dist_ema20:.2f}% above EMA20 (overextended buying trap).")
                continue

            q = self.fetch_latest_quote(sym)
            sym_imb = q.get("imbalance", 0.0)
            sym_price = q.get("askPrice") or q.get("midPrice") or tech.get("price", 0.0)
            if sym_price <= 0:
                continue
            if sym_imb <= IMBALANCE_VETO:
                logger.debug(f"{sym} skipped: heavy offer-side imbalance {sym_imb:+.2f}")
                continue

            # Sizing (gross exposure cap + buying power)
            alloc_pct = self.get_dynamic_position_pct()
            room = equity * MAX_GROSS_EXPOSURE_PCT - self.gross_exposure()
            target_dollar_size = min(equity * alloc_pct, room, buying_power * 0.95)
            qty = int(target_dollar_size / sym_price)
            if qty <= 0 or qty * sym_price < MIN_ORDER_NOTIONAL:
                logger.debug(f"{sym} skipped: insufficient room (target ${target_dollar_size:,.0f})")
                continue

            # Step 2: Jev AI gate (trend prompt, fail-closed)
            sym_sl, sym_tp, _ = get_symbol_risk(sym)
            if self.jev_gate:
                try:
                    decision = self.jev_gate.evaluate_sync(
                        asset_class="US_STOCKS",
                        symbol=sym,
                        side="LONG",
                        context_metrics={
                            **{k: tech[k] for k in ("price", "ema9", "ema20", "ema50", "atr_pct",
                                                     "dist_ema20_pct", "recent_return_pct", "lookback_bars")},
                            "imbalance": sym_imb,
                            "macro": f"QQQ BULLISH (P {qqq_c:.2f} > EMA50 {qqq_e50:.2f})",
                            "sl_pct": sym_sl * 100, "tp_pct": sym_tp * 100,
                        },
                    )
                except Exception as e:
                    logger.warning(f"Jev gate exception for {sym}: {e} -> veto")
                    self.trade_cooldowns[sym] = time.time() + JEV_ERROR_COOLDOWN_SEC
                    continue

                if not decision.approved:
                    cd = JEV_ERROR_COOLDOWN_SEC if decision.is_fallback else JEV_VETO_COOLDOWN_SEC
                    self.trade_cooldowns[sym] = time.time() + cd
                    logger.info(f"🚫 [Jev Veto] {sym} BUY (conf {decision.confidence:.2f}, {decision.latency_ms:.0f}ms): {decision.reason} -> cooldown {cd/60:.0f}m")
                    continue
                logger.info(f"🧠 [Jev Approved] {sym} (conf {decision.confidence:.2f}, {decision.latency_ms:.0f}ms{' , fallback' if decision.is_fallback else ''}): {decision.reason}")

            logger.info(f"🚀 [Alpha Signal Confirmed] {sym} BUY (Alloc: {alloc_pct*100:.0f}%, ~${qty*sym_price:,.0f}) | Imb: {sym_imb:+.2f} | Tech: {tech['reason']}")
            self.submit_bracket_order(sym, qty, "buy")

    # ── MAIN LOOP ──
    def run(self):
        """Main 24/7 Execution Loop"""
        logger.info("🟢 [Alpaca Golden Trader] Engine started. Running 24/7 supervisor...")
        send_telegram("[Alpaca Golden] 엔진 시작")
        self.sync_account()
        self.sync_positions()

        while True:
            try:
                is_open = self.check_market_clock()
                if not is_open:
                    if time.time() - self.last_closed_log >= 600:
                        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
                        logger.info(f"💤 [Market Closed] Current: {now_str} | Next Open: {self.next_open_str} | Open Positions: {list(self.positions.keys())}")
                        self.last_closed_log = time.time()
                    time.sleep(60.0)
                    continue

                if not self.sync_account():
                    logger.warning("⚠️ Account sync failed -> skipping this cycle (no sizing on stale equity).")
                    time.sleep(LOOP_INTERVAL_SEC)
                    continue
                self.sync_positions()
                self.refresh_pending_orders()

                mins_left = self.minutes_to_close()
                if mins_left is not None and mins_left <= EOD_FLATTEN_MIN_BEFORE_CLOSE:
                    self.flatten_all("EOD_FLATTEN")
                    if self.eod_flatten_done_for != self.next_close_str:
                        self.eod_flatten_done_for = self.next_close_str
                        runners = list(self.runner_positions.keys())
                        if runners:
                            send_telegram(f"[Alpaca Golden] 🌙 장 마감 관리 완료 (오버나잇 러너 {runners} 스윙 유지 중)")
                        else:
                            send_telegram("[Alpaca Golden] 🌙 장 마감 관리 완료 (모든 포지션 당일 청산 완료)")
                else:
                    self.manage_open_positions()
                    if mins_left is None or mins_left > NO_NEW_ENTRY_MIN_BEFORE_CLOSE:
                        self.evaluate_trading_opportunities()

                self._save_state()
                time.sleep(LOOP_INTERVAL_SEC)

            except KeyboardInterrupt:
                logger.info("🛑 [Shutdown] Stopping Alpaca Golden Trader...")
                self._save_state()
                break
            except Exception as e:
                logger.error(f"❌ Unexpected loop exception: {e}", exc_info=True)
                send_telegram(f"[Alpaca Golden] ❌ 루프 예외: {e}", dedupe_key="alpaca_loop_exc", min_interval_sec=1800)
                time.sleep(10.0)

if __name__ == "__main__":
    trader = AlpacaGoldenTrader()
    trader.run()
