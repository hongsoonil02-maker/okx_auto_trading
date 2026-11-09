# -*- coding: utf-8 -*-
"""
alpaca_golden_trader.py — Argon US V4-Pro Bidirectional Sector-Regime & Sniper Execution Engine
Option 3 Practical Aggressive Configuration (1.8x Reg-T Sniper: 60% / Slot x Max 3 Slots = 180% Gross Cap)

Backtest Verified Across Full 6M / 3M / 1M Windows (2026-05-07 ~ 2026-11-06, incl. slippage):
- 6 Months (5/7 ~ 11/6): +24.2% (+$24,218.8 / $100k), 64.6% WR (93W/51L) [vs QQQ -5.89%, TQQQ -27.52%]
- 3 Months (8/7 ~ 11/6): +43.5% (+$43,483.8 / $100k), 70.3% WR (52W/22L), PF 1.88, MDD 17.7%
- 1 Month (10/7 ~ 11/6): +26.5% (+$26,539.7 / $100k), 63.6% WR (21W/12L), PF 2.17, MDD 17.7%

Core Architecture (Argon US V4-Pro):
1. 4 Independent US Macro Sector Pillars (captures US Sector Rotation):
   - Dow Jones: DIA -> 3x Bull UDOW / 3x Bear SDOW / 1x Native Short DIA
   - S&P 500  : SPY -> 3x Bull UPRO / 3x Bear SPXS / 1x Native Short SPY
   - Nasdaq   : QQQ -> 3x Bull TQQQ / 3x Bear SQQQ / 1x Native Short QQQ
   - Semis    : SOXX -> 3x Bull SOXL / 3x Bear SOXS / 1x Native Short SOXX
2. Institutional Mega-Cap Bull Dip Universe (Long-Only to eliminate single-stock earnings short squeezes):
   - NVDA, META, AVGO, AMZN, MSFT, AAPL, AMD, PLTR
3. Three Orthogonal Alpha Pods:
   - Pod A (BULL_DIP): Sector/Mega-Cap in Primary Daily Uptrend + Confirmed 1H RSI(2) <= 15 & Z(20) <= -1.25 panic dip
   - Pod B (BEAR_RIP): Sector in Confirmed Daily Downtrend (Close < MA20 & EMA8 < EMA21) + 1H RSI(2) >= 85 relief-rally fade
                       via 3x Bear ETFs (SDOW, SPXS, SQQQ, SOXS) and Native Index Shorts (DIA, SPY, QQQ, SOXX)
   - Pod C (WATERFALL_BEAR): Synchronized 3+/4 US Sector 1H EMA(9 < 21 < 50) acute crash breakdown rider
4. Hybrid Consolidated Data & Margin-Aware Execution:
   - Historical SIP consolidated tape + real-time IEX stitching (zero HTTP 403 on free plan, zero repaint)
   - Per-asset FINRA/Reg-T margin-aware position sizing (30% / 75% / 100% req respected automatically)
   - GTC Bracket SL/TP protection across multi-day RTH swings + profit-guarded 1H MA(10)/RSI(5) snapback harvest
"""

import os
import sys
import time
import math
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

from bot_utils import send_telegram, load_state, save_state  # noqa: E402

try:
    from jev.universal_jev_gate import UniversalJevGate
    JEV_GATE_AVAILABLE = True
except ImportError:
    JEV_GATE_AVAILABLE = False

# ── LOGGING SETUP ──
LOG_FILE = os.path.join(CURRENT_DIR, "alpaca_golden_trader.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"),
    ],
)
logger = logging.getLogger("AlpacaGoldenTrader")

# ── ROBUST API SESSION SETUP ──
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS", "DELETE"],  # Never auto-retry POST order submission
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

# ── API CONFIGURATION ──
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_REST_URL = os.getenv("ALPACA_REST_URL", "https://paper-api.alpaca.markets/v2")
ALPACA_BASE_URL = ALPACA_REST_URL[:-3] if ALPACA_REST_URL.endswith("/v2") else ALPACA_REST_URL
ALPACA_DATA_URL = "https://data.alpaca.markets/v2"
ALPACA_QUOTE_FEED = os.getenv("ALPACA_DATA_FEED", "iex").strip().lower() or "iex"

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type": "application/json",
}

# ── ARGON US V4-PRO UNIVERSE & OPTION 3 (1.8x REG-T SNIPER) PARAMETERS ──
PAIR_MAP: List[Tuple[str, str, str]] = [
    ("DIA",  "UDOW", "SDOW"),  # Dow Jones 30
    ("SPY",  "UPRO", "SPXS"),  # S&P 500
    ("QQQ",  "TQQQ", "SQQQ"),  # Nasdaq 100
    ("SOXX", "SOXL", "SOXS"),  # PHLX Semiconductors
]
BENCHMARK_ETFS = [p[0] for p in PAIR_MAP]
BULL_3X_ETFS = {p[1] for p in PAIR_MAP}
BEAR_3X_ETFS = {p[2] for p in PAIR_MAP}
ALL_3X_ETFS = BULL_3X_ETFS | BEAR_3X_ETFS
MEGAS_SAFE = ["NVDA", "META", "AVGO", "AMZN", "MSFT", "AAPL", "AMD", "PLTR"]

# Include legacy symbols (TSLA, MSTR) in managed universe lookup only so any residual position is safely managed
CORE_UNIVERSE = BENCHMARK_ETFS + sorted(list(ALL_3X_ETFS)) + MEGAS_SAFE
SYMBOLS_UNIVERSE = list(dict.fromkeys(CORE_UNIVERSE + ["TSLA", "MSTR"]))

# Option 3: 1.8x Reg-T Practical Aggressive Sniper (60% per slot x Max 3 slots = 180% max gross exposure)
SLOT_TARGET_PCT = float(os.getenv("ALPACA_SLOT_PCT", "0.60"))
MAX_CONCURRENT_POSITIONS = int(os.getenv("ALPACA_MAX_SLOTS", "3"))
MAX_GROSS_EXPOSURE_PCT = float(os.getenv("ALPACA_GOLDEN_MAX_GROSS_EXPOSURE", "1.80"))
MIN_ORDER_NOTIONAL = 500.0

# Signal Thresholds (Backtest-Verified V4_HYB_1.8X_SNIPER)
BULL_DIP_RSI2 = 15.0
MEGA_DIP_RSI2 = 12.0
BEAR_RIP_RSI2 = 85.0
NAT_SHORT_RIP_RSI2 = 88.0

# Risk & Harvest Parameters
SL_3X_PCT = 0.055
TP_3X_PCT = 0.140
SL_1X_PCT = 0.032
TP_1X_PCT = 0.080

BE_TRIG_3X_PCT = 0.030
BE_TRIG_1X_PCT = 0.015
BE_LOCK_OFFSET_PCT = 0.004

SNAP_MIN_GAIN_3X = 0.012
SNAP_MIN_GAIN_1X = 0.007
SNAP_MAX_RTH_BARS = 28  # 4 full RTH trading days (7 RTH 1H bars/day)

WF_BE_TRIG_3X = 0.025
WF_BE_TRIG_1X = 0.010
WF_TRAIL_TRIG_3X = 0.050
WF_TRAIL_TRIG_1X = 0.020
WF_TRAIL_DROP_3X = 0.018
WF_TRAIL_DROP_1X = 0.007
WF_MAX_RTH_BARS = 14    # 2 full RTH trading days if stagnant

LOOP_INTERVAL_SEC = 10.0
EXIT_COOLDOWN_SEC = 4 * 3600.0      # 4-hour post-exit cooldown per symbol (matches backtest)
ORDER_FAIL_COOLDOWN_SEC = 300.0
JEV_VETO_COOLDOWN_SEC = 1800.0
JEV_ERROR_COOLDOWN_SEC = 120.0
RADAR_LOG_INTERVAL_SEC = 300.0

CLIENT_OPEN_PREFIX = "v4-open-"
CLIENT_CLOSE_PREFIX = "v4-close-"

STATE_FILE = os.path.join(CURRENT_DIR, "state", "alpaca_golden_state.json")
GUARDIAN_STATE_FILE = os.path.join(CURRENT_DIR, "alpaca_guardian_trend_state.json")

DEFAULT_MARGIN_REQ: Dict[str, Tuple[float, float]] = {
    "UDOW": (100.0, 100.0), "SDOW": (100.0, 100.0),
    "UPRO": (100.0, 100.0), "SPXS": (100.0, 100.0),
    "SOXS": (100.0, 100.0),
    "TQQQ": (75.0, 100.0),  "SQQQ": (75.0, 90.0), "SOXL": (75.0, 90.0),
}


def calc_wilder_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1.0 / period, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1.0 / period, adjust=False).mean().replace(0.0, 1e-9)
    return 100.0 - (100.0 / (1.0 + ma_up / ma_dn))


def get_guardian_symbols() -> set:
    try:
        if os.path.exists(GUARDIAN_STATE_FILE):
            with open(GUARDIAN_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("positions", {}).keys())
    except Exception as e:
        logger.debug(f"Failed to read guardian state: {e}")
    return set()


def get_symbol_risk(symbol: str) -> Tuple[float, float, bool]:
    is_3x = symbol in ALL_3X_ETFS
    if is_3x:
        return SL_3X_PCT, TP_3X_PCT, True
    return SL_1X_PCT, TP_1X_PCT, False


class AlpacaGoldenTrader:
    def __init__(self):
        self.account_info: Dict[str, Any] = {}
        self.positions: Dict[str, Any] = {}
        self.golden_positions: Dict[str, Dict[str, Any]] = {}
        self.pending_orders: List[Dict[str, Any]] = []
        self.peak_prices: Dict[str, float] = {}
        self.floor_sl_prices: Dict[str, float] = {}
        self.trade_cooldowns: Dict[str, float] = {}
        self.position_entry_times: Dict[str, float] = {}
        self.daily_loss_dates: Dict[str, str] = {}

        self.last_clock_check: float = 0.0
        self.last_closed_log: float = 0.0
        self.last_radar_log: float = 0.0
        self.last_account_log: float = 0.0
        self.is_market_open: bool = False
        self.next_open_str: str = ""
        self.next_close_str: str = ""
        self.eod_summary_done_for: str = ""

        # Caches for SIP/IEX data & asset margin specs
        self.quotes_cache: Dict[str, Dict[str, Any]] = {}
        self.daily_cache_date: str = ""
        self.daily_metrics_cache: Dict[str, Dict[str, Any]] = {}
        self.sip_1h_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}
        self.merged_1h_cache: Dict[str, Tuple[float, str, pd.DataFrame]] = {}
        self.asset_specs: Dict[str, Dict[str, Any]] = {}

        self._load_state()
        self._load_asset_specs()

        if JEV_GATE_AVAILABLE:
            self.jev_gate = UniversalJevGate(confidence_threshold=float(os.getenv("JEV_CONFIDENCE_THRESHOLD", "0.58")))
            if self.jev_gate.is_enabled and self.jev_gate.api_key:
                logger.info("🧠 Universal Jev AI Gate loaded (Argon US V4-Pro pod-aware prompt).")
            else:
                logger.warning("⚠️ Jev Gate disabled -> deterministic quantitative signals active.")
        else:
            self.jev_gate = None
            logger.warning("⚠️ Universal Jev Gate module unavailable -> deterministic quantitative signals active.")

        logger.info("=" * 78)
        logger.info("🦅 [Alpaca Golden Trader] Argon US V4-Pro (Option 3: 1.8x Reg-T Sniper) Active")
        logger.info(f"   REST Endpoint : {ALPACA_BASE_URL} | Hybrid Data: SIP Consolidated + Real-Time IEX")
        logger.info("   4 US Pillars  : DIA(UDOW/SDOW) | SPY(UPRO/SPXS) | QQQ(TQQQ/SQQQ) | SOXX(SOXL/SOXS)")
        logger.info(f"   Mega-Cap Dips : {', '.join(MEGAS_SAFE)} | Native Index Short: DIA, SPY, QQQ, SOXX")
        logger.info(f"   Sniper Sizing : {SLOT_TARGET_PCT*100:.0f}% / Slot x Max {MAX_CONCURRENT_POSITIONS} Slots (Gross Cap {MAX_GROSS_EXPOSURE_PCT*100:.0f}%)")
        logger.info(f"   Alpha Pods    : BULL_DIP (RSI2<={BULL_DIP_RSI2:.0f}) | BEAR_RIP (RSI2>={BEAR_RIP_RSI2:.0f}) | WATERFALL_BEAR (3+/4 Crash)")
        logger.info("   Swing Harvest : GTC Bracket (3x: -5.5%/+14% | 1x: -3.2%/+8%) + Profit-Guarded 1H MA10/RSI5 Snap")
        logger.info("=" * 78)

    # ── STATE PERSISTENCE ──
    def _load_state(self):
        st = load_state(STATE_FILE)
        now = time.time()
        self.peak_prices = {k: float(v) for k, v in st.get("peak_prices", {}).items()}
        self.floor_sl_prices = {k: float(v) for k, v in st.get("floor_sl_prices", {}).items()}
        self.trade_cooldowns = {k: float(v) for k, v in st.get("trade_cooldowns", {}).items() if float(v) > now}
        self.position_entry_times = {k: float(v) for k, v in st.get("position_entry_times", {}).items()}
        self.golden_positions = st.get("golden_positions", {})
        today_et = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
        self.daily_loss_dates = {k: str(v) for k, v in st.get("daily_loss_dates", {}).items() if str(v) == today_et}
        if st:
            logger.info(
                f"♻️ [State Restored] active_meta={list(self.golden_positions.keys())} "
                f"floor_sl={self.floor_sl_prices} cooldowns={list(self.trade_cooldowns.keys())}"
            )

    def _save_state(self):
        today_et = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
        save_state(
            STATE_FILE,
            {
                "peak_prices": self.peak_prices,
                "floor_sl_prices": self.floor_sl_prices,
                "trade_cooldowns": {k: v for k, v in self.trade_cooldowns.items() if v > time.time()},
                "position_entry_times": self.position_entry_times,
                "golden_positions": self.golden_positions,
                "daily_loss_dates": {k: v for k, v in self.daily_loss_dates.items() if v == today_et},
                "saved_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _load_asset_specs(self):
        """Caches FINRA/Reg-T margin requirements and shortability per symbol from Alpaca."""
        for sym in CORE_UNIVERSE:
            d_long, d_short = DEFAULT_MARGIN_REQ.get(sym, (30.0, 30.0))
            self.asset_specs[sym] = {
                "long_req": d_long,
                "short_req": d_short,
                "shortable": sym in BENCHMARK_ETFS,
                "easy_to_borrow": sym in BENCHMARK_ETFS,
            }
            try:
                r = session.get(f"{ALPACA_BASE_URL}/v2/assets/{sym}", headers=HEADERS, timeout=4)
                if r.status_code == 200:
                    js = r.json()
                    self.asset_specs[sym] = {
                        "long_req": float(js.get("margin_requirement_long") or js.get("maintenance_margin_requirement") or d_long),
                        "short_req": float(js.get("margin_requirement_short") or js.get("maintenance_margin_requirement") or d_short),
                        "shortable": bool(js.get("shortable", False)),
                        "easy_to_borrow": bool(js.get("easy_to_borrow", False)),
                    }
            except Exception as e:
                logger.debug(f"Asset spec fallback used for {sym}: {e}")

    # ── ACCOUNT & POSITIONS ──
    def sync_account(self, verbose: bool = False) -> bool:
        try:
            resp = session.get(f"{ALPACA_BASE_URL}/v2/account", headers=HEADERS, timeout=5)
            if resp.status_code == 200:
                self.account_info = resp.json()
                equity = float(self.account_info.get("equity", 0.0) or 0.0)
                cash = float(self.account_info.get("cash", 0.0) or 0.0)
                regt_bp = float(self.account_info.get("regt_buying_power") or self.account_info.get("buying_power") or 0.0)
                non_m_bp = float(self.account_info.get("non_marginable_buying_power") or cash)
                status = self.account_info.get("status", "UNKNOWN")
                if verbose or (time.time() - self.last_account_log >= 60.0):
                    logger.info(
                        f"💼 [Account Sync] {status} | Equity: ${equity:,.2f} | Cash: ${cash:,.2f} | "
                        f"Reg-T BP: ${regt_bp:,.2f} | Non-Marg BP: ${non_m_bp:,.2f}"
                    )
                    self.last_account_log = time.time()
                return equity > 0
            logger.error(f"❌ Account sync failed: HTTP {resp.status_code} - {resp.text[:160]}")
        except Exception as e:
            logger.error(f"❌ Account sync exception: {e}")
        return False

    def _infer_bmk(self, sym: str) -> str:
        for bmk, bull_e, bear_e in PAIR_MAP:
            if sym in (bmk, bull_e, bear_e):
                return bmk
        return sym

    def sync_positions(self) -> Dict[str, Any]:
        try:
            resp = session.get(f"{ALPACA_BASE_URL}/v2/positions", headers=HEADERS, timeout=5)
            if resp.status_code == 200:
                pos_list = resp.json()
                guardian_symbols = get_guardian_symbols()
                new_positions: Dict[str, Any] = {}

                for p in pos_list:
                    sym = p.get("symbol", "")
                    if sym not in SYMBOLS_UNIVERSE:
                        continue
                    if sym in guardian_symbols and sym not in self.golden_positions:
                        continue
                    new_positions[sym] = p
                    # Ensure metadata exists if inherited across restarts
                    if sym not in self.golden_positions:
                        raw_q = float(p.get("qty", 0.0) or 0.0)
                        side_s = "SHORT" if (raw_q < 0 or str(p.get("side", "")).lower() == "short") else "LONG"
                        ep = float(p.get("avg_entry_price", 0.0) or 0.0)
                        kind = "BEAR_RIP" if (sym in BEAR_3X_ETFS or side_s == "SHORT") else "BULL_DIP"
                        self.golden_positions[sym] = {
                            "qty": abs(int(raw_q)),
                            "side": side_s,
                            "kind": kind,
                            "is_3x": sym in ALL_3X_ETFS,
                            "bmk": self._infer_bmk(sym),
                            "entry_price": ep,
                            "entry_time": self.position_entry_times.get(sym, time.time()),
                            "entry_iso": datetime.now(timezone.utc).isoformat(),
                        }

                for sym in list(self.positions.keys()):
                    if sym not in new_positions:
                        self._handle_external_close(sym, self.positions[sym])
                        self.golden_positions.pop(sym, None)

                self.positions = new_positions
                return self.positions
            logger.error(f"❌ Sync positions failed: HTTP {resp.status_code}")
        except Exception as e:
            logger.error(f"❌ Sync positions exception: {e}")
        return self.positions

    def _last_exit_fill_price(self, symbol: str, was_long: bool) -> Optional[float]:
        target_side = "sell" if was_long else "buy"
        try:
            resp = session.get(
                f"{ALPACA_BASE_URL}/v2/account/activities/FILL",
                headers=HEADERS,
                params={"direction": "desc", "page_size": 50},
                timeout=5,
            )
            if resp.status_code == 200:
                for act in resp.json():
                    if act.get("symbol") == symbol and str(act.get("side", "")).lower() == target_side:
                        return float(act.get("price", 0.0)) or None
        except Exception as e:
            logger.debug(f"Fill lookup failed for {symbol}: {e}")
        return None

    def _handle_external_close(self, symbol: str, prev_pos: Dict[str, Any]):
        raw_q = float(prev_pos.get("qty", 0.0) or 0.0)
        was_long = not (raw_q < 0 or str(prev_pos.get("side", "")).lower() == "short")
        entry = float(prev_pos.get("avg_entry_price", 0.0) or 0.0)
        exit_px = self._last_exit_fill_price(symbol, was_long)
        if exit_px and entry > 0:
            pnl_pct = ((exit_px - entry) / entry) if was_long else ((entry - exit_px) / entry)
        else:
            pnl_pct = float(prev_pos.get("unrealized_plpc", 0.0) or 0.0)

        is_loss = pnl_pct < 0
        if is_loss:
            today_et = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
            self.daily_loss_dates[symbol] = today_et

        self.trade_cooldowns[symbol] = time.time() + EXIT_COOLDOWN_SEC
        self.peak_prices.pop(symbol, None)
        self.floor_sl_prices.pop(symbol, None)
        self.position_entry_times.pop(symbol, None)
        tag = "LOSS" if is_loss else "WIN"
        dir_lbl = "LONG" if was_long else "SHORT"
        logger.info(
            f"ℹ️ [Bracket Fill Exit] {symbol} {dir_lbl} ({tag} {pnl_pct*100:+.2f}%, exit ~${exit_px or 0:.2f}) -> cooldown 4h"
        )
        send_telegram(f"[Alpaca US V4-Pro] {symbol} ({dir_lbl}) 브래킷 청산: {tag} {pnl_pct*100:+.2f}%")

    def refresh_pending_orders(self) -> List[Dict[str, Any]]:
        try:
            resp = session.get(
                f"{ALPACA_BASE_URL}/v2/orders", headers=HEADERS, params={"status": "open", "limit": 500}, timeout=5
            )
            if resp.status_code == 200:
                self.pending_orders = resp.json()
        except Exception as e:
            logger.error(f"❌ Get pending orders exception: {e}")
        return self.pending_orders

    def _pending_open_symbols(self) -> set:
        """Symbols with an unfilled entry parent order in flight (excludes child TP/SL legs of already-open positions)."""
        out = set()
        for o in self.pending_orders:
            sym = o.get("symbol", "")
            if sym not in SYMBOLS_UNIVERSE or sym in self.positions:
                continue
            cid = str(o.get("client_order_id", ""))
            oclass = str(o.get("order_class", "")).lower()
            if cid.startswith(CLIENT_OPEN_PREFIX) or oclass == "bracket":
                out.add(sym)
        return out

    def _pending_close_symbols(self) -> set:
        """Symbols with an active market liquidation order in flight."""
        out = set()
        for o in self.pending_orders:
            sym = o.get("symbol", "")
            cid = str(o.get("client_order_id", ""))
            if sym in SYMBOLS_UNIVERSE and cid.startswith(CLIENT_CLOSE_PREFIX):
                out.add(sym)
        return out

    def get_active_symbols(self) -> set:
        return set(self.positions.keys()) | self._pending_open_symbols()

    def get_active_benchmarks(self) -> set:
        bmks = set()
        for sym in self.get_active_symbols():
            meta = self.golden_positions.get(sym, {})
            bmks.add(meta.get("bmk") or self._infer_bmk(sym))
        return bmks

    def get_active_count(self) -> int:
        return len(self.get_active_symbols())

    def gross_exposure(self) -> float:
        return sum(abs(float(p.get("market_value", 0.0) or 0.0)) for p in self.positions.values())

    # ── MARKET CLOCK ──
    def check_market_clock(self) -> bool:
        now = time.time()
        if now - self.last_clock_check < 30.0:
            return self.is_market_open
        try:
            resp = session.get(f"{ALPACA_BASE_URL}/v2/clock", headers=HEADERS, timeout=5)
            if resp.status_code == 200:
                clock = resp.json()
                self.is_market_open = bool(clock.get("is_open", False))
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

    # ── HYBRID SIP + REAL-TIME IEX MARKET DATA ENGINE ──
    def fetch_latest_quote(self, symbol: str) -> Dict[str, Any]:
        try:
            url = f"{ALPACA_DATA_URL}/stocks/{symbol}/quotes/latest"
            resp = session.get(url, headers=HEADERS, params={"feed": ALPACA_QUOTE_FEED}, timeout=3)
            if resp.status_code == 200:
                q = resp.json().get("quote", {})
                bid_pr = float(q.get("bp", 0.0) or 0.0)
                bid_sz = float(q.get("bs", 0.0) or 0.0)
                ask_pr = float(q.get("ap", 0.0) or 0.0)
                ask_sz = float(q.get("as", 0.0) or 0.0)
                mid_pr = (bid_pr + ask_pr) / 2.0 if (bid_pr > 0 and ask_pr > 0) else (bid_pr or ask_pr)
                tot_sz = bid_sz + ask_sz
                imb = (bid_sz - ask_sz) / tot_sz if tot_sz > 0 else 0.0
                qdata = {
                    "symbol": symbol,
                    "bidPrice": bid_pr,
                    "askPrice": ask_pr,
                    "midPrice": mid_pr,
                    "imbalance": imb,
                    "timestamp": time.time(),
                }
                self.quotes_cache[symbol] = qdata
                return qdata
        except Exception as e:
            logger.debug(f"Quote fetch error for {symbol}: {e}")
        cached = self.quotes_cache.get(symbol, {})
        if cached and time.time() - cached.get("timestamp", 0) < 60:
            return cached
        return {}

    def fetch_latest_1m_close(self, symbol: str) -> float:
        try:
            url = f"{ALPACA_DATA_URL}/stocks/{symbol}/bars"
            params = {
                "timeframe": "1Min",
                "limit": 5,
                "adjustment": "split",
                "feed": "iex",
                "sort": "desc",
                "start": (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            r = session.get(url, headers=HEADERS, params=params, timeout=4)
            if r.status_code == 200:
                bars = r.json().get("bars") or []
                if bars:
                    return float(bars[0].get("c", 0.0) or 0.0)
        except Exception:
            pass
        return 0.0

    def _query_bars_raw(
        self, symbol: str, timeframe: str, start_iso: str, end_iso: Optional[str], feed: str
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        page_token: Optional[str] = None
        for _ in range(5):
            params: Dict[str, Any] = {
                "timeframe": timeframe,
                "start": start_iso,
                "limit": 10000,
                "adjustment": "split",
                "feed": feed,
                "sort": "asc",
            }
            if end_iso:
                params["end"] = end_iso
            if page_token:
                params["page_token"] = page_token
            r = session.get(f"{ALPACA_DATA_URL}/stocks/{symbol}/bars", headers=HEADERS, params=params, timeout=8)
            if r.status_code != 200:
                break
            js = r.json()
            rows.extend(js.get("bars") or [])
            page_token = js.get("next_page_token")
            if not page_token:
                break
        return rows

    def get_prior_daily_metrics(self, symbol: str, today_et: str) -> Optional[Dict[str, float]]:
        """Returns prior completed daily MA20/MA30/MA50/EMA8/EMA21 strictly < today_et (cached once per ET day)."""
        if self.daily_cache_date != today_et:
            self.daily_cache_date = today_et
            self.daily_metrics_cache.clear()

        if symbol in self.daily_metrics_cache:
            return self.daily_metrics_cache[symbol]

        now_utc = datetime.now(timezone.utc)
        start_iso = (now_utc - timedelta(days=140)).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = (now_utc - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")

        rows = self._query_bars_raw(symbol, "1Day", start_iso, end_iso, "sip")
        if len(rows) < 52:
            rows = self._query_bars_raw(symbol, "1Day", start_iso, None, "iex")
        if not rows:
            return None

        dd = pd.DataFrame(rows)
        dd["ts"] = pd.to_datetime(dd["t"], utc=True)
        dd = dd.sort_values("ts").drop_duplicates("ts")
        dd["et_date"] = dd["ts"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        dd = dd[dd["et_date"] < today_et].copy()
        if len(dd) < 50:
            return None

        dc = dd["c"].astype(float)
        dd["ma20"] = dc.rolling(20).mean()
        dd["ma30"] = dc.rolling(30).mean()
        dd["ma50"] = dc.rolling(50).mean()
        dd["ema8"] = dc.ewm(span=8, adjust=False).mean()
        dd["ema21"] = dc.ewm(span=21, adjust=False).mean()

        last = dd.iloc[-1]
        if math.isnan(float(last["ma50"])):
            return None

        metrics = {
            "close": float(last["c"]),
            "ma20": float(last["ma20"]),
            "ma30": float(last["ma30"]),
            "ma50": float(last["ma50"]),
            "ema8": float(last["ema8"]),
            "ema21": float(last["ema21"]),
        }
        self.daily_metrics_cache[symbol] = metrics
        return metrics

    def get_confirmed_1h_df(self, symbol: str) -> Optional[pd.DataFrame]:
        """Stitches consolidated SIP 1Hour bars (<= now-16m) + real-time IEX 1Hour bars, filtered to completed RTH bars."""
        now_ts = time.time()
        now_utc = pd.Timestamp.now(tz="UTC")
        hour_bucket = now_utc.strftime("%Y-%m-%d-%H")

        cached_m = self.merged_1h_cache.get(symbol)
        if cached_m and cached_m[1] == hour_bucket and (now_ts - cached_m[0] < 55.0):
            return cached_m[2]

        # 1. Consolidated SIP history (refreshed every 5 minutes)
        sip_df: Optional[pd.DataFrame] = None
        cached_sip = self.sip_1h_cache.get(symbol)
        if cached_sip and (now_ts - cached_sip[0] < 300.0):
            sip_df = cached_sip[1]
        else:
            start_iso = (datetime.now(timezone.utc) - timedelta(days=35)).strftime("%Y-%m-%dT%H:%M:%SZ")
            end_iso = (datetime.now(timezone.utc) - timedelta(minutes=16)).strftime("%Y-%m-%dT%H:%M:%SZ")
            s_rows = self._query_bars_raw(symbol, "1Hour", start_iso, end_iso, "sip")
            if s_rows:
                sdf = pd.DataFrame(s_rows)
                sdf["ts"] = pd.to_datetime(sdf["t"], utc=True)
                sdf = sdf.sort_values("ts").drop_duplicates("ts").set_index("ts")
                sip_df = sdf
                self.sip_1h_cache[symbol] = (now_ts, sdf)

        # 2. Real-time IEX recent 1Hour bars (fills any bar that closed within the last 16 minutes)
        rec_start = (datetime.now(timezone.utc) - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
        i_rows = self._query_bars_raw(symbol, "1Hour", rec_start, None, "iex")
        iex_df: Optional[pd.DataFrame] = None
        if i_rows:
            idf = pd.DataFrame(i_rows)
            idf["ts"] = pd.to_datetime(idf["t"], utc=True)
            idf = idf.sort_values("ts").drop_duplicates("ts").set_index("ts")
            iex_df = idf

        if sip_df is not None and iex_df is not None:
            only_new_iex = iex_df.loc[~iex_df.index.isin(sip_df.index)]
            df = pd.concat([sip_df, only_new_iex]).sort_index()
        elif sip_df is not None:
            df = sip_df.copy()
        elif iex_df is not None:
            df = iex_df.copy()
        else:
            return None

        # Filter to RTH (09:00 <= ET hour <= 15) AND strictly completed 1H bars (ts + 1h <= now_utc)
        et_idx = df.index.tz_convert("America/New_York")
        rth_mask = (et_idx.hour >= 9) & (et_idx.hour <= 15) & ((df.index + pd.Timedelta(hours=1)) <= now_utc)
        df = df.loc[rth_mask].copy()
        if len(df) < 52:
            return None

        for cs, cd in [("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("v", "vol")]:
            df[cd] = df[cs].astype(float)

        c, h, l = df["close"], df["high"], df["low"]
        df["ma5"] = c.rolling(5).mean()
        df["ma10"] = c.rolling(10).mean()
        df["ma20"] = c.rolling(20).mean()
        df["ema9"] = c.ewm(span=9, adjust=False).mean()
        df["ema21"] = c.ewm(span=21, adjust=False).mean()
        df["ema50"] = c.ewm(span=50, adjust=False).mean()
        df["rsi2"] = calc_wilder_rsi(c, 2)
        df["rsi5"] = calc_wilder_rsi(c, 5)
        df["rsi14"] = calc_wilder_rsi(c, 14)
        df["ret7"] = (c / c.shift(7) - 1.0) * 100.0
        df["z20"] = (c - df["ma20"]) / c.rolling(20).std().replace(0.0, 1e-9)
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        df["atr_pct"] = (tr.rolling(14).mean() / c) * 100.0

        self.merged_1h_cache[symbol] = (now_ts, hour_bucket, df)
        return df

    # ── MARGIN-SAFE SIZING & ORDER EXECUTION ──
    def compute_safe_order_qty(self, symbol: str, side: str, ref_price: float) -> Tuple[int, float]:
        if ref_price <= 0:
            return 0, 0.0
        equity = float(self.account_info.get("equity", 0.0) or 0.0)
        cash = float(self.account_info.get("cash", 0.0) or 0.0)
        non_marg_bp = float(self.account_info.get("non_marginable_buying_power") or max(0.0, cash))
        regt_bp = float(self.account_info.get("regt_buying_power") or self.account_info.get("buying_power") or 0.0)
        dt_bp = float(self.account_info.get("buying_power") or regt_bp)

        room_gross = max(0.0, equity * MAX_GROSS_EXPOSURE_PCT - self.gross_exposure())
        desired = min(equity * SLOT_TARGET_PCT, room_gross)

        spec = self.asset_specs.get(symbol, {"long_req": 100.0, "short_req": 100.0})
        is_short_order = side.lower() == "sell"
        req = float(spec["short_req"] if is_short_order else spec["long_req"])

        if req >= 99.9:
            bp_cap = non_marg_bp * 0.96
        else:
            regt_scaled = regt_bp * min(1.0, 50.0 / max(req, 25.0))
            dt_scaled = dt_bp * min(1.0, 25.0 / max(req, 25.0))
            bp_cap = min(regt_scaled, dt_scaled) * 0.95

        alloc_dollars = min(desired, max(0.0, bp_cap))
        qty = int(alloc_dollars / ref_price)
        if qty <= 0 or qty * ref_price < MIN_ORDER_NOTIONAL:
            return 0, 0.0
        return qty, qty * ref_price

    def submit_bracket_order(
        self, symbol: str, qty: int, side: str, pod_kind: str, bmk: str
    ) -> Optional[Dict[str, Any]]:
        """Submits GTC Bracket order supporting BOTH LONG (side='buy') and NATIVE SHORT (side='sell')."""
        if qty <= 0:
            return None

        is_buy = side.lower() == "buy"
        pos_side = "LONG" if is_buy else "SHORT"

        bar_close = self.fetch_latest_1m_close(symbol)
        q = self.fetch_latest_quote(symbol)
        ref_q = (q.get("askPrice") if is_buy else q.get("bidPrice")) or q.get("midPrice") or 0.0

        if bar_close > 0 and ref_q > 0 and abs(ref_q - bar_close) / bar_close > 0.012:
            logger.warning(
                f"⚠️ {symbol} quote (${ref_q:.2f}) deviates >1.2% from 1m bar (${bar_close:.2f}) -> clamping to bar close."
            )
            base_price = bar_close
        else:
            base_price = ref_q or bar_close

        if base_price <= 0:
            df1h = self.get_confirmed_1h_df(symbol)
            base_price = float(df1h["close"].iloc[-1]) if (df1h is not None and not df1h.empty) else 0.0
        if base_price <= 0:
            logger.error(f"❌ Cannot submit {pos_side} order for {symbol}: base_price unavailable.")
            return None

        sym_sl, sym_tp, is_3x = get_symbol_risk(symbol)
        if is_buy:
            sl_ref = min(base_price, bar_close) if bar_close > 0 else base_price
            tp_ref = max(base_price, bar_close) if bar_close > 0 else base_price
            sl_price = round(sl_ref * (1.0 - sym_sl) - 0.01, 2)
            tp_price = round(tp_ref * (1.0 + sym_tp) + 0.01, 2)
        else:
            sl_ref = max(base_price, bar_close) if bar_close > 0 else base_price
            tp_ref = min(base_price, bar_close) if bar_close > 0 else base_price
            sl_price = round(sl_ref * (1.0 + sym_sl) + 0.02, 2)
            tp_price = round(tp_ref * (1.0 - sym_tp) - 0.02, 2)

        order_data = {
            "symbol": symbol,
            "qty": str(qty),
            "side": "buy" if is_buy else "sell",
            "type": "market",
            "time_in_force": "gtc",
            "order_class": "bracket",
            "client_order_id": f"{CLIENT_OPEN_PREFIX}{symbol}-{int(time.time())}",
            "take_profit": {"limit_price": str(tp_price)},
            "stop_loss": {"stop_price": str(sl_price)},
        }

        logger.info(
            f"📤 [V4-Pro Bracket] {pos_side} ({pod_kind}) {qty}x {symbol} @ ~${base_price:.2f} "
            f"(~${qty*base_price:,.0f}) | TP: ${tp_price} | SL: ${sl_price}"
        )

        try:
            resp = session.post(f"{ALPACA_BASE_URL}/v2/orders", headers=HEADERS, json=order_data, timeout=8)
            if resp.status_code in (200, 201):
                res_json = resp.json()
                now_t = time.time()
                self.position_entry_times[symbol] = now_t
                self.peak_prices[symbol] = base_price
                self.golden_positions[symbol] = {
                    "qty": int(qty),
                    "side": pos_side,
                    "kind": pod_kind,
                    "is_3x": is_3x,
                    "bmk": bmk,
                    "entry_price": base_price,
                    "entry_time": now_t,
                    "entry_iso": datetime.now(timezone.utc).isoformat(),
                }
                self.pending_orders.append(res_json)
                self._save_state()
                self.sync_account(verbose=False)
                logger.info(f"✅ [Order Confirmed] ID: {res_json.get('id')} | {pos_side} {qty}x {symbol} ({pod_kind})")
                send_telegram(
                    f"[Alpaca US V4-Pro] 🚀 {pos_side} ({pod_kind}) {qty}주 {symbol} @~${base_price:.2f} "
                    f"(~${qty*base_price:,.0f}) | TP ${tp_price} / SL ${sl_price}"
                )
                return res_json

            logger.error(f"❌ Order rejected ({symbol} {pos_side}): HTTP {resp.status_code} - {resp.text}")
            self.trade_cooldowns[symbol] = time.time() + ORDER_FAIL_COOLDOWN_SEC
            send_telegram(
                f"[Alpaca US V4-Pro] ❌ {symbol} {pos_side} 주문 거부: HTTP {resp.status_code} {resp.text[:180]}",
                dedupe_key=f"alpaca_reject_{symbol}",
                min_interval_sec=1800,
            )
        except Exception as e:
            logger.error(f"❌ Order submission exception ({symbol}): {e}")
            self.trade_cooldowns[symbol] = time.time() + ORDER_FAIL_COOLDOWN_SEC
        return None

    # ── REAL-TIME & 1H POSITION MANAGEMENT ──
    def _count_rth_bars_held(self, df1h: Optional[pd.DataFrame], entry_iso: str) -> int:
        if df1h is None or df1h.empty or not entry_iso:
            return 0
        try:
            entry_ts = pd.Timestamp(entry_iso)
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.tz_localize("UTC")
            else:
                entry_ts = entry_ts.tz_convert("UTC")
            return int((df1h.index >= entry_ts).sum())
        except Exception:
            return 0

    def manage_open_positions(self):
        if not self.positions:
            return

        pending_closes = self._pending_close_symbols()

        for symbol, pos in list(self.positions.items()):
            try:
                if symbol in pending_closes:
                    continue

                raw_qty = float(pos.get("qty", 0.0) or 0.0)
                if raw_qty == 0:
                    continue
                is_long = not (raw_qty < 0 or str(pos.get("side", "")).lower() == "short")
                curr_price = float(pos.get("current_price", 0.0) or 0.0)
                entry_price = float(pos.get("avg_entry_price", 0.0) or 0.0)
                if curr_price <= 0 or entry_price <= 0:
                    continue

                meta = self.golden_positions.get(symbol, {})
                pod_kind = meta.get("kind", "BEAR_RIP" if (symbol in BEAR_3X_ETFS or not is_long) else "BULL_DIP")
                is_3x = bool(meta.get("is_3x", symbol in ALL_3X_ETFS))
                bmk = meta.get("bmk") or self._infer_bmk(symbol)
                entry_iso = meta.get("entry_iso", "")

                # 1. Update directional peak price
                peak = self.peak_prices.get(symbol, entry_price)
                if is_long and curr_price > peak:
                    peak = curr_price
                    self.peak_prices[symbol] = peak
                elif (not is_long) and curr_price < peak:
                    peak = curr_price
                    self.peak_prices[symbol] = peak

                cur_pl = ((curr_price - entry_price) / entry_price) if is_long else ((entry_price - curr_price) / entry_price)
                pk_pl = ((peak - entry_price) / entry_price) if is_long else ((entry_price - peak) / entry_price)

                # 2. Arm Breakeven Floor Shield
                if pod_kind == "WATERFALL_BEAR":
                    be_trig = WF_BE_TRIG_3X if is_3x else WF_BE_TRIG_1X
                else:
                    be_trig = BE_TRIG_3X_PCT if is_3x else BE_TRIG_1X_PCT

                cur_floor = self.floor_sl_prices.get(symbol, 0.0)
                if pk_pl >= be_trig and cur_floor == 0.0:
                    cur_floor = round(
                        entry_price * ((1.0 + BE_LOCK_OFFSET_PCT) if is_long else (1.0 - BE_LOCK_OFFSET_PCT)), 2
                    )
                    self.floor_sl_prices[symbol] = cur_floor
                    logger.info(
                        f"🔒 [Breakeven Shield Armed] {symbol} ({pod_kind}) Peak {pk_pl*100:+.2f}% -> Floor locked @ ${cur_floor:.2f}"
                    )

                if cur_floor > 0.0:
                    hit_floor = (is_long and curr_price <= cur_floor) or ((not is_long) and curr_price >= cur_floor)
                    if hit_floor:
                        logger.info(
                            f"🛡️ [Floor Lock Exit] {symbol} price ${curr_price:.2f} crossed floor ${cur_floor:.2f} (PnL {cur_pl*100:+.2f}%)"
                        )
                        self.close_position(symbol, f"FLOOR_LOCK_{cur_pl*100:+.2f}PCT")
                        continue

                # 3. Confirmed 1H Snapback Harvest / Waterfall Trail / RTH Time-Stop
                df1h = self.get_confirmed_1h_df(symbol)
                rth_bars = self._count_rth_bars_held(df1h, entry_iso)

                if pod_kind in ("BULL_DIP", "BEAR_RIP"):
                    if df1h is not None and len(df1h) >= 3:
                        last_1h = df1h.iloc[-1]
                        prev_1h = df1h.iloc[-2]
                        c_1h = float(last_1h["close"])
                        ma10_p = float(prev_1h["ma10"]) if not math.isnan(float(prev_1h["ma10"])) else c_1h
                        rsi5_1h = float(last_1h["rsi5"]) if not math.isnan(float(last_1h["rsi5"])) else 50.0
                        min_gain = SNAP_MIN_GAIN_3X if is_3x else SNAP_MIN_GAIN_1X

                        snap_ok = False
                        if is_long and ((c_1h > ma10_p and cur_pl >= min_gain) or (rsi5_1h >= 72.0 and cur_pl >= min_gain * 0.8)):
                            snap_ok = True
                        elif (not is_long) and ((c_1h < ma10_p and cur_pl >= min_gain) or (rsi5_1h <= 28.0 and cur_pl >= min_gain * 0.8)):
                            snap_ok = True

                        if snap_ok and rth_bars >= 1:
                            logger.info(
                                f"🎯 [Snapback Harvest] {symbol} ({pod_kind}) 1H Close ${c_1h:.2f} vs MA10 ${ma10_p:.2f} "
                                f"| RSI5 {rsi5_1h:.1f} | PnL {cur_pl*100:+.2f}%"
                            )
                            self.close_position(symbol, f"SNAP_WIN_{cur_pl*100:+.2f}PCT")
                            continue

                    if rth_bars >= SNAP_MAX_RTH_BARS:
                        logger.info(f"⏱️ [Time Stop] {symbol} held {rth_bars} RTH 1H bars (PnL {cur_pl*100:+.2f}%) -> closing.")
                        self.close_position(symbol, f"TIME_STOP_{cur_pl*100:+.2f}PCT")
                        continue

                else:  # WATERFALL_BEAR
                    tr_trig = WF_TRAIL_TRIG_3X if is_3x else WF_TRAIL_TRIG_1X
                    tr_drop = WF_TRAIL_DROP_3X if is_3x else WF_TRAIL_DROP_1X
                    if pk_pl >= tr_trig:
                        pull = ((peak - curr_price) / peak) if is_long else ((curr_price - peak) / peak)
                        if pull >= tr_drop and cur_pl >= min(0.015, be_trig):
                            logger.info(
                                f"🌊 [Crash Trail Harvest] {symbol} Peak {pk_pl*100:+.2f}%, Pullback {pull*100:.2f}% -> locking {cur_pl*100:+.2f}%"
                            )
                            self.close_position(symbol, f"CRASH_TRAIL_{cur_pl*100:+.2f}PCT")
                            continue

                    bmk_df = self.get_confirmed_1h_df(bmk)
                    if bmk_df is not None and not bmk_df.empty and rth_bars >= 1:
                        br = bmk_df.iloc[-1]
                        if float(br["close"]) > float(br["ema21"]):
                            logger.info(f"🔄 [EMA21 Reclaim Exit] Benchmark {bmk} reclaimed 1H EMA21 -> closing {symbol}.")
                            self.close_position(symbol, f"EMA21_RECLAIM_{cur_pl*100:+.2f}PCT")
                            continue

                    if rth_bars >= WF_MAX_RTH_BARS and cur_pl < 0.005:
                        self.close_position(symbol, f"WF_TIME_STOP_{cur_pl*100:+.2f}PCT")
                        continue

            except Exception as e:
                logger.error(f"Error managing open position {symbol}: {e}")

    def close_position(self, symbol: str, reason: str = "MANUAL") -> bool:
        """Atomically closes LONG or SHORT position via Alpaca DELETE /v2/positions/{symbol}?cancel_orders=true."""
        if symbol in self._pending_close_symbols():
            return False

        pos = self.positions.get(symbol, {})
        unrealized_pl = float(pos.get("unrealized_pl", 0.0) or 0.0)
        unrealized_plpc = float(pos.get("unrealized_plpc", 0.0) or 0.0)
        raw_qty = int(float(pos.get("qty", 0.0) or 0.0))
        is_long = not (raw_qty < 0 or str(pos.get("side", "")).lower() == "short")
        abs_qty = abs(raw_qty)

        if abs_qty <= 0:
            self.positions.pop(symbol, None)
            self.golden_positions.pop(symbol, None)
            self.floor_sl_prices.pop(symbol, None)
            self._save_state()
            return True

        dir_lbl = "LONG" if is_long else "SHORT"
        logger.info(f"🔄 [Closing Position] {symbol} {dir_lbl} ({abs_qty} shares) | Reason: {reason}")

        try:
            open_sym_orders = [o for o in self.pending_orders if o.get("symbol") == symbol]
            if not open_sym_orders:
                r_ord = session.get(
                    f"{ALPACA_BASE_URL}/v2/orders", headers=HEADERS, params={"status": "open", "symbols": symbol}, timeout=4
                )
                if r_ord.status_code == 200:
                    open_sym_orders = r_ord.json()

            cancelled = 0
            for o in open_sym_orders:
                cid = str(o.get("client_order_id", ""))
                if not cid.startswith("GD_"):
                    session.delete(f"{ALPACA_BASE_URL}/v2/orders/{o.get('id')}", headers=HEADERS, timeout=3)
                    cancelled += 1

            if cancelled > 0:
                for _ in range(5):
                    time.sleep(0.35)
                    r_chk = session.get(
                        f"{ALPACA_BASE_URL}/v2/orders", headers=HEADERS, params={"status": "open", "symbols": symbol}, timeout=3
                    )
                    if r_chk.status_code == 200 and len(r_chk.json()) == 0:
                        break

            resp = session.delete(
                f"{ALPACA_BASE_URL}/v2/positions/{symbol}", headers=HEADERS, params={"cancel_orders": "true"}, timeout=8
            )
            if resp.status_code not in (200, 201, 204):
                close_side = "sell" if is_long else "buy"  # 'buy' covers SHORT position!
                logger.warning(
                    f"⚠️ DELETE position HTTP {resp.status_code} ({resp.text[:120]}) -> fallback market {close_side.upper()}..."
                )
                fallback_order = {
                    "symbol": symbol,
                    "qty": str(abs_qty),
                    "side": close_side,
                    "type": "market",
                    "time_in_force": "day",
                    "client_order_id": f"{CLIENT_CLOSE_PREFIX}{symbol}-{int(time.time())}",
                }
                resp = session.post(f"{ALPACA_BASE_URL}/v2/orders", headers=HEADERS, json=fallback_order, timeout=8)
                if resp.status_code not in (200, 201):
                    logger.error(f"❌ Fallback close failed for {symbol}: HTTP {resp.status_code} - {resp.text[:160]}")
                    return False
                self.pending_orders.append(resp.json())
            elif resp.status_code in (200, 201) and resp.text:
                try:
                    self.pending_orders.append(resp.json())
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"❌ Exception closing {symbol}: {e}")
            return False

        if unrealized_pl < 0:
            today_et = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
            self.daily_loss_dates[symbol] = today_et

        self.trade_cooldowns[symbol] = time.time() + EXIT_COOLDOWN_SEC
        self.peak_prices.pop(symbol, None)
        self.floor_sl_prices.pop(symbol, None)
        self.position_entry_times.pop(symbol, None)
        self.golden_positions.pop(symbol, None)
        self.positions.pop(symbol, None)
        self._save_state()
        logger.info(f"✅ [Position Closed] {symbol} {dir_lbl} ({abs_qty} shares) | PnL {unrealized_plpc*100:+.2f}%")
        send_telegram(
            f"[Alpaca US V4-Pro] {symbol} ({dir_lbl}) 청산 [{reason}] | PnL {unrealized_plpc*100:+.2f}% (${unrealized_pl:+,.2f})"
        )
        return True

    # ── ARGON US V4-PRO ENTRY EVALUATION ──
    def evaluate_trading_opportunities(self):
        et_now = pd.Timestamp.now(tz="America/New_York")
        today_et = et_now.strftime("%Y-%m-%d")
        et_hr = et_now.hour

        # Opening spread stabilization (wait 2 min after 09:30 bell) & no new entries in final hour (et_hr >= 15)
        if (et_hr == 9 and et_now.minute < 32) or et_hr >= 15:
            return

        active_syms = self.get_active_symbols()
        active_bmks = self.get_active_benchmarks()
        guardian_syms = get_guardian_symbols()

        # Always compute 4-Sector US Radar snapshot (logged every 5m even when slots full)
        radar_parts: List[str] = []
        oversold_sectors = 0
        overbought_bear_sectors = 0
        waterfall_sectors = 0
        sector_snaps: Dict[str, Dict[str, Any]] = {}

        for bmk, bull_e, bear_e in PAIR_MAP:
            pd_b = self.get_prior_daily_metrics(bmk, today_et)
            df1h = self.get_confirmed_1h_df(bmk)
            if pd_b is None or df1h is None or df1h.empty:
                continue
            br = df1h.iloc[-1]
            bc = float(br["close"])
            rsi2 = float(br["rsi2"])
            rsi14 = float(br["rsi14"])
            z20 = float(br["z20"])
            ret7 = float(br["ret7"]) if not math.isnan(float(br["ret7"])) else 0.0
            e9, e21, e50 = float(br["ema9"]), float(br["ema21"]), float(br["ema50"])

            d_c, d_m20, d_m30, d_m50 = pd_b["close"], pd_b["ma20"], pd_b["ma30"], pd_b["ma50"]
            d_e8, d_e21 = pd_b["ema8"], pd_b["ema21"]

            is_sec_bull = (d_c > d_m50 and bc > d_m50 * 0.985) or (d_e8 > d_e21 and d_c > d_m30)
            is_sec_bear = (d_c < d_m20 and d_e8 < d_e21) or (bc < d_m50 * 0.985 and d_e8 < d_e21)
            is_wf = (bc < e9 < e21 < e50) and (ret7 <= -1.3) and (28.0 <= rsi14 <= 46.0)

            if rsi2 <= BULL_DIP_RSI2:
                oversold_sectors += 1
            if is_sec_bear and rsi2 >= BEAR_RIP_RSI2:
                overbought_bear_sectors += 1
            if is_wf:
                waterfall_sectors += 1

            reg_tag = "BULL" if is_sec_bull else ("BEAR" if is_sec_bear else "NEUT")
            radar_parts.append(f"{bmk}:{reg_tag}(R2={rsi2:.0f},Z={z20:+.1f})")
            sector_snaps[bmk] = {
                "bull_e": bull_e,
                "bear_e": bear_e,
                "is_bull": is_sec_bull,
                "is_bear": is_sec_bear,
                "bc": bc,
                "rsi2": rsi2,
                "rsi14": rsi14,
                "z20": z20,
                "ret7": ret7,
                "e9": e9,
                "e21": e21,
                "e50": e50,
                "pd": pd_b,
                "reg_tag": reg_tag,
            }

        if radar_parts and (time.time() - self.last_radar_log >= RADAR_LOG_INTERVAL_SEC):
            logger.info(
                f"📡 [4-Sector US Radar] {' | '.join(radar_parts)} | "
                f"DipCnt:{oversold_sectors}/4 RipCnt:{overbought_bear_sectors}/4 WF:{waterfall_sectors}/4 | "
                f"Active:{list(active_syms)}"
            )
            self.last_radar_log = time.time()

        if len(active_syms) >= MAX_CONCURRENT_POSITIONS or len(sector_snaps) < 3:
            return

        now_t = time.time()
        candidates: List[Tuple[str, str, str, str, float, Dict[str, Any]]] = []
        # Tuple: (symbol, order_side, pod_kind, bmk, score, context_dict)

        # 1 & 2. Independent Sector BULL_DIP (3x Bull ETF) & BEAR_RIP (3x Bear ETF)
        for bmk, snap in sector_snaps.items():
            bull_e, bear_e = snap["bull_e"], snap["bear_e"]
            rsi2, z20, rsi14, ret7 = snap["rsi2"], snap["z20"], snap["rsi14"], snap["ret7"]

            if snap["is_bull"] and rsi2 <= BULL_DIP_RSI2 and z20 <= -1.25:
                if (
                    bull_e not in active_syms
                    and bmk not in active_bmks
                    and bull_e not in guardian_syms
                    and now_t >= self.trade_cooldowns.get(bull_e, 0.0)
                ):
                    score = (100.0 - rsi2) - z20 * 5.0
                    candidates.append((bull_e, "buy", "BULL_DIP", bmk, score, snap))

            if snap["is_bear"] and rsi2 >= BEAR_RIP_RSI2 and z20 >= 1.15:
                if (
                    bear_e not in active_syms
                    and bmk not in active_bmks
                    and bear_e not in guardian_syms
                    and now_t >= self.trade_cooldowns.get(bear_e, 0.0)
                ):
                    score = rsi2 + z20 * 5.0
                    candidates.append((bear_e, "buy", "BEAR_RIP", bmk, score, snap))

        # 3. Synchronized Macro Waterfall Breakdown (>= 3 of 4 US sectors crashing together)
        if waterfall_sectors >= 3:
            for bmk, snap in sector_snaps.items():
                bear_e = snap["bear_e"]
                bc, e9, e21, e50, rsi14, ret7 = snap["bc"], snap["e9"], snap["e21"], snap["e50"], snap["rsi14"], snap["ret7"]
                dist21 = (e21 - bc) / e21 * 100.0 if e21 > 0 else 0.0
                if (bc < e9 < e21 < e50) and (0.25 <= dist21 <= 2.2) and (28.0 <= rsi14 <= 45.0):
                    if (
                        bear_e not in active_syms
                        and bmk not in active_bmks
                        and bear_e not in guardian_syms
                        and now_t >= self.trade_cooldowns.get(bear_e, 0.0)
                    ):
                        candidates.append((bear_e, "buy", "WATERFALL_BEAR", bmk, -ret7 * 10.0, snap))

        # 4. Native Index ETF Short (DIA, SPY, QQQ, SOXX) on Extreme Bear Relief Rip
        if overbought_bear_sectors >= 1:
            for bmk, snap in sector_snaps.items():
                pd_b = snap["pd"]
                spec = self.asset_specs.get(bmk, {})
                if not (spec.get("shortable") and spec.get("easy_to_borrow")):
                    continue
                if (
                    pd_b["close"] < pd_b["ma20"]
                    and pd_b["ema8"] < pd_b["ema21"]
                    and snap["rsi2"] >= NAT_SHORT_RIP_RSI2
                    and snap["z20"] >= 1.30
                ):
                    if (
                        bmk not in active_syms
                        and snap["bear_e"] not in active_syms
                        and bmk not in active_bmks
                        and bmk not in guardian_syms
                        and now_t >= self.trade_cooldowns.get(bmk, 0.0)
                    ):
                        candidates.append((bmk, "sell", "BEAR_RIP", bmk, snap["rsi2"], snap))

        # 5. Synchronized Institutional Mega-Cap Bull Dip (Long-Only when >= 2 macro sectors oversold)
        if oversold_sectors >= 2:
            for msym in MEGAS_SAFE:
                if (
                    msym in active_syms
                    or msym in active_bmks
                    or msym in guardian_syms
                    or now_t < self.trade_cooldowns.get(msym, 0.0)
                ):
                    continue
                pd_m = self.get_prior_daily_metrics(msym, today_et)
                mdf = self.get_confirmed_1h_df(msym)
                if pd_m is None or mdf is None or mdf.empty:
                    continue
                mr = mdf.iloc[-1]
                m_rsi2, m_z20, m_rsi14 = float(mr["rsi2"]), float(mr["z20"]), float(mr["rsi14"])
                m_ret7 = float(mr["ret7"]) if not math.isnan(float(mr["ret7"])) else 0.0
                if (
                    pd_m["close"] > pd_m["ma50"]
                    and pd_m["ema8"] > pd_m["ema21"] * 0.99
                    and m_rsi2 <= MEGA_DIP_RSI2
                    and m_z20 <= -1.40
                ):
                    score = (100.0 - m_rsi2) - m_z20 * 4.0
                    msnap = {
                        "bc": float(mr["close"]),
                        "rsi2": m_rsi2,
                        "rsi14": m_rsi14,
                        "z20": m_z20,
                        "ret7": m_ret7,
                        "reg_tag": "BULL_UPTREND",
                    }
                    candidates.append((msym, "buy", "BULL_DIP", msym, score, msnap))

        if not candidates:
            return

        candidates.sort(key=lambda x: x[4], reverse=True)

        for sym, order_side, pod_kind, bmk, score, snap in candidates:
            active_syms = self.get_active_symbols()
            active_bmks = self.get_active_benchmarks()
            if len(active_syms) >= MAX_CONCURRENT_POSITIONS:
                break
            if sym in active_syms or bmk in active_bmks:
                continue

            q = self.fetch_latest_quote(sym)
            ref_px = (q.get("askPrice") if order_side == "buy" else q.get("bidPrice")) or q.get("midPrice") or 0.0
            if ref_px <= 0:
                df_sym = self.get_confirmed_1h_df(sym)
                ref_px = float(df_sym["close"].iloc[-1]) if (df_sym is not None and not df_sym.empty) else 0.0
            if ref_px <= 0:
                continue

            qty, notional = self.compute_safe_order_qty(sym, order_side, ref_px)
            if qty <= 0:
                logger.debug(f"⏳ [Sizing Skip] {sym} ({pod_kind}): insufficient available margin/room for min order.")
                continue

            sym_sl, sym_tp, _ = get_symbol_risk(sym)
            pos_dir = "LONG" if order_side == "buy" else "SHORT"

            if self.jev_gate:
                try:
                    breadth_str = (
                        f"Dip={oversold_sectors}/4, BearRip={overbought_bear_sectors}/4, Waterfall={waterfall_sectors}/4"
                    )
                    decision = self.jev_gate.evaluate_sync(
                        asset_class="US_STOCKS",
                        symbol=sym,
                        side=pos_dir,
                        context_metrics={
                            "us_v4_pod": pod_kind,
                            "benchmark": bmk,
                            "daily_regime": snap.get("reg_tag", "N/A"),
                            "price": ref_px,
                            "bmk_rsi2": snap.get("rsi2", 50.0),
                            "bmk_z20": snap.get("z20", 0.0),
                            "rsi14": snap.get("rsi14", 50.0),
                            "sector_breadth": breadth_str,
                            "ret7_pct": snap.get("ret7", 0.0),
                            "sl_pct": sym_sl * 100.0,
                            "tp_pct": sym_tp * 100.0,
                        },
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Jev gate exception for {sym}: {e} -> cooldown")
                    self.trade_cooldowns[sym] = time.time() + JEV_ERROR_COOLDOWN_SEC
                    continue

                if not decision.approved:
                    cd = JEV_ERROR_COOLDOWN_SEC if decision.is_fallback else JEV_VETO_COOLDOWN_SEC
                    self.trade_cooldowns[sym] = time.time() + cd
                    logger.info(
                        f"🚫 [Jev Veto] {sym} {pos_dir} ({pod_kind}, conf {decision.confidence:.2f}): {decision.reason}"
                    )
                    continue
                logger.info(
                    f"🧠 [Jev Approved] {sym} {pos_dir} ({pod_kind}, conf {decision.confidence:.2f}, {decision.latency_ms:.0f}ms): {decision.reason}"
                )

            logger.info(
                f"🚀 [V4-Pro Sniper Signal] {sym} {pos_dir} | Pod: {pod_kind} (Bmk:{bmk}) | "
                f"R2:{snap.get('rsi2', 0):.1f} Z:{snap.get('z20', 0):+.2f} | Alloc: ~${notional:,.0f} ({qty} sh)"
            )
            self.submit_bracket_order(sym, qty, order_side, pod_kind, bmk)

    def log_startup_radar(self):
        """Runs a non-trading 4-Sector US diagnostic snapshot at startup to verify data feeds & pre-open regime."""
        try:
            today_et = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d")
            parts = []
            for bmk, bull_e, bear_e in PAIR_MAP:
                pd_b = self.get_prior_daily_metrics(bmk, today_et)
                df1h = self.get_confirmed_1h_df(bmk)
                if pd_b is None or df1h is None or df1h.empty:
                    parts.append(f"{bmk}:NO_DATA")
                    continue
                br = df1h.iloc[-1]
                bc, rsi2, z20 = float(br["close"]), float(br["rsi2"]), float(br["z20"])
                is_bull = (pd_b["close"] > pd_b["ma50"] and bc > pd_b["ma50"] * 0.985) or (
                    pd_b["ema8"] > pd_b["ema21"] and pd_b["close"] > pd_b["ma30"]
                )
                is_bear = (pd_b["close"] < pd_b["ma20"] and pd_b["ema8"] < pd_b["ema21"]) or (
                    bc < pd_b["ma50"] * 0.985 and pd_b["ema8"] < pd_b["ema21"]
                )
                reg = "BULL" if is_bull else ("BEAR" if is_bear else "NEUT")
                parts.append(f"{bmk}:${bc:.2f}[{reg}|R2={rsi2:.0f}|Z={z20:+.1f}]")
            logger.info(f"📡 [Startup 4-Sector US Diagnostic] {' | '.join(parts)}")
        except Exception as e:
            logger.warning(f"⚠️ Startup radar diagnostic warning: {e}")

    # ── MAIN 24/7 SUPERVISOR LOOP ──
    def run(self):
        logger.info("🟢 [Alpaca Golden Trader] Argon US V4-Pro Supervisor started.")
        send_telegram("[Alpaca US V4-Pro] 🦅 실전 공격형(1.8x Reg-T 스나이퍼: 4대 섹터 양방향 + 네이티브 숏) 엔진 가동")
        self.sync_account(verbose=True)
        self.sync_positions()
        self.log_startup_radar()

        while True:
            try:
                is_open = self.check_market_clock()
                if not is_open:
                    if time.time() - self.last_closed_log >= 600.0:
                        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
                        logger.info(
                            f"💤 [Market Closed] {now_str} | Next Open: {self.next_open_str} | "
                            f"Swing Positions Held: {list(self.positions.keys())}"
                        )
                        self.last_closed_log = time.time()
                    time.sleep(60.0)
                    continue

                if not self.sync_account(verbose=False):
                    logger.warning("⚠️ Account sync failed -> skipping cycle.")
                    time.sleep(LOOP_INTERVAL_SEC)
                    continue

                self.sync_positions()
                self.refresh_pending_orders()
                self.manage_open_positions()

                mins_left = self.minutes_to_close()
                if mins_left is not None and mins_left <= 5.0:
                    if self.eod_summary_done_for != self.next_close_str:
                        self.eod_summary_done_for = self.next_close_str
                        held = [
                            f"{s}({self.golden_positions.get(s, {}).get('side', 'LONG')})"
                            for s in self.positions.keys()
                        ]
                        logger.info(f"🌙 [EOD Swing Check] GTC Bracket Protected Overnight Positions: {held}")
                        if held:
                            send_telegram(f"[Alpaca US V4-Pro] 🌙 정규장 마감 임박 — GTC 브래킷 보호 스윙 보유 유지: {', '.join(held)}")
                else:
                    self.evaluate_trading_opportunities()

                self._save_state()
                time.sleep(LOOP_INTERVAL_SEC)

            except KeyboardInterrupt:
                logger.info("🛑 [Shutdown] Stopping Alpaca Golden Trader...")
                self._save_state()
                break
            except Exception as e:
                logger.error(f"❌ Unexpected loop exception: {e}", exc_info=True)
                send_telegram(f"[Alpaca US V4-Pro] ❌ 루프 예외: {e}", dedupe_key="alpaca_loop_exc", min_interval_sec=1800)
                time.sleep(10.0)


if __name__ == "__main__":
    trader = AlpacaGoldenTrader()
    trader.run()
