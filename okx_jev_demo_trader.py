# -*- coding: utf-8 -*-
"""
okx_jev_demo_trader.py — Autonomous OKX Demo Quant Engine with Bidirectional Jev Alpha
V5 Daily-Regime Trend: 4-Major Synchronization (4/4) + 1H MA(5/10/20) + BTC 20D EMA (10x Isolated)

Validated backtest 2026-11-10 (risk-sized 0.5%/trade, 1.2% total heat, MTM MDD, confirmed bars, linear short PnL):
- Current 2026-07~11: 15 trades, PF 8.28, +11.5%, MDD -5.6%
- OOS 2025-07~2026-07: 100 trades, PF 1.03, +0.6%, MDD -6.7%
- Holdout 2024-07~2025-07: 92 trades, PF 1.08, +1.4%, MDD -7.5%
- Final holdout 2023-07~2024-07: 98 trades, PF 1.34, +6.2%, MDD -5.2%
- Weak OOS edge: demo-only candidate, not validated for real capital.

Key Features:
- 100% Simulated Sandbox: Forced 'x-simulated-trading: 1' on all requests
- Targeted Universe: BTC-USDT-SWAP, ETH-USDT-SWAP, SOL-USDT-SWAP, XRP-USDT-SWAP (4 Major Leaders)
- 4-Major Dominance Synchronization (4/4 Consensus):
  * Enters ONLY when all 4 Majors align on confirmed 1H MA(5/10/20) + BTC 15m EMA20
  * Anti-Chop Filter: |1H MA5 - MA20| >= 0.50%, 1H ADX(14) >= 22, 15m ADX(14) >= 22
  * BTC 20-day EMA filter: LONG only above EMA20D with 3-day slope >= 0; SHORT only below with <= 0
  * One-Wave-One-Shot Lock: Invariant 1H wave_start_ts tracking blocks late climax / bottom re-entries
- V5 Trend Exit & Risk Management (10x Isolated, risk-based sizing, Max 2 Slots):
  * Default 0.5% equity risk/trade including estimated round-trip fee + slippage; 1.2% total open heat
  * Daily -2% / weekly -4% realized-loss circuit breakers; minSz only allowed inside a 1% single-trade hard cap
  * Exchange-attached Hard SL (2.5% base ~4.0% dynamic ATR) only; no fixed TP, no Stage 1/2/3 harvesting
  * Exit on confirmed 1H MA20 structure reversal (LONG close < MA20-buffer & MA5 < MA20; SHORT mirror)
- Execution & Ops: Passive 1-Tick Post-Only Maker (0.02% fee) w/ Market fallback, Universal Jev AI Gate
  (strict fail-closed), account-mode guard, restart-safe state, JSONL trade ledger/MFE-MAE/hourly equity snapshots.
"""

import os
import sys
import time
import math
import hmac
import base64
import json
import logging
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

from bot_utils import send_telegram, load_state, save_state  # noqa: E402 (needs env loaded first)
from risk_policy import (  # noqa: E402
    calculate_target_notional,
    daily_trend_ok,
    loss_limit_breached,
    minimum_contract_is_acceptable,
    projected_risk_usdt,
)

# Import Universal Jev Gate
try:
    from jev.universal_jev_gate import UniversalJevGate
    JEV_GATE_AVAILABLE = True
except ImportError:
    JEV_GATE_AVAILABLE = False

# ── LOGGING SETUP ──
# File logging is done here; systemd stdout goes to journald (avoids duplicated lines in the log file).
LOG_FILE = os.path.join(CURRENT_DIR, "okx_jev_demo_trader.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    ]
)
logger = logging.getLogger("OKXJevDemoTrader")

# ── CREDENTIALS (from .env only — never hardcode) ──
OKX_API_KEY = os.getenv("OKX_DEMO_API_KEY", "")
OKX_SECRET_KEY = os.getenv("OKX_DEMO_SECRET", "")
OKX_PASSPHRASE = os.getenv("OKX_DEMO_PASSPHRASE", "")
OKX_BASE_URL = "https://www.okx.com"

# ── CONFIGURATION (V5 DAILY-REGIME: 4-MAJOR 4/4 WAVE RIDER — 10x, 4구간 백테스트 검증 2026-11-10) ──
TARGET_SYMBOLS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "XRP-USDT-SWAP"]
LEGACY_MANAGE_SYMBOLS = ["DOGE-USDT-SWAP", "SUI-USDT-SWAP"]
LEVERAGE = 10                     # 10x Isolated (corrected S8 MTM MDD -32.4%; risk sizing cuts MDD to -3.4%)
MAX_CONCURRENT_POSITIONS = 2
MAX_POSITION_MARGIN_PCT = max(0.0, min(1.0, float(os.getenv("OKX_JEV_MAX_POSITION_MARGIN_PCT", "0.20"))))
MAX_MARGIN_PER_TRADE = 5000.0     # Max $5,000 margin per slot ($50k notional @ 10x -> stays in OKX Tier-1/2 MMR)
RISK_PER_TRADE_PCT = max(0.0005, min(0.01, float(os.getenv("OKX_JEV_RISK_PER_TRADE", "0.005"))))
MAX_TOTAL_OPEN_RISK_PCT = max(0.001, min(0.05, float(os.getenv("OKX_JEV_MAX_TOTAL_OPEN_RISK", "0.012"))))
MAX_SINGLE_TRADE_RISK_PCT = max(RISK_PER_TRADE_PCT, min(0.02, float(os.getenv("OKX_JEV_MAX_SINGLE_TRADE_RISK", "0.010"))))
DAILY_LOSS_LIMIT_PCT = max(0.001, min(0.20, float(os.getenv("OKX_JEV_DAILY_LOSS_LIMIT", "0.02"))))
WEEKLY_LOSS_LIMIT_PCT = max(DAILY_LOSS_LIMIT_PCT, min(0.30, float(os.getenv("OKX_JEV_WEEKLY_LOSS_LIMIT", "0.04"))))
RISK_FEE_ROUND_TRIP_PCT = max(0.0, float(os.getenv("OKX_JEV_ROUND_TRIP_FEE_EST", "0.001")))
RISK_SLIPPAGE_PCT = max(0.0, float(os.getenv("OKX_JEV_SLIPPAGE_EST", "0.001")))
LOOP_INTERVAL_SEC = 10
POST_LOSS_COOLDOWN_SEC = 3600.0   # 60-minute lockout after stop-loss / wave exit (matches V4-Pro backtest)
NORMAL_COOLDOWN_SEC = 3600.0      # 60-minute post-wave cooldown
ENTRY_COOLDOWN_SEC = 180.0
ORDER_FAIL_COOLDOWN_SEC = 300.0   # 5-minute cooldown after a rejected order
JEV_VETO_COOLDOWN_SEC = 1800.0    # 30-minute cooldown after a Jev veto
JEV_ERROR_COOLDOWN_SEC = 120.0    # 2-minute cooldown after a Jev API error
ACCOUNT_MODE_RECHECK_SEC = 300.0
RADAR_LOG_INTERVAL_SEC = 300.0    # Log 4-Major Dominance Radar summary every 5 minutes
CANDLE_CACHE_TTL_SEC = 18.0       # Cache confirmed 1H/15m candles for 18s to avoid redundant REST polling
STATE_FILE = os.path.join(CURRENT_DIR, "state", "okx_jev_demo_state.json")
LEDGER_FILE = os.path.join(CURRENT_DIR, "state", "okx_jev_demo_ledger.jsonl")
SNAPSHOT_FILE = os.path.join(CURRENT_DIR, "state", "okx_jev_demo_snapshots.jsonl")

# ── ARGON V4.1-PRO WAVE & RISK PARAMETERS (in spot price terms) ──
MIN_SYNC_BREADTH = 4              # Require 4/4 Majors aligned — V5 daily-regime candidate (2023-2026 four-period validation)
MIN_BTC_MA20_SLOPE3_PCT = 0.18    # Minimum |BTC 1H MA20 3-bar slope| (%) to confirm standard market wave
MIN_BTC_IMPULSE_SLOPE3_PCT = 0.10 # Minimum |BTC 1H MA20 3-bar slope| (%) on fresh compression-box breakout
MIN_BTC_IMPULSE_MA5_SLOPE1_PCT = 0.30 # Minimum |BTC 1H MA5 1-bar slope| (%) on fresh compression-box breakout
MIN_BTC_IMPULSE_RET3_PCT = 1.50   # Minimum |BTC 1H 3-bar return| (%) to confirm vertical compression breakout
MIN_1H_MA_SPREAD_PCT = 0.50       # Minimum |1H MA5 - 1H MA20| / MA20 (%) to block tangled sideways chop
MIN_1H_ADX = 22.0                 # Minimum 1H Wilder ADX(14)
MIN_15M_ADX = 22.0                # Minimum 15m Wilder ADX(14)
MIN_15M_VOL_RATIO = 0.75          # Minimum 15m Vol(3)/Vol(20) participation ratio
MIN_DIST_1H_MA20_PCT = 0.15       # Minimum directional breakout clearance beyond 1H MA20 (%)
MAX_DIST_1H_MA20_PCT = 3.10       # Maximum distance from 1H MA20 (%) — blocks late blow-off / waterfall chasing
MAX_IMPULSE_DIST_1H_MA20_PCT = 4.80 # Maximum distance from 1H MA20 (%) on first vertical compression-breakout bar
MAX_BTC_WAVE_AGE_HRS = 10         # Block late-wave entries when BTC wave > 10h AND coin wave > 6h
MAX_SYM_LATE_WAVE_HRS = 6

# Multi-Stage Wave Harvest & Protection (Spot % -> multiplied by 10x for ROE %) — S8 tuned
BASE_SL_CORE = 0.025              # -2.50% base spot SL for BTC/ETH (-25% ROE @ 10x, safely inside ~9% liq)
BASE_SL_ALT = 0.025               # -2.50% base spot SL for SOL/XRP (S8: 1.6~2.0%는 풀손절 반복으로 PF 악화)
MAX_DYN_SL_PCT = 0.040            # -4.00% max dynamic ATR SL (still >= 5% buffer inside 10x isolated liq)
DYN_SL_ATR_MULT = 3.0             # dynamic SL = max(base, min(cap, 3.0 x 15m ATR ratio)) — S8 (기존 2.2)
WAVE_TP_ENABLED = False           # V5 candidate holds to 1H MA20 exit; no fixed take-profit
WAVE_TP_PCT = 0.050               # Legacy TP value (unused while WAVE_TP_ENABLED is False)
STAGE_EXITS_ENABLED = False       # V5 candidate removes Stage 1/2/3 harvesting
DAILY_TREND_FILTER = True         # BTC 20-day EMA direction must agree with the wave side
DAILY_EMA_SPAN_BARS = 480         # 20 days of confirmed 1H bars
DAILY_SLOPE_BARS = 72             # 3 days of confirmed 1H bars
DAILY_SLOPE_MIN_PCT = 0.0
DAILY_BARS_FETCH = 1200           # ~50 days for stable EMA warm-up
DAILY_CACHE_TTL_SEC = 600.0
STAGE1_BE_ARM_PCT = 0.0150        # Stage 1: Peak >= +1.50% spot (+15% ROE @ 10x) -> Lock +0.25% spot (+2.5% ROE)
STAGE1_BE_FLOOR_PCT = 0.0025
STAGE2_MID_ARM_PCT = 0.0220       # Stage 2: Peak >= +2.20% spot (+22% ROE @ 10x) -> Lock +1.10% spot (+11% ROE)
STAGE2_MID_FLOOR_PCT = 0.0110
STAGE3_TRAIL_ARM_PCT = 0.0320     # Stage 3: Peak >= +3.20% spot (+32% ROE @ 10x) -> Trail on 1.00% pullback from peak
STAGE3_TRAIL_DROP_PCT = 0.0100
STAGE3_MIN_PROFIT_PCT = 0.0100    # Minimum +1.00% spot profit required on Stage 3 trail exit
MA20_REV_BUFFER_PCT = 0.0020      # Stage 4: 0.20% clearance across 1H MA20 + MA5 cross confirms wave invalidation

SAME_SIDE_MAX_STREAK = 2          # 2 consecutive same-side losses -> 4-hour directional lockout
SAME_SIDE_LOCKOUT_SEC = 14400.0
MAX_LONG_FUNDING_RATE = 0.0008    # +0.08% extreme funding cap to guard 10x Long Squeeze
MIN_SHORT_FUNDING_RATE = -0.0006  # -0.06% extreme funding floor to guard 10x Short Squeeze


def _round_step(value: float, step: str, rounding=ROUND_DOWN) -> Decimal:
    step_d = Decimal(step)
    return (Decimal(str(value)) / step_d).to_integral_value(rounding=rounding) * step_d


def _fmt(d: Decimal, step: str) -> str:
    """Formats a Decimal with the same number of decimals as the exchange step (e.g. '0.01')."""
    exp = Decimal(step).normalize().as_tuple().exponent
    decimals = max(0, -exp) if isinstance(exp, int) else 0
    return f"{d:.{decimals}f}"


class OKXJevDemoTrader:
    def __init__(self):
        self.api_key = OKX_API_KEY
        self.secret_key = OKX_SECRET_KEY
        self.passphrase = OKX_PASSPHRASE
        self.session = requests.Session()
        self.last_error = ""
        self.server_error_count = 0

        self.account_equity = 0.0
        self.usdt_equity = 0.0
        self.available_usdt = 0.0
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.peak_prices: Dict[str, float] = {}
        self.trough_prices: Dict[str, float] = {}
        self.trade_cooldowns: Dict[str, float] = {}
        self.order_fail_counts: Dict[str, int] = {}
        self.position_entry_times: Dict[str, float] = {}
        self.position_params: Dict[str, Dict[str, float]] = {}
        self.side_loss_streak: Dict[str, int] = {}
        self.side_lockout_until: Dict[str, float] = {}
        self.traded_waves: List[str] = []
        self.inst_cache: Dict[str, Dict[str, str]] = {}
        self.candle_cache: Dict[Tuple[str, str], Tuple[float, pd.DataFrame]] = {}
        self.daily_regime_cache: Tuple[float, Optional[Dict[str, Any]]] = (0.0, None)
        self.trading_enabled = False
        self.last_mode_check = 0.0
        self.last_radar_log = 0.0
        self.last_daily_block_warn = 0.0
        self.last_risk_block_warn = 0.0
        self.daily_pnl_date = ""
        self.daily_start_usdt = 0.0
        self.daily_realized_pnl = 0.0
        self.weekly_pnl_key = ""
        self.weekly_start_usdt = 0.0
        self.weekly_realized_pnl = 0.0
        self.last_snapshot_hour = ""
        self._load_state()

        if JEV_GATE_AVAILABLE:
            self.jev_gate = UniversalJevGate(
                confidence_threshold=float(os.getenv("JEV_CONFIDENCE_THRESHOLD", "0.60")),
                fail_closed=True,
            )
            if self.jev_gate.is_enabled and self.jev_gate.api_key:
                logger.info("🧠 Universal Jev AI Gate loaded (4-Major Wave prompt, strict fail-closed).")
            else:
                logger.warning("⚠️ Jev Gate disabled (USE_JEV_PREDICTION=false or no API key) -> technical signals only.")
        else:
            self.jev_gate = None
            logger.warning("⚠️ Universal Jev Gate unavailable -> technical signals only.")

        logger.info("=" * 76)
        logger.info("🚀 [OKX Jev Demo Trader] V5 Daily-Regime Trend: 4-Major Dominance Wave Rider Active")
        logger.info(f"   Target Universe : {', '.join(TARGET_SYMBOLS)} ({MIN_SYNC_BREADTH}/4 Synchronization Required)")
        logger.info(
            f"   Leverage & Risk : {LEVERAGE}x Isolated | Risk {RISK_PER_TRADE_PCT*100:.2f}%/trade, "
            f"total open {MAX_TOTAL_OPEN_RISK_PCT*100:.2f}%, daily stop {DAILY_LOSS_LIMIT_PCT*100:.1f}%, "
            f"weekly stop {WEEKLY_LOSS_LIMIT_PCT*100:.1f}% (Max {MAX_CONCURRENT_POSITIONS} Slots, "
            f"margin cap {MAX_POSITION_MARGIN_PCT*100:.0f}%/${MAX_MARGIN_PER_TRADE:,.0f})"
        )
        logger.info(
            f"   Wave Entry Gate : {MIN_SYNC_BREADTH}/4 + BTC Slope>={MIN_BTC_MA20_SLOPE3_PCT:.2f}% + "
            f"Spread>={MIN_1H_MA_SPREAD_PCT:.2f}% + ADX>={MIN_1H_ADX:.0f}/{MIN_15M_ADX:.0f} + "
            f"BTC 20D EMA filter={'ON' if DAILY_TREND_FILTER else 'OFF'}"
        )
        logger.info(
            "   Exit Rule       : Exchange SL + 1H MA20 structure exit"
            if not STAGE_EXITS_ENABLED and not WAVE_TP_ENABLED
            else f"   Harvest         : TP={'ON' if WAVE_TP_ENABLED else 'OFF'} / Stages={'ON' if STAGE_EXITS_ENABLED else 'OFF'} / 1H MA20 Exit"
        )
        logger.info("   Execution & Safe: 1-Tick Passive Maker (0.02%) -> Market Fallback | 100% Demo (x-simulated-trading: 1)")
        logger.info("=" * 76)

    # ── STATE ──
    def _load_state(self):
        st = load_state(STATE_FILE)
        now = time.time()
        self.peak_prices = {k: float(v) for k, v in st.get("peak_prices", {}).items()}
        self.trough_prices = {k: float(v) for k, v in st.get("trough_prices", {}).items()}
        self.trade_cooldowns = {k: float(v) for k, v in st.get("trade_cooldowns", {}).items() if float(v) > now}
        self.position_entry_times = {k: float(v) for k, v in st.get("position_entry_times", {}).items()}
        self.position_params = st.get("position_params", {}) or {}
        self.positions = st.get("positions", {}) or {}
        self.side_loss_streak = {k: int(v) for k, v in st.get("side_loss_streak", {}).items() if int(v) > 0}
        self.side_lockout_until = {k: float(v) for k, v in st.get("side_lockout_until", {}).items() if float(v) > now}
        raw_waves = st.get("traded_waves", [])
        self.traded_waves = [str(x) for x in raw_waves[-80:]] if isinstance(raw_waves, list) else []
        self.daily_pnl_date = str(st.get("daily_pnl_date", ""))
        self.daily_start_usdt = float(st.get("daily_start_usdt", 0.0) or 0.0)
        self.daily_realized_pnl = float(st.get("daily_realized_pnl", 0.0) or 0.0)
        self.weekly_pnl_key = str(st.get("weekly_pnl_key", ""))
        self.weekly_start_usdt = float(st.get("weekly_start_usdt", 0.0) or 0.0)
        self.weekly_realized_pnl = float(st.get("weekly_realized_pnl", 0.0) or 0.0)
        self.last_snapshot_hour = str(st.get("last_snapshot_hour", ""))
        if st:
            logger.info(
                f"♻️ [State Restored] peaks={self.peak_prices} cooldowns={list(self.trade_cooldowns)} "
                f"waves={len(self.traded_waves)} streaks={self.side_loss_streak} lockouts={list(self.side_lockout_until)}"
            )

    def _save_state(self):
        now = time.time()
        save_state(STATE_FILE, {
            "peak_prices": self.peak_prices,
            "trough_prices": self.trough_prices,
            "trade_cooldowns": {k: v for k, v in self.trade_cooldowns.items() if v > now},
            "position_entry_times": self.position_entry_times,
            "position_params": self.position_params,
            "side_loss_streak": {k: v for k, v in self.side_loss_streak.items() if v > 0},
            "side_lockout_until": {k: v for k, v in self.side_lockout_until.items() if v > now},
            "traded_waves": self.traded_waves[-80:],
            "positions": {k: {"avgPx": v.get("avgPx"), "pos": v.get("pos"), "upl": v.get("upl"), "uplRatio": v.get("uplRatio")}
                          for k, v in self.positions.items()},
            "daily_pnl_date": self.daily_pnl_date,
            "daily_start_usdt": self.daily_start_usdt,
            "daily_realized_pnl": self.daily_realized_pnl,
            "weekly_pnl_key": self.weekly_pnl_key,
            "weekly_start_usdt": self.weekly_start_usdt,
            "weekly_realized_pnl": self.weekly_realized_pnl,
            "last_snapshot_hour": self.last_snapshot_hour,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        })

    def _mark_wave_traded(self, inst_id: str, btc_wave_ts: str, sym_wave_ts: str):
        for wk in (f"{inst_id}:BTC:{btc_wave_ts}", f"{inst_id}:SYM:{sym_wave_ts}"):
            if wk not in self.traded_waves:
                self.traded_waves.append(wk)
        if len(self.traded_waves) > 80:
            self.traded_waves = self.traded_waves[-80:]

    def _record_trade_outcome(self, inst_id: str, side_desc: str, is_loss: bool):
        key = f"{inst_id}:{side_desc}"
        if is_loss:
            streak = self.side_loss_streak.get(key, 0) + 1
            if streak >= SAME_SIDE_MAX_STREAK:
                self.side_lockout_until[key] = time.time() + SAME_SIDE_LOCKOUT_SEC
                self.side_loss_streak[key] = 0
                logger.warning(
                    f"🛑 [2-Strike Directional Lockout] {inst_id} {side_desc} hit {streak} consecutive losses -> "
                    f"locked out from {side_desc} for {SAME_SIDE_LOCKOUT_SEC/3600:.1f}h!"
                )
            else:
                self.side_loss_streak[key] = streak
        else:
            self.side_loss_streak.pop(key, None)

    # ── RISK PERIODS, LEDGER & SNAPSHOTS ──
    @staticmethod
    def _period_keys(now: Optional[datetime] = None) -> Tuple[str, str]:
        now = now or datetime.now(timezone.utc)
        iso = now.isocalendar()
        return now.date().isoformat(), f"{iso.year}-W{iso.week:02d}"

    def _roll_risk_periods(self):
        day_key, week_key = self._period_keys()
        equity = self.usdt_equity if self.usdt_equity > 0 else self.account_equity
        if self.daily_pnl_date != day_key:
            if self.daily_pnl_date:
                logger.info(
                    f"📅 [Risk Period] New UTC day {day_key}: prior day realized "
                    f"{self.daily_realized_pnl:+.2f} USDT"
                )
            self.daily_pnl_date = day_key
            self.daily_start_usdt = equity
            self.daily_realized_pnl = 0.0
        if self.weekly_pnl_key != week_key:
            if self.weekly_pnl_key:
                logger.info(
                    f"📅 [Risk Period] New ISO week {week_key}: prior week realized "
                    f"{self.weekly_realized_pnl:+.2f} USDT"
                )
            self.weekly_pnl_key = week_key
            self.weekly_start_usdt = equity
            self.weekly_realized_pnl = 0.0
        if self.daily_start_usdt <= 0 and equity > 0:
            self.daily_start_usdt = equity
        if self.weekly_start_usdt <= 0 and equity > 0:
            self.weekly_start_usdt = equity

    def _append_jsonl(self, path: str, record: Dict[str, Any]):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            logger.error(f"❌ [Journal Write Failed] {path}: {e}")

    def _append_ledger(self, event: str, payload: Dict[str, Any]):
        self._append_jsonl(LEDGER_FILE, {
            "event": event,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
        })

    def _record_realized_pnl(self, net_pnl_usdt: float, symbol: str, reason: str, source: str):
        self._roll_risk_periods()
        self.daily_realized_pnl += float(net_pnl_usdt)
        self.weekly_realized_pnl += float(net_pnl_usdt)
        logger.info(
            f"💰 [Realized PnL] {symbol} {net_pnl_usdt:+.2f} USDT ({reason}, {source}) | "
            f"Day {self.daily_realized_pnl:+.2f} / Week {self.weekly_realized_pnl:+.2f}"
        )
        for label, pnl, start_eq, limit_pct in (
            ("DAILY", self.daily_realized_pnl, self.daily_start_usdt, DAILY_LOSS_LIMIT_PCT),
            ("WEEKLY", self.weekly_realized_pnl, self.weekly_start_usdt, WEEKLY_LOSS_LIMIT_PCT),
        ):
            if loss_limit_breached(pnl, start_eq, limit_pct):
                send_telegram(
                    f"[OKX Demo] 🛑 {label} LOSS LIMIT: {pnl:+.2f} / start {start_eq:,.2f} USDT "
                    f"({pnl/start_eq*100:+.2f}%, limit -{limit_pct*100:.1f}%) -> new entries blocked",
                    dedupe_key=f"okx_{label.lower()}_loss_limit_{self.weekly_pnl_key if label == 'WEEKLY' else self.daily_pnl_date}",
                    min_interval_sec=1800,
                )

    def _risk_block_reason(self) -> Optional[str]:
        self._roll_risk_periods()
        if loss_limit_breached(self.daily_realized_pnl, self.daily_start_usdt, DAILY_LOSS_LIMIT_PCT):
            return (
                f"DAILY_LOSS_LIMIT {self.daily_realized_pnl:+.2f}/{self.daily_start_usdt:,.2f} USDT "
                f"(limit -{DAILY_LOSS_LIMIT_PCT*100:.1f}%)"
            )
        if loss_limit_breached(self.weekly_realized_pnl, self.weekly_start_usdt, WEEKLY_LOSS_LIMIT_PCT):
            return (
                f"WEEKLY_LOSS_LIMIT {self.weekly_realized_pnl:+.2f}/{self.weekly_start_usdt:,.2f} USDT "
                f"(limit -{WEEKLY_LOSS_LIMIT_PCT*100:.1f}%)"
            )
        return None

    def _position_notional_usdt(self, pos: Dict[str, Any]) -> float:
        notional = abs(float(pos.get("notionalUsd") or 0.0))
        if notional > 0:
            return notional
        margin = abs(float(pos.get("margin") or 0.0))
        return margin * LEVERAGE

    def _slot_available(self, pending_opens: int = 0) -> bool:
        return len(self.positions) + max(0, int(pending_opens)) < MAX_CONCURRENT_POSITIONS

    def _current_open_risk_usdt(self) -> float:
        total = 0.0
        for sym, pos in self.positions.items():
            params = self.position_params.get(sym, {})
            default_sl = BASE_SL_CORE if sym in ("BTC-USDT-SWAP", "ETH-USDT-SWAP") else BASE_SL_ALT
            sl_pct = float(params.get("sl_pct", default_sl) or default_sl)
            total += projected_risk_usdt(
                self._position_notional_usdt(pos), sl_pct, RISK_FEE_ROUND_TRIP_PCT, RISK_SLIPPAGE_PCT
            )
        return total

    def _snapshot_if_due(self):
        if self.usdt_equity <= 0:
            return
        hour_key = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")
        if hour_key == self.last_snapshot_hour:
            return
        self.last_snapshot_hour = hour_key
        self._append_jsonl(SNAPSHOT_FILE, {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "total_equity_usdt": round(self.account_equity, 6),
            "usdt_equity": round(self.usdt_equity, 6),
            "available_usdt": round(self.available_usdt, 6),
            "positions": len(self.positions),
            "open_risk_usdt": round(self._current_open_risk_usdt(), 6),
            "daily_realized_pnl": round(self.daily_realized_pnl, 6),
            "weekly_realized_pnl": round(self.weekly_realized_pnl, 6),
            "daily_start_usdt": round(self.daily_start_usdt, 6),
            "weekly_start_usdt": round(self.weekly_start_usdt, 6),
        })

    # ── HTTP ──
    def _sign_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        pre_hash = ts + method.upper() + path + body
        sign = base64.b64encode(
            hmac.new(self.secret_key.encode("utf-8"), pre_hash.encode("utf-8"), digestmod="sha256").digest()
        ).decode("utf-8")
        return {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": sign,
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
            "x-simulated-trading": "1"  # CRITICAL: Always Demo Mode
        }

    def _request(self, method: str, path: str, params: Optional[Dict] = None, data: Optional[Any] = None) -> Optional[Dict]:
        """Signed request. `path` must include the query string (it is part of the signature)."""
        body = json.dumps(data) if data else ""
        headers = self._sign_headers(method, path, body)
        url = f"{OKX_BASE_URL}{path}"
        self.last_error = ""
        try:
            if method.upper() == "GET":
                resp = self.session.get(url, headers=headers, params=params, timeout=5)
            elif method.upper() == "POST":
                resp = self.session.post(url, headers=headers, data=body, timeout=5)
            else:
                return None

            if resp.status_code == 200:
                res_json = resp.json()
                if res_json.get("code") == "0":
                    self.server_error_count = 0
                    return res_json
                details = [f"{d.get('sCode')}:{d.get('sMsg')}" for d in res_json.get("data", [])
                           if isinstance(d, dict) and d.get("sCode")]
                self.last_error = f"{res_json.get('code')} {res_json.get('msg')} {details}".strip()
                logger.warning(f"⚠️ OKX API code error: {res_json.get('code')} - {res_json.get('msg')} {details} ({path})")
            elif resp.status_code in (500, 502, 503, 504):
                self.server_error_count = getattr(self, "server_error_count", 0) + 1
                backoff = min(60.0, 5.0 * (2 ** min(self.server_error_count - 1, 3)))
                self.last_error = f"HTTP {resp.status_code} Temporary Gateway/Maintenance"
                logger.warning(f"⚠️ OKX 서버 일시 점검/지연 (HTTP {resp.status_code}) [{path}] - {backoff:.0f}s 백오프 대기 (연속 {self.server_error_count}회)")
                time.sleep(backoff)
            else:
                self.last_error = f"HTTP {resp.status_code} {resp.text[:200]}"
                logger.error(f"❌ OKX HTTP error: {resp.status_code} - {resp.text[:300]} ({path})")
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"❌ OKX Request exception ({path}): {e}")
            time.sleep(2.0)
        return None

    # ── ACCOUNT ──
    def check_account_mode(self) -> bool:
        """SWAP trading requires Futures mode or above (acctLv >= 2). acctLv=1 (Spot mode) rejects every order."""
        now = time.time()
        if self.trading_enabled or now - self.last_mode_check < ACCOUNT_MODE_RECHECK_SEC:
            return self.trading_enabled
        self.last_mode_check = now
        res = self._request("GET", "/api/v5/account/config")
        if not res or not res.get("data"):
            return False
        cfg = res["data"][0]
        acct_lv, pos_mode = cfg.get("acctLv"), cfg.get("posMode")
        if acct_lv in ("2", "3", "4") and pos_mode == "net_mode":
            if not self.trading_enabled:
                logger.info(f"✅ [Account Mode OK] acctLv={acct_lv} posMode={pos_mode} -> trading enabled")
            self.trading_enabled = True
        else:
            msg = (f"OKX 데모 계좌 모드가 선물 거래 불가 상태입니다 (acctLv={acct_lv}, posMode={pos_mode}). "
                   f"OKX 웹/앱 데모 트레이딩 > 설정 > 계좌 모드를 'Futures mode'(단방향 net)로 변경하세요. "
                   f"{ACCOUNT_MODE_RECHECK_SEC/60:.0f}분마다 재확인합니다.")
            logger.error(f"⛔ [Account Mode] {msg}")
            send_telegram(f"[OKX Demo] ⛔ {msg}", dedupe_key="okx_acct_mode", min_interval_sec=6 * 3600)
            self.trading_enabled = False
        return self.trading_enabled

    def sync_account(self) -> bool:
        """Fetches demo account equity, total USDT equity, and available USDT"""
        res = self._request("GET", "/api/v5/account/balance")
        if res and res.get("data"):
            acc = res["data"][0]
            self.account_equity = float(acc.get("totalEq", 0.0) or 0.0)
            for d in acc.get("details", []):
                if d.get("ccy") == "USDT":
                    self.available_usdt = float(d.get("availBal") or d.get("availEq") or 0.0)
                    self.usdt_equity = float(d.get("eq") or self.available_usdt or 0.0)
            self._sync_log_cnt = getattr(self, "_sync_log_cnt", 0) + 1
            if self._sync_log_cnt % 30 == 1:
                logger.info(
                    f"💼 [Demo Account Sync] Total Eq: ${self.account_equity:,.2f} | "
                    f"USDT Eq: ${self.usdt_equity:,.2f} | Avail USDT: ${self.available_usdt:,.2f}"
                )
            self._roll_risk_periods()
            return True
        return False

    def sync_positions(self) -> Dict[str, Dict[str, Any]]:
        """Fetches active open positions across TARGET_SYMBOLS and any legacy symbols"""
        res = self._request("GET", "/api/v5/account/positions?instType=SWAP")
        if res:
            managed_set = set(TARGET_SYMBOLS) | set(LEGACY_MANAGE_SYMBOLS) | set(self.positions.keys())
            new_pos = {}
            for p in res.get("data", []):
                inst_id = p.get("instId")
                if inst_id in managed_set and abs(float(p.get("pos", 0.0) or 0.0)) > 0:
                    new_pos[inst_id] = p

            closed = [(sym, pos) for sym, pos in self.positions.items() if sym not in new_pos]
            self.positions = new_pos
            for sym, prev_pos in closed:
                self._handle_external_close(sym, prev_pos)
        return self.positions

    def _fetch_latest_position_history(self, inst_id: str, not_before_ms: int = 0) -> Optional[Dict[str, Any]]:
        res = self._request(
            "GET", f"/api/v5/account/positions-history?instType=SWAP&instId={inst_id}&limit=10"
        )
        candidates = []
        for h in (res or {}).get("data", []):
            try:
                if int(float(h.get("uTime") or 0)) >= not_before_ms:
                    candidates.append(h)
            except (TypeError, ValueError):
                continue
        if not candidates:
            return None
        return max(candidates, key=lambda h: int(float(h.get("uTime") or 0)))

    @staticmethod
    def _history_breakdown(h: Dict[str, Any]) -> Dict[str, float]:
        def num_opt(key: str) -> Optional[float]:
            raw = h.get(key)
            if raw in (None, ""):
                return None
            try:
                return float(raw)
            except (TypeError, ValueError):
                return None

        pnl = num_opt("pnl")
        fee = num_opt("fee") or 0.0
        funding = num_opt("fundingFee") or 0.0
        liq_penalty = num_opt("liqPenalty") or 0.0
        realized = num_opt("realizedPnl")
        if realized is None:
            realized = (pnl or 0.0) + fee + funding + liq_penalty
        if pnl is None:
            pnl = realized - fee - funding - liq_penalty
        return {
            "gross_pnl_usdt": pnl,
            "fee_usdt": fee,
            "funding_usdt": funding,
            "liq_penalty_usdt": liq_penalty,
            "net_pnl_usdt": realized,
        }

    def _mfe_mae_pct(self, inst_id: str, avg_px: float, is_long: bool) -> Tuple[float, float]:
        if avg_px <= 0:
            return 0.0, 0.0
        peak = float(self.peak_prices.get(inst_id, avg_px) or avg_px)
        trough = float(self.trough_prices.get(inst_id, avg_px) or avg_px)
        if is_long:
            mfe = peak / avg_px - 1.0
            mae = trough / avg_px - 1.0
        else:
            mfe = (avg_px - peak) / avg_px
            mae = (avg_px - trough) / avg_px
        return max(0.0, mfe), min(0.0, mae)

    def _handle_external_close(self, inst_id: str, prev_pos: Dict[str, Any]):
        avg_px = float(prev_pos.get("avgPx", 0.0) or 0.0)
        is_long = float(prev_pos.get("pos", 0.0) or 0.0) > 0
        side_desc = "LONG" if is_long else "SHORT"
        entry_time = self.position_entry_times.get(inst_id, 0.0)
        h = self._fetch_latest_position_history(inst_id, int(entry_time * 1000) - 5000 if entry_time else 0)
        if h:
            breakdown = self._history_breakdown(h)
            pnl, src = breakdown["net_pnl_usdt"], "positions-history"
            close_avg_px = float(h.get("closeAvgPx") or 0.0)
        else:
            breakdown = {
                "gross_pnl_usdt": float(prev_pos.get("upl") or 0.0),
                "fee_usdt": 0.0,
                "funding_usdt": 0.0,
                "liq_penalty_usdt": 0.0,
                "net_pnl_usdt": float(prev_pos.get("upl") or 0.0),
            }
            pnl, src, close_avg_px = breakdown["net_pnl_usdt"], "last-upl", 0.0
        mfe_pct, mae_pct = self._mfe_mae_pct(inst_id, avg_px, is_long)
        is_loss = pnl < 0
        self._record_trade_outcome(inst_id, side_desc, is_loss)
        cooldown = POST_LOSS_COOLDOWN_SEC if is_loss else NORMAL_COOLDOWN_SEC
        self.trade_cooldowns[inst_id] = time.time() + cooldown
        self._record_realized_pnl(pnl, inst_id, "EXTERNAL_CLOSE", src)
        self._append_ledger("EXIT", {
            "inst_id": inst_id,
            "side": side_desc,
            "reason": "EXTERNAL_SL_TP",
            "source": src,
            "entry_avg_px": avg_px,
            "close_avg_px": close_avg_px,
            "mfe_pct": round(mfe_pct * 100.0, 6),
            "mae_pct": round(mae_pct * 100.0, 6),
            **{k: round(v, 8) for k, v in breakdown.items()},
            "equity_usdt": round(self.usdt_equity, 8),
            "daily_realized_pnl": round(self.daily_realized_pnl, 8),
            "weekly_realized_pnl": round(self.weekly_realized_pnl, 8),
        })
        self.peak_prices.pop(inst_id, None)
        self.trough_prices.pop(inst_id, None)
        self.position_entry_times.pop(inst_id, None)
        self.position_params.pop(inst_id, None)
        self._cancel_algo_orders(inst_id)
        self._save_state()
        tag = "LOSS" if is_loss else "WIN"
        logger.info(f"ℹ️ [Position Closed Externally] {inst_id} {tag} net={pnl:+.2f} USDT ({src}) -> cooldown {cooldown/60:.0f}m")
        send_telegram(f"[OKX Demo] {inst_id} 거래소 SL/TP 청산 ({tag} net {pnl:+.2f} USDT)")

    # ── MARKET DATA & WAVE INDICATORS ──
    def get_bbo_and_imbalance(self, inst_id: str) -> Tuple[float, float, float, float]:
        """Fetches L1 BBO (mid, imbalance, best_bid, best_ask) and top-5 orderbook imbalance"""
        try:
            url = f"{OKX_BASE_URL}/api/v5/market/books?instId={inst_id}&sz=5"
            resp = self.session.get(url, timeout=3)
            if resp.status_code == 200:
                d = resp.json().get("data", [])
                if d:
                    bids = d[0].get("bids", [])
                    asks = d[0].get("asks", [])
                    if bids and asks:
                        best_bid = float(bids[0][0])
                        best_ask = float(asks[0][0])
                        mid_price = (best_bid + best_ask) / 2.0
                        bid_vol = sum(float(b[1]) for b in bids)
                        ask_vol = sum(float(a[1]) for a in asks)
                        tot_vol = bid_vol + ask_vol
                        imbalance = (bid_vol - ask_vol) / tot_vol if tot_vol > 0 else 0.0
                        return mid_price, imbalance, best_bid, best_ask
        except Exception as e:
            logger.debug(f"Ticker/Imb error for {inst_id}: {e}")
        return 0.0, 0.0, 0.0, 0.0

    def get_ticker_and_imbalance(self, inst_id: str) -> Tuple[float, float]:
        mid_price, imbalance, _, _ = self.get_bbo_and_imbalance(inst_id)
        return mid_price, imbalance

    def get_funding_rate(self, inst_id: str) -> float:
        """Fetches real-time 8H funding rate for perpetual swap to guard against Long/Short Squeezes"""
        try:
            url = f"{OKX_BASE_URL}/api/v5/public/funding-rate?instId={inst_id}"
            resp = self.session.get(url, timeout=3)
            if resp.status_code == 200:
                d = resp.json().get("data", [])
                if d and d[0].get("fundingRate"):
                    return float(d[0]["fundingRate"])
        except Exception as e:
            logger.debug(f"Funding rate error for {inst_id}: {e}")
        return 0.0

    def fetch_bars(self, inst_id: str, bar: str = "15m", limit: int = 100, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Fetches CONFIRMED candlestick bars only (the live, still-forming candle is dropped to avoid repaint)"""
        cache_key = (inst_id, bar)
        now = time.time()
        if use_cache and cache_key in self.candle_cache:
            ts_cached, df_cached = self.candle_cache[cache_key]
            if now - ts_cached < CANDLE_CACHE_TTL_SEC and len(df_cached) >= min(25, limit - 5):
                return df_cached
        try:
            url = f"{OKX_BASE_URL}/api/v5/market/candles?instId={inst_id}&bar={bar}&limit={limit}"
            resp = self.session.get(url, timeout=4)
            if resp.status_code == 200:
                raw = resp.json().get("data", [])
                if not raw:
                    return None
                raw = sorted(raw, key=lambda x: int(x[0]))
                df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"])
                df = df[df["confirm"] == "1"].reset_index(drop=True)
                for col in ("open", "high", "low", "close", "vol"):
                    df[col] = df[col].astype(float)
                self.candle_cache[cache_key] = (now, df)
                return df
        except Exception as e:
            logger.debug(f"Bars error for {inst_id} ({bar}): {e}")
        return None

    def fetch_history_bars(self, inst_id: str, bar: str = "1H", limit: int = 1200) -> Optional[pd.DataFrame]:
        """Fetches a long confirmed history via paginated history-candles (newest first, then backwards)."""
        rows: List[List[str]] = []
        after = ""
        try:
            for _ in range(max(1, math.ceil(limit / 100))):
                params = {"instId": inst_id, "bar": bar, "limit": "100"}
                if after:
                    params["after"] = after
                resp = self.session.get(f"{OKX_BASE_URL}/api/v5/market/history-candles", params=params, timeout=5)
                if resp.status_code != 200:
                    break
                raw = resp.json().get("data", [])
                if not raw:
                    break
                raw = sorted(raw, key=lambda x: int(x[0]))
                rows.extend(raw)
                after = str(int(raw[0][0]))
                if len(rows) >= limit:
                    break
                time.sleep(0.12)
            if not rows:
                return None
            rows = sorted({int(r[0]): r for r in rows}.values(), key=lambda x: int(x[0]))[-limit:]
            df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"])
            df = df[df["confirm"] == "1"].reset_index(drop=True)
            for col in ("open", "high", "low", "close", "vol"):
                df[col] = df[col].astype(float)
            return df
        except Exception as e:
            logger.warning(f"⚠️ History bars error for {inst_id} ({bar}): {e}")
            return None

    def btc_daily_regime(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """BTC 20-day EMA + 3-day slope regime, cached for 10 minutes."""
        now = time.time()
        ts_cached, state = self.daily_regime_cache
        if not force and state is not None and now - ts_cached < DAILY_CACHE_TTL_SEC:
            return state
        if not DAILY_TREND_FILTER:
            state = {"long_ok": True, "short_ok": True, "close": 0.0, "ema": 0.0, "slope_pct": 0.0}
            self.daily_regime_cache = (now, state)
            return state

        df = self.fetch_history_bars("BTC-USDT-SWAP", bar="1H", limit=DAILY_BARS_FETCH)
        if df is None or len(df) < DAILY_EMA_SPAN_BARS + 1:
            self.daily_regime_cache = (now, None)
            return None

        closes = df["close"]
        ema = closes.ewm(span=DAILY_EMA_SPAN_BARS, adjust=False).mean()
        close_last = float(closes.iloc[-1])
        ema_last = float(ema.iloc[-1])
        ema_prev = float(ema.iloc[-1 - DAILY_SLOPE_BARS])
        slope_pct = (ema_last / ema_prev - 1.0) * 100.0 if ema_prev > 0 else 0.0
        state = {
            "long_ok": bool(daily_trend_ok(closes, True, DAILY_EMA_SPAN_BARS, DAILY_SLOPE_BARS, DAILY_SLOPE_MIN_PCT)),
            "short_ok": bool(daily_trend_ok(closes, False, DAILY_EMA_SPAN_BARS, DAILY_SLOPE_BARS, DAILY_SLOPE_MIN_PCT)),
            "close": close_last,
            "ema": ema_last,
            "slope_pct": slope_pct,
        }
        self.daily_regime_cache = (now, state)
        return state

    @staticmethod
    def _calc_wilder_adx(highs: pd.Series, lows: pd.Series, closes: pd.Series) -> float:
        tr = pd.concat([highs - lows, (highs - closes.shift()).abs(), (lows - closes.shift()).abs()], axis=1).max(axis=1)
        up = highs.diff()
        dn = -lows.diff()
        plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=closes.index)
        minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=closes.index)
        atr14 = tr.ewm(alpha=1/14, adjust=False).mean().replace(0, 1e-9)
        pdi = 100.0 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14)
        mdi = 100.0 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14)
        dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-9)
        adx_val = float(dx.ewm(alpha=1/14, adjust=False).mean().iloc[-1])
        return 0.0 if math.isnan(adx_val) else adx_val

    def analyze_1h_wave(self, df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """Computes 1H MA(5/10/20), slopes, spread, Wilder ADX(14), and restart-invariant wave_ts / wave_age."""
        if df is None or len(df) < 25:
            return None
        c, h, l = df["close"], df["high"], df["low"]
        ma5_s = c.rolling(5).mean()
        ma10_s = c.rolling(10).mean()
        ma20_s = c.rolling(20).mean()

        c_last = float(c.iloc[-1])
        ma5 = float(ma5_s.iloc[-1])
        ma10 = float(ma10_s.iloc[-1])
        ma20 = float(ma20_s.iloc[-1])
        if math.isnan(ma20) or ma20 <= 0:
            return None

        ma20_prev3 = float(ma20_s.iloc[-4]) if not math.isnan(float(ma20_s.iloc[-4])) else ma20
        ma5_prev1 = float(ma5_s.iloc[-2]) if not math.isnan(float(ma5_s.iloc[-2])) else ma5
        c_prev3 = float(c.iloc[-4]) if len(c) >= 4 else c_last

        dist_ma20 = (c_last / ma20 - 1.0) * 100.0
        ma20_slope3 = (ma20 / ma20_prev3 - 1.0) * 100.0 if ma20_prev3 > 0 else 0.0
        ma5_slope1 = (ma5 / ma5_prev1 - 1.0) * 100.0 if ma5_prev1 > 0 else 0.0
        ma_spread_pct = abs(ma5 - ma20) / ma20 * 100.0
        ret3 = (c_last / c_prev3 - 1.0) * 100.0 if c_prev3 > 0 else 0.0
        adx = self._calc_wilder_adx(h, l, c)

        # Invariant wave start timestamp & bars_in_wave since Close crossed MA20
        above_cur = bool(c_last > ma20)
        wave_age = 0
        wave_ts = str(df["ts"].iloc[-1])
        for idx in range(len(df) - 2, 18, -1):
            m20_k = float(ma20_s.iloc[idx])
            if math.isnan(m20_k):
                break
            if bool(float(c.iloc[idx]) > m20_k) == above_cur:
                wave_age += 1
                wave_ts = str(df["ts"].iloc[idx])
            else:
                break

        return {
            "close": c_last,
            "ma5": ma5,
            "ma10": ma10,
            "ma20": ma20,
            "dist_ma20": dist_ma20,
            "ma20_slope3": ma20_slope3,
            "ma5_slope1": ma5_slope1,
            "ma_spread_pct": ma_spread_pct,
            "ret3": ret3,
            "adx": adx,
            "above_ma20": above_cur,
            "wave_ts": wave_ts,
            "wave_age": wave_age,
        }

    def analyze_15m_wave(self, df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """Computes 15m EMA9, EMA20, ATR(14)/Price ratio, Wilder ADX(14), and Vol(3)/Vol(20)."""
        if df is None or len(df) < 22:
            return None
        c, h, l, v = df["close"], df["high"], df["low"], df["vol"]
        c_last = float(c.iloc[-1])
        ema9 = float(c.ewm(span=9, adjust=False).mean().iloc[-1])
        ema20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        atr_ratio = (atr / c_last) if c_last > 0 and not math.isnan(atr) else 0.005
        vol_ma20 = float(v.iloc[-20:].mean())
        vol_ma3 = float(v.iloc[-3:].mean())
        vol_ratio = (vol_ma3 / vol_ma20) if vol_ma20 > 0 else 1.0
        adx = self._calc_wilder_adx(h, l, c)
        lookback = min(12, len(c) - 1)
        recent_ret_pct = ((c_last / float(c.iloc[-1 - lookback])) - 1.0) * 100.0 if lookback > 0 else 0.0

        return {
            "close": c_last,
            "ema9": ema9,
            "ema20": ema20,
            "dist_ema20_pct": (c_last / ema20 - 1.0) * 100.0 if ema20 > 0 else 0.0,
            "atr_ratio": atr_ratio,
            "vol_ratio": vol_ratio,
            "adx": adx,
            "recent_return_pct": recent_ret_pct,
            "lookback_bars": lookback,
        }

    # ── INSTRUMENTS / ORDERS ──
    def get_instrument(self, inst_id: str) -> Optional[Dict[str, str]]:
        """Returns ctVal / lotSz / minSz / tickSz (cached)"""
        if inst_id in self.inst_cache:
            return self.inst_cache[inst_id]
        res = self._request("GET", f"/api/v5/public/instruments?instType=SWAP&instId={inst_id}")
        if res and res.get("data"):
            d = res["data"][0]
            info = {k: d.get(k) for k in ("ctVal", "lotSz", "minSz", "tickSz")}
            if all(info.values()):
                self.inst_cache[inst_id] = info
                return info
        return None

    def set_leverage(self, inst_id: str):
        """Ensures isolated leverage is set cleanly (cancels any stale algo orders first)"""
        self._cancel_algo_orders(inst_id)
        data = {
            "instId": inst_id,
            "lever": str(LEVERAGE),
            "mgnMode": "isolated"
        }
        res = self._request("POST", "/api/v5/account/set-leverage", data=data)
        if res:
            logger.info(f"⚡ [Leverage Configured] {inst_id} set to {LEVERAGE}x Isolated Margin")

    def _cancel_algo_orders(self, inst_id: str):
        """Cancels leftover attached SL/TP algo orders for the instrument."""
        res = self._request("GET", f"/api/v5/trade/orders-algo-pending?ordType=conditional,oco&instId={inst_id}")
        algos = [{"algoId": a["algoId"], "instId": inst_id} for a in (res or {}).get("data", []) if a.get("algoId")]
        if algos:
            self._request("POST", "/api/v5/trade/cancel-algos", data=algos)
            logger.info(f"🧹 [Algo Cleanup] {inst_id} cancelled {len(algos)} leftover SL/TP algo order(s)")

    def open_position(
        self,
        inst_id: str,
        side: str = "buy",
        sl_pct: float = BASE_SL_ALT,
        tp_pct: float = WAVE_TP_PCT,
        btc_wave_ts: str = "",
        sym_wave_ts: str = "",
    ) -> bool:
        """Submits Passive 1-Tick Post-Only Maker order (w/ Market fallback) + exchange-attached SL/TP"""
        block = self._risk_block_reason()
        if block:
            logger.warning(f"🛑 [Entry Blocked] {inst_id}: {block}")
            return False

        info = self.get_instrument(inst_id)
        mid_pr, imb = self.get_ticker_and_imbalance(inst_id)
        if not info or mid_pr <= 0:
            logger.warning(f"⚠️ {inst_id}: instrument info / price unavailable, skip entry")
            return False
        self.set_leverage(inst_id)

        ct_val = float(info["ctVal"])
        min_sz_d = Decimal(info["minSz"])
        equity = self.usdt_equity if self.usdt_equity > 0 else self.available_usdt
        current_heat = self._current_open_risk_usdt()
        max_margin_per_trade = min(MAX_MARGIN_PER_TRADE, max(0.0, equity * MAX_POSITION_MARGIN_PCT))
        target_notional = calculate_target_notional(
            equity_usdt=equity,
            risk_per_trade_pct=RISK_PER_TRADE_PCT,
            sl_pct=sl_pct,
            fee_pct=RISK_FEE_ROUND_TRIP_PCT,
            slippage_pct=RISK_SLIPPAGE_PCT,
            current_open_risk_usdt=current_heat,
            max_total_open_risk_pct=MAX_TOTAL_OPEN_RISK_PCT,
            available_margin_usdt=self.available_usdt,
            leverage=LEVERAGE,
            max_margin_per_trade=max_margin_per_trade,
        )
        contracts = _round_step(target_notional / (mid_pr * ct_val), info["lotSz"], ROUND_DOWN)
        if contracts < min_sz_d:
            min_notional = float(min_sz_d) * ct_val * mid_pr
            if minimum_contract_is_acceptable(
                min_notional_usdt=min_notional,
                equity_usdt=equity,
                current_open_risk_usdt=current_heat,
                max_total_open_risk_pct=MAX_TOTAL_OPEN_RISK_PCT,
                max_single_risk_pct=MAX_SINGLE_TRADE_RISK_PCT,
                sl_pct=sl_pct,
                fee_pct=RISK_FEE_ROUND_TRIP_PCT,
                slippage_pct=RISK_SLIPPAGE_PCT,
                available_margin_usdt=self.available_usdt,
                leverage=LEVERAGE,
            ):
                contracts = min_sz_d
                logger.info(
                    f"ℹ️ {inst_id}: minSz {info['minSz']} accepted because projected risk "
                    f"${projected_risk_usdt(min_notional, sl_pct, RISK_FEE_ROUND_TRIP_PCT, RISK_SLIPPAGE_PCT):,.2f} "
                    f"<= hard cap ${equity * MAX_SINGLE_TRADE_RISK_PCT:,.2f}"
                )
            else:
                logger.warning(
                    f"⚠️ {inst_id}: target risk size below minSz {info['minSz']} "
                    f"(target notional ${target_notional:,.2f}, min ${min_notional:,.2f}, heat ${current_heat:,.2f}) -> skip"
                )
                self.trade_cooldowns[inst_id] = time.time() + ORDER_FAIL_COOLDOWN_SEC
                return False
        notional_val = float(contracts) * ct_val * mid_pr
        margin_to_use = notional_val / LEVERAGE
        trade_risk = projected_risk_usdt(notional_val, sl_pct, RISK_FEE_ROUND_TRIP_PCT, RISK_SLIPPAGE_PCT)
        if current_heat + trade_risk > equity * MAX_TOTAL_OPEN_RISK_PCT + 1e-9:
            logger.warning(
                f"🛑 [Portfolio Heat Block] {inst_id}: new ${trade_risk:,.2f} + open ${current_heat:,.2f} "
                f"> limit ${equity * MAX_TOTAL_OPEN_RISK_PCT:,.2f} -> skip"
            )
            self.trade_cooldowns[inst_id] = time.time() + ORDER_FAIL_COOLDOWN_SEC
            return False

        is_long = side == "buy"
        _, _, best_bid, best_ask = self.get_bbo_and_imbalance(inst_id)
        tick = info["tickSz"]
        tick_d = Decimal(tick)

        # Passive 1-Tick Offset: prevents 0ms Post-Only auto-cancel on micro BBO drift while capturing 0.02% Maker fee
        if is_long and best_bid > 0:
            passive_px_d = _round_step(best_bid, tick, ROUND_DOWN) - tick_d
            if best_ask > 0:
                ask_minus_tick = _round_step(best_ask, tick, ROUND_DOWN) - tick_d
                if passive_px_d > ask_minus_tick:
                    passive_px_d = ask_minus_tick
        elif (not is_long) and best_ask > 0:
            passive_px_d = _round_step(best_ask, tick, ROUND_UP) + tick_d
            if best_bid > 0:
                bid_plus_tick = _round_step(best_bid, tick, ROUND_UP) + tick_d
                if passive_px_d < bid_plus_tick:
                    passive_px_d = bid_plus_tick
        else:
            passive_px_d = _round_step(mid_pr, tick, ROUND_DOWN if is_long else ROUND_UP)

        maker_px = float(passive_px_d)
        sl_px = maker_px * (1 - sl_pct) if is_long else maker_px * (1 + sl_pct)
        attach_algo = [{
            "slTriggerPx": _fmt(_round_step(sl_px, tick, ROUND_HALF_UP), tick),
            "slOrdPx": "-1",
            "slTriggerPxType": "last",
        }]
        if WAVE_TP_ENABLED:
            tp_px = maker_px * (1 + tp_pct) if is_long else maker_px * (1 - tp_pct)
            attach_algo[0].update({
                "tpTriggerPx": _fmt(_round_step(tp_px, tick, ROUND_HALF_UP), tick),
                "tpOrdPx": "-1",
                "tpTriggerPxType": "last",
            })

        pos_type_desc = "LONG" if is_long else "SHORT"
        sz_str = _fmt(contracts, info["lotSz"])
        approx_usd = float(contracts) * ct_val * maker_px

        maker_order = {
            "instId": inst_id,
            "tdMode": "isolated",
            "side": side,
            "posSide": "net",
            "ordType": "post_only",
            "px": _fmt(passive_px_d, tick),
            "sz": sz_str,
            "clOrdId": f"jevv4m{int(time.time() * 1000)}",
            "attachAlgoOrds": attach_algo,
        }
        logger.info(
            f"📤 [Argon V4-Pro Maker] {pos_type_desc} {sz_str} ct {inst_id} "
            f"(Margin ~${margin_to_use:,.0f} x {LEVERAGE}x = ~${approx_usd:,.0f}) Post-Only @ ${maker_order['px']} "
            f"| Risk ~${trade_risk:,.2f} ({trade_risk/equity*100 if equity > 0 else 0:.2f}%), Heat ${current_heat:,.2f} "
            f"| SL {attach_algo[0]['slTriggerPx']} (-{sl_pct*100:.2f}%) / "
            f"{'Wave TP ' + attach_algo[0]['tpTriggerPx'] + f' (+{tp_pct*100:.1f}%)' if WAVE_TP_ENABLED else 'No TP (1H MA20 exit)'}"
        )
        res = self._request("POST", "/api/v5/trade/order", data=maker_order)
        maker_filled = False
        order_already_closed = False
        rem_contracts = contracts
        entry_avg_px = maker_px

        if res and res.get("data"):
            m_oid = res["data"][0].get("ordId")
            for _ in range(3):
                time.sleep(1.1)
                chk = self._request("GET", f"/api/v5/trade/order?instId={inst_id}&ordId={m_oid}")
                if chk and chk.get("data"):
                    st_info = chk["data"][0]
                    st_state = st_info.get("state", "")
                    acc_fill = Decimal(str(st_info.get("accFillSz") or "0"))
                    avg_fill = float(st_info.get("avgPx") or 0.0)
                    if avg_fill > 0:
                        entry_avg_px = avg_fill
                    if st_state == "filled":
                        maker_filled = True
                        order_already_closed = True
                        logger.info(f"✅ [Maker Order Filled] {inst_id} {pos_type_desc} filled as Maker (0.02% fee)! ID: {m_oid}")
                        break
                    elif st_state == "canceled":
                        order_already_closed = True
                        rem_contracts = contracts - acc_fill
                        break
            if not maker_filled and not order_already_closed:
                self._request("POST", "/api/v5/trade/cancel-order", data={"instId": inst_id, "ordId": m_oid})
                chk = self._request("GET", f"/api/v5/trade/order?instId={inst_id}&ordId={m_oid}")
                if chk and chk.get("data"):
                    st_info = chk["data"][0]
                    avg_fill = float(st_info.get("avgPx") or 0.0)
                    if avg_fill > 0:
                        entry_avg_px = avg_fill
                    if st_info.get("state") == "filled":
                        maker_filled = True
                    else:
                        acc_fill = Decimal(str(st_info.get("accFillSz") or "0"))
                        rem_contracts = contracts - acc_fill

        if not maker_filled and rem_contracts >= Decimal(info["minSz"]):
            # Recompute SL/TP off latest mid price for market fallback accuracy
            curr_mid, _ = self.get_ticker_and_imbalance(inst_id)
            ref_px = curr_mid if curr_mid > 0 else maker_px
            entry_avg_px = ref_px
            m_sl_px = ref_px * (1 - sl_pct) if is_long else ref_px * (1 + sl_pct)
            m_attach = [{
                "slTriggerPx": _fmt(_round_step(m_sl_px, tick, ROUND_HALF_UP), tick),
                "slOrdPx": "-1",
                "slTriggerPxType": "last",
            }]
            if WAVE_TP_ENABLED:
                m_tp_px = ref_px * (1 + tp_pct) if is_long else ref_px * (1 - tp_pct)
                m_attach[0].update({
                    "tpTriggerPx": _fmt(_round_step(m_tp_px, tick, ROUND_HALF_UP), tick),
                    "tpOrdPx": "-1",
                    "tpTriggerPxType": "last",
                })
            order_data = {
                "instId": inst_id,
                "tdMode": "isolated",
                "side": side,
                "posSide": "net",
                "ordType": "market",
                "sz": _fmt(rem_contracts, info["lotSz"]),
                "clOrdId": f"jevv4t{int(time.time() * 1000)}",
                "attachAlgoOrds": m_attach,
            }
            logger.info(f"⚡ [Maker Unfilled -> Market Fallback] Submitting Market {pos_type_desc} {order_data['sz']} ct {inst_id}")
            res = self._request("POST", "/api/v5/trade/order", data=order_data)
        else:
            order_data = maker_order

        oid = (res or {}).get("data", [{}])[0].get("ordId") if (res or {}).get("data") else None
        if oid:
            logger.info(f"✅ [V4-Pro Wave Entry Accepted] Order ID: {oid} | {inst_id} {pos_type_desc} ({LEVERAGE}x Isolated)")
            self.order_fail_counts.pop(inst_id, None)
            self.trade_cooldowns[inst_id] = time.time() + ENTRY_COOLDOWN_SEC
            self.position_entry_times[inst_id] = time.time()
            self.peak_prices[inst_id] = entry_avg_px
            self.trough_prices[inst_id] = entry_avg_px
            self.position_params[inst_id] = {
                "sl_pct": sl_pct,
                "tp_pct": tp_pct if WAVE_TP_ENABLED else 0.0,
                "trail_arm": STAGE3_TRAIL_ARM_PCT,
                "trail_drop": STAGE3_TRAIL_DROP_PCT,
                "risk_usdt": trade_risk,
                "notional_usdt": notional_val,
            }
            if btc_wave_ts and sym_wave_ts:
                self._mark_wave_traded(inst_id, btc_wave_ts, sym_wave_ts)
            self._append_ledger("ENTRY", {
                "inst_id": inst_id,
                "side": pos_type_desc,
                "order_id": oid,
                "contracts": float(contracts),
                "entry_px_est": round(entry_avg_px, 10),
                "notional_usdt": round(notional_val, 8),
                "margin_usdt": round(margin_to_use, 8),
                "risk_usdt": round(trade_risk, 8),
                "open_heat_usdt": round(current_heat + trade_risk, 8),
                "sl_pct": round(sl_pct * 100.0, 6),
                "tp_pct": round(tp_pct * 100.0, 6) if WAVE_TP_ENABLED else 0.0,
                "equity_usdt": round(equity, 8),
            })
            self._save_state()
            send_telegram(
                f"[OKX V4-Pro {LEVERAGE}x] 🚀 {pos_type_desc} {inst_id} {order_data['sz']}ct "
                f"(증거금 ~${margin_to_use:,.0f} / 노출 ~${approx_usd:,.0f} / 리스크 ~${trade_risk:,.2f}) "
                f"@ {entry_avg_px:,.4f} | SL -{sl_pct*100:.2f}%"
            )
            return True

        fail_n = self.order_fail_counts.get(inst_id, 0) + 1
        self.order_fail_counts[inst_id] = fail_n
        fail_cd = min(ORDER_FAIL_COOLDOWN_SEC * (2 ** (fail_n - 1)), 3600.0)
        self.trade_cooldowns[inst_id] = time.time() + fail_cd
        logger.error(f"❌ [Order Failed] {inst_id} {pos_type_desc}: {self.last_error} -> cooldown {fail_cd/60:.0f}m (연속 실패 {fail_n}회)")
        send_telegram(f"[OKX Demo] ❌ {inst_id} 주문 실패: {self.last_error}",
                      dedupe_key=f"okx_order_fail_{inst_id}", min_interval_sec=1800)
        if "51010" in self.last_error or "account mode" in self.last_error.lower():
            self.trading_enabled = False
        return False

    def close_position(self, inst_id: str, reason: str = "MANUAL", pnl_pct: float = 0.0) -> bool:
        """Closes open position with Market Close Position API (also cancels pending close orders)"""
        logger.info(f"🔄 [Closing Demo Position] {inst_id} | Reason: {reason}")
        close_started_ms = int(time.time() * 1000)
        data = {
            "instId": inst_id,
            "mgnMode": "isolated",
            "autoCxl": True,
        }
        res = self._request("POST", "/api/v5/trade/close-position", data=data)
        if not res:
            logger.error(f"❌ [Close Failed] {inst_id}: {self.last_error} (exchange SL/TP still active; retry next loop)")
            send_telegram(f"[OKX Demo] ❌ {inst_id} 청산 실패 ({reason}): {self.last_error}",
                          dedupe_key=f"okx_close_fail_{inst_id}", min_interval_sec=600)
            return False

        logger.info(f"✅ [Demo Position Closed] {inst_id} successfully closed ({reason}).")
        self._cancel_algo_orders(inst_id)
        prev_pos = self.positions.pop(inst_id, {})
        avg_px = float(prev_pos.get("avgPx", 0.0) or 0.0)
        is_long = float(prev_pos.get("pos", 0.0) or 0.0) > 0
        side_desc = "LONG" if is_long else "SHORT"
        entry_time = self.position_entry_times.get(inst_id, 0.0)
        not_before = int(entry_time * 1000) - 5000 if entry_time else close_started_ms - 5000
        h = self._fetch_latest_position_history(inst_id, not_before)
        if h is None:
            time.sleep(0.5)
            h = self._fetch_latest_position_history(inst_id, not_before)
        if h:
            breakdown = self._history_breakdown(h)
            src = "positions-history"
            close_avg_px = float(h.get("closeAvgPx") or 0.0)
        else:
            notional = self._position_notional_usdt(prev_pos)
            gross = notional * pnl_pct
            fees = notional * RISK_FEE_ROUND_TRIP_PCT
            breakdown = {
                "gross_pnl_usdt": gross,
                "fee_usdt": -fees,
                "funding_usdt": 0.0,
                "liq_penalty_usdt": 0.0,
                "net_pnl_usdt": gross - fees,
            }
            src = "estimate"
            close_avg_px = 0.0
        net_pnl = breakdown["net_pnl_usdt"]
        mfe_pct, mae_pct = self._mfe_mae_pct(inst_id, avg_px, is_long)
        is_loss = net_pnl < 0
        self._record_trade_outcome(inst_id, side_desc, is_loss)
        cooldown = POST_LOSS_COOLDOWN_SEC if is_loss else NORMAL_COOLDOWN_SEC
        self.trade_cooldowns[inst_id] = time.time() + cooldown
        self._record_realized_pnl(net_pnl, inst_id, reason, src)
        self._append_ledger("EXIT", {
            "inst_id": inst_id,
            "side": side_desc,
            "reason": reason,
            "source": src,
            "entry_avg_px": avg_px,
            "close_avg_px": close_avg_px,
            "mfe_pct": round(mfe_pct * 100.0, 6),
            "mae_pct": round(mae_pct * 100.0, 6),
            **{k: round(v, 8) for k, v in breakdown.items()},
            "equity_usdt": round(self.usdt_equity, 8),
            "daily_realized_pnl": round(self.daily_realized_pnl, 8),
            "weekly_realized_pnl": round(self.weekly_realized_pnl, 8),
        })
        self.peak_prices.pop(inst_id, None)
        self.trough_prices.pop(inst_id, None)
        self.position_entry_times.pop(inst_id, None)
        self.position_params.pop(inst_id, None)
        self._save_state()
        roe_est = pnl_pct * LEVERAGE * 100.0
        send_telegram(
            f"[OKX V4-Pro] {inst_id} 청산 ({reason}) | 현물 {pnl_pct*100:+.2f}% "
            f"(추정 ROE {roe_est:+.1f}%) | 순손익 {net_pnl:+.2f} USDT ({src})"
        )
        return True

    # ── V5 WAVE POSITION MANAGEMENT ──
    def manage_open_positions(self):
        """Monitors open positions with hard SL and confirmed 1H MA20 structure exit."""
        if not self.positions:
            return

        btc_1h = self.analyze_1h_wave(self.fetch_bars("BTC-USDT-SWAP", bar="1H", limit=60))

        for inst_id, pos in list(self.positions.items()):
            try:
                avg_px = float(pos.get("avgPx", 0.0) or 0.0)
                pos_amt = float(pos.get("pos", 0.0) or 0.0)
                curr_px, _ = self.get_ticker_and_imbalance(inst_id)

                if avg_px <= 0 or curr_px <= 0:
                    continue

                # Ensure any inherited / active position marks its wave so it won't re-enter the same wave after closing
                sym_1h = self.analyze_1h_wave(self.fetch_bars(inst_id, bar="1H", limit=60))
                if btc_1h and sym_1h:
                    self._mark_wave_traded(inst_id, btc_1h["wave_ts"], sym_1h["wave_ts"])

                is_long = pos_amt > 0
                spot_pl_pct = (curr_px - avg_px) / avg_px if is_long else (avg_px - curr_px) / avg_px

                default_sl = BASE_SL_CORE if inst_id in ("BTC-USDT-SWAP", "ETH-USDT-SWAP") else BASE_SL_ALT
                params = self.position_params.get(inst_id, {})
                pos_sl_pct = float(params.get("sl_pct", default_sl))
                pos_tp_pct = float(params.get("tp_pct", WAVE_TP_PCT))

                # Peak price tracking
                peak = self.peak_prices.get(inst_id, avg_px)
                if is_long and curr_px > peak:
                    self.peak_prices[inst_id] = curr_px
                    peak = curr_px
                elif (not is_long) and curr_px < peak:
                    self.peak_prices[inst_id] = curr_px
                    peak = curr_px

                trough = self.trough_prices.get(inst_id, avg_px)
                if (is_long and curr_px < trough) or ((not is_long) and curr_px > trough):
                    self.trough_prices[inst_id] = curr_px
                    trough = curr_px
                self.position_params.setdefault(inst_id, {})["mfe_pct"] = round(
                    (max(0.0, (peak - avg_px) / avg_px if is_long else (avg_px - peak) / avg_px)) * 100.0, 6
                )
                self.position_params[inst_id]["mae_pct"] = round(
                    (min(0.0, (trough - avg_px) / avg_px if is_long else (avg_px - trough) / avg_px)) * 100.0, 6
                )

                peak_pl_pct = (peak - avg_px) / avg_px if is_long else (avg_px - peak) / avg_px

                # 1. Hard Stop-Loss backup (Exchange SL is primary)
                if spot_pl_pct <= -pos_sl_pct:
                    logger.warning(f"🛑 [Stop Loss Trigger] {inst_id} {'LONG' if is_long else 'SHORT'} Spot PnL: {spot_pl_pct*100:+.2f}% <= -{pos_sl_pct*100:.2f}%")
                    self.close_position(inst_id, f"STOP_LOSS_{spot_pl_pct*100:.2f}PCT", spot_pl_pct)
                    continue

                # 2. Optional hard take-profit backup
                if WAVE_TP_ENABLED and spot_pl_pct >= pos_tp_pct:
                    logger.info(f"🎯 [Wave Take Profit] {inst_id} {'LONG' if is_long else 'SHORT'} Spot PnL: +{spot_pl_pct*100:.2f}% >= +{pos_tp_pct*100:.1f}%")
                    self.close_position(inst_id, f"WAVE_TP_{spot_pl_pct*100:+.2f}PCT", spot_pl_pct)
                    continue

                # 3. Stage 2 Mid-Wave Profit Floor (Peak >= +2.20% spot -> Lock +1.10% spot = +22% ROE)
                if STAGE_EXITS_ENABLED and peak_pl_pct >= STAGE2_MID_ARM_PCT and spot_pl_pct <= STAGE2_MID_FLOOR_PCT:
                    logger.info(
                        f"🛡️ [Wave Stage-2 Profit Lock] {inst_id} {'LONG' if is_long else 'SHORT'} Peak +{peak_pl_pct*100:.2f}% "
                        f"pulled back to +{spot_pl_pct*100:.2f}% (Floor +{STAGE2_MID_FLOOR_PCT*100:.2f}%)."
                    )
                    self.close_position(inst_id, f"WAVE_STAGE2_LOCK_{spot_pl_pct*100:+.2f}PCT", spot_pl_pct)
                    continue

                # 4. Stage 1 Early Breakeven Floor (Peak >= +1.50% spot -> Lock +0.25% spot = +5% ROE)
                if STAGE_EXITS_ENABLED and peak_pl_pct >= STAGE1_BE_ARM_PCT and spot_pl_pct <= STAGE1_BE_FLOOR_PCT:
                    logger.info(
                        f"🛡️ [Wave Stage-1 BE Lock] {inst_id} {'LONG' if is_long else 'SHORT'} Peak +{peak_pl_pct*100:.2f}% "
                        f"pulled back to +{spot_pl_pct*100:.2f}% (Floor +{STAGE1_BE_FLOOR_PCT*100:.2f}%)."
                    )
                    self.close_position(inst_id, f"WAVE_STAGE1_BE_{spot_pl_pct*100:+.2f}PCT", spot_pl_pct)
                    continue

                # 5. Stage 3 Big-Wave Trailing Stop (Peak >= +3.20% spot -> 1.00% pullback harvest)
                if STAGE_EXITS_ENABLED and peak_pl_pct >= STAGE3_TRAIL_ARM_PCT:
                    pullback = (peak - curr_px) / peak if is_long else (curr_px - peak) / peak
                    if pullback >= STAGE3_TRAIL_DROP_PCT and spot_pl_pct >= STAGE3_MIN_PROFIT_PCT:
                        logger.info(
                            f"🔒 [Wave Stage-3 Trail Harvest] {inst_id} {'LONG' if is_long else 'SHORT'}: "
                            f"Entry ${avg_px:,.4f} -> Peak ${peak:,.4f} (+{peak_pl_pct*100:.2f}%) -> Curr ${curr_px:,.4f} (+{spot_pl_pct*100:.2f}%)"
                        )
                        self.close_position(inst_id, f"WAVE_TRAIL_{spot_pl_pct*100:+.2f}PCT", spot_pl_pct)
                        continue

                # 6. Stage 4 Confirmed 1H MA20 Structure Reversal Early Exit (cuts fake waves early around -0.8%)
                if sym_1h is not None:
                    c1h, ma5_1h, ma20_1h = sym_1h["close"], sym_1h["ma5"], sym_1h["ma20"]
                    if is_long and (c1h < ma20_1h * (1.0 - MA20_REV_BUFFER_PCT)) and (ma5_1h < ma20_1h):
                        logger.info(
                            f"✂️ [1H MA20 Structure Exit] {inst_id} LONG invalidated: 1H Close ${c1h:,.4f} < MA20 ${ma20_1h:,.4f} & MA5 < MA20 "
                            f"(Spot PnL {spot_pl_pct*100:+.2f}%)"
                        )
                        self.close_position(inst_id, f"1H_MA20_EXIT_{spot_pl_pct*100:+.2f}PCT", spot_pl_pct)
                        continue
                    elif (not is_long) and (c1h > ma20_1h * (1.0 + MA20_REV_BUFFER_PCT)) and (ma5_1h > ma20_1h):
                        logger.info(
                            f"✂️ [1H MA20 Structure Exit] {inst_id} SHORT invalidated: 1H Close ${c1h:,.4f} > MA20 ${ma20_1h:,.4f} & MA5 > MA20 "
                            f"(Spot PnL {spot_pl_pct*100:+.2f}%)"
                        )
                        self.close_position(inst_id, f"1H_MA20_EXIT_{spot_pl_pct*100:+.2f}PCT", spot_pl_pct)
                        continue

            except Exception as e:
                logger.error(f"Position manage error for {inst_id}: {e}")

    # ── 4-MAJOR DOMINANCE WAVE ENTRY ENGINE ──
    def evaluate_trading_opportunities(self):
        """Evaluates 4-Major Synchronization (BTC, ETH, SOL, XRP) on 1H MA(5/10/20) + 15m EMA(9/20)."""
        if not self._slot_available():
            return

        block = self._risk_block_reason()
        if block:
            now = time.time()
            if now - self.last_risk_block_warn >= 300:
                self.last_risk_block_warn = now
                logger.warning(f"🛑 [New Entries Blocked] {block}")
            return

        # 1. Fetch & Analyze 1H Wave Structure across all 4 Majors
        w1h_map: Dict[str, Dict[str, Any]] = {}
        for sym in TARGET_SYMBOLS:
            info_1h = self.analyze_1h_wave(self.fetch_bars(sym, bar="1H", limit=80))
            if info_1h is None:
                return
            w1h_map[sym] = info_1h

        btc_1h = w1h_map["BTC-USDT-SWAP"]
        btc_15m = self.analyze_15m_wave(self.fetch_bars("BTC-USDT-SWAP", bar="15m", limit=60))
        if btc_15m is None:
            return
        regime = self.btc_daily_regime()

        bull_cnt, bear_cnt = 0, 0
        for sym in TARGET_SYMBOLS:
            m = w1h_map[sym]
            if m["close"] > m["ma20"] and m["ma5"] > m["ma10"]:
                bull_cnt += 1
            elif m["close"] < m["ma20"] and m["ma5"] < m["ma10"]:
                bear_cnt += 1

        bc1h, bm5, bm10, bm20 = btc_1h["close"], btc_1h["ma5"], btc_1h["ma10"], btc_1h["ma20"]
        bslope = btc_1h["ma20_slope3"]
        bm5_slope = btc_1h["ma5_slope1"]
        bret3 = btc_1h["ret3"]
        btc_wave_ts = btc_1h["wave_ts"]
        btc_wave_age = btc_1h["wave_age"]
        bc15, be20_15 = btc_15m["close"], btc_15m["ema20"]

        bull_vel_ok = (bslope >= MIN_BTC_MA20_SLOPE3_PCT) or (
            bslope >= MIN_BTC_IMPULSE_SLOPE3_PCT
            and bm5_slope >= MIN_BTC_IMPULSE_MA5_SLOPE1_PCT
            and bret3 >= MIN_BTC_IMPULSE_RET3_PCT
        )
        bear_vel_ok = (bslope <= -MIN_BTC_MA20_SLOPE3_PCT) or (
            bslope <= -MIN_BTC_IMPULSE_SLOPE3_PCT
            and bm5_slope <= -MIN_BTC_IMPULSE_MA5_SLOPE1_PCT
            and bret3 <= -MIN_BTC_IMPULSE_RET3_PCT
        )

        sync_bull = (
            (bc1h > bm20) and (bm5 > bm10) and (bc15 > be20_15)
            and (bull_cnt >= MIN_SYNC_BREADTH) and bull_vel_ok
        )
        sync_bear = (
            (bc1h < bm20) and (bm5 < bm10) and (bc15 < be20_15)
            and (bear_cnt >= MIN_SYNC_BREADTH) and bear_vel_ok
        )
        is_impulse_brk = (sync_bull and bslope < MIN_BTC_MA20_SLOPE3_PCT) or (sync_bear and bslope > -MIN_BTC_MA20_SLOPE3_PCT)
        max_dist_cap = MAX_IMPULSE_DIST_1H_MA20_PCT if is_impulse_brk else MAX_DIST_1H_MA20_PCT

        now = time.time()
        if now - self.last_radar_log >= RADAR_LOG_INTERVAL_SEC:
            self.last_radar_log = now
            regime_str = f"SYNC_BULL ({MIN_SYNC_BREADTH}/4)" if sync_bull else (f"SYNC_BEAR ({MIN_SYNC_BREADTH}/4)" if sync_bear else f"STANDBY ({bull_cnt}L/{bear_cnt}S, need {MIN_SYNC_BREADTH}/4)")
            daily_str = (
                f"BTC 20D EMA=${regime['ema']:,.1f} (Slope3D {regime['slope_pct']:+.2f}%, "
                f"L={'OK' if regime['long_ok'] else 'NO'}/S={'OK' if regime['short_ok'] else 'NO'})"
                if regime else "BTC 20D EMA unavailable -> entries blocked"
            )
            logger.info(
                f"📡 [V5 Daily-Regime Radar] {regime_str} | BTC 1H=${bc1h:,.1f} vs MA20=${bm20:,.1f} "
                f"(Slope3H {bslope:+.2f}%, MA5Slp {bm5_slope:+.2f}%, Age {btc_wave_age}h) | {daily_str} | "
                f"Open Slots: {len(self.positions)}/{MAX_CONCURRENT_POSITIONS}"
            )

        if not (sync_bull or sync_bear):
            return

        if regime is None:
            now = time.time()
            if now - self.last_daily_block_warn >= 300:
                self.last_daily_block_warn = now
                logger.warning("🛑 [Daily Regime Block] BTC daily EMA unavailable -> skip new entries")
            return

        daily_ok = regime["long_ok"] if sync_bull else regime["short_ok"]
        if not daily_ok:
            now = time.time()
            if now - self.last_daily_block_warn >= 300:
                self.last_daily_block_warn = now
                logger.info(
                    f"⏳ [Daily Regime Block] {'LONG' if sync_bull else 'SHORT'} wave ignored: "
                    f"BTC 20D EMA ${regime['ema']:,.1f}, slope3D {regime['slope_pct']:+.2f}%"
                )
            return

        target_side = "buy" if sync_bull else "sell"
        pos_desc = "LONG" if sync_bull else "SHORT"
        candidates: List[Tuple[str, float, float, Dict[str, Any], Dict[str, Any]]] = []

        # 2. Filter & Score Individual Coins in TARGET_SYMBOLS
        for sym in TARGET_SYMBOLS:
            if sym in self.positions:
                continue
            if now < self.trade_cooldowns.get(sym, 0.0):
                continue
            lock_key = f"{sym}:{pos_desc}"
            if now < self.side_lockout_until.get(lock_key, 0.0):
                continue

            s1h = w1h_map[sym]
            sym_wave_ts = s1h["wave_ts"]
            sym_wave_age = s1h["wave_age"]

            # One-Wave-One-Shot Lock: never re-enter an already-traded wave
            if f"{sym}:BTC:{btc_wave_ts}" in self.traded_waves or f"{sym}:SYM:{sym_wave_ts}" in self.traded_waves:
                continue
            if btc_wave_age > MAX_BTC_WAVE_AGE_HRS and sym_wave_age > MAX_SYM_LATE_WAVE_HRS:
                continue

            if s1h["ma_spread_pct"] < MIN_1H_MA_SPREAD_PCT or s1h["adx"] < MIN_1H_ADX:
                continue

            s15 = btc_15m if sym == "BTC-USDT-SWAP" else self.analyze_15m_wave(self.fetch_bars(sym, bar="15m", limit=60))
            if s15 is None:
                continue

            c1h, ma5, ma10, ma20 = s1h["close"], s1h["ma5"], s1h["ma10"], s1h["ma20"]
            dma20, m5_slp = s1h["dist_ma20"], s1h["ma5_slope1"]
            c15, e9_15, e20_15 = s15["close"], s15["ema9"], s15["ema20"]

            if sync_bull:
                ok_1h = (
                    (c1h > ma20) and (c1h > ma5) and (ma5 > ma10 >= ma20 * 0.999)
                    and (m5_slp > 0.05) and (MIN_DIST_1H_MA20_PCT <= dma20 <= max_dist_cap)
                )
                ok_15 = (c15 > e9_15 > e20_15) and (s15["adx"] >= MIN_15M_ADX)
            else:
                ok_1h = (
                    (c1h < ma20) and (c1h < ma5) and (ma5 < ma10 <= ma20 * 1.001)
                    and (m5_slp < -0.05) and (-max_dist_cap <= dma20 <= -MIN_DIST_1H_MA20_PCT)
                )
                ok_15 = (c15 < e9_15 < e20_15) and (s15["adx"] >= MIN_15M_ADX)

            if not (ok_1h and ok_15) or s15["vol_ratio"] < MIN_15M_VOL_RATIO:
                continue

            score = s1h["ret3"] if sync_bull else -s1h["ret3"]
            candidates.append((sym, score, s15["atr_ratio"], s1h, s15))

        if not candidates:
            return

        # 3. Rank by 3H Wave Momentum Leadership & Execute Top Slots
        candidates.sort(key=lambda x: x[1], reverse=True)
        opened_count = 0
        for sym, score, atr15, s1h, s15 in candidates:
            if not self._slot_available(opened_count):
                break

            funding_rate = self.get_funding_rate(sym)
            if sync_bull and funding_rate >= MAX_LONG_FUNDING_RATE:
                logger.info(f"⏳ [Funding Squeeze Guard] {sym} LONG skipped: Funding {funding_rate*100:+.4f}% >= +{MAX_LONG_FUNDING_RATE*100:.2f}%")
                continue
            if sync_bear and funding_rate <= MIN_SHORT_FUNDING_RATE:
                logger.info(f"⏳ [Funding Squeeze Guard] {sym} SHORT skipped: Funding {funding_rate*100:+.4f}% <= {MIN_SHORT_FUNDING_RATE*100:.2f}%")
                continue

            base_sl = BASE_SL_CORE if sym in ("BTC-USDT-SWAP", "ETH-USDT-SWAP") else BASE_SL_ALT
            dyn_sl_pct = max(base_sl, min(MAX_DYN_SL_PCT, DYN_SL_ATR_MULT * atr15))
            px, imb = self.get_ticker_and_imbalance(sym)
            ref_px = px if px > 0 else s15["close"]

            logger.info(
                f"🌊 [{MIN_SYNC_BREADTH}/4 Wave Signal] {sym} {pos_desc} ({LEVERAGE}x) | Px: ${ref_px:,.4f} | "
                f"1H DistMA20: {s1h['dist_ma20']:+.2f}% | Spread: {s1h['ma_spread_pct']:.2f}% | "
                f"ADX(1H/15m): {s1h['adx']:.1f}/{s15['adx']:.1f} | 3H Ret: {s1h['ret3']:+.2f}% | "
                f"WaveAge(BTC/{sym[:3]}): {btc_wave_age}h/{s1h['wave_age']}h | DynSL: -{dyn_sl_pct*100:.2f}%"
            )

            # 4. Universal Jev AI Gate (4-Major Wave Context)
            if self.jev_gate:
                try:
                    decision = self.jev_gate.evaluate_sync(
                        asset_class="CRYPTO",
                        symbol=sym,
                        side=pos_desc,
                        context_metrics={
                            "wave_sync_breadth": f"{bull_cnt if sync_bull else bear_cnt}/4",
                            "btc_ma20_slope3_pct": bslope,
                            "price": ref_px,
                            "dist_1h_ma20_pct": s1h["dist_ma20"],
                            "ma_spread_pct": s1h["ma_spread_pct"],
                            "adx_1h": s1h["adx"],
                            "adx_15m": s15["adx"],
                            "vol_ratio": s15["vol_ratio"],
                            "ret3_1h_pct": s1h["ret3"],
                            "btc_wave_age_h": btc_wave_age,
                            "sym_wave_age_h": s1h["wave_age"],
                            "funding_rate_pct": funding_rate * 100.0,
                            "imbalance": imb,
                            "atr_pct": atr15 * 100.0,
                            "dist_ema20_pct": s15["dist_ema20_pct"],
                            "recent_return_pct": s15["recent_return_pct"],
                            "lookback_bars": s15["lookback_bars"],
                            "macro": f"{MIN_SYNC_BREADTH}/4 Majors Synchronized {'BULL' if sync_bull else 'BEAR'} Wave (BTC 1H Slope {bslope:+.2f}%)",
                            "sl_pct": dyn_sl_pct * 100.0,
                            "tp_pct": WAVE_TP_PCT * 100.0 if WAVE_TP_ENABLED else 0.0,
                        },
                    )
                except Exception as e:
                    logger.warning(f"Jev gate exception for {sym}: {e} -> veto")
                    self.trade_cooldowns[sym] = time.time() + JEV_ERROR_COOLDOWN_SEC
                    continue

                if not decision.approved:
                    cd = JEV_ERROR_COOLDOWN_SEC if decision.is_fallback else JEV_VETO_COOLDOWN_SEC
                    self.trade_cooldowns[sym] = time.time() + cd
                    logger.info(f"🚫 [Jev Veto] {sym} {pos_desc} (conf {decision.confidence:.2f}, {decision.latency_ms:.0f}ms): {decision.reason} -> cooldown {cd/60:.0f}m")
                    continue
                logger.info(f"🧠 [Jev Approved] {sym} {pos_desc} (conf {decision.confidence:.2f}, {decision.latency_ms:.0f}ms): {decision.reason}")

            if self.open_position(
                sym,
                side=target_side,
                sl_pct=dyn_sl_pct,
                tp_pct=WAVE_TP_PCT,
                btc_wave_ts=btc_wave_ts,
                sym_wave_ts=s1h["wave_ts"],
            ):
                opened_count += 1

    # ── MAIN LOOP ──
    def run(self):
        """24/7 Supervisor Loop"""
        if not (self.api_key and self.secret_key and self.passphrase):
            logger.critical("⛔ OKX_DEMO_API_KEY / OKX_DEMO_SECRET / OKX_DEMO_PASSPHRASE missing in .env -> exiting")
            sys.exit(1)

        logger.info(f"🟢 [OKX Jev Demo Trader] V5 Daily-Regime ({LEVERAGE}x Isolated) running 24/7...")
        send_telegram(
            f"[OKX Demo] V5 Daily-Regime 가동 ({LEVERAGE}x 격리, 거래당 리스크 {RISK_PER_TRADE_PCT*100:.2f}%, "
            f"총오픈 {MAX_TOTAL_OPEN_RISK_PCT*100:.2f}%, 일손실한도 {DAILY_LOSS_LIMIT_PCT*100:.1f}%, "
            f"4대 메이저 {MIN_SYNC_BREADTH}/4 + BTC 20D EMA 추세 필터)"
        )
        self.sync_account()
        self.check_account_mode()

        while True:
            try:
                self.sync_account()
                self.sync_positions()
                self._snapshot_if_due()
                self.manage_open_positions()
                if self.check_account_mode():
                    self.evaluate_trading_opportunities()
                self._save_state()
                time.sleep(LOOP_INTERVAL_SEC)
            except KeyboardInterrupt:
                logger.info("🛑 Stopping OKX Jev Demo Trader...")
                self._save_state()
                break
            except Exception as e:
                logger.error(f"Loop exception: {e}", exc_info=True)
                send_telegram(f"[OKX Demo] ❌ 루프 예외: {e}", dedupe_key="okx_loop_exc", min_interval_sec=1800)
                time.sleep(10.0)


if __name__ == "__main__":
    trader = OKXJevDemoTrader()
    trader.run()
