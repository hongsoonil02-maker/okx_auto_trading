# -*- coding: utf-8 -*-
"""
okx_jev_demo_trader.py — Autonomous OKX Demo Quant Engine with Bidirectional Jev Alpha
Argon V4-Pro (S8): 4-Major Dominance Synchronization (3/4) + 1H MA(5/10/20) Wave Rider (10x Isolated)

Backtest Validated 2026-11-09 (122D, 2026-07-10~11-09, 2,938 1H bars, 상승/하락/횡보 포함):
- S8 (10x, 3/4 gate, SL 2.5%/dyn cap 4.0%, TP +5%): 64 Trades, 68.8% WR, PF 1.83, +49.2%, MDD -12.5%, 0 Liquidations
- Regime split: first-half PF 1.51 / second-half PF 1.88 (MDD -10.5% / -9.8%)
- Supersedes prior 30D/45D claims (단기 구간 편향으로 폐기) — see agrolib/2026-11-09 정밀분석 보고서 §6

Key Features:
- 100% Simulated Sandbox: Forced 'x-simulated-trading: 1' on all requests
- Targeted Universe: BTC-USDT-SWAP, ETH-USDT-SWAP, SOL-USDT-SWAP, XRP-USDT-SWAP (4 Major Leaders)
- 4-Major Dominance Synchronization (3/4 Consensus, S8):
  * Enters ONLY when at least 3 of 4 Majors simultaneously align on confirmed 1H MA(5/10/20) + BTC 15m EMA20
  * Anti-Spaghetti Chop Filter: |1H MA5 - MA20| >= 0.30%, 1H ADX(14) >= 18, 15m ADX(14) >= 18
  * One-Wave-One-Shot Lock: Invariant 1H wave_start_ts tracking blocks late climax / bottom re-entries
- 4-Stage Wave Harvest & Risk Management (10x Isolated, 20% Margin/Slot, Max 2 Slots):
  * Exchange-attached Hard SL (2.5% base ~4.0% dynamic ATR, safely inside ~9% 10x liquidation) & +5.0% Wave TP
  * Stage 1: Peak >= +1.50% spot (+30% ROE) -> Lock +0.25% spot (+5% ROE) Breakeven Floor
  * Stage 2: Peak >= +2.20% spot (+44% ROE) -> Lock +1.10% spot (+22% ROE) Mid-Wave Profit Floor
  * Stage 3: Peak >= +3.20% spot (+64% ROE) -> Wide 1.00% Wave Trailing Stop
  * Stage 4: Confirmed 1H MA20 Structure Reversal Exit (cuts invalidated waves early around -0.85%)
- Execution & Ops: Passive 1-Tick Post-Only Maker (0.02% fee) w/ Market fallback, Universal Jev AI Gate
  (strict fail-closed), account-mode guard, restart-safe state, Telegram alerts. Credentials from .env only.
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

# ── CONFIGURATION (ARGON V4-PRO S8: 4-MAJOR 3/4 WAVE RIDER — 10x, 122D 백테스트 검증 2026-11-09) ──
TARGET_SYMBOLS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "XRP-USDT-SWAP"]
LEGACY_MANAGE_SYMBOLS = ["DOGE-USDT-SWAP", "SUI-USDT-SWAP"]
LEVERAGE = 10                     # 10x Isolated (S8: 122D PF 1.83, MDD -12.5%; 20x는 MDD -27.5%로 기준 미달)
MAX_CONCURRENT_POSITIONS = 2
POSITION_MARGIN_PCT = 0.20        # 20% of USDT equity per slot (max 40% total margin deployed, 60% cash reserve)
MAX_MARGIN_PER_TRADE = 5000.0     # Max $5,000 margin per slot ($50k notional @ 10x -> stays in OKX Tier-1/2 MMR)
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

# ── ARGON V4.1-PRO WAVE & RISK PARAMETERS (in spot price terms) ──
MIN_SYNC_BREADTH = 3              # Require 3/4 Major Coins (BTC, ETH, SOL, XRP) aligned on 1H MA(5/10/20) — S8 (4/4는 빈도 급감·검증 지연)
MIN_BTC_MA20_SLOPE3_PCT = 0.18    # Minimum |BTC 1H MA20 3-bar slope| (%) to confirm standard market wave
MIN_BTC_IMPULSE_SLOPE3_PCT = 0.10 # Minimum |BTC 1H MA20 3-bar slope| (%) on fresh compression-box breakout
MIN_BTC_IMPULSE_MA5_SLOPE1_PCT = 0.30 # Minimum |BTC 1H MA5 1-bar slope| (%) on fresh compression-box breakout
MIN_BTC_IMPULSE_RET3_PCT = 1.50   # Minimum |BTC 1H 3-bar return| (%) to confirm vertical compression breakout
MIN_1H_MA_SPREAD_PCT = 0.30       # Minimum |1H MA5 - 1H MA20| / MA20 (%) to block tangled sideways chop
MIN_1H_ADX = 18.0                 # Minimum 1H Wilder ADX(14)
MIN_15M_ADX = 18.0                # Minimum 15m Wilder ADX(14)
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
WAVE_TP_PCT = 0.050               # +5.0% hard Wave Take-Profit (+50% ROE @ 10x) — S8: +10%는 도달률 4%로 미달
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
        self.trade_cooldowns: Dict[str, float] = {}
        self.order_fail_counts: Dict[str, int] = {}
        self.position_entry_times: Dict[str, float] = {}
        self.position_params: Dict[str, Dict[str, float]] = {}
        self.side_loss_streak: Dict[str, int] = {}
        self.side_lockout_until: Dict[str, float] = {}
        self.traded_waves: List[str] = []
        self.inst_cache: Dict[str, Dict[str, str]] = {}
        self.candle_cache: Dict[Tuple[str, str], Tuple[float, pd.DataFrame]] = {}
        self.trading_enabled = False
        self.last_mode_check = 0.0
        self.last_radar_log = 0.0
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
        logger.info("🚀 [OKX Jev Demo Trader] Argon V4-Pro S8: 4-Major Dominance Wave Rider Active")
        logger.info(f"   Target Universe : {', '.join(TARGET_SYMBOLS)} ({MIN_SYNC_BREADTH}/4 Synchronization Required)")
        logger.info(f"   Leverage & Size : {LEVERAGE}x Isolated | {POSITION_MARGIN_PCT*100:.0f}% USDT Eq/Slot (Max {MAX_CONCURRENT_POSITIONS} Slots, Cap ${MAX_MARGIN_PER_TRADE:,.0f})")
        logger.info(f"   Wave Entry Gate : {MIN_SYNC_BREADTH}/4 1H MA(5/10/20) + BTC Slope>={MIN_BTC_MA20_SLOPE3_PCT:.2f}% + Spread>={MIN_1H_MA_SPREAD_PCT:.2f}% + ADX>={MIN_1H_ADX:.0f}")
        logger.info(
            f"   4-Stage Harvest : S1 +{STAGE1_BE_ARM_PCT*100:.1f}%->+{STAGE1_BE_FLOOR_PCT*100:.2f}% | "
            f"S2 +{STAGE2_MID_ARM_PCT*100:.1f}%->+{STAGE2_MID_FLOOR_PCT*100:.2f}% | "
            f"S3 +{STAGE3_TRAIL_ARM_PCT*100:.1f}% (Drop {STAGE3_TRAIL_DROP_PCT*100:.1f}%) | S4 1H MA20 Exit"
        )
        logger.info("   Execution & Safe: 1-Tick Passive Maker (0.02%) -> Market Fallback | 100% Demo (x-simulated-trading: 1)")
        logger.info("=" * 76)

    # ── STATE ──
    def _load_state(self):
        st = load_state(STATE_FILE)
        now = time.time()
        self.peak_prices = {k: float(v) for k, v in st.get("peak_prices", {}).items()}
        self.trade_cooldowns = {k: float(v) for k, v in st.get("trade_cooldowns", {}).items() if float(v) > now}
        self.position_entry_times = {k: float(v) for k, v in st.get("position_entry_times", {}).items()}
        self.position_params = st.get("position_params", {}) or {}
        self.positions = st.get("positions", {}) or {}
        self.side_loss_streak = {k: int(v) for k, v in st.get("side_loss_streak", {}).items() if int(v) > 0}
        self.side_lockout_until = {k: float(v) for k, v in st.get("side_lockout_until", {}).items() if float(v) > now}
        raw_waves = st.get("traded_waves", [])
        self.traded_waves = [str(x) for x in raw_waves[-80:]] if isinstance(raw_waves, list) else []
        if st:
            logger.info(
                f"♻️ [State Restored] peaks={self.peak_prices} cooldowns={list(self.trade_cooldowns)} "
                f"waves={len(self.traded_waves)} streaks={self.side_loss_streak} lockouts={list(self.side_lockout_until)}"
            )

    def _save_state(self):
        now = time.time()
        save_state(STATE_FILE, {
            "peak_prices": self.peak_prices,
            "trade_cooldowns": {k: v for k, v in self.trade_cooldowns.items() if v > now},
            "position_entry_times": self.position_entry_times,
            "position_params": self.position_params,
            "side_loss_streak": {k: v for k, v in self.side_loss_streak.items() if v > 0},
            "side_lockout_until": {k: v for k, v in self.side_lockout_until.items() if v > now},
            "traded_waves": self.traded_waves[-80:],
            "positions": {k: {"avgPx": v.get("avgPx"), "pos": v.get("pos"), "upl": v.get("upl"), "uplRatio": v.get("uplRatio")}
                          for k, v in self.positions.items()},
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

            for sym in list(self.positions.keys()):
                if sym not in new_pos:
                    self._handle_external_close(sym, self.positions[sym])

            self.positions = new_pos
        return self.positions

    def _handle_external_close(self, inst_id: str, prev_pos: Dict[str, Any]):
        pnl, src = None, "positions-history"
        res = self._request("GET", f"/api/v5/account/positions-history?instType=SWAP&instId={inst_id}&limit=1")
        if res and res.get("data"):
            h = res["data"][0]
            try:
                pnl = float(h.get("realizedPnl") or h.get("pnl") or 0.0)
            except ValueError:
                pnl = None
        if pnl is None:
            src = "last upl"
            pnl = float(prev_pos.get("upl") or 0.0)
        is_loss = pnl < 0
        side_desc = "LONG" if float(prev_pos.get("pos", 0.0) or 0.0) > 0 else "SHORT"
        self._record_trade_outcome(inst_id, side_desc, is_loss)
        cooldown = POST_LOSS_COOLDOWN_SEC if is_loss else NORMAL_COOLDOWN_SEC
        self.trade_cooldowns[inst_id] = time.time() + cooldown
        self.peak_prices.pop(inst_id, None)
        self.position_entry_times.pop(inst_id, None)
        self.position_params.pop(inst_id, None)
        self._cancel_algo_orders(inst_id)
        self._save_state()
        tag = "LOSS" if is_loss else "WIN"
        logger.info(f"ℹ️ [Position Closed Externally] {inst_id} {tag} pnl={pnl:+.2f} USDT ({src}) -> cooldown {cooldown/60:.0f}m")
        send_telegram(f"[OKX Demo] {inst_id} 거래소 SL/TP 청산 ({tag} {pnl:+.2f} USDT)")

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
        info = self.get_instrument(inst_id)
        mid_pr, imb = self.get_ticker_and_imbalance(inst_id)
        if not info or mid_pr <= 0:
            logger.warning(f"⚠️ {inst_id}: instrument info / price unavailable, skip entry")
            return False
        self.set_leverage(inst_id)

        ct_val = float(info["ctVal"])
        base_usdt = self.usdt_equity if self.usdt_equity > 0 else self.available_usdt
        margin_to_use = min(base_usdt * POSITION_MARGIN_PCT, self.available_usdt * 0.90, MAX_MARGIN_PER_TRADE)
        notional_val = margin_to_use * LEVERAGE
        contracts = _round_step(notional_val / (mid_pr * ct_val), info["lotSz"], ROUND_DOWN)
        if contracts < Decimal(info["minSz"]):
            logger.warning(f"⚠️ {inst_id}: size {contracts} < minSz {info['minSz']} (margin ${margin_to_use:.0f}) -> skip")
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
        tp_px = maker_px * (1 + tp_pct) if is_long else maker_px * (1 - tp_pct)
        attach_algo = [{
            "tpTriggerPx": _fmt(_round_step(tp_px, tick, ROUND_HALF_UP), tick),
            "tpOrdPx": "-1",
            "tpTriggerPxType": "last",
            "slTriggerPx": _fmt(_round_step(sl_px, tick, ROUND_HALF_UP), tick),
            "slOrdPx": "-1",
            "slTriggerPxType": "last",
        }]

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
            f"| SL {attach_algo[0]['slTriggerPx']} (-{sl_pct*100:.2f}%) / Wave TP {attach_algo[0]['tpTriggerPx']} (+{tp_pct*100:.1f}%)"
        )
        res = self._request("POST", "/api/v5/trade/order", data=maker_order)
        maker_filled = False
        order_already_closed = False
        rem_contracts = contracts

        if res and res.get("data"):
            m_oid = res["data"][0].get("ordId")
            for _ in range(3):
                time.sleep(1.1)
                chk = self._request("GET", f"/api/v5/trade/order?instId={inst_id}&ordId={m_oid}")
                if chk and chk.get("data"):
                    st_info = chk["data"][0]
                    st_state = st_info.get("state", "")
                    acc_fill = Decimal(str(st_info.get("accFillSz") or "0"))
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
                    if st_info.get("state") == "filled":
                        maker_filled = True
                    else:
                        acc_fill = Decimal(str(st_info.get("accFillSz") or "0"))
                        rem_contracts = contracts - acc_fill

        if not maker_filled and rem_contracts >= Decimal(info["minSz"]):
            # Recompute SL/TP off latest mid price for market fallback accuracy
            curr_mid, _ = self.get_ticker_and_imbalance(inst_id)
            ref_px = curr_mid if curr_mid > 0 else maker_px
            m_sl_px = ref_px * (1 - sl_pct) if is_long else ref_px * (1 + sl_pct)
            m_tp_px = ref_px * (1 + tp_pct) if is_long else ref_px * (1 - tp_pct)
            m_attach = [{
                "tpTriggerPx": _fmt(_round_step(m_tp_px, tick, ROUND_HALF_UP), tick),
                "tpOrdPx": "-1",
                "tpTriggerPxType": "last",
                "slTriggerPx": _fmt(_round_step(m_sl_px, tick, ROUND_HALF_UP), tick),
                "slOrdPx": "-1",
                "slTriggerPxType": "last",
            }]
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

        if res and res.get("data"):
            oid = res["data"][0].get("ordId")
            logger.info(f"✅ [V4-Pro Wave Entry Accepted] Order ID: {oid} | {inst_id} {pos_type_desc} ({LEVERAGE}x Isolated)")
            self.order_fail_counts.pop(inst_id, None)
            self.trade_cooldowns[inst_id] = time.time() + ENTRY_COOLDOWN_SEC
            self.position_entry_times[inst_id] = time.time()
            self.peak_prices[inst_id] = mid_pr
            self.position_params[inst_id] = {
                "sl_pct": sl_pct,
                "tp_pct": tp_pct,
                "trail_arm": STAGE3_TRAIL_ARM_PCT,
                "trail_drop": STAGE3_TRAIL_DROP_PCT,
            }
            if btc_wave_ts and sym_wave_ts:
                self._mark_wave_traded(inst_id, btc_wave_ts, sym_wave_ts)
            self._save_state()
            send_telegram(
                f"[OKX V4-Pro {LEVERAGE}x] 🚀 {pos_type_desc} {inst_id} {order_data['sz']}ct "
                f"(증거금 ~${margin_to_use:,.0f} / 노출 ~${approx_usd:,.0f}) @ {mid_pr:,.4f} | SL -{sl_pct*100:.2f}%"
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
        side_desc = "LONG" if float(prev_pos.get("pos", 0.0) or 0.0) > 0 else "SHORT"
        is_loss = ("STOP_LOSS" in reason) or (pnl_pct < 0)
        self._record_trade_outcome(inst_id, side_desc, is_loss)
        cooldown = POST_LOSS_COOLDOWN_SEC if is_loss else NORMAL_COOLDOWN_SEC
        self.trade_cooldowns[inst_id] = time.time() + cooldown
        self.peak_prices.pop(inst_id, None)
        self.position_entry_times.pop(inst_id, None)
        self.position_params.pop(inst_id, None)
        self._save_state()
        roe_est = pnl_pct * LEVERAGE * 100.0
        send_telegram(f"[OKX V4-Pro] {inst_id} 청산 ({reason}) | 현물 {pnl_pct*100:+.2f}% (추정 ROE {roe_est:+.1f}%)")
        return True

    # ── 4-STAGE WAVE POSITION MANAGEMENT ──
    def manage_open_positions(self):
        """Monitors open positions with Argon V4-Pro 4-Stage Wave Harvest & 1H MA20 Structure Guard."""
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

                peak_pl_pct = (peak - avg_px) / avg_px if is_long else (avg_px - peak) / avg_px

                # 1. Hard Stop-Loss backup (Exchange SL is primary)
                if spot_pl_pct <= -pos_sl_pct:
                    logger.warning(f"🛑 [Stop Loss Trigger] {inst_id} {'LONG' if is_long else 'SHORT'} Spot PnL: {spot_pl_pct*100:+.2f}% <= -{pos_sl_pct*100:.2f}%")
                    self.close_position(inst_id, f"STOP_LOSS_{spot_pl_pct*100:.2f}PCT", spot_pl_pct)
                    continue

                # 2. Hard Wave Take-Profit backup (+5.0% spot = +50% ROE @ 10x)
                if spot_pl_pct >= pos_tp_pct:
                    logger.info(f"🎯 [Wave Take Profit] {inst_id} {'LONG' if is_long else 'SHORT'} Spot PnL: +{spot_pl_pct*100:.2f}% >= +{pos_tp_pct*100:.1f}%")
                    self.close_position(inst_id, f"WAVE_TP_{spot_pl_pct*100:+.2f}PCT", spot_pl_pct)
                    continue

                # 3. Stage 2 Mid-Wave Profit Floor (Peak >= +2.20% spot -> Lock +1.10% spot = +22% ROE)
                if peak_pl_pct >= STAGE2_MID_ARM_PCT and spot_pl_pct <= STAGE2_MID_FLOOR_PCT:
                    logger.info(
                        f"🛡️ [Wave Stage-2 Profit Lock] {inst_id} {'LONG' if is_long else 'SHORT'} Peak +{peak_pl_pct*100:.2f}% "
                        f"pulled back to +{spot_pl_pct*100:.2f}% (Floor +{STAGE2_MID_FLOOR_PCT*100:.2f}%)."
                    )
                    self.close_position(inst_id, f"WAVE_STAGE2_LOCK_{spot_pl_pct*100:+.2f}PCT", spot_pl_pct)
                    continue

                # 4. Stage 1 Early Breakeven Floor (Peak >= +1.50% spot -> Lock +0.25% spot = +5% ROE)
                if peak_pl_pct >= STAGE1_BE_ARM_PCT and spot_pl_pct <= STAGE1_BE_FLOOR_PCT:
                    logger.info(
                        f"🛡️ [Wave Stage-1 BE Lock] {inst_id} {'LONG' if is_long else 'SHORT'} Peak +{peak_pl_pct*100:.2f}% "
                        f"pulled back to +{spot_pl_pct*100:.2f}% (Floor +{STAGE1_BE_FLOOR_PCT*100:.2f}%)."
                    )
                    self.close_position(inst_id, f"WAVE_STAGE1_BE_{spot_pl_pct*100:+.2f}PCT", spot_pl_pct)
                    continue

                # 5. Stage 3 Big-Wave Trailing Stop (Peak >= +3.20% spot -> 1.00% pullback harvest)
                if peak_pl_pct >= STAGE3_TRAIL_ARM_PCT:
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
        if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
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
            logger.info(
                f"📡 [V4.1-Pro 4-Major Radar] {regime_str} | BTC 1H=${bc1h:,.1f} vs MA20=${bm20:,.1f} "
                f"(Slope3H {bslope:+.2f}%, MA5Slp {bm5_slope:+.2f}%, Age {btc_wave_age}h) | Open Slots: {len(self.positions)}/{MAX_CONCURRENT_POSITIONS}"
            )

        if not (sync_bull or sync_bear):
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
        for sym, score, atr15, s1h, s15 in candidates:
            if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
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
                            "tp_pct": WAVE_TP_PCT * 100.0,
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

            self.open_position(
                sym,
                side=target_side,
                sl_pct=dyn_sl_pct,
                tp_pct=WAVE_TP_PCT,
                btc_wave_ts=btc_wave_ts,
                sym_wave_ts=s1h["wave_ts"],
            )

    # ── MAIN LOOP ──
    def run(self):
        """24/7 Supervisor Loop"""
        if not (self.api_key and self.secret_key and self.passphrase):
            logger.critical("⛔ OKX_DEMO_API_KEY / OKX_DEMO_SECRET / OKX_DEMO_PASSPHRASE missing in .env -> exiting")
            sys.exit(1)

        logger.info(f"🟢 [OKX Jev Demo Trader] Argon V4-Pro ({LEVERAGE}x Isolated, 4-Major Wave Rider) running 24/7...")
        send_telegram(f"[OKX Demo] Argon V4-Pro S8 가동 ({LEVERAGE}x 격리, 증거금 {POSITION_MARGIN_PCT*100:.0f}%/슬롯, 4대 메이저 {MIN_SYNC_BREADTH}/4 파동 동조화)")
        self.sync_account()
        self.check_account_mode()

        while True:
            try:
                self.sync_account()
                self.sync_positions()
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
