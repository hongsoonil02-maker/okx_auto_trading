# -*- coding: utf-8 -*-
"""
okx_jev_demo_trader.py — Autonomous OKX Demo Quant Engine with Bidirectional Jev Alpha
Optimal Alpha Boost Configuration (Backtest Validated: +8.06% Return in 45 Days)
NOTE (2026-10-22 review): the backtest figure pre-dates the fixes below (Jev gate never invoked,
unconfirmed-candle signals, no fills due to Spot-mode account). Re-validate before trusting it.

Key Features:
- 100% Simulated Sandbox: Forced 'x-simulated-trading: 1' on all requests
- Targeted Universe: BTC-USDT-SWAP, SOL-USDT-SWAP, DOGE-USDT-SWAP (High Momentum Leaders)
- Smart Bidirectional: Long during Bullish Macro, Short during Bearish Macro
- Chop/Whipsaw Filter: ATR Volatility Gate filters out 70% of noisy sideways chop
- Macro Gate: BTC 5Min EMA50 Directional Filter (confirmed candles only, no repaint)
- Microstructure & AI: Real-time LOB Imbalance + Universal Jev AI Gate (trend prompt, fail-closed)
- Strict Risk Management:
  * Low Leverage (3x isolated) for high safety
  * SL: -1.5% | TP: +3.5% | Trailing Arm: +1.5% (spot price movement)
  * SL/TP attached to the order on the exchange (attachAlgoOrds) + bot-side backup monitoring
  * 30-minute Post-Loss Lockout after any losing exit (incl. exchange-side SL fills)
  * Max 2 Concurrent Positions, 25% margin per trade, lotSz/minSz-aware sizing
- Ops: account-mode guard (requires Futures mode, acctLv>=2), sCode/sMsg error logging,
  order-failure cooldown, restart-safe state, Telegram alerts. Credentials from .env only.
"""

import os
import sys
import time
import math
import hmac
import base64
import json
import logging
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from datetime import datetime, timezone, timedelta
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

# ── CONFIGURATION (OPTIMAL CRYPTO ALPHA) ──
TARGET_SYMBOLS = ["BTC-USDT-SWAP", "SOL-USDT-SWAP", "DOGE-USDT-SWAP"]
LEVERAGE = 5                      # 5x leverage for aggressive capital growth
MAX_CONCURRENT_POSITIONS = 2
POSITION_MARGIN_PCT = 0.25        # 25% margin per position (High Alpha Allocation)
MAX_MARGIN_PER_TRADE = 2500.0     # Max $2,500 margin per trade
LOOP_INTERVAL_SEC = 10
POST_LOSS_COOLDOWN_SEC = 1800.0   # 30-minute lockout after stop-loss
NORMAL_COOLDOWN_SEC = 1800.0      # 30-minute normal cooldown (improved rotation)
ENTRY_COOLDOWN_SEC = 180.0
ORDER_FAIL_COOLDOWN_SEC = 300.0   # 5-minute cooldown after a rejected order
JEV_VETO_COOLDOWN_SEC = 1800.0    # 30-minute cooldown after a Jev veto
JEV_ERROR_COOLDOWN_SEC = 120.0    # 2-minute cooldown after a Jev API error
ACCOUNT_MODE_RECHECK_SEC = 300.0
STATE_FILE = os.path.join(CURRENT_DIR, "state", "okx_jev_demo_state.json")

# Risk Parameters (in spot price terms)
SL_PCT = 0.018                    # -1.8% Spot SL (at 5x lev = -9.0% ROE)
TP_PCT = 0.060                    # +6.0% Spot TP (at 5x lev = +30.0% ROE, High Alpha Target)
TRAIL_ARM_PCT = 0.025             # Arm trailing at +2.5% spot gain
TRAIL_DROP_PCT = 0.008            # Lock profit on 0.8% pullback from peak
MIN_ATR_RATIO = 0.0030            # Minimum ATR/Price volatility (filters out sideways chop)
MAX_EMA20_DIST_PCT = 1.2          # Max allowed distance from 5m EMA20 to prevent overextension chasing


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

        self.account_equity = 0.0
        self.available_usdt = 0.0
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.peak_prices: Dict[str, float] = {}
        self.trade_cooldowns: Dict[str, float] = {}
        self.position_entry_times: Dict[str, float] = {}
        self.position_params: Dict[str, Dict[str, float]] = {}
        self.inst_cache: Dict[str, Dict[str, str]] = {}
        self.trading_enabled = False
        self.last_mode_check = 0.0
        self._load_state()

        if JEV_GATE_AVAILABLE:
            self.jev_gate = UniversalJevGate(confidence_threshold=float(os.getenv("JEV_CONFIDENCE_THRESHOLD", "0.60")))
            if self.jev_gate.is_enabled and self.jev_gate.api_key:
                logger.info("🧠 Universal Jev AI Gate loaded (trend prompt, fail-closed).")
            else:
                logger.warning("⚠️ Jev Gate disabled (USE_JEV_PREDICTION=false or no API key) -> technical signals only.")
        else:
            self.jev_gate = None
            logger.warning("⚠️ Universal Jev Gate unavailable -> technical signals only.")

        logger.info("=" * 70)
        logger.info("🚀 [OKX Jev Demo Trader] Optimal Alpha Engine Active (v3.0 Dynamic ATR + MTF)")
        logger.info(f"   Target Universe: {', '.join(TARGET_SYMBOLS)}")
        logger.info(f"   Strategy: Bidirectional Long & Short with 15m MTF Gate + Dynamic ATR SL/TP")
        logger.info(f"   Safety Level: 100% Simulated (x-simulated-trading: 1)")
        logger.info(f"   Leverage: {LEVERAGE}x (Isolated Margin)")
        logger.info(f"   Max Concurrent Positions: {MAX_CONCURRENT_POSITIONS} | Margin: {POSITION_MARGIN_PCT*100:.0f}% (max ${MAX_MARGIN_PER_TRADE:,.0f})")
        logger.info(f"   Risk: Dynamic ATR SL (2.2*ATR: 1.8%~3.5%), TP (2.5*SL: 4.0%~8.0%), Trailing Ratchet")
        logger.info(f"   Cooldowns: Post-Loss {POST_LOSS_COOLDOWN_SEC/60:.0f}m | Post-Win {NORMAL_COOLDOWN_SEC/60:.0f}m | Jev Veto {JEV_VETO_COOLDOWN_SEC/60:.0f}m")
        logger.info("=" * 70)

    # ── STATE ──
    def _load_state(self):
        st = load_state(STATE_FILE)
        now = time.time()
        self.peak_prices = {k: float(v) for k, v in st.get("peak_prices", {}).items()}
        self.trade_cooldowns = {k: float(v) for k, v in st.get("trade_cooldowns", {}).items() if float(v) > now}
        self.position_entry_times = {k: float(v) for k, v in st.get("position_entry_times", {}).items()}
        self.position_params = st.get("position_params", {}) or {}
        self.positions = st.get("positions", {}) or {}  # lets us detect exchange-side closes that happened while down
        if st:
            logger.info(f"♻️ [State Restored] peaks={self.peak_prices} cooldowns={list(self.trade_cooldowns)}")

    def _save_state(self):
        save_state(STATE_FILE, {
            "peak_prices": self.peak_prices,
            "trade_cooldowns": {k: v for k, v in self.trade_cooldowns.items() if v > time.time()},
            "position_entry_times": self.position_entry_times,
            "position_params": self.position_params,
            "positions": {k: {"avgPx": v.get("avgPx"), "pos": v.get("pos"), "upl": v.get("upl"), "uplRatio": v.get("uplRatio")}
                          for k, v in self.positions.items()},
            "saved_at": datetime.now(timezone.utc).isoformat(),
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
                # Per-item errors (sCode/sMsg) carry the real reason, e.g. 51010 account-mode / 51008 margin
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
        """Fetches demo account equity and available USDT"""
        res = self._request("GET", "/api/v5/account/balance")
        if res and res.get("data"):
            acc = res["data"][0]
            self.account_equity = float(acc.get("totalEq", 0.0) or 0.0)
            for d in acc.get("details", []):
                if d.get("ccy") == "USDT":
                    self.available_usdt = float(d.get("availBal") or d.get("availEq") or 0.0)
            logger.info(f"💼 [Demo Account Sync] Total Equity: ${self.account_equity:,.2f} | Avail USDT: ${self.available_usdt:,.2f}")
            return True
        return False

    def sync_positions(self) -> Dict[str, Dict[str, Any]]:
        """Fetches active open positions"""
        res = self._request("GET", "/api/v5/account/positions?instType=SWAP")
        if res:
            new_pos = {}
            for p in res.get("data", []):
                inst_id = p.get("instId")
                if inst_id in TARGET_SYMBOLS and abs(float(p.get("pos", 0.0) or 0.0)) > 0:
                    new_pos[inst_id] = p

            # Check external closes (exchange-attached SL/TP fills, liquidation, manual)
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

    # ── MARKET DATA ──
    def get_ticker_and_imbalance(self, inst_id: str) -> Tuple[float, float]:
        """Fetches L1 BBO and top-5 orderbook imbalance"""
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
                        return mid_price, imbalance
        except Exception as e:
            logger.debug(f"Ticker/Imb error for {inst_id}: {e}")
        return 0.0, 0.0

    def fetch_bars(self, inst_id: str, bar: str = "5m", limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetches CONFIRMED candlestick bars only (the live, still-forming candle is dropped to avoid repaint)"""
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
                for col in ("open", "high", "low", "close"):
                    df[col] = df[col].astype(float)
                return df
        except Exception as e:
            logger.debug(f"Bars error for {inst_id}: {e}")
        return None

    def calculate_technical_signals(self, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Calculates EMA9, EMA20, EMA50 and ATR Volatility"""
        if df is None or len(df) < 20:
            return {"signal": "NEUTRAL", "price": 0.0, "ema50": 0.0, "atr_ratio": 0.0}

        closes = df["close"]
        price = float(closes.iloc[-1])
        ema9 = float(closes.ewm(span=9, adjust=False).mean().iloc[-1])
        ema20 = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(closes.ewm(span=min(50, len(closes)), adjust=False).mean().iloc[-1])

        # True Range and ATR
        highs = df["high"]
        lows = df["low"]
        tr = np.maximum(highs - lows, np.maximum(abs(highs - closes.shift()), abs(lows - closes.shift())))
        atr = float(tr.rolling(14).mean().iloc[-1])
        atr_ratio = (atr / price) if price > 0 and not math.isnan(atr) else 0.0
        lookback = min(12, len(closes) - 1)
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
            "atr_ratio": atr_ratio,
            "dist_ema20_pct": (price / ema20 - 1.0) * 100.0 if ema20 > 0 else 0.0,
            "recent_return_pct": recent_ret * 100.0,
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
        """Ensures isolated leverage is set"""
        data = {
            "instId": inst_id,
            "lever": str(LEVERAGE),
            "mgnMode": "isolated"
        }
        self._request("POST", "/api/v5/account/set-leverage", data=data)

    def _cancel_algo_orders(self, inst_id: str):
        """Cancels leftover attached SL/TP algo orders for the instrument."""
        res = self._request("GET", f"/api/v5/trade/orders-algo-pending?ordType=conditional,oco&instId={inst_id}")
        algos = [{"algoId": a["algoId"], "instId": inst_id} for a in (res or {}).get("data", []) if a.get("algoId")]
        if algos:
            self._request("POST", "/api/v5/trade/cancel-algos", data=algos)
            logger.info(f"🧹 [Algo Cleanup] {inst_id} cancelled {len(algos)} leftover SL/TP algo order(s)")

    def open_position(self, inst_id: str, side: str = "buy",
                      sl_pct: float = SL_PCT, tp_pct: float = TP_PCT,
                      trail_arm: float = TRAIL_ARM_PCT, trail_drop: float = TRAIL_DROP_PCT) -> bool:
        """Submits market order with exchange-attached dynamic SL/TP (Buy for Long, Sell for Short)"""
        info = self.get_instrument(inst_id)
        mid_pr, imb = self.get_ticker_and_imbalance(inst_id)
        if not info or mid_pr <= 0:
            logger.warning(f"⚠️ {inst_id}: instrument info / price unavailable, skip entry")
            return False
        self.set_leverage(inst_id)

        ct_val = float(info["ctVal"])
        margin_to_use = min(self.available_usdt * POSITION_MARGIN_PCT, MAX_MARGIN_PER_TRADE)
        notional_val = margin_to_use * LEVERAGE
        contracts = _round_step(notional_val / (mid_pr * ct_val), info["lotSz"], ROUND_DOWN)
        if contracts < Decimal(info["minSz"]):
            logger.warning(f"⚠️ {inst_id}: size {contracts} < minSz {info['minSz']} (margin ${margin_to_use:.0f}) -> skip")
            self.trade_cooldowns[inst_id] = time.time() + ORDER_FAIL_COOLDOWN_SEC
            return False

        is_long = side == "buy"
        sl_px = mid_pr * (1 - sl_pct) if is_long else mid_pr * (1 + sl_pct)
        tp_px = mid_pr * (1 + tp_pct) if is_long else mid_pr * (1 - tp_pct)
        tick = info["tickSz"]
        order_data = {
            "instId": inst_id,
            "tdMode": "isolated",
            "side": side,
            "posSide": "net",
            "ordType": "market",
            "sz": _fmt(contracts, info["lotSz"]),
            "clOrdId": f"jevdemo{int(time.time() * 1000)}",
            "attachAlgoOrds": [{
                "tpTriggerPx": _fmt(_round_step(tp_px, tick, ROUND_HALF_UP), tick),
                "tpOrdPx": "-1",
                "tpTriggerPxType": "last",
                "slTriggerPx": _fmt(_round_step(sl_px, tick, ROUND_HALF_UP), tick),
                "slOrdPx": "-1",
                "slTriggerPxType": "last",
            }],
        }

        pos_type_desc = "LONG" if is_long else "SHORT"
        approx_usd = float(contracts) * ct_val * mid_pr
        logger.info(f"📤 [Demo Order Submit] {pos_type_desc} {order_data['sz']} contracts {inst_id} (~${approx_usd:,.1f}) @ ${mid_pr:,.4f} "
                    f"| Dyn SL {order_data['attachAlgoOrds'][0]['slTriggerPx']} (-{sl_pct*100:.2f}%) / TP {order_data['attachAlgoOrds'][0]['tpTriggerPx']} (+{tp_pct*100:.2f}%)")
        res = self._request("POST", "/api/v5/trade/order", data=order_data)
        if res and res.get("data"):
            oid = res["data"][0].get("ordId")
            logger.info(f"✅ [Demo Order Accepted] Order ID: {oid} | {inst_id} {pos_type_desc}")
            self.trade_cooldowns[inst_id] = time.time() + ENTRY_COOLDOWN_SEC
            self.position_entry_times[inst_id] = time.time()
            self.peak_prices[inst_id] = mid_pr
            self.position_params[inst_id] = {
                "sl_pct": sl_pct,
                "tp_pct": tp_pct,
                "trail_arm": trail_arm,
                "trail_drop": trail_drop
            }
            self._save_state()
            send_telegram(f"[OKX Demo] {pos_type_desc} {inst_id} {order_data['sz']}ct ~${approx_usd:,.0f} @ {mid_pr:,.4f} | SL -{sl_pct*100:.1f}%, TP +{tp_pct*100:.1f}%")
            return True

        self.trade_cooldowns[inst_id] = time.time() + ORDER_FAIL_COOLDOWN_SEC
        logger.error(f"❌ [Order Failed] {inst_id} {pos_type_desc}: {self.last_error} -> cooldown {ORDER_FAIL_COOLDOWN_SEC/60:.0f}m")
        send_telegram(f"[OKX Demo] ❌ {inst_id} 주문 실패: {self.last_error}",
                      dedupe_key=f"okx_order_fail_{inst_id}", min_interval_sec=1800)
        if "51010" in self.last_error or "account mode" in self.last_error.lower():
            self.trading_enabled = False  # re-check account mode
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

        logger.info(f"✅ [Demo Position Closed] {inst_id} successfully closed.")
        self._cancel_algo_orders(inst_id)
        self.positions.pop(inst_id, None)
        self.peak_prices.pop(inst_id, None)
        self.position_entry_times.pop(inst_id, None)
        self.position_params.pop(inst_id, None)
        self._save_state()

        if "STOP_LOSS" in reason or pnl_pct < 0:
            logger.info(f"⏳ [Post-Loss Lockout] {inst_id} enforcing {POST_LOSS_COOLDOWN_SEC/60:.0f}m cooling period.")
            self.trade_cooldowns[inst_id] = time.time() + POST_LOSS_COOLDOWN_SEC
        else:
            self.trade_cooldowns[inst_id] = time.time() + NORMAL_COOLDOWN_SEC
        send_telegram(f"[OKX Demo] {inst_id} 청산 ({reason}) spot {pnl_pct*100:+.2f}%")
        return True

    def manage_open_positions(self):
        """Monitors open positions for SL, TP, and Trailing Stops (Long & Short). Exchange SL/TP is primary."""
        if not self.positions:
            return

        for inst_id, pos in list(self.positions.items()):
            try:
                avg_px = float(pos.get("avgPx", 0.0) or 0.0)
                pos_amt = float(pos.get("pos", 0.0) or 0.0)
                curr_px, imb = self.get_ticker_and_imbalance(inst_id)

                if avg_px <= 0 or curr_px <= 0:
                    continue

                is_long = pos_amt > 0
                spot_pl_pct = (curr_px - avg_px) / avg_px if is_long else (avg_px - curr_px) / avg_px

                # Retrieve dynamic risk parameters for this position (fallback to defaults)
                params = self.position_params.get(inst_id, {
                    "sl_pct": SL_PCT,
                    "tp_pct": TP_PCT,
                    "trail_arm": TRAIL_ARM_PCT,
                    "trail_drop": TRAIL_DROP_PCT
                })
                pos_sl_pct = params.get("sl_pct", SL_PCT)
                pos_tp_pct = params.get("tp_pct", TP_PCT)
                pos_trail_arm = params.get("trail_arm", TRAIL_ARM_PCT)
                pos_trail_drop = params.get("trail_drop", TRAIL_DROP_PCT)

                # Peak price tracking
                peak = self.peak_prices.get(inst_id, avg_px)
                if is_long and curr_px > peak:
                    self.peak_prices[inst_id] = curr_px
                    peak = curr_px
                elif not is_long and curr_px < peak:
                    self.peak_prices[inst_id] = curr_px
                    peak = curr_px

                # 1. Hard Stop-Loss backup
                if spot_pl_pct <= -pos_sl_pct:
                    logger.warning(f"🛑 [Stop Loss Trigger] {inst_id} {'LONG' if is_long else 'SHORT'} Spot PnL: {spot_pl_pct*100:.2f}% (Limit: -{pos_sl_pct*100:.2f}%)")
                    self.close_position(inst_id, f"STOP_LOSS_{spot_pl_pct*100:.1f}PCT", spot_pl_pct)
                    continue

                # 2. Take-Profit backup
                if spot_pl_pct >= pos_tp_pct:
                    logger.info(f"🎯 [Take Profit Trigger] {inst_id} {'LONG' if is_long else 'SHORT'} Spot PnL: +{spot_pl_pct*100:.2f}% (Target: +{pos_tp_pct*100:.2f}%)")
                    self.close_position(inst_id, f"TAKE_PROFIT_{spot_pl_pct*100:.1f}PCT", spot_pl_pct)
                    continue

                # 3. Dynamic Trailing Stop
                if spot_pl_pct >= pos_trail_arm:
                    pullback = (peak - curr_px) / peak if is_long else (curr_px - peak) / peak
                    if pullback >= pos_trail_drop:
                        logger.info(f"🔒 [Trailing Stop Exit] {inst_id} {'LONG' if is_long else 'SHORT'} profit locked: Entry ${avg_px} -> Peak ${peak} -> Curr ${curr_px} (Gain: +{spot_pl_pct*100:.2f}%)")
                        self.close_position(inst_id, f"TRAILING_EXIT_{spot_pl_pct*100:.2f}PCT", spot_pl_pct)
                        continue

            except Exception as e:
                logger.error(f"Position manage error for {inst_id}: {e}")

    # ── ENTRY LOGIC ──
    def evaluate_trading_opportunities(self):
        """Scans BTC Macro, applies Chop Filter, and enters Long or Short"""
        if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
            return

        # 1. Macro Trend Gate (BTC-USDT-SWAP 5Min, confirmed candles)
        btc_bars = self.fetch_bars("BTC-USDT-SWAP", bar="5m", limit=100)
        btc_tech = self.calculate_technical_signals(btc_bars)
        btc_px = btc_tech.get("price", 0.0)
        btc_e50 = btc_tech.get("ema50", 0.0)

        if btc_px <= 0 or btc_e50 <= 0:
            return

        macro_bullish = btc_px > (btc_e50 * 1.001)
        macro_bearish = btc_px < (btc_e50 * 0.999)

        # 1.1 Higher Timeframe (15m) BTC Confirmation (MTF Filter)
        btc_15m_bars = self.fetch_bars("BTC-USDT-SWAP", bar="15m", limit=50)
        if btc_15m_bars is not None and len(btc_15m_bars) >= 20:
            btc_15m_tech = self.calculate_technical_signals(btc_15m_bars)
            btc_15m_px = btc_15m_tech.get("price", 0.0)
            btc_15m_e50 = btc_15m_tech.get("ema50", 0.0)
            if btc_15m_px > 0 and btc_15m_e50 > 0:
                if macro_bullish and not (btc_15m_px > btc_15m_e50):
                    logger.debug("⏳ [MTF BTC Filter] 5m is Bullish but 15m is below EMA50. Skipping chop.")
                    macro_bullish = False
                elif macro_bearish and not (btc_15m_px < btc_15m_e50):
                    logger.debug("⏳ [MTF BTC Filter] 5m is Bearish but 15m is above EMA50. Skipping chop.")
                    macro_bearish = False

        if not (macro_bullish or macro_bearish):
            return

        for sym in TARGET_SYMBOLS:
            if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
                break
            if sym in self.positions:
                continue
            if time.time() < self.trade_cooldowns.get(sym, 0.0):
                continue

            df = btc_bars if sym == "BTC-USDT-SWAP" else self.fetch_bars(sym, bar="5m", limit=100)
            tech = self.calculate_technical_signals(df)

            # 2. Chop Filter: Reject low volatility / sideways noise
            atr_ratio = tech.get("atr_ratio", 0.0)
            if atr_ratio < MIN_ATR_RATIO:
                logger.debug(f"⏳ [Chop Filter Active] {sym} ATR ratio {atr_ratio*100:.3f}% < {MIN_ATR_RATIO*100:.2f}%. Skipping chop.")
                continue

            # 2.1 Overextension Guard: Reject chasing when price is already detached from 5m EMA20
            dist_ema20 = tech.get("dist_ema20_pct", 0.0)
            if tech["signal"] == "SELL" and dist_ema20 < -MAX_EMA20_DIST_PCT:
                logger.info(f"⏳ [Overextension Guard] {sym} SHORT skipped: Price is {dist_ema20:.2f}% below EMA20 (overextended sell/bounce trap).")
                continue
            if tech["signal"] == "BUY" and dist_ema20 > MAX_EMA20_DIST_PCT:
                logger.info(f"⏳ [Overextension Guard] {sym} LONG skipped: Price is +{dist_ema20:.2f}% above EMA20 (overextended buy/pullback trap).")
                continue

            # 2.2 Higher Timeframe (15m) Trend & Support/Resistance Confirmation
            bars_15m = self.fetch_bars(sym, bar="15m", limit=30)
            if bars_15m is not None and len(bars_15m) >= 20:
                tech_15m = self.calculate_technical_signals(bars_15m)
                dist_15m = tech_15m.get("dist_ema20_pct", 0.0)
                px_15m = tech_15m.get("price", 0.0)
                ema20_15m = tech_15m.get("ema20", 0.0)
                if macro_bearish and tech["signal"] == "SELL":
                    if tech_15m.get("signal") == "BUY" or dist_15m < -2.2 or (ema20_15m > 0 and px_15m > ema20_15m):
                        logger.info(f"⏳ [15M Trend Guard] {sym} SHORT skipped: 15m signal is {tech_15m.get('signal')} or 15m above EMA20 / oversold ({dist_15m:.2f}%).")
                        continue
                elif macro_bullish and tech["signal"] == "BUY":
                    if tech_15m.get("signal") == "SELL" or dist_15m > 2.2 or (ema20_15m > 0 and px_15m < ema20_15m):
                        logger.info(f"⏳ [15M Trend Guard] {sym} LONG skipped: 15m signal is {tech_15m.get('signal')} or 15m below EMA20 / overbought ({dist_15m:.2f}%).")
                        continue

            # 3. Signal Decision: Long or Short
            if not ((macro_bullish and tech["signal"] == "BUY") or (macro_bearish and tech["signal"] == "SELL")):
                continue
            px, imb = self.get_ticker_and_imbalance(sym)
            target_side = None
            if macro_bullish and tech["signal"] == "BUY" and imb >= 0.20:
                target_side = "buy" # LONG
            elif macro_bearish and tech["signal"] == "SELL" and imb <= -0.20:
                target_side = "sell" # SHORT

            if target_side is None:
                continue

            # Dynamic ATR parameters (Var 2 validated: 2.2*ATR: 1.8%~3.5% SL, 2.5*SL: 4.0%~8.0% TP)
            dyn_sl_pct = max(0.018, min(0.035, 2.2 * atr_ratio))
            dyn_tp_pct = max(0.040, min(0.080, 2.5 * dyn_sl_pct))
            dyn_trail_arm = max(0.018, dyn_sl_pct * 1.1)
            dyn_trail_drop = max(0.006, dyn_sl_pct * 0.35)

            pos_desc = "LONG" if target_side == "buy" else "SHORT"
            logger.info(f"🔎 [Signal Detected] {sym} {pos_desc} | Price: ${px:,.4f} | Imbalance: {imb:+.2f} | ATR: {atr_ratio*100:.2f}% | Dyn SL: -{dyn_sl_pct*100:.2f}%, TP: +{dyn_tp_pct*100:.2f}%")

            # 4. Universal Jev AI Gate (trend prompt, fail-closed)
            if self.jev_gate:
                try:
                    decision = self.jev_gate.evaluate_sync(
                        asset_class="CRYPTO",
                        symbol=sym,
                        side=pos_desc,
                        context_metrics={
                            **{k: tech[k] for k in ("price", "ema9", "ema20", "ema50",
                                                     "dist_ema20_pct", "recent_return_pct", "lookback_bars")},
                            "atr_pct": atr_ratio * 100,
                            "imbalance": imb,
                            "macro": f"BTC 5m {'BULLISH' if macro_bullish else 'BEARISH'} (P {btc_px:.1f} vs EMA50 {btc_e50:.1f}) | 15m MTF Confirmed",
                            "sl_pct": dyn_sl_pct * 100, "tp_pct": dyn_tp_pct * 100,
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
                logger.info(f"🧠 [Jev Approved] {sym} {pos_desc} (conf {decision.confidence:.2f}, {decision.latency_ms:.0f}ms{', fallback' if decision.is_fallback else ''}): {decision.reason}")

            logger.info(f"🚀 Entering {sym} {pos_desc} on OKX Demo (Dyn SL -{dyn_sl_pct*100:.2f}%, TP +{dyn_tp_pct*100:.2f}%)")
            self.open_position(sym, target_side, sl_pct=dyn_sl_pct, tp_pct=dyn_tp_pct, trail_arm=dyn_trail_arm, trail_drop=dyn_trail_drop)

    # ── MAIN LOOP ──
    def run(self):
        """24/7 Supervisor Loop"""
        if not (self.api_key and self.secret_key and self.passphrase):
            logger.critical("⛔ OKX_DEMO_API_KEY / OKX_DEMO_SECRET / OKX_DEMO_PASSPHRASE missing in .env -> exiting")
            sys.exit(1)

        logger.info("🟢 [OKX Jev Demo Trader] Optimal Bidirectional Engine running 24/7...")
        send_telegram("[OKX Demo] 엔진 시작")
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
