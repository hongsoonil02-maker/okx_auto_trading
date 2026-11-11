# -*- coding: utf-8 -*-
"""
alpaca_etf_rotation.py — H3 Multi-Asset ETF Rotation Engine (Paper-Only Forward Test)
======================================================================================
Validated via 20-year daily swing research (tools/research_alpaca_swing_v3.py):
  - Hypothesis H3 (lb126 + ma150): IS CAGR +9.9% (MDD -11.2%) | OOS CAGR +12.0%, Sharpe 1.19 (MDD -12.8%)
  - Survivorship-bias-free ETF universe: QQQ, SPY, EFA, GLD, TLT, DBC, SHY (all fractionable)
  - 50% H1 (Dual-Momentum Top-3 above MA200 & > SHY) + 50% H2 (63d Inv-Vol Risk Parity above MA150)
  - Unmet / downtrend allocations automatically park into short-term Treasuries (SHY).
  - Monthly rebalance on first RTH session of each month using prior month-end confirmed closes
    (plus one-time initial portfolio deployment on first run). Zero PDT / shorting conflicts.
"""

import os
import sys
import time
import logging
from datetime import datetime, date, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

from bot_utils import send_telegram, load_state, save_state  # noqa: E402

LOG_FILE = os.path.join(CURRENT_DIR, "alpaca_etf_rotation.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"),
    ],
)
logger = logging.getLogger("AlpacaETFRotation")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "").strip()
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "").strip()
ALPACA_REST_URL = os.getenv("ALPACA_REST_URL", "https://paper-api.alpaca.markets/v2").rstrip("/")
if not ALPACA_REST_URL.endswith("/v2"):
    ALPACA_REST_URL = f"{ALPACA_REST_URL}/v2"
ALPACA_DATA_URL = "https://data.alpaca.markets/v2"

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type": "application/json",
}

# Strategy parameters (locked to validated H3 lb126 + ma150)
RISK_UNIVERSE = ["QQQ", "SPY", "EFA", "GLD", "TLT", "DBC"]
RP_UNIVERSE = ["SPY", "QQQ", "GLD", "TLT", "DBC"]
CASH_ETF = "SHY"
ALL_UNIVERSE = sorted(set(RISK_UNIVERSE + [CASH_ETF]))

H1_LOOKBACK_DAYS = 126
H1_MA_FILTER_DAYS = 200
H1_TOP_N = 3
H2_VOL_DAYS = 63
H2_MA_FILTER_DAYS = 150

TARGET_INVEST_PCT = 0.99          # Keep 1% cash buffer to prevent fractional rounding BP errors
MIN_ORDER_USD = 25.0              # Alpaca fractional notional safe floor
DEADBAND_WEIGHT_PCT = 0.005       # Skip tiny <0.5% portfolio drift adjustments
RTH_OPEN_BUFFER_MIN = 5           # Wait until 09:35 ET (5m after open)
RTH_CLOSE_BUFFER_MIN = 10         # Stop submitting orders 10m before close
POLL_OPEN_SEC = 60.0
POLL_CLOSED_SEC = 600.0
RETRY_COOLDOWN_SEC = 300.0
STATE_FILE = os.path.join(CURRENT_DIR, "state", "alpaca_etf_rotation_state.json")


def build_http_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def compute_h3_target_weights(
    closes: pd.DataFrame,
    lb_days: int = H1_LOOKBACK_DAYS,
    h1_ma_days: int = H1_MA_FILTER_DAYS,
    top_n: int = H1_TOP_N,
    vol_days: int = H2_VOL_DAYS,
    h2_ma_days: int = H2_MA_FILTER_DAYS,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Computes exact H3 (50% H1 Dual-Momentum Top-3 + 50% H2 Trend Risk-Parity)
    target weights using the last row (latest confirmed daily close) of `closes`.
    Matches tools/research_alpaca_swing_v3.py logic 1:1.
    """
    needed = max(h1_ma_days, lb_days + 1, h2_ma_days, vol_days + 1)
    if closes is None or len(closes) < needed:
        raise ValueError(f"Insufficient daily bars: got {0 if closes is None else len(closes)}, need >= {needed}")

    for col in ALL_UNIVERSE:
        if col not in closes.columns or pd.isna(closes[col].iloc[-1]):
            raise ValueError(f"Missing confirmed close for {col}")

    d = closes.index[-1]

    # H1: 6-asset dual momentum (return_lb > SHY return_lb AND close > MA200), top 3 equal-weight, rest SHY
    ret_lb = closes / closes.shift(lb_days) - 1.0
    ma200 = closes.rolling(h1_ma_days).mean()
    shy_ret = float(ret_lb.loc[d, CASH_ETF])
    if pd.isna(shy_ret):
        raise ValueError("SHY lookback return is NaN")

    elig = []
    for sym in RISK_UNIVERSE:
        r_val = ret_lb.loc[d, sym]
        m_val = ma200.loc[d, sym]
        px = closes.loc[d, sym]
        if pd.notna(r_val) and pd.notna(m_val) and float(r_val) > shy_ret and float(px) > float(m_val):
            elig.append((float(r_val), sym))
    elig.sort(reverse=True)
    h1_picks = [sym for _, sym in elig[:top_n]]

    w_h1: Dict[str, float] = {sym: 1.0 / top_n for sym in h1_picks}
    rem_h1 = 1.0 - len(h1_picks) / float(top_n)
    if rem_h1 > 1e-9:
        w_h1[CASH_ETF] = w_h1.get(CASH_ETF, 0.0) + rem_h1

    # H2: 5-asset 63d inverse-volatility risk parity + MA150 trend gate, unmet -> SHY
    vol = closes[RP_UNIVERSE].pct_change().rolling(vol_days).std()
    ma_rp = closes[RP_UNIVERSE].rolling(h2_ma_days).mean()
    if vol.loc[d].isna().any() or ma_rp.loc[d].isna().any():
        raise ValueError("H2 volatility or MA150 contains NaN on signal date")

    inv_vol = 1.0 / vol.loc[d].clip(lower=1e-4)
    base_w = inv_vol / inv_vol.sum()
    w_h2: Dict[str, float] = {}
    park_h2 = 0.0
    for sym in RP_UNIVERSE:
        if float(closes.loc[d, sym]) > float(ma_rp.loc[d, sym]):
            w_h2[sym] = float(base_w[sym])
        else:
            park_h2 += float(base_w[sym])
    if park_h2 > 1e-9:
        w_h2[CASH_ETF] = park_h2

    # 50:50 ensemble
    combined: Dict[str, float] = {sym: 0.0 for sym in ALL_UNIVERSE}
    for sym, val in w_h1.items():
        combined[sym] = combined.get(sym, 0.0) + 0.5 * val
    for sym, val in w_h2.items():
        combined[sym] = combined.get(sym, 0.0) + 0.5 * val

    diag = {
        "signal_date": str(d),
        "h1_picks": h1_picks,
        "h1_weights": {k: round(v, 4) for k, v in w_h1.items()},
        "h2_weights": {k: round(v, 4) for k, v in w_h2.items()},
        "shy_ret_126d": round(shy_ret * 100.0, 2),
    }
    return combined, diag


def plan_rebalance_orders(
    equity: float,
    current_values: Dict[str, float],
    target_weights: Dict[str, float],
    invest_pct: float = TARGET_INVEST_PCT,
    min_order_usd: float = MIN_ORDER_USD,
    deadband_pct: float = DEADBAND_WEIGHT_PCT,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits rebalance actions into (sell_orders, buy_orders).
    Full liquidations (`target_weight == 0` while currently held) always sell full position.
    Partial adjustments smaller than max(min_order_usd, equity * deadband_pct) are skipped.
    """
    if equity <= 0:
        return [], []

    sells: List[Dict[str, Any]] = []
    buys: List[Dict[str, Any]] = []
    min_delta = max(min_order_usd, equity * deadband_pct)

    all_syms = sorted(set(current_values.keys()) | set(target_weights.keys()))
    for sym in all_syms:
        curr_val = float(current_values.get(sym, 0.0))
        w = float(target_weights.get(sym, 0.0))
        target_val = equity * invest_pct * w
        delta = target_val - curr_val

        if w <= 1e-6 and curr_val > 0.01:
            sells.append({"symbol": sym, "action": "CLOSE_ALL", "notional": round(curr_val, 2)})
        elif delta < -min_delta:
            sells.append({"symbol": sym, "action": "SELL_NOTIONAL", "notional": round(abs(delta), 2)})
        elif delta > min_delta:
            buys.append({"symbol": sym, "action": "BUY_NOTIONAL", "notional": round(delta, 2)})

    return sells, buys


class AlpacaETFRotationBot:
    def __init__(self):
        if "paper-api.alpaca.markets" not in ALPACA_REST_URL:
            raise RuntimeError(
                f"SAFETY ABORT: ALPACA_REST_URL ({ALPACA_REST_URL}) is not paper-api.alpaca.markets!"
            )
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY missing in .env")

        self.session = build_http_session()
        self.state = load_state(STATE_FILE)
        self.last_retry_ts = 0.0
        self.last_closed_log_ts = 0.0

        logger.info("=" * 72)
        logger.info("🚀 [Alpaca ETF Rotation] H3 Multi-Asset Engine (Paper-Only)")
        logger.info(f"   Universe    : {', '.join(ALL_UNIVERSE)} (100% Fractionable ETFs)")
        logger.info(f"   Strategy    : 50% H1 Dual-Mom(lb{H1_LOOKBACK_DAYS}/MA{H1_MA_FILTER_DAYS}/Top{H1_TOP_N}) + "
                    f"50% H2 Risk-Parity(vol{H2_VOL_DAYS}/MA{H2_MA_FILTER_DAYS})")
        logger.info(f"   Allocation  : {TARGET_INVEST_PCT*100:.0f}% Target Invested | Cash Buffer {(1-TARGET_INVEST_PCT)*100:.0f}%")
        logger.info(f"   Last Rebal  : ym={self.state.get('last_rebalance_ym')} | sig={self.state.get('last_signal_date')}")
        logger.info("=" * 72)

    def fetch_clock(self) -> Optional[Dict[str, Any]]:
        try:
            r = self.session.get(f"{ALPACA_REST_URL}/clock", headers=HEADERS, timeout=8)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.warning(f"Clock fetch error: {e}")
        return None

    def fetch_account_and_positions(self) -> Tuple[Optional[Dict[str, Any]], Dict[str, float]]:
        try:
            ra = self.session.get(f"{ALPACA_REST_URL}/account", headers=HEADERS, timeout=8)
            rp = self.session.get(f"{ALPACA_REST_URL}/positions", headers=HEADERS, timeout=8)
            if ra.status_code != 200 or rp.status_code != 200:
                logger.warning(f"Account/positions HTTP {ra.status_code}/{rp.status_code}")
                return None, {}
            acc = ra.json()
            pos_map: Dict[str, float] = {}
            for p in rp.json():
                sym = p.get("symbol")
                if sym:
                    pos_map[sym] = float(p.get("market_value", 0.0) or 0.0)
            return acc, pos_map
        except Exception as e:
            logger.warning(f"Account/positions error: {e}")
            return None, {}

    def fetch_confirmed_daily_closes(self, cutoff_date: date) -> Optional[pd.DataFrame]:
        """Fetches split+dividend adjusted 1Day closes strictly prior to `cutoff_date`."""
        start_dt = (cutoff_date - timedelta(days=380)).strftime("%Y-%m-%dT00:00:00Z")
        for feed in ("sip", "iex"):
            try:
                all_bars: Dict[str, List[Dict[str, Any]]] = {s: [] for s in ALL_UNIVERSE}
                page_token = None
                while True:
                    params = {
                        "symbols": ",".join(ALL_UNIVERSE),
                        "timeframe": "1Day",
                        "start": start_dt,
                        "limit": 10000,
                        "adjustment": "all",
                        "feed": feed,
                        "sort": "asc",
                    }
                    if page_token:
                        params["page_token"] = page_token
                    r = self.session.get(f"{ALPACA_DATA_URL}/stocks/bars", headers=HEADERS, params=params, timeout=12)
                    if r.status_code != 200:
                        logger.warning(f"Alpaca bars feed={feed} HTTP {r.status_code}: {r.text[:160]}")
                        all_bars = {}
                        break
                    payload = r.json()
                    for s, items in (payload.get("bars") or {}).items():
                        all_bars.setdefault(s, []).extend(items)
                    page_token = payload.get("next_page_token")
                    if not page_token:
                        break

                if all_bars:
                    series_map = {}
                    for s in ALL_UNIVERSE:
                        pts = {}
                        for b in all_bars.get(s, []):
                            d_bar = pd.Timestamp(b["t"]).tz_convert("America/New_York").date()
                            if d_bar < cutoff_date:
                                pts[d_bar] = float(b["c"])
                        if pts:
                            series_map[s] = pd.Series(pts).sort_index()
                    df = pd.DataFrame(series_map).sort_index()
                    if len(df) >= 210 and not df.iloc[-1].isna().any():
                        age_days = (cutoff_date - df.index[-1]).days
                        if age_days > 4:
                            logger.warning(f"⚠️ Latest confirmed bar ({df.index[-1]}) is {age_days}d behind cutoff {cutoff_date} (feed={feed})")
                        logger.info(f"📊 Loaded {len(df)} confirmed daily bars (feed={feed}, last={df.index[-1]})")
                        return df
            except Exception as e:
                logger.warning(f"Alpaca bars error (feed={feed}): {e}")

        # Fallback to yfinance
        try:
            import yfinance as yf
            raw = yf.download(ALL_UNIVERSE, period="18mo", interval="1d", auto_adjust=True, progress=False, group_by="ticker")
            df = pd.DataFrame({s: raw[s]["Close"] for s in ALL_UNIVERSE})
            df.index = pd.to_datetime(df.index).date
            df = df[df.index < cutoff_date].dropna()
            if len(df) >= 210:
                logger.info(f"📊 Loaded {len(df)} confirmed daily bars (fallback=yfinance, last={df.index[-1]})")
                return df
        except Exception as e:
            logger.error(f"yfinance fallback failed: {e}")

        return None

    def should_rebalance(self, et_now: datetime) -> Tuple[bool, str]:
        current_ym = et_now.strftime("%Y-%m")
        if not self.state.get("initialized"):
            return True, f"INITIAL_DEPLOY ({current_ym})"
        if self.state.get("last_rebalance_ym") != current_ym:
            return True, f"MONTHLY_REBALANCE ({self.state.get('last_rebalance_ym')} -> {current_ym})"
        return False, "UP_TO_DATE"

    def _cancel_all_open_orders(self):
        try:
            r = self.session.delete(f"{ALPACA_REST_URL}/orders", headers=HEADERS, timeout=8)
            if r.status_code in (200, 207):
                canceled = r.json() if r.text else []
                if canceled:
                    logger.info(f"🧹 Cancelled {len(canceled)} open order(s) before rebalance.")
                    time.sleep(1.5)
        except Exception as e:
            logger.warning(f"Cancel open orders warn: {e}")

    def _submit_market_notional(self, sym: str, side: str, notional: float, ym: str) -> Optional[str]:
        payload = {
            "symbol": sym,
            "notional": f"{notional:.2f}",
            "side": side,
            "type": "market",
            "time_in_force": "day",
            "client_order_id": f"etfrot-{ym}-{sym}-{side}-{int(time.time())}",
        }
        try:
            r = self.session.post(f"{ALPACA_REST_URL}/orders", headers=HEADERS, json=payload, timeout=10)
            if r.status_code in (200, 201):
                oid = r.json().get("id") or "SUBMITTED"
                logger.info(f"📤 [{side.upper()}] {sym} ${notional:,.2f} (ord={oid})")
                return oid
            logger.error(f"❌ Order rejected [{side} {sym} ${notional:,.2f}]: HTTP {r.status_code} {r.text[:200]}")
        except Exception as e:
            logger.error(f"❌ Order exception [{side} {sym}]: {e}")
        return None

    def _close_full_position(self, sym: str) -> Optional[str]:
        try:
            r = self.session.delete(f"{ALPACA_REST_URL}/positions/{sym}", headers=HEADERS, timeout=10)
            if r.status_code in (200, 201):
                oid = r.json().get("id") if r.text else None
                logger.info(f"📤 [CLOSE_ALL] {sym} (HTTP {r.status_code}, ord={oid})")
                return oid or "SUBMITTED"
            if r.status_code == 404:
                logger.info(f"📤 [CLOSE_ALL] {sym} already flat (HTTP 404)")
                return "ALREADY_FLAT"
            logger.error(f"❌ Close position failed [{sym}]: HTTP {r.status_code} {r.text[:200]}")
        except Exception as e:
            logger.error(f"❌ Close position exception [{sym}]: {e}")
        return None

    def _wait_for_orders_settlement(self, order_ids: List[str], timeout_sec: float = 30.0) -> bool:
        """Polls submitted sell/close order IDs until terminal status or timeout."""
        pending = {oid for oid in order_ids if oid and oid not in ("ALREADY_FLAT", "SUBMITTED")}
        if not pending:
            time.sleep(2.0)
            return True

        terminal = {"filled", "canceled", "expired", "rejected", "replaced", "done_for_day"}
        deadline = time.time() + timeout_sec
        while pending and time.time() < deadline:
            time.sleep(1.5)
            for oid in list(pending):
                try:
                    r = self.session.get(f"{ALPACA_REST_URL}/orders/{oid}", headers=HEADERS, timeout=8)
                    if r.status_code == 200:
                        st = str(r.json().get("status", "")).lower()
                        if st in terminal:
                            pending.discard(oid)
                except Exception as e:
                    logger.debug(f"Order status poll err ({oid}): {e}")
        if pending:
            logger.warning(f"⚠️ {len(pending)} sell order(s) still pending after {timeout_sec:.0f}s: {sorted(pending)}")
            return False
        logger.info("✅ All sell orders settled before buy leg.")
        return True

    def execute_rebalance(self, et_now: datetime, reason: str):
        acc, curr_vals = self.fetch_account_and_positions()
        if not acc:
            return
        equity = float(acc.get("equity", 0.0) or 0.0)
        if equity < 100.0:
            logger.warning(f"Equity too low (${equity:,.2f}) -> skipping rebalance")
            return

        closes = self.fetch_confirmed_daily_closes(et_now.date())
        if closes is None:
            self.last_retry_ts = time.time()
            return

        weights, diag = compute_h3_target_weights(closes)
        sells, buys = plan_rebalance_orders(equity, curr_vals, weights)
        current_ym = et_now.strftime("%Y-%m")

        w_str = ", ".join(f"{k}:{v*100:.1f}%" for k, v in sorted(weights.items()) if v > 0.001)
        logger.info(f"🎯 [{reason}] SignalDate={diag['signal_date']} | Targets: {w_str}")

        self._cancel_all_open_orders()

        all_ok = True
        sell_order_ids: List[str] = []
        if sells:
            for o in sells:
                oid = self._close_full_position(o["symbol"]) if o["action"] == "CLOSE_ALL" \
                    else self._submit_market_notional(o["symbol"], "sell", o["notional"], current_ym)
                if oid:
                    sell_order_ids.append(oid)
                else:
                    all_ok = False
            self._wait_for_orders_settlement(sell_order_ids, timeout_sec=30.0)
            acc2, curr_vals2 = self.fetch_account_and_positions()
            if acc2:
                equity = float(acc2.get("equity", equity) or equity)
                _, buys = plan_rebalance_orders(equity, curr_vals2, weights)

        if buys:
            acc_bp, _ = self.fetch_account_and_positions()
            buying_power = float((acc_bp or acc).get("buying_power", equity) or equity)
            total_buy = sum(b["notional"] for b in buys)
            scale = min(1.0, (buying_power * 0.98) / total_buy) if total_buy > 0 else 1.0
            for o in buys:
                amt = round(o["notional"] * scale, 2)
                if amt < MIN_ORDER_USD:
                    continue
                if not self._submit_market_notional(o["symbol"], "buy", amt, current_ym):
                    all_ok = False

        if all_ok:
            self.state["initialized"] = True
            self.state["last_rebalance_ym"] = current_ym
            self.state["last_signal_date"] = diag["signal_date"]
            self.state["last_weights"] = {k: round(v, 4) for k, v in weights.items() if v > 0.001}
            self.state["last_rebalance_at"] = datetime.now(timezone.utc).isoformat()
            save_state(STATE_FILE, self.state)

            msg = (
                f"📊 [Alpaca H3 ETF 로테이션 리밸런싱 완료]\n"
                f"• 사유: {reason} (확정봉: {diag['signal_date']})\n"
                f"• 계좌 평가액: ${equity:,.2f}\n"
                f"• 목표 비중: {w_str}\n"
                f"• 매도 {len(sells)}건 / 매수 {len(buys)}건 체결 요청 완료"
            )
            logger.info("✅ Rebalance completed and state saved.")
            send_telegram(msg)
        else:
            self.last_retry_ts = time.time()
            logger.warning("⚠️ Partial order failure during rebalance -> will retry after cooldown.")
            send_telegram(f"⚠️ [Alpaca ETF Rotation] 일부 주문 실패 — 5분 후 재시도합니다 ({reason})")

    def preview_once(self):
        """Logs current account & target weights preview at startup without placing orders."""
        et_now = pd.Timestamp.now(tz="America/New_York").to_pydatetime()
        acc, pos = self.fetch_account_and_positions()
        cutoff = et_now.date() + timedelta(days=1) if et_now.hour >= 16 else et_now.date()
        closes = self.fetch_confirmed_daily_closes(cutoff)
        if acc and closes is not None:
            eq = float(acc.get("equity", 0.0) or 0.0)
            weights, diag = compute_h3_target_weights(closes)
            w_str = ", ".join(f"{k}:{v*100:.1f}%" for k, v in sorted(weights.items()) if v > 0.001)
            logger.info(f"🔎 [Startup Preview] Equity=${eq:,.2f} | Pos={list(pos.keys())} | Sig={diag['signal_date']} -> {w_str}")

    def run(self):
        logger.info("🟢 [Alpaca ETF Rotation] Engine running 24/7 (Paper Forward Test)...")
        send_telegram("[Alpaca ETF Rotation] H3 다자산 ETF 로테이션 엔진 시작 (Paper)")
        self.preview_once()

        while True:
            try:
                clock = self.fetch_clock()
                if not clock or not clock.get("is_open"):
                    if time.time() - self.last_closed_log_ts >= POLL_CLOSED_SEC:
                        nxt = (clock or {}).get("next_open", "unknown")
                        logger.info(f"💤 [Market Closed] Next Open: {nxt} | LastRebal={self.state.get('last_rebalance_ym')}")
                        self.last_closed_log_ts = time.time()
                    time.sleep(60.0)
                    continue

                # Regular Trading Hours check (09:35 ~ 15:50 ET)
                ts_str = clock.get("timestamp")
                et_now = pd.Timestamp(ts_str).tz_convert("America/New_York").to_pydatetime()
                mins_since_midnight = et_now.hour * 60 + et_now.minute
                in_exec_window = (9 * 60 + RTH_OPEN_BUFFER_MIN) <= mins_since_midnight <= (16 * 60 - RTH_CLOSE_BUFFER_MIN)

                due, reason = self.should_rebalance(et_now)
                if due and in_exec_window and (time.time() - self.last_retry_ts >= RETRY_COOLDOWN_SEC):
                    self.execute_rebalance(et_now, reason)
                elif due and not in_exec_window:
                    logger.debug(f"Waiting for RTH execution window (09:35~15:50 ET), now={et_now.strftime('%H:%M')}")

                time.sleep(POLL_OPEN_SEC)
            except KeyboardInterrupt:
                logger.info("🛑 Stopping Alpaca ETF Rotation Bot...")
                break
            except Exception as e:
                logger.error(f"Loop exception: {e}", exc_info=True)
                send_telegram(f"[Alpaca ETF Rotation] ❌ 루프 예외: {e}", dedupe_key="etf_rot_exc", min_interval_sec=1800)
                time.sleep(30.0)


if __name__ == "__main__":
    bot = AlpacaETFRotationBot()
    bot.run()
