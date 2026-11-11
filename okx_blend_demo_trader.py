# -*- coding: utf-8 -*-
"""
okx_blend_demo_trader.py — OKX BLEND 65/35 Autonomous Demo Trader (100% Simulated)
===================================================================================
Strategy:
  - 65% Multi-Speed 4H Donchian Trend (24/12, 36/18, 48/24) across 8 Majors + BTC 1D EMA50 gate
  - 35% 1D Core-3 (BTC/ETH/SOL) RSI(2) < 10 Pullback inside MA100 uptrend -> exit > MA5
  - Long-Only + USDT cash defense during bear/chop regimes (2022~2026 OOS CAGR +16.7~18.4%, MDD -15~16%)
  - 3x Isolated leverage cap with -10% spot emergency disaster SL on exchange
  - 100% Simulated Demo Mode enforced (`x-simulated-trading: 1`)
"""

import os
import sys
import time
import hmac
import base64
import json
import logging
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import requests
import pandas as pd
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

from bot_utils import send_telegram, load_state, save_state  # noqa: E402
from okx_blend_strategy import (  # noqa: E402
    SYMS_8,
    compute_blend_target_weights,
    plan_delta_orders,
)

LOG_FILE = os.path.join(CURRENT_DIR, "okx_blend_demo_trader.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"),
    ],
)
logger = logging.getLogger("OKXBlendDemoTrader")

OKX_API_KEY = os.getenv("OKX_DEMO_API_KEY", "").strip()
OKX_SECRET_KEY = os.getenv("OKX_DEMO_SECRET", "").strip()
OKX_PASSPHRASE = os.getenv("OKX_DEMO_PASSPHRASE", "").strip()
OKX_BASE_URL = "https://www.okx.com"

LEVERAGE = 3                       # Safe 3x isolated ceiling (strategy uses ~0.0x~1.5x gross weight)
EMERGENCY_SL_PCT = 0.10            # -10% spot disaster stop on exchange (far inside 3x ~33% liq)
LOOP_INTERVAL_SEC = 15
HEARTBEAT_LOG_SEC = 300.0
ACCOUNT_MODE_RECHECK_SEC = 300.0
RETRY_COOLDOWN_SEC = 180.0
STATE_FILE = os.path.join(CURRENT_DIR, "state", "okx_blend_demo_state.json")


def _round_step(value: float, step: str, rounding=ROUND_DOWN) -> Decimal:
    step_d = Decimal(step)
    return (Decimal(str(value)) / step_d).to_integral_value(rounding=rounding) * step_d


def _fmt(d: Decimal, step: str) -> str:
    exp = Decimal(step).normalize().as_tuple().exponent
    decimals = max(0, -exp) if isinstance(exp, int) else 0
    return f"{d:.{decimals}f}"


class OKXBlendDemoTrader:
    def __init__(self):
        self.api_key = OKX_API_KEY
        self.secret_key = OKX_SECRET_KEY
        self.passphrase = OKX_PASSPHRASE
        self.session = requests.Session()
        self.last_error = ""

        self.account_equity = 0.0
        self.usdt_equity = 0.0
        self.available_usdt = 0.0
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.inst_cache: Dict[str, Dict[str, str]] = {}
        self.trading_enabled = False
        self.last_mode_check = 0.0
        self.last_heartbeat_ts = 0.0
        self.last_retry_ts = 0.0

        self.state = load_state(STATE_FILE)

        logger.info("=" * 74)
        logger.info("🚀 [OKX BLEND Demo Trader] 65% Multi-4H Donchian + 35% 1D RSI(2) Pullback")
        logger.info(f"   Universe     : {', '.join(s.split('-')[0] for s in SYMS_8)}")
        logger.info(f"   Safety Level : 100% Simulated Demo (x-simulated-trading: 1) | {LEVERAGE}x Isolated")
        logger.info(f"   Last Rebal   : 4H={self.state.get('last_4h_ts')} | 1D={self.state.get('last_1d_ts')}")
        logger.info("=" * 74)

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
            "x-simulated-trading": "1",
        }

    def _request(self, method: str, path: str, params: Optional[Dict] = None, data: Optional[Any] = None) -> Optional[Dict]:
        body = json.dumps(data) if data else ""
        headers = self._sign_headers(method, path, body)
        url = f"{OKX_BASE_URL}{path}"
        self.last_error = ""
        try:
            if method.upper() == "GET":
                resp = self.session.get(url, headers=headers, params=params, timeout=8)
            elif method.upper() == "POST":
                resp = self.session.post(url, headers=headers, data=body, timeout=8)
            else:
                return None

            if resp.status_code == 200:
                res_json = resp.json()
                if res_json.get("code") == "0":
                    return res_json
                details = [f"{d.get('sCode')}:{d.get('sMsg')}" for d in res_json.get("data", [])
                           if isinstance(d, dict) and d.get("sCode")]
                self.last_error = f"{res_json.get('code')} {res_json.get('msg')} {details}".strip()
                logger.warning(f"⚠️ OKX API code error: {self.last_error} ({path})")
            else:
                self.last_error = f"HTTP {resp.status_code} {resp.text[:200]}"
                logger.warning(f"⚠️ OKX HTTP error: {self.last_error} ({path})")
        except Exception as e:
            self.last_error = str(e)
            logger.warning(f"⚠️ OKX exception ({path}): {e}")
        return None

    # ── ACCOUNT & INSTRUMENTS ──
    def check_account_mode(self) -> bool:
        now = time.time()
        if self.trading_enabled or now - self.last_mode_check < ACCOUNT_MODE_RECHECK_SEC:
            return self.trading_enabled
        self.last_mode_check = now
        res = self._request("GET", "/api/v5/account/config")
        if not res or not res.get("data"):
            return False
        cfg = res["data"][0]
        if cfg.get("acctLv") in ("2", "3", "4") and cfg.get("posMode") == "net_mode":
            logger.info(f"✅ [Account Mode OK] acctLv={cfg.get('acctLv')} posMode=net_mode")
            self.trading_enabled = True
        else:
            logger.error(f"⛔ Unsupported OKX account mode: {cfg}")
            self.trading_enabled = False
        return self.trading_enabled

    def sync_account(self) -> bool:
        res = self._request("GET", "/api/v5/account/balance")
        if res and res.get("data"):
            acc = res["data"][0]
            self.account_equity = float(acc.get("totalEq", 0.0) or 0.0)
            for d in acc.get("details", []):
                if d.get("ccy") == "USDT":
                    self.available_usdt = float(d.get("availBal") or d.get("availEq") or 0.0)
                    self.usdt_equity = float(d.get("eq") or self.available_usdt or 0.0)
            return True
        return False

    def sync_positions(self) -> Dict[str, Dict[str, Any]]:
        res = self._request("GET", "/api/v5/account/positions?instType=SWAP")
        if res is not None:
            new_pos = {}
            for p in res.get("data", []):
                inst_id = p.get("instId")
                if inst_id and abs(float(p.get("pos", 0.0) or 0.0)) > 0:
                    new_pos[inst_id] = p
            self.positions = new_pos
        return self.positions

    def get_instrument(self, inst_id: str) -> Optional[Dict[str, str]]:
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

    def get_ticker_price(self, inst_id: str) -> float:
        try:
            r = self.session.get(f"{OKX_BASE_URL}/api/v5/market/ticker", params={"instId": inst_id}, timeout=5).json()
            d = r.get("data", [])
            if d:
                return float(d[0].get("last") or 0.0)
        except Exception:
            pass
        return 0.0

    # ── MARKET DATA (CONFIRMED BARS ONLY) ──
    def fetch_confirmed_bars(self, inst_id: str, bar: str, limit_bars: int) -> Optional[pd.DataFrame]:
        """Fetches `limit_bars` confirmed bars from live + history endpoints."""
        rows = []
        try:
            r0 = self.session.get(
                f"{OKX_BASE_URL}/api/v5/market/candles",
                params={"instId": inst_id, "bar": bar, "limit": "100"},
                timeout=6,
            ).json().get("data", [])
            rows.extend([x for x in r0 if x[8] == "1"])
        except Exception as e:
            logger.warning(f"Candles err ({inst_id} {bar}): {e}")
            return None

        after = str(min(int(x[0]) for x in rows)) if rows else ""
        while len({int(x[0]) for x in rows}) < limit_bars:
            params = {"instId": inst_id, "bar": bar, "limit": "100"}
            if after:
                params["after"] = after
            try:
                data = self.session.get(
                    f"{OKX_BASE_URL}/api/v5/market/history-candles",
                    params=params,
                    timeout=6,
                ).json().get("data", [])
            except Exception as e:
                logger.warning(f"History candles err ({inst_id} {bar}): {e}")
                break
            if not data:
                break
            rows.extend([x for x in data if x[8] == "1"])
            after = str(min(int(x[0]) for x in data))
            time.sleep(0.04)

        dedup = {int(x[0]): x for x in rows if x[8] == "1"}
        if not dedup:
            return None
        ordered = [dedup[k] for k in sorted(dedup.keys())[-limit_bars:]]
        df = pd.DataFrame(ordered, columns=["ts", "open", "high", "low", "close", "vol", "vc", "vcq", "confirm"])
        df["dt"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
        for col in ("open", "high", "low", "close"):
            df[col] = df[col].astype(float)
        return df.set_index("dt")[["open", "high", "low", "close"]]

    def get_latest_confirmed_timestamps(self) -> Tuple[Optional[str], Optional[str]]:
        """Lightweight poll of BTC 4H & 1D latest confirmed bar timestamps."""
        try:
            r4 = self.session.get(
                f"{OKX_BASE_URL}/api/v5/market/candles",
                params={"instId": "BTC-USDT-SWAP", "bar": "4H", "limit": "5"},
                timeout=5,
            ).json().get("data", [])
            r1 = self.session.get(
                f"{OKX_BASE_URL}/api/v5/market/candles",
                params={"instId": "BTC-USDT-SWAP", "bar": "1D", "limit": "5"},
                timeout=5,
            ).json().get("data", [])
            c4_ts = [int(x[0]) for x in r4 if x[8] == "1"]
            c1_ts = [int(x[0]) for x in r1 if x[8] == "1"]
            ts4 = str(max(c4_ts)) if c4_ts else None
            ts1 = str(max(c1_ts)) if c1_ts else None
            return ts4, ts1
        except Exception as e:
            logger.debug(f"Latest ts poll err: {e}")
            return None, None

    # ── EXECUTION ──
    def _get_pending_algos(self, inst_id: str) -> List[Dict[str, Any]]:
        res = self._request("GET", f"/api/v5/trade/orders-algo-pending?ordType=conditional,oco&instId={inst_id}")
        return [a for a in (res or {}).get("data", []) if a.get("algoId")]

    def _cancel_algo_orders(self, inst_id: str):
        algos = [{"algoId": a["algoId"], "instId": inst_id} for a in self._get_pending_algos(inst_id)]
        if algos:
            self._request("POST", "/api/v5/trade/cancel-algos", data=algos)

    def sync_emergency_sl_orders(self, prices: Dict[str, float]) -> bool:
        """
        Ensures every open long position has an exchange-side conditional -10% SL
        matching 100% of current position contracts (fixes scale-in / reduce SL drift).
        """
        all_ok = True
        for sym, p in list(self.positions.items()):
            pos_amt = float(p.get("pos", 0.0) or 0.0)
            if pos_amt <= 1e-9:
                continue
            info = self.get_instrument(sym)
            if not info:
                all_ok = False
                continue
            lot_sz = Decimal(info["lotSz"])
            pos_sz = _round_step(pos_amt, info["lotSz"], ROUND_DOWN)
            if pos_sz < Decimal(info["minSz"]):
                continue
            ref_px = float(p.get("avgPx") or prices.get(sym, 0.0) or 0.0)
            if ref_px <= 0:
                continue
            target_sl = _round_step(ref_px * (1.0 - EMERGENCY_SL_PCT), info["tickSz"], ROUND_HALF_UP)

            pending = self._get_pending_algos(sym)
            if len(pending) == 1:
                cur_sz = Decimal(str(pending[0].get("sz") or "0"))
                cur_sl = float(pending[0].get("slTriggerPx") or 0.0)
                sz_match = abs(cur_sz - pos_sz) <= (lot_sz * Decimal("0.5"))
                sl_match = float(target_sl) > 0 and abs(cur_sl - float(target_sl)) / float(target_sl) <= 0.005
                if sz_match and sl_match:
                    continue

            if pending:
                self._cancel_algo_orders(sym)

            mgn_mode = p.get("mgnMode") or "isolated"
            sl_payload = {
                "instId": sym,
                "tdMode": mgn_mode,
                "side": "sell",
                "posSide": "net",
                "ordType": "conditional",
                "sz": _fmt(pos_sz, info["lotSz"]),
                "reduceOnly": True,
                "slTriggerPx": _fmt(target_sl, info["tickSz"]),
                "slOrdPx": "-1",
                "slTriggerPxType": "last",
            }
            res = self._request("POST", "/api/v5/trade/order-algo", data=sl_payload)
            if res and res.get("data"):
                algo_id = res["data"][0].get("algoId")
                logger.info(f"🛡️ [SL Sync] {sym} sz={sl_payload['sz']} SL=${sl_payload['slTriggerPx']} algo={algo_id}")
            else:
                logger.warning(f"⚠️ [SL Sync Failed] {sym}: {self.last_error}")
                all_ok = False
        return all_ok

    def _close_position(self, inst_id: str, mgn_mode: Optional[str] = None) -> bool:
        mode = mgn_mode or self.positions.get(inst_id, {}).get("mgnMode") or "isolated"
        res = self._request("POST", "/api/v5/trade/close-position", data={
            "instId": inst_id,
            "mgnMode": mode,
            "autoCxl": True,
        })
        if res:
            self._cancel_algo_orders(inst_id)
            logger.info(f"✅ [Close Position] {inst_id} ({mode})")
            return True
        logger.error(f"❌ [Close Failed] {inst_id}: {self.last_error}")
        return False

    def _submit_market_contracts(self, inst_id: str, side: str, usd_delta: float, px: float, attach_sl: bool) -> bool:
        info = self.get_instrument(inst_id)
        if not info or px <= 0:
            return False
        self._request("POST", "/api/v5/account/set-leverage", data={
            "instId": inst_id,
            "lever": str(LEVERAGE),
            "mgnMode": "isolated",
        })
        ct_val = float(info["ctVal"])
        contracts = _round_step(usd_delta / (px * ct_val), info["lotSz"], ROUND_DOWN)
        if contracts < Decimal(info["minSz"]):
            logger.info(f"⏭️ {inst_id}: rounded contracts {contracts} < minSz {info['minSz']} -> skip")
            return True

        order_data: Dict[str, Any] = {
            "instId": inst_id,
            "tdMode": "isolated",
            "side": side,
            "posSide": "net",
            "ordType": "market",
            "sz": _fmt(contracts, info["lotSz"]),
            "clOrdId": f"blend{int(time.time() * 1000)}",
        }
        if side == "sell":
            order_data["reduceOnly"] = True
        elif attach_sl:
            sl_px = px * (1.0 - EMERGENCY_SL_PCT)
            tick = info["tickSz"]
            order_data["attachAlgoOrds"] = [{
                "slTriggerPx": _fmt(_round_step(sl_px, tick, ROUND_HALF_UP), tick),
                "slOrdPx": "-1",
                "slTriggerPxType": "last",
            }]

        res = self._request("POST", "/api/v5/trade/order", data=order_data)
        if res and res.get("data"):
            oid = res["data"][0].get("ordId")
            approx_usd = float(contracts) * ct_val * px
            logger.info(f"📤 [{side.upper()}] {inst_id} {order_data['sz']}ct (~${approx_usd:,.1f} @ ${px:,.4f}) ord={oid}")
            return True
        logger.error(f"❌ [Order Failed] {inst_id} {side}: {self.last_error}")
        return False

    def close_legacy_positions_if_any(self) -> bool:
        """Closes any legacy short positions (or non-SYMS_8 positions) cleanly before Blend runs."""
        closed_any = False
        for sym, p in list(self.positions.items()):
            pos_amt = float(p.get("pos", 0.0) or 0.0)
            if pos_amt < 0 or sym not in SYMS_8:
                logger.info(f"🧹 [Legacy Position Cleanup] Closing {sym} (pos={pos_amt}, upl={p.get('upl')})")
                if self._close_position(sym):
                    closed_any = True
        if closed_any:
            time.sleep(2.0)
            self.sync_account()
            self.sync_positions()
        return closed_any

    def evaluate_and_rebalance(self, ts4: str, ts1: str, reason: str):
        logger.info(f"🔄 [Rebalance Check] {reason} (4H={ts4}, 1D={ts1})")
        h4_map, l4_map, c4_map, c1_map = {}, {}, {}, {}
        for s in SYMS_8:
            d4 = self.fetch_confirmed_bars(s, "4H", 240)
            d1 = self.fetch_confirmed_bars(s, "1D", 150)
            if d4 is None or d1 is None or len(d4) < 200 or len(d1) < 110:
                logger.warning(f"⚠️ Incomplete candle history for {s} -> defer rebalance")
                self.last_retry_ts = time.time()
                return
            h4_map[s], l4_map[s], c4_map[s] = d4["high"], d4["low"], d4["close"]
            c1_map[s] = d1["close"]

        h4_df = pd.DataFrame(h4_map).sort_index().ffill()
        l4_df = pd.DataFrame(l4_map).sort_index().ffill()
        c4_df = pd.DataFrame(c4_map).sort_index().ffill()
        c1_df = pd.DataFrame(c1_map).sort_index().ffill()

        target_weights, diag = compute_blend_target_weights(h4_df, l4_df, c4_df, c1_df)
        w_str = ", ".join(f"{k.split('-')[0]}:{v*100:.1f}%" for k, v in target_weights.items() if v > 0.001) or "100% USDT (Cash)"
        logger.info(
            f"🎯 [Target Signal] BTC_1D_Bull={diag['btc_1d_bull']} (${diag['btc_1d_px']:,} vs EMA50 ${diag['btc_1d_ema50']:,}) | "
            f"MR_Active={diag['mr_active']} RSI2={diag['rsi2']} | Gross={diag['gross_leverage']:.2f}x -> {w_str}"
        )

        # Clean up any legacy short first so available margin & net positions are clean
        self.close_legacy_positions_if_any()

        prices, ct_vals, min_szs, curr_cts = {}, {}, {}, {}
        for s in SYMS_8:
            info = self.get_instrument(s)
            px = self.get_ticker_price(s) or float(c4_df[s].iloc[-1])
            if not info or px <= 0:
                self.last_retry_ts = time.time()
                return
            prices[s] = px
            ct_vals[s] = float(info["ctVal"])
            min_szs[s] = float(info["minSz"])
            curr_cts[s] = float(self.positions.get(s, {}).get("pos", 0.0) or 0.0)

        sells, buys = plan_delta_orders(
            equity_usdt=self.usdt_equity,
            prices=prices,
            current_contracts=curr_cts,
            target_weights=target_weights,
            ct_vals=ct_vals,
            min_szs=min_szs,
        )

        all_ok = True
        executed = 0

        for o in sells:
            sym = o["symbol"]
            if o["action"] in ("CLOSE_LONG", "CLOSE_BELOW_MIN", "CLOSE_LEGACY_SHORT"):
                ok = self._close_position(sym)
            else:
                ok = self._submit_market_contracts(sym, "sell", o["delta_usd"], prices[sym], attach_sl=False)
            if ok:
                executed += 1
            else:
                all_ok = False

        if sells:
            time.sleep(2.0)
            self.sync_account()
            self.sync_positions()

        if buys:
            # Cap buys within available margin * LEVERAGE * 0.90 safety buffer
            max_new_notional = self.available_usdt * LEVERAGE * 0.90
            tot_buy = sum(b["delta_usd"] for b in buys)
            scale = min(1.0, max_new_notional / tot_buy) if tot_buy > 0 else 1.0
            for o in buys:
                sym = o["symbol"]
                amt = o["delta_usd"] * scale
                is_fresh = curr_cts.get(sym, 0.0) <= 1e-9
                ok = self._submit_market_contracts(sym, "buy", amt, prices[sym], attach_sl=is_fresh)
                if ok:
                    executed += 1
                else:
                    all_ok = False

        if executed > 0 or self.positions:
            if executed > 0:
                time.sleep(1.5)
                self.sync_account()
                self.sync_positions()
            if self.positions:
                if not self.sync_emergency_sl_orders(prices):
                    logger.warning("⚠️ Emergency SL sync incomplete (will retry next bar)")

        if all_ok:
            self.state["last_4h_ts"] = ts4
            self.state["last_1d_ts"] = ts1
            self.state["last_weights"] = {k: round(v, 4) for k, v in target_weights.items() if v > 0.001}
            self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
            save_state(STATE_FILE, self.state)

            if executed > 0 or reason == "INITIAL_SYNC":
                send_telegram(
                    f"🪙 [OKX Demo BLEND 65/35 리밸런싱]\n"
                    f"• 사유: {reason} | 체결 {executed}건\n"
                    f"• USDT 자본: ${self.usdt_equity:,.2f} (레버리지 {diag['gross_leverage']:.2f}x)\n"
                    f"• BTC 일봉 추세: {'상승(EMA50 위)' if diag['btc_1d_bull'] else '하락/관망(EMA50 아래)'}\n"
                    f"• 목표 비중: {w_str}"
                )
        else:
            self.last_retry_ts = time.time()
            logger.warning("⚠️ Partial order failure -> retrying after cooldown")

    def run(self):
        if not (self.api_key and self.secret_key and self.passphrase):
            logger.critical("⛔ OKX Demo API credentials missing in .env -> exiting")
            sys.exit(1)

        logger.info("🟢 [OKX BLEND Demo Trader] Engine running 24/7...")
        send_telegram("[OKX Demo] BLEND 65/35 (4H 추세 + 1D RSI2) 엔진 시작")

        while True:
            try:
                if not self.sync_account():
                    time.sleep(LOOP_INTERVAL_SEC)
                    continue
                self.sync_positions()

                if self.check_account_mode():
                    ts4, ts1 = self.get_latest_confirmed_timestamps()
                    if ts4 and ts1:
                        is_init = not self.state.get("last_4h_ts")
                        new_4h = ts4 != self.state.get("last_4h_ts")
                        new_1d = ts1 != self.state.get("last_1d_ts")
                        if (is_init or new_4h or new_1d) and (time.time() - self.last_retry_ts >= RETRY_COOLDOWN_SEC):
                            reason = "INITIAL_SYNC" if is_init else ("NEW_1D_BAR" if new_1d else "NEW_4H_BAR")
                            self.evaluate_and_rebalance(ts4, ts1, reason)

                if time.time() - self.last_heartbeat_ts >= HEARTBEAT_LOG_SEC:
                    pos_desc = {s.split("-")[0]: f"{float(p.get('pos',0)):+.2f}ct(${float(p.get('upl',0)):+.1f})"
                                for s, p in self.positions.items()}
                    logger.info(
                        f"💼 [Demo Sync] USDT Eq: ${self.usdt_equity:,.2f} | Avail: ${self.available_usdt:,.2f} | "
                        f"Pos: {pos_desc or '100% USDT Cash'}"
                    )
                    self.last_heartbeat_ts = time.time()

                time.sleep(LOOP_INTERVAL_SEC)
            except KeyboardInterrupt:
                logger.info("🛑 Stopping OKX BLEND Demo Trader...")
                save_state(STATE_FILE, self.state)
                break
            except Exception as e:
                logger.error(f"Loop exception: {e}", exc_info=True)
                send_telegram(f"[OKX Demo] ❌ 루프 예외: {e}", dedupe_key="okx_blend_exc", min_interval_sec=1800)
                time.sleep(15.0)


if __name__ == "__main__":
    trader = OKXBlendDemoTrader()
    trader.run()

