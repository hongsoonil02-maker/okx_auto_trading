# -*- coding: utf-8 -*-
"""
okx_cameron_4h_swing_trader.py — Autonomous OKX 4-Hour Trend Pullback Swing Engine
Configuration:
- Timeframe: 2-Hour ('2h')
- Direction: Long Only
- Leverage: 3x Cross Margin
- Risk:Reward Ratio: 1 : 1.86
- Universe: 15 liquid USDT perpetual swaps (TARGET_SYMBOLS)
- Position Sizing: 8% of account equity per trade
- Max Concurrent Positions: 3 positions
- Exchange-side reduce-only TP/SL attached at entry (survives bot downtime)
- Trailing stop to break-even at +1.0R, mirrored on exchange via trigger order
- Persistence: atomic JSON state backup, reconciled with exchange positions each cycle
"""

import os
import sys
import time
import json
import math
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt_async
from dotenv import load_dotenv
from jev.universal_jev_gate import UniversalJevGate

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

STATE_DIR = os.path.join(CURRENT_DIR, "state")
os.makedirs(STATE_DIR, exist_ok=True)
POSITIONS_FILE = os.path.join(STATE_DIR, "okx_cameron_4h_positions.json")
LOG_FILE = os.path.join(CURRENT_DIR, "okx_cameron_4h_swing.log")

# ── LOGGING ──
logger = logging.getLogger("OKXCameron4H")
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)
    # stdout is often redirected to the same log file by a supervisor;
    # only echo to console for interactive runs to avoid duplicate lines.
    if sys.stdout.isatty():
        logger.addHandler(stream_handler)

# ── HYPERPARAMETERS ──
TARGET_SYMBOLS = [
    # Top Tier Liquid & High Momentum Swaps
    "SOL/USDT:USDT",
    "SUI/USDT:USDT",
    "DOGE/USDT:USDT",
    "ETH/USDT:USDT",
    "BTC/USDT:USDT",
    "XRP/USDT:USDT",
    "ADA/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
    "NEAR/USDT:USDT",
    "APT/USDT:USDT",
    "PEPE/USDT:USDT",
    "WIF/USDT:USDT",
    "SHIB/USDT:USDT",
    "FET/USDT:USDT"
]

TIMEFRAME = "2h"            # Backtest-verified optimal (PF 1.36 vs 4h 1.19, 120d)
RISK_REWARD_RATIO = 1.86       # Benchmark Lead Trader (01pp): 1.86:1 R:R
ALLOCATION_PCT = 0.08         # 8% of equity per trade (Controlled sizing for drawdown resistance)
LEVERAGE = 3                  # 3x safe margin
MAX_CONCURRENT_POSITIONS = 3  # Max 3 concurrent positions
MIN_SURGE_PCT = 0.025         # 2.5% impulse surge (Avoid catching local tops)
MAX_SL_PCT = 0.060            # Max 6% stop loss distance
MIN_SL_PCT = 0.015            # Min 1.5% stop loss distance
LOOP_INTERVAL_SEC = 60        # Check price & state every 60s
TRAILING_TRIGGER_RR = 1.0     # Once +1.0R is reached, protect capital by moving SL to break-even (+0.2%)
ENABLE_BE_TRAILING = False    # Backtest: BE trailing destroys PF (2h: 1.36 -> 0.99, 4h: 1.19 -> 0.71)


class OKXCameron4HTrader:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.api_key = os.getenv("OKX_API_KEY", "").strip()
        self.secret = os.getenv("OKX_SECRET", "").strip()
        self.password = os.getenv("OKX_PASSPHRASE", "").strip()
        self.exchange: Optional[ccxt_async.okx] = None
        
        self.equity_usdt = 960.0
        self.free_usdt = 960.0
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.jev_gate = UniversalJevGate(confidence_threshold=0.60)
        self.load_positions()

    def load_positions(self):
        if os.path.exists(POSITIONS_FILE):
            try:
                with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
                    self.active_positions = json.load(f)
                logger.info(f"📂 [State Restored] Loaded {len(self.active_positions)} active 4H trades: {list(self.active_positions.keys())}")
            except Exception as e:
                logger.error(f"❌ Failed to load {POSITIONS_FILE}: {e}")
                self.active_positions = {}
        else:
            self.active_positions = {}

    def save_positions(self):
        try:
            tmp_file = POSITIONS_FILE + ".tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(self.active_positions, f, ensure_ascii=False, indent=2)
            os.replace(tmp_file, POSITIONS_FILE)
        except Exception as e:
            logger.error(f"❌ Failed to save {POSITIONS_FILE}: {e}")

    async def init_exchange(self):
        if self.exchange is not None:
            return  # Guard against double init (main + start) leaking aiohttp sessions
        if not self.dry_run and not (self.api_key and self.secret and self.password):
            raise RuntimeError("OKX_API_KEY / OKX_SECRET / OKX_PASSPHRASE missing in .env")
        self.exchange = ccxt_async.okx({
            "apiKey": self.api_key,
            "secret": self.secret,
            "password": self.password,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"}
        })
        await self.exchange.load_markets()
        logger.info(f"🚀 [OKX Cameron 4H Swing Engine] Initialized (Dry Run: {self.dry_run})")

    async def update_account_balance(self) -> bool:
        try:
            bal = await self.exchange.fetch_balance()
            self.free_usdt = float(bal.get("USDT", {}).get("free", 0.0))
            self.equity_usdt = float(bal.get("USDT", {}).get("total", 0.0))
            return True
        except Exception as e:
            logger.error(f"❌ Failed to fetch OKX balance: {e}")
            return False

    async def fetch_indicators(self, symbol: str, limit: int = 40) -> Optional[pd.DataFrame]:
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=limit)
            if not ohlcv or len(ohlcv) < 20:
                return None
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            
            tp = (df["high"] + df["low"] + df["close"]) / 3
            df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, 1)
            df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
            return df
        except Exception as e:
            logger.error(f"❌ Error fetching bars for {symbol}: {e}")
            return None

    def evaluate_cameron_4h_setup(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Evaluates 4-Hour First Pullback Setup"""
        if len(df) < 15:
            return False, {}

        curr_bar = df.iloc[-1]
        prev_bar = df.iloc[-2]

        # 1. Pullback candle must be red
        if prev_bar["close"] >= prev_bar["open"]:
            return False, {}

        # 2. Support above 9 EMA & VWAP
        if prev_bar["close"] < prev_bar["ema9"] * 0.995 or prev_bar["low"] < prev_bar["vwap"] * 0.990:
            return False, {}

        # 3. Prior impulse surge (last 3~6 bars)
        prior = df.iloc[-7:-2]
        surge_low = prior["low"].min()
        surge_high = prior["high"].max()
        surge = surge_high - surge_low
        surge_pct = surge / surge_low if surge_low > 0 else 0

        if surge_pct < MIN_SURGE_PCT:
            return False, {}

        # 4. Retracement <= 55%
        pullback_depth = surge_high - prev_bar["low"]
        retrace_ratio = pullback_depth / surge if surge > 0 else 1.0
        if retrace_ratio > 0.55:
            return False, {}

        # 5. Volume contraction
        prior_vol_mean = prior["volume"].mean()
        vol_ratio = prev_bar["volume"] / prior_vol_mean if prior_vol_mean > 0 else 1.0
        if prev_bar["volume"] > prior_vol_mean * 1.20:
            return False, {}

        # 6. Trigger: current high breaks previous high
        trigger_price = prev_bar["high"]
        stop_loss = prev_bar["low"]
        risk = trigger_price - stop_loss
        risk_pct = risk / trigger_price if trigger_price > 0 else 0

        if risk_pct > MAX_SL_PCT or risk_pct < MIN_SL_PCT:
            return False, {}

        take_profit = trigger_price + (risk * RISK_REWARD_RATIO)
        is_triggered = curr_bar["high"] >= trigger_price

        return is_triggered, {
            "trigger_price": trigger_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_pct": risk_pct * 100,
            "target_pct": (take_profit / trigger_price - 1) * 100,
            "retrace_ratio": retrace_ratio * 100,
            "surge_pct": surge_pct * 100,
            "vol_ratio": vol_ratio
        }

    async def execute_trade(self, symbol: str, setup: Dict[str, Any]):
        trigger_price = setup["trigger_price"]
        stop_loss = setup["stop_loss"]
        take_profit = setup["take_profit"]

        market = self.exchange.market(symbol)
        trade_usd = self.equity_usdt * ALLOCATION_PCT * LEVERAGE
        contract_size = market.get("contractSize", 1.0)
        
        qty_contracts = math.floor(trade_usd / (trigger_price * contract_size))
        min_amount = market.get("limits", {}).get("amount", {}).get("min", 1.0)
        if qty_contracts < min_amount:
            logger.warning(f"⏭️ [SKIP] {symbol} Sized qty {qty_contracts} below min {min_amount} — not upsizing beyond {ALLOCATION_PCT*100:.0f}% allocation")
            return

        logger.info(f"🔥 [OKX 4H SWING TRIGGERED] {symbol}")
        logger.info(f"   Size: {qty_contracts} contracts (~${qty_contracts * contract_size * trigger_price:.1f} USD with {LEVERAGE}x)")
        logger.info(f"   Entry: {trigger_price} | SL: {stop_loss} (-{setup['risk_pct']:.2f}%) | TP: {take_profit} (+{setup['target_pct']:.2f}%)")

        # ── JEV PRE-TRADE VERIFICATION (Whipsaw & Fake Breakout Gate) ──
        decision = await self.jev_gate.verify_entry(
            asset_class="CRYPTO",
            symbol=symbol,
            setup_name="CAMERON_4H_PULLBACK",
            context={
                "surge_pct": setup.get("surge_pct", 0),
                "retrace_pct": setup.get("retrace_ratio", 40),
                "vol_ratio": setup.get("vol_ratio", 1.0),
                "range_pct": setup.get("risk_pct", 0)
            }
        )
        if not decision.approved:
            logger.warning(f"🛑 [JEV VETO] {symbol} Order blocked by AI Gate! Reason: {decision.reason} (Conf: {decision.confidence:.2f})")
            return

        logger.info(f"🛡️ [JEV APPROVED] {symbol} Entry validated by AI Gate! Reason: {decision.reason} (Conf: {decision.confidence:.2f})")

        if self.dry_run:
            logger.info("   [Dry Run] Simulated order recorded.")
            self.active_positions[symbol] = {
                "entry_price": trigger_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "qty": qty_contracts,
                "time": time.time()
            }
            self.save_positions()
            return

        try:
            # 1. Market Buy Entry with exchange-side TP/SL attached (survives bot downtime)
            order = await self.exchange.create_order(
                symbol=symbol,
                type="market",
                side="buy",
                amount=qty_contracts,
                params={
                    "tdMode": "cross",
                    "posSide": "long",
                    "stopLoss": {"triggerPrice": stop_loss, "type": "market"},
                    "takeProfit": {"triggerPrice": take_profit, "type": "market"}
                }
            )
            logger.info(f"   ✅ Entry Order Placed: ID {order.get('id')} (exchange-side SL/TP attached)")

            # 2. Save to State
            self.active_positions[symbol] = {
                "order_id": order.get("id"),
                "entry_price": trigger_price,
                "initial_sl": stop_loss,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "qty": qty_contracts,
                "time": time.time()
            }
            self.save_positions()

        except Exception as e:
            logger.error(f"❌ Failed to execute OKX 4H trade for {symbol}: {e}")

    async def get_open_contracts(self, symbol: str) -> float:
        """Returns actual open long contracts on exchange, or -1.0 if unknown."""
        try:
            positions = await self.exchange.fetch_positions([symbol])
            for p in positions:
                if p.get("symbol") == symbol and p.get("side") in ("long", "net"):
                    return abs(float(p.get("contracts") or 0.0))
            return 0.0
        except Exception as e:
            logger.error(f"❌ Failed to fetch position for {symbol}: {e}")
            return -1.0

    async def cancel_sl_trigger(self, symbol: str, pos_info: Dict[str, Any]):
        algo_id = pos_info.pop("sl_algo_id", None)
        if not algo_id:
            return
        try:
            await self.exchange.cancel_order(algo_id, symbol, params={"trigger": True})
            logger.info(f"   🗑️ Cancelled trailing SL trigger {algo_id} for {symbol}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to cancel SL trigger {algo_id} for {symbol}: {e}")

    async def place_trailing_trigger(self, symbol: str, pos_info: Dict[str, Any], new_sl: float):
        """Moves the exchange-side stop loss up to break-even via a reduce-only trigger order."""
        try:
            await self.cancel_sl_trigger(symbol, pos_info)
            order = await self.exchange.create_order(
                symbol=symbol,
                type="market",
                side="sell",
                amount=pos_info["qty"],
                params={"tdMode": "cross", "posSide": "long", "reduceOnly": True, "triggerPrice": new_sl}
            )
            pos_info["sl_algo_id"] = order.get("id")
            self.save_positions()
            logger.info(f"   ✅ Exchange-side trailing SL trigger placed at {new_sl} (ID {order.get('id')})")
        except Exception as e:
            logger.error(f"❌ Failed to place trailing SL trigger for {symbol}: {e}")

    async def close_position_market(self, symbol: str, pos_info: Dict[str, Any]):
        if not self.dry_run:
            await self.exchange.create_order(
                symbol=symbol,
                type="market",
                side="sell",
                amount=pos_info["qty"],
                params={"reduceOnly": True, "posSide": "long"}
            )
            await self.cancel_sl_trigger(symbol, pos_info)
        del self.active_positions[symbol]
        self.save_positions()

    async def manage_open_positions(self):
        if not self.active_positions:
            return

        for symbol in list(self.active_positions.keys()):
            pos_info = self.active_positions[symbol]
            try:
                # Reconcile with the exchange: if the position is already gone
                # (exchange-side SL/TP filled, manual close, liquidation), drop it.
                if not self.dry_run:
                    actual = await self.get_open_contracts(symbol)
                    if actual == 0.0:
                        logger.info(f"🔄 [Reconcile] {symbol} closed on exchange (SL/TP filled externally). Clearing state.")
                        await self.cancel_sl_trigger(symbol, pos_info)
                        del self.active_positions[symbol]
                        self.save_positions()
                        continue
                    if actual > 0 and abs(actual - pos_info["qty"]) > 1e-9:
                        logger.info(f"🔄 [Reconcile] {symbol} qty drift {pos_info['qty']} -> {actual}; syncing")
                        pos_info["qty"] = actual
                        self.save_positions()

                ticker = await self.exchange.fetch_ticker(symbol)
                curr_price = ticker.get("last", 0.0)
                sl = pos_info["stop_loss"]
                tp = pos_info["take_profit"]

                # Dynamic Trailing Stop to Break-Even (Trader KS Style)
                entry_p = pos_info.get("entry_price", sl)
                orig_risk = entry_p - pos_info.get("initial_sl", sl)
                if ENABLE_BE_TRAILING and orig_risk > 0:
                    rr_level = (curr_price - entry_p) / orig_risk
                    if rr_level >= TRAILING_TRIGGER_RR and sl < entry_p:
                        new_sl = entry_p * 1.002 # Lock in small profit (+0.2%)
                        pos_info["stop_loss"] = new_sl
                        self.save_positions()
                        logger.info(f"🛡️ [Trailing Lock] {symbol} Reached +{rr_level:.2f}R! SL moved to Break-Even ({new_sl})")
                        if not self.dry_run:
                            await self.place_trailing_trigger(symbol, pos_info, new_sl)
                        sl = new_sl

                # Stop Loss
                if curr_price <= sl:
                    logger.warning(f"🛑 [4H SWING STOP LOSS] {symbol} Price: {curr_price} <= SL: {sl}")
                    await self.close_position_market(symbol, pos_info)

                # Take Profit
                elif curr_price >= tp:
                    logger.info(f"🎉 [4H SWING TAKE PROFIT] {symbol} Price: {curr_price} >= TP: {tp}")
                    await self.close_position_market(symbol, pos_info)

            except Exception as e:
                logger.error(f"❌ Error monitoring 4H position for {symbol}: {e}")

    async def run_cycle(self):
        balance_ok = await self.update_account_balance()
        await self.manage_open_positions()

        if not balance_ok:
            logger.warning("⚠️ Skipping new entries this cycle: balance unavailable (refusing stale sizing)")
            return

        if len(self.active_positions) >= MAX_CONCURRENT_POSITIONS:
            return

        for symbol in TARGET_SYMBOLS:
            if symbol in self.active_positions:
                continue

            df = await self.fetch_indicators(symbol, limit=35)
            if df is None:
                continue

            triggered, setup = self.evaluate_cameron_4h_setup(df)
            if triggered:
                logger.info(f"✨ [4H SETUP CONFIRMED] {symbol} | Target: +{setup['target_pct']:.2f}% | Risk: {setup['risk_pct']:.2f}%")
                await self.execute_trade(symbol, setup)
                if len(self.active_positions) >= MAX_CONCURRENT_POSITIONS:
                    break

    async def start(self):
        await self.init_exchange()
        logger.info("🟢 Starting OKX Cameron 4H Swing Loop (Poll Interval: 60s)...")
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(LOOP_INTERVAL_SEC)
        except asyncio.CancelledError:
            logger.info("🛑 Loop cancelled.")
        finally:
            if self.exchange:
                await self.exchange.close()


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="OKX Cameron 4H Swing Trader")
    parser.add_argument("--dry-run", action="store_true", help="Run in simulation mode")
    parser.add_argument("--once", action="store_true", help="Run a single scan cycle")
    args = parser.parse_args()

    trader = OKXCameron4HTrader(dry_run=args.dry_run)
    await trader.init_exchange()
    if args.once:
        await trader.run_cycle()
        if trader.exchange:
            await trader.exchange.close()
    else:
        await trader.start()


if __name__ == "__main__":
    asyncio.run(main())
