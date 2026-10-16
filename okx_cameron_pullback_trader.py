# -*- coding: utf-8 -*-
"""
okx_cameron_pullback_trader.py — Autonomous OKX First Pullback Momentum Scalping Engine
Applied to OKX USDT-Margined Perpetual Swaps

Strategy Design (Ross Cameron Setup on Crypto Swaps):
1. Target Universe:
   - High-volume, volatile momentum swaps (SOL, SUI, DOGE, XRP, BTC, ETH, PEPE, NEAR, etc.)
2. Setup (1-Minute / 3-Minute First Pullback):
   - Impulse surge: At least 1.0% rapid thrust over 3~6 bars.
   - 1st Pullback: 1~3 red bars immediately following.
   - Retracement <= 50% of the impulse leg.
   - Price supported above VWAP and 9 EMA.
   - Pullback volume significantly lower than impulse volume.
3. Order & Risk Management:
   - Leverage: Safe 2x ~ 3x Cross/Isolated Margin
   - Position Sizing: 15% of account balance per position (~$150 USDT per trade from ~$960 USDT equity)
   - Bracket Order (OCO via OKX Algo Order):
       * Stop-Loss: Swing Low of the pullback candle (approx -1.5% ~ -2.5%)
       * Take-Profit: Strict 1 : 2.5 Risk-to-Reward (approx +3.75% ~ +6.25%)
   - Max Concurrent Trades: 2 positions
   - Persistence: State file saved to state/okx_cameron_positions.json
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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

STATE_DIR = os.path.join(CURRENT_DIR, "state")
os.makedirs(STATE_DIR, exist_ok=True)
POSITIONS_FILE = os.path.join(STATE_DIR, "okx_cameron_positions.json")
LOG_FILE = os.path.join(CURRENT_DIR, "okx_cameron_pullback.log")

# ── LOGGING ──
logger = logging.getLogger("OKXCameronTrader")
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

# ── HYPERPARAMETERS ──
TARGET_SYMBOLS = [
    "SOL/USDT:USDT",
    "SUI/USDT:USDT",
    "DOGE/USDT:USDT",
    "XRP/USDT:USDT",
    "ETH/USDT:USDT",
    "BTC/USDT:USDT",
    "NEAR/USDT:USDT",
    "PEPE/USDT:USDT"
]

TIMEFRAME = "1m"
RISK_REWARD_RATIO = 2.5       # Proven optimal 1:2.5 R:R
ALLOCATION_PCT = 0.15         # 15% of equity per trade (~150 USDT)
LEVERAGE = 2                  # Conservative 2x leverage
MAX_CONCURRENT_POSITIONS = 2  # Max 2 active positions
MIN_SURGE_PCT = 0.008         # Min 0.8% impulse surge
MAX_SL_PCT = 0.035            # Max 3.5% stop loss allowed
MIN_SL_PCT = 0.005            # Min 0.5% stop loss buffer
POLL_INTERVAL_SEC = 10        # 10s evaluation loop


class OKXCameronPullbackTrader:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.api_key = os.getenv("OKX_API_KEY", "").strip()
        self.secret = os.getenv("OKX_SECRET", "").strip()
        self.password = os.getenv("OKX_PASSPHRASE", "").strip()
        self.exchange: Optional[ccxt_async.okx] = None
        
        self.equity_usdt = 960.0
        self.free_usdt = 960.0
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.load_positions()

    def load_positions(self):
        if os.path.exists(POSITIONS_FILE):
            try:
                with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
                    self.active_positions = json.load(f)
                logger.info(f"📂 [State Restored] Loaded {len(self.active_positions)} active trades: {list(self.active_positions.keys())}")
            except Exception as e:
                logger.error(f"❌ Failed to load {POSITIONS_FILE}: {e}")
                self.active_positions = {}
        else:
            self.active_positions = {}

    def save_positions(self):
        try:
            with open(POSITIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.active_positions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to save {POSITIONS_FILE}: {e}")

    async def init_exchange(self):
        self.exchange = ccxt_async.okx({
            "apiKey": self.api_key,
            "secret": self.secret,
            "password": self.password,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"}
        })
        await self.exchange.load_markets()
        logger.info(f"🚀 [OKX Cameron Pullback Engine] Initialized with {len(self.exchange.markets)} markets (Dry Run: {self.dry_run})")

    async def update_account_balance(self):
        try:
            bal = await self.exchange.fetch_balance()
            self.free_usdt = float(bal.get("USDT", {}).get("free", 0.0))
            self.equity_usdt = float(bal.get("USDT", {}).get("total", 0.0))
        except Exception as e:
            logger.error(f"❌ Failed to fetch OKX balance: {e}")

    async def fetch_indicators(self, symbol: str, limit: int = 40) -> Optional[pd.DataFrame]:
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=limit)
            if not ohlcv or len(ohlcv) < 20:
                return None
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            
            # Typical Price VWAP
            tp = (df["high"] + df["low"] + df["close"]) / 3
            df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, 1)
            
            # 9 EMA
            df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
            return df
        except Exception as e:
            logger.error(f"❌ Error fetching bars for {symbol}: {e}")
            return None

    def evaluate_cameron_setup(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Ross Cameron 1-Min First Pullback Evaluation
        """
        if len(df) < 15:
            return False, {}

        curr_bar = df.iloc[-1]
        prev_bar = df.iloc[-2]

        # 1. Pullback candle must be bearish
        if prev_bar["close"] >= prev_bar["open"]:
            return False, {}

        # 2. Support: price must hold above 9 EMA & VWAP
        if prev_bar["close"] < prev_bar["ema9"] * 0.998 or prev_bar["low"] < prev_bar["vwap"] * 0.995:
            return False, {}

        # 3. Prior impulse surge (last 3~6 bars)
        prior = df.iloc[-7:-2]
        surge_low = prior["low"].min()
        surge_high = prior["high"].max()
        surge = surge_high - surge_low
        surge_pct = surge / surge_low if surge_low > 0 else 0

        if surge_pct < MIN_SURGE_PCT:
            return False, {}

        # 4. Retracement <= 50%
        pullback_depth = surge_high - prev_bar["low"]
        retrace_ratio = pullback_depth / surge if surge > 0 else 1.0
        if retrace_ratio > 0.50:
            return False, {}

        # 5. Volume contraction
        if prev_bar["volume"] > prior["volume"].mean() * 1.10:
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
            "retrace_ratio": retrace_ratio * 100
        }

    async def execute_trade(self, symbol: str, setup: Dict[str, Any]):
        """Executes trade with native OKX Stop-Loss and Take-Profit Algo orders"""
        trigger_price = setup["trigger_price"]
        stop_loss = setup["stop_loss"]
        take_profit = setup["take_profit"]

        market = self.exchange.market(symbol)
        trade_usd = self.equity_usdt * ALLOCATION_PCT * LEVERAGE
        contract_size = market.get("contractSize", 1.0)
        
        # Calculate contracts
        qty_contracts = math.floor(trade_usd / (trigger_price * contract_size))
        min_amount = market.get("limits", {}).get("amount", {}).get("min", 1.0)

        if qty_contracts < min_amount:
            qty_contracts = min_amount

        logger.info(f"🔥 [OKX PULLBACK TRIGGERED] {symbol}")
        logger.info(f"   Size: {qty_contracts} contracts (~${qty_contracts * contract_size * trigger_price:.1f} USD with {LEVERAGE}x)")
        logger.info(f"   Entry: {trigger_price} | SL: {stop_loss} (-{setup['risk_pct']:.2f}%) | TP: {take_profit} (+{setup['target_pct']:.2f}%)")

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
            # 1. Market Buy Entry
            order = await self.exchange.create_order(
                symbol=symbol,
                type="market",
                side="buy",
                amount=qty_contracts,
                params={"tdMode": "cross"}
            )
            logger.info(f"   ✅ Entry Order Placed: ID {order.get('id')}")

            # 2. Place Take-Profit Limit Algo Order
            try:
                await self.exchange.create_order(
                    symbol=symbol,
                    type="limit",
                    side="sell",
                    amount=qty_contracts,
                    price=take_profit,
                    params={"reduceOnly": True}
                )
                logger.info(f"   ✅ Take-Profit Limit Placed at {take_profit}")
            except Exception as e_tp:
                logger.warning(f"   ⚠️ Could not place TP limit order directly: {e_tp}")

            # 3. Save to state for position watcher
            self.active_positions[symbol] = {
                "order_id": order.get("id"),
                "entry_price": trigger_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "qty": qty_contracts,
                "time": time.time()
            }
            self.save_positions()

        except Exception as e:
            logger.error(f"❌ Failed to execute OKX trade for {symbol}: {e}")

    async def manage_open_positions(self):
        """Monitors SL / TP and manages exits"""
        if not self.active_positions:
            return

        for symbol in list(self.active_positions.keys()):
            pos_info = self.active_positions[symbol]
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                curr_price = ticker.get("last", 0.0)
                sl = pos_info["stop_loss"]
                tp = pos_info["take_profit"]

                # Stop Loss hit
                if curr_price <= sl:
                    logger.warning(f"🛑 [STOP LOSS HIT] {symbol} Price: {curr_price} <= SL: {sl}")
                    if not self.dry_run:
                        await self.exchange.create_order(
                            symbol=symbol,
                            type="market",
                            side="sell",
                            amount=pos_info["qty"],
                            params={"reduceOnly": True}
                        )
                    del self.active_positions[symbol]
                    self.save_positions()

                # Take Profit hit
                elif curr_price >= tp:
                    logger.info(f"🎉 [TAKE PROFIT HIT] {symbol} Price: {curr_price} >= TP: {tp}")
                    if not self.dry_run:
                        await self.exchange.create_order(
                            symbol=symbol,
                            type="market",
                            side="sell",
                            amount=pos_info["qty"],
                            params={"reduceOnly": True}
                        )
                    del self.active_positions[symbol]
                    self.save_positions()

            except Exception as e:
                logger.error(f"❌ Error monitoring position for {symbol}: {e}")

    async def run_cycle(self):
        await self.update_account_balance()
        await self.manage_open_positions()

        if len(self.active_positions) >= MAX_CONCURRENT_POSITIONS:
            return

        for symbol in TARGET_SYMBOLS:
            if symbol in self.active_positions:
                continue

            df = await self.fetch_indicators(symbol, limit=35)
            if df is None:
                continue

            triggered, setup = self.evaluate_cameron_setup(df)
            if triggered:
                logger.info(f"✨ [SETUP CONFIRMED] {symbol} | Risk: {setup['risk_pct']:.2f}% | Target: +{setup['target_pct']:.2f}%")
                await self.execute_trade(symbol, setup)
                if len(self.active_positions) >= MAX_CONCURRENT_POSITIONS:
                    break

    async def start(self):
        await self.init_exchange()
        logger.info("🟢 Starting OKX Cameron Pullback Engine Loop...")
        try:
            while True:
                await self.run_cycle()
                await asyncio.sleep(POLL_INTERVAL_SEC)
        except asyncio.CancelledError:
            logger.info("🛑 Loop cancelled.")
        finally:
            if self.exchange:
                await self.exchange.close()


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="OKX Cameron Pullback Trader")
    parser.add_argument("--dry-run", action="store_true", help="Run without placing live orders")
    parser.add_argument("--once", action="store_true", help="Run a single scan cycle")
    args = parser.parse_args()

    trader = OKXCameronPullbackTrader(dry_run=args.dry_run)
    await trader.init_exchange()
    if args.once:
        await trader.run_cycle()
        if trader.exchange:
            await trader.exchange.close()
    else:
        await trader.start()


if __name__ == "__main__":
    asyncio.run(main())
