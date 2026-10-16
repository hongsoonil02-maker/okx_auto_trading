# -*- coding: utf-8 -*-
"""
okx_ross_cameron_swing.py — Autonomous OKX 1H Multi-Indicator Swing Engine v1.0
Based on Ross Cameron's Modified RSI Divergence + EMA200 + MACD + Engulfing Candle Strategy

Core Specifications:
- Universe: BTC/USDT:USDT, ETH/USDT:USDT, SOL/USDT:USDT (Top 3 Liquid Majors)
- Timeframe: 1-Hour ('1h')
- Trend Filter: EMA 200
- Momentum: RSI(14) + MACD(12, 26, 9)
- Trigger: Bullish / Bearish Engulfing Candle Confirmation (On Bar Close)
- Divergence: Regular Bullish / Bearish Divergence against 15-bar rolling pivot window
- Risk / Reward: Strict 1 : 2 R:R with Swing Low / High Stop Loss
- Protection: Giant Candle Filter (>3.5% risk rejected), Max 30% Equity per Trade, 3x Leverage
- Persistence: JSON state backup for seamless crash/reboot recovery
- Alerts: Telegram webhook notifications
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
from logging.handlers import RotatingFileHandler
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt_async
from dotenv import load_dotenv

# ── DIRECTORY & CONFIG SETUP ──
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

STATE_DIR = os.path.join(CURRENT_DIR, "state")
os.makedirs(STATE_DIR, exist_ok=True)
POSITIONS_FILE = os.path.join(STATE_DIR, "ross_cameron_positions.json")
LOG_FILE = os.path.join(CURRENT_DIR, "okx_ross_cameron_swing.log")

# ── TELEGRAM ALERTS ──
try:
    from utils_telegram import send_telegram_alert
except ImportError:
    def send_telegram_alert(msg: str):
        pass

# ── LOGGING SETUP ──
logger = logging.getLogger("RossCameronSwing")
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(stream_handler)

# ── STRATEGY HYPERPARAMETERS ──
SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
TIMEFRAME = "1h"
EMA_PERIOD = 200
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
DIVERGENCE_LOOKBACK = 15      # Bars to scan for pivot lows/highs
RISK_REWARD_RATIO = 2.0      # 1:2 R:R
MAX_SL_PCT = 0.035           # Max 3.5% risk allowed per candle (giant candle protection)
MIN_SL_PCT = 0.003           # Min 0.3% stop loss buffer
LEVERAGE = 3                 # 3x safe margin
MAX_ALLOC_PER_TRADE = 0.28   # 28% of equity per trade (total 3 positions = 84% max allocation)
LOOP_INTERVAL_SEC = 10       # Check price & orders every 10s


class RossCameronSwingBot:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.api_key = os.getenv("OKX_API_KEY", "").strip()
        self.secret = os.getenv("OKX_SECRET", "").strip()
        self.password = os.getenv("OKX_PASSPHRASE", "").strip()
        self.exchange: Optional[ccxt_async.okx] = None
        
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.last_candle_time: Dict[str, int] = {s: 0 for s in SYMBOLS}
        self.account_equity = 0.0
        self.account_free = 0.0
        self.markets: Dict[str, Any] = {}
        
        self.load_positions()

    def load_positions(self):
        """Loads open positions from state JSON"""
        if os.path.exists(POSITIONS_FILE):
            try:
                with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
                    self.positions = json.load(f)
                logger.info(f"📂 [State Restored] Loaded {len(self.positions)} active swing trades: {list(self.positions.keys())}")
            except Exception as e:
                logger.error(f"❌ Failed to load positions from {POSITIONS_FILE}: {e}")
                self.positions = {}
        else:
            self.positions = {}

    def save_positions(self):
        """Saves open positions to state JSON"""
        try:
            with open(POSITIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.positions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to save positions to {POSITIONS_FILE}: {e}")

    async def init_exchange(self):
        """Initializes CCXT OKX async exchange instance"""
        self.exchange = ccxt_async.okx({
            "apiKey": self.api_key,
            "secret": self.secret,
            "password": self.password,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"}
        })
        self.markets = await self.exchange.load_markets()
        logger.info("=" * 70)
        logger.info("🚀 [Ross Cameron Swing Engine] Initialized OKX 1H Multi-Indicator Bot")
        logger.info(f"   Universe: {', '.join(SYMBOLS)}")
        logger.info(f"   Timeframe: {TIMEFRAME} | EMA Filter: {EMA_PERIOD} | RSI: {RSI_PERIOD} | MACD: ({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})")
        logger.info(f"   Target Risk:Reward: 1 : {RISK_REWARD_RATIO} | Leverage: {LEVERAGE}x | Max Trade Risk: {MAX_SL_PCT*100:.1f}%")
        logger.info(f"   Mode: {'DRY RUN (Simulation)' if self.dry_run else 'LIVE OKX SWAP'}")
        logger.info("=" * 70)

    async def update_balance(self) -> float:
        """Fetches latest USDT equity and free balance"""
        try:
            bal = await self.exchange.fetch_balance()
            self.account_total = float(bal.get("USDT", {}).get("total", 0.0))
            self.account_free = float(bal.get("USDT", {}).get("free", 0.0))
            self.account_equity = self.account_total
            return self.account_equity
        except Exception as e:
            logger.error(f"❌ Balance fetch error: {e}")
            return self.account_equity

    async def setup_leverage(self, symbol: str):
        """Sets leverage and cross margin mode for symbol"""
        if self.dry_run:
            return
        try:
            for s in ["long", "short"]:
                await self.exchange.set_leverage(LEVERAGE, symbol, {"mgnMode": "cross", "posSide": s})
        except Exception as e:
            logger.debug(f"Leverage set for {symbol}: {e}")

    async def fetch_indicators(self, symbol: str, limit: int = 250) -> Optional[pd.DataFrame]:
        """Fetches 1h OHLCV and calculates EMA200, RSI14, MACD, and Engulfing patterns"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=limit)
            if not ohlcv or len(ohlcv) < EMA_PERIOD + 20:
                return None
            
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            
            # 1. EMA 200
            df["ema200"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
            
            # 2. RSI 14
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0.0).rolling(window=RSI_PERIOD).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(window=RSI_PERIOD).mean()
            rs = gain / (loss + 1e-9)
            df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
            
            # 3. MACD (12, 26, 9)
            ema_fast = df["close"].ewm(span=MACD_FAST, adjust=False).mean()
            ema_slow = df["close"].ewm(span=MACD_SLOW, adjust=False).mean()
            df["macd"] = ema_fast - ema_slow
            df["signal"] = df["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
            
            # 4. Engulfing Candle Pattern
            df["prev_open"] = df["open"].shift(1)
            df["prev_close"] = df["close"].shift(1)
            df["prev_high"] = df["high"].shift(1)
            df["prev_low"] = df["low"].shift(1)
            
            # Bullish Engulfing: Prev red, Curr green, Curr Open <= Prev Close, Curr Close >= Prev Open
            df["bull_engulf"] = (
                (df["prev_close"] < df["prev_open"]) &
                (df["close"] > df["open"]) &
                (df["open"] <= df["prev_close"] * 1.0008) &
                (df["close"] >= df["prev_open"] * 0.9992)
            )
            
            # Bearish Engulfing: Prev green, Curr red, Curr Open >= Prev Close, Curr Close <= Prev Open
            df["bear_engulf"] = (
                (df["prev_close"] > df["prev_open"]) &
                (df["close"] < df["open"]) &
                (df["open"] >= df["prev_close"] * 0.9992) &
                (df["close"] <= df["prev_open"] * 1.0008)
            )
            
            return df
        except Exception as e:
            logger.error(f"❌ Failed to fetch indicators for {symbol}: {e}")
            return None

    def check_divergence(self, df: pd.DataFrame, idx: int) -> Tuple[bool, bool, Dict[str, float]]:
        """
        Detects Regular Bullish / Bearish Divergence using a rolling window
        - Bullish: Price <= rolling min(Low) while RSI > rolling min(RSI) + 2.0
        - Bearish: Price >= rolling max(High) while RSI < rolling max(RSI) - 2.0
        """
        if idx < DIVERGENCE_LOOKBACK + 5:
            return False, False, {}

        window = df.iloc[idx - DIVERGENCE_LOOKBACK : idx]
        curr_low = float(df.loc[idx, "low"])
        curr_high = float(df.loc[idx, "high"])
        curr_rsi = float(df.loc[idx, "rsi"])

        min_price = float(window["low"].min())
        min_rsi = float(window["rsi"].min())

        max_price = float(window["high"].max())
        max_rsi = float(window["rsi"].max())

        # Bullish Divergence: Price near/below prior low, but RSI visibly higher
        bull_div = (curr_low <= min_price * 1.0025) and (curr_rsi >= min_rsi + 2.0)

        # Bearish Divergence: Price near/above prior high, but RSI visibly lower
        bear_div = (curr_high >= max_price * 0.9975) and (curr_rsi <= max_rsi - 2.0)

        details = {
            "curr_rsi": curr_rsi,
            "min_rsi": min_rsi,
            "max_rsi": max_rsi,
            "curr_low": curr_low,
            "min_price": min_price,
            "curr_high": curr_high,
            "max_price": max_price
        }
        return bull_div, bear_div, details

    async def evaluate_symbol_on_bar_close(self, symbol: str):
        """Evaluates entry signals strictly on completed 1h bar"""
        if symbol in self.positions:
            return  # Already holding this symbol

        df = await self.fetch_indicators(symbol, limit=250)
        if df is None or len(df) < 50:
            return

        # We evaluate the LAST COMPLETED bar (df.iloc[-2]), because df.iloc[-1] is the ongoing/unclosed bar!
        last_closed_idx = len(df) - 2
        bar = df.iloc[last_closed_idx]
        bar_ts = int(bar["timestamp"].timestamp())

        # Avoid duplicate evaluations on the same closed bar
        if bar_ts <= self.last_candle_time.get(symbol, 0):
            return
        self.last_candle_time[symbol] = bar_ts

        close_px = float(bar["close"])
        ema200 = float(bar["ema200"])
        macd_val = float(bar["macd"])
        sig_val = float(bar["signal"])
        rsi_val = float(bar["rsi"])
        bull_engulf = bool(bar["bull_engulf"])
        bear_engulf = bool(bar["bear_engulf"])

        bull_div, bear_div, div_info = self.check_divergence(df, last_closed_idx)

        logger.info(
            f"🔍 [{symbol} 1H Close Check ({bar['timestamp']})] "
            f"Close: ${close_px:,.2f} | EMA200: ${ema200:,.2f} | "
            f"RSI: {rsi_val:.1f} | MACD: {macd_val:.2f}/{sig_val:.2f} | "
            f"Engulf: {'BULL' if bull_engulf else ('BEAR' if bear_engulf else 'None')} | "
            f"Div: {'BULL' if bull_div else ('BEAR' if bear_div else 'None')}"
        )

        # ── 1. LONG ENTRY CHECK ──
        if (close_px > ema200) and bull_div and (macd_val > sig_val) and bull_engulf:
            # Swing Low is min of current and previous bar low
            swing_low = min(float(bar["low"]), float(bar["prev_low"])) * 0.998
            risk_dist = close_px - swing_low
            risk_pct = risk_dist / close_px

            if risk_pct > MAX_SL_PCT:
                logger.warning(f"⚠️ [Long Skipped] {symbol} Candle risk ({risk_pct*100:.2f}%) exceeds max limit ({MAX_SL_PCT*100:.1f}%)")
                return
            if risk_pct < MIN_SL_PCT:
                swing_low = close_px * (1.0 - MIN_SL_PCT)
                risk_dist = close_px - swing_low

            target_tp = close_px + (risk_dist * RISK_REWARD_RATIO)
            logger.info(f"🎯 [LONG SIGNAL CONFIRMED] {symbol} | Entry: ${close_px:,.2f} | SL: ${swing_low:,.2f} (-{risk_pct*100:.2f}%) | TP: ${target_tp:,.2f} (+{risk_pct*RISK_REWARD_RATIO*100:.2f}%)")
            await self.execute_entry(symbol, "long", close_px, swing_low, target_tp)

        # ── 2. SHORT ENTRY CHECK ──
        elif (close_px < ema200) and bear_div and (macd_val < sig_val) and bear_engulf:
            # Swing High is max of current and previous bar high
            swing_high = max(float(bar["high"]), float(bar["prev_high"])) * 1.002
            risk_dist = swing_high - close_px
            risk_pct = risk_dist / close_px

            if risk_pct > MAX_SL_PCT:
                logger.warning(f"⚠️ [Short Skipped] {symbol} Candle risk ({risk_pct*100:.2f}%) exceeds max limit ({MAX_SL_PCT*100:.1f}%)")
                return
            if risk_pct < MIN_SL_PCT:
                swing_high = close_px * (1.0 + MIN_SL_PCT)
                risk_dist = swing_high - close_px

            target_tp = close_px - (risk_dist * RISK_REWARD_RATIO)
            logger.info(f"🎯 [SHORT SIGNAL CONFIRMED] {symbol} | Entry: ${close_px:,.2f} | SL: ${swing_high:,.2f} (-{risk_pct*100:.2f}%) | TP: ${target_tp:,.2f} (+{risk_pct*RISK_REWARD_RATIO*100:.2f}%)")
            await self.execute_entry(symbol, "short", close_px, swing_high, target_tp)

    async def execute_entry(self, symbol: str, side: str, price: float, sl_price: float, tp_price: float):
        """Calculates order size, sets leverage, and submits market order"""
        await self.update_balance()
        alloc_usdt = self.account_equity * MAX_ALLOC_PER_TRADE
        if alloc_usdt < 20.0:
            logger.error(f"❌ Insufficient equity for {symbol}: ${alloc_usdt:.2f}")
            return

        market = self.markets.get(symbol, {})
        contract_size = float(market.get("contractSize", 1.0))
        notional_target = alloc_usdt * LEVERAGE
        
        # Calculate contracts
        coins = notional_target / price
        contracts = coins / contract_size
        
        # Respect limits
        contracts_str = self.exchange.amount_to_precision(symbol, contracts)
        contracts = float(contracts_str)
        min_amount = float(market.get("limits", {}).get("amount", {}).get("min", 0.01))
        if contracts < min_amount:
            contracts = min_amount

        logger.info(f"📤 [Submitting Entry] {symbol} {side.upper()} {contracts} contracts (~${alloc_usdt:,.2f} margin x {LEVERAGE}x)")
        
        if self.dry_run:
            self.positions[symbol] = {
                "symbol": symbol,
                "side": side,
                "entry_price": price,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "contracts": contracts,
                "entry_time": datetime.now(timezone.utc).isoformat()
            }
            self.save_positions()
            send_telegram_alert(f"🧪 [SIMULATION ENTRY] {symbol} {side.upper()} {contracts}ct @ ${price:,.2f}\nSL: ${sl_price:,.2f} | TP: ${tp_price:,.2f}")
            return

        try:
            await self.setup_leverage(symbol)
            order_side = "buy" if side == "long" else "sell"
            params = {
                "tdMode": "cross",
                "posSide": side
            }
            res = await self.exchange.create_market_order(symbol, order_side, contracts, params=params)
            logger.info(f"✅ [Order Filled] {symbol} {side.upper()} ID: {res.get('id')}")

            # Place exchange-side hard stop loss algo order for absolute safety
            try:
                inst_id = symbol.replace("/", "-").split(":")[0] + "-SWAP" if ":" in symbol else symbol
                sl_side = "sell" if side == "long" else "buy"
                sl_price_str = self.exchange.price_to_precision(symbol, sl_price)
                algo_params = {
                    "instId": inst_id,
                    "tdMode": "cross",
                    "side": sl_side,
                    "posSide": side,
                    "ordType": "conditional",
                    "sz": str(contracts),
                    "slTriggerPx": str(sl_price_str),
                    "slOrdPx": "-1",
                    "slTriggerPxType": "mark",
                    "reduceOnly": True
                }
                res_algo = await self.exchange.request("trade/order-algo", api="private", method="POST", params=algo_params)
                if res_algo.get("code") == "0":
                    logger.info(f"🛡️ [Exchange SL Placed] {symbol} {side} @ {sl_price_str}")
                else:
                    logger.warning(f"⚠️ [Exchange SL Warning]: {res_algo.get('msg')}")
            except Exception as ex_sl:
                logger.warning(f"⚠️ [Exchange SL Exception]: {ex_sl}")
            
            # Record in positions state
            self.positions[symbol] = {
                "symbol": symbol,
                "side": side,
                "entry_price": price,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "contracts": contracts,
                "order_id": res.get("id"),
                "entry_time": datetime.now(timezone.utc).isoformat()
            }
            self.save_positions()
            send_telegram_alert(f"🚀 [OKX SWING ENTRY] {symbol} {side.upper()}\n수량: {contracts} 계약\n진입가: ${price:,.2f}\n손절(SL): ${sl_price:,.2f}\n목표(TP): ${tp_price:,.2f} (1:2 R:R)")
        except Exception as e:
            logger.error(f"❌ Order execution failed for {symbol}: {e}")

    async def monitor_open_positions(self):
        """Monitors active positions in real-time, executing TP / SL exits"""
        if not self.positions:
            return

        for symbol, pos in list(self.positions.items()):
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                curr_px = float(ticker["last"])
                side = pos["side"]
                entry_px = float(pos["entry_price"])
                sl_px = float(pos["sl_price"])
                tp_px = float(pos["tp_price"])
                contracts = float(pos["contracts"])

                if side == "long":
                    pnl_pct = (curr_px - entry_px) / entry_px * 100.0
                    # Check Take Profit
                    if curr_px >= tp_px:
                        logger.info(f"🎉 [Take Profit Hit] {symbol} Long reached ${curr_px:,.2f} >= Target ${tp_px:,.2f} (+{pnl_pct:.2f}%)")
                        await self.close_position(symbol, "TAKE_PROFIT_1_TO_2", curr_px)
                    # Check Stop Loss
                    elif curr_px <= sl_px:
                        logger.warning(f"🛑 [Stop Loss Hit] {symbol} Long fell to ${curr_px:,.2f} <= SL ${sl_px:,.2f} ({pnl_pct:.2f}%)")
                        await self.close_position(symbol, "SWING_LOW_STOP_LOSS", curr_px)

                elif side == "short":
                    pnl_pct = (entry_px - curr_px) / entry_px * 100.0
                    # Check Take Profit
                    if curr_px <= tp_px:
                        logger.info(f"🎉 [Take Profit Hit] {symbol} Short fell to ${curr_px:,.2f} <= Target ${tp_px:,.2f} (+{pnl_pct:.2f}%)")
                        await self.close_position(symbol, "TAKE_PROFIT_1_TO_2", curr_px)
                    # Check Stop Loss
                    elif curr_px >= sl_px:
                        logger.warning(f"🛑 [Stop Loss Hit] {symbol} Short rose to ${curr_px:,.2f} >= SL ${sl_px:,.2f} ({pnl_pct:.2f}%)")
                        await self.close_position(symbol, "SWING_HIGH_STOP_LOSS", curr_px)

            except Exception as e:
                logger.error(f"❌ Error monitoring position for {symbol}: {e}")

    async def close_position(self, symbol: str, reason: str, exit_px: float):
        """Closes position via market order"""
        pos = self.positions.get(symbol)
        if not pos:
            return
        
        side = pos["side"]
        contracts = pos["contracts"]
        entry_px = pos["entry_price"]
        pnl_pct = ((exit_px - entry_px) / entry_px * 100.0) if side == "long" else ((entry_px - exit_px) / entry_px * 100.0)

        logger.info(f"🔄 [Closing Position] {symbol} {side.upper()} | Reason: {reason} | PnL: {pnl_pct:+.2f}%")

        if not self.dry_run:
            # 1. Cancel any pending exchange SL algos for this symbol
            try:
                inst_id = symbol.replace("/", "-").split(":")[0] + "-SWAP" if ":" in symbol else symbol
                pending_algos = await self.exchange.request("trade/orders-algo-pending", api="private", method="GET", params={"instId": inst_id, "ordType": "conditional"})
                algo_ids = [a["algoId"] for a in pending_algos.get("data", []) if "algoId" in a]
                if algo_ids:
                    cancel_list = [{"instId": inst_id, "algoId": a_id} for a_id in algo_ids]
                    await self.exchange.request("trade/cancel-algos", api="private", method="POST", params=cancel_list)
                    logger.info(f"🧹 [Algo SL Canceled] {symbol} cleared {len(algo_ids)} SL algo(s)")
            except Exception as ex_c:
                logger.debug(f"Algo cancel note: {ex_c}")

            try:
                close_side = "sell" if side == "long" else "buy"
                params = {
                    "reduceOnly": True,
                    "tdMode": "cross",
                    "posSide": side
                }
                res = await self.exchange.create_market_order(symbol, close_side, contracts, params=params)
                logger.info(f"✅ [Position Closed] {symbol} | Close ID: {res.get('id')}")
            except Exception as e:
                logger.error(f"❌ Error submitting close order for {symbol}: {e}")

        tag = "WIN 🟢" if pnl_pct > 0 else "LOSS 🔴"
        send_telegram_alert(f"{tag} [OKX SWING EXIT] {symbol} {side.upper()}\n사유: {reason}\n진입가: ${entry_px:,.2f} → 청산가: ${exit_px:,.2f}\n손익률: {pnl_pct:+.2f}%")
        self.positions.pop(symbol, None)
        self.save_positions()

    async def run(self):
        """Main autonomous execution loop"""
        await self.init_exchange()
        await self.update_balance()
        logger.info(f"💼 [Account Ready] Total Equity: ${self.account_equity:,.2f} USDT | Free: ${self.account_free:,.2f} USDT")

        last_balance_log = 0.0

        while True:
            try:
                now = time.time()
                
                # Hourly Balance Heartbeat
                if now - last_balance_log >= 3600.0:
                    await self.update_balance()
                    active_info = [f"{s}({p['side']})" for s, p in self.positions.items()]
                    logger.info(f"💓 [Heartbeat] Equity: ${self.account_equity:,.2f} | Active Positions: {len(self.positions)}/3 ({active_info})")
                    last_balance_log = now

                # 1. Scan for Bar-Close Entry Signals
                for symbol in SYMBOLS:
                    await self.evaluate_symbol_on_bar_close(symbol)

                # 2. Monitor Active Positions for Real-time TP/SL
                await self.monitor_open_positions()

                await asyncio.sleep(LOOP_INTERVAL_SEC)

            except Exception as e:
                logger.error(f"⚠️ Loop exception: {e}")
                await asyncio.sleep(LOOP_INTERVAL_SEC)

    async def close(self):
        if self.exchange:
            await self.exchange.close()


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    bot = RossCameronSwingBot(dry_run=dry_run)
    try:
        asyncio.run(bot.run())
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 Bot terminated gracefully.")
        asyncio.run(bot.close())
