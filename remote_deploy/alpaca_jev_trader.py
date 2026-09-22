# -*- coding: utf-8 -*-
"""
alpaca_jev_trader.py — Autonomous Nasdaq Quant Execution Engine
Integrated with Jev AI Microstructure Gating & Alpaca Paper Trading

Features:
- Live BBO & Imbalance stream from local Bun Sidecar (port 8020)
- Jev Sub-Second Gating (/predict) for trade confirmation
- Dual-Directional Alpha:
  * Bullish Momentum -> Long TQQQ, SOXL, NVDA, TSLA, AAPL, QQQ
  * Bearish Momentum -> Long Inverse ETFs SQQQ, SOXS (or Short QQQ)
- Alpaca Paper Trading Account Portfolio Management ($100k Capital)
- Automatic Position Sizing (~15% per trade, max 4 concurrent positions)
- Bracket / Trailing Stop Management (SL -1.2%, TP +2.5% ~ +3.5%)
- Market Session Awareness (/v2/clock) with 24/7 graceful supervision
"""

import os
import sys
import time
import math
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import requests
import pandas as pd
from dotenv import load_dotenv

# Ensure local dir modules can be imported
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

# Import local Jev client
try:
    from jev_nasdaq_client import query_jev_nasdaq, get_quotes_cache
except ImportError:
    # Fallback import if running from parent dir
    sys.path.append(CURRENT_DIR)
    from jev_nasdaq_client import query_jev_nasdaq, get_quotes_cache

# ── LOGGING SETUP ──
LOG_FILE = os.path.join(CURRENT_DIR, "alpaca_jev_trader.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ]
)
logger = logging.getLogger("AlpacaJevTrader")

# ── CONFIGURATION ──
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "PKVGU7IXDJMAF6JBWWVMP2I6R3")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "CCnTinES4eWwkQqifr9879ASCcXSpm9kjuvbZnCKpHM3")
ALPACA_REST_URL = os.getenv("ALPACA_REST_URL", "https://paper-api.alpaca.markets/v2")
ALPACA_DATA_URL = "https://data.alpaca.markets/v2"

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type": "application/json"
}

# Universe & Pairing
SYMBOLS_UNIVERSE = ["QQQ", "TQQQ", "SQQQ", "SOXL", "SOXS", "NVDA", "TSLA", "AAPL"]
LEVERAGED_3X = {"TQQQ", "SQQQ", "SOXL", "SOXS"}
MAX_CONCURRENT_POSITIONS = 4
MAX_POSITION_PCT = 0.15     # 15% of equity per position (~$15,000 on $100k equity)
LOOP_INTERVAL_SEC = 10      # Poll market every 10 seconds during open hours

def get_symbol_risk(symbol: str) -> Tuple[float, float, float]:
    """Returns (stop_loss_pct, take_profit_pct, trailing_arm_pct) tailored to asset volatility"""
    if symbol in LEVERAGED_3X:
        return 0.024, 0.042, 0.020  # 3x ETF: 2.4% SL, 4.2% TP, 2.0% Trail Arm
    return 0.012, 0.028, 0.015       # 1x Equities: 1.2% SL, 2.8% TP, 1.5% Trail Arm

class AlpacaJevTrader:
    def __init__(self, simulation_mode: Optional[bool] = None):
        self.simulation_mode: bool = (
            simulation_mode if simulation_mode is not None
            else (os.getenv("JEV_SIMULATION_MODE", "false").lower() == "true" or "--simulation" in sys.argv)
        )
        self.account_info: Dict[str, Any] = {}
        self.positions: Dict[str, Any] = {}
        self.last_clock_check: float = 0.0
        self.is_market_open: bool = False
        self.next_open_str: str = ""
        self.next_close_str: str = ""
        self.trade_cooldowns: Dict[str, float] = {}  # Symbol -> cooldown timestamp
        
        logger.info("=" * 65)
        logger.info(f"🚀 [Alpaca Jev Trader] Initializing Autonomous Nasdaq Quant Engine (Mode: {'SIMULATION' if self.simulation_mode else 'LIVE PAPER'})")
        logger.info(f"   REST Endpoint: {ALPACA_REST_URL}")
        logger.info(f"   Tracked Universe: {', '.join(SYMBOLS_UNIVERSE)}")
        logger.info(f"   Max Positions: {MAX_CONCURRENT_POSITIONS} | Position Sizing: {MAX_POSITION_PCT*100:.1f}%")
        logger.info(f"   Risk Config: Hard SL -{STOP_LOSS_PCT*100:.1f}%, TP +{TAKE_PROFIT_PCT*100:.1f}%, Trail Arm +{TRAILING_ARM_PCT*100:.1f}%")
        logger.info("=" * 65)

    def sync_account(self) -> bool:
        """Fetches account details & cash / buying power"""
        try:
            resp = requests.get(f"{ALPACA_REST_URL}/account", headers=HEADERS, timeout=5)
            if resp.status_code == 200:
                self.account_info = resp.json()
                equity = float(self.account_info.get("equity", 0.0))
                cash = float(self.account_info.get("cash", 0.0))
                bp = float(self.account_info.get("buying_power", 0.0))
                status = self.account_info.get("status", "UNKNOWN")
                logger.info(f"💼 [Account Sync] Status: {status} | Equity: ${equity:,.2f} | Cash: ${cash:,.2f} | Buying Power: ${bp:,.2f}")
                return True
            else:
                logger.error(f"❌ Account sync failed: HTTP {resp.status_code} - {resp.text}")
                return False
        except Exception as e:
            logger.error(f"❌ Account sync exception: {e}")
            return False

    def sync_positions(self) -> Dict[str, Any]:
        """Fetches current open positions from Alpaca"""
        if self.simulation_mode:
            return self.positions
        try:
            resp = requests.get(f"{ALPACA_REST_URL}/positions", headers=HEADERS, timeout=5)
            if resp.status_code == 200:
                pos_list = resp.json()
                self.positions = {p["symbol"]: p for p in pos_list}
                return self.positions
            else:
                logger.error(f"❌ Positions sync failed: HTTP {resp.status_code}")
                return {}
        except Exception as e:
            logger.error(f"❌ Positions sync exception: {e}")
            return {}

    def check_market_clock(self) -> bool:
        """Checks if US stock market is open"""
        now = time.time()
        if now - self.last_clock_check < 30.0 and self.last_clock_check > 0:
            return self.is_market_open

        try:
            resp = requests.get(f"{ALPACA_REST_URL}/clock", headers=HEADERS, timeout=5)
            if resp.status_code == 200:
                clock = resp.json()
                self.is_market_open = clock.get("is_open", False)
                self.next_open_str = clock.get("next_open", "")
                self.next_close_str = clock.get("next_close", "")
                self.last_clock_check = now
                return self.is_market_open
        except Exception as e:
            logger.warning(f"⚠️ Clock check error: {e}")
        return self.is_market_open

    def fetch_bars(self, symbol: str, timeframe: str = "1Min", limit: int = 40) -> Optional[pd.DataFrame]:
        """Fetches recent bars from Alpaca Market Data API"""
        try:
            url = f"{ALPACA_DATA_URL}/stocks/bars"
            params = {
                "symbols": symbol,
                "timeframe": timeframe,
                "limit": limit,
                "feed": "iex",
                "sort": "desc"
            }
            resp = requests.get(url, headers=HEADERS, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                bars = data.get("bars", {}).get(symbol, [])
                if not bars:
                    return None
                df = pd.DataFrame(bars)
                df = df.rename(columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
                df["close"] = pd.to_numeric(df["close"])
                df["high"] = pd.to_numeric(df["high"])
                df["low"] = pd.to_numeric(df["low"])
                df = df.iloc[::-1].reset_index(drop=True) # Ascending order
                return df
        except Exception as e:
            logger.debug(f"Bars fetch failed for {symbol}: {e}")
        return None

    def calculate_technical_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Computes EMA9, EMA21, RSI(14) on candle dataframe"""
        if df is None or len(df) < 25:
            return {"signal": "NEUTRAL", "reason": "INSUFFICIENT_BARS"}

        close = df["close"]
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))

        curr_close = close.iloc[-1]
        curr_ema9 = ema9.iloc[-1]
        prev_ema9 = ema9.iloc[-2]
        curr_ema21 = ema21.iloc[-1]
        prev_ema21 = ema21.iloc[-2]
        curr_ema50 = ema50.iloc[-1]
        curr_rsi = rsi.iloc[-1]

        # Momentum Crossover Logic
        bullish_cross = (curr_ema9 > curr_ema21) and (curr_close > curr_ema9) and (curr_rsi > 48.0)
        bearish_cross = (curr_ema9 < curr_ema21) and (curr_close < curr_ema9) and (curr_rsi < 52.0)

        if bullish_cross:
            return {
                "signal": "BUY",
                "rsi": curr_rsi,
                "ema9": curr_ema9,
                "ema21": curr_ema21,
                "ema50": curr_ema50,
                "price": curr_close,
                "reason": f"BULLISH_EMA_CROSS (RSI: {curr_rsi:.1f})"
            }
        elif bearish_cross:
            return {
                "signal": "SELL",
                "rsi": curr_rsi,
                "ema9": curr_ema9,
                "ema21": curr_ema21,
                "ema50": curr_ema50,
                "price": curr_close,
                "reason": f"BEARISH_EMA_CROSS (RSI: {curr_rsi:.1f})"
            }

        return {"signal": "NEUTRAL", "rsi": curr_rsi, "price": curr_close, "ema50": curr_ema50, "reason": "NO_CLEAR_CROSS"}

    def submit_alpaca_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        stop_loss_pct: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Submits bracket order to Alpaca Paper Trading REST API"""
        if qty <= 0:
            logger.warning(f"⚠️ Invalid order qty for {symbol}: {qty}")
            return None

        # Fetch latest price to compute bracket SL/TP
        quotes = get_quotes_cache()
        q = quotes.get(symbol, {})
        base_price = limit_price or q.get("askPrice") or q.get("bidPrice") or 0.0

        order_data: Dict[str, Any] = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side.lower(),
            "type": order_type.lower(),
            "time_in_force": "day"
        }

        if order_type.lower() == "limit" and limit_price:
            order_data["limit_price"] = str(round(limit_price, 2))

        # Use adaptive risk per asset class
        sym_sl, sym_tp, sym_trail = get_symbol_risk(symbol)
        sl_pct = stop_loss_pct if stop_loss_pct is not None else sym_sl
        tp_pct = take_profit_pct if take_profit_pct is not None else sym_tp

        # Add bracket stop-loss and take-profit if price is known
        if base_price > 0:
            if side.lower() == "buy":
                tp_price = round(base_price * (1.0 + tp_pct), 2)
                sl_price = round(base_price * (1.0 - sl_pct), 2)
            else:
                tp_price = round(base_price * (1.0 - tp_pct), 2)
                sl_price = round(base_price * (1.0 + sl_pct), 2)

            order_data["order_class"] = "bracket"
            order_data["take_profit"] = {"limit_price": str(tp_price)}
            order_data["stop_loss"] = {"stop_price": str(sl_price)}

        logger.info(f"📤 [Order Submit] {side.upper()} {qty}x {symbol} | Type: {order_type} | Data: {order_data}")

        if self.simulation_mode:
            logger.info(f"🧪 [SIMULATION FILL] {side.upper()} {qty}x {symbol} @ ${base_price:.2f} | Bracket TP: ${tp_price} / SL: ${sl_price}")
            sim_pos = {
                "symbol": symbol,
                "qty": str(qty),
                "side": side.lower(),
                "avg_entry_price": str(base_price),
                "current_price": str(base_price),
                "unrealized_plpc": 0.0,
                "unrealized_pl": 0.0
            }
            self.positions[symbol] = sim_pos
            self.trade_cooldowns[symbol] = time.time() + 60
            return {"id": f"sim_{int(time.time())}", "status": "simulated", "symbol": symbol}

        try:
            resp = requests.post(f"{ALPACA_REST_URL}/orders", headers=HEADERS, json=order_data, timeout=8)
            if resp.status_code in [200, 201]:
                res_json = resp.json()
                logger.info(f"✅ [Order Confirmed] ID: {res_json.get('id')} | Status: {res_json.get('status')} | {symbol}")
                self.trade_cooldowns[symbol] = time.time() + 180 # 3 min cooldown
                return res_json
            else:
                logger.error(f"❌ Order submission rejected: HTTP {resp.status_code} - {resp.text}")
                return None
        except Exception as e:
            logger.error(f"❌ Order submission exception: {e}")
            return None

    def manage_open_positions(self):
        """Monitors unrealized PnL and executes dynamic trailing stop exits"""
        if not self.positions:
            return

        quotes = get_quotes_cache()
        for symbol, pos in list(self.positions.items()):
            qty = abs(float(pos.get("qty", 0)))
            side = pos.get("side", "long")
            avg_entry = float(pos.get("avg_entry_price", 0.0))
            unrealized_plpc = float(pos.get("unrealized_plpc", 0.0)) # Profit percentage
            current_price = float(pos.get("current_price", 0.0))

            sym_sl, sym_tp, sym_arm = get_symbol_risk(symbol)
            q = quotes.get(symbol, {})
            imb = q.get("imbalance", 0.0)

            # Check if trailing take profit should trigger
            if unrealized_plpc >= sym_tp:
                logger.info(f"🎯 [Take Profit Trigger] {symbol} {side.upper()} reached +{unrealized_plpc*100:.2f}% (Target: {sym_tp*100:.1f}%)")
                self.close_position(symbol, f"TAKE_PROFIT_{unrealized_plpc*100:.1f}PCT")

            # Check if hard stop loss triggered
            elif unrealized_plpc <= -sym_sl:
                logger.warning(f"🛑 [Stop Loss Trigger] {symbol} {side.upper()} hit loss -{abs(unrealized_plpc)*100:.2f}% (Max SL: {sym_sl*100:.1f}%)")
                self.close_position(symbol, f"STOP_LOSS_{unrealized_plpc*100:.1f}PCT")

            # Check adverse orderbook pressure
            elif (side == "long" and imb <= -0.75 and unrealized_plpc < -0.010):
                logger.warning(f"⚠️ [Microstructure Exit] {symbol} Long facing heavy ask wall (Imbalance: {imb:.2f}). Exiting to preserve capital.")
                self.close_position(symbol, "HEAVY_ASK_WALL_EXIT")

    def close_position(self, symbol: str, reason: str = "MANUAL"):
        """Closes an open position via Alpaca REST API"""
        logger.info(f"🔄 [Closing Position] {symbol} | Reason: {reason}")
        if self.simulation_mode:
            logger.info(f"🧪 [SIMULATION EXIT] {symbol} closed successfully | Reason: {reason}")
            self.positions.pop(symbol, None)
            self.trade_cooldowns[symbol] = time.time() + 120
            return

        try:
            resp = requests.delete(f"{ALPACA_REST_URL}/positions/{symbol}", headers=HEADERS, timeout=8)
            if resp.status_code in [200, 204]:
                logger.info(f"✅ [Position Closed] {symbol} closed successfully.")
                self.positions.pop(symbol, None)
                self.trade_cooldowns[symbol] = time.time() + 300 # 5 min cooldown
            else:
                logger.error(f"❌ Failed to close {symbol}: HTTP {resp.status_code} - {resp.text}")
        except Exception as e:
            logger.error(f"❌ Exception closing {symbol}: {e}")

    def evaluate_trading_opportunities(self):
        """Scans Universe, checks Signals, consults Jev AI, and executes trades"""
        current_active = len(self.positions)
        if current_active >= MAX_CONCURRENT_POSITIONS:
            logger.debug(f"At max concurrent positions ({current_active}/{MAX_CONCURRENT_POSITIONS}). Skipping new entries.")
            return

        equity = float(self.account_info.get("equity", 100000.0))
        target_dollar_size = equity * MAX_POSITION_PCT

        quotes = get_quotes_cache()

        # Step 1: Scan QQQ benchmark for macro regime
        qqq_bars = self.fetch_bars("QQQ", timeframe="1Min", limit=55)
        qqq_tech = self.calculate_technical_signals(qqq_bars)
        qqq_quote = quotes.get("QQQ", {})
        qqq_imb = qqq_quote.get("imbalance", 0.0)
        qqq_c = qqq_tech.get("price", 0.0)
        qqq_e50 = qqq_tech.get("ema50", 0.0)

        # Macro Trend Gate: QQQ > EMA50 for Long, QQQ < EMA50 for Inverse
        is_bullish = (qqq_c > qqq_e50) and (qqq_tech["signal"] == "BUY" or qqq_imb >= 0.15)
        is_bearish = (qqq_c < qqq_e50) and (qqq_tech["signal"] == "SELL" or qqq_imb <= -0.15)

        candidate_symbols = []
        if is_bullish:
            # Bullish: Prioritize Tech Leaders and TQQQ (Backtest validated best Sharpe)
            candidate_symbols = [("TQQQ", "buy"), ("NVDA", "buy"), ("AAPL", "buy"), ("SOXL", "buy"), ("TSLA", "buy"), ("QQQ", "buy")]
        elif is_bearish:
            # Bearish: Prioritize Inverse ETFs
            candidate_symbols = [("SQQQ", "buy"), ("SOXS", "buy")]
        else:
            # Neutral macro: selectively scan top resilient tech equities
            candidate_symbols = [("AAPL", "buy"), ("NVDA", "buy"), ("TQQQ", "buy"), ("QQQ", "buy")]

        for sym, side in candidate_symbols:
            if current_active >= MAX_CONCURRENT_POSITIONS:
                break
            if sym in self.positions:
                continue
            if time.time() < self.trade_cooldowns.get(sym, 0.0):
                continue

            # Fetch symbol bars
            df = self.fetch_bars(sym, timeframe="1Min", limit=30)
            tech = self.calculate_technical_signals(df)

            # Require technical alignment or strong LOB push
            sym_q = quotes.get(sym, {})
            sym_imb = sym_q.get("imbalance", 0.0)
            sym_price = sym_q.get("midPrice") or sym_q.get("askPrice") or 0.0

            if sym_price <= 0:
                continue

            signal_viable = (tech["signal"] == "BUY") or (sym_imb >= 0.25)
            if not signal_viable:
                continue

            # Step 2: Query Jev AI Sub-Second Orderbook Decision
            logger.info(f"🔎 [Signal Detected] {sym} {side.upper()} | Tech: {tech['reason']} | Imbalance: {sym_imb:+.2f}")
            jev_resp = query_jev_nasdaq(sym, side)

            is_approved = jev_resp.get("approved", False)
            score = jev_resp.get("score", 0.50)
            latency = jev_resp.get("latency_ms", 0.0)
            reason = jev_resp.get("reason", "N/A")
            target_px = jev_resp.get("target_price")
            order_type = jev_resp.get("order_type", "MARKET")

            logger.info(f"⚡ [Jev Decision] {sym} | Approved: {is_approved} | Score: {score:.3f} | Latency: {latency:.1f}ms | Reason: {reason}")

            if is_approved and score >= 0.55:
                # Calculate share quantity
                shares = int(target_dollar_size / sym_price)
                if shares < 1:
                    shares = 1

                logger.info(f"🎯 [Executing Trade] Approved by Jev AI! Sizing: {shares} shares (~${shares * sym_price:,.2f}) of {sym}")
                order_res = self.submit_alpaca_order(
                    symbol=sym,
                    side=side,
                    qty=shares,
                    order_type="market" if not target_px else "limit",
                    limit_price=target_px
                )
                if order_res:
                    current_active += 1

    def run_cycle(self):
        """Single loop execution cycle"""
        # 1. Check Clock
        is_open = self.check_market_clock()
        if not is_open:
            if self.simulation_mode:
                logger.info(f"🧪 [Simulation Mode Active] Off-market hours (Next Open: {self.next_open_str}). Executing offline simulation cycle...")
            else:
                logger.info(f"🌙 [Market Closed] Next Open: {self.next_open_str} | Waiting for regular trading hours...")
                time.sleep(60)
                return

        # 2. Sync State
        self.sync_account()
        self.sync_positions()

        # 3. Manage Open Trades
        self.manage_open_positions()

        # 4. Search & Execute Opportunities
        self.evaluate_trading_opportunities()

    def start(self):
        """Main 24/7 autonomous loop"""
        logger.info("🟢 [Daemon Started] Entering continuous trading loop...")
        self.sync_account()
        self.sync_positions()

        while True:
            try:
                self.run_cycle()
                time.sleep(LOOP_INTERVAL_SEC)
            except KeyboardInterrupt:
                logger.info("🛑 [Stop Signal] Stopping Alpaca Jev Trader gracefully.")
                break
            except Exception as e:
                logger.error(f"❌ [Main Loop Exception] {e}", exc_info=True)
                time.sleep(10)

if __name__ == "__main__":
    trader = AlpacaJevTrader()
    trader.start()
