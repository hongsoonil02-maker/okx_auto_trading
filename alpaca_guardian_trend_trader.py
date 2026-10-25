"""
alpaca_guardian_trend_trader.py — Autonomous Alpaca 1H Trend Following Swing Engine
- Strategy: Modified Cameron/HMA(100) + MACD + RSI Momentum Trend Swing
- Universe: High-Beta Trend Leaders (TSLA, TQQQ, NVDA, PLTR)
- Execution: 1H Confirmed Bars, Bracket/Stop-Loss Protection
- Risk Guard: 1% Account Risk per trade, Max 10% Equity per symbol (Max 30% gross swing exposure)
- Position Defense: +2R Take Profit 1 (50% scale out) + Move SL to Break-Even + 1H Signal Exit
- Isolation: Tagged client_order_id ('GD_') & state persistence to prevent interference with Golden Trader
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. Config & Parameters
# ==========================================
SYMBOLS = ["TSLA", "TQQQ", "NVDA", "PLTR"]
TIMEFRAME = "1Hour"
RISK_PER_TRADE = 0.01          # 1% equity risk per trade
MAX_POS_EQUITY_PCT = 0.10      # Max 10% equity per symbol (Swing allocation cap)
MAX_CONCURRENT_POS = 3         # Max 3 concurrent swing positions (30% max gross)
RR_RATIO = 2.0                 # 1:2 Risk/Reward for TP1
STATE_FILE = "alpaca_guardian_trend_state.json"
LOG_FILE = "alpaca_guardian_trend_trader.log"

# Logging setup
logger = logging.getLogger("AlpacaGuardianTrend")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Alpaca Credentials & URLs
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_REST_URL", "https://paper-api.alpaca.markets/v2")
if ALPACA_BASE_URL.endswith("/v2"):
    ALPACA_BASE_URL = ALPACA_BASE_URL[:-3]
ALPACA_DATA_URL = "https://data.alpaca.markets/v2"

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type": "application/json"
}

# ==========================================
# 2. Indicator Math (NumPy/Pandas)
# ==========================================
def calc_hma(series: pd.Series, length: int = 100) -> pd.Series:
    def wma(s, l):
        weights = np.arange(1, l + 1)
        return s.rolling(l).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    half_l = int(length / 2)
    sqrt_l = int(np.sqrt(length))
    diff = 2 * wma(series, half_l) - wma(series, length)
    return wma(diff, sqrt_l)

def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    f_ema = series.ewm(span=fast, adjust=False).mean()
    s_ema = series.ewm(span=slow, adjust=False).mean()
    macd = f_ema - s_ema
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig

def calc_rsi(series: pd.Series, length=14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

# ==========================================
# 3. Guardian Trend Engine
# ==========================================
class AlpacaGuardianTrendBot:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.last_bar_times: Dict[str, str] = {}
        self.load_state()

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    data = json.load(f)
                    self.positions = data.get("positions", {})
                    self.last_bar_times = data.get("last_bar_times", {})
                logger.info(f"Loaded state: {len(self.positions)} active swing positions.")
            except Exception as e:
                logger.error(f"State load failed: {e}")

    def save_state(self):
        try:
            with open(STATE_FILE, "w") as f:
                json.dump({
                    "positions": self.positions,
                    "last_bar_times": self.last_bar_times,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"State save failed: {e}")

    def is_market_open(self) -> bool:
        try:
            r = self.session.get(f"{ALPACA_BASE_URL}/v2/clock", timeout=5)
            if r.status_code == 200:
                return r.json().get("is_open", False)
        except Exception as e:
            logger.error(f"Clock check error: {e}")
        return False

    def get_account_equity(self) -> float:
        try:
            r = self.session.get(f"{ALPACA_BASE_URL}/v2/account", timeout=5)
            if r.status_code == 200:
                return float(r.json().get("equity", 10000.0))
        except Exception as e:
            logger.error(f"Account fetch error: {e}")
        return 10000.0

    def fetch_1h_bars(self, symbol: str, limit: int = 130) -> pd.DataFrame:
        url = f"{ALPACA_DATA_URL}/stocks/{symbol}/bars"
        start_time = (datetime.now(timezone.utc) - pd.Timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
        params = {
            "timeframe": "1Hour",
            "limit": limit,
            "feed": "iex",
            "sort": "desc",
            "start": start_time,
            "adjustment": "all"
        }
        try:
            r = self.session.get(url, params=params, timeout=8)
            if r.status_code == 200:
                bars = r.json().get("bars", [])
                if bars:
                    # Reverse to ascending order
                    df = pd.DataFrame(bars[::-1])
                    df.rename(columns={"t": "datetime", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}, inplace=True)
                    for col in ("open", "high", "low", "close", "volume"):
                        df[col] = df[col].astype(float)
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df.sort_values("datetime", inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    return df
        except Exception as e:
            logger.error(f"[{symbol}] Failed to fetch 1H bars: {e}")
        return pd.DataFrame()

    def evaluate_signals(self, symbol: str, df: pd.DataFrame):
        if len(df) < 115:
            return False, False, 0.0, 0.0, ""

        df['hma100'] = calc_hma(df['close'], 100)
        df['macd'], df['macd_signal'] = calc_macd(df['close'], 12, 26, 9)
        df['rsi'] = calc_rsi(df['close'], 14)
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        df['swing_low'] = df['low'].shift(1).rolling(10).min()

        # Last closed completed bar is -2 (-1 is live uncompleted bar)
        c_close = df['close'].iloc[-2]
        c_hma = df['hma100'].iloc[-2]
        p_hma = df['hma100'].iloc[-3]
        hma_rising = (c_hma > p_hma) and (c_close > c_hma)

        c_macd = df['macd'].iloc[-2]
        c_sig = df['macd_signal'].iloc[-2]
        p_macd = df['macd'].iloc[-3]
        p_sig = df['macd_signal'].iloc[-3]

        c_rsi = df['rsi'].iloc[-2]
        p_rsi = df['rsi'].iloc[-3]
        c_vol = df['volume'].iloc[-2]
        c_vol_ma = df['vol_ma20'].iloc[-2]
        vol_ok = c_vol > c_vol_ma

        swing_low = df['swing_low'].iloc[-2]
        bar_time = str(df['datetime'].iloc[-2])

        entry = False
        if hma_rising and vol_ok:
            if (p_macd <= p_sig) and (c_macd > c_sig) and (50 <= c_rsi < 70):
                entry = True
            elif (c_macd > c_sig) and (p_rsi < 50 and c_rsi >= 50):
                entry = True

        exit_sig = False
        if (p_macd >= p_sig and c_macd < c_sig) or (p_rsi >= 50 and c_rsi < 50):
            exit_sig = True

        return entry, exit_sig, swing_low, c_close, bar_time

    def enter_swing(self, symbol: str, current_price: float, swing_low: float):
        if len(self.positions) >= MAX_CONCURRENT_POS:
            logger.info(f"[{symbol}] Max concurrent swing positions ({MAX_CONCURRENT_POS}) reached. Skipping.")
            return

        equity = self.get_account_equity()
        raw_sl = swing_low * 0.999
        risk_dist = current_price - raw_sl

        # Guard: Minimum risk distance 0.5%
        min_dist = current_price * 0.005
        if risk_dist < min_dist:
            raw_sl = current_price - min_dist
            risk_dist = min_dist

        # Position sizing: 1% equity risk
        target_risk_dollars = equity * RISK_PER_TRADE
        desired_qty = target_risk_dollars / risk_dist

        # Cap at 10% equity per symbol
        max_notional = equity * MAX_POS_EQUITY_PCT
        max_qty = max_notional / current_price
        final_qty = int(min(desired_qty, max_qty))

        if final_qty < 1:
            logger.warning(f"[{symbol}] Calculated qty < 1 share. Skipping.")
            return

        tp1_price = round(current_price + (risk_dist * RR_RATIO), 2)
        raw_sl = round(raw_sl, 2)
        client_order_id = f"GD_BUY_{symbol}_{int(time.time())}"

        logger.info(f">>> [ALPACASWING ENTRY] {symbol} | Qty: {final_qty} | Price: ${current_price:.2f} | SL: ${raw_sl:.2f} | TP1(+2R): ${tp1_price:.2f}")

        order_data = {
            "symbol": symbol,
            "qty": str(final_qty),
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": client_order_id
        }

        try:
            r = self.session.post(f"{ALPACA_BASE_URL}/v2/orders", json=order_data, timeout=8)
            if r.status_code in (200, 201):
                order = r.json()
                logger.info(f"[{symbol}] Entry order placed: id={order.get('id')}")
                self.positions[symbol] = {
                    "entry_price": current_price,
                    "qty": final_qty,
                    "stop_loss": raw_sl,
                    "tp1_price": tp1_price,
                    "tp1_hit": False,
                    "entry_time": datetime.now(timezone.utc).isoformat()
                }
                self.save_state()
            else:
                logger.error(f"[{symbol}] Entry order failed ({r.status_code}): {r.text}")
        except Exception as e:
            logger.error(f"[{symbol}] Entry order exception: {e}")

    def manage_positions(self):
        symbols_to_remove = []
        for symbol, pos in list(self.positions.items()):
            df = self.fetch_1h_bars(symbol, limit=20)
            if df.empty:
                continue

            current_price = float(df['close'].iloc[-1])
            sl = pos["stop_loss"]
            tp1 = pos["tp1_price"]
            qty = pos["qty"]
            ep = pos["entry_price"]

            # 1. Stop Loss Hit
            if current_price <= sl:
                logger.warning(f"<<< [{symbol} SL HIT] Current: ${current_price:.2f} <= SL: ${sl:.2f}")
                self.close_position(symbol, qty, "STOP_LOSS")
                symbols_to_remove.append(symbol)
                continue

            # 2. +2R Take Profit 1 (50% scale out & Move SL to Break-Even)
            if not pos.get("tp1_hit", False) and current_price >= tp1:
                half_qty = max(1, qty // 2)
                logger.info(f"<<< [{symbol} TP1 +2R HIT] Selling 50% ({half_qty} shares) at ${current_price:.2f}")
                self.close_position(symbol, half_qty, "TP1_HALF")
                pos["qty"] -= half_qty
                pos["tp1_hit"] = True
                pos["stop_loss"] = ep  # Move Stop Loss to Break-Even
                self.save_state()
                logger.info(f">>> [{symbol} FREE-RIDE] Stop Loss moved to Break-Even (${ep:.2f})")

            # 3. Technical Indicator Exit
            _, exit_sig, _, _, _ = self.evaluate_signals(symbol, df)
            if exit_sig:
                logger.info(f"<<< [{symbol} SIGNAL EXIT] 1H MACD Dead-Cross or RSI < 50 triggered at ${current_price:.2f}")
                self.close_position(symbol, pos["qty"], "SIGNAL_EXIT")
                symbols_to_remove.append(symbol)

        for s in symbols_to_remove:
            if s in self.positions:
                del self.positions[s]
        if symbols_to_remove:
            self.save_state()

    def close_position(self, symbol: str, qty: int, reason: str):
        client_order_id = f"GD_SELL_{symbol}_{int(time.time())}"
        order_data = {
            "symbol": symbol,
            "qty": str(qty),
            "side": "sell",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": client_order_id
        }
        try:
            r = self.session.post(f"{ALPACA_BASE_URL}/v2/orders", json=order_data, timeout=8)
            if r.status_code in (200, 201):
                logger.info(f"[{symbol}] Close order executed ({reason}): {qty} shares")
            else:
                logger.error(f"[{symbol}] Close order failed: {r.text}")
        except Exception as e:
            logger.error(f"[{symbol}] Close order exception: {e}")

    def run(self):
        logger.info("==================================================")
        logger.info("Alpaca Guardian 1H Trend Swing Trader Started")
        logger.info(f"Universe: {', '.join(SYMBOLS)} | Timeframe: {TIMEFRAME}")
        logger.info(f"Max Positions: {MAX_CONCURRENT_POS} | Allocation: {MAX_POS_EQUITY_PCT*100}%/symbol")
        logger.info("==================================================")

        while True:
            try:
                is_open = self.is_market_open()
                if not is_open:
                    logger.info("[Market Closed] US Market is closed. Guardian Trend engine waiting (checking every 60s)...")
                    time.sleep(60)
                    continue

                # Manage existing swing positions
                if self.positions:
                    self.manage_positions()

                # Scan for new swing entries
                for symbol in SYMBOLS:
                    if symbol in self.positions:
                        continue

                    df = self.fetch_1h_bars(symbol, limit=130)
                    if df.empty:
                        continue

                    entry, exit_sig, swing_low, close_px, bar_time = self.evaluate_signals(symbol, df)
                    last_time = self.last_bar_times.get(symbol, "")

                    if entry and bar_time != last_time:
                        self.last_bar_times[symbol] = bar_time
                        self.enter_swing(symbol, close_px, swing_low)
                    else:
                        logger.info(f"[1H Scan] {symbol}: ${close_px:.2f} | HMA100: ${df['hma100'].iloc[-2]:.2f} | RSI: {df['rsi'].iloc[-2]:.1f}")

                time.sleep(60)

            except Exception as e:
                logger.error(f"Main loop exception: {e}")
                time.sleep(30)

if __name__ == '__main__':
    bot = AlpacaGuardianTrendBot()
    if len(sys.argv) > 1 and sys.argv[1] == '--dry-run':
        open_status = bot.is_market_open()
        eq = bot.get_account_equity()
        print(f"Alpaca Dry-Run Check: MarketOpen={open_status}, Equity=${eq:,.2f}")
        for s in SYMBOLS:
            df = bot.fetch_1h_bars(s, limit=130)
            entry, exit_sig, sl, px, t = bot.evaluate_signals(s, df)
            print(f"[{s}] Close=${px:.2f}, EntrySignal={entry}, ExitSignal={exit_sig}, LastBar={t}")
    else:
        bot.run()
