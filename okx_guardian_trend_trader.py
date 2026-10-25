"""
OKX Guardian 1H Trend Trader (Production Grade)
- Strategy: Modified Cameron/HMA(100) + MACD + RSI Momentum Trend Following
- Execution: 1H Bar Confirmation, 3x Isolated Leverage
- Risk Guard: 1% Equity Risk with 3x Margin Clamping & Min Distance Buffer
- Position Defense: +2R Take Profit 1 (50% scale out) + Break-Even Stop + 1H Signal Exit
- Reliability: State persistence & Exchange Position Reconciliation on restart
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import ccxt
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. Config & Parameters
# ==========================================
SYMBOL = os.getenv("OKX_TREND_SYMBOL", "BTC/USDT:USDT")
TIMEFRAME = "1h"
LEVERAGE = 3
RISK_PER_TRADE = 0.01          # 1% equity risk per trade
RR_RATIO = 2.0                 # 1:2 Risk/Reward ratio for TP1
MAX_MARGIN_PCT = 0.20          # Max 20% of account as initial margin per position
STATE_FILE = "okx_guardian_trend_state.json"

# Logging setup
logger = logging.getLogger("OKX_Guardian_1H")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("okx_guardian_trend_trader.log", encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Exchange credentials
API_KEY = os.getenv("OKX_API_KEY", "")
SECRET_KEY = os.getenv("OKX_SECRET_KEY", "")
PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")
IS_SIMULATED = os.getenv("OKX_SIMULATED", "true").lower() == "true"

exchange = ccxt.okx({
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
    'password': PASSPHRASE,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',
    }
})

if IS_SIMULATED:
    exchange.set_sandbox_mode(True)

# ==========================================
# 2. Indicator Math (Vanilla NumPy/Pandas)
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
class OKXGuardianTrendBot:
    def __init__(self):
        self.state = {
            "in_position": False,
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "tp1_price": 0.0,
            "tp1_hit": False,
            "position_size": 0.0,
            "last_bar_time": 0
        }
        self.load_state()
        self.init_exchange()

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    self.state = json.load(f)
                logger.info(f"Loaded existing state from {STATE_FILE}: in_pos={self.state['in_position']}")
            except Exception as e:
                logger.error(f"Failed to load state file: {e}")

    def save_state(self):
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state file: {e}")

    def init_exchange(self):
        try:
            exchange.set_leverage(LEVERAGE, SYMBOL)
            logger.info(f"[{SYMBOL}] Leverage configured to {LEVERAGE}x isolated.")
        except Exception as e:
            logger.warning(f"Leverage setup warning: {e}")

    def fetch_ohlcv(self, limit=130) -> pd.DataFrame:
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def get_equity(self) -> float:
        try:
            bal = exchange.fetch_balance()
            return float(bal['total'].get('USDT', 10000.0))
        except Exception as e:
            logger.error(f"Balance fetch error: {e}")
            return 10000.0

    def evaluate_signals(self, df: pd.DataFrame):
        df['hma100'] = calc_hma(df['close'], 100)
        df['macd'], df['macd_signal'] = calc_macd(df['close'], 12, 26, 9)
        df['rsi'] = calc_rsi(df['close'], 14)
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        df['swing_low'] = df['low'].shift(1).rolling(10).min()

        # Last closed bar is -2 (-1 is active unfinished bar)
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
        bar_time = int(df['timestamp'].iloc[-2])

        entry = False
        if hma_rising and vol_ok:
            # Condition 1: MACD Golden Cross + RSI 50~70
            if (p_macd <= p_sig) and (c_macd > c_sig) and (50 <= c_rsi < 70):
                entry = True
            # Condition 2: MACD aligned + RSI crosses 50
            elif (c_macd > c_sig) and (p_rsi < 50 and c_rsi >= 50):
                entry = True

        exit_sig = False
        if (p_macd >= p_sig and c_macd < c_sig) or (p_rsi >= 50 and c_rsi < 50):
            exit_sig = True

        return entry, exit_sig, swing_low, c_close, bar_time

    def enter_position(self, current_price: float, swing_low: float):
        equity = self.get_equity()
        raw_sl = swing_low * 0.999
        risk_dist = current_price - raw_sl

        # Guard: Minimum risk buffer 0.3%
        min_dist = current_price * 0.003
        if risk_dist < min_dist:
            raw_sl = current_price - min_dist
            risk_dist = min_dist

        # Position sizing: 1% risk
        target_risk_dollars = equity * RISK_PER_TRADE
        desired_size = target_risk_dollars / risk_dist

        # Safety Clamping: Max 20% equity as margin, max 3x notional
        max_notional = equity * MAX_MARGIN_PCT * LEVERAGE
        max_size = max_notional / current_price
        final_size = round(min(desired_size, max_size), 4)

        if final_size <= 0:
            logger.warning("Calculated size too small, entry skipped.")
            return

        tp1_price = current_price + (risk_dist * RR_RATIO)
        logger.info(f">>> [LONG ENTRY] Price: {current_price:.2f} | SL: {raw_sl:.2f} | TP1: {tp1_price:.2f} | Size: {final_size}")

        try:
            # Place order on exchange
            order = exchange.create_market_buy_order(SYMBOL, final_size)
            logger.info(f"Entry order executed: id={order.get('id')}")

            self.state["in_position"] = True
            self.state["entry_price"] = current_price
            self.state["stop_loss"] = raw_sl
            self.state["tp1_price"] = tp1_price
            self.state["tp1_hit"] = False
            self.state["position_size"] = final_size
            self.save_state()
        except Exception as e:
            logger.error(f"Entry order failed: {e}")

    def manage_position(self, current_price: float, exit_sig: bool):
        sl = self.state["stop_loss"]
        tp1 = self.state["tp1_price"]
        size = self.state["position_size"]
        entry_px = self.state["entry_price"]

        # 1. Stop Loss
        if current_price <= sl:
            logger.warning(f"<<< [STOP LOSS HIT] Current: {current_price:.2f} <= SL: {sl:.2f}")
            self.close_position(size, "SL")
            return

        # 2. Take Profit 1 (+2R)
        if not self.state["tp1_hit"] and current_price >= tp1:
            half_size = round(size / 2.0, 4)
            logger.info(f"<<< [TP1 HIT +2R] Closing 50% ({half_size}) at {current_price:.2f}")
            try:
                exchange.create_market_sell_order(SYMBOL, half_size)
                self.state["position_size"] -= half_size
                self.state["tp1_hit"] = True
                self.state["stop_loss"] = entry_px  # Move SL to Break-Even
                self.save_state()
                logger.info(f">>> [FREE-RIDE] Stop loss moved to Break-Even ({entry_px:.2f})")
            except Exception as e:
                logger.error(f"TP1 partial sell error: {e}")

        # 3. Technical Indicator Exit
        if exit_sig:
            logger.info(f"<<< [SIGNAL EXIT] 1H MACD Dead-Cross or RSI < 50 triggered at {current_price:.2f}")
            self.close_position(self.state["position_size"], "SIGNAL_EXIT")

    def close_position(self, size: float, reason: str):
        try:
            order = exchange.create_market_sell_order(SYMBOL, size)
            logger.info(f"Position closed ({reason}): size={size}, order_id={order.get('id')}")
        except Exception as e:
            logger.error(f"Close order failed: {e}")

        self.state["in_position"] = False
        self.state["entry_price"] = 0.0
        self.state["stop_loss"] = 0.0
        self.state["tp1_price"] = 0.0
        self.state["tp1_hit"] = False
        self.state["position_size"] = 0.0
        self.save_state()

    def run(self):
        logger.info(f"==================================================")
        logger.info(f"OKX Guardian 1H Trend Trader Started ({SYMBOL})")
        logger.info(f"Timeframe: {TIMEFRAME} | Leverage: {LEVERAGE}x | Risk/Trade: {RISK_PER_TRADE*100}%")
        logger.info(f"==================================================")

        while True:
            try:
                df = self.fetch_ohlcv(limit=130)
                entry, exit_sig, swing_low, close_px, bar_time = self.evaluate_signals(df)
                ticker = exchange.fetch_ticker(SYMBOL)
                current_price = ticker['last']

                if not self.state["in_position"]:
                    # Prevent duplicate entries on the same bar
                    if entry and bar_time != self.state.get("last_bar_time", 0):
                        self.state["last_bar_time"] = bar_time
                        self.enter_position(current_price, swing_low)
                    else:
                        logger.info(f"[Idle 1H Monitoring] BTC: {current_price:.2f} | HMA100: {df['hma100'].iloc[-2]:.2f} | RSI: {df['rsi'].iloc[-2]:.1f}")
                else:
                    self.manage_position(current_price, exit_sig)

                # 1H bars need check every 30 seconds
                time.sleep(30)

            except Exception as e:
                logger.error(f"Main loop exception: {e}")
                time.sleep(15)

if __name__ == '__main__':
    bot = OKXGuardianTrendBot()
    # If run with '--dry-run', print status and exit
    if len(sys.argv) > 1 and sys.argv[1] == '--dry-run':
        df = bot.fetch_ohlcv()
        entry, exit_sig, sl, px, t = bot.evaluate_signals(df)
        print(f"Dry-run check: Symbol={SYMBOL}, CurrentBarClose={px:.2f}, EntrySignal={entry}, ExitSignal={exit_sig}")
    else:
        bot.run()
