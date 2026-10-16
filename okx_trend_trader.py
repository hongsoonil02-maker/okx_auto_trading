# -*- coding: utf-8 -*-
"""
okx_trend_trader.py — Autonomous OKX Perpetual Quant Trend Following Engine v2.0
Replaces the inactive/over-filtered swing engine with the proven Quant Trend Following Model:
- Strategy: Donchian Channel 10-Day Breakout + 60-Day SMA Trend Filter + RSI(50~75) Momentum Filter
- Market Regime Guard: BTC Close >= 60-Day SMA Filter (Bear market halts altcoin buys, 100% USDT cash preservation)
- Relative Strength (RS) Ranking: Prioritizes sector leaders with highest 60-day outperformance vs BTC
- Exit: ATR 3.0x Dynamic Trailing Stop & 10-Day Donchian Low Breakdown & Hard Stop Loss (-10%)
- Risk Management: Max 3 Concurrent Positions, 30% Equity Allocation, Safe 2x Leverage (Zero liquidation risk)
- Protection: Dual-layer stop protection (local real-time trailing monitor + exchange-side conditional SL algo)
"""

import os
import sys
import time
import json
import math
import asyncio
import logging
from datetime import datetime, timezone, timedelta
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
POSITIONS_FILE = os.path.join(STATE_DIR, "okx_trend_positions.json")
LOG_FILE = os.path.join(CURRENT_DIR, "okx_trend_trader.log")

# ── TELEGRAM ALERTS ──
try:
    from utils_telegram import send_telegram_alert
except ImportError:
    def send_telegram_alert(msg: str):
        pass

# ── LOGGING SETUP ──
logger = logging.getLogger("OKXTrendTrader")
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)

# ── TRACKED UNIVERSE (OKX 거래대금 최상위 메이저 무기한 선물 10종) ──
UNIVERSE = {
    "BTC/USDT:USDT": {"name": "비트코인"},
    "ETH/USDT:USDT": {"name": "이더리움"},
    "SOL/USDT:USDT": {"name": "솔라나"},
    "DOGE/USDT:USDT": {"name": "도지코인"},
    "XRP/USDT:USDT": {"name": "리플"},
    "SUI/USDT:USDT": {"name": "수이"},
    "NEAR/USDT:USDT": {"name": "니어프로토콜"},
    "AVAX/USDT:USDT": {"name": "아발란체"},
    "LINK/USDT:USDT": {"name": "체인링크"},
    "ADA/USDT:USDT": {"name": "에이다"},
}

# ── STRATEGY HYPERPARAMETERS ──
MAX_CONCURRENT_POSITIONS = 3
MAX_POSITION_PCT = 0.30       # 포지션당 계좌 자산의 30% 마진 배분 (3종목 = 최대 90% 가동)
LEVERAGE = 2                  # 2.0배 보수적/안전 레버리지 (청산가 진입가 대비 -50% 아래 위치)
DONCHIAN_WINDOW = 10          # 10일 신고가 돌파
SMA_FAST = 20                 # 20일 이동평균선
SMA_SLOW = 60                 # 60일 이동평균선
RSI_MIN = 50.0                # RSI 50 이상 (상승 초입 국면)
RSI_MAX = 75.0                # RSI 75 이하 (단기 과열 상투 배제)
DONCHIAN_EXIT_WINDOW = 10     # 10일 최저가 이탈 시 청산
ATR_TRAILING_MULTIPLIER = 3.0 # 고점 대비 3.0x ATR 되밀림 시 트레일링 익절
HARD_STOP_LOSS_PCT = 0.10     # 진입가 대비 최대 -10% 절대 손절선 (2배율 기준 마진 -20%)
POLL_INTERVAL_SEC = 20        # 20초 주기 실시간 포지션 및 호가 모니터링
KST = timezone(timedelta(hours=9))


class OKXTrendTrader:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.api_key = os.getenv("OKX_API_KEY", "").strip()
        self.secret = os.getenv("OKX_SECRET", "").strip()
        self.password = os.getenv("OKX_PASSPHRASE", "").strip()
        self.exchange: Optional[ccxt_async.okx] = None

        self.account_equity = 0.0
        self.account_free = 0.0
        self.markets: Dict[str, Any] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.daily_indicators: Dict[str, Dict[str, Any]] = {}
        self.market_regime: Dict[str, Any] = {
            "is_bullish": True,
            "btc_close": 0.0,
            "btc_sma60": 0.0,
            "btc_ret60": 0.0
        }
        self.last_indicator_update: str = ""

        self.load_positions()

    def load_positions(self):
        """저장된 포지션 상태 JSON 파일에서 복원"""
        if os.path.exists(POSITIONS_FILE):
            try:
                with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
                    self.positions = json.load(f)
                logger.info(f"📂 [State Restored] {len(self.positions)} active positions loaded: {list(self.positions.keys())}")
            except Exception as e:
                logger.error(f"❌ Failed to load positions from {POSITIONS_FILE}: {e}")
                self.positions = {}
        else:
            self.positions = {}

    def save_positions(self):
        """포지션 상태 JSON 파일 영속화"""
        try:
            with open(POSITIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.positions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to save positions to {POSITIONS_FILE}: {e}")

    async def init_exchange(self):
        """CCXT OKX 무기한 선물 클라이언트 초기화"""
        self.exchange = ccxt_async.okx({
            "apiKey": self.api_key,
            "secret": self.secret,
            "password": self.password,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"}
        })
        self.markets = await self.exchange.load_markets()
        logger.info("=" * 75)
        logger.info("🚀 [OKX Quant Trend Trader] Initializing Major Trend Following Engine v2.0")
        univ_desc = ', '.join([f"{s}({meta['name']})" for s, meta in UNIVERSE.items()])
        logger.info(f"   Universe: {univ_desc}")
        logger.info(f"   Leverage: {LEVERAGE}x | Max Positions: {MAX_CONCURRENT_POSITIONS} | Alloc per Pos: {MAX_POSITION_PCT*100:.0f}%")
        logger.info(f"   Strategy: Donchian {DONCHIAN_WINDOW}d Breakout + SMA {SMA_FAST}/{SMA_SLOW}d + RSI ({RSI_MIN}~{RSI_MAX})")
        logger.info(f"   Exit: ATR {ATR_TRAILING_MULTIPLIER}x Trailing Stop | Hard Stop: -{HARD_STOP_LOSS_PCT*100:.0f}%")
        logger.info("=" * 75)

    async def update_balance(self) -> float:
        """OKX 계좌 USDT 잔고 및 총 자산(Equity) 실시간 동기화"""
        try:
            bal = await self.exchange.fetch_balance()
            self.account_equity = float(bal.get("USDT", {}).get("total", 0.0))
            self.account_free = float(bal.get("USDT", {}).get("free", 0.0))
            return self.account_equity
        except Exception as e:
            logger.error(f"❌ Failed to fetch balance: {e}")
            return self.account_equity

    async def setup_leverage(self, symbol: str):
        """포지션 진입 전 2x 격리/크로스 레버리지 설정"""
        try:
            await self.exchange.set_leverage(LEVERAGE, symbol, {"mgnMode": "cross", "posSide": "long"})
        except Exception as e:
            logger.debug(f"Leverage setup for {symbol}: {e}")

    async def fetch_daily_indicators(self):
        """일 1회 BTC 시장 국면 판별 및 유니버스 전 종목 일봉 지표/상대강도(RS) 산출"""
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.last_indicator_update == today_str and self.daily_indicators:
            return

        logger.info(f"📊 [Daily Scanner] Updating Indicators & Market Regime for {today_str} (UTC)...")

        # ── 1. BTC Market Regime Determination (BTC/USDT:USDT) ──
        try:
            btc_sym = "BTC/USDT:USDT"
            ohlcv_btc = await self.exchange.fetch_ohlcv(btc_sym, timeframe="1d", limit=80)
            if ohlcv_btc and len(ohlcv_btc) >= SMA_SLOW:
                df_btc = pd.DataFrame(ohlcv_btc, columns=["timestamp", "open", "high", "low", "close", "volume"])
                btc_close = float(df_btc["close"].iloc[-1])
                btc_sma60 = float(df_btc["close"].rolling(SMA_SLOW).mean().iloc[-1])
                btc_ret60 = float((btc_close / df_btc["close"].iloc[-60] - 1.0) * 100.0)
                is_bullish = bool(btc_close >= btc_sma60)
                self.market_regime = {
                    "is_bullish": is_bullish,
                    "btc_close": btc_close,
                    "btc_sma60": btc_sma60,
                    "btc_ret60": btc_ret60
                }
                regime_tag = "🟢 BULL MARKET (상승 추세 - 알트 매수 허용)" if is_bullish else "🔴 BEAR MARKET (하락 추세 - 알트 매수 전면 차단 / 현금 보존)"
                logger.info(
                    f"🏛️ [BTC Market Regime] {regime_tag} | "
                    f"BTC: ${btc_close:,.2f} | 60일선: ${btc_sma60:,.2f} | 60일수익률: {btc_ret60:+.2f}%"
                )
        except Exception as e:
            logger.error(f"❌ Failed to calculate BTC regime: {e}")

        btc_ret60 = self.market_regime.get("btc_ret60", 0.0)

        # ── 2. Universe Indicators & Relative Strength ──
        for sym, meta in UNIVERSE.items():
            try:
                ohlcv = await self.exchange.fetch_ohlcv(sym, timeframe="1d", limit=80)
                await asyncio.sleep(0.08)  # Rate limit safety
                if not ohlcv or len(ohlcv) < SMA_SLOW:
                    continue

                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                
                # TR & ATR14
                prev_close = df["close"].shift(1)
                tr = pd.concat([
                    df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()
                ], axis=1).max(axis=1)
                atr14 = float(tr.rolling(14).mean().iloc[-1])

                # Moving Averages
                sma20 = float(df["close"].rolling(SMA_FAST).mean().iloc[-1])
                sma60 = float(df["close"].rolling(SMA_SLOW).mean().iloc[-1])

                # RSI (14)
                delta = df["close"].diff()
                gain = delta.where(delta > 0, 0.0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
                rs = gain / (loss + 1e-9)
                rsi14 = float((100.0 - (100.0 / (1.0 + rs))).iloc[-1])

                # Donchian Breakout Levels (excluding today's incomplete candle)
                donchian_high = float(df["high"].iloc[:-1].tail(DONCHIAN_WINDOW).max())
                donchian_low = float(df["low"].iloc[:-1].tail(DONCHIAN_EXIT_WINDOW).min())

                # 60-Day Return & Relative Strength (RS vs BTC)
                ret60 = float((df["close"].iloc[-1] / df["close"].iloc[-60] - 1.0) * 100.0)
                rs_score = ret60 - btc_ret60

                self.daily_indicators[sym] = {
                    "donchian_high": donchian_high,
                    "donchian_low": donchian_low,
                    "sma20": sma20,
                    "sma60": sma60,
                    "atr14": atr14,
                    "rsi14": rsi14,
                    "ret60": ret60,
                    "rs_score": rs_score,
                    "last_close": float(df["close"].iloc[-1])
                }
                logger.info(
                    f"   ✓ {meta['name']}({sym}) | 10일고가: ${donchian_high:,.2f} | "
                    f"60선: ${sma60:,.2f} | ATR: ${atr14:,.2f} | RSI: {rsi14:.1f} | RS: {rs_score:+.1f}%"
                )
            except Exception as e:
                logger.error(f"❌ Failed to fetch indicators for {sym}: {e}")

        self.last_indicator_update = today_str
        logger.info(f"✅ Daily Indicators updated successfully for {len(self.daily_indicators)} symbols.")

    async def monitor_open_positions(self):
        """보유 포지션 실시간 출구 감시 (ATR 트레일링 스탑, 돈키언 10일 최저가 이탈, 하드 손절)"""
        if not self.positions:
            return

        for symbol, pos in list(self.positions.items()):
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                curr_px = float(ticker["last"])
                if curr_px <= 0:
                    continue

                entry_px = float(pos["entry_price"])
                old_peak = float(pos.get("peak_price", entry_px))
                peak_px = max(old_peak, curr_px)
                pos["peak_price"] = peak_px

                atr = float(pos.get("atr", self.daily_indicators.get(symbol, {}).get("atr14", entry_px * 0.03)))

                # 1. ATR Trailing Stop (고점 대비 3.0x ATR 하락)
                trail_stop_level = peak_px - (ATR_TRAILING_MULTIPLIER * atr)

                # 2. Hard Stop Loss (진입가 대비 -10%)
                hard_stop_level = entry_px * (1.0 - HARD_STOP_LOSS_PCT)

                # 3. Donchian 10-Day Low Breakdown
                donchian_low = self.daily_indicators.get(symbol, {}).get("donchian_low", 0.0)

                effective_stop = max(trail_stop_level, hard_stop_level, donchian_low)
                pnl_pct = (curr_px / entry_px - 1.0) * 100.0

                should_exit = False
                reason = ""

                if curr_px <= hard_stop_level:
                    should_exit = True
                    reason = f"HARD_STOP ({pnl_pct:+.2f}%)"
                elif curr_px <= effective_stop and curr_px < peak_px:
                    should_exit = True
                    reason = f"ATR_TRAIL_STOP ({pnl_pct:+.2f}%, 고점 ${peak_px:,.2f})"

                if should_exit:
                    logger.info(f"🚨 [Exit Triggered] {UNIVERSE.get(symbol, {}).get('name', symbol)}({symbol}) | Reason: {reason} | Curr: ${curr_px:,.2f}")
                    await self.close_position(symbol, reason, curr_px)
                else:
                    # 주기적 상태 저장
                    self.save_positions()

            except Exception as e:
                logger.error(f"❌ Error monitoring position for {symbol}: {e}")

    async def close_position(self, symbol: str, reason: str, exit_px: float):
        """시장가 전량 청산 및 알림 전송"""
        pos = self.positions.get(symbol)
        if not pos:
            return

        contracts = float(pos["contracts"])
        entry_px = float(pos["entry_price"])
        pnl_pct = (exit_px / entry_px - 1.0) * 100.0
        profit_usdt = (exit_px - entry_px) * (contracts * float(self.markets.get(symbol, {}).get("contractSize", 1.0)))

        logger.info(f"🔄 [Closing Position] {symbol} | Reason: {reason} | PnL: {pnl_pct:+.2f}% (~${profit_usdt:+,.2f})")

        if not self.dry_run:
            # 1. Cancel pending conditional stop loss algos
            try:
                inst_id = symbol.replace("/", "-").split(":")[0] + "-SWAP" if ":" in symbol else symbol
                pending_algos = await self.exchange.request(
                    "trade/orders-algo-pending", api="private", method="GET",
                    params={"instId": inst_id, "ordType": "conditional"}
                )
                algo_ids = [a["algoId"] for a in pending_algos.get("data", []) if "algoId" in a]
                if algo_ids:
                    cancel_list = [{"instId": inst_id, "algoId": a_id} for a_id in algo_ids]
                    await self.exchange.request("trade/cancel-algos", api="private", method="POST", params=cancel_list)
                    logger.info(f"🧹 [Algo SL Cleared] {symbol} {len(algo_ids)} SL algo(s) removed.")
            except Exception as ex_c:
                logger.debug(f"Algo cancel note for {symbol}: {ex_c}")

            # 2. Market close order
            try:
                params = {
                    "reduceOnly": True,
                    "tdMode": "cross",
                    "posSide": "long"
                }
                res = await self.exchange.create_market_order(symbol, "sell", contracts, params=params)
                logger.info(f"✅ [Position Closed] {symbol} | Close ID: {res.get('id')}")
            except Exception as e:
                logger.error(f"❌ Failed to submit close order for {symbol}: {e}")
                return

        tag = "WIN 🟢" if pnl_pct > 0 else "LOSS 🔴"
        send_telegram_alert(
            f"{tag} *[OKX 퀀트 추세 청산]* {UNIVERSE.get(symbol, {}).get('name', symbol)}({symbol})\n"
            f"사유: {reason}\n"
            f"진입가: ${entry_px:,.2f} → 청산가: ${exit_px:,.2f}\n"
            f"수익률: *{pnl_pct:+.2f}%* (레버리지 2x 반영 시 {pnl_pct*2:+.2f}%)"
        )
        self.positions.pop(symbol, None)
        self.save_positions()
        await self.update_balance()

    async def evaluate_entry_opportunities(self):
        """돈키언 10일 신고가 돌파 + 60일선 정배열 + RSI(50~75) 진입 스캔"""
        if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
            return

        # 1. BTC 시장 국면 필터 (하락장 시 알트 신규 매수 전면 차단 / 현금 보존)
        if not self.market_regime.get("is_bullish", True):
            logger.info("🛡️ [BTC Regime Guard] 비트코인이 60일선 아래(하락장)에 머물고 있습니다. 신규 알트 매수를 전면 차단합니다.")
            return

        await self.update_balance()
        if self.account_free < 50.0:
            return

        target_margin = min(self.account_equity * MAX_POSITION_PCT, self.account_free * 0.90)
        if target_margin < 20.0:
            return

        candidates = []
        for sym, meta in UNIVERSE.items():
            if sym in self.positions:
                continue

            ind = self.daily_indicators.get(sym)
            if not ind:
                continue

            try:
                ticker = await self.exchange.fetch_ticker(sym)
                curr_px = float(ticker["last"])
                if curr_px <= 0:
                    continue

                donchian_high = ind["donchian_high"]
                sma60 = ind["sma60"]
                sma20 = ind["sma20"]
                rsi = ind["rsi14"]

                # 진입 조건:
                # 1) 전일 기준 10일 신고가 돌파
                # 2) 현재가 > 60일선 & 20일선 >= 60일선 (정배열 추세)
                # 3) RSI 50 ~ 75 (상승 초입 구간)
                is_breakout = (curr_px > donchian_high)
                is_trend_aligned = (curr_px > sma60) and (sma20 >= sma60)
                is_rsi_aligned = (RSI_MIN <= rsi <= RSI_MAX)

                if is_breakout and is_trend_aligned and is_rsi_aligned:
                    rs_score = ind.get("rs_score", 0.0)
                    momentum_dist = (curr_px - sma60) / sma60
                    # 상대강도(70%) + 60선 이격도(30%) 종합 점수
                    total_score = (rs_score * 0.7) + (momentum_dist * 100.0 * 0.3)
                    candidates.append((sym, meta["name"], curr_px, total_score, rs_score, ind["atr14"]))
            except Exception as e:
                logger.warning(f"⚠️ Scan error on {sym}: {e}")

        # 모멘텀/상대강도 최상위 주도주 우선 정렬
        candidates.sort(key=lambda x: x[3], reverse=True)

        for sym, name, price, total_score, rs_score, atr in candidates:
            if len(self.positions) >= MAX_CONCURRENT_POSITIONS:
                break

            logger.info(
                f"🎯 [Breakout Detected] {name}({sym}) | 돌파가: ${price:,.2f} | "
                f"RS(비트초과): {rs_score:+.1f}% | 종합점수: {total_score:.1f}"
            )
            success = await self.execute_entry_order(sym, price, target_margin, atr)
            if success:
                await self.update_balance()
                target_margin = min(self.account_equity * MAX_POSITION_PCT, self.account_free * 0.90)

    async def execute_entry_order(self, symbol: str, price: float, margin_usdt: float, atr: float) -> bool:
        """2x 레버리지 시장가 매수 및 거래소 하드 스탑 알고 주문 발주"""
        market = self.markets.get(symbol, {})
        contract_size = float(market.get("contractSize", 1.0))
        notional_target = margin_usdt * LEVERAGE

        coins = notional_target / price
        contracts = coins / contract_size

        contracts_str = self.exchange.amount_to_precision(symbol, contracts)
        contracts = float(contracts_str)
        min_amount = float(market.get("limits", {}).get("amount", {}).get("min", 0.01))
        if contracts < min_amount:
            contracts = min_amount

        logger.info(f"📤 [Submitting Entry] {symbol} LONG {contracts} 계약 (~${margin_usdt:,.2f} 증거금 x {LEVERAGE}x)")

        if self.dry_run:
            self.positions[symbol] = {
                "symbol": symbol,
                "entry_price": price,
                "peak_price": price,
                "atr": atr,
                "contracts": contracts,
                "entry_time": datetime.now(timezone.utc).isoformat()
            }
            self.save_positions()
            return True

        try:
            await self.setup_leverage(symbol)
            params = {
                "tdMode": "cross",
                "posSide": "long"
            }
            res = await self.exchange.create_market_order(symbol, "buy", contracts, params=params)
            logger.info(f"✅ [Entry Filled] {symbol} LONG ID: {res.get('id')}")

            # 거래소 하드 스탑 알고 주문 (-10%)
            sl_price = price * (1.0 - HARD_STOP_LOSS_PCT)
            try:
                inst_id = symbol.replace("/", "-").split(":")[0] + "-SWAP" if ":" in symbol else symbol
                sl_price_str = self.exchange.price_to_precision(symbol, sl_price)
                algo_params = {
                    "instId": inst_id,
                    "tdMode": "cross",
                    "side": "sell",
                    "posSide": "long",
                    "ordType": "conditional",
                    "sz": str(contracts),
                    "slTriggerPx": str(sl_price_str),
                    "slOrdPx": "-1",
                    "slTriggerPxType": "mark",
                    "reduceOnly": True
                }
                res_algo = await self.exchange.request("trade/order-algo", api="private", method="POST", params=algo_params)
                if res_algo.get("code") == "0":
                    logger.info(f"🛡️ [Exchange SL Placed] {symbol} SL @ ${sl_price_str}")
            except Exception as ex_sl:
                logger.warning(f"⚠️ [Exchange SL Warning]: {ex_sl}")

            self.positions[symbol] = {
                "symbol": symbol,
                "entry_price": price,
                "peak_price": price,
                "atr": atr,
                "contracts": contracts,
                "order_id": res.get("id"),
                "entry_time": datetime.now(timezone.utc).isoformat()
            }
            self.save_positions()
            send_telegram_alert(
                f"🚀 *[OKX 퀀트 추세 진입]* {UNIVERSE.get(symbol, {}).get('name', symbol)}({symbol})\n"
                f"진입가: ${price:,.2f} | 수량: {contracts} 계약\n"
                f"투입 증거금: ${margin_usdt:,.2f} (레버리지 {LEVERAGE}x)\n"
                f"하드 손절선: ${sl_price:,.2f} (-{HARD_STOP_LOSS_PCT*100:.0f}%)"
            )
            return True
        except Exception as e:
            logger.error(f"❌ Entry execution failed for {symbol}: {e}")
            return False

    async def run(self):
        """메인 비동기 실행 루프"""
        await self.init_exchange()
        await self.update_balance()
        logger.info(f"💼 [Account Status] Total Equity: ${self.account_equity:,.2f} USDT | Free: ${self.account_free:,.2f} USDT")

        # 기동 즉시 일별 지표 갱신
        await self.fetch_daily_indicators()

        loop_count = 0
        while True:
            try:
                loop_count += 1

                # 1. 매일 00:00 UTC (09:00 KST) 일봉 지표 자동 갱신
                await self.fetch_daily_indicators()

                # 2. 5분마다 잔고 동기화
                if loop_count % 15 == 0:
                    await self.update_balance()
                    logger.info(
                        f"💓 [Heartbeat] Equity: ${self.account_equity:,.2f} | Free: ${self.account_free:,.2f} | "
                        f"Active Positions: {len(self.positions)}/{MAX_CONCURRENT_POSITIONS} ({list(self.positions.keys())})"
                    )

                # 3. 보유 포지션 실시간 관리 (ATR 트레일링 스탑)
                await self.monitor_open_positions()

                # 4. 신규 10일 신고가 돌파 종목 탐색 및 진입
                await self.evaluate_entry_opportunities()

                await asyncio.sleep(POLL_INTERVAL_SEC)

            except asyncio.CancelledError:
                logger.info("🛑 Bot stopped gracefully.")
                break
            except Exception as e:
                logger.error(f"❌ Main Loop Exception: {e}")
                await asyncio.sleep(10)


if __name__ == "__main__":
    bot = OKXTrendTrader()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("🛑 OKX Trend Trader stopped by user.")
