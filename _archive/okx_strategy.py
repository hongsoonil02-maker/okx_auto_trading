#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
okx_strategy.py — OKX Hybrid Strategy Engine (The Brain)
- Task 1: Shadow Copy Engine (1등 리드 트레이더의 카피 서브포지션 오픈/클로즈 모니터링)
- Task 2: Auto Trading Engine (거래대금 상위 코인 Supertrend + RSI 스윙/단타 자동매매)
"""

import os
import sys
import time
import json
import logging
import asyncio
import aiohttp
from datetime import datetime
import pytz
from dotenv import load_dotenv
import pandas as pd
import ccxt.async_support as ccxt_async

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ── 로컬 모듈 임포트 ──
try:
    from okx_copy_engine import OKXCopyEngine
    from webhook_spec import WebhookPayload, ActionType, SideType
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    sys.exit(1)

# ── LOGGING ──
logger = logging.getLogger("OKX_Hybrid_Strategy_Brain")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(BASE_DIR, "okx_strategy.log"), encoding="utf-8")
fh.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))
logger.addHandler(fh)
logger.addHandler(ch)

# ── CONFIG ──
MASTER_WEBHOOK_URL = "http://localhost:8009/webhook"
POLL_INTERVAL = 3.0  # 섀도우 카피 3초 주기
AUTO_TRADE_INTERVAL = 60.0  # 자동 매매 60초 주기
DEFAULT_QTY = float(os.getenv("OKX_SHADOW_COPY_QTY", "1"))

API_KEY    = os.getenv("OKX_API_KEY", "")
API_SECRET = os.getenv("OKX_SECRET", "") or os.getenv("OKX_API_SECRET", "")
PASSPHRASE = os.getenv("OKX_PASSPHRASE", "") or os.getenv("OKX_PASSWORD", "")


# ── TA Helpers (No external dependencies) ──
def calc_supertrend(df, period=10, multiplier=3.0):
    hl2 = (df['h'] + df['l']) / 2
    atr = (df['h'].combine(df['c'].shift(), max) - df['l'].combine(df['c'].shift(), min)).rolling(period).mean()
    
    final_upperband = hl2 + (multiplier * atr)
    final_lowerband = hl2 - (multiplier * atr)
    
    st_dir = pd.Series(1, index=df.index, dtype='int')
    st_val = pd.Series(0.0, index=df.index, dtype='float64')
    
    for i in range(period, len(df)):
        if df['c'].iloc[i] > final_upperband.iloc[i-1]:
            st_dir.iloc[i] = 1
        elif df['c'].iloc[i] < final_lowerband.iloc[i-1]:
            st_dir.iloc[i] = -1
        else:
            st_dir.iloc[i] = st_dir.iloc[i-1]
            if st_dir.iloc[i] == 1 and final_lowerband.iloc[i] < final_lowerband.iloc[i-1]:
                final_lowerband.iloc[i] = final_lowerband.iloc[i-1]
            if st_dir.iloc[i] == -1 and final_upperband.iloc[i] > final_upperband.iloc[i-1]:
                final_upperband.iloc[i] = final_upperband.iloc[i-1]
                
        if st_dir.iloc[i] == 1:
            st_val.iloc[i] = final_lowerband.iloc[i]
        else:
            st_val.iloc[i] = final_upperband.iloc[i]
            
    return st_dir, st_val

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_stoch_rsi(series, period=14, smooth_k=3, smooth_d=3):
    rsi = calc_rsi(series, period)
    stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    k = stoch_rsi.rolling(smooth_k).mean() * 100
    d = k.rolling(smooth_d).mean()
    return k, d


class OKXHybridStrategyBrain:
    def __init__(self):
        self.copy_engine = OKXCopyEngine()
        self.known_positions = {}  # { subPosId : dict_data }
        self.session = None
        self.exchange = None
        
        # 자동매매 상태 관리
        self.auto_active_pos = {}  # {(symbol, side): {'size': float, 'avgPrice': float}}
        self.max_price_state = {}  # {(symbol, side): float}

    async def init_session(self):
        self.session = aiohttp.ClientSession()
        self.exchange = ccxt_async.okx({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "password": PASSPHRASE,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"}
        })

    async def close_session(self):
        if self.session:
            await self.session.close()
        if self.exchange:
            await self.exchange.close()

    async def send_webhook(self, side: SideType, symbol: str, qty: float):
        payload = WebhookPayload(
            action=ActionType.EXEC,
            side=side,
            symbol=symbol,
            qty=qty,
            signal_strength="STRONG"
        )
        try:
            json_data = json.loads(payload.to_json())
            json_data["market"] = "okx_swap"
            async with self.session.post(
                MASTER_WEBHOOK_URL,
                json=json_data,
                timeout=5
            ) as resp:
                if resp.status in [200, 201]:
                    logger.info(f"✅ Webhook 발송 완료: {side.value} {qty} {symbol}")
                else:
                    logger.error(f"❌ Webhook 발송 실패: {resp.status} - {await resp.text()}")
        except Exception as e:
            logger.error(f"❌ Webhook 전송 예외: {e}")

    # =====================================================================
    # [Task 1: Shadow Copy Logic]
    # =====================================================================
    async def run_shadow_copy_loop(self):
        logger.info("🚫 [Shadow Copy] 엔진이 비활성화되었습니다. (OKX 공식 카피 트레이딩 기능에 전적으로 위임)")
        while True:
            await asyncio.sleep(3600)

    # =====================================================================
    # [Task 2: Auto Trading Logic (Supertrend + RSI)]
    # =====================================================================
    def is_us_market_active(self):
        ny_tz = pytz.timezone('America/New_York')
        ny_now = datetime.now(ny_tz)
        return ny_now.weekday() < 5 and ((ny_now.hour == 9 and ny_now.minute >= 30) or (10 <= ny_now.hour < 16))

    async def get_target_symbols(self):
        try:
            tickers = await self.exchange.fetch_tickers()
            markets = await self.exchange.load_markets()
            
            # Track A (1h): 메이저 코인 전용
            majors = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'AVAX', 'LINK', 'DOT', 'BNB', 'TRX']
            blacklist = ['KR200', 'SKHYNIX']
            
            data = []
            for s, t in tickers.items():
                if s in markets and markets[s].get('swap') and 'USDT' in s:
                    # Check if symbol base is in majors (e.g. BTC-USDT-SWAP -> BTC)
                    base_coin = s.split('-')[0]
                    if base_coin in majors and not any(b in s for b in blacklist):
                        data.append({'symbol': s, 'vol': t.get('quoteVolume', 0)})
            
            df = pd.DataFrame(data).sort_values(by='vol', ascending=False)
            if df.empty: return []
            return df['symbol'].tolist()
        except Exception as e:
            logger.error(f"⚠️ [Auto Trade] 심볼 로드 실패: {e}")
            return []

    async def check_auto_logic(self, symbol):
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, '1h', limit=300)
            if not ohlcv or len(ohlcv) < 30:
                return
            df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
            
            
            # 진입용 (Loose) Supertrend (Multiplier 4.0)
            st_d_loose, st_v_loose = calc_supertrend(df, 10, 4.0)
            # 청산/방어용 (Tight) Supertrend (Multiplier 2.5)
            st_d_tight, st_v_tight = calc_supertrend(df, 10, 2.5)
            
            df['st_d_loose'] = st_d_loose
            df['st_v_loose'] = st_v_loose
            df['st_d_tight'] = st_d_tight
            df['st_v_tight'] = st_v_tight
            
            # Volume MA 20
            df['vol_ma'] = df['v'].rolling(20).mean()
            
            # Volume MA 20
            df['vol_ma'] = df['v'].rolling(20).mean()
            
            # StochRSI 적용 (K, D선)
            k, d = calc_stoch_rsi(df['c'], 14, 3, 3)
            df['stoch_k'] = k
            df['stoch_d'] = d
            
            prev, curr = df.iloc[-2], df.iloc[-1]
            
            pos_long = self.auto_active_pos.get((symbol, 'long'))
            has_long = pos_long is not None
            avg_price_long = pos_long['avgPrice'] if has_long else 0
            if has_long and avg_price_long > 0:
                self.max_price_state[(symbol, 'long')] = max(self.max_price_state.get((symbol, 'long'), avg_price_long), curr['c'])
                
            pos_short = self.auto_active_pos.get((symbol, 'short'))
            has_short = pos_short is not None
            avg_price_short = pos_short['avgPrice'] if has_short else 0
            if has_short and avg_price_short > 0:
                self.max_price_state[(symbol, 'short')] = min(self.max_price_state.get((symbol, 'short'), avg_price_short), curr['c'])

            vol_cond = curr['v'] > prev['vol_ma'] * 1.2

            # A. 추적 청산 (Trailing Stop 기반)
            if has_long and avg_price_long > 0:
                is_profit = curr['c'] > avg_price_long * 1.015
                st_v_long = curr['st_v_tight'] if is_profit else curr['st_v_loose']
                st_d_long = curr['st_d_tight'] if is_profit else curr['st_d_loose']
                
                close_long_sig = st_d_long == -1 or curr['c'] < st_v_long
                # Breakeven Stop
                if self.max_price_state.get((symbol, 'long'), avg_price_long) > avg_price_long * 1.02:
                    if curr['c'] < avg_price_long:
                        close_long_sig = True
                        
                if close_long_sig:
                    logger.info(f"💨 [Auto Trade] 롱 청산 시그널 (ALL): {symbol}")
                    await self.send_webhook(SideType.CLOSE_LONG, symbol, 0)
            
            if has_short and avg_price_short > 0:
                is_profit = curr['c'] < avg_price_short * 0.985
                st_v_short = curr['st_v_tight'] if is_profit else curr['st_v_loose']
                st_d_short = curr['st_d_tight'] if is_profit else curr['st_d_loose']
                
                close_short_sig = st_d_short == 1 or curr['c'] > st_v_short
                # Breakeven Stop
                if self.max_price_state.get((symbol, 'short'), avg_price_short) < avg_price_short * 0.98:
                    if curr['c'] > avg_price_short:
                        close_short_sig = True
                        
                if close_short_sig:
                    logger.info(f"💨 [Auto Trade] 숏 청산 시그널 (ALL): {symbol}")
                    await self.send_webhook(SideType.CLOSE_SHORT, symbol, 0)

            # B. 신규 진입 (포션 3% 사용)
            # 1. 돌파 진입 (Supertrend 추세 전환)
            is_long_breakout = prev['st_d_loose'] == -1 and curr['st_d_loose'] == 1 and vol_cond
            is_short_breakout = prev['st_d_loose'] == 1 and curr['st_d_loose'] == -1 and vol_cond

            is_long_pullback = curr['st_d_loose'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20 and vol_cond
            is_short_pullback = curr['st_d_loose'] == -1 and prev['stoch_k'] > 80 and curr['stoch_k'] <= 80 and vol_cond

            if is_long_breakout or is_long_pullback:
                if (symbol, 'long') not in self.auto_active_pos:
                    await self.execute_auto_entry(symbol, SideType.BUY)
            elif is_short_breakout or is_short_pullback:
                if (symbol, 'short') not in self.auto_active_pos:
                    await self.execute_auto_entry(symbol, SideType.SELL)
        except Exception as e:
            logger.error(f"⚠️ [Auto Trade] 로직 체크 실패 ({symbol}): {e}")

    async def execute_auto_entry(self, symbol: str, side: SideType, portion: float = 0.05):
        try:
            balance = await self.exchange.fetch_balance()
            free_usdt = balance.get('USDT', {}).get('free', 0)
            if not free_usdt:
                free_usdt = balance.get('free', {}).get('USDT', 0)
            ticker = await self.exchange.fetch_ticker(symbol)
            price = ticker.get('last')
            
            if not free_usdt or not price:
                return
                
            leverage = int(os.getenv("OKX_LEVERAGE", "10"))
            raw_amount = (free_usdt * portion * leverage) / price
            
            market_info = self.exchange.markets.get(symbol)
            contract_size = market_info.get('contractSize', 1) if market_info else 1
            raw_contracts = raw_amount / float(contract_size)
            
            amount = self.exchange.amount_to_precision(symbol, raw_contracts)
            amount = float(amount)
            
            if amount > 0:
                # 주문 가능 여부 검증 (최소/최대 수량, 증거금 여유)
                min_amount = market_info.get('limits', {}).get('amount', {}).get('min', 0) if market_info else 0
                max_amount = market_info.get('limits', {}).get('market', {}).get('max', 0) if market_info else 0
                
                if min_amount and amount < min_amount:
                    logger.warning(f"⚠️ 진입 수량({amount})이 최소 수량({min_amount}) 미만 (보유 USDT: {free_usdt:.2f})")
                    return
                if max_amount and amount > max_amount:
                    amount = float(self.exchange.amount_to_precision(symbol, max_amount))
                    
                required_margin = (amount * float(contract_size) * price) / leverage
                if required_margin > free_usdt * 0.95:
                    logger.warning(f"⚠️ USDT 증거금 부족 - 필요: {required_margin:.2f}, 보유: {free_usdt:.2f} (수량: {amount})")
                    return
                    
                logger.info(f"🔥 [Auto Trade] 신규 진입 시그널: {side.value} {symbol} (수량: {amount})")
                await self.send_webhook(side, symbol, amount)
        except Exception as e:
            logger.error(f"⚠️ [Auto Trade] 진입 수량 계산 실패 ({symbol}): {e}")

    async def run_auto_trade_loop(self):
        logger.info("🚀 [Auto Trade] 자동매매 엔진 시작 (Supertrend + RSI)")
        
        while True:
            try:
                # 24시간 가동을 위해 미 본장 시간 제한 해제
                # if not self.is_us_market_active():
                #     logger.info("⏳ [Auto Trade] 미 본장(US Market) 휴장/야간 대기중...")
                #     await asyncio.sleep(60)
                #     continue

                symbols = await self.get_target_symbols()
                if not symbols:
                    await asyncio.sleep(60)
                    continue
                
                # 현재 활성 포지션 동기화
                positions = await self.exchange.fetch_positions()
                self.auto_active_pos = {}
                active_keys = set()
                for p in positions:
                    if float(p.get('contracts', 0)) > 0:
                        sym = p.get('symbol')
                        s = p.get('side')
                        active_keys.add((sym, s))
                        self.auto_active_pos[(sym, s)] = {
                            'size': float(p['contracts']),
                            'avgPrice': float(p.get('avgPrice', p.get('price', 0)))
                        }
                
                for k in list(self.max_price_state.keys()):
                    if k not in active_keys:
                        del self.max_price_state[k]

                # 코인별 로직 검사
                for symbol in symbols:
                    await self.check_auto_logic(symbol)
                    await asyncio.sleep(0.1)  # Rate limit 방어
                
            except Exception as e:
                logger.error(f"❌ [Auto Trade] 예외 발생: {e}")
            
            await asyncio.sleep(AUTO_TRADE_INTERVAL)

    # =====================================================================
    # [Main Execution]
    # =====================================================================
    async def run_all(self):
        await self.init_session()
        try:
            # 두 엔진을 병렬 실행
            await asyncio.gather(
                self.run_shadow_copy_loop(),
                self.run_auto_trade_loop()
            )
        finally:
            await self.close_session()

if __name__ == "__main__":
    brain = OKXHybridStrategyBrain()
    try:
        asyncio.run(brain.run_all())
    except KeyboardInterrupt:
        logger.info("🛑 프로그램 종료")
