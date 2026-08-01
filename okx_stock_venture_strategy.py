#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
okx_stock_venture_strategy.py — OKX Stock Token Dedicated Auto Trading Engine (VENTURE/MEME)
- Trades OKX US Stock Tokens (SPACEX, OPENAI, RDDT, etc.)
- Strict US Market time filtering
- Supertrend + StochRSI logic (15m timeframe)
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

try:
    from webhook_spec import WebhookPayload, ActionType, SideType
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    sys.exit(1)

# ── LOGGING ──
logger = logging.getLogger("OKX_Stock_Venture_Strategy")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(BASE_DIR, "okx_stock_venture_strategy.log"), encoding="utf-8")
fh.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))
logger.addHandler(fh)
logger.addHandler(ch)

# ── CONFIG ──
MASTER_WEBHOOK_URL = "http://localhost:8009/webhook"
AUTO_TRADE_INTERVAL = 60.0

API_KEY    = os.getenv("OKX_API_KEY", "")
API_SECRET = os.getenv("OKX_SECRET", "") or os.getenv("OKX_API_SECRET", "")
PASSPHRASE = os.getenv("OKX_PASSPHRASE", "") or os.getenv("OKX_PASSWORD", "")


# ── TA Helpers ──
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


class OKXStockVentureStrategyBrain:
    def __init__(self):
        self.session = None
        self.exchange = None
        self.auto_active_pos = {}
        self.dca_state = {}

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
    # [Stock Token Logic]
    # =====================================================================
    def is_us_market_active(self):
        # US Market active time filter
        ny_tz = pytz.timezone('America/New_York')
        ny_now = datetime.now(ny_tz)
        return ny_now.weekday() < 5 and ((ny_now.hour == 9 and ny_now.minute >= 30) or (10 <= ny_now.hour < 16))

    async def get_target_symbols(self):
        try:
            tickers = await self.exchange.fetch_tickers()
            markets = await self.exchange.load_markets()
            # Track B (15m): 벤처/신규 상장 토큰 전용
            stock_keywords = ['SPACEX', 'OPENAI', 'ANTHROPIC', 'RDDT', 'XSTOCKS']
            blacklist = ['KR200', 'SKHYNIX']
            
            data = []
            for s, t in tickers.items():
                if s in markets and markets[s].get('swap') and 'USDT' in s:
                    if any(k in s for k in stock_keywords) and not any(b in s for b in blacklist):
                        data.append({'symbol': s, 'vol': t.get('quoteVolume', 0)})
            
            df = pd.DataFrame(data).sort_values(by='vol', ascending=False)
            if df.empty: return []
            return df['symbol'].tolist()
        except Exception as e:
            logger.error(f"⚠️ [Stock Trade] 심볼 로드 실패: {e}")
            return []

    async def check_auto_logic(self, symbol):
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, '15m', limit=100)
            if not ohlcv or len(ohlcv) < 15:
                return
            df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
            
            st_d, st_v = calc_supertrend(df, 10, 3.0)
            df['st_d'] = st_d
            df['st_v'] = st_v
            
            k, d = calc_stoch_rsi(df['c'], 14, 3, 3)
            df['stoch_k'] = k
            df['stoch_d'] = d
            
            prev, curr = df.iloc[-2], df.iloc[-1]
            t_curr = curr['t']
            
            dca = self.dca_state.setdefault(symbol, {'entry_count': 0, 'exit_count': 0, 'last_entry_t': 0, 'last_exit_t': 0})
            
            has_long = (symbol, 'long') in self.auto_active_pos
            has_short = (symbol, 'short') in self.auto_active_pos
            
            is_long_breakout = prev['st_d'] == -1 and curr['st_d'] == 1
            is_long_pullback = curr['st_d'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
            is_long_sig = is_long_breakout or is_long_pullback
            
            is_short_breakout = prev['st_d'] == 1 and curr['st_d'] == -1
            is_short_pullback = curr['st_d'] == -1 and prev['stoch_k'] > 80 and curr['stoch_k'] <= 80
            is_short_sig = is_short_breakout or is_short_pullback
            
            close_long_sig = curr['st_d'] == -1 or curr['c'] < curr['st_v']
            close_short_sig = curr['st_d'] == 1 or curr['c'] > curr['st_v']

            if has_long:
                if close_long_sig:
                    if dca['exit_count'] < 8 and dca.get('last_exit_t') != t_curr:
                        qty = self.auto_active_pos[(symbol, 'long')]['size']
                        sell_qty = qty / (8 - dca['exit_count'])
                        
                        if market_info:
                            min_amount = market_info.get('limits', {}).get('amount', {}).get('min', 0)
                            if min_amount and sell_qty < min_amount:
                                sell_qty = min_amount
                        if sell_qty >= qty:
                            sell_qty = 0
                        else:
                            sell_qty = float(self.exchange.amount_to_precision(symbol, sell_qty))
                        
                        if sell_qty >= 0:
                            logger.info(f"💨 [DCA Trade] 롱 분할 청산 ({dca['exit_count']+1}/8): {symbol} (수량: {sell_qty if sell_qty > 0 else 'ALL'})")
                            await self.send_webhook(SideType.CLOSE_LONG, symbol, sell_qty)
                        dca['exit_count'] += 1
                        dca['last_exit_t'] = t_curr
                        if dca['exit_count'] >= 8:
                            dca['entry_count'] = 0
                            dca['exit_count'] = 0
                else:
                    if dca['entry_count'] < 8 and dca.get('last_entry_t') != t_curr:
                        logger.info(f"🔥 [DCA Trade] 롱 분할 진입 ({dca['entry_count']+1}/8): {symbol}")
                        await self.execute_auto_entry(symbol, SideType.BUY, portion=(0.05 / 8.0))
                        dca['entry_count'] += 1
                        dca['last_entry_t'] = t_curr
                        
            elif has_short:
                if close_short_sig:
                    if dca['exit_count'] < 8 and dca.get('last_exit_t') != t_curr:
                        qty = self.auto_active_pos[(symbol, 'short')]['size']
                        sell_qty = qty / (8 - dca['exit_count'])
                        
                        if market_info:
                            min_amount = market_info.get('limits', {}).get('amount', {}).get('min', 0)
                            if min_amount and sell_qty < min_amount:
                                sell_qty = min_amount
                        if sell_qty >= qty:
                            sell_qty = 0
                        else:
                            sell_qty = float(self.exchange.amount_to_precision(symbol, sell_qty))
                        
                        if sell_qty >= 0:
                            logger.info(f"💨 [DCA Trade] 숏 분할 청산 ({dca['exit_count']+1}/8): {symbol} (수량: {sell_qty if sell_qty > 0 else 'ALL'})")
                            await self.send_webhook(SideType.CLOSE_SHORT, symbol, sell_qty)
                        dca['exit_count'] += 1
                        dca['last_exit_t'] = t_curr
                        if dca['exit_count'] >= 8:
                            dca['entry_count'] = 0
                            dca['exit_count'] = 0
                else:
                    if dca['entry_count'] < 8 and dca.get('last_entry_t') != t_curr:
                        logger.info(f"🔥 [DCA Trade] 숏 분할 진입 ({dca['entry_count']+1}/8): {symbol}")
                        await self.execute_auto_entry(symbol, SideType.SELL, portion=(0.05 / 8.0))
                        dca['entry_count'] += 1
                        dca['last_entry_t'] = t_curr
                        
            else:
                if is_long_sig and dca.get('last_entry_t') != t_curr:
                    logger.info(f"🔥 [DCA Trade] 롱 분할 최초 진입 (1/8): {symbol}")
                    await self.execute_auto_entry(symbol, SideType.BUY, portion=(0.05 / 8.0))
                    dca['entry_count'] = 1
                    dca['exit_count'] = 0
                    dca['last_entry_t'] = t_curr
                elif is_short_sig and dca.get('last_entry_t') != t_curr:
                    logger.info(f"🔥 [DCA Trade] 숏 분할 최초 진입 (1/8): {symbol}")
                    await self.execute_auto_entry(symbol, SideType.SELL, portion=(0.05 / 8.0))
                    dca['entry_count'] = 1
                    dca['exit_count'] = 0
                    dca['last_entry_t'] = t_curr
                    
        except Exception as e:
            logger.error(f"⚠️ [Stock Trade] 로직 체크 실패 ({symbol}): {e}")

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
                    
                logger.info(f"🔥 [Stock Trade] 신규 진입 시그널: {side.value} {symbol} (수량: {amount})")
                await self.send_webhook(side, symbol, amount)
        except Exception as e:
            logger.error(f"⚠️ [Stock Trade] 진입 수량 계산 실패 ({symbol}): {e}")

    async def run_auto_trade_loop(self):
        logger.info("🚀 [Stock Trade] 미국 주식 토큰 자동매매 엔진 시작 (Supertrend + StochRSI)")
        
        while True:
            try:
                # 24시간 가동을 위해 미 본장 시간 제한 해제 (주식 연계 토큰도 24시간 변동성 활용)
                # if not self.is_us_market_active():
                #     logger.info("⏳ [Stock Trade] 미 본장(US Market) 휴장/야간 대기중...")
                #     await asyncio.sleep(60)
                #     continue

                symbols = await self.get_target_symbols()
                if not symbols:
                    await asyncio.sleep(60)
                    continue
                
                positions = await self.exchange.fetch_positions()
                self.auto_active_pos = {}
                for p in positions:
                    if float(p.get('contracts', 0)) > 0:
                        sym = p.get('symbol')
                        s = p.get('side')
                        self.auto_active_pos[(sym, s)] = {'size': float(p['contracts'])}

                for symbol in symbols:
                    await self.check_auto_logic(symbol)
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ [Stock Trade] 예외 발생: {e}")
            
            await asyncio.sleep(AUTO_TRADE_INTERVAL)

    async def run_all(self):
        await self.init_session()
        try:
            await self.run_auto_trade_loop()
        finally:
            await self.close_session()

if __name__ == "__main__":
    brain = OKXStockVentureStrategyBrain()
    try:
        asyncio.run(brain.run_all())
    except KeyboardInterrupt:
        logger.info("🛑 프로그램 종료")
