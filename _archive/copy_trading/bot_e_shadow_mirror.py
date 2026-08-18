#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_e_shadow_mirror.py — Bot E: Shadow Mirror Bot
세계 1등 리드 트레이더의 카피 서브포지션(진입/청산)을 완벽하게 복제하여,
내 수동 계정(Bot C)으로 정밀한 수량(목표 USDT 기반)과 함께 미러링 주문을 쏘는 전용 데몬입니다.
"""

import asyncio
import json
import os
import time
import logging
import logging.handlers
import uuid
import sys
import psutil
from dotenv import load_dotenv

import ccxt.async_support as ccxt_async
import aiohttp
from okx_copy_engine import OKXCopyEngine
from webhook_spec import WebhookPayload, ActionType, SideType, sign_payload, WEBHOOK_SIGNATURE_HEADER

load_dotenv(override=True)

# ── 설정 변수 ──
POLL_INTERVAL = 3.0  # 감시 주기(초)
SHADOW_TRADE_USDT = float(os.getenv("OKX_SHADOW_USDT", "100"))  # 1회 진입 목표 금액 (USDT)
LEVERAGE = float(os.getenv("OKX_LEVERAGE", "10"))
WEBHOOK_URL = "http://localhost:8009/webhook"  # Master → Bot C (포지션 제한 통과)

API_KEY = os.getenv("OKX_API_KEY", "")
API_SECRET = os.getenv("OKX_SECRET", "") or os.getenv("OKX_API_SECRET", "")
PASSPHRASE = os.getenv("OKX_PASSPHRASE", "") or os.getenv("OKX_PASSWORD", "")

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - [BOT_E_MIRROR] %(message)s',
    handlers=[
        # systemd가 stdout/stderr를 같은 파일로 append하므로 StreamHandler 제거 (중복 방지)
        logging.handlers.RotatingFileHandler(
            "bot_e_shadow_mirror.log", encoding="utf-8",
            maxBytes=10*1024*1024, backupCount=3,
        ),
    ]
)
logger = logging.getLogger("BotE_Mirror")

class ShadowMirrorBot:
    def __init__(self):
        self.copy_engine = OKXCopyEngine()
        self.exchange = None
        self.known_positions = {}
        self.session = None
        
    async def init(self):
        self.exchange = ccxt_async.okx({
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "password": PASSPHRASE,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"}
        })
        await self.exchange.load_markets()
        self.session = aiohttp.ClientSession()
        logger.info(f"✅ [Shadow Mirror] 초기화 완료 (1회 진입목표: {SHADOW_TRADE_USDT} USDT | 레버리지: {LEVERAGE}x)")
        
    async def close(self):
        if self.exchange:
            await self.exchange.close()
        if self.session:
            await self.session.close()

    async def calculate_contract_qty(self, symbol: str) -> float:
        """
        주어진 symbol (예: BTC-USDT-SWAP)에 대해
        SHADOW_TRADE_USDT * LEVERAGE 만큼의 포지션을 잡기 위한 계약(Contract) 수를 계산합니다.
        """
        try:
            ccxt_symbol = symbol.replace("-SWAP", "").replace("-", "/", 1)
            if ":" not in ccxt_symbol:
                ccxt_symbol += ":USDT"
                
            market = self.exchange.market(ccxt_symbol)
            contract_size = market.get('contractSize', 1.0)
            
            ticker = await self.exchange.fetch_ticker(ccxt_symbol)
            price = ticker.get('last')
            
            if not price or price <= 0:
                logger.error(f"❌ {symbol} 현재가 조회 실패")
                return 1.0
                
            balance = await self.exchange.fetch_balance()
            free_usdt = balance.get('USDT', {}).get('free', 0)
            if not free_usdt:
                free_usdt = balance.get('free', {}).get('USDT', 0)
            
            # 필요 계약 수 = (가용 잔고의 40% * 레버리지) / (현재가 * 계약단위)
            target_usdt = free_usdt * 0.40
            target_notional = target_usdt * LEVERAGE
            qty = target_notional / (price * contract_size)
            
            # 소수점 처리 (계약 수는 정수여야 하는 경우가 많음)
            qty = round(qty)
            if qty < 1.0:
                qty = 1.0
                
            logger.info(f"🧮 [수량 계산] {symbol} | 현재가: {price} | 계약단위: {contract_size} | 산출 계약수: {qty}개")
            return qty
            
        except Exception as e:
            logger.error(f"❌ 수량 계산 중 예외 발생 ({symbol}): {e}")
            return 1.0

    async def send_webhook(self, side: SideType, symbol: str, qty: float):
        payload = WebhookPayload(
            action=ActionType.EXEC,
            signal_id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            qty=qty,
            price=0.0,
            signal_strength="STRONG"
        )
        json_data = json.loads(payload.to_json())
        json_data["market"] = "okx_swap"
        body = json.dumps(json_data)
        headers = {"Content-Type": "application/json", WEBHOOK_SIGNATURE_HEADER: sign_payload(body)}
        try:
            async with self.session.post(WEBHOOK_URL, data=body, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    logger.info(f"🚀 [Webhook 발송 성공] {side.value} | {symbol} | 수량: {qty}")
                else:
                    err = await resp.text()
                    logger.error(f"⚠️ [Webhook 발송 실패] {resp.status} - {err}")
        except Exception as e:
            logger.error(f"❌ [Webhook 예외] {e}")

    async def run(self):
        await self.init()
        
        logger.info("🔎 [Shadow Mirror] 초기 포지션 스냅샷 로드 중...")
        init_positions = self.copy_engine.get_active_subpositions()
        for p in init_positions:
            pid = p.get("subPosId")
            if pid:
                self.known_positions[pid] = p
        logger.info(f"✅ [Shadow Mirror] 초기 활성 카피 포지션 {len(self.known_positions)}개 감지됨 (중복 진입 방지)")

        while True:
            try:
                current_positions = self.copy_engine.get_active_subpositions()
                if not isinstance(current_positions, list):
                    logger.warning("⚠️ 카피 포지션 조회 실패. 다음 주기 대기.")
                    await asyncio.sleep(POLL_INTERVAL)
                    continue

                current_ids = {p.get("subPosId") for p in current_positions if p.get("subPosId")}
                known_ids = set(self.known_positions.keys())

                # 1. 신규 진입 감지 (Entry)
                new_ids = current_ids - known_ids
                for p in current_positions:
                    pid = p.get("subPosId")
                    if pid in new_ids:
                        inst_id = p.get("instId", "UNKNOWN")
                        pos_side = p.get("posSide", "long").lower()
                        logger.info(f"🔥 [진입 감지] 리드 트레이더 신규 매매 포착! {inst_id} ({pos_side}) | ID: {pid}")
                        
                        side_enum = SideType.BUY if pos_side == "long" else SideType.SELL
                        qty = await self.calculate_contract_qty(inst_id)
                        
                        await self.send_webhook(side_enum, inst_id, qty)
                        self.known_positions[pid] = p

                # 2. 청산 감지 (Exit)
                closed_ids = known_ids - current_ids
                for pid in closed_ids:
                    old_p = self.known_positions[pid]
                    inst_id = old_p.get("instId", "UNKNOWN")
                    pos_side = old_p.get("posSide", "long").lower()
                    logger.info(f"💨 [청산 감지] 리드 트레이더 포지션 청산 포착! {inst_id} ({pos_side}) | ID: {pid}")
                    
                    close_side = SideType.CLOSE_LONG if pos_side == "long" else SideType.CLOSE_SHORT
                    # Bot C의 전액 청산 트리거: qty=0
                    await self.send_webhook(close_side, inst_id, 0.0)
                    del self.known_positions[pid]

            except Exception as e:
                logger.error(f"❌ [Shadow Mirror] 루프 예외 발생: {e}")
            
            await asyncio.sleep(POLL_INTERVAL)

# ── PID LOCK ──
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PID_FILE = os.path.join(_BASE_DIR, "bot_e_shadow_mirror.pid")

def acquire_pid_lock():
    try:
        if os.path.exists(_PID_FILE):
            with open(_PID_FILE) as f:
                old_pid_raw = f.read().strip()
            if old_pid_raw:
                old_pid = int(old_pid_raw)
                if psutil.pid_exists(old_pid):
                    logger.critical(f"🔴 Already running bot_e_shadow_mirror ({old_pid}) → ABORT")
                    sys.exit(1)
                else:
                    os.remove(_PID_FILE)
    except Exception:
        pass
    with open(_PID_FILE, "w") as f:
        f.write(str(os.getpid()))

def _cleanup_pid_lock():
    try:
        if os.path.exists(_PID_FILE):
            os.remove(_PID_FILE)
    except Exception:
        pass

if __name__ == "__main__":
    try:
        acquire_pid_lock()
        bot = ShadowMirrorBot()
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("🛑 프로그램 종료")
    except SystemExit as e:
        _cleanup_pid_lock()
        sys.exit(e.code)
    finally:
        _cleanup_pid_lock()
