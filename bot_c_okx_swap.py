# -*- coding: utf-8 -*-
"""
bot_c_okx_swap.py — Bot C: OKX Futures 실제 주문 엔진 v2.0
- ccxt async로 실제 OKX API 호출
- Hedge Mode: Long/Short 동시 포지션
- 시장가 주문 (Market Order)
- 포트: 8003
"""
import asyncio
import os
import sys
import time
import logging
from aiohttp import web
from dotenv import load_dotenv
from utils_telegram import send_telegram_alert

try:
    import ccxt.async_support as ccxt_async
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt", "-q"])
    import ccxt.async_support as ccxt_async

from webhook_spec import WebhookPayload, SideType

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - [BOT_C_OKX] %(message)s',
    handlers=[
        logging.FileHandler("bot_c_okx_swap.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("BotC_OKX")

API_KEY    = os.getenv("OKX_API_KEY", "")
API_SECRET = os.getenv("OKX_SECRET", "") or os.getenv("OKX_API_SECRET", "")
PASSPHRASE = os.getenv("OKX_PASSPHRASE", "") or os.getenv("OKX_PASSWORD", "")

if not all([API_KEY, API_SECRET, PASSPHRASE]):
    logger.critical("❌ OKX 자격증명 누락 — .env에 OKX_API_KEY / OKX_SECRET / OKX_PASSPHRASE 필요")
else:
    logger.info(f"🔑 OKX API 로드 완료: [{API_KEY[:4]}****{API_KEY[-4:]}]")


class BotCOKXSwap:
    def __init__(self):
        self._lock = None
        self.exchange = None

    async def init(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
        self.exchange = ccxt_async.okx({
            "apiKey":    API_KEY,
            "secret":    API_SECRET,
            "password":  PASSPHRASE,
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",
            },
        })
        logger.info("✅ OKX ccxt 엔진 초기화 완료")

    async def close(self):
        if self.exchange:
            await self.exchange.close()

    async def execute_order(self, payload: WebhookPayload):
        """실제 OKX 시장가 주문"""
        if not self.exchange or not self._lock:
            logger.error("❌ exchange 미초기화")
            return

        async with self._lock:
            symbol = payload.symbol  # e.g. "BTC-USDT-SWAP" → ccxt용 "BTC/USDT:USDT"
            # OKX ccxt 심볼 변환: "BTC-USDT-SWAP" → "BTC/USDT:USDT"
            ccxt_symbol = symbol.replace("-SWAP", "").replace("-", "/", 1)
            if ":" not in ccxt_symbol:
                ccxt_symbol = ccxt_symbol + ":USDT"

            side = "buy" if payload.side == SideType.BUY else "sell"
            amount = payload.qty  # 계약 수량

            # 청산 처리
            if payload.side == SideType.CLOSE_LONG:
                side = "sell"
                params = {"reduceOnly": True, "posSide": "long"}
            elif payload.side == SideType.CLOSE_SHORT:
                side = "buy"
                params = {"reduceOnly": True, "posSide": "short"}
            elif payload.side == SideType.BUY:
                params = {"posSide": "long"}
            else:  # SELL
                params = {"posSide": "short"}

            # [수정] 주문 전 레버리지 자동 세팅 (환경변수 OKX_LEVERAGE 적용)
            try:
                leverage = int(os.getenv("OKX_LEVERAGE", "10"))
                await self.exchange.set_leverage(leverage, ccxt_symbol, {"mgnMode": "cross"})
                logger.info(f"⚙️ [레버리지 설정] {ccxt_symbol} -> {leverage}x (Cross)")
            except Exception as e:
                logger.warning(f"⚠️ [레버리지 설정 실패] (이미 설정되어 있거나 API 제한일 수 있음): {e}")

            max_retries = 3
            last_err = ""
            for attempt in range(max_retries):
                try:
                    start = time.time()
                    order = await asyncio.wait_for(
                        self.exchange.create_market_order(
                            ccxt_symbol, side, amount, params=params
                        ),
                        timeout=10.0,
                    )
                    latency = time.time() - start
                    order_id = order.get("id", "N/A")
                    avg_price = order.get("average") or order.get("price", 0)
                    logger.info(
                        f"✅ [실주문 성공] {side.upper()} {amount} {ccxt_symbol} "
                        f"@ {avg_price} | ID: {order_id} | Latency: {latency:.3f}s"
                    )
                    return order
                except asyncio.TimeoutError:
                    last_err = f"타임아웃 (시도 {attempt+1}/{max_retries})"
                    logger.error(f"⚠️ OKX 주문 {last_err}")
                except Exception as e:
                    last_err = str(e)
                    logger.error(f"⚠️ OKX API 에러 (시도 {attempt+1}/{max_retries}): {e}")
                    if "price limit" in last_err.lower() or "limit mechanism" in last_err.lower():
                        logger.critical(
                            f"🚨 [PRICE_LIMIT] {side.upper()} {amount} {ccxt_symbol} 주문이 가격제한으로 차단됨: {last_err}"
                        )
                        try:
                            send_telegram_alert(
                                f"🚨 [Bot C] OKX price limit 차단: {side.upper()} {amount} {ccxt_symbol} | {last_err}"
                            )
                        except Exception:
                            pass
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))

            logger.critical(f"🚨 [FATAL] {max_retries}회 시도 모두 실패. 주문 폐기. last_err={last_err}")
            try:
                send_telegram_alert(f"🚨 [FATAL] Bot C (OKX) 주문 {max_retries}회 실패: {payload.side.value} {symbol} | {last_err}")
            except Exception:
                pass


# ── PID LOCK ──
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PID_FILE = os.path.join(_BASE_DIR, "bot_c_okx_swap.pid")


def acquire_pid_lock():
    """PID 락 획득 (중복 실행 방지 + stale pidfile 자동 정리)"""
    try:
        if os.path.exists(_PID_FILE):
            with open(_PID_FILE) as f:
                old_pid_raw = f.read().strip()
            if old_pid_raw:
                old_pid = int(old_pid_raw)
                if psutil.pid_exists(old_pid):
                    logger.critical(f"🔴 Already running bot_c_okx_swap ({old_pid}) → ABORT")
                    sys.exit(1)
                else:
                    logger.warning(f"⚠️ stale pidfile 발견: {old_pid} → 정지한 것으로 추정, pidfile 삭제")
                    os.remove(_PID_FILE)
    except Exception:
        pass
    with open(_PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    logger.info(f"✅ PID 락 획득: {os.getpid()}")


# ── PID LOCK cleanup helper ──
def _cleanup_pid_lock():
    try:
        if os.path.exists(_PID_FILE):
            os.remove(_PID_FILE)
    except Exception:
        pass


bot = BotCOKXSwap()


async def handle_webhook(request):
    try:
        data = await request.text()
        payload = WebhookPayload.from_json(data)
        logger.info(f"📥 Webhook 수신: {payload.side.value} {payload.qty} {payload.symbol}")
        asyncio.create_task(bot.execute_order(payload))
        return web.json_response({"status": "ok"})
    except Exception as e:
        logger.error(f"Webhook 파싱 오류: {e}")
        try:
            send_telegram_alert(f"⚠️ [Error] Bot C (OKX) Webhook 에러: {e}")
        except Exception:
            pass
        return web.json_response({"error": str(e)}, status=400)


async def handle_health(request):
    balance_info = "N/A"
    try:
        if bot.exchange:
            bal = await bot.exchange.fetch_balance()
            usdt = bal.get("USDT", {}).get("free", 0)
            balance_info = f"{usdt:.2f} USDT"
    except Exception:
        pass
    return web.json_response({
        "status": "ok",
        "bot": "Bot_C_OKX_Swap",
        "balance": balance_info,
        "timestamp": time.time()
    })


async def handle_status(request):
    balance_info = "N/A"
    try:
        if bot.exchange:
            bal = await bot.exchange.fetch_balance()
            usdt = bal.get("USDT", {}).get("free", 0)
            balance_info = f"{usdt:.2f} USDT"
    except Exception as e:
        balance_info = f"Error: {e}"
    return web.json_response({
        "bot_id": "BOT_C_OKX_SWAP",
        "status": "active",
        "is_running": True,
        "balance": balance_info,
        "timestamp": time.time()
    })


async def main():
    await bot.init()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8013
    app = web.Application()
    app.router.add_post("/webhook", handle_webhook)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/status", handle_status)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    
    for attempt in range(5):
        try:
            await site.start()
            break
        except OSError as e:
            if attempt == 4:
                raise e
            logger.warning(f"⚠️  Port {port} 사용 중, 2초 후 재시도... ({attempt+1}/5)")
            await asyncio.sleep(2)

    logger.info(f"🚀 Bot C (OKX Swap 실주문) 시작 — 포트 {port}")
    try:
        await asyncio.Event().wait()
    finally:
        await bot.close()
        await runner.cleanup()


if __name__ == "__main__":
    try:
        acquire_pid_lock()
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 프로그램 종료")
    except SystemExit as e:
        _cleanup_pid_lock()
        sys.exit(e.code)
    finally:
        _cleanup_pid_lock()
