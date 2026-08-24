# -*- coding: utf-8 -*-
"""
bot_c_okx_swap.py — Bot C: OKX Futures 실제 주문 엔진 v2.0
- ccxt async로 실제 OKX API 호출
- Hedge Mode: Long/Short 동시 포지션
- 시장가 주문 (Market Order)
- 포트: 8013 (기본값, 실행 인자로 조정 가능)
"""
import asyncio
import json
import os
import re
import sys
import time
import uuid
import logging
import logging.handlers
import psutil
from aiohttp import web
from dotenv import load_dotenv
from utils_telegram import send_telegram_alert
from bot_config import BotConfig

try:
    import ccxt.async_support as ccxt_async
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt", "-q"])
    import ccxt.async_support as ccxt_async

from webhook_spec import WebhookPayload, SideType, verify_webhook_signature, WEBHOOK_SIGNATURE_HEADER

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - [BOT_C_OKX] %(message)s',
    handlers=[
        # systemd가 stdout/stderr를 같은 파일로 append하므로 StreamHandler 제거 (중복 방지)
        logging.handlers.RotatingFileHandler(
            "bot_c_okx_swap.log", encoding="utf-8",
            maxBytes=10*1024*1024, backupCount=3,
        ),
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
        self._balance_cache = (0.0, None)  # [Fix] (timestamp, "X.XX USDT") 헬스체크 캐시

    async def get_balance_str(self, max_age: float = 25.0) -> str:
        """[Fix] 잔고 캐시 — 30초 주기 헬스체크의 중복 fetch_balance 방지."""
        ts, cached = self._balance_cache
        if cached is not None and time.time() - ts < max_age:
            return cached
        try:
            bal = await self.exchange.fetch_balance()
            usdt = bal.get("USDT", {}).get("free", 0)
            s = f"{float(usdt):.2f} USDT"
        except Exception:
            return cached if cached is not None else "N/A"
        self._balance_cache = (time.time(), s)
        return s

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
        """실제 OKX 시장가 주문. 결과 dict 반환 (status: ok/failed/rejected)."""
        if not self.exchange or not self._lock:
            logger.error("❌ exchange 미초기화")
            return {"status": "failed", "error": "exchange_not_initialized"}

        async with self._lock:
            symbol = payload.symbol  # e.g. "BTC-USDT-SWAP" → ccxt용 "BTC/USDT:USDT"

            if BotConfig.is_symbol_blacklisted(symbol, market="OKX"):
                logger.warning(f"🚫 [차단] 블랙리스트 종목 주문 거부: {symbol}")
                send_telegram_alert(f"🚫 [Bot C] 블랙리스트 종목 주문 거부: {symbol}")
                return {"status": "rejected", "reason": "blacklist"}

            # ── Last-line defense: reject new entries if max active positions reached ──
            if payload.side in (SideType.BUY, SideType.SELL):
                single_only = os.getenv("OKX_SINGLE_POSITION_ONLY", "true").lower() == "true"
                max_positions = int(os.getenv("OKX_GLOBAL_MAX_POSITIONS", "20"))
                if single_only:
                    positions = await self.exchange.fetch_positions()
                    active = [p for p in positions if float(p.get("contracts", 0)) > 0]
                    if len(active) >= max_positions:
                        logger.warning(
                            f"🚫 [Bot C Guard] 활성 포지션 {len(active)}개 >= "
                            f"최대 {max_positions}개. 진입 거부: {payload.side.value} {symbol}"
                        )
                        return {"status": "rejected", "reason": "max_positions"}

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

            # [수정] 주문 전 레버리지 자동 세팅 (개별 payload 우선, 없으면 환경변수)
            try:
                leverage = payload.leverage if payload.leverage is not None else int(os.getenv("OKX_LEVERAGE", "10"))
                await self.exchange.set_leverage(leverage, ccxt_symbol, {"mgnMode": "cross"})
                logger.info(f"⚙️ [레버리지 설정] {ccxt_symbol} -> {leverage}x (Cross)")
            except Exception as e:
                logger.warning(f"⚠️ [레버리지 설정 실패] (이미 설정되어 있거나 API 제한일 수 있음): {e}")

            max_retries = 3
            last_err = ""
            for attempt in range(max_retries):
                cl_ord_id = None
                try:
                    start = time.time()
                    
                    if amount == 0 and payload.side in [SideType.CLOSE_LONG, SideType.CLOSE_SHORT]:
                        # 수량이 0인 경우, 해당 포지션 전체 청산(close_position) 수행
                        pos_side = params.get("posSide", "long")
                        try:
                            await self.exchange.cancel_all_orders(ccxt_symbol)
                        except:
                            pass
                        
                        # [FIX] 51108 (시장가 청산 한도 초과) 방지를 위해 포지션 수량과 maxMktSz 확인 후 분할 청산
                        await self.exchange.load_markets()
                        market_info = self.exchange.markets.get(ccxt_symbol, {}).get('info', {})
                        max_mkt_sz = float(market_info.get('maxMktSz', 0)) if market_info.get('maxMktSz') else 0
                        
                        positions = await self.exchange.fetch_positions([ccxt_symbol])
                        target_pos = next((p for p in positions if p['side'] == pos_side and float(p.get('contracts', 0)) > 0), None)
                        
                        if target_pos and max_mkt_sz > 0 and float(target_pos['contracts']) > max_mkt_sz:
                            logger.info(f"⚠️ 청산 수량({target_pos['contracts']})이 시장가 최대 한도({max_mkt_sz})를 초과하여 분할 청산합니다.")
                            remain = float(target_pos['contracts'])
                            close_side = "sell" if pos_side == "long" else "buy"
                            close_params = {"reduceOnly": True, "posSide": pos_side}
                            while remain > 0:
                                chunk = min(remain, max_mkt_sz)
                                await self.exchange.create_market_order(ccxt_symbol, close_side, chunk, params=close_params)
                                remain -= chunk
                                await asyncio.sleep(0.2)
                            order = {"id": "chunked_close"}
                        else:
                            order = await asyncio.wait_for(
                                self.exchange.close_position(ccxt_symbol, side=pos_side),
                                timeout=10.0
                            )
                    else:
                        # [FIX] 51202 방지를 위해 시장가 진입 분할 처리
                        await self.exchange.load_markets()
                        market_info = self.exchange.markets.get(ccxt_symbol, {}).get('info', {})
                        max_mkt_sz = float(market_info.get('maxMktSz', 0)) if market_info.get('maxMktSz') else 0
                        
                        if max_mkt_sz > 0 and amount > max_mkt_sz:
                            logger.info(f"⚠️ 진입 수량({amount})이 시장가 최대 한도({max_mkt_sz})를 초과하여 분할 진입합니다.")
                            remain = amount
                            orders = []
                            while remain > 0:
                                chunk = min(remain, max_mkt_sz)
                                o = await self.exchange.create_market_order(ccxt_symbol, side, chunk, params=params)
                                orders.append(o)
                                remain -= chunk
                                await asyncio.sleep(0.2)
                            order = orders[-1]
                        else:
                            # [Fix] clOrdId 부여: 타임아웃 시 체결 여부 확인으로 이중 체결 방지
                            cl_ord_id = "kbot" + uuid.uuid4().hex[:20]
                            order_params = dict(params)
                            order_params["clOrdId"] = cl_ord_id
                            order = await asyncio.wait_for(
                                self.exchange.create_market_order(
                                    ccxt_symbol, side, amount, params=order_params
                                ),
                                timeout=10.0,
                            )
                        
                    latency = time.time() - start
                    order_id = order.get("id", "N/A")
                    avg_price = order.get("average") or order.get("price", 0)
                    if not avg_price:
                        # OKX 시장가 주문 응답은 average/price가 None인 경우가 많음 → 현재가 폭백
                        try:
                            _t = await self.exchange.fetch_ticker(ccxt_symbol)
                            avg_price = _t.get("last") or 0
                        except Exception:
                            pass
                    logger.info(
                        f"✅ [실주문 성공] {side.upper()} {amount if amount > 0 else 'ALL'} {ccxt_symbol} "
                        f"@ {avg_price} | ID: {order_id} | Latency: {latency:.3f}s"
                    )
                    _record_trade(ccxt_symbol, payload.side.value, amount, avg_price, order_id)
                    return {"status": "ok", "order_id": order_id, "price": avg_price}
                except asyncio.TimeoutError:
                    last_err = f"타임아웃 (시도 {attempt+1}/{max_retries})"
                    logger.error(f"⚠️ OKX 주문 {last_err}")
                    # [Fix] 타임아웃 ≠ 실패: 주문이 이미 체결됐을 수 있음 → 재시도 전 확인
                    if cl_ord_id:
                        try:
                            existing = await self.exchange.fetch_order(
                                cl_ord_id, ccxt_symbol, params={"clOrdId": cl_ord_id}
                            )
                            if existing:
                                ex_id = existing.get("id", "N/A")
                                logger.info(
                                    f"✅ 타임아웃 주문이 실제로 체결됨 (clOrdId: {cl_ord_id}) — 성공 처리"
                                )
                                _record_trade(ccxt_symbol, payload.side.value, amount,
                                              existing.get("average") or 0, ex_id)
                                return {"status": "ok", "order_id": ex_id}
                        except Exception as lookup_err:
                            logger.warning(f"⚠️ 타임아웃 주문 조회 실패 (재시도 진행): {lookup_err}")
                except Exception as e:
                    last_err = str(e)
                    logger.error(f"⚠️ OKX API 에러 (시도 {attempt+1}/{max_retries}): {e}")

                    # [FIX] 이미 청산되었거나 포지션이 없는 경우 (51023, 51169) 에러 무시하고 성공 처리
                    if "51023" in last_err or "51169" in last_err:
                        logger.info("✅ 포지션이 이미 존재하지 않거나 청산 완료됨 (성공으로 간주)")
                        return {"status": "ok", "order_id": "already_closed"}

                    # [Fix] 51008 증거금 부족: 재시도 무의미 → 즉시 중단
                    if "51008" in last_err:
                        logger.warning(
                            f"🚫 [증거금 부족] {side.upper()} {amount} {ccxt_symbol} 주문 포기 (재시도 없음)"
                        )
                        return {"status": "failed", "error": "insufficient_margin"}

                    # [Fix] 51004 포지션 한도 초과: 한도 내로 수량 클램프 후 재시도
                    if "51004" in last_err:
                        m = re.search(r"more than (\d+)\(contracts\)", last_err)
                        if m:
                            cap = float(m.group(1))
                            try:
                                positions = await self.exchange.fetch_positions([ccxt_symbol])
                                pos_side_key = params.get("posSide", "long")
                                cur = sum(
                                    float(p.get("contracts", 0)) for p in positions
                                    if p.get("side") == pos_side_key
                                )
                                allowed = cap - cur
                                if allowed > 0 and allowed < amount:
                                    logger.warning(
                                        f"⚠️ [51004] {ccxt_symbol} 포지션 한도 {cap:.0f} (보유 {cur:.0f}) — "
                                        f"수량 {amount} → {allowed:.0f} 축소 후 재시도"
                                    )
                                    amount = allowed
                                    continue
                            except Exception as clamp_err:
                                logger.warning(f"⚠️ [51004] 한도 파싱/재시도 준비 실패: {clamp_err}")
                        logger.warning(f"🚫 [51004] {ccxt_symbol} 포지션 한도 초과 — 주문 포기")
                        return {"status": "failed", "error": "position_cap_exceeded"}

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
            return {"status": "failed", "error": last_err}


# ── TRADE RECORDER (P3: 실현손익 추적용 체결 기록) ──
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_TRADES_FILE = os.path.join(_BASE_DIR, "state", "trades.jsonl")

def _record_trade(symbol: str, side: str, amount: float, price: float, order_id: str):
    try:
        os.makedirs(os.path.dirname(_TRADES_FILE), exist_ok=True)
        rec = {
            "ts": int(time.time()),
            "symbol": symbol,
            "side": side,          # BUY / SELL / CLOSE_LONG / CLOSE_SHORT
            "amount": amount,      # 0이면 전량청산
            "price": price,
            "order_id": order_id,
        }
        with open(_TRADES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"⚠️ 거래 기록 실패: {e}")


def _rotate_trades_file(max_age_days: int = 90):
    """[Fix] trades.jsonl 로테이션: 켈리 분석 윈도우(90일) 이전 기록 정리로 무한 성장 방지."""
    try:
        if not os.path.exists(_TRADES_FILE):
            return
        cutoff = time.time() - max_age_days * 86400
        kept = []
        with open(_TRADES_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    if json.loads(line).get("ts", 0) >= cutoff:
                        kept.append(line)
                except json.JSONDecodeError:
                    continue
        tmp = _TRADES_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(kept)
        os.replace(tmp, _TRADES_FILE)
        logger.info(f"✅ 거래 기록 로테이션 완료: {len(kept)}건 유지 (최근 {max_age_days}일)")
    except Exception as e:
        logger.warning(f"⚠️ 거래 기록 로테이션 실패: {e}")


# ── PID LOCK ──
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
    signature = request.headers.get(WEBHOOK_SIGNATURE_HEADER, "")
    body = await request.text()
    if not verify_webhook_signature(body, signature):
        logger.warning(f"⚠️ Bot C 웹훅 서명 검증 실패 (IP: {request.remote})")
        return web.json_response({"error": "Invalid signature"}, status=401)
    try:
        payload = WebhookPayload.from_json(body)
        logger.info(f"📥 Webhook 수신: {payload.side.value} {payload.qty} {payload.symbol}")
        # [Fix] 실제 주문 실행 완료 후 결과 반환 (기존 fire-and-forget → 마스터 상태 불일치 해소)
        result = await bot.execute_order(payload)
        result = result or {"status": "failed", "error": "unknown"}
        if result.get("status") == "ok":
            return web.json_response({"status": "ok", "order_id": result.get("order_id")})
        logger.warning(f"⚠️ 주문 실행 결과: {result}")
        return web.json_response(result)
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
            balance_info = await bot.get_balance_str()
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
    positions = []
    try:
        if bot.exchange:
            balance_info = await bot.get_balance_str()
            for p in await bot.exchange.fetch_positions():
                if float(p.get("contracts") or 0) > 0 and p.get("side") in ("long", "short"):
                    positions.append({
                        "symbol": p["symbol"], 
                        "side": p["side"],
                        "timestamp": p.get("timestamp"),
                        # [Fix] 실제 진입 시각 노출 (timestamp=uTime은 갱신 시각이라 24h 스윕 판정 왜곡)
                        "cTime": (p.get("info") or {}).get("cTime"),
                        "unrealizedPnl": p.get("unrealizedPnl"),
                        "initialMargin": p.get("initialMargin")
                    })
    except Exception as e:
        balance_info = f"Error: {e}"
    return web.json_response({
        "bot_id": "BOT_C_OKX_SWAP",
        "status": "active",
        "is_running": True,
        "balance": balance_info,
        "positions": positions,
        "timestamp": time.time()
    })


async def handle_close_all(request):
    """EMERGENCY: Close all open positions"""
    # [Fix] 서명 검증 추가: 미인증 전량 청산 요청 차단
    signature = request.headers.get(WEBHOOK_SIGNATURE_HEADER, "")
    body = await request.text()
    if not verify_webhook_signature(body, signature):
        logger.warning(f"⚠️ close_all 서명 검증 실패 (IP: {request.remote})")
        return web.json_response({"error": "Invalid signature"}, status=401)
    results = []
    try:
        if not bot.exchange:
            return web.json_response({"error": "exchange not initialized"}, status=500)

        positions = await bot.exchange.fetch_positions()
        for pos in positions:
            contracts = float(pos.get("contracts", 0))
            if contracts == 0:
                continue
            symbol = pos.get("symbol", "")
            pos_side = pos.get("side", "").lower()
            if pos_side not in ("long", "short"):
                continue

            close_side = "short" if pos_side == "long" else "long"
            ccxt_symbol = symbol.replace("-SWAP", "").replace("-", "/", 1)
            if ":" not in ccxt_symbol:
                ccxt_symbol = ccxt_symbol + ":USDT"

            try:
                await bot.exchange.cancel_all_orders(ccxt_symbol)
                order = await asyncio.wait_for(
                    bot.exchange.close_position(ccxt_symbol, side=pos_side),
                    timeout=10.0
                )
                results.append({
                    "symbol": symbol,
                    "side_closed": pos_side,
                    "contracts": contracts,
                    "status": "closed",
                })
                logger.info(f"✅ [CLOSE_ALL] {symbol} ({pos_side}) {contracts}청산 완료")
            except Exception as e:
                results.append({"symbol": symbol, "status": f"error: {e}"})
                logger.error(f"❌ [CLOSE_ALL] {symbol} 청산 실패: {e}")

        try:
            send_telegram_alert(f"🚨 [Bot C] 전량 청산 완료: {len(results)}개 포지션")
        except Exception:
            pass

    except Exception as e:
        logger.error(f"[CLOSE_ALL] 오류: {e}")
        return web.json_response({"results": results, "error": str(e)}, status=500)

    return web.json_response({"results": results, "total_closed": len(results)})


async def main():
    await bot.init()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8013
    app = web.Application()
    app.router.add_post("/webhook", handle_webhook)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/status", handle_status)
    app.router.add_post("/close_all", handle_close_all)
    # [Fix] access_log 제거: /health·/status 폴링 로그 스팸 방지
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    # [Fix] 0.0.0.0 → 127.0.0.1: 같은 호스트 마스터만 호출하므로 외부 노출 차단
    site = web.TCPSite(runner, "127.0.0.1", port)
    
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
    # [Fix] 기본 포트를 8013으로 고정 — 인자 없이 실행해도 정상 포트 사용
    # (기존 8003 기본값이 고스트 프로세스 반복 생성의 근본 원인이었음)
    try:
        acquire_pid_lock()
        _rotate_trades_file()
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 프로그램 종료")
    except SystemExit as e:
        _cleanup_pid_lock()
        sys.exit(e.code)
    finally:
        _cleanup_pid_lock()
