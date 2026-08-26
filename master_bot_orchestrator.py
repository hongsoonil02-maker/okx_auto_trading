#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
master_bot_orchestrator.py — 통합 마스터 Bot 오케스트레이터
[Andre 최종 통합 & 가동 지휘소]

목적:
1. 검증 → 배포 → 모니터링을 일괄 처리
2. 신호 큐 관리 (Webhook 집중화)
3. 각 Bot의 상태 실시간 모니터링
4. 포트폴리오 통합 리포팅
"""
import asyncio
import json
import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional
import subprocess
import aiohttp
import aiohttp.web
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# .env 먼저 로드 (webhook_spec 모듈 레벨 상수 의존성: WEBHOOK_SECRET)
# ═══════════════════════════════════════════════════════════════════════════
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    from env_auto_scanner import auto_load_env
    _ENV_SCANNER = auto_load_env()
    _ENV_AUTOLOAD_OK = True
except ImportError:
    _ENV_SCANNER = None
    _ENV_AUTOLOAD_OK = False
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_BASE_DIR, ".env"))

from webhook_spec import (
    WebhookPayload, ActionType, SideType,
    verify_webhook_signature, sign_payload, WEBHOOK_SIGNATURE_HEADER,
)
from bot_config import bot_config


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - [MASTER] %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            os.path.join(BASE_DIR, "master_orchestrator.log"),
            encoding="utf-8",
            maxBytes=10*1024*1024,
            backupCount=5
        ),
        # StreamHandler 제거: systemd가 stdout을 같은 로그파일로 append하므로 중복 기록됨
    ],
)
logger = logging.getLogger("MasterOrchestrator")


class MasterBotOrchestrator:
    """
    통합 Bot 오케스트레이터
    
    구조:
    - Brain (Strategy Engine) → Master → 4 Bots
    - Master는 다음 역할 수행:
      1. 신호 수신 (Webhook)
      2. 각 Bot으로 신호 라우팅
      3. Bot 헬스 모니터링
      4. 포트폴리오 리포팅
    """
    
    # Bot endpoints (SSH local port forwarding tunnel path: localhost:800X → remote bot)
    # Currently active bots on this server:
    #   Bot C (OKX): port 8013 (실거래)
    BOT_ENDPOINTS = {
        "Bot C (OKX)": "http://localhost:8013",
    }
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_running = False
        self.bot_health = {}
        self.signal_queue = asyncio.Queue()
        self.processed_signals = []
        self.start_time = None
        # 포지션 상태 추적: symbol -> "LONG" / "SHORT" / "FLAT"
        # single_position_only=true이면 동시에 1개 포지션만 허용
        self.position_state: Dict[str, str] = {}
        self.single_position_only = os.getenv("OKX_SINGLE_POSITION_ONLY", "true").lower() == "true"
        self.max_active_subpositions = int(os.getenv("OKX_GLOBAL_MAX_POSITIONS", "20"))
        # [Fix] DCA 물타기 허용: 심볼별 분할진입 횟수 추적
        self.dca_entry_counts: Dict[str, int] = {}
        self.max_dca_per_symbol = bot_config.okx_max_dca
        # [Fix] 헬스 로그 스팸 방지: 상태 전환 시에만 로깅
        self._prev_bot_status: Dict[str, str] = {}
        # [Fix] Sweeper 중복 청산 방지: 심볼별 마지막 스윕 시각
        self._sweep_cooldown: Dict[str, float] = {}
    
    def load_active_ports(self):
        """active_ports.json에서 동적 포트 매핑 로드 및 환경변수(Remote IP) 병합"""
        try:
            if os.path.exists("active_ports.json"):
                with open("active_ports.json", "r", encoding="utf-8") as f:
                    registry = json.load(f)
                
                mapping = {
                    "bot_c_okx_swap.py": "Bot C (OKX)",
                }
                
                for bot_file, bot_name in mapping.items():
                    if bot_file in registry:
                        port = registry[bot_file]
                        self.BOT_ENDPOINTS[bot_name] = f"http://localhost:{port}"
            
            # 원격 GCP 봇 IP 덮어쓰기 (하이브리드 분산 라우팅)
            gcp_1_ip = os.environ.get("GCP_1_IP")
            if gcp_1_ip:
                self.BOT_ENDPOINTS["Bot C (OKX)"] = f"http://{gcp_1_ip}:8008"
                
                
        except Exception as e:
            logger.error(f"❌ 설정(포트/IP) 로드 실패: {e}")
            
    async def init(self):
        """초기화"""
        self.session = aiohttp.ClientSession()
        self.start_time = datetime.now()
        self.load_active_ports()
        logger.info("✅ Master Orchestrator 초기화 완료")
    
    async def close(self):
        """종료"""
        if self.session:
            await self.session.close()
        logger.info("✅ Master Orchestrator 종료")
    
    async def check_bot_health(self) -> Dict:
        """모든 Bot의 헬스 상태 확인"""
        health_status = {}
        
        for bot_name, endpoint in self.BOT_ENDPOINTS.items():
            try:
                async with self.session.get(
                    f"{endpoint}/health",
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        health_status[bot_name] = {
                            "status": "online",
                            "data": data,
                        }
                        # [Fix] ONLINE 로그는 상태 전환 시 1회만 (30초 주기 스팸 제거)
                        if self._prev_bot_status.get(bot_name) != "online":
                            logger.info(f"✅ {bot_name}: ONLINE")
                    else:
                        health_status[bot_name] = {
                            "status": "error",
                            "code": resp.status,
                        }
                        if self._prev_bot_status.get(bot_name) != "error":
                            logger.warning(f"⚠️  {bot_name}: HTTP {resp.status}")
            
            except asyncio.TimeoutError:
                health_status[bot_name] = {
                    "status": "timeout",
                }
                if self._prev_bot_status.get(bot_name) != "timeout":
                    logger.warning(f"⚠️  {bot_name}: TIMEOUT")
            
            except Exception as e:
                health_status[bot_name] = {
                    "status": "unreachable",
                    "error": str(e),
                }
                if self._prev_bot_status.get(bot_name) != "unreachable":
                    logger.error(f"❌ {bot_name}: {e}")
        
        self.bot_health = health_status
        self._prev_bot_status = {name: h.get("status", "unknown") for name, h in health_status.items()}
        return health_status
    
    async def route_signal_to_bot(
        self,
        bot_name: str,
        webhook_payload: Dict,
    ) -> Dict:
        """
        신호를 특정 Bot으로 라우팅
        
        Args:
            bot_name: Bot 이름
            webhook_payload: Webhook Payload (JSON)
        
        Returns:
            주문 결과
        """
        if bot_name not in self.BOT_ENDPOINTS:
            return {
                "error": f"Unknown bot: {bot_name}",
                "status": "failed",
            }
        
        endpoint = self.BOT_ENDPOINTS[bot_name]
        
        try:
            body = json.dumps(webhook_payload)
            headers = {"Content-Type": "application/json", WEBHOOK_SIGNATURE_HEADER: sign_payload(body)}
            async with self.session.post(
                f"{endpoint}/webhook",
                data=body,
                headers=headers,
                # [Fix] Bot C가 실제 주문 완료 후 응답하므로 주문 소요시간만큼 타임아웃 확보
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status in [200, 201]:
                    result = await resp.json()
                    # [Fix] 주문 실패 응답 구분 로깅 (상태 불일치 진단용)
                    if result.get("status") == "ok" or result.get("order_id"):
                        logger.info(f"✅ {bot_name} 신호 처리 완료: {result.get('order_id', 'N/A')}")
                    else:
                        logger.warning(f"⚠️ {bot_name} 주문 실패 응답: {result}")
                    
                    # 신호 로깅
                    self.processed_signals.append({
                        "timestamp": datetime.now().isoformat(),
                        "bot": bot_name,
                        "signal_id": webhook_payload.get("signal_id"),
                        "result": result,
                    })
                    # [Fix] 메모리 무한 성장 방지: 최근 500건만 유지
                    if len(self.processed_signals) > 500:
                        del self.processed_signals[:-500]
                    
                    return result
                else:
                    error_text = await resp.text()
                    logger.error(f"❌ {bot_name} 오류: {resp.status} - {error_text}")
                    return {
                        "error": f"Bot API error: {resp.status}",
                        "status": "failed",
                    }
        
        except asyncio.TimeoutError:
            logger.error(f"❌ {bot_name} 타임아웃")
            return {
                "error": "Bot API timeout",
                "status": "timeout",
            }
        
        except Exception as e:
            logger.error(f"❌ {bot_name} 예외: {e}")
            return {
                "error": str(e),
                "status": "failed",
            }
    
    async def process_signal_queue(self):
        """신호 큐 처리 (백그라운드 태스크)"""
        logger.info("🔄 신호 큐 처리 시작")
        
        while self.is_running:
            try:
                # 신호 수신 (타임아웃 3초)
                signal = await asyncio.wait_for(
                    self.signal_queue.get(),
                    timeout=3.0,
                )
                
                market = signal.get("market")
                bot_mapping = {
                    "okx_swap": "Bot C (OKX)",
                }
                
                bot_name = bot_mapping.get(market)
                if not bot_name:
                    logger.warning(f"⚠️  Unknown market: {market}")
                    continue

                # ── 포지션 상태 검사 (15개 동시 진입 방지) ──
                side = signal.get("side", "")
                symbol = signal.get("symbol", "")
                active_count = sum(1 for p in self.position_state.values() if p != "FLAT")

                if side in ("BUY", "SELL"):
                    sig_dir = "LONG" if side == "BUY" else "SHORT"
                    cur_state = self.position_state.get(symbol, "FLAT")
                    is_dca = cur_state == sig_dir  # 같은 방향 보유 중 = 물타기

                    if cur_state != "FLAT" and not is_dca:
                        logger.warning(
                            f"🚫 [Master Guard] {symbol} 반대 방향 보유 "
                            f"({cur_state}). 진입 거부."
                        )
                        continue
                    if is_dca and self.dca_entry_counts.get(symbol, 0) >= self.max_dca_per_symbol:
                        logger.warning(
                            f"🚫 [Master Guard] {symbol} DCA {self.max_dca_per_symbol}회 초과. 진입 거부."
                        )
                        continue
                    if cur_state == "FLAT" and self.single_position_only:
                        if active_count >= self.max_active_subpositions:
                            logger.warning(
                                f"🚫 [Master Guard] 활성 포지션 {active_count}개 ≥ "
                                f"최대 {self.max_active_subpositions}개 제한. "
                                f"진입 거부: {side} {symbol}"
                            )
                            continue

                    # 순차 라우팅 (병렬 create_task 제거 → 레이스 컨디션 방지)
                    result = await self.route_signal_to_bot(bot_name, signal)
                    if result.get("status") == "ok" or result.get("order_id"):
                        self.position_state[symbol] = sig_dir
                        self.dca_entry_counts[symbol] = self.dca_entry_counts.get(symbol, 0) + 1
                        if is_dca:
                            logger.info(
                                f"✅ [DCA 허용] {symbol} {sig_dir} 물타기 "
                                f"({self.dca_entry_counts[symbol]}/{self.max_dca_per_symbol})"
                            )
                    elif not is_dca:
                        self.position_state[symbol] = "FLAT"

                elif side in ("CLOSE_LONG", "CLOSE_SHORT"):
                    # 청산 신호 (qty=0이면 전량청산, qty>0이면 부분청산 → 포지션 유지)
                    result = await self.route_signal_to_bot(bot_name, signal)
                    try:
                        close_qty = float(signal.get("qty", 0) or 0)
                    except (TypeError, ValueError):
                        close_qty = 0
                    if close_qty == 0:
                        self.position_state[symbol] = "FLAT"
                        self.dca_entry_counts[symbol] = 0
            
            except asyncio.TimeoutError:
                # 신호 없음 (정상)
                continue
            
            except Exception as e:
                logger.error(f"❌ 신호 큐 처리 오류: {e}")
    
    async def generate_status_report(self) -> str:
        """현황 보고서 생성"""
        uptime = datetime.now() - self.start_time
        
        report = [
            "=" * 80,
            "[MASTER BOT ORCHESTRATOR STATUS REPORT]",
            "=" * 80,
            f"Timestamp: {datetime.now().isoformat()}",
            f"Uptime: {uptime}",
            "",
            "[BOT HEALTH STATUS]",
        ]
        
        for bot_name, health in self.bot_health.items():
            status = health.get("status", "unknown")
            if status == "online":
                report.append(f"  ✅ {bot_name}: ONLINE")
            else:
                report.append(f"  ⚠️  {bot_name}: {status.upper()}")
        
        report.extend([
            "",
            "[SIGNAL PROCESSING]",
            f"  Total Signals Processed: {len(self.processed_signals)}",
            f"  Signal Queue Size: {self.signal_queue.qsize()}",
            "",
            "[RECENT SIGNALS]",
        ])
        
        # 최근 신호 10개
        for signal in self.processed_signals[-10:]:
            report.append(
                f"  {signal['timestamp']}: {signal['bot']} - {signal['signal_id']}"
            )
        
        report.extend([
            "",
            "[CONFIGURATION]",
            f"  Master Port: 8009",
            f"  Bot Endpoints:",
        ])
        
        for bot_name, endpoint in self.BOT_ENDPOINTS.items():
            report.append(f"    {bot_name}: {endpoint}")
        
        report.extend([
            "",
            "[NEXT STEPS]",
            "1. Monitor signal queue: curl http://localhost:8009/status",
            "2. View bot status: curl http://localhost:8009/bots/health",
            "3. Send test signal: POST http://localhost:8009/webhook",
            "=" * 80,
        ])
        
        return "\n".join(report)
    
    async def run_webhook_server(self):
        """Webhook 수신 서버 실행 (Master)"""
        
        async def handle_webhook(request):
            """POST /webhook - Master Webhook 수신 (HMAC 서명 검증)"""
            signature = request.headers.get(WEBHOOK_SIGNATURE_HEADER, "")
            body = await request.text()
            if not verify_webhook_signature(body, signature):
                logger.warning(f"⚠️ 웹훅 서명 검증 실패 (IP: {request.remote})")
                return aiohttp.web.json_response(
                    {"error": "Invalid signature", "status": "denied"},
                    status=401,
                )
            try:
                data = json.loads(body)
                await self.signal_queue.put(data)
                logger.info(f"📥 신호 수신: {data.get('signal_id')} ({data.get('market')})")
                return aiohttp.web.json_response({
                    "status": "queued",
                    "signal_id": data.get("signal_id"),
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception as e:
                logger.error(f"❌ Webhook 처리 오류: {e}")
                return aiohttp.web.json_response(
                    {"error": str(e), "status": "failed"},
                    status=400,
                )
        
        async def handle_status(request):
            """GET /status - 현황 조회"""
            report = await self.generate_status_report()
            return aiohttp.web.Response(text=report)
        
        async def handle_health(request):
            """GET /bots/health - Bot 헬스 조회"""
            health = await self.check_bot_health()
            return aiohttp.web.json_response(health)
        
        # Webhook 서버 설정
        app = aiohttp.web.Application()
        app.router.add_post("/webhook", handle_webhook)
        app.router.add_get("/status", handle_status)
        app.router.add_get("/bots/health", handle_health)
        
        for attempt in range(5):
            # [Fix] access_log 제거: 폴링 로그 스팸 방지
            runner = aiohttp.web.AppRunner(app, access_log=None)
            await runner.setup()
            # [Fix] 0.0.0.0 → 127.0.0.1: 브레인/봇 모두 같은 호스트이므로 외부 노출 차단
            site = aiohttp.web.TCPSite(runner, "127.0.0.1", 8009)
            try:
                await site.start()
                break
            except OSError as e:
                await runner.cleanup()
                if attempt == 4:
                    raise e
                logger.warning(f"⚠️  Master Port 8009 사용 중, 2초 후 재시도... ({attempt+1}/5)")
                await asyncio.sleep(2)
        
        logger.info("✅ Master Webhook 서버 시작: http://127.0.0.1:8009")
        
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("🛑 서버 종료 중...")
            await runner.cleanup()
    
    async def run(self):
        """메인 루프"""
        await self.init()
        self.is_running = True
        
        logger.info("🚀 Master Bot Orchestrator 시작")
        
        # 헬스 체크 (초기)
        await self.check_bot_health()
        # [Fix] 재시작 시 실제 보유 포지션 동기화 (state 초기화로 인한 중복 진입 방지)
        await self._sync_positions_from_bots()
        
        # 백그라운드 태스크
        tasks = [
            asyncio.create_task(self.process_signal_queue()),
            asyncio.create_task(self.run_webhook_server()),
            asyncio.create_task(self._periodic_health_check()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        finally:
            self.is_running = False
            await self.close()
    
    async def _sync_positions_from_bots(self):
        """부팅 시 Bot C /status에서 실제 보유 포지션을 읽어 position_state 시딩"""
        url = f"{self.BOT_ENDPOINTS.get('Bot C (OKX)', 'http://localhost:8013')}/status"
        last_err = None
        for attempt in range(5):
            try:
                async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()
                n = 0
                for p in data.get("positions", []):
                    sym, side = p.get("symbol"), p.get("side")
                    if sym and side in ("long", "short"):
                        self.position_state[sym] = side.upper()
                        self.dca_entry_counts.setdefault(sym, 1)
                        n += 1
                logger.info(f"🔄 [Master] 실제 보유 포지션 {n}개 동기화 완료")
                return
            except Exception as e:
                last_err = e
                await asyncio.sleep(3)
        logger.warning(f"⚠️ [Master] 포지션 동기화 실패(빈 상태로 시작): {last_err}")

    async def _periodic_health_check(self):
        """주기적 헬스 체크 (30초)"""
        while self.is_running:
            try:
                await asyncio.sleep(30)
                self.load_active_ports() # 동적 포트 변경 감지
                await self.check_bot_health()
                # [Fix] /status 1회 조회로 리컨실 + 스윕 공용 사용
                data = await self._fetch_bot_status()
                if data:
                    self._reconcile_positions(data)
                    await self._sweep_stagnant_positions(data)
            except Exception as e:
                logger.error(f"❌ 주기적 헬스 체크 오류: {e}")

    async def _fetch_bot_status(self) -> Optional[Dict]:
        """Bot C /status 조회 (실패 시 None)"""
        try:
            url = f"{self.BOT_ENDPOINTS.get('Bot C (OKX)', 'http://localhost:8013')}/status"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                return await resp.json()
        except Exception as e:
            logger.warning(f"⚠️ [Master] Bot 상태 조회 실패: {e}")
            return None

    def _reconcile_positions(self, data: Dict):
        """
        [Fix] 실제 포지션과 position_state 주기 동기화.
        기존엔 부팅 시 1회만 동기화 → 거래소 측 청산(스탑아웃/수동/강제) 후
        마스터 상태가 남아 재진입 가드가 오작동하는 문제 해소.
        """
        try:
            actual: Dict[str, str] = {}
            for p in data.get("positions", []):
                sym, side = p.get("symbol"), p.get("side")
                if sym and side in ("long", "short"):
                    actual[sym] = side.upper()

            for sym in list(self.position_state.keys()):
                if sym not in actual and self.position_state[sym] != "FLAT":
                    logger.info(f"🔄 [Reconcile] {sym} 실포지션 없음 → 상태 FLAT 정리")
                    self.position_state[sym] = "FLAT"
                    self.dca_entry_counts[sym] = 0

            for sym, side in actual.items():
                if self.position_state.get(sym, "FLAT") == "FLAT":
                    logger.info(f"🔄 [Reconcile] {sym} 실포지션({side}) 반영")
                    self.position_state[sym] = side
                    self.dca_entry_counts.setdefault(sym, 1)
        except Exception as e:
            logger.error(f"⚠️ 포지션 리컨실 오류: {e}")

    async def _sweep_stagnant_positions(self, data: Dict):
        """진입 후 24시간 경과 & 수익률 5% 미만인 데드 포지션 강제 청산"""
        try:
            now_ms = time.time() * 1000
            for p in data.get("positions", []):
                sym = p.get("symbol")
                side = p.get("side")
                # [Fix] 진입 시각은 cTime 기준 (timestamp=uTime은 DCA 시 갱신되어 24h 판정 왜곡)
                ts = p.get("cTime") or p.get("timestamp")
                pnl = p.get("unrealizedPnl")
                margin = p.get("initialMargin")
                
                if not ts or not margin or float(margin) == 0:
                    continue

                # [Fix] 심볼별 스윕 쿨다운: 청산 완료 전까지 30초마다 중복 신호 발송 방지
                last_sweep = self._sweep_cooldown.get(sym, 0)
                if now_ms - last_sweep < 15 * 60 * 1000:
                    continue
                
                # 24시간 경과 여부 (86400 * 1000 ms)
                held_ms = now_ms - float(ts)
                if held_ms > 129600000:  # 36시간 (기존 24시간에서 완화: 수익 거래 평균 보유 28h)
                    raw_pnl = float(pnl)
                    # [Fix] abs() 제거: 수익 포지션은 스윕 대상에서 제외.
                    # 손실 또는 극소 수익(마진 대비 3% 미만) 정체 포지션만 강제 청산.
                    if raw_pnl > float(margin) * 0.03:
                        continue  # 충분한 수익 중 → 스윕 제외
                    pnl_pct = raw_pnl / float(margin)
                    # 실제 변동 0.5% (레버리지 10x 적용된 pnl_pct가 0.05 미만)
                    if pnl_pct > -0.03:  # 손실이 마진 대비 -3% 초과 (즉 -3%~+3%) 구간만 정체로 간주
                        logger.warning(f"🧹 [Sweeper] 24시간 정체 포지션 발견: {sym} (PnL: {float(pnl):.2f}). 강제 청산 시도!")
                        # Webhook 생성 및 발송
                        close_side = "CLOSE_LONG" if side == "long" else "CLOSE_SHORT"
                        payload = {
                            "signal_id": f"sweep_{int(time.time())}",
                            "market": "okx_swap",
                            "symbol": sym.replace("/", "-").replace(":USDT", "-SWAP"),
                            "side": close_side,
                            "qty": 0
                        }
                        await self.signal_queue.put(payload)
                        self._sweep_cooldown[sym] = now_ms
                        
        except Exception as e:
            logger.error(f"⚠️ 정체 포지션 스윕 중 에러: {e}")


async def main():
    """메인 진입점"""
    orchestrator = MasterBotOrchestrator()
    
    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        logger.info("🛑 Master Orchestrator 종료")
        sys.exit(0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
