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
# [No Hardcoding 강제] 임포트 즉시 워크스페이스 .env 자동 스캔 & 병합
# ═══════════════════════════════════════════════════════════════════════════
try:
    from env_auto_scanner import auto_load_env
    _ENV_SCANNER = auto_load_env()  # 런타임 시작 시 .env 재귀 스캔 → os.environ 병합
    _ENV_AUTOLOAD_OK = True
except ImportError:
    _ENV_SCANNER = None
    _ENV_AUTOLOAD_OK = False
    print("⚠️  env_auto_scanner 미발견 - 시스템 환경변수만 사용")


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
        logging.StreamHandler(),
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
    
    # SSH 로컬 포트 포워딩 터널 경유 (localhost:800X → 원격 봇)
    # 터널: ssh -L 8001:localhost:8001 ... (방화벽 우회)
    BOT_ENDPOINTS = {
        "Bot A (KR Stock)": "http://localhost:8001",
        "Bot C (OKX)": "http://localhost:8013",
    }
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_running = False
        self.bot_health = {}
        self.signal_queue = asyncio.Queue()
        self.processed_signals = []
        self.start_time = None
    
    def load_active_ports(self):
        """active_ports.json에서 동적 포트 매핑 로드 및 환경변수(Remote IP) 병합"""
        try:
            if os.path.exists("active_ports.json"):
                with open("active_ports.json", "r", encoding="utf-8") as f:
                    registry = json.load(f)
                
                mapping = {
                    "bot_a_kr_stock.py": "Bot A (KR Stock)",
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
                
            gcp_2_ip = os.environ.get("GCP_2_IP")
            if gcp_2_ip:
                self.BOT_ENDPOINTS["Bot D (Upbit)"] = f"http://{gcp_2_ip}:8006"
                
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
                        logger.info(f"✅ {bot_name}: ONLINE")
                    else:
                        health_status[bot_name] = {
                            "status": "error",
                            "code": resp.status,
                        }
                        logger.warning(f"⚠️  {bot_name}: HTTP {resp.status}")
            
            except asyncio.TimeoutError:
                health_status[bot_name] = {
                    "status": "timeout",
                }
                logger.warning(f"⚠️  {bot_name}: TIMEOUT")
            
            except Exception as e:
                health_status[bot_name] = {
                    "status": "unreachable",
                    "error": str(e),
                }
                logger.error(f"❌ {bot_name}: {e}")
        
        self.bot_health = health_status
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
            async with self.session.post(
                f"{endpoint}/webhook",
                json=webhook_payload,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status in [200, 201]:
                    result = await resp.json()
                    logger.info(f"✅ {bot_name} 신호 처리 완료: {result.get('order_id', 'N/A')}")
                    
                    # 신호 로깅
                    self.processed_signals.append({
                        "timestamp": datetime.now().isoformat(),
                        "bot": bot_name,
                        "signal_id": webhook_payload.get("signal_id"),
                        "result": result,
                    })
                    
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
                    "kr_stock": "Bot A (KR Stock)",
                    "nasdaq": "Bot B (Nasdaq)",
                    "okx_swap": "Bot C (OKX)",
                    "upbit": "Bot D (Upbit)",
                }
                
                bot_name = bot_mapping.get(market)
                if not bot_name:
                    logger.warning(f"⚠️  Unknown market: {market}")
                    continue
                
                # Bot으로 라우팅
                result = await self.route_signal_to_bot(bot_name, signal)
            
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
            """POST /webhook - Master Webhook 수신"""
            try:
                data = await request.json()
                
                # 신호 큐에 추가
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
            runner = aiohttp.web.AppRunner(app)
            await runner.setup()
            site = aiohttp.web.TCPSite(runner, "0.0.0.0", 8009)
            try:
                await site.start()
                break
            except OSError as e:
                await runner.cleanup()
                if attempt == 4:
                    raise e
                logger.warning(f"⚠️  Master Port 8009 사용 중, 2초 후 재시도... ({attempt+1}/5)")
                await asyncio.sleep(2)
        
        logger.info("✅ Master Webhook 서버 시작: http://0.0.0.0:8009")
        
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
    
    async def _periodic_health_check(self):
        """주기적 헬스 체크 (30초)"""
        while self.is_running:
            try:
                await asyncio.sleep(30)
                self.load_active_ports() # 동적 포트 변경 감지
                await self.check_bot_health()
            except Exception as e:
                logger.error(f"❌ 주기적 헬스 체크 오류: {e}")


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
