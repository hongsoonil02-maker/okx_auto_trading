#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
api_failover_engine.py — API 레이턴시 모니터링 & 자동 Failover
[Genspark 에이전트 - 50ms 초과 시 대체 엔드포인트 우회]

목적:
1. 각 거래소 API 응답 시간 실시간 모니터링
2. 50ms 초과 시 대체 엔드포인트로 즉시 우회
3. 비동기 코루틴 기반 병렬 요청
4. API 응답 지연 로그 상세 기록
"""
import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import aiohttp


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - [API_FAILOVER] %(message)s',
)
logger = logging.getLogger("APIFailoverEngine")


class APIEndpointConfig:
    """거래소별 API 엔드포인트 설정 (Primary + Fallback)"""
    
    ENDPOINTS = {
        "kis": {
            "name": "한국투자증권 (KIS API)",
            "primary": "https://openapi.koreainvestment.com:9443",
            "fallback": "https://openapivip.koreainvestment.com:9443",
            "timeout_ms": 50,
            "health_check": "/oauth2/authorization/token",
        },
        "ib": {
            "name": "Interactive Brokers",
            "primary": "https://api.ibkr.com:443",
            "fallback": "https://api-sandbox.ibkr.com:443",
            "timeout_ms": 50,
            "health_check": "/orders",
        },
        "okx": {
            "name": "OKX (CCXT)",
            "primary": "https://www.okx.com/api/v5",
            "fallback": "https://aws.okx.com/api/v5",
            "timeout_ms": 50,
            "health_check": "/market/tickers",
        },
        "upbit": {
            "name": "Upbit",
            "primary": "https://api.upbit.com/v1",
            "fallback": "https://api-backup.upbit.com/v1",
            "timeout_ms": 50,
            "health_check": "/ticker",
        },
    }


class LatencyMonitor:
    """API 레이턴시 모니터링"""
    
    def __init__(self):
        self.latency_history = {}  # {exchange: [list of latencies]}
        self.failover_status = {}  # {exchange: 'primary'|'fallback'}
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def init(self):
        """초기화"""
        self.session = aiohttp.ClientSession()
        
        # 초기 상태: 모두 Primary
        for exchange in APIEndpointConfig.ENDPOINTS:
            self.latency_history[exchange] = []
            self.failover_status[exchange] = "primary"
    
    async def close(self):
        """종료"""
        if self.session:
            await self.session.close()
    
    async def ping_endpoint(
        self,
        exchange: str,
        endpoint: str,
        use_primary: bool = True,
    ) -> Tuple[bool, float]:
        """
        API 엔드포인트 핑 테스트
        
        Args:
            exchange: 거래소명
            endpoint: 엔드포인트 URL
            use_primary: Primary 사용 여부
        
        Returns:
            (성공 여부, 레이턴시_ms)
        """
        try:
            start_time = time.time()
            
            async with self.session.get(
                endpoint,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                elapsed_ms = (time.time() - start_time) * 1000
                
                if resp.status in [200, 400]:  # 200: OK, 400: Bad Request (정상 응답)
                    return True, elapsed_ms
                else:
                    return False, elapsed_ms
        
        except asyncio.TimeoutError:
            logger.warning(f"⏱️  {exchange}: Timeout (Primary={use_primary})")
            return False, 5000.0  # 5초 = 타임아웃
        
        except Exception as e:
            logger.error(f"❌ {exchange}: {e}")
            return False, 0.0
    
    async def check_latency(self, exchange: str) -> Tuple[str, float]:
        """
        거래소 API 레이턴시 체크 & Failover 판단
        
        Args:
            exchange: 거래소명
        
        Returns:
            (선택 엔드포인트 ('primary'|'fallback'), 레이턴시_ms)
        """
        config = APIEndpointConfig.ENDPOINTS.get(exchange)
        if not config:
            return "primary", 0.0
        
        threshold_ms = config["timeout_ms"]
        primary_url = config["primary"] + config["health_check"]
        fallback_url = config["fallback"] + config["health_check"]
        
        # Primary 테스트
        primary_ok, primary_latency = await self.ping_endpoint(
            exchange, primary_url, use_primary=True
        )
        
        # Primary 정상 & 레이턴시 OK
        if primary_ok and primary_latency < threshold_ms:
            self.failover_status[exchange] = "primary"
            self.latency_history[exchange].append(primary_latency)
            
            logger.info(
                f"✅ {exchange}: Primary OK ({primary_latency:.2f}ms < {threshold_ms}ms)"
            )
            
            return "primary", primary_latency
        
        # Primary 레이턴시 초과 → Fallback 테스트
        if primary_ok and primary_latency >= threshold_ms:
            logger.warning(
                f"⚠️  {exchange}: Primary 레이턴시 초과 ({primary_latency:.2f}ms >= {threshold_ms}ms)"
            )
            
            fallback_ok, fallback_latency = await self.ping_endpoint(
                exchange, fallback_url, use_primary=False
            )
            
            if fallback_ok and fallback_latency < threshold_ms:
                self.failover_status[exchange] = "fallback"
                self.latency_history[exchange].append(fallback_latency)
                
                logger.warning(
                    f"⚠️  {exchange}: ✅ Fallback로 우회 ({fallback_latency:.2f}ms < {threshold_ms}ms)"
                )
                
                return "fallback", fallback_latency
        
        # Primary 실패 → Fallback 시도
        if not primary_ok:
            logger.error(f"❌ {exchange}: Primary 실패")
            
            fallback_ok, fallback_latency = await self.ping_endpoint(
                exchange, fallback_url, use_primary=False
            )
            
            if fallback_ok:
                self.failover_status[exchange] = "fallback"
                self.latency_history[exchange].append(fallback_latency)
                
                logger.warning(
                    f"⚠️  {exchange}: ✅ Fallback으로 우회 ({fallback_latency:.2f}ms)"
                )
                
                return "fallback", fallback_latency
        
        # 모든 엔드포인트 실패
        logger.error(f"❌❌ {exchange}: Primary & Fallback 모두 실패!")
        self.latency_history[exchange].append(5000.0)
        
        return "unavailable", 5000.0
    
    def get_latency_stats(self, exchange: str) -> Dict[str, float]:
        """
        레이턴시 통계
        
        Args:
            exchange: 거래소명
        
        Returns:
            {min, max, avg, current}
        """
        history = self.latency_history.get(exchange, [])
        
        if not history:
            return {
                "min_ms": 0.0,
                "max_ms": 0.0,
                "avg_ms": 0.0,
                "current_ms": 0.0,
            }
        
        return {
            "min_ms": min(history),
            "max_ms": max(history),
            "avg_ms": sum(history) / len(history),
            "current_ms": history[-1] if history else 0.0,
        }
    
    def get_failover_report(self) -> str:
        """Failover 상태 보고서"""
        report = [
            "=" * 80,
            "[API FAILOVER STATUS REPORT]",
            f"Timestamp: {datetime.now().isoformat()}",
            "=" * 80,
            "",
        ]
        
        for exchange, config in APIEndpointConfig.ENDPOINTS.items():
            status = self.failover_status.get(exchange, "unknown")
            stats = self.get_latency_stats(exchange)
            
            status_icon = "✅" if status == "primary" else "⚠️" if status == "fallback" else "❌"
            
            report.append(f"{status_icon} [{exchange.upper()}] {config['name']}")
            report.append(f"   상태: {status.upper()}")
            report.append(f"   레이턴시: {stats['current_ms']:.2f}ms (Avg: {stats['avg_ms']:.2f}ms, Min: {stats['min_ms']:.2f}ms, Max: {stats['max_ms']:.2f}ms)")
            report.append(f"   임계값: {config['timeout_ms']}ms")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


async def main():
    """메인 진입점"""
    monitor = LatencyMonitor()
    
    try:
        await monitor.init()
        
        print("=" * 80)
        print("[API Failover & 레이턴시 모니터링 테스트]")
        print("=" * 80)
        
        # 각 거래소 레이턴시 체크
        for exchange in APIEndpointConfig.ENDPOINTS:
            endpoint, latency = await monitor.check_latency(exchange)
        
        # 보고서 출력
        report = monitor.get_failover_report()
        print("\n" + report)
        
        # 보고서 저장
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        report_path = os.path.join(base_dir, "api_failover_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        logger.info(f"✅ Failover 보고서 저장: {report_path}")
        
        return 0
    
    finally:
        await monitor.close()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
