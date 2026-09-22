#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_latency.py — Jev AI & OKX LOB Feed Latency Benchmark (Phase 1)
- Measures RTT (Round Trip Time) between GCP Server and Typesafe AI API (https://api.typesafe.ai/v1/systemone)
- Measures OKX Public WebSocket connect time and LOB message ingestion latency
- Computes Min, Max, Mean, Median (p50), p95, p99 latencies
- Verifies Phase 1 target: Latency < 300ms
"""
import asyncio
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s")

from jev.okx_lob_feed import OKXLOBFeed
from jev.typesafe_client import TypesafeJevClient


SAMPLE_LOB_STATE = (
    "Symbol: BTC-USDT-SWAP\n"
    "Best Bid: 63500.1 (Size: 4.250) | Best Ask: 63500.5 (Size: 1.120)\n"
    "Spread: 0.4000 (0.63 bps) | MicroPrice: 63500.416\n"
    "Top 5 Bids [Price:Size]: [63500.1:4.25, 63499.8:8.50, 63499.0:15.00, 63498.5:6.20, 63497.0:10.00]\n"
    "Top 5 Asks [Price:Size]: [63500.5:1.12, 63501.0:2.30, 63501.5:5.00, 63502.0:7.10, 63503.0:12.40]\n"
    "LOB Imbalance: +0.582 (BID_HEAVY)\n"
    "Market Context: Supertrend bullish continuation signal on 1h\n"
)


async def benchmark_okx_ws(symbols=["BTC-USDT-SWAP", "ETH-USDT-SWAP"], duration_sec=5):
    print("\n" + "=" * 60)
    print("📡 [TEST 1] OKX Public WebSocket LOB 수신 속도 측정")
    print("=" * 60)
    feed = OKXLOBFeed()
    feed.subscribe(symbols)
    
    start_conn = time.perf_counter()
    await feed.start()
    
    print(f"⏳ {duration_sec}초 동안 LOB 데이터 수집 중...")
    await asyncio.sleep(duration_sec)
    
    for sym in symbols:
        lob = feed.get_lob(sym)
        if lob:
            age_ms = (time.time() - lob.timestamp) * 1000.0
            print(f"  ✅ [{sym}] 수신 완료 | Best Bid: {lob.best_bid} | Best Ask: {lob.best_ask} | Age: {age_ms:.1f}ms")
        else:
            print(f"  ⚠️ [{sym}] LOB 미수신 (심볼 또는 네트워크 확인 필요)")
            
    await feed.stop()


async def benchmark_typesafe_api(iterations=10, timeout_ms=1000):
    print("\n" + "=" * 60)
    print(f"⚡ [TEST 2] Typesafe AI Jev System One API 레이턴시 벤치마크 ({iterations}회)")
    print("=" * 60)
    
    api_key = os.getenv("TYPESAFE_AI_API_KEY", "")
    if not api_key:
        print("⚠️ TYPESAFE_AI_API_KEY가 설정되지 않았습니다. (.env에 키 설정 필요)")
        print("ℹ️ 로컬 휴리스틱 폴백 엔진으로 레이턴시 시뮬레이션을 수행합니다.")
    else:
        print(f"🔑 API Key 감지됨: {api_key[:4]}****{api_key[-4:]}")
        
    client = TypesafeJevClient(api_key=api_key, timeout_ms=timeout_ms)
    latencies = []
    success_count = 0
    fallback_count = 0
    
    print(f"{'No.':<5} | {'Latency (ms)':<15} | {'up_in_10':<10} | {'Action':<10} | {'Status':<15}")
    print("-" * 65)
    
    for i in range(1, iterations + 1):
        t0 = time.perf_counter()
        decision = await client.predict_orderbook(SAMPLE_LOB_STATE, lob_imbalance=0.58)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(decision.latency_ms if decision.latency_ms > 0 else elapsed_ms)
        
        status = "FALLBACK" if decision.is_fallback else "OK (LIVE)"
        if decision.is_fallback:
            fallback_count += 1
        else:
            success_count += 1
            
        print(f"{i:<5} | {decision.latency_ms:<15.2f} | {decision.up_in_10:<10.3f} | {decision.action:<10} | {status:<15}")
        await asyncio.sleep(0.1)  # 100ms interval between calls
        
    await client.close()
    
    if latencies:
        arr = np.array(latencies)
        print("-" * 65)
        print("📊 [벤치마크 결과 통계]")
        print(f"  - 총 테스트 횟수: {iterations}회 (성공: {success_count}, 폴백: {fallback_count})")
        print(f"  - 최소 레이턴시 (Min):    {np.min(arr):.2f} ms")
        print(f"  - 평균 레이턴시 (Mean):   {np.mean(arr):.2f} ms")
        print(f"  - 중앙값 (Median / p50): {np.median(arr):.2f} ms")
        print(f"  - 95 백분위수 (p95):      {np.percentile(arr, 95):.2f} ms")
        print(f"  - 99 백분위수 (p99):      {np.percentile(arr, 99):.2f} ms")
        print(f"  - 최대 레이턴시 (Max):    {np.max(arr):.2f} ms")
        
        if np.mean(arr) < 300.0:
            print("  🎯 [판정] 목표치 달성! (평균 레이턴시 < 300ms 통과)")
        else:
            print("  ⚠️ [판정] 평균 레이턴시 300ms 초과. 네트워크 경로 또는 연결 풀 최적화 검토 필요.")
    print("=" * 60 + "\n")


async def main():
    print("🚀 Jev AI 초단타 모듈 시스템 벤치마크 시작")
    await benchmark_okx_ws()
    await benchmark_typesafe_api(iterations=10)


if __name__ == "__main__":
    asyncio.run(main())
