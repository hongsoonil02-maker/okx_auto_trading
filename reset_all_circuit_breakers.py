#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reset_all_circuit_breakers.py
3개 계정(메인, 서브1, 서브2)의 실제 OKX 잔고를 각각 조회하여,
모든 서킷브레이커 상태 파일(Major, Venture)을 최신 실제 잔고 기준으로 리셋하고
주간 락(weekly_locked) 및 쿨다운을 완전 해제합니다.
"""
import os
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import ccxt.async_support as ccxt

DIRS = [
    ("/home/hongsoonil02/quant_system", "메인 계정"),
    ("/home/hongsoonil02/quant_system_20x", "서브계정 1"),
    ("/home/hongsoonil02/quant_system_dontworry", "서브계정 2")
]

async def get_balance(env_path):
    load_dotenv(env_path, override=True)
    api_key = os.getenv("OKX_API_KEY")
    secret = os.getenv("OKX_SECRET") or os.getenv("OKX_SECRET_KEY")
    password = os.getenv("OKX_PASSPHRASE")
    
    ex = ccxt.okx({
        'apiKey': api_key,
        'secret': secret,
        'password': password,
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })
    try:
        bal = await ex.fetch_balance()
        usdt = float(bal.get('USDT', {}).get('total', 0) or bal.get('total', {}).get('USDT', 0) or 0)
        await ex.close()
        return usdt
    except Exception as e:
        print(f"잔고 조회 실패 ({env_path}): {e}")
        await ex.close()
        return None

async def main():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    current_week = f"{datetime.utcnow().isocalendar()[0]}-W{datetime.utcnow().isocalendar()[1]}"
    print("=" * 70)
    print(f"🔄 3개 계정 서킷브레이커 현실화 리셋 시작 (일자: {today}, 주차: {current_week})")
    print("=" * 70)

    for dir_path, label in DIRS:
        env_file = os.path.join(dir_path, ".env")
        equity = await get_balance(env_file)
        if not equity or equity <= 0:
            print(f"❌ [{label}] 잔고를 가져올 수 없어 스킵합니다.")
            continue

        state_dir = os.path.join(dir_path, "state")
        os.makedirs(state_dir, exist_ok=True)
        
        target_files = [
            "circuit_breaker_OKX_Major_Strategy.json",
            "circuit_breaker_OKX_Venture.json"
        ]

        print(f"\n📌 [{label}] 디렉터리: {dir_path}")
        print(f"   💰 실제 확인 잔고: {equity:,.2f} USDT")

        for fname in target_files:
            fpath = os.path.join(state_dir, fname)
            new_state = {
                "date": today,
                "anchor_equity": equity,
                "tripped": False,
                "week": current_week,
                "weekly_peak": equity,
                "cooldown_until": 0,
                "weekly_locked": False
            }
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(new_state, f)
            print(f"   ✅ 리셋 완료: {fname} (anchor={equity:.2f}, weekly_locked=False)")

    print("\n" + "=" * 70)
    print("🎉 3개 계정의 모든 서킷브레이커가 실제 잔고 기준으로 완벽히 초기화되었습니다!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
