#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_lead_bot_status.py
메인 계정의 OKX 카피트레이딩 리드 트레이더 설정 및 거래소 봇 상태 조회
"""
import os
import json
import asyncio
from dotenv import load_dotenv
import ccxt.async_support as ccxt_async

async def main():
    load_dotenv("/home/hongsoonil02/quant_system/.env", override=True)
    api_key = os.getenv("OKX_API_KEY", "")
    secret = os.getenv("OKX_SECRET", "") or os.getenv("OKX_SECRET_KEY", "")
    password = os.getenv("OKX_PASSPHRASE", "")

    ex = ccxt_async.okx({
        "apiKey": api_key,
        "secret": secret,
        "password": password,
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

    print("=" * 60)
    print("🔍 OKX 메인 계정 Lead / Copy Trading & Trading Bot 상태 확인")
    print("=" * 60)

    # 1. 카피트레이딩 관련 엔드포인트 확인
    endpoints = [
        ("GET", "copytrading/current-lead-traders", {"instType": "SWAP"}),
        ("GET", "copytrading/current-subpositions", {}),
        ("GET", "copytrading/lead-trader-status", {}),
        ("GET", "copytrading/copy-settings", {}),
        ("GET", "tradingBot/grid/orders-algo-list", {"algoOrdType": "grid"}),
        ("GET", "tradingBot/recurring/orders-algo-list", {}),
    ]

    for method, path, params in endpoints:
        try:
            res = await ex.request(path, api="private", method=method, params=params)
            code = res.get("code")
            msg = res.get("msg")
            data = res.get("data", [])
            print(f"\n📌 {path}: code={code}, msg='{msg}', data count={len(data)}")
            if data:
                print(f"   내용: {json.dumps(data, indent=2, ensure_ascii=False)[:300]}...")
        except Exception as e:
            print(f"\n📌 {path}: 호출 실패 ({e})")

    await ex.close()

if __name__ == "__main__":
    asyncio.run(main())
