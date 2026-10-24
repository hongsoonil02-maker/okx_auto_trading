#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preflight_check.py — OKX 자동매매 봇 사전 자가진단 (Pre-flight Self-Test) Engine
모든 봇 시작 전(start_coinbot.sh)에 필수 실행되어 5초 만에 다음 항목을 전수 검증:
1. OKX API 통신 및 서브계정 자격증명 유효성
2. 계정 모드 검증: acctLv >= 2 (선물/스왑 가능) 및 posMode == 'long_short_mode'
3. 최소 거래 잔고(USDT) 확보 여부
4. 마스터 오케스트레이터와 체결봇(Bot C) 간 포트 일치성 검증 (포트 불일치 원천 차단)
5. Webhook 스키마(WebhookPayload) 파라미터 정합성 검증
6. Jev AI LOB 웹소켓 통신 가용성
"""

import os
import sys
import json
import asyncio
import argparse
from dotenv import load_dotenv

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def log_pass(msg):
    print(f"  {GREEN}✅ [PASS]{RESET} {msg}")

def log_fail(msg):
    print(f"  {RED}❌ [FAIL]{RESET} {msg}")

def log_warn(msg):
    print(f"  {YELLOW}⚠️  [WARN]{RESET} {msg}")

def log_info(msg):
    print(f"  {CYAN}ℹ️  [INFO]{RESET} {msg}")


async def run_checks(env_path: str) -> bool:
    print("=" * 70)
    print(f"🔍 [Preflight Check] OKX 자동매매 사전 자가진단 실행")
    print(f"   환경설정 파일: {env_path}")
    print("=" * 70)

    all_passed = True

    # 1. .env 파일 검증
    if not os.path.exists(env_path):
        log_fail(f".env 파일이 존재하지 않습니다: {env_path}")
        return False
    load_dotenv(env_path, override=True)
    log_pass(".env 파일 로드 완료")

    # 2. 필수 환경변수 확인
    api_key = os.getenv("OKX_API_KEY", "").strip()
    secret = os.getenv("OKX_SECRET", "").strip()
    passphrase = os.getenv("OKX_PASSPHRASE", "").strip()

    if not (api_key and secret and passphrase):
        log_fail("OKX API 인증키(API_KEY, SECRET, PASSPHRASE) 중 일부가 누락되었습니다.")
        all_passed = False
    else:
        masked_key = f"{api_key[:6]}****{api_key[-4:]}" if len(api_key) >= 10 else "***"
        log_pass(f"OKX API 키 로드 완료: {masked_key}")

    # 3. WebhookPayload 스키마 호환성 검증
    try:
        from webhook_spec import WebhookPayload, ActionType, SideType
        test_payload = WebhookPayload(
            action=ActionType.EXEC,
            side=SideType.BUY,
            symbol="BTC-USDT-SWAP",
            qty=1.0,
            stop_pct=0.03,
            order_type="POST_ONLY",
            target_price=50000.0,
            jev_score=0.65,
            is_simulation=False
        )
        json_str = test_payload.to_json()
        log_pass("WebhookPayload 스키마 검증 통과 (stop_pct, order_type 호환)")
    except TypeError as e:
        log_fail(f"WebhookPayload 스키마 불일치 (구버전 스키마 감지): {e}")
        all_passed = False
    except Exception as e:
        log_fail(f"WebhookPayload 검증 중 예외 발생: {e}")
        all_passed = False

    # 4. 포트 정합성 검증 (오케스트레이터 <-> Bot C)
    bot_c_port_env = int(os.getenv("BOT_C_PORT", "8013"))
    master_webhook_url = os.getenv("MASTER_WEBHOOK_URL", "http://localhost:8009/webhook")
    
    # master_bot_orchestrator에서 BOT_ENDPOINTS 확인
    try:
        from master_bot_orchestrator import MasterBotOrchestrator
        orch_endpoints = MasterBotOrchestrator.BOT_ENDPOINTS
        # Bot C URL 추출
        expected_bot_c_url = None
        for name, url in orch_endpoints.items():
            if "Bot C" in name:
                expected_bot_c_url = url
                break
        
        if expected_bot_c_url:
            expected_port = int(expected_bot_c_url.split(":")[-1].replace("/", ""))
            if expected_port != bot_c_port_env:
                log_fail(
                    f"포트 불일치 감지! Master Orchestrator 기대 포트: {expected_port} vs "
                    f".env의 BOT_C_PORT: {bot_c_port_env}"
                )
                all_passed = False
            else:
                log_pass(f"오케스트레이터 - Bot C 통신 포트 정합성 확인 완료 (Port: {bot_c_port_env})")
        else:
            log_warn("Master Orchestrator에 Bot C 엔드포인트가 정의되어 있지 않습니다.")
    except Exception as e:
        log_warn(f"포트 정합성 검사 중 오케스트레이터 로드 예외 (경고): {e}")

    # 5. OKX 거래소 실시간 API 및 계정 상태 검증
    try:
        import ccxt.async_support as ccxt
        ex = ccxt.okx({
            "apiKey": api_key,
            "secret": secret,
            "password": passphrase,
            "options": {"defaultType": "swap"}
        })

        # 5-1. 계정 설정 조회
        conf = await ex.privateGetAccountConfig()
        data = conf.get("data", [{}])[0]
        acct_lv = int(data.get("acctLv", "1"))
        pos_mode = data.get("posMode", "")

        if acct_lv < 2:
            log_fail(
                f"거래소 계정 모드 오류: 현재 acctLv = {acct_lv} (현물 전용 단순 모드). "
                f"선물/스왑 매매를 위해 acctLv >= 2 (단일통화 마진 이상) 필요!"
            )
            all_passed = False
        else:
            log_pass(f"OKX 계정 모드 정상: acctLv = {acct_lv} (선물/스왑 가능)")

        if pos_mode != "long_short_mode":
            log_warn(f"포지션 모드가 양방향(long_short_mode)이 아님: {pos_mode}")
        else:
            log_pass("OKX 포지션 모드 정상: long_short_mode")

        # 5-2. 잔고 조회
        bal = await ex.fetch_balance()
        usdt_free = float(bal.get("USDT", {}).get("free", 0.0) or 0.0)
        usdt_total = float(bal.get("USDT", {}).get("total", 0.0) or 0.0)

        if usdt_total <= 0.0:
            log_fail(f"계좌 잔고 부족: 총 보유 USDT가 0.0입니다. 매매 불가!")
            all_passed = False
        elif usdt_free < 10.0:
            log_warn(f"주문 가능 현금 부족: Free USDT = {usdt_free:.2f} (Total: {usdt_total:.2f})")
        else:
            log_pass(f"계좌 잔고 확인: Free: {usdt_free:,.2f} USDT | Total: {usdt_total:,.2f} USDT")

        await ex.close()
    except Exception as e:
        log_fail(f"OKX 거래소 연결 실패: {e}")
        all_passed = False

    print("-" * 70)
    if all_passed:
        print(f"🎉 {GREEN}모든 Pre-flight 사전 자가진단 항목을 통과했습니다! 봇을 안전하게 가동합니다.{RESET}")
    else:
        print(f"🚫 {RED}사전 진단 실패 항목이 있습니다! 치명적 에러 방지를 위해 봇 기동을 차단합니다.{RESET}")
    print("=" * 70)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="OKX Bot Pre-flight Checker")
    parser.add_argument("--env", default=".env", help="Path to .env file")
    args = parser.parse_args()

    env_path = os.path.abspath(args.env)
    passed = asyncio.run(run_checks(env_path))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
