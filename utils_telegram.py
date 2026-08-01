# -*- coding: utf-8 -*-
"""
utils_telegram.py: 텔레그램 강제 통신 및 유효성 검증 유틸리티
"""
import os
import sys
import requests
import threading
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def validate_telegram_token():
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ [Warning] TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID가 .env에 존재하지 않습니다.")
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getMe"
    try:
        response = requests.get(url, timeout=4)
        if response.status_code != 200:
            print(f"⚠️ [Warning] 텔레그램 토큰 검증 실패 (Status: {response.status_code})")
        else:
            print("[Telegram Validation] Telegram Bot Token Validation Passed.")
    except Exception as e:
        print(f"⚠️ [Warning] 텔레그램 API 통신 오류 (검증 스킵): {e}")

def _send_sync(url, payload):
    try:
        response = requests.post(url, json=payload, timeout=4)
        response.raise_for_status()
    except Exception as e:
        print(f"🚨 [Error] 텔레그램 메시지 전송 실패: {e}")

def send_telegram_alert(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    # 비동기 이벤트 루프 락을 막기 위해 백그라운드 스레드로 전송 (Fire-and-Forget)
    threading.Thread(target=_send_sync, args=(url, payload), daemon=True).start()

# 모듈 로드 시 최초 1회 검증 수행 (네트워크 오류 시 종료 안 함)
validate_telegram_token()

