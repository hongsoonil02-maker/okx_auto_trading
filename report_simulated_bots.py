#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report_simulated_bots.py — 토리(Tori) 및 카메론(Cameron) 모의매매 현황 및 성과 통합 리포터
"""

import os
import json
from datetime import datetime

QUANT_DIR = "/home/hongsoonil02/quant_system"
TORI_LOG = os.path.join(QUANT_DIR, "kis_tori_kr_trader.log")
TORI_POS = os.path.join(QUANT_DIR, "kis_tori_kr_positions.json")
PENNY_LOG = os.path.join(QUANT_DIR, "kis_cameron_penny.log")
PENNY_POS = os.path.join(QUANT_DIR, "state", "kis_cameron_penny_positions.json")
TORI_US_LOG = os.path.join(QUANT_DIR, "kis_tori_swing_trader.log")


def load_json(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def tail_log(filepath, lines=5):
    if not os.path.exists(filepath):
        return ["(로그 파일 없음)"]
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return [l.strip() for l in f.readlines()[-lines:] if l.strip()]


def print_report():
    print("=" * 75)
    print(f"📊 [모의매매 통합 현황 보고서] - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)

    # 1. 토리(Tori) 국내주식 모의매매
    print("\n🟢 [1] 토리(Tori) 국내 추세선 스윙 (KODEX레버리지, SK하이닉스)")
    print("   • 실행 주기: 매일 09:01, 15:20 KST (kis_tori_kr.timer)")
    tori_pos = load_json(TORI_POS)
    if tori_pos:
        print("   • 📌 현재 보유 포지션:")
        for code, p in tori_pos.items():
            print(f"     - {p.get('name', code)}({code}): {p.get('shares')}주 @ {p.get('entry_price', 0):,.0f}원 (세이프티: {p.get('safety_line', 0):,.0f}원)")
    else:
        print("   • 📌 현재 보유 포지션: 없음 (100% 현금 관망 중)")
    
    print("   • 📜 최근 평가 로그:")
    for l in tail_log(TORI_LOG, 4):
        print(f"     {l}")

    # 2. 카메론(Cameron) 국내 동전주 모의매매
    print("\n🟢 [2] 카메론(Cameron) 국내 동전주 30분봉 풀백 (덕성, 모비스 등 6종목)")
    print("   • 실행 주기: 장중 매 30분 (kis_cameron_penny.timer)")
    penny_pos = load_json(PENNY_POS)
    if penny_pos:
        print("   • 📌 현재 보유 포지션:")
        for code, p in penny_pos.items():
            print(f"     - {p.get('name', code)}({code}): {p.get('qty')}주 @ {p.get('entry_price', 0):,.0f}원 | SL: {p.get('stop_loss', 0):,.0f}원 | TP: {p.get('take_profit', 0):,.0f}원")
    else:
        print("   • 📌 현재 보유 포지션: 없음 (진입 대기 중)")

    print("   • 📜 최근 체결/손절 히스토리 & 평가 로그:")
    for l in tail_log(PENNY_LOG, 5):
        print(f"     {l}")

    # 3. 토리(Tori) 해외주식 모의매매
    print("\n🟢 [3] 토리(Tori) 미국주식 60분봉 스윙 (SPY, NVDA)")
    print("   • 실행 주기: 매 1시간 (kis_tori_swing.timer)")
    print("   • 📜 최근 평가 로그:")
    for l in tail_log(TORI_US_LOG, 3):
        print(f"     {l}")

    print("\n" + "=" * 75)
    print("💡 터미널 확인 명령어:")
    print("   • 통합 현황: python3 /home/hongsoonil02/quant_system/report_simulated_bots.py")
    print("   • 카메론 실시간 로그: tail -f /home/hongsoonil02/quant_system/kis_cameron_penny.log")
    print("   • 토리 실시간 로그: tail -f /home/hongsoonil02/quant_system/kis_tori_kr_trader.log")
    print("=" * 75)


if __name__ == "__main__":
    print_report()
