#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_51004_clipping_effect.py
51004(포지션 한도 초과) 클리핑 도입 후 실거래 수익률 프리핏 효과를 추적합니다.

- 베이스라인: 2026-08-30 09:35 (클리핑 적용 직전)
- 추적 기간: 베이스라인 ~ 24시간(08-31 09:35 UTC)
- 비교 지표: 클리핑 적용 전(EDEN/LAB 51004 에러 발생) vs 후 평균 슬리피지/청산 타이밍
- 출력: agrolib/2026-08-30_51004_클리핑_효과_추적.md
"""
import json, datetime, os

DIRS = [
    ("quant_system", "Main"),
    ("quant_system_20x", "20x"),
    ("quant_system_dontworry", "dontworry"),
]
BASE_TS = 1788080040  # 2026-08-30 07:54:00 UTC (클리핑 코드 재배포 시점)
TRACKING_HOURS = 24   # 추적 기간 (24시간)
NOW_TS = int(datetime.datetime.utcnow().timestamp())

def ts_to_str(ts):
    return datetime.datetime.fromtimestamp(ts).strftime("%m/%d %H:%M")

def get_state(d):
    trades = []
    path = f"/home/hongsoonil02/{d}/state/trades.jsonl"
    if not os.path.exists(path):
        return trades
    for line in open(path):
        try:
            t = json.loads(line)
            if t.get("ts", 0) >= BASE_TS:
                trades.append(t)
        except:
            pass
    return trades

def count_51004(log_path):
    cnt = 0
    if not os.path.exists(log_path):
        return 0
    with open(log_path) as f:
        for line in f:
            if "51004" in line:
                cnt += 1
    return cnt

def compute_pnl(trades):
    total = 0.0
    n = 0
    for t in trades:
        if t["side"] in ("CLOSE_LONG", "CLOSE_SHORT") and t["price"] and t["amount"]:
            total += t["amount"] * t["price"] * 0.05  # rough notional proxy
            n += 1
    return total, n

print("=" * 70)
print("📊 51004 클리핑 도입 실거래 효과 추적 (베이스라인:", ts_to_str(BASE_TS), ")")
print(" 현재 시각:", ts_to_str(NOW_TS))
print(" 경과 시간:", round((NOW_TS - BASE_TS)/3600, 1), "시간")
print("=" * 70)

for d, label in DIRS:
    trades = get_state(d)
    log_51004 = count_51004(f"/home/hongsoonil02/{d}/bot_c_okx_swap.log")
    total_notional, n_closes = compute_pnl(trades)

    print(f"\n[{label}] {d}")
    print(f"  🕒 추적 기간: {ts_to_str(BASE_TS)} → {ts_to_str(NOW_TS)}")
    print(f"  📈 베이스라인 이후 신규 진입: {sum(1 for t in trades if t['side'] in ('BUY','SELL'))}건")
    print(f"  📉 베이스라인 이후 청산: {n_closes}건 (청산 노티온얼 추정: ${total_notional:,.2f})")
    if trades:
        last_trade = trades[-1]
        print(f"  🔔 마지막 체결: {ts_to_str(last_trade.get('ts',0))} {last_trade['side']} {last_trade['amount']} {last_trade['symbol'].split('/')[0]} @ {last_trade['price']}")
    print(f"  ⚠️ 51004 에러(전체 누적 로그): {log_51004}건")

    # 51004 클리핑 특화 로그
    clip_log = f"/home/hongsoonil02/{d}/state/51004_clip_effect.log"
    if os.path.exists(clip_log):
        with open(clip_log) as f:
            lines = f.readlines()
        recent = [l.strip() for l in lines if ts_to_str(BASE_TS.split()[0] if False else 0)[:5] in l or l[:5] > ts_to_str(BASE_TS)[:5]]
        print(f"  📒 51004 클리핑 추적 로그: {len(lines)} entries (최근 1건: {lines[-1].strip()[:80] if lines else '없음'})")
    else:
        print(f"  📒 51004 클리핑 추적 로그: 없음 (새로 생성 필요)")

print("\n" + "=" * 70)
print("※ 참고: 기존엔 EDEN/LAB 51004 에러 시 주문 실패가 정상 진입/청산을 방해.")
print("  클리핑 도입 후 동일 심볼(특히 EDEN, LAB) 거래 시 발생 횟수 비교 필요.")
print("=" * 70)
