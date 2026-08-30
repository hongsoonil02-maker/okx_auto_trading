#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cut_losers_all_accounts.py
alv*** 벤치마킹 실행 도구:
- 3개 계정(메인, 서브1, 서브2)의 모든 포지션을 조회
- 손실 중인 포지션(PnL < 0%, 특히 -8% 이상 물린 좀비 포지션)을 시장가로 즉시 칼손절
- 수익 중인 포지션(PnL >= 0%: ZEC, OKB, PUMP, LAB, TURBO 등)은 샹들리에 트레일링을 위해 절대 청산하지 않고 보존
"""
import os
import sys
import asyncio
from dotenv import load_dotenv
import ccxt.async_support as ccxt_async

DIRS = [
    ("/home/hongsoonil02/quant_system", "메인 계정"),
    ("/home/hongsoonil02/quant_system_20x", "서브계정 1 (20x)"),
    ("/home/hongsoonil02/quant_system_dontworry", "서브계정 2 (dontworry)")
]

async def cut_losers_for_account(dir_path, label):
    load_dotenv(os.path.join(dir_path, ".env"), override=True)
    api_key = os.getenv("OKX_API_KEY", "")
    secret = os.getenv("OKX_SECRET", "") or os.getenv("OKX_SECRET_KEY", "")
    password = os.getenv("OKX_PASSPHRASE", "")

    exchange = ccxt_async.okx({
        "apiKey": api_key,
        "secret": secret,
        "password": password,
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

    print("\n" + "=" * 65)
    print(f"🔍 [{label}] 포지션 점검 및 손실 종목 칼손절 실행")
    print("=" * 65)

    try:
        await exchange.load_markets()
        positions = await exchange.fetch_positions()
        active = [p for p in positions if float(p.get('contracts') or 0) > 0]

        if not active:
            print("   ℹ️ 보유 중인 포지션이 없습니다.")
            return

        closed_cnt = 0
        kept_cnt = 0

        for pos in active:
            symbol = pos["symbol"]
            contracts = float(pos["contracts"])
            side = pos["side"]
            pnl_pct = float(pos.get("percentage") or 0)
            close_side = "sell" if side == "long" else "buy"

            if pnl_pct < 0:
                # 손실 포지션: alv*** 원칙에 따라 시장가 즉시 칼손절
                print(f"   🔪 [칼손절 대상] {symbol:<26} | {side:<5} | {contracts:.4f}계약 | PnL {pnl_pct:+.2f}%")
                try:
                    await exchange.create_order(
                        symbol=symbol,
                        type="market",
                        side=close_side,
                        amount=contracts,
                        params={"reduceOnly": True, "posSide": side},
                    )
                    print(f"      ✅ 시장가 손절 완료 (증거금 회수)")
                    closed_cnt += 1
                except Exception as e:
                    print(f"      ❌ 손절 주문 실패: {e}")
                await asyncio.sleep(0.3)
            else:
                # 수익 포지션: 샹들리에 트레일링을 위해 유지
                print(f"   💎 [수익 보존] {symbol:<26} | {side:<5} | {contracts:.4f}계약 | PnL {pnl_pct:+.2f}% (빅스윙 홀딩)")
                kept_cnt += 1

        print(f"\n   📊 [{label} 결과]: 손실 정리 {closed_cnt}건 / 수익 보존 {kept_cnt}건")

    except Exception as e:
        print(f"   ❌ 계정 처리 중 오류: {e}")
    finally:
        await exchange.close()

async def main():
    print("=" * 70)
    print("🚀 3개 계정 손실 포지션 칼손절 & 수익 포지션 보존 실행")
    print("=" * 70)
    for dir_path, label in DIRS:
        await cut_losers_for_account(dir_path, label)
    print("\n" + "=" * 70)
    print("🎉 모든 손실 포지션 정리가 완료되었습니다. 수익 포지션은 트레일링으로 계속 달립니다!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
