#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
close_selected.py — 선택 종목 시장가 청산
[1] 손실 반복 종목 9개 + [2] 48h+ 묵은 포지션 6개
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

try:
    import ccxt.async_support as ccxt_async
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt", "-q"])
    import ccxt.async_support as ccxt_async

TARGET_BASES = {
    # [1] 손실 반복 + 블랙리스트
    "PEOPLE", "XPL", "GRVT", "APR", "UB",
    "FARTCOIN", "SKDD", "STX", "CAP",
    # [2] 48h+ 묵은 포지션
    "POL", "OKB", "GIGGLE", "MOVE", "BONK", "TRB",
}

async def close_selected():
    print("=" * 55)
    print("🎯 선택 포지션 청산 스크립트")
    print(f"   대상 종목: {len(TARGET_BASES)}개")
    print("=" * 55)

    exchange = ccxt_async.okx({
        "apiKey":   os.getenv("OKX_API_KEY", ""),
        "secret":   os.getenv("OKX_SECRET", "") or os.getenv("OKX_API_SECRET", ""),
        "password": os.getenv("OKX_PASSPHRASE", "") or os.getenv("OKX_PASSWORD", ""),
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

    try:
        await exchange.load_markets()
        positions = await exchange.fetch_positions()

        to_close = []
        for pos in positions:
            symbol = pos.get("symbol", "")
            base = symbol.split("/")[0].split("-")[0]
            contracts = float(pos.get("contracts") or 0)
            side = pos.get("side", "")
            if base in TARGET_BASES and contracts > 0 and side:
                to_close.append(pos)

        if not to_close:
            print("ℹ️  청산할 포지션이 없습니다 (이미 청산됨).")
            return

        print(f"\n📋 실제 청산 대상: {len(to_close)}개\n")
        closed, failed = 0, 0

        for pos in to_close:
            symbol    = pos["symbol"]
            contracts = float(pos["contracts"])
            side      = pos["side"]
            pnl_pct   = pos.get("percentage") or 0
            close_side = "sell" if side == "long" else "buy"

            print(f"  🔪 {symbol:<30} | {side:<5} | {contracts:.4f}계약 | PnL {pnl_pct:+.2f}%")
            try:
                await exchange.create_order(
                    symbol=symbol,
                    type="market",
                    side=close_side,
                    amount=contracts,
                    params={"reduceOnly": True, "posSide": side},
                )
                print(f"     ✅ 청산 완료")
                closed += 1
            except Exception as e:
                print(f"     ❌ 청산 실패: {e}")
                failed += 1
            await asyncio.sleep(0.3)

        print()
        print("=" * 55)
        print(f"✅ 완료: 성공 {closed}건 / 실패 {failed}건")
        print("=" * 55)

    except Exception as e:
        print(f"❌ 오류: {e}")
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(close_selected())
