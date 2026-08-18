#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
manage_copy_trading.py — OKX 카피 트레이딩 간편 제어 및 진단 유틸리티

사용법:
  python3 manage_copy_trading.py --status   # 현재 카피 트레이딩 상태 진단
  python3 manage_copy_trading.py --start    # .env 설정에 따라 리드 트레이더 카피 시작/변경
  python3 manage_copy_trading.py --stop     # 카피 트레이딩 일시 중단 및 청산
"""

import os
import sys
import json
import argparse
import logging
from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("CopyTradingCLI")

try:
    from okx_copy_engine import OKXCopyEngine, get_copy_engine
except ImportError as e:
    logger.critical(f"❌ okx_copy_engine.py 임포트 에러: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="OKX 카피 트레이딩 관리 CLI")
    parser.add_argument("--status", action="store_true", help="현재 카피 상태 및 서브포지션 진단")
    parser.add_argument("--start", action="store_true", help="리드 트레이더 카피 트레이딩 개시 또는 설정 변경")
    parser.add_argument("--auto", action="store_true", help="수익률 1등 트레이더 자동 선정 후 카피 개시")
    parser.add_argument("--top", action="store_true", help="수익률 1등 후보만 조회(카피 개시 안 함)")
    parser.add_argument("--stop", action="store_true", help="카피 트레이딩 중단 및 카피 포지션 청산")
    parser.add_argument("--code", type=str, help="리드 트레이더 Unique Code (미지정 시 .env 기본값 사용)")
    parser.add_argument("--amt", type=str, default="800", help="할당 총 자본금 USDT (기본 800)")
    parser.add_argument("--ratio", type=str, default="1", help="비율 카피 비율 (기본 1 = 동일 비율)")
    parser.add_argument("--mode", type=str, default=os.getenv("OKX_COPY_MODE", "SMART_COPY"), help="카피 모드 (SMART_COPY, ratio_copy)")

    args = parser.parse_args()

    ce = get_copy_engine()
    if not ce.copy_api:
        logger.error("❌ OKX API 클라이언트가 초기화되지 않았습니다. .env 자격증명을 확인하세요.")
        sys.exit(1)

    target_codes_str = os.getenv("OKX_LEAD_TRADER_UNIQUE_CODES")
    if args.code:
        target_codes = [args.code]
    elif target_codes_str:
        target_codes = [c.strip() for c in target_codes_str.split(",") if c.strip()]
    else:
        target_codes = [ce.unique_code]

    if args.top:
        top = ce.select_top_trader()
        if top:
            print(json.dumps({
                "uniqueCode": top.get("uniqueCode"),
                "nickName": top.get("nickName"),
                "pnlRatio": top.get("pnlRatio"),
                "winRatio": top.get("winRatio"),
                "aum": top.get("aum"),
                "leadDays": top.get("leadDays"),
                "copyTraderNum": top.get("copyTraderNum"),
                "maxCopyTraderNum": top.get("maxCopyTraderNum"),
            }, indent=2, ensure_ascii=False))
        else:
            logger.error("❌ 수익률 1등 후보 선정 실패")

    elif args.auto:
        logger.info(f"🏆 수익률 1등 자동 선정 및 카피 개시 (자본=${args.amt}, 비율={args.ratio})...")
        res = ce.auto_copy_top_trader(copy_total_amt=args.amt, copy_ratio=args.ratio)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        print("\n[상태 진단 결과]")
        print(json.dumps(ce.diagnose_copy_status(), indent=2, ensure_ascii=False))

    elif args.start:
        for t_code in target_codes:
            logger.info(f"🚀 리드 트레이더({t_code}) 카피 트레이딩 설정 시작 (모드={args.mode}, 자본=${args.amt}, 비율={args.ratio})...")
            res = ce.setup_or_update_copy_trading(
                unique_code=t_code,
                copy_mode=args.mode,
                copy_ratio=args.ratio,
                copy_total_amt=args.amt,
                copy_inst_id_type="copy",
                sub_pos_close_type="copy_close"
            )
            print(json.dumps(res, indent=2, ensure_ascii=False))
        print("\n[상태 진단 결과]")
        print(json.dumps(ce.diagnose_copy_status(), indent=2, ensure_ascii=False))

    elif args.stop:
        for t_code in target_codes:
            logger.info(f"🛑 리드 트레이더({t_code}) 카피 트레이딩 중단 요청...")
            res = ce.stop_copy_trading(unique_code=t_code, sub_pos_close_type="market_close")
            if res:
                logger.info(f"✅ {t_code} 중단 성공!")
            else:
                logger.error(f"❌ {t_code} 중단 실패")

    else:
        # 기본 동작 또는 --status: 상태 진단
        report = ce.diagnose_copy_status()
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
