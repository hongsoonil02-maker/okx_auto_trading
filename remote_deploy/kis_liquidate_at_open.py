# -*- coding: utf-8 -*-
"""
kis_liquidate_at_open.py — KIS 미국주식 장초반(09:30 ET) 일괄 전량 청산 스크립트

동작 절차:
1. 미국 정규장 개장(09:30:05 ET)까지 대기 (또는 --now 플래그 시 즉시 실행)
2. 미체결 주문 전량 취소
3. 보유 해외주식 전 종목 조회 (TTTS3012R)
4. 체결우선 지정가(-1.0%)로 전량 매도 (TTTT1006U)
5. 포지션 상태 파일(kis_nasdaq_jev_positions.json, kis_qqq_regime_state.json) 초기화
6. 결과 텔레그램 및 로그 알림
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, ".."))
load_dotenv(os.path.join(CURRENT_DIR, "..", ".env"))

TOKEN_FILE = os.path.join(CURRENT_DIR, "kis_token.json")
LOG_FILE = os.path.join(CURRENT_DIR, "kis_liquidate_at_open.log")
JEV_POS_FILE = os.path.join(CURRENT_DIR, "kis_nasdaq_jev_positions.json")
QQQ_STATE_FILE = os.path.join(CURRENT_DIR, "kis_qqq_regime_state.json")

logger = logging.getLogger("KisLiquidator")
logger.setLevel(logging.INFO)
logger.propagate = False

_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setFormatter(_formatter)
logger.addHandler(_fh)

_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_formatter)
logger.addHandler(_sh)

session = requests.Session()
_adapter = HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
                                         allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]))
session.mount("https://", _adapter)

KIS_APP_KEY = os.getenv("KIS_API_KEY", "")
KIS_APP_SECRET = os.getenv("KIS_SECRET", "")
_acc = os.getenv("KIS_ACCOUNT_NO", "")
KIS_CANO, KIS_PRDT_ABRV = _acc.split("-") if "-" in _acc else (_acc[:8], _acc[8:10])
KIS_URL = os.getenv("KIS_URL", "https://openapi.koreainvestment.com:9443")


def send_alert(msg: str):
    logger.info(msg)
    try:
        from utils_telegram import send_telegram_alert
        send_telegram_alert(msg)
    except Exception as e:
        logger.warning(f"⚠️ Telegram 실패: {e}")


class KisLiquidator:
    def __init__(self):
        self.access_token = ""
        self.token_expired_at = 0.0

    def issue_token(self) -> bool:
        now = time.time()
        if self.access_token and now < self.token_expired_at:
            return True
        if os.path.exists(TOKEN_FILE):
            try:
                with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                    d = json.load(f)
                if d.get("access_token") and d.get("expires_at", 0) > now + 300:
                    self.access_token, self.token_expired_at = d["access_token"], d["expires_at"]
                    return True
            except Exception:
                pass
        try:
            r = session.post(
                f"{KIS_URL}/oauth2/tokenP",
                json={"grant_type": "client_credentials", "appkey": KIS_APP_KEY, "appsecret": KIS_APP_SECRET},
                timeout=10
            )
            if r.status_code == 200 and "access_token" in r.json():
                d = r.json()
                self.access_token = d["access_token"]
                self.token_expired_at = now + int(d.get("expires_in", 86400)) - 600
                with open(TOKEN_FILE, "w", encoding="utf-8") as f:
                    json.dump({"access_token": self.access_token, "expires_at": self.token_expired_at}, f)
                return True
        except Exception as e:
            logger.error(f"❌ 토큰 발급 실패: {e}")
        return False

    def headers(self, tr_id: str) -> dict:
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.access_token}",
            "appkey": KIS_APP_KEY,
            "appsecret": KIS_APP_SECRET,
            "tr_id": tr_id,
            "custtype": "P"
        }

    def fetch_holdings(self) -> Dict[str, int]:
        if not self.issue_token():
            return {}
        try:
            r = session.get(
                f"{KIS_URL}/uapi/overseas-stock/v1/trading/inquire-balance",
                headers=self.headers("TTTS3012R"),
                params={"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "OVRS_EXCG_CD": "NASD",
                        "TR_CRCY_CD": "USD", "CTX_AREA_FK200": "", "CTX_AREA_NK200": ""},
                timeout=10
            )
            if r.status_code == 200 and r.json().get("rt_cd") == "0":
                held = {}
                for row in r.json().get("output1", []):
                    sym = (row.get("ovrs_pdno") or "").strip().upper()
                    try:
                        qty = int(float(row.get("ovrs_cblc_qty", 0)))
                    except (ValueError, TypeError):
                        continue
                    if sym and qty > 0:
                        held[sym] = qty
                return held
        except Exception as e:
            logger.error(f"❌ 보유 조회 실패: {e}")
        return {}

    def get_price(self, sym: str) -> Optional[float]:
        try:
            r = session.get(
                f"{KIS_URL}/uapi/overseas-price/v1/quotations/price",
                headers=self.headers("HHDFS00000300"),
                params={"AUTH": "", "EXCD": "NAS", "SYMB": sym},
                timeout=5
            )
            if r.status_code == 200:
                last = float(r.json().get("output", {}).get("last", 0.0))
                if last > 0:
                    return last
        except Exception as e:
            logger.warning(f"⚠️ [{sym}] 시세 조회 실패: {e}")
        return None

    def cancel_open_orders(self):
        """미체결 주문 조회 후 전량 취소"""
        try:
            r = session.get(
                f"{KIS_URL}/uapi/overseas-stock/v1/trading/inquire-nccs",
                headers=self.headers("TTTS3018R"),
                params={"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "OVRS_EXCG_CD": "NASD",
                        "SORT_SQN": "DS", "CTX_AREA_FK200": "", "CTX_AREA_NK200": ""},
                timeout=8
            )
            if r.status_code != 200 or r.json().get("rt_cd") != "0":
                return
            for row in r.json().get("output", []):
                odno = row.get("odno", "")
                sym = (row.get("pdno") or "").strip().upper()
                qty = row.get("ft_ord_qty", "0")
                if not odno or not sym:
                    continue
                body = {
                    "CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "OVRS_EXCG_CD": "NASD", "PDNO": sym,
                    "ORG_ODNO": str(odno), "RVSE_CNCL_DVSN_CD": "02", "ORD_QTY": str(qty),
                    "OVRS_ORD_UNPR": "0", "ORD_SVR_DVSN_CD": "0"
                }
                try:
                    session.post(f"{KIS_URL}/uapi/overseas-stock/v1/trading/order-rvsecncl",
                                 headers=self.headers("TTTT1004U"), json=body, timeout=10)
                    logger.info(f"   🗑️ 미체결 취소: {sym} ODNO {odno}")
                except Exception as e:
                    logger.warning(f"   취소 실패 {sym}: {e}")
        except Exception as e:
            logger.warning(f"⚠️ 미체결 조회 실패: {e}")

    def liquidate_all(self):
        send_alert("🚨 [KIS 미국주식 일괄 청산 시작] 보유 종목 전량 매도 진행")
        self.cancel_open_orders()
        time.sleep(1)

        holdings = self.fetch_holdings()
        if not holdings:
            send_alert("ℹ️ [KIS 청산 완료] 현재 보유 중인 미국주식이 없습니다.")
            self.clean_state_files()
            return

        send_alert(f"📋 청산 대상 보유 종목: {holdings}")

        for sym, qty in holdings.items():
            px = self.get_price(sym)
            if not px or px <= 0:
                logger.warning(f"⚠️ {sym} 실시간 시세 조회 실패, 폴백 시도")
                # 시세 재시도
                time.sleep(1)
                px = self.get_price(sym)

            if not px or px <= 0:
                logger.error(f"❌ {sym} 시세 조회 불가로 매도 실패")
                continue

            # 체결 우선을 위해 현재가 대비 -1.0% 지정가 발주 (시장가 즉시 체결)
            limit_px = round(px * 0.99, 2)
            body = {
                "CANO": KIS_CANO,
                "ACNT_PRDT_CD": KIS_PRDT_ABRV,
                "OVRS_EXCG_CD": "NASD",
                "PDNO": sym,
                "ORD_QTY": str(qty),
                "OVRS_ORD_UNPR": f"{limit_px:.2f}",
                "ORD_SVR_DVSN_CD": "0",
                "ORD_DVSN": "00"
            }
            try:
                r = session.post(f"{KIS_URL}/uapi/overseas-stock/v1/trading/order",
                                 headers=self.headers("TTTT1006U"), json=body, timeout=10)
                d = r.json()
                if r.status_code == 200 and d.get("rt_cd") == "0":
                    send_alert(f"✅ [전량 매도 접수] {sym} {qty}주 @ ${limit_px:.2f} (시장가 체결 유도)")
                else:
                    logger.error(f"❌ 매도 거부 {sym}: {d.get('msg1')}")
            except Exception as e:
                logger.error(f"❌ 매도 발주 예외 {sym}: {e}")

        time.sleep(3)
        self.clean_state_files()
        send_alert("🏁 [KIS 미국주식 청산 완료] 전량 현금화 완료 및 상태 리셋 완료")

    def clean_state_files(self):
        """기존 봇의 포지션 json 파일 초기화"""
        try:
            with open(JEV_POS_FILE, "w", encoding="utf-8") as f:
                json.dump({"positions": {}, "cooldowns": {}}, f, indent=2)
            logger.info("   🧹 JEV positions 파일 초기화 완료")
        except Exception as e:
            logger.warning(f"JEV position reset error: {e}")

        try:
            with open(QQQ_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump({"legacy_liquidate": [], "legacy_seeded": true, "open_orders": {}, "last_decision_date": "", "last_attempt_ts": 0, "last_regime": "CASH"}, f, indent=2)
            logger.info("   🧹 QQQ state 파일 초기화 완료")
        except Exception as e:
            logger.warning(f"QQQ state reset error: {e}")


def wait_until_market_open():
    """뉴욕 시간 기준 정규장 개장(09:30:05 ET)까지 대기"""
    tz_ny = ZoneInfo("America/New_York")
    logger.info("⏳ [스케줄러 가동] 뉴욕 정규장(09:30 ET / 한국 22:30 KST) 개장 대기 중...")
    while True:
        now_ny = datetime.now(tz_ny)
        # 평일 월~금 체크 (0=월, 4=금)
        if now_ny.weekday() < 5:
            # 09:30:05 ~ 15:59:00 사이이면 개장 상태
            if now_ny.hour == 9 and now_ny.minute >= 30 and now_ny.second >= 5:
                logger.info(f"🔔 [정규장 개장 감지!] EST {now_ny.strftime('%H:%M:%S')} — 일괄 청산 즉시 실행")
                break
            elif now_ny.hour > 9 and now_ny.hour < 16:
                logger.info(f"🔔 [정규장 운영 중 감지!] EST {now_ny.strftime('%H:%M:%S')} — 일괄 청산 즉시 실행")
                break
        time.sleep(5)


if __name__ == "__main__":
    liquidator = KisLiquidator()
    if "--now" in sys.argv:
        logger.info("⚡ [--now] 플래그 지정: 개장 대기 없이 즉시 청산 시도")
        liquidator.liquidate_all()
    else:
        wait_until_market_open()
        liquidator.liquidate_all()
