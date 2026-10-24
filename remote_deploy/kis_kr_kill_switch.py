# -*- coding: utf-8 -*-
"""
kis_kr_kill_switch.py — KIS 국내주식 슬리브 킬스위치 (Account-Level Circuit Breaker)

감시 범위: 국내주식 계좌 총평가액(KRW, TTTC8434R tot_evlu_amt)
발동 조건 (둘 중 하나):
  - 주간 드로다운: 현재 평가 < 7일 전 × (1 - WEEKLY_DD_PCT)
  - 일간 드로다운: 현재 평가 < 24시간 전 × (1 - DAILY_DD_PCT)
발동 시:
  1. 국내주식 전 종목 시장가 매도 (장 마감 시 휴지 중 반복 재시도)
  2. kis_trend_trader.service 정지
  3. 48시간 휴지 후 서비스 재개 + 기준선 리셋
※ 텔레그램 알림 없음 (사용자 요청), 로그만 기록
테스트: python3 kis_kr_kill_switch.py --once --dry-run
"""

import os
import sys
import json
import time
import logging
import subprocess
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, "..", ".env"))

HISTORY_FILE = os.path.join(CURRENT_DIR, "kis_kr_equity_history.json")
HALT_FILE = os.path.join(CURRENT_DIR, "kis_kr_kill_switch_halt.json")
TOKEN_FILE = os.path.join(CURRENT_DIR, "kis_token.json")
LOG_FILE = os.path.join(CURRENT_DIR, "kis_kr_kill_switch.log")

LOOP_SEC = 60
RECORD_EVERY_SEC = 600
WEEKLY_DD_PCT = 0.15
DAILY_DD_PCT = 0.10
WEEK_SEC = 7 * 86400
DAY_SEC = 86400
REF_TOLERANCE_SEC = 12 * 3600
COOLDOWN_SEC = 48 * 3600
FLATTEN_RETRY_SEC = 1800   # 장 마감으로 매도 불가 시 30분마다 재시도

MANAGED_SERVICES = ["kis_trend_trader.service"]
UNSELLABLE = {"007630", "259290"}  # 상폐/거래불가 (폴루스바이오팜, 폴루스) — 매도 시도 제외

logger = logging.getLogger("KisKrKillSwitch")
logger.setLevel(logging.INFO)
logger.propagate = False
_fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_fh)
if sys.stdout.isatty():
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
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


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


class KisKrKillSwitch:
    def __init__(self):
        self.access_token = ""
        self.token_expired_at = 0.0
        self.history = load_json(HISTORY_FILE, [])
        self.halt = load_json(HALT_FILE, None)
        self.last_record = 0.0
        self.last_flatten_try = 0.0

    def issue_token(self) -> bool:
        now = time.time()
        if self.access_token and now < self.token_expired_at:
            return True
        if os.path.exists(TOKEN_FILE):
            d = load_json(TOKEN_FILE, {})
            if d.get("access_token") and d.get("expires_at", 0) > now + 300:
                self.access_token, self.token_expired_at = d["access_token"], d["expires_at"]
                return True
        try:
            r = session.post(f"{KIS_URL}/oauth2/tokenP", json={"grant_type": "client_credentials",
                             "appkey": KIS_APP_KEY, "appsecret": KIS_APP_SECRET}, timeout=10)
            if r.status_code == 200 and "access_token" in r.json():
                d = r.json()
                self.access_token = d["access_token"]
                self.token_expired_at = now + int(d.get("expires_in", 86400)) - 600
                return True
        except Exception as e:
            logger.error(f"❌ 토큰 발급 실패: {e}")
        return False

    def headers(self, tr_id: str) -> dict:
        return {"content-type": "application/json; charset=utf-8",
                "authorization": f"Bearer {self.access_token}",
                "appkey": KIS_APP_KEY, "appsecret": KIS_APP_SECRET, "tr_id": tr_id}

    def fetch_account(self):
        """(총평가액 KRW, 보유 {종목코드: 수량}) 또는 (None, {}) — 조회 실패 구분."""
        if not self.issue_token():
            return None, {}
        try:
            r = session.get(f"{KIS_URL}/uapi/domestic-stock/v1/trading/inquire-balance",
                            headers=self.headers("TTTC8434R"),
                            params={"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "AFHR_FLPR_YN": "N",
                                    "OFL_YN": "", "INQR_DVSN": "02", "UNPR_DVSN": "01",
                                    "FUND_STTL_ICLD_YN": "N", "FNCG_AMT_AUTO_RDPT_YN": "N",
                                    "PRCS_DVSN": "00", "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""}, timeout=8)
            d = r.json()
            if r.status_code == 200 and d.get("rt_cd") == "0":
                equity = 0.0
                out2 = d.get("output2", [])
                if out2:
                    equity = float(out2[0].get("tot_evlu_amt", 0.0))
                held = {}
                for row in d.get("output1", []):
                    sym = (row.get("pdno") or "").strip()
                    try:
                        qty = int(float(row.get("hldg_qty", 0)))
                    except (ValueError, TypeError):
                        continue
                    if sym and qty > 0 and sym not in UNSELLABLE:
                        held[sym] = qty

                # 통합증거금/외화대용 주문가능금액(TTTC8908R) 반영
                try:
                    r_psbl = session.get(f"{KIS_URL}/uapi/domestic-stock/v1/trading/inquire-psbl-order",
                                         headers=self.headers("TTTC8908R"),
                                         params={"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "PDNO": "005930",
                                                 "ORD_UNPR": "0", "ORD_DVSN": "01", "CMA_EVLU_AMT_ICLD_YN": "Y", "OVRS_ICLD_YN": "Y"}, timeout=5)
                    d_psbl = r_psbl.json()
                    if d_psbl.get("rt_cd") == "0":
                        max_buy = float(d_psbl.get("output", {}).get("max_buy_amt", 0.0))
                        equity = max(equity, max_buy)
                except Exception as e:
                    logger.warning(f"inquire-psbl-order error: {e}")

                return equity, held
            logger.error(f"❌ 잔고 조회 실패: {d.get('msg1')}")
        except Exception as e:
            logger.error(f"❌ 잔고 조회 예외: {e}")
        return None, {}

    def flatten_all(self, holdings: Dict[str, int]):
        """국내주식 전 종목 시장가 매도 (장 마감 시 주문 거부 → 휴지 중 반복 재시도)"""
        for sym, qty in holdings.items():
            body = {"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "PDNO": sym,
                    "ORD_DVSN": "01", "ORD_QTY": str(qty), "ORD_UNPR": "0"}
            try:
                r = session.post(f"{KIS_URL}/uapi/domestic-stock/v1/trading/order-cash",
                                 headers=self.headers("TTTC0801U"), json=body, timeout=8)
                d = r.json()
                if d.get("rt_cd") == "0":
                    logger.warning(f"   🚨 전량 매도: {sym} x {qty}")
                else:
                    logger.error(f"   매도 거부 {sym}: {d.get('msg1')} (장 시간 외면 재시도 예정)")
            except Exception as e:
                logger.error(f"   매도 예외 {sym}: {e}")

    def ref_equity(self, age_sec, now):
        target = now - age_sec
        best = None
        for ts, eq in self.history:
            if ts <= target:
                best = (ts, eq)
            else:
                break
        if best is None or target - best[0] > REF_TOLERANCE_SEC:
            return None
        return best[1]

    def record(self, now, equity):
        if now - self.last_record < RECORD_EVERY_SEC:
            return
        self.history.append([now, equity])
        cutoff = now - 10 * 86400
        self.history = [h for h in self.history if h[0] >= cutoff]
        save_json(HISTORY_FILE, self.history)
        self.last_record = now

    def systemctl(self, action):
        for svc in MANAGED_SERVICES:
            try:
                subprocess.run(["systemctl", "--user", action, svc], capture_output=True, timeout=30)
                logger.info(f"   systemctl {action} {svc} done")
            except Exception as e:
                logger.error(f"   systemctl {action} {svc} 실패: {e}")

    def trigger(self, reason, holdings, dry_run):
        now = time.time()
        logger.warning(f"🚨🚨 [KIS KR KILL SWITCH 발동] {reason}")
        if not dry_run:
            self.flatten_all(holdings)
            self.systemctl("stop")
            self.halt = {"halted_at": now, "resume_at": now + COOLDOWN_SEC, "reason": reason}
            save_json(HALT_FILE, self.halt)

    def maybe_resume(self, now, equity, holdings, dry_run):
        if not self.halt:
            return
        # 장 마감으로 못 판 포지션이 남아있으면 휴지 중에도 반복 매도 시도
        if holdings and not dry_run and now - self.last_flatten_try >= FLATTEN_RETRY_SEC:
            self.last_flatten_try = now
            logger.warning(f"🔁 [휴지 중 잔량 청산 재시도] {len(holdings)}종목")
            self.flatten_all(holdings)
        if now < self.halt["resume_at"]:
            logger.info(f"🧊 휴지 중... 재개까지 {(self.halt['resume_at']-now)/3600:.1f}시간 (사유: {self.halt['reason']})")
            return
        logger.info("✅ 휴지 종료 — 서비스 재개 + 기준선 리셋")
        if not dry_run:
            self.halt = None
            try:
                os.remove(HALT_FILE)
            except FileNotFoundError:
                pass
            self.history = [[now, equity]]
            save_json(HISTORY_FILE, self.history)
            self.systemctl("start")

    def cycle(self, dry_run=False):
        now = time.time()
        equity, holdings = self.fetch_account()
        if equity is None or equity <= 0:
            logger.error("❌ 평가액 조회 실패(이번 사이클 스킵)")
            return
        logger.info(f"💰 KIS 국내주식 평가: ₩{equity:,.0f} (보유 {len(holdings)}종목)")
        self.record(now, equity)

        self.maybe_resume(now, equity, holdings, dry_run)
        if self.halt:
            return

        ref_w = self.ref_equity(WEEK_SEC, now)
        ref_d = self.ref_equity(DAY_SEC, now)
        if ref_w and equity < ref_w * (1 - WEEKLY_DD_PCT):
            self.trigger(f"주간 DD: ₩{ref_w:,.0f} → ₩{equity:,.0f} ({(equity/ref_w-1)*100:.1f}%)", holdings, dry_run)
        elif ref_d and equity < ref_d * (1 - DAILY_DD_PCT):
            self.trigger(f"일간 DD: ₩{ref_d:,.0f} → ₩{equity:,.0f} ({(equity/ref_d-1)*100:.1f}%)", holdings, dry_run)

    def run(self, once, dry_run):
        logger.info(f"🛡️ KIS KR Kill Switch 시작 (주간 -{WEEKLY_DD_PCT*100:.0f}% / 일간 -{DAILY_DD_PCT*100:.0f}%, 60s 감시)")
        while True:
            try:
                self.cycle(dry_run=dry_run)
            except Exception as e:
                logger.error(f"❌ 사이클 예외: {e}", exc_info=True)
            if once:
                break
            time.sleep(LOOP_SEC)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    KisKrKillSwitch().run(once=args.once, dry_run=args.dry_run)
