# -*- coding: utf-8 -*-
"""
kis_kill_switch.py — KIS 해외주식(미국) 슬리브 킬스위치 (Account-Level Circuit Breaker)

감시 범위: 해외주식 계좌 평가액(USD, TTTS3012R tot_evlu_pfls_amt) — 국내주식 슬리브 제외
발동 조건 (둘 중 하나):
  - 주간 드로다운: 현재 평가 < 7일 전 × (1 - WEEKLY_DD_PCT)
  - 일간 드로다운: 현재 평가 < 24시간 전 × (1 - DAILY_DD_PCT)
발동 시:
  1. 미체결 주문 전량 취소 (best-effort)
  2. 해외주식 전 종목 체결우선 제한가(-0.5%) 매도
  3. kis_qqq_regime_trader / kis_nasdaq_jev_trader 서비스 정지
  4. 48시간 휴지 후 서비스 재개 + 기준선 리셋
테스트: python3 kis_kill_switch.py --once --dry-run
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
sys.path.append(os.path.join(CURRENT_DIR, ".."))
load_dotenv(os.path.join(CURRENT_DIR, "..", ".env"))

HISTORY_FILE = os.path.join(CURRENT_DIR, "kis_equity_history.json")
HALT_FILE = os.path.join(CURRENT_DIR, "kis_kill_switch_halt.json")
TOKEN_FILE = os.path.join(CURRENT_DIR, "kis_token.json")
LOG_FILE = os.path.join(CURRENT_DIR, "kis_kill_switch.log")

LOOP_SEC = 60
RECORD_EVERY_SEC = 600
WEEKLY_DD_PCT = 0.15
DAILY_DD_PCT = 0.10
WEEK_SEC = 7 * 86400
DAY_SEC = 86400
REF_TOLERANCE_SEC = 12 * 3600
COOLDOWN_SEC = 48 * 3600
SELL_OFFSET = 0.995

MANAGED_SERVICES = ["kis_qqq_regime_trader.service", "kis_nasdaq_jev_trader.service"]

logger = logging.getLogger("KisKillSwitch")
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


def send_alert(msg: str):
    try:
        from utils_telegram import send_telegram_alert
        send_telegram_alert(msg)
    except Exception as e:
        logger.warning(f"⚠️ Telegram 실패: {e}")


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


class KisKillSwitch:
    def __init__(self):
        self.access_token = ""
        self.token_expired_at = 0.0
        self.history = load_json(HISTORY_FILE, [])
        self.halt = load_json(HALT_FILE, None)
        self.last_record = 0.0

    def issue_token(self) -> bool:
        now = time.time()
        if self.access_token and now < self.token_expired_at:
            return True
        if os.path.exists(TOKEN_FILE):
            try:
                d = load_json(TOKEN_FILE, {})
                if d.get("access_token") and d.get("expires_at", 0) > now + 300:
                    self.access_token, self.token_expired_at = d["access_token"], d["expires_at"]
                    return True
            except Exception:
                pass
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
                "appkey": KIS_APP_KEY, "appsecret": KIS_APP_SECRET, "tr_id": tr_id, "custtype": "P"}

    def fetch_equity(self) -> Optional[float]:
        """해외주식 계좌 총평가(USD). 조회 실패 시 None."""
        if not self.issue_token():
            return None
        try:
            r = session.get(f"{KIS_URL}/uapi/overseas-stock/v1/trading/inquire-balance",
                            headers=self.headers("TTTS3012R"),
                            params={"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "OVRS_EXCG_CD": "NASD",
                                    "TR_CRCY_CD": "USD", "CTX_AREA_FK200": "", "CTX_AREA_NK200": ""}, timeout=8)
            if r.status_code == 200 and r.json().get("rt_cd") == "0":
                return float(r.json().get("output2", {}).get("tot_evlu_pfls_amt", 0.0))
        except Exception as e:
            logger.error(f"❌ 평가액 조회 실패: {e}")
        return None

    def fetch_holdings(self) -> Dict[str, dict]:
        if not self.issue_token():
            return {}
        try:
            r = session.get(f"{KIS_URL}/uapi/overseas-stock/v1/trading/inquire-balance",
                            headers=self.headers("TTTS3012R"),
                            params={"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "OVRS_EXCG_CD": "NASD",
                                    "TR_CRCY_CD": "USD", "CTX_AREA_FK200": "", "CTX_AREA_NK200": ""}, timeout=8)
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
            r = session.get(f"{KIS_URL}/uapi/overseas-price/v1/quotations/price",
                            headers=self.headers("HHDFS00000300"),
                            params={"AUTH": "", "EXCD": "NAS", "SYMB": sym}, timeout=5)
            if r.status_code == 200:
                last = float(r.json().get("output", {}).get("last", 0.0))
                return last if last > 0 else None
        except Exception:
            pass
        return None

    def cancel_open_orders(self):
        """미체결 주문 조회 후 전량 취소 (best-effort)"""
        try:
            r = session.get(f"{KIS_URL}/uapi/overseas-stock/v1/trading/inquire-nccs",
                            headers=self.headers("TTTS3018R"),
                            params={"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "OVRS_EXCG_CD": "NASD",
                                    "SORT_SQN": "DS", "CTX_AREA_FK200": "", "CTX_AREA_NK200": ""}, timeout=8)
            if r.status_code != 200 or r.json().get("rt_cd") != "0":
                return
            for row in r.json().get("output", []):
                odno, sym = row.get("odno", ""), (row.get("pdno") or "").strip().upper()
                qty = row.get("ft_ord_qty", "0")
                if not odno or not sym:
                    continue
                body = {"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "OVRS_EXCG_CD": "NASD", "PDNO": sym,
                        "ORG_ODNO": str(odno), "RVSE_CNCL_DVSN_CD": "02", "ORD_QTY": str(qty),
                        "OVRS_ORD_UNPR": "0", "ORD_SVR_DVSN_CD": "0"}
                try:
                    session.post(f"{KIS_URL}/uapi/overseas-stock/v1/trading/order-rvsecncl",
                                 headers=self.headers("TTTT1004U"), json=body, timeout=10)
                    logger.info(f"   🗑️ 미체결 취소: {sym} ODNO {odno}")
                except Exception as e:
                    logger.warning(f"   취소 실패 {sym}: {e}")
        except Exception as e:
            logger.warning(f"⚠️ 미체결 조회 실패: {e}")

    def flatten_all(self, holdings: Dict[str, int]):
        for sym, qty in holdings.items():
            px = self.get_price(sym)
            if not px:
                logger.error(f"   시세 조회 실패로 매도 보류: {sym}")
                continue
            limit_px = round(px * SELL_OFFSET, 2)
            body = {"CANO": KIS_CANO, "ACNT_PRDT_CD": KIS_PRDT_ABRV, "OVRS_EXCG_CD": "NASD", "PDNO": sym,
                    "ORD_QTY": str(qty), "OVRS_ORD_UNPR": f"{limit_px:.2f}", "ORD_SVR_DVSN_CD": "0", "ORD_DVSN": "00"}
            try:
                r = session.post(f"{KIS_URL}/uapi/overseas-stock/v1/trading/order",
                                 headers=self.headers("TTTT1006U"), json=body, timeout=10)
                d = r.json()
                if r.status_code == 200 and d.get("rt_cd") == "0":
                    logger.warning(f"   🚨 전량 매도: {sym} x {qty} @ ${limit_px:.2f}")
                else:
                    logger.error(f"   매도 거부 {sym}: {d.get('msg1')}")
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

    def trigger(self, reason, dry_run):
        now = time.time()
        logger.warning(f"🚨🚨 [KIS KILL SWITCH 발동] {reason}")
        send_alert(f"🚨 KIS 미국주식 킬스위치 발동\n{reason}\n전 종목 매도 + 48시간 매매 정지")
        if not dry_run:
            self.cancel_open_orders()
            self.flatten_all(self.fetch_holdings())
            self.systemctl("stop")
            self.halt = {"halted_at": now, "resume_at": now + COOLDOWN_SEC, "reason": reason}
            save_json(HALT_FILE, self.halt)

    def maybe_resume(self, now, equity, dry_run):
        if not self.halt:
            return
        if now < self.halt["resume_at"]:
            logger.info(f"🧊 휴지 중... 재개까지 {(self.halt['resume_at']-now)/3600:.1f}시간")
            return
        logger.info("✅ 휴지 종료 — 서비스 재개 + 기준선 리셋")
        send_alert(f"✅ KIS 킬스위치 휴지 종료, 매매 재개\n현재 평가: ${equity:,.2f}")
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
        equity = self.fetch_equity()
        if equity is None or equity <= 0:
            logger.error("❌ 평가액 조회 실패(이번 사이클 스킵)")
            return
        logger.info(f"💰 KIS 해외주식 평가: ${equity:,.2f}")
        self.record(now, equity)

        self.maybe_resume(now, equity, dry_run)
        if self.halt:
            return

        ref_w = self.ref_equity(WEEK_SEC, now)
        ref_d = self.ref_equity(DAY_SEC, now)
        if ref_w and equity < ref_w * (1 - WEEKLY_DD_PCT):
            self.trigger(f"주간 DD: ${ref_w:,.2f} → ${equity:,.2f} ({(equity/ref_w-1)*100:.1f}%)", dry_run)
        elif ref_d and equity < ref_d * (1 - DAILY_DD_PCT):
            self.trigger(f"일간 DD: ${ref_d:,.2f} → ${equity:,.2f} ({(equity/ref_d-1)*100:.1f}%)", dry_run)

    def run(self, once, dry_run):
        logger.info(f"🛡️ KIS Kill Switch 시작 (주간 -{WEEKLY_DD_PCT*100:.0f}% / 일간 -{DAILY_DD_PCT*100:.0f}%, 60s 감시)")
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
    KisKillSwitch().run(once=args.once, dry_run=args.dry_run)
