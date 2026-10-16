# -*- coding: utf-8 -*-
"""
okx_kill_switch.py — OKX 계정 전체 킬스위치 (Account-Level Circuit Breaker)

감시: 60초마다 선물(SWAP) 계정 USDT 총자산 추적, state/okx_equity_history.json에 기록
발동 조건 (둘 중 하나):
  - 주간 드로다운: 현재 자산 < 7일 전 자산 × (1 - WEEKLY_DD_PCT)
  - 일간 드로다운: 현재 자산 < 24시간 전 자산 × (1 - DAILY_DD_PCT)
발동 시:
  1. 전 종목 미체결 주문/트리거 주문 취소
  2. 전 포지션 시장가 청산 (reduceOnly)
  3. OKX 관련 systemd 서비스 정지 (okx_cameron_4h_swing, okx_tori_executor)
  4. 48시간 휴지 후 서비스 재개 + 기준선 리셋
테스트: python3 okx_kill_switch.py --once   (조회만, 주문/정지 없음)
"""

import os
import sys
import json
import time
import asyncio
import logging
import subprocess
from datetime import datetime, timezone

import ccxt.async_support as ccxt_async
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

STATE_DIR = os.path.join(CURRENT_DIR, "state")
os.makedirs(STATE_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(STATE_DIR, "okx_equity_history.json")
HALT_FILE = os.path.join(STATE_DIR, "okx_kill_switch_halt.json")
LOG_FILE = os.path.join(CURRENT_DIR, "okx_kill_switch.log")

LOOP_SEC = 60
RECORD_EVERY_SEC = 600          # 10분마다 자산 기록
WEEKLY_DD_PCT = 0.15            # 주간 -15% 발동
DAILY_DD_PCT = 0.10             # 24시간 -10% 발동
WEEK_SEC = 7 * 86400
DAY_SEC = 86400
REF_TOLERANCE_SEC = 12 * 3600   # 기준 샘플 허용 오차 (이것보다 오래된 기준 없으면 해당 트리거 스킵)
COOLDOWN_SEC = 48 * 3600        # 48시간 휴지

MANAGED_SERVICES = [
    "okx_cameron_4h_swing.service",
    "okx_tori_executor.timer",
    "okx_tori_executor.service",
]

logger = logging.getLogger("OKXKillSwitch")
logger.setLevel(logging.INFO)
logger.propagate = False
_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_fh)
if sys.stdout.isatty():
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_sh)


def send_alert(msg: str):
    try:
        from utils_telegram import send_telegram_alert
        send_telegram_alert(msg)
    except Exception as e:
        logger.warning(f"⚠️ Telegram alert failed: {e}")


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


class KillSwitch:
    def __init__(self):
        self.exchange = None
        self.history = load_json(HISTORY_FILE, [])   # [[ts, equity], ...]
        self.halt = load_json(HALT_FILE, None)       # {"halted_at":..., "resume_at":..., "reason":...}
        self.last_record = 0.0

    async def init(self):
        self.exchange = ccxt_async.okx({
            "apiKey": os.getenv("OKX_API_KEY", "").strip(),
            "secret": os.getenv("OKX_SECRET", "").strip(),
            "password": os.getenv("OKX_PASSPHRASE", "").strip(),
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        await self.exchange.load_markets()

    async def fetch_equity(self):
        bal = await self.exchange.fetch_balance()
        return float(bal.get("USDT", {}).get("total", 0.0))

    def ref_equity(self, age_sec: float, now: float):
        """age_sec 전에 가장 가까운 기록. 충분히 오래된 샘플이 없으면 None."""
        target = now - age_sec
        best = None
        for ts, eq in self.history:
            if ts <= target:
                best = (ts, eq)
            else:
                break
        if best is None:
            return None
        if target - best[0] > REF_TOLERANCE_SEC:
            return None
        return best[1]

    def record(self, now: float, equity: float):
        if now - self.last_record < RECORD_EVERY_SEC:
            return
        self.history.append([now, equity])
        cutoff = now - 10 * 86400
        self.history = [h for h in self.history if h[0] >= cutoff]
        save_json(HISTORY_FILE, self.history)
        self.last_record = now

    def systemctl(self, action: str):
        for svc in MANAGED_SERVICES:
            try:
                subprocess.run(["systemctl", "--user", action, svc],
                               capture_output=True, timeout=30)
                logger.info(f"   systemctl {action} {svc} done")
            except Exception as e:
                logger.error(f"   systemctl {action} {svc} failed: {e}")

    async def flatten_all(self):
        """전 포지션 시장가 청산 + 전 주문 취소."""
        # 1) 일반 + 트리거 미체결 주문 취소
        try:
            open_orders = await self.exchange.fetch_open_orders()
            for o in open_orders:
                try:
                    await self.exchange.cancel_order(o["id"], o["symbol"])
                    logger.info(f"   🗑️ 주문 취소: {o['symbol']} {o['id']}")
                except Exception as e:
                    logger.error(f"   주문 취소 실패 {o['symbol']}: {e}")
        except Exception as e:
            logger.error(f"   미체결 조회 실패: {e}")
        try:
            algo_orders = await self.exchange.fetch_open_orders(params={"trigger": True})
            for o in algo_orders:
                try:
                    await self.exchange.cancel_order(o["id"], o["symbol"], params={"trigger": True})
                    logger.info(f"   🗑️ 트리거 취소: {o['symbol']} {o['id']}")
                except Exception as e:
                    logger.error(f"   트리거 취소 실패 {o['symbol']}: {e}")
        except Exception as e:
            logger.error(f"   트리거 조회 실패: {e}")

        # 2) 포지션 청산
        try:
            positions = await self.exchange.fetch_positions()
            for p in positions:
                qty = abs(float(p.get("contracts") or 0.0))
                if qty <= 0:
                    continue
                side = "sell" if p.get("side") == "long" else "buy"
                pos_side = p.get("side") if p.get("side") in ("long", "short") else "net"
                try:
                    await self.exchange.create_order(
                        symbol=p["symbol"], type="market", side=side, amount=qty,
                        params={"tdMode": "cross", "posSide": pos_side, "reduceOnly": True},
                    )
                    logger.warning(f"   🚨 청산: {p['symbol']} {p.get('side')} {qty}")
                except Exception as e:
                    logger.error(f"   청산 실패 {p['symbol']}: {e}")
        except Exception as e:
            logger.error(f"   포지션 조회 실패: {e}")

    async def trigger(self, reason: str, dry_run: bool):
        now = time.time()
        logger.warning(f"🚨🚨 [KILL SWITCH 발동] {reason}")
        send_alert(f"🚨 OKX 킬스위치 발동\n{reason}\n전 포지션 청산 + 48시간 매매 정지")
        if not dry_run:
            await self.flatten_all()
            self.systemctl("stop")
            self.halt = {"halted_at": now, "resume_at": now + COOLDOWN_SEC, "reason": reason}
            save_json(HALT_FILE, self.halt)
        logger.warning(f"🛑 48시간 휴지 시작 (재개 예정: {datetime.fromtimestamp(now + COOLDOWN_SEC, timezone.utc).isoformat()})")

    async def maybe_resume(self, now: float, equity: float, dry_run: bool):
        if not self.halt:
            return
        if now < self.halt["resume_at"]:
            remain = (self.halt["resume_at"] - now) / 3600
            logger.info(f"🧊 휴지 중... 재개까지 {remain:.1f}시간 (사유: {self.halt['reason']})")
            return
        logger.info("✅ 휴지 기간 종료 — 서비스 재개 및 기준선 리셋")
        send_alert(f"✅ OKX 킬스위치 휴지 종료, 매매 재개\n현재 자산: {equity:.2f} USDT")
        if not dry_run:
            self.halt = None
            try:
                os.remove(HALT_FILE)
            except FileNotFoundError:
                pass
            self.history = [[now, equity]]     # 기준선 리셋
            save_json(HISTORY_FILE, self.history)
            self.systemctl("start")

    async def cycle(self, dry_run: bool = False):
        now = time.time()
        try:
            equity = await self.fetch_equity()
        except Exception as e:
            logger.error(f"❌ 자산 조회 실패(이번 사이클 스킵): {e}")
            return
        logger.info(f"💰 OKX 선물 자산: {equity:.2f} USDT")
        self.record(now, equity)

        await self.maybe_resume(now, equity, dry_run)
        if self.halt:
            return

        ref_w = self.ref_equity(WEEK_SEC, now)
        ref_d = self.ref_equity(DAY_SEC, now)
        if ref_w and equity < ref_w * (1 - WEEKLY_DD_PCT):
            await self.trigger(f"주간 DD: {ref_w:.2f} → {equity:.2f} USDT ({(equity/ref_w-1)*100:.1f}%)", dry_run)
        elif ref_d and equity < ref_d * (1 - DAILY_DD_PCT):
            await self.trigger(f"일간 DD: {ref_d:.2f} → {equity:.2f} USDT ({(equity/ref_d-1)*100:.1f}%)", dry_run)

    async def run(self, once: bool, dry_run: bool):
        await self.init()
        logger.info(f"🛡️ OKX Kill Switch 시작 (주간 -{WEEKLY_DD_PCT*100:.0f}% / 일간 -{DAILY_DD_PCT*100:.0f}%, 60s 감시)")
        try:
            while True:
                await self.cycle(dry_run=dry_run)
                if once:
                    break
                await asyncio.sleep(LOOP_SEC)
        finally:
            if self.exchange:
                await self.exchange.close()


async def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="한 사이클만 (조회 위주)")
    ap.add_argument("--dry-run", action="store_true", help="발동해도 주문/정지 실행 안 함")
    args = ap.parse_args()
    ks = KillSwitch()
    await ks.run(once=args.once, dry_run=args.dry_run)


if __name__ == "__main__":
    asyncio.run(main())
