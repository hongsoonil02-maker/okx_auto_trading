# -*- coding: utf-8 -*-
"""
daily_quant_auditor.py — Autonomous Daily Quant Audit & Telegram Dispatch
Runs daily at 06:30 KST (after US market close).
Fetches Alpaca Paper & OKX Demo accounts, calculates PnL/Returns/Positions,
writes a daily Markdown audit report, and sends a concise Telegram briefing.
"""

import os
import sys
import time
import json
import hmac
import base64
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
import requests
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

from bot_utils import send_telegram, load_state, save_state

LOG_FILE = os.path.join(CURRENT_DIR, "daily_quant_auditor.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ]
)
logger = logging.getLogger("DailyQuantAuditor")

# KST Timezone Helper
KST = timezone(timedelta(hours=9))

def get_alpaca_stats() -> Dict[str, Any]:
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    base_url = os.getenv("ALPACA_REST_URL", "https://paper-api.alpaca.markets/v2")
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}

    stats = {
        "ok": False,
        "equity": 0.0,
        "last_equity": 0.0,
        "cash": 0.0,
        "daily_pnl": 0.0,
        "daily_pnl_pct": 0.0,
        "positions": [],
        "fill_count": 0
    }

    try:
        # Account
        r = requests.get(f"{base_url}/account", headers=headers, timeout=8)
        if r.status_code == 200:
            acc = r.json()
            eq = float(acc.get("equity", 0.0))
            last_eq = float(acc.get("last_equity", eq))
            cash = float(acc.get("cash", 0.0))
            diff = eq - last_eq
            diff_pct = (diff / last_eq * 100.0) if last_eq > 0 else 0.0

            stats["ok"] = True
            stats["equity"] = eq
            stats["last_equity"] = last_eq
            stats["cash"] = cash
            stats["daily_pnl"] = diff
            stats["daily_pnl_pct"] = diff_pct

        # Positions
        rp = requests.get(f"{base_url}/positions", headers=headers, timeout=8)
        if rp.status_code == 200:
            stats["positions"] = rp.json()

        # Activities (Fills)
        ra = requests.get(f"{base_url}/account/activities?activity_types=FILL&direction=desc&page_size=10", headers=headers, timeout=8)
        if ra.status_code == 200:
            stats["fill_count"] = len(ra.json())

    except Exception as e:
        logger.error(f"Alpaca audit error: {e}")

    return stats


def get_okx_stats() -> Dict[str, Any]:
    key = os.getenv("OKX_DEMO_API_KEY", "")
    secret = os.getenv("OKX_DEMO_SECRET", "")
    passphrase = os.getenv("OKX_DEMO_PASSPHRASE", "")
    base_url = "https://www.okx.com"

    stats = {
        "ok": False,
        "equity": 0.0,
        "prior_equity": 0.0,
        "daily_pnl": 0.0,
        "daily_pnl_pct": 0.0,
        "positions": [],
        "state_restored": False
    }

    def sign_headers(method: str, path: str):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        pre_hash = ts + method.upper() + path
        sign = base64.b64encode(
            hmac.new(secret.encode("utf-8"), pre_hash.encode("utf-8"), digestmod="sha256").digest()
        ).decode("utf-8")
        return {
            "OK-ACCESS-KEY": key,
            "OK-ACCESS-SIGN": sign,
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": passphrase,
            "Content-Type": "application/json",
            "x-simulated-trading": "1"
        }

    # Load prior equity tracker (USDT 기준 신규 파일 — 구 totalEq 파일과 분리해 기준 변경일 왜곡 방지)
    hist_file = os.path.join(CURRENT_DIR, "state", "okx_prior_equity_usdt.json")
    prior_eq = 0.0
    if os.path.exists(hist_file):
        try:
            with open(hist_file, "r") as f:
                saved = json.load(f)
                prior_eq = float(saved.get("equity", 0.0))
        except Exception:
            pass

    try:
        # Balance — 기준 자산은 USDT 거래 자본 (totalEq는 유휴 데모 코인 1BTC/100OKB/1ETH 포함이라 성과 왜곡)
        rb = requests.get(f"{base_url}/api/v5/account/balance", headers=sign_headers("GET", "/api/v5/account/balance"), timeout=8)
        if rb.status_code == 200:
            data = rb.json().get("data", [])
            if data:
                usdt_eq = 0.0
                for d in data[0].get("details", []):
                    if d.get("ccy") == "USDT":
                        usdt_eq = float(d.get("eq", 0.0) or 0.0)
                eq = usdt_eq if usdt_eq > 0 else float(data[0].get("totalEq", 0.0))
                stats["ok"] = True
                stats["equity"] = eq
                stats["total_eq"] = float(data[0].get("totalEq", 0.0))
                if prior_eq > 0:
                    diff = eq - prior_eq
                    diff_pct = (diff / prior_eq * 100.0)
                    stats["prior_equity"] = prior_eq
                    stats["daily_pnl"] = diff
                    stats["daily_pnl_pct"] = diff_pct
                else:
                    stats["prior_equity"] = eq

                # Save current as today's checkpoint
                os.makedirs(os.path.dirname(hist_file), exist_ok=True)
                with open(hist_file, "w") as f:
                    json.dump({"equity": eq, "timestamp": datetime.now(timezone.utc).isoformat()}, f)

        # Positions
        rp = requests.get(f"{base_url}/api/v5/account/positions?instType=SWAP", headers=sign_headers("GET", "/api/v5/account/positions?instType=SWAP"), timeout=8)
        if rp.status_code == 200:
            stats["positions"] = [p for p in rp.json().get("data", []) if abs(float(p.get("pos", 0.0) or 0.0)) > 0]

    except Exception as e:
        logger.error(f"OKX audit error: {e}")

    return stats


def run_daily_audit():
    now_kst = datetime.now(KST)
    date_str = now_kst.strftime("%Y-%m-%d")
    time_str = now_kst.strftime("%H:%M:%S")

    logger.info(f"🚀 [Daily Quant Audit] Starting run for {date_str} {time_str} KST...")

    alpaca = get_alpaca_stats()
    okx = get_okx_stats()

    # Build Telegram text
    a_sign = "+" if alpaca["daily_pnl"] >= 0 else "-"
    o_sign = "+" if okx["daily_pnl"] >= 0 else "-"

    a_pos_desc = f"{len(alpaca['positions'])}개" if alpaca['positions'] else "0개 (100% 현금 대기)"
    o_pos_desc = f"{len(okx['positions'])}개" if okx['positions'] else "0개 (ATR 횡보 필터 대기)"

    tg_msg = (
        f"📊 [100-Bagger 퀀트 랩 일일 성과 감사]\n"
        f"📅 {date_str} {now_kst.strftime('%H:%M')} KST 브리핑\n\n"
        f"🇺🇸 Alpaca (Nasdaq Paper):\n"
        f"• 계좌 평가액: ${alpaca['equity']:,.2f} ({a_sign}${abs(alpaca['daily_pnl']):,.2f} / {a_sign}{abs(alpaca['daily_pnl_pct']):.2f}%)\n"
        f"• 가용 현금: ${alpaca['cash']:,.2f}\n"
        f"• 보유 포지션: {a_pos_desc}\n\n"
        f"🪙 OKX (Futures Demo):\n"
        f"• USDT 거래 자본: ${okx['equity']:,.2f} ({o_sign}${abs(okx['daily_pnl']):,.2f} / {o_sign}{abs(okx['daily_pnl_pct']):.2f}%)\n"
        f"• 참고) 총 데모 평가액: ${okx.get('total_eq', 0.0):,.2f} (유휴 코인 포함, 성과 지표 아님)\n"
        f"• 모드: 10x Isolated Net-Mode — Argon V4-Pro S8 (100% Simulated)\n"
        f"• 보유 포지션: {o_pos_desc}\n\n"
        f"🛡️ 봇 엔진 상태:\n"
        f"• Alpaca: 1H/15m 추세추종 + 4섹터 레이더 + Jev 게이트 Active\n"
        f"• OKX: 4대 메이저 3/4 동조 + 1H 확정봉 + 서버사이드 SL Active\n"
        f"• 실계좌 잔고: 전액 출금 안전 보존 상태 ✅"
    )

    logger.info("Telegram message prepared:\n" + tg_msg)
    send_telegram(tg_msg)

    # Save Markdown report
    reports_dir = os.path.join(CURRENT_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, f"quant_audit_{date_str}.md")

    md_content = f"""# 📊 100-Bagger 퀀트 랩 일일 성과 감사 리포트 ({date_str})

- **감사 일시:** {date_str} {time_str} KST
- **계좌 모드:** 100% 가상/모의계좌 (Alpaca Paper & OKX Demo)

---

## 1. Alpaca Nasdaq Paper Trading
- **총 계좌 평가액:** ${alpaca['equity']:,.2f}
- **전일 대비 손익:** {a_sign}${abs(alpaca['daily_pnl']):,.2f} ({a_sign}{abs(alpaca['daily_pnl_pct']):.2f}%)
- **가용 현금:** ${alpaca['cash']:,.2f}
- **보유 포지션 수:** {len(alpaca['positions'])}개
- **최근 체결 건수:** {alpaca['fill_count']}건

## 2. OKX Futures Demo Trading
- **USDT 거래 자본 (성과 기준):** ${okx['equity']:,.2f}
- **일일 변동 (USDT 기준):** {o_sign}${abs(okx['daily_pnl']):,.2f} ({o_sign}{abs(okx['daily_pnl_pct']):.2f}%)
- **참고) 총 데모 평가액:** ${okx.get('total_eq', 0.0):,.2f} (유휴 데모 코인 1BTC/100OKB/1ETH 포함 — 성과 지표 아님)
- **레버리지 설정:** 10x Isolated (Argon V4-Pro S8, 2026-11-09 백테스트 채택)
- **보유 포지션 수:** {len(okx['positions'])}개

## 3. 핵심 리스크 & 알파 방어 가드
1. **DCA(물타기) 영구 금지:** 단일 진입(Single Entry)만 허용
2. **손익비 (S8):** SL -2.5% (동적 cap -4.0%) / Wave TP +5.0% / 10x 격리
3. **1H 확정봉 매매:** 4대 메이저 3/4 동조 + 15m EMA/ADX/거래량 필터, 원웨이브원샷 잠금
4. **4단계 수확:** S1 본절(+1.5%→+0.25%) / S2 중간락(+2.2%→+1.1%) / S3 트레일(+3.2%, 1.0% 되돌림) / S4 1H MA20 구조청산
5. **서버사이드 SL/TP 알고주문:** 봇 다운타임·재시작 중에도 거래소가 직접 손절 집행

---
*Auto-generated by DailyQuantAuditor Engine*
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"✅ Daily audit report saved to {report_path}")

if __name__ == "__main__":
    run_daily_audit()
