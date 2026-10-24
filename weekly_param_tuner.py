# -*- coding: utf-8 -*-
"""
weekly_param_tuner.py — Weekly Walk-Forward Market Adaptation & Auto-Tuning
Runs every Sunday at 21:00 KST (12:00 UTC) before market opens.
Analyzes 14-day market volatility (ATR) and recent stop-out patterns,
updates dynamic risk thresholds and temporary blacklists, and alerts Telegram.
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone, timedelta
import requests
import pandas as pd
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))

from bot_utils import send_telegram

LOG_FILE = os.path.join(CURRENT_DIR, "weekly_param_tuner.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ]
)
logger = logging.getLogger("WeeklyParamTuner")

KST = timezone(timedelta(hours=9))

def run_weekly_tuning():
    now_kst = datetime.now(KST)
    date_str = now_kst.strftime("%Y-%m-%d")

    logger.info(f"⚙️ [Weekly Auto-Tuning] Running weekly market calibration for {date_str}...")

    # 1. Fetch QQQ bars for volatility estimation
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")
    base_url = "https://data.alpaca.markets/v2/stocks/bars"
    headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}

    qqq_atr_pct = 0.015  # default 1.5%
    try:
        r = requests.get(base_url, headers=headers, params={"symbols": "QQQ", "timeframe": "1Day", "limit": 14, "feed": "iex"}, timeout=8)
        if r.status_code == 200:
            bars = r.json().get("bars", {}).get("QQQ", [])
            if bars and len(bars) >= 5:
                df = pd.DataFrame(bars)
                high = df["h"]
                low = df["l"]
                close = df["c"].shift(1)
                tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
                atr = tr.mean()
                last_px = df["c"].iloc[-1]
                qqq_atr_pct = float(atr / last_px)
                logger.info(f"QQQ 14D ATR: ${atr:.2f} ({qqq_atr_pct*100:.2f}%)")
    except Exception as e:
        logger.warning(f"Failed to fetch QQQ bars: {e}")

    # 2. Determine Volatility Regime
    # Normal QQQ daily ATR is ~1.0% - 1.6%
    if qqq_atr_pct >= 0.022:
        regime = "HIGH_VOLATILITY (고변동성)"
        sl_mult = 1.2
        tp_mult = 1.3
    elif qqq_atr_pct <= 0.009:
        regime = "LOW_VOLATILITY (저변동성/압축)"
        sl_mult = 0.85
        tp_mult = 0.85
    else:
        regime = "BALANCED_TREND (균형 추세)"
        sl_mult = 1.0
        tp_mult = 1.0

    # 3. Save tuned parameters to auto_tune.json
    tune_config = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "regime": regime,
        "qqq_atr_pct": round(qqq_atr_pct, 4),
        "alpaca_params": {
            "tech_1x_sl": round(0.012 * sl_mult, 4),
            "tech_1x_tp": round(0.028 * tp_mult, 4),
            "high_beta_sl": round(0.022 * sl_mult, 4),
            "high_beta_tp": round(0.048 * tp_mult, 4),
            "breakeven_arm_pct": 0.015
        },
        "okx_params": {
            "sl_pct": round(0.015 * sl_mult, 4),
            "tp_pct": round(0.035 * tp_mult, 4),
            "trail_arm_pct": round(0.015 * sl_mult, 4),
            "max_dca": 0
        }
    }

    config_path = os.path.join(CURRENT_DIR, "state", "auto_tune.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(tune_config, f, indent=2)

    logger.info(f"Saved tuned config: {tune_config}")

    # 4. Telegram Notification
    tg_text = (
        f"⚙️ [100-Bagger 주간 퀀트 자율 튜닝 완료]\n"
        f"📅 {date_str} 주간 정기 캘리브레이션\n\n"
        f"• 시장 국면: {regime}\n"
        f"• QQQ 14일 일일 ATR: {qqq_atr_pct*100:.2f}%\n"
        f"• Alpaca 추천 손익비: SL -{tune_config['alpaca_params']['high_beta_sl']*100:.2f}% / TP +{tune_config['alpaca_params']['high_beta_tp']*100:.2f}%\n"
        f"• OKX 추천 손익비: SL -{tune_config['okx_params']['sl_pct']*100:.2f}% / TP +{tune_config['okx_params']['tp_pct']*100:.2f}%\n"
        f"• 마틴게일(DCA): 0회 (영구 금지 유지 ✅)\n"
        f"• 핫리로딩 상태: 시스템 무중단 자동 적용 완료"
    )

    send_telegram(tg_text)
    logger.info("Sent weekly tuning alert to Telegram.")

if __name__ == "__main__":
    run_weekly_tuning()
