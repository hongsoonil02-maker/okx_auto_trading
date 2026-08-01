#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
daily_analyzer.py
매일 자정에 실행되어 거래 로그 분석, 파라미터 최적화, 텔레그램 리포팅 수행
"""
import os
import json
import csv
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pyupbit
import pandas as pd
from utils_telegram import send_telegram_alert

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "sizing_trade_log.csv")
CONFIG_FILE = os.path.join(BASE_DIR, "auto_tune_config.json")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
KST = ZoneInfo("Asia/Seoul")

# [시장별 폐장 맞춤 자가진화 스케줄]
# - KR Stock: 18:30 KST (장후 시간에외 마감 30분 후)
# - US Nasdaq: 11:30 KST (미국 애프터마켓 마감 2시간 30분 후)
# - Crypto:   00:30 KST (자정 30분 후)
MARKET_SCHEDULES = {
    "KR_STOCK": "18:30",
    "US_NASDAQ": "11:30",
    "CRYPTO":    "00:30"
}

MIN_SAMPLES_PER_MARKET = 5
UPBIT_CONSERVATIVE_BLACKLIST = []

DEFAULT_UPBIT_PARAMS = {
    "TRAILING_PCT": 0.02,
    "HARD_SL_PCT": 0.015,
    "ORB_LOOKBACK": 10,
    "POSITION_PCT": 0.05,
    "MIN_HOLD_SEC": 900,
    "COOLDOWN_SEC": 3600,
    "UPBIT_MIN_RANGE_PCT": 0.35,
    "UPBIT_ENTRY_BUFFER_PCT": 0.0012,
    "UPBIT_MIN_MOMENTUM_PCT": 0.0008
}

DEFAULT_OKX_PARAMS = {
    "TRAILING_PCT": 0.03,
    "HARD_SL_PCT": 0.035,
    "ORB_LOOKBACK": 4,
    "POSITION_PCT": 0.18,
    "OKX_MIN_RANGE_PCT": 0.6,
    "OKX_ENTRY_BUFFER_PCT": 0.0004,
    "OKX_MIN_MOMENTUM_PCT": 0.0003
}


def _apply_regime_tuning(upbit_p: dict, okx_p: dict, volatility_pct: float):
    """레짐에 따라 entry hurdle / momentum / range threshold를 자동 조정한다."""
    if volatility_pct > 1.2:
        # 확장장: 휩쏘 방지로 문턱은 조금 더 높이고, 손절은 넓힌다.
        upbit_p.update({
            "POSITION_PCT": 0.04,
            "MIN_HOLD_SEC": 900,
            "COOLDOWN_SEC": 3600,
            "UPBIT_MIN_RANGE_PCT": 0.50,
            "UPBIT_ENTRY_BUFFER_PCT": 0.0020,
            "UPBIT_MIN_MOMENTUM_PCT": 0.0015,
            "TRAILING_PCT": 0.03,
            "HARD_SL_PCT": 0.07,
            "ORB_LOOKBACK": 20,
        })
        okx_p.update({
            "POSITION_PCT": 0.20,
            "OKX_MIN_RANGE_PCT": 0.80,
            "OKX_ENTRY_BUFFER_PCT": 0.0006,
            "OKX_MIN_MOMENTUM_PCT": 0.0005,
            "TRAILING_PCT": 0.03,
            "HARD_SL_PCT": 0.07,
            "ORB_LOOKBACK": 8,
        })
        regime = "High Volatility (Expansion)"
    elif volatility_pct < 0.5:
        # 수축장: 더 타이트하고 민감하게, 대신 레짐 필터가 이미 수축장은 차단.
        upbit_p.update({
            "POSITION_PCT": 0.05,
            "MIN_HOLD_SEC": 900,
            "COOLDOWN_SEC": 3600,
            "UPBIT_MIN_RANGE_PCT": 0.35,
            "UPBIT_ENTRY_BUFFER_PCT": 0.0012,
            "UPBIT_MIN_MOMENTUM_PCT": 0.0008,
            "TRAILING_PCT": 0.015,
            "HARD_SL_PCT": 0.03,
            "ORB_LOOKBACK": 10,
        })
        okx_p.update({
            "POSITION_PCT": 0.18,
            "OKX_MIN_RANGE_PCT": 0.60,
            "OKX_ENTRY_BUFFER_PCT": 0.0004,
            "OKX_MIN_MOMENTUM_PCT": 0.0003,
            "TRAILING_PCT": 0.015,
            "HARD_SL_PCT": 0.03,
            "ORB_LOOKBACK": 4,
        })
        regime = "Low Volatility (Tighter Stops)"
    else:
        upbit_p.update({
            "POSITION_PCT": 0.05,
            "MIN_HOLD_SEC": 900,
            "COOLDOWN_SEC": 3600,
            "UPBIT_MIN_RANGE_PCT": 0.40,
            "UPBIT_ENTRY_BUFFER_PCT": 0.0015,
            "UPBIT_MIN_MOMENTUM_PCT": 0.0010,
            "TRAILING_PCT": 0.02,
            "HARD_SL_PCT": 0.05,
            "ORB_LOOKBACK": 15,
        })
        okx_p.update({
            "POSITION_PCT": 0.20,
            "OKX_MIN_RANGE_PCT": 0.60,
            "OKX_ENTRY_BUFFER_PCT": 0.0004,
            "OKX_MIN_MOMENTUM_PCT": 0.0003,
            "TRAILING_PCT": 0.02,
            "HARD_SL_PCT": 0.05,
            "ORB_LOOKBACK": 6,
        })
        regime = "Normal Volatility"

    return upbit_p, okx_p, regime

def analyze_trades(lookback_days: int = 30):
    if not os.path.exists(LOG_FILE):
        return {
            "bad_hours": [],
            "bad_symbols": {"OKX": [], "UPBIT": []},
            "summary": {},
            "lookback_days": lookback_days,
            "trade_count": 0,
        }

    trades = []
    cutoff = datetime.now(KST) - timedelta(days=lookback_days)
    with open(LOG_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # timezone aware parsing (assume KST if naive or parse iso)
                ts = datetime.fromisoformat(row["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=KST)
                if ts.astimezone(KST) < cutoff:
                    continue
                row["ts"] = ts
                row["pnl"] = float(row["pnl_pct"])
                side = (row.get("side") or "").upper()

                # 진입(BUY/SELL) 행은 손익이 0으로 남는 경우가 많아 성과 분석에서 제외.
                # 청산(CLOSE_*/EXIT_*) 행 위주로 실제 실현손익을 집계한다.
                if side in {"BUY", "SELL"} and row["pnl"] == 0:
                    continue
                trades.append(row)
            except Exception:
                continue

    if not trades:
        return {
            "bad_hours": [],
            "bad_symbols": {"OKX": [], "UPBIT": []},
            "summary": {},
            "lookback_days": lookback_days,
            "trade_count": 0,
        }

    # 1. 시간대/종목 통계 (시장별)
    hour_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})
    hour_market_stats = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0}))
    sym_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "market": ""})
    sym_market_stats = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0}))

    for t in trades:
        h = t["ts"].astimezone(KST).hour
        sym = t["symbol"]
        pnl = t["pnl"]
        market = t.get("market", "").upper()
        
        # 시간대 통계
        hour_stats[h]["pnl"] += pnl
        hour_market_stats[market][h]["pnl"] += pnl
        if pnl > 0:
            hour_stats[h]["wins"] += 1
            hour_market_stats[market][h]["wins"] += 1
        else:
            hour_stats[h]["losses"] += 1
            hour_market_stats[market][h]["losses"] += 1

        # 종목 통계 (market 정보 포함)
        sym_stats[sym]["pnl"] += pnl
        sym_stats[sym]["market"] = market
        sym_market_stats[market][sym]["pnl"] += pnl
        if pnl > 0:
            sym_stats[sym]["wins"] += 1
            sym_market_stats[market][sym]["wins"] += 1
        else:
            sym_stats[sym]["losses"] += 1
            sym_market_stats[market][sym]["losses"] += 1

    summary = {}
    market_rows = defaultdict(list)
    for t in trades:
        market_rows[t.get("market", "").upper()].append(t)

    for market, rows in market_rows.items():
        total = len(rows)
        wins = sum(1 for r in rows if r["pnl"] > 0)
        losses = sum(1 for r in rows if r["pnl"] < 0)
        pnl_sum = sum(r["pnl"] for r in rows)
        summary[market] = {
            "trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / total * 100.0) if total else 0.0,
            "avg_pnl_pct": (pnl_sum / total) if total else 0.0,
            "sum_pnl_pct": pnl_sum,
        }

    # 최악의 시간대 도출 (표본 5개 이상, 평균손익 -0.4% 이하)
    bad_hours = []
    for h, s in hour_stats.items():
        total = s["wins"] + s["losses"]
        if total >= 5:
            avg_pnl = s["pnl"] / total
            if avg_pnl <= -0.4:
                bad_hours.append(h)

    # 불량 종목 도출 — market별 dict 구조로 분류
    # (표본 3개 이상, 평균손익 -1.0% 이하)
    bad_symbols = {"OKX": [], "UPBIT": []}
    for sym, s in sym_stats.items():
        total = s["wins"] + s["losses"]
        if total >= 3:
            avg_pnl = s["pnl"] / total
            if avg_pnl <= -1.0:
                market = s["market"]
                if market in bad_symbols:
                    bad_symbols[market].append(sym)

    bad_hours = sorted(set(bad_hours))

    # 시장별 차단 시간대 (각 시장 최악 3개 시간대)
    bad_hours_by_market = {"OKX": [], "UPBIT": []}
    for market in ("OKX", "UPBIT"):
        candidates = []
        for h, s in hour_market_stats[market].items():
            total = s["wins"] + s["losses"]
            if total < 5:
                continue
            avg_pnl = s["pnl"] / total
            if avg_pnl <= -0.6:
                candidates.append((avg_pnl, h))
        candidates.sort(key=lambda x: x[0])
        bad_hours_by_market[market] = [h for _, h in candidates[:3]]

    bad_symbols["OKX"] = sorted(set(bad_symbols["OKX"]))
    bad_symbols["UPBIT"] = sorted(set(bad_symbols["UPBIT"]))

    return {
        "bad_hours": bad_hours,
        "bad_symbols": bad_symbols,
        "bad_hours_by_market": bad_hours_by_market,
        "summary": summary,
        "lookback_days": lookback_days,
        "trade_count": len(trades),
    }


def load_existing_config() -> dict:
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def get_market_volatility():
    """BTC의 최근 24시간 변동성 측정"""
    try:
        df = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=24)
        if df is not None and not df.empty:
            volatility_pct = ((df['high'] - df['low']) / df['close']).mean() * 100
            return volatility_pct
    except Exception as e:
        print(f"변동성 계산 실패: {e}")
    return 1.0 # 기본값


def calculate_dynamic_params(volatility_pct):
    """변동성에 따라 스탑로스 및 ORB 기간 튜닝"""
    upbit_p = DEFAULT_UPBIT_PARAMS.copy()
    okx_p = DEFAULT_OKX_PARAMS.copy()

    return _apply_regime_tuning(upbit_p, okx_p, volatility_pct)


def run_analyzer():
    print(f"[{datetime.now(KST)}] 일일 분석 및 자동 튜닝 시작...")
    analysis = analyze_trades(lookback_days=30)
    bad_hours = analysis["bad_hours"]
    bad_symbols = analysis["bad_symbols"]
    bad_hours_by_market = analysis.get("bad_hours_by_market", {"OKX": [], "UPBIT": []})
    volatility = get_market_volatility()
    upbit_p, okx_p, regime = calculate_dynamic_params(volatility)
    existing = load_existing_config()

    # 표본이 너무 적은 시장은 기존 블랙리스트/차단시간 유지
    summary = analysis.get("summary", {})
    okx_n = int(summary.get("OKX", {}).get("trades", 0))
    upbit_n = int(summary.get("UPBIT", {}).get("trades", 0))
    if okx_n < MIN_SAMPLES_PER_MARKET:
        bad_symbols["OKX"] = existing.get("blacklisted_symbols", {}).get("OKX", bad_symbols["OKX"])
        bad_hours_by_market["OKX"] = existing.get("blocked_hours_kst_okx", bad_hours_by_market["OKX"])
    if upbit_n < MIN_SAMPLES_PER_MARKET:
        bad_symbols["UPBIT"] = existing.get("blacklisted_symbols", {}).get("UPBIT", bad_symbols["UPBIT"])
        bad_hours_by_market["UPBIT"] = existing.get("blocked_hours_kst_upbit", bad_hours_by_market["UPBIT"])

    # Upbit 손실이 지속되면, 차단 시간대를 전체 bad_hours로 확장하고 블랙리스트도 보수적으로 유지
    upbit_summary = summary.get("UPBIT", {})
    if upbit_summary.get("avg_pnl_pct", 0.0) < 0:
        bad_hours_by_market["UPBIT"] = sorted(set(bad_hours_by_market.get("UPBIT", [])) | set(bad_hours))
        bad_symbols["UPBIT"] = sorted(
            set(bad_symbols.get("UPBIT", []))
            | set(existing.get("blacklisted_symbols", {}).get("UPBIT", []))
            | set(UPBIT_CONSERVATIVE_BLACKLIST)
        )

    merged_blocked = sorted(set(bad_hours_by_market.get("OKX", [])) | set(bad_hours_by_market.get("UPBIT", [])))
    if not merged_blocked:
        merged_blocked = existing.get("blocked_hours_kst", [12, 13, 21])

    config = {
        "blocked_hours_kst": merged_blocked,
        "blocked_hours_kst_okx": sorted(set(bad_hours_by_market.get("OKX", []))),
        "blocked_hours_kst_upbit": sorted(set(bad_hours_by_market.get("UPBIT", []))),
        "blacklisted_symbols": bad_symbols,
        "upbit_params": upbit_p,
        "okx_params": okx_p
    }

    # config 저장
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 텔레그램 리포트 작성
    okx_summary = analysis["summary"].get("OKX", {})
    upbit_summary = analysis["summary"].get("UPBIT", {})

    okx_blacklist_text = ", ".join(bad_symbols.get("OKX", [])) or "None"
    upbit_blacklist_text = ", ".join(bad_symbols.get("UPBIT", [])) or "None"

    report = (
        "📊 [Bot Daily Auto-Tuning Report]\n"
        "=========================\n"
        f"📦 Lookback: {analysis['lookback_days']}d | Trades: {analysis['trade_count']}\n"
        f"📈 Market Regime: {regime}\n"
        f"🔹 BTC 24h Avg Hourly Volatility: {volatility:.2f}%\n"
        "\n"
        "📌 [Market Summary]\n"
        f"OKX  | N={okx_summary.get('trades', 0)} | WR={okx_summary.get('win_rate', 0.0):.1f}% | Avg={okx_summary.get('avg_pnl_pct', 0.0):+.3f}% | Sum={okx_summary.get('sum_pnl_pct', 0.0):+.2f}%\n"
        f"UPBIT| N={upbit_summary.get('trades', 0)} | WR={upbit_summary.get('win_rate', 0.0):.1f}% | Avg={upbit_summary.get('avg_pnl_pct', 0.0):+.3f}% | Sum={upbit_summary.get('sum_pnl_pct', 0.0):+.2f}%\n"
        "\n"
        "⛔ [Blocked KST Hours]\n"
        f"{config['blocked_hours_kst']}\n"
        f"OKX={config['blocked_hours_kst_okx']} | UPBIT={config['blocked_hours_kst_upbit']}\n"
        "\n"
        "🚫 [Blacklisted Symbols - OKX]\n"
        f"{okx_blacklist_text}\n"
        "\n"
        "🚫 [Blacklisted Symbols - UPBIT]\n"
        f"{upbit_blacklist_text}\n"
        "\n"
        "⚙️ [Adjusted Params]\n"
        f"Upbit: Pos {upbit_p['POSITION_PCT']*100:.1f}%, TS {upbit_p['TRAILING_PCT']*100:.1f}%, SL {upbit_p['HARD_SL_PCT']*100:.1f}%, ORB {upbit_p['ORB_LOOKBACK']} candles, CD {upbit_p['COOLDOWN_SEC']//60}m, EB {upbit_p['UPBIT_ENTRY_BUFFER_PCT']*100:.2f}%, MOM {upbit_p['UPBIT_MIN_MOMENTUM_PCT']*100:.2f}%, RNG {upbit_p['UPBIT_MIN_RANGE_PCT']:.2f}%\n"
        f"OKX: Pos {okx_p['POSITION_PCT']*100:.1f}%, TS {okx_p['TRAILING_PCT']*100:.1f}%, SL {okx_p['HARD_SL_PCT']*100:.1f}%, ORB {okx_p['ORB_LOOKBACK']} candles, EB {okx_p['OKX_ENTRY_BUFFER_PCT']*100:.2f}%, MOM {okx_p['OKX_MIN_MOMENTUM_PCT']*100:.2f}%, RNG {okx_p['OKX_MIN_RANGE_PCT']:.2f}%\n"
        "=========================\n"
        "✅ 파라미터 자동 반영 완료."
    )

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_file = os.path.join(REPORT_DIR, f"daily_report_{datetime.now(KST).strftime('%Y%m%d')}.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print("Config Update Completed.")
    print(report)
    send_telegram_alert(report)


if __name__ == "__main__":
    run_analyzer()
