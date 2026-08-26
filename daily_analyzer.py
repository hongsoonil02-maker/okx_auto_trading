#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
daily_analyzer.py
매일 자정에 실행되어 거래 로그 분석, 파라미터 최적화, 텔레그램 리포팅 수행 (OKX Only)
"""
import os
import json
import csv
from collections import defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import ccxt
import pandas as pd
from utils_telegram import send_telegram_alert

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "sizing_trade_log.csv")
TRADES_JSONL = os.path.join(BASE_DIR, "state", "trades.jsonl")  # [Fix] 실거래 데이터 원본 파일
CONFIG_FILE = os.path.join(BASE_DIR, "auto_tune_config.json")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
KST = ZoneInfo("Asia/Seoul")

MIN_SAMPLES_PER_MARKET = 5

DEFAULT_OKX_PARAMS = {
    "TRAILING_PCT": 0.03,
    "HARD_SL_PCT": 0.035,
    "ORB_LOOKBACK": 4,
    "POSITION_PCT": 0.18,
    "OKX_MIN_RANGE_PCT": 0.6,
    "OKX_ENTRY_BUFFER_PCT": 0.0004,
    "OKX_MIN_MOMENTUM_PCT": 0.0003
}

def _apply_regime_tuning(okx_p: dict, volatility_pct: float):
    """레짐에 따라 entry hurdle / momentum / range threshold를 자동 조정한다."""
    if volatility_pct > 1.2:
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
        okx_p.update({
            "POSITION_PCT": 0.05,
            "OKX_MIN_RANGE_PCT": 0.60,
            "OKX_ENTRY_BUFFER_PCT": 0.0004,
            "OKX_MIN_MOMENTUM_PCT": 0.0003,
            "TRAILING_PCT": 0.02,
            "HARD_SL_PCT": 0.05,
            "ORB_LOOKBACK": 6,
        })
        regime = "Normal Volatility"

    return okx_p, regime

def _load_trades_jsonl(lookback_days: int):
    """[Fix] state/trades.jsonl 실거래 데이터를 읽어서 청산(CLOSE 포지션)만 반환.
    [Fix #2] DCA 물타기 평균가: 마지막 매수가 대신 가중평균진입가(WAAP) 추적.
    """
    if not os.path.exists(TRADES_JSONL):
        return []
    cutoff_ts = (datetime.now() - timedelta(days=lookback_days)).timestamp()
    # sym -> {'qty': float, 'total_cost': float}  가중평균 진입가 추적용
    sym_pos = {}
    rows = []
    try:
        with open(TRADES_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                except Exception:
                    continue
                ts = t.get('ts', 0)
                if ts > 1e10:
                    ts = ts / 1000
                if ts < cutoff_ts:
                    continue
                sym = t.get('symbol', '')
                side = (t.get('side') or '').upper()
                price = t.get('price') or 0
                amount = float(t.get('amount') or 0)
                if not price:
                    continue
                price = float(price)

                if 'BUY' in side:
                    # [Fix] DCA 가중평균: 기존 qty × 기존 avg + 신규 qty × 신규 px
                    pos = sym_pos.get(sym, {'qty': 0.0, 'total_cost': 0.0})
                    new_qty = pos['qty'] + amount
                    new_cost = pos['total_cost'] + price * amount
                    sym_pos[sym] = {'qty': new_qty, 'total_cost': new_cost}

                elif 'CLOSE' in side or 'SELL' in side:
                    pos = sym_pos.get(sym)
                    if pos and pos['qty'] > 0 and pos['total_cost'] > 0:
                        avg_entry = pos['total_cost'] / pos['qty']
                        pnl_pct = ((price - avg_entry) / avg_entry * 100) if avg_entry else 0.0
                        # 청산 수량 차감 (부분 청산 지원)
                        close_qty = amount if amount > 0 else pos['qty']
                        remaining = max(0.0, pos['qty'] - close_qty)
                        if remaining > 1e-9:
                            sym_pos[sym] = {
                                'qty': remaining,
                                'total_cost': avg_entry * remaining,
                            }
                        else:
                            sym_pos[sym] = {'qty': 0.0, 'total_cost': 0.0}
                    else:
                        pnl_pct = 0.0
                    rows.append({
                        'symbol': sym,
                        'market': 'OKX',
                        'side': side,
                        'pnl': pnl_pct,
                        'ts': ts,
                    })
    except Exception as e:
        print(f"[daily_analyzer] trades.jsonl 로드 오류: {e}")
    return rows


def analyze_trades(lookback_days: int = 30):
    trades = []

    # 1) sizing_trade_log.csv (원래 소스)
    if os.path.exists(LOG_FILE):
        cutoff = datetime.now(KST) - timedelta(days=lookback_days)
        with open(LOG_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = datetime.fromisoformat(row["timestamp"])
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=KST)
                    if ts.astimezone(KST) < cutoff:
                        continue
                    row["ts"] = ts
                    row["pnl"] = float(row["pnl_pct"])
                    side = (row.get("side") or "").upper()
                    if side in {"BUY", "SELL"} and row["pnl"] == 0:
                        continue
                    if row.get("market", "").upper() == "UPBIT":
                        continue
                    trades.append(row)
                except Exception:
                    continue

    # 2) [Fix] state/trades.jsonl (실거래 원본)
    jsonl_trades = _load_trades_jsonl(lookback_days)
    trades.extend(jsonl_trades)
    print(f"[daily_analyzer] CSV={len(trades)-len(jsonl_trades)}건, JSONL={len(jsonl_trades)}건 합산={len(trades)}건")

    if not trades:
        return {
            "bad_symbols": {"OKX": []},
            "summary": {},
            "lookback_days": lookback_days,
            "trade_count": 0,
        }

    sym_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})
    for t in trades:
        sym = t["symbol"]
        pnl = t["pnl"]
        sym_stats[sym]["pnl"] += pnl
        if pnl > 0:
            sym_stats[sym]["wins"] += 1
        else:
            sym_stats[sym]["losses"] += 1

    summary = {"OKX": {}}
    total = len(trades)
    wins = sum(1 for r in trades if r["pnl"] > 0)
    losses = sum(1 for r in trades if r["pnl"] < 0)
    pnl_sum = sum(r["pnl"] for r in trades)
    summary["OKX"] = {
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / total * 100.0) if total else 0.0,
        "avg_pnl_pct": (pnl_sum / total) if total else 0.0,
        "sum_pnl_pct": pnl_sum,
    }

    bad_symbols = {"OKX": []}
    for sym, s in sym_stats.items():
        total_sym = s["wins"] + s["losses"]
        if total_sym >= 3:
            avg_pnl = s["pnl"] / total_sym
            if avg_pnl <= -1.0:
                bad_symbols["OKX"].append(sym)

    bad_symbols["OKX"] = sorted(set(bad_symbols["OKX"]))

    return {
        "bad_symbols": bad_symbols,
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
        ex = ccxt.okx()
        ohlcv = ex.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=24)
        if ohlcv:
            vol_sum = 0
            for row in ohlcv:
                high, low, close = row[2], row[3], row[4]
                vol_sum += (high - low) / close
            return (vol_sum / len(ohlcv)) * 100
    except Exception as e:
        print(f"변동성 계산 실패: {e}")
    return 1.0

def calculate_dynamic_params(volatility_pct):
    """변동성에 따라 파라미터 자동 튜닝"""
    okx_p = DEFAULT_OKX_PARAMS.copy()
    return _apply_regime_tuning(okx_p, volatility_pct)

def run_analyzer():
    print(f"[{datetime.now(KST)}] 일일 분석 및 자동 튜닝 시작...")
    analysis = analyze_trades(lookback_days=30)
    bad_symbols = analysis["bad_symbols"]
    
    volatility = get_market_volatility()
    okx_p, regime = calculate_dynamic_params(volatility)
    existing = load_existing_config()

    summary = analysis.get("summary", {})
    okx_n = int(summary.get("OKX", {}).get("trades", 0))
    
    if okx_n < MIN_SAMPLES_PER_MARKET:
        bad_symbols["OKX"] = existing.get("blacklisted_symbols", {}).get("OKX", bad_symbols["OKX"])

    config = {
        "blocked_hours_kst": [],
        "blocked_hours_kst_okx": [],
        "blacklisted_symbols": bad_symbols,
        "okx_params": okx_p
    }

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    okx_summary = summary.get("OKX", {})
    okx_blacklist_text = ", ".join(bad_symbols.get("OKX", [])) or "None"

    report = (
        "📊 [Bot Daily Auto-Tuning Report]\n"
        "=========================\n"
        f"📦 Lookback: {analysis['lookback_days']}d | Trades: {analysis['trade_count']}\n"
        f"📈 Market Regime: {regime}\n"
        f"🔹 BTC 24h Avg Hourly Volatility: {volatility:.2f}%\n"
        "\n"
        "📌 [Market Summary]\n"
        f"OKX  | N={okx_summary.get('trades', 0)} | WR={okx_summary.get('win_rate', 0.0):.1f}% | Avg={okx_summary.get('avg_pnl_pct', 0.0):+.3f}% | Sum={okx_summary.get('sum_pnl_pct', 0.0):+.2f}%\n"
        "\n"
        "⛔ [Blocked KST Hours]\n"
        "None (Crypto runs 24/7)\n"
        "\n"
        "🚫 [Blacklisted Symbols - OKX]\n"
        f"{okx_blacklist_text}\n"
        "\n"
        "⚙️ [Adjusted Params]\n"
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
