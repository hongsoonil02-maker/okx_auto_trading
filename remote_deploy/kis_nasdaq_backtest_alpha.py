# -*- coding: utf-8 -*-
"""
kis_nasdaq_backtest_alpha.py — 알파 개선 3대 아이디어 비교 백테스트
1. 거래량 폭발 (Volume Spike >= 1.5x Vol MA20)
2. 개장 15분 쿼런틴 (Opening Quarantine: 09:30~09:45 ET 진입 금지)
3. QQQ 대비 상대 강도 (Relative Strength: 최근 15m 지수 초과수익 종목 우선)

조건별 매트릭스:
- Baseline (현행 v2.5 로직)
- [1] Volume Spike 단독
- [2] Opening Quarantine 단독
- [3] Relative Strength 단독
- [1+2] Volume + Quarantine
- [1+3] Volume + RS
- [2+3] Quarantine + RS
- [1+2+3] All Combined

각 조합별로 (1) 현행 기본 수수료(편도 0.25%, 왕복 0.5%)와 (2) 우대 수수료(편도 0.07%, 왕복 0.14%) 모두 비교!
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

UNIVERSE = ["TQQQ", "SQQQ", "NVDA", "TSLA", "MSTR", "PLTR", "AMD", "COIN", "AVGO", "META", "AAPL"]
BULLISH_ETFS = {"TQQQ", "SOXL", "QQQ"}
BEARISH_ETFS = {"SQQQ", "SOXS"}
LEVERAGED_3X = {"TQQQ", "SQQQ", "SOXL", "SOXS"}
HIGH_BETA = {"MSTR", "PLTR", "COIN", "TSLA"}

START_EQUITY = 7800.0
MAX_POS = 3
ALLOC = 0.35
BREAKEVEN_FLOOR = 0.007
POST_LOSS_CD_MIN = 30
NORMAL_CD_MIN = 5


def base_risk(sym):
    if sym in LEVERAGED_3X:
        return 0.024, 0.080, 0.025, 0.010
    if sym in HIGH_BETA:
        return 0.025, 0.090, 0.028, 0.012
    if sym in {"NVDA", "AMD", "AVGO", "META"}:
        return 0.015, 0.060, 0.020, 0.008
    return 0.012, 0.045, 0.015, 0.008


def add_indicators(df):
    c = df["close"]
    v = df["volume"] if "volume" in df else pd.Series(0, index=df.index)
    df["ema9"] = c.ewm(span=9, adjust=False).mean()
    df["ema21"] = c.ewm(span=21, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    d = c.diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    l = (-d.where(d < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + g / (l + 1e-9))
    # Volume MA20 & Volume Ratio
    df["vol_ma20"] = v.rolling(20).mean()
    df["vol_ratio"] = v / (df["vol_ma20"] + 1e-6)
    # 15분 모멘텀 (상대강도 계산용)
    df["ret_15m"] = c.pct_change(15)
    return df


def load_data():
    print("📥 1분봉 데이터 다운로드 중 (최근 7일, Universe 11개 + QQQ)...")
    data_1m, data_5m = {}, {}
    for sym in UNIVERSE + ["QQQ"]:
        df = yf.Ticker(sym).history(period="7d", interval="1m")
        if df is None or df.empty:
            print(f"  ⚠️ {sym}: 데이터 없음")
            continue
        df = df.rename(columns={"Close": "close", "High": "high", "Low": "low", "Open": "open", "Volume": "volume"})
        df = df.between_time("09:30", "16:00")
        data_1m[sym] = add_indicators(df[["close", "high", "low", "open", "volume"]].copy())
        df5 = df[["close", "high", "low", "open", "volume"]].resample("5min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        data_5m[sym] = add_indicators(df5)
    print(f"  ✓ {len(data_1m)}개 종목 로드 완료.")
    return data_1m, data_5m


def simulate(d1m, d5m, fee_side, use_volume_filter=False, use_quarantine=False, use_relative_strength=False):
    qqq1, qqq5 = d1m["QQQ"], d5m["QQQ"]
    clock = qqq1.index
    q5 = qqq5.reindex(clock, method="ffill")
    bull = (q5["close"] > q5["ema50"] * 1.0015) & (
        ((q5["ema9"] > q5["ema21"]) & (q5["close"] > q5["ema9"]) & (q5["rsi"] > 48.0)) | (q5["close"] > q5["ema21"])
    )
    bear = (q5["close"] < q5["ema50"] * 0.9985) & (
        ((q5["ema9"] < q5["ema21"]) & (q5["close"] < q5["ema9"]) & (q5["rsi"] < 52.0)) | (q5["close"] < q5["ema21"])
    )

    qqq_ret_15m = d1m["QQQ"]["ret_15m"].reindex(clock, method="ffill")

    S = {}
    for sym in UNIVERSE:
        if sym not in d1m:
            continue
        S[sym] = d1m[sym].reindex(clock, method="ffill")
        s5 = d5m[sym].reindex(clock, method="ffill")
        S[sym]["e21_5m"] = s5["ema21"]
        S[sym]["c_5m"] = s5["close"]
        S[sym]["sell_5m"] = (s5["ema9"] < s5["ema21"]) & (s5["close"] < s5["ema9"]) & (s5["rsi"] < 52.0)

    cash = START_EQUITY
    pos = {}
    cooldown = {}
    trades = []
    equity_curve = []

    for i in range(60, len(clock)):
        current_time = clock[i]
        is_bull, is_bear = bool(bull.iloc[i]), bool(bear.iloc[i])
        minute = i

        # ── Exits ──
        for sym in list(pos.keys()):
            df = S[sym]
            px = df["close"].iloc[i]
            if not np.isfinite(px) or px <= 0:
                continue
            p = pos[sym]
            entry, qty = p["entry"], p["qty"]
            p["peak"] = max(p["peak"], px)
            peak = p["peak"]
            pnl = px / entry - 1.0
            max_ret = peak / entry - 1.0
            sl, tp, arm, delta = base_risk(sym)
            delta_eff = delta * 2.0 if p["partial"] else delta
            drop = (peak - px) / peak

            def sell(sell_qty, reason):
                nonlocal cash
                proceeds = sell_qty * px * (1 - fee_side)
                cash += proceeds
                cost = sell_qty * entry
                trades.append((proceeds - cost) / cost * 100.0 if cost else 0.0)
                p["qty"] -= sell_qty
                if p["qty"] <= 0:
                    pos.pop(sym, None)
                    cooldown[sym] = minute + (POST_LOSS_CD_MIN if reason == "SL" else NORMAL_CD_MIN)

            # 분할 익절 & 러너 트레일링 로직
            if (not p["partial"]) and qty >= 2 and max_ret >= arm:
                sell(max(1, qty // 2), "PARTIAL")
                p["partial"] = True
            elif max_ret >= arm and drop >= delta_eff:
                sell(qty, "TRAIL")
            elif max_ret >= 0.015 and pnl <= BREAKEVEN_FLOOR:
                sell(qty, "BE")
            elif (not p["partial"]) and pnl >= tp:
                sell(qty, "TP")
            elif pnl <= -sl:
                sell(qty, "SL")

        # ── Entries ──
        if len(pos) < MAX_POS:
            # [아이디어 2] 개장 15분 쿼런틴 (09:30~09:45 ET 진입 금지)
            if use_quarantine:
                time_str = current_time.strftime("%H:%M")
                if "09:30" <= time_str < "09:45":
                    equity_now = cash + sum(p["qty"] * S[s]["close"].iloc[i] for s, p in pos.items() if np.isfinite(S[s]["close"].iloc[i]))
                    equity_curve.append(equity_now)
                    continue

            if is_bull:
                pool = ["TQQQ", "MSTR", "PLTR", "NVDA", "TSLA", "AMD", "COIN", "AVGO", "META"]
            elif is_bear:
                pool = ["SQQQ"]
            else:
                pool = ["PLTR", "NVDA", "TSLA", "MSTR", "META", "AAPL"]

            has_bull_etf = any(s in pos for s in BULLISH_ETFS)
            has_bear_etf = any(s in pos for s in BEARISH_ETFS)
            equity = cash + sum(p["qty"] * S[s]["close"].iloc[i] for s, p in pos.items() if np.isfinite(S[s]["close"].iloc[i]))
            alloc = equity * ALLOC

            scored = []
            q_ret = qqq_ret_15m.iloc[i] if np.isfinite(qqq_ret_15m.iloc[i]) else 0.0

            for sym in pool:
                if sym not in S or sym in pos or minute < cooldown.get(sym, 0):
                    continue
                if sym in BEARISH_ETFS and has_bull_etf:
                    continue
                if sym in BULLISH_ETFS and has_bear_etf:
                    continue
                df = S[sym]
                c, e9, e21, rsi = df["close"].iloc[i], df["ema9"].iloc[i], df["ema21"].iloc[i], df["rsi"].iloc[i]
                if not all(np.isfinite(v) for v in (c, e9, e21, rsi)):
                    continue
                # 기본 기술적 신호 (1분봉 EMA 골든크로스 + RSI)
                if not (e9 > e21 and c > e9 and rsi > 48.0):
                    continue

                # 5분봉 정렬 확인
                if bool(df["sell_5m"].iloc[i]):
                    continue
                e21_5, c_5 = df["e21_5m"].iloc[i], df["c_5m"].iloc[i]
                if np.isfinite(e21_5) and np.isfinite(c_5) and c_5 < e21_5:
                    continue

                # [아이디어 1] 거래량 폭발 필터 (Volume Surge >= 1.5x)
                if use_volume_filter:
                    vol_r = df["vol_ratio"].iloc[i]
                    if np.isfinite(vol_r) and vol_r < 1.4:
                        continue  # 거래량이 터지지 않은 크로스는 가짜 돌파로 간주

                # [아이디어 3] QQQ 대비 상대 강도 (Relative Strength)
                sym_ret = df["ret_15m"].iloc[i] if np.isfinite(df["ret_15m"].iloc[i]) else 0.0
                rs = sym_ret - q_ret
                if use_relative_strength:
                    if rs < 0:  # 지수보다 약한 종목 제외
                        continue
                    # RS 가중치 반영 점수
                    score = (e9 / e21 - 1.0) * 100.0 * 0.4 + max(0.0, rsi - 50.0) * 0.2 + (rs * 100.0) * 0.4
                else:
                    score = (e9 / e21 - 1.0) * 100.0 * 0.6 + max(0.0, rsi - 50.0) * 0.4

                scored.append((score, sym, c))

            scored.sort(reverse=True)

            for score, sym, c in scored:
                if len(pos) >= MAX_POS:
                    break
                qty = int(alloc / c)
                if qty < 1 and cash >= c:
                    qty = 1
                if qty < 1 or qty * c * (1 + fee_side) > cash:
                    continue
                cash -= qty * c * (1 + fee_side)
                pos[sym] = {"qty": qty, "entry": c, "peak": c, "partial": False}

        equity_now = cash + sum(p["qty"] * S[s]["close"].iloc[i] for s, p in pos.items() if np.isfinite(S[s]["close"].iloc[i]))
        equity_curve.append(equity_now)

    final_eq = equity_curve[-1] if equity_curve else START_EQUITY
    ret = (final_eq / START_EQUITY - 1.0) * 100.0
    n = len(trades)
    wins = sum(1 for t in trades if t > 0)
    arr = np.array(equity_curve) if equity_curve else np.array([START_EQUITY])
    mdd = ((np.maximum.accumulate(arr) - arr) / np.maximum.accumulate(arr)).max() * 100.0
    return {
        "ret": ret,
        "trades": n,
        "win_rate": wins / n * 100.0 if n else 0.0,
        "mdd": mdd,
        "avg": float(np.mean(trades)) if trades else 0.0,
    }


def run_experiment():
    d1m, d5m = load_data()
    if "QQQ" not in d1m:
        print("❌ QQQ 데이터 누락으로 중단")
        return

    configs = [
        ("1. Baseline (현재 봇 상태)", False, False, False),
        ("2. [아이디어 1] 거래량 폭발 (Vol >= 1.4x)", True, False, False),
        ("3. [아이디어 2] 개장 15분 쿼런틴 (09:30~09:45 진입차단)", False, True, False),
        ("4. [아이디어 3] 상대 강도 (RS vs QQQ > 0)", False, False, True),
        ("5. [조합 1+2] 거래량 폭발 + 개장 15분 쿼런틴", True, True, False),
        ("6. [조합 1+3] 거래량 폭발 + 상대 강도", True, False, True),
        ("7. [조합 2+3] 개장 15분 쿼런틴 + 상대 강도", False, True, True),
        ("8. [조합 1+2+3] 풀 알파 (거래량 + 쿼런틴 + 상대강도)", True, True, True),
    ]

    fee_scenarios = [
        ("현재 기본 수수료 (왕복 0.50% / 편도 0.25%)", 0.0025),
        ("우대 수수료 적용 시 (왕복 0.14% / 편도 0.07%)", 0.0007),
    ]

    for fee_label, fee_rate in fee_scenarios:
        print("\n" + "=" * 88)
        print(f"📊 시나리오: {fee_label}")
        print("=" * 88)
        print(f"{'전략 설정':<40} | {'수익률(%)':<9} | {'거래수':<6} | {'승률(%)':<7} | {'MDD(%)':<7} | {'평균손익(%)':<10}")
        print("-" * 88)

        for name, vol_f, quar_f, rs_f in configs:
            res = simulate(d1m, d5m, fee_rate, vol_f, quar_f, rs_f)
            print(
                f"{name:<40} | {res['ret']:+6.2f}%   | {res['trades']:<6} | {res['win_rate']:5.1f}%  | {res['mdd']:5.2f}%  | {res['avg']:+6.2f}%"
            )


if __name__ == "__main__":
    run_experiment()
