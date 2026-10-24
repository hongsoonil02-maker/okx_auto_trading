# -*- coding: utf-8 -*-
"""
kis_nasdaq_tradeoff_test.py — 승률 vs 손익비/수익률 트레이드오프 정밀 탐색
사용자 질문: "승률을 좀 낮추면 수익률은 어떻게 변할까?"
1. 거래량 필터 완화 (Vol Multiplier 1.0x ~ 1.5x): 진입 빈도 늘려 총 누적수익 극대화 시도
2. 트레일링 Arm/Delta 확대 (Arm x1.0 ~ x1.5, Delta x1.0 ~ x1.5): 자잘한 익절 대신 큰 파동 추구
3. 분할 익절 비율 및 TP 목표 확대

기본 수수료(0.50%) 및 우대 수수료(0.14%) 양쪽 환경에서 탐색!
"""
import os
import sys
import itertools
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
    df["vol_ma20"] = v.rolling(20).mean()
    df["vol_ratio"] = v / (df["vol_ma20"] + 1e-6)
    return df


def load_data():
    data_1m, data_5m = {}, {}
    for sym in UNIVERSE + ["QQQ"]:
        df = yf.Ticker(sym).history(period="7d", interval="1m")
        if df is None or df.empty:
            continue
        df = df.rename(columns={"Close": "close", "High": "high", "Low": "low", "Open": "open", "Volume": "volume"})
        df = df.between_time("09:30", "16:00")
        data_1m[sym] = add_indicators(df[["close", "high", "low", "open", "volume"]].copy())
        df5 = df[["close", "high", "low", "open", "volume"]].resample("5min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        data_5m[sym] = add_indicators(df5)
    return data_1m, data_5m


def simulate(d1m, d5m, fee_side, vol_thresh, arm_mult, delta_mult):
    qqq1, qqq5 = d1m["QQQ"], d5m["QQQ"]
    clock = qqq1.index
    q5 = qqq5.reindex(clock, method="ffill")
    bull = (q5["close"] > q5["ema50"] * 1.0015) & (
        ((q5["ema9"] > q5["ema21"]) & (q5["close"] > q5["ema9"]) & (q5["rsi"] > 48.0)) | (q5["close"] > q5["ema21"])
    )
    bear = (q5["close"] < q5["ema50"] * 0.9985) & (
        ((q5["ema9"] < q5["ema21"]) & (q5["close"] < q5["ema9"]) & (q5["rsi"] < 52.0)) | (q5["close"] < q5["ema21"])
    )

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

        # Exits
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
            arm *= arm_mult
            delta *= delta_mult
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

        # Entries (개장 15분 쿼런틴 포함)
        time_str = current_time.strftime("%H:%M")
        if "09:30" <= time_str < "09:45":
            equity_now = cash + sum(p["qty"] * S[s]["close"].iloc[i] for s, p in pos.items() if np.isfinite(S[s]["close"].iloc[i]))
            equity_curve.append(equity_now)
            continue

        if len(pos) < MAX_POS:
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
                if not (e9 > e21 and c > e9 and rsi > 48.0):
                    continue
                if bool(df["sell_5m"].iloc[i]):
                    continue
                e21_5, c_5 = df["e21_5m"].iloc[i], df["c_5m"].iloc[i]
                if np.isfinite(e21_5) and np.isfinite(c_5) and c_5 < e21_5:
                    continue

                # 거래량 필터
                if vol_thresh > 0:
                    vol_r = df["vol_ratio"].iloc[i]
                    if np.isfinite(vol_r) and vol_r < vol_thresh:
                        continue

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


def main():
    d1m, d5m = load_data()

    # 테스트 케이스 1: 거래량 필터 강도 조정 (1.0x ~ 1.5x)
    # 개장 쿼런틴 기본 탑재 상태
    vol_tests = [0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    for fee_label, fee_rate in [("기본 수수료 (0.50%)", 0.0025), ("우대 수수료 (0.14%)", 0.0007)]:
        print("\n" + "=" * 90)
        print(f"📊 [테스트 1] 거래량 필터 강도별 승률 vs 수익률 ({fee_label})")
        print("=" * 90)
        print(f"{'거래량 기준':<20} | {'수익률(%)':<10} | {'거래수':<6} | {'승률(%)':<8} | {'MDD(%)':<8} | {'평균손익(%)':<10}")
        print("-" * 90)
        for v in vol_tests:
            label = "필터 없음 (0.0x)" if v == 0.0 else f"Vol >= {v:.1f}x"
            r = simulate(d1m, d5m, fee_rate, v, 1.0, 1.0)
            print(f"{label:<20} | {r['ret']:+7.2f}%   | {r['trades']:<6} | {r['win_rate']:5.1f}%   | {r['mdd']:5.2f}%   | {r['avg']:+6.2f}%")

    # 테스트 케이스 2: 트레일링 Arm/Delta 확대 (더 큰 파동 추구 vs 승률 낮추기)
    # Vol >= 1.2x 및 Vol >= 1.4x 기준
    for fee_label, fee_rate in [("기본 수수료 (0.50%)", 0.0025), ("우대 수수료 (0.14%)", 0.0007)]:
        print("\n" + "=" * 90)
        print(f"📊 [테스트 2] 트레일링 익절폭(Arm/Delta) 확대 탐색 ({fee_label})")
        print("=" * 90)
        print(f"{'설정 (Vol / Arm / Delta)':<30} | {'수익률(%)':<10} | {'거래수':<6} | {'승률(%)':<8} | {'MDD(%)':<8} | {'평균손익(%)':<10}")
        print("-" * 90)
        grid_params = [
            (1.2, 1.0, 1.0),
            (1.2, 1.2, 1.0),
            (1.2, 1.2, 1.3),
            (1.2, 1.5, 1.5),
            (1.4, 1.0, 1.0),
            (1.4, 1.2, 1.0),
            (1.4, 1.2, 1.3),
            (1.4, 1.5, 1.5),
        ]
        for v, arm, delta in grid_params:
            label = f"Vol {v:.1f}x | Arm x{arm:.1f} | Dlt x{delta:.1f}"
            r = simulate(d1m, d5m, fee_rate, v, arm, delta)
            print(f"{label:<30} | {r['ret']:+7.2f}%   | {r['trades']:<6} | {r['win_rate']:5.1f}%   | {r['mdd']:5.2f}%   | {r['avg']:+6.2f}%")


if __name__ == "__main__":
    main()
