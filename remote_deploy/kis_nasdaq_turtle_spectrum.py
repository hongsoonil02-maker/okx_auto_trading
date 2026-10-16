# -*- coding: utf-8 -*-
"""
kis_nasdaq_turtle_spectrum.py — 터틀 트레이딩 스타일 승률 스펙트럼 비교 (80% -> 70% -> 60% -> 50% -> 40%)
사용자 의도: "터틀트레이딩처럼 승률을 70%, 60%, 50%로 과감하게 낮추고 손익비를 극대화했을 때의 수익률과 성과 비교"

핵심 가설:
- 터틀 스타일: 손실은 극도로 짧게 자르고(Tight SL), 분할익절 없이 큰 추세를 끝까지 먹음(Huge Runner).
- 승률은 40~50%로 떨어지지만, 손익비(Win Avg / Loss Avg)가 3.0~4.0 이상으로 폭등하여 총수익이 어떻게 되는지 관찰!
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


def simulate_turtle(d1m, d5m, fee_side, sl_pct, arm_pct, drop_pct, use_partial, use_be):
    """
    sl_pct: 손절폭 (예: 0.010 = 1.0% 타이트 손절, 0.025 = 2.5%)
    arm_pct: 트레일링 익절 시작선 (예: 0.025 = 2.5%, 0.050 = 5.0%, 0.080 = 8.0%)
    drop_pct: 고점 대비 허용 낙폭 (예: 0.010 = 1.0%, 0.025 = 2.5%)
    use_partial: 분할익절 여부 (False면 터틀 스타일 전량 추세 추종)
    use_be: 본절 방어선 사용 여부
    """
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

            # 1. 분할 익절 (사용 시)
            if use_partial and (not p["partial"]) and qty >= 2 and max_ret >= arm_pct:
                sell(max(1, qty // 2), "PARTIAL")
                p["partial"] = True
            # 2. 트레일링 익절 (터틀: 큰 파동 끝까지 먹기)
            elif max_ret >= arm_pct and drop >= drop_pct:
                sell(qty, "TRAIL")
            # 3. 본절 락 (사용 시)
            elif use_be and max_ret >= 0.015 and pnl <= BREAKEVEN_FLOOR:
                sell(qty, "BE")
            # 4. 칼손절 (터틀의 핵심: Cut Loss Short)
            elif pnl <= -sl_pct:
                sell(qty, "SL")

        # ── Entries (개장 15분 쿼런틴 포함) ──
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

                # 거래량 필터 (Vol >= 1.2x: 적절한 진입 기회 확보)
                vol_r = df["vol_ratio"].iloc[i]
                if np.isfinite(vol_r) and vol_r < 1.2:
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
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    win_rate = len(wins) / n * 100.0 if n else 0.0
    arr = np.array(equity_curve) if equity_curve else np.array([START_EQUITY])
    mdd = ((np.maximum.accumulate(arr) - arr) / np.maximum.accumulate(arr)).max() * 100.0
    
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    # 손익비 (Payoff Ratio) = 평균수익 / 평균손실(절대값)
    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    return {
        "ret": ret,
        "trades": n,
        "win_rate": win_rate,
        "mdd": mdd,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff": payoff_ratio,
    }


def main():
    d1m, d5m = load_data()

    # 승률 스펙트럼 설계:
    # 1. 초고승률형 (80%대): 타이트 익절 + 분할익절 + 넓은 손절
    # 2. 고승률형 (70%대): 일반 트레일링 + 분할익절
    # 3. 중승률형 (60%대): 분할익절 제거 + Arm 3.5% + 타이트 손절 1.5%
    # 4. 터틀 스타일 (50%대): 분할익절 제거 + Arm 5.0% + 초타이트 손절 1.2%
    # 5. 극단 터틀 (40%대): 분할익절 제거 + Arm 7.0% + 초타이트 손절 1.0%
    
    profiles = [
        # (이름, SL, Arm, Drop, use_partial, use_be)
        ("A. [현행 고승률형] 승률 80%대", 0.022, 0.025, 0.010, True, True),
        ("B. [완만 익절형] 승률 70%대",   0.018, 0.030, 0.012, True, True),
        ("C. [터틀 하이브리드] 승률 60%대", 0.015, 0.040, 0.015, False, False),
        ("D. [정통 터틀 스타일] 승률 50%대", 0.012, 0.055, 0.018, False, False),
        ("E. [극단적 터틀/빅런] 승률 40%대", 0.010, 0.075, 0.022, False, False),
    ]

    for fee_label, fee_rate in [("기본 수수료 (0.50%)", 0.0025), ("우대 수수료 (0.14%)", 0.0007)]:
        print("\n" + "=" * 105)
        print(f"🐢 터틀 스타일 승률 스펙트럼 비교: 80% ➜ 70% ➜ 60% ➜ 50% ➜ 40% ({fee_label})")
        print("=" * 105)
        print(f"{'전략 프로필':<30} | {'승률(%)':<8} | {'손익비':<7} | {'평균수익':<9} | {'평균손실':<9} | {'7일 수익률':<11} | {'거래수':<6} | {'MDD(%)':<7}")
        print("-" * 105)
        for name, sl, arm, drop, partial, be in profiles:
            r = simulate_turtle(d1m, d5m, fee_rate, sl, arm, drop, partial, be)
            print(
                f"{name:<30} | {r['win_rate']:5.1f}%   | {r['payoff']:4.2f}:1  | {r['avg_win']:+6.2f}%   | {r['avg_loss']:+6.2f}%   | {r['ret']:+8.2f}%   | {r['trades']:<6} | {r['mdd']:5.2f}%"
            )


if __name__ == "__main__":
    main()
