# -*- coding: utf-8 -*-
"""
kis_nasdaq_backtest.py — JEV 전략 1분봉 백테스트 & 파라미터 그리드 서치
kis_nasdaq_trader.py의 전략 로직(레짐 게이트, 1m+5m 신호, 분할익절, 트레일링, 본절방어, SL)을
yfinance 1분봉(최근 7일)으로 재현하고 파라미터 조합의 수수료 차감 후 순수익을 비교한다.

주의: 1분봉 7일 표본은 상대 비교용이며 절대 성과 보장 근거가 아님. 슬리피지 0, 종가 체결 가정.
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

FEE_SIDE = float(os.getenv("KIS_FEE_PCT_PER_SIDE", "0.0025"))
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
    df["ema9"] = c.ewm(span=9, adjust=False).mean()
    df["ema21"] = c.ewm(span=21, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    d = c.diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    l = (-d.where(d < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + g / (l + 1e-9))
    return df


def load_data():
    print("📥 Downloading 1m bars (7d) for universe + QQQ ...")
    data_1m, data_5m = {}, {}
    for sym in UNIVERSE + ["QQQ"]:
        df = yf.Ticker(sym).history(period="7d", interval="1m")
        if df is None or df.empty:
            print(f"  ⚠️ {sym}: no data")
            continue
        df = df.rename(columns={"Close": "close", "High": "high", "Low": "low", "Open": "open"})
        df = df.between_time("09:30", "16:00")
        data_1m[sym] = add_indicators(df[["close", "high", "low", "open"]].copy())
        df5 = df[["close", "high", "low", "open"]].resample("5min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        data_5m[sym] = add_indicators(df5)
    print(f"  ✓ loaded {len(data_1m)} symbols, {len(data_1m.get('QQQ', []))} 1m bars each (approx)")
    return data_1m, data_5m


def simulate(d1m, d5m, arm_mult, delta_mult, use_partial, use_mtf):
    qqq1, qqq5 = d1m["QQQ"], d5m["QQQ"]
    clock = qqq1.index
    # 5m 레짐을 1m 시계에 매핑
    q5 = qqq5.reindex(clock, method="ffill")
    bull = (q5["close"] > q5["ema50"] * 1.0015) & (
        ((q5["ema9"] > q5["ema21"]) & (q5["close"] > q5["ema9"]) & (q5["rsi"] > 48.0)) | (q5["close"] > q5["ema21"]))
    bear = (q5["close"] < q5["ema50"] * 0.9985) & (
        ((q5["ema9"] < q5["ema21"]) & (q5["close"] < q5["ema9"]) & (q5["rsi"] < 52.0)) | (q5["close"] < q5["ema21"]))

    # 심볼별 1m 지표를 마스터 시계에 정렬
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
    pos = {}          # sym -> {qty, entry, peak, partial}
    cooldown = {}     # sym -> minute index until
    trades = []       # net pct per closed round trip (qty-weighted usd pnl)
    equity_curve = []

    for i in range(60, len(clock)):  # 지표 워밍업
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
            arm *= arm_mult
            delta *= delta_mult
            delta_eff = delta * 2.0 if p["partial"] else delta
            drop = (peak - px) / peak

            def sell(sell_qty, reason):
                nonlocal cash
                proceeds = sell_qty * px * (1 - FEE_SIDE)
                cash += proceeds
                cost = sell_qty * entry
                trades.append((proceeds - cost) / cost * 100.0 if cost else 0.0)
                p["qty"] -= sell_qty
                if p["qty"] <= 0:
                    pos.pop(sym, None)
                    cooldown[sym] = minute + (POST_LOSS_CD_MIN if reason == "SL" else NORMAL_CD_MIN)

            if use_partial and (not p["partial"]) and qty >= 2 and max_ret >= arm:
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
                if use_mtf:
                    if bool(df["sell_5m"].iloc[i]):
                        continue
                    e21_5, c_5 = df["e21_5m"].iloc[i], df["c_5m"].iloc[i]
                    if np.isfinite(e21_5) and np.isfinite(c_5) and c_5 < e21_5:
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
                if qty < 1 or qty * c * (1 + FEE_SIDE) > cash:
                    continue
                cash -= qty * c * (1 + FEE_SIDE)
                pos[sym] = {"qty": qty, "entry": c, "peak": c, "partial": False}

        equity_now = cash + sum(p["qty"] * S[s]["close"].iloc[i] for s, p in pos.items() if np.isfinite(S[s]["close"].iloc[i]))
        equity_curve.append(equity_now)

    # 종료 시 평가
    final_eq = equity_curve[-1] if equity_curve else START_EQUITY
    ret = (final_eq / START_EQUITY - 1.0) * 100.0
    n = len(trades)
    wins = sum(1 for t in trades if t > 0)
    arr = np.array(equity_curve) if equity_curve else np.array([START_EQUITY])
    mdd = ((np.maximum.accumulate(arr) - arr) / np.maximum.accumulate(arr)).max() * 100.0
    return {"ret": ret, "trades": n, "win_rate": wins / n * 100.0 if n else 0.0, "mdd": mdd,
            "avg": float(np.mean(trades)) if trades else 0.0}


def main():
    d1m, d5m = load_data()
    if "QQQ" not in d1m:
        print("QQQ 데이터 없음 — 중단")
        sys.exit(1)

    grid = list(itertools.product(
        [0.75, 1.0, 1.25],   # trail arm 배수
        [0.75, 1.0, 1.5],    # trail delta 배수
        [False, True],       # 분할 익절
        [False, True],       # 5분봉 확인
    ))
    print(f"\n🔬 Grid search: {len(grid)} combinations (fees {FEE_SIDE*2*100:.2f}% round-trip)\n")
    results = []
    for arm_m, delta_m, partial, mtf in grid:
        r = simulate(d1m, d5m, arm_m, delta_m, partial, mtf)
        r.update({"arm": arm_m, "delta": delta_m, "partial": partial, "mtf": mtf})
        results.append(r)
        print(f"  arm x{arm_m:<4} delta x{delta_m:<4} partial={int(partial)} mtf={int(mtf)} | "
              f"ret {r['ret']:+.2f}% | {r['trades']} trades | win {r['win_rate']:.0f}% | mdd {r['mdd']:.1f}%")

    results.sort(key=lambda x: x["ret"], reverse=True)
    print("\n" + "=" * 78)
    print("🏆 TOP 5 (순수익 기준, 수수료 차감 후)")
    print("=" * 78)
    for r in results[:5]:
        print(f"  ret {r['ret']:+.2f}% | arm x{r['arm']} delta x{r['delta']} partial={int(r['partial'])} mtf={int(r['mtf'])} "
              f"| {r['trades']}t win {r['win_rate']:.0f}% mdd {r['mdd']:.1f}% avg {r['avg']:+.2f}%")
    base = [r for r in results if r["arm"] == 1.0 and r["delta"] == 1.0 and r["partial"] and r["mtf"]][0]
    print(f"\n📍 현행 설정 (arm x1.0 delta x1.0 partial on mtf on): ret {base['ret']:+.2f}% | {base['trades']}t | win {base['win_rate']:.0f}% | mdd {base['mdd']:.1f}%")
    print("\n⚠️ 1분봉 7일 표본의 상대 비교 결과이며, 슬리피지 0·종가 체결 가정. 절대 성과 근거로 사용 금지.")


if __name__ == "__main__":
    main()
