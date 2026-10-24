# -*- coding: utf-8 -*-
"""
run_cameron_30m_rr_sweep.py — Alpaca Cameron 30분봉 손익비(Risk:Reward) 파라미터 스윕 백테스트
- 대상 손익비: 1:1.5, 1:2.0, 1:2.5, 1:3.0, 1:3.5
- 컷오프: 14:30 ET (현재 봇 적용 컷오프) 및 15:30 ET (기존 컷오프) 비교 포함
- 캐시 데이터(state/bt_cache/) 100% 활용
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))
CACHE = os.path.join(CURRENT_DIR, "state", "bt_cache")

MIN_PRICE, MAX_PRICE = 2.0, 50.0
MIN_GAP_PCT = 3.0
WATCH_TOP = 5
POS_PCT = 0.10
MAX_POS = 3
MIN_SL, MAX_SL = 0.01, 0.04
SLIP = 0.001
EQUITY0 = 100_000.0
WINDOW = 40


def run_30m_rr(rr_target, daily, universe, calendar, cutoff=(14, 30)):
    trades = []
    equity = EQUITY0
    curve = []
    tf = "30Min"

    for day in calendar:
        movers = []
        for sym, df in universe.items():
            row = df[df["time"].dt.normalize() == day]
            if row.empty:
                continue
            i = row.index[0]
            if i == 0:
                continue
            pc = df["close"].loc[i - 1]
            px = row.iloc[0]["close"]
            if pd.isna(pc) or pc <= 0:
                continue
            pct = (px / pc - 1) * 100
            if pct >= MIN_GAP_PCT and MIN_PRICE <= px <= MAX_PRICE:
                movers.append((pct, sym))
        movers.sort(reverse=True)
        watch = [s for _, s in movers[:WATCH_TOP]]
        if not watch:
            curve.append((day, equity))
            continue

        hcache = os.path.join(CACHE, f"alpaca_{tf}_{day.date()}.pkl")
        if not os.path.exists(hcache):
            curve.append((day, equity))
            continue

        with open(hcache, "rb") as f:
            intraday = pickle.load(f)

        day_positions = 0
        for sym in watch:
            if day_positions >= MAX_POS:
                break
            df = intraday.get(sym)
            if df is None or len(df) < 16:
                continue
            df = df.copy()
            tp = (df["high"] + df["low"] + df["close"]) / 3
            pv = (tp * df["volume"]).rolling(WINDOW).sum()
            vv = df["volume"].rolling(WINDOW).sum().replace(0, 1)
            df["vwap"] = pv / vv
            df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
            day_idx = df.index[df["time"].dt.normalize() == day].tolist()

            for j in day_idx:
                if j < 15:
                    continue
                bar_time = df.loc[j, "time"]
                et = bar_time - pd.Timedelta(hours=4)  # EDT
                if (et.hour, et.minute) >= cutoff:
                    break

                prev, curr = df.loc[j - 1], df.loc[j]
                if not (prev["close"] < prev["open"]):
                    continue
                if prev["close"] < prev["ema9"] * 0.99 or prev["low"] < prev["vwap"] * 0.985:
                    continue

                prior = df.loc[j - 6:j - 2]
                if len(prior) < 5:
                    continue
                i_low, i_high = prior["low"].min(), prior["high"].max()
                surge = i_high - i_low
                if surge <= 0 or surge / i_low < 0.02:
                    continue
                if (i_high - prev["low"]) / surge > 0.50:
                    continue

                ivol = prior["volume"].mean()
                if ivol > 0 and prev["volume"] > ivol * 1.1:
                    continue

                trig, sl = prev["high"], prev["low"]
                risk = trig - sl
                if risk <= 0:
                    continue
                rpct = risk / trig
                if rpct > MAX_SL or rpct < MIN_SL:
                    continue
                if curr["high"] < trig:
                    continue

                entry = trig * (1 + SLIP)
                tp_price = trig + (risk * rr_target)
                value = equity * POS_PCT
                exit_p = reason = None

                for k in [k for k in day_idx if k > j]:
                    b = df.loc[k]
                    hit_sl, hit_tp = b["low"] <= sl, b["high"] >= tp_price
                    if hit_sl and hit_tp:
                        exit_p, reason = sl * (1 - SLIP), "SL_AMBIG"
                        break
                    if hit_sl:
                        exit_p, reason = sl * (1 - SLIP), "SL"
                        break
                    if hit_tp:
                        exit_p, reason = tp_price * (1 - SLIP), "TP"
                        break

                if exit_p is None:
                    exit_p = df.loc[day_idx[-1], "close"] * (1 - SLIP)
                    reason = "EOD"

                ret = exit_p / entry - 1
                pnl = value * ret
                equity += pnl
                day_positions += 1
                trades.append(dict(day=str(day.date()), sym=sym, ret=ret * 100, pnl=pnl, reason=reason))
                break

        curve.append((day, equity))
    return trades, curve


def evaluate(name, rr, trades, curve):
    eq = pd.Series({d: v for d, v in curve}).sort_index()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.01)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    mdd = ((eq.cummax() - eq) / eq.cummax()).max()

    if trades:
        arr = np.array([t["pnl"] for t in trades])
        reasons = [t["reason"] for t in trades]
        gw = arr[arr > 0].sum()
        gl = -arr[arr <= 0].sum()
        pf = gw / gl if gl > 0 else float("inf")
        wr = (arr > 0).mean() * 100
        n = len(arr)
        ret = (eq.iloc[-1] / EQUITY0 - 1) * 100
        tp_cnt = reasons.count("TP")
        sl_cnt = reasons.count("SL") + reasons.count("SL_AMBIG")
        eod_cnt = reasons.count("EOD")
        tp_pct = (tp_cnt / n) * 100
        sl_pct = (sl_cnt / n) * 100
        eod_pct = (eod_cnt / n) * 100
        avg_win = arr[arr > 0].mean() if len(arr[arr > 0]) > 0 else 0
        avg_loss = -arr[arr <= 0].mean() if len(arr[arr <= 0]) > 0 else 0
        realized_rr = avg_win / avg_loss if avg_loss > 0 else 0
    else:
        pf = wr = ret = cagr = mdd = tp_pct = sl_pct = eod_pct = realized_rr = 0.0
        n = tp_cnt = sl_cnt = eod_cnt = 0

    return {
        "name": name,
        "rr": f"1:{rr:.1f}",
        "trades": n,
        "win_rate": round(wr, 1),
        "pf": round(pf, 2),
        "total_ret": round(ret, 2),
        "cagr": round(cagr * 100, 1),
        "mdd": round(mdd * 100, 2),
        "tp_pct": round(tp_pct, 1),
        "sl_pct": round(sl_pct, 1),
        "eod_pct": round(eod_pct, 1),
        "realized_rr": round(realized_rr, 2)
    }


def main():
    import glob
    daily_files = glob.glob(os.path.join(CACHE, "alpaca_daily_*.pkl"))
    if not daily_files:
        raise FileNotFoundError("No alpaca_daily_*.pkl found in cache")
    daily_cache_path = sorted(daily_files)[-1]
    print(f"📂 데일리 캐시 로드: {os.path.basename(daily_cache_path)}")
    with open(daily_cache_path, "rb") as f:
        daily = pickle.load(f)

    # 파싱: alpaca_daily_2026-04-13_2026-10-13.pkl
    base = os.path.basename(daily_cache_path).replace("alpaca_daily_", "").replace(".pkl", "")
    s_str, e_str = base.split("_")
    start = datetime.strptime(s_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(e_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    universe = {}
    for sym, df in daily.items():
        last = df.iloc[-1]
        dv = (df["close"] * df["volume"]).tail(20).median()
        if MIN_PRICE <= last["close"] <= MAX_PRICE and dv >= 1_000_000:
            universe[sym] = df

    calendar = sorted({t.normalize() for df in universe.values() for t in df["time"]})
    calendar = [d for d in calendar if d >= pd.Timestamp(start.date(), tz="UTC")]

    print(f"=========================================================================================")
    print(f"📊 [Alpaca 30분봉 손익비(RR) 스윕 백테스트] 기간: {start.date()} ~ {end.date()} (6개월)")
    print(f"   유니버스: {len(universe)}개 종목 | 거래일: {len(calendar)}일")
    print(f"=========================================================================================")

    rr_list = [1.5, 2.0, 2.5, 3.0, 3.5]
    results = []

    # 1. 현재 봇 적용 컷오프 (14:30 EST)
    print("\n▶ [14:30 컷오프 (현재 봇 최적화 버전)]")
    for rr in rr_list:
        trades, curve = run_30m_rr(rr, daily, universe, calendar, cutoff=(14, 30))
        res = evaluate(f"30M (14:30, 1:{rr:.1f})", rr, trades, curve)
        results.append(res)
        print(f"  RR 1:{rr:<3.1f} | N: {res['trades']:>3} | 승률: {res['win_rate']:>5.1f}% | PF: {res['pf']:>4.2f} | "
              f"수익률: {res['total_ret']:>+6.2f}% | MDD: {res['mdd']:>4.2f}% | TP비율: {res['tp_pct']:>4.1f}% | 실현RR: {res['realized_rr']:>4.2f}")

    # 2. 기존 컷오프 (15:30 EST)
    print("\n▶ [15:30 컷오프 (기존 버전)]")
    for rr in rr_list:
        trades, curve = run_30m_rr(rr, daily, universe, calendar, cutoff=(15, 30))
        res = evaluate(f"30M (15:30, 1:{rr:.1f})", rr, trades, curve)
        results.append(res)
        print(f"  RR 1:{rr:<3.1f} | N: {res['trades']:>3} | 승률: {res['win_rate']:>5.1f}% | PF: {res['pf']:>4.2f} | "
              f"수익률: {res['total_ret']:>+6.2f}% | MDD: {res['mdd']:>4.2f}% | TP비율: {res['tp_pct']:>4.1f}% | 실현RR: {res['realized_rr']:>4.2f}")

    # CSV 저장
    df_res = pd.DataFrame(results)
    out_csv = os.path.join(CURRENT_DIR, "reports", "alpaca_30m_rr_sweep_results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_res.to_csv(out_csv, index=False)
    print(f"\n✅ 결과 파일 저장 완료: {out_csv}")


if __name__ == "__main__":
    main()
