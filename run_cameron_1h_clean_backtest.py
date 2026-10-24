# -*- coding: utf-8 -*-
"""
run_cameron_1h_clean_backtest.py — Lookahead-Free Alpaca Cameron 1H Sweep Backtest
- Clean Morning Open Gap (09:30 AM) instead of Lookahead Daily Close
- Evaluates Cutoffs (14:30, 15:30) x Risk:Reward (1:1.5 ~ 1:3.5)
- High-Speed Pre-Indexed Pipeline (< 10 seconds total execution)
"""

import os
import sys
import glob
import pickle
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
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


def evaluate(name, rr_target, cutoff_str, trades, curve):
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
        "cutoff": cutoff_str,
        "rr": f"1:{rr_target:.1f}",
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
    daily_files = glob.glob(os.path.join(CACHE, "alpaca_daily_*.pkl"))
    if not daily_files:
        raise FileNotFoundError("No alpaca_daily_*.pkl found in cache")
    daily_cache_path = sorted(daily_files)[-1]
    with open(daily_cache_path, "rb") as f:
        daily = pickle.load(f)

    universe = {}
    for sym, df in daily.items():
        last = df.iloc[-1]
        dv = (df["close"] * df["volume"]).tail(20).median()
        if MIN_PRICE <= last["close"] <= MAX_PRICE and dv >= 1_000_000:
            universe[sym] = df

    date_map = {}
    for sym, df in universe.items():
        times = df["time"].dt.date.values
        opens = df["open"].values
        closes = df["close"].values
        for i in range(1, len(df)):
            d = times[i]
            if d not in date_map:
                date_map[d] = []
            date_map[d].append((sym, opens[i], closes[i], closes[i - 1]))

    start_date = pd.to_datetime("2026-04-13").date()
    calendar = sorted([d for d in date_map.keys() if d >= start_date])

    print("=========================================================================================", flush=True)
    print(f"📊 [클린 1시간봉(1H) 무미래참조 백테스트] 2026-04-13 ~ 2026-10-13 (6개월)", flush=True)
    print(f"   스캐너: 당일 09:30 시가(Open) 갭상승률 >= +{MIN_GAP_PCT}% (미래 종가 Lookahead 100% 제거)", flush=True)
    print(f"   유니버스: {len(universe)}개 유동성 종목 | 거래일: {len(calendar)}일", flush=True)
    print("=========================================================================================", flush=True)

    cutoffs = [(14, 30), (15, 30)]
    rr_list = [1.5, 2.0, 2.5, 3.0, 3.5]

    configs = []
    for c in cutoffs:
        c_str = f"{c[0]:02d}:{c[1]:02d}"
        for rr in rr_list:
            configs.append({
                "cutoff": c,
                "cutoff_str": c_str,
                "rr": rr,
                "equity": EQUITY0,
                "trades": [],
                "curve": []
            })

    tf = "1Hour"
    t0 = time.time()

    for idx, day in enumerate(calendar):
        entries = date_map.get(day, [])
        movers = []
        for sym, op, cl, pc in entries:
            # [LOOKAHEAD REMOVAL] 당일 09:30 시가(Open)로만 갭상승률 측정
            if MIN_PRICE <= op <= MAX_PRICE and pc > 0:
                pct = (op / pc - 1) * 100
                if pct >= MIN_GAP_PCT:
                    movers.append((pct, sym))

        movers.sort(reverse=True)
        watch = [s for _, s in movers[:WATCH_TOP]]

        if not watch:
            for cfg in configs:
                cfg["curve"].append((day, cfg["equity"]))
            continue

        hcache = os.path.join(CACHE, f"alpaca_{tf}_{day}.pkl")
        if not os.path.exists(hcache):
            for cfg in configs:
                cfg["curve"].append((day, cfg["equity"]))
            continue

        with open(hcache, "rb") as f:
            intraday = pickle.load(f)

        sym_signals = {}
        for sym in watch:
            df = intraday.get(sym)
            if df is None or len(df) < 16:
                continue
            df = df.copy()
            tp = (df["high"] + df["low"] + df["close"]) / 3
            pv = (tp * df["volume"]).rolling(WINDOW).sum()
            vv = df["volume"].rolling(WINDOW).sum().replace(0, 1)
            df["vwap"] = pv / vv
            df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
            day_idx = df.index[df["time"].dt.date == day].tolist()

            signals = []
            for j in day_idx:
                if j < 15:
                    continue
                bar_time = df.loc[j, "time"]
                et = bar_time - pd.Timedelta(hours=4)  # EDT

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
                future_bars = [df.loc[k] for k in day_idx if k > j]
                eod_close = df.loc[day_idx[-1], "close"]

                signals.append({
                    "j": j,
                    "et_hm": (et.hour, et.minute),
                    "trig": trig,
                    "sl": sl,
                    "risk": risk,
                    "entry": entry,
                    "future_bars": future_bars,
                    "eod_close": eod_close
                })
                break

            if signals:
                sym_signals[sym] = signals[0]

        for cfg in configs:
            cutoff = cfg["cutoff"]
            rr = cfg["rr"]
            day_pos = 0

            for sym in watch:
                if day_pos >= MAX_POS:
                    break
                sig = sym_signals.get(sym)
                if not sig:
                    continue
                if sig["et_hm"] >= cutoff:
                    continue

                trig = sig["trig"]
                sl = sig["sl"]
                risk = sig["risk"]
                entry = sig["entry"]
                tp_price = trig + (risk * rr)
                value = cfg["equity"] * POS_PCT

                exit_p = reason = None
                for b in sig["future_bars"]:
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
                    exit_p = sig["eod_close"] * (1 - SLIP)
                    reason = "EOD"

                ret = exit_p / entry - 1
                pnl = value * ret
                cfg["equity"] += pnl
                day_pos += 1
                cfg["trades"].append(dict(day=str(day), sym=sym, ret=ret * 100, pnl=pnl, reason=reason))

            cfg["curve"].append((day, cfg["equity"]))

    # 결과 평가 및 출력
    results = []
    print("\n" + "=" * 95, flush=True)
    print(f"📊 [클린 1시간봉(1Hour) 무미래참조 백테스트 결과 요약표] (연산시간: {time.time()-t0:.1f}초)", flush=True)
    print("=" * 95, flush=True)
    header = f"{'컷오프':^7} | {'RR':^6} | {'거래수':^5} | {'승률':^6} | {'PF':^5} | {'수익률':^8} | {'CAGR':^7} | {'MDD':^6} | {'TP도달율':^7} | {'실현RR':^6}"
    print(header, flush=True)
    print("-" * 95, flush=True)

    for cfg in configs:
        res = evaluate(f"{cfg['cutoff_str']}_{cfg['rr']}", cfg["rr"], cfg["cutoff_str"], cfg["trades"], cfg["curve"])
        results.append(res)
        row = (f"{res['cutoff']:^7} | {res['rr']:^6} | {res['trades']:>5} | {res['win_rate']:>5.1f}% | "
               f"{res['pf']:>5.2f} | {res['total_ret']:>+7.2f}% | {res['cagr']:>6.1f}% | {res['mdd']:>5.2f}% | "
               f"{res['tp_pct']:>6.1f}% | {res['realized_rr']:>6.2f}")
        print(row, flush=True)

    df_res = pd.DataFrame(results)
    out_csv = os.path.join(CURRENT_DIR, "reports", "alpaca_1h_clean_sweep_results.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_res.to_csv(out_csv, index=False)
    print("\n" + "=" * 95, flush=True)
    print(f"✅ 클린 결과 파일 저장 완료: {out_csv}", flush=True)
    print("=" * 95, flush=True)


if __name__ == "__main__":
    main()
