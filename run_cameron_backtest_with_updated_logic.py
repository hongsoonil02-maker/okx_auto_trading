# -*- coding: utf-8 -*-
"""
run_cameron_backtest_with_updated_logic.py
수정된 로직(동적 켈리 공식, 20% 비중, 5개 동시진입, 30분봉 유연화 필터) 종합 백테스트
기간: 2026-04-13 ~ 2026-10-13 (6개월, 126 거래일)
단일 패스 고속 최적화 (사전 인덱싱 및 단일 패스 캐시 로드)
"""

import os
import sys
import pickle
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
SLIP = 0.001
EQUITY0 = 100_000.0
WINDOW = 40
CUTOFF = (14, 30)  # 14:30 ET


def evaluate(name, trades, curve):
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
        final_equity = eq.iloc[-1]
    else:
        pf = wr = ret = cagr = mdd = tp_pct = sl_pct = eod_pct = realized_rr = 0.0
        n = tp_cnt = sl_cnt = eod_cnt = 0
        final_equity = EQUITY0

    return {
        "name": name,
        "trades": n,
        "win_rate": round(wr, 1),
        "pf": round(pf, 2),
        "total_ret": round(ret, 2),
        "final_equity": round(final_equity, 2),
        "cagr": round(cagr * 100, 1),
        "mdd": round(mdd * 100, 2),
        "tp_pct": round(tp_pct, 1),
        "sl_pct": round(sl_pct, 1),
        "eod_pct": round(eod_pct, 1),
        "realized_rr": round(realized_rr, 2)
    }


def calc_dynamic_kelly(vol_ratio: float, retrace_ratio: float, max_pct: float = 0.20) -> float:
    """Half-Kelly: 16% ~ 20% 동적 사이징"""
    conviction = 1.0
    if vol_ratio > 1.1:
        conviction -= 0.10
    if retrace_ratio > 0.45:
        conviction -= 0.10
    return max(0.16, min(max_pct, max_pct * conviction))


def extract_signals(df, day, is_flexible: bool, max_sl: float):
    """주어진 종목 데이터프레임에서 셋업 시그널 추출"""
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

        if (et.hour, et.minute) >= CUTOFF:
            continue

        prev, curr = df.loc[j - 1], df.loc[j]

        # 캔들 형태 검증
        if is_flexible:
            body = abs(prev["close"] - prev["open"])
            rng = prev["high"] - prev["low"]
            is_bearish = prev["close"] < prev["open"]
            is_doji = rng > 0 and (body / rng <= 0.45)
            if not (is_bearish or is_doji):
                continue
            # VWAP / EMA9 유연 버퍼 (3%)
            if prev["close"] < prev["ema9"] * 0.97 or prev["low"] < prev["vwap"] * 0.97:
                continue
        else:
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

        retrace = (i_high - prev["low"]) / surge
        max_retrace = 0.60 if is_flexible else 0.50
        if retrace > max_retrace:
            continue

        ivol = prior["volume"].mean()
        max_vol_mult = 1.35 if is_flexible else 1.10
        if ivol > 0 and prev["volume"] > ivol * max_vol_mult:
            continue

        trig, sl = prev["high"], prev["low"]
        risk = trig - sl
        if risk <= 0:
            continue
        rpct = risk / trig
        if rpct > max_sl or rpct < 0.01:
            continue
        if curr["high"] < trig:
            continue

        vol_ratio = prev["volume"] / ivol if ivol > 0 else 1.0
        future_bars = [df.loc[k] for k in day_idx if k > j]
        eod_close = df.loc[day_idx[-1], "close"]

        return {
            "trig": trig,
            "sl": sl,
            "risk": risk,
            "entry": trig * (1 + SLIP),
            "vol_ratio": vol_ratio,
            "retrace": retrace,
            "future_bars": future_bars,
            "eod_close": eod_close
        }
    return None


def main():
    import glob
    daily_files = glob.glob(os.path.join(CACHE, "alpaca_daily_*.pkl"))
    if not daily_files:
        raise FileNotFoundError("No alpaca_daily_*.pkl found in cache")
    daily_cache_path = sorted(daily_files)[-1]
    print(f"📂 데일리 캐시 로드: {os.path.basename(daily_cache_path)}", flush=True)
    with open(daily_cache_path, "rb") as f:
        daily = pickle.load(f)

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

    # 초고속 데일리 룩업 인덱싱
    print("⚡ 데일리 모멘텀 사전 인덱싱 중...", flush=True)
    daily_movers = {}
    for sym, df in universe.items():
        df_temp = df.copy()
        df_temp["day"] = df_temp["time"].dt.normalize()
        df_temp["prev_close"] = df_temp["close"].shift(1)
        df_temp["pct"] = (df_temp["close"] / df_temp["prev_close"] - 1) * 100
        for _, r in df_temp.iterrows():
            d = r["day"]
            if d not in daily_movers:
                daily_movers[d] = []
            if r["pct"] >= MIN_GAP_PCT and MIN_PRICE <= r["close"] <= MAX_PRICE:
                daily_movers[d].append((r["pct"], sym))

    for d in daily_movers:
        daily_movers[d].sort(reverse=True)

    calendar = sorted(daily_movers.keys())
    calendar = [d for d in calendar if d >= pd.Timestamp(start.date(), tz="UTC")]

    print(f"=========================================================================================", flush=True)
    print(f"📊 [수정된 로직 반영 종합 백테스트] 기간: {start.date()} ~ {end.date()} (6개월 / 126 거래일)", flush=True)
    print(f"   유니버스: {len(universe)}개 종목 | 초기 자본: ${EQUITY0:,.2f}", flush=True)
    print(f"=========================================================================================", flush=True)

    configs = [
        {
            "name": "1. [기존 30M] Flat 10% / Max 3개 / SL 4% 제한",
            "tf": "30Min", "is_flexible": False, "max_sl": 0.04, "pos_cap": 3, "sizing": "flat_10",
            "equity": EQUITY0, "trades": [], "curve": []
        },
        {
            "name": "2. [개선 30M] Flat 20% / Max 5개 / SL 7% 허용",
            "tf": "30Min", "is_flexible": True, "max_sl": 0.07, "pos_cap": 5, "sizing": "flat_20",
            "equity": EQUITY0, "trades": [], "curve": []
        },
        {
            "name": "3. [개선 30M] 동적 Half-Kelly (16~20%) / Max 5개 / SL 7%",
            "tf": "30Min", "is_flexible": True, "max_sl": 0.07, "pos_cap": 5, "sizing": "dynamic_kelly",
            "equity": EQUITY0, "trades": [], "curve": []
        },
        {
            "name": "4. [개선 1H] 동적 Half-Kelly (16~20%) / Max 5개 / SL 5%",
            "tf": "1Hour", "is_flexible": True, "max_sl": 0.05, "pos_cap": 5, "sizing": "dynamic_kelly",
            "equity": EQUITY0, "trades": [], "curve": []
        }
    ]

    t0 = datetime.now()
    rr_target = 3.0

    for idx, day in enumerate(calendar):
        if (idx + 1) % 25 == 0 or idx == len(calendar) - 1:
            elapsed = (datetime.now() - t0).total_seconds()
            print(f"  ⏳ 진행률: {idx + 1}/{len(calendar)} 일 완료 ({elapsed:.1f}초 경과)", flush=True)

        movers = daily_movers.get(day, [])
        watch = [s for _, s in movers[:WATCH_TOP]]

        if not watch:
            for cfg in configs:
                cfg["curve"].append((day, cfg["equity"]))
            continue

        # 캐시 로드 (30M, 1H 딱 1번만 로드)
        intraday_data = {}
        for tf in ["30Min", "1Hour"]:
            hcache = os.path.join(CACHE, f"alpaca_{tf}_{day.date()}.pkl")
            if os.path.exists(hcache):
                with open(hcache, "rb") as f:
                    intraday_data[tf] = pickle.load(f)
            else:
                intraday_data[tf] = {}

        # 4개 설정별 시뮬레이션
        for cfg in configs:
            tf = cfg["tf"]
            intra = intraday_data.get(tf, {})
            pos_cap = cfg["pos_cap"]
            sizing = cfg["sizing"]
            is_flexible = cfg["is_flexible"]
            max_sl = cfg["max_sl"]

            day_pos = 0
            for sym in watch:
                if day_pos >= pos_cap:
                    break
                df = intra.get(sym)
                if df is None or len(df) < 16:
                    continue
                df = df.copy()

                sig = extract_signals(df, day, is_flexible, max_sl)
                if not sig:
                    continue

                # 포지션 사이징
                if sizing == "flat_10":
                    alloc_pct = 0.10
                elif sizing == "flat_20":
                    alloc_pct = 0.20
                elif sizing == "dynamic_kelly":
                    alloc_pct = calc_dynamic_kelly(sig["vol_ratio"], sig["retrace"], max_pct=0.20)

                value = cfg["equity"] * alloc_pct
                entry = sig["entry"]
                sl = sig["sl"]
                tp_price = sig["trig"] + (sig["risk"] * rr_target)

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
                cfg["trades"].append(dict(day=str(day.date()), sym=sym, ret=ret * 100, pnl=pnl, reason=reason, alloc_pct=alloc_pct))

            cfg["curve"].append((day, cfg["equity"]))

    # 결과 평가 및 출력
    results = []
    print("\n" + "=" * 115, flush=True)
    print("📊 [최종 백테스트 결과 비교표 (6개월 / 126 거래일)]", flush=True)
    print("=" * 115, flush=True)
    header = f"{'전략 시나리오':<38} | {'거래수':>5} | {'승률':>6} | {'PF':>5} | {'수익률':>8} | {'최종자산':>12} | {'CAGR':>6} | {'MDD':>6} | {'실현RR':>6}"
    print(header, flush=True)
    print("-" * 115, flush=True)
    for cfg in configs:
        r = evaluate(cfg["name"], cfg["trades"], cfg["curve"])
        results.append(r)
        row = (f"{r['name']:<38} | {r['trades']:>5} | {r['win_rate']:>5.1f}% | {r['pf']:>5.2f} | "
               f"{r['total_ret']:>+7.2f}% | ${r['final_equity']:>11,.2f} | {r['cagr']:>5.1f}% | {r['mdd']:>5.2f}% | {r['realized_rr']:>6.2f}")
        print(row, flush=True)
    print("=" * 115, flush=True)

    # Save to report CSV
    df_res = pd.DataFrame(results)
    out_csv = os.path.join(CURRENT_DIR, "reports", "updated_logic_backtest_results.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\n💾 리포트 CSV 저장 완료: {out_csv}", flush=True)


if __name__ == "__main__":
    main()
