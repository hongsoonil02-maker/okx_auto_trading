# -*- coding: utf-8 -*-
"""
bt_alpaca_cameron_tf.py — 카메론 풀백 전략 타임프레임 비교 (1m/5m/15m/30m/1H)
- 모든 TF에 동일 파라미터 (surge 2%, 되돌림 ≤50%, vol 1.1x, SL 1~4%, RR 1:2, 10% 사이징, 최대 3건, 15:30 컷오프)
- 1m에는 원전 스캘핑 변형(10:30 컷오프) 추가
- 스캐너(일봉 상승률 상위 5종목)는 bt_alpaca_cameron.py와 동일하게 재현
- 지표 VWAP/EMA9는 라이브 봇과 동일하게 최근 40봉 롤링 윈도우
"""
import os
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from bt_alpaca_cameron import get_bars, CACHE, MIN_PRICE, MAX_PRICE, MIN_GAP_PCT, WATCH_TOP, \
    POS_PCT, MAX_POS, RR, MIN_SL, MAX_SL, SLIP, EQUITY0

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

TFS = ["1Min", "5Min", "15Min", "30Min", "1Hour"]
WINDOW = 40  # 라이브 fetch limit=40과 동일한 롤링 VWAP 윈도우


def run_tf(tf, daily, universe, calendar, cutoff=(15, 30)):
    trades = []
    equity = EQUITY0
    curve = []

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
        if os.path.exists(hcache):
            intraday = pickle.load(open(hcache, "rb"))
        else:
            warm = 1 if tf == "1Min" else 4
            intraday = get_bars(watch, tf, (day - timedelta(days=warm)).to_pydatetime(),
                                (day + timedelta(days=1)).to_pydatetime())
            pickle.dump(intraday, open(hcache, "wb"))
            time.sleep(0.15)

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
                et = bar_time - pd.Timedelta(hours=4)  # EDT (백테스트 기간 전부 서머타임)
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
                tp_price = trig + risk * RR
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


def report(name, trades, curve):
    eq = pd.Series({d: v for d, v in curve}).sort_index()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.01)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    mdd = ((eq.cummax() - eq) / eq.cummax()).max()
    if trades:
        arr = np.array([t["pnl"] for t in trades])
        gw = arr[arr > 0].sum()
        gl = -arr[arr <= 0].sum()
        pf = gw / gl if gl > 0 else float("inf")
        wr = (arr > 0).mean() * 100
        n = len(arr)
        ret = (eq.iloc[-1] / EQUITY0 - 1) * 100
    else:
        pf = wr = ret = 0.0
        n = 0
    print(f"  {name:<22} | N {n:>3} | WR {wr:5.1f}% | PF {pf:5.2f} | 수익 {ret:+6.1f}% | CAGR {cagr*100:+6.1f}% | MDD {mdd*100:4.1f}%",
          flush=True)
    return dict(name=name, n=n, wr=wr, pf=pf, ret=ret, cagr=cagr * 100, mdd=mdd * 100)


def main():
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=int(6 * 30.5))
    daily = pickle.load(open(os.path.join(CACHE, f"alpaca_daily_{start.date()}_{end.date()}.pkl"), "rb"))
    universe = {}
    for sym, df in daily.items():
        last = df.iloc[-1]
        dv = (df["close"] * df["volume"]).tail(20).median()
        if MIN_PRICE <= last["close"] <= MAX_PRICE and dv >= 1_000_000:
            universe[sym] = df
    calendar = sorted({t.normalize() for df in universe.values() for t in df["time"]})
    calendar = [d for d in calendar if d >= pd.Timestamp(start.date(), tz="UTC")]
    print(f"유니버스 {len(universe)}종목, {len(calendar)}일 — TF별 비교 시작", flush=True)

    rows = []
    for tf in TFS:
        trades, curve = run_tf(tf, daily, universe, calendar)
        rows.append(report(f"{tf} (15:30)", trades, curve))
    # 원전 스캘핑 변형
    trades, curve = run_tf("1Min", daily, universe, calendar, cutoff=(10, 30))
    rows.append(report("1Min (10:30 원전)", trades, curve))

    print("\n결과 요약 CSV 저장")
    pd.DataFrame(rows).to_csv(os.path.join(CACHE, "alpaca_tf_compare.csv"), index=False)


if __name__ == "__main__":
    main()
