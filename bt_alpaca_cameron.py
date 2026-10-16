# -*- coding: utf-8 -*-
"""
bt_alpaca_cameron.py — Alpaca Cameron 1H Pullback 전략 백테스트 (라이브 봇 로직 재현)
- 스캐너 재현: 매일 (전일종가 대비 +3% 이상 & $2~50) 상승률 상위 5종목 워치리스트
- 셋업: 1H 퍼스트 풀백 (적색봉, EMA9/VWAP 지지, 임펄스 2%+, 되돌림 ≤50%, 거래량 축소)
- 진입: 풀백봉 고가 돌파 시 시장가 (슬리피지 0.1%), SL=풀백봉 저가, TP=1:2
- 사이징: 자산 10%/건, 동시 최대 3건, 15:30 ET 이후 신규 진입 없음, 당일 종가 청산
  (라이브는 day-TIF 브래킷이라 익일 이월 포지션은 무관리 — 백테스트는 당일 청산 근사)
- 페이퍼 수수료 0, 데이터: Alpaca IEX feed
"""
import os
import time
import pickle
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))
CACHE = os.path.join(CURRENT_DIR, "state", "bt_cache")
os.makedirs(CACHE, exist_ok=True)

HEADERS = {"APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY"),
           "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY")}
DATA = "https://data.alpaca.markets/v2"

MONTHS = 6
MIN_PRICE, MAX_PRICE = 2.0, 50.0
MIN_GAP_PCT = 3.0
WATCH_TOP = 5
POS_PCT = 0.10
MAX_POS = 3
RR = 2.0
MIN_SL, MAX_SL = 0.01, 0.04
SLIP = 0.001
_c = os.getenv("BT_CUTOFF", "15:30").split(":")
CUTOFF = (int(_c[0]), int(_c[1]))  # ET — BT_CUTOFF=12:00 이면 오전 모멘텀만 (룩어헤드 민감도 테스트)
EQUITY0 = 100_000.0


def get_bars(symbols, timeframe, start, end, limit=10000):
    """multi-symbol bars with pagination -> {sym: df}"""
    out = {s: [] for s in symbols}
    page_token = None
    for _ in range(60):
        params = {
            "symbols": ",".join(symbols), "timeframe": timeframe,
            "start": start.isoformat(), "end": end.isoformat(),
            "limit": limit, "feed": "iex", "adjustment": "split",
        }
        if page_token:
            params["page_token"] = page_token
        r = requests.get(f"{DATA}/stocks/bars", headers=HEADERS, params=params, timeout=30)
        if r.status_code != 200:
            print(f"  bars error {r.status_code}: {r.text[:120]}")
            break
        d = r.json()
        for sym, bars in d.get("bars", {}).items():
            out.setdefault(sym, []).extend(bars)
        page_token = d.get("next_page_token")
        if not page_token:
            break
        time.sleep(0.15)
    res = {}
    for sym, bars in out.items():
        if bars:
            df = pd.DataFrame(bars).rename(columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
            df["time"] = pd.to_datetime(df["time"])
            res[sym] = df.sort_values("time").reset_index(drop=True)
    return res


def main():
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=int(MONTHS * 30.5))

    # 1) 전 종목 일봉 (캐시)
    daily_cache = os.path.join(CACHE, f"alpaca_daily_{start.date()}_{end.date()}.pkl")
    if os.path.exists(daily_cache):
        daily = pickle.load(open(daily_cache, "rb"))
    else:
        print("자산 목록 조회...", flush=True)
        r = requests.get("https://paper-api.alpaca.markets/v2/assets",
                         headers=HEADERS, params={"status": "active", "asset_class": "us_equity"}, timeout=30)
        assets = [a["symbol"] for a in r.json()
                  if a.get("tradable") and a.get("exchange") in ("NASDAQ", "NYSE", "AMEX")
                  and "etf" not in (a.get("attributes") or [])]
        print(f"  대상 {len(assets)}종목 일봉 수집 중...", flush=True)
        daily = {}
        for i in range(0, len(assets), 150):
            chunk = assets[i:i + 150]
            for sym, df in get_bars(chunk, "1Day", start, end).items():
                if len(df) >= 30:
                    daily[sym] = df
            if i % 1500 == 0:
                print(f"  {i}/{len(assets)}...", flush=True)
            time.sleep(0.2)
        pickle.dump(daily, open(daily_cache, "wb"))
    print(f"일봉 확보: {len(daily)}종목", flush=True)

    # 2) 유동성/가격 필터
    universe = {}
    for sym, df in daily.items():
        last = df.iloc[-1]
        dv = (df["close"] * df["volume"]).tail(20).median()
        if MIN_PRICE <= last["close"] <= MAX_PRICE and dv >= 1_000_000:
            universe[sym] = df
    print(f"유니버스: {len(universe)}종목 (가격 ${MIN_PRICE}~{MAX_PRICE}, 일중앙값 거래대금 ≥$1M)", flush=True)

    # 3) 일별 스캐너 리플레이
    calendar = sorted({t.normalize() for df in universe.values() for t in df["time"]})
    calendar = [d for d in calendar if d >= pd.Timestamp(start.date(), tz="UTC")]
    prev_close = {s: df["close"].shift(1) for s, df in universe.items()}

    trades = []
    equity = EQUITY0
    curve = []

    for day in calendar:
        # 당일 %change 상위 5종목
        movers = []
        for sym, df in universe.items():
            row = df[df["time"].dt.normalize() == day]
            if row.empty:
                continue
            i = row.index[0]
            pc = prev_close[sym].loc[i]
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

        # 1H 봉 (전 2일 포함) — 일 단위 캐시
        hcache = os.path.join(CACHE, f"alpaca_1h_{day.date()}.pkl")
        if os.path.exists(hcache):
            hourly = pickle.load(open(hcache, "rb"))
        else:
            hourly = get_bars(watch, "1Hour", (day - timedelta(days=4)).to_pydatetime(),
                              (day + timedelta(days=1)).to_pydatetime())
            pickle.dump(hourly, open(hcache, "wb"))
            time.sleep(0.15)

        day_positions = 0
        for sym in watch:
            if day_positions >= MAX_POS:
                break
            df = hourly.get(sym)
            if df is None or len(df) < 16:
                continue
            df = df.copy()
            tp = (df["high"] + df["low"] + df["close"]) / 3
            df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, 1)
            df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
            # 당일 봉만 순회 (미국시간 기준)
            dmask = df["time"].dt.normalize() == day
            day_idx = df.index[dmask].tolist()
            for j in day_idx:
                if j < 15:
                    continue
                bar_time = df.loc[j, "time"]
                et_hour = (bar_time - pd.Timedelta(hours=4)).hour  # EDT 근사 (백테스트 기간 여름)
                et_min = (bar_time - pd.Timedelta(hours=4)).minute
                if (et_hour, et_min) >= CUTOFF:
                    break
                # 셋업: j-1 = 풀백봉, j = 트리거봉
                prev = df.loc[j - 1]
                curr = df.loc[j]
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
                # 진입!
                entry = trig * (1 + SLIP)
                tp_price = trig + risk * RR
                value = equity * POS_PCT
                exit_p, reason = None, None
                # 이후 봉으로 청산 시뮬 (같은 날만)
                later = [k for k in day_idx if k > j]
                for k in later:
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
                break  # 종목당 1회 (라이브: watchlist 순회 후 break는 아니지만 slot 제한 근사)
        curve.append((day, equity))

    # 4) 리포트
    eq = pd.Series({d: v for d, v in curve}).sort_index()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.01)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    mdd = ((eq.cummax() - eq) / eq.cummax()).max()
    print("\n" + "=" * 74)
    print(f"📊 Alpaca Cameron 1H Pullback 백테스트 — 최근 {MONTHS}개월 (IEX, 페이퍼 수수료 0)")
    print("=" * 74)
    if trades:
        arr = np.array([t["pnl"] for t in trades])
        wins = arr[arr > 0]
        gw, gl = wins.sum(), -arr[arr <= 0].sum()
        pf = gw / gl if gl > 0 else float("inf")
        print(f"거래수: {len(arr)} | 승률: {(arr > 0).mean() * 100:.1f}% | PF: {pf:.2f}")
        print(f"최종 자산: ${eq.iloc[-1]:,.0f} (시작 $100,000) | CAGR: {cagr * 100:+.1f}% | MDD: {mdd * 100:.1f}%")
        print(f"평균 손익: ${arr.mean():+.0f} | 총손익: ${arr.sum():+,.0f}")
        reasons = {}
        for t in trades:
            reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
        print(f"청산 사유: {reasons}")
        tdf = pd.DataFrame(trades)
        tdf["month"] = tdf["day"].str[:7]
        print("\n월별 손익:")
        for m, g in tdf.groupby("month"):
            print(f"  {m}: {len(g):>3}건, ${g['pnl'].sum():+9,.0f}, WR {(g['pnl'] > 0).mean() * 100:.0f}%")
        print("\n최근 15건:")
        for t in trades[-15:]:
            print(f"  {t['day']} {t['sym']:<6} {t['reason']:<9} {t['ret']:+.2f}% ${t['pnl']:+.0f}")
    else:
        print("거래 없음")
    pickle.dump(trades, open(os.path.join(CACHE, "alpaca_cameron_trades.pkl"), "wb"))


if __name__ == "__main__":
    main()
