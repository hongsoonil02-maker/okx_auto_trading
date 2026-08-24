#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_time_to_scale.py — 19K -> 1.9M(100x) 도달 기간 시뮬레이션
백테스트 구간의 일별 자산곡선을 복리 적용하여 전진 예측.
자산 규모별 슬리피지 페널티를 스케일링 백테스트 결과에서 보간 적용.
"""
import asyncio
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt_async

LEVERAGE = 10
FEE_RATE = 0.0005
MAX_POS = 15
WEIGHT = 1.5
NEW_RATIO = 0.50
INITIAL_EQUITY = 19000.0

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "DOGE/USDT:USDT", "ADA/USDT:USDT", "XRP/USDT:USDT", "AVAX/USDT:USDT",
    "ZRO/USDT:USDT", "CAP/USDT:USDT", "VVV/USDT:USDT", "LTC/USDT:USDT",
    "AXTI/USDT:USDT", "CHIP/USDT:USDT", "CRCL/USDT:USDT", "AAVE/USDT:USDT",
    "LIT/USDT:USDT", "FIL/USDT:USDT", "HYPE/USDT:USDT", "BOME/USDT:USDT",
]

SLIPPAGE_PER_RATIO = 0.0005
VOLUME_LIMIT_RATIO = 0.03


def calc_supertrend(df, period=10, multiplier=3.0):
    hl2 = (df['h'] + df['l']) / 2
    atr = (df['h'].combine(df['c'].shift(), max) - df['l'].combine(df['c'].shift(), min)).rolling(period).mean()
    fu = hl2 + multiplier * atr
    fl = hl2 - multiplier * atr
    sd = pd.Series(1, index=df.index, dtype='int')
    for i in range(period, len(df)):
        if df['c'].iloc[i] > fu.iloc[i - 1]:
            sd.iloc[i] = 1
        elif df['c'].iloc[i] < fl.iloc[i - 1]:
            sd.iloc[i] = -1
        else:
            sd.iloc[i] = sd.iloc[i - 1]
    return sd


def calc_stoch_k(series, period=14, smooth=3):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    stoch = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    return stoch.rolling(smooth).mean() * 100


async def fetch_all(ex, sym, tf, since_ms):
    out = []
    since = since_ms
    for _ in range(20):
        batch = await ex.fetch_ohlcv(sym, tf, since=since, limit=300)
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 300:
            break
        since = batch[-1][0] + 1
        await asyncio.sleep(0.05)
    df = pd.DataFrame(out, columns=['t', 'o', 'h', 'l', 'c', 'v']).drop_duplicates('t')
    return df.sort_values('t').reset_index(drop=True)


def simulate_daily(data, initial_equity):
    """백테스트 구간에서 일별 자산 곡선 반환"""
    cash = initial_equity
    positions = {}
    avg_vol_15m = {sym: df['v'].mean() for sym, df in data.items()}

    all_ts = sorted(set().union(*[set(df['t']) for df in data.values()]))
    warmup = 60
    daily_equity = {}  # day_start_timestamp -> equity

    for idx, t in enumerate(all_ts):
        if idx < warmup:
            continue
        # 하루 시작(UTC 00:00) 기록
        dt = datetime.fromtimestamp(t / 1000, tz=timezone.utc)
        if dt.hour == 0 and dt.minute == 0:
            eq = cash
            for sym, plist in positions.items():
                for p in plist:
                    pnl_pct = (p['last_px'] - p['entry']) / p['entry'] * LEVERAGE * p['dir']
                    eq += p['margin'] * (1 + pnl_pct)
            daily_equity[t] = eq

        for sym, df in data.items():
            rows = df.index[df['t'] == t]
            if len(rows) == 0:
                continue
            i = rows[0]
            if i < 1:
                continue
            curr, prev = df.iloc[i], df.iloc[i - 1]
            px = float(curr['c'])
            plist = positions.get(sym, [])
            remaining = []
            for p in plist:
                p['last_px'] = px
                pnl_pct = (px - p['entry']) / p['entry'] * LEVERAGE * p['dir']
                p['extreme'] = max(p['extreme'], pnl_pct) if p['dir'] == 1 else min(pos['extreme'], pnl_pct)
                best = p['extreme']
                exit_now = pnl_pct <= -0.30 or pnl_pct <= -0.15
                if p['dir'] == 1:
                    exit_now = exit_now or (best > 0.4 and pnl_pct < best * 0.5) or (best > 0.2 and pnl_pct < 0.05)
                else:
                    exit_now = exit_now or (best < -0.4 and pnl_pct > best * 0.5) or (best < -0.2 and pnl_pct > -0.05)
                exit_now = exit_now or pnl_pct >= 0.50
                if exit_now:
                    gross = p['margin'] * pnl_pct
                    fee = p['margin'] * LEVERAGE * FEE_RATE
                    cash += p['margin'] + gross - fee
                else:
                    remaining.append(p)
            positions[sym] = remaining
            if remaining:
                continue
            long_sig = curr['st_dir'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
            if not long_sig:
                continue
            equity = sum(p['margin'] * (1 + (p['last_px'] - p['entry']) / p['entry'] * LEVERAGE * p['dir']) for s, pl in positions.items() for p in pl) + cash
            margin = (equity / MAX_POS) * WEIGHT * NEW_RATIO
            if margin < 100 or margin > cash:
                continue
            max_notional = avg_vol_15m.get(sym, 1.0) * 96 * VOLUME_LIMIT_RATIO * px
            if margin * LEVERAGE > max_notional:
                margin = max_notional / LEVERAGE
                if margin < 100:
                    continue
            order_ratio = (margin * LEVERAGE) / (avg_vol_15m.get(sym, 1.0) * px) if avg_vol_15m.get(sym, 1.0) > 0 else 0
            slip = min(order_ratio * SLIPPAGE_PER_RATIO, 0.01)
            entry_px = px * (1 + slip)
            fee = margin * LEVERAGE * FEE_RATE
            cash -= margin + fee
            positions[sym] = [{'entry': entry_px, 'margin': margin, 'dir': 1, 'extreme': 0.0, 'last_px': entry_px}]
    return daily_equity


def project_forward(daily_equity, target_multipliers):
    """
    일별 자산곡선의 일간 수익률을 복리 적용하여 전진.
    자산 규모에 따른 ROI 체감 적용.
    """
    days = sorted(daily_equity.keys())
    if len(days) < 2:
        return []

    # 일별 수익률 시계열
    rets = []
    for i in range(1, len(days)):
        prev_eq = daily_equity[days[i - 1]]
        curr_eq = daily_equity[days[i]]
        daily_ret = (curr_eq - prev_eq) / prev_eq if prev_eq > 0 else 0
        rets.append(daily_ret)

    if not rets:
        return []

    avg_daily_ret = np.mean(rets)
    std_daily_ret = np.std(rets)
    median_ret = np.median(rets)

    # 스케일 체감 계수 (run_scaling_backtest 결과에서 도출)
    # 19K: 1.00, 190K: 1.03, 1.9M: 0.81, 19M: 0.34
    def scale_penalty(eq):
        if eq < 190000:
            return 1.0
        elif eq < 1900000:
            return 1.0 + (eq / 1900000) * 0.03
        elif eq < 19000000:
            return 0.81 + (1 - eq / 19000000) * 0.22
        else:
            return 0.34 + min(0.5, (eq - 19000000) / 200000000)

    results = {}
    for mult in target_multipliers:
        target = INITIAL_EQUITY * mult
        eq = INITIAL_EQUITY
        day_count = 0
        max_days = 365 * 3  # 3년 상한
        # 최근 수일의 평균으로 추정
        recent_rets = rets[-5:] if len(rets) >= 5 else rets
        avg_recent = np.mean(recent_rets)
        while eq < target and day_count < max_days:
            ret = avg_recent * scale_penalty(eq)
            eq *= (1 + ret)
            day_count += 1
            if eq <= 0:
                break
        results[mult] = (day_count, eq)
    return results, avg_daily_ret, std_daily_ret, median_ret, len(rets)


async def main():
    ex = ccxt_async.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    since = int(datetime(2026, 8, 18, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    try:
        print("데이터 수집 중...")
        data = {}
        for sym in SYMBOLS:
            try:
                df = await fetch_all(ex, sym, "15m", since)
                if len(df) >= 200:
                    df['st_dir'] = calc_supertrend(df)
                    df['stoch_k'] = calc_stoch_k(df['c'])
                    df = df.dropna().reset_index(drop=True)
                    data[sym] = df
            except Exception:
                pass
        print(f"{len(data)}개 심볼 수집")

        daily_eq = simulate_daily(data, INITIAL_EQUITY)
        print(f"\n백테스트 구간 일별 자산 곡선 ({len(daily_eq)}일):")
        for t, eq in sorted(daily_eq.items()):
            dt = datetime.fromtimestamp(t / 1000, tz=timezone.utc)
            print(f"  {dt.strftime('%m-%d')}: {eq:>10.0f} USDT")

        targets = [10, 100, 1000, 10000]
        results, avg_ret, std_ret, med_ret, n_days = project_forward(daily_eq, targets)
        print(f"\n일별 수익률: 평균 {avg_ret*100:+.2f}%, 중간값 {med_ret*100:+.2f}%, 표준편차 {std_ret*100:.2f}%")
        print(f"\n{'목표 규모':18s} {'도달일':>8s} {'경과년월':>10s}")
        for mult, (days, final_eq) in results.items():
            label = f"19K x{mult} = {INITIAL_EQUITY*mult/1000:.0f}K"
            yrs = days / 365
            mos = (days % 365) / 30
            print(f"  {label:18s} {days:>8d}일   {yrs:.0f}년 {mos:.0f}개월  (최종 {final_eq/1e6:.1f}M)")
    finally:
        await ex.close()


if __name__ == "__main__":
    asyncio.run(main())
