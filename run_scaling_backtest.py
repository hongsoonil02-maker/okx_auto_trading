#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_scaling_backtest.py — 자산 규모별 수익률 변화 검증
19K → 1.9M → 19M → 190M → 1.9B 스케일에서 슬리피지·유동성 한계 반영

슬리피지 모델: 주문규모 / 평균 15분 거래량 비율에 비례하여 체결가 불리하게 이동.
수용 한도: 심볼별 (24h 거래량 × 허용비율) → 초과 분은 포기.
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
CHOP_START = datetime(2026, 8, 21, 14, 0, tzinfo=timezone.utc).timestamp() * 1000

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "DOGE/USDT:USDT", "ADA/USDT:USDT", "XRP/USDT:USDT", "AVAX/USDT:USDT",
    "ZRO/USDT:USDT", "CAP/USDT:USDT", "VVV/USDT:USDT", "LTC/USDT:USDT",
    "AXTI/USDT:USDT", "CHIP/USDT:USDT", "CRCL/USDT:USDT", "AAVE/USDT:USDT",
    "LIT/USDT:USDT", "FIL/USDT:USDT", "HYPE/USDT:USDT", "BOME/USDT:USDT",
]

# 슬리피지: (order_size / avg_volume_15m) 비율 → 0.01(1%)당 0.05% 슬리피지
SLIPPAGE_PER_RATIO = 0.0005  # 거래량 대비 1% 주문 시 0.05% 슬리피지
# 심볼별 수용 한도: 24h 거대비 최대 3%까지
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


def simulate_scaled(data, initial_equity, slippage_enabled, volume_cap_enabled):
    """
    slippage_enabled: 주문규모/거래량 비례 슬리피지 반영
    volume_cap_enabled: 심볼별 24h 거래량 대비 VOLUME_LIMIT_RATIO 이상 포지션 불가
    """
    cash = initial_equity
    positions = {}
    trades = []
    fees = 0.0
    slippage_cost = 0.0

    # 심볼별 평균 15분 거래량 (관용 한도 계산용)
    avg_vol_15m = {}
    for sym, df in data.items():
        avg_vol_15m[sym] = df['v'].mean() if len(df) > 0 else 1.0

    all_ts = sorted(set().union(*[set(df['t']) for df in data.values()]))
    warmup = 60

    for idx, t in enumerate(all_ts):
        if idx < warmup:
            continue

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
            # 청산
            remaining = []
            for p in plist:
                p['last_px'] = px
                pnl_pct = (px - p['entry']) / p['entry'] * LEVERAGE * p['dir']
                p['extreme'] = max(p['extreme'], pnl_pct) if p['dir'] == 1 else min(p['extreme'], pnl_pct)
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
                    fees += fee
                    trades.append({'pnl': gross - fee})
                else:
                    remaining.append(p)
            positions[sym] = remaining

            if remaining:
                continue

            # 진입
            long_sig = curr['st_dir'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
            if not long_sig:
                continue

            equity = sum(p['margin'] * (1 + (p['last_px'] - p['entry']) / p['entry'] * LEVERAGE * p['dir']) for s, pl in positions.items() for p in pl) + cash
            margin = (equity / MAX_POS) * WEIGHT * NEW_RATIO

            if margin < 100 or margin > cash:
                continue

            # 수용 한도 체크 (volume_cap)
            if volume_cap_enabled:
                max_notional = avg_vol_15m.get(sym, 1.0) * 96 * VOLUME_LIMIT_RATIO * px  # 96 = 하루 15분 캔들 수
                if margin * LEVERAGE > max_notional:
                    margin = max_notional / LEVERAGE
                    if margin < 100:
                        continue

            # 슬리피지 계산
            entry_px = px
            if slippage_enabled:
                order_ratio = (margin * LEVERAGE) / (avg_vol_15m.get(sym, 1.0) * px) if avg_vol_15m.get(sym, 1.0) > 0 else 0
                slip = min(order_ratio * SLIPPAGE_PER_RATIO, 0.01)  # 최대 1% 슬리피지
                slippage_cost += margin * LEVERAGE * slip
                entry_px = px * (1 + slip)  # 슬리피지 반영 진입가

            fee = margin * LEVERAGE * FEE_RATE
            cash -= margin + fee
            fees += fee
            positions[sym] = [{'entry': entry_px, 'margin': margin, 'dir': 1, 'extreme': 0.0, 'last_px': entry_px}]

    equity = cash
    for sym, plist in positions.items():
        for p in plist:
            pnl_pct = (p['last_px'] - p['entry']) / p['entry'] * LEVERAGE * p['dir']
            fee = p['margin'] * LEVERAGE * FEE_RATE
            equity += p['margin'] * (1 + pnl_pct) - fee
            fees += fee
            trades.append({'pnl': p['margin'] * pnl_pct - fee})

    pnl = equity - initial_equity
    roi_pct = pnl / initial_equity * 100
    return {
        'pnl': pnl, 'roi_pct': roi_pct, 'fees': fees, 'slippage': slippage_cost,
        'equity': equity, 'trades': len(trades),
    }


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
        print(f"{len(data)}개 심볼 수집\n")

        scales = [
            ("19K  (현재)", 19000),
            ("190K (10x)", 190000),
            ("1.9M (100x)", 1900000),
            ("19M  (1,000x)", 19000000),
            ("190M (10,000x)", 190000000),
        ]

        configs = [
            ("슬리피지·한도 없음 (이상)", False, False),
            ("슬리피지만 반영", True, False),
            ("슬리피지+수용한도 반영 (현실)", True, True),
        ]
        for cname, slip, vcap in configs:
            print(f"=== {cname} ===")
            print(f"{'규모':20s} {'손익':>14s} {'ROI%':>8s} {'수수료':>10s} {'슬리피지':>10s} {'거래':>4s}")
            for sname, eq in scales:
                r = simulate_scaled(data, eq, slip, vcap)
                print(f"  {sname:18s} {r['pnl']:>+14.0f} {r['roi_pct']:>7.1f}% {r['fees']:>10.0f} {r['slippage']:>10.0f} {r['trades']:4d}")
            print()
    finally:
        await ex.close()


if __name__ == "__main__":
    asyncio.run(main())
