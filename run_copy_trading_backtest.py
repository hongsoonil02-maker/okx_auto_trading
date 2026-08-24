#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_copy_trading_backtest.py — 카피트레이딩 팔로워 증가에 따른 수익률 변화
리드 자본 19K 고정, 팔로워 자축 0x~1000x 변화.
리드 체결가 악화 = 리드 주문 슬리피지 + 팔로워 주문의 시장 충격 추가.
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
CHOP_START = datetime(2026, 8, 21, 14, 0, tzinfo=timezone.utc).timestamp() * 1000

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


def simulate_copy(data, copy_multiplier):
    """
    copy_multiplier: 팔로우 자축 / 리드 자축 비율 (0=팔로워없음, 10=10배, 100=100배)
    리드 체결 슬리피지 = 리드 주문 + 팔로워 주문이 같은 호가창을 친 누적 충격.
    """
    cash = INITIAL_EQUITY
    positions = {}
    avg_vol_15m = {sym: df['v'].mean() for sym, df in data.items()}

    all_ts = sorted(set().union(*[set(df['t']) for df in data.values()]))
    warmup = 60
    total_slippage = 0.0

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

            # 수용 한도 (리드 자축 기준)
            max_notional = avg_vol_15m.get(sym, 1.0) * 96 * VOLUME_LIMIT_RATIO * px
            if margin * LEVERAGE > max_notional:
                margin = max_notional / LEVERAGE
                if margin < 100:
                    continue

            # 슬리피지 = 리드 주문 + 팔로워 주문이 같은 방향으로 부림
            # 팔로워 주문은 리드 주문의 copy_multiplier 배 → 시장 충격 누적
            lead_ratio = (margin * LEVERAGE) / (avg_vol_15m.get(sym, 1.0) * px) if avg_vol_15m.get(sym, 1.0) > 0 else 0
            total_ratio = lead_ratio * (1 + copy_multiplier)  # 리드 + 팔로워 누적
            slip = min(total_ratio * SLIPPAGE_PER_RATIO, 0.02)  # 최대 2% 슬리피지
            total_slippage += margin * LEVERAGE * slip

            entry_px = px * (1 + slip)  # 슬리피지만큼 불리한 가격에 진입
            fee = margin * LEVERAGE * FEE_RATE
            cash -= margin + fee
            positions[sym] = [{'entry': entry_px, 'margin': margin, 'dir': 1, 'extreme': 0.0, 'last_px': entry_px}]

    equity = cash
    for sym, plist in positions.items():
        for p in plist:
            pnl_pct = (p['last_px'] - p['entry']) / p['entry'] * LEVERAGE * p['dir']
            equity += p['margin'] * (1 + pnl_pct) - p['margin'] * LEVERAGE * FEE_RATE

    roi_pct = (equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100
    return {'equity': equity, 'roi_pct': roi_pct, 'slippage': total_slippage}


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

        scenarios = [
            ("팔로워 0명 (리드만)", 0),
            ("팔로워 소수 (1x)", 1),
            ("팔로워 10명 (10x)", 10),
            ("팔로워 50명 (50x)", 50),
            ("팔로워 100명 (100x)", 100),
            ("팔로워 500명 (500x)", 500),
            ("팔로워 1000명 (1000x)", 1000),
        ]

        print(f"{'시나리오':24s} {'리드 자산':>12s} {'카피 총자축':>12s} {'ROI%':>8s} {'슬리피지':>10s}")
        for name, mult in scenarios:
            r = simulate_copy(data, mult)
            copy_aum = INITIAL_EQUITY * mult
            print(f"  {name:22s} 19,000{'':>5s} {copy_aum:>12,} {r['roi_pct']:>+7.1f}% {r['slippage']:>10.0f}")
    finally:
        await ex.close()


if __name__ == "__main__":
    asyncio.run(main())
