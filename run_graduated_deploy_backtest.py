#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_graduated_deploy_backtest.py — 단계적 배포(Graduated Deployment) 검증
현재: ADX<25 전면 차단 (전부 아니면 전무)
제안: 약추세 구간에서 축소 사이즈로 진입 허용

변형:
  A. 필터 없음 (참고)
  B. 현재 방식: ADX<25 전면 차단
  C~E. 단계적: ADX [경계~25] 구간 축소사이즈 진입, 그 미만 차단
"""
import asyncio
from datetime import datetime, timezone
import pandas as pd
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


def calc_adx(df, period=14):
    h, l, c = df['h'], df['l'], df['c']
    up, dn = h.diff(), -l.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).where((pdi + mdi) != 0)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)


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


def simulate(data, adx_map, soft_boundary, soft_size):
    """
    soft_boundary: 소프트 존 하한 ADX (None = 단계적 배포 없음 → ADX<25 전면 차단)
                   "off" = 필터 완전 해제
    soft_size:     소프트 존 진입 시 사이즈 배율 (예: 0.5 = 절반)
    """
    cash = INITIAL_EQUITY
    positions = {}
    trades = []
    fees = 0.0

    all_ts = sorted(set().union(*[set(df['t']) for df in data.values()]))
    warmup = 60

    for idx, t in enumerate(all_ts):
        if idx < warmup:
            continue
        last_h = ((t // 3600000) * 3600000) - 3600000
        adx_now = adx_map.get(last_h, 30.0)

        # 배포 결정
        if soft_boundary == "off":
            deploy_scale = 1.0 if adx_now >= 25 or adx_now < 25 else 1.0  # 필터 없음
        elif soft_boundary is None:
            # 현재 방식: ADX<25 전면 차단
            if adx_now < 25:
                deploy_scale = 0.0
            else:
                deploy_scale = 1.0
        else:
            # 단계적 배포
            if adx_now >= 25:
                deploy_scale = 1.0      # 추세: 풀사이즈
            elif adx_now >= soft_boundary:
                deploy_scale = soft_size  # 약추세: 축소
            else:
                deploy_scale = 0.0        # 횡보: 차단

        for sym, df in data.items():
            rows = df.index[df['t'] == t]
            if len(rows) == 0:
                continue
            i = rows[0]
            if i < 1:
                continue
            curr, prev = df.iloc[i], df.iloc[i - 1]
            px = float(curr['c'])

            pos = positions.get(sym)
            if pos:
                pos['last_px'] = px
                pnl_pct = (px - pos['entry']) / pos['entry'] * LEVERAGE * pos['dir']
                pos['extreme'] = max(pos['extreme'], pnl_pct) if pos['dir'] == 1 else min(pos['extreme'], pnl_pct)
                best = pos['extreme']
                exit_now = pnl_pct <= -0.30 or pnl_pct <= -0.15  # 하드스탑 + 포지션손실한도
                if pos['dir'] == 1:
                    exit_now = exit_now or (best > 0.4 and pnl_pct < best * 0.5) or (best > 0.2 and pnl_pct < 0.05)
                else:
                    exit_now = exit_now or (best < -0.4 and pnl_pct > best * 0.5) or (best < -0.2 and pnl_pct > -0.05)
                exit_now = exit_now or pnl_pct >= 0.50
                if exit_now:
                    gross = pos['margin'] * pnl_pct
                    fee = pos['margin'] * LEVERAGE * FEE_RATE
                    cash += pos['margin'] + gross - fee
                    fees += fee
                    trades.append({'t': t, 'pnl': gross - fee})
                    del positions[sym]
                continue

            if deploy_scale <= 0:
                continue
            long_sig = curr['st_dir'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
            short_sig = curr['st_dir'] == -1 and prev['stoch_k'] > 80 and curr['stoch_k'] <= 80
            if not (long_sig or short_sig):
                continue
            equity = sum(p['margin'] * (1 + (p['last_px'] - p['entry']) / p['entry'] * LEVERAGE * p['dir'])
                         for p in positions.values()) + cash
            margin = (equity / MAX_POS) * WEIGHT * NEW_RATIO * deploy_scale
            if margin < 100 or margin > cash:
                continue
            d = 1 if long_sig else -1
            fee = margin * LEVERAGE * FEE_RATE
            cash -= margin + fee
            fees += fee
            positions[sym] = {'entry': px, 'margin': margin, 'dir': d, 'extreme': 0.0, 'last_px': px}

    equity = cash
    for sym, pos in positions.items():
        pnl_pct = (pos['last_px'] - pos['entry']) / pos['entry'] * LEVERAGE * pos['dir']
        fee = pos['margin'] * LEVERAGE * FEE_RATE
        equity += pos['margin'] * (1 + pnl_pct) - fee
        fees += fee
        trades.append({'pnl': pos['margin'] * pnl_pct - fee})

    pnl_total = equity - INITIAL_EQUITY
    pnl_chop = sum(tr['pnl'] for tr in trades if 't' in tr and tr['t'] >= CHOP_START)
    wins = sum(1 for tr in trades if tr['pnl'] > 0)
    return {
        'total': pnl_total, 'chop': pnl_chop, 'n': len(trades), 'fees': fees,
        'equity': equity, 'winrate': wins / len(trades) * 100 if trades else 0,
    }


async def main():
    ex = ccxt_async.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    since = int(datetime(2026, 8, 18, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    try:
        print("데이터 수집 중...")
        btc = await fetch_all(ex, "BTC/USDT:USDT", "1h", since - 260 * 3600 * 1000)
        btc['adx'] = calc_adx(btc, 14)
        adx_map = {int(r['t']): float(r['adx']) for _, r in btc.iloc[:-1].iterrows()}

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

        variants = [
            ("A. 필터 없음 (참고)",           "off", 1.0),
            ("B. 현재: ADX<25 전면차단",       None,  1.0),
            ("C. 단계적: 20~25→50%",          20,    0.5),
            ("D. 단계적: 20~25→30%",          20,    0.3),
            ("E. 단계적: 22~25→70%",          22,    0.7),
            ("F. 단계적: 15~25→50%",          15,    0.5),
            ("G. 단계적: 18~25→40%",          18,    0.4),
        ]
        print(f"{'변형':28s} {'전체손익':>10s} {'피크이후':>10s} {'거래':>4s} {'수수료':>8s} {'승률':>5s} {'최종자산':>10s}")
        for name, bnd, size in variants:
            r = simulate(data, adx_map, bnd, size)
            print(f"  {name:26s} {r['total']:+10.0f} {r['chop']:+10.0f} {r['n']:4d} {r['fees']:8.0f} {r['winrate']:4.0f}% {r['equity']:10.0f}")
    finally:
        await ex.close()


if __name__ == "__main__":
    asyncio.run(main())
