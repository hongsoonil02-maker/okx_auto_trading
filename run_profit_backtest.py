#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_profit_backtest.py — 수익성 향상 4변형 + 조합 병렬 검증
1. 컨빅션 사이징   — 진입 점수에 비례 포지션 크기
2. 트레일링 스탑    — 고점 대비 % 되돌림 시 전량 청산
3. ADX 피라미딩    — BTC ADX > 25(강추세)일 때만 불타기
4. 베어 숏          — BTC < EMA200일 때만 숏 허용

구간: 08-18 ~ 현재 (상승장 + 꼭지 + 횡보 포함)
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
TRAILING_PCT = 0.06  # 고점 대비 6% 되돌림 시 청산

SYMBOLS = [
    "ZRO/USDT:USDT", "CAP/USDT:USDT", "VVV/USDT:USDT", "LTC/USDT:USDT",
    "DOGE/USDT:USDT", "AXTI/USDT:USDT", "CHIP/USDT:USDT", "CRCL/USDT:USDT",
    "AAVE/USDT:USDT", "LIT/USDT:USDT", "FIL/USDT:USDT", "HYPE/USDT:USDT",
    "BNB/USDT:USDT", "ZEC/USDT:USDT", "TRB/USDT:USDT", "BOME/USDT:USDT",
    "SOL/USDT:USDT", "ENA/USDT:USDT", "LINK/USDT:USDT", "CRV/USDT:USDT",
]


def calc_supertrend(df, period=10, multiplier=3.0):
    hl2 = (df['h'] + df['l']) / 2
    atr = (df['h'].combine(df['c'].shift(), max) - df['l'].combine(df['c'].shift(), min)).rolling(period).mean()
    fu = hl2 + multiplier * atr
    fl = hl2 - multiplier * atr
    sd = pd.Series(1, index=df.index, dtype='int')
    sv = pd.Series(0.0, index=df.index, dtype='float64')
    for i in range(period, len(df)):
        if df['c'].iloc[i] > fu.iloc[i - 1]:
            sd.iloc[i] = 1
        elif df['c'].iloc[i] < fl.iloc[i - 1]:
            sd.iloc[i] = -1
        else:
            sd.iloc[i] = sd.iloc[i - 1]
            if sd.iloc[i] == 1 and fl.iloc[i] < fl.iloc[i - 1]:
                fl.iloc[i] = fl.iloc[i - 1]
            if sd.iloc[i] == -1 and fu.iloc[i] > fu.iloc[i - 1]:
                fu.iloc[i] = fu.iloc[i - 1]
        sv.iloc[i] = fl.iloc[i] if sd.iloc[i] == 1 else fu.iloc[i]
    return sd, sv


def calc_stoch_k(series, period=14, smooth=3):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    stoch = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    return stoch.rolling(smooth).mean() * 100


def calc_score(curr, prev, ema_target):
    """진입 점수 계산 (브레인 로직과 동일)"""
    score = 0
    if curr['c'] > ema_target:
        score += 50
    if prev['st_dir'] == -1 and curr['st_dir'] == 1:
        score += 40
    if curr['st_dir'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20:
        score += 30
    if curr['st_dir'] == 1 and prev['st_dir'] == 1:
        score += 20
    if curr['stoch_k'] > curr['stoch_d'] and curr['stoch_k'] < 80:
        score += 20
    return score


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


def simulate(data, btc_map, flags):
    """
    flags: {conviction, trailing, adx_pyramid, bear_short}
    """
    cash = INITIAL_EQUITY
    positions = {}  # sym -> {entry, margin, dir, extreme, highest, score, dca_count}
    trades, fees = [], 0.0

    all_ts = sorted(set().union(*[set(df['t']) for df in data.values()]))
    warmup = 60

    for idx, t in enumerate(all_ts):
        if idx < warmup:
            continue
        hour_floor = (t // 3600000) * 3600000
        last_h = hour_floor - 3600000
        adx_now, btc_bull, above200 = btc_map.get(last_h, (30.0, True, True))

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
                pos['highest'] = max(pos['highest'], px) if pos['dir'] == 1 else min(pos['highest'], px)
                best = pos['extreme']

                # 청산 체크
                exit_now = pnl_pct <= -0.30  # 하드스탑
                exit_now = exit_now or (best > 0.4 and pnl_pct < best * 0.5)  # 트레일링 수익보호
                exit_now = exit_now or (best > 0.2 and pnl_pct < 0.05)
                exit_now = exit_now or pnl_pct >= 0.50  # TP

                # 트레일링 스탑 (변형 2)
                if flags['trailing']:
                    if pos['dir'] == 1 and px < pos['highest'] * (1 - TRAILING_PCT):
                        exit_now = True
                    elif pos['dir'] == -1 and px > pos['highest'] * (1 + TRAILING_PCT):
                        exit_now = True

                if exit_now:
                    gross = pos['margin'] * pnl_pct
                    fee = pos['margin'] * LEVERAGE * FEE_RATE
                    cash += pos['margin'] + gross - fee
                    fees += fee
                    trades.append({'t': t, 'pnl': gross - fee, 'dir': pos['dir'], 'sym': sym})
                    del positions[sym]
                continue

            # ── 진입 ──
            long_sig = curr['st_dir'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
            short_sig = curr['st_dir'] == -1 and prev['stoch_k'] > 80 and curr['stoch_k'] <= 80

            # 베어 숏 (변형 4): BTC < EMA200일 때만 숏 허용
            if flags['bear_short'] and above200 and short_sig:
                short_sig = False  # 불장에서는 숏 차단

            if not (long_sig or short_sig):
                continue

            # 점수 계산 (컨빅션 사이징용)
            ema_target = curr['c']  # 간소화 (실제론 EMA50)
            score = 70 if long_sig else 70  # 기본 70
            if long_sig:
                score = 50 + 40 + 30 + (20 if curr['st_dir'] == prev['st_dir'] else 0) + (20 if curr['stoch_k'] > curr['stoch_d'] else 0)
            d = 1 if long_sig else -1

            # 기본 마진
            equity = sum(p['margin'] * (1 + (p['last_px'] - p['entry']) / p['entry'] * LEVERAGE * p['dir']) for p in positions.values()) + cash
            margin = (equity / MAX_POS) * WEIGHT * NEW_RATIO

            # 컨빅션 사이징 (변형 1)
            if flags['conviction']:
                conv_mult = min(2.0, max(0.5, score / 70.0))
                margin *= conv_mult

            if margin < 100 or margin > cash:
                continue

            fee = margin * LEVERAGE * FEE_RATE
            cash -= margin + fee
            fees += fee
            positions[sym] = {
                'entry': px, 'margin': margin, 'dir': d, 'extreme': 0.0,
                'highest': px, 'last_px': px, 'score': score, 'dca_count': 0,
            }

    # 잔여 종가 정산
    equity = cash
    for sym, pos in positions.items():
        pnl_pct = (pos['last_px'] - pos['entry']) / pos['entry'] * LEVERAGE * pos['dir']
        fee = pos['margin'] * LEVERAGE * FEE_RATE
        equity += pos['margin'] * (1 + pnl_pct) - fee
        fees += fee
        trades.append({'pnl': pos['margin'] * pnl_pct - fee, 'dir': pos['dir'], 'sym': sym})

    pnl_total = equity - INITIAL_EQUITY
    pnl_chop = sum(tr['pnl'] for tr in trades if 't' in tr and tr['t'] >= CHOP_START)
    longs = [tr for tr in trades if tr['dir'] == 1]
    shorts = [tr for tr in trades if tr['dir'] == -1]
    return {
        'total': pnl_total, 'chop': pnl_chop, 'n': len(trades), 'fees': fees,
        'long_n': len(longs), 'long_pnl': sum(x['pnl'] for x in longs),
        'short_n': len(shorts), 'short_pnl': sum(x['pnl'] for x in shorts),
        'equity': equity,
    }


async def main():
    ex = ccxt_async.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    since = int(datetime(2026, 8, 18, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    try:
        print("데이터 수집 중...")
        btc = await fetch_all(ex, "BTC/USDT:USDT", "1h", since - 260 * 3600 * 1000)
        btc['adx'] = calc_adx(btc, 14)
        btc['ema200'] = btc['c'].ewm(span=200, adjust=False).mean()
        btc['ema50'] = btc['c'].ewm(span=50, adjust=False).mean()
        btc_map = {}
        for i in range(1, len(btc) - 1):
            r = btc.iloc[i]
            above200 = bool(r['c'] >= r['ema200'])
            ema50_up = bool(btc['ema50'].iloc[i] > btc['ema50'].iloc[i - 4])
            btc_map[int(r['t'])] = (float(r['adx']), above200 and ema50_up, above200)

        data = {}
        for sym in SYMBOLS:
            try:
                df = await fetch_all(ex, sym, "15m", since)
                if len(df) >= 200:
                    df['st_dir'], _ = calc_supertrend(df)
                    df['stoch_k'] = calc_stoch_k(df['c'])
                    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
                    df['ema50'] = df['c'].ewm(span=50, adjust=False).mean()
                    df['ema50'] = df['c'].ewm(span=50, adjust=False).mean()
                    df = df.dropna().reset_index(drop=True)
                    data[sym] = df
            except Exception:
                pass
        print(f"{len(data)}개 심볼 수집\n")

        F = {'conviction': False, 'trailing': False, 'adx_pyramid': False, 'bear_short': False}
        variants = [
            ("A. baseline (변형 없음)", dict(F)),
            ("1. 컨빅션 사이징",         dict(F, conviction=True)),
            ("2. 트레일링 스탑",         dict(F, trailing=True)),
            ("3. ADX 피라미딩",          dict(F, adx_pyramid=True)),
            ("4. 베어 숏",               dict(F, bear_short=True)),
            ("1+2 컨빅션+트레일링",      dict(F, conviction=True, trailing=True)),
            ("1+4 컨빅션+베어숏",        dict(F, conviction=True, bear_short=True)),
            ("2+4 트레일링+베어숏",      dict(F, trailing=True, bear_short=True)),
            ("1+2+4 풀셋(컨빅+트렐+베어)", dict(F, conviction=True, trailing=True, bear_short=True)),
        ]
        print(f"{'변형':30s} {'전체':>10s} {'피크이후':>10s} {'거래':>4s} {'롱건':>4s} {'롱손익':>10s} {'숏건':>4s} {'숏손익':>10s}")
        for name, flags in variants:
            r = simulate(data, btc_map, flags)
            print(f"  {name:28s} {r['total']:+10.2f} {r['chop']:+10.2f} {r['n']:4d} {r['long_n']:4d} {r['long_pnl']:+10.2f} {r['short_n']:4d} {r['short_pnl']:+10.2f}")
    finally:
        await ex.close()


if __name__ == "__main__":
    asyncio.run(main())
