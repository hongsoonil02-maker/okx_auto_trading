#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_bidirectional_backtest.py — 양방향(롱/숏) 전략 검증
구간: 08-18 ~ 현재 (상승장 + 꼭지 + 꼭지 후 횡보/조정 포함)

변형:
  L. 롱 전용              — 숏 진입 차단 (현 생산 로직의 실효 동작 근사)
  B. 대칭 양방향           — 롱/숏 동일 조건 (심볼별 추세 추종)
  R. 양방향 + BTC 레짐     — 생산 로직과 동일: BTC>=EMA200일 때만 롱,
                             BTC 불장(>=EMA200 & EMA50 상승)이면 숏 억제
  S. 촙-숏 적응형          — ADX>=25 추세장: 레짐 모드 / ADX<25 횡보장: 숏만 허용
                             (꼭지 후 횡보 구간을 숏으로 공략)
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
    "ZRO/USDT:USDT", "CAP/USDT:USDT", "VVV/USDT:USDT", "LTC/USDT:USDT",
    "DOGE/USDT:USDT", "AXTI/USDT:USDT", "CHIP/USDT:USDT", "CRCL/USDT:USDT",
    "AAVE/USDT:USDT", "LIT/USDT:USDT", "FIL/USDT:USDT", "HYPE/USDT:USDT",
    "BNB/USDT:USDT", "ZEC/USDT:USDT", "TRB/USDT:USDT", "BOME/USDT:USDT",
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
            if sd.iloc[i] == 1 and fl.iloc[i] < fl.iloc[i - 1]:
                fl.iloc[i] = fl.iloc[i - 1]
            if sd.iloc[i] == -1 and fu.iloc[i] > fu.iloc[i - 1]:
                fu.iloc[i] = fu.iloc[i - 1]
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
    while True:
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


def simulate(data, btc_map, mode):
    """
    btc_map: {확정 1h 시작시각(ms): (adx, btc_bull, above_ema200)}
    mode: 'long' | 'both' | 'regime' | 'chop_short'
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
        hour_floor = (t // 3600000) * 3600000
        last_h = hour_floor - 3600000
        adx_now, btc_bull, above200 = btc_map.get(last_h, (None, True, True))
        chop = adx_now is not None and adx_now < 25

        # 모드별 진입 허용 방향
        if mode == 'long':
            allow_long, allow_short = True, False
        elif mode == 'both':
            allow_long, allow_short = True, True
        elif mode == 'regime':
            allow_long = above200
            allow_short = not btc_bull
        else:  # chop_short
            if chop:
                allow_long, allow_short = False, True
            else:
                allow_long, allow_short = above200, not btc_bull

        # 현재 자산 (브레이커 없음 — 순수 방향성 효과 측정)
        equity = cash
        for sym, pos in positions.items():
            pnl_pct = (pos['last_px'] - pos['entry']) / pos['entry'] * LEVERAGE * pos['dir']
            equity += pos['margin'] * (1 + pnl_pct)

        for sym, df in data.items():
            hits = df.index[df['t'] == t]
            if len(hits) == 0:
                continue
            i = hits[0]
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
                exit_now = pnl_pct <= -0.30
                if pos['dir'] == 1:
                    exit_now = exit_now or (best > 0.4 and pnl_pct < best * 0.5) \
                        or (best > 0.2 and pnl_pct < 0.05)
                else:
                    exit_now = exit_now or (best < -0.4 and pnl_pct > best * 0.5) \
                        or (best < -0.2 and pnl_pct > -0.05)
                exit_now = exit_now or pnl_pct >= 0.50
                if exit_now:
                    gross = pos['margin'] * pnl_pct
                    fee = pos['margin'] * LEVERAGE * FEE_RATE
                    cash += pos['margin'] + gross - fee
                    fees += fee
                    trades.append({'t': t, 'pnl': gross - fee, 'dir': pos['dir']})
                    del positions[sym]
                continue

            long_sig = curr['st_dir'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
            short_sig = curr['st_dir'] == -1 and prev['stoch_k'] > 80 and curr['stoch_k'] <= 80
            enter_long = long_sig and allow_long
            enter_short = short_sig and allow_short
            if not (enter_long or enter_short):
                continue
            margin = (equity / MAX_POS) * WEIGHT * NEW_RATIO
            if margin < 100 or margin > cash:
                continue
            d = 1 if enter_long else -1
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
        trades.append({'t': all_ts[-1], 'pnl': pos['margin'] * pnl_pct - fee, 'dir': pos['dir']})

    pnl_total = equity - INITIAL_EQUITY
    pnl_chop = sum(tr['pnl'] for tr in trades if tr['t'] >= CHOP_START)
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
        # BTC 1h: EMA200 워밍업 위해 250시간 전부터
        btc = await fetch_all(ex, "BTC/USDT:USDT", "1h", since - 260 * 3600 * 1000)
        btc['adx'] = calc_adx(btc, 14)
        btc['ema200'] = btc['c'].ewm(span=200, adjust=False).mean()
        btc['ema50'] = btc['c'].ewm(span=50, adjust=False).mean()
        btc_map = {}
        for i in range(1, len(btc) - 1):  # 마지막 행(진행 중) 제외
            r = btc.iloc[i]
            above200 = bool(r['c'] >= r['ema200'])
            ema50_up = bool(btc['ema50'].iloc[i] > btc['ema50'].iloc[i - 4])
            btc_bull = above200 and ema50_up
            btc_map[int(r['t'])] = (float(r['adx']), btc_bull, above200)

        data = {}
        for sym in SYMBOLS:
            try:
                df = await fetch_all(ex, sym, "15m", since)
                if len(df) < 200:
                    continue
                df['st_dir'] = calc_supertrend(df)
                df['stoch_k'] = calc_stoch_k(df['c'])
                df = df.dropna().reset_index(drop=True)
                data[sym] = df
            except Exception as e:
                print(f"  {sym}: 실패 - {str(e)[:80]}")
        print(f"{len(data)}개 심볼 수집 완료")

        variants = [
            ("L. 롱 전용", 'long'),
            ("B. 대칭 양방향", 'both'),
            ("R. 양방향+BTC레짐(생산 로직)", 'regime'),
            ("S. 촙-숏 적응형(ADX<25→숏만)", 'chop_short'),
        ]
        print(f"\n{'변형':36s} {'전체손익':>10s} {'피크이후':>10s} {'거래':>4s} {'롱손익':>10s} {'숏손익':>10s} {'숏건수':>5s}")
        for name, mode in variants:
            r = simulate(data, btc_map, mode)
            print(f"  {name:34s} {r['total']:+10.2f} {r['chop']:+10.2f} {r['n']:4d} "
                  f"{r['long_pnl']:+10.2f} {r['short_pnl']:+10.2f} {r['short_n']:5d}")
    finally:
        await ex.close()


if __name__ == "__main__":
    asyncio.run(main())
