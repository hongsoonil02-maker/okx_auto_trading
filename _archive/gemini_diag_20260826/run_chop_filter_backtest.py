#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_chop_filter_backtest.py — 횡보장 필터 + 서킷 브레이커 + 노출 스케일 검증
구간: 08-18 ~ 현재 (상승장 08-18~21 + 꼭지 후 횡보장 08-21~23 포함)
실제 손실 구간(08-21 14:00 UTC 피크 이후)에서 필터 효과가 핵심 검증 대상.

변형:
  A. baseline        — 필터 없음 (현 전략과 동일 로직)
  B. chop(ADX n)     — BTC 1h ADX < n 이면 신규 진입 차단 (n=15/20/25)
  C. chop + breaker  — B + 일손실 -6% 서킷 브레이커
  D. 풀 세트          — C + 노출 스케일 0.6
"""
import asyncio
import time
from datetime import datetime, timezone
import pandas as pd
import ccxt.async_support as ccxt_async

LEVERAGE = 10
FEE_RATE = 0.0005          # taker 0.05% (진입+청산 각각)
MAX_POS = 15
WEIGHT = 1.5               # venture 포트폴리오 가중치
NEW_RATIO = 0.50           # 신규 진입 = 목표 마진의 50%
INITIAL_EQUITY = 19000.0
CHOP_START = datetime(2026, 8, 21, 14, 0, tzinfo=timezone.utc).timestamp() * 1000  # 피크 시점

SYMBOLS = [
    # 손실 상위 (피크 이후)
    "ZRO/USDT:USDT", "CAP/USDT:USDT", "VVV/USDT:USDT", "LTC/USDT:USDT",
    "DOGE/USDT:USDT", "AXTI/USDT:USDT", "CHIP/USDT:USDT", "CRCL/USDT:USDT",
    "AAVE/USDT:USDT", "LIT/USDT:USDT", "FIL/USDT:USDT", "HYPE/USDT:USDT",
    # 수익 상위 (대조군)
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


def simulate(data, adx_map, adx_thr, breaker, scale):
    """
    data: {sym: df(15m, indicators 포함)}
    adx_map: {확정 1h 캔들 시작시각(ms): adx}
    """
    cash = INITIAL_EQUITY
    positions = {}   # sym -> dict
    trades, fees = [], 0.0
    cb_anchor, cb_date, cb_open = None, None, False

    all_ts = sorted(set().union(*[set(df['t']) for df in data.values()]))
    warmup = 60 * 4  # 최초 60캔들 워밍업 스킵

    for idx, t in enumerate(all_ts):
        if idx < warmup:
            continue
        # ── 서킷 브레이커: 자산 = 현금 + 보유 마진(+미실현) ──
        equity = cash
        for sym, pos in positions.items():
            df = data[sym]
            row = df[df['t'] == t]
            px = float(row['c'].iloc[0]) if len(row) else pos['last_px']
            pos['last_px'] = px
            pnl_pct = (px - pos['entry']) / pos['entry'] * LEVERAGE * pos['dir']
            equity += pos['margin'] * (1 + pnl_pct)

        day = datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        if breaker:
            if cb_date != day:
                cb_date, cb_anchor, cb_open = day, equity, False
            elif cb_anchor and not cb_open and (equity - cb_anchor) / cb_anchor * 100 <= -6.0:
                cb_open = True
            elif cb_open and (equity - cb_anchor) / cb_anchor * 100 > -3.0:
                cb_open = False

        # ── 횡보 판정 (마지막 확정 1h ADX) ──
        hour_floor = (t // 3600000) * 3600000
        last_closed_hour = hour_floor - 3600000
        adx_now = adx_map.get(last_closed_hour)
        chop = adx_thr is not None and adx_now is not None and adx_now < adx_thr
        blocked = chop or (breaker and cb_open)

        for sym, df in data.items():
            rows = df[df['t'] == t]
            if not len(rows):
                continue
            i = df.index[df['t'] == t][0]
            if i < 1:
                continue
            curr, prev = df.iloc[i], df.iloc[i - 1]
            px = float(curr['c'])

            pos = positions.get(sym)
            # ── 청산 로직 ──
            if pos:
                pnl_pct = (px - pos['entry']) / pos['entry'] * LEVERAGE * pos['dir']
                pos['extreme'] = max(pos['extreme'], pnl_pct) if pos['dir'] == 1 else min(pos['extreme'], pnl_pct)
                best = pos['extreme']
                exit_now = False
                if pnl_pct <= -0.30:
                    exit_now = True
                elif pos['dir'] == 1 and best > 0.4 and pnl_pct < best * 0.5:
                    exit_now = True
                elif pos['dir'] == -1 and best < -0.4 and pnl_pct > best * 0.5:
                    exit_now = True
                elif pos['dir'] == 1 and best > 0.2 and pnl_pct < 0.05:
                    exit_now = True
                elif pos['dir'] == -1 and best < -0.2 and pnl_pct > -0.05:
                    exit_now = True
                elif pnl_pct >= 0.50:
                    exit_now = True
                if exit_now:
                    gross = pos['margin'] * pnl_pct
                    fee = pos['margin'] * LEVERAGE * FEE_RATE  # 청산 수수료
                    cash += pos['margin'] + gross - fee
                    fees += fee
                    trades.append({'sym': sym, 't': t, 'pnl': gross - fee, 'side': pos['dir']})
                    del positions[sym]
                continue

            # ── 진입 로직 (차단 시 스킵) ──
            if blocked or sym in positions:
                continue
            long_sig = curr['st_dir'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
            short_sig = curr['st_dir'] == -1 and prev['stoch_k'] > 80 and curr['stoch_k'] <= 80
            if not (long_sig or short_sig):
                continue
            margin = (equity / MAX_POS) * WEIGHT * scale * NEW_RATIO
            if margin < 100 or margin > cash:
                continue
            d = 1 if long_sig else -1
            fee = margin * LEVERAGE * FEE_RATE  # 진입 수수료
            cash -= margin + fee
            fees += fee
            positions[sym] = {
                'entry': px, 'margin': margin, 'dir': d,
                'extreme': 0.0, 'last_px': px,
            }

    # 잔여 포지션 종가 정산
    equity = cash
    for sym, pos in positions.items():
        px = pos['last_px']
        pnl_pct = (px - pos['entry']) / pos['entry'] * LEVERAGE * pos['dir']
        fee = pos['margin'] * LEVERAGE * FEE_RATE
        equity += pos['margin'] * (1 + pnl_pct) - fee
        fees += fee
        trades.append({'sym': sym, 't': all_ts[-1], 'pnl': pos['margin'] * pnl_pct - fee, 'side': pos['dir']})

    pnl_total = equity - INITIAL_EQUITY
    pnl_chop = sum(tr['pnl'] for tr in trades if tr['t'] >= CHOP_START)
    return pnl_total, pnl_chop, len(trades), fees, equity


async def main():
    ex = ccxt_async.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    since = int(datetime(2026, 8, 18, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    try:
        print("데이터 수집 중...")
        btc = await fetch_all(ex, "BTC/USDT:USDT", "1h", since - 30 * 3600 * 1000)
        btc['adx'] = calc_adx(btc, 14)
        # 확정 캔들 기준 adx_map (마지막 행 제외 = 진행 중 캔들)
        adx_map = {int(r['t']): float(r['adx']) for _, r in btc.iloc[:-1].iterrows()}

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
                print(f"  {sym}: {len(df)}캔들")
            except Exception as e:
                print(f"  {sym}: 실패 - {str(e)[:80]}")

        # ADX 구간별 값 확인
        for label, ts in [("08-19 12:00(상승)", datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)),
                          ("08-21 12:00(꼭지)", datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)),
                          ("08-22 12:00(횡보)", datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)),
                          ("08-23 06:00(현재)", datetime(2026, 8, 23, 6, 0, tzinfo=timezone.utc))]:
            h = int(ts.timestamp() * 1000 // 3600000 * 3600000) - 3600000
            print(f"  BTC ADX {label}: {adx_map.get(h, float('nan')):.1f}")

        variants = [
            ("A. baseline (필터 없음)", None, False, 1.0),
            ("B1. chop ADX<15", 15, False, 1.0),
            ("B2. chop ADX<20", 20, False, 1.0),
            ("B3. chop ADX<25", 25, False, 1.0),
            ("E. 브레이커-6% 단독", None, True, 1.0),
            ("C. chop20 + 브레이커-6%", 20, True, 1.0),
            ("F. chop25 + 브레이커-6%", 25, True, 1.0),
            ("D. chop20 + 브레이커 + 스케일0.6", 20, True, 0.6),
            ("G. chop25 + 브레이커 + 스케일0.6", 25, True, 0.6),
        ]
        print(f"\n{'변형':34s} {'전체손익':>10s} {'피크이후':>10s} {'거래수':>6s} {'수수료':>8s} {'최종자산':>10s}")
        for name, thr, brk, scale in variants:
            p_tot, p_chop, n, fee, eq = simulate(data, adx_map, thr, brk, scale)
            print(f"  {name:32s} {p_tot:+10.2f} {p_chop:+10.2f} {n:6d} {fee:8.2f} {eq:10.2f}")
    finally:
        await ex.close()


if __name__ == "__main__":
    asyncio.run(main())
