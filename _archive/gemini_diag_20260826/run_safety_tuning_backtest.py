#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_safety_tuning_backtest.py — 안전장치 강도별 비교 백테스트
DCA/피라미딩/서킷브레이커/포지션손실한도 조합을 변형하여 손익 비교.
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


def simulate(data, adx_map, dca_max, pyramid_enabled, cb_lock, pos_loss_limit):
    """
    dca_max: 최대 DCA 횟수 (기존 8)
    pyramid_enabled: 불타기 허용 여부
    cb_lock: True면 당일 락(해제 불가), False면 절반 회복 시 해제
    pos_loss_limit: 포지션별 손실 한도 (마진 대비 %, None=무제한)
    """
    cash = INITIAL_EQUITY
    # positions[sym] = list of {entry, margin, dir, extreme}
    positions = {}
    trades = []
    fees = 0.0
    cb_anchor, cb_date, cb_open = None, None, False

    all_ts = sorted(set().union(*[set(df['t']) for df in data.values()]))
    warmup = 60

    for idx, t in enumerate(all_ts):
        if idx < warmup:
            continue
        hour_floor = (t // 3600000) * 3600000
        last_h = hour_floor - 3600000
        adx_now = adx_map.get(last_h, 30.0)
        chop = adx_now < 25

        # 현재 자산
        equity = cash
        for sym, plist in positions.items():
            for p in plist:
                pnl_pct = (p['last_px'] - p['entry']) / p['entry'] * LEVERAGE * p['dir']
                equity += p['margin'] * (1 + pnl_pct)

        # 서킷 브레이커
        day = datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        if cb_date != day:
            cb_anchor, cb_date, cb_open = equity, day, False
        elif not cb_open and (equity - cb_anchor) / cb_anchor * 100 <= -6.0:
            cb_open = True
        elif cb_open and not cb_lock and (equity - cb_anchor) / cb_anchor * 100 > -3.0:
            cb_open = False

        blocked = chop or cb_open

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
            open_margin = sum(p['margin'] for p in plist)

            # ── 기존 포지션 청산 로직 ──
            remaining = []
            for p in plist:
                p['last_px'] = px
                pnl_pct = (px - p['entry']) / p['entry'] * LEVERAGE * p['dir']
                p['extreme'] = max(p['extreme'], pnl_pct) if p['dir'] == 1 else min(p['extreme'], pnl_pct)
                best = p['extreme']

                # 포지션별 손실 한도 체크
                if pos_loss_limit is not None and pnl_pct <= pos_loss_limit:
                    gross = p['margin'] * pnl_pct
                    fee = p['margin'] * LEVERAGE * FEE_RATE
                    cash += p['margin'] + gross - fee
                    fees += fee
                    trades.append({'t': t, 'pnl': gross - fee, 'dir': p['dir']})
                    continue  # 이 포지션 청산

                exit_now = pnl_pct <= -0.30
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
                    trades.append({'t': t, 'pnl': gross - fee, 'dir': p['dir']})
                else:
                    remaining.append(p)
            positions[sym] = remaining

            if remaining:
                # 이미 포지션 있음 → DCA/피라미딩 체크
                total_pnl = sum((px - p['entry']) / p['entry'] * LEVERAGE * p['dir'] for p in remaining)
                avg_entry = sum(p['entry'] * p['margin'] for p in remaining) / sum(p['margin'] for p in remaining) if remaining else 0
                n_entries = len(remaining)
                direction = remaining[0]['dir']
                pnl_from_avg = (px - avg_entry) / avg_entry * LEVERAGE * direction if avg_entry > 0 else 0

                # DCA: 손실 중이고 dca_max 미달
                if n_entries < dca_max and not blocked and pnl_from_avg < -0.05 and direction == curr['st_dir']:
                    margin = (equity / MAX_POS) * WEIGHT * 0.6 * 0.25  # DCA는 목표의 25%
                    if margin >= 50 and margin <= cash:
                        fee = margin * LEVERAGE * FEE_RATE
                        cash -= margin + fee
                        fees += fee
                        remaining.append({'entry': px, 'margin': margin, 'dir': direction, 'extreme': 0.0, 'last_px': px})

                # 피라미딩: 수익 중이고 불타기 허용
                if pyramid_enabled and not blocked and total_pnl > 0.20 * sum(p['margin'] for p in remaining) / LEVERAGE * LEVERAGE and n_entries < dca_max + 2:
                    margin = (equity / MAX_POS) * WEIGHT * 0.6 * 0.35
                    if margin >= 50 and margin <= cash:
                        fee = margin * LEVERAGE * FEE_RATE
                        cash -= margin + fee
                        fees += fee
                        remaining.append({'entry': px, 'margin': margin, 'dir': direction, 'extreme': 0.0, 'last_px': px})
                continue

            # ── 신규 진입 ──
            if blocked:
                continue
            long_sig = curr['st_dir'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
            short_sig = curr['st_dir'] == -1 and prev['stoch_k'] > 80 and curr['stoch_k'] <= 80
            if not (long_sig or short_sig):
                continue
            margin = (equity / MAX_POS) * WEIGHT * 0.6 * 0.50
            if margin < 100 or margin > cash:
                continue
            d = 1 if long_sig else -1
            fee = margin * LEVERAGE * FEE_RATE
            cash -= margin + fee
            fees += fee
            positions[sym] = [{'entry': px, 'margin': margin, 'dir': d, 'extreme': 0.0, 'last_px': px}]

    # 잔여 종가 정산
    equity = cash
    for sym, plist in positions.items():
        for p in plist:
            pnl_pct = (p['last_px'] - p['entry']) / p['entry'] * LEVERAGE * p['dir']
            fee = p['margin'] * LEVERAGE * FEE_RATE
            equity += p['margin'] * (1 + pnl_pct) - fee
            fees += fee
            trades.append({'pnl': p['margin'] * pnl_pct - fee, 'dir': p['dir']})

    pnl_total = equity - INITIAL_EQUITY
    pnl_chop = sum(tr['pnl'] for tr in trades if 't' in tr and tr['t'] >= CHOP_START)
    return {
        'total': pnl_total, 'chop': pnl_chop, 'n': len(trades), 'fees': fees, 'equity': equity
    }


async def main():
    ex = ccxt_async.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    since = int(datetime(2026, 8, 18, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    try:
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
            ("A. 기존(DCA8/피라미딩O/완화)", 8, True, False, None),
            ("B. DCA 8→4", 4, True, False, None),
            ("C. DCA 8→2", 2, True, False, None),
            ("D. 피라미딩 끔", 8, False, False, None),
            ("E. 서킷락(당일락)", 8, True, True, None),
            ("F. 포지션손실 -15%", 8, True, False, -0.15),
            ("G. 포지션손실 -10%", 8, True, False, -0.10),
            ("H. 종합(4/끔/락/15%)", 4, False, True, -0.15),
            ("I. 종합(4/끔/락/10%)", 4, False, True, -0.10),
        ]
        print(f"{'변형':34s} {'전체손익':>10s} {'피크이후':>10s} {'거래수':>6s} {'수수료':>8s} {'최종자산':>10s}")
        for name, dca, pyr, lock, plim in variants:
            r = simulate(data, adx_map, dca, pyr, lock, plim)
            print(f"  {name:32s} {r['total']:+10.2f} {r['chop']:+10.2f} {r['n']:6d} {r['fees']:8.2f} {r['equity']:10.2f}")
    finally:
        await ex.close()


if __name__ == "__main__":
    asyncio.run(main())
