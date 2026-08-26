#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_short_gate_backtest.py — 숏 해제 조건 검증 (6/23 ~ 8/26 전체 레짐)
시대: E1 횡보(6/23~7/23) / E2 스케일업(7/27~8/2) / E3 랠리(8/9~8/21) / E4 붕괴(8/21~26)

변형:
  A. 현행        — BTC 불장(>=EMA200 & EMA50상승)이면 숏 차단 (생산 로직)
  B. 천정확정 5% — A + BTC가 14일 고점 대비 -5% 이하면 숏 허용
  C. 천정확정 3% — A + -3% 이하면 숏 허용 (더 민감)
  D. 풀 양방향   — 숏 항상 허용 (대조군)
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

ERAS = [
    ("E1 횡보", datetime(2026, 6, 23), datetime(2026, 7, 24)),
    ("E2 스케일업", datetime(2026, 7, 27), datetime(2026, 8, 3)),
    ("E3 랠리", datetime(2026, 8, 9), datetime(2026, 8, 21, 14)),
    ("E4 붕괴", datetime(2026, 8, 21, 14), datetime(2026, 8, 26, 10)),
]

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
    "BNB/USDT:USDT", "DOGE/USDT:USDT", "ZEC/USDT:USDT", "CRV/USDT:USDT",
    "NEAR/USDT:USDT", "LTC/USDT:USDT", "HYPE/USDT:USDT", "AAVE/USDT:USDT",
    "UNI/USDT:USDT", "TAO/USDT:USDT", "PEPE/USDT:USDT", "LIT/USDT:USDT",
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
    for _ in range(40):
        batch = await ex.fetch_ohlcv(sym, tf, since=since, limit=300)
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 300:
            break
        since = batch[-1][0] + 1
        await asyncio.sleep(0.04)
    df = pd.DataFrame(out, columns=['t', 'o', 'h', 'l', 'c', 'v']).drop_duplicates('t')
    return df.sort_values('t').reset_index(drop=True)


def simulate(data, btc1h_map, gate, sym_1h_map=None):
    """
    gate: 'current'|'dd5'|'dd3'|'both'|'btc50'|'cur+btc50'|'sym50'|'cur+sym50'
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
        _m = btc1h_map.get(last_h, (20.0, True, 0.0, False))
        adx_now, btc_bull, btc_dd, btc50_fall = _m[0], _m[1], _m[2], _m[3] if len(_m) > 3 else False

        # 배포 스케일 (현행: 연속 곡선)
        deploy = max(0.15, min(1.0, (adx_now - 8) / max(25 - 8, 1.0)))

        # 숏 허용 판정
        if gate == 'current':
            allow_short = not btc_bull
        elif gate == 'dd5':
            allow_short = (not btc_bull) or (btc_dd >= 5.0)
        elif gate == 'dd3':
            allow_short = (not btc_bull) or (btc_dd >= 3.0)
        elif gate == 'btc50':
            allow_short = btc50_fall
        elif gate == 'cur+btc50':
            allow_short = (not btc_bull) or btc50_fall
        elif gate == 'sym50':
            allow_short = False  # 심볼별 판정은 아래 루프에서
        elif gate == 'cur+sym50':
            allow_short = not btc_bull  # 합집합: 심볼 판정은 루프에서 추가
        else:
            allow_short = True
        allow_long = True

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
                exit_now = pnl_pct <= -0.30 or pnl_pct <= -0.15  # 하드스탑 + 포지션한도
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
                    trades.append({'t': t, 'pnl': gross - fee, 'dir': pos['dir']})
                    del positions[sym]
                continue

            if deploy <= 0:
                continue
            long_sig = curr['st_dir'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
            short_sig = curr['st_dir'] == -1 and prev['stoch_k'] > 80 and curr['stoch_k'] <= 80
            if gate in ('sym50', 'cur+sym50') and short_sig:
                sm = (sym_1h_map or {}).get(sym, {})
                sym50_fall = sm.get(last_h, False)
                if gate == 'sym50':
                    short_sig = short_sig and sym50_fall
                else:
                    short_sig = short_sig or sym50_fall
            if long_sig and not allow_long:
                long_sig = False
            if short_sig and not allow_short:
                short_sig = False
            if not (long_sig or short_sig):
                continue
            equity = sum(p['margin'] * (1 + (p['last_px'] - p['entry']) / p['entry'] * LEVERAGE * p['dir'])
                         for p in positions.values()) + cash
            margin = (equity / MAX_POS) * WEIGHT * NEW_RATIO * deploy
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
        trades.append({'pnl': pos['margin'] * pnl_pct - fee, 'dir': pos['dir']})

    total = equity - INITIAL_EQUITY
    era_pnls = {}
    for name, s, e in ERAS:
        s_ms = s.replace(tzinfo=timezone.utc).timestamp() * 1000
        e_ms = e.replace(tzinfo=timezone.utc).timestamp() * 1000
        era_pnls[name] = sum(tr['pnl'] for tr in trades if 't' in tr and s_ms <= tr['t'] < e_ms)
    wins = sum(1 for tr in trades if tr['pnl'] > 0)
    shorts = [tr for tr in trades if tr['dir'] == -1]
    return {
        'total': total, 'eras': era_pnls, 'n': len(trades), 'fees': fees,
        'wr': wins / max(1, len(trades)) * 100,
        'short_n': len(shorts), 'short_pnl': sum(tr['pnl'] for tr in shorts),
        'equity': equity,
    }


async def main():
    ex = ccxt_async.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    since = int(datetime(2026, 6, 20, tzinfo=timezone.utc).timestamp() * 1000)
    try:
        print("데이터 수집 중 (6/20~8/26, 15m + BTC 1h)... 몇 분 소요")
        btc = await fetch_all(ex, "BTC/USDT:USDT", "1h", since - 400 * 3600 * 1000)
        btc['adx'] = calc_adx(btc, 14)
        btc['ema200'] = btc['c'].ewm(span=200, adjust=False).mean()
        btc['ema50'] = btc['c'].ewm(span=50, adjust=False).mean()
        btc['high14d'] = btc['c'].rolling(min(14 * 24, len(btc))).max()
        btc['dd'] = (btc['high14d'] - btc['c']) / btc['high14d'] * 100

        btc_map = {}
        for i in range(1, len(btc) - 1):
            r = btc.iloc[i]
            above200 = bool(r['c'] >= r['ema200'])
            ema50_up = bool(btc['ema50'].iloc[i] > btc['ema50'].iloc[i - 4])
            ema50_falling = bool(btc['ema50'].iloc[i] < btc['ema50'].iloc[i - 6])
            btc_map[int(r['t'])] = (float(r['adx']), above200 and ema50_up, float(r['dd']), ema50_falling)

        data = {}
        sym_1h_map = {}  # sym -> {확정1h시각: ema50_falling}
        for sym in SYMBOLS:
            try:
                h1 = await fetch_all(ex, sym, "1h", since - 400 * 3600 * 1000)
                h1['ema50'] = h1['c'].ewm(span=50, adjust=False).mean()
                m = {}
                for i in range(7, len(h1) - 1):
                    m[int(h1['t'].iloc[i])] = bool(h1['ema50'].iloc[i] < h1['ema50'].iloc[i - 6])
                sym_1h_map[sym] = m
                df = await fetch_all(ex, sym, "15m", since)
                if len(df) < 3000:
                    print(f"  {sym}: 데이터 부족({len(df)}) — 제외")
                    continue
                df['st_dir'] = calc_supertrend(df)
                df['stoch_k'] = calc_stoch_k(df['c'])
                df = df.dropna().reset_index(drop=True)
                data[sym] = df
                print(f"  {sym}: {len(df)}캔들")
            except Exception as e:
                print(f"  {sym}: 실패 {str(e)[:60]}")

        print(f"\n{len(data)}개 심볼로 검증\n")
        variants = [
            ("A. 현행 (불장 숏차단)", 'current'),
            ("E. BTC 1h EMA50 하락시 숏", 'btc50'),
            ("F. 현행+BTC EMA50 하락", 'cur+btc50'),
            ("G. 심볼 1h EMA50 하락시 숏", 'sym50'),
            ("H. 현행+심볼 EMA50 하락", 'cur+sym50'),
            ("D. 풀 양방향 (대조)", 'both'),
        ]
        hdr = f"{'변형':24s} {'전체':>10s} {'E1횡보':>9s} {'E2스케일':>9s} {'E3랠리':>10s} {'E4붕괴':>10s} {'숏건수':>5s} {'숏손익':>10s}"
        print(hdr)
        print("─" * len(hdr))
        for name, g in variants:
            r = simulate(data, btc_map, g, sym_1h_map)
            e = r['eras']
            print(f"  {name:22s} {r['total']:+10.0f} {e['E1 횡보']:+9.0f} {e['E2 스케일업']:+9.0f} "
                  f"{e['E3 랠리']:+10.0f} {e['E4 붕괴']:+10.0f} {r['short_n']:5d} {r['short_pnl']:+10.0f}")
    finally:
        await ex.close()


if __name__ == "__main__":
    asyncio.run(main())
