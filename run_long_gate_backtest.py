#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_long_gate_backtest.py — 롱 게이트 강화 검증 (1h EMA50)
숏은 전면 차단 확정(전 검증) → 롱 진입 게이트만 최적화.

변형 (전부 롱 온리):
  G0. 게이트 없음 (기준)
  I.  BTC > BTC 1h EMA50 필요 (시장 레짐 게이트)
  J.  심볼 > 심볼 1h EMA50 필요 (종목 추세 게이트 — 생산 스코어링 방식)
  K.  둘 다 필요 (이중 게이트)
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


def simulate(data, btc_map, sym_map, gate, lev=10):
    """
    gate: 'none' | 'btc50' | 'sym50' | 'both'
    btc_map[t] = (adx, above50)
    sym_map[sym][t] = above50
    """
    cash = INITIAL_EQUITY
    positions = {}
    trades = []

    all_ts = sorted(set().union(*[set(df['t']) for df in data.values()]))
    warmup = 60

    for idx, t in enumerate(all_ts):
        if idx < warmup:
            continue
        last_h = ((t // 3600000) * 3600000) - 3600000
        _bm = btc_map.get(last_h, (20.0, True))
        adx_now, btc_above50 = _bm[0], _bm[1]

        deploy = max(0.15, min(1.0, (adx_now - 8) / max(25 - 8, 1.0)))

        if gate == 'none':
            allow_long = True
        elif gate == 'btc50':
            allow_long = btc_above50
        elif gate == 'sym50':
            allow_long = True  # 심볼별 판정은 루프에서
        else:  # both
            allow_long = btc_above50

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
                pnl_pct = (px - pos['entry']) / pos['entry'] * lev * pos['dir']
                # [청산 모델] 마진 -92% 도달 = 강제청산 (유지증거금 근사, 갭 반영)
                if pnl_pct <= -0.92:
                    cash += pos['margin'] - pos['margin']  # 마진 전량 소멸
                    trades.append({'t': t, 'pnl': -pos['margin'], 'lev': lev})
                    del positions[sym]
                    continue
                pos['extreme'] = max(pos['extreme'], pnl_pct)
                best = pos['extreme']
                exit_now = pnl_pct <= -0.30 or pnl_pct <= -0.15
                exit_now = exit_now or (best > 0.4 and pnl_pct < best * 0.5) or (best > 0.2 and pnl_pct < 0.05)
                exit_now = exit_now or pnl_pct >= 0.50
                if exit_now:
                    gross = pos['margin'] * pnl_pct
                    fee = pos['margin'] * lev * FEE_RATE
                    cash += pos['margin'] + gross - fee
                    trades.append({'t': t, 'pnl': gross - fee})
                    del positions[sym]
                continue

            long_sig = curr['st_dir'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
            if long_sig:
                if gate == 'sym50':
                    allow = sym_map.get(sym, {}).get(last_h, True)
                    long_sig = long_sig and allow
                elif gate == 'both':
                    allow = sym_map.get(sym, {}).get(last_h, True)
                    long_sig = long_sig and allow and btc_above50
                elif gate == 'btc50':
                    long_sig = long_sig and btc_above50
            if not (long_sig and deploy > 0):
                continue
            equity = sum(p['margin'] * (1 + (p['last_px'] - p['entry']) / p['entry'] * lev)
                         for p in positions.values()) + cash
            margin = (equity / MAX_POS) * WEIGHT * NEW_RATIO * deploy
            if margin < 100 or margin > cash:
                continue
            fee = margin * lev * FEE_RATE
            cash -= margin + fee
            positions[sym] = {'entry': px, 'margin': margin, 'dir': 1, 'extreme': 0.0, 'last_px': px}

    equity = cash
    for sym, pos in positions.items():
        pnl_pct = (pos['last_px'] - pos['entry']) / pos['entry'] * lev
        if pnl_pct <= -0.92:
            pnl_pct = -1.0
        equity += pos['margin'] * (1 + pnl_pct) - pos['margin'] * lev * FEE_RATE
        trades.append({'pnl': pos['margin'] * pnl_pct - pos['margin'] * lev * FEE_RATE})

    total = equity - INITIAL_EQUITY
    era_pnls = {}
    for name, s, e in ERAS:
        s_ms = s.replace(tzinfo=timezone.utc).timestamp() * 1000
        e_ms = e.replace(tzinfo=timezone.utc).timestamp() * 1000
        era_pnls[name] = sum(tr['pnl'] for tr in trades if 't' in tr and s_ms <= tr['t'] < e_ms)
    wins = sum(1 for tr in trades if tr['pnl'] > 0)
    return {'total': total, 'eras': era_pnls, 'n': len(trades),
            'wr': wins / max(1, len(trades)) * 100, 'equity': equity}


async def main():
    ex = ccxt_async.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    since = int(datetime(2026, 6, 20, tzinfo=timezone.utc).timestamp() * 1000)
    try:
        import pickle
        if os.path.exists("/tmp/lg_data.pkl"):
            print("캐시에서 로드")
            with open("/tmp/lg_data.pkl","rb") as f:
                btc, btc_map, data, sym_map = pickle.load(f)
            print(f"{len(data)}개 심볼 (캐시)")
        else:
            print("데이터 수집 중...")
            btc = await fetch_all(ex, "BTC/USDT:USDT", "1h", since - 400 * 3600 * 1000)
        btc['adx'] = calc_adx(btc, 14)
        btc['ema50'] = btc['c'].ewm(span=50, adjust=False).mean()
        btc_map = {}
        for i in range(1, len(btc) - 1):
            btc_map[int(btc['t'].iloc[i])] = (
                float(btc['adx'].iloc[i]),
                bool(btc['c'].iloc[i] > btc['ema50'].iloc[i]),
            )

        data = {}
        sym_map = {}
        for sym in SYMBOLS:
            try:
                h1 = await fetch_all(ex, sym, "1h", since - 400 * 3600 * 1000)
                h1['ema50'] = h1['c'].ewm(span=50, adjust=False).mean()
                sm = {}
                for i in range(1, len(h1) - 1):
                    sm[int(h1['t'].iloc[i])] = bool(h1['c'].iloc[i] > h1['ema50'].iloc[i])
                sym_map[sym] = sm

                df = await fetch_all(ex, sym, "15m", since)
                if len(df) < 3000:
                    continue
                df['st_dir'] = calc_supertrend(df)
                df['stoch_k'] = calc_stoch_k(df['c'])
                df = df.dropna().reset_index(drop=True)
                data[sym] = df
            except Exception:
                pass
            with open("/tmp/lg_data.pkl","wb") as f:
                pickle.dump((btc, btc_map, data, sym_map), f)
        print(f"{len(data)}개 심볼\n")

        print("=== 이중게이트(K) × 레버리지 비교 (청산 모델 포함) ===")
        hdr = f"{'레버리지':10s} {'전체':>10s} {'E1횡보':>9s} {'E2스케일':>9s} {'E3랠리':>10s} {'E4붕괴':>10s} {'거래':>5s} {'승률':>5s} {'MDD':>7s}"
        print(hdr)
        print("─" * len(hdr))
        for lev in (5, 10, 15, 20, 30):
            r = simulate(data, btc_map, sym_map, 'both', lev=lev)
            e = r['eras']
            print(f"  {lev:>2d}x       {r['total']:+10.0f} {e['E1 횡보']:+9.0f} {e['E2 스케일업']:+9.0f} "
                  f"{e['E3 랠리']:+10.0f} {e['E4 붕괴']:+10.0f} {r['n']:5d} {r['wr']:4.0f}%")
        print()
        print("=== 참조: 게이트별 10x ===")
        for name, g in [("G0 게이트없음", 'none'), ("I BTC50", 'btc50'), ("J 심볼50", 'sym50'), ("K 이중", 'both')]:
            r = simulate(data, btc_map, sym_map, g, lev=10)
            print(f"  {name:14s} {r['total']:+10.0f}  (거래 {r['n']}, 승률 {r['wr']:.0f}%)")
    finally:
        await ex.close()


if __name__ == "__main__":
    asyncio.run(main())
