#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_circuit_breaker_test_final.py
과거 로직(매일 리셋, 숏 10배, 하따 허용) vs 현재 스마트 로직(다이내믹 서킷, 숏 5배, 하따 금지) 비교
"""
import asyncio
from datetime import datetime, timezone
import time
import pandas as pd
import ccxt.async_support as ccxt_async

LONG_LEVERAGE = 10
FEE_RATE = 0.0005
MAX_POS = 15
WEIGHT = 1.5
NEW_RATIO = 0.50
INITIAL_EQUITY = 19000.0
CB_ROE = -4.0

ERAS = [
    ("E1 횡보(6/23~7/24)", datetime(2026, 6, 23), datetime(2026, 7, 24)),
    ("E2 랠리(7/25~8/20)", datetime(2026, 7, 25), datetime(2026, 8, 20)),
    ("E3 붕괴(8/21~8/29)", datetime(2026, 8, 21), datetime(2026, 8, 29)),
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
        if not batch: break
        out.extend(batch)
        if len(batch) < 300: break
        since = batch[-1][0] + 1
        await asyncio.sleep(0.04)
    df = pd.DataFrame(out, columns=['t', 'o', 'h', 'l', 'c', 'v']).drop_duplicates('t')
    return df.sort_values('t').reset_index(drop=True)

def simulate(data, btc1h_map, mode='CURRENT'):
    # mode 'PREVIOUS':
    #   - Daily reset CB at UTC 00:00. Tripped? Block both.
    #   - Short leverage = 10x. Only in bear market.
    #   - No Falling knife block. No weekly MDD.
    # mode 'CURRENT':
    #   - 48h cooldown CB.
    #   - Dynamic: if bear, block long but allow 5x short. if bull, block both.
    #   - Weekly MDD -10%. Falling knife > -10% block longs.
    
    cash = INITIAL_EQUITY
    positions = {}
    trades = []
    fees = 0.0
    mdd = 0.0
    eq_peak = INITIAL_EQUITY

    all_ts = sorted(set().union(*[set(df['t']) for df in data.values()]))
    warmup = 60

    cb_date = ""
    cb_anchor = INITIAL_EQUITY
    cb_cooldown_until = 0
    cb_tripped = False
    
    weekly_peak = INITIAL_EQUITY
    current_week = ""
    weekly_locked = False

    for idx, t in enumerate(all_ts):
        if idx < warmup: continue
        
        # Calculate Equity
        equity = cash
        for p in positions.values():
            lev = LONG_LEVERAGE if p['dir'] == 1 else (10 if mode == 'PREVIOUS' else 5)
            equity += p['margin'] * (1 + (p['last_px'] - p['entry']) / p['entry'] * lev * p['dir'])
                     
        if equity > eq_peak:
            eq_peak = equity
        else:
            mdd = max(mdd, (eq_peak - equity) / eq_peak * 100)
            
        dt = datetime.fromtimestamp(t / 1000, tz=timezone.utc)
        today = dt.strftime("%Y-%m-%d")
        week_str = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]}"
        
        if mode == 'CURRENT':
            if week_str != current_week:
                current_week = week_str
                weekly_peak = equity
                weekly_locked = False
            else:
                weekly_peak = max(weekly_peak, equity)
                
            weekly_mdd_pct = (equity - weekly_peak) / weekly_peak * 100 if weekly_peak > 0 else 0
            if weekly_mdd_pct <= -10.0:
                weekly_locked = True
                
            if cb_date != today and t > cb_cooldown_until:
                cb_date = today
                cb_anchor = equity
                cb_tripped = False
        else:
            # PREVIOUS: Daily reset unconditionally at UTC 00:00
            if cb_date != today:
                cb_date = today
                cb_anchor = equity
                cb_tripped = False
                
        daily_chg = (equity - cb_anchor) / cb_anchor * 100 if cb_anchor > 0 else 0
        if daily_chg <= CB_ROE and not cb_tripped:
            cb_tripped = True
            if mode == 'CURRENT':
                cb_cooldown_until = t + (48 * 3600 * 1000)

        is_cb_active = cb_tripped or (mode == 'CURRENT' and t < cb_cooldown_until)

        last_h = ((t // 3600000) * 3600000) - 3600000
        _m = btc1h_map.get(last_h, (20.0, True, 0.0, False))
        adx_now, btc_bull, btc_dd = _m[0], _m[1], _m[2]

        deploy = max(0.15, min(1.0, (adx_now - 8) / max(25 - 8, 1.0)))
        
        allow_long = True
        allow_short = True
        
        if mode == 'PREVIOUS':
            if is_cb_active:
                allow_long = False
                allow_short = False
            if btc_bull:
                allow_short = False
        else:
            if weekly_locked:
                allow_long = False
                allow_short = False
            if is_cb_active:
                long_blocked = True
                short_blocked = True
                if not btc_bull:
                    short_blocked = False
                allow_long = not long_blocked
                allow_short = not short_blocked

        for sym, df in data.items():
            hits = df.index[df['t'] == t]
            if len(hits) == 0: continue
            i = hits[0]
            if i < 1: continue
            curr, prev = df.iloc[i], df.iloc[i - 1]
            px = float(curr['c'])

            pos = positions.get(sym)
            if pos:
                lev = LONG_LEVERAGE if pos['dir'] == 1 else (10 if mode == 'PREVIOUS' else 5)
                pos['last_px'] = px
                pnl_pct = (px - pos['entry']) / pos['entry'] * lev * pos['dir']
                pos['extreme'] = max(pos['extreme'], pnl_pct) if pos['dir'] == 1 else min(pos['extreme'], pnl_pct)
                best = pos['extreme']
                exit_now = pnl_pct <= -0.30 or pnl_pct <= -0.15
                if pos['dir'] == 1:
                    exit_now = exit_now or (best > 0.4 and pnl_pct < best * 0.5) or (best > 0.2 and pnl_pct < 0.05)
                else:
                    exit_now = exit_now or (best < -0.4 and pnl_pct > best * 0.5) or (best < -0.2 and pnl_pct > -0.05)
                exit_now = exit_now or pnl_pct >= 0.50
                if pnl_pct <= -0.90: exit_now = True

                if exit_now:
                    gross = pos['margin'] * pnl_pct
                    fee = pos['margin'] * lev * FEE_RATE
                    cash += pos['margin'] + gross - fee
                    fees += fee
                    trades.append({'t': t, 'pnl': gross - fee, 'dir': pos['dir']})
                    del positions[sym]
                continue

            if deploy <= 0: continue
            
            # Falling knife check (-10% in last 24h)
            if mode == 'CURRENT' and i >= 96:
                price_24h_ago = df.iloc[i-96]['c']
                if (px - price_24h_ago) / price_24h_ago <= -0.10:
                    allow_long = False

            long_sig = curr['st_dir'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
            short_sig = curr['st_dir'] == -1 and prev['stoch_k'] > 80 and curr['stoch_k'] <= 80

            if long_sig and not allow_long: long_sig = False
            if short_sig and not allow_short: short_sig = False
            if not (long_sig or short_sig): continue
            
            d = 1 if long_sig else -1
            lev = LONG_LEVERAGE if d == 1 else (10 if mode == 'PREVIOUS' else 5)

            equity = cash
            for p in positions.values():
                elev = LONG_LEVERAGE if p['dir'] == 1 else (10 if mode == 'PREVIOUS' else 5)
                equity += p['margin'] * (1 + (p['last_px'] - p['entry']) / p['entry'] * elev * p['dir'])
                
            margin = (equity / MAX_POS) * WEIGHT * NEW_RATIO * deploy
            if margin < 100 or margin > cash: continue
            
            fee = margin * lev * FEE_RATE
            cash -= margin + fee
            fees += fee
            positions[sym] = {'entry': px, 'margin': margin, 'dir': d, 'extreme': 0.0, 'last_px': px}

    equity = cash
    for sym, pos in positions.items():
        lev = LONG_LEVERAGE if pos['dir'] == 1 else (10 if mode == 'PREVIOUS' else 5)
        pnl_pct = (pos['last_px'] - pos['entry']) / pos['entry'] * lev * pos['dir']
        fee = pos['margin'] * lev * FEE_RATE
        trades.append({'t': all_ts[-1], 'pnl': pos['margin'] * pnl_pct - fee, 'dir': pos['dir']})
        equity += pos['margin'] + (pos['margin'] * pnl_pct) - fee

    total = equity - INITIAL_EQUITY
    era_pnls = {}
    for name, s, e in ERAS:
        s_ms = s.replace(tzinfo=timezone.utc).timestamp() * 1000
        e_ms = e.replace(tzinfo=timezone.utc).timestamp() * 1000
        era_pnls[name] = sum(tr['pnl'] for tr in trades if 't' in tr and s_ms <= tr['t'] < e_ms)
    
    shorts = [tr for tr in trades if tr['dir'] == -1]
    return {
        'total': total, 'eras': era_pnls, 'n': len(trades), 'fees': fees,
        'short_n': len(shorts), 'short_pnl': sum(tr['pnl'] for tr in shorts),
        'mdd': mdd
    }

async def main():
    ex = ccxt_async.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    since = int(datetime(2026, 6, 20, tzinfo=timezone.utc).timestamp() * 1000)
    try:
        print("최근 데이터 로드 중...")
        btc = await fetch_all(ex, "BTC/USDT:USDT", "1h", since)
        btc['adx'] = calc_adx(btc, 14)
        btc['ema200'] = btc['c'].ewm(span=200, adjust=False).mean()
        btc['ema50'] = btc['c'].ewm(span=50, adjust=False).mean()
        btc['high14d'] = btc['c'].rolling(min(14 * 24, len(btc))).max()
        btc['dd'] = (btc['high14d'] - btc['c']) / btc['high14d'] * 100

        btc_map = {}
        for i in range(5, len(btc) - 1):
            r = btc.iloc[i]
            r_m5 = btc.iloc[i-5]
            above200 = bool(r['c'] >= r['ema200'])
            bullish = bool(r['c'] >= r['ema200'] and r['ema50'] > r_m5['ema50'])
            btc_map[int(r['t'])] = (float(r['adx']), bullish, float(r['dd']))

        SYMBOLS = [
            "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
            "BNB/USDT:USDT", "DOGE/USDT:USDT", "ZEC/USDT:USDT", "CRV/USDT:USDT"
        ]
        data = {}
        for sym in SYMBOLS:
            try:
                df = await fetch_all(ex, sym, "15m", since)
                if len(df) < 500: continue
                df['st_dir'] = calc_supertrend(df)
                df['stoch_k'] = calc_stoch_k(df['c'])
                df = df.dropna().reset_index(drop=True)
                data[sym] = df
            except Exception as e:
                pass

        print(f"\n[최종 비교] 과거 로직 vs 현재 스마트 로직 백테스트\n")
        variants = [
            ("과거 로직 (매일 리셋)", 'PREVIOUS'),
            ("현재 스마트 로직", 'CURRENT')
        ]
        hdr = f"{'구분':24s} {'전체수익':>10s} {'E1횡보':>10s} {'E2랠리':>10s} {'E3최근폭락':>10s} {'숏손익':>10s} {'MDD%':>6s}"
        print(hdr)
        print("─" * len(hdr))
        for name, mode in variants:
            r = simulate(data, btc_map, mode)
            e = r['eras']
            print(f"  {name:22s} {r['total']:+10.0f} {e['E1 횡보(6/23~7/24)']:+10.0f} {e['E2 랠리(7/25~8/20)']:+10.0f} "
                  f"{e['E3 붕괴(8/21~8/29)']:+10.0f} {r['short_pnl']:+10.0f} {r['mdd']:5.1f}%")
    finally:
        await ex.close()

if __name__ == "__main__":
    asyncio.run(main())
