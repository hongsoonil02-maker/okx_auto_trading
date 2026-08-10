#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bt_current.py — 현재 라이브 로직(strategy_common.BaseStrategyBrain) 재현 백테스트
파라미터 비교: ENTRY_THRESHOLD, VOL_CONFIRM_MULT, 레짐필터 on/off, 타임프레임
수수료: 테이커 0.05% × 레버리지 10x 왕복 = 마진 기준 1.0% 차감/단위
"""
import asyncio
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt_async

LEV = 10
FEE_PER_UNIT = 0.0005 * LEV * 2  # 왕복 수수료 (마진 대비)


def calc_supertrend(df, period=10, multiplier=3.0):
    hl2 = (df['h'] + df['l']) / 2
    atr = (df['h'].combine(df['c'].shift(), max) - df['l'].combine(df['c'].shift(), min)).rolling(period).mean()
    fu = hl2 + multiplier * atr
    fl = hl2 - multiplier * atr
    sd = pd.Series(1, index=df.index, dtype='int')
    sv = pd.Series(0.0, index=df.index, dtype='float64')
    for i in range(period, len(df)):
        if df['c'].iloc[i] > fu.iloc[i-1]:
            sd.iloc[i] = 1
        elif df['c'].iloc[i] < fl.iloc[i-1]:
            sd.iloc[i] = -1
        else:
            sd.iloc[i] = sd.iloc[i-1]
            if sd.iloc[i] == 1 and fl.iloc[i] < fl.iloc[i-1]:
                fl.iloc[i] = fl.iloc[i-1]
            if sd.iloc[i] == -1 and fu.iloc[i] > fu.iloc[i-1]:
                fu.iloc[i] = fu.iloc[i-1]
        sv.iloc[i] = fl.iloc[i] if sd.iloc[i] == 1 else fu.iloc[i]
    return sd, sv


def calc_stoch_k(close, period=14, smooth=3):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    return ((rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())).rolling(smooth).mean() * 100


def prepare(df, ema_period=50):
    df['st_d_l'], df['st_v_l'] = calc_supertrend(df, 10, 3.0)
    df['st_d_t'], df['st_v_t'] = calc_supertrend(df, 10, 2.0)
    df['vol_ma'] = df['v'].rolling(20).mean()
    df['stoch_k'] = calc_stoch_k(df['c'])
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['ema'] = df['c'].ewm(span=ema_period, adjust=False).mean()
    return df


def simulate(df, thr=70, vol_mult=1.0, hard_stop=-0.30, regime_flags=None,
             profit_thr=1.03, max_dca=8, tf_min=30):
    """현재 로직 근사 시뮬. 리턴: episodes 리스트({'pnl': 마진기준 수익률, 'win': bool})"""
    eps = []
    pos = None  # {'side','entry','qty','max_pnl','exit_count','entry_i'}
    for i in range(210, len(df)):
        prev, curr = df.iloc[i-1], df.iloc[i]
        vol_cond = (curr['v'] > prev['vol_ma'] * vol_mult) or \
                   (i >= 2 and prev['v'] > df.iloc[i-2]['vol_ma'] * vol_mult)

        long_score = (50 if curr['c'] > curr['ema'] else 0) \
            + (40 if prev['st_d_l'] == -1 and curr['st_d_l'] == 1 else 0) \
            + (30 if curr['st_d_l'] == 1 and prev['stoch_k'] < 20 <= curr['stoch_k'] else 0) \
            + (20 if curr['st_d_l'] == 1 and prev['st_d_l'] == 1 else 0) \
            + (20 if curr['stoch_k'] > curr['stoch_d'] and curr['stoch_k'] < 80 else 0)
        short_score = (50 if curr['c'] < curr['ema'] else 0) \
            + (40 if prev['st_d_l'] == 1 and curr['st_d_l'] == -1 else 0) \
            + (30 if curr['st_d_l'] == -1 and prev['stoch_k'] > 80 >= curr['stoch_k'] else 0) \
            + (20 if curr['st_d_l'] == -1 and prev['st_d_l'] == -1 else 0) \
            + (20 if curr['stoch_k'] < curr['stoch_d'] and curr['stoch_k'] > 20 else 0)

        regime_ok = True if regime_flags is None else regime_flags[i]
        is_long_sig = long_score >= thr and vol_cond and regime_ok
        is_short_sig = short_score >= thr and vol_cond

        if pos is not None:
            side = pos['side']
            pnl = ((curr['c'] - pos['entry']) / pos['entry'] * LEV) * (1 if side == 'L' else -1)
            pos['max_pnl'] = max(pos['max_pnl'], pnl)
            ec = pos['exit_count']
            full_close = False
            scale_out = False

            is_profit = (curr['c'] > pos['entry'] * profit_thr) if side == 'L' else (curr['c'] < pos['entry'] * (2 - profit_thr))
            st_d = curr['st_d_t'] if is_profit else curr['st_d_l']
            st_v = curr['st_v_t'] if is_profit else curr['st_v_l']
            st_close = (st_d == -1 or curr['c'] < st_v) if side == 'L' else (st_d == 1 or curr['c'] > st_v)

            if pnl <= hard_stop:
                full_close = True
            elif pos['max_pnl'] >= 0.40 and pnl <= 0.20:
                full_close = True
            elif pos['max_pnl'] >= 0.20 and pnl <= 0.0:
                full_close = True
            elif (pnl >= 0.15 and ec == 0) or (pnl >= 0.30 and ec == 1) or (pnl >= 0.50 and ec == 2):
                scale_out = True
            elif st_close:
                scale_out = True

            if not full_close and not scale_out and ec > 0:
                held = (i - pos['entry_i'])
                bad = curr['c'] < pos['entry'] if side == 'L' else curr['c'] > pos['entry']
                if bad and held >= 3:
                    full_close = True

            if full_close:
                frac = pos['qty']
                pos['realized'] += frac * (pnl - FEE_PER_UNIT)
                eps.append({'pnl': pos['realized'], 'win': pos['realized'] > 0})
                pos = None
            elif scale_out and ec < max_dca:
                frac = pos['qty'] / (max_dca - ec)
                pos['qty'] -= frac
                pos['exit_count'] += 1
                pos['realized'] += frac * (pnl - FEE_PER_UNIT)
        else:
            if is_long_sig:
                pos = {'side': 'L', 'entry': curr['c'], 'qty': 1.0, 'max_pnl': 0.0,
                       'exit_count': 0, 'entry_i': i, 'realized': 0.0}
            elif is_short_sig:
                pos = {'side': 'S', 'entry': curr['c'], 'qty': 1.0, 'max_pnl': 0.0,
                       'exit_count': 0, 'entry_i': i, 'realized': 0.0}
    if pos is not None:
        pnl = ((df.iloc[-1]['c'] - pos['entry']) / pos['entry'] * LEV) * (1 if pos['side'] == 'L' else -1)
        eps.append({'pnl': pos['realized'] + pos['qty'] * (pnl - FEE_PER_UNIT), 'win': pnl > 0})
    return eps


def report(name, eps):
    if not eps:
        print(f"  {name:<28} | trades  0")
        return {'n': 0, 'ret': 0, 'wr': 0}
    pnls = [e['pnl'] for e in eps]
    wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
    ret = sum(pnls) * 100
    avg = np.mean(pnls) * 100
    print(f"  {name:<28} | trades {len(eps):>3} | WR {wr:5.1f}% | avg {avg:+6.2f}% | total {ret:+8.2f}%")
    return {'n': len(eps), 'ret': ret, 'wr': wr}


async def main():
    ex = ccxt_async.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

    # BTC 1h 레짐 플래그
    btc1h = await ex.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=2000)
    bdf = pd.DataFrame(btc1h, columns=['t', 'o', 'h', 'l', 'c', 'v'])
    bdf['ema200'] = bdf['c'].ewm(span=200, adjust=False).mean()
    bdf['ok'] = bdf['c'] >= bdf['ema200']
    btc_regime = bdf.set_index('t')['ok']

    def flags_for(df):
        idx = np.searchsorted(btc_regime.index.values, df['t'].values, side='right') - 1
        flags = np.ones(len(df), dtype=bool)
        valid = idx >= 0
        flags[valid] = btc_regime.values[idx[valid]]
        return flags

    groups = {
        'Major 30m': (['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT'], '30m'),
        'Major 1h': (['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT'], '1h'),
        'Venture 15m': (['DOGE/USDT:USDT', 'PEPE/USDT:USDT', 'ONDO/USDT:USDT'], '15m'),
        'Stock 15m': (['TSLA/USDT:USDT', 'NVDA/USDT:USDT'], '15m'),
        'StockVenture 15m': (['SPCX/USDT:USDT', 'OPENAI/USDT:USDT'], '15m'),
    }

    configs = [
        ('현재(70/1.0/레짐ON)', dict(thr=70, vol_mult=1.0, regime=True)),
        ('이전(80/1.2/레짐OFF)', dict(thr=80, vol_mult=1.2, regime=False)),
        ('레짐OFF(70/1.0)', dict(thr=70, vol_mult=1.0, regime=False)),
        ('완화MAX(60/0.8/레짐ON)', dict(thr=60, vol_mult=0.8, regime=True)),
    ]

    for gname, (syms, tf) in groups.items():
        print(f"\n=== {gname} ===")
        dfs = {}
        for s in syms:
            try:
                o = await ex.fetch_ohlcv(s, tf, limit=2000)
                if o and len(o) >= 300:
                    d = pd.DataFrame(o, columns=['t', 'o', 'h', 'l', 'c', 'v'])
                    dfs[s] = (prepare(d), flags_for(d))
            except Exception as e:
                print(f"  {s}: fetch 실패 {e}")
            await asyncio.sleep(0.3)
        for cname, cfg in configs:
            all_eps = []
            for s, (d, fl) in dfs.items():
                all_eps += simulate(d, thr=cfg['thr'], vol_mult=cfg['vol_mult'],
                                    regime_flags=fl if cfg['regime'] else None,
                                    tf_min=30 if tf == '30m' else 15)
            report(cname, all_eps)

    await ex.close()

if __name__ == '__main__':
    asyncio.run(main())
