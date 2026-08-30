#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_venture_tf_comparison.py — 고속 배열 기반 벤처 10종목 타임프레임별 비교
"""
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone

VENTURE_SYMBOLS = [
    "DOGE/USDT:USDT", "PEPE/USDT:USDT", "ZEC/USDT:USDT", "CRV/USDT:USDT",
    "NEAR/USDT:USDT", "HYPE/USDT:USDT", "AAVE/USDT:USDT", "UNI/USDT:USDT",
    "TAO/USDT:USDT", "LIT/USDT:USDT"
]

INITIAL_EQUITY = 10000.0
LEVERAGE = 5
FEE_RATE = 0.0005
MAX_POS = 8

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

def resample_df(df_15m, rule):
    df = df_15m.copy()
    df['dt'] = pd.to_datetime(df['t'], unit='ms')
    df = df.set_index('dt')
    agg_dict = {'t': 'first', 'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'v': 'sum'}
    res = df.resample(rule).agg(agg_dict).dropna().reset_index(drop=True)
    res['st_dir'] = calc_supertrend(res)
    res['stoch_k'] = calc_stoch_k(res['c'])
    return res.dropna().reset_index(drop=True)

def build_arrays(data):
    out = {}
    for sym in VENTURE_SYMBOLS:
        if sym not in data:
            continue
        df = data[sym]
        t_arr = df['t'].to_numpy(dtype=np.int64)
        h, l, c = df['h'].to_numpy(float), df['l'].to_numpy(float), df['c'].to_numpy(float)
        out[sym] = {
            't': t_arr,
            'ix': {int(t): i for i, t in enumerate(t_arr)},
            'st_dir': df['st_dir'].to_numpy(),
            'stoch': df['stoch_k'].to_numpy(),
            'c': c, 'h': h, 'l': l,
        }
    return out

def sim_fast(arrs, btc_map, adx_min=0.0):
    cash = INITIAL_EQUITY
    positions = {}
    trades = []
    eq_peak = eq_trough = INITIAL_EQUITY
    mdd = 0.0

    all_ts = sorted(set().union(*[set(a['t'].tolist()) for a in arrs.values()]))

    for idx, t in enumerate(all_ts):
        if idx < 40:
            continue
        last_h = ((t // 3600000) * 3600000) - 3600000
        _bm = btc_map.get(last_h, (20.0, True))
        adx_now = _bm[0]

        for sym, a in arrs.items():
            i = a['ix'].get(t)
            if i is None or i < 1:
                continue
            px = a['c'][i]

            pos = positions.get(sym)
            if pos:
                pos['last_px'] = px
                pnl_pct = (px - pos['entry']) / pos['entry'] * LEVERAGE
                if pnl_pct <= -0.90:
                    trades.append({'t': t, 'pnl': -pos['margin']})
                    del positions[sym]
                    continue
                pos['extreme'] = max(pos['extreme'], pnl_pct)
                best = pos['extreme']

                reason = None
                if pnl_pct <= -0.15:
                    reason = 'sl'
                elif a['st_dir'][i] == -1:
                    reason = 'st_flip'
                elif best > 0.40 and pnl_pct < best * 0.50:
                    reason = 'trail'
                elif best > 0.20 and pnl_pct < 0.05:
                    reason = 'trail'
                elif pnl_pct >= 0.60:
                    reason = 'tp'

                if reason:
                    gross = pos['margin'] * pnl_pct
                    fee = pos['margin'] * LEVERAGE * FEE_RATE
                    cash += pos['margin'] + gross - fee
                    trades.append({'t': t, 'pnl': gross - fee})
                    del positions[sym]
                continue

            if len(positions) >= MAX_POS:
                continue

            sd, sk = a['st_dir'][i], a['stoch'][i]
            sd_p, sk_p = a['st_dir'][i - 1], a['stoch'][i - 1]

            long_sig = (sd == 1 and sk_p < 20 and sk >= 20)
            if adx_min > 0.0 and adx_now < adx_min:
                long_sig = False

            if long_sig:
                margin = (cash / MAX_POS) * 1.2
                margin = min(margin, cash * 0.20)
                if margin >= 50 and margin <= cash:
                    fee = margin * LEVERAGE * FEE_RATE
                    cash -= (margin + fee)
                    positions[sym] = {'entry': px, 'margin': margin, 'extreme': 0.0, 'last_px': px}

        equity = sum(p['margin'] * (1 + (p['last_px'] - p['entry']) / p['entry'] * LEVERAGE)
                     for p in positions.values()) + cash
        if equity > eq_peak:
            eq_peak = eq_trough = equity
        elif equity < eq_trough:
            eq_trough = equity
            mdd = max(mdd, (eq_peak - eq_trough) / eq_peak * 100)

    equity = cash
    for sym, pos in positions.items():
        pnl_pct = (pos['last_px'] - pos['entry']) / pos['entry'] * LEVERAGE
        pnl = pos['margin'] * pnl_pct - pos['margin'] * LEVERAGE * FEE_RATE
        equity += pos['margin'] + pnl
        trades.append({'t': all_ts[-1], 'pnl': pnl})

    total_pnl = equity - INITIAL_EQUITY
    ret_pct = (total_pnl / INITIAL_EQUITY) * 100
    wins = [tr['pnl'] for tr in trades if tr['pnl'] > 0]
    losses = [tr['pnl'] for tr in trades if tr['pnl'] <= 0]
    wr = len(wins) / max(1, len(trades)) * 100
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 99.9

    return {'ret_pct': ret_pct, 'total_pnl': total_pnl, 'wr': wr, 'pf': pf, 'mdd': mdd, 'trades': len(trades)}

def main():
    with open("/tmp/lg_data.pkl", "rb") as f:
        btc_map, data, sym_map = pickle.load(f)

    data_15m = {s: data[s] for s in VENTURE_SYMBOLS if s in data}
    data_30m = {s: resample_df(data[s], '30min') for s in VENTURE_SYMBOLS if s in data}
    data_1h = {s: resample_df(data[s], '1h') for s in VENTURE_SYMBOLS if s in data}

    arrs_15m = build_arrays(data_15m)
    arrs_30m = build_arrays(data_30m)
    arrs_1h = build_arrays(data_1h)

    exps = [
        ("15m (기본 노필터)", arrs_15m, 0.0),
        ("15m (+ ADX 20 필터)", arrs_15m, 20.0),
        ("15m (+ ADX 22 필터)", arrs_15m, 22.0),
        ("30m (기본 노필터)", arrs_30m, 0.0),
        ("30m (+ ADX 20 필터)", arrs_30m, 20.0),
        ("30m (+ ADX 22 필터)", arrs_30m, 22.0),
        ("1h (기본 노필터)", arrs_1h, 0.0),
        ("1h (+ ADX 20 필터)", arrs_1h, 20.0),
    ]

    print("=" * 80)
    print(" [Venture 10종목] 타임프레임별 백테스트 비교 (15m vs 30m vs 1h) ")
    print("=" * 80)
    print(f"{'설정':<25s} | {'수익률(%)':>9s} | {'손익($)':>9s} | {'승률':>5s} | {'PF':>5s} | {'MDD%':>6s} | {'매매수':>4s}")
    print("-" * 75)
    for label, arrs, adx_val in exps:
        r = sim_fast(arrs, btc_map=btc_map, adx_min=adx_val)
        print(f"{label:<25s} | {r['ret_pct']:+8.1f}% | {r['total_pnl']:+8.0f}$ | {r['wr']:4.1f}% | {r['pf']:4.2f} | {r['mdd']:5.1f}% | {r['trades']:4d}")

if __name__ == "__main__":
    main()
