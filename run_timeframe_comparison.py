#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_timeframe_comparison.py
3대 메이저(BTC, ETH, SOL) 타임프레임별(15m vs 30m vs 1h) 백테스트 비교
"""
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone

MAJORS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
INITIAL_EQUITY = 10000.0
LEVERAGE = 10
FEE_RATE = 0.0005

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
    agg_dict = {
        't': 'first',
        'o': 'first',
        'h': 'max',
        'l': 'min',
        'c': 'last',
        'v': 'sum'
    }
    res = df.resample(rule).agg(agg_dict).dropna().reset_index(drop=True)
    res['st_dir'] = calc_supertrend(res)
    res['stoch_k'] = calc_stoch_k(res['c'])
    return res.dropna().reset_index(drop=True)

def sim_tf(data_dict, tf_rule='15T', use_adx_filter=True, adx_min=20.0, btc_map=None):
    cash = INITIAL_EQUITY
    positions = {}
    trades = []
    eq_peak = eq_trough = INITIAL_EQUITY
    mdd = 0.0

    all_ts = sorted(set().union(*[set(df['t']) for df in data_dict.values()]))

    for idx, t in enumerate(all_ts):
        if idx < 40:
            continue
        last_h = ((t // 3600000) * 3600000) - 3600000
        _bm = btc_map.get(last_h, (20.0, True)) if btc_map else (20.0, True)
        adx_now = _bm[0]

        for sym, df in data_dict.items():
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
                pnl_pct = (px - pos['entry']) / pos['entry'] * LEVERAGE
                if pnl_pct <= -0.90:
                    trades.append({'t': t, 'pnl': -pos['margin']})
                    del positions[sym]
                    continue
                pos['extreme'] = max(pos['extreme'], pnl_pct)
                best = pos['extreme']

                reason = None
                if pnl_pct <= -0.20:
                    reason = 'sl'
                elif curr['st_dir'] == -1:
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

            # 신규 진입
            long_sig = (curr['st_dir'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20)
            if use_adx_filter and adx_now < adx_min:
                long_sig = False

            if long_sig:
                margin = cash * 0.30
                if margin >= 50 and margin <= cash:
                    fee = margin * LEVERAGE * FEE_RATE
                    cash -= (margin + fee)
                    positions[sym] = {
                        'entry': px, 'margin': margin, 'extreme': 0.0, 'last_px': px
                    }

        equity = sum(p['margin'] * (1 + (p['last_px'] - p['entry']) / p['entry'] * LEVERAGE)
                     for p in positions.values()) + cash
        if equity > eq_peak:
            eq_peak = eq_trough = equity
        elif equity < eq_trough:
            eq_trough = equity
            mdd = max(mdd, (eq_peak - eq_trough) / eq_peak * 100)

    # 잔여 포지션 정리
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

    return {
        'ret_pct': ret_pct,
        'total_pnl': total_pnl,
        'wr': wr,
        'pf': pf,
        'mdd': mdd,
        'trades': len(trades)
    }

def main():
    with open("/tmp/lg_data.pkl", "rb") as f:
        btc_map, data, sym_map = pickle.load(f)

    print("=" * 80)
    print(" [3대 메이저] 타임프레임별 비교 백테스트 (15m vs 30m vs 1h) ")
    print("=" * 80)

    # 15m
    data_15m = {s: data[s] for s in MAJORS}
    # 30m
    data_30m = {s: resample_df(data[s], '30T') for s in MAJORS}
    # 1h
    data_1h = {s: resample_df(data[s], '1H') for s in MAJORS}

    experiments = [
        ("15m (기본 노필터)", data_15m, False, 0.0),
        ("15m (+ ADX 20 필터)", data_15m, True, 20.0),
        ("30m (기본 노필터)", data_30m, False, 0.0),
        ("30m (+ ADX 20 필터)", data_30m, True, 20.0),
        ("1h (기본 노필터)", data_1h, False, 0.0),
        ("1h (+ ADX 20 필터)", data_1h, True, 20.0),
    ]

    print(f"{'설정':<25s} | {'수익률(%)':>9s} | {'손익($)':>9s} | {'승률':>5s} | {'PF':>5s} | {'MDD%':>6s} | {'매매수':>4s}")
    print("-" * 75)
    for label, d, adx_use, adx_val in experiments:
        r = sim_tf(d, use_adx_filter=adx_use, adx_min=adx_val, btc_map=btc_map)
        print(f"{label:<25s} | {r['ret_pct']:+8.1f}% | {r['total_pnl']:+8.0f}$ | {r['wr']:4.1f}% | {r['pf']:4.2f} | {r['mdd']:5.1f}% | {r['trades']:4d}")

if __name__ == "__main__":
    main()
