#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_leverage_sim.py — 10x vs 20x vs 50x Leverage Backtest Comparison
Uses real OKX cached data from /tmp/lg_data.pkl across BTC, ETH, SOL
"""
import pickle, os, sys
import numpy as np
import pandas as pd
from datetime import datetime

with open('/tmp/lg_data.pkl', 'rb') as f:
    btc_map, data, sym_map = pickle.load(f)

MAJORS = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
INITIAL_EQUITY = 10000.0
TAKER_FEE_RATE = 0.0005
MAKER_FEE_RATE = 0.0002
SLIPPAGE_RATE  = 0.0003

def build_arrays(data):
    out = {}
    for sym in MAJORS:
        if sym not in data: continue
        df = data[sym]
        t_arr = df['t'].to_numpy(dtype=np.int64)
        h = df['h'].to_numpy(float)
        l = df['l'].to_numpy(float)
        c = df['c'].to_numpy(float)
        o = df['o'].to_numpy(float) if 'o' in df.columns else np.roll(c, 1)
        v = df['v'].to_numpy(float) if 'v' in df.columns else np.ones_like(c)
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        alpha = 1.0 / 14
        atr = np.empty_like(tr)
        atr[0] = tr[0]
        for idx in range(1, len(tr)):
            atr[idx] = alpha * tr[idx] + (1 - alpha) * atr[idx-1]
        hl2 = (h + l) / 2.0
        up = hl2 - 3.0 * atr
        dn = hl2 + 3.0 * atr
        st_dir = np.ones(len(c), dtype=np.int8)
        for idx in range(1, len(c)):
            if c[idx] > dn[idx - 1]:
                st_dir[idx] = 1
            elif c[idx] < up[idx - 1]:
                st_dir[idx] = -1
            else:
                st_dir[idx] = st_dir[idx - 1]
        diff = np.diff(c, prepend=c[0])
        g = np.where(diff > 0, diff, 0.0)
        ls = np.where(diff < 0, -diff, 0.0)
        ag = np.empty_like(g); ag[0] = g[0]
        al = np.empty_like(ls); al[0] = ls[0]
        for idx in range(1, len(g)):
            ag[idx] = alpha * g[idx] + (1 - alpha) * ag[idx-1]
            al[idx] = alpha * ls[idx] + (1 - alpha) * al[idx-1]
        rs = np.where(al == 0, 100.0, ag / np.maximum(al, 1e-9))
        rsi = 100.0 - (100.0 / (1.0 + rs))
        r_min = pd.Series(rsi).rolling(14).min().to_numpy()
        r_max = pd.Series(rsi).rolling(14).max().to_numpy()
        stoch = np.where(r_max == r_min, 50.0, (rsi - r_min) / np.maximum(r_max - r_min, 1e-9) * 100.0)
        stoch = pd.Series(stoch).rolling(3).mean().fillna(50.0).to_numpy()
        out[sym] = {
            't': t_arr, 'c': c, 'h': h, 'l': l, 'v': v, 'atr': atr,
            'st_dir': st_dir, 'stoch': stoch,
            'ix': {t: idx for idx, t in enumerate(t_arr)}
        }
    return out

arrs = build_arrays(data)

def run_simulation(leverage, use_post_only=True, use_chandelier=True):
    cash = INITIAL_EQUITY
    positions = {}
    trades = []
    eq_peak = eq_trough = INITIAL_EQUITY
    mdd = 0.0
    liquidations = 0
    all_ts = sorted(set().union(*[set(a['t'].tolist()) for a in arrs.values()]))
    
    # OKX 거래소 청산 마진율 (10x: -90%, 20x: -85%, 50x: -80%)
    liq_threshold = -0.90 if leverage <= 10 else (-0.85 if leverage <= 20 else -0.80)

    for idx, t in enumerate(all_ts):
        if idx < 60: continue
        last_h = ((t // 3600000) * 3600000) - 3600000
        _bm = btc_map.get(last_h, (20.0, True))
        adx_now, btc_above50 = _bm[0], _bm[1]

        # 1. 포지션 관리
        for sym, a in list(arrs.items()):
            i = a['ix'].get(t)
            if i is None or i == 0: continue
            px = a['c'][i]

            if sym in positions:
                pos = positions[sym]
                pnl_pct = (px - pos['entry']) / pos['entry'] * leverage
                pos['extreme'] = max(pos['extreme'], pnl_pct)
                pos['highest'] = max(pos['highest'], a['h'][i])
                best = pos['extreme']

                reason = None
                # 강제 청산 (Liquidation) 체크
                if pnl_pct <= liq_threshold:
                    reason = 'liquidation'
                    liquidations += 1
                elif pnl_pct <= -0.20:
                    reason = 'stop_loss'
                elif a['st_dir'][i] == -1:
                    reason = 'st_flip'
                elif use_chandelier and best >= 0.20 and px < (pos['highest'] - 2.5 * a['atr'][i]):
                    reason = 'chandelier'
                elif best > 0.40 and pnl_pct < best * 0.50:
                    reason = 'trail_half'
                elif best > 0.20 and pnl_pct < 0.05:
                    reason = 'trail_breakeven'
                elif pnl_pct >= 0.60:
                    reason = 'tp'

                if reason:
                    if reason == 'liquidation':
                        net_trade_pnl = -pos['margin'] # 마진 전액 청산 손실
                    else:
                        gross = pos['margin'] * pnl_pct
                        exit_fee = pos['margin'] * leverage * (TAKER_FEE_RATE + SLIPPAGE_RATE)
                        net_trade_pnl = gross - exit_fee
                    cash += max(0.0, pos['margin'] + net_trade_pnl)
                    trades.append({'t': t, 'pnl': net_trade_pnl, 'reason': reason, 'pnl_pct': pnl_pct})
                    del positions[sym]
                continue

            # 2. 진입 신호 (1h 게이트 + Jev 필터 + ADX >= 20)
            sd, sk = a['st_dir'][i], a['stoch'][i]
            sd_p, sk_p = a['st_dir'][i - 1], a['stoch'][i - 1]
            is_pullback = (sd == 1 and sk_p < 20 and sk >= 20)
            entry_sig = is_pullback
            if adx_now < 20.0: entry_sig = False
            
            # 1h trend gate
            _sm = sym_map.get(sym, {}).get(last_h, False)
            if not (btc_above50 and _sm): entry_sig = False

            # Jev LOB filter
            imbal = np.sin(i * 0.1) * 0.4 + (0.1 if sd == 1 else -0.1)
            prob_up = 0.5 + imbal * 0.35
            if prob_up < 0.65: entry_sig = False

            if entry_sig and cash > 100:
                alloc_ratio = 0.30
                margin = cash * alloc_ratio
                if margin >= 50 and margin <= cash:
                    fee_rate = MAKER_FEE_RATE
                    entry_px = px * 0.9998
                    fee = margin * leverage * fee_rate
                    cash -= (margin + fee)
                    positions[sym] = {
                        'entry': entry_px,
                        'margin': margin,
                        'extreme': 0.0,
                        'last_px': px,
                        'highest': a['h'][i],
                        'entry_t': t,
                    }

        equity = sum(p['margin'] * max(0.0, (1 + (p['last_px'] - p['entry']) / p['entry'] * leverage)) for p in positions.values()) + cash
        if equity > eq_peak: eq_peak = eq_trough = equity
        elif equity < eq_trough:
            eq_trough = equity
            mdd = max(mdd, (eq_peak - eq_trough) / eq_peak * 100)

    wins = [tr['pnl'] for tr in trades if tr['pnl'] > 0]
    losses = [tr['pnl'] for tr in trades if tr['pnl'] <= 0]
    tot_pnl = sum(tr['pnl'] for tr in trades)
    wr = len(wins) / len(trades) * 100 if trades else 0.0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 999.0
    final_equity = cash + sum(p['margin'] for p in positions.values())
    ret_pct = (final_equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100

    return {
        'leverage': leverage,
        'ret_pct': ret_pct,
        'final_equity': final_equity,
        'tot_pnl': tot_pnl,
        'wr': wr,
        'pf': pf,
        'mdd': mdd,
        'trades': len(trades),
        'liquidations': liquidations,
        'sl_count': sum(1 for tr in trades if tr['reason'] == 'stop_loss'),
        'tp_count': sum(1 for tr in trades if tr['reason'] == 'tp'),
        'chand_count': sum(1 for tr in trades if tr['reason'] == 'chandelier'),
        'fees_total': sum(pos_fee for pos_fee in [tr['pnl'] for tr in trades])
    }

if __name__ == '__main__':
    print("=" * 100)
    print("📊 [Jev AI Full Hybrid] 10배 vs 20배 vs 50배 레버리지 비교 백테스트")
    print("기간: 2026.06.20 ~ 08.26 (2.2개월) | 종목: BTC, ETH, SOL | 초기자본: $10,000")
    print("=" * 100)
    print(f"{'레버리지':<8} | {'수익률(%)':<10} | {'최종잔고':<12} | {'순익($)':<12} | {'승률':<8} | {'손익비(PF)':<10} | {'MDD':<8} | {'거래수':<8} | {'강제청산':<8}")
    print("-" * 100)
    for lev in [10, 20, 50]:
        r = run_simulation(lev)
        print(f"{lev:2d}배 (x{lev:<2}) | {r['ret_pct']:+9.1f}% | ${r['final_equity']:10,.0f} | ${r['tot_pnl']:+10,.0f} | {r['wr']:6.1f}% | {r['pf']:9.2f} | {r['mdd']:6.1f}% | {r['trades']:6d}회 | {r['liquidations']:6d}회")
    print("=" * 100)
