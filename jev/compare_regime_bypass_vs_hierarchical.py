#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_regime_bypass_vs_hierarchical.py
Direct Backtest Comparison:
Mode A (Bypass Mode): Current strategy_common.py approach where JEV active bypasses 1h trend gate & chop block.
Mode B (Hierarchical Mode): 1h Trend Gate + ADX Chop Block maintained as prerequisite, JEV used as final execution confirmation.
Mode C (Baseline): No gate, standard 15m Supertrend.
Mode D (1h Gate only): 1h EMA50 trend gate without JEV.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MAJORS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
INITIAL_EQUITY = 10000.0
LEVERAGE = 10
TAKER_FEE_RATE = 0.0005
MAKER_FEE_RATE = 0.0002
SLIPPAGE_RATE  = 0.0003

ERAS = [
    ("E1 횡보 (6/23~7/24)", datetime(2026, 6, 23), datetime(2026, 7, 24)),
    ("E2 스케일업 (7/27~8/3)", datetime(2026, 7, 27), datetime(2026, 8, 3)),
    ("E3 랠리 (8/9~8/21 14h)", datetime(2026, 8, 9), datetime(2026, 8, 21, 14)),
    ("E4 붕괴 (8/21 14h~8/26)", datetime(2026, 8, 21, 14), datetime(2026, 8, 26, 10)),
]

def pd_ewm(arr, span):
    alpha = 1.0 / span
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out

def build_arrays(data):
    out = {}
    for sym in MAJORS:
        if sym not in data:
            continue
        df = data[sym]
        t_arr = df['t'].to_numpy(dtype=np.int64)
        h = df['h'].to_numpy(float)
        l = df['l'].to_numpy(float)
        c = df['c'].to_numpy(float)
        o = df['o'].to_numpy(float) if 'o' in df.columns else np.roll(c, 1)

        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr = pd_ewm(tr, 14)

        rng = np.maximum(h - l, 1e-6)
        close_pos = (c - l) / rng
        prob_up_proxy = np.clip(0.40 + 0.35 * close_pos + 0.15 * ((c - o) / rng), 0.10, 0.95)

        out[sym] = {
            't': t_arr,
            'ix': {int(t): i for i, t in enumerate(t_arr)},
            'st_dir': df['st_dir'].to_numpy(),
            'stoch': df['stoch_k'].to_numpy(),
            'c': c, 'h': h, 'l': l, 'o': o,
            'atr': atr,
            'prob_up': prob_up_proxy,
        }
    return out

def run_simulation(
    arrs,
    btc_map,
    sym_map,
    mode: str, # "baseline", "gate_only", "jev_bypass", "jev_hierarchical"
    use_post_only: bool = True,
    use_chandelier: bool = True,
    jev_threshold: float = 0.65,
):
    cash = INITIAL_EQUITY
    positions = {}
    trades = []
    eq_peak = eq_trough = INITIAL_EQUITY
    mdd = 0.0
    filtered_out_count = 0
    fees_saved = 0.0

    all_ts = sorted(set().union(*[set(a['t'].tolist()) for a in arrs.values()]))

    for idx, t in enumerate(all_ts):
        if idx < 60:
            continue
        last_h = ((t // 3600000) * 3600000) - 3600000
        _bm = btc_map.get(last_h, (20.0, True))
        adx_now, btc_above50 = _bm[0], _bm[1]

        # 1. Position management
        for sym, a in list(arrs.items()):
            i = a['ix'].get(t)
            if i is None or i == 0:
                continue
            px = a['c'][i]

            if sym in positions:
                pos = positions[sym]
                pos['highest'] = max(pos['highest'], a['h'][i])
                pnl_pct = ((px - pos['entry']) / pos['entry']) * LEVERAGE
                reason = None

                # Hard SL
                if pnl_pct <= -0.30:
                    reason = "hard_sl"
                elif use_chandelier and pos['highest'] > pos['entry']:
                    # Chandelier ATR trailing stop
                    atr_val = a['atr'][i]
                    trail_stop = pos['highest'] - 3.5 * atr_val
                    unrealized_ret = (pos['highest'] - pos['entry']) / pos['entry'] * LEVERAGE
                    if unrealized_ret >= 0.35 and px < trail_stop:
                        reason = "chandelier_trailing"
                elif a['st_dir'][i] == -1:
                    reason = "supertrend_flip"

                if reason:
                    gross = pos['margin'] * pnl_pct
                    exit_fee = pos['margin'] * LEVERAGE * (TAKER_FEE_RATE + SLIPPAGE_RATE)
                    net_trade_pnl = gross - exit_fee
                    cash += pos['margin'] + net_trade_pnl
                    trades.append({'t': t, 'pnl': net_trade_pnl, 'reason': reason})
                    del positions[sym]
                continue

            # 2. Entry signal evaluation
            sd, sk = a['st_dir'][i], a['stoch'][i]
            sd_p, sk_p = a['st_dir'][i - 1], a['stoch'][i - 1]
            is_pullback = (sd == 1 and sk_p < 20 and sk >= 20)

            sym_ok = sym_map.get(sym, {}).get(last_h, True)
            gate_1h_ok = (btc_above50 and sym_ok)
            chop_ok = (adx_now >= 20.0)
            prob_up = a['prob_up'][i]
            jev_ok = (prob_up >= jev_threshold)

            entry_sig = False

            if mode == "baseline":
                # No gate, standard 15m pullback
                entry_sig = is_pullback
            elif mode == "gate_only":
                # 1h Trend Gate + ADX Chop, no Jev
                entry_sig = is_pullback and gate_1h_ok and chop_ok
            elif mode == "jev_hierarchical":
                # Hierarchical: 1h Gate + ADX Chop MUST be ok, Jev confirms
                if is_pullback and gate_1h_ok and chop_ok:
                    if jev_ok:
                        entry_sig = True
                    else:
                        filtered_out_count += 1
            elif mode == "jev_bypass":
                # Current strategy_common: JEV is 1st-class engine -> bypasses 1h gate and chop block!
                # Even if gate_1h_ok is false or chop_ok is false, Jev can approve!
                # Entry triggers on pullback OR micro-momentum (prob_up >= threshold)
                if (is_pullback or prob_up >= 0.60):
                    if jev_ok:
                        entry_sig = True
                    else:
                        filtered_out_count += 1

            if entry_sig:
                alloc_ratio = 0.30
                margin = cash * alloc_ratio
                if margin >= 50 and margin <= cash:
                    if use_post_only and "jev" in mode:
                        fee_rate = MAKER_FEE_RATE
                        saved = margin * LEVERAGE * (TAKER_FEE_RATE + SLIPPAGE_RATE - MAKER_FEE_RATE)
                        fees_saved += saved
                        entry_px = px * 0.9998
                    else:
                        fee_rate = TAKER_FEE_RATE + SLIPPAGE_RATE
                        entry_px = px

                    fee = margin * LEVERAGE * fee_rate
                    cash -= (margin + fee)
                    positions[sym] = {
                        'entry': entry_px,
                        'margin': margin,
                        'extreme': 0.0,
                        'last_px': px,
                        'highest': a['h'][i],
                    }

        # Track Equity & MDD
        u_pnl = sum([((a['c'][a['ix'][t]] - p['entry']) / p['entry']) * LEVERAGE * p['margin'] for s, p in positions.items() if t in a['ix']])
        equity = cash + sum(p['margin'] for p in positions.values()) + u_pnl
        if equity > eq_peak:
            eq_peak = equity
            eq_trough = equity
        if equity < eq_trough:
            eq_trough = equity
            mdd = max(mdd, (eq_peak - eq_trough) / eq_peak * 100.0)

    total_pnl = cash - INITIAL_EQUITY
    ret_pct = (total_pnl / INITIAL_EQUITY) * 100.0

    wins = [x for x in trades if x['pnl'] > 0]
    losses = [x for x in trades if x['pnl'] <= 0]
    wr = (len(wins) / len(trades) * 100.0) if trades else 0.0
    tot_win = sum([x['pnl'] for x in wins])
    tot_loss = abs(sum([x['pnl'] for x in losses]))
    pf = (tot_win / tot_loss) if tot_loss > 0 else (99.0 if tot_win > 0 else 0.0)

    # Era breakdown
    era_pnl = {}
    for name, s_dt, e_dt in ERAS:
        s_ms = int(s_dt.timestamp() * 1000)
        e_ms = int(e_dt.timestamp() * 1000)
        ep = sum([x['pnl'] for x in trades if s_ms <= x['t'] < e_ms])
        era_pnl[name] = ep

    return {
        'ret_pct': ret_pct,
        'total_pnl': total_pnl,
        'wr': wr,
        'pf': pf,
        'mdd': mdd,
        'trades': len(trades),
        'filtered_out': filtered_out_count,
        'fees_saved': fees_saved,
        'eras': era_pnl,
    }

def main():
    cache_path = "/tmp/lg_data.pkl"
    with open(cache_path, "rb") as f:
        btc_map, data, sym_map = pickle.load(f)

    arrs = build_arrays(data)

    modes = [
        ("1. [기준 봇] Supertrend 15m (No Gate, Taker)", "baseline"),
        ("2. [추세 게이트만] 1h EMA50 + ADX 횡보 필터", "gate_only"),
        ("3. [현행 실코드 방식] Jev 바이패스 모드 (추세/횡보 무시, Jev 1순위)", "jev_bypass"),
        ("4. [제안 방식] Jev 계층적 게이트 (1h 추세/횡보 필수 통과 + Jev 확인)", "jev_hierarchical"),
    ]

    results = []
    for title, m in modes:
        res = run_simulation(arrs, btc_map, sym_map, mode=m, use_post_only=True, use_chandelier=True)
        results.append((title, res))

    print("\n" + "=" * 105)
    print(" 🔬 [정밀 백테스트 비교 분석] JEV 바이패스 모드 vs JEV 계층적 게이트 모드")
    print(" 기간: 2026.06.20 ~ 08.26 (6,451 캔들) | 대상: BTC, ETH, SOL | 자본: $10,000 | 레버리지: 10x")
    print("=" * 105)

    print(f"\n{'전략 시나리오':<50} | {'수익률(%)':<9} | {'순익($)':<10} | {'승률':<6} | {'PF':<5} | {'MDD%':<6} | {'매매수':<6} | {'차단건수'}")
    print("-" * 110)
    for title, r in results:
        print(
            f"{title:<48} | "
            f"{r['ret_pct']:+8.1f}% | "
            f"{r['total_pnl']:+9.0f}$ | "
            f"{r['wr']:5.1f}% | "
            f"{r['pf']:4.2f} | "
            f"{r['mdd']:5.1f}% | "
            f"{r['trades']:6d} | "
            f"{r['filtered_out']:4d}건"
        )

    print("\n" + "=" * 105)
    print(" 📊 [국면별 손익 분해] (E1 횡보장 vs E2 스케일업 vs E3 불장 랠리 vs E4 급락 붕괴장)")
    print("=" * 105)
    print(f"{'전략 시나리오':<48} | {'E1 횡보':<11} | {'E2 스케일':<11} | {'E3 랠리':<11} | {'E4 붕괴'}")
    print("-" * 105)
    for title, r in results:
        e = r['eras']
        print(
            f"{title:<46} | "
            f"{e['E1 횡보 (6/23~7/24)']:+10.0f}$ | "
            f"{e['E2 스케일업 (7/27~8/3)']:+10.0f}$ | "
            f"{e['E3 랠리 (8/9~8/21 14h)']:+10.0f}$ | "
            f"{e['E4 붕괴 (8/21 14h~8/26)']:+10.0f}$"
        )
    print("=" * 105 + "\n")

if __name__ == "__main__":
    main()
