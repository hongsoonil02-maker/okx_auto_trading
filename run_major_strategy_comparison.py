#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_major_strategy_comparison.py
3대 메이저 (BTC, ETH, SOL) 15m 차트 전략 및 대안 전략 정밀 백테스트 비교
- 데이터: 2026.06.20 ~ 2026.08.26 (6,451 캔들, 약 2.2개월)
- 구간별 분해: E1 횡보장, E2 스케일업, E3 불마켓 랠리, E4 급락 붕괴장
"""
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone

MAJORS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
INITIAL_EQUITY = 10000.0  # 1만불 기준
LEVERAGE = 10
FEE_RATE = 0.0005

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
        h, l, c = df['h'].to_numpy(float), df['l'].to_numpy(float), df['c'].to_numpy(float)
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr = pd_ewm(tr, 14)
        out[sym] = {
            't': t_arr,
            'ix': {int(t): i for i, t in enumerate(t_arr)},
            'st_dir': df['st_dir'].to_numpy(),
            'stoch': df['stoch_k'].to_numpy(),
            'c': c, 'h': h, 'l': l, 'atr': atr,
        }
    return out

def run_simulation(arrs, btc_map, sym_map,
                   use_1h_gate=False,
                   adx_min=0.0,
                   chand_k=None,
                   chand_arm=0.20,
                   allow_breakout=False):
    """
    arrs: BTC, ETH, SOL
    """
    cash = INITIAL_EQUITY
    positions = {}
    trades = []
    eq_peak = eq_trough = INITIAL_EQUITY
    mdd = 0.0

    all_ts = sorted(set().union(*[set(a['t'].tolist()) for a in arrs.values()]))

    for idx, t in enumerate(all_ts):
        if idx < 60:
            continue
        last_h = ((t // 3600000) * 3600000) - 3600000
        _bm = btc_map.get(last_h, (20.0, True))
        adx_now, btc_above50 = _bm[0], _bm[1]

        # 1. 기존 포지션 관리 및 청산 판정
        for sym, a in arrs.items():
            i = a['ix'].get(t)
            if i is None or i < 1:
                continue
            px = a['c'][i]

            pos = positions.get(sym)
            if pos:
                pos['last_px'] = px
                pnl_pct = (px - pos['entry']) / pos['entry'] * LEVERAGE
                # 강제청산 근사치 (-90%)
                if pnl_pct <= -0.90:
                    trades.append({'t': t, 'pnl': -pos['margin'], 'reason': 'liq'})
                    del positions[sym]
                    continue

                pos['extreme'] = max(pos['extreme'], pnl_pct)
                pos['highest'] = max(pos['highest'], a['h'][i])
                best = pos['extreme']

                reason = None
                # 하드 스탑
                if pnl_pct <= -0.20:
                    reason = 'stop_loss'
                # Supertrend 플립 역전 청산
                elif a['st_dir'][i] == -1:
                    reason = 'st_flip'
                # 샹들리에 ATR 트레일링 스탑
                elif chand_k and best >= chand_arm and px < (pos['highest'] - chand_k * a['atr'][i]):
                    reason = 'chandelier'
                # 일반 트레일링 스탑
                elif best > 0.40 and pnl_pct < best * 0.50:
                    reason = 'trail_half'
                elif best > 0.20 and pnl_pct < 0.05:
                    reason = 'trail_breakeven'
                elif pnl_pct >= 0.60:
                    reason = 'tp'

                if reason:
                    gross = pos['margin'] * pnl_pct
                    fee = pos['margin'] * LEVERAGE * FEE_RATE
                    cash += pos['margin'] + gross - fee
                    trades.append({'t': t, 'pnl': gross - fee, 'reason': reason})
                    del positions[sym]
                continue

            # 2. 신규 진입 조건 평가
            sd, sk = a['st_dir'][i], a['stoch'][i]
            sd_p, sk_p = a['st_dir'][i - 1], a['stoch'][i - 1]

            # 롱 풀백 (기본)
            is_pullback = (sd == 1 and sk_p < 20 and sk >= 20)
            # 롱 돌파
            is_breakout = (sd_p == -1 and sd == 1) if allow_breakout else False

            entry_sig = is_pullback or is_breakout

            # ADX 횡보 필터
            if adx_min > 0.0 and adx_now < adx_min:
                entry_sig = False

            # 1h 추세 게이트 (BTC 1h EMA50 + 심볼 1h EMA50)
            if use_1h_gate:
                sym_ok = sym_map.get(sym, {}).get(last_h, True)
                if not (btc_above50 and sym_ok):
                    entry_sig = False

            if entry_sig:
                # 3개 메이저 분산 (1종목당 자본의 30% 배분)
                alloc_ratio = 0.30
                margin = cash * alloc_ratio
                if margin >= 50 and margin <= cash:
                    fee = margin * LEVERAGE * FEE_RATE
                    cash -= (margin + fee)
                    positions[sym] = {
                        'entry': px, 'margin': margin, 'extreme': 0.0,
                        'last_px': px, 'highest': a['h'][i], 'entry_t': t
                    }

        # MDD 트래킹
        equity = sum(p['margin'] * (1 + (p['last_px'] - p['entry']) / p['entry'] * LEVERAGE)
                     for p in positions.values()) + cash
        if equity > eq_peak:
            eq_peak = eq_trough = equity
        elif equity < eq_trough:
            eq_trough = equity
            mdd = max(mdd, (eq_peak - eq_trough) / eq_peak * 100)

    # 잔여 포지션 정산
    equity = cash
    for sym, pos in positions.items():
        pnl_pct = (pos['last_px'] - pos['entry']) / pos['entry'] * LEVERAGE
        pnl = pos['margin'] * pnl_pct - pos['margin'] * LEVERAGE * FEE_RATE
        equity += pos['margin'] + pnl
        trades.append({'t': all_ts[-1], 'pnl': pnl, 'reason': 'close_end'})

    total_pnl = equity - INITIAL_EQUITY
    ret_pct = (total_pnl / INITIAL_EQUITY) * 100

    era_results = {}
    for name, s, e in ERAS:
        s_ms = s.replace(tzinfo=timezone.utc).timestamp() * 1000
        e_ms = e.replace(tzinfo=timezone.utc).timestamp() * 1000
        era_trades = [tr['pnl'] for tr in trades if s_ms <= tr['t'] < e_ms]
        era_results[name] = sum(era_trades) if era_trades else 0.0

    wins = [tr['pnl'] for tr in trades if tr['pnl'] > 0]
    losses = [tr['pnl'] for tr in trades if tr['pnl'] <= 0]
    wr = len(wins) / max(1, len(trades)) * 100
    profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 99.9

    return {
        'equity': equity,
        'total_pnl': total_pnl,
        'ret_pct': ret_pct,
        'wr': wr,
        'pf': profit_factor,
        'trades': len(trades),
        'mdd': mdd,
        'eras': era_results
    }

def main():
    with open("/tmp/lg_data.pkl", "rb") as f:
        btc_map, data, sym_map = pickle.load(f)
    arrs = build_arrays(data)
    print("=" * 85)
    print(" 3대 메이저(BTC, ETH, SOL) 15m 전략 백테스트 비교 분석 ")
    print(f" 기간: 2026.06.20 ~ 08.26 (약 2.2개월, 6,451 캔들) | 초기자본: $10,000 | 레버리지: 10x")
    print("=" * 85)

    configs = [
        ("1. [현행 기준] 15m 단일 (No Gate, No Filter)", False, 0.0, None),
        ("2. [+ 돌파 진입 추가] 15m (풀백 + 돌파)", False, 0.0, None, True),
        ("3. [+ 1h 추세 게이트] BTC & 심볼 1h EMA50 필터", True, 0.0, None),
        ("4. [+ ADX 횡보 필터] ADX < 20 진입 차단", False, 20.0, None),
        ("5. [+ Chandelier Exit] ATR 2.5 익절 트레일링", False, 0.0, 2.5),
        ("6. [통합 최적화] 1h 게이트 + ADX 횡보 필터 + Chandelier Exit", True, 20.0, 2.5),
        ("7. [안정형 통합] 1h 게이트 + ADX 22 + Chandelier 2.0", True, 22.0, 2.0),
    ]

    results = []
    for cfg in configs:
        name = cfg[0]
        gate = cfg[1]
        adx = cfg[2]
        chand = cfg[3]
        bo = cfg[4] if len(cfg) > 4 else False
        res = run_simulation(arrs, btc_map, sym_map,
                             use_1h_gate=gate,
                             adx_min=adx,
                             chand_k=chand,
                             allow_breakout=bo)
        results.append((name, res))

    # 표 출력
    print(f"\n{'전략명':<40s} | {'수익률(%)':>9s} | {'손익($)':>9s} | {'승률':>5s} | {'PF':>5s} | {'MDD%':>6s} | {'매매수':>4s}")
    print("-" * 88)
    for name, r in results:
        print(f"{name:<40s} | {r['ret_pct']:+8.1f}% | {r['total_pnl']:+8.0f}$ | {r['wr']:4.1f}% | {r['pf']:4.2f} | {r['mdd']:5.1f}% | {r['trades']:4d}")

    print("\n" + "=" * 88)
    print(" [국면별 손익 분해] (E1 횡보장 vs E3 랠리장 vs E4 급락 붕괴장)")
    print("=" * 88)
    print(f"{'전략명':<40s} | {'E1 횡보':>10s} | {'E2 스케일':>10s} | {'E3 랠리':>10s} | {'E4 붕괴':>10s}")
    print("-" * 88)
    for name, r in results:
        e = r['eras']
        print(f"{name:<40s} | {e['E1 횡보 (6/23~7/24)']:+10.0f}$ | {e['E2 스케일업 (7/27~8/3)']:+10.0f}$ | {e['E3 랠리 (8/9~8/21 14h)']:+10.0f}$ | {e['E4 붕괴 (8/21 14h~8/26)']:+10.0f}$")

if __name__ == "__main__":
    main()
