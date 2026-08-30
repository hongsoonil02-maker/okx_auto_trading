#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_alv_benchmark_comparison.py
alv*** 트레이더 벤치마킹 백테스트 시뮬레이터:
- 데이터: 2026.06.20 ~ 2026.08.26 (BTC, ETH, SOL 15m)
- 비교 대상:
  1. [현행 전략] 넓은 손절 (-20%), 조기 익절 (+50~60%), 롱 전용
  2. [alv*** 벤치마크 1] 타이트 손절 (-8% 컷) + 무제한 트레일링 (상한 TP 해제)
  3. [alv*** 벤치마크 2] 타이트 손절 (-8%) + 샹들리에 ATR 트레일링 + 롱/숏 양방향 추세추종
  4. [alv*** 벤치마크 3 (풀세트)] 실질 레버리지 2.5x 제한 + 타이트 손절 + 대추세 홀딩 + 숏 스위칭
"""
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone

MAJORS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
INITIAL_EQUITY = 10000.0
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

def simulate_strategy(arrs, btc_map, sym_map,
                      leverage=10,
                      hard_sl=-0.20,
                      allow_short=False,
                      tp_cap=0.60,
                      chand_k=None,
                      chand_arm=0.20,
                      alloc_ratio=0.30,
                      use_1h_gate=False,
                      adx_min=0.0):
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

        # 1. 포지션 청산 판정
        for sym, a in arrs.items():
            i = a['ix'].get(t)
            if i is None or i < 1:
                continue
            px = a['c'][i]
            pos = positions.get(sym)

            if pos:
                direction = pos['dir']
                pos['last_px'] = px
                pnl_pct = (px - pos['entry']) / pos['entry'] * leverage * direction

                # 강제청산 근사치 (-90%)
                if pnl_pct <= -0.90:
                    trades.append({'t': t, 'pnl': -pos['margin'], 'reason': 'liq'})
                    del positions[sym]
                    continue

                pos['extreme'] = max(pos['extreme'], pnl_pct)
                if direction == 1:
                    pos['best_px'] = max(pos.get('best_px', px), a['h'][i])
                else:
                    pos['best_px'] = min(pos.get('best_px', px), a['l'][i])
                best = pos['extreme']

                reason = None
                # 칼손절 (Tight Hard SL)
                if pnl_pct <= hard_sl:
                    reason = 'hard_stop'
                # Supertrend 반전
                elif (direction == 1 and a['st_dir'][i] == -1) or (direction == -1 and a['st_dir'][i] == 1):
                    reason = 'st_flip'
                # 샹들리에 ATR 트레일링 스탑
                elif chand_k and best >= chand_arm:
                    if direction == 1 and px < (pos['best_px'] - chand_k * a['atr'][i]):
                        reason = 'chandelier'
                    elif direction == -1 and px > (pos['best_px'] + chand_k * a['atr'][i]):
                        reason = 'chandelier'
                # 일반 트레일링
                elif best > 0.40 and pnl_pct < best * 0.60:
                    reason = 'trail_half'
                elif best > 0.20 and pnl_pct < 0.05:
                    reason = 'trail_breakeven'
                # 고정 익절 상한 (있을 경우만)
                elif tp_cap and pnl_pct >= tp_cap:
                    reason = 'tp_cap'

                if reason:
                    gross = pos['margin'] * pnl_pct
                    fee = pos['margin'] * leverage * FEE_RATE
                    cash += pos['margin'] + gross - fee
                    trades.append({'t': t, 'pnl': gross - fee, 'reason': reason, 'pnl_pct': pnl_pct})
                    del positions[sym]
                continue

            # 2. 신규 진입 조건
            sd, sk = a['st_dir'][i], a['stoch'][i]
            sd_p, sk_p = a['st_dir'][i - 1], a['stoch'][i - 1]

            long_sig = (sd == 1 and sk_p < 20 and sk >= 20)
            short_sig = (sd == -1 and sk_p > 80 and sk <= 80) if allow_short else False

            # ADX 필터
            if adx_min > 0.0:
                _bm = btc_map.get(last_h, (20.0, True))
                if _bm[0] < adx_min:
                    long_sig = False
                    short_sig = False

            # 1h 추세 게이트 (하락장 롱 차단, 상승장 숏 차단)
            if use_1h_gate:
                _bm = btc_map.get(last_h, (20.0, True))
                btc_above50 = _bm[1]
                sym_ok = sym_map.get(sym, {}).get(last_h, True)
                if not (btc_above50 and sym_ok):
                    long_sig = False
                if (btc_above50 or sym_ok):
                    short_sig = False

            entry_dir = 0
            if long_sig:
                entry_dir = 1
            elif short_sig:
                entry_dir = -1

            if entry_dir != 0:
                margin = cash * alloc_ratio
                if margin >= 50 and margin <= cash:
                    fee = margin * leverage * FEE_RATE
                    cash -= (margin + fee)
                    positions[sym] = {
                        'entry': px, 'margin': margin, 'dir': entry_dir,
                        'extreme': 0.0, 'last_px': px, 'best_px': px, 'entry_t': t
                    }

        # MDD 트래킹
        equity = sum(p['margin'] * (1 + (p['last_px'] - p['entry']) / p['entry'] * leverage * p['dir'])
                     for p in positions.values()) + cash
        if equity > eq_peak:
            eq_peak = eq_trough = equity
        elif equity < eq_trough:
            eq_trough = equity
            mdd = max(mdd, (eq_peak - eq_trough) / eq_peak * 100)

    # 잔여 포지션 청산
    equity = cash
    for sym, pos in positions.items():
        pnl_pct = (pos['last_px'] - pos['entry']) / pos['entry'] * leverage * pos['dir']
        pnl = pos['margin'] * pnl_pct - pos['margin'] * leverage * FEE_RATE
        equity += pos['margin'] + pnl
        trades.append({'t': all_ts[-1], 'pnl': pnl, 'reason': 'close_end', 'pnl_pct': pnl_pct})

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

    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 1.0
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    return {
        'equity': equity,
        'total_pnl': total_pnl,
        'ret_pct': ret_pct,
        'wr': wr,
        'pf': profit_factor,
        'pl_ratio': pl_ratio,
        'trades': len(trades),
        'mdd': mdd,
        'eras': era_results
    }

def main():
    with open("/tmp/lg_data.pkl", "rb") as f:
        btc_map, data, sym_map = pickle.load(f)
    arrs = build_arrays(data)

    print("=" * 95)
    print(" [alv*** 트레이더 벤치마킹] vs [현행 전략] 메이저 15m 백테스트 정밀 비교")
    print(" 기간: 2026.06.20 ~ 08.26 (약 2.2개월) | 초기자본: $10,000")
    print("=" * 95)

    configs = [
        # (전략명, leverage, hard_sl, allow_short, tp_cap, chand_k, chand_arm, alloc_ratio, use_1h_gate, adx_min)
        ("1. [현행 방식] 넓은SL(-20%), 조기익절(+60%), 롱전용", 10, -0.20, False, 0.60, None, 0.20, 0.30, False, 0.0),
        ("2. [alv 칼손절 도입] 손절 -8% 컷, 조기익절 유지", 10, -0.08, False, 0.60, None, 0.20, 0.30, False, 0.0),
        ("3. [alv 홈런익절] 손절 -8% + 조기익절 해제(빅스윙 트레일링)", 10, -0.08, False, None, 2.5, 0.20, 0.30, False, 0.0),
        ("4. [alv 횡보필터+빅스윙] 손절 -8% + 빅스윙 + ADX>20", 10, -0.08, False, None, 2.5, 0.20, 0.30, False, 20.0),
        ("5. [alv 1h추세게이트+빅스윙] 손절 -8% + 1h추세 + ADX>20", 10, -0.08, False, None, 2.5, 0.20, 0.30, True, 20.0),
        ("6. [alv 양방향 추세스위칭] 1h추세게이트 + 숏허용 + 빅스윙 + ADX>20", 10, -0.08, True, None, 2.5, 0.20, 0.30, True, 20.0),
        ("7. [alv 실질레버리지 2.5x 모델] 자본 25% 비중 + 1h게이트 + 숏허용 + 빅스윙", 10, -0.08, True, None, 2.5, 0.20, 0.25, True, 20.0),
    ]

    results = []
    for cfg in configs:
        name = cfg[0]
        res = simulate_strategy(arrs, btc_map, sym_map,
                                leverage=cfg[1],
                                hard_sl=cfg[2],
                                allow_short=cfg[3],
                                tp_cap=cfg[4],
                                chand_k=cfg[5],
                                chand_arm=cfg[6],
                                alloc_ratio=cfg[7],
                                use_1h_gate=cfg[8],
                                adx_min=cfg[9])
        results.append((name, res))

    print(f"\n{'전략명':<46s} | {'수익률(%)':>9s} | {'손익($)':>9s} | {'승률':>5s} | {'손익비':>5s} | {'PF':>5s} | {'MDD%':>6s} | {'매매수':>4s}")
    print("-" * 105)
    for name, r in results:
        print(f"{name:<46s} | {r['ret_pct']:+8.1f}% | {r['total_pnl']:+8.0f}$ | {r['wr']:4.1f}% | {r['pl_ratio']:4.2f}:1 | {r['pf']:4.2f} | {r['mdd']:5.1f}% | {r['trades']:4d}")

    print("\n" + "=" * 105)
    print(" [국면별 손익 분해] (E1 횡보 vs E2 스케일 vs E3 랠리 vs E4 붕괴장)")
    print("=" * 105)
    print(f"{'전략명':<46s} | {'E1 횡보':>10s} | {'E2 스케일':>10s} | {'E3 랠리':>10s} | {'E4 붕괴':>10s}")
    print("-" * 105)
    for name, r in results:
        e = r['eras']
        print(f"{name:<46s} | {e['E1 횡보 (6/23~7/24)']:+10.0f}$ | {e['E2 스케일업 (7/27~8/3)']:+10.0f}$ | {e['E3 랠리 (8/9~8/21 14h)']:+10.0f}$ | {e['E4 붕괴 (8/21 14h~8/26)']:+10.0f}$")

if __name__ == "__main__":
    main()
