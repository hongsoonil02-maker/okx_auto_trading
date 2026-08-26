#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_lev_sim.py — 롱온리+이중게이트 × 레버리지 (numpy 최적화, 청산모델+MDD)"""
import pickle
import numpy as np
from datetime import datetime, timezone

LEV_FEE = 0.0005
MAX_POS = 15
WEIGHT = 1.5
NEW_RATIO = 0.50
INITIAL_EQUITY = 19000.0

ERAS = [
    ("E1", datetime(2026, 6, 23), datetime(2026, 7, 24)),
    ("E2", datetime(2026, 7, 27), datetime(2026, 8, 3)),
    ("E3", datetime(2026, 8, 9), datetime(2026, 8, 21, 14)),
    ("E4", datetime(2026, 8, 21, 14), datetime(2026, 8, 26, 10)),
]


def build_arrays(data):
    """심볼별 numpy 배열 + 시간 인덱스 사전 구축"""
    out = {}
    for sym, df in data.items():
        t_arr = df['t'].to_numpy(dtype=np.int64)
        out[sym] = {
            't': t_arr,
            'ix': {int(t): i for i, t in enumerate(t_arr)},
            'st_dir': df['st_dir'].to_numpy(),
            'stoch': df['stoch_k'].to_numpy(),
            'c': df['c'].to_numpy(dtype=np.float64),
        }
    return out


def simulate(arrs, btc_map, sym_map, gate, lev, max_pos=15, exposure=0.6):
    cash = INITIAL_EQUITY
    positions = {}
    trades = []
    liq = 0
    eq_peak = eq_trough = INITIAL_EQUITY
    mdd = 0.0

    all_ts = sorted(set().union(*[set(a['t'].tolist()) for a in arrs.values()]))

    for idx, t in enumerate(all_ts):
        if idx < 60:
            continue
        last_h = ((t // 3600000) * 3600000) - 3600000
        _bm = btc_map.get(last_h, (20.0, True))
        adx_now, btc_above50 = _bm[0], _bm[1]
        deploy = max(0.15, min(1.0, (adx_now - 8) / 17.0))

        if gate == 'btc50':
            base_allow = btc_above50
        elif gate == 'none':
            base_allow = True
        else:
            base_allow = btc_above50 if gate == 'both' else True

        for sym, a in arrs.items():
            i = a['ix'].get(t)
            if i is None or i < 1:
                continue
            px = a['c'][i]

            pos = positions.get(sym)
            if pos:
                pos['last_px'] = px
                pnl_pct = (px - pos['entry']) / pos['entry'] * lev
                if pnl_pct <= -0.92:
                    liq += 1
                    trades.append((t, -pos['margin']))
                    del positions[sym]
                    continue
                pos['extreme'] = max(pos['extreme'], pnl_pct)
                best = pos['extreme']
                if (pnl_pct <= -0.30 or pnl_pct <= -0.15
                        or (best > 0.4 and pnl_pct < best * 0.5)
                        or (best > 0.2 and pnl_pct < 0.05)
                        or pnl_pct >= 0.50):
                    gross = pos['margin'] * pnl_pct
                    fee = pos['margin'] * lev * LEV_FEE
                    cash += pos['margin'] + gross - fee
                    trades.append((t, gross - fee))
                    del positions[sym]
                continue

            sd, sk = a['st_dir'][i], a['stoch'][i]
            sd_p, sk_p = a['st_dir'][i - 1], a['stoch'][i - 1]
            long_sig = sd == 1 and sk_p < 20 and sk >= 20
            if long_sig:
                if gate == 'sym50':
                    long_sig = sym_map.get(sym, {}).get(last_h, True)
                elif gate == 'both':
                    long_sig = sym_map.get(sym, {}).get(last_h, True) and btc_above50
                elif gate == 'btc50':
                    long_sig = btc_above50
            if not (long_sig and deploy > 0):
                continue
            equity = sum(p['margin'] * (1 + (p['last_px'] - p['entry']) / p['entry'] * lev)
                         for p in positions.values()) + cash
            margin = (equity / max_pos) * WEIGHT * exposure * NEW_RATIO * deploy
            if margin < 100 or margin > cash:
                continue
            cash -= margin + margin * lev * LEV_FEE
            positions[sym] = {'entry': px, 'margin': margin, 'extreme': 0.0, 'last_px': px}

        equity = sum(p['margin'] * (1 + (p['last_px'] - p['entry']) / p['entry'] * lev)
                     for p in positions.values()) + cash
        if equity > eq_peak:
            eq_peak = eq_trough = equity
        elif equity < eq_trough:
            eq_trough = equity
            mdd = max(mdd, (eq_peak - eq_trough) / eq_peak * 100)

    equity = cash
    for sym, pos in positions.items():
        pnl_pct = (pos['last_px'] - pos['entry']) / pos['entry'] * lev
        if pnl_pct <= -0.92:
            pnl_pct = -1.0
            liq += 1
        equity += pos['margin'] * (1 + pnl_pct) - pos['margin'] * lev * LEV_FEE
        trades.append((0, pos['margin'] * pnl_pct - pos['margin'] * lev * LEV_FEE))

    total = equity - INITIAL_EQUITY
    eras = {}
    for name, s, e in ERAS:
        s_ms = s.replace(tzinfo=timezone.utc).timestamp() * 1000
        e_ms = e.replace(tzinfo=timezone.utc).timestamp() * 1000
        eras[name] = sum(p for tt, p in trades if s_ms <= tt < e_ms)
    wins = sum(1 for _, p in trades if p > 0)
    return {'total': total, 'eras': eras, 'n': len(trades),
            'wr': wins / max(1, len(trades)) * 100, 'mdd': mdd, 'liq': liq}


def main():
    with open("/tmp/lg_data.pkl", "rb") as f:
        btc_map, data, sym_map = pickle.load(f)
    arrs = build_arrays(data)
    print(f"{len(data)}개 심볼 로드 (numpy 최적화)\n")

    print("=== 이중게이트(K) × 레버리지 (청산모델+MDD) ===")
    print(f"{'레버':>5s} {'전체':>10s} {'E1횡보':>9s} {'E2스케일':>9s} {'E3랠리':>10s} "
          f"{'E4붕괴':>10s} {'거래':>5s} {'승률':>5s} {'MDD%':>6s} {'청산':>4s}")
    print("─" * 84)
    for lev in (5, 10, 15, 20, 30):
        r = simulate(arrs, btc_map, sym_map, 'both', lev)
        e = r['eras']
        print(f"  {lev:>2d}x {r['total']:+10.0f} {e['E1']:+9.0f} {e['E2']:+9.0f} "
              f"{e['E3']:+10.0f} {e['E4']:+10.0f} {r['n']:5d} {r['wr']:4.0f}% {r['mdd']:5.1f}% {r['liq']:4d}")

    print("\n=== Batch A: 포지션수 × 노출스케일 (K게이트, 10x) ===")
    print(f"{'포지션수':>6s} {'노출':>6s} {'전체':>10s} {'E1횡보':>9s} {'E3랠리':>10s} {'E4붕괴':>10s} {'MDD%':>6s}")
    for mp in (10, 15, 20):
        for expo in (0.4, 0.6, 0.8, 1.0):
            r = simulate(arrs, btc_map, sym_map, 'both', 10, max_pos=mp, exposure=expo)
            e = r['eras']
            print(f"  {mp:3d}   {expo:5.2f} {r['total']:+10.0f} {e['E1']:+9.0f} {e['E3']:+10.0f} "
                  f"{e['E4']:+10.0f} {r['mdd']:5.1f}%")

    print("\n=== 게이트별 비교 (10x) ===")
    for name, g in [("G0 게이트없음", 'none'), ("I BTC50", 'btc50'), ("J 심볼50", 'sym50'), ("K 이중", 'both')]:
        r = simulate(arrs, btc_map, sym_map, g, 10)
        e = r['eras']
        print(f"  {name:12s} 총{r['total']:+9.0f} | E1 {e['E1']:+8.0f} E3 {e['E3']:+9.0f} E4 {e['E4']:+8.0f} "
              f"| MDD {r['mdd']:4.1f}% ({r['n']}거래)")


if __name__ == "__main__":
    main()
