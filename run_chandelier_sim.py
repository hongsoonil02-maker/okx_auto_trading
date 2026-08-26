#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_chandelier_sim.py — ATR Chandelier K값 × 발동임계값 검증 (K게이트, 10x, 15포지션, 0.6노출)"""
import pickle
import numpy as np
from datetime import datetime, timezone

LEV_FEE = 0.0005
MAX_POS = 15
WEIGHT = 1.5
NEW_RATIO = 0.50
EXPOSURE = 0.6
INITIAL_EQUITY = 19000.0

ERAS = [
    ("E1", datetime(2026, 6, 23), datetime(2026, 7, 24)),
    ("E2", datetime(2026, 7, 27), datetime(2026, 8, 3)),
    ("E3", datetime(2026, 8, 9), datetime(2026, 8, 21, 14)),
    ("E4", datetime(2026, 8, 21, 14), datetime(2026, 8, 26, 10)),
]


def build_arrays(data):
    out = {}
    for sym, df in data.items():
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
            'c': c, 'h': h, 'atr': atr,
        }
    return out


def pd_ewm(arr, span):
    """numpy ewm (pandas ewm alpha=1/span 동등)"""
    alpha = 1.0 / span
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def simulate(arrs, btc_map, sym_map, chand_k, chand_arm):
    cash = INITIAL_EQUITY
    positions = {}
    trades = []
    exits = {'stop15': 0, 'stop30': 0, 'trail': 0, 'tp': 0, 'chand': 0}
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
        allow_long = btc_above50

        for sym, a in arrs.items():
            i = a['ix'].get(t)
            if i is None or i < 1:
                continue
            px = a['c'][i]

            pos = positions.get(sym)
            if pos:
                pos['last_px'] = px
                pnl_pct = (px - pos['entry']) / pos['entry'] * 10
                if pnl_pct <= -0.92:
                    liq += 1
                    trades.append((t, -pos['margin']))
                    del positions[sym]
                    continue
                pos['extreme'] = max(pos['extreme'], pnl_pct)
                pos['highest'] = max(pos['highest'], a['h'][i])
                best = pos['extreme']

                reason = None
                if pnl_pct <= -0.15:
                    reason = 'stop15'
                elif pnl_pct <= -0.30:
                    reason = 'stop30'
                elif chand_k and best >= chand_arm and px < pos['highest'] - chand_k * a['atr'][i]:
                    reason = 'chand'
                elif best > 0.4 and pnl_pct < best * 0.5:
                    reason = 'trail'
                elif best > 0.2 and pnl_pct < 0.05:
                    reason = 'trail'
                elif pnl_pct >= 0.50:
                    reason = 'tp'

                if reason:
                    gross = pos['margin'] * pnl_pct
                    fee = pos['margin'] * 10 * LEV_FEE
                    cash += pos['margin'] + gross - fee
                    trades.append((t, gross - fee))
                    exits[reason] += 1
                    del positions[sym]
                continue

            sd, sk = a['st_dir'][i], a['stoch'][i]
            sd_p, sk_p = a['st_dir'][i - 1], a['stoch'][i - 1]
            long_sig = sd == 1 and sk_p < 20 and sk >= 20
            if long_sig and gate_sym(sym, last_h, sym_map, btc_above50):
                equity = sum(p['margin'] * (1 + (p['last_px'] - p['entry']) / p['entry'] * 10)
                             for p in positions.values()) + cash
                margin = (equity / MAX_POS) * WEIGHT * EXPOSURE * NEW_RATIO * deploy
                if 100 <= margin <= cash:
                    cash -= margin + margin * 10 * LEV_FEE
                    positions[sym] = {'entry': px, 'margin': margin, 'extreme': 0.0,
                                      'last_px': px, 'highest': a['h'][i]}

        equity = sum(p['margin'] * (1 + (p['last_px'] - p['entry']) / p['entry'] * 10)
                     for p in positions.values()) + cash
        if equity > eq_peak:
            eq_peak = eq_trough = equity
        elif equity < eq_trough:
            eq_trough = equity
            mdd = max(mdd, (eq_peak - eq_trough) / eq_peak * 100)

    equity = cash
    for sym, pos in positions.items():
        pnl_pct = (pos['last_px'] - pos['entry']) / pos['entry'] * 10
        if pnl_pct <= -0.92:
            pnl_pct = -1.0
        equity += pos['margin'] * (1 + pnl_pct) - pos['margin'] * 10 * LEV_FEE
        trades.append((0, pos['margin'] * pnl_pct - pos['margin'] * 10 * LEV_FEE))

    total = equity - INITIAL_EQUITY
    eras = {}
    for name, s, e in ERAS:
        s_ms = s.replace(tzinfo=timezone.utc).timestamp() * 1000
        e_ms = e.replace(tzinfo=timezone.utc).timestamp() * 1000
        eras[name] = sum(p for tt, p in trades if s_ms <= tt < e_ms)
    wins = sum(1 for _, p in trades if p > 0)
    return {'total': total, 'eras': eras, 'n': len(trades), 'wr': wins / max(1, len(trades)) * 100,
            'mdd': mdd, 'exits': exits}


def gate_sym(sym, last_h, sym_map, btc_above50):
    if not btc_above50:
        return False
    return sym_map.get(sym, {}).get(last_h, True)


def main():
    with open("/tmp/lg_data.pkl", "rb") as f:
        btc_map, data, sym_map = pickle.load(f)
    arrs = build_arrays(data)
    print(f"{len(data)}개 심볼 로드\n")

    print("=== ATR Chandelier 검증 (K게이트, 10x, 15pos, 0.6노출) ===")
    print(f"{'K값':>6s} {'발동':>6s} {'전체':>10s} {'E1횡보':>9s} {'E3랠리':>10s} {'E4붕괴':>10s} "
          f"{'MDD%':>6s} {'승률':>5s} {'Chd청산':>7s}")
    print("─" * 80)
    variants = [
        ("없음", None, 0.20),
        ("1.5", 1.5, 0.20),
        ("2.0", 2.0, 0.20),
        ("2.5*", 2.5, 0.20),
        ("3.0", 3.0, 0.20),
        ("2.5", 2.5, 0.10),
        ("2.5", 2.5, 0.30),
    ]
    for label, k, arm in variants:
        r = simulate(arrs, btc_map, sym_map, k, arm)
        e = r['eras']
        ex = r['exits']
        cur = " ←현행" if label == "2.5*" else ""
        print(f"  {label:>4s} {arm:>5.0%} {r['total']:+10.0f} {e['E1']:+9.0f} {e['E3']:+10.0f} "
              f"{e['E4']:+10.0f} {r['mdd']:5.1f}% {r['wr']:4.0f}% {ex['chand']:7d}{cur}")
    print("\n청산 사유 합계 참고:", {k: v for k, v in exits.items()})


if __name__ == "__main__":
    main()
