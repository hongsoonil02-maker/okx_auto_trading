#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_dca_sim.py — DCA/피라미딩 조건별 검증 (K게이트, 10x, 10pos, 0.8노출)"""
import pickle
import numpy as np
from datetime import datetime, timezone

LEV_FEE = 0.0005
MAX_POS = 10
WEIGHT = 1.5
EXPOSURE = 0.8
NEW_RATIO = 0.50
DCA_RATIO = 0.25
PYR_RATIO = 0.50
DCA_SPACING = 4
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
        h = df['h'].to_numpy(float)
        tr = np.maximum(h - df['l'].to_numpy(float),
                        np.maximum(np.abs(h - np.roll(df['c'].to_numpy(float), 1)),
                                   np.abs(df['l'].to_numpy(float) - np.roll(df['c'].to_numpy(float), 1))))
        tr[0] = h[0] - df['l'].to_numpy(float)[0]
        alpha = 1.0 / 14
        atr = np.empty_like(tr)
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
        out[sym] = {
            't': t_arr, 'ix': {int(t): i for i, t in enumerate(t_arr)},
            'st_dir': df['st_dir'].to_numpy(), 'stoch': df['stoch_k'].to_numpy(),
            'c': df['c'].to_numpy(float), 'h': h, 'atr': atr,
        }
    return out


def simulate(arrs, btc_map, sym_map, dca_mode, pyr_on):
    """
    dca_mode: 'D0' 없음 | 'D1' 현행(손실-10%이내 풀백 물타기, 최대3회) | 'D2' D1+심볼>1h EMA50
    pyr_on:   피라미딩 (+40% 1차, +100% 2차, 마진 base×0.5) / 피라미딩 후 +20% 유지선
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
        deploy = max(0.15, min(1.0, (adx_now - 8) / 17.0))
        base_m = (INITIAL_EQUITY / MAX_POS) * WEIGHT * EXPOSURE  # 단순화: 신규 진입 시점 자산 대신 고정 기준

        for sym, a in arrs.items():
            i = a['ix'].get(t)
            if i is None or i < 1:
                continue
            px = a['c'][i]

            pos = positions.get(sym)
            if pos:
                pos['last_px'] = px
                pos['highest'] = max(pos['highest'], a['h'][i])
                pnl_pct = (px - pos['avg']) / pos['avg'] * 10
                if pnl_pct <= -0.92:
                    trades.append((t, -pos['margin']))
                    del positions[sym]
                    continue
                pos['extreme'] = max(pos['extreme'], pnl_pct)
                best = pos['extreme']

                reason = None
                if pnl_pct <= -0.15:
                    reason = 'stop15'
                elif pnl_pct <= -0.30:
                    reason = 'stop30'
                elif pyr_on and pos['pyr'] >= 1 and pnl_pct < 0.20:
                    reason = 'pyr_guard'
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
                    del positions[sym]
                    continue

                # ── DCA ──
                if dca_mode != 'D0' and pos['count'] < 3 and pnl_pct > -0.10 \
                        and (t - pos['last_add']) >= DCA_SPACING * 15 * 60 * 1000:
                    if dca_mode == 'D1' or (dca_mode == 'D2' and sym_map.get(sym, {}).get(last_h, True)):
                        add_m = base_m * DCA_RATIO * deploy
                        if add_m >= 50 and add_m <= cash:
                            pos['avg'] = (pos['avg'] * pos['margin'] + px * add_m) / (pos['margin'] + add_m)
                            pos['margin'] += add_m
                            pos['count'] += 1
                            pos['last_add'] = t
                            cash -= add_m + add_m * 10 * LEV_FEE
                        continue  # DCA 캔들에서는 피라미딩 안 겹침

                # ── 피라미딩 ──
                if pyr_on and best >= 0.40 and pos['pyr'] == 0 \
                        and (t - pos['last_add']) >= DCA_SPACING * 15 * 60 * 1000:
                    add_m = base_m * PYR_RATIO * deploy
                    if add_m >= 50 and add_m <= cash:
                        pos['avg'] = (pos['avg'] * pos['margin'] + px * add_m) / (pos['margin'] + add_m)
                        pos['margin'] += add_m
                        pos['pyr'] += 1
                        pos['last_add'] = t
                        cash -= add_m + add_m * 10 * LEV_FEE
                continue

            # ── 신규 진입 ──
            sd, sk = a['st_dir'][i], a['stoch'][i]
            sd_p, sk_p = a['st_dir'][i - 1], a['stoch'][i - 1]
            long_sig = sd == 1 and sk_p < 20 and sk >= 20
            if long_sig:
                long_sig = btc_above50 and sym_map.get(sym, {}).get(last_h, True)  # K 이중게이트
            if not (long_sig and deploy > 0):
                continue
            equity = sum(p['margin'] * (1 + (p['last_px'] - p['avg']) / p['avg'] * 10)
                         for p in positions.values()) + cash
            margin = (equity / MAX_POS) * WEIGHT * EXPOSURE * NEW_RATIO * deploy
            if margin < 100 or margin > cash:
                continue
            cash -= margin + margin * 10 * LEV_FEE
            positions[sym] = {'avg': px, 'margin': margin, 'count': 1, 'pyr': 0,
                              'last_add': t, 'extreme': 0.0, 'highest': a['h'][i], 'last_px': px}

        equity = sum(p['margin'] * (1 + (p['last_px'] - p['avg']) / p['avg'] * 10)
                     for p in positions.values()) + cash
        if equity > eq_peak:
            eq_peak = eq_trough = equity
        elif equity < eq_trough:
            eq_trough = equity
            mdd = max(mdd, (eq_peak - eq_trough) / eq_peak * 100)

    equity = cash
    for sym, pos in positions.items():
        pnl_pct = (pos['last_px'] - pos['avg']) / pos['avg'] * 10
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
    losses = [p for _, p in trades if p <= 0]
    avg_loss = sum(losses) / len(losses) if losses else 0
    return {'total': total, 'eras': eras, 'n': len(trades),
            'wr': wins / max(1, len(trades)) * 100, 'mdd': mdd, 'avg_loss': avg_loss}


def main():
    with open("/tmp/lg_data.pkl", "rb") as f:
        btc_map, data, sym_map = pickle.load(f)
    arrs = build_arrays(data)
    print(f"{len(data)}개 심볼 로드\n")

    print("=== DCA/피라미딩 조건별 (K게이트, 10x, 10pos, 0.8노출) ===")
    print(f"{'조합':22s} {'전체':>10s} {'E1횡보':>9s} {'E3랠리':>10s} {'E4붕괴':>10s} "
          f"{'MDD%':>6s} {'승률':>5s} {'평균손실':>8s}")
    print("─" * 92)
    combos = [
        ("D0P0 기준(둘다없음)", 'D0', False),
        ("D1P0 현행DCA만", 'D1', False),
        ("D2P0 추세정렬DCA", 'D2', False),
        ("D0P1 피라미딩만", 'D0', True),
        ("D1P1 현행조합", 'D1', True),
        ("D2P1 추세정렬+피라", 'D2', True),
    ]
    for label, dm, po in combos:
        r = simulate(arrs, btc_map, sym_map, dm, po)
        e = r['eras']
        print(f"  {label:20s} {r['total']:+10.0f} {e['E1']:+9.0f} {e['E3']:+10.0f} "
              f"{e['E4']:+10.0f} {r['mdd']:5.1f}% {r['wr']:4.0f}% {r['avg_loss']:+8.1f}")


if __name__ == "__main__":
    main()
