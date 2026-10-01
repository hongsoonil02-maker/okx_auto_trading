#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_comparative_innovation_backtest.py
- Compares Step-by-Step Improvements against the Current Benchmark:
  [Benchmark] Jev Hierarchical Gate (1h Trend Gate + ADX Chop Block + Jev Filter)
- Evaluates:
  1. Benchmark (Jev Hierarchical Gate alone, No DCA, No Pyramiding)
  2. + Winner Pyramiding (불타기: 수익 +35% 달성 시 1회 증액)
  3. + BTC Dominance Shield (비트 독주장 시 알트 롱 차단 & 비트 집중)
  4. + Sideways Micro-Range Grid (ADX < 20 횡보장 전용 레인지 틱 떼기 모드)
  5. + All-in-One Optimal Hybrid (계층적 게이트 + 도미넌스 가드 + 횡보장 그리드)
- Universe: All 16 available symbols (BTC, ETH, SOL, XRP, BNB, DOGE, ZEC, CRV, NEAR, LTC, HYPE, AAVE, UNI, TAO, PEPE, LIT)
- Period: 2026.06.20 ~ 2026.08.26 (6,451 candles) | Leverage: 10x | Initial Capital: $10,000
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

INITIAL_EQUITY = 10000.0
LEVERAGE = 10
TAKER_FEE = 0.0005 + 0.0003  # 0.08%
MAKER_FEE = 0.0002           # 0.02%

ERAS = [
    ("E1 횡보 (6/23~7/24)", datetime(2026, 6, 23), datetime(2026, 7, 24)),
    ("E2 스케일 (7/27~8/3)", datetime(2026, 7, 27), datetime(2026, 8, 3)),
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

def build_universe_arrays(data):
    out = {}
    for sym, df in data.items():
        t_arr = df['t'].to_numpy(dtype=np.int64)
        c = df['c'].to_numpy(float)
        h = df['h'].to_numpy(float)
        l = df['l'].to_numpy(float)
        o = df['o'].to_numpy(float) if 'o' in df.columns else np.roll(c, 1)

        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr = pd_ewm(tr, 14)

        # Micro-momentum LOB proxy
        rng = np.maximum(h - l, 1e-6)
        close_pos = (c - l) / rng
        prob_up = np.clip(0.40 + 0.35 * close_pos + 0.15 * ((c - o) / rng), 0.10, 0.95)

        # Bollinger Bands for Range / Grid Mode (20-period, 2-std)
        sma20 = pd_ewm(c, 20)
        # rolling std approximate
        diff_sq = (c - sma20) ** 2
        var20 = pd_ewm(diff_sq, 20)
        std20 = np.sqrt(np.maximum(var20, 1e-8))
        bb_upper = sma20 + 2.0 * std20
        bb_lower = sma20 - 2.0 * std20

        out[sym] = {
            't': t_arr,
            'ix': {int(t): i for i, t in enumerate(t_arr)},
            'st_dir': df['st_dir'].to_numpy(),
            'stoch': df['stoch_k'].to_numpy(),
            'c': c, 'h': h, 'l': l, 'o': o,
            'atr': atr,
            'prob_up': prob_up,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'sma20': sma20,
        }
    return out

def run_strategy_simulation(
    arrs,
    btc_map,
    sym_map,
    use_pyramiding: bool = False,
    use_dominance_shield: bool = False,
    use_range_grid: bool = False,
    max_concurrent_positions: int = 5,
):
    cash = INITIAL_EQUITY
    positions = {}
    range_positions = {}
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

        # ── 1. Trend Positions Management ──
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

                # Hard Stop
                if pnl_pct <= -0.30:
                    reason = "hard_sl"
                # Pyramiding logic (+35% reached and not yet pyramided)
                elif use_pyramiding and (not pos.get('pyramided', False)) and pnl_pct >= 0.35:
                    add_margin = pos['margin'] * 0.50
                    if cash >= add_margin:
                        cash -= add_margin
                        # blend entry price
                        old_vol = pos['margin'] * LEVERAGE / pos['entry']
                        new_vol = add_margin * LEVERAGE / px
                        pos['entry'] = (old_vol * pos['entry'] + new_vol * px) / (old_vol + new_vol)
                        pos['margin'] += add_margin
                        pos['pyramided'] = True
                # Chandelier Trailing Stop
                elif pos['highest'] > pos['entry']:
                    trail_stop = pos['highest'] - 3.5 * a['atr'][i]
                    unrealized_ret = (pos['highest'] - pos['entry']) / pos['entry'] * LEVERAGE
                    if unrealized_ret >= 0.35 and px < trail_stop:
                        reason = "chandelier_trailing"
                elif a['st_dir'][i] == -1:
                    reason = "supertrend_flip"

                if reason:
                    gross = pos['margin'] * pnl_pct
                    fee = pos['margin'] * LEVERAGE * TAKER_FEE
                    net_pnl = gross - fee
                    cash += pos['margin'] + net_pnl
                    trades.append({'t': t, 'pnl': net_pnl, 'type': 'trend', 'reason': reason})
                    del positions[sym]
                continue

            # ── 2. Range Grid Positions Management ──
            if use_range_grid and sym in range_positions:
                rpos = range_positions[sym]
                rpnl_pct = ((px - rpos['entry']) / rpos['entry']) * LEVERAGE
                r_reason = None
                # Take profit at +1.5% spot (+15% ROE) or SMA20 cross
                if rpnl_pct >= 0.15 or px >= a['sma20'][i]:
                    r_reason = "grid_tp"
                elif rpnl_pct <= -0.15:  # Tight grid SL at -15% ROE (-1.5% spot)
                    r_reason = "grid_sl"

                if r_reason:
                    rgross = rpos['margin'] * rpnl_pct
                    rfee = rpos['margin'] * LEVERAGE * MAKER_FEE  # Maker execution in grid
                    rnet = rgross - rfee
                    cash += rpos['margin'] + rnet
                    trades.append({'t': t, 'pnl': rnet, 'type': 'grid', 'reason': r_reason})
                    del range_positions[sym]
                continue

            # ── 3. Signal Generation ──
            sd, sk = a['st_dir'][i], a['stoch'][i]
            sd_p, sk_p = a['st_dir'][i - 1], a['stoch'][i - 1]
            is_pullback = (sd == 1 and sk_p < 20 and sk >= 20)

            sym_ok = sym_map.get(sym, {}).get(last_h, True)
            gate_1h_ok = (btc_above50 and sym_ok)
            chop_ok = (adx_now >= 20.0)
            prob_up = a['prob_up'][i]
            jev_ok = (prob_up >= 0.65)

            # Trend Mode: Hierarchical Gate
            if chop_ok:
                if is_pullback and gate_1h_ok and jev_ok:
                    # Dominance Shield Check
                    # If BTC is surging (adx >= 28 and btc_above50), block Altcoins and trade only Majors
                    if use_dominance_shield and adx_now >= 28 and btc_above50:
                        if sym not in ("BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"):
                            continue  # Block weak altcoins during BTC dominance spike

                    if len(positions) < max_concurrent_positions:
                        alloc = cash * 0.18
                        if alloc >= 50 and alloc <= cash:
                            fee = alloc * LEVERAGE * MAKER_FEE
                            cash -= (alloc + fee)
                            positions[sym] = {
                                'entry': px * 0.9998,
                                'margin': alloc,
                                'highest': a['h'][i],
                                'last_px': px,
                                'pyramided': False,
                            }
            # Range Mode (ADX < 20): Mean Reversion Grid
            elif use_range_grid and not chop_ok:
                # In sideways markets: Buy when price touches lower Bollinger Band with positive Jev LOB support
                if px <= a['bb_lower'][i] and prob_up >= 0.60:
                    if len(range_positions) < 3:
                        r_alloc = cash * 0.10  # Smaller size for range grid
                        if r_alloc >= 50 and r_alloc <= cash:
                            r_fee = r_alloc * LEVERAGE * MAKER_FEE
                            cash -= (r_alloc + r_fee)
                            range_positions[sym] = {
                                'entry': px,
                                'margin': r_alloc,
                            }

        # Track Equity & Drawdown
        u_pnl_t = sum([((arrs[s]['c'][arrs[s]['ix'][t]] - p['entry']) / p['entry']) * LEVERAGE * p['margin'] for s, p in positions.items() if t in arrs[s]['ix']])
        u_pnl_r = sum([((arrs[s]['c'][arrs[s]['ix'][t]] - p['entry']) / p['entry']) * LEVERAGE * p['margin'] for s, p in range_positions.items() if t in arrs[s]['ix']])
        equity = cash + sum(p['margin'] for p in positions.values()) + sum(p['margin'] for p in range_positions.values()) + u_pnl_t + u_pnl_r

        if equity > eq_peak:
            eq_peak = equity
            eq_trough = equity
        if equity < eq_trough:
            eq_trough = equity
            mdd = max(mdd, (eq_peak - eq_trough) / eq_peak * 100.0)

    # Wrap up open
    for p in positions.values():
        cash += p['margin']
    for p in range_positions.values():
        cash += p['margin']

    total_pnl = cash - INITIAL_EQUITY
    ret_pct = total_pnl / INITIAL_EQUITY * 100.0

    wins = [x for x in trades if x['pnl'] > 0]
    losses = [x for x in trades if x['pnl'] <= 0]
    wr = (len(wins) / len(trades) * 100.0) if trades else 0.0
    tot_win = sum([x['pnl'] for x in wins])
    tot_loss = abs(sum([x['pnl'] for x in losses]))
    pf = (tot_win / tot_loss) if tot_loss > 0 else (99.0 if tot_win > 0 else 0.0)

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
        'eras': era_pnl,
    }

def main():
    cache_path = "/tmp/lg_data.pkl"
    with open(cache_path, "rb") as f:
        btc_map, data, sym_map = pickle.load(f)

    arrs = build_universe_arrays(data)

    experiments = [
        ("1. [기준 벤치마크] Jev 계층적 게이트 단독", False, False, False),
        ("2. [+ 승자 피라미딩] +35% 도달 시 1회 불타기 증액", True, False, False),
        ("3. [+ BTC 도미넌스 가드] 비트 랠리 시 알트 롱 차단 & 메이커 집중", False, True, False),
        ("4. [+ 횡보장 그리드 모드] ADX < 20 시 볼린저 밴드 틱 떼기 결합", False, False, True),
        ("5. [🔥 최적 하이브리드 조합] 계층적 게이트 + 도미넌스 가드 + 횡보장 그리드", False, True, True),
    ]

    results = []
    for title, pyr, dom, grid in experiments:
        res = run_strategy_simulation(
            arrs, btc_map, sym_map,
            use_pyramiding=pyr,
            use_dominance_shield=dom,
            use_range_grid=grid,
            max_concurrent_positions=5,
        )
        results.append((title, res))

    print("\n" + "=" * 115)
    print(" 🔬 [전략 고도화 백테스트 비교 분석] Jev 계층적 게이트 vs 단계별 혁신 아이디어 결합")
    print(" 대상: 전체 16개 유니버스 (BTC/ETH/SOL/XRP/BNB/DOGE/AAVE/UNI 등) | 레버리지: 10x | 자본: $10,000")
    print("=" * 115)

    print(f"\n{'전략 시나리오':<52} | {'수익률(%)':<9} | {'순익($)':<10} | {'승률':<6} | {'PF':<5} | {'MDD%':<6} | {'매매수'}")
    print("-" * 115)
    for title, r in results:
        print(
            f"{title:<50} | "
            f"{r['ret_pct']:+8.1f}% | "
            f"{r['total_pnl']:+9.0f}$ | "
            f"{r['wr']:5.1f}% | "
            f"{r['pf']:4.2f} | "
            f"{r['mdd']:5.1f}% | "
            f"{r['trades']:6d}"
        )

    print("\n" + "=" * 115)
    print(" 📊 [국면별 손익 분해] (E1 횡보장 vs E2 스케일업 vs E3 불장 랠리 vs E4 급락 붕괴장)")
    print("=" * 115)
    print(f"{'전략 시나리오':<48} | {'E1 횡보':<11} | {'E2 스케일':<11} | {'E3 랠리':<11} | {'E4 붕괴'}")
    print("-" * 115)
    for title, r in results:
        e = r['eras']
        print(
            f"{title:<46} | "
            f"{e['E1 횡보 (6/23~7/24)']:+10.0f}$ | "
            f"{e['E2 스케일 (7/27~8/3)']:+10.0f}$ | "
            f"{e['E3 랠리 (8/9~8/21 14h)']:+10.0f}$ | "
            f"{e['E4 붕괴 (8/21 14h~8/26)']:+10.0f}$"
        )
    print("=" * 115 + "\n")

if __name__ == "__main__":
    main()
