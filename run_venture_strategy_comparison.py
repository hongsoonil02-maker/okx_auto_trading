#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_venture_strategy_comparison.py
벤처/알트/밈(Venture/Alt/Meme) 전략 정밀 백테스트 비교
- 타깃 심볼: DOGE, PEPE, ZEC, CRV, NEAR, HYPE, AAVE, UNI, TAO, LIT (Venture 대표 유니버스)
- 기간: 2026.06.20 ~ 2026.08.26 (6,451개 15m 캔들, 약 2.2개월)
- 비교 항목:
  1. 타임프레임 비교 (15m vs 30m vs 1h)
  2. ADX 횡보 필터 효과 (노필터 vs ADX 20 vs ADX 22)
  3. Chandelier Exit (ATR 익절 트레일링) 효과 (없음 vs k=2.0 vs k=2.5 vs k=3.0)
  4. 1h 추세 게이트 (BTC & 심볼 1h EMA50) 효과
  5. 최적 통합 조합
"""
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# 메이저 및 블랙리스트(LTC) 제외 벤처 유니버스
VENTURE_SYMBOLS = [
    "DOGE/USDT:USDT", "PEPE/USDT:USDT", "ZEC/USDT:USDT", "CRV/USDT:USDT",
    "NEAR/USDT:USDT", "HYPE/USDT:USDT", "AAVE/USDT:USDT", "UNI/USDT:USDT",
    "TAO/USDT:USDT", "LIT/USDT:USDT"
]

INITIAL_EQUITY = 10000.0
LEVERAGE = 5  # Venture 기본 5x
FEE_RATE = 0.0005
MAX_POS = 8   # Venture 동시 보유 수 제한

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
                   lev=LEVERAGE,
                   hard_sl=-0.15,
                   use_1h_gate=False,
                   adx_min=0.0,
                   chand_k=None,
                   chand_arm=0.20):
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

        # 1. 청산 체크
        for sym, a in arrs.items():
            i = a['ix'].get(t)
            if i is None or i < 1:
                continue
            px = a['c'][i]

            pos = positions.get(sym)
            if pos:
                pos['last_px'] = px
                pnl_pct = (px - pos['entry']) / pos['entry'] * lev
                if pnl_pct <= -0.90:
                    trades.append({'t': t, 'pnl': -pos['margin'], 'reason': 'liq'})
                    del positions[sym]
                    continue

                pos['extreme'] = max(pos['extreme'], pnl_pct)
                pos['highest'] = max(pos['highest'], a['h'][i])
                best = pos['extreme']

                reason = None
                if pnl_pct <= hard_sl:
                    reason = 'hard_sl'
                elif a['st_dir'][i] == -1:
                    reason = 'st_flip'
                elif chand_k and best >= chand_arm and px < (pos['highest'] - chand_k * a['atr'][i]):
                    reason = 'chandelier'
                elif best > 0.40 and pnl_pct < best * 0.50:
                    reason = 'trail_half'
                elif best > 0.20 and pnl_pct < 0.05:
                    reason = 'trail_be'
                elif pnl_pct >= 0.60:
                    reason = 'tp'

                if reason:
                    gross = pos['margin'] * pnl_pct
                    fee = pos['margin'] * lev * FEE_RATE
                    cash += pos['margin'] + gross - fee
                    trades.append({'t': t, 'pnl': gross - fee, 'reason': reason})
                    del positions[sym]
                continue

            # 2. 신규 진입 체크
            if len(positions) >= MAX_POS:
                continue

            sd, sk = a['st_dir'][i], a['stoch'][i]
            sd_p, sk_p = a['st_dir'][i - 1], a['stoch'][i - 1]

            # 롱 풀백
            entry_sig = (sd == 1 and sk_p < 20 and sk >= 20)

            # ADX 횡보 필터
            if adx_min > 0.0 and adx_now < adx_min:
                entry_sig = False

            # 1h 추세 게이트
            if use_1h_gate:
                sym_ok = sym_map.get(sym, {}).get(last_h, True)
                if not (btc_above50 and sym_ok):
                    entry_sig = False

            if entry_sig:
                # 포지션당 자본의 15% 균등 분할
                margin = (cash / MAX_POS) * 1.2
                margin = min(margin, cash * 0.20)
                if margin >= 50 and margin <= cash:
                    fee = margin * lev * FEE_RATE
                    cash -= (margin + fee)
                    positions[sym] = {
                        'entry': px, 'margin': margin, 'extreme': 0.0,
                        'last_px': px, 'highest': a['h'][i], 'entry_t': t
                    }

        equity = sum(p['margin'] * (1 + (p['last_px'] - p['entry']) / p['entry'] * lev)
                     for p in positions.values()) + cash
        if equity > eq_peak:
            eq_peak = eq_trough = equity
        elif equity < eq_trough:
            eq_trough = equity
            mdd = max(mdd, (eq_peak - eq_trough) / eq_peak * 100)

    # 잔여 포지션 정리
    equity = cash
    for sym, pos in positions.items():
        pnl_pct = (pos['last_px'] - pos['entry']) / pos['entry'] * lev
        pnl = pos['margin'] * pnl_pct - pos['margin'] * lev * FEE_RATE
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

    arrs_15m = build_arrays(data)
    print("=" * 88)
    print(" [Venture/Meme/Alt 10종목] 전략 백테스트 비교 분석 ")
    print(f" 기간: 2026.06.20 ~ 08.26 (6,451 캔들) | 초기자본: $10,000 | 레버리지: 5x")
    print(f" 대상: {', '.join(VENTURE_SYMBOLS[:5])} 외 5종")
    print("=" * 88)

    experiments = [
        # (이름, arrs, lev, sl, gate, adx, chand_k, chand_arm)
        ("1. [현행 벤처 기준] 15m 단일 (5x, SL -15%, 노필터)", arrs_15m, 5, -0.15, False, 0.0, None, 0.20),
        ("2. [+ ADX 20 횡보 필터] 15m (5x)", arrs_15m, 5, -0.15, False, 20.0, None, 0.20),
        ("3. [+ ADX 22 횡보 필터] 15m (5x)", arrs_15m, 5, -0.15, False, 22.0, None, 0.20),
        ("4. [+ Chandelier 2.0] 15m (5x, ATR 2.0)", arrs_15m, 5, -0.15, False, 0.0, 2.0, 0.20),
        ("5. [+ Chandelier 2.5] 15m (5x, ATR 2.5)", arrs_15m, 5, -0.15, False, 0.0, 2.5, 0.20),
        ("6. [+ 1h 추세 게이트] 15m (5x, BTC & 심볼 1h EMA50)", arrs_15m, 5, -0.15, True, 0.0, None, 0.20),
        ("7. [벤처 통합 A] 15m + ADX 20 + Chandelier 2.5", arrs_15m, 5, -0.15, False, 20.0, 2.5, 0.20),
        ("8. [벤처 통합 B] 15m + ADX 20 + Chandelier 2.0 + 1h 게이트", arrs_15m, 5, -0.15, True, 20.0, 2.0, 0.20),
    ]

    results = []
    for exp in experiments:
        name, a, lev, sl, gate, adx, chand_k, chand_arm = exp
        r = run_simulation(a, btc_map, sym_map, lev=lev, hard_sl=sl,
                           use_1h_gate=gate, adx_min=adx,
                           chand_k=chand_k, chand_arm=chand_arm)
        results.append((name, r))

    print(f"\n{'전략명':<45s} | {'수익률(%)':>9s} | {'손익($)':>9s} | {'승률':>5s} | {'PF':>5s} | {'MDD%':>6s} | {'매매수':>4s}")
    print("-" * 92)
    for name, r in results:
        print(f"{name:<45s} | {r['ret_pct']:+8.1f}% | {r['total_pnl']:+8.0f}$ | {r['wr']:4.1f}% | {r['pf']:4.2f} | {r['mdd']:5.1f}% | {r['trades']:4d}")

    print("\n" + "=" * 92)
    print(" [국면별 손익 분해] (E1 횡보장 vs E3 랠리장 vs E4 급락 붕괴장)")
    print("=" * 92)
    print(f"{'전략명':<45s} | {'E1 횡보':>10s} | {'E2 스케일':>10s} | {'E3 랠리':>10s} | {'E4 붕괴':>10s}")
    print("-" * 92)
    for name, r in results:
        e = r['eras']
        print(f"{name:<45s} | {e['E1 횡보 (6/23~7/24)']:+10.0f}$ | {e['E2 스케일 (7/27~8/3)']:+10.0f}$ | {e['E3 랠리 (8/9~8/21 14h)']:+10.0f}$ | {e['E4 붕괴 (8/21 14h~8/26)']:+10.0f}$")

if __name__ == "__main__":
    main()
