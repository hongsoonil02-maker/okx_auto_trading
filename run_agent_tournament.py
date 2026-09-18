#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_agent_tournament.py — OKX 자동매매 10개 에이전트 토너먼트 시뮬레이터
과거 180일(2026.03 ~ 2026.09) 및 최근 하락/횡보 구간(2026.08 ~ 2026.09) 데이터를 기반으로
10종의 에이전트 전략을 대전시켜 수익률, MDD, 승률, 손익비, 샤프/칼마 비율을 산출하고
현재 로직(Baseline) 대비 Top 1, Top 2 전략을 백테스트로 정밀 비교 분석합니다.
"""

import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone

INITIAL_EQUITY = 10000.0  # 10,000 USDT 기준
FEE_RATE = 0.0005        # 편도 0.05%
HOUR_MS = 3600 * 1000

# 1. 180일 15m 캐시 데이터 로드
CACHE_FILE = "state/cache_backtest/fj_grid_15m_180d.pkl"
with open(CACHE_FILE, "rb") as f:
    data = pickle.load(f)

# numpy 배열화
arr = {}
for sym, df in data.items():
    idx = {int(t): i for i, t in enumerate(df['t'])}
    a = {
        'idx': idx,
        'c': df['c'].to_numpy(), 'h': df['h'].to_numpy(), 'l': df['l'].to_numpy(),
        'atr': df['atr'].to_numpy(), 'st': df['st_dir'].to_numpy(),
        'sk': df['stoch_k'].to_numpy(), 'ema': df['ema50'].to_numpy(),
        'adx': df['adx'].to_numpy(),
    }
    if 'ema200' in df.columns:
        a['ema200'] = df['ema200'].to_numpy()
    arr[sym] = a

U = np.array(sorted(set().union(*[set(df['t']) for df in data.values()])), dtype=np.int64)
btc_a = arr.get("BTC/USDT:USDT")
btc_idx = btc_a['idx'] if btc_a else {}

# 2. 토너먼트 참가 10대 에이전트 스펙 정의
AGENTS = {
    # [Agent 0: 현재 기준 로직 (Baseline)]
    "0. Current_Prod (현행 수정형)": {
        "leverage": 3, "max_pos": 5, "hard_stop_pct": -0.30, "soft_stop_pct": 0.0,
        "time_stop_hours": 24, "pyramiding": False, "pyramid_trigger_pct": 0.15, "pyramid_ratio": 0.5,
        "entry_threshold": 80, "short_allowed": False, "adx_filter": True, "adx_min": 20,
        "regime_filter": True, "trailing_arm_pct": 0.30, "trailing_k": 2.5, "dynamic_lev": False,
        "vol_sizing": False, "breakout_only": False
    },
    # [Agent 1: alv*** 정통 추세 스윙어 (Alv Swinger)]
    "1. Alv_Trend_Swinger": {
        "leverage": 2, "max_pos": 4, "hard_stop_pct": -0.08, "soft_stop_pct": -0.05,
        "time_stop_hours": 96, "pyramiding": True, "pyramid_trigger_pct": 0.15, "pyramid_ratio": 0.5,
        "entry_threshold": 90, "short_allowed": False, "adx_filter": True, "adx_min": 25,
        "regime_filter": True, "trailing_arm_pct": 0.30, "trailing_k": 3.0, "dynamic_lev": False,
        "vol_sizing": True, "breakout_only": False
    },
    # [Agent 2: 칼손절 하이-모멘텀 스나이퍼 (Tight Sniper)]
    "2. Strict_Momentum_Sniper": {
        "leverage": 3, "max_pos": 3, "hard_stop_pct": -0.05, "soft_stop_pct": -0.03,
        "time_stop_hours": 24, "pyramiding": False, "pyramid_trigger_pct": 0.15, "pyramid_ratio": 0.5,
        "entry_threshold": 90, "short_allowed": False, "adx_filter": True, "adx_min": 25,
        "regime_filter": True, "trailing_arm_pct": 0.15, "trailing_k": 2.0, "dynamic_lev": False,
        "vol_sizing": False, "breakout_only": True
    },
    # [Agent 3: 동적 레버리지 서바이벌 (Dynamic Vol Shield)]
    "3. Dynamic_Vol_Shield": {
        "leverage": 3, "max_pos": 5, "hard_stop_pct": -0.08, "soft_stop_pct": 0.0,
        "time_stop_hours": 48, "pyramiding": True, "pyramid_trigger_pct": 0.15, "pyramid_ratio": 0.5,
        "entry_threshold": 80, "short_allowed": False, "adx_filter": True, "adx_min": 20,
        "regime_filter": True, "trailing_arm_pct": 0.25, "trailing_k": 2.5, "dynamic_lev": True,
        "vol_sizing": True, "breakout_only": False
    },
    # [Agent 4: 무자비한 피라미딩 러너 (Pyramid Winner)]
    "4. Aggressive_Pyramid_Runner": {
        "leverage": 3, "max_pos": 4, "hard_stop_pct": -0.06, "soft_stop_pct": 0.0,
        "time_stop_hours": 72, "pyramiding": True, "pyramid_trigger_pct": 0.10, "pyramid_ratio": 0.5,
        "entry_threshold": 85, "short_allowed": False, "adx_filter": True, "adx_min": 22,
        "regime_filter": True, "trailing_arm_pct": 0.20, "trailing_k": 2.5, "dynamic_lev": False,
        "vol_sizing": False, "breakout_only": False
    },
    # [Agent 5: 듀얼 레짐 양방향 트렌더 (Dual Direction Hedger)]
    "5. Dual_Regime_Hedger": {
        "leverage": 2, "max_pos": 4, "hard_stop_pct": -0.06, "soft_stop_pct": 0.0,
        "time_stop_hours": 48, "pyramiding": False, "pyramid_trigger_pct": 0.15, "pyramid_ratio": 0.5,
        "entry_threshold": 85, "short_allowed": True, "adx_filter": True, "adx_min": 25,
        "regime_filter": False, "trailing_arm_pct": 0.20, "trailing_k": 2.0, "dynamic_lev": False,
        "vol_sizing": False, "breakout_only": False
    },
    # [Agent 6: 딥 횡보 철통 방어기 (Ultra Chop Gatekeeper)]
    "6. Ultra_Chop_Gatekeeper": {
        "leverage": 3, "max_pos": 3, "hard_stop_pct": -0.05, "soft_stop_pct": 0.0,
        "time_stop_hours": 24, "pyramiding": False, "pyramid_trigger_pct": 0.15, "pyramid_ratio": 0.5,
        "entry_threshold": 90, "short_allowed": False, "adx_filter": True, "adx_min": 28,
        "regime_filter": True, "trailing_arm_pct": 0.20, "trailing_k": 2.5, "dynamic_lev": False,
        "vol_sizing": False, "breakout_only": False
    },
    # [Agent 7: ATR 샹들리에 맥스 홀더 (Chandelier Max Runner)]
    "7. Chandelier_Max_Runner": {
        "leverage": 2, "max_pos": 5, "hard_stop_pct": -0.10, "soft_stop_pct": 0.0,
        "time_stop_hours": 120, "pyramiding": True, "pyramid_trigger_pct": 0.20, "pyramid_ratio": 0.5,
        "entry_threshold": 80, "short_allowed": False, "adx_filter": True, "adx_min": 20,
        "regime_filter": True, "trailing_arm_pct": 0.35, "trailing_k": 3.5, "dynamic_lev": False,
        "vol_sizing": True, "breakout_only": False
    },
    # [Agent 8: 퀵 테이크프로핏 스캘퍼 (Quick TP Scalper)]
    "8. Quick_TP_Scalper": {
        "leverage": 4, "max_pos": 4, "hard_stop_pct": -0.04, "soft_stop_pct": 0.0,
        "time_stop_hours": 12, "pyramiding": False, "pyramid_trigger_pct": 0.15, "pyramid_ratio": 0.5,
        "entry_threshold": 85, "short_allowed": False, "adx_filter": True, "adx_min": 20,
        "regime_filter": True, "trailing_arm_pct": 0.08, "trailing_k": 1.5, "dynamic_lev": False,
        "vol_sizing": False, "breakout_only": False
    },
    # [Agent 9: 비트코인 리더십 팔로워 (BTC Leader Beta)]
    "9. BTC_Leader_Beta": {
        "leverage": 3, "max_pos": 4, "hard_stop_pct": -0.06, "soft_stop_pct": 0.0,
        "time_stop_hours": 48, "pyramiding": True, "pyramid_trigger_pct": 0.12, "pyramid_ratio": 0.4,
        "entry_threshold": 80, "short_allowed": False, "adx_filter": True, "adx_min": 22,
        "regime_filter": True, "trailing_arm_pct": 0.25, "trailing_k": 2.5, "dynamic_lev": False,
        "vol_sizing": True, "breakout_only": False
    }
}

def simulate_agent(agent_cfg, lo_idx, hi_idx):
    cash = INITIAL_EQUITY
    peak_equity = INITIAL_EQUITY
    max_dd = 0.0
    positions = {}
    active = set()
    trades = []
    
    symbols = list(arr.keys())
    
    for gi in range(lo_idx + 100, hi_idx):
        t = int(U[gi])
        
        # BTC 지표 확인
        btc_adx = 25.0
        btc_trend_ok = True
        btc_vol = 0.5
        if btc_a is not None:
            bi = btc_idx.get(t, -1)
            if bi >= 0:
                btc_adx = float(btc_a['adx'][bi])
                btc_trend_ok = float(btc_a['c'][bi]) > float(btc_a['ema200'][bi])
                btc_vol = float(btc_a['atr'][bi]) / float(btc_a['c'][bi]) * 100

        # 동적 레버리지 산출
        eff_lev = agent_cfg["leverage"]
        if agent_cfg.get("dynamic_lev"):
            if btc_vol < 0.3:
                eff_lev = min(eff_lev + 1, 4)
            elif btc_vol > 0.6:
                eff_lev = max(eff_lev - 1, 1)

        # 1. 포지션 관리
        for sym in symbols:
            a = arr[sym]
            i = a['idx'].get(t, -1)
            if i < 2:
                continue
            px = float(a['c'][i])
            atr_val = float(a['atr'][i])

            plist = positions.get(sym)
            if plist:
                remaining = []
                for p in plist:
                    p['last_px'] = px
                    pnl_pct = (px - p['entry']) / p['entry'] * p['lev'] * p['dir']
                    if pnl_pct > p.get('extreme', 0):
                        p['extreme'] = pnl_pct
                    best = p['extreme']
                    hours_held = (t - p['entry_t']) / HOUR_MS
                    exit_now = False

                    # 하드 스탑
                    if pnl_pct <= agent_cfg['hard_stop_pct']:
                        exit_now = True
                    # 소프트 스탑
                    elif agent_cfg['soft_stop_pct'] != 0 and pnl_pct <= agent_cfg['soft_stop_pct']:
                        exit_now = True
                    # 샹들리에 트레일링 익절
                    elif agent_cfg['trailing_k'] > 0 and best >= agent_cfg['trailing_arm_pct']:
                        if p['dir'] == 1:
                            highest = max(p.get('highest', p['entry']), float(a['h'][i]))
                            p['highest'] = highest
                            trail_line = highest - agent_cfg['trailing_k'] * (p.get('entry_atr', atr_val) or atr_val)
                            if px < trail_line:
                                exit_now = True
                        else:
                            lowest = min(p.get('lowest', p['entry']), float(a['l'][i]))
                            p['lowest'] = lowest
                            trail_line = lowest + agent_cfg['trailing_k'] * (p.get('entry_atr', atr_val) or atr_val)
                            if px > trail_line:
                                exit_now = True
                    
                    # 방어 익절선 (최고수익 50% 도달 후 40% 반납 시 방어 청산)
                    if not exit_now:
                        if best >= 0.50 and pnl_pct <= best * 0.6:
                            exit_now = True
                        elif best >= 0.20 and pnl_pct <= 0.05:
                            exit_now = True
                    
                    # 시간 스탑
                    if not exit_now and agent_cfg['time_stop_hours'] > 0:
                        if hours_held >= agent_cfg['time_stop_hours'] and pnl_pct <= 0.02:
                            exit_now = True
                    
                    # 극단 수익 익절 (+80%)
                    if not exit_now and pnl_pct >= 0.80:
                        exit_now = True

                    if exit_now:
                        gross = p['margin'] * pnl_pct
                        fee = p['margin'] * p['lev'] * FEE_RATE * 2
                        net = gross - fee
                        cash += p['margin'] + net
                        trades.append({'pnl': net, 'margin': p['margin'], 'hold_h': hours_held, 'win': net > 0})
                    else:
                        remaining.append(p)
                positions[sym] = remaining
                if not remaining:
                    active.discard(sym)
            else:
                remaining = []

            # 2. 피라미딩 (불타기)
            if agent_cfg['pyramiding'] and remaining:
                for p in remaining:
                    pnl_pct = (px - p['entry']) / p['entry'] * p['lev'] * p['dir']
                    if pnl_pct >= agent_cfg['pyramid_trigger_pct'] and not p.get('pyramided'):
                        pyr_margin = p['margin'] * agent_cfg['pyramid_ratio']
                        if 30 <= pyr_margin <= cash * 0.4:
                            fee = pyr_margin * p['lev'] * FEE_RATE
                            cash -= pyr_margin + fee
                            remaining.append({
                                'entry': px, 'margin': pyr_margin, 'dir': p['dir'],
                                'lev': p['lev'], 'extreme': 0.0, 'last_px': px,
                                'entry_t': t, 'entry_atr': atr_val, 'pyramided': True
                            })
                            p['pyramided'] = True
                            break

            if remaining:
                continue

            # 3. 진입 필터 검사
            if agent_cfg['adx_filter'] and btc_adx < agent_cfg['adx_min']:
                continue
            if agent_cfg['regime_filter'] and not btc_trend_ok:
                continue
            if len(active) >= agent_cfg['max_pos']:
                continue

            # 신호 계산
            long_score = 0
            ema50 = float(a['ema'][i])
            if px > ema50 > 0: long_score += 50
            if a['st'][i - 1] == -1 and a['st'][i] == 1: long_score += 40
            if a['st'][i] == 1 and a['sk'][i - 1] < 20 and a['sk'][i] >= 20: long_score += 30
            if a['st'][i] == 1 and a['st'][i - 1] == 1: long_score += 20
            if 20 < a['sk'][i] < 80: long_score += 20

            short_score = 0
            if agent_cfg['short_allowed']:
                if px < ema50 and ema50 > 0: short_score += 50
                if a['st'][i - 1] == 1 and a['st'][i] == -1: short_score += 40
                if a['st'][i] == -1 and a['sk'][i - 1] > 80 and a['sk'][i] <= 80: short_score += 30
                if a['st'][i] == -1 and a['st'][i - 1] == -1: short_score += 20
                if 20 < a['sk'][i] < 80: short_score += 20

            entry_dir = 0
            if long_score >= agent_cfg['entry_threshold']:
                entry_dir = 1
            elif short_score >= agent_cfg['entry_threshold'] + 20 and agent_cfg['short_allowed']:
                entry_dir = -1
            
            if entry_dir == 0:
                continue

            # 돌파 전용 필터
            if agent_cfg.get("breakout_only") and (a['st'][i - 1] == a['st'][i]):
                continue

            # 자본 평가 및 사이징
            equity = cash
            for s2 in positions:
                for p in positions[s2]:
                    pnl = (p['last_px'] - p['entry']) / p['entry'] * p['lev'] * p['dir']
                    equity += p['margin'] * (1 + pnl)
            
            base_margin = (equity / agent_cfg['max_pos']) * 0.6
            if agent_cfg.get("vol_sizing"):
                vol_ratio = 1.0 / max(atr_val / px * 100, 0.5)
                base_margin *= min(max(vol_ratio, 0.7), 1.4)

            margin = base_margin
            if margin < 30 or margin > cash * 0.9:
                continue

            fee = margin * eff_lev * FEE_RATE
            cash -= margin + fee
            positions[sym] = [{
                'entry': px, 'margin': margin, 'dir': entry_dir, 'lev': eff_lev,
                'extreme': 0.0, 'last_px': px, 'entry_t': t, 'entry_atr': atr_val
            }]
            active.add(sym)

        # 현재 자산 추적 및 MDD
        equity = cash
        for s2 in positions:
            for p in positions[s2]:
                pnl = (p['last_px'] - p['entry']) / p['entry'] * p['lev'] * p['dir']
                equity += p['margin'] * (1 + pnl)
        if equity > peak_equity:
            peak_equity = equity
        dd = (equity - peak_equity) / peak_equity
        if dd < max_dd:
            max_dd = dd

    # 잔여 포지션 청산
    equity = cash
    for s2 in positions:
        for p in positions[s2]:
            pnl = (p['last_px'] - p['entry']) / p['entry'] * p['lev'] * p['dir']
            net = p['margin'] * pnl - p['margin'] * p['lev'] * FEE_RATE * 2
            equity += p['margin'] + net
            trades.append({'pnl': net, 'margin': p['margin'], 'hold_h': 0, 'win': net > 0})

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    total_net = equity - INITIAL_EQUITY
    roi = total_net / INITIAL_EQUITY * 100
    wr = len(wins) / len(trades) * 100 if trades else 0
    aw = np.mean([t['pnl'] for t in wins]) if wins else 0
    al = abs(np.mean([t['pnl'] for t in losses])) if losses else 1
    pf = (sum(t['pnl'] for t in wins)) / (abs(sum(t['pnl'] for t in losses)) + 1e-6) if losses else 99.0
    calmar = roi / max(abs(max_dd * 100), 1.0)
    
    return {
        'final_equity': equity,
        'net_pnl': total_net,
        'roi_pct': roi,
        'mdd_pct': max_dd * 100,
        'trades': len(trades),
        'win_rate': wr,
        'profit_factor': pf,
        'pl_ratio': aw / al,
        'calmar': calmar
    }

# 3. 토너먼트 실행
print("=" * 105)
print("🏆 OKX 자동매매 에이전트 10강 토너먼트 (180일 전체 구간: 2026-03-16 ~ 2026-09-13)")
print("=" * 105)

results = []
for name, cfg in AGENTS.items():
    res = simulate_agent(cfg, 0, len(U))
    res['name'] = name
    results.append(res)

results.sort(key=lambda x: (x['calmar'], x['roi_pct']), reverse=True)

print(f"{'Rank':4s} {'Agent Name':32s} {'ROI%':>10s} {'Final USDT':>12s} {'MDD%':>8s} {'Trades':>8s} {'WR%':>7s} {'PF':>6s} {'Calmar':>8s}")
print("-" * 105)
for rk, r in enumerate(results, 1):
    print(f"{rk:<4d} {r['name']:32s} {r['roi_pct']:>+9.1f}% {r['final_equity']:>11,.0f} {r['mdd_pct']:>7.1f}% {r['trades']:>8d} {r['win_rate']:>6.1f}% {r['profit_factor']:>6.2f} {r['calmar']:>8.2f}")

print("\n" + "=" * 105)
print("⚔️ 8/21 피크 하락 및 최근 하락/횡보 구간 전용 검증 (2026-08-15 ~ 2026-09-13, 30일)")
print("=" * 105)

t_815 = int(datetime(2026, 8, 15, tzinfo=timezone.utc).timestamp() * 1000)
idx_815 = int(np.searchsorted(U, [t_815])[0])

recent_results = []
for name, cfg in AGENTS.items():
    res = simulate_agent(cfg, idx_815, len(U))
    res['name'] = name
    recent_results.append(res)

recent_results.sort(key=lambda x: (x['roi_pct'], x['calmar']), reverse=True)

print(f"{'Rank':4s} {'Agent Name':32s} {'ROI%':>10s} {'Final USDT':>12s} {'MDD%':>8s} {'Trades':>8s} {'WR%':>7s} {'PF':>6s} {'Calmar':>8s}")
print("-" * 105)
for rk, r in enumerate(recent_results, 1):
    print(f"{rk:<4d} {r['name']:32s} {r['roi_pct']:>+9.1f}% {r['final_equity']:>11,.0f} {r['mdd_pct']:>7.1f}% {r['trades']:>8d} {r['win_rate']:>6.1f}% {r['profit_factor']:>6.2f} {r['calmar']:>8.2f}")
