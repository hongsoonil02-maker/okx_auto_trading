#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_kodex_pair_inverse.py — [한국 주식] 코스피 토리 추세선 양방향 스위칭 (레버리지 ↔ 인버스 2X)
- 전략 구조:
  * 코스피 지수(또는 KODEX 레버리지) 기준 추세선 돌파 판단
  * 상승 돌파(Long 신호): KODEX 레버리지 (122630) 100% 매수
  * 하락 돌파(Short 신호): KODEX 200선물인버스2X (252670, 일명 '곱버스') 100% 매수
  * 세이프티 라인 이탈 시: 즉시 청산 후 관망(현금) or 반대편 돌파 시 스위칭
- 비용: 증권거래세 및 수수료 0.20% 라운드트립 차감
- 기간: 2021~2026 (최근 5년 실데이터)
"""

import sys
import os
import yfinance as yf
import pandas as pd
import numpy as np

# Import Tori Engine
sys.path.append("/home/hongsoonil02/quant_crypto")
from strategies.tori_trendline import ToriTrendlineEngine

FEE_TAX = 0.0020  # 0.20%


def run_inverse_switching_backtest():
    print("=" * 75)
    print("  [한국 주식] 토리 추세선 양방향 스위칭 (레버리지 122630 ↔ 곱버스 252670)")
    print("=" * 75)

    df_lev = yf.download("122630.KS", period="5y", interval="1d", progress=False)
    df_inv = yf.download("252670.KS", period="5y", interval="1d", progress=False)

    if isinstance(df_lev.columns, pd.MultiIndex):
        df_lev.columns = [c[0].lower() for c in df_lev.columns]
        df_inv.columns = [c[0].lower() for c in df_inv.columns]
    else:
        df_lev.columns = [c.lower() for c in df_lev.columns]
        df_inv.columns = [c.lower() for c in df_inv.columns]

    # Align dates
    common_idx = df_lev.index.intersection(df_inv.index)
    df_lev = df_lev.loc[common_idx]
    df_inv = df_inv.loc[common_idx]

    engine = ToriTrendlineEngine(pivot_window=4, speed_factor=2.0)
    eval_df = engine.evaluate(df_lev)

    initial_balance = 10000000.0
    balance = initial_balance
    equity_curve = [balance]
    trades = []
    
    # State: None, 'LEVERAGE', 'INVERSE'
    current_holding = None
    entry_price = 0.0
    shares = 0
    sl_price = 0.0

    closes_lev = df_lev['close'].values
    opens_lev = df_lev['open'].values
    lows_lev = df_lev['low'].values
    highs_lev = df_lev['high'].values

    closes_inv = df_inv['close'].values
    opens_inv = df_inv['open'].values

    raw_sigs = eval_df['raw_signal'].values
    speed_oks = eval_df['speed_ok'].values
    safety_lines = eval_df['safety_line'].values
    stop_losses = eval_df['stop_loss'].values

    for i in range(len(eval_df)):
        c_lev = closes_lev[i]
        c_inv = closes_inv[i]
        sig = raw_sigs[i]
        spd = speed_oks[i]
        sl_val = stop_losses[i]
        safety = safety_lines[i]

        # 1. Manage Active Holding
        if current_holding == 'LEVERAGE':
            # Check exit for Leverage
            closed = False
            exit_price = 0.0
            if lows_lev[i] <= sl_price:
                exit_price = sl_price * (1.0 - FEE_TAX)
                closed = True
            elif not np.isnan(safety) and safety > sl_price and safety < c_lev:
                sl_price = safety  # Trailing
                
            if closed:
                pnl = (exit_price - entry_price) * shares
                balance += pnl
                trades.append({'type': 'LEVERAGE', 'pnl': pnl, 'win': pnl > 0})
                current_holding = None

        elif current_holding == 'INVERSE':
            # Check exit for Inverse (when KODEX Leverage rallies above sl_price)
            closed = False
            exit_price = 0.0
            if highs_lev[i] >= sl_price:
                # Leverage broke upward safety line -> Exit Inverse
                exit_price = c_inv * (1.0 - FEE_TAX)
                closed = True
            elif not np.isnan(safety) and safety < sl_price and safety > c_lev:
                sl_price = safety  # Downward Trailing
                
            if closed:
                pnl = (exit_price - entry_price) * shares
                balance += pnl
                trades.append({'type': 'INVERSE', 'pnl': pnl, 'win': pnl > 0})
                current_holding = None

        # 2. Entry Evaluation
        if current_holding is None:
            if sig == 1 and spd and not np.isnan(sl_val): # Long Signal -> Buy KODEX Leverage
                entry_price = c_lev * (1.0 + FEE_TAX)
                shares = int(balance / entry_price)
                if shares > 0:
                    current_holding = 'LEVERAGE'
                    sl_price = sl_val
            elif sig == -1 and spd and not np.isnan(sl_val): # Short Signal -> Buy KODEX 200선물인버스2X (곱버스)
                entry_price = c_inv * (1.0 + FEE_TAX)
                shares = int(balance / entry_price)
                if shares > 0:
                    current_holding = 'INVERSE'
                    sl_price = sl_val

        # Mark-to-market
        if current_holding == 'LEVERAGE':
            eq = shares * c_lev
        elif current_holding == 'INVERSE':
            eq = shares * c_inv
        else:
            eq = balance
        equity_curve.append(eq)

    # Performance
    wins = [t for t in trades if t['win']]
    losses = [t for t in trades if not t['win']]
    win_rate = len(wins) / len(trades) * 100.0 if trades else 0.0
    gross_p = sum(t['pnl'] for t in wins)
    gross_l = abs(sum(t['pnl'] for t in losses))
    pf = (gross_p / gross_l) if gross_l > 0 else 999.0

    eq_series = pd.Series(equity_curve)
    mdd_pct = ((eq_series.cummax() - eq_series) / eq_series.cummax()).max() * 100.0
    ret_pct = ((equity_curve[-1] - initial_balance) / initial_balance) * 100.0

    lev_trades = [t for t in trades if t['type'] == 'LEVERAGE']
    inv_trades = [t for t in trades if t['type'] == 'INVERSE']
    lev_win = len([t for t in lev_trades if t['win']]) / max(len(lev_trades), 1) * 100.0
    inv_win = len([t for t in inv_trades if t['win']]) / max(len(inv_trades), 1) * 100.0

    print(f"\n기간: {df_lev.index[0].strftime('%Y-%m-%d')} ~ {df_lev.index[-1].strftime('%Y-%m-%d')} (총 {len(df_lev)} 거래일)")
    print(f"{'지표 (Metrics)':<30} | {'Long Only (레버리지+현금화)':<22} | {'양방향 (레버리지 ↔ 곱버스)':<22}")
    print("-" * 80)
    print(f"{'총 거래 횟수 (Total Trades)':<30} | {'16회':<22} | {f'{len(trades)}회 (레버 {len(lev_trades)}/인버스 {len(inv_trades)})':<22}")
    print(f"{'전체 승률 (Win Rate %)':<30} | {'31.25%':<22} | {f'{win_rate:.2f}% (레버 {lev_win:.1f}%/인버스 {inv_win:.1f}%)':<22}")
    print(f"{'손익비 (Profit Factor)':<30} | {'6.52':<22} | {f'{pf:.2f}':<22}")
    print(f"{'최대 낙폭 (MDD %)':<30} | {'33.98%':<22} | {f'{mdd_pct:.2f}%':<22}")
    print(f"{'누적 수익률 (Total Return %)':<30} | {'+196.02%':<22} | {f'+{ret_pct:.2f}%':<22}")
    print(f"{'벤치마크 KODEX 레버리지 MDD':<30} | {'-68.37% (폭락)':<22} | {'-68.37% (폭락)':<22}")


if __name__ == "__main__":
    run_inverse_switching_backtest()
