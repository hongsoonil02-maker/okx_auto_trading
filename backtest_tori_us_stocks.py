#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_tori_us_stocks.py — [Phase 2] 미국 주식 대상 토리(Tori) 전략 정밀 백테스트
- 종목: QQQ, SPY (지수 ETF), TQQQ (레버리지), NVDA (대형 주도주)
- 타임프레임: 1D (일봉 스윙) 및 1H (60분봉 단기 스윙)
- 핵심 검증:
  1) Long Only + 현금화 (세이프티 라인 이탈 시 100% 현금화)
  2) 오버나이트 갭(Overnight Gap) 하락 발생 시 슬리피지/손실 반영
  3) 필터 비교: 순수 돌파(Baseline) vs 과속 필터(Speedometer)
  4) 벤치마크: Buy & Hold 대비 알파 및 MDD 방어력
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Import Tori Engine
sys.path.append("/home/hongsoonil02/quant_crypto")
from strategies.tori_trendline import ToriTrendlineEngine

FEE_RATE = 0.0005 # 0.05% 슬리피지 및 수수료 감안


def run_stock_simulation(df: pd.DataFrame, mode: str = "baseline", initial_balance: float = 10000.0) -> dict:
    """
    Simulates Long Only stock trading:
    - Entry: Breakout of downward trendline (Close > Action Line)
    - Exit: Safety line breach or stop loss
    - Overnight Gap Handling: If next Open < Stop Loss, exited at Open price (simulating real gap slip)
    """
    balance = initial_balance
    equity_curve = [balance]
    trades = []
    position = None # {'entry_price': float, 'shares': float, 'sl': float, 'entry_bar': int}
    
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    raw_signals = df['raw_signal'].values
    speed_oks = df['speed_ok'].values
    safety_lines = df['safety_line'].values
    stop_losses = df['stop_loss'].values
    
    gap_slips = []

    for i in range(len(df)):
        c_open = opens[i]
        c_high = highs[i]
        c_low = lows[i]
        c_close = closes[i]
        
        # 1. Manage Active Long Position
        if position is not None:
            entry_price = position['entry_price']
            sl = position['sl']
            shares = position['shares']
            
            # Check Gap Down at Open
            closed = False
            exit_price = 0.0
            
            if c_open <= sl:
                # Gap down below stop loss! Exited at Open (Slippage occurred)
                exit_price = c_open * (1.0 - FEE_RATE)
                gap_slip_pct = (sl - c_open) / sl * 100.0
                gap_slips.append(gap_slip_pct)
                closed = True
            elif c_low <= sl:
                # Intraday hit SL
                exit_price = sl * (1.0 - FEE_RATE)
                closed = True
            else:
                # Dynamic Safety Line Trailing update
                if not np.isnan(safety_lines[i]):
                    if safety_lines[i] > sl and safety_lines[i] < c_close:
                        position['sl'] = safety_lines[i]
                        sl = safety_lines[i]
                        
            if closed:
                pnl = (exit_price - entry_price) * shares
                balance += pnl
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'win': pnl > 0
                })
                position = None
                
        # 2. Entry Check (Long Only)
        if position is None:
            sig = raw_signals[i]
            speed_ok = speed_oks[i]
            sl_val = stop_losses[i]
            
            trigger = False
            if mode == 'baseline':
                trigger = (sig == 1)
            elif mode == 'speedometer':
                trigger = (sig == 1 and speed_ok)
                
            if trigger and not np.isnan(sl_val) and sl_val < c_close:
                # Enter Long (100% equity allocation, no margin debt)
                entry_price = c_close * (1.0 + FEE_RATE)
                shares = balance / entry_price
                position = {
                    'entry_price': entry_price,
                    'shares': shares,
                    'sl': sl_val,
                    'entry_bar': i
                }
                
        # Mark to market equity
        if position is not None:
            current_eq = position['shares'] * c_close
        else:
            current_eq = balance
        equity_curve.append(current_eq)
        
    # Stats
    eq = pd.Series(equity_curve)
    mdd_pct = ((eq.cummax() - eq) / eq.cummax()).max() * 100.0
    ret_pct = ((equity_curve[-1] - initial_balance) / initial_balance) * 100.0
    
    # Buy & Hold Stats
    bh_ret = ((closes[-1] - closes[0]) / closes[0]) * 100.0
    close_s = pd.Series(closes)
    bh_mdd = ((close_s.cummax() - close_s) / close_s.cummax()).max() * 100.0
    
    if not trades:
        return {
            'total_trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0,
            'mdd_pct': round(mdd_pct, 2), 'max_consecutive_losses': 0,
            'ret_pct': round(ret_pct, 2), 'bh_ret': round(bh_ret, 2), 'bh_mdd': round(bh_mdd, 2),
            'avg_gap_slip': 0.0
        }
        
    wins = [t for t in trades if t['win']]
    losses = [t for t in trades if not t['win']]
    win_rate = len(wins) / len(trades) * 100.0
    gross_p = sum(t['pnl'] for t in wins)
    gross_l = abs(sum(t['pnl'] for t in losses))
    pf = (gross_p / gross_l) if gross_l > 0 else 999.0
    
    max_losses = 0
    curr = 0
    for t in trades:
        if not t['win']:
            curr += 1
            if curr > max_losses:
                max_losses = curr
        else:
            curr = 0
            
    avg_gap = np.mean(gap_slips) if gap_slips else 0.0
    
    return {
        'total_trades': len(trades),
        'win_rate': round(win_rate, 2),
        'profit_factor': round(pf, 2),
        'mdd_pct': round(mdd_pct, 2),
        'max_consecutive_losses': max_losses,
        'ret_pct': round(ret_pct, 2),
        'bh_ret': round(bh_ret, 2),
        'bh_mdd': round(bh_mdd, 2),
        'avg_gap_slip': round(avg_gap, 2)
    }


def main():
    print("=" * 75)
    print("  [Phase 2] 미국 주식 대상 토리(Tori) 전략 정밀 백테스트 (KIS 연동 대상)")
    print("=" * 75)
    
    tickers = ['QQQ', 'SPY', 'TQQQ', 'NVDA']
    timeframes = [
        ('1D (일봉 스윙, 최근 5년)', '5y', '1d'),
        ('1H (60분봉 단기, 최근 2년)', '730d', '1h')
    ]
    
    engine = ToriTrendlineEngine(pivot_window=4, speed_factor=2.0)
    
    for tf_label, period, interval in timeframes:
        print(f"\n#################### 타임프레임: {tf_label} ####################")
        
        for ticker in tickers:
            print(f"\n▶ 분석 티커: {ticker}")
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if data.empty:
                print(f"  {ticker} 데이터 로드 실패")
                continue
            # Flatten multiindex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [c[0].lower() for c in data.columns]
            else:
                data.columns = [c.lower() for c in data.columns]
                
            eval_df = engine.evaluate(data)
            
            res_base = run_stock_simulation(eval_df, mode="baseline")
            res_speed = run_stock_simulation(eval_df, mode="speedometer")
            
            print(f"  기간: {data.index[0].strftime('%Y-%m-%d')} ~ {data.index[-1].strftime('%Y-%m-%d')} ({len(data)} bars)")
            print(f"  [벤치마크 Buy & Hold] 수익률: +{res_base['bh_ret']}% | MDD: {res_base['bh_mdd']}%")
            print(f"{'지표 (Metrics)':<25} | {'1) 순수 돌파(Baseline)':<20} | {'2) 과속 필터(Speed)':<20}")
            print("-" * 75)
            print(f"{'총 거래 횟수 (Trades)':<25} | {res_base['total_trades']:<20} | {res_speed['total_trades']:<20}")
            print(f"{'승률 (Win Rate %)':<25} | {str(res_base['win_rate'])+'%':<20} | {str(res_speed['win_rate'])+'%':<20}")
            print(f"{'손익비 (Profit Factor)':<25} | {res_base['profit_factor']:<20} | {res_speed['profit_factor']:<20}")
            print(f"{'전략 MDD (MDD %)':<25} | {str(res_base['mdd_pct'])+'%':<20} | {str(res_speed['mdd_pct'])+'%':<20}")
            print(f"{'최대 연속 손절 (Max Losses)':<25} | {str(res_base['max_consecutive_losses'])+'회':<20} | {str(res_speed['max_consecutive_losses'])+'회':<20}")
            print(f"{'전략 누적 수익률 (Return)':<25} | {str(res_base['ret_pct'])+'%':<20} | {str(res_speed['ret_pct'])+'%':<20}")
            print(f"{'오버나이트 갭 슬리피지':<25} | {str(res_base['avg_gap_slip'])+'%':<20} | {str(res_speed['avg_gap_slip'])+'%':<20}")


if __name__ == "__main__":
    main()
