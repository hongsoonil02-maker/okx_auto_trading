#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_tori_kr_stocks.py — [Phase 4] 한국 주식(KRX) 대상 토리(Tori) 전략 정밀 백테스트
- 종목:
  1) KODEX 레버리지 (122630.KS) — 코스피 레버리지 추세
  2) 삼성전자 (005930.KS) — 대형 반도체 주도주
  3) SK하이닉스 (000660.KS) — 고베타 반도체 주도주
  4) 현대차 (005380.KS) — 밸류업/대형 제조주
- 타임프레임: 일봉 (1D 스윙, 최근 5년: 2021~2026)
- 한국 주식 특성 반영:
  1) Long Only + 현금화 로직: 세이프티 라인 이탈 시 100% 현금화 (하락장 방어)
  2) 오버나이트 갭하락(Gap Down) 방어: 장 시작(09:00 KST) 시가가 세이프티 라인 밑이면 시가 탈출 슬리피지 반영
  3) 국내 주식 거래비용: 0.015% 수수료 + 0.18% 증권거래세 = 약 0.20% 라운드트립 비용
- 비교 매트릭스:
  순수 돌파(Baseline) vs 과속 필터(Speedometer) vs 단순 보유(Buy & Hold)
"""

import sys
import os
import yfinance as yf
import pandas as pd
import numpy as np

# Import Tori Engine
sys.path.append("/home/hongsoonil02/quant_crypto")
from strategies.tori_trendline import ToriTrendlineEngine

FEE_TAX_RATE = 0.0020  # 0.20% (거래세 0.18% + 수수료/슬리피지)


def run_kr_simulation(df: pd.DataFrame, mode: str = "baseline", initial_balance: float = 10000000.0) -> dict:
    balance = initial_balance
    equity_curve = [balance]
    trades = []
    position = None
    
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
        
        # 1. 포지션 관리 및 청산
        if position is not None:
            entry_price = position['entry_price']
            sl = position['sl']
            shares = position['shares']
            
            closed = False
            exit_price = 0.0
            
            # 오버나이트 갭하락 방어
            if c_open <= sl:
                exit_price = c_open * (1.0 - FEE_TAX_RATE)
                gap_slip_pct = (sl - c_open) / sl * 100.0
                gap_slips.append(gap_slip_pct)
                closed = True
            elif c_low <= sl: # 장중 세이프티 라인 터치
                exit_price = sl * (1.0 - FEE_TAX_RATE)
                closed = True
            else:
                # 동적 세이프티 라인 상향 트레일링
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
                
        # 2. 신규 진입 (Long Only)
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
                entry_price = c_close * (1.0 + FEE_TAX_RATE)
                shares = int(balance / entry_price)
                if shares > 0:
                    position = {
                        'entry_price': entry_price,
                        'shares': shares,
                        'sl': sl_val
                    }
                    
        curr_eq = (position['shares'] * c_close) if position is not None else balance
        equity_curve.append(curr_eq)
        
    eq = pd.Series(equity_curve)
    mdd_pct = ((eq.cummax() - eq) / eq.cummax()).max() * 100.0
    ret_pct = ((equity_curve[-1] - initial_balance) / initial_balance) * 100.0
    
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
    print("  [Phase 4] 한국 주식(KRX) 대상 토리(Tori) 추세선 전략 정밀 백테스트")
    print("=" * 75)
    
    targets = {
        '삼성전자 (005930.KS)': '005930.KS',
        'SK하이닉스 (000660.KS)': '000660.KS',
        '현대차 (005380.KS)': '005380.KS',
        'KODEX 레버리지 (122630.KS)': '122630.KS'
    }
    
    engine = ToriTrendlineEngine(pivot_window=4, speed_factor=2.0)
    
    for name, ticker in targets.items():
        print(f"\n▶ 분석 종목: {name}")
        df = yf.download(ticker, period="5y", interval="1d", progress=False)
        if df.empty:
            print(f"  {ticker} 데이터 로드 실패")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
            
        eval_df = engine.evaluate(df)
        res_base = run_kr_simulation(eval_df, mode="baseline")
        res_speed = run_kr_simulation(eval_df, mode="speedometer")
        
        print(f"  데이터 기간: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')} ({len(df)} 거래일)")
        print(f"  [벤치마크 Buy & Hold] 수익률: {res_base['bh_ret']}% | MDD: -{res_base['bh_mdd']}%")
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
