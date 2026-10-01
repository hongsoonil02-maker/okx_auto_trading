#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_krx_trend_following_backtest.py
Backtest Trend Following strategies on Korean Stocks (Top 10 Universe)
Accounting for realistic Korean market costs:
- 0.18% Securities Transaction Tax (증권거래세) on Sell
- 0.015% Brokerage Commission each way
- 0.05% Slippage each way
- Total Roundtrip Cost: ~0.26%
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Tuple

UNIVERSE = {
    "005930.KS": "삼성전자",
    "000660.KS": "SK하이닉스",
    "373220.KS": "LG에너지솔루션",
    "207940.KS": "삼성바이오로직스",
    "005380.KS": "현대차",
    "000270.KS": "기아",
    "068270.KS": "셀트리온",
    "035420.KS": "NAVER",
    "035720.KS": "카카오",
    "042700.KS": "한미반도체",
}

INITIAL_CAPITAL = 10_000_000.0  # 1천만 원
MAX_POSITIONS = 3
POSITION_SIZE_PCT = 0.30       # 종목당 최대 30%
COMMISSION_RATE = 0.00015      # 증권사 수수료 0.015%
TAX_RATE = 0.0018              # 매도 시 거래세 0.18%
SLIPPAGE = 0.0005              # 슬리피지 0.05%

def fetch_data(start_date="2023-01-01", end_date="2026-10-01") -> Dict[str, pd.DataFrame]:
    print(f"📥 [데이터 수집] {len(UNIVERSE)}개 종목 OHLCV 다운로드 중 ({start_date} ~ {end_date})...")
    data = {}
    for ticker, name in UNIVERSE.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty or len(df) < 50:
                print(f"⚠️ {name}({ticker}) 데이터 부족: {len(df)}행")
                continue
            
            # yfinance 다중 인덱스 컬럼 처리
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [col.lower() for col in df.columns]
                
            df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            df = df.dropna()
            
            # 지표 계산
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            df['atr14'] = df['tr'].rolling(14).mean()
            
            # 이동평균
            df['sma20'] = df['close'].rolling(20).mean()
            df['sma60'] = df['close'].rolling(60).mean()
            df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
            
            # 돈키언 채널 (20일 고가, 10일 저가)
            df['donchian_high20'] = df['high'].shift(1).rolling(20).max()
            df['donchian_low10'] = df['low'].shift(1).rolling(10).min()
            
            # SuperTrend 계산 (Period 10, Multiplier 3.0)
            hl2 = (df['high'] + df['low']) / 2.0
            atr10 = df['tr'].rolling(10).mean()
            upper_band = hl2 + 3.0 * atr10
            lower_band = hl2 - 3.0 * atr10
            
            st = np.ones(len(df))
            for i in range(10, len(df)):
                prev_close = df['close'].iloc[i-1]
                prev_st = st[i-1]
                if prev_close > upper_band.iloc[i-1]:
                    st[i] = 1
                elif prev_close < lower_band.iloc[i-1]:
                    st[i] = -1
                else:
                    st[i] = prev_st
            df['supertrend'] = st
            
            data[ticker] = df
            print(f"  ✓ {name} ({ticker}): {len(df)} 영업일 완료")
        except Exception as e:
            print(f"❌ {name}({ticker}) 에러: {e}")
            
    return data


def run_simulation(strategy_name: str, data: Dict[str, pd.DataFrame]) -> dict:
    """
    모든 종목의 일자별 시계열을 결합하여 포트폴리오 차원에서 백테스트 실행
    """
    # 일자 목록 통합
    all_dates = sorted(list(set.union(*[set(df.index) for df in data.values()])))
    all_dates = [d for d in all_dates if d >= pd.Timestamp("2023-04-01")] # 지표 워밍업 후 시작
    
    cash = INITIAL_CAPITAL
    positions = {} # ticker -> {'qty': int, 'entry_price': float, 'entry_date': date, 'peak_price': float, 'stop_price': float}
    trades = []
    daily_equity = []
    total_tax_paid = 0.0
    total_comm_paid = 0.0
    
    for current_date in all_dates:
        # 1. 기존 보유 종목 청산 체크 (일봉 종가 기준)
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            df = data[ticker]
            if current_date not in df.index:
                continue
                
            row = df.loc[current_date]
            close = float(row['close'])
            high = float(row['high'])
            low = float(row['low'])
            
            # 고점 갱신
            if high > pos['peak_price']:
                pos['peak_price'] = high
                
            should_exit = False
            exit_reason = ""
            exit_price = close
            
            if strategy_name == "donchian_breakout":
                # 돈키언 10일 저가 하향 돌파 시 또는 ATR 2배 트레일링 스탑
                donchian_exit = float(row['donchian_low10'])
                trail_stop = pos['peak_price'] - 2.0 * float(row['atr14'])
                if close < max(donchian_exit, trail_stop):
                    should_exit = True
                    exit_reason = "DONCHIAN_EXIT"
                    exit_price = close * (1.0 - SLIPPAGE)
                    
            elif strategy_name == "ma_trend":
                # 20일선 < 60일선 데드크로스 또는 20일선 하향 이탈
                sma20 = float(row['sma20'])
                sma60 = float(row['sma60'])
                if close < sma20 or sma20 < sma60:
                    should_exit = True
                    exit_reason = "MA_DEAD_CROSS"
                    exit_price = close * (1.0 - SLIPPAGE)
                elif close < pos['entry_price'] * 0.93: # 7% 하드 스탑
                    should_exit = True
                    exit_reason = "HARD_STOP_7%"
                    exit_price = close * (1.0 - SLIPPAGE)
                    
            elif strategy_name == "supertrend":
                # SuperTrend 방향이 -1(하락)로 반전 시 청산
                if row['supertrend'] == -1:
                    should_exit = True
                    exit_reason = "SUPERTREND_REVERSAL"
                    exit_price = close * (1.0 - SLIPPAGE)
                elif close < pos['entry_price'] * 0.95: # 5% 보호 스탑
                    should_exit = True
                    exit_reason = "PROTECT_STOP_5%"
                    exit_price = close * (1.0 - SLIPPAGE)
                    
            elif strategy_name == "kis_jev_baseline":
                # 현재 봇의 로직 모사: 2% 손절, 5% 익절, 3% 트레일링(1.5% 드랍)
                unrealized = (close - pos['entry_price']) / pos['entry_price']
                max_ret = (pos['peak_price'] - pos['entry_price']) / pos['entry_price']
                trail_drop = (pos['peak_price'] - close) / pos['peak_price']
                
                if max_ret >= 0.03 and trail_drop >= 0.015:
                    should_exit = True
                    exit_reason = "TRAILING_STOP"
                elif unrealized >= 0.05:
                    should_exit = True
                    exit_reason = "TAKE_PROFIT"
                elif unrealized <= -0.02:
                    should_exit = True
                    exit_reason = "STOP_LOSS_2%"
                exit_price = close * (1.0 - SLIPPAGE)
                
            if should_exit:
                qty = pos['qty']
                gross_proceeds = qty * exit_price
                comm = gross_proceeds * COMMISSION_RATE
                tax = gross_proceeds * TAX_RATE
                net_proceeds = gross_proceeds - comm - tax
                
                cash += net_proceeds
                total_tax_paid += tax
                total_comm_paid += comm
                
                pnl = net_proceeds - (qty * pos['entry_price'] * (1.0 + COMMISSION_RATE + SLIPPAGE))
                pnl_pct = (exit_price / pos['entry_price'] - 1.0) * 100.0
                holding_days = (current_date - pos['entry_date']).days
                
                trades.append({
                    "ticker": ticker,
                    "name": UNIVERSE[ticker],
                    "entry_date": pos['entry_date'].strftime("%Y-%m-%d"),
                    "exit_date": current_date.strftime("%Y-%m-%d"),
                    "holding_days": holding_days,
                    "entry_price": pos['entry_price'],
                    "exit_price": exit_price,
                    "qty": qty,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "reason": exit_reason
                })
                del positions[ticker]
                
        # 2. 신규 진입 탐색
        if len(positions) < MAX_POSITIONS:
            # 진입 후보 선별
            candidates = []
            for ticker, df in data.items():
                if ticker in positions:
                    continue
                if current_date not in df.index:
                    continue
                row = df.loc[current_date]
                close = float(row['close'])
                
                signal = False
                momentum_score = 0.0
                
                if strategy_name == "donchian_breakout":
                    # 20일 신고가 돌파 + 60일선 위
                    donchian_high = float(row['donchian_high20'])
                    sma60 = float(row['sma60'])
                    if close > donchian_high and close > sma60:
                        signal = True
                        momentum_score = (close - sma60) / sma60
                        
                elif strategy_name == "ma_trend":
                    # 20일선 > 60일선 정배열 + 주가가 20일선 돌파
                    sma20 = float(row['sma20'])
                    sma60 = float(row['sma60'])
                    if sma20 > sma60 and close > sma20:
                        signal = True
                        momentum_score = (sma20 - sma60) / sma60
                        
                elif strategy_name == "supertrend":
                    # SuperTrend가 1(상승) 전환 + 20일선 위
                    if row['supertrend'] == 1 and close > float(row['sma20']):
                        signal = True
                        momentum_score = (close - float(row['sma20'])) / float(row['sma20'])
                        
                elif strategy_name == "kis_jev_baseline":
                    # 단순 RSI/호가 모멘텀 흉내 (단기 5일 상승)
                    if close > df['close'].shift(5).loc[current_date]:
                        signal = True
                        momentum_score = close / df['close'].shift(5).loc[current_date]
                        
                if signal:
                    candidates.append((ticker, momentum_score, close))
                    
            # 모멘텀 순으로 정렬 후 빈 자리만큼 진입
            candidates.sort(key=lambda x: x[1], reverse=True)
            empty_slots = MAX_POSITIONS - len(positions)
            
            for ticker, score, close in candidates[:empty_slots]:
                target_alloc = cash / (MAX_POSITIONS - len(positions))
                target_alloc = min(target_alloc, (cash + sum(p['qty'] * close for p in positions.values())) * POSITION_SIZE_PCT)
                
                entry_price = close * (1.0 + SLIPPAGE)
                qty = int(target_alloc / entry_price)
                if qty < 1:
                    continue
                    
                cost = qty * entry_price * (1.0 + COMMISSION_RATE)
                if cost > cash:
                    continue
                    
                cash -= cost
                total_comm_paid += (qty * entry_price * COMMISSION_RATE)
                
                positions[ticker] = {
                    "qty": qty,
                    "entry_price": entry_price,
                    "entry_date": current_date,
                    "peak_price": entry_price
                }
                
        # 일별 총 평가금액 계산
        pos_val = 0.0
        for ticker, pos in positions.items():
            if current_date in data[ticker].index:
                pos_val += pos['qty'] * float(data[ticker].loc[current_date, 'close'])
            else:
                pos_val += pos['qty'] * pos['entry_price']
                
        total_equity = cash + pos_val
        daily_equity.append({"date": current_date, "equity": total_equity})

    # 성과 지표 계산
    eq_df = pd.DataFrame(daily_equity).set_index("date")
    eq_df['peak'] = eq_df['equity'].cummax()
    eq_df['drawdown'] = (eq_df['equity'] - eq_df['peak']) / eq_df['peak']
    
    total_return_pct = (eq_df['equity'].iloc[-1] / INITIAL_CAPITAL - 1.0) * 100.0
    days = (all_dates[-1] - all_dates[0]).days
    cagr = ((eq_df['equity'].iloc[-1] / INITIAL_CAPITAL) ** (365.25 / days) - 1.0) * 100.0
    mdd_pct = eq_df['drawdown'].min() * 100.0
    
    trade_df = pd.DataFrame(trades)
    if not trade_df.empty:
        win_trades = trade_df[trade_df['pnl'] > 0]
        loss_trades = trade_df[trade_df['pnl'] <= 0]
        win_rate = (len(win_trades) / len(trade_df)) * 100.0
        gross_profit = win_trades['pnl'].sum() if len(win_trades) > 0 else 0.0
        gross_loss = abs(loss_trades['pnl'].sum()) if len(loss_trades) > 0 else 1.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
        avg_hold_days = trade_df['holding_days'].mean()
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_hold_days = 0.0

    return {
        "strategy": strategy_name,
        "final_equity": eq_df['equity'].iloc[-1],
        "total_return_pct": total_return_pct,
        "cagr": cagr,
        "mdd_pct": mdd_pct,
        "total_trades": len(trade_df),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_hold_days": avg_hold_days,
        "total_tax_paid": total_tax_paid,
        "total_comm_paid": total_comm_paid,
        "trades_sample": trade_df.tail(5).to_dict(orient="records") if not trade_df.empty else []
    }


def main():
    print("=" * 80)
    print("🚀 [KRX 퀀트 백테스트] 한국 주식 유니버스 추세추종 vs 기존 봇 비교 시뮬레이션")
    print("   테스트 기간: 2023-04-01 ~ 2026-10-01 (약 3.5년)")
    print("   반영 비용: 거래세 0.18% + 수수료 0.015% + 슬리피지 0.05% (왕복 ~0.26%)")
    print("=" * 80)
    
    data = fetch_data("2023-01-01", "2026-10-01")
    if not data:
        print("❌ 데이터 수집 실패")
        return

    # 1. 벤치마크 (Buy & Hold 동일비중) 계산
    all_dates = sorted(list(set.union(*[set(df.index) for df in data.values()])))
    all_dates = [d for d in all_dates if d >= pd.Timestamp("2023-04-01")]
    bh_returns = []
    for d in all_dates:
        closes = [float(data[t].loc[d, 'close']) for t in data if d in data[t].index]
        bh_returns.append(np.mean(closes))
    bh_total_ret = (bh_returns[-1] / bh_returns[0] - 1.0) * 100.0

    strategies = [
        ("kis_jev_baseline", "기존 봇 모델 (단기진입 + 2%SL / 5%TP)"),
        ("ma_trend", "이동평균 추세추종 (20/60 정배열 + 추세이탈 청산)"),
        ("donchian_breakout", "터틀 돈키언 채널 돌파 (20일 신고가 + ATR 트레일링)"),
        ("supertrend", "슈퍼트렌드 변동성 추세 (SuperTrend 10,3.0 + 보호스탑)"),
    ]

    results = []
    for strat_key, strat_label in strategies:
        print(f"\n⚙️ [{strat_label}] 백테스트 연산 중...")
        res = run_simulation(strat_key, data)
        res['label'] = strat_label
        results.append(res)

    print("\n" + "=" * 90)
    print("📊 [백테스트 결과 종합 요약 표]")
    print(f"📌 벤치마크 (Top 10 동일비중 단순 보유 Buy & Hold): +{bh_total_ret:.2f}%")
    print("-" * 90)
    header = f"{'전략명':<32} | {'총수익률':<9} | {'CAGR':<8} | {'MDD':<8} | {'승률':<7} | {'PF':<6} | {'거래수':<6} | {'세금비용(원)':<10}"
    print(header)
    print("-" * 90)
    for r in results:
        print(f"{r['label']:<30} | {r['total_return_pct']:>8.2f}% | {r['cagr']:>7.2f}% | {r['mdd_pct']:>7.2f}% | {r['win_rate']:>6.1f}% | {r['profit_factor']:>5.2f} | {r['total_trades']:>6} | ₩{r['total_tax_paid']:>9,.0f}")
    print("=" * 90)

if __name__ == "__main__":
    main()
