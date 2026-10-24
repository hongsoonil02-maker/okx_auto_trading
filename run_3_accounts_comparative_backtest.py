#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_3_accounts_comparative_backtest.py
OKX 3개 계정(전략) 성과 비교 정밀 백테스트
1. 계정 1 (dontworry): Major Strategy (Top 10 대형 코인, 1h봉, 10x 레버리지, Flip on Close)
2. 계정 2 (freedom01): Venture Strategy (알트/밈 20종, 15m봉, 5x 레버리지, Trailing Stop)
3. 메인 계정 (main): 1 + 2 Dual Engine (Major 50% + Venture 50% 동시 분산 운용)
"""

import asyncio
import os
import sys
import time
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt_async

# 공통 설정
INITIAL_EQUITY = 3000.0  # 각 계정 초기 자본 3,000 USDT 동일 표준화
FEE_RATE = 0.0005        # 편도 0.05% (OKX Taker 기준 보수적 적용)

# 심볼 유니버스
MAJOR_SYMBOLS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT',
    'ADA/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT', 'DOT/USDT:USDT',
    'SUI/USDT:USDT', 'TRX/USDT:USDT'
]

VENTURE_SYMBOLS = [
    'DOGE/USDT:USDT', 'SHIB/USDT:USDT', 'PEPE/USDT:USDT', 'NEAR/USDT:USDT',
    'APT/USDT:USDT', 'WIF/USDT:USDT', 'ARB/USDT:USDT', 'OP/USDT:USDT',
    'INJ/USDT:USDT', 'LDO/USDT:USDT', 'ETC/USDT:USDT', 'TIA/USDT:USDT',
    'RENDER/USDT:USDT', 'ENA/USDT:USDT', 'SEI/USDT:USDT', 'BONK/USDT:USDT',
    'HBAR/USDT:USDT', 'CRV/USDT:USDT', 'AAVE/USDT:USDT', 'FET/USDT:USDT'
]

def calc_supertrend(df, period=10, multiplier=3.0):
    hl2 = (df['h'] + df['l']) / 2
    h, l, c = df['h'], df['l'], df['c']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    fu = hl2 + multiplier * atr
    fl = hl2 - multiplier * atr
    sd = pd.Series(1, index=df.index, dtype='int')
    sv = pd.Series(0.0, index=df.index, dtype='float64')
    for i in range(period, len(df)):
        if df['c'].iloc[i] > fu.iloc[i - 1]:
            sd.iloc[i] = 1
        elif df['c'].iloc[i] < fl.iloc[i - 1]:
            sd.iloc[i] = -1
        else:
            sd.iloc[i] = sd.iloc[i - 1]
            if sd.iloc[i] == 1 and fl.iloc[i] < fl.iloc[i - 1]:
                fl.iloc[i] = fl.iloc[i - 1]
            if sd.iloc[i] == -1 and fu.iloc[i] > fu.iloc[i - 1]:
                fu.iloc[i] = fu.iloc[i - 1]
        sv.iloc[i] = fl.iloc[i] if sd.iloc[i] == 1 else fu.iloc[i]
    return sd, sv, atr

def calc_stoch_rsi(series, period=14, smooth_k=3):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.fillna(50)
    stoch = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min()).replace(0, np.nan)
    stoch = stoch.fillna(0.5)
    k = stoch.rolling(smooth_k).mean() * 100
    return k

def calc_adx(df, period=14):
    h, l, c = df['h'], df['l'], df['c']
    up, dn = h.diff(), -l.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    mdi = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(20.0)

async def fetch_ohlcv_history(ex, sym, tf, since_ms):
    all_rows = []
    since = since_ms
    for _ in range(12):  # 최대 3,600개 캔들
        try:
            batch = await ex.fetch_ohlcv(sym, tf, since=since, limit=300)
            if not batch:
                break
            all_rows.extend(batch)
            if len(batch) < 300:
                break
            since = batch[-1][0] + 1
            await asyncio.sleep(0.04)
        except Exception:
            break
    if not all_rows:
        return None
    df = pd.DataFrame(all_rows, columns=['t', 'o', 'h', 'l', 'c', 'v']).drop_duplicates('t')
    df = df.sort_values('t').reset_index(drop=True)
    if len(df) < 50:
        return None
    df['st_dir'], df['st_val'], df['atr'] = calc_supertrend(df)
    df['stoch_k'] = calc_stoch_rsi(df['c'])
    df['adx'] = calc_adx(df)
    return df

class StrategyBacktest:
    def __init__(self, mode: str, equity: float = 3000.0):
        self.mode = mode  # 'major', 'venture', 'dual'
        self.equity = equity
        self.initial_equity = equity
        self.trades = []
        self.equity_curve = []
        
        # 모델별 파라미터
        if mode == 'major':
            self.leverage = 10
            self.max_positions = 5
            self.pos_size_pct = 0.18
            self.hard_sl = -0.30
            self.trailing_k = 2.0
            self.flip_on_close = True
            self.adx_threshold = 20.0
        elif mode == 'venture':
            self.leverage = 5
            self.max_positions = 8
            self.pos_size_pct = 0.12
            self.hard_sl = -0.25
            self.trailing_k = 3.0
            self.flip_on_close = False
            self.adx_threshold = 18.0
        else:  # dual
            self.leverage_major = 10
            self.leverage_venture = 5
            self.max_positions = 8
            self.pos_size_pct = 0.12
            self.hard_sl = -0.28
            self.adx_threshold = 19.0

    def run(self, major_dfs: dict, venture_dfs: dict):
        positions = {}
        all_timestamps = set()
        
        if self.mode in ('major', 'dual'):
            for df in major_dfs.values():
                all_timestamps.update(df['t'])
        if self.mode in ('venture', 'dual'):
            for df in venture_dfs.values():
                all_timestamps.update(df['t'])
                
        sorted_ts = sorted(all_timestamps)
        
        # 각 캔들 시점별 인덱스 캐시
        major_pointers = {sym: 0 for sym in major_dfs}
        venture_pointers = {sym: 0 for sym in venture_dfs}
        
        for t in sorted_ts:
            # 1. 포지션 업데이트 및 청산 평가
            to_close = []
            for (sym, pos_type), pos in positions.items():
                is_major = sym in major_dfs
                df = major_dfs[sym] if is_major else venture_dfs[sym]
                ptr = major_pointers[sym] if is_major else venture_pointers[sym]
                
                # 현재 시점의 봉 찾기
                if ptr < len(df) and df['t'].iloc[ptr] <= t:
                    row = df.iloc[ptr]
                    curr_price = row['c']
                    high_price = row['h']
                    low_price = row['l']
                    atr = row['atr']
                    st_dir = row['st_dir']
                    
                    # 최고/최저가 추적
                    if pos['side'] == 'long':
                        pos['peak'] = max(pos['peak'], high_price)
                        unrealized_pct = (curr_price - pos['entry_price']) / pos['entry_price'] * pos['lev']
                        drawdown_from_peak = (pos['peak'] - curr_price) / pos['entry_price'] * pos['lev']
                        
                        # 청산 조건 1: 하드 손절
                        if unrealized_pct <= self.hard_sl:
                            to_close.append(((sym, pos_type), curr_price, 'hard_sl'))
                        # 청산 조건 2: 트레일링 스탑
                        elif pos['peak'] > pos['entry_price'] * 1.01 and drawdown_from_peak >= (atr * 2.5 / pos['entry_price'] * pos['lev']):
                            to_close.append(((sym, pos_type), curr_price, 'trailing_stop'))
                        # 청산 조건 3: Supertrend 방향 전환
                        elif st_dir == -1:
                            to_close.append(((sym, pos_type), curr_price, 'trend_flip'))
                            
                    elif pos['side'] == 'short':
                        pos['peak'] = min(pos['peak'], low_price)
                        unrealized_pct = (pos['entry_price'] - curr_price) / pos['entry_price'] * pos['lev']
                        drawdown_from_peak = (curr_price - pos['peak']) / pos['entry_price'] * pos['lev']
                        
                        if unrealized_pct <= self.hard_sl:
                            to_close.append(((sym, pos_type), curr_price, 'hard_sl'))
                        elif pos['peak'] < pos['entry_price'] * 0.99 and drawdown_from_peak >= (atr * 2.5 / pos['entry_price'] * pos['lev']):
                            to_close.append(((sym, pos_type), curr_price, 'trailing_stop'))
                        elif st_dir == 1:
                            to_close.append(((sym, pos_type), curr_price, 'trend_flip'))
                            
            # 청산 실행
            for (sym, pos_type), exit_price, reason in to_close:
                pos = positions.pop((sym, pos_type))
                if pos['side'] == 'long':
                    raw_pct = (exit_price - pos['entry_price']) / pos['entry_price']
                else:
                    raw_pct = (pos['entry_price'] - exit_price) / pos['entry_price']
                    
                net_pnl = pos['margin'] * raw_pct * pos['lev']
                fee = (pos['margin'] * pos['lev'] * 2) * FEE_RATE
                realized_pnl = net_pnl - fee
                
                self.equity += realized_pnl
                self.trades.append({
                    'symbol': sym,
                    'side': pos['side'],
                    'entry_t': pos['entry_t'],
                    'exit_t': t,
                    'entry_px': pos['entry_price'],
                    'exit_px': exit_price,
                    'margin': pos['margin'],
                    'pnl': realized_pnl,
                    'pnl_pct': (realized_pnl / pos['margin']) * 100,
                    'reason': reason,
                    'is_major': sym in major_dfs,
                })
                
            # 2. 신규 진입 평가
            # Major 진입 검토
            if self.mode in ('major', 'dual') and len(positions) < (self.max_positions if self.mode != 'dual' else 8):
                for sym, df in major_dfs.items():
                    if (sym, 'major') in positions:
                        continue
                    ptr = major_pointers[sym]
                    if ptr < len(df) and df['t'].iloc[ptr] == t and ptr > 20:
                        row = df.iloc[ptr]
                        prev_row = df.iloc[ptr - 1]
                        
                        # ADX 필터
                        if row['adx'] < 20:
                            continue
                            
                        # Supertrend + StochRSI 시그널
                        long_signal = (row['st_dir'] == 1 and prev_row['stoch_k'] <= 25 and row['stoch_k'] > 25)
                        short_signal = (row['st_dir'] == -1 and prev_row['stoch_k'] >= 75 and row['stoch_k'] < 75)
                        
                        lev = 10
                        alloc_pct = 0.15 if self.mode == 'major' else 0.10
                        margin = max(30.0, self.equity * alloc_pct)
                        
                        if long_signal and self.equity > margin:
                            positions[(sym, 'major')] = {
                                'side': 'long', 'entry_price': row['c'], 'entry_t': t,
                                'margin': margin, 'lev': lev, 'peak': row['h']
                            }
                        elif short_signal and self.equity > margin:
                            positions[(sym, 'major')] = {
                                'side': 'short', 'entry_price': row['c'], 'entry_t': t,
                                'margin': margin, 'lev': lev, 'peak': row['l']
                            }
                            
            # Venture 진입 검토
            if self.mode in ('venture', 'dual') and len(positions) < (self.max_positions if self.mode != 'venture' else 8):
                for sym, df in venture_dfs.items():
                    if (sym, 'venture') in positions:
                        continue
                    ptr = venture_pointers[sym]
                    if ptr < len(df) and df['t'].iloc[ptr] == t and ptr > 20:
                        row = df.iloc[ptr]
                        prev_row = df.iloc[ptr - 1]
                        
                        # 변동성 필터
                        if row['adx'] < 18:
                            continue
                            
                        # 모멘텀 돌파 시그널
                        long_signal = (row['st_dir'] == 1 and prev_row['stoch_k'] <= 30 and row['stoch_k'] > 30 and row['c'] > row['o'])
                        short_signal = (row['st_dir'] == -1 and prev_row['stoch_k'] >= 70 and row['stoch_k'] < 70 and row['c'] < row['o'])
                        
                        lev = 5
                        alloc_pct = 0.10 if self.mode == 'venture' else 0.08
                        margin = max(25.0, self.equity * alloc_pct)
                        
                        if long_signal and self.equity > margin:
                            positions[(sym, 'venture')] = {
                                'side': 'long', 'entry_price': row['c'], 'entry_t': t,
                                'margin': margin, 'lev': lev, 'peak': row['h']
                            }
                        elif short_signal and self.equity > margin:
                            positions[(sym, 'venture')] = {
                                'side': 'short', 'entry_price': row['c'], 'entry_t': t,
                                'margin': margin, 'lev': lev, 'peak': row['l']
                            }

            # 포인터 갱신
            for sym, df in major_dfs.items():
                if major_pointers[sym] < len(df) and df['t'].iloc[major_pointers[sym]] <= t:
                    major_pointers[sym] += 1
            for sym, df in venture_dfs.items():
                if venture_pointers[sym] < len(df) and df['t'].iloc[venture_pointers[sym]] <= t:
                    venture_pointers[sym] += 1
                    
            self.equity_curve.append({'t': t, 'equity': self.equity})

    def get_metrics(self):
        if not self.trades:
            return {'total_trades': 0}
            
        pnls = [t['pnl'] for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        win_rate = len(wins) / len(pnls) * 100
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 1e-6
        profit_factor = gross_profit / gross_loss
        
        total_pnl = self.equity - self.initial_equity
        total_return_pct = (total_pnl / self.initial_equity) * 100
        
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        
        # MDD 계산
        eqs = [e['equity'] for e in self.equity_curve]
        peak = -1e9
        mdd = 0.0
        for eq in eqs:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > mdd:
                mdd = dd
                
        # 샤프 지수
        returns = pd.Series([t['pnl_pct'] for t in self.trades])
        sharpe = (returns.mean() / returns.std() * np.sqrt(365)) if len(returns) > 1 and returns.std() > 0 else 0.0

        return {
            'mode': self.mode,
            'initial_equity': self.initial_equity,
            'final_equity': self.equity,
            'total_return_pct': total_return_pct,
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'payoff_ratio': payoff,
            'mdd_pct': mdd,
            'sharpe_ratio': sharpe,
        }

async def main():
    print("=" * 75)
    print("🚀 [OKX 3개 계정 전략 비교 백테스트] 캔들 데이터 수집 및 시뮬레이션 시작")
    print("=" * 75)
    
    ex = ccxt_async.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    # 최근 90일 데이터 수집
    days = 90
    since_ms = int((time.time() - days * 86400) * 1000)
    
    major_dfs = {}
    venture_dfs = {}
    
    try:
        print(f"📥 1. Major 전략 대상 코인 (1h봉, 10종) 90일 캔들 수집 중...")
        for sym in MAJOR_SYMBOLS:
            try:
                df = await fetch_ohlcv_history(ex, sym, '1h', since_ms)
                if df is not None:
                    major_dfs[sym] = df
                    print(f"   - {sym:18s}: {len(df)} 캔들 로드 완료")
            except Exception as e:
                print(f"   ⚠️ {sym} 수집 실패: {e}")
                
        print(f"\n📥 2. Venture 전략 대상 알트코인 (15m봉, 20종) 90일 캔들 수집 중...")
        for sym in VENTURE_SYMBOLS:
            try:
                df = await fetch_ohlcv_history(ex, sym, '15m', since_ms)
                if df is not None:
                    venture_dfs[sym] = df
                    print(f"   - {sym:18s}: {len(df)} 캔들 로드 완료")
            except Exception as e:
                print(f"   ⚠️ {sym} 수집 실패: {e}")
                
    finally:
        await ex.close()
        
    print(f"\n📊 데이터 수집 완료: Major {len(major_dfs)}개 / Venture {len(venture_dfs)}개")
    print("-" * 75)
    
    # 3개 계정 시뮬레이션 실행
    # 1. Major (dontworry)
    print("⚙️ [계정 1: dontworry (Major Strategy)] 백테스트 시뮬레이션 실행 중...")
    bt_major = StrategyBacktest(mode='major', equity=3000.0)
    bt_major.run(major_dfs, venture_dfs)
    res_major = bt_major.get_metrics()
    
    # 2. Venture (freedom01)
    print("⚙️ [계정 2: freedom01 (Venture Strategy)] 백테스트 시뮬레이션 실행 중...")
    bt_venture = StrategyBacktest(mode='venture', equity=3000.0)
    bt_venture.run(major_dfs, venture_dfs)
    res_venture = bt_venture.get_metrics()
    
    # 3. Dual (main)
    print("⚙️ [계정 3: Main (1+2 Dual Engine)] 백테스트 시뮬레이션 실행 중...")
    bt_dual = StrategyBacktest(mode='dual', equity=3000.0)
    bt_dual.run(major_dfs, venture_dfs)
    res_dual = bt_dual.get_metrics()
    
    # 결과 비교표 출력
    print("\n" + "=" * 85)
    print("🏆 [OKX 3개 계정 전략 백테스트 성과 종합 비교 (최근 90일 기준)]")
    print("=" * 85)
    
    header = f"{'지표 항목':<25} | {'서브 1: dontworry (Major)':<22} | {'서브 2: freedom01 (Venture)':<24} | {'메인 계정 (1+2 Dual)':<22}"
    print(header)
    print("-" * 85)
    
    rows = [
        ("초기 자본", f"{res_major['initial_equity']:,.0f} USDT", f"{res_venture['initial_equity']:,.0f} USDT", f"{res_dual['initial_equity']:,.0f} USDT"),
        ("최종 자산", f"{res_major['final_equity']:,.2f} USDT", f"{res_venture['final_equity']:,.2f} USDT", f"{res_dual['final_equity']:,.2f} USDT"),
        ("누적 수익률", f"{res_major['total_return_pct']:+.2f}%", f"{res_venture['total_return_pct']:+.2f}%", f"{res_dual['total_return_pct']:+.2f}%"),
        ("총 거래 횟수", f"{res_major['total_trades']} 회", f"{res_venture['total_trades']} 회", f"{res_dual['total_trades']} 회"),
        ("승률 (Win Rate)", f"{res_major['win_rate']:.1f}% ({res_major['wins']}승/{res_major['losses']}패)", f"{res_venture['win_rate']:.1f}% ({res_venture['wins']}승/{res_venture['losses']}패)", f"{res_dual['win_rate']:.1f}% ({res_dual['wins']}승/{res_dual['losses']}패)"),
        ("손익비 (Profit Factor)", f"{res_major['profit_factor']:.2f}", f"{res_venture['profit_factor']:.2f}", f"{res_dual['profit_factor']:.2f}"),
        ("평균 이익 / 평균 손실", f"+{res_major['avg_win']:.1f} / {res_major['avg_loss']:.1f} USDT", f"+{res_venture['avg_win']:.1f} / {res_venture['avg_loss']:.1f} USDT", f"+{res_dual['avg_win']:.1f} / {res_dual['avg_loss']:.1f} USDT"),
        ("손익비율 (Payoff Ratio)", f"{res_major['payoff_ratio']:.2f}배", f"{res_venture['payoff_ratio']:.2f}배", f"{res_dual['payoff_ratio']:.2f}배"),
        ("최대 낙폭 (MDD)", f"-{res_major['mdd_pct']:.2f}%", f"-{res_venture['mdd_pct']:.2f}%", f"-{res_dual['mdd_pct']:.2f}%"),
        ("샤프 지수 (Sharpe)", f"{res_major['sharpe_ratio']:.2f}", f"{res_venture['sharpe_ratio']:.2f}", f"{res_dual['sharpe_ratio']:.2f}"),
    ]
    
    for title, m, v, d in rows:
        print(f"{title:<25} | {m:<22} | {v:<24} | {d:<22}")
    print("=" * 85)
    
    # JSON 파일로 저장
    out = {
        'timestamp': datetime.now().isoformat(),
        'period_days': days,
        'dontworry_major': res_major,
        'freedom01_venture': res_venture,
        'main_dual': res_dual,
    }
    with open('reports/okx_3accounts_comparison_result.json', 'w', encoding='utf-8') as f:
        import json
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n✅ 성과 비교 결과가 'reports/okx_3accounts_comparison_result.json'에 저장되었습니다.")

if __name__ == '__main__':
    asyncio.run(main())
