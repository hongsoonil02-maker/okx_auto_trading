#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import os
import pandas as pd
import ccxt.async_support as ccxt_async
from datetime import datetime, timezone

# 설정값
SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
LEVERAGES = [10, 20, 50, 100]
LEV_FEE = 0.0005
INITIAL_EQUITY = 10000.0

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

async def fetch_1h_data(ex, sym, limit=4000):
    out = []
    # ~4000 hours is about 166 days
    since = int((datetime.now(timezone.utc).timestamp() - limit * 3600) * 1000)
    for _ in range(40):
        batch = await ex.fetch_ohlcv(sym, '1h', since=since, limit=300)
        if not batch: break
        out.extend(batch)
        if len(batch) < 300: break
        since = batch[-1][0] + 1
        await asyncio.sleep(0.1)
    df = pd.DataFrame(out, columns=['t', 'o', 'h', 'l', 'c', 'v']).drop_duplicates('t')
    df = df.sort_values('t').reset_index(drop=True)
    return df

def simulate_symbol(df, lev):
    cash = INITIAL_EQUITY
    pos = None
    trades = []
    liq_count = 0
    mdd = 0.0
    eq_peak = INITIAL_EQUITY
    eq_trough = INITIAL_EQUITY

    for i in range(1, len(df)):
        c = df['c'].iloc[i]
        sd = df['st_dir'].iloc[i]
        sd_prev = df['st_dir'].iloc[i-1]
        ema = df['ema50'].iloc[i]
        ema_prev = df['ema50'].iloc[i-1]
        ema_slope = ema - ema_prev

        is_sync_long = (sd == 1) and (ema_slope > 0)
        is_sync_short = (sd == -1) and (ema_slope < 0)

        if pos:
            if pos['dir'] == 'long':
                pnl_pct = (c - pos['entry']) / pos['entry'] * lev
            else:
                pnl_pct = (pos['entry'] - c) / pos['entry'] * lev

            if pnl_pct <= -0.90:
                liq_count += 1
                trades.append((df['t'].iloc[i], -pos['margin']))
                cash -= pos['margin']
                pos = None
                continue
            
            exit_long = pos['dir'] == 'long' and not is_sync_long
            exit_short = pos['dir'] == 'short' and not is_sync_short
            
            if exit_long or exit_short:
                gross = pos['margin'] * pnl_pct
                fee = pos['margin'] * lev * LEV_FEE
                trades.append((df['t'].iloc[i], gross - fee))
                cash += gross - fee
                pos = None
                continue
        
        if not pos:
            if is_sync_long:
                margin = cash * 0.95
                if margin < 10: continue
                cash -= margin + (margin * lev * LEV_FEE)
                pos = {'dir': 'long', 'entry': c, 'margin': margin}
            elif is_sync_short:
                margin = cash * 0.95
                if margin < 10: continue
                cash -= margin + (margin * lev * LEV_FEE)
                pos = {'dir': 'short', 'entry': c, 'margin': margin}
        
        equity = cash
        if pos:
            if pos['dir'] == 'long':
                p = (c - pos['entry']) / pos['entry'] * lev
            else:
                p = (pos['entry'] - c) / pos['entry'] * lev
            equity += pos['margin'] * (1 + p)
        
        if equity > eq_peak:
            eq_peak = equity
            eq_trough = equity
        elif equity < eq_trough:
            eq_trough = equity
            mdd = max(mdd, (eq_peak - eq_trough) / eq_peak * 100)

    if pos:
        if pos['dir'] == 'long':
            pnl_pct = (df['c'].iloc[-1] - pos['entry']) / pos['entry'] * lev
        else:
            pnl_pct = (pos['entry'] - df['c'].iloc[-1]) / pos['entry'] * lev
        gross = pos['margin'] * pnl_pct
        fee = pos['margin'] * lev * LEV_FEE
        cash += gross - fee
        trades.append((df['t'].iloc[-1], gross - fee))

    wins = sum(1 for _, p in trades if p > 0)
    total_trades = len(trades)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    total_return = ((cash - INITIAL_EQUITY) / INITIAL_EQUITY) * 100

    return {
        'total_trades': total_trades,
        'wins': wins,
        'win_rate': win_rate,
        'final_equity': cash,
        'total_return_pct': total_return,
        'mdd': mdd,
        'liq_count': liq_count
    }

async def main():
    ex = ccxt_async.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    results = []
    
    try:
        for sym in SYMBOLS:
            print(f"[{sym}] 데이터 수집 중...")
            df = await fetch_1h_data(ex, sym, limit=4000)
            
            df['st_dir'] = calc_supertrend(df)
            df['ema50'] = df['c'].ewm(span=50, adjust=False).mean()
            df = df.dropna().reset_index(drop=True)
            
            print(f"[{sym}] 백테스트 진행 중...")
            for lev in LEVERAGES:
                res = simulate_symbol(df, lev)
                res['symbol'] = sym
                res['leverage'] = lev
                results.append(res)
    finally:
        await ex.close()
    
    print("\n" + "="*80)
    print(f"{'Symbol':<15} | {'Lev':<4} | {'Trades':<6} | {'Win%':<6} | {'Return%':<10} | {'MDD%':<6} | {'Liq':<4}")
    print("-" * 80)
    for r in results:
        print(f"{r['symbol']:<15} | {r['leverage']:<4d} | {r['total_trades']:<6d} | {r['win_rate']:<6.1f} | {r['total_return_pct']:<10.1f} | {r['mdd']:<6.1f} | {r['liq_count']:<4d}")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
