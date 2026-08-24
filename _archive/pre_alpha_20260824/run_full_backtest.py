#!/usr/bin/env python3
"""Before/After 비교 백테스트 — 개선안 4가지 적용 효과 측정"""
import asyncio, os, sys, json, time
import pandas as pd, numpy as np
import ccxt.async_support as ccxt_async

def calc_supertrend(df, period=10, multiplier=3.0):
    hl2 = (df['h'] + df['l']) / 2
    atr = (df['h'].combine(df['c'].shift(), max) - df['l'].combine(df['c'].shift(), min)).rolling(period).mean()
    fu = hl2 + (multiplier * atr); fl = hl2 - (multiplier * atr)
    sd = pd.Series(1, index=df.index, dtype='int'); sv = pd.Series(0.0, index=df.index, dtype='float64')
    for i in range(period, len(df)):
        if df['c'].iloc[i] > fu.iloc[i-1]: sd.iloc[i] = 1
        elif df['c'].iloc[i] < fl.iloc[i-1]: sd.iloc[i] = -1
        else:
            sd.iloc[i] = sd.iloc[i-1]
            if sd.iloc[i] == 1 and fl.iloc[i] < fl.iloc[i-1]: fl.iloc[i] = fl.iloc[i-1]
            if sd.iloc[i] == -1 and fu.iloc[i] > fu.iloc[i-1]: fu.iloc[i] = fu.iloc[i-1]
        sv.iloc[i] = fl.iloc[i] if sd.iloc[i] == 1 else fu.iloc[i]
    return sd, sv

def calc_stoch_rsi(series, period=14, sk=3, sd=3):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    sr = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    k = sr.rolling(sk).mean() * 100; d = k.rolling(sd).mean()
    return k, d

def simulate(df, tight_mult, profit_thr, vol_mult, min_hold_candles, max_dca=3):
    """Unified simulator with configurable parameters for Before/After comparison."""
    trades = []
    long_pos = None; short_pos = None

    # Pre-calculate tight supertrend
    df['st_d_t'], df['st_v_t'] = calc_supertrend(df, 10, tight_mult)

    for i in range(20, len(df)):
        prev = df.iloc[i-1]; curr = df.iloc[i]
        vol_cond = curr['v'] > prev['vol_ma'] * vol_mult

        is_long_bo = prev['st_d_l'] == -1 and curr['st_d_l'] == 1 and vol_cond
        is_long_pb = curr['st_d_l'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
        is_long_sig = is_long_bo or is_long_pb

        is_short_bo = prev['st_d_l'] == 1 and curr['st_d_l'] == -1 and vol_cond
        is_short_pb = curr['st_d_l'] == -1 and prev['stoch_k'] > 80 and curr['stoch_k'] <= 80
        is_short_sig = is_short_bo or is_short_pb

        # ── Long ──
        if long_pos:
            ep = long_pos['entry']
            is_pft = curr['c'] > ep * profit_thr
            stv = curr['st_v_t'] if is_pft else curr['st_v_l']
            std = curr['st_d_t'] if is_pft else curr['st_d_l']
            close_l = std == -1 or curr['c'] < stv
            force_l = False
            if long_pos['exit_count'] > 0 and curr['c'] < ep:
                candles = i - long_pos.get('first_i', i)
                if candles >= min_hold_candles:
                    force_l = True

            if force_l:
                pnl = (curr['c'] - ep) / ep * long_pos['size']
                trades.append({'side': 'long', 'pnl_pct': pnl, 'type': 'be_stop'})
                long_pos = None
            elif close_l and long_pos['exit_count'] < max_dca:
                rem = max_dca - long_pos['exit_count']
                sf = long_pos['size'] / rem if rem > 0 else long_pos['size']
                pnl = (curr['c'] - ep) / ep * sf
                trades.append({'side': 'long', 'pnl_pct': pnl, 'type': 'dca_exit'})
                long_pos['size'] -= sf; long_pos['exit_count'] += 1
                if long_pos['exit_count'] >= max_dca or long_pos['size'] < 0.001: long_pos = None
            elif not close_l and long_pos['entry_count'] < max_dca:
                long_pos['entry_count'] += 1
                add = 1.0 / max_dca
                na = (long_pos['entry'] * long_pos['size'] + curr['c'] * add) / (long_pos['size'] + add)
                long_pos['entry'] = na; long_pos['size'] += add

        # ── Short ──
        if short_pos:
            ep = short_pos['entry']
            is_pft = curr['c'] < ep * (2.0 - profit_thr)
            stv = curr['st_v_t'] if is_pft else curr['st_v_l']
            std = curr['st_d_t'] if is_pft else curr['st_d_l']
            close_s = std == 1 or curr['c'] > stv
            force_s = False
            if short_pos['exit_count'] > 0 and curr['c'] > ep:
                candles = i - short_pos.get('first_i', i)
                if candles >= min_hold_candles:
                    force_s = True

            if force_s:
                pnl = (ep - curr['c']) / ep * short_pos['size']
                trades.append({'side': 'short', 'pnl_pct': pnl, 'type': 'be_stop'})
                short_pos = None
            elif close_s and short_pos['exit_count'] < max_dca:
                rem = max_dca - short_pos['exit_count']
                sf = short_pos['size'] / rem if rem > 0 else short_pos['size']
                pnl = (ep - curr['c']) / ep * sf
                trades.append({'side': 'short', 'pnl_pct': pnl, 'type': 'dca_exit'})
                short_pos['size'] -= sf; short_pos['exit_count'] += 1
                if short_pos['exit_count'] >= max_dca or short_pos['size'] < 0.001: short_pos = None

        # ── New entry ──
        active = (1 if long_pos else 0) + (1 if short_pos else 0)
        if active >= 3: continue
        if not long_pos and is_long_sig:
            long_pos = {'entry': curr['c'], 'size': 1.0/max_dca, 'entry_count': 1, 'exit_count': 0, 'first_i': i}
        if not short_pos and is_short_sig:
            short_pos = {'entry': curr['c'], 'size': 1.0/max_dca, 'entry_count': 1, 'exit_count': 0, 'first_i': i}

    # Close remaining
    lc = df.iloc[-1]['c']
    if long_pos: trades.append({'side': 'long', 'pnl_pct': (lc - long_pos['entry'])/long_pos['entry']*long_pos['size'], 'type': 'eod'})
    if short_pos: trades.append({'side': 'short', 'pnl_pct': (short_pos['entry'] - lc)/short_pos['entry']*short_pos['size'], 'type': 'eod'})
    return trades

def stats(trades, bh_ret):
    if not trades: return {'ret': 0, 'bh': bh_ret, 'alpha': -bh_ret, 'wr': 0, 'n': 0, 'be': 0, 'sharpe': 0, 'max_dd': 0}
    pnls = [t['pnl_pct'] for t in trades]
    ret = sum(pnls) * 100
    wins = sum(1 for p in pnls if p > 0)
    be = sum(1 for t in trades if t['type'] == 'be_stop')
    wr = wins / len(pnls) * 100
    sharpe = (np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls))) if len(pnls) > 1 and np.std(pnls) > 0 else 0
    # Max drawdown
    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    dd = (cumulative - peak)
    max_dd = float(np.min(dd)) * 100 if len(dd) > 0 else 0
    return {'ret': round(ret, 3), 'bh': round(bh_ret, 3), 'alpha': round(ret - bh_ret, 3),
            'wr': round(wr, 1), 'n': len(trades), 'be': be, 'sharpe': round(sharpe, 2), 'max_dd': round(max_dd, 3)}

async def main():
    print("=" * 90)
    print("  ARCA — Before / After 비교 백테스트 (개선안 4가지 적용)")
    print("=" * 90)

    configs = [
        {"bot": "Venture (Meme/Alt)",
         "symbols": ["DOGE/USDT:USDT", "PEPE/USDT:USDT", "WIF/USDT:USDT", "SHIB/USDT:USDT"],
         "before": {"tight": 2.0, "profit": 1.025, "vol": 1.2, "hold": 0},
         "after":  {"tight": 2.5, "profit": 1.025, "vol": 1.0, "hold": 3}},
        {"bot": "Stock Venture",
         "symbols": ["AAPL/USDT:USDT", "TSLA/USDT:USDT", "NVDA/USDT:USDT"],
         "before": {"tight": 2.5, "profit": 1.02, "vol": 1.2, "hold": 0},
         "after":  {"tight": 2.5, "profit": 1.02, "vol": 1.1, "hold": 3}},
    ]

    exchange = ccxt_async.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

    try:
        for cfg in configs:
            print(f"\n{'─'*90}")
            print(f"  📊 {cfg['bot']}")
            print(f"  Before: tight={cfg['before']['tight']}, vol_mult={cfg['before']['vol']}, min_hold={cfg['before']['hold']}")
            print(f"  After:  tight={cfg['after']['tight']}, vol_mult={cfg['after']['vol']}, min_hold={cfg['after']['hold']}")
            print(f"{'─'*90}")
            print(f"  {'Symbol':<20} │ {'Metric':<12} │ {'BEFORE':>10} │ {'AFTER':>10} │ {'Δ':>10}")
            print(f"  {'─'*20}─┼─{'─'*12}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}")

            all_before = []; all_after = []

            for symbol in cfg['symbols']:
                try:
                    ohlcv = await exchange.fetch_ohlcv(symbol, '15m', limit=1440)
                    if not ohlcv or len(ohlcv) < 50:
                        print(f"  {symbol:<20} │ ⚠️ 데이터 부족"); continue

                    df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
                    df['st_d_l'], df['st_v_l'] = calc_supertrend(df, 10, 4.0)
                    df['vol_ma'] = df['v'].rolling(20).mean()
                    k, d = calc_stoch_rsi(df['c'], 14, 3, 3)
                    df['stoch_k'] = k; df['stoch_d'] = d

                    bh = (df.iloc[-1]['c'] - df.iloc[0]['c']) / df.iloc[0]['c'] * 100

                    tb = simulate(df.copy(), cfg['before']['tight'], cfg['before']['profit'], cfg['before']['vol'], cfg['before']['hold'])
                    ta = simulate(df.copy(), cfg['after']['tight'], cfg['after']['profit'], cfg['after']['vol'], cfg['after']['hold'])

                    sb = stats(tb, bh); sa = stats(ta, bh)
                    all_before.extend(tb); all_after.extend(ta)

                    metrics = [
                        ('Return %', 'ret'), ('Alpha %', 'alpha'), ('Win Rate %', 'wr'),
                        ('Trades', 'n'), ('BE Stops', 'be'), ('Sharpe', 'sharpe'), ('Max DD %', 'max_dd')
                    ]
                    for j, (label, key) in enumerate(metrics):
                        sym_label = symbol.split('/')[0] if j == 0 else ''
                        bv = sb[key]; av = sa[key]; dv = av - bv
                        icon = "🟢" if dv > 0 else ("🔴" if dv < 0 else "⚪")
                        if key in ('be',) and dv < 0: icon = "🟢"  # fewer BE stops = better
                        if key == 'max_dd' and dv > 0: icon = "🟢"  # less negative DD = better
                        print(f"  {sym_label:<20} │ {label:<12} │ {bv:>10.2f} │ {av:>10.2f} │ {dv:>+9.2f} {icon}")
                    print(f"  {'─'*20}─┼─{'─'*12}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}")

                except Exception as e:
                    print(f"  {symbol:<20} │ ❌ {e}")
                await asyncio.sleep(0.3)

            # ── 합산 ──
            if all_before and all_after:
                bh_avg = 0  # 합산이므로 의미 없음
                sb_all = stats(all_before, bh_avg); sa_all = stats(all_after, bh_avg)
                print(f"  {'★ 합산':<20} │ {'Return %':<12} │ {sb_all['ret']:>10.2f} │ {sa_all['ret']:>10.2f} │ {sa_all['ret']-sb_all['ret']:>+9.2f} {'🟢' if sa_all['ret'] > sb_all['ret'] else '🔴'}")
                print(f"  {'':<20} │ {'Win Rate %':<12} │ {sb_all['wr']:>10.1f} │ {sa_all['wr']:>10.1f} │ {sa_all['wr']-sb_all['wr']:>+9.1f} {'🟢' if sa_all['wr'] > sb_all['wr'] else '🔴'}")
                print(f"  {'':<20} │ {'Trades':<12} │ {sb_all['n']:>10d} │ {sa_all['n']:>10d} │ {sa_all['n']-sb_all['n']:>+9d}")
                print(f"  {'':<20} │ {'BE Stops':<12} │ {sb_all['be']:>10d} │ {sa_all['be']:>10d} │ {sa_all['be']-sb_all['be']:>+9d} {'🟢' if sa_all['be'] <= sb_all['be'] else '🔴'}")
                print(f"  {'':<20} │ {'Sharpe':<12} │ {sb_all['sharpe']:>10.2f} │ {sa_all['sharpe']:>10.2f} │ {sa_all['sharpe']-sb_all['sharpe']:>+9.2f} {'🟢' if sa_all['sharpe'] > sb_all['sharpe'] else '🔴'}")
                print(f"  {'':<20} │ {'Max DD %':<12} │ {sb_all['max_dd']:>10.2f} │ {sa_all['max_dd']:>10.2f} │ {sa_all['max_dd']-sb_all['max_dd']:>+9.2f} {'🟢' if sa_all['max_dd'] > sb_all['max_dd'] else '🔴'}")

    finally:
        await exchange.close()

    print(f"\n{'='*90}")
    print("  개선안 요약:")
    print("  #1 Volume 확인 완화: Venture 1.2→1.0 | Stock 1.2→1.1")
    print("  #2 Venture Tight ST: 2.0→2.5 (whipsaw 감소)")
    print("  #3 Breakeven 최소 보유: 0→3캔들 (45분)")
    print(f"{'='*90}")

if __name__ == "__main__":
    asyncio.run(main())
