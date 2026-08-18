#!/usr/bin/env python3
import asyncio, pandas as pd, numpy as np
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

def simulate(df, ema_period=200, tight_mult=2.5, loose_mult=4.0, vol_mult=1.2, min_hold=3, max_dca=12, tp_thr=1.02, scale_out=True):
    trades = []
    long_pos = None; short_pos = None
    df['st_d_t'], df['st_v_t'] = calc_supertrend(df, 10, tight_mult)
    df['st_d_l'], df['st_v_l'] = calc_supertrend(df, 10, loose_mult)
    df['ema_target'] = df['c'].ewm(span=ema_period, adjust=False).mean()
    df['ema_slope'] = df['ema_target'].diff(5)

    for i in range(250, len(df)):
        prev = df.iloc[i-1]; curr = df.iloc[i]
        vol_cond = curr['v'] > prev['vol_ma'] * vol_mult

        is_long_bo = prev['st_d_l'] == -1 and curr['st_d_l'] == 1
        is_short_bo = prev['st_d_l'] == 1 and curr['st_d_l'] == -1

        is_long_pb = curr['st_d_l'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
        is_short_pb = curr['st_d_l'] == -1 and prev['stoch_k'] > 80 and curr['stoch_k'] <= 80

        ls = 60 if curr['c'] > curr['ema_target'] else 0
        if is_long_bo: ls += 40
        if is_long_pb: ls += 40
        
        ss = 60 if curr['c'] < curr['ema_target'] else 0
        if is_short_bo: ss += 40
        if is_short_pb: ss += 40
        
        is_long_sig = (ls >= 100) and vol_cond and curr['ema_slope'] > 0
        is_short_sig = (ss >= 100) and vol_cond and curr['ema_slope'] < 0

        # ── Long ──
        if long_pos:
            ep = long_pos['entry']
            is_pft = curr['c'] > ep * tp_thr
            stv = curr['st_v_t'] if is_pft else curr['st_v_l']
            std = curr['st_d_t'] if is_pft else curr['st_d_l']
            
            # 24시간(96캔들) 보유 & 0.5% 미만 변동 시 강제 청산
            held_candles = i - long_pos.get('first_i', i)
            if held_candles >= 96 and abs(curr['c'] - ep) / ep < 0.005:
                trades.append({'pnl': (curr['c'] - ep) / ep * long_pos['size'] * 10.0})
                long_pos = None
                continue

            # Supertrend가 꺾여도 EMA 기울기가 급격히 꺾이지 않았으면 유지 (우상향 중이면 홀딩)
            st_close_l = std == -1 or curr['c'] < stv
            close_l = st_close_l and curr['ema_slope'] < 0
            force_l = False
            if long_pos['exit_count'] > 0 and curr['c'] < ep:
                if (i - long_pos.get('first_i', i)) >= min_hold: force_l = True

            if force_l:
                trades.append({'pnl': (curr['c'] - ep) / ep * long_pos['size'] * 10.0})
                long_pos = None
            elif close_l and long_pos['exit_count'] < max_dca:
                if scale_out:
                    rem = max_dca - long_pos['exit_count']
                    sf = long_pos['size'] / rem if rem > 0 else long_pos['size']
                    trades.append({'pnl': (curr['c'] - ep) / ep * sf * 10.0})
                    long_pos['size'] -= sf; long_pos['exit_count'] += 1
                    if long_pos['exit_count'] >= max_dca or long_pos['size'] < 0.001: long_pos = None
                else:
                    trades.append({'pnl': (curr['c'] - ep) / ep * long_pos['size'] * 10.0})
                    long_pos = None
            elif not close_l and long_pos['entry_count'] < max_dca:
                # 불타기/물타기 스텝 세분화 (가격이 평단에서 0.5% 이상 벗어날 때만 추가 진입)
                if abs(curr['c'] - ep) / ep >= 0.005:
                    long_pos['entry_count'] += 1
                    add = 1.0 / max_dca
                    long_pos['entry'] = (ep * long_pos['size'] + curr['c'] * add) / (long_pos['size'] + add)
                    long_pos['size'] += add

        # ── Short ──
        if short_pos:
            ep = short_pos['entry']
            is_pft = curr['c'] < ep * (2.0 - tp_thr)
            stv = curr['st_v_t'] if is_pft else curr['st_v_l']
            std = curr['st_d_t'] if is_pft else curr['st_d_l']
            
            # 24시간(96캔들) 보유 & 0.5% 미만 변동 시 강제 청산
            held_candles = i - short_pos.get('first_i', i)
            if held_candles >= 96 and abs(curr['c'] - ep) / ep < 0.005:
                trades.append({'pnl': (ep - curr['c']) / ep * short_pos['size'] * 10.0})
                short_pos = None
                continue

            # Supertrend가 꺾여도 EMA 기울기가 꺾이지 않았으면 유지 (우하향 중이면 홀딩)
            st_close_s = std == 1 or curr['c'] > stv
            close_s = st_close_s and curr['ema_slope'] > 0
            force_s = False
            if short_pos['exit_count'] > 0 and curr['c'] > ep:
                if (i - short_pos.get('first_i', i)) >= min_hold: force_s = True

            if force_s:
                trades.append({'pnl': (ep - curr['c']) / ep * short_pos['size'] * 10.0})
                short_pos = None
            elif close_s and short_pos['exit_count'] < max_dca:
                if scale_out:
                    rem = max_dca - short_pos['exit_count']
                    sf = short_pos['size'] / rem if rem > 0 else short_pos['size']
                    trades.append({'pnl': (ep - curr['c']) / ep * sf * 10.0})
                    short_pos['size'] -= sf; short_pos['exit_count'] += 1
                    if short_pos['exit_count'] >= max_dca or short_pos['size'] < 0.001: short_pos = None
                else:
                    trades.append({'pnl': (ep - curr['c']) / ep * short_pos['size'] * 10.0})
                    short_pos = None
            elif not close_s and short_pos['entry_count'] < max_dca:
                # 불타기/물타기 스텝 세분화 (가격이 평단에서 0.5% 이상 벗어날 때만 추가 진입)
                if abs(curr['c'] - ep) / ep >= 0.005:
                    short_pos['entry_count'] += 1
                    add = 1.0 / max_dca
                    short_pos['entry'] = (ep * short_pos['size'] + curr['c'] * add) / (short_pos['size'] + add)
                    short_pos['size'] += add

        active = (1 if long_pos else 0) + (1 if short_pos else 0)
        if active >= 3: continue
            
        if not long_pos and is_long_sig:
            long_pos = {'entry': curr['c'], 'size': 1.0/max_dca, 'entry_count': 1, 'exit_count': 0, 'first_i': i}
        if not short_pos and is_short_sig:
            short_pos = {'entry': curr['c'], 'size': 1.0/max_dca, 'entry_count': 1, 'exit_count': 0, 'first_i': i}

    lc = df.iloc[-1]['c']
    if long_pos: trades.append({'pnl': (lc - long_pos['entry'])/long_pos['entry']*long_pos['size'] * 10.0})
    if short_pos: trades.append({'pnl': (short_pos['entry'] - lc)/short_pos['entry']*short_pos['size'] * 10.0})
    return trades

def detailed_stats(trades):
    if not trades: return {'ret': 0, 'win_rate': 0, 'n': 0}
    pnls = [t['pnl'] for t in trades]
    ret = sum(pnls) * 100
    wins = [p for p in pnls if p > 0]
    wr = (len(wins) / len(pnls)) * 100
    return {'ret': round(ret, 2), 'win_rate': round(wr, 1), 'n': len(trades)}

async def main():
    ex = ccxt_async.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    
    print("="*60)
    print(" 🚀 Final System Backtest (2 Bots, 10x Leverage, Exact Settings)")
    print("="*60)
    
    async def process_bot(name, syms, tf, ema, tp, scale_out):
        print(f"\n[{name}] - {tf} TF, EMA {ema}, TP {tp}, ScaleOut: {scale_out}, DCA 8")
        print(f"{'Symbol':<15} | {'Leveraged Return (Trades)':<25}")
        for sym in syms:
            try:
                ohlcv = await ex.fetch_ohlcv(sym, tf, limit=2000)
                if not ohlcv or len(ohlcv) < 200:
                    continue
                df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
                df['vol_ma'] = df['v'].rolling(20).mean()
                delta = df['c'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss.replace(0, np.nan); rsi = 100 - (100 / (1 + rs))
                df['stoch_k'] = ((rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())).rolling(3).mean() * 100

                tr = simulate(df.copy(), ema_period=ema, tp_thr=tp, max_dca=8, scale_out=scale_out)
                s = detailed_stats(tr)
                print(f"{sym:<15} | Ret: {s['ret']:>6.2f}% | WR: {s['win_rate']:>5.1f}% | Trades: {s['n']:>2d}")
            except Exception as e:
                print(f"{sym:<15} | ❌ Error: {str(e)}")
            await asyncio.sleep(0.5)

    await process_bot("1. Major Crypto", ["BTC/USDT:USDT", "ETH/USDT:USDT"], "1h", 50, 1.01, False)
    await process_bot("2. Venture Alts", ["DOGE/USDT:USDT", "PEPE/USDT:USDT", "WIF/USDT:USDT"], "15m", 200, 1.02, True)

    await ex.close()

if __name__ == "__main__":
    asyncio.run(main())
