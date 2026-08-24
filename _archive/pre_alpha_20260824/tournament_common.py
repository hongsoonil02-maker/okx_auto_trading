import asyncio
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt_async
import json
import os
import time

from strategy_common import calc_supertrend, calc_stoch_rsi

BACKTEST_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_HISTORY_FILE = os.path.join(BACKTEST_DIR, "tournament_results_history.json")
PROMOTED_CONFIG_FILE = os.path.join(BACKTEST_DIR, "promoted_strategy.json")


async def fetch_data(ex, symbol, timeframe="15m", limit=2000):
    try:
        ohlcv = await ex.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv or len(ohlcv) < 200:
            return None
        df = pd.DataFrame(ohlcv, columns=['t', 'o', 'h', 'l', 'c', 'v'])
        df['vol_ma'] = df['v'].rolling(20).mean()
        delta = df['c'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        df['stoch_k'] = ((rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())).rolling(3).mean() * 100
        return df
    except Exception as e:
        return None


async def fetch_data_batch(ex, symbols, timeframe="15m", limit=2000, delay=0.5):
    data_map = {}
    for sym in symbols:
        df = await fetch_data(ex, sym, timeframe, limit)
        if df is not None:
            data_map[sym] = df
        await asyncio.sleep(delay)
    return data_map


def simulate(df, ema_period=200, tight_mult=2.5, loose_mult=4.0, vol_mult=1.2, min_hold=3, max_dca=8, tp_thr=1.02, scale_out=True):
    trades = []
    long_pos = None
    short_pos = None
    df['st_d_t'], df['st_v_t'] = calc_supertrend(df, 10, tight_mult)
    df['st_d_l'], df['st_v_l'] = calc_supertrend(df, 10, loose_mult)
    df['ema_target'] = df['c'].ewm(span=ema_period, adjust=False).mean()
    df['ema_slope'] = df['ema_target'].diff(5)

    for i in range(250, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        vol_cond = curr['v'] > prev['vol_ma'] * vol_mult

        is_long_bo = prev['st_d_l'] == -1 and curr['st_d_l'] == 1
        is_short_bo = prev['st_d_l'] == 1 and curr['st_d_l'] == -1

        is_long_pb = curr['st_d_l'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
        is_short_pb = curr['st_d_l'] == -1 and prev['stoch_k'] > 80 and curr['stoch_k'] <= 80

        ls = 60 if curr['c'] > curr['ema_target'] else 0
        if is_long_bo:
            ls += 40
        if is_long_pb:
            ls += 40

        ss = 60 if curr['c'] < curr['ema_target'] else 0
        if is_short_bo:
            ss += 40
        if is_short_pb:
            ss += 40

        is_long_sig = (ls >= 100) and vol_cond and curr['ema_slope'] > 0
        is_short_sig = (ss >= 100) and vol_cond and curr['ema_slope'] < 0

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
                if (i - long_pos.get('first_i', i)) >= min_hold:
                    force_l = True

            if force_l:
                trades.append({'pnl': (curr['c'] - ep) / ep * long_pos['size'] * 10.0})
                long_pos = None
            elif close_l and long_pos['exit_count'] < max_dca:
                rem = max_dca - long_pos['exit_count']
                sf = long_pos['size'] / rem if rem > 0 else long_pos['size']
                trades.append({'pnl': (curr['c'] - ep) / ep * sf * 10.0})
                long_pos['size'] -= sf
                long_pos['exit_count'] += 1
                if long_pos['exit_count'] >= max_dca or long_pos['size'] < 0.001:
                    long_pos = None
            elif not close_l and long_pos['entry_count'] < max_dca:
                # 불타기/물타기 스텝 세분화 (가격이 평단에서 0.5% 이상 벗어날 때만 추가 진입)
                if abs(curr['c'] - ep) / ep >= 0.005:
                    long_pos['entry_count'] += 1
                    add = 1.0 / max_dca
                    long_pos['entry'] = (ep * long_pos['size'] + curr['c'] * add) / (long_pos['size'] + add)
                    long_pos['size'] += add

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
                if (i - short_pos.get('first_i', i)) >= min_hold:
                    force_s = True

            if force_s:
                trades.append({'pnl': (ep - curr['c']) / ep * short_pos['size'] * 10.0})
                short_pos = None
            elif close_s and short_pos['exit_count'] < max_dca:
                rem = max_dca - short_pos['exit_count']
                sf = short_pos['size'] / rem if rem > 0 else short_pos['size']
                trades.append({'pnl': (ep - curr['c']) / ep * sf * 10.0})
                short_pos['size'] -= sf
                short_pos['exit_count'] += 1
                if short_pos['exit_count'] >= max_dca or short_pos['size'] < 0.001:
                    short_pos = None
            elif not close_s and short_pos['entry_count'] < max_dca:
                # 불타기/물타기 스텝 세분화 (가격이 평단에서 0.5% 이상 벗어날 때만 추가 진입)
                if abs(curr['c'] - ep) / ep >= 0.005:
                    short_pos['entry_count'] += 1
                    add = 1.0 / max_dca
                    short_pos['entry'] = (ep * short_pos['size'] + curr['c'] * add) / (short_pos['size'] + add)
                    short_pos['size'] += add

        active = (1 if long_pos else 0) + (1 if short_pos else 0)
        if active >= 3:
            continue

        if not long_pos and is_long_sig:
            long_pos = {'entry': curr['c'], 'size': 1.0 / max_dca, 'entry_count': 1, 'exit_count': 0, 'first_i': i}
        if not short_pos and is_short_sig:
            short_pos = {'entry': curr['c'], 'size': 1.0 / max_dca, 'entry_count': 1, 'exit_count': 0, 'first_i': i}

    lc = df.iloc[-1]['c']
    if long_pos:
        trades.append({'pnl': (lc - long_pos['entry']) / long_pos['entry'] * long_pos['size'] * 10.0})
    if short_pos:
        trades.append({'pnl': (short_pos['entry'] - lc) / short_pos['entry'] * short_pos['size'] * 10.0})
    return trades


def simulate_vibe(df, bb_period=20, bb_std=2.5, rsi_period=14, tp_pct=0.005, dca_pct=0.005, max_dca=8):
    df['sma20'] = df['c'].rolling(bb_period).mean()
    df['std'] = df['c'].rolling(bb_period).std()
    df['upper_bb'] = df['sma20'] + (df['std'] * bb_std)
    df['lower_bb'] = df['sma20'] - (df['std'] * bb_std)

    delta = df['c'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    trades = []
    pos = None

    for i in range(50, len(df)):
        curr = df.iloc[i]

        is_long_sig = curr['c'] < curr['lower_bb'] and curr['rsi'] < 30
        is_short_sig = curr['c'] > curr['upper_bb'] and curr['rsi'] > 70

        if pos:
            ep = pos['entry']
            if pos['side'] == 'long':
                is_pft = curr['c'] >= ep * (1 + tp_pct)
                is_mean_rev = curr['c'] >= curr['sma20']
                if is_pft or is_mean_rev:
                    trades.append({'side': 'long', 'pnl': (curr['c'] - ep) / ep * pos['size'] * 50.0})
                    pos = None
                elif curr['c'] <= ep * (1 - dca_pct) and pos['dca_count'] < max_dca:
                    pos['dca_count'] += 1
                    add = 1.0 / max_dca
                    pos['entry'] = (ep * pos['size'] + curr['c'] * add) / (pos['size'] + add)
                    pos['size'] += add
            else:
                is_pft = curr['c'] <= ep * (1 - tp_pct)
                is_mean_rev = curr['c'] <= curr['sma20']
                if is_pft or is_mean_rev:
                    trades.append({'side': 'short', 'pnl': (ep - curr['c']) / ep * pos['size'] * 50.0})
                    pos = None
                elif curr['c'] >= ep * (1 + dca_pct) and pos['dca_count'] < max_dca:
                    pos['dca_count'] += 1
                    add = 1.0 / max_dca
                    pos['entry'] = (ep * pos['size'] + curr['c'] * add) / (pos['size'] + add)
                    pos['size'] += add
        else:
            if is_long_sig:
                pos = {'side': 'long', 'entry': curr['c'], 'size': 1.0 / max_dca, 'dca_count': 1}
            elif is_short_sig:
                pos = {'side': 'short', 'entry': curr['c'], 'size': 1.0 / max_dca, 'dca_count': 1}

    if pos:
        ep = pos['entry']
        lc = df.iloc[-1]['c']
        if pos['side'] == 'long':
            trades.append({'side': 'long', 'pnl': (lc - ep) / ep * pos['size'] * 50.0})
        else:
            trades.append({'side': 'short', 'pnl': (ep - lc) / ep * pos['size'] * 50.0})

    return trades


def detailed_stats(trades):
    if not trades:
        return {'ret': 0.0, 'n': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0}
    pnls = [t['pnl'] for t in trades]
    ret = sum(pnls) * 100
    wins = len([p for p in pnls if p > 0])
    losses = len(trades) - wins
    win_rate = (wins / len(trades)) * 100
    return {'ret': round(ret, 2), 'n': len(trades), 'wins': wins, 'losses': losses, 'win_rate': round(win_rate, 2)}


def stats(trades):
    if not trades:
        return {'ret': 0, 'n': 0}
    pnls = [t['pnl'] for t in trades]
    ret = sum(pnls) * 100
    return {'ret': round(ret, 2), 'n': len(trades)}


def run_tournament(agent_name, symbols, strategy_func, strategy_params, timeframe="15m", limit=2000):
    ex = ccxt_async.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    data_map = asyncio.run(fetch_data_batch(ex, symbols, timeframe, limit))
    asyncio.run(ex.close())

    if not data_map:
        return None

    total_ret = 0.0
    total_trades = 0
    total_wins = 0
    sym_stats = {}

    for sym, df in data_map.items():
        tr = strategy_func(df.copy(), **strategy_params)
        s = detailed_stats(tr)
        sym_stats[sym] = s
        total_ret += s['ret']
        total_trades += s['n']
        total_wins += s['wins']

    overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
    return {
        'agent_name': agent_name,
        'overall_ret': round(total_ret, 2),
        'total_trades': total_trades,
        'overall_win_rate': round(overall_win_rate, 2),
        'symbols': sym_stats,
    }


def save_results_history(results):
    history = []
    if os.path.exists(RESULTS_HISTORY_FILE):
        try:
            with open(RESULTS_HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except (json.JSONDecodeError, IOError):
            history = []

    history.append({
        'timestamp': time.time(),
        'datetime': pd.Timestamp.now().isoformat(),
        'results': results,
    })

    with open(RESULTS_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4, ensure_ascii=False)


def promote_strategy(best_result, params):
    promotion = {
        'timestamp': time.time(),
        'datetime': pd.Timestamp.now().isoformat(),
        'winning_agent': best_result['agent_name'],
        'winning_ret': best_result['overall_ret'],
        'winning_win_rate': best_result['overall_win_rate'],
        'winning_trades': best_result['total_trades'],
        'promoted_params': params,
    }

    with open(PROMOTED_CONFIG_FILE, 'w') as f:
        json.dump(promotion, f, indent=4, ensure_ascii=False)

    return promotion


def load_promoted_config():
    if not os.path.exists(PROMOTED_CONFIG_FILE):
        return None
    try:
        with open(PROMOTED_CONFIG_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None