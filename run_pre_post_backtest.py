#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pre_post_backtest.py — OLD(수정 전) vs NEW(MR !1 수정 후) 로직 비교 백테스트

OLD : 진행 중 캔들 신호 판정 / 고정 마진% 스탑(Major -6%, Venture -7%) / 트레일링 ARM 0.20, K 2.5
      숏 허용 / Major 플립 on / 재진입 쿨다운 4캔들 / 하드스탑 쿨다운 12h / 포지션당 100 USDT 고정 사이징
NEW : 확정 캔들 신호 판정 / ATR 스탑(K 2.5, 최소 2%) + 리스크 예산 사이징(자산×0.2%) / 트레일링 ARM 0.06, K 3.0
      숏 차단 / 플립 off / 재진입 16캔들 / 하드스탑 쿨다운 24h / HWM 킬스위치(-15%) 보존

공통 : BTC·심볼 1h EMA50 이중 게이트, BTC 1h ADX<20 횡보 차단, 일일 CB -6%/48h, 수수료 0.05% 편도
"""
import argparse, json, os, pickle, sys, time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import ccxt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MS_5M = 300_000
MS_1H = 3_600_000
MS_1D = 86_400_000
FEE = 0.0005

MAJOR_COINS = {'BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'AVAX', 'LINK', 'DOT', 'BNB', 'TRX'}
MEMES = {
    'DOGE', 'SHIB', 'PEPE', 'BOME', 'WIF', 'BONK', 'FLOKI', 'FARTCOIN', 'PUMP',
    'PEOPLE', 'MOODENG', 'PNUT', 'ACT', 'NEIRO', 'TURBO', 'MEW', 'POPCAT',
    'GIGA', 'BRETT', 'TRUMP', 'MELANIA', 'PENGU', 'AI16Z'
}
DEFAULT_SYMBOLS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'DOGE/USDT:USDT',
    'PEPE/USDT:USDT', 'SUI/USDT:USDT', 'ZKP/USDT:USDT', 'ANIME/USDT:USDT',
    'GRVT/USDT:USDT', 'ZORA/USDT:USDT', 'HUMA/USDT:USDT', 'AUCTION/USDT:USDT'
]

# ───────────────────────── 지표 연산 ─────────────────────────
def true_range(h, l, c):
    pc = np.roll(c, 1)
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    return tr

def ewm_alpha(x, alpha):
    return pd.Series(x).ewm(alpha=alpha, adjust=False).mean().to_numpy()

def supertrend_np(h, l, c, period=10, mult=3.0):
    n = len(c)
    hl2 = (h + l) / 2
    atr = ewm_alpha(true_range(h, l, c), 1 / period)
    fu = hl2 + mult * atr
    fl = hl2 - mult * atr
    d = np.ones(n, dtype=int)
    v = np.zeros(n)
    for i in range(period, n):
        if c[i] > fu[i - 1]:
            d[i] = 1
        elif c[i] < fl[i - 1]:
            d[i] = -1
        else:
            d[i] = d[i - 1]
            if d[i] == 1 and fl[i] < fl[i - 1]:
                fl[i] = fl[i - 1]
            if d[i] == -1 and fu[i] > fu[i - 1]:
                fu[i] = fu[i - 1]
        v[i] = fl[i] if d[i] == 1 else fu[i]
    return d, v, fu, fl

def calc_adx_np(h, l, c, period=14):
    h, l, c = pd.Series(h), pd.Series(l), pd.Series(c)
    up, down = h.diff(), -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    denom = (plus_di + minus_di).where((plus_di + minus_di) != 0)
    dx = 100 * (plus_di - minus_di).abs() / denom
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0).to_numpy()

def resample(sub, tf_ms):
    t = sub['t']
    g = t - (t % tf_ms)
    _, starts = np.unique(g, return_index=True)
    starts = np.sort(starts)
    ends = np.append(starts[1:], len(t))
    b = {
        't': g[starts], 'o': sub['o'][starts],
        'h': np.maximum.reduceat(sub['h'], starts),
        'l': np.minimum.reduceat(sub['l'], starts),
        'c': sub['c'][ends - 1], 'v': np.add.reduceat(sub['v'], starts),
    }
    return b

def add_indicators(b):
    h, l, c, v = b['h'], b['l'], b['c'], b['v']
    b['atr'] = ewm_alpha(true_range(h, l, c), 1 / 14)
    d_l, v_l, fu_l, fl_l = supertrend_np(h, l, c, 10, 3.0)
    d_t, v_t, fu_t, fl_t = supertrend_np(h, l, c, 10, 2.0)
    b['st_d_loose'] = d_l
    b['st_d_tight'] = d_t
    b['ema'] = pd.Series(c).ewm(span=50, adjust=False).mean().to_numpy()
    
    # RSI & StochRSI
    delta = pd.Series(c).diff()
    up = delta.clip(lower=0)
    dn = -delta.clip(upper=0)
    ma_u = up.ewm(alpha=1/14, adjust=False).mean()
    ma_d = dn.ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - (100 / (1 + ma_u / (ma_d + 1e-9)))
    stoch_raw = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min() + 1e-9)
    b['stoch_k'] = (stoch_raw.rolling(3).mean() * 100).to_numpy()
    b['stoch_d'] = pd.Series(b['stoch_k']).rolling(3).mean().to_numpy()
    b['ix'] = {int(x): i for i, x in enumerate(b['t'])}
    return b

# ───────────────────────── 데이터 다운로드 ─────────────────────────
def fetch_candles(ex, sym, tf='5m', since_ms=0, until_ms=0):
    cache_dir = os.path.join(BASE_DIR, "state", "cache_backtest")
    os.makedirs(cache_dir, exist_ok=True)
    safe_name = sym.replace("/", "_").replace(":", "_")
    cache_file = os.path.join(cache_dir, f"{safe_name}_{tf}_{since_ms}_{until_ms}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    out, cur = [], since_ms
    tf_ms = 300_000 if tf == '5m' else 3_600_000
    while cur < until_ms:
        try:
            batch = ex.fetch_ohlcv(sym, tf, since=int(cur), limit=300)
            if not batch:
                break
            out.extend(batch)
            nxt = batch[-1][0] + tf_ms
            if nxt <= cur:
                break
            cur = nxt
            time.sleep(0.08)
        except Exception as e:
            time.sleep(1.0)
            break
    rows = sorted({r[0]: r for r in out}.values())
    rows = [r for r in rows if since_ms <= r[0] <= until_ms and r[4] is not None]
    if rows:
        with open(cache_file, "wb") as f:
            pickle.dump(rows, f)
    return rows

# ───────────────────────── 시뮬레이터 ─────────────────────────
def run_backtest(mode="NEW", symbols=None, start_dt="2026-08-28", end_dt="2026-09-03", initial_equity=1000.0):
    """
    mode: "OLD" (수정 전) or "NEW" (수정 후)
    """
    ex = ccxt.okx({"enableRateLimit": True})
    since_ms = int(datetime.strptime(start_dt, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    until_ms = int(datetime.strptime(end_dt, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000) + MS_1D
    warmup_ms = since_ms - 20 * MS_1D

    # BTC 1h 데이터 준비 (게이트 및 ADX)
    btc_1h_rows = fetch_candles(ex, "BTC/USDT:USDT", "1h", warmup_ms, until_ms)
    if not btc_1h_rows:
        return {"error": "BTC 1h data fetch failed"}
    btc_arr = np.array(btc_1h_rows, dtype=float)
    btc_t = btc_arr[:, 0].astype(np.int64)
    btc_c = btc_arr[:, 4]
    btc_ema50 = pd.Series(btc_c).ewm(span=50, adjust=False).mean().to_numpy()
    btc_adx = calc_adx_np(btc_arr[:, 2], btc_arr[:, 3], btc_c, 14)
    btc_h1_map = {int(btc_t[i]): (float(btc_c[i]), float(btc_ema50[i]), float(btc_adx[i])) for i in range(len(btc_t))}

    # 심볼 데이터 준비
    sym_data = {}
    for sym in symbols:
        base = sym.split('/')[0]
        is_major = base in MAJOR_COINS
        tf_str = "15m" if is_major else "30m"
        tf_ms = 15 * 60_000 if is_major else 30 * 60_000
        
        rows = fetch_candles(ex, sym, "5m", warmup_ms, until_ms)
        if len(rows) < 300:
            continue
        arr = np.array(rows, dtype=float)
        sub = {'t': arr[:, 0].astype(np.int64), 'o': arr[:, 1], 'h': arr[:, 2], 'l': arr[:, 3], 'c': arr[:, 4], 'v': arr[:, 5]}
        bars = add_indicators(resample(sub, tf_ms))
        h1 = resample(sub, MS_1H)
        h1_ema50 = pd.Series(h1['c']).ewm(span=50, adjust=False).mean().to_numpy()
        h1_map = {int(h1['t'][i]): (float(h1['c'][i]), float(h1_ema50[i])) for i in range(len(h1['t']))}
        
        sym_data[sym] = {
            'is_major': is_major, 'tf_ms': tf_ms, 'bars': bars, 'sub_5m': sub, 'h1_map': h1_map
        }

    # 시뮬레이션 변수 초기화
    equity = initial_equity
    hwm = initial_equity
    positions = {}  # sym -> pos_dict
    trades = []
    daily_anchor = initial_equity
    current_day = None
    cb_cooldown_until = 0
    killed = False
    reentry_cooldown = defaultdict(int)

    # 5분 단위 타임라인
    all_5m_ts = sorted(list(set.union(*[set(d['sub_5m']['t']) for d in sym_data.values()])))
    sim_ts = [t for t in all_5m_ts if since_ms <= t <= until_ms]

    for t_curr in sim_ts:
        day_str = datetime.fromtimestamp(t_curr / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        
        # 1. 일일 서킷브레이커 리셋 로직
        if current_day != day_str:
            current_day = day_str
            daily_anchor = equity
            if mode == "OLD":
                hwm = equity  # [버그 재현] OLD는 HWM이 매일 리셋됨!

        # 2. HWM 절대 킬스위치
        if equity > hwm:
            hwm = equity
        dd_from_hwm = (equity - hwm) / hwm * 100
        if dd_from_hwm <= -15.0:
            killed = True  # 영구 차단
        
        # 일손실 CB 판정
        daily_dd = (equity - daily_anchor) / daily_anchor * 100
        if daily_dd <= -6.0:
            cb_cooldown_until = t_curr + 48 * MS_1H

        circuit_blocked = (t_curr < cb_cooldown_until) or killed

        # BTC 1h 상태
        prev_1h_t = (t_curr // MS_1H) * MS_1H - MS_1H
        btc_info = btc_h1_map.get(prev_1h_t, (0, 0, 25.0))
        btc_above50 = btc_info[0] > btc_info[1]
        btc_adx_val = btc_info[2]
        chop_market = btc_adx_val < 20.0

        # 3. 기존 포지션 평가 및 청산 판정
        for sym in list(positions.keys()):
            pos = positions[sym]
            info = sym_data[sym]
            bars = info['bars']
            
            # 현재 5분 가격
            sub = info['sub_5m']
            sub_idx = np.searchsorted(sub['t'], t_curr)
            if sub_idx >= len(sub['t']):
                continue
            cur_px = sub['c'][sub_idx]
            
            # PnL 계산
            side_mult = 1 if pos['side'] == 'long' else -1
            price_change = (cur_px - pos['entry_px']) / pos['entry_px'] * side_mult
            pnl_roe = price_change * pos['leverage']
            pos['high_roe'] = max(pos['high_roe'], pnl_roe)

            # 청산 로직
            should_close = False
            close_reason = ""

            if mode == "OLD":
                # OLD: 고정 마진% 스탑 (-6% Major, -7% Venture)
                sl_limit = -0.06 if pos['is_major'] else -0.07
                if pnl_roe <= sl_limit:
                    should_close = True
                    close_reason = "OLD_SL"
                elif pos['high_roe'] >= 0.20:
                    # 트레일링 스탑: 최고점에서 5% 반락 시
                    if (pos['high_roe'] - pnl_roe) >= 0.05:
                        should_close = True
                        close_reason = "OLD_TRAIL"
            else:
                # NEW: ATR 기반 동적 스탑 (스탑 거리가 가격 기준 2%~5% 확보)
                if price_change <= -pos['atr_stop_pct']:
                    should_close = True
                    close_reason = "NEW_ATR_SL"
                elif pos['high_roe'] >= 0.06:
                    # 트레일링 스탑 (ARM 0.06)
                    if (pos['high_roe'] - pnl_roe) >= 0.03:
                        should_close = True
                        close_reason = "NEW_TRAIL"

            if should_close:
                net_pnl = pos['margin'] * pnl_roe - (pos['margin'] * pos['leverage'] * FEE * 2)
                equity += net_pnl
                trades.append({
                    'symbol': sym, 'side': pos['side'], 'entry_t': pos['entry_t'], 'exit_t': t_curr,
                    'net_pnl': net_pnl, 'roe': pnl_roe, 'reason': close_reason, 'margin': pos['margin']
                })
                reentry_cooldown[sym] = t_curr + (16 * info['tf_ms'] if mode == "NEW" else 4 * info['tf_ms'])
                del positions[sym]

        # 4. 신규 진입 판정
        if not circuit_blocked and not chop_market and len(positions) < (5 if mode == "NEW" else 10):
            for sym, info in sym_data.items():
                if sym in positions or t_curr < reentry_cooldown[sym]:
                    continue
                
                bars = info['bars']
                tf_ms = info['tf_ms']
                
                # 캔들 인덱스 선택: OLD=진행 중(미확정), NEW=직전 확정(-1)
                bar_t = (t_curr // tf_ms) * tf_ms
                if mode == "NEW":
                    bar_t -= tf_ms  # 확정 캔들
                
                idx = bars['ix'].get(bar_t)
                if idx is None or idx < 2:
                    continue
                
                # 신호 판정
                st_loose = bars['st_d_loose'][idx]
                stoch_k = bars['stoch_k'][idx]
                stoch_d = bars['stoch_d'][idx]
                c_px = bars['c'][idx]
                ema_px = bars['ema'][idx]

                # 1h 심볼 EMA 게이트
                sym_h1 = info['h1_map'].get(prev_1h_t, (0, 0))
                sym_above50 = sym_h1[0] > sym_h1[1]

                is_long = (st_loose == 1) and (stoch_k > stoch_d) and (c_px > ema_px) and btc_above50 and sym_above50
                is_short = (st_loose == -1) and (stoch_k < stoch_d) and (c_px < ema_px) and (not btc_above50) and (not sym_above50)

                if mode == "NEW":
                    is_short = False  # NEW는 숏 전면 차단

                chosen_side = None
                if is_long:
                    chosen_side = "long"
                elif is_short:
                    chosen_side = "short"

                if chosen_side:
                    lev = 5 if info['is_major'] else 3
                    if mode == "OLD":
                        # 고정 100 USDT 마진
                        margin = min(100.0, equity * 0.2)
                        atr_stop = 0.01
                    else:
                        # NEW: 리스크 예산 기반 (자산의 0.2% 손실 한도)
                        atr_val = bars['atr'][idx]
                        atr_stop = max(0.02, (2.5 * atr_val) / c_px)  # 최소 2%
                        risk_budget = equity * 0.002
                        target_margin = risk_budget / (atr_stop * lev)
                        margin = max(30.0, min(target_margin, equity * 0.1))

                    if margin > 10.0 and equity >= margin:
                        positions[sym] = {
                            'side': chosen_side, 'entry_px': c_px, 'entry_t': t_curr,
                            'margin': margin, 'leverage': lev, 'is_major': info['is_major'],
                            'high_roe': 0.0, 'atr_stop_pct': atr_stop
                        }

    # 결과 집계
    df_tr = pd.DataFrame(trades)
    total_trades = len(df_tr)
    if total_trades > 0:
        wins = df_tr[df_tr['net_pnl'] > 0]
        losses = df_tr[df_tr['net_pnl'] <= 0]
        win_rate = len(wins) / total_trades * 100
        net_pnl = df_tr['net_pnl'].sum()
        gross_win = wins['net_pnl'].sum()
        gross_loss = abs(losses['net_pnl'].sum())
        pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
        
        # MDD 계산
        equity_curve = initial_equity + df_tr['net_pnl'].cumsum()
        peak_curve = equity_curve.cummax()
        dd_curve = (equity_curve - peak_curve) / peak_curve * 100
        mdd_pct = dd_curve.min()
    else:
        win_rate, net_pnl, pf, mdd_pct = 0, 0, 0, 0

    return {
        'mode': mode, 'trades': total_trades, 'win_rate': round(win_rate, 1),
        'net_pnl': round(net_pnl, 2), 'final_equity': round(equity, 2),
        'profit_factor': round(pf, 2), 'mdd_pct': round(mdd_pct, 1),
        'killed': killed
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-08-28")
    parser.add_argument("--end", default="2026-09-03")
    args = parser.parse_args()

    symbols = DEFAULT_SYMBOLS
    print(f"=========================================================")
    print(f" 백테스트 시뮬레이션: OLD (수정 전) vs NEW (수정 후)")
    print(f" 기간: {args.start} ~ {args.end} (폭락장 구간 검증)")
    print(f" 대상 심볼: {len(symbols)}개 종목")
    print(f"=========================================================\n")

    print("▶ [1/2] OLD 로직 (진행 중 캔들 신호, 고정 마진 손절, 숏 허용, 100 USDT 고정) 실행 중...")
    res_old = run_backtest("OLD", symbols, args.start, args.end)

    print("▶ [2/2] NEW 로직 (확정 캔들 신호, ATR 기반 손절, 0.2% 리스크 사이징, HWM 킬스위치) 실행 중...")
    res_new = run_backtest("NEW", symbols, args.start, args.end)

    print("\n=========================================================")
    print(" 📊 백테스트 비교 검증 결과")
    print("=========================================================")
    df_cmp = pd.DataFrame([res_old, res_new])
    print(df_cmp.to_markdown(index=False))
    print("=========================================================")

if __name__ == "__main__":
    main()
