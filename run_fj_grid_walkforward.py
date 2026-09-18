#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_fj_grid_walkforward.py — FJ 스타일 그리드 서치 + 워크포워드 OOS 검증 + 레짐 무매수 필터

1) FJ(C) 파라미터 그리드 서치 (2x~4x, 손절 -3~-8%, 불타기 트리거 0.10~0.20, ADX 15~25 등)
2) 워크포워드 OOS: 과거 구간에서 최적 파라미터 선택 -> 미래 구간(Out-of-Sample)에서 평가
3) 추가 검증:
   - BTC Buy & Hold 벤치마크 (구간별)
   - BTC 레짐 무매수 필터 (BTC가 200EMA 아래면 신규 진입 차단)
"""
import asyncio
import os
import pickle
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np
import ccxt.async_support as ccxt_async

INITIAL_EQUITY = 3250.0
FEE_RATE = 0.0005

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
    "BNB/USDT:USDT", "XRP/USDT:USDT", "ADA/USDT:USDT",
    "DOGE/USDT:USDT", "HYPE/USDT:USDT", "LINK/USDT:USDT",
    "AVAX/USDT:USDT", "ENA/USDT:USDT", "PEPE/USDT:USDT",
    "WIF/USDT:USDT", "SUI/USDT:USDT", "ZEC/USDT:USDT",
]

DATA_START = datetime(2026, 3, 16, 0, 0, tzinfo=timezone.utc)
DATA_END = datetime(2026, 9, 13, 0, 0, tzinfo=timezone.utc)
WINDOW_DAYS = 60
HOUR_MS = 3600 * 1000
DAY_MS = 24 * HOUR_MS


def log(msg):
    print(msg, flush=True)


def calc_supertrend(df, period=10, multiplier=3.0):
    hl2 = (df['h'] + df['l']) / 2
    h, l, c = df['h'], df['l'], df['c']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    fu = hl2 + multiplier * atr
    fl = hl2 - multiplier * atr
    sd = pd.Series(1, index=df.index, dtype='int')
    for i in range(period, len(df)):
        if c.iloc[i] > fu.iloc[i - 1]:
            sd.iloc[i] = 1
        elif c.iloc[i] < fl.iloc[i - 1]:
            sd.iloc[i] = -1
        else:
            sd.iloc[i] = sd.iloc[i - 1]
            if sd.iloc[i] == 1 and fl.iloc[i] < fl.iloc[i - 1]:
                fl.iloc[i] = fl.iloc[i - 1]
            if sd.iloc[i] == -1 and fu.iloc[i] > fu.iloc[i - 1]:
                fu.iloc[i] = fu.iloc[i - 1]
    return sd, atr


def calc_stoch_k(series, period=14, smooth=3):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    stoch = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    return stoch.rolling(smooth).mean() * 100


def calc_adx(df, period=14):
    h, l, c = df['h'], df['l'], df['c']
    up, down = h.diff(), -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr
    denom = (plus_di + minus_di).where((plus_di + minus_di) != 0)
    dx = 100 * (plus_di - minus_di).abs() / denom
    return dx.ewm(alpha=1.0 / period, adjust=False).mean().fillna(0.0)


async def fetch_all(ex, sym, tf, since_ms, end_ms):
    out = []
    since = since_ms
    for _ in range(120):
        try:
            batch = await ex.fetch_ohlcv(sym, tf, since=since, limit=300)
        except Exception:
            await asyncio.sleep(0.5)
            continue
        if not batch:
            break
        if batch[-1][0] >= end_ms:
            batch = [r for r in batch if r[0] <= end_ms]
            out.extend(batch)
            break
        out.extend(batch)
        if len(batch) < 300:
            break
        since = batch[-1][0] + 1
        await asyncio.sleep(0.03)
    df = pd.DataFrame(out, columns=['t', 'o', 'h', 'l', 'c', 'v']).drop_duplicates('t')
    return df.sort_values('t').reset_index(drop=True)


CACHE_FILE = "/home/hongsoonil02/quant_system/state/cache_backtest/fj_grid_15m_180d.pkl"


async def fetch_one(ex, sym, s_ms, e_ms):
    df = await fetch_all(ex, sym, "15m", s_ms, e_ms)
    if len(df) < 200:
        log(f"  ❌ {sym}: {len(df)} candles 부족")
        return None
    df['st_dir'], df['atr'] = calc_supertrend(df)
    df['stoch_k'] = calc_stoch_k(df['c'])
    df['ema50'] = df['c'].ewm(span=50, adjust=False).mean()
    df['adx'] = calc_adx(df)
    if sym == "BTC/USDT:USDT":
        df['ema200'] = df['c'].ewm(span=200, adjust=False).mean()
    df = df.dropna().reset_index(drop=True)
    log(f"  ✅ {sym}: {len(df)} candles")
    return df


async def get_data():
    s_ms = int(DATA_START.timestamp() * 1000)
    e_ms = int(DATA_END.timestamp() * 1000)
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                data = pickle.load(f)
            if all(sym in data for sym in SYMBOLS):
                log(f"📦 캐시 로드: {CACHE_FILE}")
                return data
        except Exception:
            pass
    ex = ccxt_async.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    data = {}
    try:
        sem = asyncio.Semaphore(5)
        async def guarded(sym):
            async with sem:
                return sym, await fetch_one(ex, sym, s_ms, e_ms)
        for sym, df in await asyncio.gather(*[guarded(s) for s in SYMBOLS]):
            if df is not None:
                data[sym] = df
    finally:
        await ex.close()
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(data, f)
    log(f"💾 캐시 저장: {CACHE_FILE}")
    return data


class Strategy:
    def __init__(self, name, leverage, max_pos, max_dca, hard_stop_pct,
                 soft_stop_pct, time_stop_hours, pyramiding,
                 pyramid_trigger_pct, pyramid_ratio,
                 entry_threshold, short_allowed, adx_filter, adx_min,
                 trailing_arm_pct, trailing_k, regime_filter=False):
        self.name = name
        self.leverage = leverage
        self.max_pos = max_pos
        self.max_dca = max_dca
        self.hard_stop_pct = hard_stop_pct
        self.soft_stop_pct = soft_stop_pct
        self.time_stop_hours = time_stop_hours
        self.pyramiding = pyramiding
        self.pyramid_trigger_pct = pyramid_trigger_pct
        self.pyramid_ratio = pyramid_ratio
        self.entry_threshold = entry_threshold
        self.short_allowed = short_allowed
        self.adx_filter = adx_filter
        self.adx_min = adx_min
        self.trailing_arm_pct = trailing_arm_pct
        self.trailing_k = trailing_k
        self.regime_filter = regime_filter


def build_idx(data):
    return {sym: {int(t): i for i, t in enumerate(df['t'])} for sym, df in data.items()}


def prep_arrays(data):
    """numpy-based hot-loop arrays. idx maps timestamp -> row in symbol df."""
    arr = {}
    for sym, df in data.items():
        idx = {int(t): i for i, t in enumerate(df['t'])}
        a = {
            'idx': idx,
            'c': df['c'].to_numpy(), 'h': df['h'].to_numpy(), 'l': df['l'].to_numpy(),
            'atr': df['atr'].to_numpy(), 'st': df['st_dir'].to_numpy(),
            'sk': df['stoch_k'].to_numpy(), 'ema': df['ema50'].to_numpy(),
            'adx': df['adx'].to_numpy(),
        }
        if 'ema200' in df.columns:
            a['ema200'] = df['ema200'].to_numpy()
        arr[sym] = a
    return arr


def simulate(arr, U, lo, hi, btc_a, btc_idx, strat):
    cash = INITIAL_EQUITY
    peak_equity = INITIAL_EQUITY
    max_dd = 0.0
    positions = {}
    active = set()
    trades = []

    symbols = [sym for sym in arr]

    for gi in range(lo + 100, hi):
        t = int(U[gi])

        btc_adx = 25.0
        btc_trend_ok = True
        if btc_a is not None:
            bi = btc_idx.get(t, -1)
            if bi >= 0:
                btc_adx = float(btc_a['adx'][bi])
                btc_trend_ok = float(btc_a['c'][bi]) > float(btc_a['ema200'][bi])

        for sym in symbols:
            a = arr[sym]
            i = a['idx'].get(t, -1)
            if i < 2:
                continue
            px = float(a['c'][i])
            atr_val = float(a['atr'][i])

            plist = positions.get(sym)
            if plist:
                remaining = []
                for p in plist:
                    p['last_px'] = px
                    pnl_pct = (px - p['entry']) / p['entry'] * strat.leverage * p['dir']
                    if pnl_pct > p.get('extreme', 0):
                        p['extreme'] = pnl_pct
                    best = p['extreme']
                    hours_held = (t - p['entry_t']) / HOUR_MS
                    exit_now = False

                    if pnl_pct <= strat.hard_stop_pct:
                        exit_now = True
                    elif strat.soft_stop_pct != 0 and pnl_pct <= strat.soft_stop_pct:
                        exit_now = True
                    elif strat.trailing_k > 0 and best >= strat.trailing_arm_pct:
                        if p['dir'] == 1:
                            highest = p.get('highest', p['entry'])
                            if float(a['h'][i]) > highest:
                                highest = float(a['h'][i])
                            p['highest'] = highest
                            trail_line = highest - strat.trailing_k * (p.get('entry_atr', atr_val) or atr_val)
                            if px < trail_line:
                                exit_now = True
                        else:
                            lowest = p.get('lowest', p['entry'])
                            if float(a['l'][i]) < lowest:
                                lowest = float(a['l'][i])
                            p['lowest'] = lowest
                            trail_line = lowest + strat.trailing_k * (p.get('entry_atr', atr_val) or atr_val)
                            if px > trail_line:
                                exit_now = True
                    if not exit_now:
                        if best >= 0.50 and pnl_pct <= best * 0.6:
                            exit_now = True
                        elif best >= 0.20 and pnl_pct <= 0.05:
                            exit_now = True
                    if not exit_now and strat.time_stop_hours > 0:
                        if hours_held >= strat.time_stop_hours and pnl_pct <= 0.02:
                            exit_now = True
                    if not exit_now and pnl_pct >= 0.80:
                        exit_now = True

                    if exit_now:
                        gross = p['margin'] * pnl_pct
                        fee = p['margin'] * strat.leverage * FEE_RATE * 2
                        net = gross - fee
                        cash += p['margin'] + net
                        trades.append({'pnl': net, 'margin': p['margin']})
                    else:
                        remaining.append(p)
                positions[sym] = remaining
                if not remaining:
                    active.discard(sym)
            else:
                remaining = []

            if strat.pyramiding and remaining:
                for p in remaining:
                    pnl_pct = (px - p['entry']) / p['entry'] * strat.leverage * p['dir']
                    if pnl_pct >= strat.pyramid_trigger_pct and not p.get('pyramided'):
                        pyr_margin = p['margin'] * strat.pyramid_ratio
                        if pyr_margin >= 30 and pyr_margin <= cash * 0.5:
                            fee = pyr_margin * strat.leverage * FEE_RATE
                            cash -= pyr_margin + fee
                            remaining.append({
                                'entry': px, 'margin': pyr_margin, 'dir': p['dir'],
                                'extreme': 0.0, 'last_px': px, 'entry_t': t,
                                'entry_atr': atr_val, 'pyramided': True,
                            })
                            p['pyramided'] = True
                            break

            if remaining:
                continue
            if strat.adx_filter and btc_adx < strat.adx_min:
                continue
            if strat.regime_filter and not btc_trend_ok:
                continue
            if len(active) >= strat.max_pos:
                continue

            long_score = 0
            ema50 = float(a['ema'][i])
            if px > ema50 > 0: long_score += 50
            if a['st'][i - 1] == -1 and a['st'][i] == 1: long_score += 40
            if a['st'][i] == 1 and a['sk'][i - 1] < 20 and a['sk'][i] >= 20: long_score += 30
            if a['st'][i] == 1 and a['st'][i - 1] == 1: long_score += 20
            if 20 < a['sk'][i] < 80: long_score += 20

            short_score = 0
            if strat.short_allowed:
                if px < ema50 and ema50 > 0: short_score += 50
                if a['st'][i - 1] == 1 and a['st'][i] == -1: short_score += 40
                if a['st'][i] == -1 and a['sk'][i - 1] > 80 and a['sk'][i] <= 80: short_score += 30
                if a['st'][i] == -1 and a['st'][i - 1] == -1: short_score += 20
                if 20 < a['sk'][i] < 80: short_score += 20

            entry_dir = 0
            if long_score >= strat.entry_threshold: entry_dir = 1
            elif short_score >= strat.entry_threshold + 20 and strat.short_allowed: entry_dir = -1
            if entry_dir == 0: continue

            equity = cash
            for s2 in positions:
                for p in positions[s2]:
                    pnl = (p['last_px'] - p['entry']) / p['entry'] * strat.leverage * p['dir']
                    equity += p['margin'] * (1 + pnl)
            margin = (equity / strat.max_pos) * 0.5
            if margin < 30 or margin > cash * 0.95: continue

            fee = margin * strat.leverage * FEE_RATE
            cash -= margin + fee
            positions[sym] = [{'entry': px, 'margin': margin, 'dir': entry_dir,
                               'extreme': 0.0, 'last_px': px, 'entry_t': t, 'entry_atr': atr_val}]
            active.add(sym)

        equity = cash
        for s2 in positions:
            for p in positions[s2]:
                pnl = (p['last_px'] - p['entry']) / p['entry'] * strat.leverage * p['dir']
                equity += p['margin'] * (1 + pnl)
        if equity > peak_equity: peak_equity = equity
        dd = (equity - peak_equity) / peak_equity
        if dd < max_dd: max_dd = dd

    equity = cash
    for s2 in positions:
        for p in positions[s2]:
            pnl = (p['last_px'] - p['entry']) / p['entry'] * strat.leverage * p['dir']
            net = p['margin'] * pnl - p['margin'] * strat.leverage * FEE_RATE * 2
            equity += p['margin'] + net
            trades.append({'pnl': net, 'margin': p['margin']})

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    wr = len(wins) / len(trades) * 100 if trades else 0
    aw = np.mean([t['pnl'] for t in wins]) if wins else 0
    al = abs(np.mean([t['pnl'] for t in losses])) if losses else 1
    roi = (equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100
    return {'equity': equity, 'roi_pct': roi, 'mdd_pct': max_dd * 100,
            'trades': len(trades), 'win_rate': wr, 'pl_ratio': aw / al if al else 0,
            'avg_win': aw, 'avg_loss': -(abs(np.mean([t['pnl'] for t in losses])) if losses else 0)}


def risk_adj(r):
    return r['roi_pct'] / max(abs(r['mdd_pct']), 1.0)


FJ_BASE = dict(
    name='FJ', leverage=3, max_pos=5, max_dca=0, hard_stop_pct=-0.05,
    soft_stop_pct=0.0, time_stop_hours=72, pyramiding=True,
    pyramid_trigger_pct=0.15, pyramid_ratio=0.50,
    entry_threshold=90, short_allowed=False, adx_filter=True, adx_min=20,
    trailing_arm_pct=0.06, trailing_k=2.0,
)


def make_strat(name, p):
    kw = dict(FJ_BASE)
    kw.update(p)
    kw['name'] = name
    return Strategy(**kw)


GRID = {}
for lev in (2, 3, 4):
    for stop in (-0.03, -0.05, -0.08):
        for pyra in (0.10, 0.15, 0.20):
            for thr in (80, 90):
                for adx in (15, 20, 25):
                    GRID[(lev, stop, pyra, thr, adx)] = dict(
                        leverage=lev, hard_stop_pct=stop, pyramid_trigger_pct=pyra,
                        entry_threshold=thr, adx_min=adx)


def grid_params():
    return GRID


def run_candidates(arr, U, lo, hi, btc_a, btc_idx, cands):
    out = []
    for name, params, extra in cands:
        st = make_strat(name, {**params, **extra})
        r = simulate(arr, U, lo, hi, btc_a, btc_idx, st)
        out.append((name, params, extra, r))
    return out


_ARR = None
_U = None
_BTC_A = None
_BTC_IDX = None


def init_worker(arr, U, btc_a, btc_idx):
    global _ARR, _U, _BTC_A, _BTC_IDX
    _ARR = arr
    _U = U
    _BTC_A = btc_a
    _BTC_IDX = btc_idx


def run_sim_worker(args):
    name, params, extra, lo, hi = args
    st = make_strat(name, {**params, **extra})
    r = simulate(_ARR, _U, lo, hi, _BTC_A, _BTC_IDX, st)
    return name, params, extra, r


def grid_search(arr, U, lo, hi, btc_a, btc_idx):
    tasks = [(k, p, {}, lo, hi) for k, p in GRID.items()]
    results = []
    with ProcessPoolExecutor(max_workers=2, initializer=init_worker,
                             initargs=(arr, U, btc_a, btc_idx)) as ex:
        for i, res in enumerate(ex.map(run_sim_worker, tasks, chunksize=4)):
            name, params, extra, r = res
            results.append((name, params, extra, r))
            if (i + 1) % 25 == 0:
                log(f"    grid {i+1}/{len(tasks)} done")
    results.sort(key=lambda x: risk_adj(x[3]), reverse=True)
    return results


def fmt_date(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime('%m-%d')


def main():
    np.seterr(divide='ignore', invalid='ignore')
    log("=" * 100)
    log("📊 FJ 그리드 서치 + 워크포워드 OOS + 레짐 무매수 필터 검증")
    log(f"   기간: {DATA_START:%Y-%m-%d} ~ {DATA_END:%Y-%m-%d} | 시드: ${INITIAL_EQUITY:,.0f} | 15m | 심볼 {len(SYMBOLS)}개")
    log("=" * 100)
    log("\n🔄 데이터 수집/로드 중...")
    data = asyncio.run(get_data())
    log(f"✅ {len(data)}개 심볼 확보\n")

    arr = prep_arrays(data)
    U = np.array(sorted(set().union(*[set(df['t']) for df in data.values()])), dtype=np.int64)
    btc_a = arr.get("BTC/USDT:USDT")
    btc_idx = btc_a['idx'] if btc_a else {}
    btc_df = data.get("BTC/USDT:USDT")

    def win_lo_hi(t_s, t_e):
        lo = int(np.searchsorted(U, [t_s], side='left')[0])
        hi = int(np.searchsorted(U, [t_e], side='right')[0])
        return lo, hi

    s_ms = int(DATA_START.timestamp() * 1000)
    e_ms = int(DATA_END.timestamp() * 1000)
    seg_ts = [s_ms + i * WINDOW_DAYS * DAY_MS for i in range(3 + 1)]
    log("구간 분할:")
    for i in range(3):
        log(f"  seg{i+1}: {fmt_date(seg_ts[i])} ~ {fmt_date(seg_ts[i+1])}")

    # BTC Buy & Hold per segment
    log("\n📈 BTC Buy & Hold 벤치마크")
    bt = data["BTC/USDT:USDT"]
    for i in range(3):
        a = bt[(bt['t'] >= seg_ts[i]) & (bt['t'] < seg_ts[i + 1])]
        if len(a) >= 2:
            ret = (a['c'].iloc[-1] / a['c'].iloc[0] - 1) * 100
            log(f"  seg{i+1} B&H: {ret:+.1f}%")

    gsize = len(grid_params())
    log(f"\n🔍 그리드 서치 (총 {gsize}개 조합) — 2개 워커 병렬")

    folds = [
        ("Fold-1", seg_ts[0], seg_ts[1], seg_ts[1], seg_ts[2], "seg1 -> seg2"),
        ("Fold-2", seg_ts[0], seg_ts[2], seg_ts[2], seg_ts[3], "seg1+2 -> seg3"),
    ]

    tr_lohi = {f[0]: win_lo_hi(f[1], f[2]) for f in folds}
    te_lohi = {f[0]: win_lo_hi(f[3], f[4]) for f in folds}

    all_rows = []
    for fname, tr_s, tr_e, te_s, te_e, desc in folds:
        tr_lo, tr_hi = tr_lohi[fname]
        te_lo, te_hi = te_lohi[fname]
        log(f"\n{'='*90}\n{'>'} {fname}: train {desc} 구간 그리드 최적화 시작...")
        top = grid_search(arr, U, tr_lo, tr_hi, btc_a, btc_idx)
        log(f"  ✅ train 그리드 완료. Top-3:")
        for name, params, _, r in top[:3]:
            log(f"     {name} | roi {r['roi_pct']:+.1f}% mdd {r['mdd_pct']:+.1f}% ra {risk_adj(r):.3f}")

        # out-of-sample evaluation on test window
        cands = []
        oos_top = []
        for rank, (name, params, _, r) in enumerate(top[:3], 1):
            cands.append((f"OOS-{rank}", params, {}))
            oos_top.append((rank, params))
        cands.append(("FJ-기준(C)", {}, {}))
        for rank, params in oos_top:
            cands.append((f"OOS-{rank}-레짐", params, {"regime_filter": True}))

        log(f"\n  🔎 Test(Out-of-Sample) 평가: {fmt_date(te_s)} ~ {fmt_date(te_e)}")
        log(f"  {'전략':16s} {'수익률':>8s} {'MDD':>8s} {'거래수':>6s} {'승률':>7s} {'손익비':>7s} {'리스크조정':>9s}")
        for name, params, extra, r in run_candidates(arr, U, te_lo, te_hi, btc_a, btc_idx, cands):
            log(f"  {name:16s} {r['roi_pct']:>+7.1f}% {r['mdd_pct']:>+7.1f}% {r['trades']:>5d} {r['win_rate']:>6.1f}% {r['pl_ratio']:>6.2f}x {risk_adj(r):>8.3f}")
            all_rows.append((fname, name, params, r))

    log("\n" + "=" * 90)
    log("📋 워크포워드 OOS 최종 요약")
    log(f"  {'Fold':8s} {'전략':18s} {'수익률':>8s} {'MDD':>8s} {'승률':>7s} {'손익비':>7s} {'리스크조정':>9s}")
    for fname, name, params, r in all_rows:
        log(f"  {fname:8s} {name:18s} {r['roi_pct']:>+7.1f}% {r['mdd_pct']:>+7.1f}% {r['win_rate']:>6.1f}% {r['pl_ratio']:>6.2f}x {risk_adj(r):>8.3f}")

    log("\n📌 Fold별 OOS Top-1 파라미터")
    for fname in ("Fold-1", "Fold-2"):
        best = next((x for x in all_rows if x[0] == fname and x[1] == "OOS-1"), None)
        if best:
            log(f"  {fname}: {best[2]}")

    log("\n" + "=" * 90)
    log("전체 데이터(seg1~seg3, ~180일)에서 후보들 최종 비교...")
    final_cands = [("FJ-기준(C)", {}, {}), ("FJ-기준+레짐무매수", {}, {"regime_filter": True})]
    for fname in ("Fold-1", "Fold-2"):
        best = next((x for x in all_rows if x[0] == fname and x[1] == "OOS-1"), None)
        if best:
            final_cands.append((f"{fname}-OOS-1", best[2], {}))
            final_cands.append((f"{fname}-OOS-1+레짐", best[2], {"regime_filter": True}))
    f_lo, f_hi = win_lo_hi(s_ms, e_ms)
    rows = run_candidates(arr, U, f_lo, f_hi, btc_a, btc_idx, final_cands)
    log(f"  {'전략':18s} {'수익률':>8s} {'MDD':>8s} {'거래수':>6s} {'승률':>7s} {'손익비':>7s} {'리스크조정':>9s}")
    for name, _, _, r in rows:
        log(f"  {name:22s} {r['roi_pct']:>+7.1f}% {r['mdd_pct']:>+7.1f}% {r['trades']:>5d} {r['win_rate']:>6.1f}% {r['pl_ratio']:>6.2f}x {risk_adj(r):>8.3f}")
    log("\n✅ 검증 완료")


if __name__ == "__main__":
    main()