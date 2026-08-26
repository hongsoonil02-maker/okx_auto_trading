#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_param_compare_backtest.py — REENTRY / KELLY / PORTFOLIO_WEIGHT 조합 비교 백테스트
비교 구간: 2026-08-18 ~ 2026-08-26 (8일, 실제 손실 구간)

변형 A: 현재 설정 (8/18 배포 기준 — 문제 설정)
  REENTRY=True, KELLY=0.50, PORTFOLIO_WEIGHT=1.5

변형 B: 보수 복귀 (8/18 이전 설정)
  REENTRY=False, KELLY=0.25, PORTFOLIO_WEIGHT=1.0

변형 C: 절충안 (오늘 적용한 Score 상향 + 일부 보수화)
  REENTRY=False, KELLY=0.35, PORTFOLIO_WEIGHT=1.0, Score THR 80
"""
import asyncio
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt_async

# ── 공통 설정 ─────────────────────────────────────
LEVERAGE       = 10
FEE_RATE       = 0.0005   # 0.05% taker
INITIAL_EQUITY = 19000.0
MAX_POS        = 15
DCA_MAX        = 3
ADX_CHOP_THR   = 25       # 오늘부터 적용된 Chop 필터
HARD_STOP      = -0.30    # -30% 하드스탑 (마진 대비)
POS_LOSS_LIMIT = -0.15    # -15% 포지션 손실 한도

START_MS = int(datetime(2026, 8, 18, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
END_MS   = int(datetime(2026, 8, 26, 4, 0, tzinfo=timezone.utc).timestamp() * 1000)

SYMBOLS = [
    "CAP/USDT:USDT", "DOGE/USDT:USDT", "HYPE/USDT:USDT",
    "ZEC/USDT:USDT", "WIF/USDT:USDT", "FARTCOIN/USDT:USDT",
    "PEOPLE/USDT:USDT", "RE/USDT:USDT", "AAVE/USDT:USDT",
    "CRV/USDT:USDT", "BNB/USDT:USDT", "UNI/USDT:USDT",
    "STX/USDT:USDT", "VIRTUAL/USDT:USDT", "INJ/USDT:USDT",
]

VARIANTS = [
    {
        "label":  "A: 문제설정 (8/18배포)",
        "reentry": True,  "kelly": 0.50, "weight": 1.5,
        "score_thr": 70,  "chop_filter": False,
    },
    {
        "label":  "B: 보수복귀 (8/18이전)",
        "reentry": False, "kelly": 0.25, "weight": 1.0,
        "score_thr": 70,  "chop_filter": True,
    },
    {
        "label":  "C: 절충안 (오늘적용)",
        "reentry": False, "kelly": 0.35, "weight": 1.0,
        "score_thr": 80,  "chop_filter": True,
    },
]

# ── 지표 계산 ──────────────────────────────────────
def calc_supertrend(df, period=10, mult=3.0):
    hl2 = (df['h'] + df['l']) / 2
    tr  = pd.concat([df['h']-df['l'], (df['h']-df['c'].shift()).abs(),
                     (df['l']-df['c'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    fu, fl = hl2 + mult*atr, hl2 - mult*atr
    sd = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        sd.iloc[i] = 1 if df['c'].iloc[i] > fu.iloc[i-1] else (
                    -1 if df['c'].iloc[i] < fl.iloc[i-1] else sd.iloc[i-1])
        if sd.iloc[i] == 1 and fl.iloc[i] < fl.iloc[i-1]:  fl.iloc[i] = fl.iloc[i-1]
        if sd.iloc[i] ==-1 and fu.iloc[i] > fu.iloc[i-1]:  fu.iloc[i] = fu.iloc[i-1]
    return sd

def calc_adx(df, p=14):
    h, l, c = df['h'], df['l'], df['c']
    up, dn = h.diff(), -l.diff()
    pdm = up.where((up>dn)&(up>0), 0.0)
    mdm = dn.where((dn>up)&(dn>0), 0.0)
    tr  = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/p,adjust=False).mean()
    pdi = 100*pdm.ewm(alpha=1/p,adjust=False).mean()/atr
    mdi = 100*mdm.ewm(alpha=1/p,adjust=False).mean()/atr
    dx  = 100*(pdi-mdi).abs()/(pdi+mdi).where((pdi+mdi)!=0)
    return dx.ewm(alpha=1/p,adjust=False).mean().fillna(0)

def calc_rsi(s, p=14):
    d = s.diff()
    g = d.where(d>0,0).rolling(p).mean()
    l = (-d.where(d<0,0)).rolling(p).mean()
    return 100 - 100/(1+g/l.replace(0,1e-9))

# ── 데이터 수집 ────────────────────────────────────
async def fetch_ohlcv(ex, sym, tf, since_ms, end_ms):
    out, since = [], since_ms
    for _ in range(30):
        batch = await ex.fetch_ohlcv(sym, tf, since=since, limit=300)
        if not batch: break
        out.extend([r for r in batch if r[0] <= end_ms])
        if len(batch) < 300 or batch[-1][0] >= end_ms: break
        since = batch[-1][0] + 1
        await asyncio.sleep(0.05)
    if not out: return None
    df = pd.DataFrame(out, columns=['t','o','h','l','c','v']).drop_duplicates('t')
    return df.sort_values('t').reset_index(drop=True)

# ── 시뮬레이션 ─────────────────────────────────────
def simulate(data_map, btc_15m, variant):
    kelly   = variant['kelly']
    weight  = variant['weight']
    reentry = variant['reentry']
    thr     = variant['score_thr']
    chop    = variant['chop_filter']

    cash = INITIAL_EQUITY
    pos  = {}   # sym -> {margin, entry, dca_n, last_exit_t, side}
    trades_log = []
    total_fees = 0.0
    reentry_cooldown = {}  # sym -> last_exit_ts (ms)

    # BTC ADX 사전 계산
    if btc_15m is not None and len(btc_15m) > 14:
        btc_adx_s = calc_adx(btc_15m).values
        btc_ts    = btc_15m['t'].values
    else:
        btc_adx_s, btc_ts = None, None

    def get_btc_adx(ts):
        if btc_adx_s is None: return 30.0
        idx = np.searchsorted(btc_ts, ts) - 1
        return float(btc_adx_s[max(0, idx)])

    all_ts = sorted(set(t for df in data_map.values() for t in df['t'].tolist()))
    idx_map = {sym: 0 for sym in data_map}

    for ts in all_ts:
        adx_val = get_btc_adx(ts)
        chop_blocked = chop and (adx_val < ADX_CHOP_THR)

        for sym, df in data_map.items():
            # advance index
            while idx_map[sym] < len(df)-1 and df['t'].iloc[idx_map[sym]+1] <= ts:
                idx_map[sym] += 1
            i = idx_map[sym]
            if i < 20: continue
            row = df.iloc[:i+1]
            price = float(df['c'].iloc[i])

            # 지표
            st   = calc_supertrend(row).iloc[-1]
            rsi  = calc_rsi(row['c']).iloc[-1]
            adx  = calc_adx(row).iloc[-1]

            # 간이 Score (실제 봇의 70점 임계를 단순화)
            score = 50
            if st == 1: score += 15
            if rsi < 45: score += 10
            if adx > 20: score += 10
            if adx > 30: score += 5

            long_signal  = (score >= thr) and (st == 1)
            short_signal = False   # Venture는 롱 위주

            # ── 포지션 관리 ──
            if sym in pos:
                p   = pos[sym]
                pnl_pct = (price - p['entry']) / p['entry'] if p['side']=='long' else \
                          (p['entry'] - price) / p['entry']
                margin_pnl = pnl_pct * LEVERAGE

                # 손실 한도 / 하드스탑
                if margin_pnl <= POS_LOSS_LIMIT or margin_pnl <= HARD_STOP:
                    realized = p['margin'] * (1 + margin_pnl)
                    fee      = p['margin'] * FEE_RATE * LEVERAGE
                    cash    += realized - fee
                    total_fees += fee
                    trades_log.append(margin_pnl)
                    reentry_cooldown[sym] = ts
                    del pos[sym]
                    continue

                # SuperTrend 반전 → 청산
                if (p['side']=='long' and st == -1):
                    realized = p['margin'] * (1 + margin_pnl)
                    fee      = p['margin'] * FEE_RATE * LEVERAGE
                    cash    += realized - fee
                    total_fees += fee
                    trades_log.append(margin_pnl)
                    reentry_cooldown[sym] = ts
                    del pos[sym]
                    continue

                # DCA (Chop 중에도 기존 포지션 DCA는 제한)
                if (not chop_blocked) and long_signal and \
                   p['side']=='long' and p['dca_n'] < DCA_MAX and \
                   margin_pnl < -0.05:
                    dca_margin = (cash * kelly * weight / MAX_POS) * 0.25
                    if dca_margin > 50 and cash > dca_margin:
                        fee = dca_margin * FEE_RATE * LEVERAGE
                        cash -= dca_margin + fee
                        total_fees += fee
                        # 평단 재계산
                        old_m = p['margin']
                        new_m = old_m + dca_margin
                        p['entry'] = (old_m*p['entry'] + dca_margin*price) / new_m
                        p['margin'] = new_m
                        p['dca_n'] += 1

            else:
                # 신규 진입
                if chop_blocked: continue
                if not long_signal: continue
                if len(pos) >= MAX_POS: continue

                # 재진입 쿨다운 (4캔들 = 60분)
                last_exit = reentry_cooldown.get(sym, 0)
                if not reentry and (ts - last_exit) < 4 * 15 * 60 * 1000:
                    continue
                elif reentry and (ts - last_exit) < 2 * 15 * 60 * 1000:
                    pass  # reentry=True면 짧은 쿨다운

                target_margin = (cash * kelly * weight) / MAX_POS
                target_margin = min(target_margin, cash * 0.95)
                if target_margin < 50: continue

                fee  = target_margin * FEE_RATE * LEVERAGE
                cash -= target_margin + fee
                total_fees += fee
                pos[sym] = {'margin': target_margin, 'entry': price,
                            'dca_n': 0, 'side': 'long'}

    # 미청산 포지션 종가 청산
    for sym, p in pos.items():
        df  = data_map[sym]
        price = float(df['c'].iloc[-1])
        pnl_pct = (price - p['entry']) / p['entry']
        margin_pnl = pnl_pct * LEVERAGE
        realized = p['margin'] * (1 + margin_pnl)
        fee      = p['margin'] * FEE_RATE * LEVERAGE
        cash    += realized - fee
        total_fees += fee
        trades_log.append(margin_pnl)

    final_equity = cash
    ret_pct = (final_equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100
    n       = len(trades_log)
    wins    = sum(1 for t in trades_log if t > 0)
    wr      = wins/n*100 if n else 0
    avg_pnl = np.mean(trades_log)*100 if trades_log else 0

    return {
        'final': final_equity,
        'pnl':   final_equity - INITIAL_EQUITY,
        'ret':   ret_pct,
        'trades': n,
        'wr':    wr,
        'avg':   avg_pnl,
        'fees':  total_fees,
    }

# ── 메인 ───────────────────────────────────────────
async def main():
    print("📡 OHLCV 데이터 수집 중...")
    ex = ccxt_async.okx({'enableRateLimit': True, 'options': {'defaultType':'swap'}})
    try:
        await ex.load_markets()
        data_map = {}
        for sym in SYMBOLS:
            try:
                df = await fetch_ohlcv(ex, sym, '15m', START_MS, END_MS)
                if df is not None and len(df) > 30:
                    data_map[sym] = df
                    print(f"  ✅ {sym}: {len(df)}캔들")
                else:
                    print(f"  ⚠️  {sym}: 데이터 부족")
            except Exception as e:
                print(f"  ❌ {sym}: {e}")
            await asyncio.sleep(0.1)

        print(f"\n📡 BTC 15m 수집 중...")
        btc_df = await fetch_ohlcv(ex, 'BTC/USDT:USDT', '15m', START_MS, END_MS)
        print(f"  BTC: {len(btc_df) if btc_df is not None else 0}캔들")

    finally:
        await ex.close()

    print(f"\n🔢 시뮬레이션 실행 (대상 {len(data_map)}종목, {(END_MS-START_MS)//86400000}일)")
    print("=" * 70)
    print(f"{'변형':<25} {'최종자산':>10} {'손익':>10} {'수익률':>8} {'거래수':>6} {'승률':>6} {'수수료':>8}")
    print("-" * 70)

    results = []
    for v in VARIANTS:
        r = simulate(data_map, btc_df, v)
        results.append((v['label'], r))
        arrow = "▲" if r['pnl'] > 0 else "▼"
        print(f"{v['label']:<25} {r['final']:>10,.0f} {arrow}{abs(r['pnl']):>9,.0f} "
              f"{r['ret']:>+7.1f}% {r['trades']:>6} {r['wr']:>5.1f}% {r['fees']:>8,.0f}")

    print("=" * 70)
    print()
    print("📌 핵심 비교:")
    a_pnl = results[0][1]['pnl']
    for label, r in results[1:]:
        diff = r['pnl'] - a_pnl
        print(f"  {label} vs A: {diff:+,.0f} USDT ({diff/INITIAL_EQUITY*100:+.1f}%)")

    print()
    print("✅ 백테스트 완료")

asyncio.run(main())
