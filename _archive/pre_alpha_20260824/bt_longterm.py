#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bt_longterm.py — 장기 백테스트 (P0/P1/P2 통합)
- 6개월 페이지네이션 OHLCV
- 라이브 DCA 로직 완전 재현 (보유 중 매 캔들 물타기, 분할익절, 하드스탑, 본절스탑)
- 변형 실험: 레짐필터, 트레일링 스탑, 재진입, 타임프레임
- 지표: 거래수, 승률, 총수익, MDD, 최대연속손실, Profit Factor
"""
import asyncio
import time
import numpy as np
import pandas as pd
import ccxt.async_support as ccxt_async

LEV = 10
FEE_PER_UNIT = 0.0005 * LEV * 2  # 왕복 테이커 수수료 (마진 기준)
DAYS = 180


async def fetch_all(ex, symbol, tf, days=DAYS):
    tf_ms = ex.parse_timeframe(tf) * 1000
    now = ex.milliseconds()
    since = now - days * 86400 * 1000
    out = []
    while since < now:
        batch = await ex.fetch_ohlcv(symbol, tf, since=since, limit=300)
        if not batch:
            # 상장일 이전 등으로 공백이면 7일씩 전진
            since += 7 * 86400 * 1000
            await asyncio.sleep(ex.rateLimit / 1000)
            continue
        out += batch
        if len(batch) < 300:
            break
        since = batch[-1][0] + tf_ms
        await asyncio.sleep(ex.rateLimit / 1000)
    seen = {}
    for c in out:
        seen[c[0]] = c
    return [seen[k] for k in sorted(seen)]


def calc_supertrend(c, h, l, period=10, multiplier=3.0):
    n = len(c)
    sd = np.ones(n, dtype=np.int8)
    sv = np.zeros(n)
    hl2 = (h + l) / 2
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum(h, prev_c) - np.minimum(l, prev_c)
    atr = pd.Series(tr).rolling(period).mean().values
    fu = hl2 + multiplier * atr
    fl = hl2 - multiplier * atr
    for i in range(period, n):
        if c[i] > fu[i-1]:
            sd[i] = 1
        elif c[i] < fl[i-1]:
            sd[i] = -1
        else:
            sd[i] = sd[i-1]
            if sd[i] == 1 and fl[i] < fl[i-1]:
                fl[i] = fl[i-1]
            if sd[i] == -1 and fu[i] > fu[i-1]:
                fu[i] = fu[i-1]
        sv[i] = fl[i] if sd[i] == 1 else fu[i]
    return sd, sv


def prepare(o):
    df = pd.DataFrame(o, columns=['t', 'o', 'h', 'l', 'c', 'v'])
    c, h, l = df['c'].values, df['h'].values, df['l'].values
    df['st_d_l'], df['st_v_l'] = calc_supertrend(c, h, l, 10, 3.0)
    df['st_d_t'], df['st_v_t'] = calc_supertrend(c, h, l, 10, 2.0)
    df['st_d_tt'], df['st_v_tt'] = calc_supertrend(c, h, l, 10, 1.5)  # 트레일링 변형용
    df['vol_ma'] = df['v'].rolling(20).mean()
    delta = df['c'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    df['stoch_k'] = ((rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())).rolling(3).mean() * 100
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['ema'] = df['c'].ewm(span=50, adjust=False).mean()
    return df


def simulate(df, thr=70, vol_mult=1.0, hard_stop=-0.30, regime_flags=None,
             max_dca=8, trail=False, reentry=False, cooldown=4, dca_step=None,
             flip=False, stflip=False):
    """라이브 로직 재현: DCA 물타기 + 분할익절.
    flip=True면 트레일링 전량청산 시 즉시 반대방향 진입,
    stflip=True면 슈퍼트렌드 반전 신호 시 전량청산+즉시 반대방향 진입"""
    eps = []
    pos = None
    last_close_i = -10**9
    last_close_side = None

    def open_pos(side, i, price):
        return {'side': side, 'avg': price, 'qty': 1.0, 'entries': 1,
                'max_pnl': 0.0, 'ec': 0, 'entry_i': i, 'realized': 0.0}

    for i in range(210, len(df)):
        p = df.iloc[i-1]
        u = df.iloc[i]
        vol_cond = (u['v'] > p['vol_ma'] * vol_mult) or \
                   (i >= 2 and p['v'] > df.iloc[i-2]['vol_ma'] * vol_mult)
        long_score = (50 if u['c'] > u['ema'] else 0) \
            + (40 if p['st_d_l'] == -1 and u['st_d_l'] == 1 else 0) \
            + (30 if u['st_d_l'] == 1 and p['stoch_k'] < 20 <= u['stoch_k'] else 0) \
            + (20 if u['st_d_l'] == 1 and p['st_d_l'] == 1 else 0) \
            + (20 if u['stoch_k'] > u['stoch_d'] and u['stoch_k'] < 80 else 0)
        short_score = (50 if u['c'] < u['ema'] else 0) \
            + (40 if p['st_d_l'] == 1 and u['st_d_l'] == -1 else 0) \
            + (30 if u['st_d_l'] == -1 and p['stoch_k'] > 80 >= u['stoch_k'] else 0) \
            + (20 if u['st_d_l'] == -1 and p['st_d_l'] == -1 else 0) \
            + (20 if u['stoch_k'] < u['stoch_d'] and u['stoch_k'] > 20 else 0)

        regime_ok = True if regime_flags is None else regime_flags[i]
        long_ok = long_score >= thr and vol_cond and regime_ok
        short_ok = short_score >= thr and vol_cond

        # 재진입 변형: 청산 직후 같은 방향 추세 유지 시 점수 무관 재진입
        if reentry and pos is None and last_close_side and i - last_close_i >= cooldown:
            if last_close_side == 'L' and u['st_d_l'] == 1 and u['c'] > u['ema'] and regime_ok:
                pos = open_pos('L', i, u['c'])
            elif last_close_side == 'S' and u['st_d_l'] == -1 and u['c'] < u['ema']:
                pos = open_pos('S', i, u['c'])

        if pos is None:
            if long_ok:
                pos = open_pos('L', i, u['c'])
            elif short_ok:
                pos = open_pos('S', i, u['c'])
            continue

        sgn = 1 if pos['side'] == 'L' else -1
        pnl = (u['c'] - pos['avg']) / pos['avg'] * LEV * sgn
        pos['max_pnl'] = max(pos['max_pnl'], pnl)
        ec = pos['ec']
        full = False
        scale = False
        trailing_close = False

        is_profit = u['c'] > pos['avg'] * 1.03 if pos['side'] == 'L' else u['c'] < pos['avg'] * 0.97
        if trail and is_profit:
            st_d, st_v = u['st_d_tt'], u['st_v_tt']
        else:
            st_d = u['st_d_t'] if is_profit else u['st_d_l']
            st_v = u['st_v_t'] if is_profit else u['st_v_l']
        st_close = (st_d == -1 or u['c'] < st_v) if pos['side'] == 'L' else (st_d == 1 or u['c'] > st_v)

        if pnl <= hard_stop:
            full = True
            trailing_close = False
        elif pos['max_pnl'] >= 0.40 and pnl <= 0.20:
            full = True
            trailing_close = True
        elif pos['max_pnl'] >= 0.20 and pnl <= 0.0:
            full = True
            trailing_close = True
        elif stflip and st_close and pos['max_pnl'] >= 0.50:
            # [실험] 고점 대비 조건(max_pnl≥50%=가격 5% 상승 경험) + 슈퍼트렌드 반전 시만 플립
            full = True
        elif (pnl >= 0.15 and ec == 0) or (pnl >= 0.30 and ec == 1) or (pnl >= 0.50 and ec == 2):
            scale = True
        elif st_close:
            scale = True

        if not full and not scale and ec > 0:
            bad = u['c'] < pos['avg'] if pos['side'] == 'L' else u['c'] > pos['avg']
            if bad and (i - pos['entry_i']) >= 3:
                full = True

        if full:
            pos['realized'] += pos['qty'] * (pnl - FEE_PER_UNIT)
            eps.append(pos['realized'])
            last_close_i = i
            last_close_side = pos['side']
            flip_side = 'S' if pos['side'] == 'L' else 'L'
            pos = None
            # [실험] 트레일링/슈퍼트렌드 전량청산 시 즉시 반대방향 진입
            if (flip and trailing_close) or (stflip and st_close):
                pos = open_pos(flip_side, i, u['c'])
        elif scale and ec < max_dca:
            frac = pos['qty'] / (max_dca - ec)
            pos['qty'] -= frac
            pos['ec'] += 1
            pos['realized'] += frac * (pnl - FEE_PER_UNIT)
        else:
            # 라이브 DCA: 청산 신호 없으면 매 캔들 물타기 (최대 max_dca회)
            if pos['entries'] < max_dca:
                if dca_step is not None:
                    # 역행 간격 조건: 현재 손익이 -step × 추가횟수 이하여야만 물타기
                    if pnl > -dca_step * pos['entries']:
                        continue
                unit = 1.0
                pos['avg'] = (pos['avg'] * pos['qty'] + u['c'] * unit) / (pos['qty'] + unit)
                pos['qty'] += unit
                pos['entries'] += 1

    if pos is not None:
        u = df.iloc[-1]
        pnl = (u['c'] - pos['avg']) / pos['avg'] * LEV * (1 if pos['side'] == 'L' else -1)
        eps.append(pos['realized'] + pos['qty'] * (pnl - FEE_PER_UNIT))
    return eps


def metrics(eps):
    if not eps:
        return None
    arr = np.array(eps)
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    mdd = (peak - cum).max() if len(cum) else 0
    streak = mx = 0
    for x in arr:
        streak = streak + 1 if x <= 0 else 0
        mx = max(mx, streak)
    gp = arr[arr > 0].sum()
    gl = -arr[arr <= 0].sum()
    pf = gp / gl if gl > 0 else float('inf')
    return dict(n=len(arr), wr=(arr > 0).mean() * 100, tot=cum[-1] * 100,
                mdd=mdd * 100, streak=mx, pf=pf)


def show(name, m):
    if not m:
        print(f"  {name:<30} | 거래 없음")
        return
    print(f"  {name:<30} | N {m['n']:>3} | WR {m['wr']:5.1f}% | 총수익 {m['tot']:+8.1f}% | "
          f"MDD {m['mdd']:6.1f}% | 연속손실 {m['streak']:>2} | PF {m['pf']:4.2f}")


async def main():
    ex = ccxt_async.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

    print("BTC 1h 레짐 데이터 수집 중...")
    btc = await fetch_all(ex, 'BTC/USDT:USDT', '1h', DAYS + 10)
    bdf = pd.DataFrame(btc, columns=['t', 'o', 'h', 'l', 'c', 'v'])
    bdf['ema200'] = bdf['c'].ewm(span=200, adjust=False).mean()
    bdf['ok'] = bdf['c'] >= bdf['ema200']
    btc_regime = bdf.set_index('t')['ok']

    def flags_for(df):
        idx = np.searchsorted(btc_regime.index.values, df['t'].values, side='right') - 1
        f = np.ones(len(df), dtype=bool)
        v = idx >= 0
        f[v] = btc_regime.values[idx[v]]
        return f

    configs = [
        ('현재(70/레짐ON)', dict(thr=70, regime=True)),
        ('ST플립(조걶: +50%경험)', dict(thr=70, regime=True, stflip=True)),
        ('ST플립+재진입', dict(thr=70, regime=True, stflip=True, reentry=True)),
    ]

    groups = [
        ('Venture', ['DOGE/USDT:USDT', 'PEPE/USDT:USDT', 'ONDO/USDT:USDT'], ['15m']),
        ('Major', ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT'], ['30m']),
    ]

    for gname, syms, tfs in groups:
        for tf in tfs:
            print(f"\n=== {gname} {tf} (최근 {DAYS}일) ===")
            dfs = []
            for s in syms:
                try:
                    o = await fetch_all(ex, s, tf)
                    if len(o) >= 500:
                        d = prepare(o)
                        dfs.append((s, d, flags_for(d)))
                        print(f"  [{s}] 캔들 {len(d)}개 수집")
                    else:
                        print(f"  [{s}] 데이터 부족({len(o)}) — 스킵")
                except Exception as e:
                    print(f"  [{s}] 수집 실패: {e}")
            for cname, cfg in configs:
                eps = []
                for s, d, fl in dfs:
                    eps += simulate(d, thr=cfg['thr'], vol_mult=cfg.get('vol_mult', 1.0),
                                    regime_flags=fl if cfg.get('regime') else None,
                                    trail=cfg.get('trail', False), reentry=cfg.get('reentry', False),
                                    dca_step=cfg.get('dca_step'), flip=cfg.get('flip', False),
                                    stflip=cfg.get('stflip', False))
                show(cname, metrics(eps))

    await ex.close()


if __name__ == '__main__':
    asyncio.run(main())
