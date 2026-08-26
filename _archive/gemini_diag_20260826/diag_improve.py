"""
diag_improve.py — 전략 로직 개선 연구 (라이브 규칙 반영 정밀 백테스트)
데이터: verify_data_cache.pkl 재사용 (90일 15m OHLCV, 공개 데이터만 사용)

라이브 규칙 반영 (strategy_common.py 기준):
  - 소프트 스탑 -12% (마진 기준), 하드 스탑 -30%(메이저)/-15%(벤처)
  - 하드/소프 스탑아웃 후 심볼당 12시간 재진입 쿨다운
  - ATR 트레일링: 마진수익 +20% 도달 후 샹들리에 (최고가 - 2.5*ATR / 역방향 동일)
  - 배포 스케일: BTC 1h ADX -> clamp((adx-8)/(25-8), 0.25, 1.0) 진입 마진에 곱함
  - 컨빅션 사이징, 균등분할 마진(999슬롯+최소마진 플로어)

개선 후보 (ablation):
  A0 baseline (라이브 재현)
  A1 숏 전면 금지
  A2 저ADX 진입 차단 (BTC ADX<15 진입 금지 — 배포 바닥 25% 폐지)
  A3 Stoch 과열 추격 방지 (stoch_k>=55에서 크로스 진입 금지)
  A4 A1+A2 조합
검증: 전반기 45d / 후반기 45d / 전체 90d 분할 — 양쪽 모두 개선돼야 채택.
"""

import asyncio
import pickle
from datetime import datetime, timezone
import pandas as pd

FEE_RATE = 0.0005
INITIAL_EQUITY = 19000.0
LIVE_EXPOSURE_SCALE = 0.6
LIVE_MIN_MARGIN = 100.0
WEIGHT_MAJOR_LIVE = 0.5
WEIGHT_VENTURE_LIVE = 1.5
NEW_RATIO = 0.50
MAJORS = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'AVAX', 'LINK', 'DOT', 'BNB', 'TRX']

# 라이브 리스크 파라미터
SOFT_STOP = -0.12
HARD_STOP_MAJOR = -0.30
HARD_STOP_VENTURE = -0.15
ATR_TRAIL_K = 2.5
ATR_TRAIL_ARM = 0.20
COOLDOWN_MS = 12 * 3600 * 1000

CHOP_FLOOR_ADX = 8.0
CHOP_THRESHOLD_ADX = 25.0
CHOP_FLOOR = 0.25


def calc_supertrend(df, period=10, multiplier=3.0):
    hl2 = (df['h'] + df['l']) / 2
    atr = (df['h'].combine(df['c'].shift(), max) - df['l'].combine(df['c'].shift(), min)).rolling(period).mean()
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
    return sd, sv


def calc_stoch_k(series, period=14, smooth=3):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    stoch = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
    return stoch.rolling(smooth).mean() * 100


def calc_adx(df, period=14):
    h, l, c = df['h'], df['l'], df['c']
    up, dn = h.diff(), -l.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).where((pdi + mdi) != 0)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)


async def fetch_all(ex, sym, tf, since_ms):
    out = []
    since = since_ms
    for _ in range(400):
        batch = await ex.fetch_ohlcv(sym, tf, since=since, limit=300)
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 300:
            break
        since = batch[-1][0] + 1
        await asyncio.sleep(0.05)
    df = pd.DataFrame(out, columns=['t', 'o', 'h', 'l', 'c', 'v']).drop_duplicates('t')
    return df.sort_values('t').reset_index(drop=True)


def prepare_data():
    with open("verify_data_cache.pkl", "rb") as f:
        btc_map, data = pickle.load(f)
    # 지표 추가: atr14, ema200 (15m)
    for sym, df in data.items():
        h, l, c = df['h'], df['l'], df['c']
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1 / 14, adjust=False).mean()
        df['ema200'] = c.ewm(span=200, adjust=False).mean()
        data[sym] = df
    return btc_map, data


def simulate(btc_map, data, cfg, start_ts=None, end_ts=None, log_trades=False, tidx=None):
    cash = INITIAL_EQUITY
    positions = {}
    cooldown_until = {}
    trades = []
    eq_v = []
    all_ts = sorted(set().union(*[set(df['t']) for df in data.values()]))
    if start_ts:
        all_ts = [t for t in all_ts if t >= start_ts]
    if end_ts:
        all_ts = [t for t in all_ts if t <= end_ts]
    if tidx is None:
        tidx = {sym: {int(tv): int(iv) for tv, iv in zip(df['t'].values, df.index.values)}
                for sym, df in data.items()}

    def is_major(sym):
        return sym.split('/')[0] in MAJORS

    def get_lev(sym):
        # 라이브: 메이저 15x, 벤처 5x (현재 적용값 기준)
        return 15 if is_major(sym) else 5

    def get_weight(sym):
        return WEIGHT_MAJOR_LIVE if is_major(sym) else WEIGHT_VENTURE_LIVE

    def get_hard_stop(sym):
        return HARD_STOP_MAJOR if is_major(sym) else HARD_STOP_VENTURE

    def deploy_scale(last_h):
        adx_now, _, _ = btc_map.get(last_h, (30.0, True, True))
        raw = (adx_now - CHOP_FLOOR_ADX) / max(CHOP_THRESHOLD_ADX - CHOP_FLOOR_ADX, 1.0)
        s = max(CHOP_FLOOR, min(1.0, raw))
        return s, adx_now

    for t in all_ts:
        hour_floor = (t // 3600000) * 3600000
        last_h = hour_floor - 3600000
        _, btc_bull, above200 = btc_map.get(last_h, (30.0, True, True))

        mtm = cash
        for sym, pos in positions.items():
            pnl_pct = (pos['last_px'] - pos['entry']) / pos['entry'] * get_lev(sym) * pos['dir']
            mtm += pos['margin'] * (1 + pnl_pct)
        eq_v.append(mtm)

        for sym, df in data.items():
            i = tidx[sym].get(t)
            if i is None:
                continue
            if i < 210:
                continue
            curr = df.iloc[i]
            prev = df.iloc[i - 1]
            px = float(curr['c'])

            pos = positions.get(sym)
            if pos:
                lev = get_lev(sym)
                pos['last_px'] = px
                pnl_pct = (px - pos['entry']) / pos['entry'] * lev * pos['dir']
                pos['max_pnl'] = max(pos['max_pnl'], pnl_pct)
                pos['extreme_px'] = max(pos['extreme_px'], px) if pos['dir'] == 1 else min(pos['extreme_px'], px)
                exit_now = False
                reason = ""
                # 소프트/하드 스탑 (마진 기준, 라이브 동일)
                if pnl_pct <= get_hard_stop(sym):
                    exit_now, reason = True, "hard"
                elif pnl_pct <= SOFT_STOP:
                    exit_now, reason = True, "soft"
                # ATR 트레일링 (마진수익 20% 도달 후)
                elif pos['max_pnl'] >= ATR_TRAIL_ARM:
                    atr = float(curr['atr'])
                    if pos['dir'] == 1 and px < pos['extreme_px'] - ATR_TRAIL_K * atr:
                        exit_now, reason = True, "atrtrail"
                    elif pos['dir'] == -1 and px > pos['extreme_px'] + ATR_TRAIL_K * atr:
                        exit_now, reason = True, "atrtrail"

                if exit_now:
                    gross = pos['margin'] * pnl_pct
                    fee = pos['margin'] * lev * FEE_RATE
                    cash += pos['margin'] + gross - fee
                    trades.append({'sym': sym, 'dir': pos['dir'], 'pnl': gross - fee,
                                   't': t, 'reason': reason, 'lev': lev})
                    del positions[sym]
                    if reason in ("hard", "soft"):
                        cooldown_until[sym] = t + COOLDOWN_MS
                continue

            long_sig = curr['st_dir'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
            short_sig = curr['st_dir'] == -1 and prev['stoch_k'] > 80 and curr['stoch_k'] <= 80
            # 라이브 bear_short: BTC가 200EMA 위면 숏 금지
            if above200 and short_sig:
                short_sig = False
            if not (long_sig or short_sig):
                continue

            d = 1 if long_sig else -1

            # ---- 개선 게이트 (cfg) ----
            if cfg.get('no_short') and d == -1:
                continue
            if cooldown_until.get(sym, 0) > t:
                continue
            if cfg.get('no_low_adx'):
                adx_now, _, _ = btc_map.get(last_h, (30.0, True, True))
                if adx_now < 15:
                    continue
            if cfg.get('no_overchase'):
                if d == 1 and float(curr['stoch_k']) >= 55:
                    continue
                if d == -1 and float(curr['stoch_k']) <= 45:
                    continue

            scale, _adxs = deploy_scale(last_h)
            conv_mult = 2.0 if long_sig else 1.0  # 원본 스코어링 결과와 동일
            w = get_weight(sym)
            equity = mtm
            base = (equity / 999.0) * w * LIVE_EXPOSURE_SCALE
            target = base * NEW_RATIO * conv_mult * scale
            if target < LIVE_MIN_MARGIN:
                if cash >= LIVE_MIN_MARGIN * 1.1:
                    target = LIVE_MIN_MARGIN
                else:
                    continue
            target = min(target, cash * 0.95)
            if target < LIVE_MIN_MARGIN:
                continue
            lev = get_lev(sym)
            fee = target * lev * FEE_RATE
            cash -= target + fee
            positions[sym] = {'entry': px, 'margin': target, 'dir': d,
                              'max_pnl': 0.0, 'extreme_px': px, 'last_px': px}

    # 미청산 정산
    for sym, pos in positions.items():
        lev = get_lev(sym)
        pnl_pct = (pos['last_px'] - pos['entry']) / pos['entry'] * lev * pos['dir']
        fee = pos['margin'] * lev * FEE_RATE
        cash += pos['margin'] * (1 + pnl_pct) - fee
        trades.append({'sym': sym, 'dir': pos['dir'], 'pnl': pos['margin'] * pnl_pct - fee,
                       't': None, 'reason': 'open', 'lev': lev})

    final = cash
    peak, mdd = INITIAL_EQUITY, 0.0
    for v in eq_v:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak)
    pnls = [tr['pnl'] for tr in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else float('inf')
    wr = len(wins) / len(pnls) * 100 if pnls else 0.0
    long_pnl = sum(tr['pnl'] for tr in trades if tr['dir'] == 1)
    short_pnl = sum(tr['pnl'] for tr in trades if tr['dir'] == -1)
    res = {'ret': (final - INITIAL_EQUITY) / INITIAL_EQUITY * 100, 'mdd': mdd * 100,
           'pf': pf, 'wr': wr, 'n': len(trades), 'long': long_pnl, 'short': short_pnl}
    if log_trades:
        res['_trades'] = trades
    return res


async def main():
    btc_map, data = prepare_data()
    ts_all = sorted(set().union(*[set(df['t']) for df in data.values()]))
    mid = ts_all[len(ts_all) // 2]

    configs = [
        ("A0_baseline", {}),
        ("A1_no_short", {'no_short': True}),
        ("A2_no_lowAD", {'no_low_adx': True}),
        ("A3_nochase", {'no_overchase': True}),
        ("A4_noSh_noLo", {'no_short': True, 'no_low_adx': True}),
    ]

    print("== 분할 검증 (메이저15x/벤처5x, live999 사이즈, 라이브 청산규칙) ==")
    hdr = f"{'설정':14s} {'창':6s} {'수익률%':>8s} {'MDD%':>6s} {'PF':>5s} {'승률%':>5s} {'N':>5s} {'롱손익':>9s} {'숏손익':>9s}"
    print(hdr)

    # 진단: 베이스라인 거래 로그
    base_full = simulate(btc_map, data, {}, log_trades=True)
    trs = base_full.pop('_trades')

    for name, cfg in configs:
        for wlabel, s, e in [("45d앞", None, mid), ("45d뒤", mid, None), ("90일", None, None)]:
            r = simulate(btc_map, data, cfg, start_ts=s, end_ts=e)
            print(f"{name:14s} {wlabel:6s} {r['ret']:>+8.1f} {r['mdd']:>6.1f} {r['pf']:>5.2f} "
                  f"{r['wr']:>5.1f} {r['n']:>5d} {r['long']:>+9.0f} {r['short']:>+9.0f}")
        print()

    print("== 진단: 베이스라인(A0) 청산 사유별 손익 ==")
    by_reason = {}
    for tr in trs:
        by_reason.setdefault(tr['reason'], []).append(tr['pnl'])
    for k, v in sorted(by_reason.items()):
        print(f"  {k:10s} N={len(v):4d} 합={sum(v):>+9.0f} 평균={sum(v)/len(v):>+7.1f}")

    print("== 진단: 상위 손실 심볼 TOP10 ==")
    by_sym = {}
    for tr in trs:
        by_sym.setdefault(tr['sym'], []).append(tr['pnl'])
    worst = sorted(by_sym.items(), key=lambda kv: sum(kv[1]))[:10]
    for k, v in worst:
        print(f"  {k:22s} N={len(v):3d} 합={sum(v):>+9.0f}")


if __name__ == "__main__":
    asyncio.run(main())
