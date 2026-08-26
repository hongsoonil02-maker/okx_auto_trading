"""
verify_gemini_changes.py — 재검증: 어제 적용한 2가지 변경의 백테스트 결과 점검
  [변경 1] 포지션 수 무제한 (MAX_OPEN_POSITIONS 15 -> 999)
  [변경 2] 레버리지 차등 (일률 10x -> 메이저 15x / 벤처 5x)

원본 run_dual_leverage_backtest.py 의 지표/시뮬레이션 로직을 그대로 사용하되:
  - 기간을 90일로 확장 (원본은 7일)
  - 실전 마진 공식 반영 모드 추가 (equity/999 * weight * 0.6, 최소마진 100 USDT 플로어)
  - MDD / 승률 / Profit Factor / 최대 동시 포지션 통계 추가
순수 오프라인 검증 스크립트 — 라이브 봇 코드/설정에 영향 없음 (공개 OHLCV만 조회).
"""

import asyncio
from datetime import datetime, timezone
import pandas as pd
import ccxt.async_support as ccxt_async

FEE_RATE = 0.0005
INITIAL_EQUITY = 19000.0
TRAILING_PCT = 0.06
CHOP_START = datetime(2026, 8, 21, 14, 0, tzinfo=timezone.utc).timestamp() * 1000

# 실전 파라미터 (strategy_common.py / 라이브 환경변수 기준)
LIVE_EXPOSURE_SCALE = 0.6
LIVE_MIN_MARGIN = 100.0
WEIGHT_MAJOR_LIVE = 0.5
WEIGHT_VENTURE_LIVE = 1.5
WEIGHT_SIM = 1.5          # 원본 백테스트가 모든 심볼에 1.5 사용
NEW_RATIO = 0.50

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "LINK/USDT:USDT", "BNB/USDT:USDT",
    "ZRO/USDT:USDT", "CAP/USDT:USDT", "VVV/USDT:USDT", "LTC/USDT:USDT",
    "DOGE/USDT:USDT", "AXTI/USDT:USDT", "CHIP/USDT:USDT", "CRCL/USDT:USDT",
    "AAVE/USDT:USDT", "LIT/USDT:USDT", "FIL/USDT:USDT", "HYPE/USDT:USDT",
    "ZEC/USDT:USDT", "TRB/USDT:USDT", "BOME/USDT:USDT", "ENA/USDT:USDT", "CRV/USDT:USDT",
]

MAJORS = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'AVAX', 'LINK', 'DOT', 'BNB', 'TRX']


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


async def fetch_all(ex, sym, tf, since_ms, until_ms=None):
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
        if until_ms and since > until_ms:
            break
        await asyncio.sleep(0.05)
    df = pd.DataFrame(out, columns=['t', 'o', 'h', 'l', 'c', 'v']).drop_duplicates('t')
    return df.sort_values('t').reset_index(drop=True)


def simulate(data, btc_map, lev_major=10, lev_venture=10,
             sizing="sim15", start_ts=None):
    """sizing:
      sim15     = 원본 백테스트 방식 (슬롯 15개 균등분할, weight 1.5 고정)
      live15cap = 실전 공식이지만 슬롯 15개 (변경 전 라이브)
      live999   = 실전 공식 + 슬롯 999 (현재 라이브: 무제한 + 최소마진 플로어)
    """
    slots = {"sim15": 15, "live15cap": 15, "live999": 999}[sizing]
    cash = INITIAL_EQUITY
    positions = {}
    trades = []
    eq_t, eq_v = [], []
    max_concurrent = 0
    all_ts = sorted(set().union(*[set(df['t']) for df in data.values()]))
    if start_ts:
        all_ts = [t for t in all_ts if t >= start_ts]
    warmup_skipped = 0

    def get_lev(sym):
        base = sym.split('/')[0]
        return lev_major if base in MAJORS else lev_venture

    def get_weight(sym):
        if sizing == "sim15":
            return WEIGHT_SIM
        base = sym.split('/')[0]
        return WEIGHT_MAJOR_LIVE if base in MAJORS else WEIGHT_VENTURE_LIVE

    def calc_margin(sym, equity, free):
        """진입 마진 산출 — 모드별."""
        w = get_weight(sym)
        if sizing == "sim15":
            m = (equity / slots) * WEIGHT_SIM * NEW_RATIO
            # conviction
            m *= conv_mult
            if m < LIVE_MIN_MARGIN or m > cash:
                return 0.0
            return m
        # live 방식: 균등분할 * weight * exposure_scale * ratio, 최소마진 플로어
        base = (equity / max(1, slots)) * w * LIVE_EXPOSURE_SCALE
        target = base * NEW_RATIO * conv_mult
        if target < LIVE_MIN_MARGIN:
            if free >= LIVE_MIN_MARGIN * 1.1:
                target = LIVE_MIN_MARGIN
            else:
                return 0.0
        target = min(target, free * 0.95)
        if target < LIVE_MIN_MARGIN or target > cash:
            return 0.0
        return target

    conv_mult = 1.0  # 진입 시 갱신

    # 시간->행인덱스 조회 최적화 (90일 창 대비)
    tidx = {sym: {int(tv): int(iv) for tv, iv in zip(df['t'].values, df.index.values)}
            for sym, df in data.items()}

    for idx, t in enumerate(all_ts):
        hour_floor = (t // 3600000) * 3600000
        last_h = hour_floor - 3600000
        adx_now, btc_bull, above200 = btc_map.get(last_h, (30.0, True, True))

        # --- 기록: 시점별 총자산 ---
        mtm = cash
        for sym, pos in positions.items():
            lev = get_lev(sym)
            pnl_pct = (pos['last_px'] - pos['entry']) / pos['entry'] * lev * pos['dir']
            mtm += pos['margin'] * (1 + pnl_pct)
        eq_t.append(t)
        eq_v.append(mtm)
        max_concurrent = max(max_concurrent, len(positions))

        for sym, df in data.items():
            i = tidx[sym].get(t)
            if i is None:
                continue
            if i < 1:
                continue
            curr, prev = df.iloc[i], df.iloc[i - 1]
            px = float(curr['c'])

            pos = positions.get(sym)
            if pos:
                pos['last_px'] = px
                lev = get_lev(sym)
                pnl_pct = (px - pos['entry']) / pos['entry'] * lev * pos['dir']
                pos['extreme'] = max(pos['extreme'], pnl_pct) if pos['dir'] == 1 else min(pos['extreme'], pnl_pct)
                pos['highest'] = max(pos['highest'], px) if pos['dir'] == 1 else min(pos['highest'], px)
                best = pos['extreme']

                exit_now = pnl_pct <= -0.15
                exit_now = exit_now or (best > 0.4 and pnl_pct < best * 0.5)
                exit_now = exit_now or (best > 0.2 and pnl_pct < 0.05)
                exit_now = exit_now or pnl_pct >= 0.50
                if pos['dir'] == 1 and px < pos['highest'] * (1 - TRAILING_PCT):
                    exit_now = True
                elif pos['dir'] == -1 and px > pos['highest'] * (1 + TRAILING_PCT):
                    exit_now = True

                if exit_now:
                    gross = pos['margin'] * pnl_pct
                    fee = pos['margin'] * lev * FEE_RATE
                    cash += pos['margin'] + gross - fee
                    trades.append({'t': t, 'pnl': gross - fee})
                    del positions[sym]
                continue

            long_sig = curr['st_dir'] == 1 and prev['stoch_k'] < 20 and curr['stoch_k'] >= 20
            short_sig = curr['st_dir'] == -1 and prev['stoch_k'] > 80 and curr['stoch_k'] <= 80
            if above200 and short_sig:
                short_sig = False
            if not (long_sig or short_sig):
                continue

            d = 1 if long_sig else -1
            # 원본과 동일한 스코어링 (롱 120+, 숏 70)
            score = 70
            if long_sig:
                score = 50 + 40 + 30 + (20 if curr['st_dir'] == prev['st_dir'] else 0) + (20 if curr['stoch_k'] > curr['stoch_d'] else 0)
            conv_mult = min(2.0, max(0.5, score / 70.0))

            equity = mtm
            margin = calc_margin(sym, equity, cash)
            if margin <= 0:
                continue
            lev = get_lev(sym)
            fee = margin * lev * FEE_RATE
            cash -= margin + fee
            positions[sym] = {'entry': px, 'margin': margin, 'dir': d,
                              'extreme': 0.0, 'highest': px, 'last_px': px}

    # 미청산 포지션 정산
    for sym, pos in positions.items():
        lev = get_lev(sym)
        pnl_pct = (pos['last_px'] - pos['entry']) / pos['entry'] * lev * pos['dir']
        fee = pos['margin'] * lev * FEE_RATE
        cash += pos['margin'] * (1 + pnl_pct) - fee
        trades.append({'pnl': pos['margin'] * pnl_pct - fee})

    final_equity = cash
    pnl_total = final_equity - INITIAL_EQUITY

    # MDD
    peak, mdd = INITIAL_EQUITY, 0.0
    for v in eq_v:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak)

    pnls = [tr['pnl'] for tr in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else float('inf')
    winrate = len(wins) / len(pnls) * 100 if pnls else 0.0

    return {
        'total': pnl_total,
        'ret_pct': pnl_total / INITIAL_EQUITY * 100,
        'mdd_pct': mdd * 100,
        'pf': pf,
        'winrate': winrate,
        'n': len(trades),
        'max_concurrent': max_concurrent,
        'final_equity': final_equity,
    }


async def main():
    import pickle, os
    cache = "verify_data_cache.pkl"
    ex = ccxt_async.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    now = datetime.now(timezone.utc)
    since90 = int((now - pd.Timedelta(days=97)).timestamp() * 1000)
    since07 = int(datetime(2026, 8, 18, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    start07 = int(datetime(2026, 8, 18, 4, 0, tzinfo=timezone.utc).timestamp() * 1000)

    try:
        if os.path.exists(cache):
            print("loading cached data ...")
            with open(cache, "rb") as f:
                btc_map, data = pickle.load(f)
        else:
            print("fetching BTC 1h ...")
            btc = await fetch_all(ex, "BTC/USDT:USDT", "1h", since90 - 260 * 3600 * 1000)
            btc['adx'] = calc_adx(btc, 14)
            btc['ema200'] = btc['c'].ewm(span=200, adjust=False).mean()
            btc['ema50'] = btc['c'].ewm(span=50, adjust=False).mean()
            btc_map = {}
            for i in range(1, len(btc) - 1):
                r = btc.iloc[i]
                above200 = bool(r['c'] >= r['ema200'])
                ema50_up = bool(btc['ema50'].iloc[i] > btc['ema50'].iloc[i - 4])
                btc_map[int(r['t'])] = (float(r['adx']), above200 and ema50_up, above200)

            data = {}
            for sym in SYMBOLS:
                try:
                    df = await fetch_all(ex, sym, "15m", since90)
                    if len(df) >= 200:
                        df['st_dir'], _ = calc_supertrend(df)
                        df['stoch_k'] = calc_stoch_k(df['c'])
                        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
                        df = df.dropna().reset_index(drop=True)
                        data[sym] = df
                    print(f"  {sym}: {len(df)} bars")
                except Exception as e:
                    print(f"  {sym}: FETCH FAIL {e}")
            with open(cache, "wb") as f:
                pickle.dump((btc_map, data), f)

        combos = [("10x/10x", 10, 10), ("5x/5x", 5, 5), ("10x/5x", 10, 5), ("15x/5x", 15, 5)]

        for label, start_ts in [("== 최근 7일 (원본 창, 재현) ==", start07),
                                 ("== 최근 90일 (확장 창) ==", None)]:
            print("\n" + label)
            hdr = f"{'모드':10s} {'레버리지':9s} {'수익률%':>9s} {'MDD%':>7s} {'PF':>6s} {'승률%':>6s} {'거래수':>6s} {'최대동시':>7s}"
            print(hdr)
            for sizing in ["sim15", "live15cap", "live999"]:
                for name, lm, lv in combos:
                    r = simulate(dict(data), btc_map, lm, lv, sizing=sizing, start_ts=start_ts)
                    print(f"{sizing:10s} {name:9s} {r['ret_pct']:>+9.1f} {r['mdd_pct']:>7.1f} "
                          f"{r['pf']:>6.2f} {r['winrate']:>6.1f} {r['n']:>6d} {r['max_concurrent']:>7d}")
    finally:
        await ex.close()


if __name__ == "__main__":
    asyncio.run(main())
