#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/research_okx_alpha.py
=============================================================================
OKX 무기한 선물 장기 우상향 알파 탐색 — 2022~2026 (약 5년) 1D/4H 실데이터 검증
(2026-11-11 00:15 UTC)

배경: 기존 1H 봇(S8/V5)은 1H 노이즈로 OOS PF 0.49(S8) / 1.03~1.08(V5)에 그침.
      저빈도 일봉/4H 추세·모멘텀·변동성타겟팅 전략군에서 2022 약세장부터 2026 현재까지
      전 구간 일관된 양의 기대값을 갖는 알파 전략을 탐색.

■ 사전 등록 채택 기준 (실행 전 확정 — 결과 보고 기준 완화 금지)
  - IS  : 2022-03-01 ~ 2024-06-30 (약 2.33년: 2022 루나/FTX 폭락장 + 2023 횡보 + 2024 강세)
  - OOS : 2024-07-01 ~ 2026-11-10 (약 2.36년: 2024 말 강세 + 2025~2026 변동·조정장)
  - AC1 : IS CAGR >= 15% AND OOS CAGR >= 15%
  - AC2 : IS Sharpe >= 1.0 AND OOS Sharpe >= 1.2
  - AC3 : IS MDD >= -25% AND OOS MDD >= -25%
  - AC4 : OOS 리밸런스/진입 횟수 >= 25 (일봉/4H 단위 충분 표본)
  - AC5 : 최근 365일 순수익 > 0%
  - AC6 : 동일 전략군 이웃 그리드 중 >= 2/3가 (AC1 & AC2 & AC3) 충족

■ 체결 모델 (룩어헤드 0%, 가혹 비용):
  - 신호: 바 t 확정 종가 → 집행: 바 t+1 시가
  - 비용: 편도 10bps (0.10%) = 왕복 20bps (0.20%, OKX 테이커 0.05%*2 + 슬리피지/펀딩 여유분 0.10%)
"""
import os
import time
import requests
import numpy as np
import pandas as pd

DATA_FILE = "/home/hongsoonil02/quant_system/data/okx_8majors_1d_4h_2022_2026.pkl"
SYMS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "XRP-USDT-SWAP",
        "DOGE-USDT-SWAP", "AVAX-USDT-SWAP", "LINK-USDT-SWAP", "ADA-USDT-SWAP"]

FEE = 0.0010          # 10 bps one-way (20 bps round-trip)
IS_START = "2022-03-01"
IS_END = "2024-06-30"
OOS_START = "2024-07-01"
START_MS = 1640995200000  # 2022-01-01 00:00:00 UTC


def fetch_okx_history(inst: str, bar: str, start_ms: int = START_MS) -> pd.DataFrame:
    url_hist = "https://www.okx.com/api/v5/market/history-candles"
    url_live = "https://www.okx.com/api/v5/market/candles"
    rows = []
    try:
        r0 = requests.get(url_live, params={"instId": inst, "bar": bar, "limit": "100"}, timeout=10).json().get("data", [])
        rows.extend([x for x in r0 if x[8] == "1"])
    except Exception as e:
        print("warn live:", inst, bar, e)

    after = str(min(int(x[0]) for x in rows)) if rows else ""
    while True:
        params = {"instId": inst, "bar": bar, "limit": "100"}
        if after:
            params["after"] = after
        try:
            r = requests.get(url_hist, params=params, timeout=10).json()
            data = r.get("data", [])
        except Exception as e:
            print("retry err:", inst, bar, e)
            time.sleep(0.5)
            continue
        if not data:
            break
        rows.extend(data)
        oldest = min(int(x[0]) for x in data)
        after = str(oldest)
        if oldest <= start_ms:
            break
        time.sleep(0.05)

    dedup = {int(x[0]): x for x in rows if x[8] == "1" and int(x[0]) >= start_ms}
    ordered = [dedup[k] for k in sorted(dedup.keys())]
    df = pd.DataFrame(ordered, columns=["ts", "open", "high", "low", "close", "vol", "vc", "vcq", "confirm"])
    df["dt"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "vol"):
        df[col] = df[col].astype(float)
    df = df.set_index("dt")
    return df[["open", "high", "low", "close", "vol"]]


def load_all():
    if os.path.exists(DATA_FILE):
        return pd.read_pickle(DATA_FILE)
    print("📥 Downloading OKX 8-Majors 1D & 4H data (2022-01 ~ 2026-11)...")
    d1, h4 = {}, {}
    for s in SYMS:
        d1[s] = fetch_okx_history(s, "1D")
        print(f"  1D {s}: {len(d1[s])} bars ({d1[s].index[0].date()} ~ {d1[s].index[-1].date()})")
        h4[s] = fetch_okx_history(s, "4H")
        print(f"  4H {s}: {len(h4[s])} bars")

    o1 = pd.DataFrame({s: d1[s]["open"] for s in SYMS}).sort_index().ffill()
    h1 = pd.DataFrame({s: d1[s]["high"] for s in SYMS}).sort_index().ffill()
    l1 = pd.DataFrame({s: d1[s]["low"] for s in SYMS}).sort_index().ffill()
    c1 = pd.DataFrame({s: d1[s]["close"] for s in SYMS}).sort_index().ffill()

    o4 = pd.DataFrame({s: h4[s]["open"] for s in SYMS}).sort_index().ffill()
    h4p = pd.DataFrame({s: h4[s]["high"] for s in SYMS}).sort_index().ffill()
    l4 = pd.DataFrame({s: h4[s]["low"] for s in SYMS}).sort_index().ffill()
    c4 = pd.DataFrame({s: h4[s]["close"] for s in SYMS}).sort_index().ffill()

    pack = ((o1, h1, l1, c1), (o4, h4p, l4, c4))
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    pd.to_pickle(pack, DATA_FILE)
    return pack


def run_weights(T, o, c, fee=FEE):
    """T decided at bar t close -> executed at bar t+1 open. W2 gap + W1 intra - fee*|dW|."""
    T = T.reindex(c.index).fillna(0.0)
    gap = (o / c.shift(1) - 1.0).fillna(0.0)
    intra = (c / o - 1.0).fillna(0.0)
    W1, W2 = T.shift(1).fillna(0.0), T.shift(2).fillna(0.0)
    ret = (W2 * gap).sum(1) + (W1 * intra).sum(1) - fee * (W1 - W2).abs().sum(1)
    # Count meaningful position entry / rebalance events
    entries = ((W1.abs() > 1e-4) & (W2.abs() <= 1e-4)).sum(1)
    rebals = ((W1 - W2).abs().sum(1) > 1e-3).astype(int)
    return ret, entries, rebals


# ── K1: Cross-Sectional Crypto Momentum Rotation + Trend Gate ────────────────
def k1_mom_rot(o, c, lb=21, top_n=3, ma_n=50, rebal_days=7, vol_target=0.0, allow_short=False):
    r = c / c.shift(lb) - 1.0
    ma = c.ewm(span=ma_n, adjust=False).mean()
    vol = c.pct_change().rolling(30).std() * np.sqrt(365)  # annualized crypto vol

    T = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    for i in range(len(c)):
        if i % rebal_days != 0:
            continue
        d = c.index[i]
        if ma.iloc[i].isna().any() or r.iloc[i].isna().any():
            continue
        row = pd.Series(0.0, index=c.columns)
        btc_up = c.iloc[i]["BTC-USDT-SWAP"] > ma.iloc[i]["BTC-USDT-SWAP"]
        btc_dn = c.iloc[i]["BTC-USDT-SWAP"] < ma.iloc[i]["BTC-USDT-SWAP"]

        if btc_up:
            elig = [s for s in SYMS if c.iloc[i][s] > ma.iloc[i][s] and r.iloc[i][s] > 0]
            if elig:
                picks = r.iloc[i][elig].nlargest(top_n).index
                if vol_target > 0:
                    # Inverse-vol sizing capped at vol_target portfolio risk (no >1.5x lev)
                    raw_w = {s: min(1.0 / top_n * 1.5, (vol_target / top_n) / max(0.15, float(vol.iloc[i][s]))) for s in picks}
                    tot_w = sum(raw_w.values())
                    scale = min(1.0, 1.5 / tot_w) if tot_w > 1.5 else 1.0
                    for s, w in raw_w.items():
                        row[s] = w * scale
                else:
                    row[picks] = 1.0 / top_n
        elif allow_short and btc_dn:
            elig = [s for s in SYMS if c.iloc[i][s] < ma.iloc[i][s] and r.iloc[i][s] < 0]
            if elig:
                picks = r.iloc[i][elig].nsmallest(top_n).index
                if vol_target > 0:
                    raw_w = {s: -min(1.0 / top_n * 1.5, (vol_target / top_n) / max(0.15, float(vol.iloc[i][s]))) for s in picks}
                    tot_w = sum(abs(v) for v in raw_w.values())
                    scale = min(1.0, 1.5 / tot_w) if tot_w > 1.5 else 1.0
                    for s, w in raw_w.items():
                        row[s] = w * scale
                else:
                    row[picks] = -1.0 / top_n
        T.iloc[i] = row
    return T.ffill().fillna(0.0)


# ── K2: Donchian Channel Turtle Breakout + Volatility Sizing ─────────────────
def k2_donchian(o, h, l, c, entry_n=20, exit_n=10, allow_short=False, vol_target=0.25, btc_filter=0):
    hi_n = h.shift(1).rolling(entry_n).max()
    lo_n = l.shift(1).rolling(entry_n).min()
    lo_exit = l.shift(1).rolling(exit_n).min()
    hi_exit = h.shift(1).rolling(exit_n).max()
    vol = c.pct_change().rolling(30).std() * np.sqrt(365)
    btc_ma = c["BTC-USDT-SWAP"].ewm(span=btc_filter, adjust=False).mean() if btc_filter > 0 else None

    T = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    n_syms = len(SYMS)
    for s in SYMS:
        pos_arr = np.zeros(len(c))
        st = 0  # +1 long, -1 short, 0 flat
        for i in range(max(entry_n, 35, btc_filter), len(c)):
            px = c[s].iloc[i]
            v = max(0.20, float(vol[s].iloc[i]))
            # Per-asset weight scaled by target vol, max 2.0/n_syms per coin (max ~2x gross)
            w_abs = min(2.0 / n_syms, (vol_target / n_syms) / v) if vol_target > 0 else (1.0 / n_syms)
            btc_ok_long = True if btc_ma is None else (c["BTC-USDT-SWAP"].iloc[i] > btc_ma.iloc[i])
            btc_ok_short = True if btc_ma is None else (c["BTC-USDT-SWAP"].iloc[i] < btc_ma.iloc[i])

            if st == 1:
                if px < lo_exit[s].iloc[i]:
                    st = 0
            elif st == -1:
                if px > hi_exit[s].iloc[i]:
                    st = 0

            if st == 0:
                if px > hi_n[s].iloc[i] and btc_ok_long:
                    st = 1
                elif allow_short and px < lo_n[s].iloc[i] and btc_ok_short:
                    st = -1

            pos_arr[i] = st * w_abs
        T[s] = pos_arr
    return T


# ── K3: Dual-EMA Trend Following + Vol Target ────────────────────────────────
def k3_ema_trend(o, c, fast=10, slow=40, allow_short=False, vol_target=0.25):
    ef = c.ewm(span=fast, adjust=False).mean()
    es = c.ewm(span=slow, adjust=False).mean()
    vol = c.pct_change().rolling(30).std() * np.sqrt(365)
    n_syms = len(SYMS)
    T = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    for s in SYMS:
        w = (vol_target / n_syms) / vol[s].clip(lower=0.20)
        w = w.clip(upper=2.0 / n_syms)
        sig = pd.Series(0.0, index=c.index)
        sig[ef[s] > es[s]] = 1.0
        if allow_short:
            sig[ef[s] < es[s]] = -1.0
        T[s] = (sig * w).fillna(0.0)
    T.iloc[: slow + 5] = 0.0
    return T


# ── K4: 4H Donchian Trend + Daily BTC EMA Filter (Resampled back to daily for stats) ─
def k4_4h_trend(o4, h4, l4, c4, c1_daily, entry_bars=48, exit_bars=24, allow_short=False, vol_target=0.25):
    hi_n = h4.shift(1).rolling(entry_bars).max()
    lo_n = l4.shift(1).rolling(entry_bars).min()
    lo_exit = l4.shift(1).rolling(exit_bars).min()
    hi_exit = h4.shift(1).rolling(exit_bars).max()
    vol = c4.pct_change().rolling(180).std() * np.sqrt(365 * 6)

    # Confirmed daily BTC EMA50 aligned without lookahead
    btc_d = c1_daily["BTC-USDT-SWAP"]
    btc_ema = btc_d.ewm(span=50, adjust=False).mean().shift(1)  # strictly previous completed day
    btc_ok_l = (btc_d.shift(1) > btc_ema).reindex(c4.index, method="ffill").fillna(False)
    btc_ok_s = (btc_d.shift(1) < btc_ema).reindex(c4.index, method="ffill").fillna(False)

    T4 = pd.DataFrame(0.0, index=c4.index, columns=c4.columns)
    n_syms = len(SYMS)
    for s in SYMS:
        pos_arr = np.zeros(len(c4))
        st = 0
        for i in range(max(entry_bars, 190), len(c4)):
            px = c4[s].iloc[i]
            v = max(0.20, float(vol[s].iloc[i]))
            w_abs = min(2.0 / n_syms, (vol_target / n_syms) / v)
            if st == 1 and px < lo_exit[s].iloc[i]:
                st = 0
            elif st == -1 and px > hi_exit[s].iloc[i]:
                st = 0
            if st == 0:
                if px > hi_n[s].iloc[i] and btc_ok_l.iloc[i]:
                    st = 1
                elif allow_short and px < lo_n[s].iloc[i] and btc_ok_s.iloc[i]:
                    st = -1
            pos_arr[i] = st * w_abs
        T4[s] = pos_arr

    ret4, ent4, reb4 = run_weights(T4, o4, c4, fee=FEE)
    # Compound 4H returns into daily returns for consistent IS/OOS stats
    daily_ret = (1.0 + ret4).resample("1D").prod() - 1.0
    daily_ent = ent4.resample("1D").sum()
    return daily_ret, daily_ent


# ── Evaluation ───────────────────────────────────────────────────────────────
def stats(ret, start=None, end=None):
    r = ret.loc[start:end].dropna()
    eq = (1.0 + r).cumprod()
    yrs = len(r) / 365.0
    cagr = eq.iloc[-1] ** (1.0 / yrs) - 1.0
    sharpe = (r.mean() / r.std() * np.sqrt(365.0)) if r.std() > 0 else 0.0
    mdd = (eq / eq.cummax() - 1.0).min()
    return dict(cagr=cagr, sharpe=sharpe, mdd=mdd)


def eval_daily(name, ret, entries, periodic=False):
    i = stats(ret, IS_START, IS_END)
    oo = stats(ret, OOS_START, None)
    n_oos = int(entries.loc[OOS_START:].sum())
    y1 = (1.0 + ret.iloc[-365:]).prod() - 1.0
    d60 = (1.0 + ret.iloc[-60:]).prod() - 1.0
    # Weekly PF on OOS as trade-independent proxy
    wk = (1.0 + ret.loc[OOS_START:]).resample("W").prod() - 1.0
    wk_pos = wk[wk > 0].sum()
    wk_neg = abs(wk[wk < 0].sum())
    wk_pf = wk_pos / wk_neg if wk_neg > 0 else 9.9
    ac = {
        "AC1": i["cagr"] >= 0.15 and oo["cagr"] >= 0.15,
        "AC2": i["sharpe"] >= 1.0 and oo["sharpe"] >= 1.2,
        "AC3": i["mdd"] >= -0.25 and oo["mdd"] >= -0.25,
        "AC4": n_oos >= 25,
        "AC5": y1 > 0.0,
    }
    return dict(name=name, is_cagr=i["cagr"], is_sh=i["sharpe"], is_mdd=i["mdd"],
                oos_cagr=oo["cagr"], oos_sh=oo["sharpe"], oos_mdd=oo["mdd"],
                n_oos=n_oos, wk_pf=wk_pf, y1=y1, d60=d60, ret=ret, **ac)


def main():
    (o1, h1, l1, c1), (o4, h4, l4, c4) = load_all()
    print(f"데이터 범위: {c1.index[0].date()} ~ {c1.index[-1].date()} ({len(c1)}일 / 4H {len(c4)}봉)")
    btc_ret = c1["BTC-USDT-SWAP"].pct_change().fillna(0.0)
    b_is, b_oo = stats(btc_ret, IS_START, IS_END), stats(btc_ret, OOS_START)
    print(f"[벤치] BTC 보유 | IS CAGR {b_is['cagr']*100:5.1f}% Sh {b_is['sharpe']:.2f} MDD {b_is['mdd']*100:6.1f}% | "
          f"OOS CAGR {b_oo['cagr']*100:5.1f}% Sh {b_oo['sharpe']:.2f} MDD {b_oo['mdd']*100:6.1f}%\n")

    fams = []

    # K1: Cross-Sectional Momentum (Long-only vs Long+Short, fixed vs vol-targeted)
    k1_lo, k1_ls = [], []
    for lb in (14, 21, 30):
        for ma in (30, 50):
            T_lo = k1_mom_rot(o1, c1, lb=lb, top_n=3, ma_n=ma, rebal_days=7, vol_target=0.25, allow_short=False)
            ret, ent, reb = run_weights(T_lo, o1, c1)
            k1_lo.append(eval_daily(f"K1_LO lb{lb} ma{ma} vt25", ret, reb))

            T_ls = k1_mom_rot(o1, c1, lb=lb, top_n=3, ma_n=ma, rebal_days=7, vol_target=0.25, allow_short=True)
            ret2, ent2, reb2 = run_weights(T_ls, o1, c1)
            k1_ls.append(eval_daily(f"K1_LS lb{lb} ma{ma} vt25", ret2, reb2))
    fams += [("K1_LO", k1_lo), ("K1_LS", k1_ls)]

    # K2: Donchian Turtle (Long-only vs Long+Short)
    k2_lo, k2_ls = [], []
    for en, ex in ((15, 7), (20, 10), (30, 15), (40, 20)):
        T_lo = k2_donchian(o1, h1, l1, c1, entry_n=en, exit_n=ex, allow_short=False, vol_target=0.30)
        ret, ent, reb = run_weights(T_lo, o1, c1)
        k2_lo.append(eval_daily(f"K2_DON_LO {en}/{ex} vt30", ret, ent))

        T_ls = k2_donchian(o1, h1, l1, c1, entry_n=en, exit_n=ex, allow_short=True, vol_target=0.30)
        ret2, ent2, reb2 = run_weights(T_ls, o1, c1)
        k2_ls.append(eval_daily(f"K2_DON_LS {en}/{ex} vt30", ret2, ent2))
    fams += [("K2_DON_LO", k2_lo), ("K2_DON_LS", k2_ls)]

    # K3: Dual EMA Trend (Long-only vs Long+Short)
    k3_lo, k3_ls = [], []
    for fa, sl in ((5, 25), (10, 40), (15, 60)):
        ret, ent, reb = run_weights(k3_ema_trend(o1, c1, fa, sl, allow_short=False, vol_target=0.30), o1, c1)
        k3_lo.append(eval_daily(f"K3_EMA_LO {fa}/{sl} vt30", ret, ent))
        ret2, ent2, reb2 = run_weights(k3_ema_trend(o1, c1, fa, sl, allow_short=True, vol_target=0.30), o1, c1)
        k3_ls.append(eval_daily(f"K3_EMA_LS {fa}/{sl} vt30", ret2, ent2))
    fams += [("K3_EMA_LO", k3_lo), ("K3_EMA_LS", k3_ls)]

    # K4: 4H Donchian Trend (Long-only vs Long+Short)
    k4_lo, k4_ls = [], []
    for en, ex in ((36, 18), (60, 30), (96, 48)):
        r_lo, e_lo = k4_4h_trend(o4, h4, l4, c4, c1, entry_bars=en, exit_bars=ex, allow_short=False, vol_target=0.25)
        k4_lo.append(eval_daily(f"K4_4H_LO {en}/{ex} vt25", r_lo, e_lo))
        r_ls, e_ls = k4_4h_trend(o4, h4, l4, c4, c1, entry_bars=en, exit_bars=ex, allow_short=True, vol_target=0.25)
        k4_ls.append(eval_daily(f"K4_4H_LS {en}/{ex} vt25", r_ls, e_ls))
    fams += [("K4_4H_LO", k4_lo), ("K4_4H_LS", k4_ls)]

    # K5: Ensemble (K1 + K2 50:50)
    k5 = []
    for lb, en, ex in ((21, 20, 10), (21, 30, 15), (30, 20, 10), (30, 30, 15)):
        T1 = k1_mom_rot(o1, c1, lb=lb, top_n=3, ma_n=40, rebal_days=7, vol_target=0.30, allow_short=True)
        T2 = k2_donchian(o1, h1, l1, c1, entry_n=en, exit_n=ex, allow_short=True, vol_target=0.30)
        ret, ent, reb = run_weights(0.5 * T1 + 0.5 * T2, o1, c1)
        k5.append(eval_daily(f"K5_ENS_LS lb{lb}+{en}/{ex}", ret, reb))
    fams.append(("K5_ENS_LS", k5))

    rows = []
    for fam, items in fams:
        ok = sum(1 for r in items if r["AC1"] and r["AC2"] and r["AC3"])
        for r in items:
            r["AC6"] = ok >= len(items) * 2 / 3
        rows += items

    hdr = f"{'전략':<26}|{'IS CG':>6}{'IS Sh':>6}{'IS DD':>7}|{'OOS CG':>7}{'OOS Sh':>7}{'OOS DD':>7}|{'N':>5}{'WkPF':>6}{'1Y':>7}{'60d':>7}| AC1-6"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        flags = "".join("O" if r[k] else "x" for k in ("AC1", "AC2", "AC3", "AC4", "AC5", "AC6"))
        star = " ★PASS★" if all(r[k] for k in ("AC1", "AC2", "AC3", "AC4", "AC5", "AC6")) else ""
        print(f"{r['name']:<26}|{r['is_cagr']*100:5.1f}%{r['is_sh']:6.2f}{r['is_mdd']*100:6.1f}%|"
              f"{r['oos_cagr']*100:6.1f}%{r['oos_sh']:7.2f}{r['oos_mdd']*100:6.1f}%|"
              f"{r['n_oos']:5d}{r['wk_pf']:5.2f}x{r['y1']*100:6.1f}%{r['d60']*100:6.1f}%| {flags}{star}")


if __name__ == "__main__":
    main()

