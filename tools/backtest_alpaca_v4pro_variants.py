#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/backtest_alpaca_v4pro_variants.py
=============================================================================
Alpaca V4-Pro 롱 엔진(BULL_DIP + MEGA_DIP 포드) 개선 시나리오 비교 백테스트 — 2026-11-10
배경: 현행 엔진 실전 엣지 약함(구엔진 60d PF 1.05, 고점 대비 -12.2%). V4 포드 엔진(11-07 배포)은
      실전 표본 0건 → 백테스트로 기저 성능과 개선안을 사전 검증.

■ 사전 등록 채택 기준 (실행 전 확정) — 변형이 S0 대비 아래를 모두 충족 시에만 채택:
  AC1: 두 창(60d/5m, 730d/1h) 모두 순손익 >= S0 + $500
  AC2: 두 창 모두 PF >= max(1.30, S0 PF)
  AC3: MDD 악화 <= 1.0%p (두 창)
  AC4: 표본 감소 <= 30% (두 창 합산)
  미달 시 기각·현행 유지.

■ 충실도: 라이브 로직 재현 (Wilder RSI2, Z20, 일봉 레짐, 3x SL5.5/TP14, 1x SL3.2/TP8,
  BE arm 2.5/1.0%→+0.4%, trail arm 5/2%→drop 1.8/0.7%, 타임스탑 14 RTH 1H봉 & PnL<+0.5%,
  종목당 4h 쿨다운, 최대 3석, 벤치마크당 1석). 근사치: 인트라바 SL 우선, 비용 5bps x2,
  Jev 게이트·EOD 스마트홀드 패스스루, 슬롯 $30k 고정.
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf

PAIR_MAP = [("DIA", "UDOW"), ("SPY", "UPRO"), ("QQQ", "TQQQ"), ("SOXX", "SOXL")]
BENCHS = [p[0] for p in PAIR_MAP]
MEGAS = ["NVDA", "META", "AVGO", "AMZN", "MSFT", "AAPL", "AMD", "PLTR"]
ALL = sorted(set(BENCHS + [p[1] for p in PAIR_MAP] + MEGAS))
NOTIONAL = 30_000.0
FEE = 0.0005
DATA_DIR = "/home/hongsoonil02/quant_system/data"

BASE = dict(
    dip_rsi=15.0, dip_z=-1.25, mega_rsi=12.0, mega_z=-1.40,
    sl3=0.055, tp3=0.140, sl1=0.032, tp1=0.080,
    be_arm3=0.025, be_arm1=0.010, be_off=0.004,
    tr_arm3=0.050, tr_arm1=0.020, tr_drop3=0.018, tr_drop1=0.007,
    time_bars=14, time_min_pl=0.005,
    max_pos=3, cooldown_bars=4,   # 4h = 1H봉 4개
)
SCEN = {
    "S0_현행":            {},
    "E1_깊은딥만(R10,Z-1.75)": dict(dip_rsi=10.0, dip_z=-1.75, mega_rsi=10.0, mega_z=-1.60),
    "E2_SL타이트(3.5/2.2)":   dict(sl3=0.035, sl1=0.022),
    "E3_트레일강화":          dict(tr_arm3=0.035, tr_drop3=0.012, tr_arm1=0.015, tr_drop1=0.005),
    "E4_TP단축(8/5)":        dict(tp3=0.080, tp1=0.050),
    "E5_E1+E3":             dict(dip_rsi=10.0, dip_z=-1.75, mega_rsi=10.0, mega_z=-1.60,
                                 tr_arm3=0.035, tr_drop3=0.012, tr_arm1=0.015, tr_drop1=0.005),
    # ── Round 2 (2026-11-10 00:30, 실행 전 기준 등록): 수익 상한 구조 해소 ──
    # 등록 기준: 두 창 모두 PF >= 1.20 且 순손익 > 0 且 60d 건당 기대값 > $0 → 구조 개선 방향 확정
    "E6_BE+타임스탑 해제":    dict(be_off=None, time_bars=None),
    "E7_E6+조기러너보호":      dict(be_off=None, time_bars=None, tr_arm3=0.025, tr_drop3=0.010,
                                   tr_arm1=0.010, tr_drop1=0.005),
    "E8_E6+SL타이트":         dict(be_off=None, time_bars=None, sl3=0.030, sl1=0.018),
}

def load(tag, period, interval):
    f = os.path.join(DATA_DIR, f"v4pods_{tag}.pkl")
    if os.path.exists(f):
        return pd.read_pickle(f)
    print(f"📥 download {len(ALL)} tickers {period}/{interval}")
    raw = yf.download(ALL, period=period, interval=interval, progress=False, group_by="ticker", auto_adjust=False)
    out = {}
    for s in ALL:
        df = raw[s].dropna(subset=["Close"]).copy()
        df.columns = [c.lower() for c in df.columns]
        df.index = df.index.tz_convert("America/New_York")
        out[s] = df
    pd.to_pickle(out, f)
    return out

def wilder_rsi(c, n=2):
    d = c.diff()
    up = d.where(d > 0, 0.0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.where(d < 0, 0.0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-9))

def prep(data):
    """1H 지표 + 전일 일봉 레짐 사전계산."""
    ind1h, daily, rth = {}, {}, {}
    for s in ALL:
        df = data[s]
        r = df.between_time("09:30", "15:59")
        rth[s] = r
        h = r.resample("1h", origin="start_day", offset="9h30min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        c = h["close"]
        h["ma20"] = c.rolling(20).mean()
        h["rsi2"] = wilder_rsi(c, 2)
        h["z20"] = (c - h["ma20"]) / c.rolling(20).std().replace(0.0, np.nan)
        ind1h[s] = h
        d = r.resample("1D").agg({"close": "last"}).dropna()
        d["ma20"] = d["close"].rolling(20).mean()
        d["ma30"] = d["close"].rolling(30).mean()
        d["ma50"] = d["close"].rolling(50).mean()
        d["ema8"] = d["close"].ewm(span=8, adjust=False).mean()
        d["ema21"] = d["close"].ewm(span=21, adjust=False).mean()
        daily[s] = d
    return ind1h, daily, rth

def prior_daily(daily, s, day):
    d = daily[s]
    p = d[d.index.date < day]
    return p.iloc[-1] if len(p) else None

def run(ind1h, daily, rth, cfg0):
    cfg = dict(BASE); cfg.update(cfg0)
    timeline = ind1h["QQQ"].index
    pos = {}          # sym -> dict(entry, peak, is3x, bmk, t0idx)
    cooldown = {}     # sym -> idx until
    trades = []
    eq = 0.0          # sleeve PnL ($) 기준
    eq_curve = []
    i_of = {t: i for i, t in enumerate(timeline)}

    for t in timeline:
        day = t.date()
        # ── 포지션 관리 (1H 확정봉 기준 인트라바) ──
        for sym in list(pos.keys()):
            p = pos[sym]
            df = rth[sym]
            bars = df.loc[t:t]  # 현재 1H봉과 겹치는 원본봉들
            if bars.empty:
                continue
            hi, lo, cl = float(bars["high"].max()), float(bars["low"].min()), float(bars["close"].iloc[-1])
            ent = p["entry"]; is3 = p["is3x"]
            sl = cfg["sl3"] if is3 else cfg["sl1"]; tp = cfg["tp3"] if is3 else cfg["tp1"]
            be_arm = cfg["be_arm3"] if is3 else cfg["be_arm1"]
            tr_arm = cfg["tr_arm3"] if is3 else cfg["tr_arm1"]
            tr_drop = cfg["tr_drop3"] if is3 else cfg["tr_drop1"]
            p["peak"] = max(p["peak"], hi)
            peak_pl = p["peak"] / ent - 1
            floor = ent * (1 - sl)
            if cfg["be_off"] is not None and peak_pl >= be_arm:
                floor = max(floor, ent * (1 + cfg["be_off"]))
            if peak_pl >= tr_arm:
                floor = max(floor, p["peak"] * (1 - tr_drop))
            exit_px, reason = None, None
            if lo <= floor:
                exit_px, reason = floor, "SL/TRAIL/BE"
            elif hi >= ent * (1 + tp):
                exit_px, reason = ent * (1 + tp), "TP"
            elif cfg["time_bars"] is not None:
                held = t - p["t0"]
                if held >= pd.Timedelta(hours=cfg["time_bars"]) and cl / ent - 1 < cfg["time_min_pl"]:
                    exit_px, reason = cl, "TIME_STOP"
            if exit_px is not None:
                ret = exit_px / ent - 1 - 2 * FEE
                pnl = NOTIONAL * ret
                eq += pnl
                trades.append(dict(sym=sym, pod=p["pod"], ret=ret * 100, pnl=pnl, reason=reason,
                                   day=str(day), hold_h=round((t - p["t0"]).total_seconds() / 3600, 1)))
                cooldown[sym] = t + pd.Timedelta(hours=4)
                del pos[sym]
        eq_curve.append((t, eq))

        # ── 진입 (14:00 ET 이전, 슬롯 여유 시) ──
        if len(pos) >= cfg["max_pos"] or t.hour >= 14:
            continue
        cands = []
        oversold = 0
        snaps = {}
        for bmk in BENCHS:
            h = ind1h[bmk]
            if t not in h.index:
                continue
            br = h.loc[t]
            pd_ = prior_daily(daily, bmk, day)
            if pd_ is None:
                continue
            bc, r2, z = float(br["close"]), float(br["rsi2"]), float(br["z20"])
            if np.isnan(r2) or np.isnan(z):
                continue
            is_bull = (pd_["close"] > pd_["ma50"] and bc > pd_["ma50"] * 0.985) or \
                      (pd_["ema8"] > pd_["ema21"] and pd_["close"] > pd_["ma30"])
            snaps[bmk] = dict(bc=bc, r2=r2, z=z, bull=is_bull)
            if r2 <= cfg["dip_rsi"]:
                oversold += 1
        for bmk, s in snaps.items():
            bull_e = dict(PAIR_MAP)[bmk]
            if s["bull"] and s["r2"] <= cfg["dip_rsi"] and s["z"] <= cfg["dip_z"]:
                if bull_e not in pos and not any(p["bmk"] == bmk for p in pos.values()) \
                   and t >= cooldown.get(bull_e, pd.Timestamp.min.tz_localize("America/New_York")):
                    cands.append((bull_e, "BULL_DIP", bmk, (100 - s["r2"]) - s["z"] * 5.0))
        if oversold >= 2:
            for m in MEGAS:
                if m in pos or t < cooldown.get(m, pd.Timestamp.min.tz_localize("America/New_York")):
                    continue
                h = ind1h[m]
                if t not in h.index:
                    continue
                mr = h.loc[t]
                pdm = prior_daily(daily, m, day)
                if pdm is None:
                    continue
                if pdm["close"] > pdm["ma50"] and pdm["ema8"] > pdm["ema21"] * 0.99 \
                   and float(mr["rsi2"]) <= cfg["mega_rsi"] and float(mr["z20"]) <= cfg["mega_z"]:
                    cands.append((m, "MEGA_DIP", m, (100 - float(mr["rsi2"])) - float(mr["z20"]) * 4.0))
        cands.sort(key=lambda x: x[3], reverse=True)
        for sym, pod, bmk, _ in cands:
            if len(pos) >= cfg["max_pos"] or sym in pos:
                continue
            df = rth[sym]
            nxt = df.index[df.index > t]
            if len(nxt) == 0:
                continue
            et = nxt[0]
            if et.date() != day or et.hour >= 14:
                continue
            pos[sym] = dict(entry=float(df["open"].loc[et]), peak=float(df["open"].loc[et]),
                            is3x=sym in dict(PAIR_MAP).values(), bmk=bmk, t0=et, pod=pod)

    tr = pd.DataFrame(trades)
    eqs = pd.Series([e for _, e in eq_curve])
    if len(tr) == 0:
        return dict(n=0, wr=0, pf=0, net=0, mdd=0, tr=tr)
    w = tr[tr.pnl > 0]; l = tr[tr.pnl <= 0]
    pf = w.pnl.sum() / abs(l.pnl.sum()) if len(l) and l.pnl.sum() != 0 else float("inf")
    mdd = (eqs - eqs.cummax()).min()
    return dict(n=len(tr), wr=len(w) / len(tr) * 100, pf=pf, net=tr.pnl.sum(), mdd=mdd, tr=tr)

def main():
    d5 = load("60d5m", "60d", "5m")
    d1h = load("730d1h", "730d", "1h")
    p5 = prep(d5); p1h = prep(d1h)
    print(f"60d/5m: {p5[2]['QQQ'].index[0]} ~ {p5[2]['QQQ'].index[-1]} | 730d/1h: {p1h[2]['QQQ'].index[0]} ~ {p1h[2]['QQQ'].index[-1]}\n")
    results = {}
    base = {}
    for name, ov in SCEN.items():
        m60 = run(*p5, ov)
        m730 = run(*p1h, ov)
        results[name] = (m60, m730)
        if name == "S0_현행":
            base = dict(m60=m60, m730=m730)
    print(f"{'시나리오':<26} {'60d: n/WR/PF/net/MDD':<42} {'730d: n/WR/PF/net/MDD':<42} 판정")
    for name, (m60, m730) in results.items():
        b60, b730 = base["m60"], base["m730"]
        ok = (m60["net"] >= b60["net"] + 500 and m730["net"] >= b730["net"] + 500
              and m60["pf"] >= max(1.30, b60["pf"]) and m730["pf"] >= max(1.30, b730["pf"])
              and m60["mdd"] >= b60["mdd"] - 1000 and m730["mdd"] >= b730["mdd"] - 1000
              and m60["n"] + m730["n"] >= (b60["n"] + b730["n"]) * 0.7)
        verdict = "✅ PASS" if ok else ("— 기준선" if name == "S0_현행" else "❌ FAIL")
        print(f"{name:<26} {m60['n']:>3}건 {m60['wr']:>4.0f}% {m60['pf']:>5.2f} {m60['net']:>+8.0f}$ {m60['mdd']:>+7.0f}$   "
              f"{m730['n']:>3}건 {m730['wr']:>4.0f}% {m730['pf']:>5.2f} {m730['net']:>+8.0f}$ {m730['mdd']:>+7.0f}$   {verdict}")
    pd.to_pickle(results, os.path.join(DATA_DIR, "v4pods_results.pkl"))
    # S0 상세
    print("\n[S0 60d 거래 명세]")
    print(base["m60"]["tr"].to_string(index=False) if len(base["m60"]["tr"]) else "(거래 없음)")

if __name__ == "__main__":
    main()
