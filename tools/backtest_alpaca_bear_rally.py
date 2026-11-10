#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/backtest_alpaca_bear_rally.py
=============================================================================
Bear-Rally LONG Module (하락장 반등 롱) 신규 가설 검증 — 2026-11-09
배경: 현행 V4-Pro는 일봉 상승추세(BULL)에서만 딥 매수 → BEAR 레짐 진입 경로 전무.
      2026-11-09 나스닥 -8.9% 폭락 후 +4.08% V자 반등 같은 자리를 잡는 모듈 검증.

■ 사전 등록 채택 기준 (Pre-registered, 실행 전 확정):
  AC1: 두 창(60d/5m, 730d/1h) 모두 슬리브 순손익 > 0
  AC2: 두 창 모두 PF >= 1.30
  AC3: 합산 표본 >= 20건
  AC4: 60d 창 건당 기대값 >= +$15 (비용 대비 유의성)
  AC5: 최악 단일 거래 손실률 >= -3.0% (설계 SL 내)
  → 전부 충족 시에만 파일럿 채택, 하나라도 미달 시 기각·현행 유지

■ 모듈 스펙 (라이브 엔진 충실): 일봉 confirmed is_bear(close<MA20 & EMA8<EMA21) 국면에서
  1H 확정봉 RSI(2) 극단 과매도 + Z(20) 이탈 시 3x Bull ETF 반등 롱, 당일 청산(오버나잇 금지).
■ 근사치: 인트라바 SL/TP 동시 터치 시 SL 우선(보수적), 수수료+슬립 5bps x 2, Jev 게이트 패스스루.
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf

PAIRS = [("QQQ", "TQQQ"), ("SOXX", "SOXL")]
BENCHS = [p[0] for p in PAIRS]
ALL = sorted(set(BENCHS + [p[1] for p in PAIRS]))
NOTIONAL = 10_000.0
FEE = 0.0005
DATA_DIR = "/home/hongsoonil02/quant_system/data"

# 시나리오: (rsi2_th, z_th, sl, tp, trail_arm, trail_drop, use_1x)
SCEN = {
    "B1_기본(R10,Z-1.5,SL2,TP3)":      dict(rsi=10.0, z=-1.50, sl=0.020, tp=0.030, trail=None, x1=False),
    "B2_완화(R15,Z-1.25)":             dict(rsi=15.0, z=-1.25, sl=0.020, tp=0.030, trail=None, x1=False),
    "B3_극단(R5,Z-2.0,SL1.5,TP4)":     dict(rsi=5.0,  z=-2.00, sl=0.015, tp=0.040, trail=None, x1=False),
    "B4_B1+트레일(arm2,drop1.2)":      dict(rsi=10.0, z=-1.50, sl=0.020, tp=9.99,  trail=(0.020, 0.012), x1=False),
    "B5_1x분리(B1트리거,1x매매)":       dict(rsi=10.0, z=-1.50, sl=0.020, tp=0.030, trail=None, x1=True),
}

def load(tag, period, interval):
    f = os.path.join(DATA_DIR, f"bear_rally_{tag}.pkl")
    if os.path.exists(f):
        return pd.read_pickle(f)
    print(f"📥 download {ALL} {period}/{interval}")
    raw = yf.download(ALL, period=period, interval=interval, progress=False, group_by="ticker", auto_adjust=False)
    out = {}
    for s in ALL:
        df = raw[s].dropna(subset=["Close"]).copy()
        df.columns = [c.lower() for c in df.columns]
        df.index = df.index.tz_convert("America/New_York")  # yf intraday는 UTC → ET 변환 필수
        out[s] = df
    pd.to_pickle(out, f)
    return out

def wilder_rsi(c, n=2):
    d = c.diff()
    up = d.where(d > 0, 0.0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.where(d < 0, 0.0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-9))

def add_ind(df):
    c = df["close"]
    df["ma20"] = c.rolling(20).mean()
    df["rsi2"] = wilder_rsi(c, 2)
    df["z20"] = (c - df["ma20"]) / c.rolling(20).std().replace(0.0, np.nan)
    return df

def prep(data):
    """벤치마크: 1H 지표 + 전일 일봉 레짐. 트레이드 대상: 원본 봉."""
    bench_1h, trade, daily_regime = {}, {}, {}
    for b in BENCHS:
        df = data[b]
        rth = df.between_time("09:30", "15:59")           # 정규장
        h = rth.resample("1h", origin="start_day", offset="9h30min").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        bench_1h[b] = add_ind(h)
        dd = rth.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        dd["ma20"] = dd["close"].rolling(20).mean()
        dd["ema8"] = dd["close"].ewm(span=8, adjust=False).mean()
        dd["ema21"] = dd["close"].ewm(span=21, adjust=False).mean()
        dd["is_bear"] = (dd["close"] < dd["ma20"]) & (dd["ema8"] < dd["ema21"])
        daily_regime[b] = dd["is_bear"]
    for s in ALL:
        trade[s] = data[s].between_time("09:30", "15:59")
    return bench_1h, trade, daily_regime

def run(bench_1h, trade, daily_regime, cfg):
    """이벤트 드리븐: 1H 확정 신호 → 다음 5분봉(또는 1H봉) 시가 진입, 인트라바 SL/TP, 15:55 ET 강제청산."""
    trades = []
    open_pos = {}   # sym -> dict
    traded_today = set()
    for b, be in PAIRS:
        tgt = b if cfg["x1"] else be
        tdf = trade[tgt]
        b1h = bench_1h[b]
        sig_idx = b1h.index[(b1h["rsi2"] <= cfg["rsi"]) & (b1h["z20"] <= cfg["z"])]
        for sig_t in sig_idx:
            day = sig_t.date()
            # 전일 일봉 BEAR 확인
            reg = daily_regime[b]
            prior = reg[reg.index.date < day]
            if prior.empty or not bool(prior.iloc[-1]):
                continue
            if (day, tgt) in traded_today:
                continue
            # 진입: 신호봉 이후 첫 봉, 14:00 ET까지
            nxt = tdf.index[tdf.index > sig_t]
            if len(nxt) == 0:
                continue
            et = nxt[0]
            if et.date() != day or et.hour >= 14:
                continue
            traded_today.add((day, tgt))
            entry = float(tdf["open"].loc[et])
            # 이후 당일 봉으로 청산 시뮬
            bars = tdf.loc[et:].between_time("09:30", "15:59")
            bars = bars[bars.index.date == day]
            peak = entry
            exit_px, reason = None, "EOD"
            for bt_, br in bars.iloc[1:].iterrows():
                hi, lo = float(br["high"]), float(br["low"])
                peak = max(peak, hi)
                # 보수적: SL 우선
                if lo <= entry * (1 - cfg["sl"]):
                    exit_px, reason = entry * (1 - cfg["sl"]), "SL"
                    break
                if hi >= entry * (1 + cfg["tp"]):
                    exit_px, reason = entry * (1 + cfg["tp"]), "TP"
                    break
                if cfg["trail"]:
                    arm, drop = cfg["trail"]
                    if peak >= entry * (1 + arm) and lo <= peak * (1 - drop):
                        exit_px, reason = peak * (1 - drop), "TRAIL"
                        break
            if exit_px is None:
                exit_px = float(bars["close"].iloc[-1])
            ret = exit_px / entry - 1 - 2 * FEE
            trades.append(dict(sym=tgt, day=str(day), entry_t=str(et.time())[:5], ret=ret * 100,
                               pnl=NOTIONAL * ret, reason=reason))
    return pd.DataFrame(trades)

def metrics(tr):
    if len(tr) == 0:
        return dict(n=0, wr=0, pf=0, net=0, avg=0, worst=0)
    w = tr[tr.pnl > 0]; l = tr[tr.pnl <= 0]
    pf = w.pnl.sum() / abs(l.pnl.sum()) if len(l) and l.pnl.sum() != 0 else float("inf")
    return dict(n=len(tr), wr=len(w)/len(tr)*100, pf=pf, net=tr.pnl.sum(),
                avg=tr.pnl.mean(), worst=tr.ret.min())

def main():
    d5 = load("60d5m", "60d", "5m")
    d1h = load("730d1h", "730d", "1h")
    b5, t5, r5 = prep(d5)
    b1h, t1h, r1h = prep(d1h)
    print(f"60d/5m: {t5['QQQ'].index[0]} ~ {t5['QQQ'].index[-1]} | 730d/1h: {t1h['QQQ'].index[0]} ~ {t1h['QQQ'].index[-1]}\n")
    summary = {}
    for name, cfg in SCEN.items():
        m60 = metrics(run(b5, t5, r5, cfg))
        m730 = metrics(run(b1h, t1h, r1h, cfg))
        summary[name] = (m60, m730)
        ok = (m60["net"] > 0 and m730["net"] > 0 and m60["pf"] >= 1.3 and m730["pf"] >= 1.3
              and m60["n"] + m730["n"] >= 20 and m60["avg"] >= 15 and m60["worst"] >= -3.0)
        print(f"[{name}] {'✅ PASS' if ok else '❌ FAIL'}")
        print(f"  60d/5m : {m60['n']}건 WR {m60['wr']:.0f}% PF {m60['pf']:.2f} net {m60['net']:+.0f}$ avg {m60['avg']:+.1f}$ worst {m60['worst']:+.2f}%")
        print(f"  730d/1h: {m730['n']}건 WR {m730['wr']:.0f}% PF {m730['pf']:.2f} net {m730['net']:+.0f}$ avg {m730['avg']:+.1f}$ worst {m730['worst']:+.2f}%")
    pd.to_pickle(summary, os.path.join(DATA_DIR, "bear_rally_results.pkl"))

if __name__ == "__main__":
    main()
