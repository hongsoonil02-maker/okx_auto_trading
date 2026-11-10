#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/backtest_alpaca_trend_hybrid.py
=============================================================================
T9 하이브리드: MA200 느린 레짐 게이트 + 4섹터 일봉 레이더 빠른 방어 오버레이 — 2026-11-10
설계 근거: R1/R2에서 MA200 단독은 갈아내리기 하락 초입 후행(AC4 미달), 한편 현행 4섹터 레이더는
           11-05에 MA200과 동일 시점 방어. 섹터 레벨(close<MA20 & EMA8<EMA21)은 지수 MA200보다
           수일 빠르므로 결합 시 AC4 결함 해소 가설.

■ 규칙 (사전 등록):
  - 핵심 게이트: QQQ 확정 종가 > MA(200) (T9c는 MA150)
  - 빠른 방어: DIA/SPY/QQQ/SOXX 중 일봉 BEAR(종가<MA20 & EMA8<EMA21) >= 3/4 → 익일 시가 현금화
  - 재진입: QQQ > MA(200) 且 BEAR 카운트 <= 1/4 → 익일 시가 재진입
  - 비히클: QQQ (T9a) / TQQQ (T9b, T9c)

■ 사전 등록 채택 기준: R1과 동일 AC1~AC6
  (AC1 10y CAGR>=12% / AC2 MDD>=-35% / AC3 730d>0 / AC4 60d>=-1% / AC5 <=30회/yr / AC6 PF>=1.3)
  통과 변형 중 MDD 대비 CAGR 최고 채택. 전부 미달 시 봇 중지(사용자 사전 승인 경로).
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = "/home/hongsoonil02/quant_system/data"
FEE = 0.0005
BENCHS = ["DIA", "SPY", "QQQ", "SOXX"]
ALL = sorted(set(BENCHS + ["TQQQ"]))

VARIANTS = {
    "T9a_MA200+레이더_QQQ":  dict(ma=200, vehicle="QQQ"),
    "T9b_MA200+레이더_TQQQ": dict(ma=200, vehicle="TQQQ"),
    "T9c_MA150+레이더_TQQQ": dict(ma=150, vehicle="TQQQ"),
}

def load():
    f = os.path.join(DATA_DIR, "trend_hybrid_11y.pkl")
    if os.path.exists(f):
        return pd.read_pickle(f)
    raw = yf.download(ALL, period="11y", interval="1d", progress=False, group_by="ticker", auto_adjust=True)
    out = {}
    for s in ALL:
        df = raw[s].dropna(subset=["Close"]).copy()
        df.columns = [c.lower() for c in df.columns]
        out[s] = df
    pd.to_pickle(out, f)
    return out

def bear_breadth(data):
    cnt = None
    for b in BENCHS:
        d = data[b]["close"]
        bear = (d < d.rolling(20).mean()) & (d.ewm(span=8, adjust=False).mean() < d.ewm(span=21, adjust=False).mean())
        cnt = bear.astype(int) if cnt is None else cnt + bear.astype(int)
    return cnt

def run(data, ma, vehicle):
    q = data["QQQ"]["close"]
    above = q > q.rolling(ma).mean()
    bc = bear_breadth(data)
    idx = q.index.intersection(bc.index)
    above, bc = above.loc[idx], bc.loc[idx]
    # 상태기계: 방어 발동(bc>=3) → 현금 / 재진입(above & bc<=1)
    state, cur = [], False
    for a, b in zip(above, bc):
        if b >= 3:
            cur = False
        elif a and b <= 1:
            cur = True
        state.append(cur)
    want = pd.Series(state, index=idx)

    veh = data[vehicle][["open"]].reindex(idx).ffill()
    df = pd.DataFrame({"want": want, "open": veh["open"]}).dropna()
    df["pos"] = df["want"].shift(1).fillna(False).astype(bool)
    ret = df["open"].pct_change().fillna(0.0)
    flip = df["pos"].ne(df["pos"].shift().fillna(False))
    df["eq"] = (1 + ret * df["pos"] - flip.astype(float) * FEE).cumprod()
    flips = df.index[flip].tolist()
    trades, in_pos, ent = [], False, None
    for i_ in flips:
        if not in_pos:
            ent = i_; in_pos = True
        else:
            trades.append((ent, i_)); in_pos = False
    if in_pos:
        trades.append((ent, df.index[-1]))
    tr = pd.DataFrame([(str(a.date()), str(b.date()),
                        (df["open"].loc[b] / df["open"].loc[a] - 1 - 2 * FEE) * 100) for a, b in trades],
                      columns=["entry", "exit", "ret_pct"])
    return df, tr

def metrics(df, from_date=None):
    d = df if from_date is None else df[df.index >= from_date]
    if len(d) < 5:
        return dict(cagr=0, mdd=0, ret=0)
    eq = d["eq"] / d["eq"].iloc[0]
    yrs = (d.index[-1] - d.index[0]).days / 365.25
    cagr = (eq.iloc[-1] ** (1 / yrs) - 1) * 100 if yrs > 0 else 0
    mdd = ((eq / eq.cummax()) - 1).min() * 100
    return dict(cagr=cagr, mdd=mdd, ret=(eq.iloc[-1] - 1) * 100)

def main():
    data = load()
    print(f"기간: {data['QQQ'].index[0].date()} ~ {data['QQQ'].index[-1].date()}\n")
    best = None
    for name, v in VARIANTS.items():
        df, tr = run(data, **v)
        m_all, m_730 = metrics(df), metrics(df, df.index[-1] - pd.Timedelta(days=730))
        m_60 = metrics(df, df.index[-1] - pd.Timedelta(days=60))
        yrs = (df.index[-1] - df.index[0]).days / 365.25
        wins = tr[tr.ret_pct > 0]; losses = tr[tr.ret_pct <= 0]
        pf = wins.ret_pct.sum() / abs(losses.ret_pct.sum()) if len(losses) and losses.ret_pct.sum() != 0 else float("inf")
        ok = (m_all["cagr"] >= 12 and m_all["mdd"] >= -35 and m_730["ret"] > 0
              and m_60["ret"] >= -1 and len(tr) / yrs <= 30 and pf >= 1.3)
        score = m_all["cagr"] / abs(m_all["mdd"]) if m_all["mdd"] < 0 else 0
        if ok and (best is None or score > best[2]):
            best = (name, v, score, m_all, m_730)
        # 최근 방어 발동일 확인용
        recent_bc = bear_breadth(data).tail(8).to_dict()
        print(f"[{name}] 10y CAGR {m_all['cagr']:+.1f}% MDD {m_all['mdd']:+.1f}% | 730d {m_730['ret']:+.1f}% | "
              f"60d {m_60['ret']:+.2f}% | 매매 {len(tr)}회({len(tr)/yrs:.1f}/yr) PF {pf:.2f} | {'✅ PASS' if ok else '❌ FAIL'}")
    print(f"\n최근 8일 BEAR breadth: {[(str(k.date()), v) for k, v in recent_bc.items()]}")
    if best:
        print(f"🏆 채택 후보: {best[0]} (CAGR/MDD {best[2]:.2f})")
    else:
        print("❌ 전 변형 미달 — 봇 중지 안으로 회귀")
    pd.to_pickle(best, os.path.join(DATA_DIR, "trend_hybrid_best.pkl"))

if __name__ == "__main__":
    main()
