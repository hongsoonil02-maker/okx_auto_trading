#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/backtest_alpaca_ma200_trend.py
=============================================================================
Alpaca 후속 엔진 "Trend Guardian" 정밀 검증 — 2026-11-10
규칙: QQQ 일봉 확정 종가가 MA(N) 위 → 다음날 시가에 투자(비히클), 아래 → 현금(0%).

■ 사전 등록 채택 기준 (실행 전 확정):
  AC1: 10년 CAGR >= 12%
  AC2: 10년 MDD >= -35% (QQQ B&H의 2022 낙폭 수준 이내)
  AC3: 최근 730일 창 순수익 > 0
  AC4: 최근 60일 하락장 창 손실 >= -1%
  AC5: 연평균 왕복매매 <= 30회 (저빈도 요건)
  AC6: 거래 PF >= 1.3
  전부 충족 변형 중 "MDD 대비 CAGR 최고"를 채택. 전부 미달 시 봇 중지 안으로 회귀.

■ 근사치: 신호=확정 종가, 집행=익일 시가, 비용 5bps x2, 현금 수익률 0%.
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = "/home/hongsoonil02/quant_system/data"
FEE = 0.0005

VARIANTS = {
    "T1_MA200_TQQQ":        dict(ma=200, vehicle="TQQQ", confirm=1, slope=False),
    "T2_MA200_QQQ":         dict(ma=200, vehicle="QQQ",  confirm=1, slope=False),
    "T3_MA150_TQQQ":        dict(ma=150, vehicle="TQQQ", confirm=1, slope=False),
    "T4_MA200+기울기_TQQQ":  dict(ma=200, vehicle="TQQQ", confirm=1, slope=True),
    "T5_MA200_3일확인_TQQQ": dict(ma=200, vehicle="TQQQ", confirm=3, slope=False),
    "T6_MA250_TQQQ":        dict(ma=250, vehicle="TQQQ", confirm=1, slope=False),
    # ── Round 2 (2026-11-10 00:45, 실행 전 등록): R1 실패 원인(급락 시 MA200 이탈 지연) 겨냥 ──
    # panic: QQQ 확정 일수익률 <= -4.0% 시 익일 시가 즉시 청산, MA200 재탈환 시에만 재진입
    # 기준: R1과 동일한 AC1~AC6 (AC4 -1% 유지)
    "T7_T2+패닉컷":  dict(ma=200, vehicle="QQQ",  confirm=1, slope=False, panic=-4.0),
    "T8_T1+패닉컷":  dict(ma=200, vehicle="TQQQ", confirm=1, slope=False, panic=-4.0),
}

def load():
    f = os.path.join(DATA_DIR, "trend_guardian_10y.pkl")
    if os.path.exists(f):
        return pd.read_pickle(f)
    raw = yf.download(["QQQ", "TQQQ"], period="11y", interval="1d", progress=False, group_by="ticker", auto_adjust=True)
    out = {}
    for s in ("QQQ", "TQQQ"):
        df = raw[s].dropna(subset=["Close"]).copy()
        df.columns = [c.lower() for c in df.columns]
        out[s] = df
    pd.to_pickle(out, f)
    return out

def run(data, ma, vehicle, confirm, slope, panic=None):
    sig = data["QQQ"].copy()
    sig["ma"] = sig["close"].rolling(ma).mean()
    above = sig["close"] > sig["ma"]
    if slope:
        above = above & (sig["ma"] > sig["ma"].shift(5))
    if confirm > 1:
        above = above.rolling(confirm).sum() >= confirm
        below_ok = (sig["close"] < sig["ma"]).rolling(confirm).sum() >= confirm
        state = above.where(above | below_ok).ffill().fillna(False)  # 확인 전까지 기존 상태 유지
    else:
        state = above
    if panic is not None:
        daily_ret = sig["close"].pct_change() * 100
        panic_hit = daily_ret <= panic
        # 패닉 확정 → 현금, MA200 재탈환 시에만 해제
        state_list, cur = [], False
        for a, p in zip(state, panic_hit):
            if p:
                cur = False
            elif a:
                cur = True
            state_list.append(cur)
        state = pd.Series(state_list, index=sig.index)
    sig["want"] = state.astype(bool)

    veh = data[vehicle][["open", "close"]].copy()
    df = sig[["want"]].join(veh, how="inner").dropna()
    # 신호 확정(당일 종가) → 익일 시가 집행: 포지션은 want.shift(1), 수익은 open->open 체인
    df["pos"] = df["want"].shift(1).fillna(False).astype(bool)
    o = df["open"]
    ret = o.pct_change().fillna(0.0)          # 시가 대비 시가 수익 (보유 구간 근사)
    # 집행일 슬립: 포지션 변경 시 비용 차감
    flip = df["pos"].ne(df["pos"].shift().fillna(False))
    cost = flip.astype(float) * FEE
    strat_ret = ret * df["pos"] - cost
    df["eq"] = (1 + strat_ret).cumprod()
    # 거래 명세 (진입일~청산일)
    flips = df.index[flip].tolist()
    trades = []
    i = 0
    in_pos = False
    for idx in flips:
        if not in_pos:
            ent = idx; in_pos = True
        else:
            trades.append((ent, idx)); in_pos = False
    if in_pos:
        trades.append((ent, df.index[-1]))
    tr = pd.DataFrame([(str(a.date()), str(b.date()),
                        (df["open"].loc[b] / df["open"].loc[a] - 1 - 2 * FEE) * 100) for a, b in trades],
                      columns=["entry", "exit", "ret_pct"])
    return df, tr

def metrics(df, tr, from_date=None):
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
    # 벤치마크
    for s in ("QQQ", "TQQQ"):
        c = data[s]["close"]
        eq = c / c.iloc[0]
        yrs = (c.index[-1] - c.index[0]).days / 365.25
        mdd = ((eq / eq.cummax()) - 1).min() * 100
        print(f"[B&H {s}] 10y CAGR {((eq.iloc[-1])**(1/yrs)-1)*100:.1f}% MDD {mdd:.1f}%")
    print()
    best = None
    for name, v in VARIANTS.items():
        df, tr = run(data, **v)
        m_all = metrics(df, tr)
        m_730 = metrics(df, tr, df.index[-1] - pd.Timedelta(days=730))
        m_60 = metrics(df, tr, df.index[-1] - pd.Timedelta(days=60))
        yrs = (df.index[-1] - df.index[0]).days / 365.25
        n_tr = len(tr)
        wins = tr[tr.ret_pct > 0]; losses = tr[tr.ret_pct <= 0]
        pf = wins.ret_pct.sum() / abs(losses.ret_pct.sum()) if len(losses) and losses.ret_pct.sum() != 0 else float("inf")
        ok = (m_all["cagr"] >= 12 and m_all["mdd"] >= -35 and m_730["ret"] > 0
              and m_60["ret"] >= -1 and n_tr / yrs <= 30 and pf >= 1.3)
        score = m_all["cagr"] / abs(m_all["mdd"]) if m_all["mdd"] < 0 else 0
        if ok and (best is None or score > best[2]):
            best = (name, v, score, m_all, m_730)
        print(f"[{name}] 10y CAGR {m_all['cagr']:+.1f}% MDD {m_all['mdd']:+.1f}% | 730d {m_730['ret']:+.1f}% | "
              f"60d {m_60['ret']:+.2f}% | 매매 {n_tr}회({n_tr/yrs:.1f}/yr) PF {pf:.2f} | {'✅ PASS' if ok else '❌ FAIL'}")
    if best:
        print(f"\n🏆 채택 후보: {best[0]} (CAGR/MDD {best[2]:.2f}, 10y CAGR {best[3]['cagr']:+.1f}%, MDD {best[3]['mdd']:+.1f}%, 730d {best[4]['ret']:+.1f}%)")
    else:
        print("\n❌ 전 변형 미달 — 봇 중지 안으로 회귀")
    pd.to_pickle(best, os.path.join(DATA_DIR, "trend_guardian_best.pkl"))

if __name__ == "__main__":
    main()
