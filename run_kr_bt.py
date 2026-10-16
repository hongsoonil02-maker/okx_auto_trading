# -*- coding: utf-8 -*-
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

UNIVERSE = {
    "005930.KS": "삼성전자",
    "000660.KS": "SK하이닉스",
    "005380.KS": "현대차",
    "000270.KS": "기아",
    "068270.KS": "셀트리온",
    "042700.KS": "한미반도체",
    "012450.KS": "한화에어로스페이스",
    "064350.KS": "현대로템",
    "267260.KS": "HD현대일렉트릭",
    "298040.KS": "효성중공업",
    "196170.KQ": "알테오젠",
    "003230.KS": "삼양식품",
    "009540.KS": "HD한국조선해양",
    "034020.KS": "두산에너빌리티"
}

kospi = yf.Ticker("^KS11").history(period="3y")
kospi_sma60 = kospi["Close"].rolling(60).mean()

dfs = {}
for s in UNIVERSE:
    df = yf.Ticker(s).history(period="3y")
    if not df.empty and len(df) > 70:
        df["sma20"] = df["Close"].rolling(20).mean()
        df["sma60"] = df["Close"].rolling(60).mean()
        df["dc_h20"] = df["High"].shift(1).rolling(20).max()
        df["dc_l10"] = df["Low"].shift(1).rolling(10).min()
        tr = np.maximum(df["High"] - df["Low"], np.maximum(abs(df["High"] - df["Close"].shift(1)), abs(df["Low"] - df["Close"].shift(1))))
        df["atr14"] = tr.rolling(14).mean()
        dfs[s] = df

dates = sorted(list(set(kospi.index).intersection(*[set(df.index) for df in dfs.values()])))
capital = 10_000_000.0
positions = {}
trades = []
curve = []
FEE = 0.0020

for dt in dates[60:]:
    k_bull = kospi.loc[dt, "Close"] >= kospi_sma60.loc[dt]
    # exits
    for s in list(positions.keys()):
        row = dfs[s].loc[dt]
        pos = positions[s]
        px = row["Close"]
        pos["peak"] = max(pos["peak"], row["High"])
        trail = pos["peak"] - 2.0 * pos["atr"]
        hard = max(pos["entry"] * 0.93, pos["entry"] - 2.5 * pos["atr"])
        eff = max(trail, row["dc_l10"])
        exit_r = None
        if px < hard:
            exit_r = "HARD_STOP"
        elif px < eff and px < pos["peak"]:
            exit_r = "TRAIL"
        if exit_r:
            ret = (px / pos["entry"] - 1) - FEE * 2
            pnl = pos["qty"] * px * (1 - FEE) - pos["qty"] * pos["entry"] * (1 + FEE)
            capital += pos["qty"] * px * (1 - FEE)
            trades.append({"sym": UNIVERSE[s], "ret": ret*100, "pnl": pnl, "reason": exit_r})
            del positions[s]
            
    # entries
    if k_bull and len(positions) < 3:
        cand = []
        for s, df in dfs.items():
            if s in positions:
                continue
            r = df.loc[dt]
            px = r["Close"]
            if px > r["dc_h20"] and px > r["sma60"] and r["sma20"] >= r["sma60"]:
                rs = px / df.loc[:dt, "Close"].iloc[-60] - 1
                dist = (px - r["sma60"]) / r["sma60"]
                cand.append((s, px, rs * 0.7 + dist * 0.3, r["atr14"]))
        cand.sort(key=lambda x: x[2], reverse=True)
        for s, px, sc, atr in cand:
            if len(positions) >= 3:
                break
            eq_now = capital + sum(p["qty"] * dfs[k].loc[dt, "Close"] for k, p in positions.items())
            qty = int((eq_now * 0.30) / px)
            cost = qty * px * (1 + FEE)
            if qty > 0 and capital >= cost:
                capital -= cost
                positions[s] = {"qty": qty, "entry": px, "peak": px, "atr": atr}
                
    eq_val = capital + sum(p["qty"] * dfs[k].loc[dt, "Close"] for k, p in positions.items())
    curve.append((dt, eq_val))

s_eq = pd.Series({d: v for d, v in curve})
arr = np.array([t["pnl"] for t in trades])
pf = (arr[arr > 0].sum() / -arr[arr <= 0].sum()) if (arr <= 0).sum() != 0 else 0
cagr = (s_eq.iloc[-1] / 10_000_000.0) ** (1 / ((s_eq.index[-1] - s_eq.index[0]).days / 365.25)) - 1
mdd = ((s_eq.cummax() - s_eq) / s_eq.cummax()).max() * 100

print(f"=== KIS 한국 주식 Trend Trader 결과 (최근 3년) ===")
print(f"거래건수: {len(trades)}건 | 승률: {(arr > 0).mean()*100:.1f}% | PF: {pf:.2f}")
print(f"최종자산: ₩{s_eq.iloc[-1]:,.0f} (시작 ₩10,000,000) | 총수익률: {(s_eq.iloc[-1]/10000000.0 - 1)*100:+.1f}% | CAGR: {cagr*100:+.1f}% | MDD: {mdd:.1f}%")
print(f"평균 익절: +{np.mean([t['ret'] for t in trades if t['ret']>0]):.1f}% | 평균 손절: {np.mean([t['ret'] for t in trades if t['ret']<=0]):.1f}%")
