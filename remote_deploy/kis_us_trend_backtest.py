# -*- coding: utf-8 -*-
"""
kis_us_trend_backtest.py — KRX 추세추종 엔진(kis_trend_trader.py)의 미국 유니버스 이식 검증
전략 (kis_trend_trader.py와 동일 로직, 일봉 근사):
- 진입: 20일 돈치안 신고가 돌파 + 종가 > SMA60 + SMA20 >= SMA60 + QQQ 레짐(QQQ 종가 >= QQQ SMA60)
- 랭킹: RS(60일 초과수익 vs QQQ) 70% + 60일선 이격도 30%
- 청산: ATR(진입시점 고정) 2.0x 트레일링(고점 대비), 하드스탑 max(-7%, -2.5xATR), 10일 최저가 이탈
- 규모: 최대 3종목, 종목당 30%, 수수료 편도 0.25%
주의: 유니버스가 현재 시점 승자들이라 생존자 편향이 +방향으로 작용함. 일봉 종가 근사(실제 봇은 장중 체크).
"""
import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

_DEFAULT_UNIVERSE = "TQQQ,NVDA,TSLA,MSTR,PLTR,AMD,COIN,AVGO,META,AAPL"
UNIVERSE = [s.strip().upper() for s in os.environ.get("US_TREND_UNIVERSE", _DEFAULT_UNIVERSE).split(",") if s.strip()]
FEE_SIDE = 0.0025
TRAIL_MULT = float(os.environ.get("US_TREND_TRAIL_MULT", "2.0"))
DC_LOW_N = int(os.environ.get("US_TREND_DC_LOW", "10"))
MAX_POS = 3
ALLOC = 0.30
START = 7800.0


def load():
    tickers = UNIVERSE + ["QQQ"]
    df = yf.download(tickers, period="10y", interval="1d", auto_adjust=True,
                     progress=False, group_by="ticker", threads=True)
    out = {}
    for t in tickers:
        try:
            d = df[t].dropna().copy()
            if len(d) > 80:
                out[t] = d
        except Exception:
            pass
    return out


def prep(d):
    c, h, l = d["Close"], d["High"], d["Low"]
    d["sma20"] = c.rolling(20).mean()
    d["sma60"] = c.rolling(60).mean()
    tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
    d["atr14"] = tr.rolling(14).mean()
    d["dc_hi20"] = h.shift(1).rolling(20).max()   # 전일 기준 20일 고가 (오늘 제외)
    d["dc_lo10"] = l.shift(1).rolling(DC_LOW_N).min()
    d["ret60"] = c / c.shift(60) - 1.0
    return d


def simulate(data, qqq, label):
    cal = qqq.index
    q_ret60 = qqq["ret60"]
    q_regime = qqq["Close"] >= qqq["sma60"]

    cash = START
    pos = {}
    trades = []
    curve = []
    invested_days = 0

    for t in range(61, len(cal)):
        day = cal[t]
        bullish = bool(q_regime.iloc[t])

        # ── Exits (일봉 종가 근사) ──
        for sym in list(pos.keys()):
            if sym not in data or day not in data[sym].index:
                continue
            row = data[sym].loc[day]
            close = float(row["Close"])
            if not np.isfinite(close):
                continue
            p = pos[sym]
            p["peak"] = max(p["peak"], close)
            trail = p["peak"] - TRAIL_MULT * p["atr"]
            hard = max(p["entry"] * 0.93, p["entry"] - 2.5 * p["atr"])
            lo10 = float(row["dc_lo10"]) if np.isfinite(row["dc_lo10"]) else 0.0
            stop = max(trail, lo10)
            reason = None
            if close < hard:
                reason = "SL"
            elif close < stop and close < p["peak"]:
                reason = "TRAIL"
            if reason:
                proceeds = p["qty"] * close * (1 - FEE_SIDE)
                cash += proceeds
                ret = (close / p["entry"] - 1.0) * 100.0
                trades.append({"sym": sym, "ret": ret, "days": (day - p["entry_day"]).days, "reason": reason})
                pos.pop(sym)

        # ── Entries ──
        if bullish and len(pos) < MAX_POS:
            cands = []
            for sym in UNIVERSE:
                if sym in pos or sym not in data or day not in data[sym].index:
                    continue
                row = data[sym].loc[day]
                close = float(row["Close"])
                if not all(np.isfinite([close, row["dc_hi20"], row["sma60"], row["sma20"], row["ret60"], row["atr14"]])):
                    continue
                if close > row["dc_hi20"] and close > row["sma60"] and row["sma20"] >= row["sma60"]:
                    rs = (float(row["ret60"]) - float(q_ret60.iloc[t])) * 100.0
                    dist = (close / float(row["sma60"]) - 1.0) * 100.0
                    score = rs * 0.7 + dist * 0.3
                    cands.append((score, sym, close, float(row["atr14"])))
            cands.sort(reverse=True)
            for score, sym, close, atr in cands:
                if len(pos) >= MAX_POS:
                    break
                equity = cash + sum(p["qty"] * float(data[s].loc[day, "Close"]) for s, p in pos.items()
                                    if day in data[s].index and np.isfinite(data[s].loc[day, "Close"]))
                alloc = min(equity * ALLOC, cash * 0.95)
                qty = int(alloc / (close * (1 + FEE_SIDE)))
                if qty < 1:
                    continue
                cash -= qty * close * (1 + FEE_SIDE)
                pos[sym] = {"qty": qty, "entry": close, "peak": close, "atr": atr, "entry_day": day}

        eq = cash + sum(p["qty"] * float(data[s].loc[day, "Close"]) for s, p in pos.items()
                        if day in data[s].index and np.isfinite(data[s].loc[day, "Close"]))
        curve.append((day, eq))
        if pos:
            invested_days += 1

    eq = pd.Series([v for _, v in curve], index=[d for d, _ in curve])
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    mdd = ((eq.cummax() - eq) / eq.cummax()).max()
    sharpe = (eq.pct_change().mean() / (eq.pct_change().std() + 1e-12)) * np.sqrt(252)
    n = len(trades)
    wins = sum(1 for x in trades if x["ret"] > 0)
    avg_hold = np.mean([x["days"] for x in trades]) if trades else 0
    avg_win = np.mean([x["ret"] for x in trades if x["ret"] > 0]) if wins else 0
    losses = [x["ret"] for x in trades if x["ret"] <= 0]
    avg_loss = np.mean(losses) if losses else 0

    print(f"\n=== {label} ===")
    print(f"기간: {eq.index[0].date()} ~ {eq.index[-1].date()} ({years:.1f}y) | 최종자산: ${eq.iloc[-1]:,.0f} (시작 ${START:,.0f})")
    print(f"CAGR: {cagr*100:+.1f}% | MDD: {mdd*100:.1f}% | Sharpe: {sharpe:.2f} | 투자일수 비중: {invested_days/len(curve)*100:.0f}%")
    print(f"거래: {n}건 | 승률: {wins/n*100 if n else 0:.0f}% | 평균보유: {avg_hold:.0f}일 | 평균익절: {avg_win:+.1f}% | 평균손절: {avg_loss:+.1f}%")
    by_reason = {}
    for x in trades:
        by_reason.setdefault(x["reason"], []).append(x["ret"])
    print("청산 사유별:", {k: f"{len(v)}건(평균 {np.mean(v):+.1f}%)" for k, v in by_reason.items()})
    return {"cagr": cagr, "mdd": mdd, "sharpe": sharpe, "trades": n, "win": wins / n if n else 0, "eq": eq}


def bh(data, sym, label, cal):
    d = data[sym]
    d = d[d.index >= cal[0]]
    r = d["Close"].iloc[-1] / d["Close"].iloc[0] - 1
    years = (d.index[-1] - d.index[0]).days / 365.25
    cagr = (1 + r) ** (1 / years) - 1
    mdd = ((d["Close"].cummax() - d["Close"]) / d["Close"].cummax()).max()
    print(f"{label}: CAGR {cagr*100:+.1f}% | MDD {mdd*100:.1f}%")


def main():
    raw = load()
    print(f"로드 완료: {list(raw.keys())}")
    data = {t: prep(d.copy()) for t, d in raw.items()}
    qqq = data["QQQ"]

    simulate(data, qqq, f"Donchian 추세추종 | trail {TRAIL_MULT}xATR | exit low{DC_LOW_N} | 수수료 0.25%/편도")

    print("\n--- Buy & Hold 참조 ---")
    cal = [d for d in qqq.index][61:]
    bh(data, "QQQ", "QQQ B&H", cal)
    if "TQQQ" in data:
        bh(data, "TQQQ", "TQQQ B&H", cal)

    print("\n⚠️ 생존자 편향 주의: 유니버스는 현재 시점의 승자 10종. 일봉 종가 근사(실제 봇은 장중 30초 체크).")
    print("⚠️ 3종목 집중 전략 특성상 결과 분산이 큼. 과거 성과는 미래를 보장하지 않음.")


if __name__ == "__main__":
    main()
