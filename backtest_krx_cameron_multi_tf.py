# -*- coding: utf-8 -*-
"""
backtest_krx_cameron_multi_tf.py — Korean Stocks (KRX) Multi-Timeframe Ross Cameron Pullback Backtest
Testing:
- Timeframes: 1m (1분봉), 5m (5분봉), 15m (10~15분봉), 60m (1시간봉), Daily (일봉/4시간급 스윙)
- Tickers:
  1) KODEX 레버리지 (122630.KS)
  2) KODEX 코스닥150레버리지 (233740.KS) — 극단적 변동성
  3) 알테오젠 (196170.KQ) — 코스닥 대표 주도성장주
  4) SK하이닉스 (000660.KS) — 반도체 고베타 주도주
  5) 에코프로비엠 (247540.KQ) — 2차전지 고변동성주
- Domestic Costs:
  - 증권거래세: 0.18% (코스닥/코스피 매도 시)
  - 거래 수수료: 0.015% x 2 = 0.03%
  - 슬리피지: 0.05% x 2 = 0.10%
  - 총 라운드트립 비용: 약 0.31% 반영
"""

import sys
import os
import math
import yfinance as yf
import pandas as pd
import numpy as np

TICKERS = {
    "KODEX 레버리지": "122630.KS",
    "KODEX 코스닥150레버": "233740.KS",
    "SK하이닉스": "000660.KS",
    "알테오젠": "196170.KQ",
    "에코프로비엠": "247540.KQ"
}

INITIAL_CAPITAL = 10000000.0   # 1,000만원
POSITION_PCT = 0.20            # 1회 진입 20% (200만원)
KRX_COST_PCT = 0.0031          # 0.31% (거래세 0.18% + 수수료 0.03% + 슬리피지 0.10%)

TIMEFRAME_CONFIGS = [
    {"label": "1분봉 (1m 초단타)", "interval": "1m", "period": "5d", "min_surge": 0.004, "timeout": 30},
    {"label": "5분봉 (5m 단타)", "interval": "5m", "period": "1mo", "min_surge": 0.010, "timeout": 24},
    {"label": "15분봉 (15m 스캘핑)", "interval": "15m", "period": "1mo", "min_surge": 0.015, "timeout": 20},
    {"label": "60분봉 (1시간 스윙)", "interval": "60m", "period": "3mo", "min_surge": 0.025, "timeout": 18},
    {"label": "일봉 (1D/4시간급 스윙)", "interval": "1d", "period": "1y", "min_surge": 0.040, "timeout": 15}
]

def load_data(ticker_symbol: str, interval: str, period: str):
    try:
        df = yf.download(ticker_symbol, interval=interval, period=period, progress=False)
        if df.empty or len(df) < 30:
            return None
        # Handle multi-level columns if returned by yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        tp = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, 1)
        df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
        return df
    except Exception as e:
        return None

def simulate_cameron_kr(df: pd.DataFrame, min_surge: float, timeout_bars: int, rr_ratio: float = 2.0):
    capital = INITIAL_CAPITAL
    trades = []

    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    volumes = df["volume"].values
    vwaps = df["vwap"].values
    ema9s = df["ema9"].values
    n = len(df)

    in_pos = False
    entry_idx = 0
    entry_p = 0.0
    sl_p = 0.0
    tp_p = 0.0
    qty = 0

    for i in range(10, n):
        curr_h, curr_l, curr_c = highs[i], lows[i], closes[i]
        prev_o, prev_h, prev_l, prev_c = opens[i-1], highs[i-1], lows[i-1], closes[i-1]
        prev_v = volumes[i-1]

        # 1. 포지션 청산 감시
        if in_pos:
            # TP Hit
            if curr_h >= tp_p:
                exit_p = tp_p
                gross_pnl = (exit_p - entry_p) * qty
                net_pnl = gross_pnl - ((entry_p + exit_p) * qty * (KRX_COST_PCT / 2))
                capital += net_pnl
                trades.append({"pnl": net_pnl, "ret": (exit_p / entry_p - 1) * 100, "win": net_pnl > 0, "reason": "TP"})
                in_pos = False
                continue
            # SL Hit
            elif curr_l <= sl_p:
                exit_p = sl_p
                gross_pnl = (exit_p - entry_p) * qty
                net_pnl = gross_pnl - ((entry_p + exit_p) * qty * (KRX_COST_PCT / 2))
                capital += net_pnl
                trades.append({"pnl": net_pnl, "ret": (exit_p / entry_p - 1) * 100, "win": net_pnl > 0, "reason": "SL"})
                in_pos = False
                continue
            # Timeout
            elif (i - entry_idx) >= timeout_bars:
                exit_p = curr_c
                gross_pnl = (exit_p - entry_p) * qty
                net_pnl = gross_pnl - ((entry_p + exit_p) * qty * (KRX_COST_PCT / 2))
                capital += net_pnl
                trades.append({"pnl": net_pnl, "ret": (exit_p / entry_p - 1) * 100, "win": net_pnl > 0, "reason": "TIME"})
                in_pos = False
                continue

        # 2. 로스 카메론 1차 눌림목(First Pullback) 셋업 감지
        if not in_pos:
            prior_h = np.max(highs[max(0, i-6):i-1])
            prior_l = np.min(lows[max(0, i-6):i-1])
            prior_v_mean = np.mean(volumes[max(0, i-6):i-1])

            # 전봉이 음봉(눌림목) & 9 EMA 및 VWAP 지지
            if prev_c < prev_o and prev_c >= ema9s[i-1] * 0.998 and prev_l >= vwaps[i-1] * 0.995:
                surge = prior_h - prior_l
                if (surge / prior_l) >= min_surge:
                    retrace = (prior_h - prev_l) / surge
                    # 되돌림 55% 이내 & 거래량 감소
                    if retrace <= 0.55 and prev_v <= prior_v_mean * 1.20:
                        # 당봉에서 직전봉 고점 돌파 트리거
                        if curr_h >= prev_h:
                            risk = prev_h - prev_l
                            risk_pct = risk / prev_h
                            if 0.003 <= risk_pct <= 0.070:
                                in_pos = True
                                entry_p = prev_h
                                sl_p = prev_l
                                tp_p = prev_h + (risk * rr_ratio)
                                alloc = capital * POSITION_PCT
                                qty = math.floor(alloc / entry_p)
                                entry_idx = i

    df_t = pd.DataFrame(trades)
    if len(df_t) == 0:
        return {"net_pnl": 0, "ret_pct": 0, "win_rate": 0, "pf": 0, "trades": 0}

    wins = df_t[df_t["win"] == True]
    losses = df_t[df_t["win"] == False]
    gross_win = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 1e-6
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    return {
        "final_capital": capital,
        "net_pnl": df_t["pnl"].sum(),
        "ret_pct": (df_t["pnl"].sum() / INITIAL_CAPITAL) * 100,
        "win_rate": (len(wins) / len(df_t)) * 100,
        "pf": pf,
        "trades": len(df_t),
        "wins": len(wins),
        "losses": len(losses),
        "avg_win": wins["ret"].mean() if len(wins) > 0 else 0,
        "avg_loss": losses["ret"].mean() if len(losses) > 0 else 0
    }

def main():
    print("=" * 105)
    print("🇰🇷 [한국 주식 KRX] 로스 카메론 1차 눌림목(First Pullback) 타임프레임별 전수 백테스트")
    print(f"   초기 자본: 10,000,000원 | 1회 투입: 20% (200만원) | R:R 목표: 1:2.0")
    print(f"   국내 세금/비용: 총 0.31% (증권거래세 0.18% + 수수료 0.03% + 슬리피지 0.10%) 전건 차감")
    print(f"   테스트 종목군: {list(TICKERS.keys())}")
    print("=" * 105)

    summary_rows = []

    for cfg in TIMEFRAME_CONFIGS:
        tf_label = cfg["label"]
        print(f"\n⏳ [{tf_label}] 전 종목 시뮬레이션 중...")
        
        tf_trades_total = 0
        tf_pnl_total = 0.0
        all_trades_list = []

        for name, sym in TICKERS.items():
            df = load_data(sym, cfg["interval"], cfg["period"])
            if df is None:
                continue
            res = simulate_cameron_kr(df, min_surge=cfg["min_surge"], timeout_bars=cfg["timeout"], rr_ratio=2.0)
            tf_trades_total += res["trades"]
            tf_pnl_total += res["net_pnl"]
            if res["trades"] > 0:
                all_trades_list.append(res)

        # Aggregate metrics for timeframe
        if tf_trades_total > 0:
            total_wins = sum(r["wins"] for r in all_trades_list)
            win_rate = (total_wins / tf_trades_total) * 100
            total_gross_win = sum(r["net_pnl"] for r in all_trades_list if r["net_pnl"] > 0)
            total_gross_loss = abs(sum(r["net_pnl"] for r in all_trades_list if r["net_pnl"] < 0))
            pf = total_gross_win / total_gross_loss if total_gross_loss > 0 else float("inf")
            ret_pct = (tf_pnl_total / INITIAL_CAPITAL) * 100
        else:
            win_rate = 0.0
            pf = 0.0
            ret_pct = 0.0

        summary_rows.append({
            "label": tf_label,
            "net_pnl": tf_pnl_total,
            "ret_pct": ret_pct,
            "win_rate": win_rate,
            "pf": pf,
            "trades": tf_trades_total
        })

    print("\n" + "=" * 105)
    print(f"{'타임프레임 (Timeframe)':<28} | {'총 순손익(원)':<18} | {'수익률(%)':<10} | {'승률':<8} | {'PF':<6} | {'총 거래수':<8}")
    print("-" * 105)

    for r in summary_rows:
        pnl_str = f"{r['net_pnl']:+,.0f}원"
        ret_str = f"{r['ret_pct']:+.2f}%"
        win_str = f"{r['win_rate']:.1f}%"
        pf_str = f"{r['pf']:.2f}"
        tr_str = f"{r['trades']}회"
        print(f"{r['label']:<28} | {pnl_str:<18} | {ret_str:<10} | {win_str:<8} | {pf_str:<6} | {tr_str:<8}")

    print("=" * 105)

if __name__ == "__main__":
    main()
