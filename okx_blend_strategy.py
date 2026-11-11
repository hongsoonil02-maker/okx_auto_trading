# -*- coding: utf-8 -*-
"""
okx_blend_strategy.py — Pure Signal & Sizing Module for OKX BLEND 65/35 Strategy
================================================================================
Matches tools/research_okx_alpha_v3.py & v4.py 1:1 (zero lookahead):
  - 65% Multi-Speed 4H Donchian Trend (Long-Only, 8 Majors, BTC 1D EMA50 regime gate,
    3 speeds: 24/12, 36/18, 48/24, 30d annualized inverse-volatility target 0.55)
  - 35% 1D RSI(2) Pullback Mean-Reversion (Long-Only, Core-3 BTC/ETH/SOL,
    Close > MA100 & RSI2 < 10 entry, Close > MA5 exit, vol target 0.50)
"""

from typing import Dict, List, Sequence, Tuple, Any
import numpy as np
import pandas as pd

SYMS_8 = [
    "BTC-USDT-SWAP",
    "ETH-USDT-SWAP",
    "SOL-USDT-SWAP",
    "XRP-USDT-SWAP",
    "DOGE-USDT-SWAP",
    "AVAX-USDT-SWAP",
    "LINK-USDT-SWAP",
    "ADA-USDT-SWAP",
]
CORE_3 = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]

TREND_BLEND_WEIGHT = 0.65
MR_BLEND_WEIGHT = 0.35
TREND_SPEEDS = ((24, 12), (36, 18), (48, 24))
TREND_VOL_TARGET = 0.55
MR_VOL_TARGET = 0.50
MR_RSI_THRESHOLD = 10.0
MAX_GROSS_WEIGHT = 2.0


def compute_wilder_rsi2(closes: pd.Series) -> pd.Series:
    d = closes.diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + up / dn.replace(0, np.nan))


def compute_4h_donchian_weights(
    h4: pd.DataFrame,
    l4: pd.DataFrame,
    c4: pd.DataFrame,
    c1_btc: pd.Series,
    syms: Sequence[str] = SYMS_8,
    speeds: Sequence[Tuple[int, int]] = TREND_SPEEDS,
    vol_target: float = TREND_VOL_TARGET,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Replays confirmed 4H Donchian state machines for each speed across `syms`
    and averages target weights across `speeds`.
    `c1_btc` contains confirmed daily closes of BTC-USDT-SWAP (no lookahead).
    """
    n_syms = len(syms)
    if len(c4) < 200 or len(c1_btc) < 55:
        raise ValueError(f"Insufficient history for 4H Donchian: 4H={len(c4)}, 1D_BTC={len(c1_btc)}")

    # Align confirmed daily BTC > EMA50 onto 4H timestamps (any 4H bar at or after
    # the daily bar close timestamp sees that completed day's regime).
    btc_ema50 = c1_btc.ewm(span=50, adjust=False).mean()
    # In live usage, `c1_btc` index is bar open time (e.g. T00:00 of day D-1, closed at T+24h).
    # In backtest (`donchian_8_4h`), c1_btc was shifted by 1 day and ffilled on 4H bar open times.
    # Shifting index by +24h (one daily bar) makes each completed day visible strictly after it closes!
    btc_regime_raw = (c1_btc > btc_ema50)
    btc_regime_shifted = pd.Series(btc_regime_raw.values, index=c1_btc.index + pd.Timedelta(hours=24))
    btc_ok_long = btc_regime_shifted.reindex(c4.index, method="ffill").fillna(False)

    vol = c4.pct_change().rolling(180).std() * np.sqrt(365 * 6)
    avg_weights: Dict[str, float] = {s: 0.0 for s in syms}
    speed_diag: Dict[str, List[str]] = {}

    for entry_b, exit_b in speeds:
        hi_n = h4.shift(1).rolling(entry_b).max()
        lo_ex = l4.shift(1).rolling(exit_b).min()
        active_syms = []

        for s in syms:
            st = 0
            w_last = 0.0
            start_i = max(entry_b, 190)
            for i in range(start_i, len(c4)):
                px = float(c4[s].iloc[i])
                v_raw = float(vol[s].iloc[i])
                v = 0.20 if (pd.isna(v_raw) or v_raw < 0.20) else v_raw
                w_abs = min(2.0 / n_syms, (vol_target / n_syms) / v)

                if st == 1 and px < float(lo_ex[s].iloc[i]):
                    st = 0
                if st == 0 and px > float(hi_n[s].iloc[i]) and bool(btc_ok_long.iloc[i]):
                    st = 1
                w_last = st * w_abs
            if w_last > 0:
                avg_weights[s] += w_last / len(speeds)
                active_syms.append(s.split("-")[0])
        speed_diag[f"{entry_b}/{exit_b}"] = active_syms

    return avg_weights, {
        "btc_1d_bull": bool(btc_ok_long.iloc[-1]),
        "btc_1d_px": round(float(c1_btc.iloc[-1]), 2),
        "btc_1d_ema50": round(float(btc_ema50.iloc[-1]), 2),
        "speeds": speed_diag,
    }


def compute_1d_mr_weights(
    c1: pd.DataFrame,
    syms: Sequence[str] = CORE_3,
    rsi_th: float = MR_RSI_THRESHOLD,
    vol_target: float = MR_VOL_TARGET,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Replays confirmed 1D RSI(2) Pullback inside MA100 uptrend across `syms`.
    Matches tools/research_okx_alpha_v4.py (`mr_1d`) 1:1.
    """
    n_syms = len(syms)
    if len(c1) < 110:
        raise ValueError(f"Insufficient 1D history for MR_1D: {len(c1)} < 110")

    weights: Dict[str, float] = {s: 0.0 for s in syms}
    active = []
    rsi_snap = {}

    for s in syms:
        px_s = c1[s]
        r2 = compute_wilder_rsi2(px_s)
        ma100 = px_s.rolling(100).mean()
        ma5 = px_s.rolling(5).mean()
        vol = px_s.pct_change().rolling(30).std() * np.sqrt(365)

        held = False
        w_last = 0.0
        for i in range(105, len(px_s)):
            px = float(px_s.iloc[i])
            if (not held) and px > float(ma100.iloc[i]) and float(r2.iloc[i]) < rsi_th:
                held = True
            elif held and px > float(ma5.iloc[i]):
                held = False
            v_raw = float(vol.iloc[i])
            v = 0.20 if (pd.isna(v_raw) or v_raw < 0.20) else v_raw
            w_last = min(2.0 / n_syms, (vol_target / n_syms) / v) if held else 0.0
        weights[s] = w_last
        rsi_snap[s.split("-")[0]] = round(float(r2.iloc[-1]), 1)
        if w_last > 0:
            active.append(s.split("-")[0])

    return weights, {"mr_active": active, "rsi2": rsi_snap}


def compute_blend_target_weights(
    h4: pd.DataFrame,
    l4: pd.DataFrame,
    c4: pd.DataFrame,
    c1: pd.DataFrame,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Combines 65% Multi-Speed 4H Donchian Trend + 35% 1D Core-3 RSI(2) Pullback."""
    w_trend, diag_tr = compute_4h_donchian_weights(h4, l4, c4, c1["BTC-USDT-SWAP"])
    w_mr, diag_mr = compute_1d_mr_weights(c1)

    combined = {s: 0.0 for s in SYMS_8}
    for s in SYMS_8:
        combined[s] = TREND_BLEND_WEIGHT * w_trend.get(s, 0.0) + MR_BLEND_WEIGHT * w_mr.get(s, 0.0)

    gross = sum(combined.values())
    if gross > MAX_GROSS_WEIGHT and gross > 0:
        scale = MAX_GROSS_WEIGHT / gross
        combined = {s: v * scale for s, v in combined.items()}

    diag = {
        "4h_ts": str(c4.index[-1]),
        "1d_ts": str(c1.index[-1]),
        "gross_leverage": round(sum(combined.values()), 3),
        **diag_tr,
        **diag_mr,
    }
    return combined, diag


def plan_delta_orders(
    equity_usdt: float,
    prices: Dict[str, float],
    current_contracts: Dict[str, float],
    target_weights: Dict[str, float],
    ct_vals: Dict[str, float],
    min_szs: Dict[str, float],
    min_trade_usd: float = 20.0,
    drift_band: float = 0.12,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Plans (sells_or_closes, buys) to move current positions toward target_weights.
    Returns SELL/CLOSE orders first so margin is freed before BUY orders.
    """
    if equity_usdt <= 0:
        return [], []

    sells: List[Dict[str, Any]] = []
    buys: List[Dict[str, Any]] = []

    all_syms = sorted(set(current_contracts.keys()) | set(target_weights.keys()))
    for sym in all_syms:
        curr_ct = float(current_contracts.get(sym, 0.0))
        px = float(prices.get(sym, 0.0))
        ct_val = float(ct_vals.get(sym, 0.0))
        min_sz = float(min_szs.get(sym, 0.01))
        w = float(target_weights.get(sym, 0.0))

        # Any legacy short position (curr_ct < 0) in a long-only portfolio must be closed
        if curr_ct < -1e-9:
            sells.append({"symbol": sym, "action": "CLOSE_LEGACY_SHORT", "side": "buy", "delta_usd": abs(curr_ct) * ct_val * max(px, 0.0)})
            continue

        if px <= 0 or ct_val <= 0:
            continue

        curr_usd = curr_ct * ct_val * px
        target_usd = equity_usdt * w
        min_ct_usd = min_sz * ct_val * px

        if w <= 1e-6:
            if curr_ct > 1e-9:
                sells.append({"symbol": sym, "action": "CLOSE_LONG", "side": "sell", "delta_usd": curr_usd})
            continue

        if target_usd < min_ct_usd * 1.05:
            if curr_ct > 1e-9:
                sells.append({"symbol": sym, "action": "CLOSE_BELOW_MIN", "side": "sell", "delta_usd": curr_usd})
            continue

        delta_usd = target_usd - curr_usd
        threshold = max(min_trade_usd, min_ct_usd, target_usd * drift_band)

        if curr_ct <= 1e-9:
            # Fresh entry: submit as long as target >= min_ct_usd
            buys.append({"symbol": sym, "action": "BUY_DELTA", "side": "buy", "delta_usd": target_usd})
        elif delta_usd < -threshold:
            sells.append({"symbol": sym, "action": "REDUCE_LONG", "side": "sell", "delta_usd": abs(delta_usd)})
        elif delta_usd > threshold:
            buys.append({"symbol": sym, "action": "BUY_DELTA", "side": "buy", "delta_usd": delta_usd})

    return sells, buys
