# -*- coding: utf-8 -*-
"""Pure risk-sizing and loss-limit rules shared by the live trader and tests."""

from typing import Optional, Sequence

import pandas as pd


def daily_trend_ok(
    closes: Sequence[float],
    is_long: bool,
    ema_span: int = 480,
    slope_bars: int = 72,
    min_slope_pct: float = 0.0,
) -> Optional[bool]:
    """BTC 20-day EMA direction filter on confirmed 1H closes (no lookahead)."""
    if len(closes) < max(ema_span + 1, slope_bars + 2):
        return None
    s = pd.Series([float(x) for x in closes], dtype="float64")
    if not s.notna().all() or (s <= 0).any():
        return None
    ema = s.ewm(span=int(ema_span), adjust=False).mean()
    last = float(s.iloc[-1])
    ema_last = float(ema.iloc[-1])
    ema_prev = float(ema.iloc[-1 - int(slope_bars)])
    if ema_prev <= 0:
        return None
    slope_pct = (ema_last / ema_prev - 1.0) * 100.0
    if is_long:
        return last > ema_last and slope_pct >= float(min_slope_pct)
    return last < ema_last and slope_pct <= -float(min_slope_pct)


def price_risk_buffer(sl_pct: float, fee_pct: float, slippage_pct: float) -> float:
    return max(0.0001, float(sl_pct) + float(fee_pct) + float(slippage_pct))


def projected_risk_usdt(
    notional_usdt: float,
    sl_pct: float,
    fee_pct: float,
    slippage_pct: float,
) -> float:
    return max(0.0, float(notional_usdt)) * price_risk_buffer(sl_pct, fee_pct, slippage_pct)


def calculate_target_notional(
    equity_usdt: float,
    risk_per_trade_pct: float,
    sl_pct: float,
    fee_pct: float,
    slippage_pct: float,
    current_open_risk_usdt: float,
    max_total_open_risk_pct: float,
    available_margin_usdt: float,
    leverage: float,
    max_margin_per_trade: float,
    margin_utilization: float = 0.90,
) -> float:
    equity = max(0.0, float(equity_usdt))
    lev = max(1.0, float(leverage))
    if equity <= 0 or risk_per_trade_pct <= 0 or max_total_open_risk_pct <= 0:
        return 0.0

    buffer = price_risk_buffer(sl_pct, fee_pct, slippage_pct)
    risk_budget = equity * float(risk_per_trade_pct)
    remaining_heat = max(0.0, equity * float(max_total_open_risk_pct) - max(0.0, float(current_open_risk_usdt)))
    allowed_risk = min(risk_budget, remaining_heat)
    if allowed_risk <= 0:
        return 0.0

    by_risk = allowed_risk / buffer
    by_margin = max(0.0, float(available_margin_usdt)) * float(margin_utilization) * lev
    by_trade_cap = max(0.0, float(max_margin_per_trade)) * lev
    return max(0.0, min(by_risk, by_margin, by_trade_cap))


def loss_limit_breached(
    realized_pnl_usdt: float,
    period_start_equity_usdt: float,
    loss_limit_pct: float,
) -> bool:
    start_eq = float(period_start_equity_usdt)
    limit_pct = float(loss_limit_pct)
    if start_eq <= 0 or limit_pct <= 0:
        return False
    return float(realized_pnl_usdt) <= -(start_eq * limit_pct)


def minimum_contract_is_acceptable(
    min_notional_usdt: float,
    equity_usdt: float,
    current_open_risk_usdt: float,
    max_total_open_risk_pct: float,
    max_single_risk_pct: float,
    sl_pct: float,
    fee_pct: float,
    slippage_pct: float,
    available_margin_usdt: float,
    leverage: float,
    margin_utilization: float = 0.90,
) -> bool:
    equity = max(0.0, float(equity_usdt))
    if equity <= 0:
        return False
    risk = projected_risk_usdt(min_notional_usdt, sl_pct, fee_pct, slippage_pct)
    remaining_heat = equity * float(max_total_open_risk_pct) - max(0.0, float(current_open_risk_usdt))
    required_margin = max(0.0, float(min_notional_usdt)) / max(1.0, float(leverage))
    return (
        risk <= equity * float(max_single_risk_pct)
        and risk <= remaining_heat
        and required_margin <= max(0.0, float(available_margin_usdt)) * float(margin_utilization)
    )
