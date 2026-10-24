# -*- coding: utf-8 -*-
"""
hard_risk_veto.py — Dynamic Capital & Hard Risk Veto Engine for Trading Order Execution
- Operates immediately after Jev AI grants trade approval.
- Enforces non-negotiable program-level guardrails:
    1. Maximum parallel active positions check (VETO_MAX_POSITIONS_REACHED)
    2. Slippage tolerance limit check (VETO_SLIPPAGE_EXCEEDED)
    3. Dynamic Capital Allocation & Max Cap check (VETO_MAX_CAP_EXCEEDED)
       - Dynamically scales available margin with leverage based on remaining symbol slots
       - Enforces single-symbol safety ceiling (max_single_exposure_ratio)
    4. Flash crash Stop Loss & Liquidation distance safety check
- If ANY condition is violated, the order is strictly VETOED (rejected).
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger("HardRiskVetoEngine")


@dataclass
class VetoResult:
    is_vetoed: bool
    reason: str
    metrics: Dict[str, Any]


class HardRiskVetoEngine:
    """
    Hard-coded Risk Veto Engine with Dynamic Capital Allocation.
    Jev retains supreme decision authority, but this engine acts as the ultimate
    programmatic shield immediately before order execution to protect capital
    while maximizing capital efficiency.
    """

    def __init__(
        self,
        max_slippage_bps: float = 15.0,              # Maximum allowed slippage (15 bps = 0.15%)
        default_max_parallel_symbols: int = 4,       # Default maximum simultaneous positions (4 symbols)
        default_leverage: int = 20,                  # Default leverage (20x)
        max_single_exposure_ratio: float = 0.40,     # Max 40% margin of total equity for any single symbol
        available_margin_utilization: float = 0.90,  # Utilize up to 90% of available free margin
        max_order_notional_usdt: float = 20000.0,    # Absolute maximum ceiling for single order notional
        min_liquidation_buffer_pct: float = 0.02     # Min 2.0% gap between Stop Loss and Liquidation price
    ):
        self.max_slippage_bps = max_slippage_bps
        self.default_max_parallel_symbols = default_max_parallel_symbols
        self.default_leverage = default_leverage
        self.max_single_exposure_ratio = max_single_exposure_ratio
        self.available_margin_utilization = available_margin_utilization
        self.max_order_notional_usdt = max_order_notional_usdt
        self.min_liquidation_buffer_pct = min_liquidation_buffer_pct

    def evaluate_veto(
        self,
        symbol: str,
        side: str,                                      # "BUY" (Long) or "SELL" (Short)
        order_price: float,                             # Proposed execution price (limit or best quote)
        market_mid_price: float,                        # Current orderbook mid or last ticker price
        order_qty: float,                               # Order quantity (contracts or base coin)
        contract_size: float = 1.0,                     # OKX contract multiplier (e.g., 0.01 for BTC)
        account_equity: float = 0.0,                    # Total account equity in USDT
        stop_loss_price: Optional[float] = None,         # Absolute stop loss price
        leverage: int = 20,                             # Trading leverage
        maintenance_margin_rate: float = 0.005,         # MMR (0.5% default)
        max_parallel_symbols: Optional[int] = None,     # Max simultaneous symbols allowed (default: 4)
        current_open_positions: int = 0,                # Number of currently active open positions
        available_margin_usdt: float = 0.0,             # Real-time usable free margin in USDT
    ) -> VetoResult:
        """
        Runs all hard risk rules in sequence:
        1. Max Active Positions check
        2. Slippage tolerance check
        3. Dynamic Capital Allocation & Max Cap check
        4. Flash Crash Stop Loss & Liquidation buffer check
        """
        side_upper = side.upper()
        notional_value = order_qty * contract_size * order_price
        max_symbols = max_parallel_symbols if max_parallel_symbols is not None else self.default_max_parallel_symbols
        active_lev = leverage if leverage > 0 else self.default_leverage

        # -------------------------------------------------------------
        # 1. Max Active Positions Check (동시 진입 종목 수 초과 차단)
        # -------------------------------------------------------------
        if current_open_positions >= max_symbols:
            reason = (
                f"VETO_MAX_POSITIONS_REACHED: Active positions ({current_open_positions}) >= "
                f"max allowed ({max_symbols})"
            )
            logger.error(f"🚨 [HARD RISK VETO] {symbol} {side_upper} 기각! {reason}")
            return VetoResult(
                is_vetoed=True,
                reason=reason,
                metrics={
                    "rule": "MAX_POSITIONS",
                    "current_open_positions": current_open_positions,
                    "max_parallel_symbols": max_symbols,
                },
            )

        # -------------------------------------------------------------
        # 2. Slippage Check (슬리피지 허용 한도 초과 검증)
        # -------------------------------------------------------------
        slippage_bps = 0.0
        if market_mid_price > 0 and order_price > 0:
            if side_upper in ("BUY", "LONG"):
                slippage_ratio = (order_price - market_mid_price) / market_mid_price
            else:
                slippage_ratio = (market_mid_price - order_price) / market_mid_price

            slippage_bps = slippage_ratio * 10000.0

            if slippage_bps > self.max_slippage_bps:
                reason = (
                    f"VETO_SLIPPAGE_EXCEEDED: Slippage {slippage_bps:.2f} bps "
                    f"exceeds hard limit {self.max_slippage_bps:.2f} bps "
                    f"(Order: {order_price}, Mid: {market_mid_price})"
                )
                logger.error(f"🚨 [HARD RISK VETO] {symbol} {side_upper} 기각! {reason}")
                return VetoResult(
                    is_vetoed=True,
                    reason=reason,
                    metrics={
                        "rule": "SLIPPAGE",
                        "slippage_bps": slippage_bps,
                        "limit_bps": self.max_slippage_bps,
                    },
                )

        # -------------------------------------------------------------
        # 3. Dynamic Capital Allocation & Max Cap Check
        #    (실시간 잔여 증거금 및 동시 진입 종목 수 연동형 가동률 극대화)
        # -------------------------------------------------------------
        remaining_slots = max(1, max_symbols - current_open_positions)
        
        # 3-A. Dynamic Margin Pool:
        # 잔여 증거금(available_margin_usdt)의 90%를 활용 가능 풀로 설정
        if available_margin_usdt > 0:
            usable_margin_pool = available_margin_usdt * self.available_margin_utilization
        elif account_equity > 0:
            usable_margin_pool = account_equity * self.available_margin_utilization
        else:
            usable_margin_pool = 0.0

        if usable_margin_pool > 0:
            # 남은 슬롯 수에 맞춰 가용 마진 분할 (남은 슬롯이 1개면 잔여 증거금의 90% 전체 활용)
            slot_margin = usable_margin_pool / remaining_slots
            dynamic_notional_cap = slot_margin * active_lev

            # 3-B. 단일 종목 최대 한도 마지노선 (Single-Symbol Safety Ceiling):
            # 특정 단일 종목이 계좌 전체 리스크를 흔들지 않도록 총 자산의 40% 마진(또는 Notional) 마지노선 통제
            if account_equity > 0:
                max_single_margin = account_equity * self.max_single_exposure_ratio
                max_single_notional_ceiling = max_single_margin * active_lev
            else:
                max_single_notional_ceiling = self.max_order_notional_usdt

            effective_cap = min(dynamic_notional_cap, max_single_notional_ceiling)
            if self.max_order_notional_usdt > 0:
                effective_cap = min(effective_cap, self.max_order_notional_usdt)

            if notional_value > effective_cap:
                reason = (
                    f"VETO_MAX_CAP_EXCEEDED: Order Notional ${notional_value:.2f} USDT "
                    f"exceeds dynamic cap ${effective_cap:.2f} USDT "
                    f"(SlotMargin: ${slot_margin:.2f}, Lev: {active_lev}x, "
                    f"RemainingSlots: {remaining_slots}/{max_symbols}, "
                    f"SingleCeiling: ${max_single_notional_ceiling:.2f})"
                )
                logger.error(f"🚨 [HARD RISK VETO] {symbol} {side_upper} 기각! {reason}")
                return VetoResult(
                    is_vetoed=True,
                    reason=reason,
                    metrics={
                        "rule": "MAX_CAP",
                        "notional": notional_value,
                        "effective_cap": effective_cap,
                        "slot_margin": slot_margin,
                        "remaining_slots": remaining_slots,
                        "single_ceiling": max_single_notional_ceiling,
                        "available_margin": available_margin_usdt,
                        "account_equity": account_equity,
                    },
                )

        # -------------------------------------------------------------
        # 4. Flash Crash Stop Loss & Liquidation Safety Check
        #    (급격한 플래시 크래시 대비 하드코딩 손절선 및 청산 방지 룰)
        # -------------------------------------------------------------
        if stop_loss_price is None or stop_loss_price <= 0:
            reason = "VETO_STOP_LOSS_MISSING: Hard stop-loss must be defined for flash crash prevention"
            logger.error(f"🚨 [HARD RISK VETO] {symbol} {side_upper} 기각! {reason}")
            return VetoResult(
                is_vetoed=True,
                reason=reason,
                metrics={"rule": "STOP_LOSS_MISSING"},
            )

        # Estimated Liquidation Price Calculation
        # Long: LiqPrice = Entry * (1 - 1/Lev * (1 - MMR))
        # Short: LiqPrice = Entry * (1 + 1/Lev * (1 - MMR))
        margin_req = 1.0 / max(1, active_lev)
        if side_upper in ("BUY", "LONG"):
            estimated_liq = order_price * (1.0 - margin_req * (1.0 - maintenance_margin_rate))
            if stop_loss_price <= estimated_liq:
                reason = (
                    f"VETO_DANGEROUS_STOP_LOSS: Long SL ({stop_loss_price}) is at or below "
                    f"estimated liquidation price ({estimated_liq:.4f})"
                )
                logger.error(f"🚨 [HARD RISK VETO] {symbol} {side_upper} 기각! {reason}")
                return VetoResult(
                    is_vetoed=True,
                    reason=reason,
                    metrics={"rule": "STOP_LOSS_BELOW_LIQ", "liq": estimated_liq, "sl": stop_loss_price},
                )

            buffer_pct = (stop_loss_price - estimated_liq) / order_price
            if buffer_pct < self.min_liquidation_buffer_pct:
                reason = (
                    f"VETO_INSUFFICIENT_LIQ_BUFFER: Long buffer {buffer_pct*100:.2f}% "
                    f"< required minimum {self.min_liquidation_buffer_pct*100:.2f}% "
                    f"(SL: {stop_loss_price}, Liq: {estimated_liq:.4f})"
                )
                logger.error(f"🚨 [HARD RISK VETO] {symbol} {side_upper} 기각! {reason}")
                return VetoResult(
                    is_vetoed=True,
                    reason=reason,
                    metrics={"rule": "LIQ_BUFFER", "buffer_pct": buffer_pct, "min_buffer": self.min_liquidation_buffer_pct},
                )

        else: # SELL / SHORT
            estimated_liq = order_price * (1.0 + margin_req * (1.0 - maintenance_margin_rate))
            if stop_loss_price >= estimated_liq:
                reason = (
                    f"VETO_DANGEROUS_STOP_LOSS: Short SL ({stop_loss_price}) is at or below "
                    f"estimated liquidation price ({estimated_liq:.4f})"
                )
                logger.error(f"🚨 [HARD RISK VETO] {symbol} {side_upper} 기각! {reason}")
                return VetoResult(
                    is_vetoed=True,
                    reason=reason,
                    metrics={"rule": "STOP_LOSS_ABOVE_LIQ", "liq": estimated_liq, "sl": stop_loss_price},
                )

            buffer_pct = (estimated_liq - stop_loss_price) / order_price
            if buffer_pct < self.min_liquidation_buffer_pct:
                reason = (
                    f"VETO_INSUFFICIENT_LIQ_BUFFER: Short buffer {buffer_pct*100:.2f}% "
                    f"< required minimum {self.min_liquidation_buffer_pct*100:.2f}% "
                    f"(SL: {stop_loss_price}, Liq: {estimated_liq:.4f})"
                )
                logger.error(f"🚨 [HARD RISK VETO] {symbol} {side_upper} 기각! {reason}")
                return VetoResult(
                    is_vetoed=True,
                    reason=reason,
                    metrics={"rule": "LIQ_BUFFER", "buffer_pct": buffer_pct, "min_buffer": self.min_liquidation_buffer_pct},
                )

        # All hard rules passed
        return VetoResult(
            is_vetoed=False,
            reason="PASSED_ALL_HARD_RULES",
            metrics={
                "notional": notional_value,
                "slippage_bps": slippage_bps,
                "liq_price": estimated_liq,
                "stop_loss_price": stop_loss_price,
                "remaining_slots": remaining_slots,
                "effective_cap": locals().get("effective_cap", 0.0),
            },
        )
