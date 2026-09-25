# -*- coding: utf-8 -*-
"""
jev_signal_filter.py — Jev AI Sub-Second Decision Filter & Order Pricing
- Gates trading signals based on Jev probability scores (e.g. up_in_10 >= 0.65)
- Computes Maker Post-Only limit order prices using QUOTE_INSIDE_TICKS
- Supports Simulation Mode (Paper-trading logging without executing real balance)
- Implements Graceful Degradation / Fallback on timeout or API unavailability
"""
import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional
from .okx_lob_feed import OKXLOBFeed, LOBEntry
from .typesafe_client import TypesafeJevClient, JevDecision

logger = logging.getLogger("JevSignalFilter")


@dataclass
class FilterResult:
    approved: bool
    order_type: str  # "MARKET" or "POST_ONLY"
    target_price: Optional[float]
    jev_score: float  # up_in_10
    action: str  # 'buy', 'sell', 'neutral'
    confidence: float
    reason: str
    latency_ms: float
    is_simulation: bool = False
    is_fallback: bool = False


class JevSignalFilter:
    def __init__(
        self,
        lob_feed: Optional[OKXLOBFeed] = None,
        jev_client: Optional[TypesafeJevClient] = None,
    ):
        self.lob_feed = lob_feed or OKXLOBFeed()
        self.jev_client = jev_client or TypesafeJevClient()
        self._is_initialized = False

    @property
    def is_enabled(self) -> bool:
        return os.getenv("USE_JEV_PREDICTION", "false").lower() == "true"

    @property
    def is_simulation_mode(self) -> bool:
        return os.getenv("JEV_SIMULATION_MODE", "true").lower() == "true"

    @property
    def confidence_threshold(self) -> float:
        return float(os.getenv("JEV_CONFIDENCE_THRESHOLD", "0.65"))

    @property
    def quote_inside_ticks(self) -> int:
        return int(os.getenv("QUOTE_INSIDE_TICKS", "1"))

    @property
    def fallback_to_baseline(self) -> bool:
        return os.getenv("JEV_FALLBACK_TO_BASELINE", "true").lower() == "true"

    async def initialize(self):
        """Starts background LOB feed if enabled."""
        if not self._is_initialized:
            if self.is_enabled:
                await self.lob_feed.start()
            self._is_initialized = True

    async def close(self):
        if self._is_initialized:
            await self.lob_feed.stop()
            await self.jev_client.close()
            self._is_initialized = False

    def register_symbols(self, symbols: list):
        """Registers symbols to keep in real-time LOB buffer."""
        self.lob_feed.subscribe(symbols)

    async def evaluate_signal(
        self,
        symbol: str,
        proposed_side: str,  # "BUY" (Long) or "SELL" (Short)
        tick_size: float = 0.1,
        context_note: str = "",
    ) -> FilterResult:
        """
        Evaluates proposed trading signal using sub-second Jev orderbook prediction.
        """
        # 1. Feature flag check
        if not self.is_enabled:
            return FilterResult(
                approved=True,
                order_type="MARKET",
                target_price=None,
                jev_score=0.50,
                action="bypass",
                confidence=1.0,
                reason="JEV_FEATURE_DISABLED",
                latency_ms=0.0,
                is_simulation=False,
                is_fallback=True,
            )

        # 2. Retrieve real-time LOB snapshot
        lob: Optional[LOBEntry] = self.lob_feed.get_lob(symbol)
        if lob is None or lob.is_stale:
            if self.fallback_to_baseline:
                logger.warning(f"⚠️ [Jev] {symbol} LOB 부재 또는 Stale -> 기준 전략으로 안전 폴백")
                return FilterResult(
                    approved=True,
                    order_type="MARKET",
                    target_price=None,
                    jev_score=0.50,
                    action="fallback",
                    confidence=0.0,
                    reason="LOB_FEED_STALE_OR_MISSING_FALLBACK",
                    latency_ms=0.0,
                    is_simulation=self.is_simulation_mode,
                    is_fallback=True,
                )
            else:
                logger.warning(f"🚫 [Jev] {symbol} LOB 미가용으로 진입 거부")
                return FilterResult(
                    approved=False,
                    order_type="MARKET",
                    target_price=None,
                    jev_score=0.0,
                    action="reject",
                    confidence=0.0,
                    reason="LOB_FEED_NOT_AVAILABLE",
                    latency_ms=0.0,
                    is_simulation=self.is_simulation_mode,
                )

        # 3. Request inference from Jev model
        state_text = lob.format_for_jev(recent_note=context_note)
        decision: JevDecision = await self.jev_client.predict_orderbook(
            state_text=state_text,
            lob_imbalance=lob.imbalance,
        )

        threshold = self.confidence_threshold
        side_upper = proposed_side.upper()
        approved = False
        reason = ""
        target_price: Optional[float] = None
        order_type = "MARKET"

        # 4. Directional score evaluation
        if side_upper in ("BUY", "LONG"):
            # For BUY/LONG: check up_in_10 probability
            score = decision.up_in_10
            if score >= threshold:
                approved = True
                reason = f"UP_IN_10_{score:.2f}_GTE_{threshold:.2f}"
                # Compute Maker Post-Only price
                target_price = self._calc_post_only_price(
                    is_buy=True,
                    best_bid=lob.best_bid,
                    best_ask=lob.best_ask,
                    tick_size=tick_size,
                )
                order_type = "POST_ONLY" if target_price is not None else "MARKET"
            else:
                approved = False
                reason = f"LOW_UP_IN_10_{score:.2f}_LT_{threshold:.2f}"

        elif side_upper in ("SELL", "SHORT"):
            # For SELL/SHORT: check downward probability (1 - up_in_10)
            down_score = 1.0 - decision.up_in_10
            score = down_score
            if down_score >= threshold:
                approved = True
                reason = f"DOWN_IN_10_{down_score:.2f}_GTE_{threshold:.2f}"
                target_price = self._calc_post_only_price(
                    is_buy=False,
                    best_bid=lob.best_bid,
                    best_ask=lob.best_ask,
                    tick_size=tick_size,
                )
                order_type = "POST_ONLY" if target_price is not None else "MARKET"
            else:
                approved = False
                reason = f"LOW_DOWN_IN_10_{down_score:.2f}_LT_{threshold:.2f}"
        else:
            # Neutral / Close signals bypass gate
            approved = True
            reason = "EXIT_SIGNAL_BYPASS"
            score = decision.up_in_10

        # 5. Handle simulation mode
        sim_mode = self.is_simulation_mode
        if sim_mode:
            logger.info(
                f"📝 [JEV SIMULATION] {symbol} {side_upper} | "
                f"Approved={approved} ({reason}) | Score={score:.3f} | "
                f"Action={decision.action} ({decision.action_confidence:.2f}) | "
                f"Type={order_type} @ {target_price} | Latency={decision.latency_ms:.1f}ms"
            )

        return FilterResult(
            approved=approved,
            order_type=order_type,
            target_price=target_price,
            jev_score=score,
            action=decision.action,
            confidence=decision.action_confidence,
            reason=reason,
            latency_ms=decision.latency_ms,
            is_simulation=sim_mode,
            is_fallback=decision.is_fallback,
        )

    def _calc_post_only_price(
        self,
        is_buy: bool,
        best_bid: float,
        best_ask: float,
        tick_size: float,
    ) -> Optional[float]:
        """
        Calculates Post-Only limit order price inside the spread.
        Guarantees price never crosses opposite side to prevent taker execution.
        """
        ticks = self.quote_inside_ticks
        if is_buy:
            # Quote inside best bid towards best ask, but strictly below best ask
            desired_bid = best_bid + (ticks * tick_size)
            if desired_bid >= best_ask:
                # Constrain to best bid if spread is tight (1 tick)
                desired_bid = best_bid
            return round(desired_bid, 8)
        else:
            # Quote inside best ask towards best bid, but strictly above best bid
            desired_ask = best_ask - (ticks * tick_size)
            if desired_ask <= best_bid:
                # Constrain to best ask if spread is tight (1 tick)
                desired_ask = best_ask
            return round(desired_ask, 8)

    async def evaluate_exit(
        self,
        symbol: str,
        current_side: str,  # "LONG" or "SHORT"
        entry_price: float,
        current_price: float,
        leverage: int = 20,
    ) -> tuple:
        """
        [Jev AI Autonomous Scalping Exit]
        실시간 호가창(LOB) 수급을 마이크로초 단위로 평가하여 조기 청산 결정:
        1. Adverse Pressure Micro-Exit (호가 수급 역전 즉시 탈출: 롱인데 매도벽 쏠림, 숏인데 매수벽 쏠림)
        2. Scalp Profit-Lock Exit (20x 목표 ROE 달성 후 모멘텀 둔화 시 칼익절)
        """
        if not self.is_enabled:
            return False, "JEV_DISABLED"

        lob: Optional[LOBEntry] = self.lob_feed.get_lob(symbol)
        if lob is None or lob.is_stale:
            return False, "LOB_NOT_AVAILABLE"

        side_u = current_side.upper()
        pnl_pct = ((current_price - entry_price) / entry_price) * leverage if side_u in ("LONG", "BUY") else ((entry_price - current_price) / entry_price) * leverage

        # 1. 초고속 LOB Imbalance 평가 (네트워크 지연 0ms)
        # 극단적인 역방향 호가벽 쏠림 감지 시 즉시 탈출
        if side_u in ("LONG", "BUY") and lob.imbalance <= -0.45:
            return True, f"LOB_ADVERSE_IMBALANCE_SELL (imb: {lob.imbalance:+.2f})"
        elif side_u in ("SHORT", "SELL") and lob.imbalance >= +0.45:
            return True, f"LOB_ADVERSE_IMBALANCE_BUY (imb: {lob.imbalance:+.2f})"

        # 2. 20배 초단타 스캘핑 칼익절 (ROE +10% 이상 도달 후 수급 둔화 시 확정)
        if pnl_pct >= 0.10:
            if side_u in ("LONG", "BUY") and lob.imbalance <= 0.0:
                return True, f"SCALP_TP_REVERSAL (ROE: {pnl_pct*100:+.1f}%, imb: {lob.imbalance:+.2f})"
            elif side_u in ("SHORT", "SELL") and lob.imbalance >= 0.0:
                return True, f"SCALP_TP_REVERSAL (ROE: {pnl_pct*100:+.1f}%, imb: {lob.imbalance:+.2f})"

        # 3. Jev 서브세컨드 AI 추론 평가 (< 500ms)
        try:
            state_text = lob.format_for_jev(recent_note=f"Holding={side_u}, Lev={leverage}x, PnL={pnl_pct*100:+.1f}%")
            decision: JevDecision = await self.jev_client.predict_orderbook(
                state_text=state_text,
                lob_imbalance=lob.imbalance,
            )

            if side_u in ("LONG", "BUY"):
                # 롱 보유 중인데 Jev가 하방 예측 (up_in_10 <= 0.35 즉 down_in_10 >= 0.65)
                if decision.up_in_10 <= 0.35:
                    return True, f"JEV_ADVERSE_PRESSURE (prob_up: {decision.up_in_10:.2f} <= 0.35)"
                # 롱 이익 구간(ROE >= 8%)에서 모멘텀 둔화 시 익절
                if pnl_pct >= 0.08 and decision.action in ("sell", "neutral") and decision.up_in_10 < 0.50:
                    return True, f"JEV_SCALP_TP (ROE: {pnl_pct*100:+.1f}%, action: {decision.action})"

            elif side_u in ("SHORT", "SELL"):
                # 숏 보유 중인데 Jev가 상방 예측 (up_in_10 >= 0.65)
                if decision.up_in_10 >= 0.65:
                    return True, f"JEV_ADVERSE_PRESSURE (prob_up: {decision.up_in_10:.2f} >= 0.65)"
                # 숏 이익 구간(ROE >= 8%)에서 모멘텀 둔화 시 익절
                if pnl_pct >= 0.08 and decision.action in ("buy", "neutral") and decision.up_in_10 > 0.50:
                    return True, f"JEV_SCALP_TP (ROE: {pnl_pct*100:+.1f}%, action: {decision.action})"

        except Exception as ex_eval:
            logger.debug(f"Jev exit evaluation error ({symbol}): {ex_eval}")

        return False, "HOLD"
