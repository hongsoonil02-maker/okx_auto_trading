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
from .hard_risk_veto import HardRiskVetoEngine, VetoResult

logger = logging.getLogger("JevSignalFilter")


@dataclass
class FilterResult:
    approved: bool
    order_type: str  # "MARKET" or "POST_ONLY"
    target_price: Optional[float]
    jev_score: float  # up_in_10
    action: str  # 'buy', 'sell', 'neutral', 'HOLD'
    confidence: float
    reason: str
    latency_ms: float
    is_simulation: bool = False
    is_fallback: bool = False
    veto_reason: Optional[str] = None


class JevSignalFilter:
    def __init__(
        self,
        lob_feed: Optional[OKXLOBFeed] = None,
        jev_client: Optional[TypesafeJevClient] = None,
        veto_engine: Optional[HardRiskVetoEngine] = None,
    ):
        self.lob_feed = lob_feed or OKXLOBFeed()
        self.jev_client = jev_client or TypesafeJevClient()
        self.veto_engine = veto_engine or HardRiskVetoEngine()
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
        order_qty: float = 0.0,
        contract_size: float = 1.0,
        account_equity: float = 0.0,
        stop_loss_price: Optional[float] = None,
        leverage: int = 20,
        max_parallel_symbols: Optional[int] = None,
        current_open_positions: int = 0,
        available_margin_usdt: float = 0.0,
    ) -> FilterResult:
        """
        [Jev AI Sub-Second Decision Filter + Hard Risk Veto Pipeline]
        1. Jev AI (Model: jev-1.15-prod) 진단 및 최우선 결정권한 행사
           - Jev가 BUY/SELL을 승인하지 않으면 즉시 차단(주문 절차 시작 불가)
        2. Hard Risk Veto Engine: Jev 승인 직후 송출 직전 하드코딩 룰 검증
           - 슬리피지 허용 한도 초과 체크
           - 계좌 잔고 대비 1회 최대 금액 한도(Max Cap) 체크
           - 플래시 크래시 대비 손절선(Stop Loss) 및 청산 방지 룰 체크
        3. Exception & Fallback: 타임아웃/오류 발생 시 즉시 'HOLD' 안전 대기 전환
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

        try:
            # 2. Retrieve real-time LOB snapshot (프록시 없는 초저지연 로컬 LOB 버퍼)
            lob: Optional[LOBEntry] = self.lob_feed.get_lob(symbol)
            if lob is None or lob.is_stale:
                if self.fallback_to_baseline:
                    logger.warning(f"⚠️ [Jev] {symbol} LOB 부재 또는 Stale -> 안전 HOLD 유지")
                    return FilterResult(
                        approved=False,
                        order_type="MARKET",
                        target_price=None,
                        jev_score=0.50,
                        action="HOLD",
                        confidence=0.0,
                        reason="LOB_FEED_STALE_OR_MISSING_SAFE_HOLD",
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

            # 3. [최우선 결정권한] Jev 고정 모델(jev-1.15-prod) 서브세컨드 추론 요청
            state_text = lob.format_for_jev(recent_note=context_note)
            decision: JevDecision = await self.jev_client.predict_orderbook(
                state_text=state_text,
                lob_imbalance=lob.imbalance,
            )

            # [Rule 3: Fallback & Timeout Safe Hold]
            # 외부 API 503, 타임아웃 등으로 폴백 결정이 내려진 경우 자산 보호를 위해 즉각 HOLD
            is_error_fallback = (decision.action == "HOLD" or decision.error is not None)
            if is_error_fallback and not self.is_simulation_mode:
                logger.info(f"🛡️ [Jev Safe Fallback] {symbol} 비상 HOLD 발동 — 에이전트 불필요 루프 방지 (Error: {decision.error})")
                return FilterResult(
                    approved=False,
                    order_type="MARKET",
                    target_price=None,
                    jev_score=decision.up_in_10,
                    action="HOLD",
                    confidence=0.0,
                    reason=f"SAFE_HOLD_FALLBACK_{decision.error}",
                    latency_ms=decision.latency_ms,
                    is_simulation=self.is_simulation_mode,
                    is_fallback=True,
                )

            threshold = self.confidence_threshold
            side_upper = proposed_side.upper()
            approved = False
            reason = ""
            target_price: Optional[float] = None
            order_type = "MARKET"

            # 4. Directional score evaluation (Jev Supreme Decision Gate)
            if side_upper in ("BUY", "LONG"):
                score = decision.up_in_10
                if score >= threshold:
                    approved = True
                    reason = f"UP_IN_10_{score:.2f}_GTE_{threshold:.2f}"
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
                approved = True
                reason = "EXIT_SIGNAL_BYPASS"
                score = decision.up_in_10

            # 5. [Rule 2: Hard Risk Veto Rule]
            # Jev가 BUY/SELL을 승인했더라도, 주문 송출 바로 직전에 3대 하드 룰 기반 Veto 검증
            veto_reason = None
            if approved and order_qty > 0 and not self.is_simulation_mode:
                exec_price = target_price or (lob.best_ask if side_upper in ("BUY", "LONG") else lob.best_bid)
                mid_price = getattr(lob, 'mid_price', (lob.best_bid + lob.best_ask) / 2.0 if (lob.best_bid and lob.best_ask) else lob.micro_price)
                
                veto_res: VetoResult = self.veto_engine.evaluate_veto(
                    symbol=symbol,
                    side=side_upper,
                    order_price=exec_price,
                    market_mid_price=mid_price,
                    order_qty=order_qty,
                    contract_size=contract_size,
                    account_equity=account_equity,
                    stop_loss_price=stop_loss_price,
                    leverage=leverage,
                    max_parallel_symbols=max_parallel_symbols,
                    current_open_positions=current_open_positions,
                    available_margin_usdt=available_margin_usdt,
                )
                if veto_res.is_vetoed:
                    approved = False
                    veto_reason = veto_res.reason
                    reason = f"HARD_RISK_VETO: {veto_res.reason}"
                    logger.error(f"🛑 [VETO ACTIVATED] Jev 승인 취소 및 주문 강제 차단: {symbol} {side_upper} -> {veto_reason}")

            # 6. Handle simulation mode
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
                veto_reason=veto_reason,
            )

        except Exception as e_filter:
            # Gemini 3.8 Flash 에이전트가 예외 루프에 갇히지 않도록 완벽 봉쇄
            logger.error(f"❌ [JevSignalFilter Error] 처리 중 예외 발생 ({e_filter}) — 안전 대기(HOLD) 반환")
            return FilterResult(
                approved=False,
                order_type="MARKET",
                target_price=None,
                jev_score=0.50,
                action="HOLD",
                confidence=0.0,
                reason=f"EXCEPTION_SAFE_HOLD: {str(e_filter)}",
                latency_ms=0.0,
                is_simulation=self.is_simulation_mode,
                is_fallback=True,
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
