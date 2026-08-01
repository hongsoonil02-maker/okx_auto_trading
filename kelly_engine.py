# -*- coding: utf-8 -*-
"""
kelly_engine.py — OKX 상위 1위 카피봇 동적 자금 관리 (Kelly Criterion 변형)
[Qwen 구현 표준]

각 Bot이 주문 수량 결정 시 호출하여 Kelly Fraction을 적용한 최적 포지션 크기 연산
"""
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class PortfolioMetrics:
    """포트폴리오 메트릭"""
    total_balance: float                # 총 자산 (USDT/KRW)
    current_position_value: float       # 현재 보유 포지션 가치
    peak_balance: float                 # 고점 자산
    win_rate: float                     # 승률 (0.0 ~ 1.0)
    avg_win: float                      # 평균 승리 수익률
    avg_loss: float                     # 평균 손실 수익률
    sharpe_ratio: Optional[float] = None  # 샤프 비율 (선택사항)


class KellyCriterionEngine:
    """
    Kelly Criterion 기반 동적 주문 크기 연산
    
    [Original Kelly Formula]
    f* = (b*p - q) / b
    
    where:
    - f* = Fraction of capital to bet
    - b = Ratio of win size to loss size (R:R ratio)
    - p = Probability of win (win_rate)
    - q = Probability of loss (1 - win_rate)
    
    [Practical Modification]
    - Kelly Fraction = 0.25 (Quarter Kelly) 권장 (파산 위험 감소)
    - Portfolio Pressure Adjustment (자산 드로우다운 시 감소)
    - Market Volatility Scaling (변동성 높으면 감소)
    """
    
    def __init__(self, kelly_fraction: float = 0.25):
        """
        Args:
            kelly_fraction: Kelly Criterion 계수 (0.0 ~ 1.0, 기본 0.25)
                - 0.25 (Quarter Kelly): 보수적, 낮은 변동성
                - 0.50 (Half Kelly): 중간, 권장값
                - 1.00 (Full Kelly): 공격적, 높은 변동성
        """
        if not 0 < kelly_fraction <= 1.0:
            raise ValueError(f"kelly_fraction must be in (0, 1], got {kelly_fraction}")
        self.kelly_fraction = kelly_fraction
    
    @staticmethod
    def calculate_kelly_optimal(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float = 0.25,
    ) -> float:
        """
        Kelly Criterion 최적 배팅 비율 계산
        
        Args:
            win_rate: 승률 (0.0 ~ 1.0)
            avg_win: 평균 승리 수익률 (0.02 = 2%)
            avg_loss: 평균 손실 수익률 (-0.01 = -1%)
            kelly_fraction: Kelly 계수 (기본 0.25)
        
        Returns:
            최적 배팅 비율 (0.0 ~ 1.0)
        """
        if not 0 <= win_rate <= 1.0:
            return 0.0  # Invalid win_rate
        
        if avg_loss == 0:
            return 0.0  # Cannot divide by zero
        
        # R:R Ratio (Risk-Reward)
        b = abs(avg_win) / abs(avg_loss)  # e.g., 0.02 / 0.01 = 2.0
        p = win_rate
        q = 1.0 - win_rate
        
        # Kelly Formula
        kelly = (b * p - q) / b
        
        # 음수 Kelly는 0으로 처리 (배팅하지 않음)
        kelly = max(0.0, kelly)
        
        # Fraction 적용
        optimal_fraction = kelly * kelly_fraction
        
        # 최대 포지션 크기 제한 (100% 초과 방지)
        optimal_fraction = min(1.0, optimal_fraction)
        
        return optimal_fraction
    
    @staticmethod
    def apply_drawdown_penalty(
        kelly_fraction: float,
        current_balance: float,
        peak_balance: float,
        max_drawdown_threshold: float = 0.15,  # 15% 드로우다운
    ) -> float:
        """
        자산 드로우다운 시 Kelly Fraction 감소
        
        Args:
            kelly_fraction: 원래 Kelly 분수
            current_balance: 현재 자산
            peak_balance: 최고 자산
            max_drawdown_threshold: 최대 드로우다운 임계값 (0.15 = 15%)
        
        Returns:
            조정된 Kelly Fraction (드로우다운이 크면 더 작음)
        """
        if peak_balance == 0:
            return kelly_fraction
        
        drawdown = (peak_balance - current_balance) / peak_balance
        
        if drawdown <= 0:
            return kelly_fraction  # 신고점 갱신, 원래대로
        
        if drawdown >= max_drawdown_threshold:
            return kelly_fraction * 0.5  # 50% 감소
        
        # Linear penalty: 0 ~ 50% 감소
        penalty = (drawdown / max_drawdown_threshold) * 0.5
        return kelly_fraction * (1.0 - penalty)
    
    @staticmethod
    def calculate_position_size(
        kelly_fraction: float,
        account_balance: float,
        entry_price: float,
        leverage: int = 1,
    ) -> float:
        """
        계산된 Kelly Fraction을 실제 주문 수량으로 변환
        
        Args:
            kelly_fraction: Kelly 비율 (0.0 ~ 1.0)
            account_balance: 계좌 잔고 (USDT/KRW)
            entry_price: 진입가
            leverage: 레버리지 배수 (OKX용, 기본 1)
        
        Returns:
            주문 수량 (코인 수/주식 수)
        """
        if entry_price <= 0:
            return 0.0
        
        # 투입 자본
        capital_to_deploy = account_balance * kelly_fraction * leverage
        
        # 주문 수량
        position_size = capital_to_deploy / entry_price
        
        return max(0.0, position_size)
    
    def compute_order_quantity(
        self,
        metrics: PortfolioMetrics,
        entry_price: float,
        stop_loss_price: float,
        leverage: int = 1,
        market_volatility: Optional[float] = None,
    ) -> Tuple[float, dict]:
        """
        포트폴리오 메트릭 기반 최적 주문 수량 계산
        
        Args:
            metrics: PortfolioMetrics 객체
            entry_price: 진입가
            stop_loss_price: 손실 종료 가격
            leverage: 레버리지
            market_volatility: 시장 변동성 (선택, VIX 등)
        
        Returns:
            (주문_수량, {계산_상세_정보})
        """
        # 1. Kelly Optimal Fraction 계산
        kelly_optimal = self.calculate_kelly_optimal(
            win_rate=metrics.win_rate,
            avg_win=metrics.avg_win,
            avg_loss=metrics.avg_loss,
            kelly_fraction=self.kelly_fraction,
        )
        
        # 2. 드로우다운 페널티 적용
        kelly_adjusted = self.apply_drawdown_penalty(
            kelly_optimal,
            metrics.current_position_value,
            metrics.peak_balance,
        )
        
        # 3. 시장 변동성 스케일링 (선택사항)
        if market_volatility is not None and market_volatility > 0:
            # 변동성이 높으면 Kelly 분수 감소
            vol_penalty = min(market_volatility / 100.0, 0.5)  # 최대 50% 감소
            kelly_adjusted *= (1.0 - vol_penalty)
        
        # 4. 실제 주문 수량 계산
        order_qty = self.calculate_position_size(
            kelly_adjusted,
            metrics.total_balance,
            entry_price,
            leverage,
        )
        
        # 상세 정보 반환
        details = {
            "kelly_optimal": kelly_optimal,
            "kelly_adjusted": kelly_adjusted,
            "kelly_fraction": self.kelly_fraction,
            "order_quantity": order_qty,
            "capital_deployed": metrics.total_balance * kelly_adjusted,
            "risk_per_trade": abs(entry_price - stop_loss_price) * order_qty,
            "risk_pct": (abs(entry_price - stop_loss_price) / entry_price * 100),
        }
        
        return order_qty, details


class OrderQuantityValidator:
    """거래소별 주문 수량 유효성 검증 (최소/최대 단위)"""
    
    # 거래소별 최소/최대 주문 단위 및 정밀도
    EXCHANGE_RULES = {
        "kr_stock": {
            "min_quantity": 1,              # 최소 1주
            "max_quantity": 1_000_000,      # 최대 100만 주
            "precision": 0,                 # 정수
            "min_order_krw": 0,             # 최소 주문 금액 없음
            "description": "KIS (한국투자증권)",
        },
        "nasdaq": {
            "min_quantity": 0.001,          # 소수 주문 가능 (프랙셔널 쉐어)
            "max_quantity": 100_000,        # 최대 10만 주
            "precision": 3,                 # 소수점 3자리
            "min_order_usd": 1.0,           # 최소 1 USD
            "description": "Interactive Brokers",
        },
        "okx_swap": {
            "min_quantity": 0.001,          # 0.001 계약
            "max_quantity": 1_000_000,      # 최대 100만 계약
            "precision": 4,                 # 소수점 4자리
            "min_order_usdt": 10,           # 최소 10 USDT
            "description": "OKX V5 API",
        },
        "upbit": {
            "min_quantity": 0.00000001,     # 최소 0.00000001 코인
            "max_quantity": 1000,           # 최대 1000 BTC 등가
            "precision": 8,                 # 소수점 8자리
            "min_order_krw": 5_000,         # 최소 5,000 KRW
            "description": "Upbit API",
        },
    }
    
    @staticmethod
    def validate_and_adjust(
        market: str,
        order_quantity: float,
        entry_price: float,
    ) -> Tuple[bool, float, str]:
        """
        주문 수량 유효성 검증 및 조정
        
        Args:
            market: 시장 ("kr_stock", "nasdaq", "okx_swap", "upbit")
            order_quantity: 계산된 주문 수량
            entry_price: 진입가
        
        Returns:
            (is_valid, adjusted_quantity, message)
        """
        if market not in OrderQuantityValidator.EXCHANGE_RULES:
            return False, 0, f"Unknown market: {market}"
        
        rules = OrderQuantityValidator.EXCHANGE_RULES[market]
        min_qty = rules["min_quantity"]
        max_qty = rules["max_quantity"]
        precision = rules["precision"]
        
        # 최소 주문량 검증
        if order_quantity < min_qty:
            msg = f"⚠️ {rules['description']}: Order quantity {order_quantity} < min {min_qty}"
            return False, 0, msg
        
        # 최대 주문량 검증
        if order_quantity > max_qty:
            adjusted = max_qty
            msg = f"⚠️ {rules['description']}: Capped {order_quantity} → {adjusted} (max)"
            return True, adjusted, msg
        
        # 정밀도 조정 (반올림)
        if precision >= 0:
            adjusted = round(order_quantity, precision)
        else:
            adjusted = round(order_quantity / (10 ** abs(precision))) * (10 ** abs(precision))
        
        # 최소 주문금액 검증 (USDT/KRW)
        order_value = adjusted * entry_price
        
        min_order_usdt = rules.get("min_order_usdt", 0)
        if min_order_usdt > 0 and order_value < min_order_usdt:
            return False, 0, f"Order value {order_value:.2f} USDT < min {min_order_usdt}"
        
        min_order_krw = rules.get("min_order_krw", 0)
        if min_order_krw > 0 and order_value < min_order_krw:
            return False, 0, f"Order value {order_value:,.0f} KRW < min {min_order_krw:,}"
        
        return True, adjusted, f"✅ {rules['description']}: {adjusted} units"


if __name__ == "__main__":
    print("=" * 70)
    print("[TEST] Kelly Criterion Engine")
    print("=" * 70)
    
    # 테스트 시나리오: 점진적 성장 포트폴리오
    metrics = PortfolioMetrics(
        total_balance=10_000,           # 10,000 USDT
        current_position_value=1_500,   # 1,500 USDT 포지션
        peak_balance=11_000,            # 최고 11,000 USDT
        win_rate=0.60,                  # 60% 승률
        avg_win=0.03,                   # 평균 승리 3%
        avg_loss=-0.015,                # 평균 손실 -1.5%
    )
    
    kelly = KellyCriterionEngine(kelly_fraction=0.25)
    
    # OKX 진입가 $45,000
    entry_price = 45_000
    stop_loss = 44_000
    
    order_qty, details = kelly.compute_order_quantity(
        metrics=metrics,
        entry_price=entry_price,
        stop_loss_price=stop_loss,
        leverage=5,
    )
    
    print(f"\n[Portfolio Metrics]")
    print(f"  Total Balance: ${metrics.total_balance:,.0f}")
    print(f"  Win Rate: {metrics.win_rate * 100:.1f}%")
    print(f"  Avg Win/Loss: +{metrics.avg_win * 100:.1f}% / {metrics.avg_loss * 100:.1f}%")
    
    print(f"\n[Kelly Calculation]")
    for key, value in details.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # 주문 유효성 검증
    print(f"\n[Order Validation]")
    is_valid, adjusted_qty, msg = OrderQuantityValidator.validate_and_adjust(
        "okx_swap",
        order_qty,
        entry_price,
    )
    print(f"  {msg}")
    
    print("\n" + "=" * 70)
    print("✅ Kelly Criterion Engine 테스트 완료")
    print("=" * 70)
