#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_position_sizing.py — 포지션 사이징 로직 단위 테스트
equity 기반 균등 분할 사이징이 올바르게 작동하는지 검증
"""
import os
import sys

# .env 로드
from dotenv import load_dotenv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

from strategy_common import BaseStrategyBrain


class TestSizing:
    def __init__(self):
        self.brain = BaseStrategyBrain()
        self.passed = 0
        self.failed = 0

    def assert_eq(self, name, actual, expected, tolerance=0.01):
        if abs(actual - expected) <= tolerance:
            self.passed += 1
            print(f"  ✅ {name}: {actual:.2f} (expected ~{expected:.2f})")
        else:
            self.failed += 1
            print(f"  ❌ {name}: {actual:.2f} (expected ~{expected:.2f})")

    def assert_gte(self, name, actual, minimum):
        if actual >= minimum:
            self.passed += 1
            print(f"  ✅ {name}: {actual:.2f} >= {minimum:.2f}")
        else:
            self.failed += 1
            print(f"  ❌ {name}: {actual:.2f} < {minimum:.2f} (FAIL)")

    def assert_zero(self, name, actual):
        if actual == 0:
            self.passed += 1
            print(f"  ✅ {name}: 0 (진입 거부 정상)")
        else:
            self.failed += 1
            print(f"  ❌ {name}: {actual:.2f} (should be 0)")

    def run_all(self):
        print("=" * 70)
        print("포지션 사이징 로직 단위 테스트")
        print("=" * 70)

        self.test_basic_equity_division()
        self.test_entry_types()
        self.test_min_margin_floor()
        self.test_free_usdt_guard()
        self.test_low_equity_scenarios()
        self.test_real_scenario()

        print("\n" + "=" * 70)
        print(f"결과: {self.passed} passed, {self.failed} failed")
        print("=" * 70)
        return self.failed == 0

    def test_basic_equity_division(self):
        """기본 균등 분할 테스트: 7381 equity / 15 positions = 492 base margin"""
        print("\n📋 [Test 1] 기본 균등 분할 (equity=7381, max_pos=15)")

        # 신규 진입: 492 * 0.50 = 246 USDT
        m = self.brain._calc_target_margin(3648, 7381, "new")
        self.assert_eq("신규 진입 마진", m, 492 * 0.50, 1.0)

        # DCA: 492 * 0.25 = 123 USDT
        m = self.brain._calc_target_margin(3648, 7381, "dca")
        self.assert_eq("DCA 마진", m, 492 * 0.25, 1.0)

        # Pyramid: 492 * 0.35 = 172 USDT
        m = self.brain._calc_target_margin(3648, 7381, "pyramid")
        self.assert_eq("Pyramid 마진", m, 492 * 0.35, 1.0)

    def test_entry_types(self):
        """진입 유형별 비율 테스트"""
        print("\n📋 [Test 2] 진입 유형별 비율 (equity=10000, max_pos=15)")
        base = 10000 / 15  # 666.67

        for entry_type, ratio in [("new", 0.50), ("dca", 0.25), ("pyramid", 0.35), ("flip", 0.30), ("reentry", 0.30)]:
            m = self.brain._calc_target_margin(5000, 10000, entry_type)
            expected = base * ratio
            self.assert_eq(f"{entry_type} 마진", m, expected, 1.0)

    def test_min_margin_floor(self):
        """최소 마진 하한선 테스트: MIN_POSITION_MARGIN=100보다 작으면 100으로 상향"""
        print(f"\n📋 [Test 3] 최소 마진 하한선 (MIN={self.brain.MIN_POSITION_MARGIN})")

        # equity가 작아서 기본 계산 < 100이지만 free가 충분한 경우 → 100으로 상향
        # equity=1000, base=66.67, new=33.33 < 100 → 100으로 상향 (free=500 충분)
        m = self.brain._calc_target_margin(500, 1000, "new")
        self.assert_gte("소규모 equity 최소 마진", m, 100)

    def test_free_usdt_guard(self):
        """Free USDT 안전 가드: free가 너무 부족하면 진입 거부"""
        print("\n📋 [Test 4] Free USDT 안전 가드")

        # free=50 → MIN_POSITION_MARGIN(100) × 1.1 = 110보다 작으면 진입 불가
        m = self.brain._calc_target_margin(50, 7000, "new")
        self.assert_zero("free 50 USDT → 진입 거부", m)

        # free=120 → 충분하므로 진입 가능
        m = self.brain._calc_target_margin(120, 7000, "new")
        self.assert_gte("free 120 USDT → 진입 가능", m, 100)

    def test_low_equity_scenarios(self):
        """소규모 자산 시나리오"""
        print("\n📋 [Test 5] 소규모 자산 시나리오")

        # equity=500, free=500 → base=33.3, new=16.7 < 100
        # 하지만 free=500 >= 110이므로 최소 마진 100 적용
        m = self.brain._calc_target_margin(500, 500, "new")
        self.assert_gte("equity=500 최소 마진", m, 100)

        # equity=100, free=100 → base=6.67, new=3.33 < 100
        # free=100 < 110이므로 진입 거부
        m = self.brain._calc_target_margin(100, 100, "new")
        self.assert_zero("equity=100 진입 거부", m)

    def test_real_scenario(self):
        """현재 실제 시나리오 시뮬레이션"""
        print("\n📋 [Test 6] 실제 시나리오 (equity=7381, free=3648)")

        # 모든 진입 유형에서 100 USDT 이상인지 확인
        for entry_type in ["new", "dca", "pyramid", "flip", "reentry"]:
            m = self.brain._calc_target_margin(3648, 7381, entry_type)
            self.assert_gte(f"실제 {entry_type} >= 100", m, 100)

        # 비교: 기존 방식 시뮬레이션 (free_usdt 기반)
        # 기존: free(500) × portion(0.20) × 0.7 × 0.35 = 24.5 USDT
        print("\n  📊 기존 vs 개선 비교 (free=500 시나리오):")
        old_margin = 500 * 0.20 * 0.7 * 0.35  # 24.5
        new_margin = self.brain._calc_target_margin(500, 7381, "new")
        print(f"     기존 방식: {old_margin:.1f} USDT (free×portion×0.7×0.35)")
        print(f"     개선 방식: {new_margin:.1f} USDT (equity/15×0.50)")
        print(f"     개선율: {new_margin/old_margin:.1f}x")


if __name__ == "__main__":
    t = TestSizing()
    success = t.run_all()
    sys.exit(0 if success else 1)
