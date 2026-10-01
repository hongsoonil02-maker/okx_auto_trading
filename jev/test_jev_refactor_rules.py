# -*- coding: utf-8 -*-
"""
test_jev_refactor_rules.py — Verification of 3 Architectural Rules & Dynamic Capital Allocation:
1. Version Pinning ('jev-1.15-prod')
2. Hard Risk Veto Rule:
   - Maximum Parallel Positions Gate (VETO_MAX_POSITIONS_REACHED)
   - Dynamic Capital Max Cap (Remaining slots, Available margin 90%, Single-symbol 40% ceiling)
   - Strict Slippage tolerance check
   - Flash crash Stop Loss & Liquidation distance safety check
3. Direct low-latency session and deterministic Safe HOLD Fallback
"""
import asyncio
import os
import unittest
from jev.typesafe_client import TypesafeJevClient, JevDecision, PINNED_MODEL_VERSION
from jev.hard_risk_veto import HardRiskVetoEngine, VetoResult
from jev.jev_signal_filter import JevSignalFilter, FilterResult
from jev.okx_lob_feed import LOBEntry


class TestJevRefactorRules(unittest.IsolatedAsyncioTestCase):

    def test_rule1_version_pinning(self):
        """Rule 1: Verify version pinning enforces 'jev-1.15-prod'."""
        self.assertEqual(PINNED_MODEL_VERSION, "jev-1.15-prod")
        client = TypesafeJevClient()
        self.assertEqual(client.model, "jev-1.15-prod")

        # Even if environment variable passes "jev-latest", client pins to "jev-1.15-prod"
        os.environ["JEV_MODEL"] = "jev-latest"
        client2 = TypesafeJevClient()
        self.assertEqual(client2.model, "jev-1.15-prod")
        del os.environ["JEV_MODEL"]

    def test_rule2_hard_risk_veto_engine(self):
        """Rule 2: Verify Hard Risk Veto Engine with Dynamic Capital Allocation."""
        engine = HardRiskVetoEngine(
            max_slippage_bps=15.0,
            default_max_parallel_symbols=4,
            default_leverage=20,
            max_single_exposure_ratio=0.40,     # Max 40% margin of equity for single symbol
            available_margin_utilization=0.90,  # 90% utilization of available margin
            max_order_notional_usdt=50000.0,
            min_liquidation_buffer_pct=0.02,
        )

        # 2-A: Maximum Open Positions Reached Test
        # When active positions (4) >= max allowed (4), strictly veto entry
        res_max_pos = engine.evaluate_veto(
            symbol="BTC/USDT:USDT",
            side="BUY",
            order_price=50000.0,
            market_mid_price=50000.0,
            order_qty=0.02,
            contract_size=1.0,
            account_equity=10000.0,
            stop_loss_price=49000.0,
            leverage=20,
            max_parallel_symbols=4,
            current_open_positions=4,
            available_margin_usdt=5000.0,
        )
        self.assertTrue(res_max_pos.is_vetoed)
        self.assertIn("VETO_MAX_POSITIONS_REACHED", res_max_pos.reason)

        # 2-B: Dynamic Capital Allocation - Single Remaining Slot Expansion
        # Total equity = $10,000. Available margin = $2,000. Remaining slots = 1 (current=3, max=4).
        # Usable margin pool = $2,000 * 0.90 = $1,800.
        # Slot notional cap = $1,800 * 20 = $36,000.
        # Single symbol ceiling = $10,000 * 0.40 * 20 = $80,000. Effective cap = min(36000, 80000) = $36,000.
        # An order of $25,000 notional (qty=0.5 BTC @ 50,000) should PASS thanks to dynamic slot expansion:
        res_dynamic_pass = engine.evaluate_veto(
            symbol="BTC/USDT:USDT",
            side="BUY",
            order_price=50000.0,
            market_mid_price=50000.0,
            order_qty=0.5, # 0.5 * 50,000 = $25,000 notional < $36,000 cap
            contract_size=1.0,
            account_equity=10000.0,
            stop_loss_price=49000.0,
            leverage=20,
            max_parallel_symbols=4,
            current_open_positions=3, # 1 remaining slot
            available_margin_usdt=2000.0,
        )
        self.assertFalse(res_dynamic_pass.is_vetoed)
        self.assertEqual(res_dynamic_pass.reason, "PASSED_ALL_HARD_RULES")

        # 2-C: Single Symbol Safety Ceiling Enforcement (40% Equity Margin Limit)
        # Order of $40,000 notional exceeds the $36,000 slot limit -> Vetoed!
        res_dynamic_veto = engine.evaluate_veto(
            symbol="BTC/USDT:USDT",
            side="BUY",
            order_price=50000.0,
            market_mid_price=50000.0,
            order_qty=0.8, # 0.8 * 50,000 = $40,000 notional > $36,000 cap
            contract_size=1.0,
            account_equity=10000.0,
            stop_loss_price=49000.0,
            leverage=20,
            max_parallel_symbols=4,
            current_open_positions=3,
            available_margin_usdt=2000.0,
        )
        self.assertTrue(res_dynamic_veto.is_vetoed)
        self.assertIn("VETO_MAX_CAP_EXCEEDED", res_dynamic_veto.reason)

        # 2-D: Slippage Violation Test
        # Mid is 50,000, but Buy order price is 50,150 (30 bps slippage > 15 bps limit)
        res_slip = engine.evaluate_veto(
            symbol="BTC/USDT:USDT",
            side="BUY",
            order_price=50150.0,
            market_mid_price=50000.0,
            order_qty=0.01,
            contract_size=1.0,
            account_equity=10000.0,
            stop_loss_price=49000.0,
            leverage=20,
            max_parallel_symbols=4,
            current_open_positions=1,
            available_margin_usdt=5000.0,
        )
        self.assertTrue(res_slip.is_vetoed)
        self.assertIn("SLIPPAGE", res_slip.reason)

        # 2-E: Flash Crash Stop Loss Missing Test
        res_no_sl = engine.evaluate_veto(
            symbol="BTC/USDT:USDT",
            side="BUY",
            order_price=50000.0,
            market_mid_price=50000.0,
            order_qty=0.02,
            contract_size=1.0,
            account_equity=10000.0,
            stop_loss_price=None,
            leverage=20,
            max_parallel_symbols=4,
            current_open_positions=1,
            available_margin_usdt=5000.0,
        )
        self.assertTrue(res_no_sl.is_vetoed)
        self.assertIn("STOP_LOSS_MISSING", res_no_sl.reason)

        # 2-F: Flash Crash Dangerous Stop Loss (Below Liquidation Price) Test
        res_danger_sl = engine.evaluate_veto(
            symbol="BTC/USDT:USDT",
            side="BUY",
            order_price=50000.0,
            market_mid_price=50000.0,
            order_qty=0.02,
            contract_size=1.0,
            account_equity=10000.0,
            stop_loss_price=47000.0, # Below estimated liq (~47,512.5)
            leverage=20,
            max_parallel_symbols=4,
            current_open_positions=1,
            available_margin_usdt=5000.0,
        )
        self.assertTrue(res_danger_sl.is_vetoed)
        self.assertIn("STOP_LOSS_BELOW_LIQ", res_danger_sl.metrics.get("rule", ""))

    async def test_rule3_safe_hold_fallback(self):
        """Rule 3: Verify timeout/503 fallback deterministically yields HOLD."""
        client = TypesafeJevClient()
        fallback_res = client._safe_hold_fallback("HTTP_503_SERVICE_UNAVAILABLE", 42.0)
        self.assertTrue(fallback_res.is_fallback)
        self.assertEqual(fallback_res.action, "HOLD")
        self.assertEqual(fallback_res.up_in_10, 0.50)
        self.assertEqual(fallback_res.model_version, "jev-1.15-prod")


if __name__ == "__main__":
    unittest.main()
