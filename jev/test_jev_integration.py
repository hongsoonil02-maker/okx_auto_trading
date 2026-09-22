#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_jev_integration.py — Integration and Unit Test Suite for Jev AI Integration
"""
import asyncio
import os
import sys
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webhook_spec import WebhookPayload, ActionType, SideType
from jev.okx_lob_feed import LOBEntry, OKXLOBFeed
from jev.typesafe_client import TypesafeJevClient, JevDecision
from jev.jev_signal_filter import JevSignalFilter, FilterResult
from bot_config import bot_config


class TestJevIntegration(unittest.IsolatedAsyncioTestCase):

    def test_webhook_payload_jev_fields(self):
        """Test WebhookPayload serialization/deserialization with Jev fields."""
        payload = WebhookPayload(
            action=ActionType.EXEC,
            side=SideType.BUY,
            symbol="BTC/USDT:USDT",
            qty=0.01,
            price=63500.5,
            order_type="POST_ONLY",
            target_price=63500.5,
            jev_score=0.742,
            is_simulation=True,
        )
        json_str = payload.to_json()
        self.assertIn("POST_ONLY", json_str)
        self.assertIn("0.742", json_str)
        self.assertIn('"is_simulation": true', json_str)

        restored = WebhookPayload.from_json(json_str)
        self.assertEqual(restored.order_type, "POST_ONLY")
        self.assertEqual(restored.target_price, 63500.5)
        self.assertEqual(restored.jev_score, 0.742)
        self.assertTrue(restored.is_simulation)

    def test_lob_entry_math(self):
        """Test LOB calculations: spread, spread_bps, imbalance, micro_price."""
        bids = [(100.0, 5.0), (99.5, 10.0)]
        asks = [(100.5, 2.0), (101.0, 8.0)]
        entry = LOBEntry(
            symbol="TEST-USDT-SWAP",
            timestamp=1000.0,
            exchange_ts=1000000,
            bids=bids,
            asks=asks,
            best_bid=100.0,
            best_ask=100.5,
            best_bid_size=5.0,
            best_ask_size=2.0,
            spread=0.5,
            spread_bps=49.87,
            imbalance=(15.0 - 10.0) / (15.0 + 10.0),  # +0.20
            micro_price=(100.0 * 2.0 + 100.5 * 5.0) / 7.0,
        )
        self.assertAlmostEqual(entry.spread, 0.5)
        self.assertAlmostEqual(entry.imbalance, 0.20)
        self.assertAlmostEqual(entry.micro_price, 100.357, places=3)
        formatted = entry.format_for_jev()
        self.assertIn("Symbol: TEST-USDT-SWAP", formatted)
        self.assertIn("LOB Imbalance: +0.200", formatted)

    async def test_typesafe_client_heuristic_fallback(self):
        """Test fallback heuristic prediction when no API key is provided."""
        client = TypesafeJevClient(api_key="", timeout_ms=500)
        decision = await client.predict_orderbook("mock state", lob_imbalance=0.60)
        self.assertTrue(decision.is_fallback)
        self.assertGreaterEqual(decision.up_in_10, 0.65)
        self.assertEqual(decision.action, "buy")
        await client.close()

    async def test_jev_signal_filter_gating(self):
        """Test JevSignalFilter gating threshold logic and POST_ONLY pricing."""
        os.environ["USE_JEV_PREDICTION"] = "true"
        os.environ["JEV_CONFIDENCE_THRESHOLD"] = "0.65"
        os.environ["QUOTE_INSIDE_TICKS"] = "1"
        os.environ["JEV_SIMULATION_MODE"] = "true"

        feed = OKXLOBFeed()
        # Mock feed cache
        mock_lob = LOBEntry(
            symbol="BTC-USDT-SWAP",
            timestamp=time.time(),  # Current epoch time
            exchange_ts=100000,
            bids=[(80000.0, 5.0)],
            asks=[(80001.0, 2.0)],
            best_bid=80000.0,
            best_ask=80001.0,
            best_bid_size=5.0,
            best_ask_size=2.0,
            spread=1.0,
            spread_bps=0.125,
            imbalance=0.60,  # strong positive imbalance -> up_in_10 ~0.74
            micro_price=80000.71,
        )
        feed._lob_cache["BTC-USDT-SWAP"] = mock_lob

        client = TypesafeJevClient(api_key="", timeout_ms=500)
        filter_engine = JevSignalFilter(lob_feed=feed, jev_client=client)

        # 1. Evaluate BUY with high positive imbalance
        res_buy = await filter_engine.evaluate_signal("BTC-USDT-SWAP", "BUY", tick_size=0.1)
        self.assertTrue(res_buy.approved)
        self.assertGreaterEqual(res_buy.jev_score, 0.65)
        self.assertEqual(res_buy.order_type, "POST_ONLY")
        self.assertEqual(res_buy.target_price, 80000.1)  # best_bid + 1 * 0.1
        self.assertTrue(res_buy.is_simulation)

        # 2. Evaluate SELL with high positive imbalance -> should be rejected because down_score < 0.65
        res_sell = await filter_engine.evaluate_signal("BTC-USDT-SWAP", "SELL", tick_size=0.1)
        self.assertFalse(res_sell.approved)
        self.assertIn("LOW_DOWN_IN_10", res_sell.reason)

        await filter_engine.close()

    def test_bot_config_properties(self):
        """Test bot_config properties for Jev integration."""
        os.environ["USE_JEV_PREDICTION"] = "true"
        os.environ["JEV_CONFIDENCE_THRESHOLD"] = "0.70"
        os.environ["QUOTE_INSIDE_TICKS"] = "2"
        self.assertTrue(bot_config.use_jev_prediction)
        self.assertEqual(bot_config.jev_confidence_threshold, 0.70)
        self.assertEqual(bot_config.quote_inside_ticks, 2)


if __name__ == "__main__":
    unittest.main()
