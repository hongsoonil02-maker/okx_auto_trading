#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_end_to_end_simulation.py — End-to-End Pipeline Verification
Verifies:
1. LOB Feed buffer provides real-time state
2. JevSignalFilter evaluates and gates signal
3. WebhookPayload serializes with order_type, target_price, jev_score, and is_simulation=True
4. Bot C handles simulation mode, logging fill without calling exchange
"""
import asyncio
import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webhook_spec import WebhookPayload, ActionType, SideType
from jev.okx_lob_feed import LOBEntry, OKXLOBFeed
from jev.jev_signal_filter import JevSignalFilter
from bot_c_okx_swap import BotCOKXSwap


class TestEndToEndSimulation(unittest.IsolatedAsyncioTestCase):

    async def test_full_simulation_flow(self):
        # Set test environment
        os.environ["USE_JEV_PREDICTION"] = "true"
        os.environ["JEV_CONFIDENCE_THRESHOLD"] = "0.65"
        os.environ["QUOTE_INSIDE_TICKS"] = "1"
        os.environ["JEV_SIMULATION_MODE"] = "true"

        feed = OKXLOBFeed()
        import time
        mock_lob = LOBEntry(
            symbol="BTC-USDT-SWAP",
            timestamp=time.time(),
            exchange_ts=int(time.time() * 1000),
            bids=[(80500.0, 10.0), (80499.0, 5.0)],
            asks=[(80501.0, 2.0), (80502.0, 8.0)],
            best_bid=80500.0,
            best_ask=80501.0,
            best_bid_size=10.0,
            best_ask_size=2.0,
            spread=1.0,
            spread_bps=0.124,
            imbalance=0.55,  # Strong bid imbalance
            micro_price=80500.83,
        )
        feed._lob_cache["BTC-USDT-SWAP"] = mock_lob

        filter_engine = JevSignalFilter(lob_feed=feed)

        # 1. Evaluate Buy Signal
        eval_result = await filter_engine.evaluate_signal(
            symbol="BTC-USDT-SWAP",
            proposed_side="BUY",
            tick_size=0.1,
            context_note="E2E Pipeline Test",
        )

        self.assertTrue(eval_result.approved)
        self.assertEqual(eval_result.order_type, "POST_ONLY")
        self.assertEqual(eval_result.target_price, 80500.1)
        self.assertTrue(eval_result.is_simulation)

        # 2. Build WebhookPayload as BaseStrategyBrain would
        payload = WebhookPayload(
            action=ActionType.EXEC,
            side=SideType.BUY,
            symbol="BTC/USDT:USDT",
            qty=0.05,
            price=eval_result.target_price,
            order_type=eval_result.order_type,
            target_price=eval_result.target_price,
            jev_score=eval_result.jev_score,
            is_simulation=eval_result.is_simulation,
        )

        # 3. Process via Bot C in simulation mode
        bot_c = BotCOKXSwap()
        bot_c._lock = asyncio.Lock()
        # mock exchange with a minimal fetch_ticker
        class MockExchange:
            async def fetch_ticker(self, symbol):
                return {"last": 80500.5}
        bot_c.exchange = MockExchange()

        res = await bot_c.execute_order(payload)
        self.assertEqual(res.get("status"), "ok")
        self.assertTrue(res.get("simulation"))
        self.assertEqual(res.get("symbol"), "BTC/USDT:USDT")
        self.assertEqual(res.get("order_type"), "POST_ONLY")
        self.assertAlmostEqual(res.get("price"), 80500.5)

        print("\n✅ [E2E Simulation Flow Verified Successfully]")
        print(f"   - Jev Decision: Score={eval_result.jev_score:.3f} | Action={eval_result.action}")
        print(f"   - Quoting: Type={eval_result.order_type} @ {eval_result.target_price}")
        print(f"   - Bot C Sim Fill: Status={res.get('status')} | Sim Price={res.get('price')}")

        await filter_engine.close()


if __name__ == "__main__":
    unittest.main()
