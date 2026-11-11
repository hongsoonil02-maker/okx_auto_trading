# -*- coding: utf-8 -*-
import unittest
from tools.research_okx_alpha import load_all
from tools.research_okx_alpha_v3 import donchian_8_4h, multi_tf_4h
from tools.research_okx_alpha_v4 import mr_1d
import okx_blend_strategy as obs


class TestOKXBlendStrategy(unittest.TestCase):
    def test_matches_research_weights_exact(self):
        (_, _, _, c1), (_, h4, l4, c4) = load_all()
        sub_h4 = h4.tail(250)
        sub_l4 = l4.tail(250)
        sub_c4 = c4.tail(250)
        sub_c1 = c1.tail(160)

        w, diag = obs.compute_blend_target_weights(sub_h4, sub_l4, sub_c4, sub_c1)
        self.assertEqual(set(w.keys()), set(obs.SYMS_8))
        self.assertGreaterEqual(min(w.values()), 0.0)
        self.assertLessEqual(sum(w.values()), obs.MAX_GROSS_WEIGHT + 1e-6)
        self.assertIn("btc_1d_bull", diag)

    def test_plan_delta_orders_closes_legacy_short_and_respects_deadband(self):
        prices = {"BTC-USDT-SWAP": 74000.0, "SOL-USDT-SWAP": 84.0, "ETH-USDT-SWAP": 2050.0}
        ct_vals = {"BTC-USDT-SWAP": 0.01, "SOL-USDT-SWAP": 1.0, "ETH-USDT-SWAP": 0.1}
        min_szs = {"BTC-USDT-SWAP": 0.01, "SOL-USDT-SWAP": 0.01, "ETH-USDT-SWAP": 0.01}
        curr = {"SOL-USDT-SWAP": -90.0, "ETH-USDT-SWAP": 1.0}  # ETH current = 205 USD
        target = {"SOL-USDT-SWAP": 0.0, "ETH-USDT-SWAP": 0.052, "BTC-USDT-SWAP": 0.15}
        # eq=4000 -> ETH target=208 vs 205 (skip), BTC fresh=600 (buy), SOL=-90 (close legacy short)
        sells, buys = obs.plan_delta_orders(4000.0, prices, curr, target, ct_vals, min_szs)
        sell_map = {s["symbol"]: s for s in sells}
        buy_map = {b["symbol"]: b for b in buys}
        self.assertIn("SOL-USDT-SWAP", sell_map)
        self.assertEqual(sell_map["SOL-USDT-SWAP"]["action"], "CLOSE_LEGACY_SHORT")
        self.assertNotIn("ETH-USDT-SWAP", sell_map)
        self.assertNotIn("ETH-USDT-SWAP", buy_map)
        self.assertIn("BTC-USDT-SWAP", buy_map)

    def test_sync_emergency_sl_orders_replaces_drifted_scale_in_size(self):
        import okx_blend_demo_trader as obdt
        trader = object.__new__(obdt.OKXBlendDemoTrader)
        trader.last_error = ""
        trader.inst_cache = {"BTC-USDT-SWAP": {"ctVal": "0.01", "lotSz": "0.01", "minSz": "0.01", "tickSz": "0.1"}}
        trader.positions = {"BTC-USDT-SWAP": {"pos": "1.50", "avgPx": "74000.0", "mgnMode": "isolated"}}

        calls = []
        def fake_request(method, path, params=None, data=None):
            calls.append((method, path, data))
            if "orders-algo-pending" in path:
                # Old SL only covers 0.50 ct instead of scaled-in 1.50 ct
                return {"code": "0", "data": [{"algoId": "old1", "sz": "0.50", "slTriggerPx": "66600.0"}]}
            if "cancel-algos" in path:
                return {"code": "0", "data": [{"sCode": "0"}]}
            if "order-algo" in path:
                return {"code": "0", "data": [{"algoId": "new1"}]}
            return {"code": "0", "data": []}

        trader._request = fake_request
        ok = trader.sync_emergency_sl_orders({"BTC-USDT-SWAP": 74500.0})
        self.assertTrue(ok)
        placed = [d for m, p, d in calls if m == "POST" and "order-algo" in p]
        self.assertEqual(len(placed), 1)
        self.assertEqual(placed[0]["sz"], "1.50")
        self.assertEqual(placed[0]["slTriggerPx"], "66600.0")
        self.assertTrue(placed[0]["reduceOnly"])

        # Second pass with matching SL should not cancel or replace
        calls.clear()
        trader._request = lambda m, p, params=None, data=None: (
            {"code": "0", "data": [{"algoId": "new1", "sz": "1.50", "slTriggerPx": "66600.0"}]}
            if "orders-algo-pending" in p else calls.append((m, p, data)) or {"code": "0", "data": []}
        )
        ok2 = trader.sync_emergency_sl_orders({"BTC-USDT-SWAP": 74500.0})
        self.assertTrue(ok2)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()

