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


if __name__ == "__main__":
    unittest.main()
