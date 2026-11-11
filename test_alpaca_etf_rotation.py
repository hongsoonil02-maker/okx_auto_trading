# -*- coding: utf-8 -*-
import unittest
from datetime import datetime
import numpy as np
import pandas as pd

import alpaca_etf_rotation as bot_mod
from tools.research_alpaca_swing_v3 import load, h1_multi_div, h2_rp_multi


class TestAlpacaETFRotation(unittest.TestCase):
    def test_matches_research_backtest_weights_exact(self):
        o, c = load()
        sub = c[bot_mod.ALL_UNIVERSE].tail(260)
        weights, diag = bot_mod.compute_h3_target_weights(sub)

        T = 0.5 * h1_multi_div(o, c, lb=126, top_n=3) + 0.5 * h2_rp_multi(o, c, ma_n=150, include_dbc=True)
        expected = T.iloc[-1]

        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        for sym in bot_mod.ALL_UNIVERSE:
            self.assertAlmostEqual(weights.get(sym, 0.0), float(expected.get(sym, 0.0)), places=6, msg=f"Mismatch on {sym}")
        self.assertEqual(diag["signal_date"], str(sub.index[-1]))

    def test_bear_market_parks_100pct_in_shy(self):
        dates = pd.bdate_range("2025-01-01", periods=250)
        data = {}
        for i, sym in enumerate(bot_mod.ALL_UNIVERSE):
            if sym == "SHY":
                data[sym] = np.linspace(80.0, 82.0, 250)
            else:
                # Steadily falling prices below MA150 and MA200
                data[sym] = np.linspace(120.0 + i, 80.0 + i, 250)
        df = pd.DataFrame(data, index=dates)
        weights, _ = bot_mod.compute_h3_target_weights(df)
        self.assertAlmostEqual(weights["SHY"], 1.0, places=6)
        for sym in bot_mod.RISK_UNIVERSE:
            self.assertAlmostEqual(weights[sym], 0.0, places=6)

    def test_plan_rebalance_orders_sell_first_and_deadband(self):
        curr = {"QQQ": 20000.0, "SPY": 27900.0, "SHY": 10000.0}
        target = {"SPY": 0.285, "SHY": 0.460, "EFA": 0.255, "QQQ": 0.0}
        sells, buys = bot_mod.plan_rebalance_orders(
            equity=100000.0,
            current_values=curr,
            target_weights=target,
            invest_pct=0.99,
            min_order_usd=25.0,
            deadband_pct=0.005,
        )
        sell_map = {s["symbol"]: s for s in sells}
        buy_map = {b["symbol"]: b for b in buys}

        # QQQ target is 0 -> CLOSE_ALL
        self.assertIn("QQQ", sell_map)
        self.assertEqual(sell_map["QQQ"]["action"], "CLOSE_ALL")
        # SPY target = 99000*0.285 = 28215 vs 27900 -> delta +315 < 500 deadband -> skipped
        self.assertNotIn("SPY", sell_map)
        self.assertNotIn("SPY", buy_map)
        # SHY & EFA -> BUY_NOTIONAL
        self.assertIn("SHY", buy_map)
        self.assertIn("EFA", buy_map)

    def test_should_rebalance_monthly_idempotency(self):
        inst = object.__new__(bot_mod.AlpacaETFRotationBot)
        inst.state = {}
        due, reason = inst.should_rebalance(datetime(2026, 11, 11, 10, 0))
        self.assertTrue(due)
        self.assertIn("INITIAL_DEPLOY", reason)

        inst.state = {"initialized": True, "last_rebalance_ym": "2026-11"}
        due_same, _ = inst.should_rebalance(datetime(2026, 11, 15, 10, 0))
        self.assertFalse(due_same)

        due_next, reason_next = inst.should_rebalance(datetime(2026, 12, 1, 9, 35))
        self.assertTrue(due_next)
        self.assertIn("MONTHLY_REBALANCE", reason_next)


if __name__ == "__main__":
    unittest.main()
