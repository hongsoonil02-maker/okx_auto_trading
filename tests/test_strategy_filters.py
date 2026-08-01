import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import position_sizer
from okx_strategy import should_take_okx_entry
from upbit_strategy import should_take_upbit_entry
from orb_engine import ORBSignal


class StrategyFilterTests(unittest.TestCase):
    def test_okx_long_entry_accepts_momentum_without_strict_vwap(self):
        self.assertTrue(
            should_take_okx_entry(
                signal=ORBSignal.LONG,
                price=101.2,
                closed_price=100.0,
                vwap_ok=False,
                vwap_below=False,
                orb_range_pct=1.2,
            )
        )

    def test_okx_short_entry_accepts_momentum_without_strict_vwap(self):
        self.assertTrue(
            should_take_okx_entry(
                signal=ORBSignal.SHORT,
                price=99.0,
                closed_price=100.0,
                vwap_ok=False,
                vwap_below=False,
                orb_range_pct=1.2,
            )
        )

    def test_okx_entry_accepts_mild_momentum_when_range_is_reasonable(self):
        self.assertTrue(
            should_take_okx_entry(
                signal=ORBSignal.LONG,
                price=100.05,
                closed_price=100.0,
                vwap_ok=False,
                vwap_below=False,
                orb_range_pct=0.6,
            )
        )

    def test_upbit_entry_accepts_breakout_near_orb_high(self):
        self.assertTrue(
            should_take_upbit_entry(
                signal=ORBSignal.LONG,
                price=100.3,
                orb_high=100.0,
                vwap_ok=False,
            )
        )

    def test_upbit_entry_rejects_weak_signal(self):
        self.assertFalse(
            should_take_upbit_entry(
                signal=ORBSignal.WAIT,
                price=100.3,
                orb_high=100.0,
                vwap_ok=False,
            )
        )

    def test_sizing_defaults_are_conservative(self):
        self.assertEqual(position_sizer.KELLY_FRACTION, 0.25)
        self.assertEqual(position_sizer.MAX_PCT, 0.25)
        self.assertEqual(position_sizer.MIN_PCT, 0.03)


if __name__ == "__main__":
    unittest.main()
