# -*- coding: utf-8 -*-
import unittest

import pandas as pd

from backtest_v4pro_scenarios import (
    align_15m_to_confirmed_1h,
    linear_return,
    trim_1h_to_15m_coverage,
)


class TestBacktestCorrectness(unittest.TestCase):
    def test_linear_short_pnl_matches_live_formula(self):
        self.assertAlmostEqual(linear_return(100, 95, False), 0.05)
        self.assertAlmostEqual(linear_return(100, 102, False), -0.02)
        self.assertAlmostEqual(linear_return(100, 102, True), 0.02)

    def test_confirmed_15m_alignment_uses_t_plus_45_minutes(self):
        minute = 60 * 1000
        frame = pd.DataFrame({
            "ts": pd.Series([0, 15 * minute, 30 * minute, 45 * minute, 60 * minute], dtype="int64"),
            "close": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        aligned = align_15m_to_confirmed_1h({"BTC-USDT-SWAP": frame}, [0, 60 * minute])
        self.assertEqual(aligned["BTC-USDT-SWAP"]["ts"].tolist(), [45 * minute, 60 * minute])
        self.assertEqual(aligned["BTC-USDT-SWAP"]["close"].tolist(), [4.0, 5.0])

    def test_1h_coverage_trims_bars_without_confirmed_15m(self):
        minute = 60 * 1000
        data_1h = {
            "BTC-USDT-SWAP": pd.DataFrame({"ts": [0, 60 * minute, 120 * minute], "close": [1.0, 2.0, 3.0]}),
        }
        data_15m = {
            "BTC-USDT-SWAP": pd.DataFrame({
                "ts": [15 * minute, 30 * minute, 45 * minute, 60 * minute, 75 * minute],
                "close": [1.0, 2.0, 3.0, 4.0, 5.0],
            }),
        }
        trimmed = trim_1h_to_15m_coverage(data_1h, data_15m)
        self.assertEqual(trimmed["BTC-USDT-SWAP"]["ts"].tolist(), [0])

    def test_alignment_clamps_before_first_candle(self):
        minute = 60 * 1000
        frame = pd.DataFrame({"ts": [60 * minute, 75 * minute], "close": [1.0, 2.0]})
        aligned = align_15m_to_confirmed_1h({"BTC-USDT-SWAP": frame}, [0])
        self.assertEqual(aligned["BTC-USDT-SWAP"]["close"].tolist(), [1.0])


if __name__ == "__main__":
    unittest.main()
