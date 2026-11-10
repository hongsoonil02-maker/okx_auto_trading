# -*- coding: utf-8 -*-
import unittest

from risk_policy import (
    calculate_target_notional,
    daily_trend_ok,
    loss_limit_breached,
    minimum_contract_is_acceptable,
    price_risk_buffer,
    projected_risk_usdt,
)


class TestRiskPolicy(unittest.TestCase):
    def test_price_buffer_includes_sl_fee_and_slippage(self):
        self.assertAlmostEqual(price_risk_buffer(0.025, 0.001, 0.001), 0.027)
        self.assertAlmostEqual(price_risk_buffer(-1, 0, 0), 0.0001)

    def test_target_notional_uses_half_percent_risk(self):
        notional = calculate_target_notional(
            equity_usdt=5000,
            risk_per_trade_pct=0.005,
            sl_pct=0.025,
            fee_pct=0.001,
            slippage_pct=0.001,
            current_open_risk_usdt=0,
            max_total_open_risk_pct=0.012,
            available_margin_usdt=5000,
            leverage=10,
            max_margin_per_trade=5000,
        )
        self.assertAlmostEqual(notional, 25 / 0.027)
        self.assertAlmostEqual(projected_risk_usdt(notional, 0.025, 0.001, 0.001), 25)

    def test_total_open_risk_caps_new_position(self):
        notional = calculate_target_notional(
            equity_usdt=5000,
            risk_per_trade_pct=0.005,
            sl_pct=0.025,
            fee_pct=0.001,
            slippage_pct=0.001,
            current_open_risk_usdt=50,
            max_total_open_risk_pct=0.012,
            available_margin_usdt=5000,
            leverage=10,
            max_margin_per_trade=5000,
        )
        self.assertAlmostEqual(notional, 10 / 0.027)

    def test_margin_and_trade_caps_are_applied(self):
        notional = calculate_target_notional(
            equity_usdt=5000,
            risk_per_trade_pct=0.10,
            sl_pct=0.01,
            fee_pct=0.001,
            slippage_pct=0.001,
            current_open_risk_usdt=0,
            max_total_open_risk_pct=0.20,
            available_margin_usdt=100,
            leverage=10,
            max_margin_per_trade=250,
        )
        self.assertAlmostEqual(notional, 900)

    def test_no_heat_means_no_trade(self):
        notional = calculate_target_notional(
            equity_usdt=5000,
            risk_per_trade_pct=0.005,
            sl_pct=0.025,
            fee_pct=0.001,
            slippage_pct=0.001,
            current_open_risk_usdt=60,
            max_total_open_risk_pct=0.012,
            available_margin_usdt=5000,
            leverage=10,
            max_margin_per_trade=5000,
        )
        self.assertEqual(notional, 0)

    def test_daily_loss_limit(self):
        self.assertTrue(loss_limit_breached(-100, 5000, 0.02))
        self.assertFalse(loss_limit_breached(-99.99, 5000, 0.02))
        self.assertFalse(loss_limit_breached(-100, 0, 0.02))

    def test_daily_trend_filter_requires_price_and_slope_alignment(self):
        rising = [100.0 + i * 0.1 for i in range(600)]
        falling = [200.0 - i * 0.1 for i in range(600)]
        self.assertTrue(daily_trend_ok(rising, True))
        self.assertFalse(daily_trend_ok(rising, False))
        self.assertTrue(daily_trend_ok(falling, False))
        self.assertFalse(daily_trend_ok(falling, True))
        self.assertIsNone(daily_trend_ok(rising[:480], True))

    def test_daily_trend_filter_threshold(self):
        rising = [100.0 + i * 0.1 for i in range(600)]
        self.assertFalse(daily_trend_ok(rising, True, min_slope_pct=100.0))

    def test_minimum_contract_requires_hard_cap_and_heat(self):
        kwargs = dict(
            min_notional_usdt=1000,
            equity_usdt=5000,
            current_open_risk_usdt=0,
            max_total_open_risk_pct=0.012,
            max_single_risk_pct=0.01,
            sl_pct=0.025,
            fee_pct=0.001,
            slippage_pct=0.001,
            available_margin_usdt=5000,
            leverage=10,
        )
        self.assertTrue(minimum_contract_is_acceptable(**kwargs))
        kwargs["current_open_risk_usdt"] = 40
        self.assertFalse(minimum_contract_is_acceptable(**kwargs))


if __name__ == "__main__":
    unittest.main()
