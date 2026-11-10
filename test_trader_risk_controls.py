# -*- coding: utf-8 -*-
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from okx_jev_demo_trader import (
    DAILY_LOSS_LIMIT_PCT,
    RISK_FEE_ROUND_TRIP_PCT,
    RISK_SLIPPAGE_PCT,
    WEEKLY_LOSS_LIMIT_PCT,
    OKXJevDemoTrader,
)
from risk_policy import projected_risk_usdt


class TestTraderRiskControls(unittest.TestCase):
    def _bare_trader(self):
        trader = object.__new__(OKXJevDemoTrader)
        trader.positions = {}
        trader.position_params = {}
        trader.peak_prices = {}
        trader.trough_prices = {}
        trader.usdt_equity = 5000.0
        trader.account_equity = 5000.0
        trader.daily_pnl_date = ""
        trader.daily_start_usdt = 0.0
        trader.daily_realized_pnl = 0.0
        trader.weekly_pnl_key = ""
        trader.weekly_start_usdt = 0.0
        trader.weekly_realized_pnl = 0.0
        trader.last_snapshot_hour = ""
        trader.daily_regime_cache = (0.0, None)
        return trader

    def test_daily_loss_limit_blocks_entries(self):
        trader = self._bare_trader()
        day, week = trader._period_keys(datetime.now(timezone.utc))
        trader.daily_pnl_date = day
        trader.weekly_pnl_key = week
        trader.daily_start_usdt = 5000.0
        trader.weekly_start_usdt = 5000.0
        trader.daily_realized_pnl = -5000.0 * DAILY_LOSS_LIMIT_PCT
        self.assertIn("DAILY_LOSS_LIMIT", trader._risk_block_reason())

    def test_weekly_loss_limit_blocks_entries(self):
        trader = self._bare_trader()
        day, week = trader._period_keys(datetime.now(timezone.utc))
        trader.daily_pnl_date = day
        trader.weekly_pnl_key = week
        trader.daily_start_usdt = 5000.0
        trader.weekly_start_usdt = 5000.0
        trader.weekly_realized_pnl = -5000.0 * WEEKLY_LOSS_LIMIT_PCT
        self.assertIn("WEEKLY_LOSS_LIMIT", trader._risk_block_reason())

    def test_pending_opens_cannot_exceed_position_slots(self):
        trader = self._bare_trader()
        trader.positions = {}
        self.assertTrue(trader._slot_available(0))
        self.assertTrue(trader._slot_available(1))
        self.assertFalse(trader._slot_available(2))
        trader.positions = {"BTC-USDT-SWAP": {}, "ETH-USDT-SWAP": {}}
        self.assertFalse(trader._slot_available(0))

    def test_open_heat_uses_notional_and_stored_sl(self):
        trader = self._bare_trader()
        trader.positions = {
            "BTC-USDT-SWAP": {"notionalUsd": "1000", "pos": "1"},
            "ETH-USDT-SWAP": {"notionalUsd": "500", "pos": "-1"},
        }
        trader.position_params = {
            "BTC-USDT-SWAP": {"sl_pct": 0.025},
            "ETH-USDT-SWAP": {"sl_pct": 0.04},
        }
        expected = (
            projected_risk_usdt(1000, 0.025, RISK_FEE_ROUND_TRIP_PCT, RISK_SLIPPAGE_PCT)
            + projected_risk_usdt(500, 0.04, RISK_FEE_ROUND_TRIP_PCT, RISK_SLIPPAGE_PCT)
        )
        self.assertAlmostEqual(trader._current_open_risk_usdt(), expected)

    def test_history_breakdown_does_not_double_count_costs(self):
        h = {
            "pnl": "-243.210896",
            "fee": "-17.970519092",
            "fundingFee": "4.8134875816033714",
            "liqPenalty": "0",
            "realizedPnl": "-256.3679275103966284",
        }
        result = OKXJevDemoTrader._history_breakdown(h)
        self.assertAlmostEqual(result["net_pnl_usdt"], -256.3679275103966)
        self.assertAlmostEqual(result["gross_pnl_usdt"], -243.210896)

    def test_history_breakdown_falls_back_when_realized_missing(self):
        h = {"pnl": "10", "fee": "-0.4", "fundingFee": "-0.1", "liqPenalty": "-0.2"}
        result = OKXJevDemoTrader._history_breakdown(h)
        self.assertAlmostEqual(result["net_pnl_usdt"], 9.3)

    def test_external_close_observes_positions_already_synced(self):
        trader = self._bare_trader()
        trader.positions = {"BTC-USDT-SWAP": {"pos": "1", "avgPx": "100"}}
        seen = {}

        def fake_close(inst_id, prev_pos):
            seen["inst_id"] = inst_id
            seen["positions"] = dict(trader.positions)

        with patch.object(trader, "_request", return_value={"data": []}), \
                patch.object(trader, "_handle_external_close", side_effect=fake_close):
            trader.sync_positions()

        self.assertEqual(seen["inst_id"], "BTC-USDT-SWAP")
        self.assertEqual(seen["positions"], {})
        self.assertEqual(trader.positions, {})

    def test_btc_daily_regime_matches_price_and_slope(self):
        trader = self._bare_trader()
        rising = pd.DataFrame({"close": [100.0 + i * 0.1 for i in range(600)]})
        falling = pd.DataFrame({"close": [200.0 - i * 0.1 for i in range(600)]})
        with patch.object(trader, "fetch_history_bars", return_value=rising):
            state = trader.btc_daily_regime(force=True)
        self.assertTrue(state["long_ok"])
        self.assertFalse(state["short_ok"])
        with patch.object(trader, "fetch_history_bars", return_value=falling):
            state = trader.btc_daily_regime(force=True)
        self.assertFalse(state["long_ok"])
        self.assertTrue(state["short_ok"])

    def test_btc_daily_regime_blocks_when_history_is_short(self):
        trader = self._bare_trader()
        with patch.object(trader, "fetch_history_bars", return_value=pd.DataFrame({"close": [100.0] * 100})):
            self.assertIsNone(trader.btc_daily_regime(force=True))

    def test_jsonl_append_is_durable_and_parseable(self):
        trader = self._bare_trader()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ledger.jsonl")
            trader._append_jsonl(path, {"event": "TEST", "value": 1})
            trader._append_jsonl(path, {"event": "TEST", "value": 2})
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual([r["value"] for r in rows], [1, 2])


if __name__ == "__main__":
    unittest.main()
