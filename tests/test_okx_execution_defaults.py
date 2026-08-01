import asyncio
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bot_c_okx_swap as bot_module


class OKXExecutionDefaultsTests(unittest.TestCase):
    def test_parse_bool_env_defaults_to_requested_value(self):
        self.assertTrue(bot_module.parse_bool_env("MISSING_FLAG", True))
        self.assertFalse(bot_module.parse_bool_env("MISSING_FLAG", False))

    def test_post_only_respects_environment_configuration(self):
        self.assertEqual(bot_module.OKX_POST_ONLY, bot_module.parse_bool_env("OKX_POST_ONLY", False))

    def test_lock_is_created_inside_running_loop(self):
        async def run():
            bot = bot_module.BotCOKXSwap()
            self.assertIsNone(bot._lock)
            await bot._ensure_lock()
            self.assertIsNotNone(bot._lock)
            self.assertIsInstance(bot._lock, asyncio.Lock)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
