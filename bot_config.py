#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bot_config.py — Centralized bot configuration manager.
- Loads auto_tune_config.json (produced by daily_analyzer.py)
- Merges with .env defaults
- Provides typed access to trading parameters for all bots
- Auto-reloads when the config file changes (mtime-based)
"""
import os
import json
import time
import threading
from typing import Dict, Any, Optional, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "auto_tune_config.json")
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "auto_tune_config.json")

_lock = threading.Lock()
_cache: Optional[Dict[str, Any]] = None
_cache_mtime: float = 0.0


def _load_config_file() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def reload_config() -> Dict[str, Any]:
    """Force-reload auto_tune_config.json from disk."""
    global _cache, _cache_mtime
    with _lock:
        cfg = _load_config_file()
        _cache = cfg
        try:
            _cache_mtime = os.path.getmtime(CONFIG_FILE)
        except Exception:
            _cache_mtime = 0.0
        return cfg


def get_config() -> Dict[str, Any]:
    """Get config, auto-reloading if the file changed on disk."""
    global _cache, _cache_mtime
    with _lock:
        try:
            mtime = os.path.getmtime(CONFIG_FILE)
        except Exception:
            mtime = 0.0
        if _cache is None or mtime > _cache_mtime:
            _cache = _load_config_file()
            _cache_mtime = mtime
        return _cache if _cache else {}


class BotConfig:
    """
    Typed access to auto-tuned parameters.
    Strategy brains and execution bots should use this instead of
    hardcoded defaults to pick up daily_analyzer.py optimizations.
    """

    @staticmethod
    def _okx_params() -> Dict[str, Any]:
        cfg = get_config()
        return cfg.get("okx_params", {})



    # -- OKX / Swap trading bot --
    @property
    def okx_trailing_pct(self) -> float:
        return float(self._okx_params().get("TRAILING_PCT", 0.03))

    @property
    def okx_hard_sl_pct(self) -> float:
        return float(self._okx_params().get("HARD_SL_PCT", 0.035))

    @property
    def okx_position_pct(self) -> float:
        return float(self._okx_params().get("POSITION_PCT", 0.18))

    @property
    def okx_orb_lookback(self) -> int:
        return int(self._okx_params().get("ORB_LOOKBACK", 4))

    @property
    def okx_min_range_pct(self) -> float:
        return float(self._okx_params().get("OKX_MIN_RANGE_PCT", 0.6))

    @property
    def okx_entry_buffer_pct(self) -> float:
        return float(self._okx_params().get("OKX_ENTRY_BUFFER_PCT", 0.0004))

    @property
    def okx_min_momentum_pct(self) -> float:
        return float(self._okx_params().get("OKX_MIN_MOMENTUM_PCT", 0.0003))

    @property
    def okx_max_dca(self) -> int:
        return int(self._okx_params().get("MAX_DCA", 8))

    @property
    def okx_dca_gap_pct(self) -> float:
        return float(self._okx_params().get("OKX_DCA_GAP_PCT", 0.005))

    # -- Blocked hours / symbols --
    @property
    def blocked_hours_kst(self) -> List[int]:
        cfg = get_config()
        return cfg.get("blocked_hours_kst", [])

    @property
    def blocked_hours_kst_okx(self) -> List[int]:
        cfg = get_config()
        return cfg.get("blocked_hours_kst_okx", [])

    @property
    def blacklisted_symbols_okx(self) -> List[str]:
        cfg = get_config()
        return cfg.get("blacklisted_symbols", {}).get("OKX", [])



    # -- Convenience --
    @classmethod
    def is_symbol_blacklisted(cls, symbol: str, market: str = "OKX") -> bool:
        if market.upper() == "OKX":
            return symbol in cls().blacklisted_symbols_okx
        return False

    @classmethod
    def is_hour_blocked(cls, hour_kst: int) -> bool:
        return hour_kst in cls().blocked_hours_kst

    def refresh(self) -> None:
        """Call periodically to pick up daily_analyzer updates."""
        reload_config()


# Singleton instance for convenient import
bot_config = BotConfig()
