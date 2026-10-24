# -*- coding: utf-8 -*-
"""
bot_utils.py — Shared helpers for the golden/demo trading bots.

- send_telegram(): fire-and-forget Telegram alert with per-key rate limiting (never raises)
- load_state() / save_state(): atomic JSON persistence for restart-safe bot state
"""

import os
import json
import time
import logging
import tempfile
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger("BotUtils")

_TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
_TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
_last_sent: Dict[str, float] = {}


def send_telegram(text: str, dedupe_key: Optional[str] = None, min_interval_sec: float = 0.0) -> None:
    """Sends a Telegram message. If dedupe_key is given, suppresses repeats within min_interval_sec."""
    token = _TELEGRAM_TOKEN or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = _TELEGRAM_CHAT_ID or os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return
    if dedupe_key:
        now = time.time()
        if now - _last_sent.get(dedupe_key, 0.0) < min_interval_sec:
            return
        _last_sent[dedupe_key] = now
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:4000], "disable_web_page_preview": True},
            timeout=5,
        )
    except Exception as e:  # alerts must never break trading loops
        logger.debug(f"Telegram send failed: {e}")


def load_state(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning(f"State load failed ({path}): {e}")
        return {}


def save_state(path: str, state: Dict[str, Any]) -> None:
    """Atomic write (tmp file + rename) so a crash mid-write never corrupts the state file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except Exception as e:
        logger.warning(f"State save failed ({path}): {e}")
