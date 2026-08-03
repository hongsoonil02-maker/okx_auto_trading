#!/usr/bin/env python3
import json
import sys
import urllib.request
from typing import Dict, Tuple

TARGETS = [
    ("BotC_OKX", "http://127.0.0.1:8003/health"),
    ("BotD_Upbit", "http://127.0.0.1:8004/health"),
]


def check_health(url: str) -> Tuple[bool, Dict[str, object]]:
    req = urllib.request.Request(url, headers={"User-Agent": "coinbot-health-check"})
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            return True, data
    except Exception as exc:  # pragma: no cover - network/HTTP fallback
        return False, {"error": str(exc)}


if __name__ == "__main__":
    ok = True
    for name, url in TARGETS:
        is_ok, payload = check_health(url)
        if is_ok:
            print(f"[OK] {name}: {payload}")
        else:
            print(f"[FAIL] {name}: {payload.get('error', 'unknown')}")
            ok = False
    sys.exit(0 if ok else 1)
