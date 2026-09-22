# -*- coding: utf-8 -*-
"""
jev_nasdaq_client.py — Python Client for Bun Jev Nasdaq Sidecar Daemon
Calls http://127.0.0.1:8020/predict in < 1ms local IPC
"""
import requests
import logging

logger = logging.getLogger("JevNasdaqClient")

def query_jev_nasdaq(symbol: str, side: str = "buy", tick_size: float = 0.01) -> dict:
    url = "http://127.0.0.1:8020/predict"
    try:
        resp = requests.post(
            url,
            json={"symbol": symbol.upper(), "side": side.lower(), "tick_size": tick_size},
            timeout=5.0
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning(f"⚠️ Jev Nasdaq Daemon 쿼리 예외: {e}")
    
    # Intelligent Heuristic Fallback using cached orderbook imbalance
    quotes = get_quotes_cache()
    q = quotes.get(symbol.upper(), {})
    imb = float(q.get("imbalance", 0.0))
    prob_up = min(max(0.50 + imb * 0.35, 0.10), 0.90)
    approved = (prob_up >= 0.55) if side.lower() == "buy" else (prob_up <= 0.45)

    return {
        "symbol": symbol.upper(),
        "side": side.lower(),
        "approved": approved,
        "score": round(prob_up, 3),
        "action": "buy" if prob_up >= 0.58 else ("sell" if prob_up <= 0.42 else "neutral"),
        "order_type": "MARKET",
        "target_price": None,
        "latency_ms": 0.0,
        "is_simulation": False,
        "is_fallback": True,
        "reason": f"HEURISTIC_FALLBACK (Imbalance: {imb:+.2f})"
    }

def get_quotes_cache() -> dict:
    url = "http://127.0.0.1:8020/quotes"
    try:
        resp = requests.get(url, timeout=1.5)
        if resp.status_code == 200:
            return resp.json().get("quotes", {})
    except Exception as e:
        logger.warning(f"⚠️ Quotes cache 쿼리 예외: {e}")
    return {}

if __name__ == "__main__":
    res = query_jev_nasdaq("QQQ", "buy")
    print("Python Client Test Result:", res)
