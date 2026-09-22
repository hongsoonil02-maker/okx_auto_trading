# -*- coding: utf-8 -*-
"""
typesafe_client.py — Typesafe AI Jev (System One) Sub-Second Client
- Official Endpoint: POST https://api.typesafe.ai/v1/systemone
- Persistent aiohttp session with HTTP Keep-Alive & DNS caching for ultra-low latency
- Questions schema:
    - up_in_10: noul (probability 0.0 ~ 1.0 of upward tick in next 10s)
    - action: choice (buy / sell / neutral) with calibrated probabilities and confidence
- Hard timeout control (< 500ms) with seamless fallback
"""
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import aiohttp

logger = logging.getLogger("Typesafe_Jev")


@dataclass
class JevDecision:
    up_in_10: float = 0.50  # Probability (0.0 to 1.0)
    action: str = "neutral"  # 'buy', 'sell', or 'neutral'
    action_confidence: float = 0.0  # Confidence score (0.0 to 1.0)
    action_probabilities: Dict[str, float] = field(default_factory=lambda: {"buy": 0.33, "sell": 0.33, "neutral": 0.34})
    latency_ms: float = 0.0
    is_fallback: bool = False
    error: Optional[str] = None
    model_version: str = "jev-latest"


class TypesafeJevClient:
    API_URL = "https://api.typesafe.ai/v1/systemone"
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "jev-latest",
        timeout_ms: int = 500,
    ):
        raw_key = api_key if api_key is not None else (os.getenv("TYPESAFE_AI_API_KEY", "") or os.getenv("OPENROUTER_API_KEY", ""))
        self.api_key = raw_key.strip()
        self.model = os.getenv("JEV_MODEL", model)
        self.timeout_ms = int(os.getenv("JEV_TIMEOUT_MS", str(timeout_ms)))
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()

    @property
    def is_openrouter(self) -> bool:
        return self.api_key.startswith("sk-or-")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=10,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
                force_close=False,
            )
            client_timeout = aiohttp.ClientTimeout(
                total=max(1.0, self.timeout_ms / 1000.0),
                connect=0.25,
                sock_read=0.75,
            )
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/quant_system",
                "X-Title": "OKX Jev Trader",
            }
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=client_timeout,
                headers=headers,
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def predict_orderbook(
        self,
        state_text: str,
        timeout_override_ms: Optional[int] = None,
        lob_imbalance: float = 0.0,
    ) -> JevDecision:
        """
        Sends orderbook state to Jev model (or OpenRouter) and parses typed decision answers.
        Guarantees response within timeout; on timeout/error, returns a graceful fallback.
        """
        # If API key is not configured, trigger instant simulated heuristic fallback
        if not self.api_key:
            logger.debug("API 키 미설정 — 휴리스틱 시뮬레이션 모드로 폴백")
            return self._heuristic_simulation(lob_imbalance)

        t_limit = (timeout_override_ms or self.timeout_ms) / 1000.0

        # Automatic delegation to OpenRouter if key is OpenRouter format
        if self.is_openrouter:
            return await self._predict_openrouter(state_text, t_limit, lob_imbalance)

        payload = {
            "model": self.model,
            "state": state_text,
            "questions": {
                "up_in_10": {
                    "type": "noul",
                    "instructions": "Will the best bid or mid-price tick upward within the next 10 seconds?",
                },
                "action": {
                    "type": "choice",
                    "instructions": "Immediate micro-momentum order recommendation",
                    "criteria": {
                        "buy": "Strong bid support, asks getting absorbed, upward orderbook pressure",
                        "sell": "Heavy ask resistance, bids thinning out, downward orderbook pressure",
                        "neutral": "Balanced order book, no clear micro-trend",
                    },
                },
            },
        }

        t_limit = (timeout_override_ms or self.timeout_ms) / 1000.0
        start_time = time.perf_counter()

        try:
            session = await self._get_session()
            async with session.post(self.API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=t_limit)) as resp:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                if resp.status == 200:
                    data = await resp.json()
                    answers = data.get("answers", {})
                    
                    # Parse up_in_10 (noul: float probability)
                    noul_val = answers.get("up_in_10", {}).get("noul", 0.50)
                    
                    # Parse action (choice: string choice, probabilities, confidence)
                    action_data = answers.get("action", {})
                    choice = action_data.get("choice", "neutral")
                    confidence = float(action_data.get("confidence", 0.0))
                    probs = action_data.get("probabilities", {"buy": 0.33, "sell": 0.33, "neutral": 0.34})
                    model_ver = data.get("model", self.model)

                    return JevDecision(
                        up_in_10=float(noul_val),
                        action=choice,
                        action_confidence=confidence,
                        action_probabilities=probs,
                        latency_ms=elapsed_ms,
                        is_fallback=False,
                        model_version=model_ver,
                    )
                else:
                    err_text = await resp.text()
                    logger.warning(f"⚠️ Jev API 응답 에러 (HTTP {resp.status}): {err_text[:200]}")
                    fallback = self._heuristic_simulation(lob_imbalance)
                    fallback.latency_ms = elapsed_ms
                    fallback.is_fallback = True
                    fallback.error = f"HTTP_{resp.status}"
                    return fallback

        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.warning(f"⏱️ Jev API 타임아웃 발생 ({elapsed_ms:.1f}ms > {t_limit*1000}ms) — 즉시 폴백")
            fallback = self._heuristic_simulation(lob_imbalance)
            fallback.latency_ms = elapsed_ms
            fallback.is_fallback = True
            fallback.error = "TIMEOUT"
            return fallback

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"❌ Jev API 요청 예외: {e}")
            fallback = self._heuristic_simulation(lob_imbalance)
            fallback.latency_ms = elapsed_ms
            fallback.is_fallback = True
            fallback.error = str(e)
            return fallback

    async def _predict_openrouter(
        self,
        state_text: str,
        t_limit: float,
        lob_imbalance: float,
    ) -> JevDecision:
        model = os.getenv("OPENROUTER_MODEL", "google/gemini-3.5-flash-lite")
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an ultra-fast high-frequency trading decision model. "
                        "Analyze orderbook depth, spread, and imbalance. "
                        "Return ONLY a valid JSON object: "
                        "{\"up_in_10\": <float 0.0 to 1.0>, \"action\": <\"buy\"|\"sell\"|\"neutral\">, \"confidence\": <float 0.0 to 1.0>}. "
                        "No markdown fences, no extra text."
                    ),
                },
                {"role": "user", "content": state_text},
            ],
            "temperature": 0.0,
            "max_tokens": 60,
        }

        start_time = time.perf_counter()
        try:
            session = await self._get_session()
            async with session.post(
                self.OPENROUTER_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=t_limit),
            ) as resp:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                if resp.status == 200:
                    data = await resp.json()
                    raw_content = data["choices"][0]["message"]["content"].strip()
                    if "```" in raw_content:
                        raw_content = raw_content.split("```")[1]
                        if raw_content.startswith("json"):
                            raw_content = raw_content[4:].strip()
                    parsed = json.loads(raw_content)
                    up_in_10 = float(parsed.get("up_in_10", 0.50))
                    action = str(parsed.get("action", "neutral")).lower()
                    conf = float(parsed.get("confidence", 0.50))

                    return JevDecision(
                        up_in_10=up_in_10,
                        action=action,
                        action_confidence=conf,
                        action_probabilities={
                            "buy": round(up_in_10, 3),
                            "sell": round(1.0 - up_in_10, 3),
                            "neutral": round(max(0.0, 1.0 - abs(up_in_10 - 0.5) * 2), 3),
                        },
                        latency_ms=elapsed_ms,
                        is_fallback=False,
                        model_version=f"openrouter/{model}",
                    )
                else:
                    err_text = await resp.text()
                    logger.warning(f"⚠️ OpenRouter API 오류 (HTTP {resp.status}): {err_text[:150]}")
                    fallback = self._heuristic_simulation(lob_imbalance)
                    fallback.latency_ms = elapsed_ms
                    fallback.is_fallback = True
                    fallback.error = f"OPENROUTER_{resp.status}"
                    return fallback

        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.warning(f"⏱️ OpenRouter 타임아웃 ({elapsed_ms:.1f}ms > {t_limit*1000}ms) — 즉시 폴백")
            fallback = self._heuristic_simulation(lob_imbalance)
            fallback.latency_ms = elapsed_ms
            fallback.is_fallback = True
            fallback.error = "TIMEOUT"
            return fallback

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"❌ OpenRouter 요청 예외: {e}")
            fallback = self._heuristic_simulation(lob_imbalance)
            fallback.latency_ms = elapsed_ms
            fallback.is_fallback = True
            fallback.error = str(e)
            return fallback

    def _heuristic_simulation(self, lob_imbalance: float) -> JevDecision:
        """
        Internal fallback heuristic based on Orderbook Imbalance Ratio (OIR).
        Used when offline, during network failure, or timeout.
        """
        # Map imbalance (-1.0 to 1.0) to probability (0.0 to 1.0)
        prob_up = max(0.01, min(0.99, 0.50 + (lob_imbalance * 0.40)))
        if prob_up >= 0.65:
            action = "buy"
            confidence = (prob_up - 0.5) * 2
        elif prob_up <= 0.35:
            action = "sell"
            confidence = (0.5 - prob_up) * 2
        else:
            action = "neutral"
            confidence = 0.20

        return JevDecision(
            up_in_10=round(prob_up, 3),
            action=action,
            action_confidence=round(confidence, 3),
            action_probabilities={
                "buy": round(prob_up, 3),
                "sell": round(1.0 - prob_up, 3),
                "neutral": round(max(0.0, 1.0 - abs(prob_up - 0.5) * 2), 3),
            },
            latency_ms=1.2,
            is_fallback=True,
            model_version="heuristic-fallback-v1",
        )
