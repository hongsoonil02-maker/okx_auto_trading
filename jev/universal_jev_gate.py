# -*- coding: utf-8 -*-
"""
universal_jev_gate.py — Unified Sub-Second Decision Filter & Guardrail for All Asset Classes
Supports:
1. CRYPTO (OKX Swaps): Microstructure, Liquidation/Long Trap & Fake Breakout Filter
2. US_STOCKS (Alpaca / KIS US): Dilution/Offering Gap Classifier & False Momentum Veto
3. KR_STOCKS (KIS KR): Convertible Bond (CB/BW) Dumping & Theme Peak Filter
4. KILL_SWITCH: Emergency early exit evaluation for open positions before hitting hard SL
"""

import os
import sys
import time
import json
import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import aiohttp
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

logger = logging.getLogger("UniversalJevGate")

PINNED_MODEL_VERSION = "jev-1.15-prod"


@dataclass
class JevGateDecision:
    approved: bool
    confidence: float
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    reason: str
    action: str      # "ENTER", "VETO", "EXIT_NOW", "HOLD"
    latency_ms: float
    is_fallback: bool = False


class UniversalJevGate:
    """
    Sub-second Multi-Asset AI Gatekeeper powered by Typesafe Jev / OpenRouter
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        confidence_threshold: float = 0.60,
        timeout_ms: int = 2500,
        simulation_mode: bool = False,
        fail_closed: Optional[bool] = None
    ):
        raw_key = api_key or os.getenv("TYPESAFE_AI_API_KEY", "") or os.getenv("OPENROUTER_API_KEY", "")
        self.api_key = raw_key.strip()
        self.confidence_threshold = confidence_threshold
        self.timeout_ms = timeout_ms
        self.simulation_mode = simulation_mode or (os.getenv("JEV_SIMULATION_MODE", "false").lower() == "true")
        self.is_enabled = os.getenv("USE_JEV_PREDICTION", "true").lower() == "true"
        if fail_closed is None:
            self.fail_closed = os.getenv("JEV_FAIL_CLOSED", "true").lower() == "true"
        else:
            self.fail_closed = fail_closed
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if (
            self._session is None
            or self._session.closed
            or (current_loop is not None and getattr(self._session, "_loop", None) != current_loop)
        ):
            if self._session and not self._session.closed:
                try:
                    await self._session.close()
                except Exception:
                    pass
            timeout = aiohttp.ClientTimeout(total=self.timeout_ms / 1000.0)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def verify_entry(
        self,
        asset_class: str,  # "CRYPTO", "US_STOCKS", "KR_STOCKS"
        symbol: str,
        setup_name: str,
        context: Dict[str, Any]
    ) -> JevGateDecision:
        """
        Gates technical entry signals based on Jev multi-choice qualitative validation.
        """
        start_t = time.perf_counter()

        if not self.is_enabled or not self.api_key:
            return JevGateDecision(
                approved=True,
                confidence=1.0,
                risk_level="LOW",
                reason="Jev disabled or API key missing -> Baseline pass-through",
                action="ENTER",
                latency_ms=0.0,
                is_fallback=True
            )

        # 1. Construct asset-specific validation prompt
        prompt = self._build_prompt(asset_class, symbol, setup_name, context)

        # 2. Query Jev / Typesafe SystemOne
        try:
            session = await self._get_session()
            url = "https://openrouter.ai/api/v1/chat/completions" if self.api_key.startswith("sk-or-") else "https://api.typesafe.ai/v1/systemone"
            model_target = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001") if self.api_key.startswith("sk-or-") else PINNED_MODEL_VERSION

            payload = {
                "model": model_target,
                "messages": [
                    {"role": "system", "content": "You are a quantitative sub-second trading risk validator. Keep reason concise (under 20 words). Respond strictly in valid JSON format: {\"approved\": bool, \"confidence\": float (0.0~1.0), \"risk_level\": \"LOW\"|\"MEDIUM\"|\"HIGH\", \"reason\": string}"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 350
            }

            timeout_sec = (self.timeout_ms or int(os.getenv("JEV_TIMEOUT_MS", "2500"))) / 1000.0
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_sec)) as resp:
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                if resp.status == 200:
                    data = await resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                    
                    # Clean and resiliently parse JSON (repairs truncated strings if needed)
                    clean_content = content.replace("```json", "").replace("```", "").strip()
                    if "{" in clean_content:
                        clean_content = clean_content[clean_content.find("{"):]
                    try:
                        parsed = json.loads(clean_content)
                    except json.JSONDecodeError:
                        repaired = clean_content.rstrip()
                        if not repaired.endswith("}"):
                            if repaired.count('"') % 2 == 1:
                                repaired += '"'
                            repaired += "}"
                        parsed = json.loads(repaired)

                    confidence = float(parsed.get("confidence", 0.5))
                    approved = bool(parsed.get("approved", False)) and (confidence >= self.confidence_threshold)
                    risk_level = parsed.get("risk_level", "MEDIUM")
                    reason = parsed.get("reason", "Evaluated by Jev")

                    return JevGateDecision(
                        approved=approved,
                        confidence=confidence,
                        risk_level=risk_level,
                        reason=reason,
                        action="ENTER" if approved else "VETO",
                        latency_ms=elapsed_ms
                    )
                else:
                    logger.warning(f"⚠️ Jev API returned status {resp.status} for {symbol}. Fail-closed={self.fail_closed}.")
        except Exception as e:
            logger.warning(f"⚠️ Jev verification error ({symbol}): {e}. Fail-closed={self.fail_closed}.")

        # Fallback Handling (Fail-Closed by default to prevent blind leveraged entries on timeout/HTTP error)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        if self.fail_closed:
            return JevGateDecision(
                approved=False,
                confidence=0.0,
                risk_level="HIGH",
                reason="Fail-closed veto due to Jev API timeout/error",
                action="VETO",
                latency_ms=elapsed_ms,
                is_fallback=True
            )
        return JevGateDecision(
            approved=True,
            confidence=0.50,
            risk_level="MEDIUM",
            reason="Fallback baseline due to API timeout/error",
            action="ENTER",
            latency_ms=elapsed_ms,
            is_fallback=True
        )

    async def evaluate_kill_switch(
        self,
        asset_class: str,
        symbol: str,
        entry_price: float,
        current_price: float,
        unrealized_pnl_pct: float,
        context: Dict[str, Any]
    ) -> JevGateDecision:
        """
        Monitors open positions for emergency early liquidation before hitting hard SL.
        """
        start_t = time.perf_counter()
        if not self.is_enabled or not self.api_key:
            return JevGateDecision(approved=True, confidence=1.0, risk_level="LOW", reason="Disabled", action="HOLD", latency_ms=0.0)

        prompt = (
            f"Asset: {asset_class} | Symbol: {symbol}\n"
            f"Entry: {entry_price} | Current: {current_price} | PnL: {unrealized_pnl_pct:+.2f}%\n"
            f"Context: {context}\n"
            f"Question: Is there an immediate toxic liquidation wave, catastrophic breaking news, or structure breakdown requiring an EMERGENCY MARKET EXIT right now? (Yes/No)\n"
            f"Return JSON: {{\"exit_now\": bool, \"confidence\": float (0.0~1.0), \"reason\": string}}"
        )

        try:
            session = await self._get_session()
            url = "https://openrouter.ai/api/v1/chat/completions" if self.api_key.startswith("sk-or-") else "https://api.typesafe.ai/v1/systemone"
            payload = {
                "model": PINNED_MODEL_VERSION if not self.api_key.startswith("sk-or-") else "meta-llama/llama-3.2-3b-instruct",
                "messages": [
                    {"role": "system", "content": "You are a trading risk kill-switch. Return strictly JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 100
            }

            async with session.post(url, json=payload) as resp:
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                if resp.status == 200:
                    data = await resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                    clean_content = content.replace("```json", "").replace("```", "").strip()
                    parsed = json.loads(clean_content)

                    exit_now = bool(parsed.get("exit_now", False))
                    confidence = float(parsed.get("confidence", 0.5))

                    should_exit = exit_now and (confidence >= 0.70)
                    return JevGateDecision(
                        approved=not should_exit,
                        confidence=confidence,
                        risk_level="HIGH" if should_exit else "LOW",
                        reason=parsed.get("reason", "Kill-switch monitoring"),
                        action="EXIT_NOW" if should_exit else "HOLD",
                        latency_ms=elapsed_ms
                    )
        except Exception as e:
            logger.debug(f"Kill switch error: {e}")

        return JevGateDecision(approved=True, confidence=0.5, risk_level="LOW", reason="Hold fallback", action="HOLD", latency_ms=0.0)

    def evaluate_sync(
        self,
        asset_class: str,
        symbol: str,
        side: str,
        context_metrics: Dict[str, Any]
    ) -> JevGateDecision:
        """
        Synchronous evaluation wrapper for verify_entry (compatible with sync bot loops)
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    lambda: asyncio.run(self.verify_entry(asset_class, symbol, side, context_metrics))
                ).result()
        else:
            return loop.run_until_complete(
                self.verify_entry(asset_class, symbol, side, context_metrics)
            )

    def _build_prompt(self, asset_class: str, symbol: str, setup_name: str, context: Dict[str, Any]) -> str:
        if asset_class == "CRYPTO":
            if "wave_sync_breadth" in context:
                return (
                    f"Asset: Crypto Perpetual Swap ({symbol}) | Setup: 4-Major Dominance Synchronized 1H Wave ({setup_name})\n"
                    f"4-Major Synchronization (BTC/ETH/SOL/XRP): {context.get('wave_sync_breadth', '4/4')} aligned | "
                    f"BTC 1H MA20 3H-Slope: {context.get('btc_ma20_slope3_pct', 0.0):+.2f}%\n"
                    f"Price: ${context.get('price', 0):,.4f} | 1H Dist MA20: {context.get('dist_1h_ma20_pct', 0.0):+.2f}% | "
                    f"1H MA(5-20) Spread: {context.get('ma_spread_pct', 0.0):.2f}%\n"
                    f"1H ADX(14): {context.get('adx_1h', 0.0):.1f} | 15m ADX(14): {context.get('adx_15m', 0.0):.1f} | "
                    f"15m Vol(3/20): {context.get('vol_ratio', 1.0):.2f}x | 3H Return: {context.get('ret3_1h_pct', 0.0):+.2f}%\n"
                    f"Wave Age: BTC {context.get('btc_wave_age_h', 0)}h / {symbol} {context.get('sym_wave_age_h', 0)}h | "
                    f"Funding: {context.get('funding_rate_pct', 0.0):+.4f}% | SL: -{context.get('sl_pct', 2.0):.2f}% / Wave TP: +{context.get('tp_pct', 10.0):.1f}%\n"
                    f"Note: All 4 major crypto assets (BTC, ETH, SOL, XRP) have simultaneously aligned on confirmed 1H MA(5/10/20) and 15m EMA(9/20) early in the wave. "
                    f"Short-term L1 orderbook noise is secondary on 1H wave breakouts.\n"
                    f"Question: Is this synchronized 1H market-wide wave structurally sound for trend continuation (not an extreme funding squeeze or blow-off climax)?\n"
                    f"Approve if structural wave continuation probability >= 60%."
                )
            if "macro" in context or "dist_ema20_pct" in context:
                extra_lines = ""
                if "vol_ratio" in context or "dist_1h_ema50_pct" in context:
                    extra_lines = (
                        f"ADX(14): {context.get('adx', 0.0):.1f} | "
                        f"Volume Ratio (3/20): {context.get('vol_ratio', 1.0):.2f}x | "
                        f"1H Dist EMA50: {context.get('dist_1h_ema50_pct', 0.0):+.2f}% | "
                        f"1H RS vs BTC: {context.get('rs_vs_btc_1h_pct', 0.0):+.2f}% | "
                        f"Funding: {context.get('funding_rate_pct', 0.0):+.4f}%\n"
                    )
                return (
                    f"Asset: Crypto Swap ({symbol}) | Direction: {setup_name}\n"
                    f"Price: ${context.get('price', 0):,.4f} | Dist from EMA20: {context.get('dist_ema20_pct', 0):+.2f}%\n"
                    f"Macro Trend: {context.get('macro', 'N/A')} | LOB Imbalance: {context.get('imbalance', 0):+.2f}\n"
                    f"ATR Volatility: {context.get('atr_pct', 0):.2f}% | SL: -{context.get('sl_pct', 1.5):.1f}% | TP: +{context.get('tp_pct', 3.5):.1f}%\n"
                    f"Recent Return ({context.get('lookback_bars', 12)} bars): {context.get('recent_return_pct', 0):+.2f}%\n"
                    f"{extra_lines}"
                    f"Question: Does this {setup_name} trade have high continuation probability, or is it an overextended trap / exhaustion near support/resistance?\n"
                    f"Approve only if continuation probability >= 65%."
                )
            return (
                f"Asset: Crypto Swap ({symbol}) | Setup: {setup_name}\n"
                f"Surge: {context.get('surge_pct', 0):.2f}% | Retracement: {context.get('retrace_pct', 0):.2f}%\n"
                f"Volume Ratio: {context.get('vol_ratio', 1.0):.2f}x | Recent High-Low Range: {context.get('range_pct', 0):.2f}%\n"
                f"Question: Is this a genuine momentum continuation, or a fake breakout / liquidity grab (long trap) near local resistance?\n"
                f"Approve only if continuation probability >= 65%."
            )
        elif asset_class == "US_STOCKS":
            if "us_v4_pod" in context:
                pod = context.get("us_v4_pod", "BULL_DIP")
                bmk = context.get("benchmark", symbol)
                return (
                    f"Asset: US Equities/ETF ({symbol}) | Direction: {setup_name} | Strategy Pod: Argon US V4-Pro ({pod})\n"
                    f"Sector Benchmark ({bmk}) Daily Regime: {context.get('daily_regime', 'N/A')}\n"
                    f"Price: ${context.get('price', 0):,.2f} | Benchmark 1H RSI(2): {context.get('bmk_rsi2', 50.0):.1f} | "
                    f"Benchmark 1H Z-Score(20): {context.get('bmk_z20', 0.0):+.2f} | 1H RSI(14): {context.get('rsi14', 50.0):.1f}\n"
                    f"Cross-Sector Breadth: {context.get('sector_breadth', 'N/A')} | 7H Return: {context.get('ret7_pct', 0.0):+.2f}% | "
                    f"SL: -{context.get('sl_pct', 5.5):.1f}% / TP: +{context.get('tp_pct', 14.0):.1f}%\n"
                    f"Strategy Note: "
                    f"{'Connors RSI(2) institutional mean-reversion dip-buy inside a confirmed Daily Uptrend (low RSI2 and negative Z-score are the intended buy-low trigger, NOT a reason to veto).' if pod == 'BULL_DIP' else ('Institutional Bear-Regime relief-rally fade using 3x Inverse ETF or Index Native Short when an oversold downtrend index bounces into extreme 1H RSI(2) >= 85 overbought territory.' if pod == 'BEAR_RIP' else 'Synchronized 3+/4 US cross-sector 1H EMA(9<21<50) waterfall breakdown momentum.')}\n"
                    f"Question: Is this {pod} quantitative setup structurally valid to execute (no catastrophic data anomaly or extreme spread distortion)?\n"
                    f"Approve if structural validity probability >= 60%."
                )
            if "macro" in context or "dist_ema20_pct" in context:
                inv_note = ""
                if symbol in ("SQQQ", "SOXS", "SPXS", "PSQ"):
                    inv_note = (
                        f"Instrument Note: {symbol} is an INVERSE BEAR ETF that RISES when Nasdaq/Semiconductors fall. "
                        f"Going LONG {symbol} during a QQQ BEARISH macro trend is a trend-following trade aligned with the market selloff.\n"
                    )
                extra_eq = ""
                if "rsi" in context or "day_ret_pct" in context:
                    extra_eq = f"RSI(14): {context.get('rsi', 55.0):.1f} | Intraday Return from Open: {context.get('day_ret_pct', 0.0):+.2f}%\n"
                return (
                    f"Asset: US Equities ({symbol}) | Setup: Trend Continuation ({setup_name})\n"
                    f"{inv_note}"
                    f"Price: ${context.get('price', 0):,.2f} | Dist from EMA20: {context.get('dist_ema20_pct', 0):+.2f}%\n"
                    f"Macro Trend: {context.get('macro', 'N/A')} | L1 Imbalance: {context.get('imbalance', 0):+.2f}\n"
                    f"ATR Volatility: {context.get('atr_pct', 0):.2f}% | SL: -{context.get('sl_pct', 1.2):.1f}% | TP: +{context.get('tp_pct', 2.8):.1f}%\n"
                    f"Recent Return ({context.get('lookback_bars', 15)} bars): {context.get('recent_return_pct', 0):+.2f}%\n"
                    f"{extra_eq}"
                    f"Question: Is this a high-probability momentum continuation aligned with the macro regime, or an overextended climax trap?\n"
                    f"Approve only if continuation probability >= 65%."
                )
            surge = context.get('surge_pct') or context.get('gap_pct', 0)
            rvol = context.get('vol_ratio') or context.get('rvol', 1.0)
            retrace = context.get('retrace_pct') or context.get('retrace_ratio', 0)
            return (
                f"Asset: US Equities ({symbol}) | Setup: {setup_name}\n"
                f"Impulse Surge: {surge:.2f}% | Retracement: {retrace:.1f}%\n"
                f"Relative Vol / Contraction: {rvol:.2f}x\n"
                f"Catalyst/Context: {context.get('headline', 'Technical Momentum Pullback')} | Sector: {context.get('sector', 'SmallCap Momentum')}\n"
                f"Question: Is this setup a legitimate momentum continuation, or a fake liquidity grab / dilution pump-and-dump trap?\n"
                f"Approve only if clean continuation probability >= 60%."
            )
        else: # KR_STOCKS
            return (
                f"Asset: Korean KRX Stock ({symbol}) | Setup: {setup_name}\n"
                f"Surge: {context.get('surge_pct', 0):.2f}% | Volume: {context.get('vol_ratio', 1.0):.2f}x\n"
                f"Theme: {context.get('theme', 'General')} | Notice: {context.get('notice', 'None')}\n"
                f"Question: Is this a sustainable theme wave, or a terminal peak with impending CB/BW dump or warning?\n"
                f"Approve only if risk of dumping is below 40%."
            )
