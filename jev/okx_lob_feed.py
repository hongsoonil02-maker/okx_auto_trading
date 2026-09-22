# -*- coding: utf-8 -*-
"""
okx_lob_feed.py — OKX Real-Time Order Book (LOB) WebSocket Feed & Ring Buffer
- Connects to OKX Public WebSocket (wss://ws.okx.com:8443/ws/v5/public)
- Subscribes to 'books5' (top 5 orderbook) or 'bbo-tbt' channels
- Maintains ultra-fast, lock-free in-memory cache of LOB snapshots
- Computes Orderbook Imbalance Ratio (OIR), Micro-price, and Spread Ticks
- Provides formatted text states for Jev AI System One prompts
"""
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import websockets

logger = logging.getLogger("OKX_LOB_Feed")


@dataclass
class LOBEntry:
    symbol: str  # e.g. "BTC-USDT-SWAP"
    timestamp: float  # local epoch timestamp
    exchange_ts: int  # OKX timestamp in ms
    bids: List[Tuple[float, float]] = field(default_factory=list)  # [(price, size), ...]
    asks: List[Tuple[float, float]] = field(default_factory=list)  # [(price, size), ...]
    best_bid: float = 0.0
    best_ask: float = 0.0
    best_bid_size: float = 0.0
    best_ask_size: float = 0.0
    spread: float = 0.0
    spread_bps: float = 0.0
    imbalance: float = 0.0  # -1.0 (all ask) to +1.0 (all bid)
    micro_price: float = 0.0

    @property
    def is_stale(self) -> bool:
        """Returns True if data is older than 3 seconds."""
        return (time.time() - self.timestamp) > 3.0

    def format_for_jev(self, timeframe: str = "10s", recent_note: str = "") -> str:
        """
        Formats LOB state into concise structured text for Jev System One prompt.
        """
        bids_summary = ", ".join([f"{p:.2f}:{s:.2f}" for p, s in self.bids[:5]])
        asks_summary = ", ".join([f"{p:.2f}:{s:.2f}" for p, s in self.asks[:5]])
        bias = "BID_HEAVY" if self.imbalance > 0.15 else ("ASK_HEAVY" if self.imbalance < -0.15 else "NEUTRAL")
        
        state_str = (
            f"Symbol: {self.symbol}\n"
            f"Best Bid: {self.best_bid:.4f} (Size: {self.best_bid_size:.3f}) | "
            f"Best Ask: {self.best_ask:.4f} (Size: {self.best_ask_size:.3f})\n"
            f"Spread: {self.spread:.4f} ({self.spread_bps:.2f} bps) | MicroPrice: {self.micro_price:.4f}\n"
            f"Top 5 Bids [Price:Size]: [{bids_summary}]\n"
            f"Top 5 Asks [Price:Size]: [{asks_summary}]\n"
            f"LOB Imbalance: {self.imbalance:+.3f} ({bias})\n"
        )
        if recent_note:
            state_str += f"Market Context: {recent_note}\n"
        return state_str


class OKXLOBFeed:
    """
    Asynchronous OKX Public WebSocket Order Book Client.
    Runs continuously in the background event loop.
    """
    DEFAULT_WS_URL = "wss://ws.okx.com/ws/v5/public"

    def __init__(self, ws_url: Optional[str] = None):
        self.ws_url = ws_url or os.getenv("OKX_WS_PUBLIC_URL", self.DEFAULT_WS_URL)
        self._subscribed_symbols: Set[str] = set()
        self._lob_cache: Dict[str, LOBEntry] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._ws = None

    def normalize_symbol(self, raw_symbol: str) -> str:
        """
        Normalizes various symbol formats ('BTC/USDT:USDT', 'BTC-USDT-SWAP', 'BTC/USDT')
        into OKX WebSocket instrument ID: 'BTC-USDT-SWAP'.
        """
        s = raw_symbol.upper().replace(":USDT", "").replace("/", "-")
        if not s.endswith("-SWAP") and not s.endswith("-SPOT"):
            s = f"{s}-SWAP"
        return s

    def subscribe(self, symbols: List[str]):
        """Register symbols to listen for."""
        for sym in symbols:
            norm = self.normalize_symbol(sym)
            self._subscribed_symbols.add(norm)
            if self._ws and self._ws.open:
                asyncio.create_task(self._send_subscription([norm]))

    def get_lob(self, symbol: str) -> Optional[LOBEntry]:
        """
        Returns latest LOBEntry for given symbol.
        Returns None if not found or stale (> 3s).
        """
        norm = self.normalize_symbol(symbol)
        entry = self._lob_cache.get(norm)
        if entry is None or entry.is_stale:
            return None
        return entry

    def _is_ws_connected(self) -> bool:
        if self._ws is None:
            return False
        try:
            # Modern websockets (14+) ClientConnection
            if hasattr(self._ws, "protocol") and hasattr(self._ws.protocol, "state"):
                from websockets.protocol import State
                return self._ws.protocol.state == State.OPEN
            # Legacy websockets (<=13)
            return getattr(self._ws, "open", False) or not getattr(self._ws, "closed", True)
        except Exception:
            return False

    async def _send_subscription(self, symbols: List[str]):
        if not self._is_ws_connected():
            return
        args = [{"channel": "books5", "instId": s} for s in symbols]
        msg = {"op": "subscribe", "args": args}
        try:
            await self._ws.send(json.dumps(msg))
            logger.info(f"📡 OKX LOB 구독 요청 완료: {symbols}")
        except Exception as e:
            logger.error(f"❌ 구독 메시지 전송 실패: {e}")

    async def start(self):
        """Starts the background WebSocket task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("🚀 OKX LOB WebSocket 피드 태스크 시작")

    async def stop(self):
        """Stops the background WebSocket task."""
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 OKX LOB WebSocket 피드 태스크 중지 완료")

    async def _run_loop(self):
        while self._running:
            try:
                logger.info(f"🔗 OKX WebSocket 연결 시도: {self.ws_url}")
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=None,  # OKX uses manual text ping/pong
                    ping_timeout=None,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws
                    logger.info("✅ OKX WebSocket 연결 성공!")

                    # Re-subscribe all registered symbols
                    if self._subscribed_symbols:
                        await self._send_subscription(list(self._subscribed_symbols))

                    # Ping task
                    ping_task = asyncio.create_task(self._ping_heartbeat(ws))

                    try:
                        async for raw_msg in ws:
                            if not self._running:
                                break
                            if raw_msg == "pong":
                                continue
                            self._handle_message(raw_msg)
                    finally:
                        ping_task.cancel()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"⚠️ OKX WebSocket 에러 발생: {e}. 3초 후 재연결...")
                await asyncio.sleep(3.0)

    async def _ping_heartbeat(self, ws):
        """OKX expects 'ping' every 20-25 seconds."""
        try:
            while self._running and self._is_ws_connected():
                await asyncio.sleep(20.0)
                await ws.send("ping")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Ping heartbeat 예외: {e}")

    def _handle_message(self, raw_msg: str):
        try:
            msg = json.loads(raw_msg)
        except Exception:
            return

        if "event" in msg:
            if msg.get("event") == "error":
                logger.error(f"❌ OKX WS 오류 이벤트: {msg}")
            return

        arg = msg.get("arg", {})
        if arg.get("channel") != "books5":
            return

        inst_id = arg.get("instId")
        data_list = msg.get("data", [])
        if not inst_id or not data_list:
            return

        data = data_list[0]
        raw_bids = data.get("bids", [])
        raw_asks = data.get("asks", [])
        if not raw_bids or not raw_asks:
            return

        bids = [(float(b[0]), float(b[1])) for b in raw_bids[:10]]
        asks = [(float(a[0]), float(a[1])) for a in raw_asks[:10]]

        best_bid, best_bid_size = bids[0]
        best_ask, best_ask_size = asks[0]
        spread = max(0.0, best_ask - best_bid)
        mid = (best_ask + best_bid) / 2.0 if (best_ask + best_bid) > 0 else 1.0
        spread_bps = (spread / mid) * 10000.0

        # Calculate volume imbalance over top 5 levels
        top_bid_vol = sum(size for _, size in bids[:5])
        top_ask_vol = sum(size for _, size in asks[:5])
        total_vol = top_bid_vol + top_ask_vol
        imbalance = (top_bid_vol - top_ask_vol) / total_vol if total_vol > 0 else 0.0

        # Micro-price calculation: weighted by top-of-book volumes
        top_denom = best_bid_size + best_ask_size
        if top_denom > 0:
            micro_price = (best_bid * best_ask_size + best_ask * best_bid_size) / top_denom
        else:
            micro_price = mid

        ts_now = time.time()
        ex_ts = int(data.get("ts", int(ts_now * 1000)))

        self._lob_cache[inst_id] = LOBEntry(
            symbol=inst_id,
            timestamp=ts_now,
            exchange_ts=ex_ts,
            bids=bids,
            asks=asks,
            best_bid=best_bid,
            best_ask=best_ask,
            best_bid_size=best_bid_size,
            best_ask_size=best_ask_size,
            spread=spread,
            spread_bps=spread_bps,
            imbalance=imbalance,
            micro_price=micro_price,
        )
