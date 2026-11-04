"""Live WebSocket tests — real Binance USDⓈ-M Futures frames only.

Skipped unless NOONE_LIVE=1. Mocked frames (tests/test_ws.py) never
substitute for these checks. A handshake alone is not a pass: real
trade/mark/kline frames must arrive, parse, and keep arriving.

Run: $env:NOONE_LIVE=1; python -m pytest tests/test_live_ws.py -v
"""
import asyncio
import os
from collections import Counter

import pytest

from noone.market_data import BinanceFuturesCollector

pytestmark = pytest.mark.skipif(
    os.environ.get("NOONE_LIVE") != "1", reason="live WS tests need NOONE_LIVE=1")

STREAMS = ["btcusdt@aggTrade", "btcusdt@markPrice@1s", "btcusdt@kline_1m"]


async def _collect(seconds: float) -> list:
    events = []
    with BinanceFuturesCollector() as market:
        try:
            async with asyncio.timeout(seconds):
                async for event in market.stream_events(STREAMS):
                    events.append(event)
        except TimeoutError:
            pass
    return events


def test_live_ws_continuous_frames_all_kinds():
    events = asyncio.run(_collect(20))
    kinds = Counter(e.kind for e in events)
    assert kinds["trade"] > 5, f"too few trades: {dict(kinds)}"
    assert kinds["mark_price"] >= 3, f"too few mark prices: {dict(kinds)}"
    assert kinds["kline"] >= 3, f"too few klines: {dict(kinds)}"
    for event in events:
        assert event.source == "websocket" and event.symbol == "BTCUSDT"
        assert event.event_time is not None and event.received_at >= event.event_time
        assert event.delay_seconds is not None and event.delay_seconds < 30


def test_live_ws_reconnect_resubscribes():
    first = asyncio.run(_collect(8))
    assert first, "no frames before deliberate interruption"
    second = asyncio.run(_collect(8))  # fresh connect = resubscription
    assert second, "no frames after reconnect/resubscribe"
    assert any(e.kind == "trade" for e in second)
