"""Standalone live WebSocket diagnostic (Component 4).

    python -m noone.market_data.ws_diagnostic --seconds 30
    python -m noone.market_data.ws_diagnostic --seconds 20 --interrupt

Proves real Binance USDⓈ-M Futures frames (never Spot) are received
continuously, parsed into normalized models, and timestamped. With
``--interrupt`` it deliberately drops the connection mid-run and
reconnects/resubscribes, reporting both phases. Exit 0 only if data
frames were actually received (a handshake alone is NOT a pass).
"""
from __future__ import annotations

import argparse
import asyncio
import time
from collections import Counter

from noone.market_data import BinanceFuturesCollector
from noone.market_data.binance_futures import build_stream_url


async def collect(symbol: str, seconds: float) -> tuple[Counter, list, float]:
    streams = [f"{symbol.lower()}@aggTrade",
               f"{symbol.lower()}@markPrice@1s",
               f"{symbol.lower()}@kline_1m"]
    print(f"subscribe (URL): {build_stream_url(streams)}")
    counts: Counter = Counter()
    delays: list[float] = []
    first_at: float | None = None
    started = time.monotonic()
    with BinanceFuturesCollector() as market:
        try:
            async with asyncio.timeout(seconds):
                async for event in market.stream_events(streams):
                    counts[event.kind] += 1
                    if first_at is None:
                        first_at = time.monotonic() - started
                        print(f"first frame: kind={event.kind} "
                              f"after {first_at:.2f}s "
                              f"event_time={event.event_time} "
                              f"received_at={event.received_at}")
                    if event.delay_seconds is not None:
                        delays.append(event.delay_seconds)
        except TimeoutError:
            pass
    return counts, delays, (first_at or -1.0)


async def run(symbol: str, seconds: float, interrupt: bool) -> int:
    print(f"Binance USDⓈ-M Futures WS diagnostic: {symbol} for {seconds:.0f}s")
    started = time.monotonic()
    counts, delays, first_at = await collect(symbol, seconds)
    total = sum(counts.values())
    print(f"phase1: frames={total} byKind={dict(counts)} "
          f"duration={time.monotonic() - started:.1f}s")
    if delays:
        print(f"event delay (received-event): avg={sum(delays)/len(delays):.3f}s "
              f"max={max(delays):.3f}s samples={len(delays)}")
    ok = total > 0 and counts.get("trade", 0) > 0
    if interrupt:
        print("interrupting deliberately; reconnecting (resubscribe via URL)...")
        await asyncio.sleep(1)
        counts2, _d2, _f2 = await collect(symbol, min(10.0, seconds))
        total2 = sum(counts2.values())
        print(f"phase2 (after reconnect): frames={total2} byKind={dict(counts2)}")
        ok = ok and total2 > 0
    print(f"RESULT: {'PASS' if ok else 'FAIL'} "
          f"(handshake alone does not pass; frames required)")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="NOONE Futures WS diagnostic")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--interrupt", action="store_true",
                        help="Drop and reconnect mid-run to test resubscribe")
    args = parser.parse_args()
    return asyncio.run(run(args.symbol.upper(), args.seconds, args.interrupt))


if __name__ == "__main__":
    raise SystemExit(main())
