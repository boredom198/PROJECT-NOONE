"""Binance USDⓈ-M Futures public market-data collector (Component 4).

REST base: ``https://fapi.binance.com`` (``/fapi/v1/...``)
WS base:   ``wss://fstream.binance.com``

This is the USDⓈ-M **Futures** API family only. Spot endpoints
(``/api/v3/...``) are never used. All endpoints here are public and
need no API key. No account access, no order placement.

Failure policy: HTTP/transport/parse failures raise
:class:`MarketDataError`. Missing fields normalize to ``None``
(unavailable), never to fabricated values or Spot substitutes.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from typing import Any, AsyncIterator

import httpx

from noone.market_data.models import (
    BoundaryProbeLimitHit,
    DepthSnapshot,
    ExchangeSymbol,
    FundingBlindSpot,
    FundingConflict,
    FundingContext,
    FundingMalformedOccurrence,
    FundingHistory,
    FundingMalformedRecord,
    FundingRatePoint,
    FundingSettlement,
    Kline,
    KlineHistory,
    MarketEvent,
    MarkPrice,
    OpenInterest,
    Ticker24h,
    Trade,
    _f,
    anomalies_from_assessments,
    assess_funding_intervals,
    interval_ms,
    ms_to_dt,
    utcnow,
    validate_klines,
)

REST_BASE = "https://fapi.binance.com"
# Binance retired the legacy wss://fstream.binance.com/ws and /stream
# paths on 2026-04-23: they handshake and ACK subscriptions but
# silently deliver zero market frames (verified live, see README).
# Current routing: /market (kline, aggTrade, markPrice, tickers),
# /public (high-frequency public), /private (user data).
WS_MARKET_BASE = "wss://fstream.binance.com/market"
WS_PUBLIC_BASE = "wss://fstream.binance.com/public"
WS_BASE = WS_MARKET_BASE  # backwards-compatible alias (market data)


class MarketDataError(RuntimeError):
    """Raised when Futures market data cannot be retrieved/parsed."""

    def __init__(self, message: str, *, status: int | None = None, endpoint: str = ""):
        super().__init__(message)
        self.status = status
        self.endpoint = endpoint


# --- Pure parsers (unit-testable without network) ---

def parse_klines(symbol: str, interval: str, payload: Any) -> list[Kline]:
    if not isinstance(payload, list):
        raise MarketDataError(f"Unexpected klines payload: {type(payload).__name__}")
    now = utcnow()
    out: list[Kline] = []
    for row in payload:
        if not isinstance(row, (list, tuple)) or len(row) < 8:
            continue  # skip malformed candle, keep the rest
        close_time = ms_to_dt(row[6])
        out.append(Kline(
            symbol=symbol, interval=interval,
            open_time=ms_to_dt(row[0]), close_time=close_time,
            open=_f(row[1]), high=_f(row[2]), low=_f(row[3]), close=_f(row[4]),
            volume=_f(row[5]), quote_volume=_f(row[7]),
            trades=int(row[8]) if len(row) > 8 and row[8] is not None else None,
            taker_buy_volume=_f(row[9]) if len(row) > 9 else None,
            taker_buy_quote_volume=_f(row[10]) if len(row) > 10 else None,
            is_closed=close_time is not None and close_time <= now,
            raw=row,
        ))
    return out


def parse_ticker_24h(payload: Any) -> Ticker24h:
    if not isinstance(payload, dict) or "symbol" not in payload:
        raise MarketDataError("Unexpected ticker/24hr payload (missing symbol)")
    return Ticker24h(
        symbol=str(payload["symbol"]),
        last_price=_f(payload.get("lastPrice")),
        price_change=_f(payload.get("priceChange")),
        price_change_percent=_f(payload.get("priceChangePercent")),
        high_price=_f(payload.get("highPrice")),
        low_price=_f(payload.get("lowPrice")),
        volume=_f(payload.get("volume")),
        quote_volume=_f(payload.get("quoteVolume")),
        open_time=ms_to_dt(payload.get("openTime")),
        close_time=ms_to_dt(payload.get("closeTime")),
        raw=payload,
    )


def parse_premium_index(payload: Any) -> MarkPrice:
    if not isinstance(payload, dict) or "symbol" not in payload:
        raise MarketDataError("Unexpected premiumIndex payload (missing symbol)")
    return MarkPrice(
        symbol=str(payload["symbol"]),
        mark_price=_f(payload.get("markPrice")),
        index_price=_f(payload.get("indexPrice")),
        estimated_settle_price=_f(payload.get("estimatedSettlePrice")),
        last_funding_rate=_f(payload.get("lastFundingRate")),
        next_funding_time=ms_to_dt(payload.get("nextFundingTime")),
        event_time=ms_to_dt(payload.get("time")),
        raw=payload,
    )


def _canonical_payload(value: Any):
    """Canonical, hashable identity for a raw JSON-compatible payload.

    Dictionary key order is normalized; value types are tagged so
    ``1``, ``1.0``, ``"1"`` and ``True`` never merge. Anything that
    cannot be canonicalized safely gets a unique identity and is
    therefore never merged with another record.
    """
    if isinstance(value, dict):
        items = sorted(
            ((_canonical_payload(k), _canonical_payload(v))
             for k, v in value.items()),
            key=repr)
        return ("dict", tuple(items))
    if isinstance(value, (list, tuple)):
        return ("seq", tuple(_canonical_payload(v) for v in value))
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", repr(value))
    if isinstance(value, str):
        return ("str", value)
    if value is None:
        return ("none",)
    if isinstance(value, (bytes, bytearray)):
        return ("bytes", bytes(value))
    from decimal import Decimal

    if isinstance(value, Decimal):
        return ("decimal", str(value))
    return ("unique", object())


def _decimal(value: Any):
    """Exact Decimal from a Binance decimal string; None if invalid."""
    from decimal import Decimal, InvalidOperation

    if value is None or value == "":
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def parse_funding_settlements(
    symbol: str, payload: Any,
) -> tuple[list[FundingSettlement], list[FundingMalformedRecord]]:
    """Decimal-preserving funding parser with separate malformed list.

    A record is malformed (excluded from settlements, reported
    separately) when its symbol mismatches, its funding time is
    missing/invalid, its rate is missing/invalid, or its settlement
    mark price is missing, invalid, or non-positive. A genuine zero
    rate stays zero and distinct from missing.
    """
    if not isinstance(payload, list):
        raise MarketDataError("Unexpected fundingRate payload")
    settlements: list[FundingSettlement] = []
    malformed: list[FundingMalformedRecord] = []
    for record_index, record in enumerate(payload):
        if not isinstance(record, dict):
            malformed.append(FundingMalformedRecord(
                symbol=symbol, funding_time=None,
                reason="record_not_object", raw=record,
                record_index=record_index))
            continue
        try:
            funding_time = ms_to_dt(int(record.get("fundingTime")))
        except (TypeError, ValueError):
            funding_time = None
        record_symbol = str(record.get("symbol", "") or "")
        if record_symbol and record_symbol != symbol:
            malformed.append(FundingMalformedRecord(
                symbol=symbol, funding_time=funding_time,
                reason="symbol_mismatch", raw=record,
                record_symbol=record_symbol, record_index=record_index))
            continue
        if funding_time is None:
            malformed.append(FundingMalformedRecord(
                symbol=symbol, funding_time=None,
                reason="missing_or_invalid_funding_time", raw=record,
                record_symbol=record_symbol or None,
                record_index=record_index))
            continue
        rate = _decimal(record.get("fundingRate"))
        if rate is None:
            malformed.append(FundingMalformedRecord(
                symbol=symbol, funding_time=funding_time,
                reason="missing_or_invalid_funding_rate", raw=record,
                record_symbol=record_symbol or None,
                record_index=record_index))
            continue
        mark = _decimal(record.get("markPrice"))
        if mark is None or mark <= 0:
            malformed.append(FundingMalformedRecord(
                symbol=symbol, funding_time=funding_time,
                reason="missing_or_invalid_mark_price", raw=record,
                record_symbol=record_symbol or None,
                record_index=record_index))
            continue
        rate_type_raw = record.get("rateType")
        settlements.append(FundingSettlement(
            symbol=symbol, funding_time=funding_time, funding_rate=rate,
            mark_price=mark,
            rate_type=(str(rate_type_raw) if rate_type_raw not in (None, "")
                       else None),
            raw=record))
    return settlements, malformed


def parse_funding_history(payload: Any) -> list[FundingRatePoint]:
    if not isinstance(payload, list):
        raise MarketDataError("Unexpected fundingRate payload")
    return [FundingRatePoint(
        symbol=str(p.get("symbol", "")),
        funding_rate=_f(p.get("fundingRate")),
        funding_time=ms_to_dt(p.get("fundingTime")),
        mark_price=_f(p.get("markPrice")),
        raw=p,
    ) for p in payload if isinstance(p, dict)]


def parse_open_interest(symbol: str, payload: Any) -> OpenInterest:
    if not isinstance(payload, dict):
        raise MarketDataError("Unexpected openInterest payload")
    return OpenInterest(
        symbol=str(payload.get("symbol", symbol)),
        open_interest=_f(payload.get("openInterest")),
        event_time=ms_to_dt(payload.get("time")),
        raw=payload,
    )


def parse_depth(symbol: str, payload: Any) -> DepthSnapshot:
    if not isinstance(payload, dict):
        raise MarketDataError("Unexpected depth payload")
    def side(rows: Any) -> tuple[tuple[float | None, float | None], ...]:
        if not isinstance(rows, list):
            return ()
        return tuple((_f(r[0]), _f(r[1])) for r in rows
                     if isinstance(r, (list, tuple)) and len(r) >= 2)
    return DepthSnapshot(
        symbol=symbol, bids=side(payload.get("bids")), asks=side(payload.get("asks")),
        last_update_id=payload.get("lastUpdateId"),
        event_time=ms_to_dt(payload.get("T") or payload.get("E")),
        raw=payload,
    )


def parse_exchange_info(payload: Any) -> list[ExchangeSymbol]:
    if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
        raise MarketDataError("Unexpected exchangeInfo payload")
    return [ExchangeSymbol(
        symbol=str(s.get("symbol", "")), pair=s.get("pair"),
        contract_type=s.get("contractType"), status=s.get("status"),
        base_asset=s.get("baseAsset"), quote_asset=s.get("quoteAsset"),
        margin_asset=s.get("marginAsset"), raw=s,
    ) for s in payload["symbols"] if isinstance(s, dict)]


def build_stream_url(streams: list[str], *, category: str = "market") -> str:
    """Combined-stream URL on the current Binance routing.

    Streams are lowercased and de-duplicated (order preserved) so a
    stream is never subscribed twice. ``category`` is "market"
    (kline/aggTrade/markPrice/tickers) or "public" (high-frequency).
    """
    base = WS_PUBLIC_BASE if category == "public" else WS_MARKET_BASE

    def normalize(name: str) -> str:
        # Symbol lowercased; stream type case preserved — Binance
        # stream names are case-sensitive (aggTrade, markPrice).
        parts = name.strip().split("@")
        return "@".join([parts[0].lower(), *parts[1:]])

    unique = list(dict.fromkeys(
        normalize(s) for s in streams if s.strip()))
    if not unique:
        raise ValueError("At least one stream is required")
    return f"{base}/stream?streams={'/'.join(unique)}"


def parse_ws_message(message: Any) -> MarketEvent | None:
    """Normalize one decoded WS frame; None for control/malformed.

    Handles combined envelopes (``{"stream": ..., "data": {...}}``)
    and single-stream payloads. Supported kinds: ``trade``
    (aggTrade/trade), ``mark_price`` (markPriceUpdate), ``kline``.
    Unknown event types are preserved as raw MarketEvents. Exchange
    event time (``E``) and local receive time are both kept.
    """
    if not isinstance(message, dict):
        return None
    if "result" in message and "id" in message:
        return None  # subscription ACK / control response
    stream = str(message.get("stream", ""))
    data = message.get("data", message)
    if not isinstance(data, dict):
        return None
    kind_raw = str(data.get("e", ""))
    symbol = str(data.get("s", ""))
    event_time = ms_to_dt(data.get("E"))
    received = utcnow()
    if not stream and symbol:
        stream = f"{symbol.lower()}@{kind_raw}"
    if kind_raw in ("aggTrade", "trade"):
        trade = Trade(
            symbol=symbol,
            trade_id=data.get("a", data.get("t")),
            price=_f(data.get("p")), quantity=_f(data.get("q")),
            event_time=event_time, trade_time=ms_to_dt(data.get("T")),
            is_buyer_maker=data.get("m"), received_at=received,
            raw=data)
        return MarketEvent("trade", symbol, event_time, received,
                           stream, trade, raw=message)
    if kind_raw == "markPriceUpdate":
        mark = MarkPrice(
            symbol=symbol, mark_price=_f(data.get("p")),
            index_price=_f(data.get("i")),
            estimated_settle_price=_f(data.get("P")),
            last_funding_rate=_f(data.get("r")),
            next_funding_time=ms_to_dt(data.get("T")),
            event_time=event_time, source="websocket",
            received_at=received, raw=data)
        return MarketEvent("mark_price", symbol, event_time, received,
                           stream, mark, raw=message)
    if kind_raw == "kline" and isinstance(data.get("k"), dict):
        k = data["k"]
        kline = Kline(
            symbol=symbol, interval=str(k.get("i", "")),
            open_time=ms_to_dt(k.get("t")), close_time=ms_to_dt(k.get("T")),
            open=_f(k.get("o")), high=_f(k.get("h")), low=_f(k.get("l")),
            close=_f(k.get("c")), volume=_f(k.get("v")),
            quote_volume=_f(k.get("q")),
            trades=int(k["n"]) if k.get("n") is not None else None,
            taker_buy_volume=_f(k.get("V")),
            taker_buy_quote_volume=_f(k.get("Q")),
            is_closed=bool(k.get("x", False)), source="websocket",
            received_at=received, raw=data)
        return MarketEvent("kline", symbol, event_time, received,
                           stream, kline, raw=message)
    if not kind_raw:
        return None
    return MarketEvent(kind_raw, symbol, event_time, received,
                       stream, data, raw=message)


class BinanceFuturesCollector:
    """Polling REST collector + WebSocket stream for USDⓈ-M Futures."""

    def __init__(self, *, timeout_s: float = 15.0, max_retries: int = 3):
        self._client = httpx.Client(
            base_url=REST_BASE, timeout=timeout_s,
            headers={"User-Agent": "NOONE/0.1 (Binance USD-M Futures public data)"},
        )
        self._max_retries = max(1, max_retries)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "BinanceFuturesCollector":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.get(endpoint, params=params)
            except httpx.HTTPError as exc:
                last_exc = exc
                time.sleep(min(2 ** attempt, 8))
                continue
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise MarketDataError(
                        f"Invalid JSON from {endpoint}", endpoint=endpoint) from exc
            if resp.status_code in (418, 429):  # rate limited / IP ban
                wait = float(resp.headers.get("Retry-After", 2 ** attempt))
                time.sleep(min(wait, 30))
                last_exc = MarketDataError(
                    f"Rate limited by Binance ({resp.status_code})",
                    status=resp.status_code, endpoint=endpoint)
                continue
            raise MarketDataError(
                f"Binance Futures {endpoint} -> HTTP {resp.status_code}: "
                f"{resp.text[:200]}", status=resp.status_code, endpoint=endpoint)
        raise MarketDataError(
            f"Binance Futures {endpoint} failed after retries: {last_exc}",
            endpoint=endpoint) from last_exc

    # -- REST endpoints (all /fapi/v1, public, keyless) --
    def ping(self) -> bool:
        self._get("/fapi/v1/ping")
        return True

    def server_time_ms(self) -> int:
        data = self._get("/fapi/v1/time")
        return int(data["serverTime"])

    def exchange_info(self) -> list[ExchangeSymbol]:
        return parse_exchange_info(self._get("/fapi/v1/exchangeInfo"))

    def klines(self, symbol: str, interval: str = "1m", limit: int = 100) -> list[Kline]:
        interval_ms(interval)  # validate against supported intervals
        return parse_klines(symbol, interval, self._get(
            "/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit}))

    def historical_klines(
        self, symbol: str, *, interval: str = "1h",
        start_time: Any = None, end_time: Any = None,
        total_limit: int = 1000, page_limit: int = 1500,
        only_closed: bool = False,
    ) -> KlineHistory:
        """Paginated historical klines from USDⓈ-M Futures only.

        With ``start_time``, pages forward from that time using
        ``startTime`` cursors (``last_open + interval``) until
        ``total_limit``, ``end_time``, or a short/empty page ends the
        range. Without it, Binance returns only the most recent
        ``limit`` candles, so pages walk *backwards*: the first page
        is the most recent chunk, then ``endTime`` is set just before
        that chunk's earliest candle until enough history is
        collected. The cursor therefore only moves backwards, never
        into future timestamps, and no ``startTime`` is invented.

        Result is chronologically sorted, de-duplicated, and
        gap-checked by ``validate_klines``; gaps are reported, never
        filled. ``start_time``/``end_time`` may be UTC-aware datetimes
        or millisecond epochs. With ``only_closed=True`` the
        still-forming candle is dropped; in backward mode one extra
        older candle takes its place so ``total_limit`` closed
        candles are returned when enough history exists. Pagination
        stops on an empty page, a short page, or a page that adds no
        new candle (repeated-page protection). Page size never
        exceeds the exchange maximum of 1500. Rate limits/retries are
        inherited from ``_get()``.
        """
        from datetime import datetime, timezone

        step = interval_ms(interval)
        page_limit = max(1, min(page_limit, 1500))

        def to_ms(value: Any) -> int | None:
            if value is None:
                return None
            if isinstance(value, datetime):
                aware = value if value.tzinfo else value.replace(
                    tzinfo=timezone.utc)
                return int(aware.timestamp() * 1000)
            return int(value)

        def counts(candle: Kline) -> bool:
            # Unique candles counting toward total_limit.
            return candle.is_closed if only_closed else True

        start_ms, end_ms = to_ms(start_time), to_ms(end_time)
        collected: list[Kline] = []
        seen: set[int] = set()  # open_time ms of collected candles
        counted = 0  # unique candles toward total_limit so far
        pages = 0
        if start_ms is not None:
            cursor: int | None = start_ms
            while counted < total_limit:
                want = min(page_limit, total_limit - counted)
                params: dict[str, Any] = {
                    "symbol": symbol, "interval": interval, "limit": want,
                    "startTime": cursor}
                if end_ms is not None:
                    params["endTime"] = end_ms
                page = parse_klines(symbol, interval, self._get(
                    "/fapi/v1/klines", params))
                pages += 1
                if not page:
                    break
                fresh = [c for c in page if c.open_time is None
                         or int(c.open_time.timestamp() * 1000) not in seen]
                seen.update(int(c.open_time.timestamp() * 1000)
                            for c in fresh if c.open_time is not None)
                counted += sum(1 for c in fresh if counts(c))
                collected.extend(page)
                if not fresh or len(page) < want:
                    break
                last_open = page[-1].open_time
                if last_open is None:
                    break
                next_cursor = int(last_open.timestamp() * 1000) + step
                if cursor is not None and next_cursor <= cursor:
                    break  # cursor must strictly advance
                cursor = next_cursor
                if end_ms is not None and cursor > end_ms:
                    break
        else:
            end_cursor = end_ms  # None means "most recent available"
            while counted < total_limit:
                want = min(page_limit, total_limit - counted)
                params = {
                    "symbol": symbol, "interval": interval, "limit": want}
                if end_cursor is not None:
                    params["endTime"] = end_cursor
                page = parse_klines(symbol, interval, self._get(
                    "/fapi/v1/klines", params))
                pages += 1
                if not page:
                    break
                fresh = [c for c in page if c.open_time is None
                         or int(c.open_time.timestamp() * 1000) not in seen]
                seen.update(int(c.open_time.timestamp() * 1000)
                            for c in fresh if c.open_time is not None)
                counted += sum(1 for c in fresh if counts(c))
                collected.extend(page)
                if not fresh or len(page) < want:
                    break
                opens = [c.open_time for c in page if c.open_time is not None]
                if not opens:
                    break
                # Walk strictly before this chunk's earliest candle;
                # a page adding no new candle breaks the loop above.
                end_cursor = int(min(opens).timestamp() * 1000) - 1
        clean, gaps, duplicates = validate_klines(collected, interval)
        if start_ms is not None:
            clean = [c for c in clean if c.open_time is None
                     or c.open_time.timestamp() * 1000 >= start_ms]
        if end_ms is not None:
            clean = [c for c in clean if c.open_time is None
                     or c.open_time.timestamp() * 1000 <= end_ms]
        if only_closed:
            clean = [c for c in clean if c.is_closed]
        return KlineHistory(
            symbol=symbol, interval=interval, candles=tuple(clean),
            gaps=tuple(gaps), duplicates_removed=duplicates,
            pages_fetched=pages,
            start_time=ms_to_dt(start_ms), end_time=ms_to_dt(end_ms))

    def historical_klines_multi(
        self, symbols: list[str], **kwargs: Any,
    ) -> dict[str, KlineHistory]:
        """Historical klines for multiple perpetual symbols."""
        return {s: self.historical_klines(s, **kwargs) for s in symbols}

    def ticker_24h(self, symbol: str) -> Ticker24h:
        return parse_ticker_24h(self._get("/fapi/v1/ticker/24hr", {"symbol": symbol}))

    def mark_price(self, symbol: str) -> MarkPrice:
        return parse_premium_index(self._get("/fapi/v1/premiumIndex", {"symbol": symbol}))

    def funding_history(self, symbol: str, limit: int = 10) -> list[FundingRatePoint]:
        return parse_funding_history(self._get(
            "/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit}))

    def historical_funding_rates(
        self, symbol: str, *, start_time: Any = None, end_time: Any = None,
        total_limit: int = 1000, page_limit: int = 1000,
        context_limit: int = 0, context_max_pages: int = 10,
        _max_pages: int | None = None,
    ) -> FundingHistory:
        """Paginated historical funding settlements (USDⓈ-M Futures).

        With ``start_time``, pages forward from it; without it, pages
        walk backwards from the most recent records. Both cursors are
        *inclusive* of the boundary timestamp (``last_time`` forward,
        ``earliest_time`` backward) because distinct settlements can
        share one ``fundingTime`` (distinguished by ``rateType``) and
        a page boundary can split them: a ``+1ms``/``-1ms`` cursor
        would skip the unseen same-timestamp event. The boundary is
        re-requested and de-duplicated, and is only passed after a
        full-timestamp probe (``startTime == endTime == boundary``)
        confirms no unseen same-timestamp event remains. Results are chronological and
        de-duplicated by settlement identity; exact duplicates are
        counted, conflicting duplicates are reported separately, and
        malformed records never enter the settlement list. Suspected
        interval anomalies are detected against a locally established
        interval only (never an assumed schedule). Page size never
        exceeds Binance's 1000-record maximum. API failures raise
        ``MarketDataError`` via ``_get``; they never become an empty
        successful history.

        With ``context_limit > 0``, up to that many settlements
        immediately before ``start_time`` and immediately after
        ``end_time`` are also collected (bounded by
        ``context_max_pages`` requests per side) through the same
        paginated mechanism. Context is stored separately on the
        returned history, never merged into the requested
        settlements or coverage, and only widens the evidence the
        two-sided interval detector can use. A context failure
        preserves the requested data and is reported as a scoped
        blind spot instead of an empty success.
        """
        from datetime import datetime, timezone

        page_limit = max(1, min(page_limit, 1000))

        def to_ms(value: Any) -> int | None:
            if value is None:
                return None
            if isinstance(value, datetime):
                aware = value if value.tzinfo else value.replace(
                    tzinfo=timezone.utc)
                return int(aware.timestamp() * 1000)
            return int(value)

        start_ms, end_ms = to_ms(start_time), to_ms(end_time)
        forward = start_ms is not None
        by_identity: dict[tuple[int, str], FundingSettlement] = {}
        conflicts: list[FundingConflict] = []
        malformed: list[FundingMalformedRecord] = []
        malformed_index: dict[tuple, int] = {}  # exact-payload identity
        duplicates = 0
        pages = 0
        termination = "total_limit_reached"
        cursor = start_ms if forward else end_ms

        def in_range(ms: int) -> bool:
            if start_ms is not None and ms < start_ms:
                return False
            if end_ms is not None and ms > end_ms:
                return False
            return True

        def add_malformed(
            records: list[FundingMalformedRecord],
            request_params: dict[str, Any],
        ) -> None:
            """Stamp page provenance; dedup on exact raw payload only.

            Distinct payloads are never merged (even with identical
            reason/timestamp). An exact repeat (boundary overlap)
            keeps the first provenance and appends the new occurrence.
            """
            for record in records:
                occurrence = FundingMalformedOccurrence(
                    page_index=pages - 1,
                    page_start_ms=request_params.get("startTime"),
                    page_end_ms=request_params.get("endTime"),
                    record_index=record.record_index)
                key = (record.reason, _canonical_payload(record.raw))
                if key in malformed_index:
                    idx = malformed_index[key]
                    existing = malformed[idx]
                    malformed[idx] = replace(
                        existing,
                        occurrence_count=existing.occurrence_count + 1,
                        occurrences=existing.occurrences + (occurrence,))
                else:
                    malformed_index[key] = len(malformed)
                    malformed.append(replace(
                        record, page_index=occurrence.page_index,
                        page_start_ms=occurrence.page_start_ms,
                        page_end_ms=occurrence.page_end_ms,
                        occurrences=(occurrence,)))

        def ingest(page: list[FundingSettlement]) -> int:
            nonlocal duplicates
            fresh_count = 0
            for settlement in page:
                assert settlement.funding_time is not None
                ms = int(settlement.funding_time.timestamp() * 1000)
                if not in_range(ms):
                    continue
                identity = (ms, settlement.rate_type or "")
                kept = by_identity.get(identity)
                if kept is None:
                    by_identity[identity] = settlement
                    fresh_count += 1
                elif (kept.funding_rate == settlement.funding_rate
                        and kept.mark_price == settlement.mark_price):
                    duplicates += 1
                else:
                    conflicts.append(FundingConflict(
                        symbol=symbol, funding_time=settlement.funding_time,
                        rate_type=settlement.rate_type,
                        kept_rate=kept.funding_rate,
                        kept_mark_price=kept.mark_price,
                        conflicting_rate=settlement.funding_rate,
                        conflicting_mark_price=settlement.mark_price,
                        raw=settlement.raw))
            return fresh_count

        advanced_boundaries: set[int] = set()
        probe_limit_hits: list[BoundaryProbeLimitHit] = []

        while len(by_identity) < total_limit:
            want = min(page_limit, total_limit - len(by_identity))
            params: dict[str, Any] = {
                "symbol": symbol, "limit": want}
            if forward:
                params["startTime"] = cursor
            elif cursor is not None:
                params["endTime"] = cursor
            if end_ms is not None and forward:
                params["endTime"] = end_ms
            raw_page = self._get("/fapi/v1/fundingRate", params)
            pages += 1
            if not isinstance(raw_page, list):
                raise MarketDataError("Unexpected fundingRate payload")
            if not raw_page:
                termination = "empty_page"
                break
            page, page_malformed = parse_funding_settlements(symbol, raw_page)
            add_malformed(page_malformed, params)
            page_times = [int(s.funding_time.timestamp() * 1000)  # type: ignore[union-attr]
                          for s in page]
            page_times += [int(r.funding_time.timestamp() * 1000)
                           for r in page_malformed
                           if r.funding_time is not None]
            fresh = ingest(page)
            if len(by_identity) >= total_limit:
                termination = "total_limit_reached"
                break
            if _max_pages is not None and pages >= _max_pages:
                termination = "safety_limit"
                break
            if len(raw_page) < want:
                termination = ("range_exhausted"
                               if forward and end_ms is not None
                               else "short_page")
                break
            if not page_times:
                # No usable timestamp anywhere on a full page: cursor
                # progress cannot be established safely.
                termination = "no_usable_timestamp"
                break
            boundary = max(page_times) if forward else min(page_times)
            if fresh == 0:
                if (all(t == boundary for t in page_times)
                        and boundary not in advanced_boundaries):
                    # Probe the whole boundary timestamp before
                    # passing it, so a same-timestamp event split
                    # across the page boundary is never skipped.
                    probe_params = {
                        "symbol": symbol, "startTime": boundary,
                        "endTime": boundary, "limit": 1000}
                    probe_raw = self._get("/fapi/v1/fundingRate", probe_params)
                    pages += 1
                    if not isinstance(probe_raw, list):
                        raise MarketDataError(
                            "Unexpected fundingRate payload")
                    if len(probe_raw) == 1000:
                        # Probe hit the endpoint maximum: further
                        # records may exist at this timestamp and no
                        # documented cursor can reach them.
                        probe_limit_hits.append(BoundaryProbeLimitHit(
                            funding_time=ms_to_dt(boundary),
                            returned_count=len(probe_raw)))
                    probe, probe_malformed = parse_funding_settlements(
                        symbol, probe_raw)
                    add_malformed(probe_malformed, probe_params)
                    ingest(probe)
                    advanced_boundaries.add(boundary)
                    if len(by_identity) >= total_limit:
                        termination = "total_limit_reached"
                        break
                    cursor = boundary + 1 if forward else boundary - 1
                    if forward and end_ms is not None and cursor > end_ms:
                        termination = "range_exhausted"
                        break
                    continue
                termination = "repeated_page"
                break
            # Inclusive boundary cursor: re-request the boundary
            # timestamp next time (de-duplicated by identity).
            cursor = boundary

        settlements = tuple(sorted(
            by_identity.values(),
            key=lambda s: (s.funding_time, s.rate_type or "")))

        # --- Optional bounded context (separate from requested data).
        before_context: FundingContext | None = None
        after_context: FundingContext | None = None
        scoped_blind_spots: list[FundingBlindSpot] = []
        if context_limit > 0:
            def _context(direction: str) -> FundingContext:
                if direction == "before" and start_ms is None:
                    return FundingContext(
                        direction, (), (), context_limit, "insufficient",
                        0, True, "not_requested_no_start",
                        "requested start_time not provided")
                if direction == "after" and end_ms is None:
                    return FundingContext(
                        direction, (), (), context_limit, "insufficient",
                        0, True, "not_requested_no_end",
                        "requested end_time not provided")
                try:
                    if direction == "before":
                        sub = self.historical_funding_rates(
                            symbol, end_time=ms_to_dt(start_ms - 1),
                            total_limit=context_limit,
                            page_limit=page_limit,
                            _max_pages=context_max_pages)
                    else:
                        sub = self.historical_funding_rates(
                            symbol, start_time=ms_to_dt(end_ms + 1),
                            total_limit=context_limit,
                            page_limit=page_limit,
                            _max_pages=context_max_pages)
                except MarketDataError as exc:
                    return FundingContext(
                        direction, (), (), context_limit, "failed",
                        0, False, "api_error", str(exc))
                if sub.pagination_termination == "safety_limit":
                    status = "safety_limit"
                elif not sub.pagination_complete:
                    status = "incomplete_pagination"
                elif len(sub.settlements) >= context_limit:
                    status = "sufficient"
                elif not sub.settlements:
                    status = "empty"
                else:
                    status = "insufficient"
                return FundingContext(
                    direction, sub.settlements, sub.malformed_records,
                    context_limit, status, sub.pages_fetched,
                    sub.pagination_complete, sub.pagination_termination)

            before_context = _context("before")
            after_context = _context("after")
            for context, boundary in ((before_context, "before"),
                                      (after_context, "after")):
                if context.status == "sufficient":
                    continue
                reasons = {
                    "failed": f"{boundary}-context API request failed",
                    "empty": f"no {boundary}-context settlements available",
                    "insufficient": (
                        f"only {context.received_count} of "
                        f"{context_limit} {boundary}-context settlements "
                        "available"),
                    "incomplete_pagination": (
                        f"{boundary}-context pagination incomplete"),
                    "safety_limit": (
                        f"{boundary}-context stopped by safety limit"),
                }
                scoped_blind_spots.append(FundingBlindSpot(
                    boundary=boundary,
                    start_time=(None if boundary == "before"
                                else ms_to_dt(end_ms)),
                    end_time=(ms_to_dt(start_ms) if boundary == "before"
                              else None),
                    reason=reasons.get(context.status, context.status),
                    context_requested=context_limit,
                    context_received=context.received_count,
                    pagination_incomplete=(
                        context.status == "incomplete_pagination"),
                    safety_limit_reached=context.status == "safety_limit",
                    api_failed=context.status == "failed"))

        # Two-sided assessment over requested data widened by context;
        # only intervals intersecting the requested range are kept.
        combined = tuple(
            (before_context.settlements if before_context else ())
            + settlements
            + (after_context.settlements if after_context else ()))
        assessments = assess_funding_intervals(combined)
        if start_ms is not None or end_ms is not None:
            def _intersects(assessment) -> bool:
                previous = assessment.previous_time
                following = assessment.next_time
                if end_ms is not None and previous is not None and (
                        previous.timestamp() * 1000 > end_ms):
                    return False
                if start_ms is not None and following is not None and (
                        following.timestamp() * 1000 < start_ms):
                    return False
                return True

            assessments = tuple(a for a in assessments if _intersects(a))

        collected_at = utcnow()
        observed_start = settlements[0].funding_time if settlements else None
        observed_end = settlements[-1].funding_time if settlements else None
        if forward:
            coverage_start = ms_to_dt(start_ms)
            if termination == "total_limit_reached":
                coverage_end = observed_end
            elif end_ms is not None:
                coverage_end = ms_to_dt(end_ms)
            else:
                coverage_end = collected_at
        else:
            coverage_end = ms_to_dt(end_ms) if end_ms is not None else collected_at
            coverage_start = (None if termination in ("short_page", "empty_page")
                              else observed_start)
        return FundingHistory(
            symbol=symbol, settlements=settlements,
            requested_start_time=ms_to_dt(start_ms),
            requested_end_time=ms_to_dt(end_ms),
            coverage_start_time=coverage_start,
            coverage_end_time=coverage_end,
            collected_at=collected_at, pages_fetched=pages,
            duplicates_removed=duplicates,
            conflicting_records=tuple(conflicts),
            malformed_records=tuple(malformed),
            suspected_anomalies=anomalies_from_assessments(assessments),
            pagination_complete=termination in (
                "total_limit_reached", "short_page", "empty_page",
                "range_exhausted"),
            pagination_termination=termination,
            total_limit=total_limit, page_limit=page_limit,
            boundary_probe_limit_hits=tuple(probe_limit_hits),
            interval_assessments=assessments,
            before_context=before_context, after_context=after_context,
            scoped_blind_spots=tuple(scoped_blind_spots))

    def open_interest(self, symbol: str) -> OpenInterest:
        return parse_open_interest(symbol, self._get(
            "/fapi/v1/openInterest", {"symbol": symbol}))

    def depth(self, symbol: str, limit: int = 20) -> DepthSnapshot:
        return parse_depth(symbol, self._get(
            "/fapi/v1/depth", {"symbol": symbol, "limit": limit}))

    def snapshot(self, symbol: str, interval: str = "1m") -> dict[str, Any]:
        """One-shot bundle used by the diagnostic and downstream code."""
        return {
            "symbol": symbol,
            "ticker_24h": self.ticker_24h(symbol),
            "mark_price": self.mark_price(symbol),
            "open_interest": self.open_interest(symbol),
            "klines": self.klines(symbol, interval=interval, limit=20),
            "funding_history": self.funding_history(symbol, limit=3),
        }

    async def stream(
        self, streams: list[str], *, category: str = "market",
        reconnect_delay_s: float = 2.0,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield raw combined-stream messages, reconnecting on failure.

        Example streams: ``["btcusdt@kline_1m", "btcusdt@markPrice@1s"]``.
        Subscriptions are encoded in the URL, so a reconnect
        automatically resubscribes (no duplicate SUBSCRIBE messages).
        Prefer ``stream_events()`` for normalized models.
        """
        import websockets  # local import: REST-only use needs no WS dep

        url = build_stream_url(streams, category=category)
        delay = reconnect_delay_s
        while True:
            try:
                async with websockets.connect(
                        url, ping_interval=20, ping_timeout=20) as ws:
                    delay = reconnect_delay_s
                    async for message in ws:
                        try:
                            yield json.loads(message)
                        except ValueError:
                            continue  # skip malformed frame, keep streaming
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)

    async def stream_events(
        self, streams: list[str], *, category: str = "market",
        reconnect_delay_s: float = 2.0,
    ) -> AsyncIterator[MarketEvent]:
        """Yield normalized ``MarketEvent``s (source="websocket").

        Reconnects with capped backoff; resubscription is inherent
        (streams are in the URL). Control frames (subscription ACKs)
        and malformed frames are skipped safely.
        """
        async for message in self.stream(
                streams, category=category,
                reconnect_delay_s=reconnect_delay_s):
            event = parse_ws_message(message)
            if event is not None:
                yield event
