"""Normalized Binance USDⓈ-M Futures market-data structures.

Conventions:
- Prices/quantities are ``float``; a missing/unavailable field is
  ``None`` — never silently replaced with ``0.0``. A real ``0.0``
  (e.g. zero funding) stays ``0.0`` and is distinct from ``None``.
- ``event_time`` is the exchange event time (UTC) when provided;
  ``received_at`` is when NOONE received it (UTC).
- ``raw`` preserves the original payload for inspection.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Iterable


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ms_to_dt(ms: int | float | None) -> datetime | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def age_seconds(event_time: datetime | None, *, now: datetime | None = None) -> float | None:
    """Age of a record in seconds; ``None`` when event time is unknown."""
    if event_time is None:
        return None
    return ((now or utcnow()) - event_time).total_seconds()


def is_stale(event_time: datetime | None, max_age_s: float, *, now: datetime | None = None) -> bool:
    age = age_seconds(event_time, now=now)
    return age is None or age > max_age_s


def closed_candles_before(
    candles: list[Kline] | tuple[Kline, ...], decision_time: datetime,
) -> list[Kline]:
    """Point-in-time filter (sync foundation for Component 8).

    Only fully closed candles whose ``close_time <= decision_time``
    may inform a decision at ``decision_time``. The still-forming
    candle is always excluded, preventing lookahead.
    """
    return [c for c in candles if c.is_closed
            and c.close_time is not None and c.close_time <= decision_time]


def _f(value: Any) -> float | None:
    """Parse a numeric field; missing/empty -> None (not 0)."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


INTERVAL_MS: dict[str, int] = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000,
    "12h": 43_200_000, "1d": 86_400_000, "3d": 259_200_000,
    "1w": 604_800_000,
}
"""Supported fixed-length kline intervals (Binance USDⓈ-M Futures).

The calendar interval ``1M`` is deliberately excluded: its length
varies by month, so gap arithmetic in milliseconds cannot apply.
"""


def interval_ms(interval: str) -> int:
    """Milliseconds for a supported interval; ValueError otherwise."""
    try:
        return INTERVAL_MS[interval]
    except KeyError:
        raise ValueError(
            f"Unsupported kline interval {interval!r}; "
            f"supported: {sorted(INTERVAL_MS)}") from None


@dataclass(frozen=True)
class Kline:
    symbol: str
    interval: str
    open_time: datetime | None
    close_time: datetime | None
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None
    quote_volume: float | None
    trades: int | None
    taker_buy_volume: float | None = None
    taker_buy_quote_volume: float | None = None
    # False for the still-forming current candle. Never treat an
    # unclosed candle as final (indicators, RL features).
    is_closed: bool = True
    source: str = "rest"  # "rest" | "websocket" — never mixed silently
    received_at: datetime = field(default_factory=utcnow)
    raw: Any = None


@dataclass(frozen=True)
class KlineGap:
    """A detected hole between two consecutive candles."""

    after_open_time: datetime | None  # last candle before the gap
    before_open_time: datetime | None  # first candle after the gap
    missing_candles: int


@dataclass(frozen=True)
class KlineHistory:
    """Validated historical candle set returned by the collector."""

    symbol: str
    interval: str
    candles: tuple[Kline, ...]  # chronological, de-duplicated
    gaps: tuple[KlineGap, ...]
    duplicates_removed: int
    pages_fetched: int
    start_time: datetime | None
    end_time: datetime | None

    @property
    def incomplete_candles(self) -> tuple[Kline, ...]:
        return tuple(c for c in self.candles if not c.is_closed)


def validate_klines(
    candles: list[Kline], interval: str,
) -> tuple[list[Kline], list[KlineGap], int]:
    """Sort chronologically, drop duplicate open_times, detect gaps.

    Returns ``(clean_candles, gaps, duplicates_removed)``. Missing
    candles are *reported* as gaps — never filled with fabricated
    prices or volumes. Candles without ``open_time`` sort last and
    cannot participate in gap detection.
    """
    step = interval_ms(interval)
    ordered = sorted(
        candles, key=lambda c: (c.open_time is None, c.open_time))
    clean: list[Kline] = []
    seen: set[datetime] = set()
    duplicates = 0
    for candle in ordered:
        if candle.open_time is not None:
            if candle.open_time in seen:
                duplicates += 1
                continue
            seen.add(candle.open_time)
        clean.append(candle)
    gaps: list[KlineGap] = []
    for prev, curr in zip(clean, clean[1:]):
        if prev.open_time is None or curr.open_time is None:
            continue
        delta_ms = (curr.open_time - prev.open_time).total_seconds() * 1000
        missing = round(delta_ms / step) - 1
        if missing > 0:
            gaps.append(KlineGap(
                after_open_time=prev.open_time,
                before_open_time=curr.open_time,
                missing_candles=int(missing)))
    return clean, gaps, duplicates


@dataclass(frozen=True)
class Trade:
    """A single (aggregate) trade. WebSocket-sourced by default."""

    symbol: str
    trade_id: int | None
    price: float | None
    quantity: float | None
    event_time: datetime | None
    trade_time: datetime | None
    is_buyer_maker: bool | None
    source: str = "websocket"  # REST and WS data stay distinguishable
    received_at: datetime = field(default_factory=utcnow)
    raw: Any = None


@dataclass(frozen=True)
class MarketEvent:
    """One normalized WebSocket market event."""

    kind: str  # "trade" | "mark_price" | "kline" | raw event type
    symbol: str
    event_time: datetime | None  # exchange event time (UTC)
    received_at: datetime  # local receive time (UTC)
    stream: str  # e.g. "btcusdt@aggTrade"
    data: Any  # Trade | MarkPrice | Kline | raw dict (unknown kinds)
    source: str = "websocket"
    raw: Any = None

    @property
    def delay_seconds(self) -> float | None:
        """received_at - event_time; None when event time unknown."""
        if self.event_time is None:
            return None
        return (self.received_at - self.event_time).total_seconds()


@dataclass(frozen=True)
class Ticker24h:
    symbol: str
    last_price: float | None
    price_change: float | None
    price_change_percent: float | None
    high_price: float | None
    low_price: float | None
    volume: float | None
    quote_volume: float | None
    open_time: datetime | None
    close_time: datetime | None
    received_at: datetime = field(default_factory=utcnow)
    raw: Any = None


@dataclass(frozen=True)
class MarkPrice:
    symbol: str
    mark_price: float | None
    index_price: float | None
    estimated_settle_price: float | None
    last_funding_rate: float | None
    next_funding_time: datetime | None
    event_time: datetime | None
    source: str = "rest"
    received_at: datetime = field(default_factory=utcnow)
    raw: Any = None


@dataclass(frozen=True)
class FundingRatePoint:
    symbol: str
    funding_rate: float | None
    funding_time: datetime | None
    mark_price: float | None
    received_at: datetime = field(default_factory=utcnow)
    raw: Any = None


@dataclass(frozen=True)
class FundingSettlement:
    """One settled funding event (Component 4 historical funding).

    ``funding_rate`` and ``mark_price`` keep Binance's exact decimal
    values (never float-rounded at collection). ``funding_time`` is
    the settlement timestamp (UTC). ``rate_type`` preserves Binance's
    rate-type field when supplied (e.g. regular vs special); it is
    part of the settlement identity because a special event could
    share a timestamp with a regular one.
    """

    symbol: str
    funding_time: datetime | None
    funding_rate: Decimal | None
    mark_price: Decimal | None
    rate_type: str | None = None
    received_at: datetime = field(default_factory=utcnow)
    raw: Any = None

    @property
    def identity(self) -> tuple[str, datetime | None, str | None]:
        return (self.symbol, self.funding_time, self.rate_type)


@dataclass(frozen=True)
class FundingMalformedOccurrence:
    """One appearance of a malformed record on a specific API page."""

    page_index: int | None = None  # 0-based page within the collection
    page_start_ms: int | None = None  # actual startTime sent (None = unsent)
    page_end_ms: int | None = None  # actual endTime sent (None = unsent)
    record_index: int | None = None  # index within the returned page


@dataclass(frozen=True)
class FundingMalformedRecord:
    """A funding record that cannot be used for accounting.

    Provenance fields describe the *first* occurrence; every
    appearance (including boundary-overlap repeats) is preserved in
    ``occurrences``. ``page_start_ms``/``page_end_ms`` are the actual
    request window sent to Binance — for timestamp-less records they
    bound the possible impact period and are never the event time.
    """

    symbol: str  # symbol the collection was requested for
    funding_time: datetime | None
    reason: str
    raw: Any = None
    record_symbol: str | None = None  # symbol inside the record, if any
    page_index: int | None = None
    page_start_ms: int | None = None
    page_end_ms: int | None = None
    record_index: int | None = None
    occurrence_count: int = 1
    occurrences: tuple[FundingMalformedOccurrence, ...] = ()

    @property
    def has_usable_timestamp(self) -> bool:
        return self.funding_time is not None


@dataclass(frozen=True)
class FundingConflict:
    """Same settlement identity with differing rate/mark values."""

    symbol: str
    funding_time: datetime | None
    rate_type: str | None
    kept_rate: Decimal | None
    kept_mark_price: Decimal | None
    conflicting_rate: Decimal | None
    conflicting_mark_price: Decimal | None
    raw: Any = None


@dataclass(frozen=True)
class FundingIntervalAnomaly:
    """Suspected (not proven) missing settlement(s) between events.

    Detected only against a locally established interval (repeated
    adjacent intervals), never an assumed 8-hour schedule. An
    interval change can look identical, so acknowledgement is an
    explicit human/investigation action, never automatic.
    """

    after_time: datetime | None  # last settlement before the anomaly
    before_time: datetime | None  # first settlement after it
    interval_seconds: float  # observed gap between the two events
    reference_interval_seconds: float  # locally established interval
    estimated_missing_count: int | None  # None when not a near-multiple
    acknowledged: bool = False

    @property
    def key(self) -> tuple[datetime | None, datetime | None]:
        return (self.after_time, self.before_time)


# Two-sided interval assessment classifications.
SUSPECTED_MISSING_SETTLEMENT = "SUSPECTED_MISSING_SETTLEMENT"
OBSERVED_INTERVAL_CHANGE = "OBSERVED_INTERVAL_CHANGE"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class FundingIntervalAssessment:
    """Evidence record for one suspicious funding interval.

    Produced by the two-sided detector. ``previous_time`` and
    ``next_time`` are the exact observed settlements bracketing the
    interval; no synthetic settlement is ever created. Supporting
    interval lists (seconds) show the observations on both sides
    that drove the classification.
    """

    symbol: str
    rate_type: str | None
    previous_time: datetime | None
    next_time: datetime | None
    observed_interval_seconds: float
    reference_interval_seconds: float | None
    estimated_interval_multiple: float | None
    supporting_before_seconds: tuple[float, ...]
    supporting_after_seconds: tuple[float, ...]
    classification: str  # SUSPECTED_MISSING_SETTLEMENT|OBSERVED_INTERVAL_CHANGE|INSUFFICIENT_EVIDENCE
    reason: str
    evidence_sufficient: bool
    acknowledged: bool = False

    @property
    def key(self) -> tuple[datetime | None, datetime | None, str | None]:
        return (self.previous_time, self.next_time, self.rate_type)


@dataclass(frozen=True)
class FundingBlindSpot:
    """A scoped region the anomaly detector could not assess."""

    boundary: str  # "before" | "after" | "internal"
    start_time: datetime | None  # affected range start (None = open)
    end_time: datetime | None  # affected range end (None = open)
    reason: str
    context_requested: int = 0
    context_received: int = 0
    pagination_incomplete: bool = False
    safety_limit_reached: bool = False
    api_failed: bool = False


@dataclass(frozen=True)
class FundingContext:
    """Bounded settlement context on one side of a requested period.

    Kept separate from the requested period's settlements and
    pagination metadata. ``status`` is one of: sufficient, empty
    (request succeeded, no events available), insufficient,
    incomplete_pagination, safety_limit, failed.
    """

    direction: str  # "before" | "after"
    settlements: tuple[FundingSettlement, ...]
    malformed_records: tuple[FundingMalformedRecord, ...]
    requested_count: int
    status: str
    pages_fetched: int
    pagination_complete: bool
    pagination_termination: str
    error: str | None = None

    @property
    def received_count(self) -> int:
        return len(self.settlements)


@dataclass(frozen=True)
class BoundaryProbeLimitHit:
    """A full-timestamp probe that returned the 1,000-record maximum.

    The endpoint has no record-ID/offset/rateType cursor, so the
    collector cannot prove that no further records exist at this
    timestamp. The history must never be classified complete.
    """

    funding_time: datetime | None
    returned_count: int


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(timezone.utc).isoformat() if value else None


def _dec(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


@dataclass(frozen=True)
class FundingHistory:
    """Validated historical funding set returned by the collector.

    Distinguishes the *requested* range from the *observed* event
    range and from the *coverage* window the collection actually
    queried. First/last event timestamps alone never prove
    continuous coverage: funding intervals are variable, so coverage
    is expressed through the queried window plus pagination state,
    suspected anomalies, malformed and conflicting records.
    """

    symbol: str
    settlements: tuple[FundingSettlement, ...]  # chronological, unique
    requested_start_time: datetime | None
    requested_end_time: datetime | None
    coverage_start_time: datetime | None  # None = beginning of history
    coverage_end_time: datetime | None  # None = up to collection time
    collected_at: datetime
    pages_fetched: int
    duplicates_removed: int
    conflicting_records: tuple[FundingConflict, ...]
    malformed_records: tuple[FundingMalformedRecord, ...]
    suspected_anomalies: tuple[FundingIntervalAnomaly, ...]
    pagination_complete: bool
    pagination_termination: str  # total_limit_reached|short_page|empty_page|range_exhausted|repeated_page|no_usable_timestamp
    total_limit: int
    page_limit: int
    # Additive Stage-1 hardening fields (defaults keep existing
    # constructors and callers working unchanged).
    boundary_probe_limit_hits: tuple[BoundaryProbeLimitHit, ...] = ()
    blind_spots: tuple[str, ...] = ()  # populated by later detection stages
    # Additive Stage-3 fields.
    interval_assessments: tuple[FundingIntervalAssessment, ...] = ()
    before_context: FundingContext | None = None
    after_context: FundingContext | None = None
    scoped_blind_spots: tuple[FundingBlindSpot, ...] = ()

    @property
    def interval_transitions(self) -> tuple[FundingIntervalAssessment, ...]:
        return tuple(a for a in self.interval_assessments
                     if a.classification == OBSERVED_INTERVAL_CHANGE)

    @property
    def observed_start_time(self) -> datetime | None:
        return self.settlements[0].funding_time if self.settlements else None

    @property
    def observed_end_time(self) -> datetime | None:
        return self.settlements[-1].funding_time if self.settlements else None

    @property
    def acknowledged_anomalies(self) -> tuple[FundingIntervalAnomaly, ...]:
        return tuple(a for a in self.suspected_anomalies if a.acknowledged)

    @property
    def unacknowledged_anomalies(self) -> tuple[FundingIntervalAnomaly, ...]:
        return tuple(a for a in self.suspected_anomalies if not a.acknowledged)

    @property
    def completeness_status(self) -> str:
        """Derived three-state completeness: complete|uncertain|incomplete.

        INCOMPLETE (priority): pagination incomplete, a conflicting
        settlement exists, or a malformed record with a usable
        timestamp was collected (such records arrive only from the
        requested pages, i.e. inside the evaluated collection).
        UNCERTAIN: a boundary probe hit the 1,000-record limit, a
        suspected anomaly exists (acknowledged or not —
        acknowledgement records human review, never manufactures
        certainty), an unassessed blind spot exists, or a
        timestamp-less malformed record may affect the period.
        COMPLETE: none of the above. This attests to evidence under
        this model, never that Binance provably returned every
        settlement that should exist.
        """
        if not self.pagination_complete:
            return "incomplete"
        if self.conflicting_records:
            return "incomplete"
        if any(m.funding_time is not None for m in self.malformed_records):
            return "incomplete"
        if self.boundary_probe_limit_hits:
            return "uncertain"
        if self.suspected_anomalies:
            return "uncertain"
        if any(a.classification in (
                SUSPECTED_MISSING_SETTLEMENT, INSUFFICIENT_EVIDENCE)
                for a in self.interval_assessments):
            return "uncertain"
        if self.blind_spots:
            return "uncertain"
        if self.scoped_blind_spots:
            return "uncertain"
        if any(m.funding_time is None for m in self.malformed_records):
            return "uncertain"
        return "complete"

    def covers_period(self, period_start: datetime, period_end: datetime) -> bool:
        """Whether collection *completely* covered a candle period.

        True only when pagination completed and the queried coverage
        window spans the period. Based on the coverage window, never
        on first/last event timestamps alone. Even then it does not
        prove no settlement is missing inside the period: suspected
        anomalies (acknowledged or not), malformed records and
        conflicts remain separate visible facts in ``metadata()``.
        Acknowledging an anomaly never changes this result.
        """
        if not self.pagination_complete:
            return False
        start_ok = (self.coverage_start_time is None
                    or self.coverage_start_time <= period_start)
        end_ok = (self.coverage_end_time is None
                  or self.coverage_end_time >= period_end)
        return start_ok and end_ok

    def completeness_status_for_period(
        self, period_start: datetime, period_end: datetime,
    ) -> str:
        """Three-state completeness scoped to one evaluated period.

        Additive to (and independent of) ``covers_period`` and the
        history-level ``completeness_status``. A malformed record or
        conflict whose usable timestamp lies inside the period makes
        it incomplete; a timestamp-less malformed record whose
        *page request window* overlaps the period makes it uncertain
        (its impact cannot be placed precisely); timestamp-less
        records whose page window lies wholly outside the period do
        not affect it. Boundary ambiguity and suspected anomalies
        count only when they intersect the period. INCOMPLETE takes
        priority over UNCERTAIN.
        """
        start_ms = period_start.timestamp() * 1000
        end_ms = period_end.timestamp() * 1000

        def inside(value: datetime | None) -> bool:
            return (value is not None
                    and start_ms <= value.timestamp() * 1000 <= end_ms)

        if not self.pagination_complete:
            return "incomplete"
        if any(inside(c.funding_time) for c in self.conflicting_records):
            return "incomplete"
        if any(inside(m.funding_time) for m in self.malformed_records):
            return "incomplete"
        if any(inside(h.funding_time)
               for h in self.boundary_probe_limit_hits):
            return "uncertain"
        for anomaly in self.suspected_anomalies:
            after = (anomaly.after_time.timestamp() * 1000
                     if anomaly.after_time else None)
            before = (anomaly.before_time.timestamp() * 1000
                      if anomaly.before_time else None)
            if (after is None or after <= end_ms) and (
                    before is None or before >= start_ms):
                return "uncertain"
        for assessment in self.interval_assessments:
            if assessment.classification == OBSERVED_INTERVAL_CHANGE:
                continue  # a proven transition is not uncertainty
            after = (assessment.previous_time.timestamp() * 1000
                     if assessment.previous_time else None)
            before = (assessment.next_time.timestamp() * 1000
                      if assessment.next_time else None)
            if (after is None or after <= end_ms) and (
                    before is None or before >= start_ms):
                return "uncertain"
        if self.blind_spots:
            return "uncertain"
        for spot in self.scoped_blind_spots:
            if spot.boundary == "before":
                # Affects only periods whose start is being assessed
                # at/before the requested start boundary.
                boundary_ms = (spot.end_time.timestamp() * 1000
                               if spot.end_time else start_ms)
                if start_ms <= boundary_ms:
                    return "uncertain"
            elif spot.boundary == "after":
                boundary_ms = (spot.start_time.timestamp() * 1000
                               if spot.start_time else end_ms)
                if end_ms >= boundary_ms:
                    return "uncertain"
            else:  # internal: actual affected range must overlap
                spot_start = (spot.start_time.timestamp() * 1000
                              if spot.start_time else float("-inf"))
                spot_end = (spot.end_time.timestamp() * 1000
                            if spot.end_time else float("inf"))
                if spot_start <= end_ms and spot_end >= start_ms:
                    return "uncertain"
        for record in self.malformed_records:
            if record.funding_time is not None:
                continue
            window_start = (record.page_start_ms
                            if record.page_start_ms is not None
                            else float("-inf"))
            window_end = (record.page_end_ms
                          if record.page_end_ms is not None
                          else float("inf"))
            if window_start <= end_ms and window_end >= start_ms:
                return "uncertain"
        return "complete"

    def acknowledge_anomalies(
        self, anomalies: Iterable[FundingIntervalAnomaly],
    ) -> "FundingHistory":
        """Return a copy with the given anomalies acknowledged.

        Explicit only: keys must match existing anomalies exactly,
        original details are preserved, and acknowledging never
        clears conflicts, malformed records or pagination state.
        """
        keys = {a.key for a in anomalies}
        known = {a.key for a in self.suspected_anomalies}
        unknown = keys - known
        if unknown:
            raise ValueError(f"Unknown funding anomalies to acknowledge: {unknown}")
        updated = tuple(
            replace(a, acknowledged=True) if a.key in keys else a
            for a in self.suspected_anomalies)
        # Mirror the acknowledgement onto matching assessments for
        # audit; evidence fields are preserved and completeness is
        # unaffected (acknowledgement never upgrades certainty).
        updated_assessments = tuple(
            replace(a, acknowledged=True)
            if (a.previous_time, a.next_time) in keys else a
            for a in self.interval_assessments)
        return replace(self, suspected_anomalies=updated,
                       interval_assessments=updated_assessments)

    def metadata(self) -> dict:
        return {
            "symbol": self.symbol,
            "settlement_count": len(self.settlements),
            "requested_start_time": _iso(self.requested_start_time),
            "requested_end_time": _iso(self.requested_end_time),
            "observed_start_time": _iso(self.observed_start_time),
            "observed_end_time": _iso(self.observed_end_time),
            "coverage_start_time": _iso(self.coverage_start_time),
            "coverage_end_time": _iso(self.coverage_end_time),
            "collected_at": _iso(self.collected_at),
            "pages_fetched": self.pages_fetched,
            "duplicates_removed": self.duplicates_removed,
            "total_limit": self.total_limit,
            "page_limit": self.page_limit,
            "pagination_complete": self.pagination_complete,
            "pagination_termination": self.pagination_termination,
            "completeness_status": self.completeness_status,
            "boundary_probe_limit_hits": [{
                "funding_time": _iso(h.funding_time),
                "returned_count": h.returned_count,
            } for h in self.boundary_probe_limit_hits],
            "blind_spots": list(self.blind_spots),
            "conflicting_records": [{
                "funding_time": _iso(c.funding_time),
                "rate_type": c.rate_type,
                "kept_rate": _dec(c.kept_rate),
                "kept_mark_price": _dec(c.kept_mark_price),
                "conflicting_rate": _dec(c.conflicting_rate),
                "conflicting_mark_price": _dec(c.conflicting_mark_price),
            } for c in self.conflicting_records],
            "malformed_records": [{
                "funding_time": _iso(m.funding_time), "reason": m.reason,
                "symbol": m.symbol, "record_symbol": m.record_symbol,
                "raw": m.raw,
                "has_usable_timestamp": m.has_usable_timestamp,
                "page_index": m.page_index,
                "page_start_ms": m.page_start_ms,
                "page_end_ms": m.page_end_ms,
                "record_index": m.record_index,
                "occurrence_count": m.occurrence_count,
                "occurrences": [{
                    "page_index": o.page_index,
                    "page_start_ms": o.page_start_ms,
                    "page_end_ms": o.page_end_ms,
                    "record_index": o.record_index,
                } for o in m.occurrences],
                # For timestamp-less records the page window only
                # bounds possible impact; it is NOT an event time.
                "possible_impact_window": (
                    {"start_ms": m.page_start_ms, "end_ms": m.page_end_ms,
                     "label": "possible_impact_bound_not_event_time"}
                    if m.funding_time is None else None),
            } for m in self.malformed_records],
            "malformed_count": len(self.malformed_records),
            "malformed_with_timestamp_count": sum(
                1 for m in self.malformed_records
                if m.funding_time is not None),
            "malformed_without_timestamp_count": sum(
                1 for m in self.malformed_records
                if m.funding_time is None),
            "suspected_anomalies": [{
                "after_time": _iso(a.after_time),
                "before_time": _iso(a.before_time),
                "interval_seconds": a.interval_seconds,
                "reference_interval_seconds": a.reference_interval_seconds,
                "estimated_missing_count": a.estimated_missing_count,
                "acknowledged": a.acknowledged,
            } for a in self.suspected_anomalies],
            "acknowledged_anomalies": [
                {"after_time": _iso(a.after_time),
                 "before_time": _iso(a.before_time)}
                for a in self.acknowledged_anomalies],
            "unacknowledged_anomaly_count": len(self.unacknowledged_anomalies),
            "interval_assessments": [{
                "symbol": a.symbol, "rate_type": a.rate_type,
                "previous_time": _iso(a.previous_time),
                "next_time": _iso(a.next_time),
                "observed_interval_seconds": a.observed_interval_seconds,
                "reference_interval_seconds": a.reference_interval_seconds,
                "estimated_interval_multiple": a.estimated_interval_multiple,
                "supporting_before_seconds": list(a.supporting_before_seconds),
                "supporting_after_seconds": list(a.supporting_after_seconds),
                "classification": a.classification, "reason": a.reason,
                "evidence_sufficient": a.evidence_sufficient,
                "acknowledged": a.acknowledged,
            } for a in self.interval_assessments],
            "before_context": _context_metadata(self.before_context),
            "after_context": _context_metadata(self.after_context),
            "scoped_blind_spots": [{
                "boundary": s.boundary,
                "start_time": _iso(s.start_time),
                "end_time": _iso(s.end_time), "reason": s.reason,
                "context_requested": s.context_requested,
                "context_received": s.context_received,
                "pagination_incomplete": s.pagination_incomplete,
                "safety_limit_reached": s.safety_limit_reached,
                "api_failed": s.api_failed,
            } for s in self.scoped_blind_spots],
        }


def _context_metadata(context: FundingContext | None) -> dict | None:
    if context is None:
        return None
    return {
        "direction": context.direction, "status": context.status,
        "requested_count": context.requested_count,
        "received_count": context.received_count,
        "settlement_times": [_iso(s.funding_time)
                             for s in context.settlements],
        "malformed_count": len(context.malformed_records),
        "pages_fetched": context.pages_fetched,
        "pagination_complete": context.pagination_complete,
        "pagination_termination": context.pagination_termination,
        "error": context.error,
    }


def _within_tolerance(
    value: float, reference: float,
    tolerance_seconds: float, tolerance_ratio: float,
) -> bool:
    return (reference > 0 and abs(value - reference)
            <= max(tolerance_seconds, tolerance_ratio * reference))


def _stable_reference(
    window: list[float], tolerance_seconds: float, tolerance_ratio: float,
) -> float | None:
    """Median of the last up-to-3 intervals when >=2 agree with it."""
    from statistics import median

    if len(window) < 2:
        return None
    recent = window[-3:]
    reference = median(recent)
    if reference <= 0:
        return None
    agreeing = sum(1 for d in recent if _within_tolerance(
        d, reference, tolerance_seconds, tolerance_ratio))
    return reference if agreeing >= 2 else None


def assess_funding_intervals(
    settlements: tuple[FundingSettlement, ...] | list[FundingSettlement],
    *, tolerance_seconds: float = 120.0, tolerance_ratio: float = 0.02,
) -> tuple[FundingIntervalAssessment, ...]:
    """Two-sided funding-interval assessment (never fabricates events).

    Settlements are assessed independently per ``rate_type`` group
    (types are never mixed when establishing a reference). For each
    interval that deviates from a locally stable preceding reference,
    the following intervals decide the classification:

    - near-integer multiple (>=2x) of the reference and following
      intervals return to it -> SUSPECTED_MISSING_SETTLEMENT;
    - the new interval continues afterwards (longer or shorter)
      -> OBSERVED_INTERVAL_CHANGE;
    - anything else (no stable preceding reference, no following
      observations, or an irregular pattern) -> INSUFFICIENT_EVIDENCE,
      emitted only when the interval is long against some neighbouring
      scale so ordinary series starts are not flagged.

    Tolerance defaults to max(120s, 2% of the reference): live
    BTCUSDT/ETHUSDT/GTCUSDT settlement deltas deviated from their
    nominal intervals by at most ~7ms in audit samples, so this is
    far above observed jitter yet far below any interval multiple.
    It is an engineering safeguard, not an exchange-certified bound.
    Only actual settlements are ever examined; nothing is created.
    """
    groups: dict[str | None, list[FundingSettlement]] = {}
    for settlement in settlements:
        if settlement.funding_time is not None:
            groups.setdefault(settlement.rate_type, []).append(settlement)
    assessments: list[FundingIntervalAssessment] = []
    for rate_type, group in groups.items():
        times = sorted({s.funding_time for s in group
                        if s.funding_time is not None})
        if len(times) < 2:
            continue
        symbol = group[0].symbol
        deltas = [(b - a).total_seconds()
                  for a, b in zip(times, times[1:])]

        def make(i, classification, reference, reason, sufficient):
            delta = deltas[i]
            return FundingIntervalAssessment(
                symbol=symbol, rate_type=rate_type,
                previous_time=times[i], next_time=times[i + 1],
                observed_interval_seconds=delta,
                reference_interval_seconds=reference,
                estimated_interval_multiple=(
                    delta / reference if reference else None),
                supporting_before_seconds=tuple(deltas[max(0, i - 3):i]),
                supporting_after_seconds=tuple(deltas[i + 1:i + 4]),
                classification=classification, reason=reason,
                evidence_sufficient=sufficient)

        for i, delta in enumerate(deltas):
            before = deltas[max(0, i - 3):i]
            after = deltas[i + 1:i + 4]
            if before and _within_tolerance(
                    delta, before[-1], tolerance_seconds, tolerance_ratio):
                continue  # continuation of the immediately preceding
                # interval; any change was assessed when it first appeared
            reference = _stable_reference(
                before, tolerance_seconds, tolerance_ratio)
            if reference is not None and _within_tolerance(
                    delta, reference, tolerance_seconds, tolerance_ratio):
                continue  # ordinary interval
            recent_pair = (
                before[-1] if len(before) >= 2 and _within_tolerance(
                    before[-1], before[-2],
                    tolerance_seconds, tolerance_ratio) else None)
            if (reference is not None and delta < reference
                    and recent_pair is not None):
                # Shorter interval: a change only if it continues.
                if after and _within_tolerance(
                        after[0], delta, tolerance_seconds, tolerance_ratio):
                    assessments.append(make(
                        i, OBSERVED_INTERVAL_CHANGE, reference,
                        "interval shortened and the shorter interval "
                        "continues afterwards", True))
                elif not after:
                    assessments.append(make(
                        i, INSUFFICIENT_EVIDENCE, reference,
                        "interval shortened at the end of the observed "
                        "series; no following observations", False))
                else:
                    assessments.append(make(
                        i, INSUFFICIENT_EVIDENCE, reference,
                        "interval shortened but does not continue and "
                        "does not return to the reference", False))
                continue
            if reference is not None and delta > reference:
                ratio = delta / reference
                nearest = round(ratio)
                near_multiple = nearest >= 2 and abs(
                    delta - nearest * reference) <= max(
                        tolerance_seconds, tolerance_ratio * delta)
                returns = bool(after) and _within_tolerance(
                    after[0], reference, tolerance_seconds, tolerance_ratio)
                continues = bool(after) and _within_tolerance(
                    after[0], delta, tolerance_seconds, tolerance_ratio) \
                    and (len(after) < 2 or _within_tolerance(
                        after[1], delta, tolerance_seconds, tolerance_ratio))
                if near_multiple and returns:
                    assessments.append(make(
                        i, SUSPECTED_MISSING_SETTLEMENT, reference,
                        "gap is a near-integer multiple of the stable "
                        "preceding interval and following intervals "
                        "return to that interval", True))
                elif continues:
                    assessments.append(make(
                        i, OBSERVED_INTERVAL_CHANGE, reference,
                        "longer interval continues afterwards; no "
                        "return to the preceding interval", True))
                elif not after:
                    if recent_pair is not None:
                        assessments.append(make(
                            i, INSUFFICIENT_EVIDENCE, reference,
                            "long gap at the end of the observed series; "
                            "no following observations to distinguish a "
                            "missing settlement from an interval change",
                            False))
                else:
                    if recent_pair is not None:
                        assessments.append(make(
                            i, INSUFFICIENT_EVIDENCE, reference,
                            "long gap is not a near-integer multiple with "
                            "a return to the reference, nor a continuing "
                            "new interval", False))
                continue
            # No stable preceding reference: flag only when the gap is
            # long against a neighbouring scale, as insufficient
            # evidence (never as a suspected missing settlement). The
            # following reference must come from the intervals
            # immediately after the gap (first agreeing pair), so a
            # later regime change is not projected backwards.
            neighbour = (after[0] if len(after) >= 2 and _within_tolerance(
                after[0], after[1], tolerance_seconds, tolerance_ratio)
                else None)
            if neighbour is None and before and before[-1] > 0:
                neighbour = before[-1]
            if (neighbour is not None and delta >= 1.5 * neighbour
                    and not _within_tolerance(
                        delta, neighbour,
                        tolerance_seconds, tolerance_ratio)):
                assessments.append(make(
                    i, INSUFFICIENT_EVIDENCE, neighbour,
                    "no sufficiently stable preceding interval "
                    "reference; classification cannot be established",
                    False))
    return tuple(assessments)


def anomalies_from_assessments(
    assessments: tuple[FundingIntervalAssessment, ...]
    | list[FundingIntervalAssessment],
) -> tuple[FundingIntervalAnomaly, ...]:
    """Convert suspected-missing assessments to anomaly records."""
    anomalies: list[FundingIntervalAnomaly] = []
    for assessment in assessments:
        if assessment.classification != SUSPECTED_MISSING_SETTLEMENT:
            continue
        reference = assessment.reference_interval_seconds or 0.0
        multiple = assessment.estimated_interval_multiple
        estimated = (int(round(multiple)) - 1
                     if multiple is not None and multiple >= 2 else None)
        anomalies.append(FundingIntervalAnomaly(
            after_time=assessment.previous_time,
            before_time=assessment.next_time,
            interval_seconds=assessment.observed_interval_seconds,
            reference_interval_seconds=reference,
            estimated_missing_count=estimated,
            acknowledged=assessment.acknowledged))
    return tuple(anomalies)


def detect_funding_anomalies(
    settlements: tuple[FundingSettlement, ...] | list[FundingSettlement],
    *, tolerance_seconds: float = 120.0, tolerance_ratio: float = 0.02,
) -> tuple[FundingIntervalAnomaly, ...]:
    """Suspected missing settlements from the two-sided assessment.

    Back-compatible view over ``assess_funding_intervals``: only
    SUSPECTED_MISSING_SETTLEMENT classifications are returned, as
    ``FundingIntervalAnomaly`` records. Interval changes and
    insufficient-evidence intervals are not anomalies here.
    """
    return anomalies_from_assessments(assess_funding_intervals(
        settlements, tolerance_seconds=tolerance_seconds,
        tolerance_ratio=tolerance_ratio))


@dataclass(frozen=True)
class OpenInterest:
    symbol: str
    open_interest: float | None
    event_time: datetime | None
    received_at: datetime = field(default_factory=utcnow)
    raw: Any = None


@dataclass(frozen=True)
class DepthSnapshot:
    symbol: str
    bids: tuple[tuple[float | None, float | None], ...]
    asks: tuple[tuple[float | None, float | None], ...]
    last_update_id: int | None
    event_time: datetime | None
    received_at: datetime = field(default_factory=utcnow)
    raw: Any = None


@dataclass(frozen=True)
class ExchangeSymbol:
    symbol: str
    pair: str | None
    contract_type: str | None
    status: str | None
    base_asset: str | None
    quote_asset: str | None
    margin_asset: str | None
    raw: Any = None
