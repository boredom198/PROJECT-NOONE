"""Historical Futures data preparation for Stage 2 (Component 1).

Rules enforced here, reusing Component 4 models/validation:
chronological order, de-duplication by ``open_time``, closed candles
only, gaps detected and reported (never filled/fabricated), malformed
candles fail loudly, and a strict chronological train/eval split with
no decision-candle overlap. Indicator warm-up candles preceding the
evaluation period may be supplied as *context only* (forced HOLD,
never model decisions, never recorded in evaluation metrics).
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from noone.market_data.models import (
    INTERVAL_MS,
    Kline,
    KlineGap,
    interval_ms,
    validate_klines,
)


class DataPreparationError(ValueError):
    """Raised when historical data is unavailable/insufficient/invalid."""


@dataclass(frozen=True)
class PreparedData:
    candles: tuple[Kline, ...]
    gaps: tuple[KlineGap, ...]
    duplicates_removed: int
    dropped_unclosed: int
    symbol: str
    interval: str

    @property
    def start_time(self) -> datetime | None:
        return self.candles[0].open_time if self.candles else None

    @property
    def end_time(self) -> datetime | None:
        return self.candles[-1].close_time if self.candles else None

    def metadata(self) -> dict:
        return {
            "symbol": self.symbol, "interval": self.interval,
            "candle_count": len(self.candles),
            "start_time": _iso(self.start_time), "end_time": _iso(self.end_time),
            "duplicates_removed": self.duplicates_removed,
            "dropped_unclosed": self.dropped_unclosed,
            "gap_count": len(self.gaps),
            "missing_candles_in_gaps": sum(g.missing_candles for g in self.gaps),
            "gaps": [{
                "after_open_time": _iso(g.after_open_time),
                "before_open_time": _iso(g.before_open_time),
                "missing_candles": g.missing_candles,
            } for g in self.gaps],
        }


@dataclass(frozen=True)
class ChronologicalSplit:
    train: tuple[Kline, ...]
    evaluation: tuple[Kline, ...]
    # Last warm-up candles of the training period: context for the
    # first evaluation observations only, never evaluation decisions.
    eval_warmup_context: tuple[Kline, ...]
    split_index: int
    train_fraction: float

    def metadata(self) -> dict:
        return {
            "train_fraction": self.train_fraction,
            "split_index": self.split_index,
            "train_candle_count": len(self.train),
            "evaluation_candle_count": len(self.evaluation),
            "eval_warmup_context_count": len(self.eval_warmup_context),
            "train_start": _iso(self.train[0].open_time if self.train else None),
            "train_end": _iso(self.train[-1].close_time if self.train else None),
            "evaluation_start": _iso(
                self.evaluation[0].open_time if self.evaluation else None),
            "evaluation_end": _iso(
                self.evaluation[-1].close_time if self.evaluation else None),
        }


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(timezone.utc).isoformat() if value else None


def _ohlc_error(candle: Kline) -> str | None:
    values = {"open": candle.open, "high": candle.high,
              "low": candle.low, "close": candle.close}
    for name, value in values.items():
        if value is None:
            return f"missing {name}"
        if not math.isfinite(value):
            return f"non-finite {name}"
        if value <= 0:
            return f"non-positive {name}"
    assert candle.open is not None and candle.high is not None
    assert candle.low is not None and candle.close is not None
    if candle.high < max(candle.open, candle.close, candle.low):
        return "high below open/close/low"
    if candle.low > min(candle.open, candle.close, candle.high):
        return "low above open/close/high"
    return None


def prepare_candles(
    candles: Iterable[Kline], *, interval: str,
    symbol: str | None = None, min_candles: int = 2,
    warmup_candles: int = 0,
) -> PreparedData:
    """Validate and clean raw candles; fail clearly on bad data."""
    if interval not in INTERVAL_MS:
        raise DataPreparationError(
            f"Unsupported interval {interval!r}; supported: {sorted(INTERVAL_MS)}")
    raw = list(candles)
    if not raw:
        raise DataPreparationError(
            "No historical candles available: check symbol/interval/date "
            "range and Component 4 collector availability")
    resolved_symbol = symbol or raw[0].symbol
    for candle in raw:
        if candle.symbol != resolved_symbol:
            raise DataPreparationError(
                f"Mixed symbols in dataset: {candle.symbol!r} != {resolved_symbol!r}")
        if candle.interval != interval:
            raise DataPreparationError(
                f"Mixed intervals in dataset: {candle.interval!r} != {interval!r}")
        if candle.open_time is None:
            raise DataPreparationError("Candle without open_time cannot be ordered")
    dropped_unclosed = sum(1 for c in raw if not c.is_closed)
    closed = [c for c in raw if c.is_closed]
    if not closed:
        raise DataPreparationError(
            "No closed candles available (only unclosed/forming candles supplied)")
    for candle in closed:
        reason = _ohlc_error(candle)
        if reason is not None:
            raise DataPreparationError(
                f"Invalid candle at {candle.open_time}: {reason}")
    clean, gaps, duplicates = validate_klines(closed, interval)
    times = [c.open_time for c in clean]
    if times != sorted(times) or len(set(times)) != len(times):
        raise DataPreparationError("Candles are not strictly chronological after cleaning")
    required = max(min_candles, warmup_candles + 2)
    if len(clean) < required:
        raise DataPreparationError(
            f"Insufficient historical data: {len(clean)} closed candles, "
            f"need at least {required} (min_candles={min_candles}, "
            f"warmup_candles={warmup_candles}). Fetch a longer Futures history.")
    return PreparedData(
        candles=tuple(clean), gaps=tuple(gaps), duplicates_removed=duplicates,
        dropped_unclosed=dropped_unclosed, symbol=resolved_symbol, interval=interval)


def chronological_split(
    prepared: PreparedData | Sequence[Kline], *, train_fraction: float = 0.8,
    warmup_candles: int = 50, min_train_candles: int = 100,
    min_eval_candles: int = 50,
) -> ChronologicalSplit:
    """Strict earlier-train / later-evaluation split. No shuffling."""
    candles = list(prepared.candles if isinstance(prepared, PreparedData) else prepared)
    if not 0.0 < train_fraction < 1.0:
        raise DataPreparationError("train_fraction must be in (0, 1)")
    split_index = int(len(candles) * train_fraction)
    train, evaluation = tuple(candles[:split_index]), tuple(candles[split_index:])
    if len(train) < min_train_candles:
        raise DataPreparationError(
            f"Insufficient training candles after split: {len(train)} < {min_train_candles}")
    if len(evaluation) < min_eval_candles:
        raise DataPreparationError(
            f"Insufficient evaluation candles after split: "
            f"{len(evaluation)} < {min_eval_candles}")
    if split_index <= warmup_candles:
        raise DataPreparationError(
            f"Split index {split_index} leaves no room for {warmup_candles} "
            "warm-up candles in the training period")
    if train and evaluation and not train[-1].open_time < evaluation[0].open_time:  # type: ignore[operator]
        raise DataPreparationError("Train/evaluation overlap detected")
    warmup_context = train[-warmup_candles:] if warmup_candles else ()
    return ChronologicalSplit(
        train=train, evaluation=evaluation, eval_warmup_context=tuple(warmup_context),
        split_index=split_index, train_fraction=train_fraction)


def _parse_time(value: str) -> datetime:
    text = value.strip()
    if text.replace(".", "", 1).isdigit():
        return datetime.fromtimestamp(float(text) / 1000.0, tz=timezone.utc)
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def load_candles_csv(path: str | Path, *, symbol: str, interval: str) -> list[Kline]:
    """Load candles exported as CSV (open_time,close_time,open,high,
    low,close,volume[,quote_volume,trades]). Times: ISO-8601 or ms epoch."""
    rows = list(csv.DictReader(Path(path).open(newline="", encoding="utf-8")))
    if not rows:
        raise DataPreparationError(f"No candles in CSV file: {path}")
    candles: list[Kline] = []
    for line, row in enumerate(rows, start=2):
        try:
            open_time = _parse_time(row["open_time"])
            close_raw = (row.get("close_time") or "").strip()
            step = interval_ms(interval)
            close_time = (_parse_time(close_raw) if close_raw else
                          datetime.fromtimestamp(
                              open_time.timestamp() + step / 1000.0, tz=timezone.utc))
            candles.append(Kline(
                symbol=symbol, interval=interval, open_time=open_time,
                close_time=close_time, open=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]), close=float(row["close"]),
                volume=float(row.get("volume") or 0.0),
                quote_volume=float(row["quote_volume"]) if row.get("quote_volume") else None,
                trades=int(float(row["trades"])) if row.get("trades") else None,
                is_closed=(row.get("is_closed", "true").strip().lower()
                           not in ("false", "0", "no"))))
        except (KeyError, TypeError, ValueError) as exc:
            raise DataPreparationError(f"Invalid CSV candle at line {line}: {exc}") from exc
    return candles


def fetch_candles(
    *, symbol: str, interval: str, total_limit: int = 5_000,
    start_time: datetime | None = None, end_time: datetime | None = None,
) -> list[Kline]:
    """Fetch closed Binance USD-M Futures candles via Component 4 only."""
    from noone.market_data import BinanceFuturesCollector

    with BinanceFuturesCollector() as collector:
        history = collector.historical_klines(
            symbol, interval=interval, start_time=start_time, end_time=end_time,
            total_limit=total_limit, only_closed=True)
    if not history.candles:
        raise DataPreparationError(
            f"Component 4 returned no closed {symbol} {interval} Futures candles; "
            "check symbol, interval and requested date range")
    return list(history.candles)


def fetch_funding_history(
    *, symbol: str, start_time: datetime, end_time: datetime,
    total_limit: int = 100_000,
):
    """Fetch a full Component 4 ``FundingHistory`` with context.

    Collected with ``context_limit=3`` as required for strict
    historical funding validation. The returned history is passed
    unchanged to the trading environment; collection failures raise
    (they never fall back to constant funding). ``start_time`` and
    ``end_time`` must be the explicit range to cover — for
    evaluation this is the combined warm-up-plus-evaluation candle
    range, not the evaluation decision range alone.
    """
    from noone.market_data import BinanceFuturesCollector
    from noone.market_data.models import FundingHistory

    if start_time is None or end_time is None:
        raise DataPreparationError(
            "Historical funding requires an explicit start_time and "
            "end_time covering the candle range to be evaluated")
    if end_time < start_time:
        raise DataPreparationError(
            "Historical funding end_time is before start_time")
    with BinanceFuturesCollector() as collector:
        history = collector.historical_funding_rates(
            symbol, start_time=start_time, end_time=end_time,
            total_limit=total_limit, context_limit=3)
    if not isinstance(history, FundingHistory):
        raise DataPreparationError(
            "Component 4 did not return a FundingHistory for "
            f"{symbol} {start_time}..{end_time}")
    if history.symbol != symbol:
        raise DataPreparationError(
            f"Funding history symbol {history.symbol!r} does not "
            f"match requested symbol {symbol!r}")
    return history


# -- FundingHistory persistence (Stage 3 resume support) ---------------
#
# Versioned JSON artifact. Decimals are stored as strings and
# timestamps as ISO-8601 so reconstruction is exact. The checksum is
# SHA-256 over the canonical JSON encoding of the payload. No pickle
# or other unsafe deserialization is used.

FUNDING_HISTORY_ARTIFACT_SCHEMA_VERSION = 1


def _art_iso(value):
    return value.astimezone(timezone.utc).isoformat() if value else None


def _art_dec(value):
    return str(value) if value is not None else None


def _raw_payload(value, where: str):
    """Return ``value`` only if it is strictly JSON-native.

    Tuples, sets, Decimals, datetimes, non-string dict keys, and
    non-finite floats are rejected explicitly instead of being
    silently converted by the JSON encoder.
    """
    import math

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DataPreparationError(
                f"Cannot serialize FundingHistory: raw payload at "
                f"{where} contains a non-finite float, which is not "
                "strictly JSON-native")
        return value
    if isinstance(value, list):
        for item in value:
            _raw_payload(item, where)
        return value
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DataPreparationError(
                    f"Cannot serialize FundingHistory: raw payload at "
                    f"{where} contains a non-string object key "
                    f"{key!r}, which is not JSON-native")
            _raw_payload(item, where)
        return value
    raise DataPreparationError(
        f"Cannot serialize FundingHistory: raw payload at {where} "
        f"contains an unsupported {type(value).__name__} value; raw "
        "payloads must be JSON-native (objects, arrays, strings, "
        "numbers, booleans, or null)")


def _settlement_payload(s):
    return {"symbol": s.symbol, "funding_time": _art_iso(s.funding_time),
            "funding_rate": _art_dec(s.funding_rate),
            "mark_price": _art_dec(s.mark_price),
            "rate_type": s.rate_type,
            "received_at": _art_iso(s.received_at),
            "raw": _raw_payload(s.raw, "settlement.raw")}


def _malformed_payload(m):
    return {"symbol": m.symbol, "funding_time": _art_iso(m.funding_time),
            "reason": m.reason,
            "raw": _raw_payload(m.raw, "malformed_record.raw"),
            "record_symbol": m.record_symbol, "page_index": m.page_index,
            "page_start_ms": m.page_start_ms, "page_end_ms": m.page_end_ms,
            "record_index": m.record_index,
            "occurrence_count": m.occurrence_count,
            "occurrences": [
                {"page_index": o.page_index, "page_start_ms": o.page_start_ms,
                 "page_end_ms": o.page_end_ms, "record_index": o.record_index}
                for o in m.occurrences]}


def _context_payload(c):
    if c is None:
        return None
    return {"direction": c.direction,
            "settlements": [_settlement_payload(s) for s in c.settlements],
            "malformed_records": [_malformed_payload(m)
                                  for m in c.malformed_records],
            "requested_count": c.requested_count, "status": c.status,
            "pages_fetched": c.pages_fetched,
            "pagination_complete": c.pagination_complete,
            "pagination_termination": c.pagination_termination,
            "error": c.error}


def funding_history_to_payload(history) -> dict:
    """Serialize a full ``FundingHistory`` exactly (no metadata loss)."""
    return {
        "symbol": history.symbol,
        "settlements": [_settlement_payload(s) for s in history.settlements],
        "requested_start_time": _art_iso(history.requested_start_time),
        "requested_end_time": _art_iso(history.requested_end_time),
        "coverage_start_time": _art_iso(history.coverage_start_time),
        "coverage_end_time": _art_iso(history.coverage_end_time),
        "collected_at": _art_iso(history.collected_at),
        "pages_fetched": history.pages_fetched,
        "duplicates_removed": history.duplicates_removed,
        "conflicting_records": [{
            "symbol": c.symbol, "funding_time": _art_iso(c.funding_time),
            "rate_type": c.rate_type, "kept_rate": _art_dec(c.kept_rate),
            "kept_mark_price": _art_dec(c.kept_mark_price),
            "conflicting_rate": _art_dec(c.conflicting_rate),
            "conflicting_mark_price": _art_dec(c.conflicting_mark_price),
            "raw": _raw_payload(c.raw, "conflicting_record.raw")}
            for c in history.conflicting_records],
        "malformed_records": [_malformed_payload(m)
                              for m in history.malformed_records],
        "suspected_anomalies": [{
            "after_time": _art_iso(a.after_time),
            "before_time": _art_iso(a.before_time),
            "interval_seconds": a.interval_seconds,
            "reference_interval_seconds": a.reference_interval_seconds,
            "estimated_missing_count": a.estimated_missing_count,
            "acknowledged": a.acknowledged}
            for a in history.suspected_anomalies],
        "pagination_complete": history.pagination_complete,
        "pagination_termination": history.pagination_termination,
        "total_limit": history.total_limit,
        "page_limit": history.page_limit,
        "boundary_probe_limit_hits": [{
            "funding_time": _art_iso(h.funding_time),
            "returned_count": h.returned_count}
            for h in history.boundary_probe_limit_hits],
        "blind_spots": list(history.blind_spots),
        "interval_assessments": [{
            "symbol": a.symbol, "rate_type": a.rate_type,
            "previous_time": _art_iso(a.previous_time),
            "next_time": _art_iso(a.next_time),
            "observed_interval_seconds": a.observed_interval_seconds,
            "reference_interval_seconds": a.reference_interval_seconds,
            "estimated_interval_multiple": a.estimated_interval_multiple,
            "supporting_before_seconds": list(a.supporting_before_seconds),
            "supporting_after_seconds": list(a.supporting_after_seconds),
            "classification": a.classification, "reason": a.reason,
            "evidence_sufficient": a.evidence_sufficient,
            "acknowledged": a.acknowledged}
            for a in history.interval_assessments],
        "before_context": _context_payload(history.before_context),
        "after_context": _context_payload(history.after_context),
        "scoped_blind_spots": [{
            "boundary": s.boundary, "start_time": _art_iso(s.start_time),
            "end_time": _art_iso(s.end_time), "reason": s.reason,
            "context_requested": s.context_requested,
            "context_received": s.context_received,
            "pagination_incomplete": s.pagination_incomplete,
            "safety_limit_reached": s.safety_limit_reached,
            "api_failed": s.api_failed}
            for s in history.scoped_blind_spots],
    }


def _payload_checksum(payload: dict) -> str:
    import hashlib
    import json as _json

    canonical = _json.dumps(payload, sort_keys=True,
                            separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_funding_history(history, path: str | Path) -> dict:
    """Persist ``history`` as a versioned, checksummed JSON artifact."""
    import json as _json

    payload = funding_history_to_payload(history)
    checksum = _payload_checksum(payload)
    document = {"schema_version": FUNDING_HISTORY_ARTIFACT_SCHEMA_VERSION,
                "checksum": checksum, "payload": payload}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_json.dumps(document, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    return {"artifact_path": target.name,
            "artifact_schema_version": FUNDING_HISTORY_ARTIFACT_SCHEMA_VERSION,
            "artifact_checksum": checksum,
            "collection_identity": _art_iso(history.collected_at)}


def _bad(message: str) -> DataPreparationError:
    return DataPreparationError(f"Invalid funding history artifact: {message}")


def _req(obj, field_name, types, *, allow_none=False):
    if not isinstance(obj, dict) or field_name not in obj:
        raise _bad(f"missing required field {field_name!r}")
    value = obj[field_name]
    if value is None and allow_none:
        return None
    if isinstance(value, bool) and types is int:
        raise _bad(f"field {field_name!r} must be an integer")
    if not isinstance(value, types):
        raise _bad(f"field {field_name!r} has the wrong type")
    return value


def _present(obj, field_name):
    """Return a required field's value, which may legitimately be None."""
    if not isinstance(obj, dict) or field_name not in obj:
        raise _bad(f"missing required field {field_name!r}")
    return obj[field_name]


def _parse_time(value, field_name):
    if value is None:
        return None
    if not isinstance(value, str):
        raise _bad(f"field {field_name!r} must be an ISO timestamp or null")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _bad(f"field {field_name!r} is not a valid timestamp") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _parse_dec(value, field_name):
    if value is None:
        return None
    if not isinstance(value, str):
        raise _bad(f"field {field_name!r} must be a decimal string or null")
    from decimal import Decimal, InvalidOperation

    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise _bad(f"field {field_name!r} is not a valid decimal") from exc


def _settlement_from(data):
    from noone.market_data.models import FundingSettlement

    if not isinstance(data, dict):
        raise _bad("settlement entries must be objects")
    received = _parse_time(_req(data, "received_at", str, allow_none=True),
                           "received_at")
    return FundingSettlement(
        symbol=_req(data, "symbol", str),
        funding_time=_parse_time(
            _req(data, "funding_time", str, allow_none=True), "funding_time"),
        funding_rate=_parse_dec(
            _req(data, "funding_rate", str, allow_none=True), "funding_rate"),
        mark_price=_parse_dec(
            _req(data, "mark_price", str, allow_none=True), "mark_price"),
        rate_type=_req(data, "rate_type", str, allow_none=True),
        received_at=received or datetime.now(timezone.utc),
        raw=_present(data, "raw"))


def _malformed_from(data):
    from noone.market_data.models import (
        FundingMalformedOccurrence,
        FundingMalformedRecord,
    )

    if not isinstance(data, dict):
        raise _bad("malformed record entries must be objects")
    occurrences = _req(data, "occurrences", list)
    return FundingMalformedRecord(
        symbol=_req(data, "symbol", str),
        funding_time=_parse_time(
            _req(data, "funding_time", str, allow_none=True), "funding_time"),
        reason=_req(data, "reason", str), raw=_present(data, "raw"),
        record_symbol=_req(data, "record_symbol", str, allow_none=True),
        page_index=_req(data, "page_index", int, allow_none=True),
        page_start_ms=_req(data, "page_start_ms", int, allow_none=True),
        page_end_ms=_req(data, "page_end_ms", int, allow_none=True),
        record_index=_req(data, "record_index", int, allow_none=True),
        occurrence_count=_req(data, "occurrence_count", int),
        occurrences=tuple(FundingMalformedOccurrence(
            page_index=_req(o, "page_index", int, allow_none=True),
            page_start_ms=_req(o, "page_start_ms", int, allow_none=True),
            page_end_ms=_req(o, "page_end_ms", int, allow_none=True),
            record_index=_req(o, "record_index", int, allow_none=True))
            for o in occurrences))


def _context_from(data):
    if data is None:
        return None
    from noone.market_data.models import FundingContext

    if not isinstance(data, dict):
        raise _bad("funding context must be an object or null")
    return FundingContext(
        direction=_req(data, "direction", str),
        settlements=tuple(_settlement_from(s)
                          for s in _req(data, "settlements", list)),
        malformed_records=tuple(_malformed_from(m) for m in _req(
            data, "malformed_records", list)),
        requested_count=_req(data, "requested_count", int),
        status=_req(data, "status", str),
        pages_fetched=_req(data, "pages_fetched", int),
        pagination_complete=_req(data, "pagination_complete", bool),
        pagination_termination=_req(data, "pagination_termination", str),
        error=_req(data, "error", str, allow_none=True))


def funding_history_from_payload(payload) -> "object":
    """Reconstruct a ``FundingHistory`` from a validated payload."""
    from noone.market_data.models import (
        BoundaryProbeLimitHit,
        FundingBlindSpot,
        FundingConflict,
        FundingHistory,
        FundingIntervalAnomaly,
        FundingIntervalAssessment,
    )

    if not isinstance(payload, dict):
        raise _bad("payload must be an object")
    collected = _parse_time(_req(payload, "collected_at", str), "collected_at")
    if collected is None:
        raise _bad("collected_at is required")
    return FundingHistory(
        symbol=_req(payload, "symbol", str),
        settlements=tuple(_settlement_from(s)
                          for s in _req(payload, "settlements", list)),
        requested_start_time=_parse_time(_req(
            payload, "requested_start_time", str, allow_none=True),
            "requested_start_time"),
        requested_end_time=_parse_time(_req(
            payload, "requested_end_time", str, allow_none=True),
            "requested_end_time"),
        coverage_start_time=_parse_time(_req(
            payload, "coverage_start_time", str, allow_none=True),
            "coverage_start_time"),
        coverage_end_time=_parse_time(_req(
            payload, "coverage_end_time", str, allow_none=True),
            "coverage_end_time"),
        collected_at=collected,
        pages_fetched=_req(payload, "pages_fetched", int),
        duplicates_removed=_req(payload, "duplicates_removed", int),
        conflicting_records=tuple(FundingConflict(
            symbol=_req(c, "symbol", str),
            funding_time=_parse_time(_req(
                c, "funding_time", str, allow_none=True), "funding_time"),
            rate_type=_req(c, "rate_type", str, allow_none=True),
            kept_rate=_parse_dec(_req(
                c, "kept_rate", str, allow_none=True), "kept_rate"),
            kept_mark_price=_parse_dec(_req(
                c, "kept_mark_price", str, allow_none=True),
                "kept_mark_price"),
            conflicting_rate=_parse_dec(_req(
                c, "conflicting_rate", str, allow_none=True),
                "conflicting_rate"),
            conflicting_mark_price=_parse_dec(_req(
                c, "conflicting_mark_price", str, allow_none=True),
                "conflicting_mark_price"),
            raw=_present(c, "raw")) for c in _req(
                payload, "conflicting_records", list)),
        malformed_records=tuple(_malformed_from(m) for m in _req(
            payload, "malformed_records", list)),
        suspected_anomalies=tuple(FundingIntervalAnomaly(
            after_time=_parse_time(_req(
                a, "after_time", str, allow_none=True), "after_time"),
            before_time=_parse_time(_req(
                a, "before_time", str, allow_none=True), "before_time"),
            interval_seconds=_req(a, "interval_seconds", (int, float)),
            reference_interval_seconds=_req(
                a, "reference_interval_seconds", (int, float)),
            estimated_missing_count=_req(
                a, "estimated_missing_count", int, allow_none=True),
            acknowledged=_req(a, "acknowledged", bool))
            for a in _req(payload, "suspected_anomalies", list)),
        pagination_complete=_req(payload, "pagination_complete", bool),
        pagination_termination=_req(payload, "pagination_termination", str),
        total_limit=_req(payload, "total_limit", int),
        page_limit=_req(payload, "page_limit", int),
        boundary_probe_limit_hits=tuple(BoundaryProbeLimitHit(
            funding_time=_parse_time(_req(
                h, "funding_time", str, allow_none=True), "funding_time"),
            returned_count=_req(h, "returned_count", int))
            for h in _req(payload, "boundary_probe_limit_hits", list)),
        blind_spots=tuple(_req(payload, "blind_spots", list)),
        interval_assessments=tuple(FundingIntervalAssessment(
            symbol=_req(a, "symbol", str),
            rate_type=_req(a, "rate_type", str, allow_none=True),
            previous_time=_parse_time(_req(
                a, "previous_time", str, allow_none=True), "previous_time"),
            next_time=_parse_time(_req(
                a, "next_time", str, allow_none=True), "next_time"),
            observed_interval_seconds=_req(
                a, "observed_interval_seconds", (int, float)),
            reference_interval_seconds=_req(
                a, "reference_interval_seconds", (int, float),
                allow_none=True),
            estimated_interval_multiple=_req(
                a, "estimated_interval_multiple", (int, float),
                allow_none=True),
            supporting_before_seconds=tuple(_req(
                a, "supporting_before_seconds", list)),
            supporting_after_seconds=tuple(_req(
                a, "supporting_after_seconds", list)),
            classification=_req(a, "classification", str),
            reason=_req(a, "reason", str),
            evidence_sufficient=_req(a, "evidence_sufficient", bool),
            acknowledged=_req(a, "acknowledged", bool))
            for a in _req(payload, "interval_assessments", list)),
        before_context=_context_from(_req(
            payload, "before_context", dict, allow_none=True)),
        after_context=_context_from(_req(
            payload, "after_context", dict, allow_none=True)),
        scoped_blind_spots=tuple(FundingBlindSpot(
            boundary=_req(s, "boundary", str),
            start_time=_parse_time(_req(
                s, "start_time", str, allow_none=True), "start_time"),
            end_time=_parse_time(_req(
                s, "end_time", str, allow_none=True), "end_time"),
            reason=_req(s, "reason", str),
            context_requested=_req(s, "context_requested", int),
            context_received=_req(s, "context_received", int),
            pagination_incomplete=_req(s, "pagination_incomplete", bool),
            safety_limit_reached=_req(s, "safety_limit_reached", bool),
            api_failed=_req(s, "api_failed", bool))
            for s in _req(payload, "scoped_blind_spots", list)))


def load_funding_history(path: str | Path):
    """Load and verify a persisted ``FundingHistory`` artifact.

    Verifies the schema version and SHA-256 checksum, validates the
    payload structure, and reconstructs the exact original history
    including its ``collected_at``. Corrupt, truncated, malformed, or
    unsupported artifacts raise ``DataPreparationError``.
    """
    import json as _json

    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise _bad(f"cannot read artifact {target}: {exc}") from exc
    try:
        document = _json.loads(text)
    except _json.JSONDecodeError as exc:
        raise _bad(f"artifact is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise _bad("artifact root must be an object")
    version = document.get("schema_version")
    if version != FUNDING_HISTORY_ARTIFACT_SCHEMA_VERSION:
        raise _bad(
            f"unsupported schema_version {version!r}; this build "
            f"supports version {FUNDING_HISTORY_ARTIFACT_SCHEMA_VERSION}")
    checksum = document.get("checksum")
    payload = document.get("payload")
    if not isinstance(checksum, str) or not isinstance(payload, dict):
        raise _bad("artifact must contain a checksum and payload object")
    if _payload_checksum(payload) != checksum:
        raise _bad(
            "checksum mismatch: artifact is corrupted or was modified "
            "after persistence; refusing to use it")
    return funding_history_from_payload(payload)
