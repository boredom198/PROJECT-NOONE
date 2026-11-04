"""Unit tests: historical funding collection (mocked exchange).

Deterministic fake transport only; live proof is a separate
public-data check. No API credentials, no network.
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from noone.market_data.binance_futures import (
    BinanceFuturesCollector,
    MarketDataError,
    parse_funding_settlements,
)
from noone.market_data.models import detect_funding_anomalies

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def frow(dt, rate="0.00010000", mark="100.5", symbol="BTCUSDT", rate_type=None):
    row = {"symbol": symbol, "fundingRate": rate,
           "fundingTime": int(dt.timestamp() * 1000), "markPrice": mark}
    if rate_type is not None:
        row["rateType"] = rate_type
    return row


def settlements_for(start_idx, count, step_hours=8):
    return [frow(BASE + timedelta(hours=step_hours * (start_idx + i)))
            for i in range(count)]


class FakeFundingExchange(BinanceFuturesCollector):
    """Param-aware fake of GET /fapi/v1/fundingRate semantics."""

    def __init__(self, rows):
        self._rows = sorted(rows, key=lambda r: r["fundingTime"])
        self.calls = []

    def _get(self, endpoint, params=None):
        assert endpoint == "/fapi/v1/fundingRate"
        params = dict(params or {})
        self.calls.append(params)
        rows = self._rows
        if params.get("startTime") is not None:
            rows = [r for r in rows if r["fundingTime"] >= params["startTime"]]
        if params.get("endTime") is not None:
            rows = [r for r in rows if r["fundingTime"] <= params["endTime"]]
        limit = int(params.get("limit", 100))
        if not rows or limit <= 0:
            return []
        return rows[:limit] if "startTime" in params else rows[-limit:]


class FakePages(BinanceFuturesCollector):
    def __init__(self, pages):
        self._pages = list(pages)
        self.calls = []

    def _get(self, endpoint, params=None):
        self.calls.append(dict(params or {}))
        return self._pages.pop(0) if self._pages else []


def _times(hist):
    return [s.funding_time for s in hist.settlements]


def _expected(*idx, step_hours=8):
    return [BASE + timedelta(hours=step_hours * i) for i in idx]


def _ms(dt):
    return int(dt.timestamp() * 1000)


# -- Parser ------------------------------------------------------------

def test_parser_decimal_zero_negative_and_rate_type():
    settlements, malformed = parse_funding_settlements("BTCUSDT", [
        frow(BASE, rate="0.00000000", mark="67234.10000000"),
        frow(BASE + timedelta(hours=8), rate="-0.00375000",
             mark="67000.5", rate_type="Regular"),
    ])
    assert malformed == []
    assert settlements[0].funding_rate == Decimal("0.00000000")
    assert settlements[0].mark_price == Decimal("67234.10000000")
    assert isinstance(settlements[0].funding_rate, Decimal)
    assert settlements[1].funding_rate == Decimal("-0.00375000")
    assert settlements[1].rate_type == "Regular"


def test_parser_malformed_reported_separately():
    settlements, malformed = parse_funding_settlements("BTCUSDT", [
        frow(BASE),
        {"symbol": "BTCUSDT", "fundingRate": "0.0001",
         "fundingTime": _ms(BASE + timedelta(hours=8))},  # no mark
        frow(BASE + timedelta(hours=16), mark="0"),  # invalid mark
        {"symbol": "BTCUSDT", "fundingTime": _ms(BASE)},  # no rate
        {"symbol": "ETHUSDT", "fundingRate": "0.0001",
         "fundingTime": _ms(BASE), "markPrice": "100"},  # wrong symbol
        "not-a-record",
    ])
    assert [s.funding_time for s in settlements] == [BASE]
    reasons = [m.reason for m in malformed]
    assert reasons.count("missing_or_invalid_mark_price") == 2
    assert "missing_or_invalid_funding_rate" in reasons
    assert "symbol_mismatch" in reasons
    assert "record_not_object" in reasons


# -- Pagination --------------------------------------------------------

def test_explicit_start_smaller_than_one_page():
    fake = FakeFundingExchange(settlements_for(0, 10))
    hist = fake.historical_funding_rates(
        "BTCUSDT", start_time=BASE, total_limit=3, page_limit=10)
    assert _times(hist) == _expected(0, 1, 2)
    assert hist.pages_fetched == 1
    assert fake.calls[0]["startTime"] == _ms(BASE)


def test_explicit_start_larger_than_one_page():
    fake = FakeFundingExchange(settlements_for(0, 10))
    hist = fake.historical_funding_rates(
        "BTCUSDT", start_time=BASE, total_limit=5, page_limit=2)
    assert _times(hist) == _expected(0, 1, 2, 3, 4)
    # Inclusive boundary cursors overlap each page boundary by one
    # timestamp (de-duplicated), plus one full-timestamp probe.
    assert hist.pages_fetched == 6
    assert hist.duplicates_removed == 4
    starts = [c["startTime"] for c in fake.calls]
    assert starts == [_ms(BASE), _ms(BASE + timedelta(hours=8)),
                      _ms(BASE + timedelta(hours=16)),
                      _ms(BASE + timedelta(hours=24)),
                      _ms(BASE + timedelta(hours=24)),
                      _ms(BASE + timedelta(hours=24)) + 1]


def test_omitted_start_smaller_than_one_page():
    fake = FakeFundingExchange(settlements_for(0, 10))
    hist = fake.historical_funding_rates("BTCUSDT", total_limit=3, page_limit=10)
    assert _times(hist) == _expected(7, 8, 9)
    assert hist.pages_fetched == 1 and "startTime" not in fake.calls[0]


def test_omitted_start_larger_than_one_page():
    fake = FakeFundingExchange(settlements_for(0, 10))
    hist = fake.historical_funding_rates("BTCUSDT", total_limit=5, page_limit=2)
    assert _times(hist) == _expected(5, 6, 7, 8, 9)
    # Inclusive boundary cursors overlap each boundary (de-duplicated)
    # plus one bounded full-timestamp probe (startTime == endTime).
    assert hist.pages_fetched == 6
    assert hist.duplicates_removed == 4
    paging = [c for c in fake.calls
              if not (c.get("startTime") is not None
                      and c.get("startTime") == c.get("endTime"))]
    assert all("startTime" not in c for c in paging)
    ends = [c.get("endTime") for c in paging]
    assert ends[0] is None
    assert ends[1] == _ms(BASE + timedelta(hours=64))
    assert ends[2] == _ms(BASE + timedelta(hours=56))
    assert ends[3] == _ms(BASE + timedelta(hours=48))
    assert ends[4] == _ms(BASE + timedelta(hours=48)) - 1


def test_exactly_one_full_page_both_modes():
    for kwargs, want in (({"start_time": BASE}, _expected(0, 1)),
                         ({}, _expected(8, 9))):
        fake = FakeFundingExchange(settlements_for(0, 10))
        hist = fake.historical_funding_rates(
            "BTCUSDT", total_limit=2, page_limit=2, **kwargs)
        assert _times(hist) == want and hist.pages_fetched == 1


def test_fewer_available_and_empty_response():
    for kwargs in ({"start_time": BASE}, {}):
        fake = FakeFundingExchange(settlements_for(0, 3))
        hist = fake.historical_funding_rates(
            "BTCUSDT", total_limit=20, page_limit=4, **kwargs)
        assert _times(hist) == _expected(0, 1, 2) and hist.pages_fetched == 1
        empty = FakeFundingExchange([]).historical_funding_rates(
            "BTCUSDT", total_limit=5, page_limit=2, **kwargs)
        assert empty.settlements == () and empty.pages_fetched == 1
        assert empty.pagination_complete is True


def test_duplicates_across_page_boundaries():
    fwd = FakePages([settlements_for(0, 3), settlements_for(2, 3)])
    hist = fwd.historical_funding_rates(
        "BTCUSDT", start_time=BASE, total_limit=5, page_limit=3)
    assert _times(hist) == _expected(0, 1, 2, 3, 4)
    assert hist.duplicates_removed == 1 and hist.pages_fetched == 2
    back = FakePages([settlements_for(7, 3), settlements_for(5, 3)])
    hist = back.historical_funding_rates("BTCUSDT", total_limit=5, page_limit=3)
    assert _times(hist) == _expected(5, 6, 7, 8, 9)
    assert hist.duplicates_removed == 1


def test_conflicting_duplicate_reported_not_silently_picked():
    page1 = settlements_for(0, 2)
    conflicting = frow(BASE + timedelta(hours=8), rate="0.00020000")
    fake = FakePages([page1, [conflicting] + settlements_for(2, 1)])
    hist = fake.historical_funding_rates(
        "BTCUSDT", start_time=BASE, total_limit=3, page_limit=2)
    assert len(hist.settlements) == 3
    assert len(hist.conflicting_records) == 1
    conflict = hist.conflicting_records[0]
    assert conflict.kept_rate == Decimal("0.00010000")
    assert conflict.conflicting_rate == Decimal("0.00020000")
    assert hist.settlements[1].funding_rate == Decimal("0.00010000")


def test_chronological_ordering_and_rate_type_identity():
    fake = FakeFundingExchange(settlements_for(0, 10))
    hist = fake.historical_funding_rates("BTCUSDT", total_limit=6, page_limit=2)
    times = _times(hist)
    assert times == _expected(4, 5, 6, 7, 8, 9) and times == sorted(times)
    same_time = [frow(BASE, rate_type="Regular"), frow(BASE, rate_type="Special")]
    hist2 = FakePages([same_time]).historical_funding_rates(
        "BTCUSDT", start_time=BASE, total_limit=5, page_limit=10)
    assert len(hist2.settlements) == 2  # distinct rate types coexist


def test_malformed_records_in_collector_and_invalid_mark():
    rows = settlements_for(0, 2) + [
        {"symbol": "BTCUSDT", "fundingRate": "0.0001",
         "fundingTime": _ms(BASE + timedelta(hours=16))}]  # missing mark
    hist = FakeFundingExchange(rows).historical_funding_rates(
        "BTCUSDT", start_time=BASE, total_limit=5, page_limit=10)
    assert _times(hist) == _expected(0, 1)
    assert len(hist.malformed_records) == 1
    assert hist.malformed_records[0].reason == "missing_or_invalid_mark_price"


def test_repeated_page_terminates_and_api_error_propagates():
    class Repeating(BinanceFuturesCollector):
        def __init__(self):
            self.calls = 0

        def _get(self, endpoint, params=None):
            self.calls += 1
            if self.calls > 10:
                raise AssertionError("pagination did not terminate")
            return settlements_for(0, 2)

    fake = Repeating()
    hist = fake.historical_funding_rates("BTCUSDT", total_limit=10, page_limit=2)
    assert len(hist.settlements) == 2 and hist.pages_fetched <= 2
    assert hist.pagination_complete is False
    assert hist.pagination_termination == "repeated_page"

    class Failing(BinanceFuturesCollector):
        def _get(self, endpoint, params=None):
            raise MarketDataError("boom", endpoint=endpoint)

    with pytest.raises(MarketDataError, match="boom"):
        Failing.__new__(Failing).historical_funding_rates("BTCUSDT")


# -- Anomalies, acknowledgement, coverage ------------------------------

def _settlements_at(hours):
    from noone.market_data.models import FundingSettlement
    return tuple(FundingSettlement(
        symbol="BTCUSDT", funding_time=BASE + timedelta(hours=h),
        funding_rate=Decimal("0.0001"), mark_price=Decimal("100"))
        for h in hours)


def test_anomaly_detection_never_assumes_eight_hours():
    assert detect_funding_anomalies(_settlements_at([0, 8, 16, 24])) == ()
    assert detect_funding_anomalies(_settlements_at([0, 4, 8, 12])) == ()
    assert detect_funding_anomalies(_settlements_at([0, 1, 2, 3])) == ()
    # Transition to a shorter interval is not a suspected gap.
    assert detect_funding_anomalies(_settlements_at([0, 8, 16, 20, 24])) == ()
    missing = detect_funding_anomalies(_settlements_at([0, 8, 16, 32, 40]))
    assert len(missing) == 1
    assert missing[0].estimated_missing_count == 1
    assert missing[0].acknowledged is False
    assert missing[0].reference_interval_seconds == 8 * 3600


def test_acknowledgement_explicit_preserves_details_and_conflicts():
    rows = [frow(BASE + timedelta(hours=h)) for h in (0, 8, 16, 32, 40)]
    conflict = frow(BASE + timedelta(hours=8), rate="0.00090000")
    fake = FakePages([rows[:3], [conflict] + rows[3:]])
    hist = fake.historical_funding_rates(
        "BTCUSDT", start_time=BASE, total_limit=10, page_limit=3)
    assert len(hist.suspected_anomalies) == 1
    assert len(hist.conflicting_records) == 1
    with pytest.raises(ValueError, match="Unknown funding anomalies"):
        hist.acknowledge_anomalies([type(hist.suspected_anomalies[0])(
            after_time=BASE, before_time=BASE, interval_seconds=1,
            reference_interval_seconds=1, estimated_missing_count=None)])
    acked = hist.acknowledge_anomalies(hist.suspected_anomalies)
    assert acked.suspected_anomalies[0].acknowledged is True
    assert acked.acknowledged_anomalies == acked.suspected_anomalies
    assert acked.unacknowledged_anomalies == ()
    # Original details and unrelated blocking records are preserved.
    assert acked.suspected_anomalies[0].estimated_missing_count == 1
    assert len(acked.conflicting_records) == 1
    assert hist.suspected_anomalies[0].acknowledged is False  # original intact
    meta = acked.metadata()
    assert meta["acknowledged_anomalies"][0]["after_time"].startswith(
        "2026-01-01T16:00")
    assert meta["acknowledged_anomalies"][0]["before_time"].startswith(
        "2026-01-02T08:00")
    assert meta["unacknowledged_anomaly_count"] == 0


def test_coverage_metadata_distinguishes_requested_and_observed():
    start = BASE + timedelta(hours=8)
    end = BASE + timedelta(hours=32)
    fake = FakeFundingExchange(settlements_for(0, 10))
    hist = fake.historical_funding_rates(
        "BTCUSDT", start_time=start, end_time=end,
        total_limit=100, page_limit=2)
    meta = hist.metadata()
    assert meta["requested_start_time"].startswith("2026-01-01T08:00")
    assert meta["observed_start_time"].startswith("2026-01-01T08:00")
    assert _times(hist) == [BASE + timedelta(hours=h) for h in (8, 16, 24, 32)]
    assert hist.covers_period(start, end) is True
    assert hist.covers_period(BASE, end) is False  # before requested start
    assert hist.pagination_complete is True
    # First/last events alone do not widen coverage beyond the request.
    recent = FakeFundingExchange(settlements_for(0, 10)) \
        .historical_funding_rates("BTCUSDT", total_limit=3, page_limit=10)
    assert recent.covers_period(BASE, BASE + timedelta(hours=72)) is False


# -- Review findings: same-timestamp boundaries, coverage, malformed --

def test_forward_same_timestamp_split_across_pages():
    t8 = BASE + timedelta(hours=8)
    rows = [frow(BASE),
            frow(t8, rate="0.00010000", rate_type="Regular"),
            frow(t8, rate="0.00020000", rate_type="Special"),
            frow(BASE + timedelta(hours=16))]
    fake = FakeFundingExchange(rows)
    hist = fake.historical_funding_rates(
        "BTCUSDT", start_time=BASE, total_limit=4, page_limit=2)
    assert _times(hist) == [BASE, t8, t8, BASE + timedelta(hours=16)]
    at_t = [s for s in hist.settlements if s.funding_time == t8]
    assert {s.rate_type for s in at_t} == {"Regular", "Special"}
    assert hist.duplicates_removed >= 1  # boundary overlap de-duplicated
    assert any(c.get("startTime") == c.get("endTime") == _ms(t8)
               for c in fake.calls)  # full-timestamp probe happened


def test_backward_same_timestamp_split_across_pages():
    t8 = BASE + timedelta(hours=8)
    rows = [frow(BASE),
            frow(t8, rate="0.00010000", rate_type="Regular"),
            frow(t8, rate="0.00020000", rate_type="Special"),
            frow(BASE + timedelta(hours=16)),
            frow(BASE + timedelta(hours=24))]
    fake = FakeFundingExchange(rows)
    hist = fake.historical_funding_rates("BTCUSDT", total_limit=5, page_limit=2)
    times = _times(hist)
    assert times == [BASE, t8, t8, BASE + timedelta(hours=16),
                     BASE + timedelta(hours=24)]
    at_t = [s for s in hist.settlements if s.funding_time == t8]
    assert {s.rate_type for s in at_t} == {"Regular", "Special"}
    assert hist.duplicates_removed >= 1
    assert any(c.get("startTime") == c.get("endTime") == _ms(t8)
               for c in fake.calls)


def test_covers_period_requires_complete_pagination():
    class Repeating(BinanceFuturesCollector):
        def _get(self, endpoint, params=None):
            return settlements_for(0, 2)

    hist = Repeating.__new__(Repeating).historical_funding_rates(
        "BTCUSDT", total_limit=10, page_limit=2)
    assert hist.pagination_complete is False
    assert hist.covers_period(BASE, BASE + timedelta(days=365)) is False
    meta = hist.metadata()
    assert meta["pagination_complete"] is False
    assert meta["pagination_termination"] == "repeated_page"


def test_acknowledgement_does_not_change_coverage_or_hide_records():
    from decimal import Decimal as _D

    from noone.market_data.models import (
        FundingConflict, FundingHistory, FundingIntervalAnomaly,
        FundingMalformedRecord, FundingSettlement)
    settlements = tuple(FundingSettlement(
        symbol="BTCUSDT", funding_time=BASE + timedelta(hours=h),
        funding_rate=_D("0.0001"), mark_price=_D("100"))
        for h in (0, 8, 16, 32, 40))
    anomaly = FundingIntervalAnomaly(
        after_time=BASE + timedelta(hours=16),
        before_time=BASE + timedelta(hours=32),
        interval_seconds=16 * 3600, reference_interval_seconds=8 * 3600,
        estimated_missing_count=1)
    hist = FundingHistory(
        symbol="BTCUSDT", settlements=settlements,
        requested_start_time=BASE, requested_end_time=BASE + timedelta(hours=48),
        coverage_start_time=BASE,
        coverage_end_time=BASE + timedelta(hours=48),
        collected_at=BASE + timedelta(hours=49), pages_fetched=2,
        duplicates_removed=0,
        conflicting_records=(FundingConflict(
            symbol="BTCUSDT", funding_time=BASE + timedelta(hours=8),
            rate_type=None, kept_rate=_D("0.0001"), kept_mark_price=_D("100"),
            conflicting_rate=_D("0.0002"), conflicting_mark_price=_D("100")),),
        malformed_records=(FundingMalformedRecord(
            symbol="BTCUSDT", funding_time=None,
            reason="missing_or_invalid_funding_time", raw={}),),
        suspected_anomalies=(anomaly,), pagination_complete=True,
        pagination_termination="range_exhausted",
        total_limit=100, page_limit=1000)
    period = (BASE, BASE + timedelta(hours=48))
    assert hist.covers_period(*period) is True
    acked = hist.acknowledge_anomalies([anomaly])
    assert acked.covers_period(*period) is True  # unchanged by ack
    assert acked.coverage_start_time == hist.coverage_start_time
    assert acked.coverage_end_time == hist.coverage_end_time
    assert acked.pagination_complete is True
    meta = acked.metadata()
    assert meta["suspected_anomalies"][0]["acknowledged"] is True
    assert meta["suspected_anomalies"][0]["estimated_missing_count"] == 1
    assert len(meta["conflicting_records"]) == 1  # still visible
    assert len(meta["malformed_records"]) == 1  # still visible
    assert meta["malformed_without_timestamp_count"] == 1


def test_observed_range_alone_does_not_widen_coverage():
    fake = FakeFundingExchange(settlements_for(0, 10))
    hist = fake.historical_funding_rates(
        "BTCUSDT", start_time=BASE, total_limit=2, page_limit=2)
    assert hist.observed_start_time == BASE
    # Coverage ends at the last observed event under total_limit;
    # later periods are not covered despite the symbol having data.
    assert hist.covers_period(BASE, BASE + timedelta(hours=72)) is False


def test_malformed_without_timestamp_full_page_is_incomplete():
    no_time = {"symbol": "BTCUSDT", "fundingRate": "0.0001",
               "markPrice": "100"}
    fake = FakePages([[no_time, "junk-record", dict(no_time)]])
    hist = fake.historical_funding_rates(
        "BTCUSDT", start_time=BASE, total_limit=5, page_limit=3)
    assert hist.settlements == ()
    assert hist.pagination_complete is False
    assert hist.pagination_termination == "no_usable_timestamp"
    assert hist.covers_period(BASE, BASE + timedelta(hours=1)) is False
    assert len(hist.malformed_records) >= 2  # details preserved
    assert hist.metadata()["malformed_without_timestamp_count"] >= 2


def test_malformed_without_timestamp_mixed_page_still_reported():
    no_time = {"symbol": "BTCUSDT", "fundingRate": "0.0001",
               "markPrice": "100"}
    fake = FakePages([[frow(BASE), no_time,
                       frow(BASE + timedelta(hours=8))]])
    hist = fake.historical_funding_rates(
        "BTCUSDT", start_time=BASE, total_limit=5, page_limit=10)
    assert _times(hist) == _expected(0, 1)
    assert len(hist.malformed_records) == 1
    assert hist.malformed_records[0].funding_time is None
    assert hist.metadata()["malformed_without_timestamp_count"] == 1


def test_page_limit_never_exceeds_exchange_maximum():
    fake = FakeFundingExchange(settlements_for(0, 5))
    hist = fake.historical_funding_rates(
        "BTCUSDT", start_time=BASE, total_limit=5, page_limit=5000)
    assert len(hist.settlements) == 5
    assert all(c["limit"] <= 1000 for c in fake.calls)


# -- Stage 1 hardening: probe ambiguity + completeness ------------------

def _probe_limit_history():
    t8 = BASE + timedelta(hours=8)
    rows = [frow(BASE)]
    rows += [frow(t8, rate_type=f"Type{i:04d}") for i in range(1000)]
    rows += [frow(BASE + timedelta(hours=16))]
    fake = FakeFundingExchange(rows)
    hist = fake.historical_funding_rates(
        "BTCUSDT", start_time=BASE, total_limit=5000, page_limit=2)
    return hist, t8


def test_s1_probe_exactly_1000_records_flagged():
    hist, t8 = _probe_limit_history()
    assert len(hist.boundary_probe_limit_hits) == 1
    hit = hist.boundary_probe_limit_hits[0]
    assert hit.funding_time == t8 and hit.returned_count == 1000
    assert len([s for s in hist.settlements if s.funding_time == t8]) == 1000


def test_s1_probe_ambiguity_in_metadata_and_prevents_complete():
    hist, t8 = _probe_limit_history()
    meta = hist.metadata()
    assert meta["boundary_probe_limit_hits"] == [{
        "funding_time": t8.isoformat(), "returned_count": 1000}]
    assert meta["completeness_status"] == "uncertain"
    assert hist.completeness_status == "uncertain"  # never complete


def _status_history(**overrides):
    from noone.market_data.models import FundingHistory
    values = dict(
        symbol="BTCUSDT", settlements=_settlements_at([0, 8, 16, 24]),
        requested_start_time=BASE,
        requested_end_time=BASE + timedelta(hours=24),
        coverage_start_time=BASE,
        coverage_end_time=BASE + timedelta(hours=24),
        collected_at=BASE + timedelta(hours=25), pages_fetched=1,
        duplicates_removed=0, conflicting_records=(), malformed_records=(),
        suspected_anomalies=(), pagination_complete=True,
        pagination_termination="range_exhausted",
        total_limit=100, page_limit=1000)
    values.update(overrides)
    return FundingHistory(**values)


def _anomaly():
    from noone.market_data.models import FundingIntervalAnomaly
    return FundingIntervalAnomaly(
        after_time=BASE + timedelta(hours=16),
        before_time=BASE + timedelta(hours=32),
        interval_seconds=16 * 3600, reference_interval_seconds=8 * 3600,
        estimated_missing_count=1)


def test_s1_probe_ambiguity_survives_acknowledgement():
    from noone.market_data.models import BoundaryProbeLimitHit
    hist = _status_history(
        suspected_anomalies=(_anomaly(),),
        boundary_probe_limit_hits=(BoundaryProbeLimitHit(
            funding_time=BASE + timedelta(hours=8), returned_count=1000),))
    acked = hist.acknowledge_anomalies(hist.suspected_anomalies)
    assert acked.boundary_probe_limit_hits == hist.boundary_probe_limit_hits
    assert acked.metadata()["boundary_probe_limit_hits"][0][
        "returned_count"] == 1000
    assert acked.completeness_status == "uncertain"


def test_s1_pagination_failure_is_incomplete():
    hist = _status_history(pagination_complete=False,
                           pagination_termination="repeated_page")
    assert hist.completeness_status == "incomplete"


def test_s1_conflicting_records_are_incomplete():
    from noone.market_data.models import FundingConflict
    conflict = FundingConflict(
        symbol="BTCUSDT", funding_time=BASE + timedelta(hours=8),
        rate_type=None, kept_rate=Decimal("0.0001"),
        kept_mark_price=Decimal("100"),
        conflicting_rate=Decimal("0.0002"),
        conflicting_mark_price=Decimal("100"))
    hist = _status_history(conflicting_records=(conflict,))
    assert hist.completeness_status == "incomplete"


def test_s1_incomplete_takes_priority_over_uncertain():
    from noone.market_data.models import BoundaryProbeLimitHit
    hist = _status_history(
        pagination_complete=False,
        suspected_anomalies=(_anomaly(),),
        boundary_probe_limit_hits=(BoundaryProbeLimitHit(
            funding_time=BASE, returned_count=1000),))
    assert hist.completeness_status == "incomplete"


def test_s1_acknowledged_anomaly_remains_uncertain():
    hist = _status_history(suspected_anomalies=(_anomaly(),))
    assert hist.completeness_status == "uncertain"
    acked = hist.acknowledge_anomalies(hist.suspected_anomalies)
    assert acked.suspected_anomalies[0].acknowledged is True
    assert acked.completeness_status == "uncertain"  # ack != certainty


def test_s1_malformed_timestamp_incomplete_timestampless_uncertain():
    from noone.market_data.models import FundingMalformedRecord
    with_time = _status_history(malformed_records=(
        FundingMalformedRecord(symbol="BTCUSDT", funding_time=BASE,
                               reason="missing_or_invalid_mark_price"),))
    assert with_time.completeness_status == "incomplete"
    no_time = _status_history(malformed_records=(
        FundingMalformedRecord(symbol="BTCUSDT", funding_time=None,
                               reason="missing_or_invalid_funding_time"),))
    assert no_time.completeness_status == "uncertain"


def test_s1_clean_history_is_complete():
    hist = _status_history()
    assert hist.completeness_status == "complete"
    assert hist.metadata()["completeness_status"] == "complete"
    assert hist.metadata()["boundary_probe_limit_hits"] == []


# -- Stage 2: malformed provenance, identity, per-period status --------

def _bad_mark(dt, rate="0.00010000", **extra):
    row = {"symbol": "BTCUSDT", "fundingRate": rate,
           "fundingTime": _ms(dt)}
    row.update(extra)
    return row


def _no_time(**extra):
    row = {"symbol": "BTCUSDT", "fundingRate": "0.0001", "markPrice": "100"}
    row.update(extra)
    return row


def test_s2_malformed_with_timestamp_retains_page_provenance():
    end = BASE + timedelta(hours=24)
    fake = FakePages([[frow(BASE), _bad_mark(BASE + timedelta(hours=8)),
                       frow(BASE + timedelta(hours=16))]])
    hist = fake.historical_funding_rates(
        "BTCUSDT", start_time=BASE, end_time=end,
        total_limit=5, page_limit=10)
    (record,) = hist.malformed_records
    assert record.funding_time == BASE + timedelta(hours=8)
    assert record.page_index == 0
    assert record.page_start_ms == _ms(BASE)
    assert record.page_end_ms == _ms(end)
    assert record.record_index == 1
    assert record.occurrence_count == 1
    assert record.occurrences[0].page_index == 0
    assert record.occurrences[0].record_index == 1


def test_s2_timestampless_retains_actual_request_window():
    end = BASE + timedelta(hours=24)
    fake = FakePages([[frow(BASE), _no_time(),
                       frow(BASE + timedelta(hours=8))]])
    hist = fake.historical_funding_rates(
        "BTCUSDT", start_time=BASE, end_time=end,
        total_limit=5, page_limit=10)
    (record,) = hist.malformed_records
    assert record.funding_time is None
    assert record.page_start_ms == _ms(BASE)
    assert record.page_end_ms == _ms(end)  # window sent, not inferred
    assert record.record_index == 1


def test_s2_distinct_payloads_same_reason_and_timestamp_separate():
    t8 = BASE + timedelta(hours=8)
    fake = FakePages([[_bad_mark(t8, rate="0.00010000"),
                       _bad_mark(t8, rate="0.00020000")]])
    hist = fake.historical_funding_rates(
        "BTCUSDT", start_time=BASE, total_limit=5, page_limit=10)
    assert len(hist.malformed_records) == 2
    assert {m.reason for m in hist.malformed_records} == {
        "missing_or_invalid_mark_price"}
    assert all(m.funding_time == t8 for m in hist.malformed_records)
    assert {m.raw["fundingRate"] for m in hist.malformed_records} == {
        "0.00010000", "0.00020000"}


def test_s2_exact_duplicate_overlap_deduped_all_occurrences_kept():
    bad = _bad_mark(BASE + timedelta(hours=8))
    fake = FakePages([[frow(BASE), bad],
                      [dict(bad), frow(BASE + timedelta(hours=16))]])
    hist = fake.historical_funding_rates(
        "BTCUSDT", start_time=BASE, total_limit=10, page_limit=2)
    assert _times(hist) == _expected(0, 2)  # 0h and 16h settlements
    (record,) = hist.malformed_records
    assert record.occurrence_count == 2
    assert [o.page_index for o in record.occurrences] == [0, 1]
    assert record.page_index == 0  # first provenance preserved


def test_s2_dict_key_ordering_not_a_distinct_identity():
    t8ms = _ms(BASE + timedelta(hours=8))
    bad1 = {"symbol": "BTCUSDT", "fundingRate": "0.0001",
            "fundingTime": t8ms}
    bad2 = {"fundingTime": t8ms, "fundingRate": "0.0001",
            "symbol": "BTCUSDT"}
    fake = FakePages([[frow(BASE), bad1],
                      [bad2, frow(BASE + timedelta(hours=16))]])
    hist = fake.historical_funding_rates(
        "BTCUSDT", start_time=BASE, total_limit=10, page_limit=2)
    assert len(hist.malformed_records) == 1
    assert hist.malformed_records[0].occurrence_count == 2


def test_s2_different_json_value_types_not_merged():
    t8 = BASE + timedelta(hours=8)
    fake = FakePages([[
        {"symbol": "BTCUSDT", "fundingRate": 0.0001,
         "fundingTime": _ms(t8)},
        {"symbol": "BTCUSDT", "fundingRate": "0.0001",
         "fundingTime": _ms(t8)}]])
    hist = fake.historical_funding_rates(
        "BTCUSDT", start_time=BASE, total_limit=5, page_limit=10)
    assert len(hist.malformed_records) == 2
    types = {type(m.raw["fundingRate"]) for m in hist.malformed_records}
    assert types == {float, str}


def test_s2_nested_raw_payload_preserved():
    nested = {"levels": [1, 2, {"flag": True, "note": None}]}
    fake = FakePages([[_bad_mark(BASE + timedelta(hours=8), extra=nested)]])
    hist = fake.historical_funding_rates(
        "BTCUSDT", start_time=BASE, total_limit=5, page_limit=10)
    (record,) = hist.malformed_records
    assert record.raw["extra"] == nested
    assert hist.metadata()["malformed_records"][0]["raw"]["extra"] == nested


def test_s2_metadata_exposes_payloads_provenance_and_bounds():
    end = BASE + timedelta(hours=24)
    bad_timed = _bad_mark(BASE + timedelta(hours=8))
    fake = FakePages([[frow(BASE), bad_timed, _no_time()]])
    hist = fake.historical_funding_rates(
        "BTCUSDT", start_time=BASE, end_time=end,
        total_limit=5, page_limit=10)
    meta = hist.metadata()
    assert meta["malformed_count"] == 2
    assert meta["malformed_with_timestamp_count"] == 1
    assert meta["malformed_without_timestamp_count"] == 1
    entries = {e["reason"]: e for e in meta["malformed_records"]}
    timed = next(e for e in meta["malformed_records"]
                 if e["has_usable_timestamp"])
    assert timed["raw"] == bad_timed
    assert timed["page_index"] == 0 and timed["occurrence_count"] == 1
    assert timed["possible_impact_window"] is None
    timeless = next(e for e in meta["malformed_records"]
                    if not e["has_usable_timestamp"])
    assert timeless["funding_time"] is None
    assert timeless["possible_impact_window"] == {
        "start_ms": _ms(BASE), "end_ms": _ms(end),
        "label": "possible_impact_bound_not_event_time"}
    assert entries  # reasons listed


def test_s2_timestampless_overlapping_period_is_uncertain():
    end = BASE + timedelta(hours=24)
    fake = FakePages([[frow(BASE), _no_time(),
                       frow(BASE + timedelta(hours=8))]])
    hist = fake.historical_funding_rates(
        "BTCUSDT", start_time=BASE, end_time=end,
        total_limit=5, page_limit=10)
    assert hist.completeness_status_for_period(BASE, end) == "uncertain"
    assert hist.completeness_status == "uncertain"


def test_s2_timestampless_outside_period_does_not_invalidate():
    from noone.market_data.models import (
        FundingMalformedOccurrence, FundingMalformedRecord)
    occurrence = FundingMalformedOccurrence(
        page_index=3, page_start_ms=_ms(BASE + timedelta(hours=48)),
        page_end_ms=_ms(BASE + timedelta(hours=72)), record_index=0)
    record = FundingMalformedRecord(
        symbol="BTCUSDT", funding_time=None,
        reason="missing_or_invalid_funding_time", raw=_no_time(),
        page_index=3, page_start_ms=occurrence.page_start_ms,
        page_end_ms=occurrence.page_end_ms, record_index=0,
        occurrence_count=1, occurrences=(occurrence,))
    hist = _status_history(malformed_records=(record,))
    period = (BASE, BASE + timedelta(hours=24))
    assert hist.completeness_status_for_period(*period) == "complete"
    # History-level status still reports the unresolved record.
    assert hist.completeness_status == "uncertain"


def test_s2_malformed_with_timestamp_inside_period_is_incomplete():
    from noone.market_data.models import FundingMalformedRecord
    record = FundingMalformedRecord(
        symbol="BTCUSDT", funding_time=BASE + timedelta(hours=8),
        reason="missing_or_invalid_mark_price", raw={},
        page_index=0, record_index=1)
    hist = _status_history(malformed_records=(record,))
    assert hist.completeness_status_for_period(
        BASE, BASE + timedelta(hours=24)) == "incomplete"
    assert hist.completeness_status_for_period(
        BASE + timedelta(hours=48),
        BASE + timedelta(hours=72)) == "complete"


def test_s2_acknowledgement_preserves_malformed_evidence():
    from noone.market_data.models import (
        FundingMalformedOccurrence, FundingMalformedRecord)
    occurrence = FundingMalformedOccurrence(
        page_index=0, page_start_ms=_ms(BASE), page_end_ms=None,
        record_index=1)
    record = FundingMalformedRecord(
        symbol="BTCUSDT", funding_time=None,
        reason="missing_or_invalid_funding_time", raw=_no_time(),
        page_index=0, page_start_ms=_ms(BASE), record_index=1,
        occurrence_count=2, occurrences=(occurrence, occurrence))
    hist = _status_history(suspected_anomalies=(_anomaly(),),
                           malformed_records=(record,))
    before = hist.completeness_status_for_period(
        BASE, BASE + timedelta(hours=24))
    acked = hist.acknowledge_anomalies(hist.suspected_anomalies)
    assert acked.malformed_records == hist.malformed_records
    assert acked.malformed_records[0].occurrence_count == 2
    assert acked.malformed_records[0].occurrences == (occurrence, occurrence)
    assert acked.completeness_status == hist.completeness_status
    assert acked.completeness_status_for_period(
        BASE, BASE + timedelta(hours=24)) == before == "uncertain"
    assert acked.metadata()["malformed_records"] == \
        hist.metadata()["malformed_records"]


# -- Stage 3: context collection ----------------------------------------

def _ctx_hist(page_limit=10, context_limit=3, start_idx=6, end_idx=11,
              rows=None, **kwargs):
    fake = FakeFundingExchange(
        rows if rows is not None else settlements_for(0, 20))
    start = BASE + timedelta(hours=8 * start_idx)
    end = BASE + timedelta(hours=8 * end_idx)
    hist = fake.historical_funding_rates(
        "BTCUSDT", start_time=start, end_time=end, total_limit=100,
        page_limit=page_limit, context_limit=context_limit, **kwargs)
    return hist, start, end


def test_s3_before_and_after_context_retrieved_separately():
    hist, start, end = _ctx_hist()
    assert _times(hist) == _expected(6, 7, 8, 9, 10, 11)
    assert [s.funding_time for s in hist.before_context.settlements] == \
        _expected(3, 4, 5)
    assert [s.funding_time for s in hist.after_context.settlements] == \
        _expected(12, 13, 14)
    assert hist.before_context.status == "sufficient"
    assert hist.after_context.status == "sufficient"
    # Context never merges into requested settlements or coverage.
    assert hist.metadata()["settlement_count"] == 6
    assert hist.requested_start_time == start
    assert hist.requested_end_time == end
    assert hist.coverage_start_time == start
    assert hist.covers_period(start, end) is True
    combined = (_times(hist)
                + [s.funding_time for s in hist.before_context.settlements]
                + [s.funding_time for s in hist.after_context.settlements])
    assert len(combined) == len(set(combined))  # no boundary overlap
    assert [s.funding_time for s in hist.before_context.settlements] == \
        sorted(s.funding_time for s in hist.before_context.settlements)


def test_s3_context_pagination():
    hist, _, _ = _ctx_hist(page_limit=2)
    assert hist.before_context.pages_fetched >= 2
    assert hist.before_context.received_count == 3


def test_s3_insufficient_context_before_and_after():
    hist, _, _ = _ctx_hist(start_idx=1, end_idx=10)
    assert hist.before_context.status == "insufficient"
    assert hist.before_context.received_count == 1
    (spot,) = [s for s in hist.scoped_blind_spots if s.boundary == "before"]
    assert spot.context_requested == 3 and spot.context_received == 1
    hist2, _, _ = _ctx_hist(start_idx=6, end_idx=18)
    assert hist2.after_context.status == "insufficient"
    assert hist2.after_context.received_count == 1
    (spot2,) = [s for s in hist2.scoped_blind_spots if s.boundary == "after"]
    assert spot2.context_received == 1


def test_s3_context_safety_limit():
    hist, _, _ = _ctx_hist(page_limit=1, context_max_pages=1)
    assert hist.before_context.status == "safety_limit"
    (spot,) = [s for s in hist.scoped_blind_spots if s.boundary == "before"]
    assert spot.safety_limit_reached is True


def test_s3_context_failure_preserves_requested_data():
    start = BASE + timedelta(hours=48)

    class ContextFail(FakeFundingExchange):
        def _get(self, endpoint, params=None):
            if (params or {}).get("endTime") == _ms(start) - 1:
                raise MarketDataError("ctx boom", endpoint=endpoint)
            return super()._get(endpoint, params)

    fake = ContextFail(settlements_for(0, 20))
    hist = fake.historical_funding_rates(
        "BTCUSDT", start_time=start, end_time=BASE + timedelta(hours=88),
        total_limit=100, page_limit=10, context_limit=3)
    assert _times(hist) == _expected(6, 7, 8, 9, 10, 11)  # requested intact
    assert hist.before_context.status == "failed"
    assert "ctx boom" in (hist.before_context.error or "")
    assert hist.after_context.status == "sufficient"
    (spot,) = [s for s in hist.scoped_blind_spots if s.boundary == "before"]
    assert spot.api_failed is True


def test_s3_context_malformed_provenance_and_separation():
    rows = settlements_for(0, 20) + [_bad_mark(BASE + timedelta(hours=32))]
    hist, _, _ = _ctx_hist(rows=rows)
    assert _times(hist) == _expected(6, 7, 8, 9, 10, 11)
    assert len(hist.before_context.malformed_records) == 1
    record = hist.before_context.malformed_records[0]
    assert record.funding_time == BASE + timedelta(hours=32)
    assert record.page_index is not None
    assert len(hist.malformed_records) == 0  # not in requested data


# -- Stage 3: two-sided anomaly detection --------------------------------

from noone.market_data.models import assess_funding_intervals  # noqa: E402


def _assess(hours, **kwargs):
    return assess_funding_intervals(_settlements_at(hours), **kwargs)


def _classes(hours, **kwargs):
    return [a.classification for a in _assess(hours, **kwargs)]


def test_s3_detection_stable_missing_transition_irregular():
    assert _assess([0, 8, 16, 24, 32]) == ()
    missing = _assess([0, 8, 16, 32, 40, 48])
    assert [a.classification for a in missing] == [
        "SUSPECTED_MISSING_SETTLEMENT"]
    assert missing[0].estimated_interval_multiple == 2.0
    assert missing[0].supporting_before_seconds == (28800.0, 28800.0)
    assert missing[0].supporting_after_seconds == (28800.0, 28800.0)
    assert missing[0].evidence_sufficient is True
    assert missing[0].reason
    assert [a.classification for a in _assess([0, 8, 16, 32, 48, 64])] == [
        "OBSERVED_INTERVAL_CHANGE"]
    assert [a.classification for a in _assess([0, 8, 16, 20, 24, 28])] == [
        "OBSERVED_INTERVAL_CHANGE"]
    assert [a.classification for a in _assess([0, 8, 16, 32, 44, 60])] == [
        "INSUFFICIENT_EVIDENCE"]


def test_s3_detection_insufficient_evidence_edges_and_multiple_gaps():
    assert _classes([0, 16, 24, 32, 40]) == ["INSUFFICIENT_EVIDENCE"]
    assert _classes([0, 8, 16, 32]) == ["INSUFFICIENT_EVIDENCE"]
    assert _classes([0, 8, 16, 32, 40, 56, 64]) == [
        "SUSPECTED_MISSING_SETTLEMENT", "SUSPECTED_MISSING_SETTLEMENT"]


def test_s3_detection_rate_types_never_mixed():
    from decimal import Decimal as _D

    from noone.market_data.models import FundingSettlement
    regular = [FundingSettlement(
        symbol="BTCUSDT", funding_time=BASE + timedelta(hours=h),
        funding_rate=_D("0.0001"), mark_price=_D("100"),
        rate_type="Regular") for h in (0, 8, 16, 32, 40)]
    special = [FundingSettlement(
        symbol="BTCUSDT", funding_time=BASE + timedelta(hours=h),
        funding_rate=_D("0.0001"), mark_price=_D("100"),
        rate_type="Special") for h in (4, 12, 20)]
    assessments = assess_funding_intervals(regular + special)
    assert assessments
    assert all(a.rate_type == "Regular" for a in assessments)
    assert assessments[0].classification == "SUSPECTED_MISSING_SETTLEMENT"
    # Unknown/special group alone: preserved, never suspected missing.
    assert [a for a in assess_funding_intervals(special)
            if a.classification == "SUSPECTED_MISSING_SETTLEMENT"] == []
    assert detect_funding_anomalies(special) == ()


def test_s3_detection_tolerance_within_and_outside():
    from noone.market_data.models import FundingSettlement
    from decimal import Decimal as _D

    def at(*deltas_s):
        times = [BASE]
        for d in deltas_s:
            times.append(times[-1] + timedelta(seconds=d))
        return tuple(FundingSettlement(
            symbol="BTCUSDT", funding_time=t, funding_rate=_D("0.0001"),
            mark_price=_D("100")) for t in times)

    within = assess_funding_intervals(at(28800, 28800, 57660, 28800))
    assert [a.classification for a in within] == [
        "SUSPECTED_MISSING_SETTLEMENT"]  # 60s over a 16h multiple: tolerated
    outside = assess_funding_intervals(at(28800, 28800, 61200, 28800))
    assert [a.classification for a in outside] == ["INSUFFICIENT_EVIDENCE"]
    jittered = assess_funding_intervals(at(28830, 28770, 28800, 28800))
    assert jittered == ()  # ordinary timing deviation, no anomaly


def test_s3_detection_no_fabrication_deterministic_exact_times():
    settlements = _settlements_at([0, 8, 16, 32, 40])
    first = assess_funding_intervals(settlements)
    second = assess_funding_intervals(settlements)
    assert first == second  # deterministic
    assert len(settlements) == 5  # nothing synthetic inserted
    (assessment,) = first
    assert assessment.previous_time == BASE + timedelta(hours=16)
    assert assessment.next_time == BASE + timedelta(hours=32)
    assert {s.funding_time for s in settlements} == {
        BASE + timedelta(hours=h) for h in (0, 8, 16, 32, 40)}


# -- Stage 3: completeness integration ------------------------------------

def test_s3_completeness_missing_inside_vs_outside_period():
    fake = FakeFundingExchange(settlements_for(0, 10))
    # Remove the idx-3 settlement (24h) to create a return-to-reference gap.
    rows = [r for i, r in enumerate(settlements_for(0, 10)) if i != 3]
    fake = FakeFundingExchange(rows)
    end = BASE + timedelta(hours=72)
    hist = fake.historical_funding_rates(
        "BTCUSDT", start_time=BASE, end_time=end,
        total_limit=100, page_limit=10)
    assert hist.completeness_status_for_period(BASE, end) == "uncertain"
    assert hist.completeness_status == "uncertain"
    # The suspicious gap (16h..32h) does not touch later periods;
    # pagination itself is complete, so they stay complete.
    later = (BASE + timedelta(hours=80), BASE + timedelta(hours=100))
    assert hist.completeness_status_for_period(*later) == "complete"
    inside = (BASE + timedelta(hours=40), BASE + timedelta(hours=48))
    assert hist.completeness_status_for_period(*inside) == "complete"


def test_s3_blind_spot_scoping_before_after_internal():
    hist, start, end = _ctx_hist(start_idx=1, end_idx=10)
    assert hist.completeness_status_for_period(start, end) == "uncertain"
    later = (BASE + timedelta(hours=40), BASE + timedelta(hours=64))
    assert hist.completeness_status_for_period(*later) == "complete"
    hist2, start2, end2 = _ctx_hist(start_idx=6, end_idx=18)
    assert hist2.completeness_status_for_period(start2, end2) == "uncertain"
    earlier = (BASE + timedelta(hours=40), BASE + timedelta(hours=64))
    assert hist2.completeness_status_for_period(*earlier) == "complete"

    from noone.market_data.models import FundingBlindSpot
    manual = _status_history(scoped_blind_spots=(FundingBlindSpot(
        boundary="internal", start_time=BASE + timedelta(hours=8),
        end_time=BASE + timedelta(hours=16), reason="test blind spot"),))
    assert manual.completeness_status_for_period(
        BASE, BASE + timedelta(hours=24)) == "uncertain"
    assert manual.completeness_status_for_period(
        BASE + timedelta(hours=48),
        BASE + timedelta(hours=72)) == "complete"


def test_s3_transition_not_uncertain_and_ack_never_upgrades():
    rows = [frow(BASE + timedelta(hours=h)) for h in (0, 8, 16, 32, 48, 64)]
    hist = FakeFundingExchange(rows).historical_funding_rates(
        "BTCUSDT", start_time=BASE, end_time=BASE + timedelta(hours=64),
        total_limit=100, page_limit=10)
    assert len(hist.interval_transitions) == 1
    assert hist.completeness_status == "complete"
    # Acknowledgement never upgrades a suspected-missing history.
    rows2 = [frow(BASE + timedelta(hours=h)) for h in (0, 8, 16, 32, 40)]
    hist2 = FakeFundingExchange(rows2).historical_funding_rates(
        "BTCUSDT", start_time=BASE, end_time=BASE + timedelta(hours=40),
        total_limit=100, page_limit=10)
    assert hist2.completeness_status == "uncertain"
    acked = hist2.acknowledge_anomalies(hist2.suspected_anomalies)
    assert acked.completeness_status == "uncertain"
    assert acked.completeness_status_for_period(
        BASE, BASE + timedelta(hours=40)) == "uncertain"
    assert acked.interval_assessments[0].acknowledged is True
    assert acked.interval_assessments[0].supporting_before_seconds


def test_s3_second_gap_after_irregular_stretch_not_suppressed():
    # Deltas: 8h, 8h, 16h, 12h (irregular), 8h, 8h (stable re-forms),
    # 16h (second independent gap), 8h (return). Suppression of
    # repeated insufficient-evidence reports must not latch.
    assessments = _assess([0, 8, 16, 32, 44, 52, 60, 76, 84])
    assert [a.classification for a in assessments] == [
        "INSUFFICIENT_EVIDENCE", "SUSPECTED_MISSING_SETTLEMENT"]
    second = assessments[1]
    assert second.reference_interval_seconds == 28800.0
    # The second assessment is the second independent suspicious gap:
    # the 16h interval between the settlements at +60h and +76h.
    assert second.previous_time == BASE + timedelta(hours=60)
    assert second.next_time == BASE + timedelta(hours=76)
    assert second.observed_interval_seconds == 16 * 3600


def test_s3_existing_rules_still_intact():
    from noone.market_data.models import (
        BoundaryProbeLimitHit, FundingConflict, FundingMalformedRecord)
    conflict = FundingConflict(
        symbol="BTCUSDT", funding_time=BASE + timedelta(hours=8),
        rate_type=None, kept_rate=Decimal("0.0001"),
        kept_mark_price=Decimal("100"),
        conflicting_rate=Decimal("0.0002"),
        conflicting_mark_price=Decimal("100"))
    assert _status_history(
        conflicting_records=(conflict,)).completeness_status == "incomplete"
    assert _status_history(malformed_records=(
        FundingMalformedRecord(
            symbol="BTCUSDT", funding_time=BASE,
            reason="missing_or_invalid_mark_price"),)
    ).completeness_status == "incomplete"
    assert _status_history(boundary_probe_limit_hits=(
        BoundaryProbeLimitHit(funding_time=BASE, returned_count=1000),)
    ).completeness_status == "uncertain"
    clean = _status_history()
    assert clean.completeness_status == clean.completeness_status_for_period(
        BASE, BASE + timedelta(hours=24)) == "complete"
