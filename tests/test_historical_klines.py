"""Unit tests: historical kline pagination/validation (fake transport).

Synthetic pages test pagination logic only; live proof is in
tests/test_live.py and the diagnostic.
"""
from datetime import datetime, timedelta, timezone

import pytest

from noone.market_data.binance_futures import BinanceFuturesCollector
from noone.market_data.models import validate_klines

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def row(open_dt, close="100"):
    ms = int(open_dt.timestamp() * 1000)
    return [ms, "99", "101", "98", close, "10", ms + 59_999,
            "1000", 5, "6", "600", "0"]


def candles_for(start_idx, count, step_min=1):
    return [row(BASE + timedelta(minutes=step_min * (start_idx + i)))
            for i in range(count)]


class FakeCollector(BinanceFuturesCollector):
    def __init__(self, pages):
        self._pages = list(pages)
        self.calls = []

    def _get(self, endpoint, params=None):
        self.calls.append((endpoint, dict(params or {})))
        return self._pages.pop(0) if self._pages else []


def test_pagination_more_than_one_page():
    fake = FakeCollector([candles_for(0, 2), candles_for(2, 2),
                          candles_for(4, 1)])
    hist = fake.historical_klines("BTCUSDT", interval="1m",
                                  total_limit=5, page_limit=2)
    assert hist.pages_fetched == 3 and len(hist.candles) == 5
    assert hist.candles[0].open_time == BASE  # chronological
    assert hist.candles[-1].open_time == BASE + timedelta(minutes=4)


def test_start_end_filtering_and_multi_symbol():
    fake = FakeCollector([candles_for(0, 4)])
    hist = fake.historical_klines(
        "ETHUSDT", interval="1m",
        start_time=BASE + timedelta(minutes=1),
        end_time=BASE + timedelta(minutes=2), total_limit=10)
    assert [c.open_time for c in hist.candles] == [
        BASE + timedelta(minutes=1), BASE + timedelta(minutes=2)]
    multi_fake = FakeCollector([candles_for(0, 1), candles_for(0, 1)])
    multi = multi_fake.historical_klines_multi(
        ["BTCUSDT", "ETHUSDT"], interval="1m", total_limit=1)
    assert set(multi) == {"BTCUSDT", "ETHUSDT"}


def test_interval_validation_and_other_interval():
    with pytest.raises(ValueError):
        FakeCollector([]).historical_klines("BTCUSDT", interval="7x")
    fake = FakeCollector([candles_for(0, 2, step_min=15)])
    hist = fake.historical_klines("BTCUSDT", interval="15m", total_limit=2)
    assert len(hist.candles) == 2 and hist.gaps == ()


def test_duplicates_gaps_and_ordering():
    from noone.market_data.binance_futures import parse_klines

    raw = [row(BASE + timedelta(minutes=2)), row(BASE),
           row(BASE), row(BASE + timedelta(minutes=5))]
    parsed = parse_klines("BTCUSDT", "1m", raw)
    clean, gaps, dups = validate_klines(parsed, "1m")
    assert dups == 1 and len(clean) == 3
    assert [c.open_time for c in clean] == sorted(c.open_time for c in clean)
    # Gaps at minute 1 (0->2) and minutes 3,4 (2->5): reported, not filled.
    assert [g.missing_candles for g in gaps] == [1, 2]


def test_closed_vs_incomplete_candle():
    from noone.market_data.binance_futures import parse_klines

    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    old = parse_klines("BTCUSDT", "1m", [row(now - timedelta(minutes=5))])[0]
    current_open = now
    current = parse_klines("BTCUSDT", "1m", [row(current_open)])[0]
    assert old.is_closed is True and current.is_closed is False
    assert old.taker_buy_volume == 6.0  # column 9 preserved


class FakeExchange(BinanceFuturesCollector):
    """Param-aware fake of Binance /fapi/v1/klines semantics.

    With ``startTime`` it returns the first ``limit`` candles at/after
    it; without, it returns the most recent ``limit`` candles at/before
    ``endTime``. Pages are chronological, like the real endpoint.
    """

    def __init__(self, rows):
        self._rows = sorted(rows, key=lambda r: r[0])
        self.calls = []

    def _get(self, endpoint, params=None):
        assert endpoint == "/fapi/v1/klines"
        params = dict(params or {})
        self.calls.append(params)
        rows = self._rows
        if params.get("startTime") is not None:
            rows = [r for r in rows if r[0] >= params["startTime"]]
        if params.get("endTime") is not None:
            rows = [r for r in rows if r[0] <= params["endTime"]]
        limit = int(params.get("limit", 500))
        if not rows or limit <= 0:
            return []
        return rows[:limit] if params.get("startTime") is not None else rows[-limit:]


def _ms(dt):
    return int(dt.timestamp() * 1000)


def _opens(hist):
    return [c.open_time for c in hist.candles]


def _expected(*idx, step_min=1):
    return [BASE + timedelta(minutes=step_min * i) for i in idx]


# -- Pagination regression matrix (mocked exchange, no live calls) ----

def test_01_explicit_start_smaller_than_one_page():
    fake = FakeExchange(candles_for(0, 10))
    hist = fake.historical_klines(
        "BTCUSDT", interval="1m", start_time=BASE, total_limit=3, page_limit=10)
    assert _opens(hist) == _expected(0, 1, 2)
    assert hist.pages_fetched == 1 and len(fake.calls) == 1
    assert fake.calls[0]["startTime"] == _ms(BASE)


def test_02_explicit_start_larger_than_one_page():
    fake = FakeExchange(candles_for(0, 10))
    hist = fake.historical_klines(
        "BTCUSDT", interval="1m", start_time=BASE, total_limit=5, page_limit=2)
    assert _opens(hist) == _expected(0, 1, 2, 3, 4)
    assert hist.pages_fetched == 3
    assert [c["startTime"] for c in fake.calls] == [
        _ms(BASE), _ms(BASE + timedelta(minutes=2)),
        _ms(BASE + timedelta(minutes=4))]


def test_03_omitted_start_smaller_than_one_page():
    fake = FakeExchange(candles_for(0, 10))
    hist = fake.historical_klines("BTCUSDT", interval="1m",
                                  total_limit=3, page_limit=10)
    assert _opens(hist) == _expected(7, 8, 9)  # most recent
    assert hist.pages_fetched == 1
    assert "startTime" not in fake.calls[0]


def test_04_omitted_start_larger_than_one_page():
    fake = FakeExchange(candles_for(0, 10))
    hist = fake.historical_klines("BTCUSDT", interval="1m",
                                  total_limit=5, page_limit=2)
    assert _opens(hist) == _expected(5, 6, 7, 8, 9)  # most recent, sorted
    assert hist.pages_fetched == 3
    assert all("startTime" not in c for c in fake.calls)  # never goes future
    end_times = [c.get("endTime") for c in fake.calls]
    assert end_times[0] is None
    assert end_times[1] == _ms(BASE + timedelta(minutes=8)) - 1
    assert end_times[2] == _ms(BASE + timedelta(minutes=6)) - 1
    assert end_times[1] > end_times[2]  # strictly backwards


def test_05_exactly_one_full_page():
    for kwargs, want in (
            ({"start_time": BASE}, _expected(0, 1)),
            ({}, _expected(8, 9))):
        fake = FakeExchange(candles_for(0, 10))
        hist = fake.historical_klines("BTCUSDT", interval="1m",
                                      total_limit=2, page_limit=2, **kwargs)
        assert _opens(hist) == want
        assert hist.pages_fetched == 1 and len(fake.calls) == 1


def test_06_fewer_available_than_requested():
    for kwargs in ({"start_time": BASE}, {}):
        fake = FakeExchange(candles_for(0, 3))
        hist = fake.historical_klines("BTCUSDT", interval="1m",
                                      total_limit=20, page_limit=4, **kwargs)
        assert _opens(hist) == _expected(0, 1, 2)
        assert hist.pages_fetched == 1  # short first page ends pagination


def test_07_empty_api_response():
    for kwargs in ({"start_time": BASE}, {}):
        fake = FakeExchange([])
        hist = fake.historical_klines("BTCUSDT", interval="1m",
                                      total_limit=5, page_limit=2, **kwargs)
        assert hist.candles == () and hist.gaps == ()
        assert hist.duplicates_removed == 0 and hist.pages_fetched == 1


def test_08_duplicate_candles_across_page_boundaries():
    fwd = FakeCollector([candles_for(0, 3), candles_for(2, 3)])
    hist = fwd.historical_klines("BTCUSDT", interval="1m", start_time=BASE,
                                 total_limit=5, page_limit=3)
    assert _opens(hist) == _expected(0, 1, 2, 3, 4)
    assert hist.duplicates_removed == 1 and hist.pages_fetched == 2
    back = FakeCollector([candles_for(7, 3), candles_for(5, 3)])
    hist = back.historical_klines("BTCUSDT", interval="1m",
                                  total_limit=5, page_limit=3)
    assert _opens(hist) == _expected(5, 6, 7, 8, 9)
    assert hist.duplicates_removed == 1 and hist.pages_fetched == 2


def test_09_chronological_ordering_backward_collection():
    fake = FakeExchange(candles_for(0, 10))
    hist = fake.historical_klines("BTCUSDT", interval="1m",
                                  total_limit=6, page_limit=2)
    opens = _opens(hist)
    assert opens == _expected(4, 5, 6, 7, 8, 9)
    assert opens == sorted(opens) and len(set(opens)) == len(opens)


def test_10_no_future_or_unfinished_candles():
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    rows = [row(now + timedelta(minutes=i)) for i in range(-6, 2)]
    fake = FakeExchange(rows)
    hist = fake.historical_klines("BTCUSDT", interval="1m", total_limit=3,
                                  page_limit=2, only_closed=True)
    assert len(hist.candles) == 3
    assert all(c.is_closed for c in hist.candles)
    assert all(c.open_time <= now for c in hist.candles)
    assert all("startTime" not in c for c in fake.calls)


def test_11_repeated_page_terminates_without_loop():
    class Repeating(BinanceFuturesCollector):
        def __init__(self):
            self.calls = 0

        def _get(self, endpoint, params=None):
            self.calls += 1
            if self.calls > 10:
                raise AssertionError("pagination did not terminate")
            return candles_for(0, 2)

    for kwargs in ({"start_time": BASE}, {}):
        fake = Repeating()
        hist = fake.historical_klines("BTCUSDT", interval="1m",
                                      total_limit=10, page_limit=2, **kwargs)
        assert len(hist.candles) == 2
        assert hist.pages_fetched <= 2 and fake.calls <= 2


def test_12_existing_caller_call_shapes_stay_compatible():
    # Diagnostic shape: start + end + small pages + only_closed.
    fake = FakeExchange(candles_for(0, 10))
    hist = fake.historical_klines(
        "BTCUSDT", interval="1m", start_time=BASE + timedelta(minutes=1),
        end_time=BASE + timedelta(minutes=4), total_limit=40,
        page_limit=2, only_closed=True)
    assert _opens(hist) == _expected(1, 2, 3, 4)
    assert hist.symbol == "BTCUSDT" and hist.interval == "1m"
    assert isinstance(hist.candles, tuple)
    # Multi-symbol wrapper shape (RL/diagnostic callers).
    multi_fake = FakeExchange(candles_for(0, 3))
    multi = multi_fake.historical_klines_multi(
        ["BTCUSDT", "ETHUSDT"], interval="1m", total_limit=2)
    assert set(multi) == {"BTCUSDT", "ETHUSDT"}
    assert all(len(h.candles) == 2 for h in multi.values())


def test_failure_propagates():
    class Failing(BinanceFuturesCollector):
        def _get(self, endpoint, params=None):
            from noone.market_data.binance_futures import MarketDataError
            raise MarketDataError("boom", endpoint=endpoint)

    with pytest.raises(Exception, match="boom"):
        Failing.__new__(Failing).historical_klines("BTCUSDT", interval="1m")
