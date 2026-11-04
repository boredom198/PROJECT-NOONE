"""Unit tests: Futures parsing/normalization (recorded-shape fixtures).

Synthetic fixtures test parsing only — live proof is in test_live.py
and `python -m noone.diagnostic`.
"""
from datetime import datetime, timezone

import pytest

from noone.market_data.binance_futures import (
    MarketDataError,
    parse_depth,
    parse_exchange_info,
    parse_funding_history,
    parse_klines,
    parse_open_interest,
    parse_premium_index,
    parse_ticker_24h,
)
from noone.market_data.models import is_stale

KLINE = [[1700000000000, "100.0", "110.0", "90.0", "105.0", "12.5",
          1700000059999, "1312.5", 42, "6.0", "630.0", "0"]]


def test_parse_klines():
    (k,) = parse_klines("BTCUSDT", "1m", KLINE)
    assert (k.open, k.high, k.low, k.close) == (100.0, 110.0, 90.0, 105.0)
    assert k.trades == 42 and k.open_time is not None


def test_malformed_kline_skipped_and_bad_payload_raises():
    assert parse_klines("BTCUSDT", "1m", [["bad"], KLINE[0]])[0].close == 105.0
    with pytest.raises(MarketDataError):
        parse_klines("BTCUSDT", "1m", {"not": "a list"})


def test_ticker_missing_fields_are_none_not_zero():
    t = parse_ticker_24h({"symbol": "BTCUSDT", "lastPrice": "100.5"})
    assert t.last_price == 100.5 and t.volume is None  # unavailable != 0


def test_zero_funding_stays_zero():
    m = parse_premium_index({"symbol": "BTCUSDT", "markPrice": "100",
                             "lastFundingRate": "0.00000000", "time": 1700000000000})
    assert m.last_funding_rate == 0.0  # valid zero preserved


def test_open_interest_depth_funding_exchange():
    oi = parse_open_interest("BTCUSDT", {"symbol": "BTCUSDT",
                                         "openInterest": "123.5", "time": 1700000000000})
    assert oi.open_interest == 123.5
    d = parse_depth("BTCUSDT", {"lastUpdateId": 1, "bids": [["100", "1"]], "asks": []})
    assert d.bids == ((100.0, 1.0),) and d.asks == ()
    f = parse_funding_history([{"symbol": "BTCUSDT", "fundingRate": "0.0001",
                                "fundingTime": 1700000000000}])
    assert f[0].funding_rate == 0.0001
    info = parse_exchange_info({"symbols": [{"symbol": "BTCUSDT",
                                             "contractType": "PERPETUAL"}]})
    assert info[0].contract_type == "PERPETUAL"


def test_staleness():
    old = datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert is_stale(old, 30) is True
    assert is_stale(None, 30) is True  # unknown time counts as stale
