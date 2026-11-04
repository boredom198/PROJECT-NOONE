"""Live API tests — real data only. Skipped unless NOONE_LIVE=1.

Run:  NOONE_LIVE=1 python -m pytest tests/test_live.py -v
(PowerShell: $env:NOONE_LIVE=1; python -m pytest tests/test_live.py -v)
Synthetic fixtures elsewhere never substitute for these checks.
"""
import os
from datetime import datetime, timezone

import pytest

from noone.market_data import BinanceFuturesCollector
from noone.market_data.models import age_seconds
from noone.news_classifier import FinVADERClassifier
from noone.news_data import CryptoNewsCollector

pytestmark = pytest.mark.skipif(
    os.environ.get("NOONE_LIVE") != "1", reason="live tests need NOONE_LIVE=1")


def test_live_futures_are_futures_and_fresh():
    with BinanceFuturesCollector() as market:
        assert market.ping() is True
        info = market.exchange_info()
        assert any(s.symbol == "BTCUSDT" and s.contract_type == "PERPETUAL"
                   for s in info)
        mark = market.mark_price("BTCUSDT")
        ticker = market.ticker_24h("BTCUSDT")
        klines = market.klines("BTCUSDT", limit=5)
    assert mark.mark_price and mark.mark_price > 0
    assert ticker.last_price and ticker.last_price > 0
    assert klines and all(k.close and k.close > 0 for k in klines)
    age = age_seconds(mark.event_time, now=datetime.now(timezone.utc))
    assert age is not None and age < 120  # fresh, not stale


def test_live_historical_klines_pagination_multi_interval():
    from datetime import datetime, timedelta
    from noone.market_data.models import utcnow

    end = utcnow()
    start = end - timedelta(hours=4)
    with BinanceFuturesCollector() as market:
        multi = market.historical_klines_multi(
            ["BTCUSDT", "ETHUSDT"], interval="15m",
            start_time=start, end_time=end, total_limit=40,
            page_limit=10, only_closed=True)
    assert set(multi) == {"BTCUSDT", "ETHUSDT"}
    for hist in multi.values():
        assert hist.pages_fetched > 1  # real pagination happened
        assert len(hist.candles) >= 10
        assert all(c.is_closed for c in hist.candles)
        times = [c.open_time for c in hist.candles]
        assert times == sorted(times) and len(set(times)) == len(times)


def test_live_news_archive_roundtrip(tmp_path):
    from noone.news_data import NewsArchive

    with CryptoNewsCollector() as news:
        articles = news.fetch_latest(limit=5)
    assert articles
    classifier = FinVADERClassifier()
    db = tmp_path / "live_news.sqlite3"
    with NewsArchive(db) as arch:
        for item in classifier.classify_batch(articles):
            arch.store_classification(item)
        assert arch.count() >= 1
    with NewsArchive(db) as arch:  # restart persistence with real data
        assert arch.count() >= 1
        ranged = arch.query_time_range()
        assert ranged and all(a.title and a.url for a in ranged)


def test_live_news_and_classification():
    with CryptoNewsCollector() as news:
        articles = news.fetch_latest(limit=5)
    assert articles, "live news API returned no articles"
    assert all(a.title and a.url.startswith("http") for a in articles)
    classified = FinVADERClassifier().classify_batch(articles)
    assert len(classified) == len(articles)
    assert all(c.sentiment.label in ("positive", "negative", "neutral")
               for c in classified)
