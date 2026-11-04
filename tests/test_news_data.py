"""Unit tests: news normalization, dedup, malformed handling."""
import pytest

from noone.news_data.crypto_news import NewsDataError, normalize_article, normalize_feed

RAW = {"title": "Bitcoin rallies", "link": "https://example.com/a",
       "description": "BTC surged today.", "pubDate": "2026-11-02T10:00:00Z",
       "source": "CoinDesk", "sourceKey": "coindesk", "category": "bitcoin"}


def test_normalize_article():
    a = normalize_article(RAW)
    assert a is not None and a.source == "CoinDesk"
    assert a.url == "https://example.com/a"
    assert a.published_at is not None and a.api_sentiment is None
    assert "Bitcoin rallies" in a.text_for_classification


def test_malformed_articles_rejected():
    assert normalize_article({"title": "", "link": "https://x"}) is None
    assert normalize_article({"title": "T"}) is None  # no URL
    assert normalize_article("not-a-dict") is None
    assert normalize_article(None) is None


def test_feed_dedup_and_empty():
    feed = normalize_feed({"articles": [RAW, RAW, {"bad": 1}]})
    assert len(feed) == 1  # duplicate + malformed removed
    assert normalize_feed({"articles": []}) == []
    with pytest.raises(NewsDataError):
        normalize_feed({"unexpected": True})


def test_bad_date_is_none_not_fabricated():
    a = normalize_article({**RAW, "pubDate": "not-a-date"})
    assert a is not None and a.published_at is None
