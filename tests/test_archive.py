"""Unit tests: SQLite point-in-time news archive + sync rules."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from noone.market_data.models import Kline, closed_candles_before
from noone.news_classifier import FinVADERClassifier
from noone.news_data import NewsArchive
from noone.news_data.crypto_news import normalize_article

T0 = datetime(2026, 11, 1, 10, 0, tzinfo=timezone.utc)


def make_article(title="Bitcoin rallies", when=T0, description="BTC surged."):
    a = normalize_article({"title": title, "link": "https://example.com/a",
                           "description": description,
                           "pubDate": when.isoformat(),
                           "source": "CoinDesk"})
    return replace(a, retrieved_at=when + timedelta(minutes=5))


def test_persist_restart_dedup_timestamps(tmp_path):
    path = tmp_path / "news.sqlite3"
    article = make_article()
    with NewsArchive(path) as arch:
        assert arch.store_article(article) is True
        # Duplicate store: ignored, original timestamps preserved even
        # if a later copy claims different times.
        later = replace(article, title="Changed",
                        retrieved_at=T0 + timedelta(hours=2))
        assert arch.store_article(later) is False
        assert arch.count() == 1
    with NewsArchive(path) as arch:  # survives restart
        stored = arch.get_article(article.article_id)
        assert stored is not None and stored.title == "Bitcoin rallies"
        assert stored.published_at == T0
        assert stored.first_seen_at == T0 + timedelta(minutes=5)
        assert stored.retrieved_at == T0 + timedelta(minutes=5)


def test_time_range_and_asset_queries(tmp_path):
    classifier = FinVADERClassifier()
    with NewsArchive(tmp_path / "n.db") as arch:
        cases = [("Bitcoin surges", T0, "BTC surged."),
                 ("Ethereum upgrade live", T0 + timedelta(hours=1),
                  "Ethereum network upgrade completed."),
                 ("Market quiet day", T0 + timedelta(hours=2),
                  "Trading was quiet.")]
        for title, when, desc in cases:
            a = make_article(title, when, desc)
            arch.store_classification(classifier.classify_article(a))
        ranged = arch.query_time_range(T0, T0 + timedelta(hours=1))
        assert len(ranged) == 2
        btc = arch.query_by_asset("BTC")
        assert len(btc) == 1 and "Bitcoin" in btc[0].title
        assert arch.classification_for(btc[0].article_id)["label"] in (
            "positive", "negative", "neutral")


def test_point_in_time_excludes_future_and_unknown(tmp_path):
    with NewsArchive(tmp_path / "n.db") as arch:
        arch.store_article(make_article("Bitcoin old news", T0))
        future = make_article("Bitcoin future news", T0 + timedelta(days=1))
        arch.store_article(future)
        unknown = replace(make_article("Bitcoin dateless", T0),
                          published_at=None)
        arch.store_article(unknown)
        decision = T0 + timedelta(hours=1)
        available = arch.available_at(decision)
        assert [a.title for a in available] == ["Bitcoin old news"]
        # First-seen respected: published before decision but first
        # seen after it must also be excluded.
        late_seen = replace(make_article("Bitcoin late seen", T0),
                            retrieved_at=T0 + timedelta(days=2))
        arch.store_article(late_seen)
        assert "Bitcoin late seen" not in [
            a.title for a in arch.available_at(decision)]


def test_closed_candles_before_decision():
    def candle(minute, closed=True):
        open_t = T0 + timedelta(minutes=minute)
        return Kline(symbol="BTCUSDT", interval="1m", open_time=open_t,
                     close_time=open_t + timedelta(seconds=59),
                     open=1, high=1, low=1, close=1, volume=1,
                     quote_volume=1, trades=1, is_closed=closed)

    candles = [candle(0), candle(1), candle(2, closed=False)]
    usable = closed_candles_before(candles, T0 + timedelta(minutes=2))
    assert len(usable) == 2  # forming candle excluded (no lookahead)
