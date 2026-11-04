"""Persistent point-in-time news archive (Component 4, news side).

SQLite (stdlib ``sqlite3``) — local, CPU-friendly, no cloud/paid DB.
The archive starts when NOONE starts observing: it stores real
incoming articles with three separate timestamps that are never
overwritten on re-store:

- ``published_at``: when the article was originally published.
- ``first_seen_at``: when NOONE first observed it (first store).
- ``retrieved_at``: when NOONE retrieved it (that first retrieval).

Re-storing the same ``article_id`` is a no-op for the article row
(returns False); classifications can be attached/updated separately.
Queries support time ranges, per-asset lookup (indexed
``article_assets`` table), and the strict point-in-time rule:
``published_at <= t AND first_seen_at <= t``.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from noone.news_data.models import NewsArticle

_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    article_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    source TEXT, source_key TEXT,
    url TEXT NOT NULL,
    category TEXT,
    tags_json TEXT NOT NULL DEFAULT '[]',
    api_sentiment TEXT,
    published_at TEXT,
    first_seen_at TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    raw_json TEXT
);
CREATE TABLE IF NOT EXISTS article_assets (
    article_id TEXT NOT NULL REFERENCES articles(article_id),
    asset TEXT NOT NULL,
    association TEXT NOT NULL DEFAULT 'mentioned',
    matched_terms_json TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (article_id, asset)
);
CREATE TABLE IF NOT EXISTS classifications (
    article_id TEXT PRIMARY KEY REFERENCES articles(article_id),
    label TEXT, compound REAL, pos REAL, neg REAL, neu REAL,
    confidence_proxy REAL, uncertain INTEGER,
    token_count INTEGER, matched_terms_json TEXT,
    method TEXT, degraded INTEGER, classified_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_articles_first_seen ON articles(first_seen_at);
CREATE INDEX IF NOT EXISTS idx_assets_asset ON article_assets(asset);
"""


def _iso(dt: datetime | None) -> str | None:
    return dt.astimezone(timezone.utc).isoformat() if dt else None


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class NewsArchive:
    """SQLite-backed archive. ``path`` may be ``":memory:"``."""

    def __init__(self, path: str | Path):
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "NewsArchive":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def store_article(
        self, article: NewsArticle, *, assets: Any = None,
    ) -> bool:
        """Insert an article once. True if newly stored.

        ``assets`` is an ``AssetIdentification`` (or iterable of
        ``AssetMatch``). On duplicate, original timestamps and
        content are preserved untouched (returns False).
        """
        first_seen = article.first_seen_at or article.retrieved_at
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO articles (article_id, title, "
            "description, source, source_key, url, category, tags_json, "
            "api_sentiment, published_at, first_seen_at, retrieved_at, "
            "raw_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (article.article_id, article.title, article.description,
             article.source, article.source_key, article.url,
             article.category, json.dumps(list(article.tags)),
             article.api_sentiment, _iso(article.published_at),
             _iso(first_seen), _iso(article.retrieved_at),
             json.dumps(article.raw, default=str) if article.raw else None))
        inserted = cur.rowcount > 0
        if inserted and assets is not None:
            matches = []
            if hasattr(assets, "mentioned"):
                matches = (list(assets.mentioned) + list(assets.affected)
                           + list(assets.uncertain))
            else:
                matches = list(assets)
            for match in matches:
                self._conn.execute(
                    "INSERT OR IGNORE INTO article_assets (article_id, "
                    "asset, association, matched_terms_json) "
                    "VALUES (?,?,?,?)",
                    (article.article_id, match.asset, match.association,
                     json.dumps(list(match.matched_terms))))
        self._conn.commit()
        return inserted

    def store_classification(self, classified: Any) -> None:
        """Attach/update a ``ClassifiedArticle`` result for its article.

        Stores the article first if needed, including its identified
        assets, then upserts the classification row.
        """
        article, sentiment = classified.article, classified.sentiment
        self.store_article(article, assets=sentiment.assets)
        self._conn.execute(
            "INSERT INTO classifications (article_id, label, compound, "
            "pos, neg, neu, confidence_proxy, uncertain, token_count, "
            "matched_terms_json, method, degraded, classified_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(article_id) "
            "DO UPDATE SET label=excluded.label, compound=excluded.compound,"
            " pos=excluded.pos, neg=excluded.neg, neu=excluded.neu,"
            " confidence_proxy=excluded.confidence_proxy,"
            " uncertain=excluded.uncertain,"
            " token_count=excluded.token_count,"
            " matched_terms_json=excluded.matched_terms_json,"
            " method=excluded.method, degraded=excluded.degraded,"
            " classified_at=excluded.classified_at",
            (article.article_id, sentiment.label, sentiment.compound,
             sentiment.pos, sentiment.neg, sentiment.neu,
             sentiment.confidence_proxy, int(sentiment.uncertain),
             sentiment.token_count,
             json.dumps(list(sentiment.matched_terms)), sentiment.method,
             int(sentiment.degraded), _iso(datetime.now(timezone.utc))))
        self._conn.commit()

    def count(self) -> int:
        return int(self._conn.execute(
            "SELECT COUNT(*) FROM articles").fetchone()[0])

    def _row_to_article(self, row: tuple) -> NewsArticle:
        return NewsArticle(
            article_id=row[0], title=row[1], description=row[2],
            source=row[3], source_key=row[4], url=row[5],
            published_at=_parse(row[9]), category=row[6],
            tags=tuple(json.loads(row[7] or "[]")),
            api_sentiment=row[8], retrieved_at=_parse(row[11]),
            first_seen_at=_parse(row[10]),
            raw=json.loads(row[12]) if row[12] else None)

    _COLS = ("article_id, title, description, source, source_key, url,"
             " category, tags_json, api_sentiment, published_at,"
             " first_seen_at, retrieved_at, raw_json")
    _COLS_Q = ", ".join(f"articles.{c.strip()}" for c in _COLS.split(","))

    def get_article(self, article_id: str) -> NewsArticle | None:
        row = self._conn.execute(
            f"SELECT {self._COLS} FROM articles WHERE article_id=?",
            (article_id,)).fetchone()
        return self._row_to_article(row) if row else None

    def query_time_range(
        self, start: datetime | None = None, end: datetime | None = None,
        *, asset: str | None = None, limit: int = 500,
    ) -> list[NewsArticle]:
        """Articles with ``published_at`` in [start, end] (UTC).

        Articles with unknown ``published_at`` are excluded when a
        range bound is given (they cannot be placed in time safely).
        """
        sql = f"SELECT {self._COLS_Q if asset else self._COLS} FROM articles"
        clauses: list[str] = []
        params: list[str] = []
        if asset:
            sql += " JOIN article_assets aa ON aa.article_id = articles.article_id"
            clauses.append("aa.asset = ?")
            params.append(asset.upper())
        if start is not None:
            clauses.append("published_at >= ?")
            params.append(_iso(start))
        if end is not None:
            clauses.append("published_at <= ?")
            params.append(_iso(end))
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY published_at LIMIT ?"
        params.append(str(limit))
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_article(r) for r in rows]

    def query_by_asset(
        self, asset: str, *, limit: int = 500,
    ) -> list[NewsArticle]:
        return self.query_time_range(asset=asset, limit=limit)

    def available_at(
        self, decision_time: datetime, *, asset: str | None = None,
        limit: int = 500,
    ) -> list[NewsArticle]:
        """Point-in-time query (strict anti-lookahead rule).

        Returns only articles with ``published_at <= decision_time``
        AND ``first_seen_at <= decision_time``. Articles with unknown
        ``published_at`` are never returned here.
        """
        sql = f"SELECT {self._COLS_Q if asset else self._COLS} FROM articles"
        params: list[str] = []
        if asset:
            sql += " JOIN article_assets aa ON aa.article_id = articles.article_id"
        sql += (" WHERE published_at IS NOT NULL AND published_at <= ?"
                " AND first_seen_at <= ?")
        params += [_iso(decision_time), _iso(decision_time)]
        if asset:
            sql += " AND aa.asset = ?"
            params.append(asset.upper())
        sql += " ORDER BY published_at LIMIT ?"
        params.append(str(limit))
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_article(r) for r in rows]

    def classification_for(self, article_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT label, compound, pos, neg, neu, confidence_proxy,"
            " uncertain, method, degraded FROM classifications"
            " WHERE article_id=?", (article_id,)).fetchone()
        if not row:
            return None
        return {"label": row[0], "compound": row[1], "pos": row[2],
                "neg": row[3], "neu": row[4], "confidence_proxy": row[5],
                "uncertain": bool(row[6]), "method": row[7],
                "degraded": bool(row[8])}

    @staticmethod
    def with_first_seen(article: NewsArticle) -> NewsArticle:
        """Return a copy stamped with first_seen = retrieved (helper)."""
        if article.first_seen_at is not None:
            return article
        return replace(article, first_seen_at=article.retrieved_at)
