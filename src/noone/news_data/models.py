"""Normalized news-article structure (Component 4, news side)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_pub_date(value: Any) -> datetime | None:
    """Parse ISO-8601 ``pubDate``; unparseable -> None (unavailable)."""
    if not value or not isinstance(value, str):
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def article_id_for(url: str, title: str) -> str:
    """Stable dedup id when the API omits ``id`` (hash of URL+title)."""
    return hashlib.sha256(f"{url}|{title}".encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class NewsArticle:
    article_id: str
    title: str
    description: str  # "" when the API provides no description
    source: str | None
    source_key: str | None
    url: str
    published_at: datetime | None
    category: str | None
    tags: tuple[str, ...] = ()
    # API-provided sentiment, preserved for comparison ONLY. Never used
    # as the classifier's final output (see news_classifier).
    api_sentiment: str | None = None
    retrieved_at: datetime = field(default_factory=utcnow)
    # When NOONE first observed this article (archive assigns it on
    # first store; kept distinct from published_at/retrieved_at and
    # never overwritten). None until archived.
    first_seen_at: datetime | None = None
    raw: Any = None

    @property
    def text_for_classification(self) -> str:
        """Title + description — the text FinVADER actually scores."""
        return f"{self.title}. {self.description}".strip()
