"""Free Crypto News API collector — https://cryptocurrency.cv/api

Keyless/unauthenticated. Primary endpoint: ``GET /api/news``.
No streaming endpoint is documented for the free feed, so polling is
the supported mode (``poll`` generator below). No articles are ever
fabricated: an empty API response yields an empty list, and failures
raise :class:`NewsDataError`.
"""
from __future__ import annotations

import time
from typing import Any, Iterator

import httpx

from noone.news_data.models import NewsArticle, article_id_for, parse_pub_date

BASE_URL = "https://cryptocurrency.cv/api"


class NewsDataError(RuntimeError):
    """Raised when the news API cannot be retrieved/parsed."""


def normalize_article(raw: Any) -> NewsArticle | None:
    """Normalize one raw article; malformed entries return ``None``.

    Malformed = not a dict, or missing/blank title, or missing URL.
    Missing optional fields become ``None``/empty, never invented.
    """
    if not isinstance(raw, dict):
        return None
    title = str(raw.get("title") or "").strip()
    url = str(raw.get("link") or raw.get("url") or "").strip()
    if not title or not url:
        return None
    tags_raw = raw.get("tags")
    tags = tuple(str(t) for t in tags_raw) if isinstance(tags_raw, list) else ()
    return NewsArticle(
        article_id=str(raw.get("id") or article_id_for(url, title)),
        title=title,
        description=str(raw.get("description") or ""),
        source=raw.get("source"),
        source_key=raw.get("sourceKey"),
        url=url,
        published_at=parse_pub_date(raw.get("pubDate") or raw.get("publishedAt")),
        category=raw.get("category"),
        tags=tags,
        api_sentiment=raw.get("sentiment"),  # preserved, not trusted
        raw=raw,
    )


def normalize_feed(payload: Any) -> list[NewsArticle]:
    """Normalize a ``/api/news`` payload; dedup by article_id (first wins)."""
    if not isinstance(payload, dict) or not isinstance(payload.get("articles"), list):
        raise NewsDataError("Unexpected news payload (missing 'articles' list)")
    seen: set[str] = set()
    out: list[NewsArticle] = []
    for raw in payload["articles"]:
        article = normalize_article(raw)
        if article is None or article.article_id in seen:
            continue  # malformed or duplicate
        seen.add(article.article_id)
        out.append(article)
    return out


class CryptoNewsCollector:
    """Polling collector for the Free Crypto News API."""

    def __init__(self, *, timeout_s: float = 20.0, max_retries: int = 3):
        self._client = httpx.Client(
            base_url=BASE_URL, timeout=timeout_s,
            headers={"User-Agent": "NOONE/0.1 (Free Crypto News API)"},
        )
        self._max_retries = max(1, max_retries)
        self._seen_ids: set[str] = set()  # cross-poll dedup state

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "CryptoNewsCollector":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def fetch_latest(
        self, *, limit: int = 20, source: str | None = None,
        category: str | None = None, lang: str | None = None,
        only_new: bool = False,
    ) -> list[NewsArticle]:
        """Fetch and normalize the latest articles.

        ``only_new=True`` filters out articles already returned by an
        earlier call on this collector (duplicate handling across polls).
        """
        # The free tier caps each page (observed maxResults=3), so
        # paginate until `limit` articles are gathered or pages run dry.
        collected: list[NewsArticle] = []
        seen_here: set[str] = set()
        page = 1
        last_exc: Exception | None = None
        while len(collected) < limit:
            params: dict[str, Any] = {
                "limit": min(limit - len(collected), 100), "page": page}
            if source:
                params["source"] = source
            if category:
                params["category"] = category
            if lang:
                params["lang"] = lang
            page_articles: list[NewsArticle] | None = None
            for attempt in range(self._max_retries):
                try:
                    resp = self._client.get("/news", params=params)
                except httpx.HTTPError as exc:
                    last_exc = exc
                    time.sleep(min(2 ** attempt, 8))
                    continue
                if resp.status_code != 200:
                    raise NewsDataError(
                        f"News API /news -> HTTP {resp.status_code}: {resp.text[:200]}")
                try:
                    page_articles = normalize_feed(resp.json())
                except ValueError as exc:
                    raise NewsDataError("News API returned invalid JSON") from exc
                break
            if page_articles is None:
                raise NewsDataError(
                    f"News API unreachable after retries: {last_exc}")
            if not page_articles:
                break  # empty page: no more data, return what we have
            before = len(collected)
            for article in page_articles:
                if article.article_id not in seen_here:
                    seen_here.add(article.article_id)
                    collected.append(article)
            if len(collected) == before:
                break  # page added nothing new; avoid looping forever
            # Note: a page shorter than requested is NOT the last page
            # here — the free tier caps page size (3) below our limit.
            # Only an empty/no-new page ends pagination (plus max_pages).
            page += 1
            if page > 50:
                break
        articles = collected[:limit]
        if only_new:
            fresh = [a for a in articles if a.article_id not in self._seen_ids]
            self._seen_ids.update(a.article_id for a in articles)
            return fresh
        self._seen_ids.update(a.article_id for a in articles)
        return articles

    def poll(
        self, *, interval_s: float = 60.0, limit: int = 20,
    ) -> Iterator[list[NewsArticle]]:
        """Yield only-new article batches forever (polling mode)."""
        while True:
            yield self.fetch_latest(limit=limit, only_new=True)
            time.sleep(interval_s)
