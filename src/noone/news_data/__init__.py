"""Component 4 (news side): Free Crypto News API collector.

Source: https://cryptocurrency.cv/api (keyless, unauthenticated).
The API-provided sentiment (when present) is preserved separately and
is never treated as NOONE's classifier output.
"""
from noone.news_data.archive import NewsArchive
from noone.news_data.crypto_news import CryptoNewsCollector, NewsDataError
from noone.news_data.models import NewsArticle

__all__ = ["CryptoNewsCollector", "NewsArchive", "NewsDataError", "NewsArticle"]
