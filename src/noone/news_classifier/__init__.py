"""Component 2: News Classifier — Modified Lexicon FinVADER.

FinVADER (upstream, Apache-2.0) = NLTK VADER + SentiBignomics lexicon
(scaled x0.1) + Henry lexicon. NOONE adds a documented crypto overlay
(``CRYPTO_LEXICON``) on top. Sentiment alone does not predict prices.
"""
from noone.news_classifier.assets import (
    ASSET_ALIASES,
    AssetIdentification,
    AssetMatch,
    identify_assets,
)
from noone.news_classifier.classifier import (
    ClassifiedArticle,
    FinVADERClassifier,
    SentimentResult,
)
from noone.news_classifier.lexicon import CRYPTO_LEXICON

__all__ = [
    "ASSET_ALIASES",
    "AssetIdentification",
    "AssetMatch",
    "CRYPTO_LEXICON",
    "ClassifiedArticle",
    "FinVADERClassifier",
    "SentimentResult",
    "identify_assets",
]
