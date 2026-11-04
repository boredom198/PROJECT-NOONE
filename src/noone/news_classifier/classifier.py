"""Modified Lexicon FinVADER classifier (Component 2).

Scoring pipeline — replicates FinVADER upstream, then NOONE's overlay:
  VADER lexicon
  + SentiBignomics x 0.1   (upstream constant; skipped if unavailable)
  + Henry lexicon          (merged on top, as upstream)
  + CRYPTO_LEXICON         (NOONE modification, applied last)

Labels use the standard VADER thresholds on the compound score:
  compound >= 0.05 -> positive; <= -0.05 -> negative; else neutral.

Confidence: FinVADER/VADER provides **no probability**. We therefore
return ``confidence_proxy`` (score-based, NOT a probability):
  proxy = |compound| * (0.6 + 0.4 * min(1, matched_tokens / tokens))
clipped to [0, 1]. It grows with signal strength and lexicon
coverage. ``uncertain=True`` flags insufficient text (< 3 tokens or
zero lexicon hits): the label is then "neutral" by convention only
and must not be traded on. Sentiment alone does not predict prices.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

from noone.news_classifier.assets import AssetIdentification, identify_assets
from noone.news_classifier.lexicon import CRYPTO_LEXICON
from noone.news_data.models import NewsArticle

_TOKEN_RE = re.compile(r"[a-z']+")
_RELEVANCE_TERMS = frozenset({
    "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency",
    "token", "tokens", "blockchain", "coin", "coins", "exchange",
    "futures", "trading", "trader", "defi", "nft", "solana", "sol",
    "xrp", "binance", "coinbase", "wallet", "mining", "staking",
    "stablecoin", "altcoin", "market", "markets", "stock", "stocks",
    "shares", "investors", "fund", "etf", "price", "prices",
})
POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05
MIN_TOKENS = 3


@dataclass(frozen=True)
class SentimentResult:
    label: str  # "positive" | "negative" | "neutral"
    compound: float
    pos: float
    neg: float
    neu: float
    confidence_proxy: float  # score-based proxy, NOT a probability
    uncertain: bool
    token_count: int
    matched_terms: tuple[str, ...]
    method: str
    degraded: bool = False  # True if FinVADER lexicons were unavailable
    # Crypto asset identification (mentioned != affected; see assets.py)
    assets: AssetIdentification = AssetIdentification((), (), ())


@dataclass(frozen=True)
class ClassifiedArticle:
    article: NewsArticle  # original preserved for inspection
    sentiment: SentimentResult
    raw_scores: dict[str, float] = field(default_factory=dict)


def build_analyzer() -> tuple[Any, bool, str]:
    """Build the VADER analyzer with the FinVADER + crypto lexicons.

    Returns ``(analyzer, degraded, method_description)``. If the
    ``finvader`` package lexicons cannot be loaded, falls back to
    VADER + crypto overlay and reports ``degraded=True`` honestly
    instead of failing or silently pretending to be full FinVADER.
    """
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    try:
        analyzer = SentimentIntensityAnalyzer()
    except LookupError as exc:
        raise RuntimeError(
            "NLTK vader_lexicon not found. Run: python -c \"import nltk; "
            "nltk.download('vader_lexicon')\" (one-time download). Behind "
            "a trusted proxy you may need NLTK_ALLOW_PROXIED_URLOPEN=1."
        ) from exc
    degraded = False
    method = "FinVADER(SentiBignomics x0.1 + Henry) + NOONE crypto overlay"
    try:
        from finvader.Henry import lexicon2
        from finvader.SentiBignomics import lexicon1

        senti = lexicon1()
        senti = {k: v * 0.1 for k, v in senti.items()}  # upstream constant
        merged = {**senti, **lexicon2()}  # Henry wins, as upstream Merge
        analyzer.lexicon.update(merged)
    except Exception:
        degraded = True
        method = "VADER + NOONE crypto overlay (FinVADER lexicons unavailable)"
    analyzer.lexicon.update(CRYPTO_LEXICON)
    return analyzer, degraded, method


class FinVADERClassifier:
    """CPU-only lexicon classifier. No LLM, no API calls."""

    def __init__(self) -> None:
        self._analyzer, self._degraded, self._method = build_analyzer()

    @property
    def degraded(self) -> bool:
        return self._degraded

    def classify_text(self, text: str | None) -> SentimentResult:
        clean = (text or "").strip()
        tokens = _TOKEN_RE.findall(clean.lower())
        scores = self._analyzer.polarity_scores(clean) if clean else {
            "neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
        compound = float(scores["compound"])
        matched = tuple(t for t in tokens if t in self._analyzer.lexicon)
        uncertain = len(tokens) < MIN_TOKENS or not matched
        if uncertain:
            label = "neutral"  # convention only; see uncertain flag
        elif compound >= POS_THRESHOLD:
            label = "positive"
        elif compound <= NEG_THRESHOLD:
            label = "negative"
        else:
            label = "neutral"
        coverage = min(1.0, len(matched) / max(1, len(tokens)))
        proxy = round(min(1.0, abs(compound) * (0.6 + 0.4 * coverage)), 4)
        if uncertain:
            proxy = 0.0
        return SentimentResult(
            label=label, compound=compound,
            pos=float(scores["pos"]), neg=float(scores["neg"]),
            neu=float(scores["neu"]), confidence_proxy=proxy,
            uncertain=uncertain, token_count=len(tokens),
            matched_terms=matched, method=self._method,
            degraded=self._degraded, assets=identify_assets(clean),
        )

    def classify_article(self, article: NewsArticle) -> ClassifiedArticle:
        text = article.text_for_classification
        result = self.classify_text(text)
        # Relevance gate: the live feed mixes in non-crypto wires. An
        # article with no crypto/finance term is flagged uncertain
        # (proxy zeroed) instead of carrying a confident label.
        tokens = set(_TOKEN_RE.findall(text.lower()))
        if not tokens & _RELEVANCE_TERMS:
            result = replace(result, uncertain=True, confidence_proxy=0.0)
        scores = self._analyzer.polarity_scores(text)
        return ClassifiedArticle(article=article, sentiment=result,
                                 raw_scores={k: float(v) for k, v in scores.items()})

    def classify_batch(self, articles: list[NewsArticle]) -> list[ClassifiedArticle]:
        return [self.classify_article(a) for a in articles]
