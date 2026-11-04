"""Unit tests: Modified Lexicon FinVADER behaviour (deterministic)."""
from noone.news_classifier import FinVADERClassifier
from noone.news_data.crypto_news import normalize_article


def test_positive_negative_neutral():
    c = FinVADERClassifier()
    pos = c.classify_text("Bitcoin surged to a record high as profits soared")
    neg = c.classify_text("Exchange hacked, losses plunge amid fraud and bankruptcy")
    neu = c.classify_text("The meeting is scheduled for Tuesday in Berlin")
    assert pos.label == "positive" and pos.compound >= 0.05
    assert neg.label == "negative" and neg.compound <= -0.05
    assert neu.label == "neutral"


def test_crypto_overlay_terms_scored():
    c = FinVADERClassifier()
    result = c.classify_text("Analysts turn bullish ahead of breakout rally")
    assert "bullish" in result.matched_terms and result.label == "positive"


def test_empty_and_insufficient_text_uncertain():
    c = FinVADERClassifier()
    for text in ("", "   ", None, "Bitcoin"):
        result = c.classify_text(text)
        assert result.uncertain is True and result.label == "neutral"
        assert result.confidence_proxy == 0.0


def test_confidence_proxy_is_not_probability_contract():
    c = FinVADERClassifier()
    result = c.classify_text("Massive rally, huge surge, excellent profits soar")
    assert 0.0 <= result.confidence_proxy <= 1.0 and result.uncertain is False
    assert result.method  # method always documented on the result


def test_irrelevant_article_flagged_uncertain():
    c = FinVADERClassifier()
    article = normalize_article({
        "title": "Shop freezers in seven European countries hit by cyber-attack",
        "link": "https://example.com/freezers",
        "description": "Several stores affected across Europe today.",
    })
    classified = c.classify_article(article)
    assert classified.sentiment.uncertain is True
    assert classified.sentiment.confidence_proxy == 0.0


def test_article_preserved_and_api_sentiment_ignored():
    c = FinVADERClassifier()
    article = normalize_article({
        "title": "Crypto exchange hacked in major exploit",
        "link": "https://example.com/hack", "description": "Losses and fraud.",
        "sentiment": "bullish",  # wrong API tag must NOT drive our label
    })
    classified = c.classify_article(article)
    assert classified.article is article
    assert classified.sentiment.label == "negative"
    assert article.api_sentiment == "bullish"  # preserved separately
