"""Unit tests: crypto asset identification/mapping."""
from noone.news_classifier import FinVADERClassifier, identify_assets


def assets_of(result):
    return set(result.mentioned_assets) | set(result.affected_assets)


def test_names_and_tickers():
    r = identify_assets("Bitcoin surges as BTC hits record high")
    assert "BTC" in assets_of(r)
    r = identify_assets("Ethereum and Solana rally together")
    assert {"ETH", "SOL"} <= assets_of(r)
    r = identify_assets("Ripple news: XRP listed on exchange")
    assert "XRP" in assets_of(r)
    r = identify_assets("Binance Coin (BNB) and Dogecoin DOGE climb")
    assert {"BNB", "DOGE"} <= assets_of(r)


def test_case_and_lowercase_ticker_rejected():
    # Names are case-insensitive; bare lowercase tickers are not matched.
    assert "BTC" in assets_of(identify_assets("bitcoin is quiet today"))
    assert identify_assets("I bought some btc yesterday").mentioned == ()


def test_ambiguous_tickers_need_context():
    # No crypto context: ambiguous SOL stays uncertain, not mentioned.
    r = identify_assets("The SOL was bright over the beach")
    assert "SOL" in r.uncertain_assets and "SOL" not in assets_of(r)
    # With context: mentioned.
    r = identify_assets("SOL token trades higher on the exchange")
    assert "SOL" in assets_of(r)
    # Ordinary 'near' is not NEAR protocol.
    assert identify_assets("The shop is near the park").mentioned == ()


def test_multi_asset_and_exact_terms_preserved():
    r = identify_assets("Bitcoin and Ethereum face selloff pressure")
    assert {"BTC", "ETH"} <= assets_of(r)
    btc = [m for m in r.mentioned + r.affected if m.asset == "BTC"][0]
    assert "Bitcoin" in btc.matched_terms


def test_mentioned_vs_affected():
    affected = identify_assets("Bitcoin price surges to record high")
    assert "BTC" in affected.affected_assets
    mentioned = identify_assets("A history of Bitcoin was published")
    assert "BTC" in mentioned.mentioned_assets
    assert "BTC" not in mentioned.affected_assets


def test_sentiment_intact_with_assets():
    c = FinVADERClassifier()
    pos = c.classify_text("Bitcoin surged to a record high, BTC rally soars")
    assert pos.label == "positive" and "BTC" in (
        pos.assets.mentioned_assets + pos.assets.affected_assets)
    neg = c.classify_text("Ethereum exchange hacked, ETH plunge on exploit")
    assert neg.label == "negative"
