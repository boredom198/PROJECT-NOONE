"""NOONE's crypto lexicon overlay for FinVADER (the "modified" part).

How the full lexicon is built (see ``classifier.py``):
1. Start from NLTK's VADER lexicon (general language).
2. Apply FinVADER upstream exactly: SentiBignomics (~7,300 financial
   terms, each polarity multiplied by 0.1 as upstream does), then
   Henry's 189-word earnings lexicon merged on top (Henry wins ties,
   matching upstream ``Merge`` order).
3. Apply this crypto overlay last, so crypto senses win for these
   terms (e.g. "long"/"short" as positions, "liquidation", "rug pull").

Valences use the VADER scale (roughly -4..+4). Every entry below is a
deliberate, reviewable choice — edit this dict to tune behaviour.
Multi-word entries (e.g. "rug pull") are matched by VADER only as
single tokens, so single-token forms are what take effect; phrase
forms are kept for documentation/future n-gram handling.
"""
from __future__ import annotations

CRYPTO_LEXICON: dict[str, float] = {
    # --- Bullish / positive crypto senses ---
    "bullish": 2.2, "bullrun": 2.4, "rally": 2.0, "rallies": 2.0,
    "surge": 2.2, "surged": 2.2, "surges": 2.2, "soar": 2.4, "soared": 2.4,
    "breakout": 1.9, "ath": 1.8, "halving": 1.2, "accumulation": 1.0,
    "outperform": 1.8, "upgrade": 1.5, "upgraded": 1.5, "adoption": 1.4,
    "partnership": 1.3, "listing": 0.9, "listed": 0.7, "mainnet": 1.0,
    "staking": 0.6, "airdrop": 0.8, "buyback": 1.4, "undervalued": 1.2,
    # --- Bearish / negative crypto senses ---
    "bearish": -2.2, "crash": -2.8, "crashed": -2.8, "plunge": -2.6,
    "plunged": -2.6, "dump": -2.0, "dumped": -2.0, "selloff": -2.2,
    "liquidation": -1.9, "liquidated": -2.1, "liquidations": -1.9,
    "hack": -2.7, "hacked": -2.7, "exploit": -2.6, "exploited": -2.6,
    "scam": -2.9, "fraud": -2.8, "bankrupt": -2.9, "bankruptcy": -2.9,
    "insolvent": -2.8, "lawsuit": -1.7, "sued": -1.7, "ban": -1.9,
    "banned": -2.0, "delist": -2.0, "delisted": -2.0, "downgrade": -1.6,
    "downgraded": -1.6, "outage": -1.8, "halted": -1.6, "vulnerability": -1.9,
    "rugpull": -3.0, "ponzi": -2.9, " capitulation": -2.0,
    # --- Position / market-structure terms (mild, context-dependent) ---
    "long": 0.4, "short": -0.4, "shorts": -0.4, "squeeze": -0.8,
    "dip": -0.7, "dips": -0.7, "correction": -1.1, "volatility": -0.4,
    "volatile": -0.5, "sideways": -0.1, "consolidation": 0.0,
}

# Fix accidental leading-space key above at import time (kept explicit
# so the table stays readable); VADER tokens never contain spaces.
CRYPTO_LEXICON = {k.strip(): v for k, v in CRYPTO_LEXICON.items()}
