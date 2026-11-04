"""Cryptocurrency asset identification for news (Component 2).

Maintainable alias dictionary + conservative matching rules:

- Full names (``bitcoin``, ``solana``...) match case-insensitively.
- Tickers (``BTC``, ``ETH``...) match only when the token appears in
  ALL-CAPS in the original text, so ordinary words never trigger.
- Ambiguous tickers (``SOL``, ``NEAR``, ``OP``, ``LINK``, ``DOT``,
  ``ATOM``, ``UNI``, ``ADA``, ``SUI``, ``TRX``) additionally require a
  crypto-context word in the article; otherwise they are reported as
  *uncertain* associations, not mentions.
- Exact matched terms are always preserved.

Mentioned != affected. An asset is listed as explicitly *affected*
only when an impact term (price move, hack, ETF, lawsuit, listing,
etc.) appears within 5 tokens after the asset mention in the same
sentence. Everything else stays "mentioned" or "uncertain". This is
a transparent heuristic, not proof of impact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# symbol -> (full names/aliases, ambiguous ticker?)
ASSET_ALIASES: dict[str, tuple[tuple[str, ...], bool]] = {
    "BTC": (("bitcoin", "btc", "xbt"), False),
    "ETH": (("ethereum", "eth", "ether"), False),
    "SOL": (("solana", "sol"), True),
    "XRP": (("xrp", "ripple"), False),
    "BNB": (("bnb", "binance coin"), False),
    "DOGE": (("dogecoin", "doge"), False),
    "ADA": (("cardano", "ada"), True),
    "LINK": (("chainlink", "link"), True),
    "LTC": (("litecoin", "ltc"), False),
    "DOT": (("polkadot", "dot"), True),
    "AVAX": (("avalanche", "avax"), False),
    "MATIC": (("polygon", "matic"), True),
    "TRX": (("tron", "trx"), True),
    "UNI": (("uniswap", "uni"), True),
    "ATOM": (("cosmos", "atom"), True),
    "NEAR": (("near protocol", "near"), True),
    "APT": (("aptos", "apt"), True),
    "ARB": (("arbitrum", "arb"), True),
    "OP": (("optimism", "op"), True),
    "SUI": (("sui",), True),
    "PEPE": (("pepe",), False),
    "SHIB": (("shiba inu", "shib"), False),
}

_TICKERS = {sym: sym.lower() for sym in ASSET_ALIASES}
_NAMES: dict[str, str] = {}  # alias(lower) -> symbol
for _sym, (_aliases, _amb) in ASSET_ALIASES.items():
    for _alias in _aliases:
        _NAMES[_alias] = _sym

_CONTEXT_TERMS = frozenset({
    "crypto", "cryptocurrency", "token", "coin", "blockchain",
    "exchange", "trading", "price", "market", "futures", "defi",
    "wallet", "binance", "coinbase", "etf", "investors",
})
_IMPACT_TERMS = frozenset({
    "price", "surge", "surged", "rally", "crash", "crashed", "plunge",
    "plunged", "soar", "soared", "drop", "dropped", "falls", "fell",
    "rises", "rose", "hack", "hacked", "exploit", "lawsuit", "etf",
    "approval", "approved", "listing", "listed", "delisted", "ban",
    "theft", "stolen", "liquidation", "selloff", "breakout",
})
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']*")
_SENTENCE_RE = re.compile(r"[^.!?]+")


@dataclass(frozen=True)
class AssetMatch:
    asset: str  # canonical symbol, e.g. "BTC"
    matched_terms: tuple[str, ...]  # exact terms as they appeared
    association: str  # "mentioned" | "affected" | "uncertain"


@dataclass(frozen=True)
class AssetIdentification:
    mentioned: tuple[AssetMatch, ...]
    affected: tuple[AssetMatch, ...]
    uncertain: tuple[AssetMatch, ...]

    @property
    def mentioned_assets(self) -> tuple[str, ...]:
        return tuple(m.asset for m in self.mentioned + self.affected)

    @property
    def affected_assets(self) -> tuple[str, ...]:
        return tuple(m.asset for m in self.affected)

    @property
    def uncertain_assets(self) -> tuple[str, ...]:
        return tuple(m.asset for m in self.uncertain)


def _sentences_with_tokens(text: str) -> list[list[str]]:
    return [_TOKEN_RE.findall(s) for s in _SENTENCE_RE.findall(text or "")]


def identify_assets(text: str | None) -> AssetIdentification:
    """Identify crypto assets mentioned/affected in ``text``."""
    if not text or not text.strip():
        return AssetIdentification((), (), ())
    lower = text.lower()
    has_context = any(w in lower.split() for w in _CONTEXT_TERMS) or any(
        f" {w} " in f" {lower} " for w in _CONTEXT_TERMS)
    # Collect hits: symbol -> list of (sentence_idx, token_idx, term)
    hits: dict[str, list[tuple[int, int, str]]] = {}
    sentences = _sentences_with_tokens(text)
    for s_idx, tokens in enumerate(sentences):
        lowered = [t.lower() for t in tokens]
        i = 0
        while i < len(tokens):
            matched = False
            # Multi-word names first ("binance coin", "shiba inu")
            for width in (2, 1):
                if i + width > len(tokens):
                    continue
                phrase = " ".join(lowered[i:i + width])
                symbol = _NAMES.get(phrase)
                if symbol is None:
                    continue
                original = " ".join(tokens[i:i + width])
                is_ticker_form = phrase == _TICKERS[symbol] and width == 1
                if is_ticker_form and original != original.upper():
                    continue  # tickers require ALL-CAPS in source
                hits.setdefault(symbol, []).append((s_idx, i, original))
                i += width
                matched = True
                break
            if not matched:
                i += 1
    mentioned: list[AssetMatch] = []
    affected: list[AssetMatch] = []
    uncertain: list[AssetMatch] = []
    for symbol, occurrences in hits.items():
        terms = tuple(dict.fromkeys(o[2] for o in occurrences))
        _aliases, ambiguous = ASSET_ALIASES[symbol]
        ticker_only = all(t.lower() == _TICKERS[symbol] for t in terms)
        if ambiguous and ticker_only and not has_context:
            uncertain.append(AssetMatch(symbol, terms, "uncertain"))
            continue
        is_affected = False
        for s_idx, t_idx, _term in occurrences:
            window = [w.lower() for w in sentences[s_idx][t_idx + 1:t_idx + 6]]
            if any(w in _IMPACT_TERMS for w in window):
                is_affected = True
                break
        match = AssetMatch(
            symbol, terms, "affected" if is_affected else "mentioned")
        (affected if is_affected else mentioned).append(match)
    by_asset = lambda m: m.asset  # noqa: E731
    return AssetIdentification(
        mentioned=tuple(sorted(mentioned, key=by_asset)),
        affected=tuple(sorted(affected, key=by_asset)),
        uncertain=tuple(sorted(uncertain, key=by_asset)))
