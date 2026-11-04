# PROJECT NOONE — Market & News Signal Audit

Audit date: 2026-11-02. Scope: Component 4 (Market Real-Time & News
Data APIs) and Component 2 (News Classifier) only. This is an
inspection report; no source code was modified.

Evidence base: `src/noone/market_data/binance_futures.py`,
`src/noone/market_data/models.py`, `src/noone/news_data/crypto_news.py`,
`src/noone/news_data/models.py`, `src/noone/news_classifier/classifier.py`,
`src/noone/news_classifier/lexicon.py`, `src/noone/diagnostic.py`,
`tests/test_market_data.py`, `tests/test_news_data.py`,
`tests/test_classifier.py`, `tests/test_live.py`, `README.md`,
`pyproject.toml`. A repository search for `RSI|MACD|EMA|ATR|Bollinger|VWAP|indicator`
over `src/` and `tests/` returned zero matches.

## Executive verdict

NOONE has a verified live *latest-value* market feed (REST) and a
working live news-plus-sentiment feed. It does **not** yet have:
historical-data pagination, gap/duplicate candle validation, any
technical indicator, any persistence/time-series store, normalized
WebSocket signals, per-asset news mapping, news aggregation, or any
market/news synchronization code. Those omissions — not the live
collectors — are what block direct DQN/RL use.

## 1. Existing market signals (Task 1)

Collector: `BinanceFuturesCollector` in
`src/noone/market_data/binance_futures.py`. REST base
`https://fapi.binance.com` (`REST_BASE`), WS base
`wss://fstream.binance.com` (`WS_BASE`). All REST methods go through
`_get()` (retry, 418/429 backoff) and pure `parse_*` functions.
Missing numeric fields become `None` via `_f()` in
`src/noone/market_data/models.py`; valid `0.0` stays `0.0`.

Key distinction for every row below: `stream()` yields **raw decoded
JSON dicts** (`stream`/`data` keys) with no normalization, no model,
no locally added `received_at`, and no validation. Therefore
"available via WS" is only true as unprocessed transport. All
normalized signals below are REST-only in practice.

| Signal | Implemented | Endpoint (collector method) | Normalized format | In `snapshot()` | REST/WS | Timestamps / freshness |
|---|---|---|---|---|---|---|
| Last traded price | Yes | `GET /fapi/v1/ticker/24hr` (`ticker_24h()`) | `Ticker24h.last_price: float\|None` | Yes | REST only (normalized) | `Ticker24h.open_time`/`close_time` are the 24h window bounds, not a trade event time; `received_at` local UTC |
| Mark price | Yes | `GET /fapi/v1/premiumIndex` (`mark_price()`) | `MarkPrice.mark_price: float\|None` | Yes | REST; WS only raw (`@markPrice`, diagnostic probe) | `MarkPrice.event_time` from payload `time`; `received_at`; diagnostic checks `is_stale(event_time, 30)` |
| Index price | Yes | Same `premiumIndex` | `MarkPrice.index_price: float\|None` | Yes (same object) | REST only (normalized) | Same `event_time`/`received_at` as mark price |
| Estimated settle price | Yes (extra) | Same `premiumIndex` | `MarkPrice.estimated_settle_price: float\|None` | Yes (same object) | REST only | Same as above |
| OHLCV / candlesticks | Yes | `GET /fapi/v1/klines` (`klines()`) | `list[Kline]`: `open/high/low/close/volume/quote_volume: float\|None`, `trades: int\|None` | Yes, `limit=20` only | REST; WS only raw (`@kline_1m`, diagnostic probe) | `Kline.open_time`/`close_time` (UTC-aware via `ms_to_dt()`); `received_at` per parsed batch |
| Trading volume | Yes | Klines (`volume`, `quote_volume`) + ticker 24hr (`volume`, `quote_volume`) | `float\|None` on `Kline` and `Ticker24h` | Yes (both) | REST only (normalized) | Inherits candle/ticker timestamps |
| Taker-buy volume etc. | No | Binance kline columns 9–11 exist but `parse_klines()` ignores them | — | No | — | — |
| 24h ticker statistics | Yes (subset) | `GET /fapi/v1/ticker/24hr` (`ticker_24h()`) | `Ticker24h`: `price_change`, `price_change_percent`, `high_price`, `low_price`, `volume`, `quote_volume` | Yes | REST only | Window `open_time`/`close_time`; no per-tick event time |
| 24h weighted avg price / count / last qty | No | Present in payload (`raw` only), not parsed by `parse_ticker_24h()` | Only via `Ticker24h.raw` | Only inside `raw` | REST | — |
| Order-book depth | Yes | `GET /fapi/v1/depth` (`depth()`) | `DepthSnapshot.bids/asks: tuple[(price, qty)]`, `last_update_id` | **No** | REST only (normalized); no diff-depth WS handling | `DepthSnapshot.event_time` from payload `T` or `E`; `received_at` |
| Current funding rate | Yes | `premiumIndex` field `lastFundingRate` (`mark_price()`) | `MarkPrice.last_funding_rate: float\|None`, plus `next_funding_time` | Yes | REST only (normalized) | `event_time` + `received_at` |
| Funding-rate history | Yes | `GET /fapi/v1/fundingRate` (`funding_history()`) | `list[FundingRatePoint]`: `funding_rate`, `funding_time`, `mark_price` | Yes, `limit=3` only (method default 10) | REST only | `funding_time` (settlement time); no separate event time |
| Open interest (current) | Yes | `GET /fapi/v1/openInterest` (`open_interest()`) | `OpenInterest.open_interest: float\|None` | Yes | REST only | `event_time` from payload `time`; `received_at` |
| Open-interest history | No | `/futures/data/openInterestHist` is never called | — | No | — | — |
| Recent trades / agg trades | No | No `trades()`/`aggTrades()` method exists; only `Kline.trades` (count per candle) | — | No | WS raw only if caller subscribes | — |
| Exchange / contract metadata | Yes | `GET /fapi/v1/exchangeInfo` (`exchange_info()`) | `list[ExchangeSymbol]`: `pair`, `contract_type`, `status`, `base_asset`, `quote_asset`, `margin_asset` | No (diagnostic calls it separately) | REST only | **No timestamps at all** on `ExchangeSymbol` (no `received_at`) |
| Server time / connectivity | Yes | `GET /fapi/v1/ping` (`ping()`), `GET /fapi/v1/time` (`server_time_ms()`) | `bool`, `int` (ms) | No | REST only | Server ms; not stored alongside snapshots |

`snapshot(symbol, interval="1m")` (`binance_futures.py:233`) returns
a plain `dict[str, Any]` with keys `symbol`, `ticker_24h`,
`mark_price`, `open_interest`, `klines` (20), `funding_history` (3).
It has **no snapshot-level timestamp**, and its five REST calls run
sequentially, so members can be seconds apart (non-atomic).

Freshness: `age_seconds()` and `is_stale()` exist in
`market_data/models.py:28-37` (`is_stale(None, ...)` is `True`), but
the collector never enforces them. Only the diagnostic applies
`is_stale(mark.event_time, 30)`. Stale data is therefore *detectable
by callers*, not *rejected by the collector*.

WebSocket status: `stream()` (`binance_futures.py:244`) reconnects
with capped exponential backoff and skips malformed JSON frames, but
per `README.md:90-94` it is **not live-verified** (handshake OK, zero
frames in 25s from the development network; sandbox also blocks
asyncio sockets). Treat REST as the only verified live path today.

## 2. Historical market data (Task 2)

Implementation: `BinanceFuturesCollector.klines(symbol, interval="1m",
limit=100)` → `GET /fapi/v1/klines` → `parse_klines()`. Nothing else
in the codebase retrieves history (no `startTime`/`endTime` anywhere).

- **Intervals:** not enumerated or validated in code. Any string is
  passed straight to Binance (default `"1m"`; `snapshot()` forwards
  its `interval` argument). Whatever Binance accepts works; a typo
  surfaces as a Binance HTTP error via `_get()`, not a local check.
- **How many candles:** one request only. Caller-set `limit`
  (default 100; `snapshot()` hardcodes 20). The code sets no maximum
  and implements no continuation, so in code terms the retrievable
  history is "the most recent N candles of a single response".
- **Pagination:** **not implemented** for klines. No `startTime`/
  `endTime` parameters, no cursor loop. (The *news* collector
  paginates; the market collector does not.)
- **Missing candles:** **not detected.** `parse_klines()` silently
  skips rows shorter than 8 columns and never checks that consecutive
  `open_time` values are spaced by the interval. A gap is invisible.
- **Duplicate candles:** **not handled.** No deduplication, sorting,
  or monotonicity check; output order is raw API order.
- **Multiple symbols:** no batch/multi-symbol helper. A caller must
  loop and call `klines()` once per symbol (one HTTP request each).
- **Candle completeness:** not tracked. The parser keeps no
  "is this candle closed" flag, so the last (still-forming) candle is
  indistinguishable from closed candles — a lookahead trap if used
  naively for indicators/decisions.
- **Ignored fields:** kline columns 9–11 (taker-buy base/quote
  volume, ignore flag) are discarded; only column 8 (`trades` count)
  is kept beyond OHLCV/quote-volume.
- **Timestamps:** consistent and good. `ms_to_dt()` converts Binance
  millisecond epochs with `tz=timezone.utc`; `open_time`,
  `close_time`, and `received_at` (`utcnow()`) are all UTC-aware.

**Sufficiency for indicators:** the *fields* suffice (open, high,
low, close, volume, quote_volume — see §4), but the *retrieval shape*
does not yet suffice for serious indicator work: 20 candles from
`snapshot()` cannot warm up EMA 50/MACD/Bollinger; the default 100
barely can; and without pagination there is no way to build the
long, gap-checked history that RL training backtests need. History
length, continuity validation, and closed-candle filtering must be
added before indicators are trustworthy (see §8, P0).

## 3. Existing news signals (Task 3)

Collector: `CryptoNewsCollector.fetch_latest()` (`GET /news` on
`https://cryptocurrency.cv/api`, paginating because the free tier
caps a page at 3 articles), `poll()` for polling mode, and
`normalize_article()`/`normalize_feed()` in
`src/noone/news_data/crypto_news.py`. Model: `NewsArticle` in
`src/noone/news_data/models.py`. Classifier: `FinVADERClassifier`,
`SentimentResult`, `ClassifiedArticle` in
`src/noone/news_classifier/classifier.py`; overlay `CRYPTO_LEXICON`
in `src/noone/news_classifier/lexicon.py`.

| Item | Status | Source field / model field |
|---|---|---|
| Article title | Yes | `NewsArticle.title` (required; blank → article dropped) |
| Description / article text | Partial | `NewsArticle.description` (may be `""`); no full article body is fetched. Classified text is `text_for_classification` = title + description only |
| Publisher / source | Yes | `NewsArticle.source`, `source_key` (either may be `None`) |
| Article URL | Yes | `NewsArticle.url` from `link` or `url` (required; missing → dropped) |
| Published timestamp | Yes, nullable | `NewsArticle.published_at` via `parse_pub_date()`; unparseable/missing → `None` |
| Retrieval timestamp | Yes | `NewsArticle.retrieved_at` (UTC, set at normalization time) |
| Category | Yes, nullable | `NewsArticle.category` — a topic string (e.g. `bitcoin`, `journalism`), not an asset list |
| Asset / crypto tags | **No (not reliably)** | `NewsArticle.tags` is parsed if the API sends a list, but the live free feed sends no `tags` (README live check), so it is normally `()`. No asset extraction exists |
| API sentiment tag | Preserved only | `NewsArticle.api_sentiment`; live free feed currently sends none (`None`). Never used as classifier output |
| Sentiment label | Yes | `SentimentResult.label`: positive/negative/neutral, VADER ±0.05 on compound |
| Pos / neg / neu scores | Yes | `SentimentResult.pos`, `.neg`, `.neu` (VADER proportions, sum ≈ 1) |
| Compound score | Yes | `SentimentResult.compound` in [-1, 1]; duplicated in `ClassifiedArticle.raw_scores` |
| Confidence proxy | Yes | `SentimentResult.confidence_proxy` = \|compound\| × (0.6 + 0.4 × lexicon coverage), [0,1]; **not a probability** |
| Matched lexicon terms | Yes | `SentimentResult.matched_terms` (every token found in the merged lexicon, duplicates kept); `token_count` also kept |
| Relevance / uncertainty | Yes | `SentimentResult.uncertain` (+ proxy forced to 0). Set when <3 tokens, zero lexicon hits, or (in `classify_article()`) no term from `_RELEVANCE_TERMS` |
| Method / degradation flag | Yes | `SentimentResult.method`, `.degraded` (True = FinVADER lexicons unavailable, VADER+crypto fallback) |
| Original article preserved | Yes | `ClassifiedArticle.article` is the original `NewsArticle` object; `NewsArticle.raw` keeps the raw API payload |

Specific determinations:

1. **Affected cryptocurrencies cannot be identified reliably.**
   There is no asset field, no ticker/alias mapping (e.g. "Bitcoin" →
   `BTCUSDT`), and no extraction output. `_RELEVANCE_TERMS` (which
   includes `bitcoin`, `btc`, `eth`, `sol`, …) is used only as a
   boolean relevance gate inside `classify_article()`; its matches
   are discarded. Any per-symbol news feature today would need new
   extraction logic — and naive substring matching would misfire
   (e.g. English words colliding with tickers).
2. **Duplicates are filtered**, at three levels: within a page
   (`normalize_feed()` by `article_id`), across pages of one fetch
   (`seen_here`), and across polls (`_seen_ids`, applied when
   `only_new=True`; IDs are recorded even without `only_new`).
   Limits: `article_id` falls back to SHA-256(URL|title), so the same
   story at a different URL, or same URL with a changed title, is
   *not* deduplicated; `_seen_ids` grows unboundedly and is in-memory
   only (lost on restart).
3. **Original + output preserved together:** yes, via
   `ClassifiedArticle(article, sentiment, raw_scores)`. Nothing is
   persisted to disk/database anywhere in the project.
4. **News freshness can be calculated but no helper exists.** Both
   `published_at` and `retrieved_at` are UTC-aware, so age =
   `now - published_at` is computable by a caller. Unlike market
   data, there is no `age_seconds`/`is_stale` equivalent applied to
   news, no maximum-age filter, and `published_at=None` must be
   handled by the caller.
5. **Insufficient-information articles are handled conservatively:**
   malformed entries (non-dict, blank title, missing URL) are dropped
   by `normalize_article()` (silently — no count or log); empty
   description degrades to title-only scoring; empty/short/no-hit
   text yields `label="neutral"` *by convention* with
   `uncertain=True`, `confidence_proxy=0.0`. Non-Latin text is a real
   weakness: the tokenizer regex `[a-z']+` discards e.g. Chinese
   characters, producing zero matched terms → uncertain (observed
   live with a BlockTempo article in the diagnostic).

## 4. Technical-indicator feasibility (Task 4)

No indicator exists today (zero code matches for any indicator name).
All nine are *computable locally* from `Kline` fields with pure
Python — no additional API is needed for the math — but each needs
more closed candles than `snapshot()` provides (20). Minimum warm-ups
below are the standard parameter defaults; "reliable" history for
RL features is typically 2–5× the minimum.

| Indicator | Exists | Required inputs (all on `Kline`?) | Min candles | Extra history needed today |
|---|---|---|---|---|
| RSI (14) | No | `close` ✓ | 15 (14 changes) | Yes — use `klines(limit≥100)`, not snapshot's 20 |
| MACD (12, 26, 9) | No | `close` ✓ | 35 (26 + 9 signal) | Yes — ≥100 recommended for stable signal line |
| EMA 9 | No | `close` ✓ | 9 | No, if fed ≥9 closes; snapshot's 20 suffices minimally |
| EMA 21 | No | `close` ✓ | 21 | Marginal — snapshot's 20 is one short; use limit ≥ 50 |
| EMA 50 | No | `close` ✓ | 50 | Yes — snapshot insufficient; default `limit=100` is the floor |
| ATR (14) | No | `high`, `low`, previous `close` ✓ | 15 | Yes — ≥50 recommended |
| Bollinger Bands (20, 2σ) | No | `close` ✓ | 20 | Marginal — snapshot exactly 20 leaves no lookback context |
| VWAP | No | `volume`, `quote_volume` ✓ (or typical price × volume) | Session-dependent | Needs an anchor/reset rule (e.g. UTC day) and continuous intraday candles; no session logic exists |
| Volume change | No | `volume` ✓ | 2 | No |

Feasibility verdict: **data-feasible, code-absent, history-limited.**
The blocker is §2 (single-page retrieval, no gap checks, unclosed
last candle), not missing fields. VWAP additionally needs a defined
session anchor; quote_volume/volume over a window gives a defensible
window-VWAP without new API data.

## 5. Data synchronization assessment (Task 5)

What exists: both domains use UTC-aware `datetime` throughout —
market `ms_to_dt()` (exchange ms epochs) and `utcnow()`
(`received_at`/`retrieved_at`), news `parse_pub_date()` (ISO-8601,
naive input assumed UTC). There is **no synchronization code**: no
join, no alignment, no storage, no shared clock discipline beyond
"everything is UTC", and `snapshot()` itself is five sequential REST
calls with no snapshot timestamp.

| Concern | Market data | News data |
|---|---|---|
| Event time | `MarkPrice.event_time`, `OpenInterest.event_time`, `DepthSnapshot.event_time`, `Kline.open/close_time`, `Ticker24h` window bounds — all nullable | `published_at` — nullable (`None` on bad/missing date) |
| Local receipt time | `received_at` on every market model except `ExchangeSymbol` (set at parse time, after the HTTP round trip) | `retrieved_at` (set at normalization) |
| Missing/delayed times | `_f`/`ms_to_dt` yield `None`; `is_stale(None)=True` gives callers a guard | No stale helper; `published_at=None` silently possible |
| Stale detection | Helpers exist (`age_seconds`, `is_stale`); only the diagnostic enforces a 30s mark-price check | None |
| Clock skew | `server_time_ms()` exists but is never compared with local time; skew is unmeasured | No server-clock check |

**Preventing future news from influencing historical decisions
(lookahead discipline for future RL training):**

1. An article is usable for a decision at time *t* only if
   `published_at <= t` **and** it was actually available by *t*.
   For live-collected data, availability = `retrieved_at`
   (first-seen time). Both conditions must hold; using
   `published_at` alone leaks articles that the system had not yet
   fetched, and using `retrieved_at` alone leaks backfilled history.
2. Articles with `published_at is None` must be excluded from
   time-aligned training (or timestamped at `retrieved_at` and used
   only from then on) — never silently placed at retrieval time
   against earlier candles.
3. Align news to candles by *availability time*, and only let an
   article affect decisions on candles that **close after** it
   became available (news arriving during candle *k* is first usable
   for the decision made at/after *k*'s close). The same rule applies
   to the still-forming last kline (§2): never compute features from
   an unclosed candle as if final.
4. The current news API is a *latest-news* feed, not a point-in-time
   archive: anything fetched today carries `retrieved_at = now`, so
   it cannot honestly be replayed against last month's candles.
   Historical RL training with news requires building a
   first-seen-stamped news archive going forward (collect now, train
   later), or restricting news features to live/paper operation.
5. Persist `published_at`, `retrieved_at`/first-seen, and the
   classifier output together (currently nothing is persisted), and
   record the market snapshot's per-member `event_time`s rather than
   a single fetch time, because `snapshot()` members are not atomic.

## 6. Candidate-signal feasibility for future AI trading (Task 6)

Classification of the 12 proposed DQN features. This is a feasibility
review only; the observation space is not finalized here.

| # | Proposed feature | Classification | Basis |
|---|---|---|---|
| 1 | Price return | Can be calculated from existing data | `Kline.close` series / `Ticker24h.last_price`; needs only consecutive closes. No return function exists |
| 2 | RSI | Can be calculated from existing data | `close` series; needs ≥15–100 closed candles (§4) |
| 3 | MACD | Can be calculated from existing data | `close` series; needs ≥35–100 closed candles |
| 4 | EMA trend difference (e.g. EMA9−EMA21) | Can be calculated from existing data | `close` series; needs ≥21–50 closed candles |
| 5 | ATR / volatility | Can be calculated from existing data | `high`/`low`/`close` (ATR) or close-return dispersion |
| 6 | Volume change | Can be calculated from existing data | `Kline.volume` consecutive difference/ratio |
| 7 | Funding rate | **Already available** | `MarkPrice.last_funding_rate` (+ `FundingRatePoint` history, snapshot limit 3) |
| 8 | Open-interest change | Requires additional data collection | Only current OI exists (`OpenInterest`); change needs repeated polling with persistence or the unused open-interest-history endpoint |
| 9 | Order-book imbalance | Can be calculated from existing data | Sum `DepthSnapshot.bids` vs `asks` quantities; caveat: `depth()` is REST-only, not in `snapshot()`, and depth stales in seconds |
| 10 | News sentiment | **Already available** | Per-article `SentimentResult.compound`/`.label`; caveat: no per-asset mapping and no time-window aggregation exists (see notes) |
| 11 | News confidence proxy | **Already available** | `SentimentResult.confidence_proxy` per article (score proxy, not a probability) |
| 12 | News freshness | Can be calculated from existing data | `published_at` + `retrieved_at` are stored; age is a subtraction away, but no helper/window function exists |

Notes: features 10–11 are "already available" **per article**. A
per-symbol, per-candle news feature (e.g. "mean compound for BTCUSDT
news in the last hour") is *not* available: it additionally needs
asset mapping (§3.1) and aggregation over an availability-aligned
window (§5). Feature 8 is the only market feature whose raw history
genuinely does not exist in the current code path.

## 7. Actual implementation gaps

1. No kline pagination (`startTime`/`endTime`), no interval
   validation, no maximum-history strategy.
2. No candle continuity checks: gaps, duplicates, ordering, and the
   unclosed final candle are all unhandled.
3. No persistence anywhere: no database/files, no time-series store
   for OI, funding, depth, snapshots, or news. Every poll starts
   from zero history.
4. No open-interest history and no trades/agg-trades collection
   (README's "trades/depth" overstates: only depth + per-candle
   trade *count* exist).
5. WebSocket is a raw passthrough: no normalization into models, no
   local receipt timestamps, no sequence/update-id handling for
   depth, and not live-verified from the development network.
6. `snapshot()` is non-atomic, untimestamped as a whole, excludes
   depth, and hardcodes klines=20 / funding=3.
7. Freshness helpers exist but are unenforced (except the
   diagnostic); `ExchangeSymbol` has no timestamp; clock skew vs
   `server_time_ms()` is never measured.
8. News: no affected-asset identification, no sentiment aggregation,
   no freshness/staleness helper, no full-text fetching
   (title+description only), silent dropping of malformed articles,
   unbounded in-memory dedup set, weak non-Latin-text handling.
9. No technical indicators, returns, or feature calculations of any
   kind (by design at this stage — confirmed absent, not partial).
10. No market/news alignment, availability-time discipline, or
    point-in-time news archive — the precondition for leak-free RL
    training with news (§5).

## 8. Recommended improvements (prioritized; not implemented here)

**P0 — required before any indicator/RL work:**
1. Historical klines: `startTime`/`endTime` pagination, interval
   validation, sort + dedupe by `open_time`, gap detection against
   the interval, and an explicit closed-candle flag/filter.
2. Point-in-time news archive: persist every classified article with
   `published_at`, first-seen `retrieved_at`, and classifier output;
   enforce the §5 availability rules in any future dataset builder.
3. Affected-asset extraction for news (curated alias→symbol mapping,
   output as a new field) — without it, per-symbol news features
   (Task 6 #10–12) cannot be built honestly.

**P1 — required for the richer proposed features:**
4. Time-series collection for open interest (history endpoint or
   polled persistence) and funding, so OI change and funding
   dynamics become computable; add depth to `snapshot()` (or a
   dedicated imbalance-ready depth call).
5. Normalize WebSocket streams into the existing models with local
   receipt timestamps, then live-verify on an unrestricted network;
   until then, document REST as the production path.
6. News freshness helper + windowed sentiment aggregation
   (per asset, availability-aligned), reusing the market
   `age_seconds`/`is_stale` pattern.

**P2 — robustness and quality:**
7. Timestamp `snapshot()` as a whole, measure clock skew against
   `server_time_ms()`, and enforce staleness thresholds in the
   collector (not only the diagnostic).
8. Count/log dropped malformed articles and candles instead of
   silently skipping; bound or persist the news dedup set.
9. Improve non-Latin and short-text news handling (language tag is
   already requestable via `fetch_latest(lang=...)` but unused by
   the classifier path); consider classifying description-only
   fallback already present (title-only today) — no change needed,
   but document the behavior.
10. Only after P0: implement the §4 indicators as a separate,
    unit-tested feature module feeding (not embedding) the future
    Trading Environment / DQN observation builder.


