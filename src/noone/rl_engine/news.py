"""Point-in-time news variation for Stage 2 training (Component 1).

The Stage 1 Gymnasium adapter already enforces the strict rule via
Component 2's archive: ``published_at <= t AND first_seen_at <= t``,
classified, non-uncertain, within the staleness window. This module
adds *training variation* on top of that archive interface without
changing Component 2:

- Genuine no-news: the archive returns no eligible articles -> the
  adapter reports status ``"none"`` (market-only observation).
- Collector/API failure: the archive raises -> the adapter reports
  status ``"unavailable"``. This wrapper never fabricates that state.
- Simulated dropout: with probability ``dropout_probability``
  (default 0.1, configurable) a query returns no articles, so the
  adapter reports ``"none"`` -- *not* ``"unavailable"``. Dropped
  samples are therefore never randomly mislabelled as API failures.
  Dropout events are counted on the wrapper (``dropout_count`` /
  ``last_query_dropped``) so they stay distinguishable in logs/tests
  from a genuine empty result, even though both yield the same valid
  market-only observation.

Dropout draws come from a seeded RNG: same seed + same call sequence
reproduces the same dropout pattern.
"""
from __future__ import annotations

import random
from datetime import datetime
from typing import Any

from noone.rl_engine.features import FeatureBuilder, NewsObservation


class NewsDropoutArchive:
    """Archive-compatible wrapper adding seeded simulated dropout."""

    def __init__(self, archive: Any, *, dropout_probability: float = 0.1,
                 seed: int = 42):
        if not 0.0 <= dropout_probability <= 1.0:
            raise ValueError("dropout_probability must be in [0, 1]")
        self._archive = archive
        self.dropout_probability = float(dropout_probability)
        self._seed = seed
        self._rng = random.Random(seed)
        self.query_count = 0
        self.dropout_count = 0
        self.last_query_dropped = False

    def reset(self, seed: int | None = None) -> None:
        """Restart the dropout sequence (reproducible with same seed)."""
        self._rng = random.Random(self._seed if seed is None else seed)
        self.query_count = 0
        self.dropout_count = 0
        self.last_query_dropped = False

    def available_at(self, decision_time: datetime, *, asset: str | None = None,
                     limit: int = 500):
        self.query_count += 1
        self.last_query_dropped = False
        if self.dropout_probability and self._rng.random() < self.dropout_probability:
            self.dropout_count += 1
            self.last_query_dropped = True
            return []  # -> adapter "none" state, never "unavailable"
        # Real errors propagate: the adapter maps them to "unavailable".
        return self._archive.available_at(decision_time, asset=asset, limit=limit)

    def classification_for(self, article_id: str):
        return self._archive.classification_for(article_id)


def news_observation_at(
    archive: Any, decision_time: datetime, *, asset: str,
    feature_builder: FeatureBuilder | None = None,
) -> NewsObservation:
    """Eligible-news summary at ``decision_time`` (point-in-time only).

    Mirrors the Stage 1 adapter logic so tests/callers can inspect the
    three states directly: fresh / none / unavailable.
    """
    builder = feature_builder or FeatureBuilder()
    try:
        articles = archive.available_at(decision_time, asset=asset)
    except Exception:
        return NewsObservation(status="unavailable")
    compounds, confidences, latest_age = [], [], None
    for article in articles:
        result = archive.classification_for(article.article_id)
        if not result or result.get("uncertain"):
            continue
        if article.published_at is None:
            continue
        age = (decision_time - article.published_at).total_seconds()
        if age < 0 or age > builder.news_max_age_seconds:
            continue
        compounds.append(float(result["compound"]))
        confidences.append(float(result["confidence_proxy"]))
        latest_age = age if latest_age is None else min(latest_age, age)
    if not compounds:
        return NewsObservation(status="none")
    return NewsObservation(
        status="fresh", mean_compound=sum(compounds) / len(compounds),
        mean_confidence=sum(confidences) / len(confidences),
        count=len(compounds), age_seconds=latest_age)
