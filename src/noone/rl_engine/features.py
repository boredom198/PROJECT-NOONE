"""Shared FeatureBuilder for Component 1 (Stage 1).

Single source of observation logic for training, evaluation, paper
trading, and live inference — no separate feature calculations.
Converts a Component 8 observation plus past-only candle history
into a fixed 26-feature ``np.float32`` vector (see FEATURE_NAMES).

Missing-data convention: unavailable funding/OI are encoded as
value 0.0 with ``funding_oi_available = 0`` — never as real zero
measurements. News has three explicit states (fresh / none /
unavailable) so neutral sentiment is never confused with missing
news. All outputs are clipped and guaranteed finite.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    # Market (12)
    "log_return_1", "log_return_5", "log_return_15",
    "candle_range_pct", "close_ema9_dev", "close_ema21_dev",
    "close_ema50_dev", "volume_change", "atr_pct",
    "funding_rate", "open_interest_change", "funding_oi_available",
    # Account/position (8)
    "position_side", "position_quantity_rel_equity",
    "entry_distance_pct", "unrealized_pnl_rel_equity",
    "realized_pnl_rel_equity", "available_balance_rel_equity",
    "margin_used_rel_equity", "episode_progress",
    # News (6)
    "news_available", "news_unavailable", "news_compound",
    "news_confidence", "news_count_log", "news_age_normalized",
)
OBSERVATION_DIM = len(FEATURE_NAMES)  # 26


@dataclass(frozen=True)
class NewsObservation:
    """Eligible-news summary at a decision timestamp.

    ``status``: "fresh" (eligible news), "none" (collection worked,
    nothing eligible), "unavailable" (collector/API failure).
    """

    status: str = "none"
    mean_compound: float = 0.0
    mean_confidence: float = 0.0
    count: int = 0
    age_seconds: float | None = None


def _ema(values: Sequence[float], period: int) -> float:
    if not values:
        return 0.0
    if len(values) < period:
        return float(values[-1])
    k = 2.0 / (period + 1)
    ema = float(values[0])
    for value in values[1:]:
        ema = float(value) * k + ema * (1 - k)
    return ema


def _log_return(closes: Sequence[float], lag: int) -> float:
    if len(closes) <= lag or closes[-1 - lag] <= 0 or closes[-1] <= 0:
        return 0.0
    return math.log(closes[-1] / closes[-1 - lag])


class FeatureBuilder:
    """Deterministic 26-feature builder. Past information only."""

    def __init__(self, *, news_max_age_hours: float = 24.0):
        self.news_max_age_seconds = news_max_age_hours * 3600.0

    def build(
        self, obs: dict[str, float], history: Sequence[Any],
        *, news: NewsObservation | None = None,
        funding_rate: float | None = None,
        open_interest_change: float | None = None,
    ) -> np.ndarray:
        """Build the observation vector.

        ``history`` must contain only candles available at the
        decision timestamp (current and past, chronological), each
        exposing open/high/low/close/volume. ``funding_rate`` and
        ``open_interest_change`` stay ``None`` (missing convention)
        until Component 4 supplies aligned series.
        """
        news = news or NewsObservation()
        closes = [float(c.close) for c in history if c.close]
        highs = [float(c.high) for c in history if c.high]
        lows = [float(c.low) for c in history if c.low]
        volumes = [float(c.volume or 0.0) for c in history]
        close = closes[-1] if closes else 0.0

        range_pct = ((highs[-1] - lows[-1]) / close
                     if highs and lows and close > 0 else 0.0)
        vol_change = 0.0
        if len(volumes) >= 2 and volumes[-2] > 0:
            vol_change = volumes[-1] / volumes[-2] - 1.0
        atr_pct = 0.0
        if len(closes) >= 2 and close > 0:
            ranges = []
            for i in range(1, len(closes)):
                tr = max(highs[i] - lows[i],
                         abs(highs[i] - closes[i - 1]),
                         abs(lows[i] - closes[i - 1]))
                ranges.append(tr)
            window = ranges[-14:]
            atr_pct = (sum(window) / len(window)) / close if window else 0.0

        equity = float(obs.get("equity", 0.0)) or 1.0
        entry = float(obs.get("entry_price", 0.0))
        side = float(obs.get("position_side", 0.0))
        qty = float(obs.get("position_quantity", 0.0))

        funding_oi_available = 1.0 if (
            funding_rate is not None or open_interest_change is not None
        ) else 0.0
        age_norm = 1.0
        if news.age_seconds is not None:
            age_norm = min(news.age_seconds, self.news_max_age_seconds) \
                / self.news_max_age_seconds

        values = [
            _log_return(closes, 1), _log_return(closes, 5),
            _log_return(closes, 15), range_pct,
            (close / _ema(closes, 9) - 1.0) if closes and _ema(closes, 9) else 0.0,
            (close / _ema(closes, 21) - 1.0) if closes and _ema(closes, 21) else 0.0,
            (close / _ema(closes, 50) - 1.0) if closes and _ema(closes, 50) else 0.0,
            vol_change, atr_pct,
            float(funding_rate) if funding_rate is not None else 0.0,
            float(open_interest_change)
            if open_interest_change is not None else 0.0,
            funding_oi_available,
            side,
            side * qty * close / equity if equity else 0.0,
            side * (close / entry - 1.0) if entry > 0 and close > 0 else 0.0,
            float(obs.get("unrealized_pnl", 0.0)) / equity,
            float(obs.get("realized_pnl", 0.0)) / equity,
            float(obs.get("available_balance", 0.0)) / equity,
            float(obs.get("margin_used", 0.0)) / equity,
            float(obs.get("progress", 0.0)),
            1.0 if news.status == "fresh" else 0.0,
            1.0 if news.status == "unavailable" else 0.0,
            float(news.mean_compound) if news.status == "fresh" else 0.0,
            float(news.mean_confidence) if news.status == "fresh" else 0.0,
            math.log1p(news.count) if news.status == "fresh" else 0.0,
            age_norm if news.status == "fresh" else 1.0,
        ]
        vector = np.asarray(values, dtype=np.float64)
        vector = np.nan_to_num(vector, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(vector, -10.0, 10.0).astype(np.float32)


# --- Canonical Feature Schema Specification (Single Source of Truth) ---

FEATURE_SCHEMA_VERSION: str = "noone-features-v1"

FEATURE_CALCULATION_PARAMETERS: dict[str, Any] = {
    "version": FEATURE_SCHEMA_VERSION,
    "feature_count": OBSERVATION_DIM,
    "feature_names": FEATURE_NAMES,
    "log_return_lags": (1, 5, 15),
    "ema_periods": (9, 21, 50),
    "atr_period": 14,
    "clipping_bounds": (-10.0, 10.0),
    "clipping_nan_replacement": 0.0,
    "clipping_posinf_replacement": 10.0,
    "clipping_neginf_replacement": -10.0,
    "default_news_max_age_hours": 24.0,
}

CANONICAL_FEATURE_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "log_return_1",
        "category": "market",
        "formula": "log(closes[-1] / closes[-2])",
        "guard": "len(closes) > 1 and closes[-2] > 0 and closes[-1] > 0",
        "fallback": 0.0,
    },
    {
        "name": "log_return_5",
        "category": "market",
        "formula": "log(closes[-1] / closes[-6])",
        "guard": "len(closes) > 5 and closes[-6] > 0 and closes[-1] > 0",
        "fallback": 0.0,
    },
    {
        "name": "log_return_15",
        "category": "market",
        "formula": "log(closes[-1] / closes[-16])",
        "guard": "len(closes) > 15 and closes[-16] > 0 and closes[-1] > 0",
        "fallback": 0.0,
    },
    {
        "name": "candle_range_pct",
        "category": "market",
        "formula": "(highs[-1] - lows[-1]) / closes[-1]",
        "guard": "highs and lows and closes[-1] > 0",
        "fallback": 0.0,
    },
    {
        "name": "close_ema9_dev",
        "category": "market",
        "formula": "closes[-1] / ema(closes, 9) - 1.0",
        "guard": "closes and ema(closes, 9) > 0",
        "fallback": 0.0,
    },
    {
        "name": "close_ema21_dev",
        "category": "market",
        "formula": "closes[-1] / ema(closes, 21) - 1.0",
        "guard": "closes and ema(closes, 21) > 0",
        "fallback": 0.0,
    },
    {
        "name": "close_ema50_dev",
        "category": "market",
        "formula": "closes[-1] / ema(closes, 50) - 1.0",
        "guard": "closes and ema(closes, 50) > 0",
        "fallback": 0.0,
    },
    {
        "name": "volume_change",
        "category": "market",
        "formula": "volumes[-1] / volumes[-2] - 1.0",
        "guard": "len(volumes) >= 2 and volumes[-2] > 0",
        "fallback": 0.0,
    },
    {
        "name": "atr_pct",
        "category": "market",
        "formula": "mean(true_range[-14:]) / closes[-1]",
        "guard": "len(closes) >= 2 and closes[-1] > 0",
        "fallback": 0.0,
    },
    {
        "name": "funding_rate",
        "category": "market",
        "formula": "float(funding_rate)",
        "guard": "funding_rate is not None",
        "fallback": 0.0,
    },
    {
        "name": "open_interest_change",
        "category": "market",
        "formula": "float(open_interest_change)",
        "guard": "open_interest_change is not None",
        "fallback": 0.0,
    },
    {
        "name": "funding_oi_available",
        "category": "market",
        "formula": "1.0 if funding_rate is not None or open_interest_change is not None else 0.0",
        "guard": "always",
        "fallback": 0.0,
    },
    {
        "name": "position_side",
        "category": "account",
        "formula": "float(obs['position_side'])",
        "guard": "side in {-1, 0, 1}",
        "fallback": 0.0,
    },
    {
        "name": "position_quantity_rel_equity",
        "category": "account",
        "formula": "side * qty * close / equity",
        "guard": "equity > 0",
        "fallback": 0.0,
    },
    {
        "name": "entry_distance_pct",
        "category": "account",
        "formula": "side * (close / entry - 1.0)",
        "guard": "entry > 0 and close > 0",
        "fallback": 0.0,
    },
    {
        "name": "unrealized_pnl_rel_equity",
        "category": "account",
        "formula": "float(obs['unrealized_pnl']) / equity",
        "guard": "equity > 0",
        "fallback": 0.0,
    },
    {
        "name": "realized_pnl_rel_equity",
        "category": "account",
        "formula": "float(obs['realized_pnl']) / equity",
        "guard": "equity > 0",
        "fallback": 0.0,
    },
    {
        "name": "available_balance_rel_equity",
        "category": "account",
        "formula": "float(obs['available_balance']) / equity",
        "guard": "equity > 0",
        "fallback": 0.0,
    },
    {
        "name": "margin_used_rel_equity",
        "category": "account",
        "formula": "float(obs['margin_used']) / equity",
        "guard": "equity > 0",
        "fallback": 0.0,
    },
    {
        "name": "episode_progress",
        "category": "account",
        "formula": "float(obs['progress'])",
        "guard": "0.0 <= progress <= 1.0",
        "fallback": 0.0,
    },
    {
        "name": "news_available",
        "category": "news",
        "formula": "1.0 if news.status == 'fresh' else 0.0",
        "guard": "always",
        "fallback": 0.0,
    },
    {
        "name": "news_unavailable",
        "category": "news",
        "formula": "1.0 if news.status == 'unavailable' else 0.0",
        "guard": "always",
        "fallback": 0.0,
    },
    {
        "name": "news_compound",
        "category": "news",
        "formula": "float(news.mean_compound)",
        "guard": "news.status == 'fresh'",
        "fallback": 0.0,
    },
    {
        "name": "news_confidence",
        "category": "news",
        "formula": "float(news.mean_confidence)",
        "guard": "news.status == 'fresh'",
        "fallback": 0.0,
    },
    {
        "name": "news_count_log",
        "category": "news",
        "formula": "math.log1p(news.count)",
        "guard": "news.status == 'fresh'",
        "fallback": 0.0,
    },
    {
        "name": "news_age_normalized",
        "category": "news",
        "formula": "min(news.age_seconds, max_age_seconds) / max_age_seconds if news.status == 'fresh' else 1.0",
        "guard": "news.status == 'fresh'",
        "fallback": 1.0,
    },
)


# Pinned development-verified implementation AST digest (covers _ema, _log_return,
# FeatureBuilder.__init__, FeatureBuilder.build, and NewsObservation).
PINNED_IMPLEMENTATION_DIGEST: str = (
    "08e6c0bb9982038e0ce3ea33d17e1de12211fd3c07baf8b00eee26111dffad6b"
)

# Pinned canonical feature schema hash for the default runtime configuration (24.0h news).
PINNED_CANONICAL_FEATURE_SCHEMA_HASH: str = (
    "b08a684a49e8917b9533bae811224b5e9eac4a925c12d2378f0e9c0c2e66f6a3"
)


def verify_feature_implementation_integrity() -> bool:
    """Compare live ASTs of features.py against the pinned implementation digest.

    Returns True if live AST matches pinned digest.
    Raises RuntimeError if live AST diverges from pinned digest.
    Returns False if source code inspection is unavailable (e.g. packaged environment).
    """
    import ast
    import hashlib
    import inspect
    import json
    import textwrap

    targets = [
        _ema,
        _log_return,
        FeatureBuilder.__init__,
        FeatureBuilder.build,
        NewsObservation,
    ]
    try:
        trees = []
        for t in targets:
            src = inspect.getsource(t)
            trees.append(ast.dump(ast.parse(textwrap.dedent(src))))
        digest = hashlib.sha256(json.dumps(trees).encode("utf-8")).hexdigest()
    except (OSError, TypeError):
        return False

    if digest != PINNED_IMPLEMENTATION_DIGEST:
        raise RuntimeError(
            "FeatureBuilder implementation AST diverges from pinned implementation digest: "
            f"live={digest!r} != pinned={PINNED_IMPLEMENTATION_DIGEST!r}"
        )
    return True


def compute_feature_schema_hash(
    *,
    news_max_age_hours: float | int | None = None,
    parameters: dict[str, Any] | None = None,
    definitions: tuple[dict[str, Any], ...] | None = None,
    implementation_digest: str | None = None,
    implementation_sources: list[str] | None = None,
) -> str:
    """Compute deterministic SHA-256 fingerprint for the canonical feature schema.

    Supports parameterization of runtime feature configuration (e.g. news_max_age_hours)
    while binding all 26 feature definitions and executable implementation ASTs.
    In environments where Python source files are stripped or unavailable, safely
    falls back to the pinned verified implementation digest.
    """
    import ast
    import hashlib
    import inspect
    import json
    import textwrap

    params = (
        dict(parameters)
        if parameters is not None
        else dict(FEATURE_CALCULATION_PARAMETERS)
    )
    if news_max_age_hours is not None:
        params["news_max_age_hours"] = float(news_max_age_hours)
    elif "news_max_age_hours" not in params:
        params["news_max_age_hours"] = float(
            params.get("default_news_max_age_hours", 24.0)
        )

    defs = definitions if definitions is not None else CANONICAL_FEATURE_DEFINITIONS

    if implementation_digest is not None:
        impl_digest = implementation_digest
    elif implementation_sources is not None:
        trees = [
            ast.dump(ast.parse(textwrap.dedent(src)))
            for src in implementation_sources
        ]
        impl_digest = hashlib.sha256(json.dumps(trees).encode("utf-8")).hexdigest()
    else:
        targets = [
            _ema,
            _log_return,
            FeatureBuilder.__init__,
            FeatureBuilder.build,
            NewsObservation,
        ]
        try:
            trees = []
            for t in targets:
                src = inspect.getsource(t)
                trees.append(ast.dump(ast.parse(textwrap.dedent(src))))
            impl_digest = hashlib.sha256(json.dumps(trees).encode("utf-8")).hexdigest()
        except (OSError, TypeError):
            # Deployment-safe fallback for packaged or frozen binaries
            impl_digest = PINNED_IMPLEMENTATION_DIGEST

    payload_dict: dict[str, Any] = {
        "version": params.get("version", FEATURE_SCHEMA_VERSION),
        "parameters": params,
        "definitions": list(defs),
        "implementation_digest": impl_digest,
    }

    serialized = json.dumps(payload_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


CANONICAL_FEATURE_SCHEMA_HASH: str = PINNED_CANONICAL_FEATURE_SCHEMA_HASH
