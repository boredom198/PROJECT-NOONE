"""Gymnasium adapter around Component 8 (Stage 1, Component 1).

Thin adapter only: all accounting, risk, fees, funding, slippage,
and execution stay in ``PerpetualFuturesEnv``. No DQN, training, or
trading logic lives here.

Seeded episode sampling: ``reset(seed=...)`` deterministically
selects a start offset for the episode window (same seed → same
window and trajectory). ``options={"start_index": i}`` overrides it.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from noone.market_data.models import FundingHistory, Kline
from noone.rl_engine.features import (
    OBSERVATION_DIM,
    FeatureBuilder,
    NewsObservation,
)
from noone.trading_env import Action, PerpetualFuturesEnv, TradingConfig


class NooneGymEnv(gym.Env):
    """Discrete(5) Gymnasium view of the Component 8 simulator."""

    metadata = {"render_modes": []}

    def __init__(
        self, candles: list[Kline], config: TradingConfig | None = None,
        *, episode_window: int | None = None,
        news_archive: Any = None, news_asset: str | None = None,
        feature_builder: FeatureBuilder | None = None,
        funding_history: FundingHistory | None = None,
        invalid_action_penalty: float = 0.001,
    ):
        super().__init__()
        if len(candles) < 2:
            raise ValueError("At least 2 candles are required")
        if isinstance(invalid_action_penalty, bool) or not isinstance(invalid_action_penalty, (int, float)):
            raise TypeError(f"invalid_action_penalty must be numeric, got {type(invalid_action_penalty).__name__}")
        penalty_val = float(invalid_action_penalty)
        import math
        if not math.isfinite(penalty_val) or penalty_val < 0.0:
            raise ValueError(f"invalid_action_penalty must be finite and non-negative, got {penalty_val}")
        self._invalid_action_penalty = penalty_val
        self._candles = list(candles)
        self._config = config
        self._episode_window = episode_window
        self._news_archive = news_archive
        self._news_asset = news_asset.upper() if news_asset else None
        self._funding_history = funding_history
        self._episode_started = False
        self._completed_episode_count = 0
        self._completed_total_funding = 0.0
        self._completed_ledger_entries = 0
        self._validate_funding_coverage()
        self.features = feature_builder or FeatureBuilder()
        self.action_space = spaces.Discrete(5)  # Action enum 0..4
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(OBSERVATION_DIM,),
            dtype=np.float32)
        self._slice: tuple[int, int] = (0, len(self._candles))
        self.env = self._make_env(0)

    def _validate_funding_coverage(self) -> None:
        """Fail-fast funding checks over the adapter's full candle range.

        The original full ``FundingHistory`` is forwarded unchanged
        to every episode env; this adapter-level check only ensures
        an inadequate history fails before training starts. Each
        episode env still runs Component 8's full strict gate.
        """
        config = self._config or TradingConfig()
        history = self._funding_history
        if history is None:
            if config.funding_mode == "historical":
                raise ValueError(
                    "funding_mode='historical' requires a full "
                    "FundingHistory object via funding_history=...; "
                    "a bare list of funding events is not accepted")
            return
        if config.funding_mode != "historical":
            raise ValueError(
                "funding_history supplied but funding_mode is "
                "'constant': set funding_mode='historical' "
                "explicitly to use historical funding")
        if not isinstance(history, FundingHistory):
            raise TypeError(
                "funding_history must be a FundingHistory object, "
                f"not {type(history).__name__}")
        first, last = self._candles[0], self._candles[-1]
        start = first.open_time
        end = last.close_time or last.open_time
        if start is None or end is None:
            raise ValueError(
                "Funding coverage cannot be validated: candles need "
                "open_time and close_time for the full candle range")
        if history.symbol != first.symbol:
            raise ValueError(
                f"Funding history symbol {history.symbol!r} does not "
                f"match candle symbol {first.symbol!r}")
        if not history.covers_period(start, end):
            raise ValueError(
                "Funding history coverage does not cover the full "
                f"adapter candle range ({start}..{end}); coverage is "
                f"{history.coverage_start_time}..{history.coverage_end_time} "
                f"(requested {history.requested_start_time}.."
                f"{history.requested_end_time})")
        status = history.completeness_status_for_period(start, end)
        if status != "complete":
            raise ValueError(
                "Funding history is not complete for the full "
                f"adapter candle range ({start}..{end}): "
                f"completeness_status_for_period={status!r}")

    @property
    def funding_history(self) -> FundingHistory | None:
        return self._funding_history

    @property
    def invalid_action_penalty(self) -> float:
        """Configured penalty subtracted from reward when an action is invalid (units: fraction of initial balance)."""
        return self._invalid_action_penalty

    def funding_metadata(self) -> dict[str, Any]:
        history = self._funding_history
        config = self._config or TradingConfig()
        if history is None:
            return {"funding_mode": config.funding_mode,
                    "historical_accounting_implemented": False,
                    "funding_history": None}
        first, last = self._candles[0], self._candles[-1]
        start = first.open_time
        end = last.close_time or last.open_time
        return {
            "funding_mode": config.funding_mode,
            "historical_accounting_implemented": True,
            "completeness_status_for_full_range":
                history.completeness_status_for_period(start, end),  # type: ignore[arg-type]
            "funding_history": history.metadata(),
        }

    def episode_funding_statistics(self) -> dict[str, Any]:
        """Funding totals labelled as independent-episode aggregates."""
        current = self.env
        return {
            "basis": "independent_episodes_not_continuous_account",
            "completed_episode_count": self._completed_episode_count,
            "aggregate_total_funding_across_completed_episodes":
                self._completed_total_funding,
            "aggregate_ledger_entries_across_completed_episodes":
                self._completed_ledger_entries,
            "current_episode_total_funding": current.total_funding,
            "current_episode_ledger_entries": len(current.funding_ledger),
        }

    def _make_env(self, start: int) -> PerpetualFuturesEnv:
        end = len(self._candles)
        if self._episode_window:
            end = min(end, start + self._episode_window)
        self._slice = (start, end)
        return PerpetualFuturesEnv(
            self._candles[start:end], self._config,
            news_archive=self._news_archive, news_asset=self._news_asset,
            funding_history=self._funding_history)

    def _history(self, raw_obs: dict[str, float]) -> list[Kline]:
        index = int(raw_obs.get("index", 0))
        return self.env.candles[:index + 1]

    def _news_observation(self) -> NewsObservation:
        """Eligible news at the current decision timestamp.

        Fresh: at least one point-in-time-available, classified,
        non-uncertain article within the staleness limit. None:
        collection succeeded, nothing eligible. Unavailable: the
        archive query failed (collector/API error state).
        """
        if self._news_archive is None or not self._news_asset:
            return NewsObservation(status="none")
        candle = self.env.current_candle
        decision = candle.close_time or candle.open_time
        if decision is None:
            return NewsObservation(status="none")
        try:
            articles = self._news_archive.available_at(
                decision, asset=self._news_asset)
        except Exception:
            return NewsObservation(status="unavailable")
        compounds, confidences, latest_age = [], [], None
        for article in articles:
            result = self._news_archive.classification_for(
                article.article_id)
            if not result or result.get("uncertain"):
                continue  # irrelevant/uncertain excluded
            if article.published_at is None:
                continue
            age = (decision - article.published_at).total_seconds()
            if age > self.features.news_max_age_seconds:
                continue  # stale news is never reused indefinitely
            compounds.append(float(result["compound"]))
            confidences.append(float(result["confidence_proxy"]))
            latest_age = age if latest_age is None else min(latest_age, age)
        if not compounds:
            return NewsObservation(status="none")
        return NewsObservation(
            status="fresh",
            mean_compound=sum(compounds) / len(compounds),
            mean_confidence=sum(confidences) / len(confidences),
            count=len(compounds), age_seconds=latest_age)

    def _build(self, raw_obs: dict[str, float]) -> np.ndarray:
        obs = dict(raw_obs)
        obs["progress"] = 0.0
        return self.features.build(
            obs, self._history(raw_obs), news=self._news_observation())

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        max_start = 0
        if self._episode_window and len(self._candles) > self._episode_window:
            max_start = len(self._candles) - self._episode_window
        if options and "start_index" in options:
            start = int(options["start_index"])
        elif max_start > 0:
            start = int(self.np_random.integers(0, max_start + 1))
        else:
            start = 0
        if self._episode_started:
            self._completed_episode_count += 1
            self._completed_total_funding += self.env.total_funding
            self._completed_ledger_entries += len(self.env.funding_ledger)
        self.env = self._make_env(start)
        self._episode_started = True
        raw = self.env.reset()
        info = {"start_index": start, "slice": self._slice,
                "raw_observation": raw}
        return self._build(raw), info

    def step(self, action: int):
        result = self.env.step(Action(int(action)))
        observation = self._build(result.observation)
        info = dict(result.info)
        info["raw_observation"] = result.observation
        reward = float(result.reward)
        if bool(info.get("invalid_action", False)):
            reward_scale = (
                float(self._config.reward_scale)
                if self._config and self._config.reward_scale is not None
                else 1.0
            )
            penalty = self._invalid_action_penalty * reward_scale
            reward -= penalty
            info["invalid_action_penalty"] = penalty
        return (observation, reward, bool(result.terminated),
                bool(result.truncated), info)
