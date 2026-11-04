"""Model inference and deployment data contracts (Component 1, Stage 3).

Defines the core data contracts, Action enum, PredictionResult,
AccountSnapshot, exception hierarchy, and ModelInference engine
for standalone model loading and inference outside the Gymnasium
training environment.
"""
from __future__ import annotations

import json
import math
import warnings
import zipfile
from dataclasses import dataclass
from enum import EnumType, IntEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

def _require_torch() -> Any:
    """Lazily import PyTorch on demand, raising ModelInferenceError if unavailable."""
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ModelInferenceError(
            'PyTorch is required for model inference. Install the RL extra: pip install -e ".[rl]"'
        ) from exc


def _require_torch_and_nn() -> tuple[Any, Any]:
    """Lazily import PyTorch and torch.nn on demand, raising ModelInferenceError if unavailable."""
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as exc:
        raise ModelInferenceError(
            'PyTorch is required for model inference. Install the RL extra: pip install -e ".[rl]"'
        ) from exc

MAX_BATCH_SIZE: int = 10_000

from noone.rl_engine.features import (
    CANONICAL_FEATURE_DEFINITIONS,
    CANONICAL_FEATURE_SCHEMA_HASH,
    FEATURE_CALCULATION_PARAMETERS,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    OBSERVATION_DIM,
    PINNED_CANONICAL_FEATURE_SCHEMA_HASH,
    PINNED_IMPLEMENTATION_DIGEST,
    FeatureBuilder,
    NewsObservation,
    compute_feature_schema_hash,
    verify_feature_implementation_integrity,
)


# --- 1. Action Enum ---

class _ActionMeta(EnumType):
    """Metaclass for Action enum ensuring strict type validation.

    Specifically prevents Python's bool type (which is an int subclass)
    from being implicitly converted into Action.HOLD (False) or
    Action.OPEN_LONG (True).
    """

    def __call__(cls, value: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError(f"Boolean values are not valid Actions: {value!r}")
        return super().__call__(value, *args, **kwargs)


class Action(IntEnum, metaclass=_ActionMeta):
    """Five-action trading space for PROJECT NOONE."""

    HOLD = 0
    OPEN_LONG = 1
    CLOSE_LONG = 2
    OPEN_SHORT = 3
    CLOSE_SHORT = 4


# --- 2. Exception Hierarchy ---

class ModelInferenceError(RuntimeError):
    """Base exception for model inference failures."""


class ModelNotFoundError(FileNotFoundError):
    """Raised when a model file or checkpoint cannot be found."""


class CorruptModelError(ModelInferenceError):
    """Raised when a model artifact file is corrupt or unreadable."""


class IncompatibleModelError(ValueError):
    """Raised when model metadata, observation dimension, or policy architecture is incompatible."""


class InvalidObservationError(ValueError):
    """Raised when an observation vector has invalid shape, dtype, or non-finite values."""


class InsufficientHistoryError(ValueError):
    """Raised when closed candle history is insufficient for indicator calculation."""


class InvalidAccountStateError(ValueError, KeyError):
    """Raised when an account state or snapshot is invalid or inconsistent."""

    def __str__(self) -> str:
        # Prevent KeyError's repr-wrapping behavior for clean error messages
        return (
            self.args[0]
            if len(self.args) == 1 and isinstance(self.args[0], str)
            else super().__str__()
        )


# --- 3. Prediction Result ---

@dataclass(frozen=True)
class PredictionResult:
    """Result of a model inference forward pass.

    Represents the selected executable action, estimated action Q-values,
    Q-value margin, original unmasked model prediction, and action validity.
    Q-values represent expected cumulative returns under the policy;
    they are never represented as confidence probabilities.
    """

    action: Action
    q_values: Mapping[str, float]
    q_margin: float
    raw_action: Action | None = None
    is_valid: bool = True

    def __post_init__(self) -> None:
        # Validate and normalize action
        if isinstance(self.action, bool):
            raise ValueError(f"action must be an Action enum or integer, got bool {self.action!r}")
        try:
            act = Action(self.action)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid action {self.action!r}: must map to Action enum (0..4)") from exc
        object.__setattr__(self, "action", act)

        # Validate and normalize raw_action
        if self.raw_action is None:
            object.__setattr__(self, "raw_action", act)
        else:
            if isinstance(self.raw_action, bool):
                raise ValueError(f"raw_action must be an Action enum or integer, got bool {self.raw_action!r}")
            try:
                raw_act = Action(self.raw_action)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid raw_action {self.raw_action!r}: must map to Action enum (0..4)") from exc
            object.__setattr__(self, "raw_action", raw_act)

        # Validate is_valid
        if not isinstance(self.is_valid, bool):
            raise ValueError(f"is_valid must be bool, got {type(self.is_valid).__name__}")

        # Validate and normalize q_values
        if not isinstance(self.q_values, (dict, Mapping)):
            raise ValueError(f"q_values must be a mapping, got {type(self.q_values).__name__}")

        normalized_q: dict[str, float] = {}
        for k, v in self.q_values.items():
            if isinstance(k, Action):
                name = k.name
            elif isinstance(k, int) and not isinstance(k, bool):
                try:
                    name = Action(k).name
                except ValueError as exc:
                    raise ValueError(f"Invalid action key {k!r} in q_values") from exc
            elif isinstance(k, str):
                try:
                    name = Action[k].name
                except KeyError as exc:
                    raise ValueError(f"Invalid action key {k!r} in q_values") from exc
            else:
                raise ValueError(f"Invalid action key type {type(k).__name__} in q_values: {k!r}")

            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"Q-value for {name} must be numeric, got {v!r}")
            v_float = float(v)
            if not math.isfinite(v_float):
                raise ValueError(f"Q-value for {name} must be finite, got {v_float}")
            normalized_q[name] = v_float

        required_actions = {a.name for a in Action}
        missing = required_actions - set(normalized_q.keys())
        if missing:
            raise ValueError(f"q_values missing actions: {sorted(missing)}")

        # Wrap in immutable MappingProxyType to prevent post-init caller mutation
        object.__setattr__(self, "q_values", MappingProxyType(normalized_q))

        # Validate q_margin
        if isinstance(self.q_margin, bool) or not isinstance(self.q_margin, (int, float)):
            raise ValueError(f"q_margin must be numeric, got {self.q_margin!r}")
        margin_float = float(self.q_margin)
        if not math.isfinite(margin_float):
            raise ValueError(f"q_margin must be finite, got {margin_float}")
        if margin_float < 0.0:
            raise ValueError(f"q_margin must be non-negative, got {margin_float}")

        # Validate consistency with top two Q-values
        sorted_q = sorted(normalized_q.values(), reverse=True)
        highest_q = sorted_q[0]
        second_highest_q = sorted_q[1]
        expected_margin = highest_q - second_highest_q
        if not math.isclose(margin_float, expected_margin, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(
                f"Inconsistent q_margin {margin_float}: expected {expected_margin:.6g} "
                f"(difference between highest {highest_q:.6g} and second-highest {second_highest_q:.6g})"
            )

        object.__setattr__(self, "q_margin", margin_float)


# --- 4. Account Snapshot ---

@dataclass(frozen=True)
class AccountSnapshot:
    """Verified point-in-time account and position state for inference.

    Maps to the exact 9 account/position keys required by FeatureBuilder:
    equity, entry_price, position_side, position_quantity, unrealized_pnl,
    realized_pnl, available_balance, margin_used, and progress.
    """

    equity: float
    available_balance: float
    position_side: int = 0
    position_quantity: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    margin_used: float = 0.0
    progress: float = 0.0

    def __post_init__(self) -> None:
        # 1. Equity
        if isinstance(self.equity, bool) or not isinstance(self.equity, (int, float)):
            raise InvalidAccountStateError(f"equity must be numeric, got {self.equity!r}")
        eq = float(self.equity)
        if not math.isfinite(eq):
            raise InvalidAccountStateError("equity must be finite")
        if eq <= 0.0:
            raise InvalidAccountStateError(f"equity must be greater than zero, got {eq}")
        object.__setattr__(self, "equity", eq)

        # 2. Available balance
        if isinstance(self.available_balance, bool) or not isinstance(self.available_balance, (int, float)):
            raise InvalidAccountStateError(f"available_balance must be numeric, got {self.available_balance!r}")
        avail = float(self.available_balance)
        if not math.isfinite(avail):
            raise InvalidAccountStateError("available_balance must be finite")
        if avail < 0.0:
            raise InvalidAccountStateError(f"available_balance must be non-negative, got {avail}")
        object.__setattr__(self, "available_balance", avail)

        # 3. Position side
        if isinstance(self.position_side, bool) or not isinstance(self.position_side, (int, float)):
            raise InvalidAccountStateError(f"position_side must be -1, 0, or 1, got {self.position_side!r}")
        if self.position_side not in (-1, 0, 1):
            raise InvalidAccountStateError(f"position_side must be -1, 0, or 1, got {self.position_side}")
        side = int(self.position_side)
        object.__setattr__(self, "position_side", side)

        # 4. Numeric fields validation
        for field_name, val in [
            ("position_quantity", self.position_quantity),
            ("entry_price", self.entry_price),
            ("unrealized_pnl", self.unrealized_pnl),
            ("realized_pnl", self.realized_pnl),
            ("margin_used", self.margin_used),
        ]:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise InvalidAccountStateError(f"{field_name} must be numeric, got {val!r}")
            f_val = float(val)
            if not math.isfinite(f_val):
                raise InvalidAccountStateError(f"{field_name} must be finite, got {f_val}")
            object.__setattr__(self, field_name, f_val)

        # Sign bounds
        if self.position_quantity < 0.0:
            raise InvalidAccountStateError(f"position_quantity cannot be negative, got {self.position_quantity}")
        if self.entry_price < 0.0:
            raise InvalidAccountStateError(f"entry_price cannot be negative, got {self.entry_price}")
        if self.margin_used < 0.0:
            raise InvalidAccountStateError(f"margin_used cannot be negative, got {self.margin_used}")

        # 5. Progress
        if isinstance(self.progress, bool) or not isinstance(self.progress, (int, float)):
            raise InvalidAccountStateError(f"progress must be numeric, got {self.progress!r}")
        prog = float(self.progress)
        if not math.isfinite(prog):
            raise InvalidAccountStateError("progress must be finite")
        if not (0.0 <= prog <= 1.0):
            raise InvalidAccountStateError(f"progress must be between 0 and 1, got {prog}")
        object.__setattr__(self, "progress", prog)

        # 6. Position consistency checks
        if side == 0:
            if self.position_quantity != 0.0:
                raise InvalidAccountStateError(
                    f"Flat position (position_side=0) cannot have active position_quantity, got {self.position_quantity}"
                )
            if self.entry_price != 0.0:
                raise InvalidAccountStateError(
                    f"Flat position (position_side=0) cannot have active entry_price, got {self.entry_price}"
                )
            if self.margin_used != 0.0:
                raise InvalidAccountStateError(
                    f"Flat position (position_side=0) cannot have active margin_used, got {self.margin_used}"
                )
            if self.unrealized_pnl != 0.0:
                raise InvalidAccountStateError(
                    f"Flat position (position_side=0) cannot have unrealized_pnl, got {self.unrealized_pnl}"
                )
        else:
            if self.position_quantity <= 0.0:
                raise InvalidAccountStateError(
                    f"Active position (position_side={side}) requires positive position_quantity, got {self.position_quantity}"
                )
            if self.entry_price <= 0.0:
                raise InvalidAccountStateError(
                    f"Active position (position_side={side}) requires positive entry_price, got {self.entry_price}"
                )

    def to_obs_dict(self) -> dict[str, float]:
        """Convert snapshot into the exact dictionary required by FeatureBuilder."""
        return {
            "equity": self.equity,
            "entry_price": self.entry_price,
            "position_side": float(self.position_side),
            "position_quantity": self.position_quantity,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "available_balance": self.available_balance,
            "margin_used": self.margin_used,
            "progress": self.progress,
        }


# --- 4B. Position State & Action Validity Handling ---

def _extract_position_side(position_state: Any) -> int:
    """Extract canonical position side integer (-1, 0, 1) from various input formats."""
    if position_state is None:
        return 0
    if isinstance(position_state, AccountSnapshot):
        return int(position_state.position_side)
    if isinstance(position_state, (dict, Mapping)):
        if "position_side" in position_state:
            return _extract_position_side(position_state["position_side"])
        return 0
    if hasattr(position_state, "side"):
        return _extract_position_side(getattr(position_state, "side"))
    if isinstance(position_state, str):
        s = position_state.strip().upper()
        if s in ("FLAT", "NONE", "0"):
            return 0
        if s in ("LONG", "1", "+1"):
            return 1
        if s in ("SHORT", "-1"):
            return -1
        raise ValueError(f"Unknown position state string: {position_state!r}")
    if isinstance(position_state, (int, np.integer)) and not isinstance(position_state, bool):
        val = int(position_state)
        if val in (-1, 0, 1):
            return val
        raise ValueError(f"Invalid position side integer: {val}. Must be -1, 0, or 1")
    if isinstance(position_state, float):
        if not math.isfinite(position_state):
            raise ValueError("position_side must be finite")
        if position_state in (-1.0, 0.0, 1.0):
            return int(position_state)
        raise ValueError(f"Invalid position side float: {position_state}. Must be -1.0, 0.0, or 1.0")
    raise TypeError(
        f"Cannot extract position side from type {type(position_state).__name__}: {position_state!r}"
    )


def get_valid_actions(position_state: Any = 0) -> tuple[Action, ...]:
    """Return tuple of valid executable actions for a given position state.

    Rules:
    - Flat (side 0): Action.HOLD (0), Action.OPEN_LONG (1), Action.OPEN_SHORT (3)
    - Long (side 1): Action.HOLD (0), Action.CLOSE_LONG (2)
    - Short (side -1): Action.HOLD (0), Action.CLOSE_SHORT (4)
    """
    side = _extract_position_side(position_state)
    if side == 0:
        return (Action.HOLD, Action.OPEN_LONG, Action.OPEN_SHORT)
    elif side == 1:
        return (Action.HOLD, Action.CLOSE_LONG)
    elif side == -1:
        return (Action.HOLD, Action.CLOSE_SHORT)
    raise ValueError(f"Invalid position side: {side!r}")


def get_valid_action_mask(position_state: Any = 0) -> tuple[bool, bool, bool, bool, bool]:
    """Return 5-element boolean mask (True=valid, False=invalid) for actions 0..4."""
    valid_actions = set(get_valid_actions(position_state))
    return (
        Action.HOLD in valid_actions,
        Action.OPEN_LONG in valid_actions,
        Action.CLOSE_LONG in valid_actions,
        Action.OPEN_SHORT in valid_actions,
        Action.CLOSE_SHORT in valid_actions,
    )


# --- 5. Observation Input Validation & Preparation Helpers ---

class _DictCandleWrapper:
    """Internal lightweight adapter for candle dictionaries."""

    __slots__ = ("open", "high", "low", "close", "volume", "open_time", "close_time", "is_closed")

    def __init__(self, d: dict[str, Any]) -> None:
        self.open = float(d["open"])
        self.high = float(d["high"])
        self.low = float(d["low"])
        self.close = float(d["close"])
        self.volume = float(d.get("volume", 0.0) or 0.0)
        self.open_time = d.get("open_time")
        self.close_time = d.get("close_time")
        self.is_closed = bool(d.get("is_closed", True))


def _validate_and_prepare_candles(candles: Any) -> list[Any]:
    """Validate raw candle history and reject non-compliant data.

    Enforces:
    - At least 50 closed candles.
    - Strict chronological order (no duplicates, no out-of-order).
    - Rejects open/forming candles (is_closed must be True).
    - Requires valid open_time.
    - Validates close_time > open_time if present and comparable.
    - Numeric, finite, positive OHLC values.
    - Consistent high/low bounds (high >= max(open, close, low), low <= min(open, close, high)).
    - Numeric, finite, non-negative volume.
    """
    if candles is None or isinstance(candles, (str, bytes, dict, bool)):
        raise InvalidObservationError(
            f"candles must be a sequence of candle objects, got {type(candles).__name__}"
        )
    try:
        candle_list = list(candles)
    except TypeError as exc:
        raise InvalidObservationError(f"candles is not iterable: {exc}") from exc

    if len(candle_list) < 50:
        raise InsufficientHistoryError(
            f"Insufficient candle history: requires at least 50 closed candles, got {len(candle_list)}"
        )

    prepared: list[Any] = []
    prev_open_time: Any = None

    for i, c in enumerate(candle_list):
        if c is None or isinstance(c, (bool, int, float, str, bytes)):
            raise InvalidObservationError(f"Candle at index {i} is invalid: got {type(c).__name__}")

        if isinstance(c, dict):
            open_val = c.get("open")
            high_val = c.get("high")
            low_val = c.get("low")
            close_val = c.get("close")
            vol_val = c.get("volume")
            open_time = c.get("open_time")
            close_time = c.get("close_time")
            is_closed = c.get("is_closed", True)
        else:
            open_val = getattr(c, "open", None)
            high_val = getattr(c, "high", None)
            low_val = getattr(c, "low", None)
            close_val = getattr(c, "close", None)
            vol_val = getattr(c, "volume", None)
            open_time = getattr(c, "open_time", None)
            close_time = getattr(c, "close_time", None)
            is_closed = getattr(c, "is_closed", True)

        # 1. Reject unclosed/forming candles and falsy closure representations
        if is_closed is False or is_closed is None or is_closed == 0:
            raise InvalidObservationError(
                f"Candle at index {i} is not closed (open/forming candles are rejected)"
            )

        # 2. Open time validation
        if open_time is None:
            raise InvalidObservationError(f"Candle at index {i} missing open_time")

        if prev_open_time is not None:
            try:
                if open_time < prev_open_time:
                    raise InvalidObservationError(
                        f"Candles not in chronological order at index {i}: {open_time} < {prev_open_time}"
                    )
                if open_time == prev_open_time:
                    raise InvalidObservationError(
                        f"Duplicate candle timestamp detected at index {i}: {open_time}"
                    )
            except TypeError as exc:
                raise InvalidObservationError(
                    f"Incomparable candle timestamps at index {i}: {exc}"
                ) from exc

        prev_open_time = open_time

        # 3. Close time consistency
        if close_time is not None:
            try:
                if close_time <= open_time:
                    raise InvalidObservationError(
                        f"Invalid candle interval at index {i}: close_time ({close_time}) <= open_time ({open_time})"
                    )
            except TypeError:
                pass

        # 4. OHLC validation
        for field_name, val in (
            ("open", open_val),
            ("high", high_val),
            ("low", low_val),
            ("close", close_val),
        ):
            if val is None:
                raise InvalidObservationError(f"Candle at index {i} missing {field_name}")
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise InvalidObservationError(
                    f"Candle at index {i} has non-numeric {field_name}: {val!r}"
                )
            f_val = float(val)
            if not math.isfinite(f_val):
                raise InvalidObservationError(
                    f"Candle at index {i} has non-finite {field_name}: {f_val}"
                )
            if f_val <= 0.0:
                raise InvalidObservationError(
                    f"Candle at index {i} has non-positive {field_name}: {f_val}"
                )

        o_val = float(open_val)
        h_val = float(high_val)
        l_val = float(low_val)
        c_val = float(close_val)

        if h_val < max(o_val, c_val, l_val):
            raise InvalidObservationError(
                f"Candle at index {i} has high ({h_val}) below open/close/low ({max(o_val, c_val, l_val)})"
            )
        if l_val > min(o_val, c_val, h_val):
            raise InvalidObservationError(
                f"Candle at index {i} has low ({l_val}) above open/close/high ({min(o_val, c_val, h_val)})"
            )

        # 5. Volume validation
        if vol_val is not None:
            if isinstance(vol_val, bool) or not isinstance(vol_val, (int, float)):
                raise InvalidObservationError(
                    f"Candle at index {i} has non-numeric volume: {vol_val!r}"
                )
            f_vol = float(vol_val)
            if not math.isfinite(f_vol):
                raise InvalidObservationError(
                    f"Candle at index {i} has non-finite volume: {f_vol}"
                )
            if f_vol < 0.0:
                raise InvalidObservationError(
                    f"Candle at index {i} has negative volume: {f_vol}"
                )

        if isinstance(c, dict):
            prepared.append(_DictCandleWrapper(c))
        else:
            prepared.append(c)

    return prepared


def _validate_funding_and_oi(
    funding_rate: Any,
    open_interest_change: Any,
) -> tuple[float | None, float | None]:
    """Validate funding rate and open interest change, preserving None for missing data."""
    fr_out: float | None = None
    if funding_rate is not None:
        if isinstance(funding_rate, bool) or not isinstance(funding_rate, (int, float)):
            raise InvalidObservationError(
                f"funding_rate must be numeric or None, got {type(funding_rate).__name__} ({funding_rate!r})"
            )
        fr_float = float(funding_rate)
        if not math.isfinite(fr_float):
            raise InvalidObservationError(f"funding_rate must be finite, got {fr_float}")
        fr_out = fr_float

    oi_out: float | None = None
    if open_interest_change is not None:
        if isinstance(open_interest_change, bool) or not isinstance(open_interest_change, (int, float)):
            raise InvalidObservationError(
                f"open_interest_change must be numeric or None, got {type(open_interest_change).__name__} ({open_interest_change!r})"
            )
        oi_float = float(open_interest_change)
        if not math.isfinite(oi_float):
            raise InvalidObservationError(f"open_interest_change must be finite, got {oi_float}")
        oi_out = oi_float

    return fr_out, oi_out


def _validate_news(news: Any) -> NewsObservation:
    """Validate NewsObservation input contracts and preserve fresh/none/unavailable states."""
    if news is None:
        return NewsObservation(status="none")
    if not isinstance(news, NewsObservation):
        raise InvalidObservationError(
            f"news must be an instance of NewsObservation or None, got {type(news).__name__}"
        )
    if news.status not in ("fresh", "none", "unavailable"):
        raise InvalidObservationError(
            f"Invalid news status: {news.status!r}; must be 'fresh', 'none', or 'unavailable'"
        )

    for field_name, val in (
        ("mean_compound", news.mean_compound),
        ("mean_confidence", news.mean_confidence),
    ):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise InvalidObservationError(f"news.{field_name} must be numeric, got {val!r}")
        f_val = float(val)
        if not math.isfinite(f_val):
            raise InvalidObservationError(f"news.{field_name} must be finite, got {f_val}")

    if float(news.mean_confidence) < 0.0:
        raise InvalidObservationError(
            f"news.mean_confidence must be non-negative, got {news.mean_confidence}"
        )

    if isinstance(news.count, bool) or not isinstance(news.count, int):
        raise InvalidObservationError(f"news.count must be an integer, got {news.count!r}")
    if news.count < 0:
        raise InvalidObservationError(f"news.count must be non-negative, got {news.count}")

    if news.age_seconds is not None:
        if isinstance(news.age_seconds, bool) or not isinstance(news.age_seconds, (int, float)):
            raise InvalidObservationError(
                f"news.age_seconds must be numeric or None, got {news.age_seconds!r}"
            )
        f_age = float(news.age_seconds)
        if not math.isfinite(f_age):
            raise InvalidObservationError(f"news.age_seconds must be finite, got {f_age}")
        if f_age < 0.0:
            raise InvalidObservationError(f"news.age_seconds must be non-negative, got {f_age}")

    return news


def _validate_account(account: Any) -> AccountSnapshot:
    """Validate AccountSnapshot input contract without duplicating snapshot logic."""
    if not isinstance(account, AccountSnapshot):
        raise InvalidAccountStateError(
            f"account must be an instance of AccountSnapshot, got {type(account).__name__}"
        )
    return account


# --- 6. Model Inference Engine (Loading, Observation & Prediction) ---

class ModelInference:
    """Production model inference interface for PROJECT NOONE (Component 1, Stage 3).

    Handles model artifact validation, metadata compatibility verification,
    FeatureBuilder integration, observation construction, and deterministic
    CPU inference prediction with Q-value extraction.
    """

    def __init__(
        self,
        model: Any,
        metadata: dict[str, Any],
        *,
        model_path: Path,
        metadata_path: Path,
        is_legacy_metadata: bool = False,
        news_max_age_hours: float | None = None,
    ) -> None:
        self._model = model
        self._metadata = metadata
        self._model_path = model_path
        self._metadata_path = metadata_path
        self._is_legacy_metadata = is_legacy_metadata
        self._news_max_age_hours = news_max_age_hours
        if news_max_age_hours is not None:
            self._feature_builder: FeatureBuilder | None = FeatureBuilder(
                news_max_age_hours=news_max_age_hours
            )
        else:
            self._feature_builder = None

    @property
    def model(self) -> Any:
        """The loaded Stable-Baselines3 DQN model."""
        return self._model

    @property
    def metadata(self) -> dict[str, Any]:
        """The validated metadata dictionary."""
        return self._metadata

    @property
    def model_path(self) -> Path:
        """Path to the loaded model ZIP archive."""
        return self._model_path

    @property
    def metadata_path(self) -> Path:
        """Path to the validated metadata JSON file."""
        return self._metadata_path

    @property
    def is_legacy_metadata(self) -> bool:
        """True if model was loaded with unverified legacy metadata."""
        return self._is_legacy_metadata

    @property
    def news_max_age_hours(self) -> float | None:
        """The validated training-time news max age in hours, if present in metadata."""
        return self._news_max_age_hours

    @property
    def feature_builder(self) -> FeatureBuilder | None:
        """The initialized FeatureBuilder instance, or None if configuration was missing."""
        return self._feature_builder

    @property
    def device(self) -> str:
        """Device on which the model executes (always 'cpu')."""
        return "cpu"

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        metadata_path: str | Path | None = None,
        allow_legacy_metadata: bool = False,
        expected_net_arch: tuple[int, ...] | list[int] | None = (128, 128),
    ) -> ModelInference:
        """Load and validate a trained DQN model and its metadata for inference.

        Args:
            model_path: Path to the model .zip archive.
            metadata_path: Optional path to metadata.json. If omitted, resolved
                beside the model or in its parent run directory.
            allow_legacy_metadata: If False (default), metadata must contain
                both canonical feature_schema_hash and feature_schema_version.
                If True, permits legacy Stage 2 metadata with basic dimension
                checks and emits a warning.
            expected_net_arch: Expected hidden layer architecture (default: [128, 128]).
                If None, skips strict default architecture check while still enforcing
                metadata consistency if net_arch is declared in metadata.

        Returns:
            Validated ModelInference instance.

        Raises:
            ModelNotFoundError: If model file or metadata.json cannot be found.
            CorruptModelError: If model archive or metadata is unreadable/corrupt.
            IncompatibleModelError: If model spaces, architecture, or metadata
                fail compatibility checks.
            ModelInferenceError: Base exception for other inference failures.
        """
        model_p = Path(model_path)
        if (
            model_p.suffix != ".zip"
            and not model_p.exists()
            and model_p.with_suffix(".zip").exists()
        ):
            model_p = model_p.with_suffix(".zip")

        if not model_p.exists():
            raise ModelNotFoundError(f"Model file not found: {model_path}")

        # 1. Resolve metadata path
        if metadata_path is not None:
            meta_p = Path(metadata_path)
            if not meta_p.exists():
                raise ModelNotFoundError(f"Metadata file not found: {meta_p}")
        else:
            cand1 = model_p.parent / "metadata.json"
            cand2 = model_p.parent.parent / "metadata.json"
            if cand1.exists():
                meta_p = cand1
            elif cand2.exists():
                meta_p = cand2
            else:
                raise ModelNotFoundError(
                    f"Metadata file not found for model '{model_p}'. Looked in {cand1} and {cand2}."
                )

        # 2. Read and parse metadata JSON
        try:
            raw_meta = meta_p.read_text(encoding="utf-8")
            metadata = json.loads(raw_meta)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CorruptModelError(f"Metadata file is malformed JSON: {meta_p}") from exc
        except OSError as exc:
            raise CorruptModelError(f"Failed to read metadata file: {meta_p}") from exc

        if not isinstance(metadata, dict):
            raise CorruptModelError(f"Metadata must be a JSON object, got {type(metadata).__name__}")

        # 3. Check required metadata fields and enforce strict integer types (reject float, bool, str)
        for field_name, expected_val in (
            ("observation_dim", OBSERVATION_DIM),
            ("action_count", 5),
        ):
            if field_name not in metadata:
                raise IncompatibleModelError(f"Metadata at {meta_p} missing required field '{field_name}'")
            val = metadata[field_name]
            if isinstance(val, bool) or not isinstance(val, int):
                raise IncompatibleModelError(
                    f"Field '{field_name}' in metadata must be a strict integer, got {type(val).__name__} ({val!r})"
                )
            if val != expected_val:
                raise IncompatibleModelError(
                    f"Incompatible {field_name} in metadata: {val} != expected {expected_val}"
                )

        # 4. Feature schema verification and legacy opt-in
        schema_hash = metadata.get("feature_schema_hash")
        schema_version = metadata.get("feature_schema_version")

        # Resolve runtime news configuration if present
        meta_news = metadata.get("news")
        meta_data_cfg = (metadata.get("config") or {}).get("data") or {}

        runtime_news_age: float | None = None
        raw_news_age = None
        if isinstance(meta_news, dict) and "news_max_age_hours" in meta_news:
            raw_news_age = meta_news["news_max_age_hours"]
        elif isinstance(meta_data_cfg, dict) and "news_max_age_hours" in meta_data_cfg:
            raw_news_age = meta_data_cfg["news_max_age_hours"]

        if raw_news_age is not None:
            if isinstance(raw_news_age, bool) or not isinstance(raw_news_age, (int, float)):
                raise IncompatibleModelError(
                    f"Invalid 'news_max_age_hours' in metadata: expected numeric value, got {type(raw_news_age).__name__} ({raw_news_age!r})"
                )
            age_float = float(raw_news_age)
            if not math.isfinite(age_float) or age_float <= 0.0:
                raise IncompatibleModelError(
                    f"Invalid 'news_max_age_hours' in metadata: must be positive and finite, got {age_float}"
                )
            runtime_news_age = age_float

        is_legacy = False
        if schema_hash is None or schema_version is None:
            if not allow_legacy_metadata:
                missing = []
                if schema_hash is None:
                    missing.append("feature_schema_hash")
                if schema_version is None:
                    missing.append("feature_schema_version")
                raise IncompatibleModelError(
                    f"Model metadata at {meta_p} lacks required strict schema fields: {missing} (legacy metadata). "
                    "To load legacy models without verified feature schema, explicitly set allow_legacy_metadata=True."
                )
            warnings.warn(
                f"Legacy model metadata detected at {meta_p}: feature schema verification fields missing. "
                f"Model cannot be cryptographically verified against the canonical {OBSERVATION_DIM}-feature schema. "
                "Proceeding under explicit legacy opt-in.",
                UserWarning,
                stacklevel=2,
            )
            is_legacy = True
        else:
            # Strict mode: requires verified runtime news configuration
            if runtime_news_age is None:
                raise IncompatibleModelError(
                    f"Model metadata at {meta_p} lacks required runtime feature configuration 'news_max_age_hours' "
                    "(expected in metadata['news'] or metadata['config']['data'])."
                )

            expected_schema_hash = compute_feature_schema_hash(news_max_age_hours=runtime_news_age)

            if not isinstance(schema_hash, str) or schema_hash.strip() == "":
                raise IncompatibleModelError(f"Invalid feature_schema_hash in metadata: {schema_hash!r}")
            if schema_hash != expected_schema_hash:
                raise IncompatibleModelError(
                    f"Incompatible feature schema hash for news_max_age_hours={runtime_news_age}: "
                    f"expected {expected_schema_hash}, got {schema_hash}"
                )
            if not isinstance(schema_version, str) or schema_version != FEATURE_SCHEMA_VERSION:
                raise IncompatibleModelError(
                    f"Incompatible feature schema version: expected {FEATURE_SCHEMA_VERSION}, got {schema_version!r}"
                )

        # 5. Validate ZIP archive structure
        if not zipfile.is_zipfile(model_p):
            raise CorruptModelError(f"Model file is not a valid ZIP archive: {model_p}")

        try:
            with zipfile.ZipFile(model_p) as z:
                names = set(z.namelist())
                required_members = {"data", "policy.pth"}
                missing_members = required_members - names
                if missing_members:
                    raise CorruptModelError(
                        f"Model archive {model_p} is corrupt: missing internal members {sorted(missing_members)}"
                    )
        except zipfile.BadZipFile as exc:
            raise CorruptModelError(f"Model archive {model_p} is corrupt: {exc}") from exc

        # 6. Load model using Stable-Baselines3 on CPU
        torch, nn = _require_torch_and_nn()

        try:
            from stable_baselines3 import DQN
        except ImportError as exc:  # pragma: no cover
            raise ModelInferenceError(
                'Stable-Baselines3 is required for model inference. Install the RL extra: pip install -e ".[rl]"'
            ) from exc

        try:
            model = DQN.load(str(model_p), env=None, device="cpu")
        except Exception as exc:
            raise CorruptModelError(f"Failed to load DQN model from {model_p}: {exc}") from exc

        # Enforce CPU execution
        if hasattr(model, "device") and str(model.device) != "cpu":
            raise IncompatibleModelError(f"Expected CPU device, but model is on {model.device}")
        if (
            hasattr(model, "policy")
            and hasattr(model.policy, "device")
            and str(model.policy.device) != "cpu"
        ):
            raise IncompatibleModelError(f"Expected CPU device, but policy is on {model.policy.device}")

        # 7. Verify model spaces and architecture
        if not hasattr(model, "observation_space") or model.observation_space is None:
            raise IncompatibleModelError(f"Loaded model from {model_p} has no observation_space defined")
        if model.observation_space.shape != (OBSERVATION_DIM,):
            raise IncompatibleModelError(
                f"Loaded model observation space shape {model.observation_space.shape} != ({OBSERVATION_DIM},)"
            )

        if not hasattr(model, "action_space") or model.action_space is None:
            raise IncompatibleModelError(f"Loaded model from {model_p} has no action_space defined")
        if not (hasattr(model.action_space, "n") and int(model.action_space.n) == 5):
            raise IncompatibleModelError(f"Loaded model action space {model.action_space} != Discrete(5)")

        # Inspect actual PyTorch QNetwork architecture
        if (
            not hasattr(model, "policy")
            or not hasattr(model.policy, "q_net")
            or not hasattr(model.policy.q_net, "q_net")
        ):
            raise IncompatibleModelError("Loaded model policy does not have expected DQN q_net architecture")

        q_net_seq = model.policy.q_net.q_net
        linear_layers = [l for l in q_net_seq if isinstance(l, nn.Linear)]
        if len(linear_layers) < 2:
            raise IncompatibleModelError("Loaded model q_net has fewer than 2 linear layers")

        in_features = linear_layers[0].in_features
        out_features = linear_layers[-1].out_features
        actual_hidden_arch = [l.out_features for l in linear_layers[:-1]]

        if in_features != OBSERVATION_DIM:
            raise IncompatibleModelError(
                f"Loaded model input features {in_features} != expected {OBSERVATION_DIM}"
            )
        if out_features != 5:
            raise IncompatibleModelError(f"Loaded model output features {out_features} != expected 5")

        # Verify expected hidden architecture
        if expected_net_arch is not None:
            exp_arch = list(expected_net_arch)
            if actual_hidden_arch != exp_arch:
                raise IncompatibleModelError(
                    f"Loaded model hidden architecture {actual_hidden_arch} != expected {exp_arch}"
                )

        # Compare metadata declared architecture against actual model topology
        meta_dqn = (metadata.get("config") or {}).get("dqn") or {}
        meta_net_arch = meta_dqn.get("net_arch") or metadata.get("net_arch")
        if meta_net_arch is not None:
            declared_arch = list(meta_net_arch)
            if declared_arch != actual_hidden_arch:
                raise IncompatibleModelError(
                    f"Metadata declared net_arch {declared_arch} does not match actual model architecture {actual_hidden_arch}"
                )

        # Set policy to eval mode
        model.policy.set_training_mode(False)

        return cls(
            model=model,
            metadata=metadata,
            model_path=model_p,
            metadata_path=meta_p,
            is_legacy_metadata=is_legacy,
            news_max_age_hours=runtime_news_age,
        )

    def build_observation(
        self,
        candles: Any,
        account: AccountSnapshot,
        news: NewsObservation | None = None,
        funding_rate: float | None = None,
        open_interest_change: float | None = None,
    ) -> np.ndarray:
        """Construct verified 26-feature observation vector from raw inputs.

        Args:
            candles: Sequence of closed candles (at least 50).
            account: Verified AccountSnapshot.
            news: Optional NewsObservation summary (defaults to status="none").
            funding_rate: Optional aligned funding rate (None indicates unavailable).
            open_interest_change: Optional aligned open-interest change (None indicates unavailable).

        Returns:
            np.ndarray: Exactly 26 float32 values, guaranteed finite.

        Raises:
            IncompatibleModelError: If model metadata lacks required news_max_age_hours.
            InsufficientHistoryError: If candle history is fewer than 50 candles.
            InvalidObservationError: If candles, funding, OI, or news data is invalid/non-finite.
            InvalidAccountStateError: If account is not an AccountSnapshot.
        """
        if self._feature_builder is None:
            raise IncompatibleModelError(
                "Cannot build observation: model metadata lacks required 'news_max_age_hours' configuration. "
                "Use predict_observation() directly with a pre-constructed observation vector."
            )

        valid_candles = _validate_and_prepare_candles(candles)
        valid_account = _validate_account(account)
        valid_news = _validate_news(news)
        valid_fr, valid_oi = _validate_funding_and_oi(funding_rate, open_interest_change)

        obs_dict = valid_account.to_obs_dict()
        obs_dict["progress"] = 0.0

        vector = self._feature_builder.build(
            obs=obs_dict,
            history=valid_candles,
            news=valid_news,
            funding_rate=valid_fr,
            open_interest_change=valid_oi,
        )

        if not isinstance(vector, np.ndarray):
            raise InvalidObservationError(f"FeatureBuilder produced non-array: {type(vector).__name__}")
        if vector.shape != (OBSERVATION_DIM,):
            raise InvalidObservationError(f"FeatureBuilder produced shape {vector.shape} != ({OBSERVATION_DIM},)")
        if vector.dtype != np.float32:
            raise InvalidObservationError(f"FeatureBuilder produced dtype {vector.dtype} != float32")
        if not np.isfinite(vector).all():
            raise InvalidObservationError("FeatureBuilder produced non-finite values (NaN or infinity)")

        return vector

    def predict_observation(
        self,
        observation: np.ndarray,
        *,
        position_state: Any = None,
        valid_action_mask: Sequence[bool] | None = None,
    ) -> PredictionResult:
        """Predict trading action and extract Q-values from an observation vector.

        Supports deterministic action masking based on position state or an explicit mask.
        Raw Q-values and unmasked action predictions are preserved for diagnostics.

        Args:
            observation: Point-in-time observation vector of shape (26,) and dtype float32.
            position_state: Optional position state (e.g. AccountSnapshot, side int/str, or dict).
            valid_action_mask: Optional 5-element boolean mask for actions 0..4.

        Returns:
            PredictionResult: action, immutable Q-values mapping, Q-margin, raw_action, and is_valid.

        Raises:
            InvalidObservationError: If observation vector is invalid, non-finite, or wrong shape/dtype.
            ModelInferenceError: If model prediction fails or returns invalid outputs.
        """
        torch = _require_torch()

        if isinstance(observation, bool) or not isinstance(observation, np.ndarray):
            raise InvalidObservationError(
                f"observation must be a numpy ndarray, got {type(observation).__name__}"
            )
        if observation.shape != (OBSERVATION_DIM,):
            raise InvalidObservationError(
                f"observation shape {observation.shape} != ({OBSERVATION_DIM},)"
            )
        if observation.dtype != np.float32:
            raise InvalidObservationError(
                f"observation dtype must be float32, got {observation.dtype}"
            )
        if not np.isfinite(observation).all():
            raise InvalidObservationError(
                "observation contains non-finite values (NaN or infinity)"
            )

        # Resolve action validity mask if requested
        resolved_mask: tuple[bool, bool, bool, bool, bool] | None = None
        if valid_action_mask is not None:
            if isinstance(valid_action_mask, (list, tuple, np.ndarray)):
                if len(valid_action_mask) != 5:
                    raise ValueError(f"valid_action_mask must have length 5, got {len(valid_action_mask)}")
                bool_mask = tuple(bool(x) for x in valid_action_mask)
                if not any(bool_mask):
                    raise ValueError("valid_action_mask must contain at least one valid action")
                resolved_mask = (bool_mask[0], bool_mask[1], bool_mask[2], bool_mask[3], bool_mask[4])
            else:
                raise TypeError(f"valid_action_mask must be a sequence of booleans, got {type(valid_action_mask).__name__}")
        elif position_state is not None:
            resolved_mask = get_valid_action_mask(position_state)

        # Enforce CPU execution and safely handle read-only / non-contiguous arrays without warnings
        obs_tensor = torch.from_numpy(np.ascontiguousarray(observation).copy()).unsqueeze(0)

        with torch.no_grad():
            q_net_out = self._model.policy.q_net(obs_tensor)

        if not torch.isfinite(q_net_out).all():
            raise ModelInferenceError("Q-network produced non-finite Q-values")
        if q_net_out.ndim != 2 or q_net_out.shape != (1, 5):
            raise ModelInferenceError(f"Unexpected Q-network output shape: {q_net_out.shape} != (1, 5)")

        q_list = q_net_out.squeeze(0).tolist()
        q_dict = {
            Action(i).name: float(q_list[i])
            for i in range(5)
        }

        # Deterministic prediction via SB3 model
        sb3_act_raw, _ = self._model.predict(observation, deterministic=True)
        if hasattr(sb3_act_raw, "item"):
            sb3_act_int = int(sb3_act_raw.item())
        elif isinstance(sb3_act_raw, (int, np.integer)):
            sb3_act_int = int(sb3_act_raw)
        else:
            raise ModelInferenceError(f"Model returned invalid action type: {type(sb3_act_raw).__name__}")

        if sb3_act_int not in (0, 1, 2, 3, 4):
            raise ModelInferenceError(f"Model returned action out of range (0..4): {sb3_act_int}")

        expected_argmax = int(q_net_out.argmax(dim=1).item())
        if sb3_act_int != expected_argmax:
            raise ModelInferenceError(
                f"Deterministic action {sb3_act_int} disagrees with Q-network argmax {expected_argmax}"
            )

        raw_action = Action(sb3_act_int)
        if resolved_mask is not None:
            is_valid = bool(resolved_mask[raw_action.value])
            if is_valid:
                chosen_action = raw_action
            else:
                # Deterministic selection among valid actions with deterministic tie-breaking (lowest index)
                valid_indices = [i for i in range(5) if resolved_mask[i]]
                best_idx = max(valid_indices, key=lambda idx: q_list[idx])
                chosen_action = Action(best_idx)
        else:
            is_valid = True
            chosen_action = raw_action

        sorted_q = sorted(q_dict.values(), reverse=True)
        q_margin = sorted_q[0] - sorted_q[1]

        return PredictionResult(
            action=chosen_action,
            q_values=q_dict,
            q_margin=q_margin,
            raw_action=raw_action,
            is_valid=is_valid,
        )

    def predict_observations(
        self,
        observations: np.ndarray,
        *,
        position_states: Sequence[Any] | None = None,
        valid_action_masks: Sequence[Sequence[bool]] | None = None,
    ) -> tuple[PredictionResult, ...]:
        """Predict trading actions and extract Q-values for a 2D batch of observations.

        Supports batch deterministic action masking based on position states or masks.

        Args:
            observations: Point-in-time observation matrix of shape (N, 26) and dtype float32.
            position_states: Optional sequence of position states of length N.
            valid_action_masks: Optional sequence of 5-element boolean masks of length N.

        Returns:
            tuple[PredictionResult, ...]: Immutable tuple of PredictionResult instances.

        Raises:
            InvalidObservationError: If observations matrix is not a 2D array, violates
                shape (N, 26), has dtype other than float32, is empty (N == 0),
                exceeds 10,000 observations, or contains non-finite values (NaN/Inf).
            ModelInferenceError: If model prediction fails or Q-network outputs are non-finite.
        """
        # 1. Type validation (strictly reject bool, list, non-ndarray)
        if isinstance(observations, bool) or not isinstance(observations, np.ndarray):
            raise InvalidObservationError(
                f"observations must be a numpy ndarray, got {type(observations).__name__}"
            )

        # 2. Dimension validation
        if observations.ndim != 2:
            if observations.ndim == 1 and observations.shape == (OBSERVATION_DIM,):
                raise InvalidObservationError(
                    f"observations must be a 2D array of shape (N, {OBSERVATION_DIM}), "
                    f"got 1D array of shape {observations.shape}. Use predict_observation() for single 1D observations."
                )
            raise InvalidObservationError(
                f"observations must be a 2D array of shape (N, {OBSERVATION_DIM}), "
                f"got {observations.ndim}D array of shape {observations.shape}"
            )

        # 3. Shape validation (N, 26)
        n_samples, feat_count = observations.shape
        if feat_count != OBSERVATION_DIM:
            raise InvalidObservationError(
                f"observations feature count {feat_count} != expected {OBSERVATION_DIM}"
            )

        if n_samples == 0:
            raise InvalidObservationError("observations batch cannot be empty")

        if n_samples > MAX_BATCH_SIZE:
            raise InvalidObservationError(
                f"observations batch size {n_samples} exceeds maximum allowed {MAX_BATCH_SIZE}"
            )

        # 4. Dtype validation
        if observations.dtype != np.float32:
            raise InvalidObservationError(
                f"observations dtype must be float32, got {observations.dtype}"
            )

        # 5. Finiteness validation (validate everything before tensor allocation)
        if not np.isfinite(observations).all():
            raise InvalidObservationError(
                "observations contains non-finite values (NaN or infinity)"
            )

        # Resolve batch action validity masks if provided
        resolved_masks: list[tuple[bool, bool, bool, bool, bool] | None] | None = None
        if valid_action_masks is not None:
            if not isinstance(valid_action_masks, (list, tuple, np.ndarray)):
                raise TypeError(f"valid_action_masks must be a sequence, got {type(valid_action_masks).__name__}")
            if len(valid_action_masks) != n_samples:
                raise ValueError(f"valid_action_masks length {len(valid_action_masks)} != batch size {n_samples}")
            resolved_masks = []
            for idx, mask_row in enumerate(valid_action_masks):
                if not isinstance(mask_row, (list, tuple, np.ndarray)) or len(mask_row) != 5:
                    raise ValueError(f"valid_action_masks[{idx}] must have length 5")
                b_mask = tuple(bool(x) for x in mask_row)
                if not any(b_mask):
                    raise ValueError(f"valid_action_masks[{idx}] must contain at least one valid action")
                resolved_masks.append((b_mask[0], b_mask[1], b_mask[2], b_mask[3], b_mask[4]))
        elif position_states is not None:
            if not isinstance(position_states, (list, tuple, np.ndarray)):
                raise TypeError(f"position_states must be a sequence, got {type(position_states).__name__}")
            if len(position_states) != n_samples:
                raise ValueError(f"position_states length {len(position_states)} != batch size {n_samples}")
            resolved_masks = [get_valid_action_mask(ps) for ps in position_states]

        # 6. Safe contiguous CPU tensor allocation (defensive copy prevents mutation and non-writable warnings)
        torch = _require_torch()
        obs_tensor = torch.from_numpy(np.ascontiguousarray(observations).copy())

        # 7. CPU-only vectorized Q-network inference
        with torch.no_grad():
            q_net_out = self._model.policy.q_net(obs_tensor)

        if not torch.isfinite(q_net_out).all():
            raise ModelInferenceError("Q-network produced non-finite Q-values")
        if q_net_out.ndim != 2 or q_net_out.shape != (n_samples, 5):
            raise ModelInferenceError(
                f"Unexpected Q-network output shape: {q_net_out.shape} != ({n_samples}, 5)"
            )

        # 8. Deterministic prediction via SB3 model
        sb3_actions_raw, _ = self._model.predict(observations, deterministic=True)
        sb3_actions = np.asarray(sb3_actions_raw).reshape(-1)
        if len(sb3_actions) != n_samples:
            raise ModelInferenceError(
                f"Model returned {len(sb3_actions)} actions for batch of {n_samples} observations"
            )

        # Validate action integers and range before argmax check
        for i in range(n_samples):
            act_val = sb3_actions[i]
            if not isinstance(act_val, (int, np.integer)):
                raise ModelInferenceError(
                    f"Model returned non-integer action type: {type(act_val).__name__} at index {i}"
                )
            act_int = int(act_val)
            if act_int not in (0, 1, 2, 3, 4):
                raise ModelInferenceError(f"Model returned action out of range (0..4): {act_int} at index {i}")

        expected_argmax = q_net_out.argmax(dim=1).cpu().numpy()
        if not np.array_equal(sb3_actions, expected_argmax):
            raise ModelInferenceError(
                "Deterministic SB3 actions disagree with Q-network argmax in batch inference"
            )

        # 9. Extract all five Q-values, calculate Q-margins, and construct PredictionResults
        q_net_numpy = q_net_out.cpu().numpy()
        action_names = [a.name for a in Action]
        results: list[PredictionResult] = []

        for i in range(n_samples):
            act_int = int(sb3_actions[i])
            raw_action = Action(act_int)

            q_row = q_net_numpy[i]
            q_dict = {
                action_names[a]: float(q_row[a])
                for a in range(5)
            }
            sorted_q = sorted(q_dict.values(), reverse=True)
            q_margin = sorted_q[0] - sorted_q[1]

            mask_row = resolved_masks[i] if resolved_masks is not None else None
            if mask_row is not None:
                is_valid = bool(mask_row[raw_action.value])
                if is_valid:
                    chosen_action = raw_action
                else:
                    valid_indices = [a for a in range(5) if mask_row[a]]
                    best_idx = max(valid_indices, key=lambda a: float(q_row[a]))
                    chosen_action = Action(best_idx)
            else:
                is_valid = True
                chosen_action = raw_action

            results.append(
                PredictionResult(
                    action=chosen_action,
                    q_values=q_dict,
                    q_margin=q_margin,
                    raw_action=raw_action,
                    is_valid=is_valid,
                )
            )

        # 10. Return immutable tuple of PredictionResults
        return tuple(results)

    def predict(
        self,
        candles: Any,
        account: AccountSnapshot,
        news: NewsObservation | None = None,
        funding_rate: float | None = None,
        open_interest_change: float | None = None,
        *,
        apply_action_mask: bool = True,
    ) -> PredictionResult:
        """Validate raw inputs, generate observation, and return deterministic prediction result.

        Args:
            candles: Sequence of closed candles (at least 50).
            account: Verified AccountSnapshot.
            news: Optional NewsObservation summary (defaults to status="none").
            funding_rate: Optional aligned funding rate (None indicates unavailable).
            open_interest_change: Optional aligned open-interest change (None indicates unavailable).
            apply_action_mask: Whether to apply deterministic action masking using account position state (default True).

        Returns:
            PredictionResult: action, immutable Q-values mapping, Q-margin, raw_action, and is_valid.
        """
        obs = self.build_observation(
            candles=candles,
            account=account,
            news=news,
            funding_rate=funding_rate,
            open_interest_change=open_interest_change,
        )
        pos_state = account if apply_action_mask else None
        return self.predict_observation(obs, position_state=pos_state)


