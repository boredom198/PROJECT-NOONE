"""Unit tests for Component 1, Stage 3 inference (Step 1 and Step 2A).

Tests Action enum, PredictionResult, AccountSnapshot, validation,
to_obs_dict mapping, exception hierarchy, canonical feature schema,
and ModelInference loading and validation.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch
from stable_baselines3 import DQN

from noone.market_data.models import Kline
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
from noone.rl_engine.inference import (
    Action,
    AccountSnapshot,
    CorruptModelError,
    IncompatibleModelError,
    InsufficientHistoryError,
    InvalidAccountStateError,
    InvalidObservationError,
    MAX_BATCH_SIZE,
    ModelInference,
    ModelInferenceError,
    ModelNotFoundError,
    PredictionResult,
)


# --- 1. Action Enum Tests ---

def test_action_enum_values():
    assert Action.HOLD == 0
    assert Action.OPEN_LONG == 1
    assert Action.CLOSE_LONG == 2
    assert Action.OPEN_SHORT == 3
    assert Action.CLOSE_SHORT == 4
    assert len(Action) == 5
    assert [a.value for a in Action] == [0, 1, 2, 3, 4]


def test_action_enum_boolean_rejection():
    with pytest.raises(ValueError, match="Boolean values are not valid Actions: True"):
        Action(True)

    with pytest.raises(ValueError, match="Boolean values are not valid Actions: False"):
        Action(False)


def test_action_enum_integer_conversions():
    for i in range(5):
        act = Action(i)
        assert isinstance(act, Action)
        assert act.value == i

    with pytest.raises(ValueError):
        Action(5)

    with pytest.raises(ValueError):
        Action(-1)


# --- 2. AccountSnapshot Valid Construction & Mapping ---

def test_valid_flat_account_snapshot_defaults():
    snap = AccountSnapshot(equity=10_000.0, available_balance=10_000.0)
    assert snap.equity == 10_000.0
    assert snap.available_balance == 10_000.0
    assert snap.position_side == 0
    assert snap.position_quantity == 0.0
    assert snap.entry_price == 0.0
    assert snap.unrealized_pnl == 0.0
    assert snap.realized_pnl == 0.0
    assert snap.margin_used == 0.0
    assert snap.progress == 0.0

    obs = snap.to_obs_dict()
    assert obs == {
        "equity": 10_000.0,
        "entry_price": 0.0,
        "position_side": 0.0,
        "position_quantity": 0.0,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
        "available_balance": 10_000.0,
        "margin_used": 0.0,
        "progress": 0.0,
    }


def test_valid_active_long_account_snapshot():
    snap = AccountSnapshot(
        equity=10_500.0,
        available_balance=8_500.0,
        position_side=1,
        position_quantity=0.1,
        entry_price=65_000.0,
        unrealized_pnl=500.0,
        realized_pnl=200.0,
        margin_used=2_000.0,
        progress=0.45,
    )
    assert snap.position_side == 1
    assert snap.position_quantity == 0.1
    assert snap.entry_price == 65_000.0

    obs = snap.to_obs_dict()
    assert obs["position_side"] == 1.0
    assert obs["position_quantity"] == 0.1
    assert obs["entry_price"] == 65_000.0
    assert obs["unrealized_pnl"] == 500.0
    assert obs["margin_used"] == 2_000.0


def test_valid_active_short_account_snapshot():
    snap = AccountSnapshot(
        equity=9_800.0,
        available_balance=7_800.0,
        position_side=-1,
        position_quantity=0.05,
        entry_price=64_000.0,
        unrealized_pnl=-200.0,
        realized_pnl=0.0,
        margin_used=2_000.0,
        progress=1.0,
    )
    assert snap.position_side == -1
    obs = snap.to_obs_dict()
    assert obs["position_side"] == -1.0
    assert obs["position_quantity"] == 0.05
    assert obs["progress"] == 1.0


# --- 3. AccountSnapshot Validation & Rejection ---

def test_invalid_equity_rejected():
    with pytest.raises(InvalidAccountStateError, match="equity must be greater than zero"):
        AccountSnapshot(equity=0.0, available_balance=100.0)

    with pytest.raises(InvalidAccountStateError, match="equity must be greater than zero"):
        AccountSnapshot(equity=-50.0, available_balance=100.0)

    with pytest.raises(InvalidAccountStateError, match="equity must be finite"):
        AccountSnapshot(equity=float("inf"), available_balance=100.0)

    with pytest.raises(InvalidAccountStateError, match="equity must be finite"):
        AccountSnapshot(equity=float("nan"), available_balance=100.0)

    with pytest.raises(InvalidAccountStateError, match="equity must be numeric"):
        AccountSnapshot(equity=True, available_balance=100.0)  # type: ignore[arg-type]


def test_invalid_available_balance_rejected():
    with pytest.raises(InvalidAccountStateError, match="available_balance must be non-negative"):
        AccountSnapshot(equity=1000.0, available_balance=-1.0)

    with pytest.raises(InvalidAccountStateError, match="available_balance must be finite"):
        AccountSnapshot(equity=1000.0, available_balance=float("inf"))

    with pytest.raises(InvalidAccountStateError, match="available_balance must be finite"):
        AccountSnapshot(equity=1000.0, available_balance=float("nan"))


def test_invalid_position_side_rejected():
    for bad_side in [2, -2, 3, 99]:
        with pytest.raises(InvalidAccountStateError, match="position_side must be -1, 0, or 1"):
            AccountSnapshot(equity=1000.0, available_balance=1000.0, position_side=bad_side)

    with pytest.raises(InvalidAccountStateError, match="position_side must be -1, 0, or 1"):
        AccountSnapshot(equity=1000.0, available_balance=1000.0, position_side=0.5)  # type: ignore[arg-type]

    with pytest.raises(InvalidAccountStateError, match="position_side must be -1, 0, or 1"):
        AccountSnapshot(equity=1000.0, available_balance=1000.0, position_side=True)  # type: ignore[arg-type]


def test_invalid_progress_rejected():
    with pytest.raises(InvalidAccountStateError, match="progress must be between 0 and 1"):
        AccountSnapshot(equity=1000.0, available_balance=1000.0, progress=-0.01)

    with pytest.raises(InvalidAccountStateError, match="progress must be between 0 and 1"):
        AccountSnapshot(equity=1000.0, available_balance=1000.0, progress=1.01)

    with pytest.raises(InvalidAccountStateError, match="progress must be finite"):
        AccountSnapshot(equity=1000.0, available_balance=1000.0, progress=float("nan"))


def test_flat_position_inconsistent_fields_rejected():
    # Flat with active quantity
    with pytest.raises(InvalidAccountStateError, match="Flat position.*position_quantity"):
        AccountSnapshot(equity=1000.0, available_balance=1000.0, position_side=0, position_quantity=1.0)

    # Flat with active entry price
    with pytest.raises(InvalidAccountStateError, match="Flat position.*entry_price"):
        AccountSnapshot(equity=1000.0, available_balance=1000.0, position_side=0, entry_price=100.0)

    # Flat with active margin
    with pytest.raises(InvalidAccountStateError, match="Flat position.*margin_used"):
        AccountSnapshot(equity=1000.0, available_balance=1000.0, position_side=0, margin_used=50.0)

    # Flat with active unrealized PnL
    with pytest.raises(InvalidAccountStateError, match="Flat position.*unrealized_pnl"):
        AccountSnapshot(equity=1000.0, available_balance=1000.0, position_side=0, unrealized_pnl=10.0)


def test_active_position_missing_quantity_or_entry_rejected():
    # Long missing quantity
    with pytest.raises(InvalidAccountStateError, match="Active position.*positive position_quantity"):
        AccountSnapshot(equity=1000.0, available_balance=500.0, position_side=1, position_quantity=0.0, entry_price=100.0)

    # Long missing entry price
    with pytest.raises(InvalidAccountStateError, match="Active position.*positive entry_price"):
        AccountSnapshot(equity=1000.0, available_balance=500.0, position_side=1, position_quantity=1.0, entry_price=0.0)

    # Short with negative quantity
    with pytest.raises(InvalidAccountStateError, match="position_quantity cannot be negative"):
        AccountSnapshot(equity=1000.0, available_balance=500.0, position_side=-1, position_quantity=-1.0, entry_price=100.0)

    # Short with negative entry price
    with pytest.raises(InvalidAccountStateError, match="entry_price cannot be negative"):
        AccountSnapshot(equity=1000.0, available_balance=500.0, position_side=-1, position_quantity=1.0, entry_price=-100.0)


def test_non_finite_numeric_fields_rejected():
    with pytest.raises(InvalidAccountStateError, match="unrealized_pnl must be finite"):
        AccountSnapshot(equity=1000.0, available_balance=500.0, position_side=1, position_quantity=1.0, entry_price=100.0, unrealized_pnl=float("nan"))

    with pytest.raises(InvalidAccountStateError, match="realized_pnl must be finite"):
        AccountSnapshot(equity=1000.0, available_balance=1000.0, realized_pnl=float("inf"))


# --- 4. Exception Hierarchy Tests ---

def test_exception_inheritance_and_mro():
    # ModelInferenceError is a RuntimeError
    err = ModelInferenceError("inference failed")
    assert isinstance(err, RuntimeError)

    # ModelNotFoundError is a FileNotFoundError
    fnf = ModelNotFoundError("model not found")
    assert isinstance(fnf, FileNotFoundError)

    # CorruptModelError is a ModelInferenceError
    corrupt = CorruptModelError("corrupt zip")
    assert isinstance(corrupt, ModelInferenceError)
    assert isinstance(corrupt, RuntimeError)

    # IncompatibleModelError is a ValueError
    incompat = IncompatibleModelError("dim 25 != 26")
    assert isinstance(incompat, ValueError)

    # InvalidObservationError is a ValueError
    inv_obs = InvalidObservationError("contains NaN")
    assert isinstance(inv_obs, ValueError)

    # InsufficientHistoryError is a ValueError
    insuf = InsufficientHistoryError("< 50 candles")
    assert isinstance(insuf, ValueError)

    # InvalidAccountStateError is both ValueError and KeyError
    inv_acc = InvalidAccountStateError("equity must be greater than zero")
    assert isinstance(inv_acc, ValueError)
    assert isinstance(inv_acc, KeyError)
    assert isinstance(inv_acc, LookupError)
    assert str(inv_acc) == "equity must be greater than zero"  # clean str without KeyError repr quotes


# --- 5. PredictionResult Construction & Validation ---

def test_prediction_result_valid_construction():
    q_vals = {
        "HOLD": -0.05,
        "OPEN_LONG": 0.15,
        "CLOSE_LONG": -0.02,
        "OPEN_SHORT": 0.08,
        "CLOSE_SHORT": -0.01,
    }
    res = PredictionResult(action=Action.OPEN_LONG, q_values=q_vals, q_margin=0.07)
    assert res.action == Action.OPEN_LONG
    assert res.action.name == "OPEN_LONG"
    assert res.action.value == 1
    assert res.q_margin == 0.07
    assert res.q_values["OPEN_LONG"] == 0.15


def test_prediction_result_int_action_conversion():
    q_vals = {Action.HOLD: 0.1, Action.OPEN_LONG: 0.2, Action.CLOSE_LONG: 0.0, Action.OPEN_SHORT: -0.1, Action.CLOSE_SHORT: -0.2}
    res = PredictionResult(action=0, q_values=q_vals, q_margin=0.1)
    assert isinstance(res.action, Action)
    assert res.action == Action.HOLD


def test_prediction_result_missing_actions_rejected():
    incomplete_q = {"HOLD": 0.1, "OPEN_LONG": 0.2}
    with pytest.raises(ValueError, match="q_values missing actions"):
        PredictionResult(action=Action.HOLD, q_values=incomplete_q, q_margin=0.1)


def test_prediction_result_non_finite_q_values_rejected():
    bad_q = {"HOLD": 0.1, "OPEN_LONG": float("nan"), "CLOSE_LONG": 0.0, "OPEN_SHORT": -0.1, "CLOSE_SHORT": -0.2}
    with pytest.raises(ValueError, match="Q-value for OPEN_LONG must be finite"):
        PredictionResult(action=Action.HOLD, q_values=bad_q, q_margin=0.1)


def test_prediction_result_negative_q_margin_rejected():
    q_vals = {"HOLD": 0.1, "OPEN_LONG": 0.2, "CLOSE_LONG": 0.0, "OPEN_SHORT": -0.1, "CLOSE_SHORT": -0.2}
    with pytest.raises(ValueError, match="q_margin must be non-negative"):
        PredictionResult(action=Action.HOLD, q_values=q_vals, q_margin=-0.01)


def test_prediction_result_invalid_action_rejected():
    q_vals = {"HOLD": 0.1, "OPEN_LONG": 0.2, "CLOSE_LONG": 0.0, "OPEN_SHORT": -0.1, "CLOSE_SHORT": -0.2}
    with pytest.raises(ValueError, match="Invalid action"):
        PredictionResult(action=99, q_values=q_vals, q_margin=0.1)


def test_prediction_result_boolean_action_rejected():
    q_vals = {"HOLD": 0.1, "OPEN_LONG": 0.2, "CLOSE_LONG": 0.0, "OPEN_SHORT": -0.1, "CLOSE_SHORT": -0.2}
    with pytest.raises(ValueError, match="action must be an Action enum or integer, got bool True"):
        PredictionResult(action=True, q_values=q_vals, q_margin=0.1)

    with pytest.raises(ValueError, match="action must be an Action enum or integer, got bool False"):
        PredictionResult(action=False, q_values=q_vals, q_margin=0.1)


def test_prediction_result_inconsistent_q_margin_rejected():
    # Top two values are 0.2 and 0.1 -> expected margin is 0.1
    q_vals = {"HOLD": 0.1, "OPEN_LONG": 0.2, "CLOSE_LONG": 0.0, "OPEN_SHORT": -0.1, "CLOSE_SHORT": -0.2}
    with pytest.raises(ValueError, match="Inconsistent q_margin 0.5: expected 0.1"):
        PredictionResult(action=Action.OPEN_LONG, q_values=q_vals, q_margin=0.5)

    with pytest.raises(ValueError, match="Inconsistent q_margin 0.0: expected 0.1"):
        PredictionResult(action=Action.OPEN_LONG, q_values=q_vals, q_margin=0.0)


def test_prediction_result_tied_highest_q_values_zero_margin():
    # Tied highest values (both 0.25) -> expected margin is 0.0
    tied_q = {"HOLD": 0.25, "OPEN_LONG": 0.25, "CLOSE_LONG": 0.10, "OPEN_SHORT": 0.0, "CLOSE_SHORT": -0.10}
    res = PredictionResult(action=Action.HOLD, q_values=tied_q, q_margin=0.0)
    assert res.q_margin == 0.0

    # Non-zero margin on tied Q-values must be rejected
    with pytest.raises(ValueError, match="Inconsistent q_margin 0.05: expected 0"):
        PredictionResult(action=Action.HOLD, q_values=tied_q, q_margin=0.05)


def test_prediction_result_q_values_immutability():
    original_q = {
        "HOLD": -0.05,
        "OPEN_LONG": 0.15,
        "CLOSE_LONG": -0.02,
        "OPEN_SHORT": 0.08,
        "CLOSE_SHORT": -0.01,
    }
    res = PredictionResult(action=Action.OPEN_LONG, q_values=original_q, q_margin=0.07)

    # 1. Mutating original dict does not mutate res.q_values
    original_q["HOLD"] = 999.0
    assert res.q_values["HOLD"] == -0.05

    # 2. Mutating res.q_values directly raises TypeError
    with pytest.raises(TypeError, match="'mappingproxy' object does not support item assignment"):
        res.q_values["HOLD"] = 999.0  # type: ignore[index]

    # 3. Deleting keys from res.q_values directly raises TypeError
    with pytest.raises(TypeError, match="'mappingproxy' object does not support item deletion"):
        del res.q_values["HOLD"]  # type: ignore[attr-defined]

    # 4. Standard mapping read operations work
    assert len(res.q_values) == 5
    assert set(res.q_values.keys()) == {"HOLD", "OPEN_LONG", "CLOSE_LONG", "OPEN_SHORT", "CLOSE_SHORT"}
    assert "OPEN_LONG" in res.q_values


# --- 6. Step 2A Model Loading & Compatibility Validation Tests ---

class _DummyEnv(gym.Env):
    """Minimal Gymnasium environment fixture for SB3 DQN save/load testing."""

    def __init__(self, obs_dim: int = 26, act_count: int = 5):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(act_count)

    def reset(self, seed=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}


@pytest.fixture(scope="module")
def valid_dqn_zip(tmp_path_factory) -> Path:
    """Creates a temporary valid canonical SB3 DQN model archive [128, 128] once for the module."""
    fn = tmp_path_factory.mktemp("models") / "final_model.zip"
    env = _DummyEnv(obs_dim=26, act_count=5)
    model = DQN("MlpPolicy", env, policy_kwargs={"net_arch": [128, 128]}, seed=42, device="cpu", verbose=0)
    model.save(str(fn))
    return fn


def _write_metadata(
    path: Path,
    *,
    obs_dim: Any = 26,
    act_count: Any = 5,
    include_hash: bool = True,
    include_version: bool = True,
    schema_hash: str | None = None,
    schema_version: str | None = None,
    net_arch: list[int] | None = None,
    news_max_age_hours: Any = 24.0,
    include_news_config: bool = True,
) -> None:
    data: dict[str, Any] = {
        "observation_dim": obs_dim,
        "action_count": act_count,
    }
    if include_news_config and news_max_age_hours is not None:
        data["news"] = {"news_max_age_hours": news_max_age_hours}
    if include_hash:
        if schema_hash is not None:
            data["feature_schema_hash"] = schema_hash
        elif news_max_age_hours is not None and isinstance(news_max_age_hours, (int, float)):
            data["feature_schema_hash"] = compute_feature_schema_hash(
                news_max_age_hours=float(news_max_age_hours)
            )
        else:
            data["feature_schema_hash"] = CANONICAL_FEATURE_SCHEMA_HASH
    if include_version:
        data["feature_schema_version"] = schema_version or FEATURE_SCHEMA_VERSION
    if net_arch is not None:
        data["config"] = {"dqn": {"net_arch": net_arch}}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# --- Area 1: Feature-Schema Integrity & Coverage Tests ---

def test_feature_schema_single_source_of_truth_in_features():
    # 26 features defined in features.py
    assert len(CANONICAL_FEATURE_DEFINITIONS) == 26
    names = [d["name"] for d in CANONICAL_FEATURE_DEFINITIONS]
    assert tuple(names) == FEATURE_NAMES
    assert len(FEATURE_NAMES) == OBSERVATION_DIM == 26

    # Verify calculation parameters exist
    assert FEATURE_CALCULATION_PARAMETERS["atr_period"] == 14
    assert FEATURE_CALCULATION_PARAMETERS["log_return_lags"] == (1, 5, 15)
    assert FEATURE_CALCULATION_PARAMETERS["ema_periods"] == (9, 21, 50)
    assert FEATURE_CALCULATION_PARAMETERS["clipping_bounds"] == (-10.0, 10.0)


def test_feature_implementation_integrity_covers_all_five_targets():
    # 1. Live codebase matches pinned implementation digest
    assert verify_feature_implementation_integrity() is True

    base_hash = compute_feature_schema_hash(news_max_age_hours=24.0)

    # 2. Modifying FeatureBuilder.__init__ source changes hash
    h_init = compute_feature_schema_hash(
        implementation_sources=[
            "def _ema(v, p): return 0.0\n",
            "def _log_return(c, l): return 0.0\n",
            "def __init__(self, *, news_max_age_hours=48.0): self.news_max_age_seconds = 48.0 * 3600.0\n",
            "def build(self, obs, history, **kw): return np.zeros(26, dtype=np.float32)\n",
            "class NewsObservation: pass\n",
        ]
    )
    assert h_init != base_hash

    # 3. Modifying NewsObservation source changes hash
    h_news = compute_feature_schema_hash(
        implementation_sources=[
            "def _ema(v, p): return 0.0\n",
            "def _log_return(c, l): return 0.0\n",
            "def __init__(self, *, news_max_age_hours=24.0): self.news_max_age_seconds = 24.0 * 3600.0\n",
            "def build(self, obs, history, **kw): return np.zeros(26, dtype=np.float32)\n",
            "class NewsObservation: status: str = 'fresh'\n",
        ]
    )
    assert h_news != base_hash


def test_feature_schema_hash_sensitive_to_parameters():
    base_hash = compute_feature_schema_hash(news_max_age_hours=24.0)
    assert base_hash == CANONICAL_FEATURE_SCHEMA_HASH

    # 1. Modifying atr_period changes hash
    modified_params = dict(FEATURE_CALCULATION_PARAMETERS)
    modified_params["atr_period"] = 20
    h_atr = compute_feature_schema_hash(parameters=modified_params)
    assert h_atr != base_hash

    # 2. Modifying log_return_lags changes hash
    modified_params2 = dict(FEATURE_CALCULATION_PARAMETERS)
    modified_params2["log_return_lags"] = (1, 5, 20)
    h_lags = compute_feature_schema_hash(parameters=modified_params2)
    assert h_lags != base_hash

    # 3. Modifying clipping_bounds changes hash
    modified_params3 = dict(FEATURE_CALCULATION_PARAMETERS)
    modified_params3["clipping_bounds"] = (-5.0, 5.0)
    h_clip = compute_feature_schema_hash(parameters=modified_params3)
    assert h_clip != base_hash

    # 4. Modifying news_max_age_hours changes hash
    h_age = compute_feature_schema_hash(news_max_age_hours=12.0)
    assert h_age != base_hash


def test_deployment_safe_fallback_when_source_unavailable(monkeypatch):
    import inspect

    def mock_getsource(obj):
        raise OSError("could not get source code")

    monkeypatch.setattr(inspect, "getsource", mock_getsource)

    # In packaged deployment without .py source files:
    # 1. verify_feature_implementation_integrity returns False safely without crashing
    assert verify_feature_implementation_integrity() is False

    # 2. compute_feature_schema_hash falls back to pinned digest without crashing
    fallback_hash = compute_feature_schema_hash(news_max_age_hours=24.0)
    assert fallback_hash == PINNED_CANONICAL_FEATURE_SCHEMA_HASH


# --- Area 2: Strict Metadata Validation Tests ---

def test_strict_metadata_requires_both_hash_and_version(valid_dqn_zip, tmp_path):
    run_dir = tmp_path / "run_missing_fields"
    run_dir.mkdir()
    model_copy = run_dir / "final_model.zip"
    shutil.copy(valid_dqn_zip, model_copy)

    # Missing feature_schema_version (only hash present) -> rejected in strict mode
    _write_metadata(run_dir / "metadata.json", include_hash=True, include_version=False)
    with pytest.raises(IncompatibleModelError, match="lacks required strict schema fields.*feature_schema_version"):
        ModelInference.load(model_copy, allow_legacy_metadata=False)

    # Missing feature_schema_hash (only version present) -> rejected in strict mode
    _write_metadata(run_dir / "metadata.json", include_hash=False, include_version=True)
    with pytest.raises(IncompatibleModelError, match="lacks required strict schema fields.*feature_schema_hash"):
        ModelInference.load(model_copy, allow_legacy_metadata=False)

    # Both missing -> rejected in strict mode
    _write_metadata(run_dir / "metadata.json", include_hash=False, include_version=False)
    with pytest.raises(IncompatibleModelError, match="lacks required strict schema fields"):
        ModelInference.load(model_copy, allow_legacy_metadata=False)


def test_explicit_legacy_opt_in_with_warning(valid_dqn_zip, tmp_path):
    run_dir = tmp_path / "run_legacy_opt_in"
    run_dir.mkdir()
    model_copy = run_dir / "final_model.zip"
    shutil.copy(valid_dqn_zip, model_copy)
    _write_metadata(run_dir / "metadata.json", include_hash=False, include_version=False)

    with pytest.warns(UserWarning, match="Legacy model metadata detected"):
        inf = ModelInference.load(model_copy, allow_legacy_metadata=True)

    assert inf.is_legacy_metadata is True
    assert inf.model is not None


def test_strict_metadata_rejects_hash_mismatch(valid_dqn_zip, tmp_path):
    run_dir = tmp_path / "run_hash_mismatch"
    run_dir.mkdir()
    model_copy = run_dir / "final_model.zip"
    shutil.copy(valid_dqn_zip, model_copy)
    _write_metadata(
        run_dir / "metadata.json",
        schema_hash="0000000000000000000000000000000000000000000000000000000000000000",
    )

    with pytest.raises(IncompatibleModelError, match="Incompatible feature schema hash"):
        ModelInference.load(model_copy)


def test_strict_metadata_rejects_version_mismatch(valid_dqn_zip, tmp_path):
    run_dir = tmp_path / "run_version_mismatch"
    run_dir.mkdir()
    model_copy = run_dir / "final_model.zip"
    shutil.copy(valid_dqn_zip, model_copy)
    _write_metadata(
        run_dir / "metadata.json",
        schema_version="noone-features-v999",
    )

    with pytest.raises(IncompatibleModelError, match="Incompatible feature schema version"):
        ModelInference.load(model_copy)


# --- Area 3: Metadata Type Validation Tests ---

def test_metadata_type_validation_rejects_floats_booleans_and_strings(valid_dqn_zip, tmp_path):
    run_dir = tmp_path / "run_type_checks"
    run_dir.mkdir()
    model_copy = run_dir / "final_model.zip"
    shutil.copy(valid_dqn_zip, model_copy)

    # 1. observation_dim float 26.9
    _write_metadata(run_dir / "metadata.json", obs_dim=26.9, act_count=5)
    with pytest.raises(IncompatibleModelError, match="Field 'observation_dim' in metadata must be a strict integer, got float"):
        ModelInference.load(model_copy)

    # 2. observation_dim string "26"
    _write_metadata(run_dir / "metadata.json", obs_dim="26", act_count=5)
    with pytest.raises(IncompatibleModelError, match="Field 'observation_dim' in metadata must be a strict integer, got str"):
        ModelInference.load(model_copy)

    # 3. observation_dim boolean True
    _write_metadata(run_dir / "metadata.json", obs_dim=True, act_count=5)
    with pytest.raises(IncompatibleModelError, match="Field 'observation_dim' in metadata must be a strict integer, got bool"):
        ModelInference.load(model_copy)

    # 4. action_count float 5.0
    _write_metadata(run_dir / "metadata.json", obs_dim=26, act_count=5.0)
    with pytest.raises(IncompatibleModelError, match="Field 'action_count' in metadata must be a strict integer, got float"):
        ModelInference.load(model_copy)

    # 5. action_count string "5"
    _write_metadata(run_dir / "metadata.json", obs_dim=26, act_count="5")
    with pytest.raises(IncompatibleModelError, match="Field 'action_count' in metadata must be a strict integer, got str"):
        ModelInference.load(model_copy)

    # 6. action_count boolean False
    _write_metadata(run_dir / "metadata.json", obs_dim=26, act_count=False)
    with pytest.raises(IncompatibleModelError, match="Field 'action_count' in metadata must be a strict integer, got bool"):
        ModelInference.load(model_copy)


# --- Area 3: Runtime Configuration Compatibility Tests ---

def test_strict_loading_requires_runtime_news_configuration(valid_dqn_zip, tmp_path):
    run_dir = tmp_path / "run_no_news_cfg"
    run_dir.mkdir()
    model_copy = run_dir / "final_model.zip"
    shutil.copy(valid_dqn_zip, model_copy)

    # Strict metadata with hash and version but missing news_max_age_hours
    _write_metadata(run_dir / "metadata.json", include_news_config=False)
    with pytest.raises(IncompatibleModelError, match="lacks required runtime feature configuration 'news_max_age_hours'"):
        ModelInference.load(model_copy)


def test_strict_loading_validates_custom_news_max_age_hours(valid_dqn_zip, tmp_path):
    run_dir = tmp_path / "run_custom_age"
    run_dir.mkdir()
    model_copy = run_dir / "final_model.zip"
    shutil.copy(valid_dqn_zip, model_copy)

    # 1. Custom news_max_age_hours=12.0 with matching hash succeeds
    _write_metadata(run_dir / "metadata.json", news_max_age_hours=12.0)
    inf = ModelInference.load(model_copy)
    assert inf.model is not None

    # 2. Custom news_max_age_hours=12.0 with mismatched 24.0h canonical hash fails
    _write_metadata(run_dir / "metadata.json", news_max_age_hours=12.0, schema_hash=CANONICAL_FEATURE_SCHEMA_HASH)
    with pytest.raises(IncompatibleModelError, match="Incompatible feature schema hash for news_max_age_hours=12"):
        ModelInference.load(model_copy)


def test_strict_loading_rejects_invalid_news_max_age_hours(valid_dqn_zip, tmp_path):
    run_dir = tmp_path / "run_invalid_age"
    run_dir.mkdir()
    model_copy = run_dir / "final_model.zip"
    shutil.copy(valid_dqn_zip, model_copy)

    # String age
    _write_metadata(run_dir / "metadata.json", news_max_age_hours="24.0")
    with pytest.raises(IncompatibleModelError, match="Invalid 'news_max_age_hours' in metadata"):
        ModelInference.load(model_copy)

    # Boolean age
    _write_metadata(run_dir / "metadata.json", news_max_age_hours=True)
    with pytest.raises(IncompatibleModelError, match="Invalid 'news_max_age_hours' in metadata"):
        ModelInference.load(model_copy)

    # Negative age
    _write_metadata(run_dir / "metadata.json", news_max_age_hours=-5.0)
    with pytest.raises(IncompatibleModelError, match="Invalid 'news_max_age_hours' in metadata: must be positive"):
        ModelInference.load(model_copy)

    # Zero age
    _write_metadata(run_dir / "metadata.json", news_max_age_hours=0.0)
    with pytest.raises(IncompatibleModelError, match="Invalid 'news_max_age_hours' in metadata: must be positive"):
        ModelInference.load(model_copy)


# --- Area 4: DQN Architecture Validation Tests ---

def test_model_inference_load_canonical_architecture(valid_dqn_zip, tmp_path):
    run_dir = tmp_path / "run_arch_canonical"
    run_dir.mkdir()
    model_copy = run_dir / "final_model.zip"
    shutil.copy(valid_dqn_zip, model_copy)
    _write_metadata(run_dir / "metadata.json", net_arch=[128, 128])

    # Default expected_net_arch=(128, 128) succeeds
    inf = ModelInference.load(model_copy)
    assert inf.model is not None


def test_model_inference_rejects_architecture_mismatch(tmp_path):
    # Model trained with [16, 16] instead of canonical [128, 128]
    arch16_zip = tmp_path / "model_16.zip"
    env = _DummyEnv(obs_dim=26, act_count=5)
    model16 = DQN("MlpPolicy", env, policy_kwargs={"net_arch": [16, 16]}, seed=42, device="cpu", verbose=0)
    model16.save(str(arch16_zip))

    # Metadata without net_arch -> default expected_net_arch=(128, 128) rejects [16, 16]
    _write_metadata(tmp_path / "metadata.json")
    with pytest.raises(IncompatibleModelError, match=r"Loaded model hidden architecture \[16, 16\] != expected \[128, 128\]"):
        ModelInference.load(arch16_zip)


def test_model_inference_rejects_metadata_declared_architecture_inconsistency(valid_dqn_zip, tmp_path):
    run_dir = tmp_path / "run_declared_arch_mismatch"
    run_dir.mkdir()
    model_copy = run_dir / "final_model.zip"
    shutil.copy(valid_dqn_zip, model_copy)

    # Model is [128, 128], but metadata declares [64, 64]
    _write_metadata(run_dir / "metadata.json", net_arch=[64, 64])
    with pytest.raises(IncompatibleModelError, match=r"Metadata declared net_arch \[64, 64\] does not match actual model architecture \[128, 128\]"):
        ModelInference.load(model_copy, expected_net_arch=None)


def test_model_inference_custom_architecture_matching(tmp_path):
    # Model trained with [16, 16]
    arch16_zip = tmp_path / "model_custom16.zip"
    env = _DummyEnv(obs_dim=26, act_count=5)
    model16 = DQN("MlpPolicy", env, policy_kwargs={"net_arch": [16, 16]}, seed=42, device="cpu", verbose=0)
    model16.save(str(arch16_zip))

    meta_file = tmp_path / "metadata.json"
    _write_metadata(meta_file, net_arch=[16, 16])

    # Loading with expected_net_arch=(16, 16) matches both actual and declared architecture
    inf = ModelInference.load(arch16_zip, expected_net_arch=(16, 16))
    assert inf.model is not None


# --- General Loading & Artifact Failure Tests ---

def test_model_inference_load_from_checkpoint_in_subfolder(valid_dqn_zip, tmp_path):
    run_dir = tmp_path / "run_checkpoint"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt_model = ckpt_dir / "dqn_1000_steps.zip"
    shutil.copy(valid_dqn_zip, ckpt_model)
    _write_metadata(run_dir / "metadata.json")

    inf = ModelInference.load(ckpt_model)
    assert inf.model is not None
    assert inf.metadata_path == run_dir / "metadata.json"


def test_model_inference_load_with_explicit_metadata_path(valid_dqn_zip, tmp_path):
    run_dir = tmp_path / "run_explicit"
    run_dir.mkdir()
    model_copy = run_dir / "model.zip"
    shutil.copy(valid_dqn_zip, model_copy)
    custom_meta = tmp_path / "separate_metadata.json"
    _write_metadata(custom_meta)

    inf = ModelInference.load(model_copy, metadata_path=custom_meta)
    assert inf.metadata_path == custom_meta


def test_model_inference_load_without_gym_env(valid_dqn_zip, tmp_path):
    run_dir = tmp_path / "run_no_env"
    run_dir.mkdir()
    model_copy = run_dir / "final_model.zip"
    shutil.copy(valid_dqn_zip, model_copy)
    _write_metadata(run_dir / "metadata.json")

    inf = ModelInference.load(model_copy)
    assert inf.model.policy.training is False


def test_model_inference_missing_model_file_raises_not_found(tmp_path):
    missing_path = tmp_path / "does_not_exist.zip"
    with pytest.raises(ModelNotFoundError, match="Model file not found"):
        ModelInference.load(missing_path)


def test_model_inference_missing_metadata_raises_not_found(valid_dqn_zip, tmp_path):
    run_dir = tmp_path / "run_missing_meta"
    run_dir.mkdir()
    model_copy = run_dir / "final_model.zip"
    shutil.copy(valid_dqn_zip, model_copy)

    with pytest.raises(ModelNotFoundError, match="Metadata file not found for model"):
        ModelInference.load(model_copy)


def test_model_inference_missing_explicit_metadata_raises_not_found(valid_dqn_zip, tmp_path):
    run_dir = tmp_path / "run_missing_exp_meta"
    run_dir.mkdir()
    model_copy = run_dir / "final_model.zip"
    shutil.copy(valid_dqn_zip, model_copy)

    with pytest.raises(ModelNotFoundError, match="Metadata file not found"):
        ModelInference.load(model_copy, metadata_path=tmp_path / "nonexistent.json")


def test_model_inference_corrupt_model_zip_not_zip(tmp_path):
    run_dir = tmp_path / "run_corrupt_zip"
    run_dir.mkdir()
    bad_zip = run_dir / "final_model.zip"
    bad_zip.write_bytes(b"not a valid zip file content")
    _write_metadata(run_dir / "metadata.json")

    with pytest.raises(CorruptModelError, match="not a valid ZIP archive"):
        ModelInference.load(bad_zip)


def test_model_inference_corrupt_model_zip_missing_members(tmp_path):
    run_dir = tmp_path / "run_incomplete_zip"
    run_dir.mkdir()
    bad_zip = run_dir / "final_model.zip"
    with zipfile.ZipFile(bad_zip, "w") as z:
        z.writestr("dummy.txt", "hello")
    _write_metadata(run_dir / "metadata.json")

    with pytest.raises(CorruptModelError, match="missing internal members"):
        ModelInference.load(bad_zip)


def test_model_inference_corrupt_metadata_invalid_json(valid_dqn_zip, tmp_path):
    run_dir = tmp_path / "run_bad_json"
    run_dir.mkdir()
    model_copy = run_dir / "final_model.zip"
    shutil.copy(valid_dqn_zip, model_copy)
    (run_dir / "metadata.json").write_text("{broken json syntax: invalid", encoding="utf-8")

    with pytest.raises(CorruptModelError, match="Metadata file is malformed JSON"):
        ModelInference.load(model_copy)


def test_model_inference_corrupt_metadata_not_dict(valid_dqn_zip, tmp_path):
    run_dir = tmp_path / "run_list_json"
    run_dir.mkdir()
    model_copy = run_dir / "final_model.zip"
    shutil.copy(valid_dqn_zip, model_copy)
    (run_dir / "metadata.json").write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(CorruptModelError, match="Metadata must be a JSON object"):
        ModelInference.load(model_copy)


def test_model_inference_missing_required_metadata_fields(valid_dqn_zip, tmp_path):
    run_dir = tmp_path / "run_missing_fields"
    run_dir.mkdir()
    model_copy = run_dir / "final_model.zip"
    shutil.copy(valid_dqn_zip, model_copy)

    # Missing observation_dim
    (run_dir / "metadata.json").write_text(json.dumps({"action_count": 5}), encoding="utf-8")
    with pytest.raises(IncompatibleModelError, match="missing required field 'observation_dim'"):
        ModelInference.load(model_copy)

    # Missing action_count
    (run_dir / "metadata.json").write_text(json.dumps({"observation_dim": 26}), encoding="utf-8")
    with pytest.raises(IncompatibleModelError, match="missing required field 'action_count'"):
        ModelInference.load(model_copy)


def test_model_inference_metadata_dimension_mismatches(valid_dqn_zip, tmp_path):
    run_dir = tmp_path / "run_meta_mismatch"
    run_dir.mkdir()
    model_copy = run_dir / "final_model.zip"
    shutil.copy(valid_dqn_zip, model_copy)

    # Obs dim mismatch
    _write_metadata(run_dir / "metadata.json", obs_dim=25)
    with pytest.raises(IncompatibleModelError, match="Incompatible observation_dim in metadata: 25 != expected 26"):
        ModelInference.load(model_copy)

    # Action count mismatch
    _write_metadata(run_dir / "metadata.json", act_count=3)
    with pytest.raises(IncompatibleModelError, match="Incompatible action_count in metadata: 3 != expected 5"):
        ModelInference.load(model_copy)


def test_model_inference_loaded_observation_space_mismatch(tmp_path):
    bad_model_zip = tmp_path / "model_obs10.zip"
    env = _DummyEnv(obs_dim=10, act_count=5)
    bad_model = DQN("MlpPolicy", env, policy_kwargs={"net_arch": [128, 128]}, seed=42, device="cpu", verbose=0)
    bad_model.save(str(bad_model_zip))

    _write_metadata(tmp_path / "metadata.json", obs_dim=26, act_count=5)
    with pytest.raises(IncompatibleModelError, match=r"Loaded model observation space shape \(10,\) != \(26,\)"):
        ModelInference.load(bad_model_zip)


def test_model_inference_loaded_action_space_mismatch(tmp_path):
    bad_model_zip = tmp_path / "model_act3.zip"
    env = _DummyEnv(obs_dim=26, act_count=3)
    bad_model = DQN("MlpPolicy", env, policy_kwargs={"net_arch": [128, 128]}, seed=42, device="cpu", verbose=0)
    bad_model.save(str(bad_model_zip))

    _write_metadata(tmp_path / "metadata.json", obs_dim=26, act_count=5)
    with pytest.raises(IncompatibleModelError, match=r"Loaded model action space.*Discrete\(3\).*!= Discrete\(5\)"):
        ModelInference.load(bad_model_zip)


def test_model_inference_cpu_enforcement(valid_dqn_zip, tmp_path):
    run_dir = tmp_path / "run_cpu"
    run_dir.mkdir()
    model_copy = run_dir / "final_model.zip"
    shutil.copy(valid_dqn_zip, model_copy)
    _write_metadata(run_dir / "metadata.json")

    inf = ModelInference.load(model_copy)
    assert inf.device == "cpu"
    assert str(inf.model.device) == "cpu"
    assert str(inf.model.policy.device) == "cpu"


# --- 7. Step 2B Observation Generation & Inference Prediction Tests ---

class _MockQNet(torch.nn.Module):
    """Mock PyTorch module to simulate specific Q-network outputs for test verification."""

    def __init__(self, tensor: torch.Tensor) -> None:
        super().__init__()
        self.tensor = tensor

    def forward(self, x: Any) -> torch.Tensor:
        return self.tensor


def _make_valid_candles(count: int = 60, start_time: int = 1_000_000) -> list[dict[str, Any]]:
    candles = []
    for i in range(count):
        t = start_time + i * 60
        candles.append({
            "open": 100.0 + i * 0.1,
            "high": 105.0 + i * 0.1,
            "low": 95.0 + i * 0.1,
            "close": 102.0 + i * 0.1,
            "volume": 50.0 + i,
            "open_time": t,
            "close_time": t + 59,
            "is_closed": True,
        })
    return candles


def _make_valid_klines(count: int = 60) -> list[Kline]:
    base_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        Kline(
            symbol="BTCUSDT",
            interval="1m",
            open_time=base_dt.replace(minute=i % 60, hour=(i // 60)),
            close_time=base_dt.replace(minute=i % 60, hour=(i // 60)).replace(second=59),
            open=100.0 + i * 0.1,
            high=105.0 + i * 0.1,
            low=95.0 + i * 0.1,
            close=102.0 + i * 0.1,
            volume=50.0 + i,
            quote_volume=5100.0,
            trades=10,
            is_closed=True,
        )
        for i in range(count)
    ]


@pytest.fixture
def loaded_inference(valid_dqn_zip, tmp_path_factory) -> ModelInference:
    run_dir = tmp_path_factory.mktemp("run_inf_step2b")
    model_copy = run_dir / "final_model.zip"
    shutil.copy(valid_dqn_zip, model_copy)
    _write_metadata(run_dir / "metadata.json", news_max_age_hours=24.0)
    return ModelInference.load(model_copy)


def test_feature_builder_integration_configured_news_age(valid_dqn_zip, tmp_path):
    run_dir = tmp_path / "run_news_12h"
    run_dir.mkdir()
    model_copy = run_dir / "final_model.zip"
    shutil.copy(valid_dqn_zip, model_copy)
    _write_metadata(run_dir / "metadata.json", news_max_age_hours=12.0)

    inf = ModelInference.load(model_copy)
    assert inf.news_max_age_hours == 12.0
    assert inf.feature_builder is not None
    assert inf.feature_builder.news_max_age_seconds == 12.0 * 3600.0


def test_legacy_metadata_without_news_config_blocks_observation_building(valid_dqn_zip, tmp_path):
    run_dir = tmp_path / "run_legacy_no_news"
    run_dir.mkdir()
    model_copy = run_dir / "final_model.zip"
    shutil.copy(valid_dqn_zip, model_copy)
    # Write legacy metadata with no schema hash and no news config
    (run_dir / "metadata.json").write_text(
        json.dumps({"observation_dim": 26, "action_count": 5}), encoding="utf-8"
    )

    with pytest.warns(UserWarning, match="Legacy model metadata detected"):
        inf = ModelInference.load(model_copy, allow_legacy_metadata=True)

    assert inf.is_legacy_metadata is True
    assert inf.news_max_age_hours is None
    assert inf.feature_builder is None

    candles = _make_valid_candles(60)
    account = AccountSnapshot(equity=10_000.0, available_balance=10_000.0)

    with pytest.raises(IncompatibleModelError, match="Cannot build observation: model metadata lacks required 'news_max_age_hours'"):
        inf.build_observation(candles, account)

    with pytest.raises(IncompatibleModelError, match="Cannot build observation: model metadata lacks required 'news_max_age_hours'"):
        inf.predict(candles, account)

    # Pre-constructed observation still works
    valid_obs = np.zeros(26, dtype=np.float32)
    res = inf.predict_observation(valid_obs)
    assert isinstance(res, PredictionResult)


def test_build_observation_shape_and_dtype(loaded_inference):
    candles = _make_valid_candles(55)
    account = AccountSnapshot(equity=10_000.0, available_balance=10_000.0)
    obs = loaded_inference.build_observation(candles, account)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (26,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()


def test_build_observation_with_klines(loaded_inference):
    klines = _make_valid_klines(60)
    account = AccountSnapshot(equity=10_000.0, available_balance=10_000.0)
    obs = loaded_inference.build_observation(klines, account)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (26,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()


def test_insufficient_candle_history_rejected(loaded_inference):
    account = AccountSnapshot(equity=10_000.0, available_balance=10_000.0)

    # 49 candles
    short_candles = _make_valid_candles(49)
    with pytest.raises(InsufficientHistoryError, match="Insufficient candle history: requires at least 50 closed candles, got 49"):
        loaded_inference.build_observation(short_candles, account)

    # 0 candles
    with pytest.raises(InsufficientHistoryError, match="Insufficient candle history: requires at least 50 closed candles, got 0"):
        loaded_inference.build_observation([], account)


def test_unclosed_candles_rejected(loaded_inference):
    account = AccountSnapshot(equity=10_000.0, available_balance=10_000.0)
    candles = _make_valid_candles(50)
    candles[-1]["is_closed"] = False

    with pytest.raises(InvalidObservationError, match="Candle at index 49 is not closed"):
        loaded_inference.build_observation(candles, account)


def test_duplicate_or_out_of_order_candles_rejected(loaded_inference):
    account = AccountSnapshot(equity=10_000.0, available_balance=10_000.0)

    # Duplicate open_time
    candles_dup = _make_valid_candles(50)
    candles_dup[10]["open_time"] = candles_dup[9]["open_time"]
    with pytest.raises(InvalidObservationError, match="Duplicate candle timestamp detected at index 10"):
        loaded_inference.build_observation(candles_dup, account)

    # Out of order
    candles_ooo = _make_valid_candles(50)
    candles_ooo[10]["open_time"] = candles_ooo[9]["open_time"] - 100
    with pytest.raises(InvalidObservationError, match="Candles not in chronological order at index 10"):
        loaded_inference.build_observation(candles_ooo, account)

    # Missing open_time
    candles_missing = _make_valid_candles(50)
    candles_missing[5]["open_time"] = None
    with pytest.raises(InvalidObservationError, match="Candle at index 5 missing open_time"):
        loaded_inference.build_observation(candles_missing, account)

    # close_time <= open_time
    candles_bad_close = _make_valid_candles(50)
    candles_bad_close[7]["close_time"] = candles_bad_close[7]["open_time"]
    with pytest.raises(InvalidObservationError, match=r"Invalid candle interval at index 7: close_time.*<= open_time"):
        loaded_inference.build_observation(candles_bad_close, account)


def test_invalid_and_non_finite_ohlcv_rejected(loaded_inference):
    account = AccountSnapshot(equity=10_000.0, available_balance=10_000.0)

    # NaN close
    c_nan = _make_valid_candles(50)
    c_nan[12]["close"] = float("nan")
    with pytest.raises(InvalidObservationError, match="Candle at index 12 has non-finite close"):
        loaded_inference.build_observation(c_nan, account)

    # Inf high
    c_inf = _make_valid_candles(50)
    c_inf[12]["high"] = float("inf")
    with pytest.raises(InvalidObservationError, match="Candle at index 12 has non-finite high"):
        loaded_inference.build_observation(c_inf, account)

    # Non-positive close
    c_neg = _make_valid_candles(50)
    c_neg[12]["close"] = 0.0
    with pytest.raises(InvalidObservationError, match="Candle at index 12 has non-positive close"):
        loaded_inference.build_observation(c_neg, account)

    # High below max(open, close, low)
    c_bad_high = _make_valid_candles(50)
    c_bad_high[15]["high"] = 90.0  # lower than open 101.5
    with pytest.raises(InvalidObservationError, match="Candle at index 15 has high.*below open/close/low"):
        loaded_inference.build_observation(c_bad_high, account)

    # Low above min(open, close, high)
    c_bad_low = _make_valid_candles(50)
    c_bad_low[15]["low"] = 104.0  # higher than open 101.5
    with pytest.raises(InvalidObservationError, match="Candle at index 15 has low.*above open/close/high"):
        loaded_inference.build_observation(c_bad_low, account)

    # Negative volume
    c_bad_vol = _make_valid_candles(50)
    c_bad_vol[20]["volume"] = -1.0
    with pytest.raises(InvalidObservationError, match="Candle at index 20 has negative volume"):
        loaded_inference.build_observation(c_bad_vol, account)

    # Non-iterable candles
    with pytest.raises(InvalidObservationError, match="candles must be a sequence"):
        loaded_inference.build_observation(None, account)

    with pytest.raises(InvalidObservationError, match="candles must be a sequence"):
        loaded_inference.build_observation(True, account)


def test_funding_and_oi_missing_and_invalid(loaded_inference):
    candles = _make_valid_candles(50)
    account = AccountSnapshot(equity=10_000.0, available_balance=10_000.0)

    # Missing: both None -> funding_oi_available is 0.0 (feature index 11)
    obs_missing = loaded_inference.build_observation(candles, account, funding_rate=None, open_interest_change=None)
    assert obs_missing[9] == 0.0   # funding_rate
    assert obs_missing[10] == 0.0  # open_interest_change
    assert obs_missing[11] == 0.0  # funding_oi_available

    # Real zero funding: funding_oi_available must be 1.0!
    obs_zero = loaded_inference.build_observation(candles, account, funding_rate=0.0)
    assert obs_zero[9] == 0.0
    assert obs_zero[11] == 1.0

    # Non-zero funding and OI
    obs_active = loaded_inference.build_observation(
        candles, account, funding_rate=0.0001, open_interest_change=0.05
    )
    assert obs_active[9] == pytest.approx(0.0001)
    assert obs_active[10] == pytest.approx(0.05)
    assert obs_active[11] == 1.0

    # Rejection of NaN funding
    with pytest.raises(InvalidObservationError, match="funding_rate must be finite"):
        loaded_inference.build_observation(candles, account, funding_rate=float("nan"))

    # Rejection of Inf OI
    with pytest.raises(InvalidObservationError, match="open_interest_change must be finite"):
        loaded_inference.build_observation(candles, account, open_interest_change=float("inf"))

    # Rejection of boolean funding
    with pytest.raises(InvalidObservationError, match="funding_rate must be numeric or None"):
        loaded_inference.build_observation(candles, account, funding_rate=True)


def test_news_states_and_validation(loaded_inference):
    candles = _make_valid_candles(50)
    account = AccountSnapshot(equity=10_000.0, available_balance=10_000.0)

    idx_avail = FEATURE_NAMES.index("news_available")        # 20
    idx_unavail = FEATURE_NAMES.index("news_unavailable")    # 21
    idx_compound = FEATURE_NAMES.index("news_compound")      # 22
    idx_conf = FEATURE_NAMES.index("news_confidence")        # 23
    idx_count = FEATURE_NAMES.index("news_count_log")        # 24
    idx_age = FEATURE_NAMES.index("news_age_normalized")     # 25

    # 1. None news -> status "none"
    obs_none = loaded_inference.build_observation(candles, account, news=None)
    assert obs_none[idx_avail] == 0.0
    assert obs_none[idx_unavail] == 0.0

    # 2. Fresh news
    fresh = NewsObservation(status="fresh", mean_compound=0.6, mean_confidence=0.9, count=4, age_seconds=3600.0)
    obs_fresh = loaded_inference.build_observation(candles, account, news=fresh)
    assert obs_fresh[idx_avail] == 1.0
    assert obs_fresh[idx_unavail] == 0.0
    assert obs_fresh[idx_compound] == pytest.approx(0.6)
    assert obs_fresh[idx_conf] == pytest.approx(0.9)
    assert obs_fresh[idx_count] == pytest.approx(math.log1p(4))
    # Normalized age: 3600 / (24 * 3600) = 1/24
    assert obs_fresh[idx_age] == pytest.approx(3600.0 / (24.0 * 3600.0))

    # 3. Unavailable news
    unavail = NewsObservation(status="unavailable")
    obs_unavail = loaded_inference.build_observation(candles, account, news=unavail)
    assert obs_unavail[idx_avail] == 0.0
    assert obs_unavail[idx_unavail] == 1.0

    # 4. Invalid news status
    with pytest.raises(InvalidObservationError, match="Invalid news status: 'broken'"):
        loaded_inference.build_observation(candles, account, news=NewsObservation(status="broken"))

    # 5. Non-finite compound
    with pytest.raises(InvalidObservationError, match="news.mean_compound must be finite"):
        loaded_inference.build_observation(candles, account, news=NewsObservation(status="fresh", mean_compound=float("nan")))

    # 6. Negative confidence
    with pytest.raises(InvalidObservationError, match="news.mean_confidence must be non-negative"):
        loaded_inference.build_observation(candles, account, news=NewsObservation(status="fresh", mean_confidence=-0.1))

    # 7. Negative count
    with pytest.raises(InvalidObservationError, match="news.count must be non-negative"):
        loaded_inference.build_observation(candles, account, news=NewsObservation(status="fresh", count=-1))

    # 8. Negative age
    with pytest.raises(InvalidObservationError, match="news.age_seconds must be non-negative"):
        loaded_inference.build_observation(candles, account, news=NewsObservation(status="fresh", age_seconds=-10.0))

    # 9. Not a NewsObservation
    with pytest.raises(InvalidObservationError, match="news must be an instance of NewsObservation or None"):
        loaded_inference.build_observation(candles, account, news={"status": "fresh"})


def test_invalid_account_state_rejected(loaded_inference):
    candles = _make_valid_candles(50)

    # Not an AccountSnapshot
    with pytest.raises(InvalidAccountStateError, match="account must be an instance of AccountSnapshot, got dict"):
        loaded_inference.build_observation(candles, {"equity": 10_000.0})

    with pytest.raises(InvalidAccountStateError, match="account must be an instance of AccountSnapshot, got NoneType"):
        loaded_inference.build_observation(candles, None)


def test_predict_observation_nan_inf_shape_dtype_rejection(loaded_inference):
    # Shape mismatch
    with pytest.raises(InvalidObservationError, match=r"observation shape \(25,\) != \(26,\)"):
        loaded_inference.predict_observation(np.zeros(25, dtype=np.float32))

    with pytest.raises(InvalidObservationError, match=r"observation shape \(1, 26\) != \(26,\)"):
        loaded_inference.predict_observation(np.zeros((1, 26), dtype=np.float32))

    # Dtype mismatch
    with pytest.raises(InvalidObservationError, match="observation dtype must be float32, got float64"):
        loaded_inference.predict_observation(np.zeros(26, dtype=np.float64))

    # NaN in observation
    bad_obs_nan = np.zeros(26, dtype=np.float32)
    bad_obs_nan[0] = np.nan
    with pytest.raises(InvalidObservationError, match="observation contains non-finite values"):
        loaded_inference.predict_observation(bad_obs_nan)

    # Inf in observation
    bad_obs_inf = np.zeros(26, dtype=np.float32)
    bad_obs_inf[5] = np.inf
    with pytest.raises(InvalidObservationError, match="observation contains non-finite values"):
        loaded_inference.predict_observation(bad_obs_inf)

    # Non-array input
    with pytest.raises(InvalidObservationError, match="observation must be a numpy ndarray"):
        loaded_inference.predict_observation([0.0] * 26)


def test_deterministic_prediction_and_q_values(loaded_inference):
    candles = _make_valid_candles(60)
    account = AccountSnapshot(equity=10_000.0, available_balance=10_000.0)

    res1 = loaded_inference.predict(candles, account)
    res2 = loaded_inference.predict(candles, account)

    # Deterministic consistency
    assert res1.action == res2.action
    assert res1.q_margin == res2.q_margin
    assert res1.q_values == res2.q_values

    # Action in Action enum
    assert isinstance(res1.action, Action)
    assert res1.action in (Action.HOLD, Action.OPEN_LONG, Action.CLOSE_LONG, Action.OPEN_SHORT, Action.CLOSE_SHORT)

    # Q-values validation
    assert len(res1.q_values) == 5
    for a in Action:
        assert a.name in res1.q_values
        assert math.isfinite(res1.q_values[a.name])

    # Q-margin matches difference between highest and second-highest
    sorted_q = sorted(res1.q_values.values(), reverse=True)
    assert res1.q_margin == pytest.approx(sorted_q[0] - sorted_q[1], rel=1e-5, abs=1e-6)

    # Immutability
    with pytest.raises(TypeError):
        res1.q_values["HOLD"] = 999.0


def test_action_mapping_all_five_actions(loaded_inference, monkeypatch):
    """Confirm model prediction works and accurately maps to all five Action enum members."""
    obs = np.zeros(26, dtype=np.float32)

    for target_act in Action:
        # Create a mock QNetwork output where target_act has the highest value
        mock_q = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1]], dtype=torch.float32)
        mock_q[0, target_act.value] = 0.9

        monkeypatch.setattr(loaded_inference.model.policy.q_net, "forward", lambda x, mq=mock_q: mq)
        monkeypatch.setattr(loaded_inference.model, "predict", lambda x, deterministic=True, ta=target_act: (np.array(ta.value), None))

        res = loaded_inference.predict_observation(obs)
        assert isinstance(res.action, Action)
        assert res.action == target_act
        assert res.action.value == target_act.value
        assert res.q_values[target_act.name] == pytest.approx(0.9)
        assert res.q_margin == pytest.approx(0.8)


def test_argmax_tie_breaking_and_zero_margin(loaded_inference, monkeypatch):
    """Confirm tied highest Q-values yield q_margin == 0.0 and adhere to lowest-index tie breaking."""
    obs = np.zeros(26, dtype=np.float32)
    # Tie between Action.HOLD (0) and Action.OPEN_LONG (1) with highest value 0.5
    tied_q = torch.tensor([[0.5, 0.5, 0.2, 0.1, -0.1]], dtype=torch.float32)

    monkeypatch.setattr(loaded_inference.model.policy.q_net, "forward", lambda x: tied_q)
    # PyTorch argmax breaks ties to the lowest index (0 = Action.HOLD)
    monkeypatch.setattr(loaded_inference.model, "predict", lambda x, deterministic=True: (np.array(0), None))

    res = loaded_inference.predict_observation(obs)
    assert res.action == Action.HOLD
    assert res.q_margin == 0.0
    assert res.q_values["HOLD"] == pytest.approx(0.5)
    assert res.q_values["OPEN_LONG"] == pytest.approx(0.5)


def test_invalid_model_outputs_and_non_finite_q_values(loaded_inference, monkeypatch):
    obs = np.zeros(26, dtype=np.float32)

    # 1. Non-finite Q-values (NaN)
    nan_q = torch.tensor([[float("nan"), 0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    monkeypatch.setattr(loaded_inference.model.policy.q_net, "forward", lambda x: nan_q)
    with pytest.raises(ModelInferenceError, match="Q-network produced non-finite Q-values"):
        loaded_inference.predict_observation(obs)

    # 2. Unexpected Q-network shape
    bad_shape_q = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    monkeypatch.setattr(loaded_inference.model.policy.q_net, "forward", lambda x: bad_shape_q)
    with pytest.raises(ModelInferenceError, match=r"Unexpected Q-network output shape"):
        loaded_inference.predict_observation(obs)

    # 3. Model returns out-of-range action
    valid_q = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=torch.float32)
    monkeypatch.setattr(loaded_inference.model.policy.q_net, "forward", lambda x: valid_q)
    monkeypatch.setattr(loaded_inference.model, "predict", lambda x, deterministic=True: (np.array(99), None))
    with pytest.raises(ModelInferenceError, match="Model returned action out of range"):
        loaded_inference.predict_observation(obs)

    # 4. Model action disagrees with Q-network argmax
    # Q-network argmax is 4, but model predicts 1
    monkeypatch.setattr(loaded_inference.model, "predict", lambda x, deterministic=True: (np.array(1), None))
    with pytest.raises(ModelInferenceError, match="Deterministic action 1 disagrees with Q-network argmax 4"):
        loaded_inference.predict_observation(obs)


def test_predict_delegates_to_build_and_predict_observation(loaded_inference):
    candles = _make_valid_candles(60)
    account = AccountSnapshot(equity=10_000.0, available_balance=10_000.0)
    news = NewsObservation(status="fresh", mean_compound=0.3, count=2)

    obs = loaded_inference.build_observation(candles, account, news=news, funding_rate=0.0002)
    res_direct = loaded_inference.predict_observation(obs)
    res_predict = loaded_inference.predict(candles, account, news=news, funding_rate=0.0002)

    assert res_direct.action == res_predict.action
    assert res_direct.q_margin == res_predict.q_margin
    assert res_direct.q_values == res_predict.q_values


def test_regression_existing_feature_calculation_preserved(loaded_inference):
    """Ensure build_observation produces byte-for-byte identical features to FeatureBuilder."""
    candles = _make_valid_klines(60)
    account = AccountSnapshot(
        equity=15_000.0,
        available_balance=12_000.0,
        position_side=1,
        position_quantity=0.5,
        entry_price=101.0,
        unrealized_pnl=50.0,
        margin_used=1_000.0,
        progress=0.0,
    )
    news = NewsObservation(status="fresh", mean_compound=0.45, mean_confidence=0.85, count=5, age_seconds=7200.0)

    obs_inference = loaded_inference.build_observation(
        candles, account, news=news, funding_rate=0.00015, open_interest_change=-0.02
    )

    fb = FeatureBuilder(news_max_age_hours=24.0)
    obs_direct = fb.build(
        obs=account.to_obs_dict(),
        history=candles,
        news=news,
        funding_rate=0.00015,
        open_interest_change=-0.02,
    )

    np.testing.assert_array_equal(obs_inference, obs_direct)


# --- 8. Step 2C Batch Prediction & Package Root Tests ---

def test_batch_prediction_parity_against_sequential(loaded_inference):
    """Verify batch prediction returns identical results to sequential predict_observation."""
    rng = np.random.RandomState(42)
    obs_batch = rng.uniform(-1.0, 1.0, size=(20, 26)).astype(np.float32)

    batch_results = loaded_inference.predict_observations(obs_batch)
    assert isinstance(batch_results, tuple)
    assert len(batch_results) == 20

    for i in range(20):
        single_res = loaded_inference.predict_observation(obs_batch[i])
        b_res = batch_results[i]
        assert b_res.action == single_res.action
        assert b_res.q_margin == pytest.approx(single_res.q_margin, rel=1e-5, abs=1e-6)
        for a in Action:
            assert b_res.q_values[a.name] == pytest.approx(single_res.q_values[a.name], rel=1e-5, abs=1e-6)


def test_batch_prediction_all_five_actions(loaded_inference, monkeypatch):
    """Confirm batch prediction accurately extracts Q-values and maps all five actions."""
    obs_batch = np.zeros((5, 26), dtype=np.float32)

    # 5 rows, each row has a different argmax action (0..4)
    mock_q = torch.tensor([
        [0.9, 0.1, 0.1, 0.1, 0.1],  # Action 0 (HOLD)
        [0.1, 0.9, 0.1, 0.1, 0.1],  # Action 1 (OPEN_LONG)
        [0.1, 0.1, 0.9, 0.1, 0.1],  # Action 2 (CLOSE_LONG)
        [0.1, 0.1, 0.1, 0.9, 0.1],  # Action 3 (OPEN_SHORT)
        [0.1, 0.1, 0.1, 0.1, 0.9],  # Action 4 (CLOSE_SHORT)
    ], dtype=torch.float32)

    monkeypatch.setattr(loaded_inference.model.policy.q_net, "forward", lambda x: mock_q)
    monkeypatch.setattr(loaded_inference.model, "predict", lambda x, deterministic=True: (np.array([0, 1, 2, 3, 4]), None))

    results = loaded_inference.predict_observations(obs_batch)
    assert len(results) == 5

    expected_actions = [Action.HOLD, Action.OPEN_LONG, Action.CLOSE_LONG, Action.OPEN_SHORT, Action.CLOSE_SHORT]
    for i, exp_act in enumerate(expected_actions):
        res = results[i]
        assert isinstance(res.action, Action)
        assert res.action == exp_act
        assert res.action.value == exp_act.value
        assert res.q_values[exp_act.name] == pytest.approx(0.9)
        assert res.q_margin == pytest.approx(0.8)


def test_batch_prediction_multiple_tied_q_rows(loaded_inference, monkeypatch):
    """Confirm multiple rows with tied highest Q-values yield lowest-index action and q_margin == 0.0."""
    obs_batch = np.zeros((3, 26), dtype=np.float32)

    mock_q = torch.tensor([
        [0.5, 0.5, 0.1, 0.1, 0.1],   # Tie between 0 and 1 -> argmax selects 0
        [0.1, 0.2, 0.8, 0.8, 0.1],   # Tie between 2 and 3 -> argmax selects 2
        [0.3, 0.3, 0.3, 0.1, 0.1],   # 3-way tie (0, 1, 2) -> argmax selects 0
    ], dtype=torch.float32)

    monkeypatch.setattr(loaded_inference.model.policy.q_net, "forward", lambda x: mock_q)
    monkeypatch.setattr(loaded_inference.model, "predict", lambda x, deterministic=True: (np.array([0, 2, 0]), None))

    results = loaded_inference.predict_observations(obs_batch)
    assert len(results) == 3

    assert results[0].action == Action.HOLD
    assert results[0].q_margin == 0.0
    assert results[0].q_values["HOLD"] == pytest.approx(0.5)

    assert results[1].action == Action.CLOSE_LONG
    assert results[1].q_margin == 0.0
    assert results[1].q_values["CLOSE_LONG"] == pytest.approx(0.8)

    assert results[2].action == Action.HOLD
    assert results[2].q_margin == 0.0
    assert results[2].q_values["HOLD"] == pytest.approx(0.3)


def test_batch_prediction_shape_and_empty_rejections(loaded_inference):
    # Empty batch
    with pytest.raises(InvalidObservationError, match="observations batch cannot be empty"):
        loaded_inference.predict_observations(np.zeros((0, 26), dtype=np.float32))

    # 1D array passed to predict_observations
    with pytest.raises(InvalidObservationError, match=r"got 1D array of shape \(26,\). Use predict_observation\(\)"):
        loaded_inference.predict_observations(np.zeros(26, dtype=np.float32))

    # 3D array
    with pytest.raises(InvalidObservationError, match="got 3D array"):
        loaded_inference.predict_observations(np.zeros((2, 5, 26), dtype=np.float32))

    # Wrong feature count (N, 25)
    with pytest.raises(InvalidObservationError, match="observations feature count 25 != expected 26"):
        loaded_inference.predict_observations(np.zeros((10, 25), dtype=np.float32))

    # Non-array input
    with pytest.raises(InvalidObservationError, match="observations must be a numpy ndarray"):
        loaded_inference.predict_observations([[0.0] * 26])

    with pytest.raises(InvalidObservationError, match="observations must be a numpy ndarray"):
        loaded_inference.predict_observations(True)


def test_batch_prediction_wrong_dtype_rejection(loaded_inference):
    with pytest.raises(InvalidObservationError, match="observations dtype must be float32, got float64"):
        loaded_inference.predict_observations(np.zeros((5, 26), dtype=np.float64))

    with pytest.raises(InvalidObservationError, match="observations dtype must be float32, got int32"):
        loaded_inference.predict_observations(np.zeros((5, 26), dtype=np.int32))


def test_batch_prediction_nan_inf_rejections(loaded_inference):
    # NaN in first row
    obs_nan_0 = np.zeros((5, 26), dtype=np.float32)
    obs_nan_0[0, 3] = np.nan
    with pytest.raises(InvalidObservationError, match="observations contains non-finite values"):
        loaded_inference.predict_observations(obs_nan_0)

    # NaN in middle row
    obs_nan_mid = np.zeros((5, 26), dtype=np.float32)
    obs_nan_mid[2, 10] = np.nan
    with pytest.raises(InvalidObservationError, match="observations contains non-finite values"):
        loaded_inference.predict_observations(obs_nan_mid)

    # Inf in last row
    obs_inf_last = np.zeros((5, 26), dtype=np.float32)
    obs_inf_last[4, 25] = np.inf
    with pytest.raises(InvalidObservationError, match="observations contains non-finite values"):
        loaded_inference.predict_observations(obs_inf_last)


def test_batch_prediction_size_limit_rejections(loaded_inference):
    # Exactly 10,000 observations (MAX_BATCH_SIZE) succeeds
    obs_10k = np.zeros((MAX_BATCH_SIZE, 26), dtype=np.float32)
    res_10k = loaded_inference.predict_observations(obs_10k)
    assert len(res_10k) == MAX_BATCH_SIZE

    # 10,001 observations rejected
    obs_10k_plus = np.zeros((MAX_BATCH_SIZE + 1, 26), dtype=np.float32)
    with pytest.raises(InvalidObservationError, match=f"observations batch size 10001 exceeds maximum allowed {MAX_BATCH_SIZE}"):
        loaded_inference.predict_observations(obs_10k_plus)


def test_read_only_and_non_contiguous_arrays_safe(loaded_inference):
    """Ensure read-only and non-contiguous arrays run safely with zero PyTorch warnings and no mutations."""
    # Read-only 1D
    obs_1d = np.zeros(26, dtype=np.float32)
    obs_1d.flags.writeable = False
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        res_1d = loaded_inference.predict_observation(obs_1d)
    assert isinstance(res_1d, PredictionResult)
    # Confirm no PyTorch UserWarning about non-writable arrays was emitted
    pytorch_warnings = [w for w in recorded if "not writable" in str(w.message)]
    assert len(pytorch_warnings) == 0

    # Read-only batch
    obs_batch = np.zeros((10, 26), dtype=np.float32)
    obs_batch.flags.writeable = False
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        res_batch = loaded_inference.predict_observations(obs_batch)
    assert len(res_batch) == 10
    pytorch_warnings = [w for w in recorded if "not writable" in str(w.message)]
    assert len(pytorch_warnings) == 0

    # Non-contiguous batch (strided slice)
    large_obs = np.zeros((10, 52), dtype=np.float32)
    non_contig = large_obs[:, ::2]  # shape (10, 26), non-contiguous
    assert not non_contig.flags.c_contiguous
    res_non_contig = loaded_inference.predict_observations(non_contig)
    assert len(res_non_contig) == 10


def test_caller_array_immutability(loaded_inference):
    """Verify inference never mutates caller-provided NumPy arrays."""
    # 1D observation
    obs_1d = np.array([float(i) for i in range(26)], dtype=np.float32)
    obs_1d_copy = obs_1d.copy()
    loaded_inference.predict_observation(obs_1d)
    np.testing.assert_array_equal(obs_1d, obs_1d_copy)

    # Batch observations
    obs_batch = np.arange(130, dtype=np.float32).reshape(5, 26)
    obs_batch_copy = obs_batch.copy()
    loaded_inference.predict_observations(obs_batch)
    np.testing.assert_array_equal(obs_batch, obs_batch_copy)


def test_falsy_candle_closure_representations_rejected(loaded_inference):
    """Verify falsy closure representations (False, 0, None) are all strictly rejected."""
    account = AccountSnapshot(equity=10_000.0, available_balance=10_000.0)

    # is_closed = False
    c_false = _make_valid_candles(50)
    c_false[-1]["is_closed"] = False
    with pytest.raises(InvalidObservationError, match="Candle at index 49 is not closed"):
        loaded_inference.build_observation(c_false, account)

    # is_closed = 0
    c_zero = _make_valid_candles(50)
    c_zero[-1]["is_closed"] = 0
    with pytest.raises(InvalidObservationError, match="Candle at index 49 is not closed"):
        loaded_inference.build_observation(c_zero, account)

    # is_closed = None
    c_none = _make_valid_candles(50)
    c_none[-1]["is_closed"] = None
    with pytest.raises(InvalidObservationError, match="Candle at index 49 is not closed"):
        loaded_inference.build_observation(c_none, account)

    # is_closed = True succeeds
    c_true = _make_valid_candles(50)
    c_true[-1]["is_closed"] = True
    obs = loaded_inference.build_observation(c_true, account)
    assert obs.shape == (26,)


def test_package_root_exports_and_lazy_model_inference():
    """Verify Stage 3 exports at noone.rl_engine package root."""
    import noone.rl_engine as rl

    # Verify Stage 3 symbols in __all__
    assert "Action" in rl.__all__
    assert "PredictionResult" in rl.__all__
    assert "AccountSnapshot" in rl.__all__
    assert "MAX_BATCH_SIZE" in rl.__all__
    assert "ModelInference" in rl.__all__
    assert "ModelInferenceError" in rl.__all__
    assert "ModelNotFoundError" in rl.__all__
    assert "CorruptModelError" in rl.__all__
    assert "IncompatibleModelError" in rl.__all__
    assert "InvalidObservationError" in rl.__all__
    assert "InsufficientHistoryError" in rl.__all__
    assert "InvalidAccountStateError" in rl.__all__

    # Direct imports from package root
    from noone.rl_engine import (
        Action,
        AccountSnapshot,
        PredictionResult,
        MAX_BATCH_SIZE,
        ModelInference,
        ModelInferenceError,
        ModelNotFoundError,
        CorruptModelError,
        IncompatibleModelError,
        InvalidObservationError,
        InsufficientHistoryError,
        InvalidAccountStateError,
    )
    assert Action.HOLD == 0
    assert MAX_BATCH_SIZE == 10_000
    assert ModelInference is not None


def test_package_root_lazy_import_missing_dependency(monkeypatch):
    """Verify requesting ModelInference raises ModelInferenceError if PyTorch or SB3 is missing."""
    import noone.rl_engine as rl

    # Simulate missing torch when accessing ModelInference via __getattr__
    real_import = __import__
    def mock_import(name, *args, **kwargs):
        if name in ("torch", "stable_baselines3"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    with pytest.raises(ModelInferenceError, match="PyTorch and Stable-Baselines3 are required for ModelInference"):
        rl.__getattr__("ModelInference")


# --- 9. Clean-Process Import Tests & Batch Prediction Error-Path Tests ---

def test_clean_process_package_import_leaves_torch_and_sb3_absent():
    """Verify in an isolated Python process that importing noone.rl_engine does not load torch or sb3."""
    code = (
        "import sys\n"
        "import noone.rl_engine\n"
        "assert 'torch' not in sys.modules, f'torch loaded: {sys.modules.get(\"torch\")}'\n"
        "assert 'stable_baselines3' not in sys.modules, f'sb3 loaded: {sys.modules.get(\"stable_baselines3\")}'\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"


def test_clean_process_stage3_data_contracts_leave_torch_and_sb3_absent():
    """Verify importing Stage 3 data contracts does not load torch or sb3."""
    code = (
        "import sys\n"
        "from noone.rl_engine import (\n"
        "    Action,\n"
        "    PredictionResult,\n"
        "    AccountSnapshot,\n"
        "    MAX_BATCH_SIZE,\n"
        "    ModelInferenceError,\n"
        "    ModelNotFoundError,\n"
        "    CorruptModelError,\n"
        "    IncompatibleModelError,\n"
        "    InvalidObservationError,\n"
        "    InsufficientHistoryError,\n"
        "    InvalidAccountStateError,\n"
        ")\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'stable_baselines3' not in sys.modules\n"
        "assert Action.HOLD == 0\n"
        "assert MAX_BATCH_SIZE == 10_000\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"


def test_clean_process_stage1_and_2_symbols_functional_without_dependencies():
    """Verify Stage 1 and Stage 2 symbols import and run without torch or stable_baselines3."""
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "sys.modules['stable_baselines3'] = None\n"
        "from noone.rl_engine import (\n"
        "    FEATURE_NAMES,\n"
        "    OBSERVATION_DIM,\n"
        "    FeatureBuilder,\n"
        "    NewsObservation,\n"
        "    NooneGymEnv,\n"
        "    DataConfig,\n"
        "    DQNConfig,\n"
        "    PipelineConfig,\n"
        "    prepare_candles,\n"
        "    chronological_split,\n"
        ")\n"
        "fb = FeatureBuilder()\n"
        "assert fb is not None\n"
        "assert len(FEATURE_NAMES) == 26\n"
        "assert OBSERVATION_DIM == 26\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"


def test_clean_process_accessing_model_inference_triggers_lazy_loading():
    """Verify accessing ModelInference triggers lazy loading of both torch and sb3."""
    code = (
        "import sys\n"
        "import noone.rl_engine as rl\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'stable_baselines3' not in sys.modules\n"
        "mi = rl.ModelInference\n"
        "assert mi is not None\n"
        "assert 'torch' in sys.modules\n"
        "assert 'stable_baselines3' in sys.modules\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"


def test_clean_process_missing_dependencies_raises_informative_error():
    """Verify accessing ModelInference when dependencies are missing raises ModelInferenceError."""
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "import noone.rl_engine as rl\n"
        "try:\n"
        "    _ = rl.ModelInference\n"
        "    sys.exit(1)\n"
        "except rl.ModelInferenceError as exc:\n"
        "    assert 'PyTorch and Stable-Baselines3 are required for ModelInference' in str(exc)\n"
        "    sys.exit(0)\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"


def test_batch_prediction_non_finite_q_values_raises(loaded_inference, monkeypatch):
    """Verify predict_observations raises ModelInferenceError if Q-network output contains NaN/Inf."""
    obs_batch = np.zeros((3, 26), dtype=np.float32)

    # 1. NaN in row 1
    nan_q = torch.tensor([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [float("nan"), 0.2, 0.3, 0.4, 0.5],
        [0.1, 0.2, 0.3, 0.4, 0.5],
    ], dtype=torch.float32)
    monkeypatch.setattr(loaded_inference.model.policy.q_net, "forward", lambda x: nan_q)
    with pytest.raises(ModelInferenceError, match="Q-network produced non-finite Q-values"):
        loaded_inference.predict_observations(obs_batch)

    # 2. Inf in row 2
    inf_q = torch.tensor([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.1, 0.2, float("inf"), 0.4, 0.5],
    ], dtype=torch.float32)
    monkeypatch.setattr(loaded_inference.model.policy.q_net, "forward", lambda x: inf_q)
    with pytest.raises(ModelInferenceError, match="Q-network produced non-finite Q-values"):
        loaded_inference.predict_observations(obs_batch)


def test_batch_prediction_invalid_output_shape_raises(loaded_inference, monkeypatch):
    """Verify predict_observations raises ModelInferenceError if Q-network output shape is invalid."""
    obs_batch = np.zeros((3, 26), dtype=np.float32)

    # Wrong column count (3, 4) instead of (3, 5)
    bad_col_q = torch.tensor([[0.1, 0.2, 0.3, 0.4]] * 3, dtype=torch.float32)
    monkeypatch.setattr(loaded_inference.model.policy.q_net, "forward", lambda x: bad_col_q)
    with pytest.raises(ModelInferenceError, match=r"Unexpected Q-network output shape:.*!= \(3, 5\)"):
        loaded_inference.predict_observations(obs_batch)

    # 1D output instead of 2D
    bad_dim_q = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
    monkeypatch.setattr(loaded_inference.model.policy.q_net, "forward", lambda x: bad_dim_q)
    with pytest.raises(ModelInferenceError, match=r"Unexpected Q-network output shape"):
        loaded_inference.predict_observations(obs_batch)

    # Wrong row count (2, 5) instead of (3, 5)
    bad_row_q = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]] * 2, dtype=torch.float32)
    monkeypatch.setattr(loaded_inference.model.policy.q_net, "forward", lambda x: bad_row_q)
    with pytest.raises(ModelInferenceError, match=r"Unexpected Q-network output shape:.*!= \(3, 5\)"):
        loaded_inference.predict_observations(obs_batch)


def test_batch_prediction_wrong_number_of_actions_raises(loaded_inference, monkeypatch):
    """Verify predict_observations raises ModelInferenceError if model returns incorrect number of actions."""
    obs_batch = np.zeros((4, 26), dtype=np.float32)
    valid_q = torch.tensor([[0.5, 0.1, 0.1, 0.1, 0.1]] * 4, dtype=torch.float32)
    monkeypatch.setattr(loaded_inference.model.policy.q_net, "forward", lambda x: valid_q)

    # SB3 predict returns 2 actions for a batch of 4
    monkeypatch.setattr(
        loaded_inference.model,
        "predict",
        lambda x, deterministic=True: (np.array([0, 0]), None),
    )
    with pytest.raises(ModelInferenceError, match=r"Model returned 2 actions for batch of 4 observations"):
        loaded_inference.predict_observations(obs_batch)


def test_batch_prediction_action_disagreement_raises(loaded_inference, monkeypatch):
    """Verify predict_observations raises ModelInferenceError if deterministic SB3 action disagrees with Q-net argmax."""
    obs_batch = np.zeros((3, 26), dtype=np.float32)
    # Q-net argmax is 0 for all 3 rows
    valid_q = torch.tensor([[0.5, 0.1, 0.1, 0.1, 0.1]] * 3, dtype=torch.float32)
    monkeypatch.setattr(loaded_inference.model.policy.q_net, "forward", lambda x: valid_q)

    # SB3 predict returns action 1 for row index 1: [0, 1, 0]
    monkeypatch.setattr(
        loaded_inference.model,
        "predict",
        lambda x, deterministic=True: (np.array([0, 1, 0]), None),
    )
    with pytest.raises(
        ModelInferenceError,
        match="Deterministic SB3 actions disagree with Q-network argmax in batch inference",
    ):
        loaded_inference.predict_observations(obs_batch)


def test_batch_prediction_invalid_action_raises(loaded_inference, monkeypatch):
    """Verify predict_observations raises ModelInferenceError if model returns out-of-range or non-integer actions."""
    obs_batch = np.zeros((2, 26), dtype=np.float32)
    valid_q = torch.tensor([[0.5, 0.1, 0.1, 0.1, 0.1]] * 2, dtype=torch.float32)
    monkeypatch.setattr(loaded_inference.model.policy.q_net, "forward", lambda x: valid_q)

    # 1. Action out of range (99)
    monkeypatch.setattr(
        loaded_inference.model,
        "predict",
        lambda x, deterministic=True: (np.array([0, 99]), None),
    )
    with pytest.raises(ModelInferenceError, match=r"Model returned action out of range \(0..4\): 99 at index 1"):
        loaded_inference.predict_observations(obs_batch)

    # 2. Negative action (-1)
    monkeypatch.setattr(
        loaded_inference.model,
        "predict",
        lambda x, deterministic=True: (np.array([-1, 0]), None),
    )
    with pytest.raises(ModelInferenceError, match=r"Model returned action out of range \(0..4\): -1 at index 0"):
        loaded_inference.predict_observations(obs_batch)

    # 3. Non-integer action type (string)
    monkeypatch.setattr(
        loaded_inference.model,
        "predict",
        lambda x, deterministic=True: (np.array(["HOLD", "OPEN_LONG"], dtype=object), None),
    )
    with pytest.raises(ModelInferenceError, match=r"Model returned non-integer action type: str at index 0"):
        loaded_inference.predict_observations(obs_batch)


# --- 9. Action Validity & Masking Tests ---

from noone.rl_engine.inference import get_valid_actions, get_valid_action_mask


def test_action_masking_all_three_position_states():
    """Verify valid action lists and masks for all three position states."""
    # 1. Flat (side 0)
    flat_actions = get_valid_actions(0)
    assert set(flat_actions) == {Action.HOLD, Action.OPEN_LONG, Action.OPEN_SHORT}
    assert get_valid_action_mask(0) == (True, True, False, True, False)
    assert get_valid_actions(None) == flat_actions
    assert get_valid_actions("FLAT") == flat_actions
    assert get_valid_actions("0") == flat_actions
    assert get_valid_actions(AccountSnapshot(equity=1000.0, available_balance=1000.0, position_side=0)) == flat_actions
    assert get_valid_actions({"position_side": 0.0}) == flat_actions

    # 2. Long (side 1)
    long_actions = get_valid_actions(1)
    assert set(long_actions) == {Action.HOLD, Action.CLOSE_LONG}
    assert get_valid_action_mask(1) == (True, False, True, False, False)
    assert get_valid_actions("LONG") == long_actions
    assert get_valid_actions("+1") == long_actions
    assert get_valid_actions(AccountSnapshot(equity=1000.0, available_balance=1000.0, position_side=1, position_quantity=1.0, entry_price=100.0)) == long_actions
    assert get_valid_actions({"position_side": 1}) == long_actions

    # 3. Short (side -1)
    short_actions = get_valid_actions(-1)
    assert set(short_actions) == {Action.HOLD, Action.CLOSE_SHORT}
    assert get_valid_action_mask(-1) == (True, False, False, False, True)
    assert get_valid_actions("SHORT") == short_actions
    assert get_valid_actions("-1") == short_actions
    assert get_valid_actions(AccountSnapshot(equity=1000.0, available_balance=1000.0, position_side=-1, position_quantity=1.0, entry_price=100.0)) == short_actions
    assert get_valid_actions({"position_side": -1}) == short_actions

    # 4. Invalid position side rejections
    with pytest.raises(ValueError, match="Invalid position side"):
        get_valid_actions(2)
    with pytest.raises(ValueError, match="Unknown position state string"):
        get_valid_actions("INVALID_STATE")
    with pytest.raises(TypeError, match="Cannot extract position side"):
        get_valid_actions([1, 2, 3])


def test_action_masking_invalid_predictions_remap_to_valid_action(loaded_inference, monkeypatch):
    """Verify invalid raw predictions remap to highest-Q valid action while preserving raw diagnostics."""
    obs = np.zeros(26, dtype=np.float32)

    # Q-values: HOLD=1.0, OPEN_LONG=2.0, CLOSE_LONG=5.0, OPEN_SHORT=3.0, CLOSE_SHORT=4.0
    # Highest raw action is CLOSE_LONG (action 2) with Q=5.0
    fake_q = torch.tensor([[1.0, 2.0, 5.0, 3.0, 4.0]], dtype=torch.float32)
    monkeypatch.setattr(loaded_inference.model.policy.q_net, "forward", lambda x: fake_q)
    monkeypatch.setattr(loaded_inference.model, "predict", lambda x, deterministic=True: (2, None))

    # Case A: Position is Flat (valid: HOLD=0, OPEN_LONG=1, OPEN_SHORT=3)
    # Raw action CLOSE_LONG (2) is invalid!
    # Best valid action is OPEN_SHORT (3, Q=3.0)
    res_flat = loaded_inference.predict_observation(obs, position_state=0)
    assert res_flat.raw_action == Action.CLOSE_LONG
    assert res_flat.is_valid is False
    assert res_flat.action == Action.OPEN_SHORT
    # Raw Q-values and margin remain untouched
    assert res_flat.q_values["CLOSE_LONG"] == 5.0
    assert res_flat.q_margin == pytest.approx(1.0)  # 5.0 - 4.0

    # Case B: Position is Long (valid: HOLD=0, CLOSE_LONG=2)
    # Raw action CLOSE_LONG (2) is valid!
    res_long = loaded_inference.predict_observation(obs, position_state=1)
    assert res_long.raw_action == Action.CLOSE_LONG
    assert res_long.is_valid is True
    assert res_long.action == Action.CLOSE_LONG

    # Case C: Position is Short (valid: HOLD=0, CLOSE_SHORT=4)
    # Raw action CLOSE_LONG (2) is invalid!
    # Best valid action is CLOSE_SHORT (4, Q=4.0) vs HOLD (0, Q=1.0) -> CLOSE_SHORT
    res_short = loaded_inference.predict_observation(obs, position_state=-1)
    assert res_short.raw_action == Action.CLOSE_LONG
    assert res_short.is_valid is False
    assert res_short.action == Action.CLOSE_SHORT


def test_action_masking_deterministic_tie_breaking(loaded_inference, monkeypatch):
    """Verify deterministic tie-breaking picks the lowest action index among tied valid actions."""
    obs = np.zeros(26, dtype=np.float32)

    # Q-values: HOLD=3.0, OPEN_LONG=3.0, CLOSE_LONG=5.0, OPEN_SHORT=3.0, CLOSE_SHORT=1.0
    # Unmasked raw action: CLOSE_LONG (2, Q=5.0)
    fake_q = torch.tensor([[3.0, 3.0, 5.0, 3.0, 1.0]], dtype=torch.float32)
    monkeypatch.setattr(loaded_inference.model.policy.q_net, "forward", lambda x: fake_q)
    monkeypatch.setattr(loaded_inference.model, "predict", lambda x, deterministic=True: (2, None))

    # Flat state: valid are HOLD (0), OPEN_LONG (1), OPEN_SHORT (3), all tied at Q=3.0
    # Lowest index is 0 (HOLD)
    res = loaded_inference.predict_observation(obs, position_state=0)
    assert res.raw_action == Action.CLOSE_LONG
    assert res.is_valid is False
    assert res.action == Action.HOLD


def test_progress_feature_parity_across_pipeline(loaded_inference):
    """Verify feature 19 (progress) is strictly 0.0 across gym env, evaluation, and inference."""
    # 1. Gym env reset & step
    candles = _make_valid_klines(60)
    from noone.rl_engine.gym_env import NooneGymEnv
    gym_env = NooneGymEnv(candles)
    obs_reset, _ = gym_env.reset()
    assert obs_reset[19] == 0.0

    for act in [0, 1, 0]:
        obs_step, _, _, _, _ = gym_env.step(act)
        assert obs_step[19] == 0.0

    # 2. ModelInference build_observation
    account = AccountSnapshot(equity=10_000.0, available_balance=10_000.0)
    obs_inf = loaded_inference.build_observation(candles, account)
    assert obs_inf[19] == 0.0


def test_batch_prediction_action_masking(loaded_inference, monkeypatch):
    """Verify predict_observations applies action masking row-by-row."""
    obs_batch = np.zeros((3, 26), dtype=np.float32)

    # All rows predict CLOSE_LONG (action 2)
    fake_q = torch.tensor([[1.0, 2.0, 5.0, 3.0, 4.0]] * 3, dtype=torch.float32)
    monkeypatch.setattr(loaded_inference.model.policy.q_net, "forward", lambda x: fake_q)
    monkeypatch.setattr(loaded_inference.model, "predict", lambda x, deterministic=True: (np.array([2, 2, 2]), None))

    # Mask row 0 as Flat, row 1 as Long, row 2 as Short
    results = loaded_inference.predict_observations(obs_batch, position_states=[0, 1, -1])
    assert len(results) == 3

    # Row 0: Flat -> OPEN_SHORT (3)
    assert results[0].raw_action == Action.CLOSE_LONG
    assert results[0].is_valid is False
    assert results[0].action == Action.OPEN_SHORT

    # Row 1: Long -> CLOSE_LONG (2)
    assert results[1].raw_action == Action.CLOSE_LONG
    assert results[1].is_valid is True
    assert results[1].action == Action.CLOSE_LONG

    # Row 2: Short -> CLOSE_SHORT (4)
    assert results[2].raw_action == Action.CLOSE_LONG
    assert results[2].is_valid is False
    assert results[2].action == Action.CLOSE_SHORT


def test_cpu_only_execution_guarantee(loaded_inference):
    """Verify model execution stays strictly on CPU."""
    assert loaded_inference.device == "cpu"
    for param in loaded_inference.model.policy.parameters():
        assert param.device.type == "cpu"




