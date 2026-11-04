"""Component 1: DQN + RL Engine.

Stage 1: Gymnasium adapter around Component 8 plus the shared
FeatureBuilder. Stage 2: historical data preparation, point-in-time
news variation, SB3 DQN training (checkpoints/resume/metadata) and
deterministic held-out evaluation. Stage 3: Model inference, data
contracts, and compatibility validation. Training/evaluation entry
points and ModelInference import Stable-Baselines3 and PyTorch lazily
so Stage 1 use needs no RL extra. No live inference loops, paper
trading, or order execution.
"""
from typing import Any

from noone.rl_engine.config import DataConfig, DQNConfig, PipelineConfig
from noone.rl_engine.data import (
    ChronologicalSplit,
    DataPreparationError,
    PreparedData,
    chronological_split,
    prepare_candles,
)
from noone.rl_engine.features import (
    FEATURE_NAMES,
    OBSERVATION_DIM,
    FeatureBuilder,
    NewsObservation,
)
from noone.rl_engine.gym_env import NooneGymEnv
from noone.rl_engine.inference import (
    Action,
    AccountSnapshot,
    CorruptModelError,
    IncompatibleModelError,
    InsufficientHistoryError,
    InvalidAccountStateError,
    InvalidObservationError,
    MAX_BATCH_SIZE,
    ModelInferenceError,
    ModelNotFoundError,
    PredictionResult,
    get_valid_actions,
    get_valid_action_mask,
)
from noone.rl_engine.news import NewsDropoutArchive, news_observation_at


def __getattr__(name: str) -> Any:
    if name == "ModelInference":
        try:
            import torch
            import stable_baselines3
            from noone.rl_engine.inference import ModelInference
            return ModelInference
        except ImportError as exc:
            raise ModelInferenceError(
                'PyTorch and Stable-Baselines3 are required for ModelInference. '
                'Install the RL extra: pip install -e ".[rl]"'
            ) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Stage 1
    "FEATURE_NAMES",
    "OBSERVATION_DIM",
    "FeatureBuilder",
    "NewsObservation",
    "NooneGymEnv",
    # Stage 2
    "DataConfig",
    "DQNConfig",
    "PipelineConfig",
    "DataPreparationError",
    "PreparedData",
    "ChronologicalSplit",
    "prepare_candles",
    "chronological_split",
    "NewsDropoutArchive",
    "news_observation_at",
    # Stage 3
    "Action",
    "PredictionResult",
    "AccountSnapshot",
    "MAX_BATCH_SIZE",
    "ModelInference",
    "ModelInferenceError",
    "ModelNotFoundError",
    "CorruptModelError",
    "IncompatibleModelError",
    "InvalidObservationError",
    "InsufficientHistoryError",
    "InvalidAccountStateError",
    "get_valid_actions",
    "get_valid_action_mask",
]
