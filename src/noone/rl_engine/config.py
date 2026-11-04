"""Stage 2 pipeline configuration (Component 1).

All values are plain data so a run can be reproduced from its saved
``config.json`` without code edits. CPU-only by design: ``device``
must remain ``"cpu"`` (no CUDA requirement anywhere in Stage 2).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class DQNConfig:
    """SB3 DQN hyper-parameters (initial CPU-friendly defaults)."""

    policy: str = "MlpPolicy"
    net_arch: tuple[int, ...] = (128, 128)
    buffer_size: int = 100_000
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    train_freq: int = 4
    target_update_interval: int = 1_000
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    exploration_fraction: float = 0.25
    learning_starts: int = 2_000
    total_timesteps: int = 250_000
    seed: int = 42
    device: str = "cpu"
    checkpoint_freq: int = 10_000
    verbose: int = 0
    invalid_action_penalty: float = 0.001

    def __post_init__(self) -> None:
        if isinstance(self.invalid_action_penalty, bool) or not isinstance(self.invalid_action_penalty, (int, float)):
            raise ValueError("invalid_action_penalty must be numeric")
        if not math.isfinite(self.invalid_action_penalty) or self.invalid_action_penalty < 0.0:
            raise ValueError("invalid_action_penalty must be finite and non-negative")
        if self.device != "cpu":
            raise ValueError("Stage 2 is CPU-only: device must be 'cpu'")
        if self.policy != "MlpPolicy":
            raise ValueError("Only MlpPolicy is supported in Stage 2")
        if not self.net_arch or any(n < 1 for n in self.net_arch):
            raise ValueError("net_arch must be non-empty positive widths")
        if self.buffer_size < 1 or self.batch_size < 1:
            raise ValueError("buffer_size/batch_size must be positive")
        if not 0.0 < self.learning_rate:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.train_freq < 1 or self.target_update_interval < 1:
            raise ValueError("train_freq/target_update_interval must be >= 1")
        if not 0.0 <= self.exploration_final_eps <= self.exploration_initial_eps <= 1.0:
            raise ValueError("exploration eps must satisfy 0 <= final <= initial <= 1")
        if not 0.0 < self.exploration_fraction <= 1.0:
            raise ValueError("exploration_fraction must be in (0, 1]")
        if self.learning_starts < 0 or self.total_timesteps < 1:
            raise ValueError("learning_starts >= 0 and total_timesteps >= 1 required")
        if self.checkpoint_freq < 1:
            raise ValueError("checkpoint_freq must be >= 1")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["net_arch"] = list(self.net_arch)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "DQNConfig":
        values = dict(data)
        if "net_arch" in values:
            values["net_arch"] = tuple(int(n) for n in values["net_arch"])
        return cls(**values)


@dataclass(frozen=True)
class DataConfig:
    """Historical Futures data + split + news-variation settings."""

    symbol: str = "BTCUSDT"
    interval: str = "15m"
    total_limit: int = 5_000
    train_fraction: float = 0.8
    warmup_candles: int = 50
    min_train_candles: int = 100
    min_eval_candles: int = 50
    episode_window: int | None = None
    news_max_age_hours: float = 24.0
    # Conservative dropout: 10% of news queries return no articles,
    # so the agent regularly practises operating market-only.
    news_dropout_probability: float = 0.1
    news_asset: str | None = None  # None -> derived from symbol

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol is required")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        if self.warmup_candles < 0:
            raise ValueError("warmup_candles must be >= 0")
        if self.min_train_candles < 2 or self.min_eval_candles < 2:
            raise ValueError("min train/eval candles must be >= 2")
        if not 0.0 <= self.news_dropout_probability <= 1.0:
            raise ValueError("news_dropout_probability must be in [0, 1]")
        if self.news_max_age_hours <= 0:
            raise ValueError("news_max_age_hours must be positive")
        if self.episode_window is not None and self.episode_window < 2:
            raise ValueError("episode_window must be >= 2 when set")

    @property
    def resolved_news_asset(self) -> str:
        if self.news_asset:
            return self.news_asset.upper()
        symbol = self.symbol.upper()
        for quote in ("USDT", "USDC", "BUSD", "USD"):
            if symbol.endswith(quote):
                return symbol[: -len(quote)]
        return symbol

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DataConfig":
        return cls(**dict(data))


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level Stage 2 configuration saved with every run."""

    dqn: DQNConfig = field(default_factory=DQNConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = "runs"
    run_name: str | None = None

    def to_dict(self) -> dict:
        return {
            "dqn": self.dqn.to_dict(),
            "data": self.data.to_dict(),
            "output_dir": self.output_dir,
            "run_name": self.run_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        return cls(
            dqn=DQNConfig.from_dict(data.get("dqn", {})),
            data=DataConfig.from_dict(data.get("data", {})),
            output_dir=data.get("output_dir", "runs"),
            run_name=data.get("run_name"),
        )
