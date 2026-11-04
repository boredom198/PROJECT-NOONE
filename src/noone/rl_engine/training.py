"""SB3 DQN training workflow for Stage 2 (Component 1).

CPU-only. Each run gets its own output directory (never overwritten
by default) containing checkpoints, the final model, ``config.json``
and ``metadata.json`` (data range, split boundaries, library
versions, news-dropout settings) needed to reproduce or resume it.
Resume validates that saved metadata exists and matches the current
observation/action spaces and DQN architecture, failing with a
clear error otherwise.
"""
from __future__ import annotations

import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from noone.market_data.models import Kline
from noone.rl_engine.config import DataConfig, DQNConfig, PipelineConfig
from noone.rl_engine.data import (
    ChronologicalSplit,
    PreparedData,
    chronological_split,
    prepare_candles,
)
from noone.rl_engine.features import (
    FEATURE_SCHEMA_VERSION,
    FeatureBuilder,
    OBSERVATION_DIM,
    compute_feature_schema_hash,
)
from noone.rl_engine.gym_env import NooneGymEnv
from noone.rl_engine.news import NewsDropoutArchive
from noone.trading_env import TradingConfig

DISCLAIMER = ("Historical backtest/training results do not guarantee "
              "future performance. This is a research pipeline, not a "
              "profitability guarantee.")


def _require_sb3():
    try:
        import stable_baselines3
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import CheckpointCallback
    except ImportError as exc:  # pragma: no cover - exercised via extra
        raise ImportError(
            "Stable-Baselines3 is required for Stage 2 training. "
            'Install the RL extra: pip install -e ".[rl]" '
            "(plus CPU-only torch, see README).") from exc
    return stable_baselines3, DQN, CheckpointCallback


def make_env(
    candles: Sequence[Kline], *, trading_config: TradingConfig | None = None,
    data_config: DataConfig | None = None, news_archive: Any = None,
    episode_window: int | None = None, seed: int = 42,
    funding_history: Any = None,
    invalid_action_penalty: float = 0.001,
) -> NooneGymEnv:
    """Build the Stage 1 Gymnasium env for a candle slice."""
    data_config = data_config or DataConfig()
    archive = news_archive
    if archive is not None and not isinstance(archive, NewsDropoutArchive):
        archive = NewsDropoutArchive(
            archive, dropout_probability=data_config.news_dropout_probability,
            seed=seed)
    return NooneGymEnv(
        list(candles), trading_config,
        episode_window=(episode_window if episode_window is not None
                        else data_config.episode_window),
        news_archive=archive, news_asset=data_config.resolved_news_asset,
        feature_builder=FeatureBuilder(
            news_max_age_hours=data_config.news_max_age_hours),
        funding_history=funding_history,
        invalid_action_penalty=invalid_action_penalty)


def _funding_metadata(
    trading_config: TradingConfig | None, funding_history: Any,
    candles: Sequence[Kline], env: NooneGymEnv | None = None,
    artifact_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Funding provenance for run metadata (Stage 3, Component 8)."""
    config = trading_config or TradingConfig()
    section: dict[str, Any] = {
        "funding_mode": config.funding_mode,
        "historical_accounting_implemented":
            config.funding_mode == "historical",
        "funding_config": {
            "funding_mode": config.funding_mode,
            "funding_rate": config.funding_rate,
            "funding_times_utc": list(config.funding_times_utc),
        },
        "funding_history": None,
        "validation_status": "not_applicable_constant_mode",
        "training_episode_count": 0,
        "episode_funding_statistics": None,
        "artifact_path": None,
        "artifact_schema_version": None,
        "artifact_checksum": None,
        "collection_identity": None,
    }
    if artifact_info is not None:
        section["artifact_path"] = artifact_info.get("artifact_path")
        section["artifact_schema_version"] = artifact_info.get(
            "artifact_schema_version")
        section["artifact_checksum"] = artifact_info.get("artifact_checksum")
        section["collection_identity"] = artifact_info.get(
            "collection_identity")
    if funding_history is not None:
        start = candles[0].open_time if candles else None
        last = candles[-1] if candles else None
        end = (last.close_time or last.open_time) if last else None
        section["funding_history"] = funding_history.metadata()
        if start is not None and end is not None:
            section["validation_status"] = (
                "strict:"
                + funding_history.completeness_status_for_period(start, end))
    if env is not None:
        stats = env.episode_funding_statistics()
        section["training_episode_count"] = stats["completed_episode_count"]
        section["episode_funding_statistics"] = stats
    return section


def build_dqn(env: NooneGymEnv, config: DQNConfig):
    """Create an SB3 DQN with the configured CPU-friendly parameters."""
    _sb3, dqn_cls, _cb = _require_sb3()
    return dqn_cls(
        config.policy, env,
        policy_kwargs={"net_arch": list(config.net_arch)},
        buffer_size=config.buffer_size, batch_size=config.batch_size,
        learning_rate=config.learning_rate, gamma=config.gamma,
        train_freq=config.train_freq,
        target_update_interval=config.target_update_interval,
        exploration_initial_eps=config.exploration_initial_eps,
        exploration_final_eps=config.exploration_final_eps,
        exploration_fraction=config.exploration_fraction,
        learning_starts=config.learning_starts,
        seed=config.seed, device=config.device, verbose=config.verbose)


def _make_run_dir(output_dir: str | Path, run_name: str | None) -> Path:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    name = run_name or datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S")
    candidate, suffix = base / name, 1
    while candidate.exists():
        suffix += 1
        candidate = base / f"{name}-{suffix:03d}"
    candidate.mkdir(parents=True)
    return candidate


def _versions() -> dict[str, str]:
    import gymnasium
    import torch

    sb3, _dqn, _cb = _require_sb3()
    return {"python": platform.python_version(), "stable_baselines3": sb3.__version__,
            "gymnasium": gymnasium.__version__, "torch": torch.__version__,
            "numpy": np.__version__}


def _find_resume_metadata(checkpoint: Path) -> dict | None:
    for directory in (checkpoint.parent, checkpoint.parent.parent):
        meta = directory / "metadata.json"
        if meta.exists():
            return json.loads(meta.read_text(encoding="utf-8"))
    return None


def validate_resume(
    checkpoint_path: str | Path, env: NooneGymEnv, config: DQNConfig,
) -> dict:
    """Validate checkpoint + metadata compatibility; clear errors."""
    checkpoint = Path(checkpoint_path)
    zip_path = checkpoint if checkpoint.suffix == ".zip" else checkpoint.with_suffix(".zip")
    if not zip_path.exists() and not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    metadata = _find_resume_metadata(checkpoint)
    if metadata is None:
        raise ValueError(
            f"Cannot resume from {checkpoint_path}: no metadata.json found "
            "beside the checkpoint or in its run directory. Resume requires "
            "the original run metadata to verify compatibility.")
    if int(metadata.get("observation_dim", -1)) != OBSERVATION_DIM:
        raise ValueError(
            f"Incompatible checkpoint: metadata observation_dim="
            f"{metadata.get('observation_dim')} != current {OBSERVATION_DIM}")
    if int(metadata.get("action_count", -1)) != int(env.action_space.n):
        raise ValueError(
            f"Incompatible checkpoint: metadata action_count="
            f"{metadata.get('action_count')} != current {env.action_space.n}")
    saved_dqn = (metadata.get("config") or {}).get("dqn") or {}
    if saved_dqn:
        saved = DQNConfig.from_dict(saved_dqn)
        if list(saved.net_arch) != list(config.net_arch) or saved.policy != config.policy:
            raise ValueError(
                f"Incompatible checkpoint: saved policy/net_arch "
                f"{saved.policy}/{list(saved.net_arch)} != requested "
                f"{config.policy}/{list(config.net_arch)}")
    saved_schema_hash = metadata.get("feature_schema_hash")
    if saved_schema_hash is not None:
        news_age = getattr(env.features, "news_max_age_seconds", 86400.0) / 3600.0
        expected_hash = compute_feature_schema_hash(news_max_age_hours=news_age)
        if saved_schema_hash != expected_hash:
            raise ValueError(
                f"Incompatible checkpoint: metadata feature_schema_hash="
                f"{saved_schema_hash!r} != current {expected_hash!r}")
    _validate_resume_funding(metadata, env)
    return metadata


def _validate_resume_funding(metadata: dict, env: NooneGymEnv) -> None:
    """Reject funding-incompatible resumes (Stage 3, Component 8).

    Legacy checkpoints without funding metadata may resume only in
    constant mode (the only mode reachable through this adapter
    before Stage 3); historical resumes from them are rejected
    explicitly rather than assumed compatible.
    """
    current_config = env._config or TradingConfig()
    current_history = env._funding_history
    current_mode = current_config.funding_mode
    saved = metadata.get("funding")
    if not isinstance(saved, dict) or "funding_mode" not in saved:
        if current_mode == "historical":
            raise ValueError(
                "Incompatible checkpoint: legacy checkpoint without "
                "funding metadata cannot be resumed in historical "
                "funding mode; funding provenance is unknown")
        metadata["funding_resume_status"] = (
            "legacy_checkpoint_no_funding_metadata_constant_mode")
        return
    saved_mode = saved.get("funding_mode")
    if saved_mode != current_mode:
        raise ValueError(
            f"Incompatible checkpoint: saved funding_mode={saved_mode!r} "
            f"!= requested funding_mode={current_mode!r}")
    if saved_mode == "historical":
        if current_history is None:
            raise ValueError(
                "Incompatible checkpoint: saved funding_mode is "
                "'historical' but no FundingHistory was supplied")
        saved_history = saved.get("funding_history") or {}
        current_meta = current_history.metadata()
        for field_name, label in (
                ("symbol", "historical symbol"),
                ("requested_start_time", "requested history range"),
                ("requested_end_time", "requested history range"),
                ("settlement_count", "settlement count"),
                ("collected_at", "collection identity (collected_at)")):
            if saved_history.get(field_name) != current_meta.get(field_name):
                raise ValueError(
                    f"Incompatible checkpoint: saved {label} "
                    f"{saved_history.get(field_name)!r} != requested "
                    f"{current_meta.get(field_name)!r}")
    metadata["funding_resume_status"] = "funding_metadata_matched"


def train_dqn(
    train_candles: Sequence[Kline], *, config: PipelineConfig | None = None,
    trading_config: TradingConfig | None = None, news_archive: Any = None,
    output_dir: str | Path | None = None, run_name: str | None = None,
    resume_from: str | Path | None = None,
    prepared: PreparedData | None = None,
    split: ChronologicalSplit | None = None,
    funding_history: Any = None,
    callbacks: Sequence[Any] | None = None,
    exploration_schedule: Any | None = None,
) -> dict[str, Any]:
    """Train (or resume) DQN on ``train_candles`` only. Returns run info."""
    _sb3, dqn_cls, checkpoint_cb_cls = _require_sb3()
    import torch

    config = config or PipelineConfig()
    dqn_config, data_config = config.dqn, config.data
    random.seed(dqn_config.seed)
    np.random.seed(dqn_config.seed)
    torch.manual_seed(dqn_config.seed)

    run_dir = _make_run_dir(output_dir or config.output_dir, run_name or config.run_name)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    # Persist the exact FundingHistory before training begins so a
    # later process can resume with the original provenance.
    artifact_info = None
    if funding_history is not None:
        from noone.rl_engine.data import save_funding_history

        artifact_info = save_funding_history(
            funding_history, run_dir / "funding_history.json")

    env = make_env(train_candles, trading_config=trading_config,
                   data_config=data_config, news_archive=news_archive,
                   seed=dqn_config.seed, funding_history=funding_history,
                   invalid_action_penalty=dqn_config.invalid_action_penalty)
    if resume_from is not None:
        validate_resume(resume_from, env, dqn_config)
        model = dqn_cls.load(str(resume_from), env=env, device=dqn_config.device)
    else:
        model = build_dqn(env, dqn_config)

    schema_hash = compute_feature_schema_hash(news_max_age_hours=data_config.news_max_age_hours)
    metadata: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": DISCLAIMER,
        "config": config.to_dict(),
        "observation_dim": OBSERVATION_DIM,
        "action_count": int(env.action_space.n),
        "feature_schema_hash": schema_hash,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "seed": dqn_config.seed, "device": dqn_config.device,
        "versions": _versions(),
        "total_timesteps_requested": dqn_config.total_timesteps,
        "resume_from": str(resume_from) if resume_from else None,
        "news": {
            "dropout_probability": data_config.news_dropout_probability,
            "news_asset": data_config.resolved_news_asset,
            "news_max_age_hours": data_config.news_max_age_hours,
            "archive_supplied": news_archive is not None,
        },
        "data": prepared.metadata() if prepared else {
            "candle_count": len(train_candles),
            "start_time": train_candles[0].open_time.isoformat() if train_candles else None,
            "end_time": train_candles[-1].close_time.isoformat() if train_candles else None,
        },
        "split": split.metadata() if split else None,
        "funding": _funding_metadata(
            trading_config, funding_history, train_candles,
            artifact_info=artifact_info),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    metadata["funding"] = _funding_metadata(
        trading_config, funding_history, train_candles, env=env,
        artifact_info=artifact_info)
    if exploration_schedule is not None:
        model.exploration_schedule = exploration_schedule

    callback = checkpoint_cb_cls(
        save_freq=max(1, dqn_config.checkpoint_freq),
        save_path=str(run_dir / "checkpoints"), name_prefix="dqn",
        save_replay_buffer=False, save_vecnormalize=False, verbose=0)
    if callbacks:
        from stable_baselines3.common.callbacks import CallbackList
        learn_callback = CallbackList([callback] + list(callbacks))
    else:
        learn_callback = callback

    model.learn(total_timesteps=dqn_config.total_timesteps,
                callback=learn_callback, reset_num_timesteps=resume_from is None)
    final_path = run_dir / "final_model"
    model.save(str(final_path))
    metadata["funding"] = _funding_metadata(
        trading_config, funding_history, train_candles, env=env,
        artifact_info=artifact_info)
    metadata["final_model"] = str(final_path.with_suffix(".zip"))
    metadata["num_timesteps_trained"] = int(model.num_timesteps)
    archive = env._news_archive
    if isinstance(archive, NewsDropoutArchive):
        metadata["news"]["dropout_queries"] = archive.query_count
        metadata["news"]["dropout_dropped"] = archive.dropout_count
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return {"run_dir": str(run_dir), "final_model": metadata["final_model"],
            "metadata": metadata, "model": model}


def run_training(
    candles: Sequence[Kline], *, config: PipelineConfig | None = None,
    trading_config: TradingConfig | None = None, news_archive: Any = None,
    resume_from: str | Path | None = None,
    funding_history: Any = None,
) -> dict[str, Any]:
    """Full path: prepare -> strict split -> train on train slice only."""
    config = config or PipelineConfig()
    data_config = config.data
    prepared = prepare_candles(
        candles, interval=data_config.interval, symbol=data_config.symbol,
        warmup_candles=data_config.warmup_candles,
        min_candles=max(data_config.min_train_candles, data_config.min_eval_candles))
    split = chronological_split(
        prepared, train_fraction=data_config.train_fraction,
        warmup_candles=data_config.warmup_candles,
        min_train_candles=data_config.min_train_candles,
        min_eval_candles=data_config.min_eval_candles)
    return train_dqn(
        split.train, config=config, trading_config=trading_config,
        news_archive=news_archive, resume_from=resume_from,
        prepared=prepared, split=split, funding_history=funding_history)
