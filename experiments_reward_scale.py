"""Reward Scale and Action Masking Experiments for PROJECT NOONE Component 1.

Compares:
- Condition A: Reward scale 1.0 (seeds 42, 43, 44)
- Condition B: Reward scale 100.0 (seeds 42, 43, 44)
- Condition C: Reward scale 1000.0 (seeds 42, 43, 44)
- Condition D (Decision C): Penalty 0.0 vs Penalty 0.001 (raw vs masked actions, invalid prediction reduction)
"""
from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from noone.market_data.models import Kline
from noone.rl_engine import Action, ModelInference, get_valid_action_mask
from noone.rl_engine.config import DataConfig, DQNConfig, PipelineConfig
from noone.rl_engine.data import chronological_split, prepare_candles
from noone.rl_engine.evaluation import evaluate_model
from noone.rl_engine.training import train_dqn
from noone.trading_env import TradingConfig


def generate_market_candles(n: int = 1200, seed: int = 1337) -> list[Kline]:
    """Generate realistic synthetic 15m crypto futures candles using geometric random walk."""
    rng = np.random.RandomState(seed)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    base_price = 60_000.0
    candles: list[Kline] = []

    p = base_price
    for i in range(n):
        open_time = t0 + timedelta(minutes=15 * i)
        close_time = open_time + timedelta(minutes=14, seconds=59)

        # 15m return with slight drift and mean-reversion
        ret = rng.normal(0.0001, 0.0035) - 0.0002 * ((p - base_price) / base_price)
        close_p = p * math.exp(ret)
        low_p = min(p, close_p) * (1.0 - abs(rng.normal(0.0008, 0.0005)))
        high_p = max(p, close_p) * (1.0 + abs(rng.normal(0.0008, 0.0005)))
        vol = float(rng.lognormal(mean=3.0, sigma=0.6))
        q_vol = vol * close_p

        candles.append(
            Kline(
                symbol="BTCUSDT",
                interval="15m",
                open_time=open_time,
                close_time=close_time,
                open=float(p),
                high=float(high_p),
                low=float(low_p),
                close=float(close_p),
                volume=vol,
                quote_volume=q_vol,
                trades=int(rng.randint(20, 200)),
                is_closed=True,
            )
        )
        p = close_p

    return candles


def run_experiment(
    *,
    reward_scale: float,
    seed: int,
    invalid_action_penalty: float = 0.001,
    train_candles: list[Kline],
    eval_candles: list[Kline],
    warmup_context: list[Kline],
    output_base: Path,
    run_id: str,
) -> dict[str, Any]:
    run_dir = output_base / run_id
    trading_cfg = TradingConfig(
        fee_rate=0.0005,       # 0.05% realistic taker fee
        slippage_rate=0.0002,  # 0.02% realistic slippage
        funding_mode="constant",
        funding_rate=0.0001,   # 0.01% funding
        reward_scale=reward_scale,
        initial_balance=10_000.0,
    )
    data_cfg = DataConfig(
        symbol="BTCUSDT",
        interval="15m",
        warmup_candles=50,
        min_train_candles=100,
        min_eval_candles=50,
        news_max_age_hours=24.0,
    )
    dqn_cfg = DQNConfig(
        net_arch=(128, 128),
        learning_starts=300,
        total_timesteps=1500,
        buffer_size=10_000,
        batch_size=64,
        checkpoint_freq=1500,
        seed=seed,
        invalid_action_penalty=invalid_action_penalty,
    )
    pipe_cfg = PipelineConfig(
        dqn=dqn_cfg,
        data=data_cfg,
        output_dir=str(output_base),
        run_name=run_id,
    )

    # Train
    train_res = train_dqn(
        train_candles,
        config=pipe_cfg,
        trading_config=trading_cfg,
        output_dir=output_base,
        run_name=run_id,
    )

    model_path = run_dir / "final_model.zip"

    # Strict load through ModelInference
    inference = ModelInference.load(model_path)

    # Evaluate with action masking active
    eval_res = evaluate_model(
        inference,
        eval_candles,
        warmup_context=warmup_context,
        trading_config=trading_cfg,
        data_config=data_cfg,
        seed=seed,
        include_baselines=False,
    )

    # Extract metrics
    metrics = eval_res["metrics"]
    actions = eval_res["actions"]
    raw_actions = eval_res.get("raw_actions", [])
    invalid_count = eval_res.get("invalid_predictions_count", 0)

    # Q-values analysis over eval candles
    # Generate observations for eval candles using inference.predict_observation
    q_vals_all: list[list[float]] = []
    q_margins_all: list[float] = []
    action_names = [a.name for a in Action]

    # Sample observations across evaluation episode
    env_candles = list(warmup_context) + list(eval_candles)
    from noone.rl_engine.gym_env import NooneGymEnv
    probe_env = NooneGymEnv(env_candles, trading_cfg, feature_builder=inference.feature_builder)
    obs, _ = probe_env.reset(seed=seed, options={"start_index": 0})
    for _ in range(len(warmup_context)):
        obs, _, _, _, _ = probe_env.step(0)

    for step_act in actions:
        pred = inference.predict_observation(obs)
        q_vals_all.append([pred.q_values[name] for name in action_names])
        q_margins_all.append(pred.q_margin)
        obs, _, term, trunc, _ = probe_env.step(step_act)
        if term or trunc:
            break

    q_matrix = np.array(q_vals_all) if q_vals_all else np.zeros((1, 5))
    q_mean = float(np.mean(q_matrix))
    q_std = float(np.std(q_matrix))
    q_margin_mean = float(np.mean(q_margins_all)) if q_margins_all else 0.0

    raw_action_counts = {Action(i).name: int(np.sum(np.array(raw_actions) == i)) for i in range(5)}
    masked_action_counts = {Action(i).name: int(np.sum(np.array(actions) == i)) for i in range(5)}

    hold_freq = float(masked_action_counts["HOLD"]) / max(1, len(actions))
    raw_hold_freq = float(raw_action_counts["HOLD"]) / max(1, len(raw_actions))

    return {
        "run_id": run_id,
        "reward_scale": reward_scale,
        "seed": seed,
        "invalid_action_penalty": invalid_action_penalty,
        "q_mean": q_mean,
        "q_std": q_std,
        "q_margin_mean": q_margin_mean,
        "raw_action_counts": raw_action_counts,
        "masked_action_counts": masked_action_counts,
        "raw_invalid_count": invalid_count,
        "raw_invalid_pct": float(invalid_count) / max(1, len(raw_actions)) * 100.0,
        "hold_freq": hold_freq,
        "raw_hold_freq": raw_hold_freq,
        "return_pct": float(metrics.get("total_return", 0.0)) * 100.0,
        "max_drawdown_pct": float(metrics.get("max_drawdown", 0.0)) * 100.0,
        "trade_count": int(metrics.get("trade_count", 0)),
        "final_equity": float(eval_res.get("final_equity", 10_000.0)),
    }


def main():
    print("Generating deterministic historical dataset (1200 candles)...")
    all_candles = generate_market_candles(1200, seed=1337)
    prepared = prepare_candles(all_candles, interval="15m", warmup_candles=50)
    split = chronological_split(
        prepared,
        train_fraction=0.8,
        warmup_candles=50,
        min_train_candles=200,
        min_eval_candles=100,
    )
    print(f"Data split: {len(split.train)} train candles, {len(split.evaluation)} eval candles, {len(split.eval_warmup_context)} warmup context.")

    exp_dir = Path("runs/experiments_reward_scale")
    if exp_dir.exists():
        shutil.rmtree(exp_dir, ignore_errors=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    scales = [1.0, 100.0, 1000.0]
    seeds = [42, 43, 44]

    results: list[dict[str, Any]] = []

    print("\n--- Part B: Reward Scale Comparison across 3 Seeds ---")
    for scale in scales:
        for seed in seeds:
            run_id = f"scale_{int(scale)}_seed_{seed}"
            print(f"Running {run_id} (scale={scale}, seed={seed}, penalty=0.001)...")
            res = run_experiment(
                reward_scale=scale,
                seed=seed,
                invalid_action_penalty=0.001,
                train_candles=split.train,
                eval_candles=split.evaluation,
                warmup_context=split.eval_warmup_context,
                output_base=exp_dir,
                run_id=run_id,
            )
            results.append(res)
            print(f"  Done: Q_mean={res['q_mean']:.4f}, Q_std={res['q_std']:.4f}, HOLD={res['hold_freq']*100:.1f}%, RawInvalid={res['raw_invalid_count']} ({res['raw_invalid_pct']:.1f}%), Return={res['return_pct']:.3f}%")

    print("\n--- Part C: Invalid-Action Penalty Ablation (Decision C) ---")
    # Compare with vs without penalty (seed 42, reward_scale 100.0)
    unpenalized_id = "scale_100_unpenalized_seed_42"
    print(f"Running {unpenalized_id} (scale=100.0, seed=42, penalty=0.0)...")
    res_unpenalized = run_experiment(
        reward_scale=100.0,
        seed=42,
        invalid_action_penalty=0.0,
        train_candles=split.train,
        eval_candles=split.evaluation,
        warmup_context=split.eval_warmup_context,
        output_base=exp_dir,
        run_id=unpenalized_id,
    )
    results.append(res_unpenalized)
    print(f"  Unpenalized: RawInvalid={res_unpenalized['raw_invalid_count']} ({res_unpenalized['raw_invalid_pct']:.1f}%), RawActions={res_unpenalized['raw_action_counts']}, MaskedActions={res_unpenalized['masked_action_counts']}")

    penalized_match = next(r for r in results if r["run_id"] == "scale_100_seed_42")
    print(f"  Penalized (0.001): RawInvalid={penalized_match['raw_invalid_count']} ({penalized_match['raw_invalid_pct']:.1f}%), RawActions={penalized_match['raw_action_counts']}, MaskedActions={penalized_match['masked_action_counts']}")

    # Save all results to JSON
    summary_path = exp_dir / "experiment_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved experiment summary to {summary_path}")


if __name__ == "__main__":
    main()
