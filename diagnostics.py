"""Deep Diagnostics for Component 1 (DQN + RL Engine).

Audits:
1. Q-value distributions by position state (Flat, Long, Short) for scale 1, 100, 1000.
2. Step-by-step training trace: action counts, valid vs invalid, flat vs positioned steps, completed round trips.
3. Training loss over time, gradient norms, reward distributions (economic vs penalty).
4. Exploration schedule analysis.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from noone.market_data.models import Kline
from noone.rl_engine import Action, ModelInference, get_valid_action_mask, get_valid_actions
from noone.rl_engine.config import DataConfig, DQNConfig, PipelineConfig
from noone.rl_engine.data import chronological_split, prepare_candles
from noone.rl_engine.features import FeatureBuilder
from noone.rl_engine.gym_env import NooneGymEnv
from noone.rl_engine.inference import AccountSnapshot
from noone.trading_env import TradingConfig
from experiments_reward_scale import generate_market_candles


def audit_q_values_by_position_state(model_path: str | Path, eval_candles: list[Kline], warmup: list[Kline]) -> dict[str, Any]:
    inference = ModelInference.load(model_path)
    fb = inference.feature_builder or FeatureBuilder()

    # Create dummy account snapshots representing Flat, Long, and Short states
    snap_flat = AccountSnapshot(equity=10_000.0, available_balance=10_000.0, position_side=0)
    snap_long = AccountSnapshot(equity=10_000.0, available_balance=9_000.0, position_side=1, position_quantity=0.1, entry_price=60_000.0)
    snap_short = AccountSnapshot(equity=10_000.0, available_balance=9_000.0, position_side=-1, position_quantity=0.1, entry_price=60_000.0)

    # For each evaluation candle, build observation under Flat, Long, and Short, and evaluate Q-values
    full_candles = list(warmup) + list(eval_candles)
    history_start = len(warmup)

    q_flat_list, q_long_list, q_short_list = [], [], []

    for idx in range(history_start, len(full_candles)):
        current_history = full_candles[:idx + 1]

        # Flat state
        obs_flat = fb.build(snap_flat.to_obs_dict(), current_history)
        p_flat = inference.predict_observation(obs_flat, position_state=0)
        q_flat_list.append([p_flat.q_values[a.name] for a in Action])

        # Long state
        obs_long = fb.build(snap_long.to_obs_dict(), current_history)
        p_long = inference.predict_observation(obs_long, position_state=1)
        q_long_list.append([p_long.q_values[a.name] for a in Action])

        # Short state
        obs_short = fb.build(snap_short.to_obs_dict(), current_history)
        p_short = inference.predict_observation(obs_short, position_state=-1)
        q_short_list.append([p_short.q_values[a.name] for a in Action])

    q_flat = np.array(q_flat_list)
    q_long = np.array(q_long_list)
    q_short = np.array(q_short_list)

    action_names = [a.name for a in Action]

    flat_means = {action_names[i]: float(np.mean(q_flat[:, i])) for i in range(5)}
    long_means = {action_names[i]: float(np.mean(q_long[:, i])) for i in range(5)}
    short_means = {action_names[i]: float(np.mean(q_short[:, i])) for i in range(5)}

    # Differences when Flat
    diff_flat_hold_long = flat_means["HOLD"] - flat_means["OPEN_LONG"]
    diff_flat_hold_short = flat_means["HOLD"] - flat_means["OPEN_SHORT"]

    # Differences when Long
    diff_long_hold_close = long_means["HOLD"] - long_means["CLOSE_LONG"]

    # Differences when Short
    diff_short_hold_close = short_means["HOLD"] - short_means["CLOSE_SHORT"]

    return {
        "flat_q_means": flat_means,
        "long_q_means": long_means,
        "short_q_means": short_means,
        "diff_flat_hold_minus_open_long": diff_flat_hold_long,
        "diff_flat_hold_minus_open_short": diff_flat_hold_short,
        "diff_long_hold_minus_close_long": diff_long_hold_close,
        "diff_short_hold_minus_close_short": diff_short_hold_close,
    }


def trace_training_transitions(
    train_candles: list[Kline],
    reward_scale: float = 100.0,
    seed: int = 42,
    invalid_action_penalty: float = 0.001,
    total_timesteps: int = 1500,
    learning_starts: int = 300,
) -> dict[str, Any]:
    """Instrument and trace a complete training run to record step-level transitions, rewards, and losses."""
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback

    trading_cfg = TradingConfig(
        fee_rate=0.0005,
        slippage_rate=0.0002,
        funding_mode="constant",
        funding_rate=0.0001,
        reward_scale=reward_scale,
        initial_balance=10_000.0,
    )
    env = NooneGymEnv(
        train_candles,
        config=trading_cfg,
        feature_builder=FeatureBuilder(news_max_age_hours=24.0),
        invalid_action_penalty=invalid_action_penalty,
    )

    losses = []
    grad_norms = []
    actions_taken = []
    economic_rewards = []
    penalty_rewards = []
    total_rewards = []
    valid_actions = []
    positions_held = []  # 0=flat, 1=long, -1=short
    round_trips = 0

    class DiagnosticCallback(BaseCallback):
        def _on_step(self) -> bool:
            # env info from last step
            infos = self.locals.get("infos", [{}])
            actions = self.locals.get("actions", [0])
            rewards = self.locals.get("rewards", [0.0])

            info = infos[0] if infos else {}
            act = int(actions[0])
            r = float(rewards[0])

            pen = float(info.get("invalid_action_penalty", 0.0))
            econ_r = r + pen  # raw economic reward before penalty

            actions_taken.append(act)
            total_rewards.append(r)
            economic_rewards.append(econ_r)
            penalty_rewards.append(pen)
            is_invalid = bool(info.get("invalid_action", False))
            valid_actions.append(not is_invalid)

            pos_side = int(info.get("raw_observation", {}).get("position_side", 0))
            positions_held.append(pos_side)

            if info.get("event") in ("close", "liquidation", "stop_loss", "take_profit"):
                nonlocal round_trips
                round_trips += 1

            # Check if gradient update happened in this step
            model_self = self.model
            logger = model_self.logger
            if logger and hasattr(logger, "name_to_value"):
                l_val = logger.name_to_value.get("train/loss")
                if l_val is not None and (not losses or l_val != losses[-1]):
                    losses.append(float(l_val))

            return True

    cb = DiagnosticCallback()
    dqn_model = DQN(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=10_000,
        learning_starts=learning_starts,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.25,
        policy_kwargs={"net_arch": [128, 128]},
        seed=seed,
        device="cpu",
        verbose=0,
    )

    dqn_model.learn(total_timesteps=total_timesteps, callback=cb)

    # Analyze transition statistics
    act_arr = np.array(actions_taken)
    val_arr = np.array(valid_actions)
    pos_arr = np.array(positions_held)

    # Count valid and invalid actions per action enum
    counts_valid = {}
    counts_invalid = {}
    for a in range(5):
        name = Action(a).name
        counts_valid[name] = int(np.sum((act_arr == a) & val_arr))
        counts_invalid[name] = int(np.sum((act_arr == a) & ~val_arr))

    flat_steps = int(np.sum(pos_arr == 0))
    long_steps = int(np.sum(pos_arr == 1))
    short_steps = int(np.sum(pos_arr == -1))

    # Exploration rate calculation
    # Schedule: 1.0 down to 0.05 over exploration_fraction * total_timesteps (375 steps)
    eps_at_starts = 1.0 - (learning_starts / (0.25 * total_timesteps)) * (1.0 - 0.05)
    eps_at_starts = max(0.05, eps_at_starts)

    return {
        "total_timesteps": len(actions_taken),
        "learning_starts": learning_starts,
        "exploration_duration_steps": int(0.25 * total_timesteps),
        "eps_at_learning_starts": eps_at_starts,
        "round_trips_completed": round_trips,
        "flat_steps": flat_steps,
        "flat_pct": flat_steps / max(1, len(actions_taken)) * 100.0,
        "long_steps": long_steps,
        "long_pct": long_steps / max(1, len(actions_taken)) * 100.0,
        "short_steps": short_steps,
        "short_pct": short_steps / max(1, len(actions_taken)) * 100.0,
        "valid_action_counts": counts_valid,
        "invalid_action_counts": counts_invalid,
        "total_invalid_actions": int(np.sum(~val_arr)),
        "total_invalid_pct": float(np.sum(~val_arr)) / max(1, len(actions_taken)) * 100.0,
        "economic_reward_mean": float(np.mean(economic_rewards)),
        "economic_reward_std": float(np.std(economic_rewards)),
        "penalty_reward_mean": float(np.mean(penalty_rewards)),
        "penalty_reward_count": int(np.sum(np.array(penalty_rewards) > 0)),
        "net_reward_mean": float(np.mean(total_rewards)),
        "recorded_losses_count": len(losses),
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "min_loss": float(np.min(losses)) if losses else None,
        "max_loss": float(np.max(losses)) if losses else None,
    }


def main():
    print("Generating market dataset...")
    candles = generate_market_candles(1200, seed=1337)
    prepared = prepare_candles(candles, interval="15m", warmup_candles=50)
    split = chronological_split(prepared, train_fraction=0.8, warmup_candles=50, min_train_candles=200, min_eval_candles=100)

    print("\n=======================================================")
    print("1. Q-VALUE BREAKDOWN BY POSITION STATE (EVALUATION)")
    print("=======================================================")

    scales = [1, 100, 1000]
    for s in scales:
        model_p = Path(f"runs/experiments_reward_scale/scale_{s}_seed_42/final_model.zip")
        if model_p.exists():
            q_res = audit_q_values_by_position_state(model_p, split.evaluation, split.eval_warmup_context)
            print(f"\n--- Model Scale {s} (Seed 42) ---")
            print("When Flat Q-Means:")
            for k, v in q_res["flat_q_means"].items():
                print(f"  {k:12s}: {v:+.6f}")
            print(f"  Gap Q(HOLD) - Q(OPEN_LONG) : {q_res['diff_flat_hold_minus_open_long']:+.6f}")
            print(f"  Gap Q(HOLD) - Q(OPEN_SHORT): {q_res['diff_flat_hold_minus_open_short']:+.6f}")

            print("\nWhen Long Q-Means:")
            for k, v in q_res["long_q_means"].items():
                print(f"  {k:12s}: {v:+.6f}")
            print(f"  Gap Q(HOLD) - Q(CLOSE_LONG): {q_res['diff_long_hold_minus_close_long']:+.6f}")

            print("\nWhen Short Q-Means:")
            for k, v in q_res["short_q_means"].items():
                print(f"  {k:12s}: {v:+.6f}")
            print(f"  Gap Q(HOLD) - Q(CLOSE_SHORT): {q_res['diff_short_hold_minus_close_short']:+.6f}")

    # Unpenalized model Q-values
    unpen_p = Path("runs/experiments_reward_scale/scale_100_unpenalized_seed_42/final_model.zip")
    if unpen_p.exists():
        q_res_unpen = audit_q_values_by_position_state(unpen_p, split.evaluation, split.eval_warmup_context)
        print(f"\n--- Model Scale 100 Unpenalized (Seed 42) ---")
        print("When Flat Q-Means:")
        for k, v in q_res_unpen["flat_q_means"].items():
            print(f"  {k:12s}: {v:+.6f}")
        print(f"  Gap Q(HOLD) - Q(OPEN_LONG) : {q_res_unpen['diff_flat_hold_minus_open_long']:+.6f}")
        print(f"  Gap Q(HOLD) - Q(OPEN_SHORT): {q_res_unpen['diff_flat_hold_minus_open_short']:+.6f}")

    print("\n=======================================================")
    print("2. STEP-BY-STEP TRAINING DYNAMICS AUDIT (1500 STEPS)")
    print("=======================================================")

    print("\nTracing Scale 100.0 Training Run (with penalty 0.001)...")
    t_pen = trace_training_transitions(split.train, reward_scale=100.0, seed=42, invalid_action_penalty=0.001)
    print(json.dumps(t_pen, indent=2))

    print("\nTracing Scale 100.0 Training Run (Unpenalized, penalty 0.0)...")
    t_unpen = trace_training_transitions(split.train, reward_scale=100.0, seed=42, invalid_action_penalty=0.0)
    print(json.dumps(t_unpen, indent=2))

    print("\nTracing Scale 1.0 Training Run (with penalty 0.001)...")
    t_scale1 = trace_training_transitions(split.train, reward_scale=1.0, seed=42, invalid_action_penalty=0.001)
    print(json.dumps(t_scale1, indent=2))

    # Save diagnostics
    diag_path = Path("runs/experiments_reward_scale/diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump({
            "scale_100_penalized_trace": t_pen,
            "scale_100_unpenalized_trace": t_unpen,
            "scale_1_trace": t_scale1,
        }, f, indent=2)
    print(f"\nSaved diagnostic traces to {diag_path}")


if __name__ == "__main__":
    main()

