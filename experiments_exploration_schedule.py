"""Exploration Schedule & Extended Training Budget Experiments for PROJECT NOONE Component 1.

Investigates:
- Experiment A: 10,000 timesteps, learning_starts=1,000, exploration decay duration=3,000 steps (seeds 42, 43, 44)
- Experiment B: 25,000 timesteps, learning_starts=1,000, exploration decay duration=8,000 steps (seeds 42, 43, 44)

Decoupled exploration schedule:
- Epsilon = 1.0 until learning_starts (t <= 1,000)
- Epsilon linearly decays from 1.0 -> 0.05 over decay_duration
- Epsilon = 0.05 clamped thereafter
- Verifies epsilon at start, middle, and end of decay period.

Metrics collected:
- Training: total transitions, valid action breakdown (0..4), invalid actions and penalties,
  position duration (flat, long, short), round trips, epsilon progression & verification,
  loss history, Q-value and target-Q distributions.
- Evaluation: raw action distribution, executed masked action distribution, raw invalid rate,
  HOLD frequency, completed trades, return %, max drawdown %, win rate %, avg win, avg loss,
  fees, slippage rate (0.02%), funding costs.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from noone.market_data.models import Kline
from noone.rl_engine import Action, ModelInference
from noone.rl_engine.config import DataConfig, DQNConfig, PipelineConfig
from noone.rl_engine.data import chronological_split, prepare_candles
from noone.rl_engine.evaluation import evaluate_model
from noone.rl_engine.training import train_dqn
from noone.trading_env import TradingConfig
from stable_baselines3.common.callbacks import BaseCallback


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


def make_exploration_schedule(
    total_timesteps: int,
    learning_starts: int,
    decay_duration: int,
    initial_eps: float = 1.0,
    final_eps: float = 0.05,
):
    """Factory for linear exploration schedule decoupled from learning_starts."""
    decay_end = learning_starts + decay_duration

    def schedule(progress_remaining: float) -> float:
        current_step = (1.0 - progress_remaining) * total_timesteps
        if current_step <= learning_starts:
            return initial_eps
        elif current_step >= decay_end:
            return final_eps
        else:
            fraction = (current_step - learning_starts) / float(decay_duration)
            return initial_eps - fraction * (initial_eps - final_eps)

    return schedule


class TrainingDiagnosticsCallback(BaseCallback):
    """Tracks training diagnostics, verifies epsilon schedule, and records metrics."""

    def __init__(
        self,
        *,
        total_timesteps: int,
        learning_starts: int,
        decay_duration: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.learning_starts = learning_starts
        self.decay_duration = decay_duration
        self.decay_end = learning_starts + decay_duration
        self.midpoint = learning_starts + decay_duration // 2

        self.total_transitions = 0
        self.valid_action_counts = {a.name: 0 for a in Action}
        self.invalid_action_counts = {a.name: 0 for a in Action}
        self.total_penalty_applied = 0.0

        self.time_flat_steps = 0
        self.time_long_steps = 0
        self.time_short_steps = 0

        self.completed_round_trips = 0
        self.trade_pnls: list[float] = []

        self.epsilon_checks: dict[str, dict[str, Any]] = {}
        self.epsilon_progression: list[dict[str, Any]] = []
        self.loss_history: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        self.total_transitions += 1
        current_step = self.num_timesteps

        # Track position state prior to / during step
        target = self.training_env.envs[0]
        while hasattr(target, "env"):
            if hasattr(target, "position"):
                break
            target = target.env
        pos = getattr(target, "position", None)
        if pos is None:
            self.time_flat_steps += 1
        elif pos.side == 1:
            self.time_long_steps += 1
        elif pos.side == -1:
            self.time_short_steps += 1

        # Track action and validity
        action_idx = int(self.locals["actions"][0])
        action = Action(action_idx)
        info = self.locals["infos"][0]
        is_invalid = bool(info.get("invalid_action", False))

        if is_invalid:
            self.invalid_action_counts[action.name] += 1
            self.total_penalty_applied += float(info.get("invalid_action_penalty", 0.0))
        else:
            self.valid_action_counts[action.name] += 1

        # Track completed trades / round trips
        if info.get("event") in ("close", "liquidation", "stop_loss", "take_profit", "daily_limit_close") or "realized_pnl" in info:
            self.completed_round_trips += 1
            self.trade_pnls.append(float(info.get("realized_pnl", 0.0)))

        # Track epsilon and verify at start, middle, and end of decay
        current_eps = float(self.model.exploration_rate)
        if current_step == self.learning_starts:
            self.epsilon_checks["beginning"] = {
                "step": current_step,
                "epsilon": current_eps,
                "expected": 1.0,
                "verified": abs(current_eps - 1.0) < 0.01,
            }
        elif current_step == self.midpoint:
            expected_mid = 1.0 - 0.5 * (1.0 - 0.05)
            self.epsilon_checks["middle"] = {
                "step": current_step,
                "epsilon": current_eps,
                "expected": expected_mid,
                "verified": abs(current_eps - expected_mid) < 0.01,
            }
        elif current_step == self.decay_end + 1 or (current_step == self.decay_end and "end" not in self.epsilon_checks):
            self.epsilon_checks["end"] = {
                "step": current_step,
                "epsilon": current_eps,
                "expected": 0.05,
                "verified": abs(current_eps - 0.05) < 0.01,
            }

        # Sample epsilon trajectory
        if current_step % 500 == 0 or current_step in (self.learning_starts, self.midpoint, self.decay_end, self.total_timesteps):
            self.epsilon_progression.append({"step": current_step, "epsilon": current_eps})

        # Track training loss
        loss_val = self.logger.name_to_value.get("train/loss")
        if loss_val is not None:
            if not self.loss_history or self.loss_history[-1]["step"] != current_step:
                self.loss_history.append({"step": current_step, "loss": float(loss_val)})

        return True


def compute_q_distributions(model: Any, n_samples: int = 1000) -> dict[str, float]:
    """Compute Q-value and target-Q distributions from the replay buffer."""
    import torch

    rb = model.replay_buffer
    actual_samples = min(n_samples, rb.size())
    if actual_samples == 0:
        return {}

    batch = rb.sample(actual_samples)
    with torch.no_grad():
        q_vals = model.policy.q_net(batch.observations)
        next_q_vals = model.policy.q_net_target(batch.next_observations)
        max_next_q, _ = next_q_vals.max(dim=1, keepdim=True)
        target_q = batch.rewards + (1.0 - batch.dones) * model.gamma * max_next_q
        chosen_q = torch.gather(q_vals, dim=1, index=batch.actions.long())

    q_all = q_vals.cpu().numpy()
    target_q_np = target_q.cpu().numpy().flatten()
    chosen_q_np = chosen_q.cpu().numpy().flatten()

    return {
        "q_mean": float(np.mean(q_all)),
        "q_std": float(np.std(q_all)),
        "q_min": float(np.min(q_all)),
        "q_max": float(np.max(q_all)),
        "target_q_mean": float(np.mean(target_q_np)),
        "target_q_std": float(np.std(target_q_np)),
        "target_q_min": float(np.min(target_q_np)),
        "target_q_max": float(np.max(target_q_np)),
        "chosen_q_mean": float(np.mean(chosen_q_np)),
        "chosen_q_std": float(np.std(chosen_q_np)),
    }


def run_single_experiment(
    *,
    experiment_name: str,
    total_timesteps: int,
    learning_starts: int,
    decay_duration: int,
    seed: int,
    train_candles: list[Kline],
    eval_candles: list[Kline],
    warmup_context: list[Kline],
    output_base: Path,
) -> dict[str, Any]:
    run_id = f"{experiment_name}_seed_{seed}"
    run_dir = output_base / run_id

    trading_cfg = TradingConfig(
        fee_rate=0.0005,       # 0.05% taker fee
        slippage_rate=0.0002,  # 0.02% slippage
        funding_mode="constant",
        funding_rate=0.0001,   # 0.01% funding
        reward_scale=100.0,
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
        learning_starts=learning_starts,
        total_timesteps=total_timesteps,
        buffer_size=50_000,
        batch_size=64,
        checkpoint_freq=total_timesteps,
        seed=seed,
        invalid_action_penalty=0.001,
        device="cpu",
    )
    pipe_cfg = PipelineConfig(
        dqn=dqn_cfg,
        data=data_cfg,
        output_dir=str(output_base),
        run_name=run_id,
    )

    schedule_fn = make_exploration_schedule(
        total_timesteps=total_timesteps,
        learning_starts=learning_starts,
        decay_duration=decay_duration,
        initial_eps=1.0,
        final_eps=0.05,
    )

    diag_callback = TrainingDiagnosticsCallback(
        total_timesteps=total_timesteps,
        learning_starts=learning_starts,
        decay_duration=decay_duration,
    )

    print(f"[{run_id}] Starting training: T={total_timesteps}, L_starts={learning_starts}, Decay={decay_duration}...")
    t0 = datetime.now()

    train_res = train_dqn(
        train_candles,
        config=pipe_cfg,
        trading_config=trading_cfg,
        output_dir=output_base,
        run_name=run_id,
        callbacks=[diag_callback],
        exploration_schedule=schedule_fn,
    )

    train_seconds = (datetime.now() - t0).total_seconds()
    print(f"[{run_id}] Training completed in {train_seconds:.1f}s.")

    model = train_res["model"]
    q_dist = compute_q_distributions(model, n_samples=2000)

    # Strict load through ModelInference for evaluation
    model_path = run_dir / "final_model.zip"
    inference = ModelInference.load(model_path)

    # Evaluate deterministically on held-out slice with action masking active
    eval_res = evaluate_model(
        inference,
        eval_candles,
        warmup_context=warmup_context,
        trading_config=trading_cfg,
        data_config=data_cfg,
        seed=seed,
        include_baselines=True,
    )

    metrics = eval_res["metrics"]
    actions = eval_res["actions"]
    raw_actions = eval_res.get("raw_actions", [])
    raw_invalid_count = eval_res.get("invalid_predictions_count", 0)
    trade_pnls = eval_res.get("trade_pnls", [])

    raw_action_counts = {Action(i).name: int(np.sum(np.array(raw_actions) == i)) for i in range(5)}
    masked_action_counts = {Action(i).name: int(np.sum(np.array(actions) == i)) for i in range(5)}

    hold_freq = float(masked_action_counts["HOLD"]) / max(1, len(actions))
    raw_hold_freq = float(raw_action_counts["HOLD"]) / max(1, len(raw_actions))
    raw_invalid_rate = float(raw_invalid_count) / max(1, len(raw_actions))

    winning_trades = [p for p in trade_pnls if p > 0]
    losing_trades = [p for p in trade_pnls if p < 0]
    avg_win = float(np.mean(winning_trades)) if winning_trades else 0.0
    avg_loss = float(np.mean(losing_trades)) if losing_trades else 0.0

    # Training time distributions
    total_tr = max(1, diag_callback.total_transitions)
    time_flat_pct = diag_callback.time_flat_steps / total_tr * 100.0
    time_long_pct = diag_callback.time_long_steps / total_tr * 100.0
    time_short_pct = diag_callback.time_short_steps / total_tr * 100.0

    # Recent training losses
    losses = [entry["loss"] for entry in diag_callback.loss_history]
    loss_initial = float(losses[0]) if losses else 0.0
    loss_final = float(np.mean(losses[-50:])) if len(losses) >= 50 else (float(losses[-1]) if losses else 0.0)

    result = {
        "experiment": experiment_name,
        "run_id": run_id,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "learning_starts": learning_starts,
        "decay_duration": decay_duration,
        "train_seconds": train_seconds,
        "training_diagnostics": {
            "total_transitions": diag_callback.total_transitions,
            "valid_actions": diag_callback.valid_action_counts,
            "invalid_actions": diag_callback.invalid_action_counts,
            "total_invalid_actions": sum(diag_callback.invalid_action_counts.values()),
            "total_penalties_applied": diag_callback.total_penalty_applied,
            "time_flat_steps": diag_callback.time_flat_steps,
            "time_long_steps": diag_callback.time_long_steps,
            "time_short_steps": diag_callback.time_short_steps,
            "time_flat_pct": time_flat_pct,
            "time_long_pct": time_long_pct,
            "time_short_pct": time_short_pct,
            "completed_round_trips": diag_callback.completed_round_trips,
            "epsilon_checks": diag_callback.epsilon_checks,
            "epsilon_progression": diag_callback.epsilon_progression,
            "loss_initial": loss_initial,
            "loss_final_mean": loss_final,
            "loss_history_samples": diag_callback.loss_history[::max(1, len(diag_callback.loss_history) // 20)],
            "q_distributions": q_dist,
        },
        "evaluation_diagnostics": {
            "raw_action_counts": raw_action_counts,
            "masked_action_counts": masked_action_counts,
            "raw_invalid_count": raw_invalid_count,
            "raw_invalid_rate_pct": raw_invalid_rate * 100.0,
            "hold_frequency_pct": hold_freq * 100.0,
            "raw_hold_frequency_pct": raw_hold_freq * 100.0,
            "num_trades": int(metrics.get("num_trades", 0)),
            "total_return_pct": float(metrics.get("total_return", 0.0)) * 100.0,
            "max_drawdown_pct": float(metrics.get("max_drawdown", 0.0)) * 100.0,
            "win_rate_pct": float(metrics.get("win_rate", 0.0)) * 100.0,
            "avg_winning_trade": avg_win,
            "avg_losing_trade": avg_loss,
            "total_fees": float(metrics.get("total_fees", 0.0)),
            "slippage_rate_pct": 0.02,
            "total_funding": float(metrics.get("total_funding", 0.0)),
            "final_equity": float(metrics.get("final_equity", 10_000.0)),
            "baseline_always_hold_return_pct": float(eval_res.get("baselines", {}).get("always_hold", {}).get("total_return", 0.0)) * 100.0,
            "baseline_random_return_pct": float(eval_res.get("baselines", {}).get("random_fixed_seed", {}).get("total_return", 0.0)) * 100.0,
        },
    }

    return result


def main():
    print("=== PROJECT NOONE: Component 1 Exploration Schedule Investigation ===")
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

    output_base = Path("runs/experiments_exploration_schedule")
    output_base.mkdir(parents=True, exist_ok=True)

    experiments = [
        {"name": "Exp_A_10k", "timesteps": 10_000, "learning_starts": 1_000, "decay": 3_000},
        {"name": "Exp_B_25k", "timesteps": 25_000, "learning_starts": 1_000, "decay": 8_000},
    ]
    seeds = [42, 43, 44]

    all_results: list[dict[str, Any]] = []

    for exp in experiments:
        print(f"\n=======================================================")
        print(f"Starting {exp['name']}: Total={exp['timesteps']}, Learning Starts={exp['learning_starts']}, Decay={exp['decay']}")
        print(f"=======================================================")
        for seed in seeds:
            res = run_single_experiment(
                experiment_name=exp["name"],
                total_timesteps=exp["timesteps"],
                learning_starts=exp["learning_starts"],
                decay_duration=exp["decay"],
                seed=seed,
                train_candles=split.train,
                eval_candles=split.evaluation,
                warmup_context=split.eval_warmup_context,
                output_base=output_base,
            )
            all_results.append(res)
            # Print intermediate summary
            ed = res["evaluation_diagnostics"]
            td = res["training_diagnostics"]
            print(f"--- Summary {res['run_id']} ---")
            print(f"  Training: transitions={td['total_transitions']}, round_trips={td['completed_round_trips']}, invalid={td['total_invalid_actions']}, flat/long/short={td['time_flat_pct']:.1f}%/{td['time_long_pct']:.1f}%/{td['time_short_pct']:.1f}%")
            print(f"  Epsilon verification: {td['epsilon_checks']}")
            print(f"  Q-distributions: Q_mean={td['q_distributions'].get('q_mean', 0.0):.4f}, Target_Q_mean={td['q_distributions'].get('target_q_mean', 0.0):.4f}")
            print(f"  Evaluation: HOLD={ed['hold_frequency_pct']:.1f}%, RawInvalid={ed['raw_invalid_count']} ({ed['raw_invalid_rate_pct']:.1f}%), Trades={ed['num_trades']}, Return={ed['total_return_pct']:.3f}%, WinRate={ed['win_rate_pct']:.1f}%")

    # Save summary
    summary_path = output_base / "experiment_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nAll experiments complete. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
