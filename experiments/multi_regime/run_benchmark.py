"""Multi-Regime DQN Benchmark Runner for PROJECT NOONE Component 1.

Investigates whether the existing DQN pipeline learns regime-dependent behaviour:
1. Training configurations:
   - Multi-Regime (Bull + Bear + Sideways, 1200 candles)
   - Trend Only (Bull + Bear, 800 candles) -> tests generalization to unseen Sideways
   - Range Only (Sideways, 400 candles) -> tests generalization to unseen Bull & Bear
2. Multiple random seeds (42, 43)
3. Evaluates on 4 held-out environments:
   - Bull (200 candles)
   - Bear (200 candles)
   - Sideways (200 candles)
   - Composite (600 candles continuous sequence)
4. Benchmarks against:
   - Always HOLD (remain flat)
   - Buy-and-Hold (long from bar 0)
   - Simple Momentum (EMA9 vs EMA21 trend follower)
   All under identical Component 8 accounting, fees, slippage, and leverage.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from experiments.multi_regime.generate_datasets import (
    analyze_segment,
    generate_benchmark_suite,
)
from noone.market_data.models import Kline
from noone.rl_engine import Action, ModelInference
from noone.rl_engine.config import DataConfig, DQNConfig, PipelineConfig
from noone.rl_engine.evaluation import evaluate_model
from noone.rl_engine.gym_env import NooneGymEnv
from noone.rl_engine.training import train_dqn
from noone.trading_env import TradingConfig


def make_exploration_schedule(
    total_timesteps: int,
    learning_starts: int,
    decay_duration: int,
    initial_eps: float = 1.0,
    final_eps: float = 0.05,
):
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


def run_evaluation_with_telemetry(
    *,
    inference: ModelInference,
    eval_candles: list[Kline],
    warmup_context: list[Kline],
    trading_cfg: TradingConfig,
    seed: int,
) -> dict[str, Any]:
    """Runs deterministic evaluation with complete action, exposure, and Q-value telemetry."""
    full_candles = list(warmup_context) + list(eval_candles)
    env = NooneGymEnv(full_candles, trading_cfg, feature_builder=inference.feature_builder)
    obs, _ = env.reset(seed=seed, options={"start_index": 0})
    for _ in range(len(warmup_context)):
        obs, _, term, trunc, _ = env.step(0)

    actions: list[int] = []
    raw_actions: list[int] = []
    raw_invalid_count = 0
    trade_pnls: list[float] = []
    q_vals_all: list[dict[str, float]] = []

    steps_flat = 0
    steps_long = 0
    steps_short = 0

    done = False
    while not done:
        pos = env.env.position
        pos_side = 0 if pos is None else pos.side
        if pos_side == 0:
            steps_flat += 1
        elif pos_side == 1:
            steps_long += 1
        elif pos_side == -1:
            steps_short += 1

        pred = inference.predict_observation(obs, position_state=pos_side)
        if not pred.is_valid:
            raw_invalid_count += 1

        act = int(pred.action.value)
        raw_act = int(pred.raw_action.value)
        actions.append(act)
        raw_actions.append(raw_act)
        q_vals_all.append(pred.q_values)

        obs, reward, term, trunc, info = env.step(act)
        if info.get("event") in ("close", "liquidation", "daily_limit_close") or "realized_pnl" in info:
            trade_pnls.append(float(info.get("realized_pnl", 0.0)))

        done = bool(term or trunc)

    total_steps = max(1, len(actions))
    time_flat_pct = steps_flat / total_steps * 100.0
    time_long_pct = steps_long / total_steps * 100.0
    time_short_pct = steps_short / total_steps * 100.0

    winning_trades = [p for p in trade_pnls if p > 0]
    losing_trades = [p for p in trade_pnls if p < 0]
    win_rate = (len(winning_trades) / len(trade_pnls) * 100.0) if trade_pnls else 0.0
    avg_win = float(np.mean(winning_trades)) if winning_trades else 0.0
    avg_loss = float(np.mean(losing_trades)) if losing_trades else 0.0

    # Q-values summary
    action_names = [a.name for a in Action]
    q_means = {
        name: float(np.mean([q[name] for q in q_vals_all])) if q_vals_all else 0.0
        for name in action_names
    }

    # Action counts
    masked_counts = {Action(i).name: int(np.sum(np.array(actions) == i)) for i in range(5)}
    raw_counts = {Action(i).name: int(np.sum(np.array(raw_actions) == i)) for i in range(5)}

    # Equity curve and return
    init_bal = trading_cfg.initial_balance
    final_eq = float(env.env.equity)
    tot_ret_pct = (final_eq / init_bal - 1.0) * 100.0

    return {
        "total_return_pct": tot_ret_pct,
        "final_equity": final_eq,
        "completed_trades": len(trade_pnls),
        "win_rate_pct": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "total_fees": float(env.env.total_fees),
        "total_funding": float(env.env.total_funding),
        "time_flat_pct": time_flat_pct,
        "time_long_pct": time_long_pct,
        "time_short_pct": time_short_pct,
        "raw_invalid_count": raw_invalid_count,
        "raw_invalid_pct": raw_invalid_count / total_steps * 100.0,
        "masked_action_counts": masked_counts,
        "raw_action_counts": raw_counts,
        "mean_q_values": q_means,
    }


def run_baseline(
    *,
    baseline_type: str,
    eval_candles: list[Kline],
    warmup_context: list[Kline],
    trading_cfg: TradingConfig,
    seed: int = 42,
) -> dict[str, Any]:
    """Runs a baseline strategy through the exact same gym environment."""
    full_candles = list(warmup_context) + list(eval_candles)
    env = NooneGymEnv(full_candles, trading_cfg)
    obs, _ = env.reset(seed=seed, options={"start_index": 0})
    for _ in range(len(warmup_context)):
        obs, _, term, trunc, _ = env.step(0)

    actions: list[int] = []
    trade_pnls: list[float] = []
    steps_flat = 0
    steps_long = 0
    steps_short = 0

    done = False
    step_idx = 0
    while not done:
        pos = env.env.position
        pos_side = 0 if pos is None else pos.side
        if pos_side == 0:
            steps_flat += 1
        elif pos_side == 1:
            steps_long += 1
        elif pos_side == -1:
            steps_short += 1

        if baseline_type == "always_hold":
            act = Action.HOLD
        elif baseline_type == "buy_and_hold":
            if step_idx == 0 and pos_side == 0:
                act = Action.OPEN_LONG
            else:
                act = Action.HOLD
        elif baseline_type == "momentum":
            # Feature 4: close_ema9_dev, Feature 5: close_ema21_dev
            ema9_dev = obs[4]
            ema21_dev = obs[5]
            if ema9_dev > ema21_dev:
                if pos_side == 0:
                    act = Action.OPEN_LONG
                elif pos_side == -1:
                    act = Action.CLOSE_SHORT
                else:
                    act = Action.HOLD
            elif ema9_dev < ema21_dev:
                if pos_side == 0:
                    act = Action.OPEN_SHORT
                elif pos_side == 1:
                    act = Action.CLOSE_LONG
                else:
                    act = Action.HOLD
            else:
                act = Action.HOLD
        else:
            raise ValueError(f"Unknown baseline: {baseline_type}")

        actions.append(act.value)
        obs, reward, term, trunc, info = env.step(act.value)
        if info.get("event") in ("close", "liquidation", "daily_limit_close") or "realized_pnl" in info:
            trade_pnls.append(float(info.get("realized_pnl", 0.0)))

        done = bool(term or trunc)
        step_idx += 1

    total_steps = max(1, len(actions))
    winning_trades = [p for p in trade_pnls if p > 0]
    losing_trades = [p for p in trade_pnls if p < 0]
    win_rate = (len(winning_trades) / len(trade_pnls) * 100.0) if trade_pnls else 0.0
    avg_win = float(np.mean(winning_trades)) if winning_trades else 0.0
    avg_loss = float(np.mean(losing_trades)) if losing_trades else 0.0

    final_eq = float(env.env.equity)
    tot_ret_pct = (final_eq / trading_cfg.initial_balance - 1.0) * 100.0

    return {
        "baseline": baseline_type,
        "total_return_pct": tot_ret_pct,
        "final_equity": final_eq,
        "completed_trades": len(trade_pnls),
        "win_rate_pct": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "total_fees": float(env.env.total_fees),
        "total_funding": float(env.env.total_funding),
        "time_flat_pct": steps_flat / total_steps * 100.0,
        "time_long_pct": steps_long / total_steps * 100.0,
        "time_short_pct": steps_short / total_steps * 100.0,
        "action_counts": {Action(i).name: int(np.sum(np.array(actions) == i)) for i in range(5)},
    }


def main():
    print("=== MULTI-REGIME BENCHMARK FOR PROJECT NOONE COMPONENT 1 ===")
    out_dir = Path("experiments/multi_regime/results")
    model_dir = Path("experiments/multi_regime/models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Generating multi-regime dataset suite...")
    suite = generate_benchmark_suite()
    training_sets = suite["training_sets"]
    eval_suite = suite["eval_suite"]

    trading_cfg = TradingConfig(
        fee_rate=0.0005,
        slippage_rate=0.0002,
        funding_mode="constant",
        funding_rate=0.0001,
        reward_scale=100.0,
        initial_balance=10_000.0,
    )
    data_cfg = DataConfig(symbol="BTCUSDT", interval="15m", warmup_candles=50)

    # 1. Run all Baselines across all evaluation regimes
    print("\n--- Running Baseline Strategies across Evaluation Regimes ---")
    baseline_results: dict[str, dict[str, Any]] = {}
    for eval_name, eval_data in eval_suite.items():
        baseline_results[eval_name] = {}
        for b_name in ["always_hold", "buy_and_hold", "momentum"]:
            res = run_baseline(
                baseline_type=b_name,
                eval_candles=eval_data["candles"],
                warmup_context=eval_data["warmup"],
                trading_cfg=trading_cfg,
            )
            baseline_results[eval_name][b_name] = res
            print(f"  [{eval_name:<10}] {b_name:<15}: Return={res['total_return_pct']:+6.2f}%, Trades={res['completed_trades']:2d}, Flat/Long/Short={res['time_flat_pct']:.0f}%/{res['time_long_pct']:.0f}%/{res['time_short_pct']:.0f}%, Fees=${res['total_fees']:.2f}")

    # 2. Train and Evaluate DQN Models
    # Train 3 models across 2 seeds (42, 43):
    # A. multi_regime (15k timesteps, decay 5k)
    # B. trend_only (10k timesteps, decay 4k)
    # C. range_only (10k timesteps, decay 4k)
    train_configs = [
        {"name": "multi_regime", "data_key": "multi_regime", "timesteps": 15000, "decay": 5000},
        {"name": "trend_only", "data_key": "trend_only", "timesteps": 10000, "decay": 4000},
        {"name": "range_only", "data_key": "range_only", "timesteps": 10000, "decay": 4000},
    ]
    seeds = [42, 43]

    dqn_results: list[dict[str, Any]] = []

    for tcfg in train_configs:
        t_name = tcfg["name"]
        t_candles = training_sets[tcfg["data_key"]]
        t_steps = tcfg["timesteps"]
        t_decay = tcfg["decay"]

        for seed in seeds:
            run_id = f"{t_name}_seed_{seed}"
            run_dir = model_dir / run_id
            print(f"\n=======================================================")
            print(f"Training DQN: {run_id} (Candles={len(t_candles)}, Steps={t_steps}, Decay={t_decay})...")
            print(f"=======================================================")

            dqn_cfg = DQNConfig(
                net_arch=(128, 128),
                learning_starts=1000,
                total_timesteps=t_steps,
                buffer_size=50000,
                batch_size=64,
                checkpoint_freq=t_steps,
                seed=seed,
                invalid_action_penalty=0.001,
                device="cpu",
            )
            pipe_cfg = PipelineConfig(
                dqn=dqn_cfg,
                data=data_cfg,
                output_dir=str(model_dir),
                run_name=run_id,
            )

            schedule_fn = make_exploration_schedule(
                total_timesteps=t_steps,
                learning_starts=1000,
                decay_duration=t_decay,
            )

            t0 = datetime.now()
            train_res = train_dqn(
                t_candles,
                config=pipe_cfg,
                trading_config=trading_cfg,
                output_dir=model_dir,
                run_name=run_id,
                exploration_schedule=schedule_fn,
            )
            elapsed = (datetime.now() - t0).total_seconds()
            print(f"[{run_id}] Trained in {elapsed:.1f}s.")

            # Load strictly through ModelInference
            model_path = run_dir / "final_model.zip"
            inference = ModelInference.load(model_path)

            # Evaluate on all 4 evaluation regimes
            eval_outcomes: dict[str, Any] = {}
            for eval_name, eval_data in eval_suite.items():
                ev_res = run_evaluation_with_telemetry(
                    inference=inference,
                    eval_candles=eval_data["candles"],
                    warmup_context=eval_data["warmup"],
                    trading_cfg=trading_cfg,
                    seed=seed,
                )
                eval_outcomes[eval_name] = ev_res
                print(f"  Eval on {eval_name:<10}: Return={ev_res['total_return_pct']:+6.2f}%, Trades={ev_res['completed_trades']:2d}, Flat/Long/Short={ev_res['time_flat_pct']:.0f}%/{ev_res['time_long_pct']:.0f}%/{ev_res['time_short_pct']:.0f}%, Fees=${ev_res['total_fees']:.2f}, RawInv={ev_res['raw_invalid_count']}")

            dqn_results.append({
                "model_name": t_name,
                "run_id": run_id,
                "seed": seed,
                "train_candles": len(t_candles),
                "total_timesteps": t_steps,
                "train_seconds": elapsed,
                "evaluations": eval_outcomes,
            })

    # Save full summary
    full_benchmark = {
        "dataset_statistics": {
            k: analyze_segment(v, k) for k, v in suite["segments"].items()
        },
        "baselines": baseline_results,
        "dqn_models": dqn_results,
    }

    summary_file = out_dir / "benchmark_results.json"
    summary_file.write_text(json.dumps(full_benchmark, indent=2, default=str), encoding="utf-8")
    print(f"\nBenchmark complete. Full results written to {summary_file}")


if __name__ == "__main__":
    main()
