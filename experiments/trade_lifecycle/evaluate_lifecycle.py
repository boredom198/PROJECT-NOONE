"""Trade Lifecycle Evaluation Engine for PROJECT NOONE Component 1.

Evaluates whether trained DQN checkpoints independently execute complete trade lifecycles:
1. Select OPEN_LONG or OPEN_SHORT.
2. Maintain and manage the open position.
3. Independently select CLOSE_LONG or CLOSE_SHORT.
4. Return to a valid FLAT state.

Captures step-by-step decision telemetry, raw Q-values, action masks,
and strict distinction between policy closes and episode-end truncations.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root and src are in sys.path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from experiments.multi_regime.generate_datasets import generate_benchmark_suite
from experiments.multi_regime.generate_transition_datasets import build_all_transition_sequences
from noone.market_data.models import Kline
from noone.rl_engine import Action, ModelInference
from noone.rl_engine.gym_env import NooneGymEnv
from noone.trading_env import TradingConfig


def compute_max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd * 100.0


def extract_key_observations(obs: np.ndarray) -> dict[str, float]:
    """Extracts interpretable features from the canonical 26-dim observation."""
    return {
        "log_return_15": float(obs[0]),
        "close_ema9_dev": float(obs[4]),
        "close_ema21_dev": float(obs[5]),
        "close_ema50_dev": float(obs[6]),
        "rsi_14": float(obs[10]),
        "adx_14": float(obs[11]),
        "funding_rate": float(obs[15]),
        "position_side": float(obs[16]),
        "position_size": float(obs[17]),
        "unrealized_pnl": float(obs[18]),
    }


def evaluate_checkpoint_lifecycle(
    *,
    inference: ModelInference,
    checkpoint_name: str,
    group: str,
    seed: int,
    env_name: str,
    candles: list[Kline],
    warmup_context: list[Kline],
    trading_cfg: TradingConfig,
) -> dict[str, Any]:
    full_candles = list(warmup_context) + list(candles)
    env = NooneGymEnv(full_candles, trading_cfg, feature_builder=inference.feature_builder)
    obs, _ = env.reset(seed=seed, options={"start_index": 0})

    # Warmup phase (HOLD for all warmup candles)
    for _ in range(len(warmup_context)):
        obs, _, term, trunc, _ = env.step(0)

    # Evaluation phase
    actions: list[int] = []
    raw_actions: list[int] = []
    raw_invalid_count = 0
    equity_curve: list[float] = [float(env.env.equity)]

    steps_flat = 0
    steps_long = 0
    steps_short = 0

    trades: list[dict[str, Any]] = []
    current_trade: dict[str, Any] | None = None
    trade_counter = 0

    done = False
    step_idx = 0
    init_balance = trading_cfg.initial_balance

    while not done:
        pos = env.env.position
        pos_side = 0 if pos is None else pos.side
        cur_price = float(candles[step_idx].close)
        cur_time = candles[step_idx].close_time.isoformat() if candles[step_idx].close_time else None

        if pos_side == 0:
            steps_flat += 1
            if current_trade is not None and current_trade.get("exit_step") is not None:
                current_trade["steps_flat_after_exit"] += 1
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

        q_dict = {a.name: float(pred.q_values[a.name]) for a in Action}
        key_obs = extract_key_observations(obs)

        # Check if an entry is triggered at this step
        if pos_side == 0 and act in (Action.OPEN_LONG.value, Action.OPEN_SHORT.value):
            trade_counter += 1
            direction = "LONG" if act == Action.OPEN_LONG.value else "SHORT"
            current_trade = {
                "trade_id": trade_counter,
                "direction": direction,
                "entry_step": step_idx,
                "entry_time": cur_time,
                "entry_action": Action(act).name,
                "entry_price": 0.0,  # updated from info
                "position_size": 0.0,
                "margin_used": 0.0,
                "entry_fee": 0.0,
                "entry_q_values": q_dict,
                "entry_raw_action": Action(raw_act).name,
                "entry_is_valid": pred.is_valid,
                "entry_observations": key_obs,
                "duration_steps": 0,
                "exit_step": None,
                "exit_time": None,
                "exit_action": None,
                "exit_price": None,
                "exit_fee": 0.0,
                "realized_pnl": 0.0,
                "funding_accrued": 0.0,
                "net_pnl": 0.0,
                "exit_type": None,
                "exit_q_values": None,
                "exit_raw_action": None,
                "exit_is_valid": None,
                "exit_observations": None,
                "returned_to_flat": False,
                "steps_flat_after_exit": 0,
            }

        # Check if an exit is triggered at this step while in position
        elif pos_side != 0 and act in (Action.CLOSE_LONG.value, Action.CLOSE_SHORT.value):
            if current_trade is not None and current_trade.get("exit_step") is None:
                current_trade["exit_step"] = step_idx
                current_trade["exit_time"] = cur_time
                current_trade["exit_action"] = Action(act).name
                current_trade["exit_type"] = "policy_close"
                current_trade["exit_q_values"] = q_dict
                current_trade["exit_raw_action"] = Action(raw_act).name
                current_trade["exit_is_valid"] = pred.is_valid
                current_trade["exit_observations"] = key_obs

        obs, reward, term, trunc, info = env.step(act)

        # Track execution updates from info
        event = info.get("event")
        if event in ("open_long", "open_short") and current_trade is not None:
            current_trade["entry_price"] = float(info.get("entry_price", cur_price))
            current_trade["position_size"] = float(info.get("quantity", 0.0))
            current_trade["margin_used"] = float(info.get("margin", 0.0))
            current_trade["entry_fee"] = float(info.get("fee", 0.0))

        if current_trade is not None and current_trade.get("exit_step") is None:
            current_trade["duration_steps"] += 1
            if "funding_payment" in info:
                current_trade["funding_accrued"] += float(info["funding_payment"])

        # If a close event occurred (policy close or liquidation or daily limit)
        if event in ("close", "liquidation", "stop_loss", "take_profit", "daily_limit_close"):
            if current_trade is not None:
                if current_trade.get("exit_step") is None:
                    # Non-policy close (e.g. liquidation)
                    current_trade["exit_step"] = step_idx
                    current_trade["exit_time"] = cur_time
                    current_trade["exit_action"] = event.upper()
                    current_trade["exit_type"] = event
                    current_trade["exit_q_values"] = q_dict
                    current_trade["exit_raw_action"] = Action(raw_act).name
                    current_trade["exit_is_valid"] = pred.is_valid
                    current_trade["exit_observations"] = key_obs

                current_trade["exit_price"] = float(info.get("exit_price", cur_price))
                current_trade["exit_fee"] = float(info.get("fee", 0.0))
                current_trade["realized_pnl"] = float(info.get("realized_pnl", 0.0))
                current_trade["net_pnl"] = (
                    current_trade["realized_pnl"]
                    - current_trade["entry_fee"]
                    - current_trade["exit_fee"]
                    + current_trade["funding_accrued"]
                )
                current_trade["returned_to_flat"] = (env.env.position is None)
                trades.append(dict(current_trade))

        equity_curve.append(float(env.env.equity))
        done = bool(term or trunc)
        step_idx += 1

    # Episode ended: check if a trade remained open unclosed
    if env.env.position is not None and current_trade is not None:
        last_pos = env.env.position
        unrl_pnl = float(env.env.unrealized_pnl())
        current_trade["exit_step"] = step_idx - 1
        current_trade["exit_time"] = candles[-1].close_time.isoformat() if candles[-1].close_time else None
        current_trade["exit_action"] = "HELD_UNTIL_END"
        current_trade["exit_type"] = "held_until_end"
        current_trade["exit_price"] = float(candles[-1].close)
        current_trade["exit_fee"] = 0.0
        current_trade["realized_pnl"] = unrl_pnl
        current_trade["net_pnl"] = unrl_pnl - current_trade["entry_fee"] + current_trade["funding_accrued"]
        current_trade["returned_to_flat"] = False
        trades.append(dict(current_trade))

    total_steps = max(1, len(actions))
    final_eq = float(env.env.equity)
    tot_ret_pct = (final_eq / init_balance - 1.0) * 100.0
    max_dd = compute_max_drawdown(equity_curve)

    permanently_flat = (len(trades) == 0 and steps_flat == total_steps)
    held_until_end = (len(trades) > 0 and trades[-1]["exit_type"] == "held_until_end")
    policy_completed_trades = [t for t in trades if t["exit_type"] == "policy_close"]

    action_counts = {Action(i).name: int(np.sum(np.array(actions) == i)) for i in range(5)}
    raw_action_counts = {Action(i).name: int(np.sum(np.array(raw_actions) == i)) for i in range(5)}

    return {
        "checkpoint": checkpoint_name,
        "group": group,
        "seed": seed,
        "environment": env_name,
        "total_steps": total_steps,
        "total_return_pct": tot_ret_pct,
        "final_equity": final_eq,
        "max_drawdown_pct": max_dd,
        "total_fees": float(env.env.total_fees),
        "total_funding": float(env.env.total_funding),
        "time_flat_pct": steps_flat / total_steps * 100.0,
        "time_long_pct": steps_long / total_steps * 100.0,
        "time_short_pct": steps_short / total_steps * 100.0,
        "permanently_flat": permanently_flat,
        "held_until_end": held_until_end,
        "total_trades_opened": len(trades),
        "policy_closed_trades_count": len(policy_completed_trades),
        "liquidations_count": sum(1 for t in trades if t["exit_type"] == "liquidation"),
        "raw_invalid_count": raw_invalid_count,
        "raw_invalid_pct": raw_invalid_count / total_steps * 100.0,
        "action_counts": action_counts,
        "raw_action_counts": raw_action_counts,
        "trades": trades,
    }


def main():
    print("=" * 90)
    print("PROJECT NOONE - TRADE LIFECYCLE INVESTIGATION")
    print("=" * 90)

    out_dir = Path("experiments/trade_lifecycle/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Loading Benchmark Datasets...")
    bench_suite = generate_benchmark_suite()
    trans_suite = build_all_transition_sequences()

    eval_datasets = {
        "strong_bull": {
            "candles": bench_suite["eval_suite"]["bull"]["candles"],
            "warmup": bench_suite["eval_suite"]["bull"]["warmup"],
            "description": "Strong Bull (+7.5% over 200 candles)",
        },
        "sustained_bear": {
            "candles": bench_suite["eval_suite"]["bear"]["candles"],
            "warmup": bench_suite["eval_suite"]["bear"]["warmup"],
            "description": "Sustained Bear (-7.5% over 200 candles)",
        },
        "high_vol_sideways": {
            "candles": bench_suite["eval_suite"]["sideways"]["candles"],
            "warmup": bench_suite["eval_suite"]["sideways"]["warmup"],
            "description": "High-Vol Sideways (0.0% chop over 200 candles)",
        },
        "bull_to_bear": {
            "candles": trans_suite["bull_to_bear"]["candles"],
            "warmup": trans_suite["bull_to_bear"]["warmup"],
            "description": "Bull -> Bear (+7.5% then -7.5%, 400 candles)",
        },
        "bear_to_bull": {
            "candles": trans_suite["bear_to_bull"]["candles"],
            "warmup": trans_suite["bear_to_bull"]["warmup"],
            "description": "Bear -> Bull (-7.5% then +7.5%, 400 candles)",
        },
        "composite_regime": {
            "candles": bench_suite["eval_suite"]["composite"]["candles"],
            "warmup": bench_suite["eval_suite"]["composite"]["warmup"],
            "description": "Composite (Bull + Bear + Sideways, 600 candles)",
        },
    }
    print(f"Loaded {len(eval_datasets)} evaluation environments.")

    # Checkpoints to test
    target_checkpoints = [
        # Group 1: Multi-Regime / Synthetic Trained
        ("multi_regime_seed_42", "multi_regime", 42, Path("experiments/multi_regime/models/multi_regime_seed_42/final_model.zip")),
        ("multi_regime_seed_43", "multi_regime", 43, Path("experiments/multi_regime/models/multi_regime_seed_43/final_model.zip")),
        ("trend_only_seed_42", "trend_only", 42, Path("experiments/multi_regime/models/trend_only_seed_42/final_model.zip")),
        ("trend_only_seed_43", "trend_only", 43, Path("experiments/multi_regime/models/trend_only_seed_43/final_model.zip")),
        ("range_only_seed_42", "range_only", 42, Path("experiments/multi_regime/models/range_only_seed_42/final_model.zip")),
        ("range_only_seed_43", "range_only", 43, Path("experiments/multi_regime/models/range_only_seed_43/final_model.zip")),
        # Group 2: Exploration Schedule on Historical BTCUSDT (10k)
        ("Exp_A_10k_seed_42", "exploration_10k", 42, Path("runs/experiments_exploration_schedule/Exp_A_10k_seed_42/final_model.zip")),
        ("Exp_A_10k_seed_43", "exploration_10k", 43, Path("runs/experiments_exploration_schedule/Exp_A_10k_seed_43/final_model.zip")),
        ("Exp_A_10k_seed_44", "exploration_10k", 44, Path("runs/experiments_exploration_schedule/Exp_A_10k_seed_44/final_model.zip")),
        # Group 3: Exploration Schedule on Historical BTCUSDT (25k)
        ("Exp_B_25k_seed_42", "exploration_25k", 42, Path("runs/experiments_exploration_schedule/Exp_B_25k_seed_42/final_model.zip")),
        ("Exp_B_25k_seed_43", "exploration_25k", 43, Path("runs/experiments_exploration_schedule/Exp_B_25k_seed_43/final_model.zip")),
        ("Exp_B_25k_seed_44", "exploration_25k", 44, Path("runs/experiments_exploration_schedule/Exp_B_25k_seed_44/final_model.zip")),
        # Group 4: Reward Scale 100 Baseline
        ("scale_100_seed_42", "reward_scale_100", 42, Path("runs/experiments_reward_scale/scale_100_seed_42/final_model.zip")),
        ("scale_100_seed_43", "reward_scale_100", 43, Path("runs/experiments_reward_scale/scale_100_seed_43/final_model.zip")),
        ("scale_100_seed_44", "reward_scale_100", 44, Path("runs/experiments_reward_scale/scale_100_seed_44/final_model.zip")),
    ]

    trading_cfg = TradingConfig(
        fee_rate=0.0005,
        slippage_rate=0.0002,
        funding_mode="constant",
        funding_rate=0.0001,
        reward_scale=100.0,
        initial_balance=10_000.0,
    )

    print(f"\n[2/3] Evaluating {len(target_checkpoints)} Checkpoints across {len(eval_datasets)} Environments ({len(target_checkpoints)*len(eval_datasets)} runs)...")

    results_by_checkpoint: dict[str, dict[str, Any]] = {}
    all_runs_summary: list[dict[str, Any]] = []

    for c_name, group, seed, m_path in target_checkpoints:
        if not m_path.exists():
            print(f"Skipping {c_name}: path {m_path} does not exist.")
            continue

        print(f"\n--- Checkpoint: {c_name} (Group: {group}, Seed: {seed}) ---")
        try:
            inference = ModelInference.load(m_path, allow_legacy_metadata=True)
        except Exception as e:
            print(f"Error loading {c_name}: {e}")
            continue

        results_by_checkpoint[c_name] = {}

        for env_name, env_data in eval_datasets.items():
            run_res = evaluate_checkpoint_lifecycle(
                inference=inference,
                checkpoint_name=c_name,
                group=group,
                seed=seed,
                env_name=env_name,
                candles=env_data["candles"],
                warmup_context=env_data["warmup"],
                trading_cfg=trading_cfg,
            )
            results_by_checkpoint[c_name][env_name] = run_res

            # Summary item
            n_trades = run_res["total_trades_opened"]
            n_policy_close = run_res["policy_closed_trades_count"]
            held = run_res["held_until_end"]
            flat = run_res["permanently_flat"]
            status_str = "FLAT" if flat else (f"{n_trades} trades ({n_policy_close} policy closed)" + (" [HELD TO END]" if held else ""))
            print(f"  [{env_name:<18}] Ret: {run_res['total_return_pct']:>+6.2f}% | MaxDD: {run_res['max_drawdown_pct']:>5.2f}% | Status: {status_str}")

            all_runs_summary.append({
                "checkpoint": c_name,
                "group": group,
                "seed": seed,
                "environment": env_name,
                "total_return_pct": run_res["total_return_pct"],
                "max_drawdown_pct": run_res["max_drawdown_pct"],
                "permanently_flat": flat,
                "held_until_end": held,
                "trades_opened": n_trades,
                "policy_closed": n_policy_close,
                "time_flat_pct": run_res["time_flat_pct"],
                "time_long_pct": run_res["time_long_pct"],
                "time_short_pct": run_res["time_short_pct"],
                "raw_invalid_count": run_res["raw_invalid_count"],
            })

    # Save full detailed results
    payload = {
        "generated_at": datetime.now().isoformat(),
        "summary": all_runs_summary,
        "detailed_results": results_by_checkpoint,
    }

    res_file = out_dir / "lifecycle_evaluation_results.json"
    with open(res_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[3/3] Saved lifecycle results to {res_file}")


if __name__ == "__main__":
    main()

