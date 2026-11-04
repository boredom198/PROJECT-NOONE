"""Regime-Transition Evaluation and Reversal Mechanism Analysis.

PROJECT NOONE - Component 1 (DQN + RL Engine)
Investigates dynamic regime switching across 6 trained checkpoints:
- multi_regime (seeds 42, 43)
- trend_only (seeds 42, 43)
- range_only (seeds 42, 43)

Across 6 pairwise transition sequences:
1. Bull -> Bear
2. Bear -> Bull
3. Bull -> Sideways
4. Bear -> Sideways
5. Sideways -> Bull
6. Sideways -> Bear

Evaluates:
- Complete episode performance (Return, Equity, MaxDD, Fees, Funding, Trades)
- Phase 1 (Pre-transition) vs Phase 2 (Post-transition) exposure and returns
- Reversal dynamics: position at boundary, time to close, time to open opposite
- Q-value trajectory and advantage across the regime boundary (steps 195-205)
- Baselines: Always HOLD, Buy-and-Hold, Simple Momentum
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


def evaluate_transition_run(
    *,
    inference: ModelInference,
    seq_data: dict[str, Any],
    trading_cfg: TradingConfig,
    seed: int = 42,
) -> dict[str, Any]:
    warmup_context = seq_data["warmup"]
    eval_candles = seq_data["candles"]
    transition_idx = seq_data["transition_index"]  # 200

    full_candles = list(warmup_context) + list(eval_candles)
    env = NooneGymEnv(full_candles, trading_cfg, feature_builder=inference.feature_builder)
    obs, _ = env.reset(seed=seed, options={"start_index": 0})

    # Warmup phase (HOLD for all warmup candles)
    for _ in range(len(warmup_context)):
        obs, _, term, trunc, _ = env.step(0)

    # Evaluation phase
    step_records: list[dict[str, Any]] = []
    actions: list[int] = []
    raw_actions: list[int] = []
    raw_invalid_count = 0
    trade_pnls: list[float] = []
    equity_curve: list[float] = [float(env.env.equity)]

    # Tracking per phase
    p1_steps_flat = 0
    p1_steps_long = 0
    p1_steps_short = 0
    p2_steps_flat = 0
    p2_steps_long = 0
    p2_steps_short = 0

    p1_q_vals: list[dict[str, float]] = []
    p2_q_vals: list[dict[str, float]] = []

    pos_at_boundary: int = 0
    equity_at_boundary: float = float(env.env.equity)

    done = False
    step_idx = 0
    while not done:
        pos = env.env.position
        pos_side = 0 if pos is None else pos.side
        pos_qty = 0.0 if pos is None else pos.quantity
        unrealized_pnl = float(env.env.unrealized_pnl())
        cur_eq = float(env.env.equity)

        # Record boundary state exactly before step 200 action
        if step_idx == transition_idx:
            pos_at_boundary = pos_side
            equity_at_boundary = cur_eq

        # Phase tracking
        if step_idx < transition_idx:
            if pos_side == 0:
                p1_steps_flat += 1
            elif pos_side == 1:
                p1_steps_long += 1
            elif pos_side == -1:
                p1_steps_short += 1
        else:
            if pos_side == 0:
                p2_steps_flat += 1
            elif pos_side == 1:
                p2_steps_long += 1
            elif pos_side == -1:
                p2_steps_short += 1

        pred = inference.predict_observation(obs, position_state=pos_side)
        if not pred.is_valid:
            raw_invalid_count += 1

        act = int(pred.action.value)
        raw_act = int(pred.raw_action.value)
        actions.append(act)
        raw_actions.append(raw_act)

        q_dict = {a.name: float(pred.q_values[a.name]) for a in Action}
        if step_idx < transition_idx:
            p1_q_vals.append(q_dict)
        else:
            p2_q_vals.append(q_dict)

        # Record step telemetry (especially boundary window 190-210)
        c_price = eval_candles[step_idx].close
        step_rec = {
            "step": step_idx,
            "regime": "R1" if step_idx < transition_idx else "R2",
            "price": c_price,
            "pos_side": pos_side,
            "pos_qty": pos_qty,
            "unrealized_pnl": unrealized_pnl,
            "equity": cur_eq,
            "action": Action(act).name,
            "raw_action": Action(raw_act).name,
            "is_valid": pred.is_valid,
            "q_values": q_dict,
        }
        step_records.append(step_rec)

        obs, reward, term, trunc, info = env.step(act)
        if info.get("event") in ("close", "liquidation", "daily_limit_close") or "realized_pnl" in info:
            trade_pnls.append(float(info.get("realized_pnl", 0.0)))

        equity_curve.append(float(env.env.equity))
        done = bool(term or trunc)
        step_idx += 1

    total_steps = max(1, len(actions))
    init_bal = trading_cfg.initial_balance
    final_eq = float(env.env.equity)
    tot_ret_pct = (final_eq / init_bal - 1.0) * 100.0

    r1_steps = transition_idx
    r2_steps = total_steps - transition_idx
    r1_return_pct = (equity_at_boundary / init_bal - 1.0) * 100.0
    r2_return_pct = (final_eq / equity_at_boundary - 1.0) * 100.0 if equity_at_boundary > 0 else 0.0

    max_dd = compute_max_drawdown(equity_curve)

    winning_trades = [p for p in trade_pnls if p > 0]
    losing_trades = [p for p in trade_pnls if p < 0]
    win_rate = (len(winning_trades) / len(trade_pnls) * 100.0) if trade_pnls else 0.0
    avg_win = float(np.mean(winning_trades)) if winning_trades else 0.0
    avg_loss = float(np.mean(losing_trades)) if losing_trades else 0.0

    action_names = [a.name for a in Action]
    p1_mean_q = {
        name: float(np.mean([q[name] for q in p1_q_vals])) if p1_q_vals else 0.0
        for name in action_names
    }
    p2_mean_q = {
        name: float(np.mean([q[name] for q in p2_q_vals])) if p2_q_vals else 0.0
        for name in action_names
    }

    # Action counts per phase
    p1_actions = actions[:transition_idx]
    p2_actions = actions[transition_idx:]
    p1_act_counts = {Action(i).name: int(np.sum(np.array(p1_actions) == i)) for i in range(5)}
    p2_act_counts = {Action(i).name: int(np.sum(np.array(p2_actions) == i)) for i in range(5)}

    # Adaptation / Reversal Dynamics:
    # 1. time_to_close: steps after step 200 until initial position is closed
    time_to_close: int | None = None
    if pos_at_boundary != 0:
        for offset, rec in enumerate(step_records[transition_idx:], start=1):
            if rec["pos_side"] == 0:
                time_to_close = offset
                break

    # 2. time_to_open_opposite: steps after step 200 until opposite position is opened
    time_to_open_opposite: int | None = None
    opposite_side = -pos_at_boundary if pos_at_boundary != 0 else (1 if "bull" in seq_data["regime_2"] else -1)
    for offset, rec in enumerate(step_records[transition_idx:], start=1):
        if rec["pos_side"] == opposite_side:
            time_to_open_opposite = offset
            break

    # Boundary window telemetry (steps 195 to 205)
    boundary_window = step_records[max(0, transition_idx - 5) : min(len(step_records), transition_idx + 6)]

    return {
        "sequence": seq_data["name"],
        "regime_1": seq_data["regime_1"],
        "regime_2": seq_data["regime_2"],
        "total_return_pct": tot_ret_pct,
        "final_equity": final_eq,
        "max_drawdown_pct": max_dd,
        "completed_trades": len(trade_pnls),
        "win_rate_pct": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "total_fees": float(env.env.total_fees),
        "total_funding": float(env.env.total_funding),
        "raw_invalid_count": raw_invalid_count,
        "raw_invalid_pct": raw_invalid_count / total_steps * 100.0,
        # Reversal dynamics
        "pos_at_boundary": pos_at_boundary,
        "time_to_close": time_to_close,
        "time_to_open_opposite": time_to_open_opposite,
        # Phase 1 metrics
        "p1_return_pct": r1_return_pct,
        "p1_flat_pct": p1_steps_flat / r1_steps * 100.0,
        "p1_long_pct": p1_steps_long / r1_steps * 100.0,
        "p1_short_pct": p1_steps_short / r1_steps * 100.0,
        "p1_actions": p1_act_counts,
        "p1_mean_q": p1_mean_q,
        # Phase 2 metrics
        "p2_return_pct": r2_return_pct,
        "p2_flat_pct": p2_steps_flat / r2_steps * 100.0,
        "p2_long_pct": p2_steps_long / r2_steps * 100.0,
        "p2_short_pct": p2_steps_short / r2_steps * 100.0,
        "p2_actions": p2_act_counts,
        "p2_mean_q": p2_mean_q,
        # Boundary window
        "boundary_window": boundary_window,
    }


def evaluate_transition_baseline(
    *,
    baseline_type: str,
    seq_data: dict[str, Any],
    trading_cfg: TradingConfig,
    seed: int = 42,
) -> dict[str, Any]:
    warmup_context = seq_data["warmup"]
    eval_candles = seq_data["candles"]
    transition_idx = seq_data["transition_index"]

    full_candles = list(warmup_context) + list(eval_candles)
    env = NooneGymEnv(full_candles, trading_cfg)
    obs, _ = env.reset(seed=seed, options={"start_index": 0})
    for _ in range(len(warmup_context)):
        obs, _, term, trunc, _ = env.step(0)

    actions: list[int] = []
    trade_pnls: list[float] = []
    equity_curve: list[float] = [float(env.env.equity)]

    p1_steps_flat = 0
    p1_steps_long = 0
    p1_steps_short = 0
    p2_steps_flat = 0
    p2_steps_long = 0
    p2_steps_short = 0

    pos_at_boundary = 0
    equity_at_boundary = float(env.env.equity)

    done = False
    step_idx = 0
    while not done:
        pos = env.env.position
        pos_side = 0 if pos is None else pos.side

        if step_idx == transition_idx:
            pos_at_boundary = pos_side
            equity_at_boundary = float(env.env.equity)

        if step_idx < transition_idx:
            if pos_side == 0:
                p1_steps_flat += 1
            elif pos_side == 1:
                p1_steps_long += 1
            elif pos_side == -1:
                p1_steps_short += 1
        else:
            if pos_side == 0:
                p2_steps_flat += 1
            elif pos_side == 1:
                p2_steps_long += 1
            elif pos_side == -1:
                p2_steps_short += 1

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

        equity_curve.append(float(env.env.equity))
        done = bool(term or trunc)
        step_idx += 1

    total_steps = max(1, len(actions))
    init_bal = trading_cfg.initial_balance
    final_eq = float(env.env.equity)
    tot_ret_pct = (final_eq / init_bal - 1.0) * 100.0

    r1_steps = transition_idx
    r2_steps = total_steps - transition_idx
    r1_return_pct = (equity_at_boundary / init_bal - 1.0) * 100.0
    r2_return_pct = (final_eq / equity_at_boundary - 1.0) * 100.0 if equity_at_boundary > 0 else 0.0

    max_dd = compute_max_drawdown(equity_curve)

    winning_trades = [p for p in trade_pnls if p > 0]
    losing_trades = [p for p in trade_pnls if p < 0]
    win_rate = (len(winning_trades) / len(trade_pnls) * 100.0) if trade_pnls else 0.0
    avg_win = float(np.mean(winning_trades)) if winning_trades else 0.0
    avg_loss = float(np.mean(losing_trades)) if losing_trades else 0.0

    p1_act_counts = {Action(i).name: int(np.sum(np.array(actions[:transition_idx]) == i)) for i in range(5)}
    p2_act_counts = {Action(i).name: int(np.sum(np.array(actions[transition_idx:]) == i)) for i in range(5)}

    return {
        "baseline": baseline_type,
        "sequence": seq_data["name"],
        "total_return_pct": tot_ret_pct,
        "final_equity": final_eq,
        "max_drawdown_pct": max_dd,
        "completed_trades": len(trade_pnls),
        "win_rate_pct": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "total_fees": float(env.env.total_fees),
        "total_funding": float(env.env.total_funding),
        "pos_at_boundary": pos_at_boundary,
        "p1_return_pct": r1_return_pct,
        "p1_flat_pct": p1_steps_flat / r1_steps * 100.0,
        "p1_long_pct": p1_steps_long / r1_steps * 100.0,
        "p1_short_pct": p1_steps_short / r1_steps * 100.0,
        "p1_actions": p1_act_counts,
        "p2_return_pct": r2_return_pct,
        "p2_flat_pct": p2_steps_flat / r2_steps * 100.0,
        "p2_long_pct": p2_steps_long / r2_steps * 100.0,
        "p2_short_pct": p2_steps_short / r2_steps * 100.0,
        "p2_actions": p2_act_counts,
    }


def main():
    print("=" * 80)
    print("PROJECT NOONE - REGIME TRANSITION & REVERSAL BENCHMARK")
    print("=" * 80)

    out_dir = Path("experiments/multi_regime/results")
    model_dir = Path("experiments/multi_regime/models")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Generating 6 Controlled Transition Sequences (400 candles each)...")
    transition_suite = build_all_transition_sequences()
    print(f"Generated {len(transition_suite)} transition sequences successfully.")

    trading_cfg = TradingConfig(
        fee_rate=0.0005,
        slippage_rate=0.0002,
        funding_mode="constant",
        funding_rate=0.0001,
        reward_scale=100.0,
        initial_balance=10_000.0,
    )

    # 1. Run Baselines across all 6 transition sequences
    print("\n[2/3] Evaluating Baselines (Always HOLD, Buy-and-Hold, Momentum)...")
    baseline_results: dict[str, dict[str, Any]] = {}
    for seq_name, seq_data in transition_suite.items():
        baseline_results[seq_name] = {}
        for b_name in ["always_hold", "buy_and_hold", "momentum"]:
            res = evaluate_transition_baseline(
                baseline_type=b_name,
                seq_data=seq_data,
                trading_cfg=trading_cfg,
            )
            baseline_results[seq_name][b_name] = res

    # 2. Evaluate Trained Models
    print("\n[3/3] Evaluating 6 Trained Models across all 6 Transition Sequences...")
    model_names = [
        "multi_regime_seed_42",
        "multi_regime_seed_43",
        "trend_only_seed_42",
        "trend_only_seed_43",
        "range_only_seed_42",
        "range_only_seed_43",
    ]

    model_results: dict[str, dict[str, Any]] = {}
    boundary_telemetry: dict[str, dict[str, Any]] = {}

    for m_name in model_names:
        m_path = model_dir / m_name / "final_model.zip"
        if not m_path.exists():
            print(f"Warning: {m_path} does not exist!")
            continue

        print(f"\n--- Loading and Evaluating Model: {m_name} ---")
        inference = ModelInference.load(m_path)
        model_results[m_name] = {}
        boundary_telemetry[m_name] = {}

        for seq_name, seq_data in transition_suite.items():
            run_res = evaluate_transition_run(
                inference=inference,
                seq_data=seq_data,
                trading_cfg=trading_cfg,
                seed=42,
            )
            # Separate boundary window for clean serialization
            b_win = run_res.pop("boundary_window")
            model_results[m_name][seq_name] = run_res
            boundary_telemetry[m_name][seq_name] = b_win

            # Quick progress log
            pos_b = run_res["pos_at_boundary"]
            pos_str = "FLAT" if pos_b == 0 else ("LONG" if pos_b == 1 else "SHORT")
            t_close = run_res["time_to_close"]
            t_opp = run_res["time_to_open_opposite"]
            print(
                f"  [{seq_name:<18}] Ret={run_res['total_return_pct']:+6.2f}% | MaxDD={run_res['max_drawdown_pct']:5.2f}% | "
                f"BoundPos={pos_str:<5} | TimeToClose={str(t_close):<5} | TimeToOpp={str(t_opp):<5} | "
                f"P1(L={run_res['p1_long_pct']:3.0f}%,S={run_res['p1_short_pct']:3.0f}%) -> P2(L={run_res['p2_long_pct']:3.0f}%,S={run_res['p2_short_pct']:3.0f}%)"
            )

    # Save structured results
    final_payload = {
        "generated_at": datetime.now().isoformat(),
        "trading_config": {
            "fee_rate": trading_cfg.fee_rate,
            "slippage_rate": trading_cfg.slippage_rate,
            "funding_rate": trading_cfg.funding_rate,
            "reward_scale": trading_cfg.reward_scale,
            "initial_balance": trading_cfg.initial_balance,
        },
        "baselines": baseline_results,
        "models": model_results,
    }

    out_file = out_dir / "transition_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\n[Saved] Transition evaluation results to {out_file}")

    telemetry_file = out_dir / "boundary_telemetry.json"
    with open(telemetry_file, "w", encoding="utf-8") as f:
        json.dump(boundary_telemetry, f, indent=2)
    print(f"[Saved] Boundary telemetry to {telemetry_file}")


if __name__ == "__main__":
    main()

