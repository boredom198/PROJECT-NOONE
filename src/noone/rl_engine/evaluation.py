"""Deterministic held-out evaluation for Stage 2 (Component 1).

Evaluation runs only on the later, disjoint evaluation slice, uses
deterministic predictions, and never updates model weights (a
parameter checksum is taken before/after and reported). Accounting,
fees, slippage and funding all come from Component 8 via the Stage 1
adapter. Warm-up context candles are stepped with forced HOLD only:
they are never model decisions and are excluded from metrics and
the reported evaluation date range.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable, Sequence

from noone.market_data.models import Kline
from noone.rl_engine.config import DataConfig
from noone.rl_engine.training import DISCLAIMER, _require_sb3, make_env
from noone.trading_env import TradingConfig

_CLOSE_EVENTS = {"close", "liquidation", "stop_loss", "take_profit",
                 "daily_limit_close"}


def compute_metrics(
    equity_curve: Sequence[float], *, initial_balance: float,
    trade_pnls: Sequence[float], total_fees: float, total_funding: float,
    final_balance: float,
) -> dict[str, Any]:
    """Pure metrics from an equity curve and closed-trade PnLs."""
    if initial_balance <= 0:
        raise ValueError("initial_balance must be positive")
    equities = [float(e) for e in equity_curve] or [float(initial_balance)]
    final_equity = equities[-1]
    peak, max_dd = equities[0], 0.0
    for value in equities:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, (peak - value) / peak)
    trades = list(trade_pnls)
    wins = sum(1 for pnl in trades if pnl > 0)
    return {
        "total_return": final_equity / initial_balance - 1.0,
        "max_drawdown": max_dd,
        "num_trades": len(trades),
        "win_rate": (wins / len(trades)) if trades else 0.0,
        "total_fees": float(total_fees),
        # Component 8 total_funding is signed cash received (+ credit).
        "total_funding": float(total_funding),
        "funding_cost": float(max(0.0, -total_funding)),
        "funding_credit": float(max(0.0, total_funding)),
        "final_balance": float(final_balance),
        "final_equity": float(final_equity),
        "initial_balance": float(initial_balance),
    }


def _weights_checksum(model: Any) -> str | None:
    try:
        import torch

        state = model.policy.state_dict()
        digest = hashlib.sha256()
        for key in sorted(state):
            digest.update(key.encode())
            digest.update(state[key].detach().cpu().numpy().tobytes())
        return digest.hexdigest()
    except Exception:
        return None


def _funding_summary(
    trading_env: Any, trading_config: TradingConfig | None,
    funding_history: Any, candles: Sequence[Kline],
    warmup: Sequence[Kline],
) -> dict[str, Any]:
    """Per-episode funding provenance and ledger summary (Stage 3)."""
    from decimal import Decimal

    config = trading_config or TradingConfig()
    historical = config.funding_mode == "historical"
    ledger = list(trading_env.funding_ledger)
    ledger_total = sum(
        (entry.funding_payment for entry in ledger), Decimal("0"))
    reconciliation = "not_applicable_constant_mode"
    if historical:
        trading_env.reconcile_historical_funding()
        reconciliation = "passed"
    return {
        "funding_mode": config.funding_mode,
        "historical_accounting_implemented": historical,
        "funding_history": (funding_history.metadata()
                            if funding_history is not None else None),
        "evaluation_range": {
            "start": candles[0].open_time.isoformat() if candles else None,
            "end": (candles[-1].close_time.isoformat()
                    if candles and candles[-1].close_time else None),
        },
        "warmup_range": {
            "start": warmup[0].open_time.isoformat() if warmup else None,
            "end": (warmup[-1].close_time.isoformat()
                    if warmup and warmup[-1].close_time else None),
            "candle_count": len(warmup),
        },
        "ledger_entry_count": len(ledger),
        "ledger_funding_total": float(ledger_total),
        "ledger_funding_total_decimal": str(ledger_total),
        "total_funding": float(trading_env.total_funding),
        "reconciliation_status": reconciliation,
    }


def _run_episode(
    candles: Sequence[Kline], warmup: Sequence[Kline],
    action_fn: Callable[[Any], int], *, trading_config: TradingConfig | None,
    data_config: DataConfig, news_archive: Any, seed: int,
    funding_history: Any = None,
) -> dict[str, Any]:
    env_candles = list(warmup) + list(candles)
    env = make_env(env_candles, trading_config=trading_config,
                   data_config=data_config, news_archive=news_archive,
                   seed=seed, funding_history=funding_history)
    obs, _info = env.reset(seed=seed, options={"start_index": 0})
    equity_curve: list[float] = []
    trade_pnls: list[float] = []
    actions: list[int] = []
    # Warm-up context: forced HOLD, unrecorded, no position can exist.
    for _ in range(len(warmup)):
        obs, _r, terminated, truncated, _i = env.step(0)
        if terminated or truncated:
            break
    done = False
    while not done:
        pos_side = int(env.env.position.side if env.env.position else 0)
        try:
            action = int(action_fn(obs, pos_side))
        except TypeError:
            action = int(action_fn(obs))
        actions.append(action)
        obs, _reward, terminated, truncated, info = env.step(action)
        raw = info.get("raw_observation") or {}
        equity_curve.append(float(raw.get("equity", env.env.equity)))
        if info.get("event") in _CLOSE_EVENTS and "realized_pnl" in info:
            trade_pnls.append(float(info["realized_pnl"]))
        done = bool(terminated or truncated)
    trading_env = env.env
    metrics = compute_metrics(
        equity_curve, initial_balance=trading_env.config.initial_balance,
        trade_pnls=trade_pnls, total_fees=trading_env.total_fees,
        total_funding=trading_env.total_funding, final_balance=trading_env.balance)
    return {"metrics": metrics, "equity_curve": equity_curve,
            "trade_pnls": trade_pnls, "actions": actions,
            "final_balance": trading_env.balance,
            "final_equity": trading_env.equity,
            "funding": _funding_summary(
                trading_env, trading_config, funding_history,
                candles, warmup)}


def evaluate_model(
    model_or_path: Any, eval_candles: Sequence[Kline], *,
    warmup_context: Sequence[Kline] = (),
    trading_config: TradingConfig | None = None,
    data_config: DataConfig | None = None, news_archive: Any = None,
    seed: int = 42, include_baselines: bool = True,
    funding_history: Any = None,
    allow_legacy_metadata: bool = False,
    expected_net_arch: tuple[int, ...] | list[int] | None = None,
) -> dict[str, Any]:
    """Deterministically evaluate a model on held-out candles only."""
    if len(eval_candles) < 2:
        raise ValueError("At least 2 evaluation candles are required")
    data_config = data_config or DataConfig()
    _sb3, dqn_cls, _cb = _require_sb3()
    inference = None
    if isinstance(model_or_path, (str, Path)):
        from noone.rl_engine.inference import ModelInference

        inference = ModelInference.load(
            str(model_or_path),
            allow_legacy_metadata=allow_legacy_metadata,
            expected_net_arch=expected_net_arch,
        )
        model = inference.model
    elif hasattr(model_or_path, "model") and hasattr(model_or_path, "predict_observation"):
        inference = model_or_path
        model = inference.model
    else:
        model = model_or_path
    before = _weights_checksum(model)

    invalid_predictions_count = 0
    raw_actions: list[int] = []

    def model_action(obs: Any, pos_side: int | None = None) -> int:
        nonlocal invalid_predictions_count
        if inference is not None:
            pred = inference.predict_observation(obs, position_state=pos_side)
            if not pred.is_valid:
                invalid_predictions_count += 1
            raw_actions.append(int(pred.raw_action.value))
            return int(pred.action.value)
        action, _state = model.predict(obs, deterministic=True)
        raw_act = int(action)
        raw_actions.append(raw_act)
        if pos_side is None:
            return raw_act
        from noone.rl_engine.inference import get_valid_action_mask
        mask = get_valid_action_mask(pos_side)
        if mask[raw_act]:
            return raw_act
        invalid_predictions_count += 1
        if hasattr(model, "policy") and hasattr(model.policy, "q_net"):
            from noone.rl_engine.inference import _require_torch
            _torch = _require_torch()
            with _torch.no_grad():
                q_out = model.policy.q_net(
                    _torch.as_tensor(obs, dtype=_torch.float32).unsqueeze(0)
                ).squeeze(0).tolist()
            valid_indices = [i for i in range(5) if mask[i]]
            return max(valid_indices, key=lambda i: q_out[i])
        valid_indices = [i for i in range(5) if mask[i]]
        return valid_indices[0]

    result = _run_episode(
        eval_candles, warmup_context, model_action,
        trading_config=trading_config, data_config=data_config,
        news_archive=news_archive, seed=seed,
        funding_history=funding_history)
    result["invalid_predictions_count"] = invalid_predictions_count
    result["raw_actions"] = raw_actions
    after = _weights_checksum(model)
    baselines: dict[str, Any] = {}
    baselines_funding: dict[str, Any] = {}
    if include_baselines:
        hold_run = _run_episode(
            eval_candles, warmup_context, lambda _obs: 0,
            trading_config=trading_config, data_config=data_config,
            news_archive=news_archive, seed=seed,
            funding_history=funding_history)
        baselines["always_hold"] = hold_run["metrics"]
        baselines_funding["always_hold"] = hold_run["funding"]
        rng = random.Random(seed)

        def random_action(_obs: Any) -> int:
            return rng.randrange(5)

        random_run = _run_episode(
            eval_candles, warmup_context, random_action,
            trading_config=trading_config, data_config=data_config,
            news_archive=news_archive, seed=seed,
            funding_history=funding_history)
        baselines["random_fixed_seed"] = random_run["metrics"]
        baselines_funding["random_fixed_seed"] = random_run["funding"]
    return {
        "disclaimer": DISCLAIMER,
        "evaluation_start": eval_candles[0].open_time.isoformat() if eval_candles else None,
        "evaluation_end": eval_candles[-1].close_time.isoformat() if eval_candles else None,
        "evaluation_candle_count": len(eval_candles),
        "warmup_context_count": len(warmup_context),
        "deterministic": True,
        "weights_checksum_before": before,
        "weights_checksum_after": after,
        "weights_changed": (before != after) if before and after else None,
        "metrics": result["metrics"],
        "equity_curve": result["equity_curve"],
        "trade_pnls": result["trade_pnls"],
        "actions": result["actions"],
        "invalid_predictions_count": result["invalid_predictions_count"],
        "raw_actions": result["raw_actions"],
        "baselines": baselines,
        "funding": result["funding"],
        "baselines_funding": baselines_funding,
    }


def save_evaluation(results: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return out
