"""Evaluation CLI: ``python -m noone.rl_engine.evaluate`` (Stage 2).

Loads a saved DQN model and evaluates it deterministically on the
held-out (later, disjoint) chronological slice only. No training,
no weight updates, no live/paper trading or order execution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from noone.rl_engine.config import DataConfig
from noone.rl_engine.data import (
    chronological_split,
    fetch_candles,
    fetch_funding_history,
    load_candles_csv,
    prepare_candles,
)
from noone.rl_engine.train import _parse_time
from noone.trading_env import TradingConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate NOONE DQN on held-out data")
    parser.add_argument("--model", required=True, help="Saved model .zip")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--total-limit", type=int, default=5000)
    parser.add_argument("--start-time", default=None)
    parser.add_argument("--end-time", default=None)
    parser.add_argument("--candles-csv", default=None)
    parser.add_argument("--news-archive", default=None)
    parser.add_argument("--news-asset", default=None)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="JSON results path (default: <model dir>/evaluation.json)")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--allow-legacy-metadata", action="store_true",
                        help="Allow legacy model metadata without feature schema hash")
    parser.add_argument("--funding-mode", choices=("constant", "historical"),
                        default="constant",
                        help="Funding accounting mode (default constant)")
    args = parser.parse_args(argv)

    if args.candles_csv:
        candles = load_candles_csv(args.candles_csv, symbol=args.symbol,
                                   interval=args.interval)
    else:
        candles = fetch_candles(
            symbol=args.symbol, interval=args.interval,
            total_limit=args.total_limit,
            start_time=_parse_time(args.start_time),
            end_time=_parse_time(args.end_time))
    data_config = DataConfig(symbol=args.symbol, interval=args.interval,
                             train_fraction=args.train_fraction,
                             news_asset=args.news_asset,
                             news_dropout_probability=0.0)
    prepared = prepare_candles(candles, interval=args.interval, symbol=args.symbol,
                               warmup_candles=data_config.warmup_candles)
    split = chronological_split(
        prepared, train_fraction=args.train_fraction,
        warmup_candles=data_config.warmup_candles,
        min_train_candles=data_config.min_train_candles,
        min_eval_candles=data_config.min_eval_candles)

    # Historical funding must cover warm-up plus evaluation candles.
    funding_history = None
    if args.funding_mode == "historical":
        combined = list(split.eval_warmup_context) + list(split.evaluation)
        funding_history = fetch_funding_history(
            symbol=args.symbol, start_time=combined[0].open_time,
            end_time=combined[-1].close_time)

    archive = None
    try:
        if args.news_archive:
            from noone.news_data import NewsArchive

            archive = NewsArchive(args.news_archive)
        from noone.rl_engine.evaluation import evaluate_model, save_evaluation

        results = evaluate_model(
            args.model, split.evaluation,
            warmup_context=split.eval_warmup_context,
            trading_config=TradingConfig(funding_mode=args.funding_mode),
            data_config=data_config,
            news_archive=archive, seed=args.seed,
            include_baselines=not args.no_baselines,
            funding_history=funding_history,
            allow_legacy_metadata=args.allow_legacy_metadata)
    finally:
        if archive is not None:
            archive.close()
    out = Path(args.output) if args.output else Path(args.model).parent / "evaluation.json"
    save_evaluation(results, out)
    print(json.dumps(results["metrics"], indent=2))
    print(f"Evaluation range: {results['evaluation_start']} .. {results['evaluation_end']}")
    print(f"Results saved to: {out}")
    print("Historical backtest results do not guarantee future performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
