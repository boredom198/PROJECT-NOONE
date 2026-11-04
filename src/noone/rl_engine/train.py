"""Training CLI: ``python -m noone.rl_engine.train`` (Stage 2).

Fetches closed Binance USD-M Futures candles via Component 4 (or
loads ``--candles-csv``), prepares/splits them chronologically, and
trains SB3 DQN on the training slice only. No live/paper trading,
no order execution, no credentials.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from noone.rl_engine.config import DataConfig, DQNConfig, PipelineConfig
from noone.rl_engine.data import (
    DataPreparationError,
    fetch_candles,
    fetch_funding_history,
    load_candles_csv,
    load_funding_history,
)
from noone.trading_env import TradingConfig


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _load_resume_funding_history(resume_from: str):
    """Load the exact persisted FundingHistory for a historical resume.

    Locates the checkpoint's run metadata, resolves the artifact it
    references relative to the run directory, verifies it via
    ``load_funding_history`` (schema and checksum), and returns the
    reconstructed original history. Fresh collection is never used
    as a substitute for a missing or invalid artifact.
    """
    import json
    from pathlib import Path

    checkpoint = Path(resume_from)
    candidates = [checkpoint.parent / "metadata.json",
                  checkpoint.parent.parent / "metadata.json"]
    metadata_path = next((p for p in candidates if p.exists()), None)
    if metadata_path is None:
        raise DataPreparationError(
            f"Cannot resume historical training from {resume_from}: "
            "no metadata.json found beside the checkpoint or in its "
            "run directory; the original FundingHistory cannot be "
            "located. Start a new historical run instead.")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataPreparationError(
            f"Cannot resume historical training: run metadata at "
            f"{metadata_path} is unreadable: {exc}") from exc
    funding = metadata.get("funding") if isinstance(metadata, dict) else None
    if not isinstance(funding, dict) or funding.get("funding_mode") != "historical":
        raise DataPreparationError(
            "Cannot resume historical training: the checkpoint has "
            "no historical funding provenance. If this is a legacy "
            "historical checkpoint, its original FundingHistory was "
            "never persisted and cannot be reconstructed from "
            "metadata alone; start a new historical run instead.")
    artifact_rel = funding.get("artifact_path")
    if not artifact_rel:
        raise DataPreparationError(
            "Cannot resume historical training: the run metadata "
            "does not reference a persisted FundingHistory artifact. "
            "The original history cannot be reconstructed from "
            "metadata alone; start a new historical run instead.")
    if not isinstance(artifact_rel, str) or Path(artifact_rel).is_absolute():
        raise DataPreparationError(
            "Cannot resume historical training: the run metadata "
            f"artifact_path {artifact_rel!r} is not a relative path; "
            "absolute artifact paths are not allowed. The artifact "
            "must remain inside the run directory.")
    run_dir = metadata_path.parent.resolve()
    artifact_path = (run_dir / artifact_rel).resolve()
    if not artifact_path.is_relative_to(run_dir):
        raise DataPreparationError(
            "Cannot resume historical training: the persisted "
            f"FundingHistory artifact path {artifact_rel!r} resolves "
            f"to {artifact_path}, which escapes the run directory "
            f"{run_dir}. Refusing to load an artifact outside the "
            "run directory.")
    if not artifact_path.exists():
        raise DataPreparationError(
            f"Cannot resume historical training: the persisted "
            f"FundingHistory artifact is missing at {artifact_path}. "
            "It will not be replaced with newly collected history; "
            "restore the artifact or start a new historical run.")
    history = load_funding_history(artifact_path)
    expected_checksum = funding.get("artifact_checksum")
    if expected_checksum:
        from noone.rl_engine.data import (
            funding_history_to_payload,
            _payload_checksum,
        )

        actual = _payload_checksum(funding_history_to_payload(history))
        if actual != expected_checksum:
            raise DataPreparationError(
                "Cannot resume historical training: the persisted "
                "FundingHistory artifact checksum does not match the "
                "run metadata; the artifact is corrupted or belongs "
                "to a different run.")
    return history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train NOONE DQN (CPU, historical only)")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--total-limit", type=int, default=5000)
    parser.add_argument("--start-time", default=None)
    parser.add_argument("--end-time", default=None)
    parser.add_argument("--candles-csv", default=None,
                        help="Offline CSV instead of fetching via Component 4")
    parser.add_argument("--news-archive", default=None,
                        help="Path to Component 2 SQLite news archive")
    parser.add_argument("--news-asset", default=None)
    parser.add_argument("--news-dropout", type=float, default=0.1)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--resume", default=None, help="Checkpoint .zip to resume from")
    parser.add_argument("--episode-window", type=int, default=None)
    parser.add_argument("--funding-mode", choices=("constant", "historical"),
                        default="constant",
                        help="Funding accounting mode (default constant)")
    parser.add_argument("--smoke", action="store_true",
                        help="Short pipeline-validation run (256 timesteps)")
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
    timesteps = 256 if args.smoke else args.timesteps
    config = PipelineConfig(
        dqn=DQNConfig(total_timesteps=timesteps, seed=args.seed,
                      checkpoint_freq=(64 if args.smoke else args.checkpoint_freq)),
        data=DataConfig(symbol=args.symbol, interval=args.interval,
                        total_limit=args.total_limit,
                        train_fraction=args.train_fraction,
                        news_dropout_probability=args.news_dropout,
                        news_asset=args.news_asset,
                        episode_window=args.episode_window),
        output_dir=args.output_dir, run_name=args.run_name)

    # Historical funding: a resume reuses the exact persisted
    # artifact from the original run (never recollected, so the
    # original collected_at is preserved). A new run collects the
    # full history for the loaded candle range (context_limit=3).
    # Failure raises; there is no silent fallback to constant funding.
    funding_history = None
    if args.funding_mode == "historical":
        if args.resume:
            funding_history = _load_resume_funding_history(args.resume)
        else:
            if not candles:
                raise ValueError("No candles loaded for historical funding")
            funding_history = fetch_funding_history(
                symbol=args.symbol, start_time=candles[0].open_time,
                end_time=candles[-1].close_time)

    archive = None
    try:
        if args.news_archive:
            from noone.news_data import NewsArchive

            archive = NewsArchive(args.news_archive)
        from noone.rl_engine.training import run_training

        result = run_training(
            candles, config=config,
            trading_config=TradingConfig(funding_mode=args.funding_mode),
            news_archive=archive, resume_from=args.resume,
            funding_history=funding_history)
    finally:
        if archive is not None:
            archive.close()
    print(f"Run directory: {result['run_dir']}")
    print(f"Final model:   {result['final_model']}")
    print("Historical training results do not guarantee future performance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
