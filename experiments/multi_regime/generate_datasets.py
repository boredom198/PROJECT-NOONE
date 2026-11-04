"""Multi-regime synthetic dataset generator for PROJECT NOONE Component 1.

Generates controlled benchmark datasets:
1. Strong Bull: +15.0% cumulative return (train: 400 candles, eval: 200 candles at +7.5%)
2. Sustained Bear: -15.0% cumulative return (train: 400 candles, eval: 200 candles at -7.5%)
3. High-Volatility Sideways: 0.0% cumulative return with meaningful swings (train: 400 candles, eval: 200 candles at 0.0%)

Each dataset includes:
- 50 warmup candles
- Exact target cumulative returns with realistic geometric random-walk innovations
- Independent reproducible seeds
- Regime statistics logging
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta, timezone
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

from noone.market_data.models import Kline


def make_kline(
    *,
    symbol: str,
    interval: str,
    open_time: datetime,
    close_time: datetime,
    open_p: float,
    high_p: float,
    low_p: float,
    close_p: float,
    volume: float,
    quote_volume: float,
    trades: int,
) -> Kline:
    return Kline(
        symbol=symbol,
        interval=interval,
        open_time=open_time,
        close_time=close_time,
        open=float(open_p),
        high=float(high_p),
        low=float(low_p),
        close=float(close_p),
        volume=float(volume),
        quote_volume=float(quote_volume),
        trades=int(trades),
        is_closed=True,
    )


def generate_calibrated_segment(
    *,
    target_return: float,
    n_candles: int,
    start_price: float,
    start_time: datetime,
    sigma: float,
    wave_amp: float = 0.0,
    seed: int = 42,
) -> list[Kline]:
    rng = np.random.RandomState(seed)
    total_log_ret = math.log(1.0 + target_return)
    eps = rng.normal(0.0, sigma, n_candles)
    if wave_amp > 0:
        wave = np.array([wave_amp * math.cos(2 * math.pi * i / 80) for i in range(n_candles)])
        eps = eps + wave
    adjusted_rets = eps - np.mean(eps) + (total_log_ret / n_candles)

    candles: list[Kline] = []
    p = start_price

    for i, r in enumerate(adjusted_rets):
        open_time = start_time + timedelta(minutes=15 * i)
        close_time = open_time + timedelta(minutes=14, seconds=59)

        close_p = p * math.exp(r)

        # Wicks
        wick_h = abs(rng.normal(0.0010, 0.0006))
        wick_l = abs(rng.normal(0.0010, 0.0006))
        high_p = max(p, close_p) * (1.0 + wick_h)
        low_p = min(p, close_p) * (1.0 - wick_l)

        vol = float(rng.lognormal(mean=3.0, sigma=0.5))
        q_vol = vol * close_p

        candles.append(
            make_kline(
                symbol="BTCUSDT",
                interval="15m",
                open_time=open_time,
                close_time=close_time,
                open_p=p,
                high_p=high_p,
                low_p=low_p,
                close_p=close_p,
                volume=vol,
                quote_volume=q_vol,
                trades=int(rng.randint(20, 200)),
            )
        )
        p = close_p

    return candles


def analyze_segment(candles: list[Kline], name: str = "") -> dict[str, Any]:
    closes = np.array([float(c.close) for c in candles])
    highs = np.array([float(c.high) for c in candles])
    lows = np.array([float(c.low) for c in candles])
    opens = np.array([float(c.open) for c in candles])

    p0 = opens[0]
    p_end = closes[-1]
    ret_pct = (p_end - p0) / p0 * 100.0

    log_rets = np.diff(np.log(closes))
    vol_15m = float(np.std(log_rets))
    vol_ann = vol_15m * math.sqrt(365.25 * 24 * 4) * 100.0

    cum_max = np.maximum.accumulate(closes)
    dds = (cum_max - closes) / cum_max
    max_dd = float(np.max(dds)) * 100.0

    up_bars = int(np.sum(closes > opens))
    down_bars = int(np.sum(closes < opens))

    return {
        "name": name,
        "count": len(candles),
        "start_time": candles[0].open_time.isoformat(),
        "end_time": candles[-1].close_time.isoformat(),
        "start_price": float(p0),
        "end_price": float(p_end),
        "min_price": float(np.min(lows)),
        "max_price": float(np.max(highs)),
        "return_pct": float(ret_pct),
        "vol_annual_pct": float(vol_ann),
        "max_drawdown_pct": float(max_dd),
        "up_bars": up_bars,
        "down_bars": down_bars,
        "up_bar_ratio": float(up_bars / len(candles)),
    }


def stitch_candles(segments: list[list[Kline]], start_time: datetime | None = None) -> list[Kline]:
    """Stitches candle segments with strictly continuous timestamps and matching open-close boundaries."""
    stitched: list[Kline] = []
    t_cursor = start_time or segments[0][0].open_time

    for seg in segments:
        for k in seg:
            open_time = t_cursor
            close_time = open_time + timedelta(minutes=14, seconds=59)
            stitched.append(
                make_kline(
                    symbol=k.symbol,
                    interval=k.interval,
                    open_time=open_time,
                    close_time=close_time,
                    open_p=k.open,
                    high_p=k.high,
                    low_p=k.low,
                    close_p=k.close,
                    volume=k.volume,
                    quote_volume=k.quote_volume,
                    trades=k.trades,
                )
            )
            t_cursor = open_time + timedelta(minutes=15)
    return stitched


def generate_benchmark_suite():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    base_price = 60_000.0

    # 1. Warmup context: 50 neutral candles
    warmup_neutral = generate_calibrated_segment(
        target_return=0.00, n_candles=50, start_price=base_price, start_time=t0, sigma=0.0020, seed=10
    )
    t_step = warmup_neutral[-1].close_time + timedelta(seconds=1)

    # 2. Bull regime: Train +15.0%, Eval +7.5%
    bull_train = generate_calibrated_segment(
        target_return=0.15, n_candles=400, start_price=base_price, start_time=t_step, sigma=0.0032, seed=101
    )
    t_bull_eval_warm = bull_train[-1].close_time + timedelta(seconds=1)
    bull_eval_warm = generate_calibrated_segment(
        target_return=0.00, n_candles=50, start_price=bull_train[-1].close, start_time=t_bull_eval_warm, sigma=0.0025, seed=102
    )
    t_bull_eval = bull_eval_warm[-1].close_time + timedelta(seconds=1)
    bull_eval = generate_calibrated_segment(
        target_return=0.075, n_candles=200, start_price=bull_eval_warm[-1].close, start_time=t_bull_eval, sigma=0.0032, seed=103
    )

    # 3. Bear regime: Train -15.0%, Eval -7.5%
    bear_train = generate_calibrated_segment(
        target_return=-0.15, n_candles=400, start_price=base_price, start_time=t_step, sigma=0.0038, seed=201
    )
    t_bear_eval_warm = bear_train[-1].close_time + timedelta(seconds=1)
    bear_eval_warm = generate_calibrated_segment(
        target_return=0.00, n_candles=50, start_price=bear_train[-1].close, start_time=t_bear_eval_warm, sigma=0.0025, seed=202
    )
    t_bear_eval = bear_eval_warm[-1].close_time + timedelta(seconds=1)
    bear_eval = generate_calibrated_segment(
        target_return=-0.075, n_candles=200, start_price=bear_eval_warm[-1].close, start_time=t_bear_eval, sigma=0.0038, seed=203
    )

    # 4. Sideways regime: Train 0.0% (cyclical swings), Eval 0.0%
    side_train = generate_calibrated_segment(
        target_return=0.00, n_candles=400, start_price=base_price, start_time=t_step, sigma=0.0042, wave_amp=0.0008, seed=301
    )
    t_side_eval_warm = side_train[-1].close_time + timedelta(seconds=1)
    side_eval_warm = generate_calibrated_segment(
        target_return=0.00, n_candles=50, start_price=side_train[-1].close, start_time=t_side_eval_warm, sigma=0.0025, seed=302
    )
    t_side_eval = side_eval_warm[-1].close_time + timedelta(seconds=1)
    side_eval = generate_calibrated_segment(
        target_return=0.00, n_candles=200, start_price=side_eval_warm[-1].close, start_time=t_side_eval, sigma=0.0042, wave_amp=0.0008, seed=303
    )

    # Combined datasets
    # Training configurations:
    # A. Multi-Regime (Bull + Bear + Sideways in sequence, 1200 candles)
    train_multi = stitch_candles([bull_train, bear_train, side_train], start_time=t_step)
    # B. Trend Only (Bull + Bear, 800 candles)
    train_trend = stitch_candles([bull_train, bear_train], start_time=t_step)
    # C. Range Only (Sideways, 400 candles)
    train_range = stitch_candles([side_train], start_time=t_step)

    # Evaluation datasets (each with its exact warmup context)
    eval_suite = {
        "bull": {"candles": stitch_candles([bull_eval]), "warmup": stitch_candles([bull_eval_warm])},
        "bear": {"candles": stitch_candles([bear_eval]), "warmup": stitch_candles([bear_eval_warm])},
        "sideways": {"candles": stitch_candles([side_eval]), "warmup": stitch_candles([side_eval_warm])},
    }

    # Composite Evaluation (Bull -> Bear -> Sideways in continuous chronological sequence: 600 candles)
    composite_eval = stitch_candles([bull_eval, bear_eval, side_eval])
    eval_suite["composite"] = {"candles": composite_eval, "warmup": stitch_candles([warmup_neutral])}

    return {
        "warmup_neutral": warmup_neutral,
        "segments": {
            "bull_train": bull_train,
            "bear_train": bear_train,
            "side_train": side_train,
            "bull_eval": bull_eval,
            "bear_eval": bear_eval,
            "side_eval": side_eval,
        },
        "training_sets": {
            "multi_regime": train_multi,
            "trend_only": train_trend,
            "range_only": train_range,
        },
        "eval_suite": eval_suite,
    }


if __name__ == "__main__":
    suite = generate_benchmark_suite()
    print("=== MULTI-REGIME BENCHMARK SUITE CREATED ===")
    for k, v in suite["segments"].items():
        st = analyze_segment(v, k)
        print(f"{k:<15}: Count={st['count']:3d}, Ret={st['return_pct']:+6.2f}%, VolAnn={st['vol_annual_pct']:4.1f}%, MaxDD={st['max_drawdown_pct']:5.2f}%, Min=${st['min_price']:.1f}, Max=${st['max_price']:.1f}")

    print("\n--- Training Set Sizes ---")
    for k, v in suite["training_sets"].items():
        print(f"  {k:<15}: {len(v)} candles ({len(v)*0.25:.1f} hours)")

    print("\n--- Evaluation Set Sizes ---")
    for k, v in suite["eval_suite"].items():
        print(f"  {k:<15}: {len(v['candles'])} candles ({len(v['candles'])*0.25:.1f} hours), Warmup={len(v['warmup'])}")
