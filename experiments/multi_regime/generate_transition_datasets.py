"""Controlled Regime-Transition Dataset Generator for PROJECT NOONE Component 1.

Constructs 6 pairwise transition sequences:
1. Bull -> Bear
2. Bear -> Bull
3. Bull -> Sideways
4. Bear -> Sideways
5. Sideways -> Bull
6. Sideways -> Bear

Each sequence:
- 50 warmup candles (neutral flat context)
- Regime 1: 200 candles (50 hours)
- Regime 2: 200 candles (50 hours)
- Total evaluation length: 400 candles (100 hours)
- Transition point: exactly at candle index 200
- Price continuity: Regime 2 starts at the exact close price of Regime 1 (no artificial gap)
- Reproducible independent seeds
"""
from __future__ import annotations

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

from experiments.multi_regime.generate_datasets import (
    analyze_segment,
    generate_calibrated_segment,
    make_kline,
    stitch_candles,
)
from noone.market_data.models import Kline


def create_transition_sequence(
    *,
    name: str,
    regime_1: str,
    regime_2: str,
    ret_1: float,
    ret_2: float,
    sigma_1: float,
    sigma_2: float,
    wave_amp_1: float = 0.0,
    wave_amp_2: float = 0.0,
    seed_1: int = 401,
    seed_2: int = 501,
    start_time: datetime,
    base_price: float = 60_000.0,
) -> dict[str, Any]:
    # 50 warmup context candles (neutral flat at base_price)
    warmup = generate_calibrated_segment(
        target_return=0.00,
        n_candles=50,
        start_price=base_price,
        start_time=start_time,
        sigma=0.0020,
        seed=10,
    )
    t_regime1 = warmup[-1].close_time + timedelta(seconds=1)

    # Regime 1: 200 candles
    seg1 = generate_calibrated_segment(
        target_return=ret_1,
        n_candles=200,
        start_price=base_price,
        start_time=t_regime1,
        sigma=sigma_1,
        wave_amp=wave_amp_1,
        seed=seed_1,
    )
    t_regime2 = seg1[-1].close_time + timedelta(seconds=1)
    p_regime2_start = seg1[-1].close

    # Regime 2: 200 candles, starting at exact close price of Regime 1
    seg2 = generate_calibrated_segment(
        target_return=ret_2,
        n_candles=200,
        start_price=p_regime2_start,
        start_time=t_regime2,
        sigma=sigma_2,
        wave_amp=wave_amp_2,
        seed=seed_2,
    )

    combined_eval = stitch_candles([seg1, seg2], start_time=t_regime1)

    return {
        "name": name,
        "regime_1": regime_1,
        "regime_2": regime_2,
        "transition_index": 200,
        "warmup": stitch_candles([warmup], start_time=start_time),
        "seg1": seg1,
        "seg2": seg2,
        "candles": combined_eval,
        "stats_seg1": analyze_segment(seg1, f"{name}_R1_{regime_1}"),
        "stats_seg2": analyze_segment(seg2, f"{name}_R2_{regime_2}"),
        "stats_total": analyze_segment(combined_eval, f"{name}_TOTAL"),
    }


def build_all_transition_sequences() -> dict[str, dict[str, Any]]:
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    base_price = 60_000.0

    transitions = {
        # 1. Bull -> Bear: +7.5% then -7.5%
        "bull_to_bear": create_transition_sequence(
            name="bull_to_bear",
            regime_1="bull",
            regime_2="bear",
            ret_1=0.075,
            ret_2=-0.075,
            sigma_1=0.0032,
            sigma_2=0.0038,
            seed_1=411,
            seed_2=511,
            start_time=t0,
            base_price=base_price,
        ),
        # 2. Bear -> Bull: -7.5% then +7.5%
        "bear_to_bull": create_transition_sequence(
            name="bear_to_bull",
            regime_1="bear",
            regime_2="bull",
            ret_1=-0.075,
            ret_2=0.075,
            sigma_1=0.0038,
            sigma_2=0.0032,
            seed_1=412,
            seed_2=512,
            start_time=t0,
            base_price=base_price,
        ),
        # 3. Bull -> Sideways: +7.5% then 0.0% chop
        "bull_to_sideways": create_transition_sequence(
            name="bull_to_sideways",
            regime_1="bull",
            regime_2="sideways",
            ret_1=0.075,
            ret_2=0.000,
            sigma_1=0.0032,
            sigma_2=0.0042,
            wave_amp_2=0.0008,
            seed_1=413,
            seed_2=513,
            start_time=t0,
            base_price=base_price,
        ),
        # 4. Bear -> Sideways: -7.5% then 0.0% chop
        "bear_to_sideways": create_transition_sequence(
            name="bear_to_sideways",
            regime_1="bear",
            regime_2="sideways",
            ret_1=-0.075,
            ret_2=0.000,
            sigma_1=0.0038,
            sigma_2=0.0042,
            wave_amp_2=0.0008,
            seed_1=414,
            seed_2=514,
            start_time=t0,
            base_price=base_price,
        ),
        # 5. Sideways -> Bull: 0.0% chop then +7.5% breakout
        "sideways_to_bull": create_transition_sequence(
            name="sideways_to_bull",
            regime_1="sideways",
            regime_2="bull",
            ret_1=0.000,
            ret_2=0.075,
            sigma_1=0.0042,
            sigma_2=0.0032,
            wave_amp_1=0.0008,
            seed_1=415,
            seed_2=515,
            start_time=t0,
            base_price=base_price,
        ),
        # 6. Sideways -> Bear: 0.0% chop then -7.5% breakdown
        "sideways_to_bear": create_transition_sequence(
            name="sideways_to_bear",
            regime_1="sideways",
            regime_2="bear",
            ret_1=-0.000,
            ret_2=-0.075,
            sigma_1=0.0042,
            sigma_2=0.0038,
            wave_amp_1=0.0008,
            seed_1=416,
            seed_2=516,
            start_time=t0,
            base_price=base_price,
        ),
    }

    return transitions


if __name__ == "__main__":
    suite = build_all_transition_sequences()
    print("=== REGIME TRANSITION SUITE CREATED ===")
    for k, v in suite.items():
        s1 = v["stats_seg1"]
        s2 = v["stats_seg2"]
        st = v["stats_total"]
        print(f"\n[{k.upper()}] (Total: {st['count']} candles, Return={st['return_pct']:+.2f}%, VolAnn={st['vol_annual_pct']:.1f}%, MaxDD={st['max_drawdown_pct']:.2f}%):")
        print(f"  R1 ({v['regime_1']:<8}): Ret={s1['return_pct']:+6.2f}%, VolAnn={s1['vol_annual_pct']:4.1f}%, MaxDD={s1['max_drawdown_pct']:5.2f}%, P=${s1['start_price']:.1f} -> ${s1['end_price']:.1f}")
        print(f"  R2 ({v['regime_2']:<8}): Ret={s2['return_pct']:+6.2f}%, VolAnn={s2['vol_annual_pct']:4.1f}%, MaxDD={s2['max_drawdown_pct']:5.2f}%, P=${s2['start_price']:.1f} -> ${s2['end_price']:.1f}")
