import math
from datetime import datetime
import numpy as np
import pandas as pd

from experiments_exploration_schedule import generate_market_candles
from noone.rl_engine.data import prepare_candles, chronological_split

all_candles = generate_market_candles(1200, seed=1337)
prepared = prepare_candles(all_candles, interval="15m", warmup_candles=50)
split = chronological_split(
    prepared,
    train_fraction=0.8,
    warmup_candles=50,
    min_train_candles=200,
    min_eval_candles=100,
)

train = split.train
eval_c = split.evaluation
warmup = split.eval_warmup_context

def analyze_candles(candles, name=""):
    closes = np.array([float(c.close) for c in candles])
    highs = np.array([float(c.high) for c in candles])
    lows = np.array([float(c.low) for c in candles])
    opens = np.array([float(c.open) for c in candles])
    vols = np.array([float(c.volume) for c in candles])
    
    t_start = candles[0].open_time
    t_end = candles[-1].close_time
    duration_hours = len(candles) * 0.25
    duration_days = duration_hours / 24.0
    
    p0 = opens[0]
    p_end = closes[-1]
    ret = (p_end - p0) / p0
    
    log_rets = np.diff(np.log(closes))
    vol_15m = np.std(log_rets)
    vol_annual = vol_15m * math.sqrt(365.25 * 24 * 4)
    
    # Drawdown
    cum_max = np.maximum.accumulate(closes)
    dds = (cum_max - closes) / cum_max
    max_dd = np.max(dds)
    
    # Up vs Down bars
    up_bars = np.sum(closes > opens)
    down_bars = np.sum(closes < opens)
    
    return {
        "name": name,
        "count": len(candles),
        "start": t_start.isoformat(),
        "end": t_end.isoformat(),
        "days": duration_days,
        "p_start": p0,
        "p_end": p_end,
        "p_min": np.min(lows),
        "p_max": np.max(highs),
        "total_return_pct": ret * 100.0,
        "vol_15m_pct": vol_15m * 100.0,
        "vol_annual_pct": vol_annual * 100.0,
        "max_drawdown_pct": max_dd * 100.0,
        "up_bars": int(up_bars),
        "down_bars": int(down_bars),
        "up_bar_ratio": up_bars / len(candles),
    }

print("=== OVERALL DATASET ===")
print(analyze_candles(all_candles, "All 1200 Candles"))
print("\n=== TRAIN SPLIT ===")
print(analyze_candles(train, "Train Split (960)"))
print("\n=== EVALUATION SPLIT ===")
print(analyze_candles(eval_c, "Evaluation Split (240)"))
print("\n=== WARMUP CONTEXT ===")
print(analyze_candles(warmup, "Warmup Context (50)"))

# Sub-segment analysis of training to see regimes
n_train = len(train)
q_size = n_train // 4
print("\n=== TRAIN SUB-REGIMES (Quartiles of 240 candles each) ===")
for i in range(4):
    sub = train[i*q_size : (i+1)*q_size]
    res = analyze_candles(sub, f"Train Q{i+1} (steps {i*q_size}..{(i+1)*q_size})")
    print(f"Q{i+1}: Start={res['p_start']:.1f}, End={res['p_end']:.1f}, Return={res['total_return_pct']:+.2f}%, MaxDD={res['max_drawdown_pct']:.2f}%, VolAnn={res['vol_annual_pct']:.1f}%")

# Sub-segment analysis of evaluation (halves of 120 candles each)
print("\n=== EVALUATION SUB-REGIMES (Halves of 120 candles each) ===")
for i in range(2):
    sub = eval_c[i*120 : (i+1)*120]
    res = analyze_candles(sub, f"Eval H{i+1}")
    print(f"H{i+1}: Start={res['p_start']:.1f}, End={res['p_end']:.1f}, Return={res['total_return_pct']:+.2f}%, MaxDD={res['max_drawdown_pct']:.2f}%, VolAnn={res['vol_annual_pct']:.1f}%")

