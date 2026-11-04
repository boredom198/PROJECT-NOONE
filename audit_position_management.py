import json
from pathlib import Path
import numpy as np
import pandas as pd

from experiments_exploration_schedule import generate_market_candles
from noone.rl_engine import Action, ModelInference
from noone.rl_engine.config import DataConfig
from noone.rl_engine.data import prepare_candles, chronological_split
from noone.rl_engine.gym_env import NooneGymEnv
from noone.trading_env import TradingConfig

# Prepare dataset
all_candles = generate_market_candles(1200, seed=1337)
prepared = prepare_candles(all_candles, interval="15m", warmup_candles=50)
split = chronological_split(
    prepared,
    train_fraction=0.8,
    warmup_candles=50,
    min_train_candles=200,
    min_eval_candles=100,
)
eval_candles = split.evaluation
warmup = split.eval_warmup_context
full_eval_candles = list(warmup) + list(eval_candles)

trading_cfg = TradingConfig(
    fee_rate=0.0005,
    slippage_rate=0.0002,
    funding_mode="constant",
    funding_rate=0.0001,
    reward_scale=100.0,
    initial_balance=10_000.0,
)

seeds = [42, 43, 44]
models = {}
for seed in seeds:
    model_path = Path(f"runs/experiments_exploration_schedule/Exp_B_25k_seed_{seed}/final_model.zip")
    models[seed] = ModelInference.load(model_path)

# Analyze evaluation trajectory for each model
print("=== POSITION MANAGEMENT ANALYSIS (Exp_B_25k models) ===")

for seed, inf in models.items():
    print(f"\n=======================================================")
    print(f"MODEL: Exp_B_25k_seed_{seed}")
    print(f"=======================================================")
    
    env = NooneGymEnv(full_eval_candles, trading_cfg, feature_builder=inf.feature_builder)
    obs, _ = env.reset(seed=seed, options={"start_index": 0})
    for _ in range(len(warmup)):
        obs, _, term, trunc, _ = env.step(0)
    
    # Run evaluation episode, recording Q-values and margins
    records = []
    step_idx = 0
    done = False
    while not done:
        pos = env.env.position
        pos_side = 0 if pos is None else pos.side
        
        # Predict with masking and get Q-values
        pred = inf.predict_observation(obs, position_state=pos_side)
        q_vals = pred.q_values
        
        # Raw Q-values
        q_hold = q_vals["HOLD"]
        q_open_long = q_vals["OPEN_LONG"]
        q_close_long = q_vals["CLOSE_LONG"]
        q_open_short = q_vals["OPEN_SHORT"]
        q_close_short = q_vals["CLOSE_SHORT"]
        
        # Differences
        diff_close_hold_long = q_close_long - q_hold
        diff_close_hold_short = q_close_short - q_hold
        
        candle = env.env.current_candle
        close_p = float(candle.close)
        
        records.append({
            "step": step_idx,
            "candle_index": env.env._index,
            "open_time": candle.open_time.isoformat() if candle.open_time else None,
            "close": close_p,
            "pos_side": pos_side,
            "unrealized_pnl": float(env.env.unrealized_pnl()),
            "equity": float(env.env.equity),
            "raw_action": pred.raw_action.name,
            "exec_action": pred.action.name,
            "q_hold": q_hold,
            "q_open_long": q_open_long,
            "q_close_long": q_close_long,
            "q_open_short": q_open_short,
            "q_close_short": q_close_short,
            "diff_close_hold_long": diff_close_hold_long,
            "diff_close_hold_short": diff_close_hold_short,
            "log_ret_1": float(obs[0]),
            "log_ret_15": float(obs[2]),
            "ema9_dev": float(obs[4]),
            "ema21_dev": float(obs[5]),
            "ema50_dev": float(obs[6]),
        })
        
        act = int(pred.action.value)
        obs, reward, term, trunc, info = env.step(act)
        done = bool(term or trunc)
        step_idx += 1

    df = pd.DataFrame(records)
    
    # Statistical analysis of Q-differences
    long_diffs = df["diff_close_hold_long"]
    short_diffs = df["diff_close_hold_short"]
    
    print(f"Evaluated {len(df)} steps. Actions taken: {df['exec_action'].value_counts().to_dict()}")
    print(f"Raw actions: {df['raw_action'].value_counts().to_dict()}")
    print(f"Q(CLOSE_LONG) - Q(HOLD):")
    print(f"  Mean: {long_diffs.mean():.4f}, Std: {long_diffs.std():.4f}")
    print(f"  Min:  {long_diffs.min():.4f}, Max: {long_diffs.max():.4f}")
    print(f"  Count where Q(CLOSE_LONG) > Q(HOLD): {(long_diffs > 0).sum()} of {len(df)}")
    
    print(f"Q(CLOSE_SHORT) - Q(HOLD):")
    print(f"  Mean: {short_diffs.mean():.4f}, Std: {short_diffs.std():.4f}")
    print(f"  Min:  {short_diffs.min():.4f}, Max: {short_diffs.max():.4f}")
    print(f"  Count where Q(CLOSE_SHORT) > Q(HOLD): {(short_diffs > 0).sum()} of {len(df)}")

    # Correlation between price drop and Q(CLOSE_LONG) - Q(HOLD)
    # If the model recognizes adverse movement, diff_close_hold_long should be negatively correlated with log_ret_15 or ema_dev
    corr_ret15 = df["diff_close_hold_long"].corr(df["log_ret_15"])
    corr_ema50 = df["diff_close_hold_long"].corr(df["ema50_dev"])
    corr_pnl = df["diff_close_hold_long"].corr(df["unrealized_pnl"])
    print(f"Correlations with (Q(CLOSE_LONG) - Q(HOLD)):")
    print(f"  corr(diff, log_ret_15): {corr_ret15:.4f}")
    print(f"  corr(diff, ema50_dev):  {corr_ema50:.4f}")
    print(f"  corr(diff, unrealized_pnl): {corr_pnl:.4f}")
    
    # Peak price and subsequent drawdown
    peak_idx = df["close"].idxmax()
    peak_row = df.loc[peak_idx]
    trough_after_peak_idx = df.loc[peak_idx:]["close"].idxmin()
    trough_row = df.loc[trough_after_peak_idx]
    drop_pct = (trough_row["close"] - peak_row["close"]) / peak_row["close"] * 100.0
    
    print(f"\nPrice Peak at step {peak_row['step']} (price={peak_row['close']:.1f}, time={peak_row['open_time']})")
    print(f"Trough after peak at step {trough_row['step']} (price={trough_row['close']:.1f}, time={trough_row['open_time']})")
    print(f"Subsequent Drawdown: {drop_pct:.2f}%")
    print(f"At Peak: Q(HOLD)={peak_row['q_hold']:.4f}, Q(CLOSE_LONG)={peak_row['q_close_long']:.4f}, diff={peak_row['diff_close_hold_long']:.4f}")
    print(f"At Trough: Q(HOLD)={trough_row['q_hold']:.4f}, Q(CLOSE_LONG)={trough_row['q_close_long']:.4f}, diff={trough_row['diff_close_hold_long']:.4f}")

    # Check a window around the peak and drop
    drop_window = df.loc[peak_idx:peak_idx+20, ["step", "close", "log_ret_15", "ema21_dev", "q_hold", "q_close_long", "diff_close_hold_long"]]
    print("\nSample steps during post-peak decline:")
    print(drop_window.head(10).to_string(index=False))
