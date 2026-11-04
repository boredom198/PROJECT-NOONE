import pandas as pd
import numpy as np
from pathlib import Path
from experiments_exploration_schedule import generate_market_candles
from noone.rl_engine import ModelInference
from noone.rl_engine.data import prepare_candles, chronological_split
from noone.rl_engine.evaluation import evaluate_model
from noone.trading_env import TradingConfig

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
trading_cfg = TradingConfig(reward_scale=100.0)

# Break train into 4 quartiles of 240 candles each (with 50 warmup candles before each if available)
q_size = 240
quartiles = []
for i in range(4):
    start_idx = i * q_size
    end_idx = (i + 1) * q_size
    warmup_start = max(0, start_idx - 50)
    w = train[warmup_start:start_idx]
    c = train[start_idx:end_idx]
    quartiles.append((f"Train_Q{i+1}", w, c))

print("=== REGIME GENERALIZATION ON EXISTING TRAINING PERIODS ===")
for seed in [42, 43, 44]:
    model_path = Path(f"runs/experiments_exploration_schedule/Exp_B_25k_seed_{seed}/final_model.zip")
    inf = ModelInference.load(model_path)
    print(f"\n--- Model Exp_B_25k_seed_{seed} ---")
    for q_name, w, c in quartiles:
        if len(w) < 2:
            w = c[:50]
            c_eval = c[50:]
        else:
            c_eval = c
        res = evaluate_model(
            inf, c_eval, warmup_context=w, trading_config=trading_cfg, seed=seed, include_baselines=False
        )
        actions = res["actions"]
        raw = res.get("raw_actions", [])
        m = res["metrics"]
        print(f"  {q_name} (240 bars): ExecActions={dict(pd.Series(actions).value_counts())}, RawActions={dict(pd.Series(raw).value_counts())}, Trades={m['num_trades']}, Return={m['total_return']*100:+.2f}%, MaxDD={m['max_drawdown']*100:.2f}%")

