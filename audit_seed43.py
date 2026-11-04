import json
from pathlib import Path
import numpy as np
import pandas as pd

from experiments_exploration_schedule import generate_market_candles
from noone.rl_engine import Action, ModelInference
from noone.rl_engine.data import prepare_candles, chronological_split
from noone.rl_engine.evaluation import evaluate_model
from noone.rl_engine.gym_env import NooneGymEnv
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
eval_candles = split.evaluation
warmup = split.eval_warmup_context

trading_cfg = TradingConfig(
    fee_rate=0.0005,
    slippage_rate=0.0002,
    funding_mode="constant",
    funding_rate=0.0001,
    reward_scale=100.0,
    initial_balance=10_000.0,
)

model_path = Path("runs/experiments_exploration_schedule/Exp_A_10k_seed_43/final_model.zip")
inf = ModelInference.load(model_path)

# Run evaluation and trace all steps
env_candles = list(warmup) + list(eval_candles)
env = NooneGymEnv(env_candles, trading_cfg, feature_builder=inf.feature_builder)
obs, _ = env.reset(seed=43, options={"start_index": 0})
for _ in range(len(warmup)):
    obs, _, _, _, _ = env.step(0)

records = []
step_idx = 0
done = False
while not done:
    pos = env.env.position
    pos_side = 0 if pos is None else pos.side
    pred = inf.predict_observation(obs, position_state=pos_side)
    act = int(pred.action.value)
    
    candle = env.env.current_candle
    close_p = float(candle.close)
    
    obs, reward, term, trunc, info = env.step(act)
    done = bool(term or trunc)
    
    records.append({
        "step": step_idx,
        "time": candle.open_time.isoformat() if candle.open_time else None,
        "close": close_p,
        "raw_action": pred.raw_action.name,
        "exec_action": pred.action.name,
        "event": info.get("event"),
        "realized_pnl": info.get("realized_pnl"),
        "fees": info.get("fee", 0.0),
        "total_fees": float(env.env.total_fees),
        "equity": float(env.env.equity),
        "q_hold": pred.q_values["HOLD"],
        "q_open_long": pred.q_values["OPEN_LONG"],
        "q_close_long": pred.q_values["CLOSE_LONG"],
        "diff_close_hold": pred.q_values["CLOSE_LONG"] - pred.q_values["HOLD"],
        "log_ret_15": float(obs[2]),
        "ema21_dev": float(obs[5]),
    })
    step_idx += 1

df = pd.DataFrame(records)
trades_df = df[df["event"].isin(["open_long", "close"])]
print("=== TRADES IN Exp_A_10k_seed_43 EVALUATION ===")
print(f"Total events found: {len(trades_df)}")
print(trades_df[["step", "time", "close", "exec_action", "event", "realized_pnl", "fees", "total_fees", "equity", "q_hold", "q_close_long", "diff_close_hold", "log_ret_15"]].to_string())

