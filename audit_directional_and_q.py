import numpy as np
import torch
from pathlib import Path
from experiments_exploration_schedule import generate_market_candles
from noone.rl_engine import Action, ModelInference
from noone.rl_engine.data import prepare_candles, chronological_split
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
full_eval = list(warmup) + list(eval_candles)

trading_cfg = TradingConfig(reward_scale=100.0)

for seed in [42, 43, 44]:
    model_path = Path(f"runs/experiments_exploration_schedule/Exp_B_25k_seed_{seed}/final_model.zip")
    inf = ModelInference.load(model_path)
    model = inf.model
    
    # Generate observations across training candles in 3 simulated states: FLAT, LONG, SHORT
    # FeatureBuilder builds canonical 26-dim observations
    train_candles = split.train
    fb = inf.feature_builder
    
    q_flat = []
    q_long = []
    q_short = []
    
    for i in range(50, len(train_candles)):
        history = train_candles[:i+1]
        candle = train_candles[i]
        price = float(candle.close)
        
        # 1. FLAT state
        raw_flat = {
            "index": i, "price": price, "balance": 10000.0, "equity": 10000.0,
            "margin_used": 0.0, "available_balance": 10000.0,
            "position": None, "unrealized_pnl": 0.0, "realized_pnl": 0.0, "total_fees": 0.0, "progress": 0.0,
        }
        obs_f = fb.build(raw_flat, history)
        
        # 2. LONG state (+1% unrealized pnl, $5000 notional)
        raw_long = {
            "index": i, "price": price, "balance": 10000.0, "equity": 10050.0,
            "margin_used": 1000.0, "available_balance": 9000.0,
            "position": {"side": 1, "entry_price": price * 0.99, "quantity": 5000.0 / (price * 0.99), "margin": 1000.0},
            "unrealized_pnl": 50.0, "realized_pnl": 0.0, "total_fees": 2.50, "progress": 0.0,
        }
        obs_l = fb.build(raw_long, history)
        
        # 3. SHORT state (-1% unrealized pnl, $5000 notional)
        raw_short = {
            "index": i, "price": price, "balance": 10000.0, "equity": 9950.0,
            "margin_used": 1000.0, "available_balance": 9000.0,
            "position": {"side": -1, "entry_price": price * 0.99, "quantity": 5000.0 / (price * 0.99), "margin": 1000.0},
            "unrealized_pnl": -50.0, "realized_pnl": 0.0, "total_fees": 2.50, "progress": 0.0,
        }
        obs_s = fb.build(raw_short, history)
        
        with torch.no_grad():
            q_flat.append(model.policy.q_net(torch.as_tensor(obs_f, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy())
            q_long.append(model.policy.q_net(torch.as_tensor(obs_l, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy())
            q_short.append(model.policy.q_net(torch.as_tensor(obs_s, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy())
            
    q_flat_arr = np.array(q_flat)
    q_long_arr = np.array(q_long)
    q_short_arr = np.array(q_short)
    
    print(f"\n=======================================================")
    print(f"STATE-CONDITIONAL Q-VALUE PROFILE: Exp_B_25k_seed_{seed} (N={len(q_flat)})")
    print(f"=======================================================")
    action_names = [a.name for a in Action]
    
    for state_name, arr in [("FLAT (pos=0)", q_flat_arr), ("LONG (pos=1)", q_long_arr), ("SHORT (pos=-1)", q_short_arr)]:
        m = arr.mean(axis=0)
        s = arr.std(axis=0)
        q_str = ", ".join([f"{name}: {m[i]:.4f} (±{s[i]:.4f})" for i, name in enumerate(action_names)])
        print(f"State {state_name}:")
        print(f"  Mean Q -> {q_str}")
        if "FLAT" in state_name:
            print(f"  Advantage: OPEN_LONG - HOLD = {m[1] - m[0]:+.4f}, OPEN_SHORT - HOLD = {m[3] - m[0]:+.4f}")
            print(f"  Dominant action: {action_names[np.argmax(m)]}")
        elif "LONG" in state_name:
            print(f"  Advantage: HOLD - CLOSE_LONG = {m[0] - m[2]:+.4f}")
            print(f"  Dominant valid action: {'HOLD' if m[0] > m[2] else 'CLOSE_LONG'}")
        elif "SHORT" in state_name:
            print(f"  Advantage: HOLD - CLOSE_SHORT = {m[0] - m[4]:+.4f}")
            print(f"  Dominant valid action: {'HOLD' if m[0] > m[4] else 'CLOSE_SHORT'}")
