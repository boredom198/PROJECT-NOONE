import numpy as np
import torch
from pathlib import Path
from noone.rl_engine import ModelInference
from noone.rl_engine.features import FEATURE_NAMES

for seed in [42, 43, 44]:
    model_path = Path(f"runs/experiments_exploration_schedule/Exp_B_25k_seed_{seed}/final_model.zip")
    inf = ModelInference.load(model_path)
    model = inf.model
    
    # Inspect first layer weights of q_net
    # Policy net_arch: [128, 128]
    # In SB3, q_net has q_net (MLP)
    w0 = model.policy.q_net.q_net[0].weight.data.abs().mean(dim=0).cpu().numpy()
    
    print(f"\n=======================================================")
    print(f"FIRST LAYER WEIGHT MAGNITUDES: Exp_B_25k_seed_{seed}")
    print(f"=======================================================")
    
    sorted_idx = np.argsort(w0)[::-1]
    for rank, idx in enumerate(sorted_idx):
        print(f"  Rank {rank+1:2d}: Feature {idx:2d} ({FEATURE_NAMES[idx]:<30}) = {w0[idx]:.5f}")

