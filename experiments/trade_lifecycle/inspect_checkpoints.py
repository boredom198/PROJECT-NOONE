"""Inspect all available trained checkpoints in the workspace.

Verifies:
1. Model loading via ModelInference.load().
2. Training configuration (total timesteps, learning starts, decay, seeds, reward scale).
3. Metadata schema integrity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from noone.rl_engine import ModelInference

def find_and_inspect_checkpoints():
    checkpoint_dirs = [
        Path("experiments/multi_regime/models"),
        Path("runs/experiments_exploration_schedule"),
        Path("runs/experiments_reward_scale"),
        Path("runs/stage2-realdata"),
    ]

    found_models: list[dict[str, Any]] = []

    for base_dir in checkpoint_dirs:
        if not base_dir.exists():
            continue
        for p in base_dir.rglob("final_model.zip"):
            run_dir = p.parent
            meta_p = run_dir / "metadata.json"
            cfg_p = run_dir / "config.json"
            
            meta_data = {}
            if meta_p.exists():
                try:
                    with open(meta_p, "r", encoding="utf-8") as f:
                        meta_data = json.load(f)
                except Exception as e:
                    meta_data = {"error": str(e)}

            cfg_data = {}
            if cfg_p.exists():
                try:
                    with open(cfg_p, "r", encoding="utf-8") as f:
                        cfg_data = json.load(f)
                except Exception as e:
                    cfg_data = {"error": str(e)}

            # Try loading
            load_success = False
            load_error = None
            try:
                inf = ModelInference.load(p, allow_legacy_metadata=True)
                load_success = True
            except Exception as e:
                load_error = str(e)

            dqn_cfg = cfg_data.get("dqn", {})
            found_models.append({
                "path": str(p),
                "run_name": run_dir.name,
                "group": base_dir.name,
                "load_success": load_success,
                "load_error": load_error,
                "total_timesteps": dqn_cfg.get("total_timesteps", meta_data.get("total_timesteps")),
                "seed": dqn_cfg.get("seed", meta_data.get("training_seed")),
                "learning_rate": dqn_cfg.get("learning_rate"),
                "learning_starts": dqn_cfg.get("learning_starts"),
                "reward_scale": cfg_data.get("trading", {}).get("reward_scale"),
                "invalid_action_penalty": dqn_cfg.get("invalid_action_penalty"),
                "schema_hash": meta_data.get("feature_schema_hash", "legacy/missing"),
            })

    print(f"Total model checkpoints found: {len(found_models)}")
    print(f"{'Group':<30} | {'Run Name':<30} | {'Load':<5} | {'Steps':>7} | {'Seed':>4} | {'Penalty':>7} | {'Scale':>5}")
    print("-" * 96)
    for m in sorted(found_models, key=lambda x: (x['group'], x['run_name'])):
        status = "OK" if m["load_success"] else "FAIL"
        steps = str(m["total_timesteps"]) if m["total_timesteps"] is not None else "-"
        seed = str(m["seed"]) if m["seed"] is not None else "-"
        pen = str(m["invalid_action_penalty"]) if m["invalid_action_penalty"] is not None else "-"
        scale = str(m["reward_scale"]) if m["reward_scale"] is not None else "-"
        print(f"{m['group']:<30} | {m['run_name']:<30} | {status:<5} | {steps:>7} | {seed:>4} | {pen:>7} | {scale:>5}")

    out_file = Path("experiments/trade_lifecycle/checkpoint_inventory.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(found_models, f, indent=2)
    print(f"\nInventory saved to {out_file}")

if __name__ == "__main__":
    find_and_inspect_checkpoints()

