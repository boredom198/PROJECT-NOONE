import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from experiments.multi_regime.generate_transition_datasets import build_all_transition_sequences
from noone.rl_engine import Action, ModelInference
from noone.rl_engine.gym_env import NooneGymEnv
from noone.trading_env import TradingConfig

seq = build_all_transition_sequences()["bear_to_bull"]
m = ModelInference.load("experiments/multi_regime/models/multi_regime_seed_42/final_model.zip")
cfg = TradingConfig(
    fee_rate=0.0005,
    slippage_rate=0.0002,
    funding_mode="constant",
    funding_rate=0.0001,
    reward_scale=100.0,
    initial_balance=10000.0,
)
env = NooneGymEnv(list(seq["warmup"]) + list(seq["candles"]), cfg, feature_builder=m.feature_builder)
obs, _ = env.reset(seed=42, options={"start_index": 0})
for _ in range(50):
    obs, _, _, _, _ = env.step(0)

print("Step | Price   | Pos   | Unrl PnL | Action      | Raw Action  | Q(HOLD) | Q(CLOSE_S) | Q(OPEN_L) | Diff Q(CS)-Q(H)")
print("-" * 105)
for step in range(235):
    pos = env.env.position
    side = 0 if pos is None else pos.side
    unrl = float(env.env.unrealized_pnl())
    pred = m.predict_observation(obs, position_state=side)
    q = pred.q_values
    p_close = seq["candles"][step].close
    if 210 <= step <= 225:
        diff = q["CLOSE_SHORT"] - q["HOLD"]
        pos_s = "FLAT" if side == 0 else ("LONG" if side == 1 else "SHORT")
        print(
            f"{step:4d} | {p_close:7.1f} | {pos_s:>5} | {unrl:+8.2f} | "
            f"{pred.action.name:11s} | {pred.raw_action.name:11s} | "
            f"{q['HOLD']:+.4f} | {q['CLOSE_SHORT']:+.4f} | {q['OPEN_LONG']:+.4f} | {diff:+.4f}"
        )
    obs, _, _, _, _ = env.step(pred.action.value)

