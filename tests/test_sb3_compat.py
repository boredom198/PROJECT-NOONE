"""SB3 compatibility tests (NOT Stage 2 training).

Verifies the real Stable-Baselines3 library works with NooneGymEnv:
check_env, a tiny DQN smoke run, save/load, deterministic predict.
Skipped when stable-baselines3 is not installed (`.[rl]` extra).
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

sb3 = pytest.importorskip("stable_baselines3")
from stable_baselines3 import DQN  # noqa: E402
from stable_baselines3.common.env_checker import check_env  # noqa: E402

from noone.market_data.models import Kline  # noqa: E402
from noone.rl_engine import NooneGymEnv  # noqa: E402
from noone.trading_env import TradingConfig  # noqa: E402

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_env():
    def c(i, p=100.0):
        return Kline(symbol="BTCUSDT", interval="1m",
                     open_time=T0 + timedelta(minutes=i),
                     close_time=T0 + timedelta(minutes=i, seconds=59),
                     open=p, high=p, low=p, close=p, volume=10.0,
                     quote_volume=1000.0, trades=5, is_closed=True)

    return NooneGymEnv(
        [c(i, 100 + i * 0.1) for i in range(60)],
        config=TradingConfig(fee_rate=0, slippage_rate=0, funding_rate=0))


def test_check_env_passes():
    check_env(make_env(), warn=True)


def test_dqn_smoke_save_load_deterministic(tmp_path):
    env = make_env()
    model = DQN("MlpPolicy", env, policy_kwargs={"net_arch": [32, 32]},
                buffer_size=500, learning_starts=10, batch_size=16,
                train_freq=1, verbose=0, seed=42, device="cpu")
    model.learn(total_timesteps=128)  # smoke only, not training
    path = str(tmp_path / "dqn_smoke")
    model.save(path)
    loaded = DQN.load(path, env=make_env())
    obs, _ = make_env().reset(seed=7)
    # Deterministic prediction must be stable across calls and loads.
    a1, _ = model.predict(obs, deterministic=True)
    a2, _ = loaded.predict(obs, deterministic=True)
    assert int(a1) == int(a2) and 0 <= int(a1) < 5
    obs2, _info = make_env().reset(seed=7)
    assert np.array_equal(obs, obs2)
