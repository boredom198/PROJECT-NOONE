"""Component 1 Stage 1 tests: Gymnasium adapter + FeatureBuilder.

Offline/deterministic only — no DQN, no training (Stage 2 scope).
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")

from noone.market_data.models import Kline  # noqa: E402
from noone.rl_engine import FEATURE_NAMES, FeatureBuilder, NooneGymEnv  # noqa: E402
from noone.rl_engine.features import NewsObservation  # noqa: E402
from noone.trading_env import TradingConfig  # noqa: E402

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def candle(i, price=100.0, start=T0):
    p = float(price)
    return Kline(symbol="BTCUSDT", interval="1m",
                 open_time=start + timedelta(minutes=i),
                 close_time=start + timedelta(minutes=i, seconds=59),
                 open=p, high=p, low=p, close=p, volume=10.0,
                 quote_volume=1000.0, trades=5, is_closed=True)


def ramp(n=60, start=T0):
    return [candle(i, 100.0 + i * 0.1, start) for i in range(n)]


def make_env(**kw):
    kw.setdefault("config", TradingConfig(
        fee_rate=0.0, slippage_rate=0.0, funding_rate=0.0))
    return NooneGymEnv(ramp(), **kw)


def test_spaces_and_reset_step_formats():
    env = make_env()
    assert env.action_space.n == 5
    assert env.observation_space.shape == (26,)
    assert env.observation_space.dtype == np.float32
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray) and obs.dtype == np.float32
    assert env.observation_space.contains(obs) and "start_index" in info
    out = env.step(0)
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert isinstance(reward, float) and isinstance(terminated, bool)


def test_action_mapping_opens_positions():
    env = make_env()
    env.reset(seed=1)
    env.step(1)  # OPEN_LONG
    assert env.env.position is not None and env.env.position.side == 1
    env.step(3)  # OPEN_SHORT while long: Component 8 no-op
    assert env.env.position.side == 1
    env.step(2)  # CLOSE_LONG
    assert env.env.position is None
    env.step(3)
    assert env.env.position.side == -1


def test_seeded_reproducibility_and_truncation():
    def run():
        env = make_env(episode_window=20)
        obs, info = env.reset(seed=123)
        rewards = []
        for action in [1, 0, 0, 2, 3, 0, 4, 0]:
            _o, r, _t, _tr, _i = env.step(action)
            rewards.append(r)
        return info["start_index"], rewards

    assert run() == run()  # same seed -> same window + trajectory
    env = make_env()
    env.reset(seed=0)
    done = False
    while not done:
        _o, _r, terminated, truncated, _i = env.step(0)
        done = terminated or truncated
    assert truncated  # end of candles truncates


def test_feature_builder_order_missing_and_finite():
    builder = FeatureBuilder()
    assert len(FEATURE_NAMES) == 26
    assert FEATURE_NAMES[0] == "log_return_1"
    assert FEATURE_NAMES[-1] == "news_age_normalized"
    obs = {"equity": 10_000.0, "unrealized_pnl": 0.0, "realized_pnl": 0.0,
           "available_balance": 10_000.0, "margin_used": 0.0,
           "position_side": 0.0, "position_quantity": 0.0,
           "entry_price": 0.0, "progress": 0.0}
    vec = builder.build(obs, ramp(60))
    assert vec.shape == (26,) and np.isfinite(vec).all()
    idx = FEATURE_NAMES.index("funding_oi_available")
    assert vec[idx] == 0.0  # missing funding/OI never faked as real zero
    assert vec[FEATURE_NAMES.index("funding_rate")] == 0.0
    vec2 = builder.build(obs, ramp(60), funding_rate=0.0001,
                         open_interest_change=0.02)
    assert vec2[idx] == 1.0
    # Extreme history still clips to finite bounded output.
    wild = [candle(i, 100.0 * (1.5 ** i)) for i in range(20)]
    vec3 = builder.build(obs, wild)
    assert np.isfinite(vec3).all() and abs(vec3).max() <= 10.0


def test_news_three_states_and_staleness():
    builder = FeatureBuilder(news_max_age_hours=24)
    obs = {"equity": 1.0}
    fresh = builder.build(obs, ramp(5), news=NewsObservation(
        status="fresh", mean_compound=0.5, mean_confidence=0.8,
        count=3, age_seconds=3600))
    assert fresh[FEATURE_NAMES.index("news_available")] == 1.0
    assert fresh[FEATURE_NAMES.index("news_compound")] == pytest.approx(0.5, abs=1e-6)
    none = builder.build(obs, ramp(5), news=NewsObservation(status="none"))
    assert none[FEATURE_NAMES.index("news_available")] == 0.0
    assert none[FEATURE_NAMES.index("news_unavailable")] == 0.0
    assert none[FEATURE_NAMES.index("news_compound")] == 0.0  # != neutral
    down = builder.build(obs, ramp(5), news=NewsObservation(status="unavailable"))
    assert down[FEATURE_NAMES.index("news_unavailable")] == 1.0


def _archive_with_article(tmp_path, published, first_seen_offset_days=0,
                          name="a.db"):
    from dataclasses import replace

    from noone.news_classifier import FinVADERClassifier
    from noone.news_data import NewsArchive
    from noone.news_data.crypto_news import normalize_article

    article = normalize_article({
        "title": "Bitcoin surges in rally", "link": "https://example.com/x",
        "description": "BTC breakout soars", "source": "CoinDesk",
        "pubDate": published.isoformat()})
    article = replace(
        article, retrieved_at=published + timedelta(days=first_seen_offset_days))
    path = tmp_path / name
    with NewsArchive(path) as arch:
        arch.store_classification(FinVADERClassifier().classify_article(article))
    return path


def test_adapter_news_point_in_time_and_stale(tmp_path):
    from noone.news_data import NewsArchive

    # Published 1h before candle 0: eligible and fresh.
    path = _archive_with_article(tmp_path, T0 - timedelta(hours=1))
    with NewsArchive(path) as arch:
        env = NooneGymEnv(ramp(), config=TradingConfig(
            fee_rate=0, slippage_rate=0, funding_rate=0),
            news_archive=arch, news_asset="BTC")
        obs, _ = env.reset(seed=0)
        assert obs[FEATURE_NAMES.index("news_available")] == 1.0
    # Published 48h ago: stale (>24h) -> treated as missing.
    path = _archive_with_article(tmp_path, T0 - timedelta(hours=48),
                                 name="stale.db")
    with NewsArchive(path) as arch:
        env = NooneGymEnv(ramp(), config=TradingConfig(
            fee_rate=0, slippage_rate=0, funding_rate=0),
            news_archive=arch, news_asset="BTC")
        obs, _ = env.reset(seed=0)
        assert obs[FEATURE_NAMES.index("news_available")] == 0.0
        assert obs[FEATURE_NAMES.index("news_unavailable")] == 0.0


def test_no_future_candle_leakage():
    # Identical prefixes, different futures -> identical observations
    # until the prefix ends.
    a = ramp(30)
    b = ramp(30)
    for i in range(20, 30):
        b[i] = candle(i, 50_000.0)
    env_a, env_b = NooneGymEnv(a), NooneGymEnv(b)
    obs_a, _ = env_a.reset(seed=0)
    obs_b, _ = env_b.reset(seed=0)
    assert np.array_equal(obs_a, obs_b)
    for _ in range(5):
        obs_a, *_ = env_a.step(0)
        obs_b, *_ = env_b.step(0)
    assert np.array_equal(obs_a, obs_b)


def test_adapter_delegates_accounting_to_component8():
    env = make_env()
    env.reset(seed=0)
    _o, reward, _t, _tr, info = env.step(1)
    # Reward/accounting come from Component 8 unchanged (fee-free cfg:
    # open + ramp move). Underlying env equity moved identically.
    assert info["raw_observation"]["equity"] == pytest.approx(env.env.equity)
    assert env.env.position is not None


def test_invalid_action_penalty_validation():
    """Verify invalid_action_penalty validation in NooneGymEnv."""
    # Valid constructions
    env = NooneGymEnv(ramp(30), invalid_action_penalty=0.002)
    assert env.invalid_action_penalty == 0.002
    assert NooneGymEnv(ramp(30), invalid_action_penalty=0.0).invalid_action_penalty == 0.0

    # Invalid constructions
    with pytest.raises(TypeError, match="invalid_action_penalty must be numeric"):
        NooneGymEnv(ramp(30), invalid_action_penalty="0.001")
    with pytest.raises(TypeError, match="invalid_action_penalty must be numeric"):
        NooneGymEnv(ramp(30), invalid_action_penalty=True)
    with pytest.raises(ValueError, match="invalid_action_penalty must be finite and non-negative"):
        NooneGymEnv(ramp(30), invalid_action_penalty=-0.001)
    with pytest.raises(ValueError, match="invalid_action_penalty must be finite and non-negative"):
        NooneGymEnv(ramp(30), invalid_action_penalty=float("nan"))


def test_invalid_action_penalty_all_15_position_action_combinations():
    """Verify all 15 position-action combinations: 7 valid receive no penalty, 8 invalid receive penalty exactly once."""
    candles = ramp(100)
    cfg = TradingConfig(fee_rate=0.0, slippage_rate=0.0, funding_rate=0.0, reward_scale=1.0)
    penalty_val = 0.001

    # 1. FLAT STATE (env starts flat)
    # Valid: HOLD (0), OPEN_LONG (1), OPEN_SHORT (3) -> NO penalty
    for valid_act in [0]:
        env = NooneGymEnv(candles, config=cfg, invalid_action_penalty=penalty_val)
        env.reset(seed=0)
        _, reward, _, _, info = env.step(valid_act)
        assert info.get("invalid_action") is False
        assert "invalid_action_penalty" not in info
        assert reward == 0.0

    # Invalid while flat: CLOSE_LONG (2), CLOSE_SHORT (4) -> PENALTY
    for invalid_act in [2, 4]:
        env = NooneGymEnv(candles, config=cfg, invalid_action_penalty=penalty_val)
        env.reset(seed=0)
        _, reward, _, _, info = env.step(invalid_act)
        assert info.get("invalid_action") is True
        assert info.get("invalid_action_penalty") == penalty_val
        assert reward == pytest.approx(-penalty_val)

    # 2. LONG STATE
    # Valid: HOLD (0), CLOSE_LONG (2) -> NO penalty
    for valid_act in [0, 2]:
        env = NooneGymEnv(candles, config=cfg, invalid_action_penalty=penalty_val)
        env.reset(seed=0)
        env.step(1)  # Open Long
        assert env.env.position is not None and env.env.position.side == 1
        _, reward, _, _, info = env.step(valid_act)
        assert info.get("invalid_action") is False
        assert "invalid_action_penalty" not in info

    # Invalid while Long: OPEN_LONG (1), OPEN_SHORT (3), CLOSE_SHORT (4) -> PENALTY
    for invalid_act in [1, 3, 4]:
        env = NooneGymEnv(candles, config=cfg, invalid_action_penalty=penalty_val)
        env.reset(seed=0)
        env.step(1)  # Open Long
        assert env.env.position is not None and env.env.position.side == 1
        raw_eq_before = env.env.equity
        _, reward, _, _, info = env.step(invalid_act)
        assert info.get("invalid_action") is True
        assert info.get("invalid_action_penalty") == penalty_val
        # Component 8 reward for pseudo-HOLD + penalty subtraction
        comp8_reward = (env.env.equity - raw_eq_before) / cfg.initial_balance * cfg.reward_scale
        assert reward == pytest.approx(comp8_reward - penalty_val)

    # 3. SHORT STATE
    # Valid: HOLD (0), CLOSE_SHORT (4) -> NO penalty
    for valid_act in [0, 4]:
        env = NooneGymEnv(candles, config=cfg, invalid_action_penalty=penalty_val)
        env.reset(seed=0)
        env.step(3)  # Open Short
        assert env.env.position is not None and env.env.position.side == -1
        _, reward, _, _, info = env.step(valid_act)
        assert info.get("invalid_action") is False
        assert "invalid_action_penalty" not in info

    # Invalid while Short: OPEN_LONG (1), CLOSE_LONG (2), OPEN_SHORT (3) -> PENALTY
    for invalid_act in [1, 2, 3]:
        env = NooneGymEnv(candles, config=cfg, invalid_action_penalty=penalty_val)
        env.reset(seed=0)
        env.step(3)  # Open Short
        assert env.env.position is not None and env.env.position.side == -1
        raw_eq_before = env.env.equity
        _, reward, _, _, info = env.step(invalid_act)
        assert info.get("invalid_action") is True
        assert info.get("invalid_action_penalty") == penalty_val
        comp8_reward = (env.env.equity - raw_eq_before) / cfg.initial_balance * cfg.reward_scale
        assert reward == pytest.approx(comp8_reward - penalty_val)


def test_invalid_action_penalty_scales_proportionately_with_reward_scale():
    """Verify penalty scales with reward_scale so gradient proportions remain consistent."""
    candles = ramp(50)
    cfg_100 = TradingConfig(fee_rate=0.0, slippage_rate=0.0, funding_rate=0.0, reward_scale=100.0)
    env = NooneGymEnv(candles, config=cfg_100, invalid_action_penalty=0.001)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(2)  # Invalid action while flat
    assert info.get("invalid_action") is True
    assert info.get("invalid_action_penalty") == pytest.approx(0.10)  # 0.001 * 100.0
    assert reward == pytest.approx(-0.10)

