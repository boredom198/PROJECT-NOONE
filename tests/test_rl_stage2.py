"""Component 1 Stage 2 tests: data prep, news variation, training, evaluation.

Small synthetic fixtures only; no live APIs. SB3 tests skip when the
RL extra is not installed.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from noone.market_data.models import Kline
from noone.rl_engine.config import DataConfig, DQNConfig, PipelineConfig
from noone.rl_engine.data import (
    DataPreparationError,
    chronological_split,
    prepare_candles,
)
from noone.rl_engine.news import NewsDropoutArchive, news_observation_at

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def candle(i, price=100.0, *, closed=True, start=T0):
    p = float(price)
    return Kline(symbol="BTCUSDT", interval="1m",
                 open_time=start + timedelta(minutes=i),
                 close_time=start + timedelta(minutes=i, seconds=59),
                 open=p, high=p, low=p, close=p, volume=10.0,
                 quote_volume=1000.0, trades=5, is_closed=closed)


def ramp(n=200, start=T0):
    return [candle(i, 100.0 + i * 0.05, start=start) for i in range(n)]


# -- Data preparation -------------------------------------------------

def test_prepare_sorts_dedups_and_drops_unclosed():
    raw = [candle(2), candle(0), candle(0), candle(1), candle(3, closed=False)]
    prepared = prepare_candles(raw, interval="1m")
    assert [c.open_time for c in prepared.candles] == sorted(
        c.open_time for c in prepared.candles)
    assert prepared.duplicates_removed == 1
    assert prepared.dropped_unclosed == 1
    assert len(prepared.candles) == 3


def test_prepare_gap_detection_not_filled():
    prepared = prepare_candles([candle(0), candle(1), candle(4)], interval="1m")
    assert len(prepared.candles) == 3  # gap reported, never fabricated
    assert len(prepared.gaps) == 1 and prepared.gaps[0].missing_candles == 2
    assert prepared.metadata()["missing_candles_in_gaps"] == 2


def test_prepare_invalid_and_insufficient_errors():
    bad = candle(0)
    object.__setattr__(bad, "close", -1.0)
    with pytest.raises(DataPreparationError, match="Invalid candle"):
        prepare_candles([bad, candle(1)], interval="1m")
    with pytest.raises(DataPreparationError, match="Insufficient"):
        prepare_candles(ramp(10), interval="1m", warmup_candles=50)
    with pytest.raises(DataPreparationError, match="No historical candles"):
        prepare_candles([], interval="1m")
    with pytest.raises(DataPreparationError, match="No closed candles"):
        prepare_candles([candle(0, closed=False), candle(1, closed=False)],
                        interval="1m")


def test_chronological_split_no_overlap_and_warmup_context():
    prepared = prepare_candles(ramp(200), interval="1m", warmup_candles=50)
    split = chronological_split(prepared, train_fraction=0.8, warmup_candles=50,
                                min_train_candles=100, min_eval_candles=20)
    assert len(split.train) == 160 and len(split.evaluation) == 40
    assert split.train[-1].open_time < split.evaluation[0].open_time
    assert not {c.open_time for c in split.train} & {
        c.open_time for c in split.evaluation}
    # Warm-up context is the tail of train, excluded from eval decisions.
    assert split.eval_warmup_context == split.train[-50:]
    assert not {c.open_time for c in split.eval_warmup_context} & {
        c.open_time for c in split.evaluation}
    meta = split.metadata()
    assert meta["train_end"] < meta["evaluation_start"]


def test_split_insufficient_periods_error():
    prepared = prepare_candles(ramp(60), interval="1m")
    with pytest.raises(DataPreparationError):
        chronological_split(prepared, train_fraction=0.8, warmup_candles=50,
                            min_train_candles=100, min_eval_candles=50)


# -- News -------------------------------------------------------------

class FakeArchive:
    """Archive stub honouring published/first_seen point-in-time rules."""

    def __init__(self, articles, classifications):
        self._articles, self._cls = articles, classifications

    def available_at(self, decision, *, asset=None, limit=500):
        return [a for a in self._articles
                if a.published_at and a.published_at <= decision
                and a.first_seen_at and a.first_seen_at <= decision]

    def classification_for(self, article_id):
        return self._cls.get(article_id)


def _article(aid, published, first_seen=None, compound=0.6, uncertain=False):
    art = SimpleNamespace(article_id=aid, published_at=published,
                          first_seen_at=first_seen or published)
    cls = {"compound": compound, "confidence_proxy": 0.8, "uncertain": uncertain}
    return art, cls


def test_news_point_in_time_stale_and_uncertain():
    decision = T0 + timedelta(hours=2)
    fresh, fresh_cls = _article("fresh", T0 + timedelta(hours=1))
    future, future_cls = _article("future", T0 + timedelta(hours=5))
    stale, stale_cls = _article("stale", T0 - timedelta(hours=48))
    uncert, uncert_cls = _article("uncert", T0 + timedelta(hours=1), uncertain=True)
    archive = FakeArchive(
        [fresh, future, stale, uncert],
        {"fresh": fresh_cls, "future": future_cls,
         "stale": stale_cls, "uncert": uncert_cls})
    obs = news_observation_at(archive, decision, asset="BTC")
    assert obs.status == "fresh" and obs.count == 1  # only fresh counts
    only_future = FakeArchive([future], {"future": future_cls})
    assert news_observation_at(only_future, decision, asset="BTC").status == "none"


def test_news_dropout_distinct_from_failure_and_reproducible():
    art, cls = _article("a", T0)
    archive = FakeArchive([art], {"a": cls})
    dropped = NewsDropoutArchive(archive, dropout_probability=1.0, seed=7)
    assert dropped.available_at(T0, asset="BTC") == []
    assert dropped.last_query_dropped and dropped.dropout_count == 1
    # Dropout yields the "none" (market-only) state, never "unavailable".
    assert news_observation_at(dropped, T0, asset="BTC").status == "none"

    class Failing:
        def available_at(self, *a, **k):
            raise RuntimeError("API down")

        def classification_for(self, _id):
            return None

    assert news_observation_at(Failing(), T0, asset="BTC").status == "unavailable"
    seqs = []
    for _ in range(2):
        wrapper = NewsDropoutArchive(archive, dropout_probability=0.5, seed=123)
        seqs.append([wrapper.available_at(T0, asset="BTC") == []
                     for _ in range(20)])
    assert seqs[0] == seqs[1] and any(seqs[0]) and not all(seqs[0])


# -- Training / evaluation (SB3) ---------------------------------------

def _small_config(tmp_path, **dqn_kw):
    dqn = dict(net_arch=(32, 32), buffer_size=500, batch_size=16,
               learning_starts=10, train_freq=1, total_timesteps=128,
               checkpoint_freq=64, seed=42)
    dqn.update(dqn_kw)
    return PipelineConfig(
        dqn=DQNConfig(**dqn),
        data=DataConfig(symbol="BTCUSDT", interval="1m", warmup_candles=10,
                        min_train_candles=20, min_eval_candles=10),
        output_dir=str(tmp_path / "runs"))


def test_smoke_training_checkpoint_metadata_and_resume(tmp_path):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import DQN

    from noone.rl_engine.training import train_dqn, validate_resume
    from noone.trading_env import TradingConfig

    cfg = _small_config(tmp_path)
    free = TradingConfig(fee_rate=0.0, slippage_rate=0.0, funding_rate=0.0)
    result = train_dqn(ramp(80), config=cfg, trading_config=free)
    run_dir = tmp_path / "runs"
    assert result["final_model"].endswith("final_model.zip")
    import json
    from pathlib import Path

    meta = json.loads((Path(result["run_dir"]) / "metadata.json").read_text())
    assert (Path(result["run_dir"]) / "config.json").exists()
    assert meta["observation_dim"] == 26 and meta["action_count"] == 5
    assert meta["feature_schema_version"] == "noone-features-v1"
    assert "feature_schema_hash" in meta and len(meta["feature_schema_hash"]) == 64
    assert meta["data"]["candle_count"] == 80
    assert list((Path(result["run_dir"]) / "checkpoints").glob("*.zip"))

    env_candles = ramp(80)
    from noone.rl_engine.training import make_env

    loaded = DQN.load(result["final_model"])
    obs, _ = make_env(env_candles, trading_config=free,
                      data_config=cfg.data).reset(seed=3)
    a1, _ = loaded.predict(obs, deterministic=True)
    a2, _ = loaded.predict(obs, deterministic=True)
    assert int(a1) == int(a2)

    # Resume with identical config works; incompatible arch / missing
    # metadata fail with clear errors.
    resumed = train_dqn(ramp(80), config=_small_config(
        tmp_path, total_timesteps=32), trading_config=free,
        resume_from=result["final_model"])
    assert Path(resumed["final_model"]).exists()
    bad_env = make_env(env_candles, trading_config=free, data_config=cfg.data)
    with pytest.raises(ValueError, match="Incompatible checkpoint"):
        validate_resume(result["final_model"], bad_env,
                        DQNConfig(net_arch=(64, 64)))
    isolated = tmp_path / "isolated.zip"
    import shutil

    shutil.copy(result["final_model"], isolated)
    with pytest.raises(ValueError, match="no metadata"):
        validate_resume(isolated, bad_env, cfg.dqn)
    with pytest.raises(FileNotFoundError):
        validate_resume(tmp_path / "missing.zip", bad_env, cfg.dqn)
    assert run_dir.exists()


def test_evaluation_deterministic_no_weight_update_and_no_trade(tmp_path):
    pytest.importorskip("stable_baselines3")
    from noone.rl_engine.evaluation import compute_metrics, evaluate_model
    from noone.rl_engine.training import build_dqn, make_env
    from noone.trading_env import TradingConfig

    free = TradingConfig(fee_rate=0.0, slippage_rate=0.0, funding_rate=0.0)
    cfg = _small_config(tmp_path)
    env = make_env(ramp(60), trading_config=free, data_config=cfg.data)
    model = build_dqn(env, cfg.dqn)
    eval_candles = ramp(40, start=T0 + timedelta(minutes=500))
    first = evaluate_model(model, eval_candles, trading_config=free,
                           data_config=cfg.data, seed=9)
    second = evaluate_model(model, eval_candles, trading_config=free,
                            data_config=cfg.data, seed=9)
    assert first["actions"] == second["actions"]  # deterministic
    assert first["weights_changed"] is False
    assert first["metrics"]["final_equity"] == pytest.approx(
        second["metrics"]["final_equity"])
    for key in ("total_return", "max_drawdown", "num_trades", "win_rate",
                "total_fees", "total_funding", "final_balance", "final_equity"):
        assert key in first["metrics"]
    hold = first["baselines"]["always_hold"]
    assert hold["num_trades"] == 0 and hold["win_rate"] == 0.0
    assert "random_fixed_seed" in first["baselines"]
    no_trade = compute_metrics([10_000.0, 10_000.0], initial_balance=10_000.0,
                               trade_pnls=[], total_fees=0.0,
                               total_funding=0.0, final_balance=10_000.0)
    assert no_trade["num_trades"] == 0 and no_trade["win_rate"] == 0.0
    assert no_trade["total_return"] == 0.0 and no_trade["max_drawdown"] == 0.0


def test_end_to_end_training_to_strict_inference_and_evaluation(tmp_path):
    """Verify full end-to-end pipeline: train -> ModelInference.load (strict) -> predict -> evaluate."""
    pytest.importorskip("stable_baselines3")
    import numpy as np
    from noone.rl_engine import AccountSnapshot, Action, ModelInference
    from noone.rl_engine.evaluation import evaluate_model
    from noone.rl_engine.training import train_dqn
    from noone.trading_env import TradingConfig

    cfg = _small_config(tmp_path, total_timesteps=64)
    free = TradingConfig(fee_rate=0.0, slippage_rate=0.0, funding_rate=0.0)

    # 1. Train model
    train_res = train_dqn(ramp(80), config=cfg, trading_config=free)
    model_path = train_res["final_model"]

    # 2. Strict ModelInference.load (must succeed without warnings and without allow_legacy_metadata)
    mi = ModelInference.load(model_path, allow_legacy_metadata=False, expected_net_arch=(32, 32))
    assert mi is not None
    assert mi.news_max_age_hours == cfg.data.news_max_age_hours
    assert mi.is_legacy_metadata is False

    # 3. Single observation inference
    candles = ramp(60)
    account = AccountSnapshot(equity=10_000.0, available_balance=10_000.0)
    pred = mi.predict(candles, account)
    assert isinstance(pred.action, Action)
    assert len(pred.q_values) == 5
    assert pred.q_margin >= 0.0

    # 4. Vectorized batch inference
    obs_batch = np.zeros((10, 26), dtype=np.float32)
    batch_preds = mi.predict_observations(obs_batch)
    assert len(batch_preds) == 10
    assert all(isinstance(p.action, Action) for p in batch_preds)

    # 5. Deterministic evaluation routing through ModelInference
    eval_candles = ramp(40, start=T0 + timedelta(minutes=500))
    warmup_context = ramp(20, start=T0 + timedelta(minutes=480))
    eval_res = evaluate_model(model_path, eval_candles, warmup_context=warmup_context,
                              trading_config=free, data_config=cfg.data, seed=123)
    assert eval_res["deterministic"] is True
    assert eval_res["weights_changed"] is False
    assert len(eval_res["actions"]) == len(eval_candles) - 1


def test_evaluate_model_rejects_incompatible_or_corrupt_metadata(tmp_path):
    """Verify evaluate_model enforces strict metadata validation on model paths."""
    pytest.importorskip("stable_baselines3")
    import json
    import shutil
    from pathlib import Path
    from noone.rl_engine import IncompatibleModelError, ModelNotFoundError
    from noone.rl_engine.evaluation import evaluate_model
    from noone.rl_engine.training import train_dqn
    from noone.trading_env import TradingConfig

    cfg = _small_config(tmp_path, total_timesteps=32)
    free = TradingConfig(fee_rate=0.0, slippage_rate=0.0, funding_rate=0.0)
    train_res = train_dqn(ramp(80), config=cfg, trading_config=free)
    model_path = Path(train_res["final_model"])
    meta_path = Path(train_res["run_dir"]) / "metadata.json"

    eval_candles = ramp(30, start=T0 + timedelta(minutes=500))

    # 1. Missing metadata raises ModelNotFoundError
    isolated_dir = tmp_path / "isolated_eval"
    isolated_dir.mkdir()
    isolated_model = isolated_dir / "model.zip"
    shutil.copy(model_path, isolated_model)
    with pytest.raises(ModelNotFoundError):
        evaluate_model(isolated_model, eval_candles, data_config=cfg.data)

    # 2. Corrupt metadata schema hash raises IncompatibleModelError
    corrupt_meta = json.loads(meta_path.read_text())
    corrupt_meta["feature_schema_hash"] = "0" * 64
    (isolated_dir / "metadata.json").write_text(json.dumps(corrupt_meta))
    with pytest.raises(IncompatibleModelError, match="Incompatible feature schema hash"):
        evaluate_model(isolated_model, eval_candles, data_config=cfg.data)

    # 3. Legacy metadata (lacking hash) fails in strict mode but succeeds when allow_legacy_metadata=True
    del corrupt_meta["feature_schema_hash"]
    del corrupt_meta["feature_schema_version"]
    (isolated_dir / "metadata.json").write_text(json.dumps(corrupt_meta))
    with pytest.raises(IncompatibleModelError, match="lacks required strict schema fields"):
        evaluate_model(isolated_model, eval_candles, data_config=cfg.data, allow_legacy_metadata=False)

    legacy_res = evaluate_model(isolated_model, eval_candles, data_config=cfg.data, allow_legacy_metadata=True)
    assert legacy_res["deterministic"] is True


def test_evaluate_model_in_memory_sb3_and_inference_instances(tmp_path):
    """Verify evaluate_model supports both in-memory ModelInference and raw SB3 model."""
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import DQN
    from noone.rl_engine import ModelInference
    from noone.rl_engine.evaluation import evaluate_model
    from noone.rl_engine.training import train_dqn
    from noone.trading_env import TradingConfig

    cfg = _small_config(tmp_path, total_timesteps=32)
    free = TradingConfig(fee_rate=0.0, slippage_rate=0.0, funding_rate=0.0)
    train_res = train_dqn(ramp(80), config=cfg, trading_config=free)
    eval_candles = ramp(30, start=T0 + timedelta(minutes=500))

    # In-memory ModelInference instance
    mi = ModelInference.load(train_res["final_model"], expected_net_arch=(32, 32))
    res_mi = evaluate_model(mi, eval_candles, data_config=cfg.data, seed=42)
    assert res_mi["deterministic"] is True

    # In-memory SB3 DQN model directly
    raw_sb3 = DQN.load(train_res["final_model"], device="cpu")
    res_sb3 = evaluate_model(raw_sb3, eval_candles, data_config=cfg.data, seed=42)
    assert res_sb3["deterministic"] is True
    assert res_mi["actions"] == res_sb3["actions"]


def test_checkpoint_resume_schema_hash_consistency(tmp_path):
    """Verify validate_resume rejects checkpoints with mismatched feature schema hash."""
    pytest.importorskip("stable_baselines3")
    import json
    from pathlib import Path
    from noone.rl_engine.training import make_env, train_dqn, validate_resume
    from noone.trading_env import TradingConfig

    cfg = _small_config(tmp_path, total_timesteps=32)
    free = TradingConfig(fee_rate=0.0, slippage_rate=0.0, funding_rate=0.0)
    train_res = train_dqn(ramp(80), config=cfg, trading_config=free)
    model_path = train_res["final_model"]
    run_dir = Path(train_res["run_dir"])

    env = make_env(ramp(80), trading_config=free, data_config=cfg.data)
    # Valid resume succeeds
    meta = validate_resume(model_path, env, cfg.dqn)
    assert meta["feature_schema_version"] == "noone-features-v1"

    # Corrupt feature_schema_hash in metadata causes validate_resume to reject
    meta_path = run_dir / "metadata.json"
    raw_meta = json.loads(meta_path.read_text())
    raw_meta["feature_schema_hash"] = "f" * 64
    meta_path.write_text(json.dumps(raw_meta))
    with pytest.raises(ValueError, match="Incompatible checkpoint: metadata feature_schema_hash"):
        validate_resume(model_path, env, cfg.dqn)

