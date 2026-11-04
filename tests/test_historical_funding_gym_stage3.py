"""Stage 3: historical funding through Gymnasium / training integration.

The full Component 4 ``FundingHistory`` is forwarded unchanged to
every episode env; constant mode, the 26-feature observation, and
Component 8 accounting are unchanged.
"""
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")

from noone.market_data.models import (  # noqa: E402
    FundingContext,
    FundingHistory,
    FundingSettlement,
    Kline,
)
from noone.rl_engine.config import DataConfig, DQNConfig, PipelineConfig  # noqa: E402
from noone.rl_engine.features import FEATURE_NAMES  # noqa: E402
from noone.rl_engine.gym_env import NooneGymEnv  # noqa: E402
from noone.trading_env import TradingConfig  # noqa: E402

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_candle(open_time, price=100.0):
    return Kline(symbol="BTCUSDT", interval="1h", open_time=open_time,
                 close_time=open_time + timedelta(minutes=59),
                 open=price, high=price, low=price, close=price,
                 volume=10.0, quote_volume=1000.0, trades=5, is_closed=True)


def hourly(n=20, start=T0):
    return [make_candle(start + timedelta(hours=i)) for i in range(n)]


def at(hours=0, minutes=0, start=T0):
    return start + timedelta(hours=hours, minutes=minutes)


def settlement(dt, rate="0.001", mark="100", rate_type=None):
    return FundingSettlement(symbol="BTCUSDT", funding_time=dt,
                             funding_rate=Decimal(rate),
                             mark_price=Decimal(mark), rate_type=rate_type)


def context(direction, times):
    return FundingContext(
        direction=direction,
        settlements=tuple(settlement(t) for t in times),
        malformed_records=(), requested_count=3, status="sufficient",
        pages_fetched=1, pagination_complete=True,
        pagination_termination="range_exhausted")


def history_for(cs, settlements):
    start, last = cs[0].open_time, cs[-1]
    end = last.close_time
    return FundingHistory(
        symbol="BTCUSDT", settlements=tuple(settlements),
        requested_start_time=start, requested_end_time=end,
        coverage_start_time=start, coverage_end_time=end,
        collected_at=end + timedelta(hours=1), pages_fetched=1,
        duplicates_removed=0, conflicting_records=(), malformed_records=(),
        suspected_anomalies=(), pagination_complete=True,
        pagination_termination="range_exhausted",
        total_limit=100, page_limit=1000,
        before_context=context(
            "before", [start - timedelta(hours=h) for h in (24, 16, 8)]),
        after_context=context(
            "after", [end + timedelta(hours=h) for h in (8, 16, 24)]),
        scoped_blind_spots=())


def hist_cfg(**kw):
    base = dict(fee_rate=0.0, slippage_rate=0.0, funding_mode="historical")
    base.update(kw)
    return TradingConfig(**base)


def hist_gym(cs, settlements, **kw):
    hist = history_for(cs, settlements)
    return NooneGymEnv(cs, hist_cfg(), funding_history=hist, **kw), hist


# -- Gymnasium integration -------------------------------------------------

def test_historical_funding_through_gymnasium():
    env, _ = hist_gym(hourly(8), [settlement(at(2, 30))])
    obs, _ = env.reset(seed=0, options={"start_index": 0})
    assert obs.shape == (26,)
    env.step(1)  # OPEN_LONG
    _, _, _, _, info = env.step(0)
    assert env.env.total_funding == pytest.approx(-5.0)
    assert len(env.env.funding_ledger) == 1
    assert info["funding_mode"] == "historical"


def test_constant_mode_gym_compatibility():
    cfg = TradingConfig(fee_rate=0.0, slippage_rate=0.0, funding_rate=0.01,
                        funding_times_utc=("00:00", "08:00", "16:00"))
    env = NooneGymEnv(hourly(10), cfg)
    env.reset(seed=0, options={"start_index": 0})
    env.step(1)
    for _ in range(7):
        env.step(0)
    assert env.env.total_funding == pytest.approx(-50.0)
    assert env.env.funding_ledger == []


def test_full_history_identity_preserved_across_episodes():
    env, hist = hist_gym(hourly(20), [settlement(at(10, 30))],
                         episode_window=8)
    env.reset(seed=1, options={"start_index": 0})
    first_env = env.env
    assert first_env.funding_history is hist
    env.reset(seed=1, options={"start_index": 8})
    assert env.env is not first_env
    assert env.env.funding_history is hist
    assert env.env.funding_history.before_context is hist.before_context
    assert env.env.funding_history.after_context is hist.after_context
    assert env.funding_history is hist


def test_fail_fast_coverage_validation():
    cs = hourly(20)
    partial = history_for(cs[:8], [settlement(at(2, 30))])
    with pytest.raises(ValueError, match="full.*candle range|coverage"):
        NooneGymEnv(cs, hist_cfg(), episode_window=5,
                    funding_history=partial)


def test_episode_specific_settlement_accounting():
    cs = hourly(20)
    env, _ = hist_gym(cs, [settlement(at(2, 30)),
                           settlement(at(10, 30))], episode_window=8)
    env.reset(seed=0, options={"start_index": 0})
    env.step(1)
    for _ in range(3):
        env.step(0)
    assert [e.funding_time for e in env.env.funding_ledger] == [at(2, 30)]
    env.reset(seed=0, options={"start_index": 8})
    env.step(1)
    for _ in range(3):
        env.step(0)
    assert [e.funding_time for e in env.env.funding_ledger] == [at(10, 30)]
    assert env.env.total_funding == pytest.approx(-5.0)


def test_gym_first_open_and_final_close_boundaries():
    cs = hourly(8)
    env, _ = hist_gym(cs, [settlement(at(1)), settlement(at(5, 59))],
                      episode_window=8)
    env.reset(seed=0, options={"start_index": 0})
    env.step(1)  # opens at 01:00 settlement: not charged
    assert env.env.funding_ledger == []
    for _ in range(4):
        env.step(0)  # through candle 5, close 05:59 settlement charged
    assert [e.funding_time for e in env.env.funding_ledger] == [at(5, 59)]


def test_repeated_resets_reproduce_and_clear_ledger():
    def run():
        env, _ = hist_gym(hourly(8), [settlement(at(2, 30))])
        env.reset(seed=7, options={"start_index": 0})
        env.step(1)
        env.step(0)
        return env.env.total_funding, len(env.env.funding_ledger), env

    total1, count1, env = run()
    env.reset(seed=7, options={"start_index": 0})
    assert env.env.funding_ledger == [] and env.env.total_funding == 0.0
    env.step(1)
    env.step(0)
    assert env.env.total_funding == total1 and len(env.env.funding_ledger) == count1


def test_observation_shape_and_feature_invariance():
    assert len(FEATURE_NAMES) == 26
    assert FEATURE_NAMES[0] == "log_return_1"
    assert FEATURE_NAMES[-1] == "news_age_normalized"
    const_env = NooneGymEnv(hourly(8), TradingConfig(funding_rate=0.0))
    hist_env_, _ = hist_gym(hourly(8), [settlement(at(2, 30))])
    obs_c, _ = const_env.reset(seed=3, options={"start_index": 0})
    obs_h, _ = hist_env_.reset(seed=3, options={"start_index": 0})
    assert obs_c.shape == obs_h.shape == (26,)
    assert obs_c.dtype == obs_h.dtype == np.float32
    assert np.array_equal(obs_c, obs_h)


# -- Warm-up and evaluation --------------------------------------------------

def _run(action_fn, warm, evalc, settlements):
    from noone.rl_engine.evaluation import _run_episode

    combined = list(warm) + list(evalc)
    return _run_episode(
        evalc, warm, action_fn, trading_config=hist_cfg(),
        data_config=DataConfig(symbol="BTCUSDT", interval="1h"),
        news_archive=None, seed=42,
        funding_history=history_for(combined, settlements))


def test_warmup_settlements_consumed_without_charging():
    warm = hourly(4, start=T0 - timedelta(hours=4))
    evalc = hourly(6)
    calls = {"n": 0}

    def act(_obs):
        calls["n"] += 1
        return 1 if calls["n"] == 1 else 0  # open on first eval action

    result = _run(act, warm, evalc,
                  [settlement(at(-2, 30)), settlement(at(2, 30))])
    funding = result["funding"]
    assert funding["ledger_entry_count"] == 1
    assert funding["total_funding"] == pytest.approx(-5.0)
    assert funding["reconciliation_status"] == "passed"
    assert funding["warmup_range"]["candle_count"] == 4
    assert funding["funding_history"]["settlement_count"] == 2


def test_evaluation_history_must_cover_warmup_range():
    warm = hourly(4, start=T0 - timedelta(hours=4))
    evalc = hourly(6)
    with pytest.raises(ValueError, match="requested"):
        _run_eval_only_history(warm, evalc)


def _run_eval_only_history(warm, evalc):
    from noone.rl_engine.evaluation import _run_episode

    return _run_episode(
        evalc, warm, lambda _o: 0, trading_config=hist_cfg(),
        data_config=DataConfig(symbol="BTCUSDT", interval="1h"),
        news_archive=None, seed=42,
        funding_history=history_for(evalc, [settlement(at(2, 30))]))


def test_independent_evaluation_ledgers():
    warm, evalc = [], hourly(8)
    opened = _run(lambda _o: 1, warm, evalc, [settlement(at(2, 30))])
    # First action opens; subsequent _run calls are independent envs.
    held = _run(lambda _o: 0, warm, evalc, [settlement(at(2, 30))])
    assert opened["funding"]["ledger_entry_count"] >= 1
    assert held["funding"]["ledger_entry_count"] == 0
    assert held["funding"]["total_funding"] == 0.0
    assert opened["funding"] is not held["funding"]


def test_evaluation_metadata_fields():
    result = _run(lambda _o: 0, hourly(2, start=T0 - timedelta(hours=2)),
                  hourly(6), [settlement(at(2, 30))])
    funding = result["funding"]
    assert funding["funding_mode"] == "historical"
    assert funding["historical_accounting_implemented"] is True
    assert funding["evaluation_range"]["start"] == T0.isoformat()
    assert funding["warmup_range"]["candle_count"] == 2
    assert funding["ledger_funding_total"] == pytest.approx(
        funding["total_funding"])
    assert funding["funding_history"]["before_context"]["status"] == \
        "sufficient"


# -- Data helper ---------------------------------------------------------------

def test_fetch_funding_history_uses_context_limit_3(monkeypatch):
    import noone.market_data as market_data
    from noone.rl_engine.data import fetch_funding_history

    hist = history_for(hourly(4), [settlement(at(1, 30))])
    calls = {}

    class FakeCollector:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def historical_funding_rates(self, symbol, **kwargs):
            calls.update(kwargs)
            calls["symbol"] = symbol
            return hist

    monkeypatch.setattr(market_data, "BinanceFuturesCollector", FakeCollector)
    result = fetch_funding_history(symbol="BTCUSDT", start_time=T0,
                                   end_time=at(3, 59))
    assert result is hist
    assert calls["context_limit"] == 3
    assert calls["symbol"] == "BTCUSDT"


# -- Training metadata and resume ----------------------------------------------

def _small_pipeline(tmp_path, timesteps=64):
    return PipelineConfig(
        dqn=DQNConfig(net_arch=(32, 32), buffer_size=500, batch_size=16,
                      learning_starts=10, train_freq=1,
                      total_timesteps=timesteps, checkpoint_freq=32,
                      seed=42),
        data=DataConfig(symbol="BTCUSDT", interval="1h", warmup_candles=5,
                        min_train_candles=20, min_eval_candles=10),
        output_dir=str(tmp_path / "runs"))


def test_training_metadata_historical(tmp_path):
    pytest.importorskip("stable_baselines3")
    from noone.rl_engine.training import train_dqn

    cs = hourly(80)
    hist = history_for(cs, [settlement(at(10, 30)), settlement(at(40, 30))])
    result = train_dqn(cs, config=_small_pipeline(tmp_path),
                       trading_config=hist_cfg(), funding_history=hist)
    meta = json.loads((Path(result["run_dir"]) / "metadata.json")
                      .read_text())
    funding = meta["funding"]
    assert funding["funding_mode"] == "historical"
    assert funding["historical_accounting_implemented"] is True
    assert funding["funding_history"]["settlement_count"] == 2
    assert funding["validation_status"] == "strict:complete"
    assert funding["episode_funding_statistics"]["basis"] == \
        "independent_episodes_not_continuous_account"
    assert "training_episode_count" in funding


def test_training_metadata_constant(tmp_path):
    pytest.importorskip("stable_baselines3")
    from noone.rl_engine.training import train_dqn

    result = train_dqn(hourly(80), config=_small_pipeline(tmp_path),
                       trading_config=TradingConfig(funding_rate=0.0))
    meta = json.loads((Path(result["run_dir"]) / "metadata.json")
                      .read_text())
    assert meta["funding"]["funding_mode"] == "constant"
    assert meta["funding"]["funding_history"] is None


def _write_checkpoint(tmp_path, metadata):
    ckpt = tmp_path / "model.zip"
    ckpt.write_bytes(b"placeholder")
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    return ckpt


def _base_metadata():
    return {"observation_dim": 26, "action_count": 5}


def test_resume_funding_match_and_mismatches(tmp_path):
    from noone.rl_engine.training import make_env, validate_resume

    cs = hourly(20)
    hist = history_for(cs, [settlement(at(10, 30))])
    hist_env = make_env(cs, trading_config=hist_cfg(),
                        data_config=DataConfig(interval="1h"),
                        funding_history=hist)
    const_env = make_env(cs, trading_config=TradingConfig(),
                         data_config=DataConfig(interval="1h"))
    saved = _base_metadata()
    saved["funding"] = {"funding_mode": "historical",
                        "funding_history": hist.metadata()}
    ckpt = _write_checkpoint(tmp_path, saved)
    assert validate_resume(ckpt, hist_env, DQNConfig())[
        "funding_resume_status"] == "funding_metadata_matched"
    with pytest.raises(ValueError, match="funding_mode"):
        validate_resume(ckpt, const_env, DQNConfig())
    other = history_for(cs, [settlement(at(10, 30)),
                             settlement(at(12, 30))])
    other_env = make_env(cs, trading_config=hist_cfg(),
                         data_config=DataConfig(interval="1h"),
                         funding_history=other)
    with pytest.raises(ValueError, match="settlement count"):
        validate_resume(ckpt, other_env, DQNConfig())


def test_resume_legacy_checkpoint_handling(tmp_path):
    from noone.rl_engine.training import make_env, validate_resume

    cs = hourly(20)
    hist = history_for(cs, [settlement(at(10, 30))])
    hist_env = make_env(cs, trading_config=hist_cfg(),
                        data_config=DataConfig(interval="1h"),
                        funding_history=hist)
    const_env = make_env(cs, trading_config=TradingConfig(),
                         data_config=DataConfig(interval="1h"))
    ckpt = _write_checkpoint(tmp_path, _base_metadata())
    result = validate_resume(ckpt, const_env, DQNConfig())
    assert result["funding_resume_status"] == \
        "legacy_checkpoint_no_funding_metadata_constant_mode"
    with pytest.raises(ValueError, match="legacy checkpoint"):
        validate_resume(ckpt, hist_env, DQNConfig())
