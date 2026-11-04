"""FundingHistory persistence and historical CLI resume (Stage 3 fix).

The exact original FundingHistory is persisted as a versioned,
checksummed JSON artifact so historical training can resume across
processes with its original collected_at and provenance intact.
"""
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from noone.market_data.models import (
    BoundaryProbeLimitHit,
    FundingBlindSpot,
    FundingConflict,
    FundingContext,
    FundingHistory,
    FundingIntervalAnomaly,
    FundingIntervalAssessment,
    FundingMalformedOccurrence,
    FundingMalformedRecord,
    FundingSettlement,
    Kline,
)
from noone.rl_engine.data import (
    DataPreparationError,
    load_funding_history,
    save_funding_history,
)
from noone.trading_env import TradingConfig

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def at(hours=0, minutes=0):
    return T0 + timedelta(hours=hours, minutes=minutes)


def settlement(dt, rate="0.00010000", mark="67234.10000000", rate_type=None):
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


def rich_history():
    anomaly = FundingIntervalAnomaly(
        after_time=at(16), before_time=at(32), interval_seconds=57600.0,
        reference_interval_seconds=28800.0, estimated_missing_count=1,
        acknowledged=True)
    assessment = FundingIntervalAssessment(
        symbol="BTCUSDT", rate_type=None, previous_time=at(16),
        next_time=at(32), observed_interval_seconds=57600.0,
        reference_interval_seconds=28800.0, estimated_interval_multiple=2.0,
        supporting_before_seconds=(28800.0, 28800.0),
        supporting_after_seconds=(28800.0,), 
        classification="SUSPECTED_MISSING_SETTLEMENT",
        reason="gap is a near-integer multiple", evidence_sufficient=True,
        acknowledged=True)
    occurrence = FundingMalformedOccurrence(
        page_index=0, page_start_ms=1, page_end_ms=2, record_index=3)
    malformed = FundingMalformedRecord(
        symbol="BTCUSDT", funding_time=at(8),
        reason="missing_or_invalid_mark_price", raw={"fundingRate": "0.1"},
        page_index=0, page_start_ms=1, page_end_ms=2, record_index=3,
        occurrence_count=1, occurrences=(occurrence,))
    return FundingHistory(
        symbol="BTCUSDT",
        settlements=(settlement(at(0)), settlement(at(8), rate_type="Regular"),
                     settlement(at(8), rate="0.00020000", rate_type="Special")),
        requested_start_time=T0, requested_end_time=at(48),
        coverage_start_time=T0, coverage_end_time=at(48),
        collected_at=at(49, 15), pages_fetched=3, duplicates_removed=2,
        conflicting_records=(FundingConflict(
            symbol="BTCUSDT", funding_time=at(8), rate_type=None,
            kept_rate=Decimal("0.0001"), kept_mark_price=Decimal("100"),
            conflicting_rate=Decimal("0.0002"),
            conflicting_mark_price=Decimal("101"), raw={"x": 1}),),
        malformed_records=(malformed,),
        suspected_anomalies=(anomaly,), pagination_complete=True,
        pagination_termination="range_exhausted", total_limit=100,
        page_limit=1000,
        boundary_probe_limit_hits=(BoundaryProbeLimitHit(
            funding_time=at(8), returned_count=1000),),
        blind_spots=("legacy-blind-spot",),
        interval_assessments=(assessment,),
        before_context=context("before", [at(-24), at(-16), at(-8)]),
        after_context=context("after", [at(56), at(64), at(72)]),
        scoped_blind_spots=(FundingBlindSpot(
            boundary="internal", start_time=at(8), end_time=at(16),
            reason="test", context_requested=3, context_received=2,
            pagination_incomplete=True),))


def hourly(n=20, start=T0):
    return [Kline(symbol="BTCUSDT", interval="1h",
                  open_time=start + timedelta(hours=i),
                  close_time=start + timedelta(hours=i, minutes=59),
                  open=100.0, high=100.0, low=100.0, close=100.0,
                  volume=10.0, quote_volume=1000.0, trades=5,
                  is_closed=True) for i in range(n)]


def simple_history(cs=None):
    cs = cs or hourly()
    start, end = cs[0].open_time, cs[-1].close_time
    return FundingHistory(
        symbol="BTCUSDT", settlements=(settlement(at(2, 30), "0.001", "100"),),
        requested_start_time=start, requested_end_time=end,
        coverage_start_time=start, coverage_end_time=end,
        collected_at=at(30), pages_fetched=1, duplicates_removed=0,
        conflicting_records=(), malformed_records=(), suspected_anomalies=(),
        pagination_complete=True, pagination_termination="range_exhausted",
        total_limit=100, page_limit=1000,
        before_context=context(
            "before", [start - timedelta(hours=h) for h in (24, 16, 8)]),
        after_context=context(
            "after", [end + timedelta(hours=h) for h in (8, 16, 24)]),
        scoped_blind_spots=())


# -- Serialization -----------------------------------------------------------

def test_complete_round_trip_preserves_everything(tmp_path):
    hist = rich_history()
    path = tmp_path / "funding_history.json"
    info = save_funding_history(hist, path)
    assert info["artifact_schema_version"] == 1
    assert info["collection_identity"] == hist.collected_at.isoformat()
    loaded = load_funding_history(path)
    assert loaded == hist
    assert loaded.collected_at == hist.collected_at
    assert loaded.settlements[0].funding_rate == Decimal("0.00010000")
    assert loaded.settlements[0].mark_price == Decimal("67234.10000000")
    assert loaded.before_context.settlements[0].funding_time == at(-24)
    assert loaded.suspected_anomalies[0].acknowledged is True
    assert loaded.interval_assessments[0].acknowledged is True
    assert loaded.conflicting_records[0].conflicting_rate == Decimal("0.0002")
    assert loaded.malformed_records[0].occurrences[0].record_index == 3
    assert loaded.boundary_probe_limit_hits[0].returned_count == 1000
    assert loaded.scoped_blind_spots[0].boundary == "internal"
    assert loaded.pagination_termination == "range_exhausted"


def test_schema_version_rejected(tmp_path):
    path = tmp_path / "a.json"
    save_funding_history(simple_history(), path)
    doc = json.loads(path.read_text())
    doc["schema_version"] = 999
    path.write_text(json.dumps(doc))
    with pytest.raises(DataPreparationError, match="schema_version"):
        load_funding_history(path)


def test_corrupted_checksum_rejected(tmp_path):
    path = tmp_path / "a.json"
    save_funding_history(simple_history(), path)
    doc = json.loads(path.read_text())
    doc["payload"]["settlements"] = []
    path.write_text(json.dumps(doc))
    with pytest.raises(DataPreparationError, match="checksum"):
        load_funding_history(path)


def test_missing_and_malformed_artifact_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_funding_history(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(DataPreparationError, match="JSON|malformed|Invalid"):
        load_funding_history(bad)
    truncated = tmp_path / "t.json"
    save_funding_history(simple_history(), truncated)
    text = truncated.read_text()
    truncated.write_text(text[: len(text) // 2])
    with pytest.raises(DataPreparationError):
        load_funding_history(truncated)


# -- Resume provenance ---------------------------------------------------------

def _resume_env(hist, candles=None):
    from noone.rl_engine.config import DataConfig
    from noone.rl_engine.training import make_env

    return make_env(candles or hourly(), data_config=DataConfig(interval="1h"),
                    trading_config=TradingConfig(
                        fee_rate=0.0, slippage_rate=0.0,
                        funding_mode="historical"),
                    funding_history=hist)


def test_changed_collected_at_rejected(tmp_path):
    from noone.rl_engine.config import DQNConfig
    from noone.rl_engine.training import validate_resume

    hist = simple_history()
    saved = {"observation_dim": 26, "action_count": 5,
             "funding": {"funding_mode": "historical",
                         "funding_history": hist.metadata()}}
    ckpt = tmp_path / "model.zip"
    ckpt.write_bytes(b"x")
    (tmp_path / "metadata.json").write_text(json.dumps(saved, default=str))
    recollected = replace(hist, collected_at=hist.collected_at
                          + timedelta(seconds=30))
    with pytest.raises(ValueError, match="collected_at"):
        validate_resume(ckpt, _resume_env(recollected), DQNConfig())
    # Identical collection identity resumes.
    assert validate_resume(ckpt, _resume_env(hist), DQNConfig())[
        "funding_resume_status"] == "funding_metadata_matched"


# -- CLI resume ------------------------------------------------------------------

def _setup_run(tmp_path, hist=None, with_artifact=True, funding_section=None):
    run = tmp_path / "run"
    run.mkdir()
    ckpt = run / "final_model.zip"
    ckpt.write_bytes(b"model")
    if funding_section is None:
        funding_section = {"funding_mode": "historical",
                           "funding_history": (hist or simple_history()).metadata()}
    if with_artifact and hist is not None:
        info = save_funding_history(hist, run / "funding_history.json")
        funding_section.update(info)
        # artifact_path from save is just the filename; keep relative.
    (run / "metadata.json").write_text(
        json.dumps(funding_section and {"funding": funding_section},
                   default=str))
    return run, ckpt


def _run_cli_resume(monkeypatch, tmp_path, ckpt, hist_candles=None):
    import noone.rl_engine.train as train_cli
    import noone.rl_engine.training as training

    captured = {}

    def fake_fetch(**kwargs):
        raise AssertionError("fresh funding collector must not be called")

    def fake_run_training(candles, **kwargs):
        captured.update(kwargs)
        captured["candles"] = candles
        return {"run_dir": str(tmp_path / "newrun"),
                "final_model": "x.zip", "metadata": {}, "model": None}

    monkeypatch.setattr(train_cli, "fetch_candles",
                        lambda **kw: hist_candles or hourly())
    monkeypatch.setattr(train_cli, "fetch_funding_history", fake_fetch)
    monkeypatch.setattr(training, "run_training", fake_run_training)
    code = train_cli.main(["--funding-mode", "historical",
                           "--resume", str(ckpt), "--symbol", "BTCUSDT"])
    assert code == 0
    return captured


def test_cli_resume_loads_artifact_without_recollecting(tmp_path, monkeypatch):
    hist = simple_history()
    _, ckpt = _setup_run(tmp_path, hist=hist)
    captured = _run_cli_resume(monkeypatch, tmp_path, ckpt)
    loaded = captured["funding_history"]
    assert loaded == hist
    assert loaded.collected_at == hist.collected_at
    assert captured["resume_from"] == str(ckpt)


def test_cli_resume_missing_artifact_fails_without_recollecting(
        tmp_path, monkeypatch):
    hist = simple_history()
    run, ckpt = _setup_run(tmp_path, hist=hist)
    (run / "funding_history.json").unlink()
    with pytest.raises(DataPreparationError, match="missing"):
        _run_cli_resume(monkeypatch, tmp_path, ckpt)


def test_cli_resume_corrupted_artifact_fails(tmp_path, monkeypatch):
    hist = simple_history()
    run, ckpt = _setup_run(tmp_path, hist=hist)
    path = run / "funding_history.json"
    doc = json.loads(path.read_text())
    doc["payload"]["symbol"] = "ETHUSDT"
    path.write_text(json.dumps(doc))
    with pytest.raises(DataPreparationError, match="checksum"):
        _run_cli_resume(monkeypatch, tmp_path, ckpt)


def test_cli_resume_legacy_historical_without_artifact_rejected(
        tmp_path, monkeypatch):
    _, ckpt = _setup_run(tmp_path, hist=None, with_artifact=False,
                         funding_section={"funding_mode": "historical"})
    with pytest.raises(DataPreparationError, match="cannot be reconstructed"):
        _run_cli_resume(monkeypatch, tmp_path, ckpt)


def test_cli_constant_resume_unchanged(tmp_path, monkeypatch):
    import noone.rl_engine.train as train_cli
    import noone.rl_engine.training as training

    ckpt = tmp_path / "model.zip"
    ckpt.write_bytes(b"x")
    captured = {}

    def fake_run_training(candles, **kwargs):
        captured.update(kwargs)
        return {"run_dir": str(tmp_path), "final_model": "x.zip",
                "metadata": {}, "model": None}

    monkeypatch.setattr(train_cli, "fetch_candles", lambda **kw: hourly())
    monkeypatch.setattr(training, "run_training", fake_run_training)
    code = train_cli.main(["--resume", str(ckpt)])
    assert code == 0
    assert captured["funding_history"] is None
    assert captured["trading_config"].funding_mode == "constant"


# -- Hardening: path containment, required fields, raw payloads --------------

def _doc_with_payload(hist, mutate):
    from noone.rl_engine.data import _payload_checksum, funding_history_to_payload

    payload = funding_history_to_payload(hist)
    mutate(payload)
    return {"schema_version": 1, "checksum": _payload_checksum(payload),
            "payload": payload}


def _write_run(tmp_path, hist, artifact_path_value, artifact_file=None):
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    ckpt = run / "final_model.zip"
    ckpt.write_bytes(b"model")
    info = save_funding_history(hist, artifact_file or (run / "funding_history.json"))
    funding = {"funding_mode": "historical",
               "funding_history": hist.metadata(), **info,
               "artifact_path": artifact_path_value}
    (run / "metadata.json").write_text(json.dumps({"funding": funding},
                                                  default=str))
    return run, ckpt


def test_resume_artifact_path_normal_relative_succeeds(tmp_path):
    import noone.rl_engine.train as train_cli

    hist = simple_history()
    _, ckpt = _write_run(tmp_path, hist, "funding_history.json")
    loaded = train_cli._load_resume_funding_history(str(ckpt))
    assert loaded == hist


def test_resume_artifact_absolute_path_rejected(tmp_path):
    import noone.rl_engine.train as train_cli

    hist = simple_history()
    run, ckpt = _write_run(
        tmp_path, hist, str((tmp_path / "run" / "funding_history.json").resolve()))
    with pytest.raises(DataPreparationError, match="absolute"):
        train_cli._load_resume_funding_history(str(ckpt))


def test_resume_artifact_traversal_rejected(tmp_path):
    import noone.rl_engine.train as train_cli

    hist = simple_history()
    outside = tmp_path / "outside.json"
    _, ckpt = _write_run(tmp_path, hist, "../outside.json",
                         artifact_file=outside)
    with pytest.raises(DataPreparationError, match="escapes the run directory"):
        train_cli._load_resume_funding_history(str(ckpt))


def test_resume_artifact_nested_traversal_rejected(tmp_path):
    import noone.rl_engine.train as train_cli

    hist = simple_history()
    outside = tmp_path / "outside.json"
    run, ckpt = _write_run(
        tmp_path, hist, "checkpoints/../../outside.json",
        artifact_file=outside)
    (run / "checkpoints").mkdir(exist_ok=True)
    with pytest.raises(DataPreparationError, match="escapes the run directory"):
        train_cli._load_resume_funding_history(str(ckpt))


def test_resume_artifact_symlink_escape_rejected(tmp_path):
    import noone.rl_engine.train as train_cli

    hist = simple_history()
    run, ckpt = _write_run(tmp_path, hist, "linked.json")
    outside = tmp_path / "outside.json"
    save_funding_history(hist, outside)
    link = run / "linked.json"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation not supported on this platform")
    with pytest.raises(DataPreparationError, match="escapes the run directory"):
        train_cli._load_resume_funding_history(str(ckpt))


def test_missing_before_and_after_context_rejected(tmp_path):
    hist = simple_history()
    for field in ("before_context", "after_context"):
        doc = _doc_with_payload(hist, lambda p, f=field: p.pop(f))
        path = tmp_path / f"{field}.json"
        path.write_text(json.dumps(doc))
        with pytest.raises(DataPreparationError, match=field):
            load_funding_history(path)


def test_legitimate_none_context_and_missing_nested_field(tmp_path):
    hist = simple_history()
    doc = _doc_with_payload(hist, lambda p: p.update(before_context=None))
    path = tmp_path / "none_ctx.json"
    path.write_text(json.dumps(doc))
    loaded = load_funding_history(path)
    assert loaded.before_context is None
    assert loaded.after_context is not None

    def drop_rate(p):
        del p["settlements"][0]["funding_rate"]

    path2 = tmp_path / "missing_rate.json"
    path2.write_text(json.dumps(_doc_with_payload(hist, drop_rate)))
    with pytest.raises(DataPreparationError, match="funding_rate"):
        load_funding_history(path2)

    def drop_raw(p):
        del p["settlements"][0]["raw"]

    path3 = tmp_path / "missing_raw.json"
    path3.write_text(json.dumps(_doc_with_payload(hist, drop_raw)))
    with pytest.raises(DataPreparationError, match="raw"):
        load_funding_history(path3)


def test_raw_json_native_round_trip_and_unsupported_rejected(tmp_path):
    hist = simple_history()
    raw = {"a": [1, "x", True, None, {"b": 2.5}], "c": {"d": []}}
    native = replace(hist, settlements=(
        replace(hist.settlements[0], raw=raw),))
    path = tmp_path / "raw.json"
    save_funding_history(native, path)
    assert load_funding_history(path).settlements[0].raw == raw

    for bad_raw, kind in (({"levels": (1, 2)}, "tuple"),
                          ({"x": Decimal("1.5")}, "Decimal"),
                          ({"t": T0}, "datetime")):
        bad = replace(hist, settlements=(
            replace(hist.settlements[0], raw=bad_raw),))
        with pytest.raises(DataPreparationError, match="JSON-native"):
            save_funding_history(bad, tmp_path / f"bad_{kind}.json")
        assert not (tmp_path / f"bad_{kind}.json").exists()


def test_new_historical_training_persists_artifact(tmp_path):
    pytest.importorskip("stable_baselines3")
    from noone.rl_engine.config import DataConfig, DQNConfig, PipelineConfig
    from noone.rl_engine.training import train_dqn

    cs = hourly(80)
    hist = simple_history(cs)
    cfg = PipelineConfig(
        dqn=DQNConfig(net_arch=(32, 32), buffer_size=500, batch_size=16,
                      learning_starts=10, train_freq=1, total_timesteps=32,
                      checkpoint_freq=32, seed=42),
        data=DataConfig(symbol="BTCUSDT", interval="1h", warmup_candles=5,
                        min_train_candles=20, min_eval_candles=10),
        output_dir=str(tmp_path / "runs"))
    result = train_dqn(cs, config=cfg, funding_history=hist,
                       trading_config=TradingConfig(
                           fee_rate=0.0, slippage_rate=0.0,
                           funding_mode="historical"))
    run = Path(result["run_dir"])
    artifact = run / "funding_history.json"
    assert artifact.exists()
    assert load_funding_history(artifact) == hist
    meta = json.loads((run / "metadata.json").read_text())
    assert meta["funding"]["artifact_path"] == "funding_history.json"
    assert meta["funding"]["artifact_schema_version"] == 1
    assert meta["funding"]["collection_identity"] == \
        hist.collected_at.isoformat()
    assert meta["funding"]["artifact_checksum"]
