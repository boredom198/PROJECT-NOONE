"""Stage 1: historical funding mode contract and strict gate.

Component 8 must accept only a full Component 4 ``FundingHistory``
in explicit historical mode, validate it strictly at construction,
and surface auditable funding metadata. Historical settlement
application itself is Stage 2; constant mode is unchanged.
"""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from noone.market_data.models import (
    FundingBlindSpot,
    FundingConflict,
    FundingContext,
    FundingHistory,
    FundingIntervalAnomaly,
    FundingMalformedRecord,
    FundingSettlement,
    Kline,
    KlineGap,
    KlineHistory,
)
from noone.trading_env import PerpetualFuturesEnv, TradingConfig

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def candle(i, price=100.0):
    return Kline(symbol="BTCUSDT", interval="1h",
                 open_time=T0 + timedelta(hours=i),
                 close_time=T0 + timedelta(hours=i, minutes=59),
                 open=price, high=price, low=price, close=price,
                 volume=10.0, quote_volume=1000.0, trades=5, is_closed=True)


def candles(n=10):
    return [candle(i) for i in range(n)]


def settlement(hours, mark="100"):
    return FundingSettlement(
        symbol="BTCUSDT", funding_time=T0 + timedelta(hours=hours),
        funding_rate=Decimal("0.0001"), mark_price=Decimal(mark))


def context(direction, hours_list):
    return FundingContext(
        direction=direction,
        settlements=tuple(settlement(h) for h in hours_list),
        malformed_records=(), requested_count=3, status="sufficient",
        pages_fetched=1, pagination_complete=True,
        pagination_termination="range_exhausted")


def valid_history(cs=None):
    cs = cs if cs is not None else candles()
    start = cs[0].open_time
    end = cs[-1].close_time
    return FundingHistory(
        symbol="BTCUSDT",
        settlements=tuple(settlement(h) for h in (0, 8)),
        requested_start_time=start, requested_end_time=end,
        coverage_start_time=start, coverage_end_time=end,
        collected_at=end + timedelta(hours=1), pages_fetched=1,
        duplicates_removed=0, conflicting_records=(), malformed_records=(),
        suspected_anomalies=(), pagination_complete=True,
        pagination_termination="range_exhausted",
        total_limit=100, page_limit=1000,
        before_context=context("before", (-24, -16, -8)),
        after_context=context("after", (16, 24, 32)),
        scoped_blind_spots=())


def hist_cfg(**kw):
    base = dict(fee_rate=0.0, slippage_rate=0.0, funding_rate=0.0,
                funding_mode="historical")
    base.update(kw)
    return TradingConfig(**base)


# -- Mode and input contract -------------------------------------------

def test_config_defaults_to_constant_and_rejects_unknown_mode():
    assert TradingConfig().funding_mode == "constant"
    with pytest.raises(ValueError, match="funding_mode"):
        TradingConfig(funding_mode="live")


def test_historical_mode_requires_full_funding_history():
    with pytest.raises(ValueError, match="FundingHistory"):
        PerpetualFuturesEnv(candles(), hist_cfg())
    bare = [settlement(0), settlement(8)]
    with pytest.raises((TypeError, ValueError), match="FundingHistory"):
        PerpetualFuturesEnv(candles(), hist_cfg(), funding_history=bare)


def test_funding_history_in_constant_mode_is_rejected():
    with pytest.raises(ValueError, match="funding_mode"):
        PerpetualFuturesEnv(candles(), TradingConfig(),
                            funding_history=valid_history())


def test_valid_historical_history_accepted_and_metadata_surfaced():
    hist = valid_history()
    env = PerpetualFuturesEnv(candles(), hist_cfg(), funding_history=hist)
    assert env.funding_history is hist
    meta = env.funding_metadata()
    assert meta["funding_mode"] == "historical"
    assert meta["completeness_status_for_period"] == "complete"
    assert meta["funding_history"]["settlement_count"] == 2
    assert meta["funding_history"]["before_context"]["received_count"] == 3
    assert meta["funding_history"]["after_context"]["received_count"] == 3
    assert meta["unacknowledged_anomaly_count"] == 0


def test_from_history_accepts_both_histories():
    cs = candles()
    khist = KlineHistory(symbol="BTCUSDT", interval="1h",
                         candles=tuple(cs), gaps=(), duplicates_removed=0,
                         pages_fetched=1, start_time=None, end_time=None)
    env = PerpetualFuturesEnv.from_history(
        khist, hist_cfg(), funding_history=valid_history(cs))
    assert env.funding_metadata()["funding_mode"] == "historical"


def test_historical_env_constructs_and_steps_after_stage2():
    # Stage 1 blocked historical stepping; Stage 2 implements the
    # accounting, so a valid historical env now steps normally.
    env = PerpetualFuturesEnv(candles(), hist_cfg(),
                              funding_history=valid_history())
    env.reset()
    result = env.step(0)
    assert env._index == 1
    assert result.info["funding_mode"] == "historical"
    assert result.info["historical_accounting_implemented"] is True
    assert env.funding_metadata()[
        "historical_accounting_implemented"] is True


def test_constant_mode_steps_and_funding_unchanged_with_new_info_field():
    cfg = TradingConfig(fee_rate=0.0, slippage_rate=0.0, funding_rate=0.01,
                        funding_times_utc=("00:00", "08:00", "16:00"))
    env = PerpetualFuturesEnv(candles(10), cfg)
    assert env.funding_metadata()[
        "historical_accounting_implemented"] is False
    env.reset()
    result = env.step(1)  # OPEN_LONG at 01:00, qty 50 @100
    assert result.info["historical_accounting_implemented"] is False
    for _ in range(7):
        result = env.step(0)  # HOLD through the 08:00 settlement candle
    assert env.total_funding == pytest.approx(-50.0)
    assert result.info["funding_payment"] == pytest.approx(-50.0)


def test_constant_mode_without_history_still_works():
    env = PerpetualFuturesEnv(candles(), TradingConfig(funding_rate=0.0))
    assert env.funding_history is None
    assert env.funding_metadata()["funding_mode"] == "constant"
    env.reset()
    assert env.step(0).observation["total_funding"] == 0.0


# -- Strict gate ---------------------------------------------------------

def test_gate_rejects_symbol_mismatch():
    hist = replace(valid_history(), symbol="ETHUSDT")
    with pytest.raises(ValueError, match="symbol"):
        PerpetualFuturesEnv(candles(), hist_cfg(), funding_history=hist)


def test_gate_rejects_requested_range_not_covering_period():
    hist = replace(valid_history(),
                   requested_start_time=T0 + timedelta(hours=1))
    with pytest.raises(ValueError, match="requested"):
        PerpetualFuturesEnv(candles(), hist_cfg(), funding_history=hist)
    hist = replace(valid_history(),
                   requested_end_time=T0 + timedelta(hours=1))
    with pytest.raises(ValueError, match="requested"):
        PerpetualFuturesEnv(candles(), hist_cfg(), funding_history=hist)


def test_gate_rejects_incomplete_pagination():
    hist = replace(valid_history(), pagination_complete=False,
                   pagination_termination="repeated_page")
    with pytest.raises(ValueError, match="pagination"):
        PerpetualFuturesEnv(candles(), hist_cfg(), funding_history=hist)


def test_complete_without_context_does_not_prove_boundary():
    hist = replace(valid_history(), before_context=None, after_context=None)
    assert hist.completeness_status == "complete"
    with pytest.raises(ValueError, match="context"):
        PerpetualFuturesEnv(candles(), hist_cfg(), funding_history=hist)


def test_gate_rejects_insufficient_context_and_wrong_context_limit():
    bad = replace(context("before", (-24, -16)), status="insufficient",
                  requested_count=3)
    # received_count is len(settlements) == 2 here.
    hist = replace(valid_history(), before_context=bad)
    with pytest.raises(ValueError, match="context"):
        PerpetualFuturesEnv(candles(), hist_cfg(), funding_history=hist)
    wrong_limit = replace(context("after", (16, 24, 32)), requested_count=2)
    hist = replace(valid_history(), after_context=wrong_limit)
    with pytest.raises(ValueError, match="context_limit=3|context"):
        PerpetualFuturesEnv(candles(), hist_cfg(), funding_history=hist)


def test_gate_rejects_before_and_after_blind_spots():
    for boundary in ("before", "after"):
        spot = FundingBlindSpot(
            boundary=boundary, start_time=None, end_time=None,
            reason="test blind spot", context_requested=3,
            context_received=3)
        hist = replace(valid_history(), scoped_blind_spots=(spot,))
        with pytest.raises(ValueError, match="blind spot"):
            PerpetualFuturesEnv(candles(), hist_cfg(), funding_history=hist)


def test_gate_rejects_conflict_and_malformed_inside_period():
    conflict = FundingConflict(
        symbol="BTCUSDT", funding_time=T0 + timedelta(hours=8),
        rate_type=None, kept_rate=Decimal("0.0001"),
        kept_mark_price=Decimal("100"), conflicting_rate=Decimal("0.0002"),
        conflicting_mark_price=Decimal("100"))
    hist = replace(valid_history(), conflicting_records=(conflict,))
    with pytest.raises(ValueError, match="complete"):
        PerpetualFuturesEnv(candles(), hist_cfg(), funding_history=hist)
    malformed = FundingMalformedRecord(
        symbol="BTCUSDT", funding_time=T0 + timedelta(hours=8),
        reason="missing_or_invalid_mark_price", raw={})
    hist = replace(valid_history(), malformed_records=(malformed,))
    with pytest.raises(ValueError, match="complete"):
        PerpetualFuturesEnv(candles(), hist_cfg(), funding_history=hist)


def test_gate_rejects_invalid_settlement_mark_and_acknowledged_anomaly():
    bad = FundingSettlement(symbol="BTCUSDT", funding_time=T0,
                            funding_rate=Decimal("0.0001"),
                            mark_price=Decimal("0"))
    hist = replace(valid_history(), settlements=(bad, settlement(8)))
    with pytest.raises(ValueError, match="mark"):
        PerpetualFuturesEnv(candles(), hist_cfg(), funding_history=hist)
    anomaly = FundingIntervalAnomaly(
        after_time=T0, before_time=T0 + timedelta(hours=16),
        interval_seconds=16 * 3600, reference_interval_seconds=8 * 3600,
        estimated_missing_count=1, acknowledged=True)
    hist = replace(valid_history(), suspected_anomalies=(anomaly,))
    with pytest.raises(ValueError, match="complete"):
        PerpetualFuturesEnv(candles(), hist_cfg(), funding_history=hist)
