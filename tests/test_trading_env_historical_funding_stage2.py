"""Stage 2: historical funding settlement accounting (Component 8).

Settlements come only from a full Component 4 ``FundingHistory``:
actual rate and settlement mark per event, split around candle open,
applied exactly once by monotonic consumption, ledgered per event,
and reconciled against ``total_funding`` and balance adjustments.
Constant mode is unchanged.
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from noone.market_data.models import (
    FundingContext,
    FundingHistory,
    FundingSettlement,
    Kline,
)
from noone.trading_env import Action, PerpetualFuturesEnv, TradingConfig

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_candle(open_time, close_time, price=100.0, high=None, low=None):
    return Kline(symbol="BTCUSDT", interval="1h", open_time=open_time,
                 close_time=close_time, open=price,
                 high=float(high if high is not None else price),
                 low=float(low if low is not None else price), close=price,
                 volume=10.0, quote_volume=1000.0, trades=5, is_closed=True)


def hourly(n=8, start=T0):
    return [make_candle(start + timedelta(hours=i),
                        start + timedelta(hours=i, minutes=59))
            for i in range(n)]


def settlement(dt, rate="0.001", mark="100", rate_type=None):
    return FundingSettlement(
        symbol="BTCUSDT", funding_time=dt,
        funding_rate=Decimal(rate) if rate is not None else None,
        mark_price=Decimal(mark) if mark is not None else None,
        rate_type=rate_type)


def context(direction, times):
    return FundingContext(
        direction=direction,
        settlements=tuple(settlement(t) for t in times),
        malformed_records=(), requested_count=3, status="sufficient",
        pages_fetched=1, pagination_complete=True,
        pagination_termination="range_exhausted")


def history_for(cs, settlements):
    start = cs[0].open_time
    last = cs[-1]
    end = last.close_time or last.open_time
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


def hist_env(cs, settlements, **cfg_kw):
    base = dict(fee_rate=0.0, slippage_rate=0.0, funding_mode="historical")
    base.update(cfg_kw)
    env = PerpetualFuturesEnv(cs, TradingConfig(**base),
                              funding_history=history_for(cs, settlements))
    env.reset()
    return env


def at(hours=0, minutes=0):
    return T0 + timedelta(hours=hours, minutes=minutes)


def single(rate="0.001", mark="100", action=Action.OPEN_LONG, when=None):
    """Qty 50 @100; settlement inside candle 2; magnitude 5 at defaults."""
    env = hist_env(hourly(6), [settlement(when or at(2, 30), rate, mark)])
    env.step(action)
    result = env.step(Action.HOLD)
    return env, result


# -- Funding calculation -------------------------------------------------

def test_long_positive_rate_pays():
    env, result = single()
    assert env.total_funding == pytest.approx(-5.0)
    assert env.balance == pytest.approx(9_995.0)
    assert result.info["funding_payment"] == pytest.approx(-5.0)
    (entry,) = env.funding_ledger
    assert entry.position_side == 1
    assert entry.funding_payment == Decimal("-5")


def test_short_positive_rate_receives():
    env, _ = single(action=Action.OPEN_SHORT)
    assert env.total_funding == pytest.approx(5.0)
    assert env.funding_ledger[0].position_side == -1


def test_long_negative_rate_receives():
    env, _ = single(rate="-0.001")
    assert env.total_funding == pytest.approx(5.0)


def test_short_negative_rate_pays():
    env, _ = single(rate="-0.001", action=Action.OPEN_SHORT)
    assert env.total_funding == pytest.approx(-5.0)


def test_zero_rate_creates_zero_ledger_entry():
    env, result = single(rate="0")
    assert env.total_funding == 0.0 and env.balance == 10_000.0
    (entry,) = env.funding_ledger
    assert entry.funding_payment == Decimal("0")
    assert entry.settlement_notional > 0
    assert result.info["funding_payment"] == 0.0


def test_decimal_calculation_uses_actual_mark_and_rate():
    env, _ = single(mark="200", rate="0.001")
    (entry,) = env.funding_ledger
    assert entry.settlement_mark_price == Decimal("200")
    assert entry.funding_rate == Decimal("0.001")
    assert entry.position_quantity == Decimal("50.0")
    assert entry.settlement_notional == Decimal("10000.0")
    assert entry.funding_payment == Decimal("-10")
    assert entry.balance_adjustment == entry.funding_payment
    # Candle close is 100; candle-close pricing would have given -5.
    assert env.total_funding == pytest.approx(-10.0)


# -- Settlement timing -----------------------------------------------------

def test_settlement_at_open_applies_before_action():
    env = hist_env(hourly(6), [settlement(at(2), "0.001")])
    env.step(Action.OPEN_LONG)          # candle 1
    result = env.step(Action.HOLD)      # candle 2, open == settlement
    assert len(env.funding_ledger) == 1
    assert env.funding_ledger[0].funding_time == at(2)
    assert result.info["funding_payment"] == pytest.approx(-5.0)


def test_settlement_at_close_applies_after_action():
    env = hist_env(hourly(6), [settlement(at(2, 59), "0.001")])
    env.step(Action.OPEN_LONG)
    env.step(Action.HOLD)               # candle 2 close == settlement
    assert len(env.funding_ledger) == 1
    assert env.total_funding == pytest.approx(-5.0)


def test_position_opened_at_settlement_not_charged():
    env = hist_env(hourly(6), [settlement(at(1), "0.001")])
    env.step(Action.OPEN_LONG)          # executes exactly at settlement
    env.step(Action.HOLD)
    assert env.funding_ledger == []
    assert env.total_funding == 0.0


def test_position_closed_at_settlement_remains_eligible():
    env = hist_env(hourly(6), [settlement(at(2), "0.001")])
    env.step(Action.OPEN_LONG)
    env.step(Action.CLOSE_LONG)         # closes at settlement open time
    assert len(env.funding_ledger) == 1
    assert env.total_funding == pytest.approx(-5.0)
    assert env.balance == pytest.approx(9_995.0)


def test_multiple_settlements_in_one_candle_independent():
    env = hist_env(hourly(6), [settlement(at(2, 15), "0.001"),
                               settlement(at(2, 45), "0.002")])
    env.step(Action.OPEN_LONG)
    result = env.step(Action.HOLD)
    assert len(env.funding_ledger) == 2
    assert env.total_funding == pytest.approx(-15.0)
    assert result.info["funding_payment"] == pytest.approx(-15.0)
    assert len(result.info["funding_ledger_entries"]) == 2


def test_settlement_between_candles_applies_next_pre_open():
    cs = [make_candle(at(0), at(0, 59)), make_candle(at(1), at(1, 59)),
          make_candle(at(5), at(5, 59)), make_candle(at(6), at(6, 59)),
          make_candle(at(7), at(7, 59))]
    env = hist_env(cs, [settlement(at(3), "0.001")])
    env.step(Action.OPEN_LONG)          # candle at 01:00
    env.step(Action.HOLD)               # candle at 05:00 (gap before it)
    (entry,) = env.funding_ledger
    assert entry.funding_time == at(3)
    assert env.total_funding == pytest.approx(-5.0)


# -- Duplicate prevention and identity -------------------------------------

def test_shared_boundary_settlement_applied_exactly_once():
    cs = [make_candle(at(i), at(i + 1)) for i in range(5)]
    env = hist_env(cs, [settlement(at(2), "0.001")])
    env.step(Action.OPEN_LONG)          # candle 01:00-02:00
    env.step(Action.HOLD)               # candle 02:00-03:00
    env.step(Action.HOLD)
    assert len(env.funding_ledger) == 1
    assert env.total_funding == pytest.approx(-5.0)


def test_two_rate_types_same_timestamp_both_processed():
    env = hist_env(hourly(6), [
        settlement(at(2, 30), "0.001", rate_type="Regular"),
        settlement(at(2, 30), "0.002", rate_type="Special")])
    env.step(Action.OPEN_LONG)
    env.step(Action.HOLD)
    assert len(env.funding_ledger) == 2
    assert {e.rate_type for e in env.funding_ledger} == {"Regular", "Special"}
    assert env.total_funding == pytest.approx(-15.0)


def test_settlement_consumed_while_flat_not_charged_later():
    env = hist_env(hourly(6), [settlement(at(1, 30), "0.001")])
    env.step(Action.HOLD)               # flat through the settlement
    env.step(Action.OPEN_LONG)
    env.step(Action.HOLD)
    assert env.funding_ledger == []
    assert env.total_funding == 0.0


def test_same_timestamp_records_not_collapsed_by_timestamp():
    env = hist_env(hourly(6), [
        settlement(at(2, 30), "0.001", rate_type="Regular"),
        settlement(at(2, 30), "0.001", rate_type="Special")])
    env.step(Action.OPEN_LONG)
    env.step(Action.HOLD)
    times = [e.funding_time for e in env.funding_ledger]
    assert times == [at(2, 30), at(2, 30)]
    assert env.total_funding == pytest.approx(-10.0)


# -- Risk and accounting -----------------------------------------------------

def test_liquidated_position_not_charged_post_open_settlement():
    cs = hourly(6)
    cs[2] = make_candle(at(2), at(2, 59), price=80.0, high=100.0, low=70.0)
    env = hist_env(cs, [settlement(at(2, 30), "0.001")])
    env.step(Action.OPEN_LONG)
    result = env.step(Action.HOLD)      # liquidation inside candle 2
    assert result.terminated and result.info["event"] == "liquidation"
    assert env.funding_ledger == []
    assert env.total_funding == 0.0


def test_ledger_reconciles_with_total_and_balance_adjustments():
    env = hist_env(hourly(8), [settlement(at(2, 30), "0.001"),
                               settlement(at(4, 30), "-0.002"),
                               settlement(at(6, 30), "0.003")])
    env.step(Action.OPEN_LONG)
    for _ in range(6):
        env.step(Action.HOLD)
    ledger_total = sum((e.funding_payment for e in env.funding_ledger),
                       Decimal("0"))
    assert len(env.funding_ledger) == 3
    assert float(ledger_total) == pytest.approx(env.total_funding)
    # Fees are zero and price is flat, so balance moved only by funding.
    assert env.balance - 10_000.0 == pytest.approx(env.total_funding)
    assert env.reconcile_historical_funding() is True
    meta = env.funding_metadata()
    assert meta["historical_accounting_implemented"] is True


def test_multiple_settlements_preserve_cumulative_totals():
    env = hist_env(hourly(8), [settlement(at(2, 30), "0.001"),
                               settlement(at(4, 30), "0.001")])
    env.step(Action.OPEN_LONG)
    env.step(Action.HOLD)
    assert env.total_funding == pytest.approx(-5.0)
    env.step(Action.HOLD)
    env.step(Action.HOLD)
    assert env.total_funding == pytest.approx(-10.0)
    assert [e.funding_payment for e in env.funding_ledger] == [
        Decimal("-5"), Decimal("-5")]


def test_reset_clears_ledger_and_reapplies_from_start():
    env, _ = single()
    assert len(env.funding_ledger) == 1
    env.reset()
    assert env.funding_ledger == [] and env.total_funding == 0.0
    env.step(Action.OPEN_LONG)
    env.step(Action.HOLD)
    assert len(env.funding_ledger) == 1
    assert env.total_funding == pytest.approx(-5.0)


def test_settlements_outside_episode_not_applied():
    env = hist_env(hourly(6), [settlement(at(-1), "0.001"),
                               settlement(at(2, 30), "0.001"),
                               settlement(at(100), "0.001")])
    env.step(Action.OPEN_LONG)
    for _ in range(4):
        env.step(Action.HOLD)
    assert [e.funding_time for e in env.funding_ledger] == [at(2, 30)]


# -- Safety and compatibility -------------------------------------------------

def test_historical_never_falls_back_to_constant_funding():
    env = hist_env(hourly(6), [settlement(at(2, 30), "0.0001")],
                   funding_rate=0.01)
    env.step(Action.OPEN_LONG)
    result = env.step(Action.HOLD)
    # Constant logic would charge 50 * 100 * 0.01 = 50.
    assert env.total_funding == pytest.approx(-0.5)
    assert result.info["funding_mode"] == "historical"
    assert result.info["historical_accounting_implemented"] is True
    assert result.info["total_funding"] == pytest.approx(-0.5)


def test_historical_rejects_invalid_settlement_rate():
    cs = hourly(4)
    bad = FundingSettlement(symbol="BTCUSDT", funding_time=at(1),
                            funding_rate=None, mark_price=Decimal("100"))
    with pytest.raises(ValueError, match="rate"):
        PerpetualFuturesEnv(
            cs, TradingConfig(funding_mode="historical"),
            funding_history=history_for(cs, [bad]))


def test_historical_rejects_invalid_settlement_mark():
    cs = hourly(4)
    with pytest.raises(ValueError, match="mark"):
        hist_env(cs, [settlement(at(1), "0.001", mark="-5")])


def test_historical_rejects_missing_candle_close_time():
    cs = hourly(4)
    cs[2] = make_candle(at(2), None)
    with pytest.raises(ValueError, match="close"):
        hist_env(cs, [settlement(at(1), "0.001")])


# -- Final hardening corrections -------------------------------------------

def test_duplicate_settlement_identity_rejected_distinct_rate_types_ok():
    cs = hourly(6)
    dup = [settlement(at(2, 30), "0.001", rate_type="Regular"),
           settlement(at(2, 30), "0.001", rate_type="Regular")]
    with pytest.raises(ValueError, match="Duplicate funding settlement identity"):
        hist_env(cs, dup)
    distinct = [settlement(at(2, 30), "0.001", rate_type="Regular"),
                settlement(at(2, 30), "0.001", rate_type="Special")]
    env = hist_env(cs, distinct)
    env.step(Action.OPEN_LONG)
    env.step(Action.HOLD)
    assert len(env.funding_ledger) == 2
    assert {e.rate_type for e in env.funding_ledger} == {"Regular", "Special"}


@pytest.mark.parametrize("mark", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_settlement_mark_rejected_with_value_error(mark):
    cs = hourly(4)
    with pytest.raises(ValueError, match="mark"):
        hist_env(cs, [settlement(at(1), "0.001", mark=mark)])


@pytest.mark.parametrize("rate", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_funding_rate_rejected_with_value_error(rate):
    cs = hourly(4)
    with pytest.raises(ValueError, match="rate"):
        hist_env(cs, [settlement(at(1), rate)])


def test_failed_settlement_does_not_advance_cursor():
    from dataclasses import replace as dc_replace

    env = hist_env(hourly(6), [settlement(at(2, 30), "0.001")])
    env.step(Action.OPEN_LONG)
    # Bypass the construction gate by corrupting the prepared
    # settlement (NaN mark); application must fail and stay unconsumed.
    bad = dc_replace(env._historical_settlements[0],
                     mark_price=Decimal("NaN"))
    env._historical_settlements = (bad,) + env._historical_settlements[1:]
    env._historical_funding_cursor = 0
    with pytest.raises(ValueError, match="mark"):
        env.step(Action.HOLD)
    assert env._historical_funding_cursor == 0
    assert env.funding_ledger == []
    assert env.total_funding == 0.0


def test_final_truncated_step_includes_historical_fields():
    env, _ = single()
    assert len(env.funding_ledger) == 1
    env._index = len(env.candles) - 1
    env._truncated = False
    result = env.step(Action.HOLD)
    assert result.truncated and result.info["reason"] == "end_of_data"
    assert result.info["funding_mode"] == "historical"
    assert result.info["historical_accounting_implemented"] is True
    assert result.info["funding_payment"] == 0.0
    assert result.info["funding_ledger_entries"] == []
    assert result.info["funding_ledger_count"] == 1
    assert result.info["total_funding"] == pytest.approx(-5.0)


def test_constant_mode_retains_existing_behaviour():
    cfg = TradingConfig(fee_rate=0.0, slippage_rate=0.0, funding_rate=0.01,
                        funding_times_utc=("00:00", "08:00", "16:00"))
    env = PerpetualFuturesEnv(hourly(10), cfg)
    env.reset()
    env.step(Action.OPEN_LONG)
    for _ in range(7):
        env.step(Action.HOLD)
    assert env.total_funding == pytest.approx(-50.0)
    assert env.funding_ledger == []
