"""Tests for Risk Gate and Exposure Controller (Component 3).

Verifies the 8-step sizing workflow, margin buffer engine, daily loss limit (T11),
uninitialized/invalid baseline rejection (T14-A, T14-B), external transfer accounting (T15),
account-wide exposure detection (T16), unexplained open orders (T17), and fresh pre-dispatch checks (T18).
Conforms strictly to PROJECT NOONE Technical Design Specification V2.2.4 Sections 9 and 10.
"""

from decimal import Decimal
import pytest

from noone.rl_engine.inference import AccountSnapshot
from noone.exchange.config import Component3Config
from noone.exchange.errors import (
    DailyLossLimitError,
    DailyPnLUninitializedError,
    FilterValidationError,
    RiskGateError,
    SafetyFreezeError,
    StaleMarketDataError,
)
from noone.exchange.journal import ExecutionJournal
from noone.exchange.precision import ParsedSymbolFilters
from noone.exchange.risk_gate import RiskGate


@pytest.fixture
def risk_env(tmp_path):
    config = Component3Config(
        testnet=True,
        symbol="BTCUSDT",
        leverage=10,
        margin_allocation_fraction=0.10,
        max_daily_loss_pct=0.03,
        database_path=tmp_path / "risk_test.db",
    )
    journal = ExecutionJournal(config.database_path)
    gate = RiskGate(config, journal)
    return config, journal, gate


@pytest.fixture
def default_filters():
    return ParsedSymbolFilters(
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        max_qty=Decimal("100.0"),
        min_notional=Decimal("10.0"),
    )


def test_8_step_sizing_workflow_success(risk_env, default_filters):
    """Verify 8-step sizing computes correct quantized quantity and validates margin buffer."""
    _, _, gate = risk_env
    snapshot = AccountSnapshot(
        equity=10000.0,
        available_balance=9500.0,
        position_side=0,
        position_quantity=0.0,
        entry_price=0.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        margin_used=0.0,
        progress=0.0,
    )
    # Intended margin: 10000 * 0.10 = 1000 USDT
    # Notional: 1000 * 10 = 10000 USDT
    # Mark price: 50000 -> raw qty = 0.2 BTC
    # Quantized: 0.200 BTC
    # Fee (0.05%): 5.0 USDT, Buffer (0.5%): 50.0 USDT -> Required margin = 1000 + 5 + 50 = 1055.0 <= 9500.0
    qty = gate.compute_entry_sizing(snapshot, Decimal("50000.0"), default_filters)
    assert qty == Decimal("0.200")


def test_sizing_insufficient_available_balance(risk_env, default_filters):
    """Verify RiskGateError is raised when available balance is insufficient for margin + buffers."""
    _, _, gate = risk_env
    snapshot = AccountSnapshot(
        equity=10000.0,
        available_balance=500.0,  # Required is 1055.0, so 500.0 is insufficient!
        position_side=0,
        position_quantity=0.0,
        entry_price=0.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        margin_used=0.0,
        progress=0.0,
    )
    with pytest.raises(RiskGateError) as exc:
        gate.compute_entry_sizing(snapshot, Decimal("50000.0"), default_filters)
    assert "exceeds available balance" in str(exc.value)


def test_t11_asymmetric_daily_loss_limit(risk_env):
    """Test T11: Asymmetric daily loss limit breaches must raise DailyLossLimitError and trigger freeze."""
    _, journal, gate = risk_env
    baseline = Decimal("10000.0")
    # Max allowed loss is 3% = 300.0 USDT
    # Current equity = 9650.0, transfers = 0 -> Daily PnL = -350.0 <= -300.0 -> BREACH!
    with pytest.raises(DailyLossLimitError) as exc:
        gate.check_daily_pnl_limit(
            current_equity=Decimal("9650.0"),
            baseline_equity=baseline,
            net_external_transfers=Decimal("0.0"),
            verified_status="VERIFIED_UTC_BASELINE",
        )
    assert "Daily loss limit breached" in str(exc.value)
    assert journal.has_active_freeze("DAILY_LOSS_LIMIT") is True


def test_t14_a_and_t14_b_unverified_daily_baseline_rejection(risk_env):
    """Test T14-A & T14-B: Missing or OPERATOR_ACKNOWLEDGED_STARTUP_BASELINE must block entries."""
    _, journal, gate = risk_env

    # 1. T14-B: OPERATOR_ACKNOWLEDGED_STARTUP_BASELINE cannot authorize entries
    with pytest.raises(DailyPnLUninitializedError) as exc:
        gate.check_daily_pnl_limit(
            current_equity=Decimal("10000.0"),
            baseline_equity=Decimal("10000.0"),
            net_external_transfers=Decimal("0.0"),
            verified_status="OPERATOR_ACKNOWLEDGED_STARTUP_BASELINE",
        )
    assert "daily baseline is uninitialized or unverified" in str(exc.value)
    assert journal.has_active_freeze("DAILY_PNL_UNINITIALIZED") is True

    # 2. T14-A: INVALID status
    with pytest.raises(DailyPnLUninitializedError):
        gate.check_daily_pnl_limit(
            current_equity=Decimal("10000.0"),
            baseline_equity=Decimal("10000.0"),
            net_external_transfers=Decimal("0.0"),
            verified_status="INVALID",
        )


def test_t15_external_transfer_accounting(risk_env):
    """Test T15: Daily PnL = Equity - Net Transfers - Baseline correctly accounts for deposits."""
    _, journal, gate = risk_env
    baseline = Decimal("10000.0")
    # Equity appears to increase to 10500.0, BUT external deposit of 1000.0 occurred:
    # Daily PnL = 10500.0 - 1000.0 - 10000.0 = -500.0 (Loss of 500 USDT, breaching 300 limit!)
    with pytest.raises(DailyLossLimitError):
        gate.check_daily_pnl_limit(
            current_equity=Decimal("10500.0"),
            baseline_equity=baseline,
            net_external_transfers=Decimal("1000.0"),
            verified_status="VERIFIED_UTC_BASELINE",
        )
    assert journal.has_active_freeze("DAILY_LOSS_LIMIT") is True


def test_t16_account_wide_exposure_detection(risk_env, default_filters):
    """Test T16: Detection of positions in non-configured symbols triggers freeze and blocks pre-dispatch."""
    _, journal, gate = risk_env
    snapshot = AccountSnapshot(
        equity=10000.0, available_balance=9000.0, position_side=0, position_quantity=0.0
    )

    all_positions = [
        {"symbol": "BTCUSDT", "positionAmt": "0.0"},
        {"symbol": "ETHUSDT", "positionAmt": "1.5"},  # Non-configured symbol position!
    ]

    with pytest.raises(RiskGateError) as exc:
        gate.validate_pre_dispatch(
            account_snapshot=snapshot,
            mark_price=Decimal("50000.0"),
            filters=default_filters,
            target_pos_amt=Decimal("0.0"),
            all_positions=all_positions,
            open_orders=[],
            is_entry=True,
        )
    assert "external position detected on ETHUSDT" in str(exc.value)
    assert journal.has_active_freeze("ACCOUNT_WIDE_EXPOSURE_DETECTED") is True


def test_t17_unexplained_open_orders_detection(risk_env, default_filters):
    """Test T17: Resting open orders trigger ACCOUNT_WIDE_EXPOSURE_DETECTED freeze."""
    _, journal, gate = risk_env
    snapshot = AccountSnapshot(
        equity=10000.0, available_balance=9000.0, position_side=0, position_quantity=0.0
    )

    open_orders = [{"symbol": "BTCUSDT", "orderId": 12345}]

    with pytest.raises(RiskGateError) as exc:
        gate.validate_pre_dispatch(
            account_snapshot=snapshot,
            mark_price=Decimal("50000.0"),
            filters=default_filters,
            target_pos_amt=Decimal("0.0"),
            all_positions=[{"symbol": "BTCUSDT", "positionAmt": "0.0"}],
            open_orders=open_orders,
            is_entry=True,
        )
    assert "open orders resting on exchange" in str(exc.value)
    assert journal.has_active_freeze("ACCOUNT_WIDE_EXPOSURE_DETECTED") is True


def test_t18_active_safety_freeze_blocks_pre_dispatch(risk_env, default_filters):
    """Test T18: Any active safety freeze blocks entry orders during fresh pre-dispatch validation."""
    _, journal, gate = risk_env
    journal.trigger_safety_freeze("NETWORK_COMMUNICATION_FAILURE", "Socket timeout")

    snapshot = AccountSnapshot(
        equity=10000.0, available_balance=9000.0, position_side=0, position_quantity=0.0
    )

    with pytest.raises(SafetyFreezeError) as exc:
        gate.validate_pre_dispatch(
            account_snapshot=snapshot,
            mark_price=Decimal("50000.0"),
            filters=default_filters,
            target_pos_amt=Decimal("0.0"),
            all_positions=[{"symbol": "BTCUSDT", "positionAmt": "0.0"}],
            open_orders=[],
            is_entry=True,
        )
    assert "active safety freezes" in str(exc.value)


def test_sizing_clamps_to_max_notional_filter(risk_env):
    """Defect 6: Sizing clamps notional to parsed maxNotional filter."""
    _, _, gate = risk_env
    snapshot = AccountSnapshot(
        equity=10000.0, available_balance=9500.0, position_side=0, position_quantity=0.0
    )
    # Intended notional: 10000 * 0.10 * 10 = 10000 USDT
    # But max_notional = 5000 USDT
    filters = ParsedSymbolFilters(
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        max_qty=Decimal("100.0"),
        min_notional=Decimal("10.0"),
        max_notional=Decimal("5000.0"),
    )
    # Mark price = 50000. Capped notional 5000 / 50000 = 0.100 BTC
    qty = gate.compute_entry_sizing(snapshot, Decimal("50000.0"), filters)
    assert qty == Decimal("0.100")


def test_sizing_clamps_to_bracket_max_notional(risk_env, default_filters):
    """Defect 5: Sizing clamps notional to leverage bracket max notional."""
    _, _, gate = risk_env
    snapshot = AccountSnapshot(
        equity=10000.0, available_balance=9500.0, position_side=0, position_quantity=0.0
    )
    # Intended notional = 10000 USDT. Bracket cap = 4000 USDT.
    # Mark price = 50000. Capped notional 4000 / 50000 = 0.080 BTC.
    qty = gate.compute_entry_sizing(
        snapshot,
        Decimal("50000.0"),
        default_filters,
        bracket_max_notional=Decimal("4000.0"),
    )
    assert qty == Decimal("0.080")


def test_pre_dispatch_leverage_mismatch_rejected(risk_env, default_filters):
    """Defect 5: Leverage mismatch triggers freeze and blocks pre-dispatch."""
    _, journal, gate = risk_env
    snapshot = AccountSnapshot(
        equity=10000.0, available_balance=9000.0, position_side=0, position_quantity=0.0
    )

    with pytest.raises(RiskGateError) as exc:
        gate.validate_pre_dispatch(
            account_snapshot=snapshot,
            mark_price=Decimal("50000.0"),
            filters=default_filters,
            target_pos_amt=Decimal("0.0"),
            all_positions=[{"symbol": "BTCUSDT", "positionAmt": "0.0"}],
            open_orders=[],
            is_entry=True,
            actual_leverage=20,  # config is 10!
        )
    assert "does not match configured leverage" in str(exc.value)
    assert journal.has_active_freeze("LEVERAGE_MISMATCH") is True


def test_pre_dispatch_stale_timestamp_rejected(risk_env, default_filters):
    """Defect 7: Stale pre-dispatch timestamp blocks dispatch."""
    _, _, gate = risk_env
    snapshot = AccountSnapshot(
        equity=10000.0, available_balance=9000.0, position_side=0, position_quantity=0.0
    )
    now_ms = 1_000_000
    timestamps = {
        "mark_price": now_ms - 6_000,  # 6000ms > 5000ms limit!
        "target_position": now_ms - 100,
        "account_snapshot": now_ms - 100,
        "all_positions": now_ms - 100,
        "open_orders": now_ms - 100,
        "leverage": now_ms - 100,
        "filters": now_ms - 100,
        "brackets": now_ms - 100,
    }

    with pytest.raises(StaleMarketDataError) as exc:
        gate.validate_pre_dispatch(
            account_snapshot=snapshot,
            mark_price=Decimal("50000.0"),
            filters=default_filters,
            target_pos_amt=Decimal("0.0"),
            all_positions=[{"symbol": "BTCUSDT", "positionAmt": "0.0"}],
            open_orders=[],
            is_entry=True,
            timestamps=timestamps,
            current_time_ms=now_ms,
        )
    assert "mark price data is stale" in str(exc.value)


def test_pre_dispatch_future_timestamp_rejected(risk_env, default_filters):
    """Defect 7: Future-dated timestamp blocks dispatch."""
    _, _, gate = risk_env
    snapshot = AccountSnapshot(
        equity=10000.0, available_balance=9000.0, position_side=0, position_quantity=0.0
    )
    now_ms = 1_000_000
    timestamps = {
        "mark_price": now_ms + 2_000,  # Future dated > now + 1000ms!
        "target_position": now_ms,
        "account_snapshot": now_ms,
        "all_positions": now_ms,
        "open_orders": now_ms,
        "leverage": now_ms,
        "filters": now_ms,
        "brackets": now_ms,
    }

    with pytest.raises(RiskGateError) as exc:
        gate.validate_pre_dispatch(
            account_snapshot=snapshot,
            mark_price=Decimal("50000.0"),
            filters=default_filters,
            target_pos_amt=Decimal("0.0"),
            all_positions=[{"symbol": "BTCUSDT", "positionAmt": "0.0"}],
            open_orders=[],
            is_entry=True,
            timestamps=timestamps,
            current_time_ms=now_ms,
        )
    assert "future-dated timestamp for 'mark_price'" in str(exc.value)


def test_pre_dispatch_missing_timestamp_key_rejected(risk_env, default_filters):
    """Defect 7: Incomplete timestamp dictionary blocks dispatch."""
    _, _, gate = risk_env
    snapshot = AccountSnapshot(
        equity=10000.0, available_balance=9000.0, position_side=0, position_quantity=0.0
    )
    now_ms = 1_000_000
    # Missing 'brackets' key
    timestamps = {
        "mark_price": now_ms,
        "target_position": now_ms,
        "account_snapshot": now_ms,
        "all_positions": now_ms,
        "open_orders": now_ms,
        "leverage": now_ms,
        "filters": now_ms,
    }

    with pytest.raises(RiskGateError) as exc:
        gate.validate_pre_dispatch(
            account_snapshot=snapshot,
            mark_price=Decimal("50000.0"),
            filters=default_filters,
            target_pos_amt=Decimal("0.0"),
            all_positions=[{"symbol": "BTCUSDT", "positionAmt": "0.0"}],
            open_orders=[],
            is_entry=True,
            timestamps=timestamps,
            current_time_ms=now_ms,
        )
    assert "missing timestamp for 'brackets'" in str(exc.value)

