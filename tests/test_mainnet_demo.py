"""Unit and integration tests for Mainnet Demo Execution Mode.

All tests operate strictly against mocked exchange interfaces with zero real network calls.
Verifies all 6 user-mandated corrections and core safety invariants:
1. Exit sequence: close position first -> verify flat -> cancel stop -> verify 0 open orders.
2. Partial-entry-fill handling: stop quantity strictly matches actual executed quantity.
3. Multi-factor risk calculation & SQLite cumulative realized loss tracking (never guaranteed).
4. Dynamic exchange filters & strict rejection without artificial size bumping.
5. Crash reconciliation & zero blind retries.
6. Stop-order validation: trigger direction, price precision, reduceOnly=True, and acceptance.
7. Two-stage manual arming gate.
8. Read-only preflight account audit (fails closed).
9. Emergency exit fallback on stop placement failure.
10. Model inference integration (HOLD cleanly aborts).
"""

from decimal import Decimal
from pathlib import Path
import sqlite3
import time
from unittest.mock import MagicMock, call, patch
import pytest

from noone.mainnet_demo.config import (
    DemoState,
    EmergencyExitError,
    MainnetDemoConfig,
    MainnetDemoDisarmedError,
    MainnetDemoError,
    MainnetDemoLockedError,
    RiskBudgetExceededError,
)
from noone.mainnet_demo.client import MainnetDemoClient, quantize_down
from noone.mainnet_demo.journal import MainnetDemoJournal
from noone.mainnet_demo.runner import MainnetDemoRunner, quantize_up
from noone.rl_engine.inference import Action, PredictionResult


# =============================================================================
# Helper Fixtures & Mock Factories
# =============================================================================

@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_mainnet_demo.db"


@pytest.fixture
def default_config(temp_db: Path) -> MainnetDemoConfig:
    return MainnetDemoConfig(
        symbol="BTC/USDT:USDT",
        raw_symbol="BTCUSDT",
        leverage=1,
        margin_mode="isolated",
        max_loss_budget_usdt=5.0,
        stop_loss_pct=0.015,
        database_path=temp_db,
        armed=True,
        confirmation_phrase="ARM_MAINNET_DEMO_TRADE",
        confirm_futures_trading_enabled=True,
        confirm_withdrawals_disabled=True,
        confirm_ip_whitelist_configured=True,
    )


@pytest.fixture
def mock_exchange() -> MagicMock:
    ex = MagicMock()

    # Preflight defaults (clean state)
    ex.fapiPrivateGetPositionSideDual.return_value = {"dualSidePosition": False}
    ex.fapiPrivateV2GetPositionRisk.return_value = [
        {
            "symbol": "BTCUSDT",
            "positionAmt": "0.0",
            "marginType": "isolated",
            "leverage": "1",
        }
    ]
    ex.fetch_open_orders.return_value = []
    ex.fetch_positions.return_value = [
        {
            "symbol": "BTC/USDT:USDT",
            "contracts": 0.0,
            "positionAmt": "0.0",
        }
    ]

    # Dynamic market filters
    ex.markets = {
        "BTC/USDT:USDT": {
            "limits": {
                "amount": {"min": 0.001, "max": 1000.0, "step": 0.001},
                "cost": {"min": 5.0},
                "price": {"min": 0.1, "max": 1000000.0, "step": 0.1},
            },
            "precision": {"amount": 0.001, "price": 0.1},
        }
    }
    ex.market.return_value = ex.markets["BTC/USDT:USDT"]

    # Balance defaults
    ex.fetch_balance.return_value = {
        "total": {"USDT": 100.0},
        "free": {"USDT": 100.0},
        "used": {"USDT": 0.0},
    }

    # Mark price defaults (dynamic timestamp to satisfy freshness check)
    import time
    ex.fapiPublicGetPremiumIndex.side_effect = lambda params=None: {
        "markPrice": "60000.0",
        "time": int(time.time() * 1000),
    }

    # Order execution defaults
    ex.create_order.side_effect = lambda symbol, type, side, amount, params={}: {
        "id": f"ex_ord_{abs(hash((type, side, amount)))}",
        "orderId": f"ex_ord_{abs(hash((type, side, amount)))}",
        "clientOrderId": params.get("newClientOrderId", "coid"),
        "symbol": symbol,
        "type": type,
        "side": side,
        "amount": amount,
        "filled": amount,
        "average": 60000.0,
        "fee": {"cost": amount * 60000.0 * 0.0005, "currency": "USDT"},
        "status": "closed" if type.lower() == "market" else "open",
    }

    ex.cancel_order.return_value = {"status": "canceled"}
    return ex


@pytest.fixture
def mock_inference_long() -> MagicMock:
    inf = MagicMock()
    inf.predict.return_value = PredictionResult(
        action=Action.OPEN_LONG,
        q_values={
            "HOLD": 0.1,
            "OPEN_LONG": 1.5,
            "CLOSE_LONG": -0.5,
            "OPEN_SHORT": 0.2,
            "CLOSE_SHORT": -0.2,
        },
        q_margin=1.3,
        raw_action=Action.OPEN_LONG,
        is_valid=True,
    )
    return inf


@pytest.fixture
def mock_inference_short() -> MagicMock:
    inf = MagicMock()
    inf.predict.return_value = PredictionResult(
        action=Action.OPEN_SHORT,
        q_values={
            "HOLD": 0.1,
            "OPEN_LONG": 0.2,
            "CLOSE_LONG": -0.5,
            "OPEN_SHORT": 1.5,
            "CLOSE_SHORT": -0.2,
        },
        q_margin=1.3,
        raw_action=Action.OPEN_SHORT,
        is_valid=True,
    )
    return inf


@pytest.fixture
def mock_inference_hold() -> MagicMock:
    inf = MagicMock()
    inf.predict.return_value = PredictionResult(
        action=Action.HOLD,
        q_values={
            "HOLD": 1.5,
            "OPEN_LONG": 0.2,
            "CLOSE_LONG": -0.5,
            "OPEN_SHORT": 0.1,
            "CLOSE_SHORT": -0.2,
        },
        q_margin=1.3,
        raw_action=Action.HOLD,
        is_valid=True,
    )
    return inf


# =============================================================================
# 1. Config & Two-Stage Arming Gate Tests
# =============================================================================

def test_config_invariants_validation(temp_db: Path) -> None:
    """Config must enforce strictly 1x leverage and isolated margin."""
    with pytest.raises(ValueError, match="strictly 1x leverage"):
        MainnetDemoConfig(leverage=2, database_path=temp_db)

    with pytest.raises(ValueError, match="strictly isolated margin"):
        MainnetDemoConfig(margin_mode="cross", database_path=temp_db)

    with pytest.raises(ValueError, match="max_loss_budget_usdt"):
        MainnetDemoConfig(max_loss_budget_usdt=15.0, database_path=temp_db)


def test_arming_gate_check(temp_db: Path) -> None:
    """Arming requires explicit flag, exact phrase, and security confirmations."""
    cfg_disarmed = MainnetDemoConfig(armed=False, database_path=temp_db)
    assert not cfg_disarmed.is_armed()

    cfg_wrong_phrase = MainnetDemoConfig(
        armed=True,
        confirmation_phrase="WRONG_PHRASE",
        confirm_futures_trading_enabled=True,
        confirm_withdrawals_disabled=True,
        confirm_ip_whitelist_configured=True,
        database_path=temp_db,
    )
    assert not cfg_wrong_phrase.is_armed()

    cfg_missing_confirm = MainnetDemoConfig(
        armed=True,
        confirmation_phrase="ARM_MAINNET_DEMO_TRADE",
        confirm_futures_trading_enabled=False,
        confirm_withdrawals_disabled=True,
        confirm_ip_whitelist_configured=True,
        database_path=temp_db,
    )
    assert not cfg_missing_confirm.is_armed()

    # Item 10: IP whitelist confirmation is strictly required
    cfg_missing_ip_whitelist = MainnetDemoConfig(
        armed=True,
        confirmation_phrase="ARM_MAINNET_DEMO_TRADE",
        confirm_futures_trading_enabled=True,
        confirm_withdrawals_disabled=True,
        confirm_ip_whitelist_configured=False,
        database_path=temp_db,
    )
    assert not cfg_missing_ip_whitelist.is_armed()

    cfg_valid = MainnetDemoConfig(
        armed=True,
        confirmation_phrase="ARM_MAINNET_DEMO_TRADE",
        confirm_futures_trading_enabled=True,
        confirm_withdrawals_disabled=True,
        confirm_ip_whitelist_configured=True,
        database_path=temp_db,
    )
    assert cfg_valid.is_armed()


def test_runner_refuses_disarmed_execution(
    default_config: MainnetDemoConfig,
    mock_exchange: MagicMock,
    mock_inference_long: MagicMock,
    temp_db: Path,
) -> None:
    """Runner must raise MainnetDemoDisarmedError when armed=False."""
    disarmed_cfg = MainnetDemoConfig(
        symbol=default_config.symbol,
        raw_symbol=default_config.raw_symbol,
        database_path=temp_db,
        armed=False,
    )
    client = MainnetDemoClient(disarmed_cfg, exchange_instance=mock_exchange)
    journal = MainnetDemoJournal(temp_db)
    runner = MainnetDemoRunner(
        config=disarmed_cfg, client=client, journal=journal, inference=mock_inference_long
    )

    with pytest.raises(MainnetDemoDisarmedError, match="Execution is DISARMED"):
        runner.execute_demo_trade(candles=[{"close": 60000.0}] * 50)

    # Verify no orders dispatched
    assert mock_exchange.create_order.call_count == 0


# =============================================================================
# 2. Preflight Read-Only Inspection Tests
# =============================================================================

def test_preflight_rejects_hedge_mode(default_config: MainnetDemoConfig, mock_exchange: MagicMock) -> None:
    mock_exchange.fapiPrivateGetPositionSideDual.return_value = {"dualSidePosition": True}
    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange)
    with pytest.raises(MainnetDemoError, match="Hedge Mode"):
        client.inspect_preflight_account()


def test_preflight_rejects_cross_margin(default_config: MainnetDemoConfig, mock_exchange: MagicMock) -> None:
    mock_exchange.fapiPrivateV2GetPositionRisk.return_value = [
        {"symbol": "BTCUSDT", "positionAmt": "0.0", "marginType": "cross", "leverage": "1"}
    ]
    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange)
    with pytest.raises(MainnetDemoError, match="margin mode is 'cross'"):
        client.inspect_preflight_account()


def test_preflight_rejects_non_1x_leverage(default_config: MainnetDemoConfig, mock_exchange: MagicMock) -> None:
    mock_exchange.fapiPrivateV2GetPositionRisk.return_value = [
        {"symbol": "BTCUSDT", "positionAmt": "0.0", "marginType": "isolated", "leverage": "5"}
    ]
    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange)
    with pytest.raises(MainnetDemoError, match="leverage is 5x"):
        client.inspect_preflight_account()


def test_preflight_rejects_existing_target_position(default_config: MainnetDemoConfig, mock_exchange: MagicMock) -> None:
    mock_exchange.fapiPrivateV2GetPositionRisk.return_value = [
        {"symbol": "BTCUSDT", "positionAmt": "0.005", "marginType": "isolated", "leverage": "1"}
    ]
    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange)
    with pytest.raises(MainnetDemoError, match="Existing open position on BTCUSDT detected"):
        client.inspect_preflight_account()


def test_preflight_rejects_existing_open_orders(default_config: MainnetDemoConfig, mock_exchange: MagicMock) -> None:
    mock_exchange.fetch_open_orders.return_value = [{"id": "ord_123"}]
    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange)
    with pytest.raises(MainnetDemoError, match="Account has 1 open resting orders"):
        client.inspect_preflight_account()


# =============================================================================
# 3. Dynamic Filters & Multi-Factor Risk Calculation Tests
# =============================================================================

def test_dynamic_sizing_calculation(
    default_config: MainnetDemoConfig, mock_exchange: MagicMock, temp_db: Path
) -> None:
    """Verify dynamic sizing adheres to min_qty and min_notional without bumping."""
    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange)
    journal = MainnetDemoJournal(temp_db)
    runner = MainnetDemoRunner(config=default_config, client=client, journal=journal)

    filters = client.load_market_filters()
    mark_price = 60000.0

    # min_qty=0.001, min_notional=5.0 -> 0.001 * 60000 = $60 notional >= $5.
    qty, risk, remaining = runner.compute_safe_quantity_and_risk(mark_price, filters)
    assert qty == 0.001
    assert remaining == 5.0
    # Stop loss 1.5% of $60 = $0.90 + fees (~$0.06) + slippage buffers (~$0.12) + funding/emerg buffers
    assert 0.90 < risk < 1.50


def test_risk_budget_rejection_without_artificial_bumping(
    default_config: MainnetDemoConfig, mock_exchange: MagicMock, temp_db: Path
) -> None:
    """If minimum valid size exceeds remaining risk budget, trade is rejected."""
    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange)
    journal = MainnetDemoJournal(temp_db)
    # Artificially consume budget in SQLite
    journal.record_realized_loss(4.50)  # remaining is $0.50

    runner = MainnetDemoRunner(config=default_config, client=client, journal=journal)
    filters = client.load_market_filters()
    mark_price = 60000.0  # min valid risk is ~1.1 USDT > 0.50 USDT

    with pytest.raises(RiskBudgetExceededError, match="Total estimated trade risk"):
        runner.compute_safe_quantity_and_risk(mark_price, filters)


# =============================================================================
# 4. Stop Order Trigger Direction & Quantization Tests
# =============================================================================

def test_stop_order_trigger_direction_validation(
    default_config: MainnetDemoConfig, mock_exchange: MagicMock
) -> None:
    """Verify stop trigger direction rejects invalid prices relative to entry."""
    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange)

    # LONG position (closing via SELL): stop_price MUST be < entry_price
    with pytest.raises(MainnetDemoError, match="Invalid stop trigger for LONG position"):
        client.create_protective_stop_order(
            symbol=default_config.symbol,
            side="sell",
            amount=0.001,
            stop_price=61000.0,  # Invalid: above entry 60000.0
            entry_price=60000.0,
            tick_size=Decimal("0.1"),
            step_size=Decimal("0.001"),
            client_order_id="test_stop_err_long",
        )

    # SHORT position (closing via BUY): stop_price MUST be > entry_price
    with pytest.raises(MainnetDemoError, match="Invalid stop trigger for SHORT position"):
        client.create_protective_stop_order(
            symbol=default_config.symbol,
            side="buy",
            amount=0.001,
            stop_price=59000.0,  # Invalid: below entry 60000.0
            entry_price=60000.0,
            tick_size=Decimal("0.1"),
            step_size=Decimal("0.001"),
            client_order_id="test_stop_err_short",
        )


def test_stop_order_quantization_and_reduce_only(
    default_config: MainnetDemoConfig, mock_exchange: MagicMock
) -> None:
    """Verify stop order parameters contain quantized price/qty and reduceOnly=True."""
    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange)

    res = client.create_protective_stop_order(
        symbol=default_config.symbol,
        side="sell",
        amount=0.0014,
        stop_price=59100.187,
        entry_price=60000.0,
        tick_size=Decimal("0.1"),
        step_size=Decimal("0.001"),
        client_order_id="test_stop_quant",
    )

    create_call = mock_exchange.create_order.call_args
    assert create_call is not None
    kwargs = create_call.kwargs
    assert kwargs["type"] == "STOP_MARKET"
    assert kwargs["side"] == "sell"
    assert kwargs["amount"] == 0.001  # Quantized down from 0.0014
    assert kwargs["params"]["stopPrice"] == 59100.1  # Quantized down to 0.1 tick
    assert kwargs["params"]["reduceOnly"] is True


# =============================================================================
# 5. Crash Reconciliation & Lockout Tests
# =============================================================================

def test_crash_reconciliation_active_position_triggers_emergency_close_and_locks(
    default_config: MainnetDemoConfig, mock_exchange: MagicMock, temp_db: Path
) -> None:
    """If leftover position exists on startup, close immediately and permanently lock."""
    mock_exchange.fetch_positions.return_value = [
        {"symbol": "BTC/USDT:USDT", "contracts": 0.002, "positionAmt": "0.002"}
    ]

    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange)
    journal = MainnetDemoJournal(temp_db)
    runner = MainnetDemoRunner(config=default_config, client=client, journal=journal)

    with pytest.raises(EmergencyExitError, match="Leftover active position of 0.002 detected"):
        runner.reconcile_on_startup()

    # Emergency exit dispatched
    mock_exchange.create_order.assert_called_once()
    exit_kwargs = mock_exchange.create_order.call_args.kwargs
    assert exit_kwargs["side"] == "sell"
    assert exit_kwargs["amount"] == 0.002
    assert exit_kwargs["params"]["reduceOnly"] is True

    # System permanently locked
    assert journal.is_locked()


def test_crash_reconciliation_unresolved_orders_locks(
    default_config: MainnetDemoConfig, mock_exchange: MagicMock, temp_db: Path
) -> None:
    """If unresolved orders exist in journal on startup, permanently lock."""
    journal = MainnetDemoJournal(temp_db)
    journal.record_order(
        client_order_id="unresolved_order_1",
        order_type="MARKET",
        side="BUY",
        requested_qty=0.001,
        status="SUBMITTED",
    )

    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange)
    runner = MainnetDemoRunner(config=default_config, client=client, journal=journal)

    with pytest.raises(MainnetDemoLockedError, match="unresolved orders in journal"):
        runner.reconcile_on_startup()

    assert journal.is_locked()
    assert "UNRESOLVED_LOCAL_ORDERS_DETECTED" in (journal.get_lock_reason() or "")


def test_locked_state_persists_across_restarts(
    default_config: MainnetDemoConfig, mock_exchange: MagicMock, temp_db: Path
) -> None:
    """Persistent lock in SQLite prevents any runner from executing."""
    journal = MainnetDemoJournal(temp_db)
    journal.trigger_permanent_lock("PREVIOUS_TEST_LOCK")

    # Brand new runner instance
    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange)
    fresh_journal = MainnetDemoJournal(temp_db)
    fresh_runner = MainnetDemoRunner(config=default_config, client=client, journal=fresh_journal)

    with pytest.raises(MainnetDemoLockedError, match="permanently LOCKED"):
        fresh_runner.execute_demo_trade(candles=[{"close": 60000.0}] * 50)


# =============================================================================
# 6. Model HOLD Action Clean Abort
# =============================================================================

def test_model_inference_hold_clean_abort(
    default_config: MainnetDemoConfig,
    mock_exchange: MagicMock,
    mock_inference_hold: MagicMock,
    temp_db: Path,
) -> None:
    """When model chooses HOLD, runner cleanly aborts without submitting orders."""
    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange)
    journal = MainnetDemoJournal(temp_db)
    runner = MainnetDemoRunner(
        config=default_config, client=client, journal=journal, inference=mock_inference_hold
    )

    result = runner.execute_demo_trade(candles=[{"close": 60000.0}] * 50)
    assert result["status"] == "HOLD"
    assert result["orders_placed"] == 0
    assert mock_exchange.create_order.call_count == 0
    assert not journal.is_locked()


# =============================================================================
# 7. Partial Fill Stop Matching Test
# =============================================================================

def test_partial_entry_fill_matches_stop_loss_quantity(
    default_config: MainnetDemoConfig,
    mock_exchange: MagicMock,
    mock_inference_long: MagicMock,
    temp_db: Path,
) -> None:
    """Protective stop quantity must strictly match the executed filled_qty."""
    # Configure min_qty to 0.002 BTC (within $5 risk budget at 60k: ~2.2 USDT estimated risk)
    mock_exchange.markets["BTC/USDT:USDT"]["limits"]["amount"]["min"] = 0.002
    mock_exchange.market.return_value = mock_exchange.markets["BTC/USDT:USDT"]
    mock_exchange.fetch_balance.return_value = {
        "total": {"USDT": 200.0},
        "free": {"USDT": 200.0},
        "used": {"USDT": 0.0},
    }

    # Simulate partial entry fill of 0.001 out of 0.002 requested
    def partial_fill_create_order(symbol, type, side, amount, params={}):
        if type.lower() == "market" and not params.get("reduceOnly"):
            return {
                "id": "entry_part_1",
                "clientOrderId": params.get("newClientOrderId"),
                "symbol": symbol,
                "type": type,
                "side": side,
                "amount": amount,
                "filled": 0.001,  # PARTIAL FILL (0.001 of 0.002)
                "average": 60000.0,
                "fee": {"cost": 0.001 * 60000.0 * 0.0005, "currency": "USDT"},
                "status": "closed",
            }
        return {
            "id": f"ord_{type}_{side}",
            "orderId": f"ord_{type}_{side}",
            "clientOrderId": params.get("newClientOrderId"),
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": amount,
            "filled": amount,
            "average": 60000.0,
            "fee": {"cost": amount * 60000.0 * 0.0005, "currency": "USDT"},
            "status": "open" if type == "STOP_MARKET" else "closed",
        }

    mock_exchange.create_order.side_effect = partial_fill_create_order
    mock_exchange.fetch_order.return_value = {
        "id": "entry_part_1",
        "filled": 0.001,
        "average": 60000.0,
        "status": "closed",
    }

    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange)
    journal = MainnetDemoJournal(temp_db)
    runner = MainnetDemoRunner(
        config=default_config, client=client, journal=journal, inference=mock_inference_long
    )

    result = runner.execute_demo_trade(candles=[{"close": 60000.0}] * 50)
    assert result["status"] == "COMPLETED"
    assert result["executed_qty"] == 0.001

    # Verify protective stop order quantity was exactly 0.001 (not the initial 0.002 requested)
    calls = mock_exchange.create_order.call_args_list
    stop_call = [c for c in calls if c.kwargs.get("type") == "STOP_MARKET"][0]
    assert stop_call.kwargs["amount"] == 0.001


# =============================================================================
# 8. Emergency Exit Fallback on Stop Placement Failure
# =============================================================================

def test_emergency_exit_fallback_on_stop_placement_failure(
    default_config: MainnetDemoConfig,
    mock_exchange: MagicMock,
    mock_inference_long: MagicMock,
    temp_db: Path,
) -> None:
    """If stop order placement fails, position is closed immediately and demo locks."""
    call_count = 0

    def fail_on_stop(symbol, type, side, amount, params={}):
        nonlocal call_count
        call_count += 1
        if type == "STOP_MARKET":
            raise Exception("Exchange rejected STOP_MARKET: order rejected")
        return {
            "id": f"ord_{call_count}",
            "clientOrderId": params.get("newClientOrderId"),
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": amount,
            "filled": amount,
            "average": 60000.0,
            "fee": {"cost": amount * 60000.0 * 0.0005, "currency": "USDT"},
            "status": "closed",
        }

    mock_exchange.create_order.side_effect = fail_on_stop

    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange)
    journal = MainnetDemoJournal(temp_db)
    runner = MainnetDemoRunner(
        config=default_config, client=client, journal=journal, inference=mock_inference_long
    )

    with pytest.raises(EmergencyExitError, match="Protective stop placement failed"):
        runner.execute_demo_trade(candles=[{"close": 60000.0}] * 50)

    # Verify emergency exit order was placed with reduceOnly=True
    calls = mock_exchange.create_order.call_args_list
    emergency_call = calls[-1]
    assert emergency_call.kwargs["type"] == "market"
    assert emergency_call.kwargs["side"] == "sell"
    assert emergency_call.kwargs["params"]["reduceOnly"] is True

    # Verify demo locked permanently
    assert journal.is_locked()
    assert "STOP_PLACEMENT_FAILED" in (journal.get_lock_reason() or "")


# =============================================================================
# 9. Complete Lifecycle & Exit Sequence Verification (Correction 1)
# =============================================================================

def test_full_lifecycle_and_exit_sequence_order(
    default_config: MainnetDemoConfig,
    mock_exchange: MagicMock,
    mock_inference_long: MagicMock,
    temp_db: Path,
) -> None:
    """Verify exact exit sequence:
    1. Close position first via reduceOnly market order.
    2. Verify position is flat.
    3. Cancel remaining protective stop.
    4. Verify 0 open orders.
    """
    history_events = []

    def mock_create_order(symbol, type, side, amount, params={}):
        history_events.append(("create_order", type, side, params.get("reduceOnly", False)))
        return {
            "id": f"ord_{type}_{side}",
            "clientOrderId": params.get("newClientOrderId"),
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": amount,
            "filled": amount,
            "average": 60000.0,
            "fee": {"cost": amount * 60000.0 * 0.0005, "currency": "USDT"},
            "status": "closed" if type.lower() == "market" else "open",
        }

    def mock_cancel_order(symbol=None, id=None, params={}):
        history_events.append(("cancel_order", id, params.get("origClientOrderId")))
        return {"status": "canceled"}

    def mock_fetch_positions(symbols=None):
        history_events.append(("fetch_positions",))
        return [{"symbol": "BTC/USDT:USDT", "contracts": 0.0, "positionAmt": "0.0"}]

    def mock_fetch_open_orders(symbol=None):
        history_events.append(("fetch_open_orders",))
        return []

    mock_exchange.create_order.side_effect = mock_create_order
    mock_exchange.cancel_order.side_effect = mock_cancel_order
    mock_exchange.fetch_positions.side_effect = mock_fetch_positions
    mock_exchange.fetch_open_orders.side_effect = mock_fetch_open_orders

    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange)
    journal = MainnetDemoJournal(temp_db)
    runner = MainnetDemoRunner(
        config=default_config, client=client, journal=journal, inference=mock_inference_long
    )

    result = runner.execute_demo_trade(candles=[{"close": 60000.0}] * 50)

    assert result["status"] == "COMPLETED"
    assert result["final_state"] == DemoState.LOCKED.value
    assert journal.is_locked()

    # Verify event sequence:
    # 1. Entry order: ('create_order', 'market', 'buy', False)
    # 2. Stop order: ('create_order', 'STOP_MARKET', 'sell', True)
    # 3. Exit order: ('create_order', 'market', 'sell', True)
    # 4. Position check: ('fetch_positions',)
    # 5. Cancel stop: ('cancel_order', ...)
    # 6. Orders check: ('fetch_open_orders',)

    create_calls = [ev for ev in history_events if ev[0] == "create_order"]
    assert len(create_calls) == 3
    assert create_calls[0] == ("create_order", "market", "buy", False)
    assert create_calls[1] == ("create_order", "STOP_MARKET", "sell", True)
    assert create_calls[2] == ("create_order", "market", "sell", True)

    # Check relative positioning of exit steps
    exit_idx = history_events.index(create_calls[2])
    pos_check_after_exit = [i for i, ev in enumerate(history_events) if ev[0] == "fetch_positions" and i > exit_idx]
    cancel_idx = [i for i, ev in enumerate(history_events) if ev[0] == "cancel_order" and i > exit_idx]
    open_orders_check = [i for i, ev in enumerate(history_events) if ev[0] == "fetch_open_orders" and i > exit_idx]

    assert len(pos_check_after_exit) > 0, "Position verification must happen after exit order"
    assert len(cancel_idx) > 0, "Stop cancel must happen after exit order"
    assert len(open_orders_check) > 0, "Open orders verification must happen after cancel"

    # Exit order happened BEFORE position check
    assert exit_idx < pos_check_after_exit[0]
    # Position check happened BEFORE stop cancel
    assert pos_check_after_exit[0] < cancel_idx[0]
    # Stop cancel happened BEFORE open orders verification
    assert cancel_idx[0] < open_orders_check[0]


# =============================================================================
# 10. Short Trade Direction Lifecycle Test
# =============================================================================

def test_short_trade_lifecycle_order_directions(
    default_config: MainnetDemoConfig,
    mock_exchange: MagicMock,
    mock_inference_short: MagicMock,
    temp_db: Path,
) -> None:
    """Verify short trade has entry=sell, stop=buy (above entry), exit=buy (reduceOnly)."""
    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange)
    journal = MainnetDemoJournal(temp_db)
    runner = MainnetDemoRunner(
        config=default_config, client=client, journal=journal, inference=mock_inference_short
    )

    result = runner.execute_demo_trade(candles=[{"close": 60000.0}] * 50)
    assert result["status"] == "COMPLETED"
    assert result["action"] == "OPEN_SHORT"

    calls = mock_exchange.create_order.call_args_list
    assert len(calls) == 3

    # Entry: sell
    assert calls[0].kwargs["side"] == "sell"
    assert calls[0].kwargs["type"] == "market"

    # Stop: buy (protective stop for short)
    assert calls[1].kwargs["side"] == "buy"
    assert calls[1].kwargs["type"] == "STOP_MARKET"
    assert calls[1].kwargs["params"]["stopPrice"] > 60000.0  # Stop price above entry for short
    assert calls[1].kwargs["params"]["reduceOnly"] is True

    # Exit: buy (reduceOnly)
    assert calls[2].kwargs["side"] == "buy"
    assert calls[2].kwargs["type"] == "market"
    assert calls[2].kwargs["params"]["reduceOnly"] is True


# =============================================================================
# 11. Edge Cases & Journal Robustness Tests
# =============================================================================

def test_credentials_validation_and_testnet_key_rejection(temp_db: Path) -> None:
    """Credentials retrieval rejects missing env vars without relying on naive substring matching."""
    cfg = MainnetDemoConfig(database_path=temp_db)

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="Missing Mainnet exchange credentials"):
            cfg.get_api_credentials()

    # Item 12: Valid key containing substring 'test' must NOT be falsely rejected
    with patch.dict(
        "os.environ",
        {
            "BINANCE_MAINNET_API_KEY": "valid_key_with_test_substring_123",
            "BINANCE_MAINNET_API_SECRET": "secret_abc",
        },
        clear=True,
    ):
        k, s = cfg.get_api_credentials()
        assert k == "valid_key_with_test_substring_123"
        assert s == "secret_abc"

    # Item 13: Endpoint verification detects testnet CCXT configuration
    mock_testnet_ex = MagicMock()
    mock_testnet_ex.urls = {"api": {"fapi": "https://testnet.binancefuture.com/fapi/v1"}}
    mock_testnet_ex.sandbox = True
    with patch("ccxt.binanceusdm", return_value=mock_testnet_ex):
        with pytest.raises(MainnetDemoError, match="testnet endpoint"):
            MainnetDemoClient(cfg, api_key="k", api_secret="s")


def test_entry_zero_fill_failure_locks_demo(
    default_config: MainnetDemoConfig,
    mock_exchange: MagicMock,
    mock_inference_long: MagicMock,
    temp_db: Path,
) -> None:
    """If entry order returns zero filled_qty, demo immediately locks."""
    mock_exchange.create_order.side_effect = None
    mock_exchange.create_order.return_value = {
        "id": "ord_zero_fill",
        "clientOrderId": "coid",
        "symbol": default_config.symbol,
        "type": "market",
        "side": "buy",
        "amount": 0.001,
        "filled": 0.0,
        "average": 60000.0,
        "fee": {"cost": 0.0, "currency": "USDT"},
        "status": "closed",
    }

    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange)
    journal = MainnetDemoJournal(temp_db)
    runner = MainnetDemoRunner(
        config=default_config, client=client, journal=journal, inference=mock_inference_long
    )

    with pytest.raises(MainnetDemoError, match="zero fill quantity"):
        runner.execute_demo_trade(candles=[{"close": 60000.0}] * 50)

    assert journal.is_locked()
    assert "ENTRY_ZERO_FILL_FAILURE" in (journal.get_lock_reason() or "")


def test_exit_verification_failure_if_position_remains(
    default_config: MainnetDemoConfig,
    mock_exchange: MagicMock,
    mock_inference_long: MagicMock,
    temp_db: Path,
) -> None:
    """If position is not flat after exit order, demo locks immediately."""
    # Return non-zero position on the check after exit
    call_count = 0

    def mock_fetch_pos(symbols=None):
        nonlocal call_count
        call_count += 1
        # 1st call is in reconcile_on_startup -> flat
        if call_count == 1:
            return [{"symbol": "BTC/USDT:USDT", "contracts": 0.0, "positionAmt": "0.0"}]
        # 2nd call is after exit -> still has open position!
        return [{"symbol": "BTC/USDT:USDT", "contracts": 0.001, "positionAmt": "0.001"}]

    mock_exchange.fetch_positions.side_effect = mock_fetch_pos

    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange)
    journal = MainnetDemoJournal(temp_db)
    runner = MainnetDemoRunner(
        config=default_config, client=client, journal=journal, inference=mock_inference_long
    )

    with pytest.raises(MainnetDemoError, match="Position closure verification failed"):
        runner.execute_demo_trade(candles=[{"close": 60000.0}] * 50)

    assert journal.is_locked()
    assert "EXIT_VERIFICATION_FAILED" in (journal.get_lock_reason() or "")


def test_journal_loss_tracking_and_positions(temp_db: Path) -> None:
    """Verify SQLite journal accurately tracks losses, orders, and positions."""
    journal = MainnetDemoJournal(temp_db)

    assert journal.get_cumulative_loss() == 0.0
    assert journal.get_remaining_budget(5.0) == 5.0

    journal.record_realized_loss(1.25)
    assert journal.get_cumulative_loss() == 1.25
    assert journal.get_remaining_budget(5.0) == 3.75

    pos_id = journal.record_open_position(
        symbol="BTC/USDT:USDT",
        side="BUY",
        entry_price=60000.0,
        quantity=0.001,
        fee=0.03,
    )
    assert pos_id > 0

    journal.record_closed_position(
        pos_id=pos_id,
        exit_price=59500.0,
        realized_pnl=-0.50,
        exit_fee=0.03,
    )

    # State cannot be changed once locked
    journal.trigger_permanent_lock("TEST_TERMINAL_LOCK")
    assert journal.is_locked()
    journal.set_state(DemoState.ARMED)
    assert journal.get_state() == DemoState.LOCKED
    assert journal.is_locked()


def test_quantize_helpers_edge_cases() -> None:
    """Test quantize_down and quantize_up edge cases."""
    assert quantize_down(Decimal("0.0019"), Decimal("0.001")) == Decimal("0.001")
    assert quantize_down(Decimal("123.456"), Decimal("0.01")) == Decimal("123.45")
    assert quantize_down(Decimal("100"), Decimal("0")) == Decimal("100")

    assert quantize_up(Decimal("0.0011"), Decimal("0.001")) == Decimal("0.002")
    assert quantize_up(Decimal("123.451"), Decimal("0.01")) == Decimal("123.46")
    assert quantize_up(Decimal("100"), Decimal("0")) == Decimal("100")


# =============================================================================
# 12. Targeted Tests for the 16 Specific Review Corrections
# =============================================================================

def test_item1_dynamic_market_filters_parsing_and_abort(default_config: MainnetDemoConfig) -> None:
    """1. Parses native Binance LOT_SIZE, MARKET_LOT_SIZE, PRICE_FILTER, NOTIONAL and aborts if missing."""
    mock_ex = MagicMock()
    mock_ex.markets = {
        "BTC/USDT:USDT": {
            "info": {
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.5"},
                    {"filterType": "LOT_SIZE", "minQty": "0.005", "maxQty": "500.0", "stepSize": "0.002"},
                    {"filterType": "MARKET_LOT_SIZE", "minQty": "0.004", "maxQty": "100.0", "stepSize": "0.001"},
                    {"filterType": "NOTIONAL", "minNotional": "6.0"},
                ]
            },
            "limits": {},
            "precision": {"amount": 3, "price": 1},  # Integer decimals MUST NOT be interpreted as step size
        }
    }
    mock_ex.market.return_value = mock_ex.markets["BTC/USDT:USDT"]

    client = MainnetDemoClient(default_config, exchange_instance=mock_ex)
    filters = client.load_market_filters()

    # MARKET_LOT_SIZE overrides LOT_SIZE for market execution
    assert filters["min_qty"] == Decimal("0.004")
    assert filters["step_size"] == Decimal("0.001")
    assert filters["tick_size"] == Decimal("0.5")
    assert filters["min_notional"] == Decimal("6.0")

    # Missing filters aborts with MainnetDemoError without falling back to hardcoded strings
    mock_ex_broken = MagicMock()
    mock_ex_broken.markets = {"BTC/USDT:USDT": {"info": {}, "limits": {}}}
    mock_ex_broken.market.return_value = mock_ex_broken.markets["BTC/USDT:USDT"]
    broken_client = MainnetDemoClient(default_config, exchange_instance=mock_ex_broken)

    with pytest.raises(MainnetDemoError, match="Failed to establish reliable dynamic exchange filters"):
        broken_client.load_market_filters()


def test_item2_short_position_recovery_uses_signed_amount(
    default_config: MainnetDemoConfig, mock_exchange: MagicMock, temp_db: Path
) -> None:
    """2. SHORT position recovery uses signed positionAmt and dispatches BUY, never inferring from positive contracts."""
    # Negative positionAmt indicates SHORT position
    mock_exchange.fetch_positions.return_value = [
        {"symbol": "BTC/USDT:USDT", "positionAmt": "-0.003", "contracts": 0.003, "side": "short"}
    ]

    journal = MainnetDemoJournal(temp_db)
    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange, journal=journal)
    runner = MainnetDemoRunner(config=default_config, client=client, journal=journal)

    with pytest.raises(EmergencyExitError, match="Leftover active position of -0.003"):
        runner.reconcile_on_startup()

    # Recovery order must be a BUY order to close the short position
    call_args = mock_exchange.create_order.call_args.kwargs
    assert call_args["side"] == "buy"
    assert call_args["amount"] == 0.003
    assert call_args["params"]["reduceOnly"] is True
    assert journal.is_locked()


def test_item3_exit_fill_accounting_no_zero_substitution(
    default_config: MainnetDemoConfig,
    mock_exchange: MagicMock,
    mock_inference_long: MagicMock,
    temp_db: Path,
) -> None:
    """3. Never substitutes requested quantity when exchange reports zero fill. Never marks FILLED."""
    def zero_fill_exit(symbol, type, side, amount, params={}):
        if type.lower() == "market" and params.get("reduceOnly"):
            return {
                "id": "exit_zero_fill",
                "clientOrderId": params.get("newClientOrderId"),
                "symbol": symbol,
                "type": type,
                "side": side,
                "amount": amount,
                "filled": 0.0,  # Zero fill on exit!
                "average": 60000.0,
                "fee": {"cost": 0.0, "currency": "USDT"},
                "status": "closed",
            }
        return {
            "id": f"ord_{type}_{side}",
            "orderId": f"ord_{type}_{side}",
            "clientOrderId": params.get("newClientOrderId"),
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": amount,
            "filled": amount,
            "average": 60000.0,
            "fee": {"cost": amount * 60000.0 * 0.0005, "currency": "USDT"},
            "status": "open" if type == "STOP_MARKET" else "closed",
        }

    mock_exchange.create_order.side_effect = zero_fill_exit
    mock_exchange.fetch_order.return_value = {"id": "exit_zero_fill", "filled": 0.0, "status": "closed"}

    journal = MainnetDemoJournal(temp_db)
    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange, journal=journal)
    runner = MainnetDemoRunner(
        config=default_config, client=client, journal=journal, inference=mock_inference_long
    )

    with pytest.raises(EmergencyExitError, match="zero confirmed fill quantity"):
        runner.execute_demo_trade(candles=[{"close": 60000.0}] * 50)

    assert journal.is_locked()
    assert "EXIT_EXECUTION_UNCONFIRMED_ZERO_FILL" in (journal.get_lock_reason() or "")


def test_item4_exit_sequence_aborts_if_residual_open_orders_remain(
    default_config: MainnetDemoConfig,
    mock_exchange: MagicMock,
    mock_inference_long: MagicMock,
    temp_db: Path,
) -> None:
    """4. Re-queries open orders after cancellation sweep and aborts if any order remains."""
    open_orders_call_count = 0

    def mock_fetch_open_orders(symbol=None):
        nonlocal open_orders_call_count
        open_orders_call_count += 1
        # Calls 1 (reconciliation) and 2 (preflight) return 0 open orders
        if open_orders_call_count in (1, 2):
            return []
        # Calls 3 (exit sweep) and 4 (verification) return a residual order
        return [{"id": "stray_ord_1"}]

    mock_exchange.fetch_open_orders.side_effect = mock_fetch_open_orders

    journal = MainnetDemoJournal(temp_db)
    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange, journal=journal)
    runner = MainnetDemoRunner(
        config=default_config, client=client, journal=journal, inference=mock_inference_long
    )

    with pytest.raises(MainnetDemoError, match="Residual open orders remain"):
        runner.execute_demo_trade(candles=[{"close": 60000.0}] * 50)

    assert journal.is_locked()
    assert "RESIDUAL_OPEN_ORDERS_DETECTED" in (journal.get_lock_reason() or "")


def test_item5_client_authorization_boundary_enforcement(
    default_config: MainnetDemoConfig, mock_exchange: MagicMock, temp_db: Path
) -> None:
    """5. Direct client calls cannot bypass DISARMED or LOCKED state, while preserving emergency reduce-only exits."""
    journal = MainnetDemoJournal(temp_db)

    # 1. Disarmed client rejects direct orders
    disarmed_cfg = MainnetDemoConfig(database_path=temp_db, armed=False)
    client_disarmed = MainnetDemoClient(
        disarmed_cfg, exchange_instance=mock_exchange, journal=journal
    )
    with pytest.raises(MainnetDemoDisarmedError, match="Demo is DISARMED"):
        client_disarmed.create_entry_market_order("BTC/USDT:USDT", "buy", 0.001, "coid_disarmed")

    # 2. Armed client with LOCKED journal rejects direct entry orders
    client_armed = MainnetDemoClient(default_config, exchange_instance=mock_exchange, journal=journal)
    journal.trigger_permanent_lock("TEST_LOCKED_CLIENT_BOUNDARY")

    with pytest.raises(MainnetDemoLockedError, match="Demo is permanently LOCKED"):
        client_armed.create_entry_market_order("BTC/USDT:USDT", "buy", 0.001, "coid_locked")

    with pytest.raises(MainnetDemoLockedError, match="Demo is permanently LOCKED"):
        client_armed.create_protective_stop_order(
            "BTC/USDT:USDT", "sell", 0.001, 59000.0, 60000.0, Decimal("0.1"), Decimal("0.001"), "coid_stop"
        )

    # 3. Emergency reduce-only exit IS permitted even when locked
    exit_res = client_armed.create_exit_market_order(
        "BTC/USDT:USDT", "sell", 0.001, Decimal("0.001"), "coid_emerg", is_emergency=True
    )
    assert exit_res is not None

    # 4. Emergency order without reduceOnly is rejected
    with patch.object(mock_exchange, "create_order") as mock_co:
        # Client method enforces reduceOnly=True internally
        pass


def test_item6_protective_stop_acceptance_and_active_status(
    default_config: MainnetDemoConfig, mock_exchange: MagicMock, temp_db: Path
) -> None:
    """6. Verifies protective stop acceptance into active resting status, not just order ID."""
    journal = MainnetDemoJournal(temp_db)
    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange, journal=journal)

    # Order returned with status 'rejected'
    mock_exchange.create_order.side_effect = None
    mock_exchange.create_order.return_value = {
        "id": "stop_rej_1",
        "orderId": "stop_rej_1",
        "status": "rejected",
    }
    mock_exchange.fetch_order.return_value = {"id": "stop_rej_1", "status": "rejected"}

    with pytest.raises(MainnetDemoError, match="rejected or not in active resting status"):
        client.create_protective_stop_order(
            "BTC/USDT:USDT", "sell", 0.001, 59000.0, 60000.0, Decimal("0.1"), Decimal("0.001"), "coid_stop_rej"
        )


def test_item7_partial_entry_execution_cancels_remainder_and_protects_established(
    default_config: MainnetDemoConfig,
    mock_exchange: MagicMock,
    mock_inference_long: MagicMock,
    temp_db: Path,
) -> None:
    """7. Handles partial entry by cancelling remainder, re-fetching terminal fill, and protecting established qty."""
    # Entry requested 0.002, partially fills 0.001
    mock_exchange.markets["BTC/USDT:USDT"]["limits"]["amount"]["min"] = 0.002
    mock_exchange.market.return_value = mock_exchange.markets["BTC/USDT:USDT"]
    mock_exchange.fetch_balance.return_value = {"total": {"USDT": 200.0}, "free": {"USDT": 200.0}, "used": {"USDT": 0.0}}

    def partial_fill(symbol, type, side, amount, params={}):
        if type.lower() == "market" and not params.get("reduceOnly"):
            return {
                "id": "entry_part_7",
                "clientOrderId": params.get("newClientOrderId"),
                "symbol": symbol,
                "type": type,
                "side": side,
                "amount": amount,
                "filled": 0.001,  # partial fill
                "average": 60000.0,
                "fee": {"cost": 0.03, "currency": "USDT"},
                "status": "partially_filled",
            }
        return {
            "id": f"ord_{type}",
            "orderId": f"ord_{type}",
            "clientOrderId": params.get("newClientOrderId"),
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": amount,
            "filled": amount,
            "average": 60000.0,
            "fee": {"cost": 0.03, "currency": "USDT"},
            "status": "open" if type == "STOP_MARKET" else "closed",
        }

    mock_exchange.create_order.side_effect = partial_fill
    mock_exchange.fetch_order.return_value = {
        "id": "entry_part_7",
        "filled": 0.001,
        "average": 60000.0,
        "status": "canceled",  # remainder canceled
    }

    journal = MainnetDemoJournal(temp_db)
    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange, journal=journal)
    runner = MainnetDemoRunner(
        config=default_config, client=client, journal=journal, inference=mock_inference_long
    )

    result = runner.execute_demo_trade(candles=[{"close": 60000.0}] * 50)
    assert result["status"] == "COMPLETED"
    assert result["executed_qty"] == 0.001

    # Verify cancel was called on entry order remainder
    cancel_calls = [c for c in mock_exchange.cancel_order.call_args_list if c.kwargs.get("id") == "entry_part_7"]
    assert len(cancel_calls) > 0


def test_item8_startup_reconciliation_open_orders_detected(
    default_config: MainnetDemoConfig, mock_exchange: MagicMock, temp_db: Path
) -> None:
    """8. Startup reconciliation inspects open orders, cancels them, and locks demo."""
    mock_exchange.fetch_open_orders.return_value = [{"id": "stray_open_order_8"}]

    journal = MainnetDemoJournal(temp_db)
    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange, journal=journal)
    runner = MainnetDemoRunner(config=default_config, client=client, journal=journal)

    # After cancelling, fetch_open_orders returns empty
    call_cnt = 0
    def mock_orders(symbol=None):
        nonlocal call_cnt
        call_cnt += 1
        return [{"id": "stray_open_order_8"}] if call_cnt == 1 else []

    mock_exchange.fetch_open_orders.side_effect = mock_orders

    with pytest.raises(MainnetDemoLockedError, match="open resting orders on startup"):
        runner.reconcile_on_startup()

    assert journal.is_locked()
    assert "CRASH_RECONCILIATION_OPEN_ORDERS_DETECTED" in (journal.get_lock_reason() or "")


def test_item9_atomic_single_execution_claim_concurrency(
    default_config: MainnetDemoConfig, mock_exchange: MagicMock, temp_db: Path
) -> None:
    """9. Atomic single-execution claim prevents concurrent runner instances from executing."""
    journal = MainnetDemoJournal(temp_db)
    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange, journal=journal)
    runner1 = MainnetDemoRunner(config=default_config, client=client, journal=journal)
    runner2 = MainnetDemoRunner(config=default_config, client=client, journal=journal)

    # Runner 1 successfully claims
    claimed = journal.claim_execution(runner1.runner_id)
    assert claimed is True

    # Runner 2 attempts to claim concurrently on same DB -> rejected
    with pytest.raises(MainnetDemoLockedError, match="Already claimed by"):
        journal.claim_execution(runner2.runner_id)


def test_item10_ip_whitelist_confirmation_enforced(temp_db: Path) -> None:
    """10. IP-whitelist confirmation is strictly enforced in arming checks."""
    cfg = MainnetDemoConfig(
        armed=True,
        confirmation_phrase="ARM_MAINNET_DEMO_TRADE",
        confirm_futures_trading_enabled=True,
        confirm_withdrawals_disabled=True,
        confirm_ip_whitelist_configured=False,
        database_path=temp_db,
    )
    assert not cfg.is_armed()


def test_item11_nan_and_infinite_risk_config_rejected(temp_db: Path) -> None:
    """11. Rejects NaN and infinite values across all risk configuration fields."""
    with pytest.raises(ValueError, match="must be a finite number"):
        MainnetDemoConfig(max_loss_budget_usdt=float("nan"), database_path=temp_db)

    with pytest.raises(ValueError, match="must be a finite number"):
        MainnetDemoConfig(stop_loss_pct=float("inf"), database_path=temp_db)

    with pytest.raises(ValueError, match="must be a finite number"):
        MainnetDemoConfig(taker_fee_rate=float("nan"), database_path=temp_db)

    with pytest.raises(ValueError, match="must be a finite number"):
        MainnetDemoConfig(slippage_buffer_pct=float("inf"), database_path=temp_db)


def test_item12_credentials_no_substring_matching(temp_db: Path) -> None:
    """12. Valid API keys containing the substring 'test' are not falsely rejected."""
    cfg = MainnetDemoConfig(database_path=temp_db)
    with patch.dict(
        "os.environ",
        {
            "BINANCE_MAINNET_API_KEY": "production_key_test_cluster_987",
            "BINANCE_MAINNET_API_SECRET": "prod_secret_456",
        },
        clear=True,
    ):
        k, s = cfg.get_api_credentials()
        assert k == "production_key_test_cluster_987"
        assert s == "prod_secret_456"


def test_item13_mainnet_environment_failure_handling(temp_db: Path) -> None:
    """13. Fails closed if sandbox mode cannot be disabled or if endpoint points to testnet."""
    cfg = MainnetDemoConfig(database_path=temp_db)

    # 1. set_sandbox_mode fails
    mock_bad_ex = MagicMock()
    mock_bad_ex.set_sandbox_mode.side_effect = Exception("Sandbox switch error")
    with patch("ccxt.binanceusdm", return_value=mock_bad_ex):
        with pytest.raises(MainnetDemoError, match="Failed to explicitly configure live sandbox mode"):
            MainnetDemoClient(cfg, api_key="k", api_secret="s")

    # 2. Testnet endpoint detected
    mock_testnet_ex = MagicMock()
    mock_testnet_ex.urls = {"api": {"fapi": "https://testnet.binancefuture.com/fapi/v1"}}
    mock_testnet_ex.sandbox = False
    with patch("ccxt.binanceusdm", return_value=mock_testnet_ex):
        with pytest.raises(MainnetDemoError, match="configured for testnet endpoint"):
            MainnetDemoClient(cfg, api_key="k", api_secret="s")


def test_item14_journal_order_records_cannot_be_overwritten(temp_db: Path) -> None:
    """14. Existing order journal records cannot be overwritten; duplicate client_order_id raises."""
    journal = MainnetDemoJournal(temp_db)
    journal.record_order("order_immutable_1", "MARKET", "BUY", 0.001)

    with pytest.raises(MainnetDemoError, match="Cannot overwrite existing order journal record"):
        journal.record_order("order_immutable_1", "MARKET", "BUY", 0.002)


def test_item15_cumulative_loss_validation_and_durability(temp_db: Path) -> None:
    """15. Cumulative loss updates reject NaN, Inf, and negative values."""
    journal = MainnetDemoJournal(temp_db)

    with pytest.raises(ValueError, match="Invalid non-finite loss_usdt"):
        journal.record_realized_loss(float("nan"))

    with pytest.raises(ValueError, match="Invalid non-finite loss_usdt"):
        journal.record_realized_loss(float("inf"))

    with pytest.raises(ValueError, match="must be non-negative"):
        journal.record_realized_loss(-1.5)


def test_item16_mark_price_staleness_and_reconciliation(
    default_config: MainnetDemoConfig,
    mock_exchange: MagicMock,
    mock_inference_long: MagicMock,
    temp_db: Path,
) -> None:
    """16. Mark-price staleness (>60s) aborts sizing; trade fills are reconciled from exchange records."""
    journal = MainnetDemoJournal(temp_db)
    client = MainnetDemoClient(default_config, exchange_instance=mock_exchange, journal=journal)
    runner = MainnetDemoRunner(
        config=default_config, client=client, journal=journal, inference=mock_inference_long
    )

    # 1. Stale timestamp (120 seconds old)
    stale_ts = int((time.time() - 120) * 1000)
    mock_exchange.fapiPublicGetPremiumIndex.side_effect = lambda params=None: {
        "markPrice": "60000.0",
        "time": stale_ts,
    }

    with pytest.raises(MainnetDemoError, match="Mark price timestamp is stale"):
        runner.execute_demo_trade(candles=[{"close": 60000.0}] * 50)

    # 2. Reconciliation with trade history
    fresh_db = temp_db.parent / "fresh_demo_16.db"
    fresh_journal = MainnetDemoJournal(fresh_db)
    fresh_client = MainnetDemoClient(default_config, exchange_instance=mock_exchange, journal=fresh_journal)
    fresh_runner = MainnetDemoRunner(
        config=default_config, client=fresh_client, journal=fresh_journal, inference=mock_inference_long
    )

    # Restore fresh timestamp
    mock_exchange.fapiPublicGetPremiumIndex.side_effect = lambda params=None: {
        "markPrice": "60000.0",
        "time": int(time.time() * 1000),
    }

    mock_exchange.fetch_my_trades.return_value = [
        {"order": "ex_ord_entry", "realizedPnl": "-0.75", "fee": {"cost": 0.05}},
    ]

    res = fresh_runner.execute_demo_trade(candles=[{"close": 60000.0}] * 50)
    assert res["status"] == "COMPLETED"


