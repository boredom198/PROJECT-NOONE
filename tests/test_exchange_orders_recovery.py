"""Tests for order recovery and safety freeze manager (Component 3).

Verifies network failure handling (T02), filled order recovery (T03-A),
active NEW order cancellation (T03-B), partial fill recovery (T03-C),
cancellation timeout (T03-D), canceled order recovery (T04),
inconclusive OrderNotFound handling (T05), and startup orphan recovery (T06-A).
Conforms strictly to PROJECT NOONE Technical Design Specification V2.2.4 Sections 6 and 11.
"""

from unittest.mock import MagicMock, patch
import pytest

from noone.exchange.client import BinanceFuturesClient
from noone.exchange.config import Component3Config
from noone.exchange.errors import OrderRecoveryError
from noone.exchange.journal import ExecutionJournal
from noone.exchange.reconciliation import ReconciliationManager


@pytest.fixture
def recovery_env(tmp_path):
    config = Component3Config(
        testnet=True,
        symbol="BTCUSDT",
        database_path=tmp_path / "recovery_test.db",
    )
    journal = ExecutionJournal(config.database_path)
    mock_client = MagicMock(spec=BinanceFuturesClient)
    mock_client.config = config
    mock_client.fetch_my_trades_paginated.side_effect = (
        lambda *args, **kwargs: mock_client.fetch_my_trades.return_value
        if isinstance(mock_client.fetch_my_trades.return_value, list)
        else []
    )
    mgr = ReconciliationManager(config, mock_client, journal)
    return config, journal, mock_client, mgr


def test_t03_a_filled_order_recovery(recovery_env):
    """Test T03-A: Fully filled order recovery through 9-step sequence."""
    _, journal, mock_client, mgr = recovery_env
    cid = "noone_fill_recovery_001"
    journal.create_order_intent(cid, "BTCUSDT", "OPEN_LONG", "BUY", 0.1)
    journal.transition_order_state(cid, "DISPATCHING")

    # Mock exchange responses
    mock_client.fetch_order_by_client_id.return_value = {
        "id": "ex_111222",
        "symbol": "BTC/USDT:USDT",
        "status": "FILLED",
        "filled": 0.1,
        "cost": 5000.0,
        "average": 50000.0,
    }
    mock_client.fetch_my_trades.return_value = [
        {
            "id": "tr_1",
            "order": "ex_111222",
            "price": 50000.0,
            "amount": 0.1,
            "fee": {"cost": 2.5, "currency": "USDT"},
            "info": {"realizedPnl": "0.0", "clientOrderId": cid},
            "timestamp": 1730000000000,
        }
    ]
    mock_client.fetch_position_risk_v2.return_value = [{"symbol": "BTCUSDT", "positionAmt": "0.1"}]
    mock_client.fetch_account_v2.return_value = {"totalMarginBalance": "10000.0"}
    mock_client.fetch_open_orders.return_value = []

    outcome = mgr.recover_unknown_order(cid)

    assert outcome.is_authoritative is True
    assert outcome.final_order_state == "FILLED"
    assert outcome.reconciliation_state == "VERIFIED"
    assert outcome.filled_quantity == 0.1
    assert outcome.avg_price == 50000.0

    # Freeze should be cleared
    assert journal.has_active_freeze("UNKNOWN_ORDER_OUTCOME") is False

    # Order in journal should be updated to FILLED and VERIFIED
    stored = journal.get_order(cid)
    assert stored.state == "FILLED"
    assert stored.reconciliation_state == "VERIFIED"


def test_t03_b_active_new_order_cancellation(recovery_env):
    """Test T03-B: Active NEW order must be canceled using original clientOrderId."""
    _, journal, mock_client, mgr = recovery_env
    cid = "noone_new_cancel_001"
    journal.create_order_intent(cid, "BTCUSDT", "OPEN_LONG", "BUY", 0.2)
    journal.transition_order_state(cid, "DISPATCHING")

    # Order is resting as NEW, then verified CANCELED after cancel request
    mock_client.fetch_order_by_client_id.side_effect = [
        {"id": "ex_333444", "symbol": "BTC/USDT:USDT", "status": "NEW", "filled": 0.0, "cost": 0.0, "average": 0.0},
        {"id": "ex_333444", "symbol": "BTC/USDT:USDT", "status": "CANCELED", "filled": 0.0, "cost": 0.0, "average": 0.0},
    ]
    mock_client.fetch_my_trades.return_value = []
    mock_client.fetch_position_risk_v2.return_value = [{"symbol": "BTCUSDT", "positionAmt": "0.0"}]
    mock_client.fetch_account_v2.return_value = {"totalMarginBalance": "10000.0"}
    mock_client.fetch_open_orders.return_value = []
    mock_client.cancel_order_by_client_id.return_value = {"status": "CANCELED"}

    outcome = mgr.recover_unknown_order(cid)

    assert outcome.is_authoritative is True
    assert outcome.final_order_state == "CANCELED"
    assert outcome.reconciliation_state == "VERIFIED"

    # Verify cancellation was called with original clientOrderId
    mock_client.cancel_order_by_client_id.assert_called_once_with(cid)

    # Freeze cleared
    assert journal.has_active_freeze("UNKNOWN_ORDER_OUTCOME") is False


def test_t03_c_partially_filled_order_cancellation_and_fill_recovery(recovery_env):
    """Test T03-C: Partially filled active order cancellation and partial fill capture."""
    _, journal, mock_client, mgr = recovery_env
    cid = "noone_partial_fill_001"
    journal.create_order_intent(cid, "BTCUSDT", "OPEN_LONG", "BUY", 0.5)
    journal.transition_order_state(cid, "DISPATCHING")

    # 0.2 of 0.5 filled, then verified CANCELED on requery
    mock_client.fetch_order_by_client_id.side_effect = [
        {"id": "ex_555666", "symbol": "BTC/USDT:USDT", "status": "PARTIALLY_FILLED", "filled": 0.2, "cost": 10000.0, "average": 50000.0},
        {"id": "ex_555666", "symbol": "BTC/USDT:USDT", "status": "CANCELED", "filled": 0.2, "cost": 10000.0, "average": 50000.0},
    ]
    mock_client.fetch_my_trades.return_value = [
        {
            "id": "tr_partial_1",
            "order": "ex_555666",
            "price": 50000.0,
            "amount": 0.2,
            "fee": {"cost": 5.0, "currency": "USDT"},
            "info": {"realizedPnl": "0.0", "clientOrderId": cid},
            "timestamp": 1730000000000,
        }
    ]
    mock_client.fetch_position_risk_v2.return_value = [{"symbol": "BTCUSDT", "positionAmt": "0.2"}]
    mock_client.fetch_account_v2.return_value = {"totalMarginBalance": "10000.0"}
    mock_client.fetch_open_orders.return_value = []
    mock_client.cancel_order_by_client_id.return_value = {"status": "CANCELED"}

    outcome = mgr.recover_unknown_order(cid)

    assert outcome.is_authoritative is True
    assert outcome.final_order_state == "PARTIALLY_FILLED_CANCELED"
    assert outcome.filled_quantity == 0.2
    assert journal.has_active_freeze("PARTIAL_FILL_UNRESOLVED") is True

    # Verify partial fill recorded in SQLite
    fills = journal.get_order_fills(cid)
    assert len(fills) == 1
    assert fills[0].quantity == 0.2


def test_t03_d_cancellation_failure_retains_freeze(recovery_env):
    """Test T03-D: When cancellation of an active order fails, retain UNKNOWN and safety freeze."""
    _, journal, mock_client, mgr = recovery_env
    cid = "noone_cancel_fail_001"
    journal.create_order_intent(cid, "BTCUSDT", "OPEN_LONG", "BUY", 0.1)

    mock_client.fetch_order_by_client_id.return_value = {
        "id": "ex_777888",
        "status": "NEW",
        "filled": 0.0,
    }
    mock_client.cancel_order_by_client_id.side_effect = Exception("Exchange 504 Gateway Timeout on cancel")

    outcome = mgr.recover_unknown_order(cid)

    assert outcome.is_authoritative is False
    assert journal.has_active_freeze("UNKNOWN_ORDER_OUTCOME") is True


def test_t04_canceled_order_recovery(recovery_env):
    """Test T04: Canceled terminal order recovery."""
    _, journal, mock_client, mgr = recovery_env
    cid = "noone_canceled_terminal_001"
    journal.create_order_intent(cid, "BTCUSDT", "OPEN_LONG", "BUY", 0.1)

    mock_client.fetch_order_by_client_id.return_value = {
        "id": "ex_888999",
        "status": "CANCELED",
        "filled": 0.0,
        "cost": 0.0,
        "average": 0.0,
    }
    mock_client.fetch_my_trades.return_value = []
    mock_client.fetch_position_risk_v2.return_value = [{"symbol": "BTCUSDT", "positionAmt": "0.0"}]
    mock_client.fetch_account_v2.return_value = {"totalMarginBalance": "10000.0"}
    mock_client.fetch_open_orders.return_value = []

    outcome = mgr.recover_unknown_order(cid)

    assert outcome.is_authoritative is True
    assert outcome.final_order_state == "CANCELED"
    assert journal.has_active_freeze("UNKNOWN_ORDER_OUTCOME") is False


def test_t05_inconclusive_order_not_found(recovery_env):
    """Test T05: OrderNotFound (-2013) must retain UNKNOWN state, keep freeze, and never auto-resubmit."""
    _, journal, mock_client, mgr = recovery_env
    cid = "noone_not_found_001"
    journal.create_order_intent(cid, "BTCUSDT", "OPEN_LONG", "BUY", 0.1)
    journal.transition_order_state(cid, "DISPATCHING")

    # CCXT raises OrderRecoveryError("Order not found")
    mock_client.fetch_order_by_client_id.side_effect = OrderRecoveryError("Order not found on exchange.")

    outcome = mgr.recover_unknown_order(cid)

    assert outcome.is_authoritative is False
    assert outcome.final_order_state == "UNKNOWN"
    assert outcome.reconciliation_state == "DISCREPANCY"

    # Freeze MUST remain active
    assert journal.has_active_freeze("UNKNOWN_ORDER_OUTCOME") is True

    # Order must remain UNKNOWN in SQLite
    order = journal.get_order(cid)
    assert order.state in ("DISPATCHING", "UNKNOWN")


def test_t06_a_startup_orphan_order_recovery(recovery_env):
    """Test T06-A: Startup orphan order recovery from prior crashed session."""
    _, journal, mock_client, mgr = recovery_env
    cid = "noone_orphan_startup_001"
    journal.create_order_intent(cid, "BTCUSDT", "OPEN_LONG", "BUY", 0.3)
    journal.transition_order_state(cid, "DISPATCHING")

    unresolved = journal.get_unresolved_orders()
    assert len(unresolved) == 1
    assert unresolved[0].client_order_id == cid

    # Exchange confirms it was filled while agent was offline, with matching trade fill
    mock_client.fetch_order_by_client_id.return_value = {
        "id": "ex_orphan_1",
        "status": "FILLED",
        "filled": 0.3,
        "cost": 15000.0,
        "average": 50000.0,
    }
    mock_client.fetch_my_trades.return_value = [
        {
            "id": "tr_orphan_1",
            "order": "ex_orphan_1",
            "price": 50000.0,
            "amount": 0.3,
            "fee": {"cost": 7.5, "currency": "USDT"},
            "info": {"realizedPnl": "0.0", "clientOrderId": cid},
            "timestamp": 1730000000000,
        }
    ]
    mock_client.fetch_position_risk_v2.return_value = [{"symbol": "BTCUSDT", "positionAmt": "0.3"}]
    mock_client.fetch_account_v2.return_value = {"totalMarginBalance": "10000.0"}
    mock_client.fetch_open_orders.return_value = []

    outcome = mgr.recover_unknown_order(cid)
    assert outcome.is_authoritative is True
    assert outcome.final_order_state == "FILLED"

    # No unresolved orders should remain
    assert len(journal.get_unresolved_orders()) == 0


def test_cancel_acknowledgement_without_verified_closure_fails_closed(recovery_env):
    """Defect 2: Cancellation ack alone is insufficient; if order is still active on requery, retain freeze."""
    _, journal, mock_client, mgr = recovery_env
    cid = "noone_cancel_stuck_001"
    journal.create_order_intent(cid, "BTCUSDT", "OPEN_LONG", "BUY", 0.1)

    # Requery after cancel returns NEW (stuck in matching engine)
    mock_client.fetch_order_by_client_id.side_effect = [
        {"id": "ex_stuck", "status": "NEW", "filled": 0.0},
        {"id": "ex_stuck", "status": "NEW", "filled": 0.0},
    ]
    mock_client.cancel_order_by_client_id.return_value = {"status": "CANCELED"}

    outcome = mgr.recover_unknown_order(cid)
    assert outcome.is_authoritative is False
    assert journal.has_active_freeze("UNKNOWN_ORDER_OUTCOME") is True


def test_recovery_fails_closed_when_fills_incomplete(recovery_env):
    """Defect 2: Incomplete or contradictory fills evidence must retain freeze."""
    _, journal, mock_client, mgr = recovery_env
    cid = "noone_fill_mismatch_001"
    journal.create_order_intent(cid, "BTCUSDT", "OPEN_LONG", "BUY", 0.2)

    # Order says filled 0.2, but trade fill says amount 0.1
    mock_client.fetch_order_by_client_id.return_value = {
        "id": "ex_mismatch", "status": "FILLED", "filled": 0.2, "cost": 10000.0, "average": 50000.0
    }
    mock_client.fetch_my_trades.return_value = [
        {
            "id": "tr_mis", "order": "ex_mismatch", "price": 50000.0, "amount": 0.1,
            "fee": {"cost": 2.5, "currency": "USDT"}, "timestamp": 1730000000000,
        }
    ]

    outcome = mgr.recover_unknown_order(cid)
    assert outcome.is_authoritative is False
    assert outcome.reconciliation_state == "DISCREPANCY"
    assert journal.has_active_freeze("UNKNOWN_ORDER_OUTCOME") is True


def test_recovery_fails_closed_when_position_query_fails(recovery_env):
    """Defect 2: Failure in Step 5 (position query) must halt recovery and retain freeze."""
    _, journal, mock_client, mgr = recovery_env
    cid = "noone_pos_fail_001"
    journal.create_order_intent(cid, "BTCUSDT", "OPEN_LONG", "BUY", 0.1)

    mock_client.fetch_order_by_client_id.return_value = {
        "id": "ex_pos_fail", "status": "CANCELED", "filled": 0.0, "cost": 0.0, "average": 0.0
    }
    mock_client.fetch_my_trades.return_value = []
    mock_client.fetch_position_risk_v2.side_effect = Exception("500 Internal Server Error on positionRisk")

    outcome = mgr.recover_unknown_order(cid)
    assert outcome.is_authoritative is False
    assert journal.has_active_freeze("UNKNOWN_ORDER_OUTCOME") is True


def test_recovery_multiple_uncertain_orders_retains_freeze_if_unresolved_remain(recovery_env):
    """Defect 4: In multi-uncertain scenario, recovering order A must NOT clear freeze for uncertain order B."""
    _, journal, mock_client, mgr = recovery_env
    cid_a = "noone_order_a"
    cid_b = "noone_order_b"

    journal.create_order_intent(cid_a, "BTCUSDT", "OPEN_LONG", "BUY", 0.1)
    journal.transition_order_state(cid_a, "DISPATCHING")
    journal.trigger_safety_freeze("UNKNOWN_ORDER_OUTCOME", f"Order {cid_a} dropped", client_order_id=cid_a)

    journal.create_order_intent(cid_b, "BTCUSDT", "OPEN_LONG", "BUY", 0.1)
    journal.transition_order_state(cid_b, "DISPATCHING")
    journal.trigger_safety_freeze("UNKNOWN_ORDER_OUTCOME", f"Order {cid_b} dropped", client_order_id=cid_b)

    # Recover order A successfully
    mock_client.fetch_order_by_client_id.return_value = {
        "id": "ex_a_123", "symbol": "BTC/USDT:USDT", "status": "FILLED", "filled": 0.1, "cost": 5000.0, "average": 50000.0
    }
    mock_client.fetch_my_trades.return_value = [
        {"id": "tr_a_1", "order": "ex_a_123", "price": 50000.0, "amount": 0.1, "fee": {"cost": 2.5, "currency": "USDT"}, "timestamp": 1730000000000}
    ]
    mock_client.fetch_position_risk_v2.return_value = [{"symbol": "BTCUSDT", "positionAmt": "0.1"}]
    mock_client.fetch_account_v2.return_value = {"totalMarginBalance": "10000.0"}
    mock_client.fetch_open_orders.return_value = []

    outcome_a = mgr.recover_unknown_order(cid_a)
    assert outcome_a.is_authoritative is True
    assert outcome_a.final_order_state == "FILLED"

    # Freeze for order A is cleared, BUT freeze for order B remains active!
    active_freezes = journal.get_active_safety_freezes()
    assert any(f.reason == "UNKNOWN_ORDER_OUTCOME" and f.client_order_id == cid_b for f in active_freezes)

    can_enter, reason = mgr.can_execute_entry()
    assert can_enter is False
    assert "Unresolved orders exist" in str(reason) or "UNKNOWN_ORDER_OUTCOME" in str(reason)


def test_recovery_fails_closed_when_position_delta_contradicts_fills(recovery_env):
    """Defect 4: Position delta check fails closed when exchange position disagrees with fill amount."""
    _, journal, mock_client, mgr = recovery_env
    cid = "noone_pos_mismatch_001"
    journal.create_order_intent(cid, "BTCUSDT", "OPEN_LONG", "BUY", 0.2)

    # Order says filled 0.2, trades say filled 0.2, but position is 0.0 (contradiction for an entry)
    mock_client.fetch_order_by_client_id.return_value = {
        "id": "ex_pos_mis", "status": "FILLED", "filled": 0.2, "cost": 10000.0, "average": 50000.0
    }
    mock_client.fetch_my_trades.return_value = [
        {"id": "tr_pm_1", "order": "ex_pos_mis", "price": 50000.0, "amount": 0.2, "fee": {"cost": 5.0, "currency": "USDT"}, "timestamp": 1730000000000}
    ]
    mock_client.fetch_position_risk_v2.return_value = [{"symbol": "BTCUSDT", "positionAmt": "0.0"}]
    mock_client.fetch_account_v2.return_value = {"totalMarginBalance": "10000.0"}
    mock_client.fetch_open_orders.return_value = []

    outcome = mgr.recover_unknown_order(cid)
    assert outcome.is_authoritative is False
    assert outcome.reconciliation_state == "DISCREPANCY"
    assert journal.has_active_freeze("UNKNOWN_ORDER_OUTCOME") is True

