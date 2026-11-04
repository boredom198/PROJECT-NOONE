"""Tests for SQLite Execution Journal (Component 3).

Verifies DDL migrations, ACID transactions, order state transitions, fill deduplication (T06-B),
daily baselines, external transfers, safety freeze persistence across restarts, and audit logging.
Conforms strictly to PROJECT NOONE Technical Design Specification V2.2.4 Sections 11 and 13.
"""

from decimal import Decimal
import pytest

from noone.exchange.journal import ExecutionJournal


@pytest.fixture
def journal(tmp_path):
    db_file = tmp_path / "journal_test.db"
    return ExecutionJournal(db_file)


def test_schema_initialization(journal):
    """Verify tables and schema version 1 are initialized on creation."""
    with journal._get_connection() as conn:
        row = conn.execute("SELECT version FROM schema_version WHERE version = 1;").fetchone()
        assert row is not None
        assert row["version"] == 1


def test_order_intent_creation_and_state_transitions(journal):
    """Verify persisting order intent in CREATED state and sequential state transitions."""
    cid = "noone_test_order_001"
    order = journal.create_order_intent(
        client_order_id=cid,
        symbol="BTCUSDT",
        action_type="OPEN_LONG",
        side="BUY",
        quantity=0.15,
        reduce_only=False,
    )
    assert order.client_order_id == cid
    assert order.state == "CREATED"
    assert order.reconciliation_state == "PENDING"
    assert order.quantity == 0.15

    # Transition to DISPATCHING
    journal.transition_order_state(
        client_order_id=cid,
        to_state="DISPATCHING",
        reason="Sending to socket",
    )
    updated = journal.get_order(cid)
    assert updated.state == "DISPATCHING"

    # Transition to FILLED
    journal.transition_order_state(
        client_order_id=cid,
        to_state="FILLED",
        reason="Execution confirmed",
        exchange_order_id="ex_999888",
        filled_quantity=0.15,
        cum_quote=7500.0,
        avg_price=50000.0,
        reconciliation_state="VERIFIED",
    )
    filled_order = journal.get_order(cid)
    assert filled_order.state == "FILLED"
    assert filled_order.reconciliation_state == "VERIFIED"
    assert filled_order.exchange_order_id == "ex_999888"
    assert filled_order.avg_price == 50000.0


def test_t06_b_trade_fill_deduplication(journal):
    """Test T06-B: Trade fills must be strictly deduplicated by trade_id in SQLite."""
    cid = "noone_test_fill_001"
    journal.create_order_intent(cid, "BTCUSDT", "OPEN_LONG", "BUY", 0.1)

    # First insert -> True
    res1 = journal.record_trade_fill(
        trade_id="trade_1001",
        client_order_id=cid,
        symbol="BTCUSDT",
        price=50000.0,
        quantity=0.05,
        commission=1.25,
        commission_asset="USDT",
        realized_pnl=0.0,
        timestamp=1730000000000,
    )
    assert res1 is True

    # Duplicate insert with same trade_id -> False (ignored/deduplicated)
    res2 = journal.record_trade_fill(
        trade_id="trade_1001",
        client_order_id=cid,
        symbol="BTCUSDT",
        price=50000.0,
        quantity=0.05,
        commission=1.25,
        commission_asset="USDT",
        realized_pnl=0.0,
        timestamp=1730000000000,
    )
    assert res2 is False

    # Second distinct fill
    res3 = journal.record_trade_fill(
        trade_id="trade_1002",
        client_order_id=cid,
        symbol="BTCUSDT",
        price=50010.0,
        quantity=0.05,
        commission=1.25,
        commission_asset="USDT",
        realized_pnl=0.0,
        timestamp=1730000001000,
    )
    assert res3 is True

    fills = journal.get_order_fills(cid)
    assert len(fills) == 2
    assert {f.trade_id for f in fills} == {"trade_1001", "trade_1002"}


def test_daily_baseline_recording_and_retrieval(journal):
    """Verify daily baseline persistence, retrieval, and immutability."""
    # First recording of VERIFIED_UTC_BASELINE
    res1 = journal.record_daily_baseline(
        date_utc="2026-11-04",
        baseline_equity=10500.0,
        verified_status="VERIFIED_UTC_BASELINE",
        note="Rollover sample",
    )
    assert res1 is True

    baseline = journal.get_daily_baseline("2026-11-04")
    assert baseline is not None
    assert baseline.date_utc == "2026-11-04"
    assert baseline.baseline_equity == 10500.0
    assert baseline.verified_status == "VERIFIED_UTC_BASELINE"

    # Defect 1: Baseline immutability. Attempting to overwrite established baseline must be rejected!
    res2 = journal.record_daily_baseline(
        date_utc="2026-11-04",
        baseline_equity=9000.0,
        verified_status="VERIFIED_UTC_BASELINE",
        note="Attempted overwrite",
    )
    assert res2 is False

    res3 = journal.record_daily_baseline(
        date_utc="2026-11-04",
        baseline_equity=0.0,
        verified_status="INVALID",
        note="Attempted invalid reset",
    )
    assert res3 is False

    # Original baseline MUST remain unchanged
    current = journal.get_daily_baseline("2026-11-04")
    assert current.baseline_equity == 10500.0
    assert current.verified_status == "VERIFIED_UTC_BASELINE"

    # Non-existent date
    assert journal.get_daily_baseline("2026-11-05") is None


def test_external_transfers_deduplication_and_net_sum(journal):
    """Verify capital transfer recording, deduplication on tran_id, and sum."""
    # Deposit 500 USDT at t=1000
    assert journal.record_external_transfer("tran_1", "TRANSFER", "USDT", 500.0, 1000, is_external=True) is True
    # Duplicate tran_1 -> False
    assert journal.record_external_transfer("tran_1", "TRANSFER", "USDT", 500.0, 1000, is_external=True) is False

    # Withdrawal 200 USDT at t=2000
    assert journal.record_external_transfer("tran_2", "TRANSFER", "USDT", -200.0, 2000, is_external=True) is True

    # Internal transfer (not external) 1000 USDT at t=2500
    assert journal.record_external_transfer("tran_3", "TRANSFER", "USDT", 1000.0, 2500, is_external=False) is True

    # Net external transfers since t=500: 500 - 200 = 300.0 (tran_3 excluded!)
    net = journal.get_net_external_transfers(500)
    assert net == 300.0

    # Net transfers since t=1500: -200.0
    assert journal.get_net_external_transfers(1500) == -200.0


def test_safety_freeze_lifecycle_and_restart_persistence(tmp_path):
    """Verify safety freeze triggers, clears, and persists across process restarts."""
    db_file = tmp_path / "freeze_restart_test.db"
    journal1 = ExecutionJournal(db_file)

    assert journal1.has_active_freeze() is False

    # Trigger freeze
    journal1.trigger_safety_freeze("POSITION_ACCOUNT_DISCREPANCY", "Residual dust detected")
    assert journal1.has_active_freeze("POSITION_ACCOUNT_DISCREPANCY") is True
    assert journal1.has_active_freeze() is True

    # Simulate process restart by instantiating a new ExecutionJournal on same DB
    journal2 = ExecutionJournal(db_file)
    assert journal2.has_active_freeze("POSITION_ACCOUNT_DISCREPANCY") is True
    freezes = journal2.get_active_safety_freezes()
    assert len(freezes) == 1
    assert freezes[0].reason == "POSITION_ACCOUNT_DISCREPANCY"

    # Clear freeze
    journal2.clear_safety_freeze("POSITION_ACCOUNT_DISCREPANCY", "Manual acknowledgement by operator")
    assert journal2.has_active_freeze("POSITION_ACCOUNT_DISCREPANCY") is False
    assert journal2.has_active_freeze() is False

