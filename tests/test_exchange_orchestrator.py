"""Tests for ExchangeOrchestrator runtime pipeline (Component 3).

Verifies the pre-flight handshake, end-to-end DQN Action execution (HOLD, OPEN_LONG,
CLOSE_LONG, OPEN_SHORT, CLOSE_SHORT), intent persistence before dispatch,
network failure isolation (T02), session realized PnL tracking, UTC midnight baseline sampling (Defect 1),
transfer sync integrity (Defect 3), residual dust safety (Defect 4), leverage enforcement (Defect 5),
maxNotional filters (Defect 6), pre-dispatch freshness (Defect 7), partial fills (Defect 8),
and hard production lock (Defect 9).
Conforms strictly to PROJECT NOONE Technical Design Specification V2.2.4.
"""

from datetime import datetime, timezone
from decimal import Decimal
import time
from unittest.mock import MagicMock, patch
import pytest

from noone.rl_engine.inference import Action
from noone.exchange.client import BinanceFuturesClient
from noone.exchange.config import Component3Config
from noone.exchange.errors import (
    AccountSyncError,
    DailyLossLimitError,
    DailyPnLUninitializedError,
    ExchangeError,
    InvalidBaselineError,
    ResidualPositionError,
    RiskGateError,
    SafetyFreezeError,
    StateDiscrepancyError,
    StaleMarketDataError,
)
from noone.exchange.journal import ExecutionJournal
from noone.exchange.orchestrator import ExchangeOrchestrator


@pytest.fixture
def orchestrator_env(tmp_path):
    config = Component3Config(
        testnet=True,
        symbol="BTCUSDT",
        leverage=10,
        margin_allocation_fraction=0.10,
        max_daily_loss_pct=0.03,
        database_path=tmp_path / "orchestrator_test.db",
    )
    journal = ExecutionJournal(config.database_path)
    mock_client = MagicMock(spec=BinanceFuturesClient)
    mock_client.config = config

    # Preload mock exchange info with minNotional & maxNotional
    mock_client.fetch_exchange_info.return_value = {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "filters": [
                    {
                        "filterType": "MARKET_LOT_SIZE",
                        "minQty": "0.001",
                        "maxQty": "100.0",
                        "stepSize": "0.001",
                    },
                    {
                        "filterType": "NOTIONAL",
                        "minNotional": "10.0",
                        "maxNotional": "100000.0",
                        "applyMinToMarket": True,
                        "applyMaxToMarket": True,
                    },
                ],
            }
        ]
    }

    # Preload mock leverage bracket
    mock_client.fetch_leverage_bracket.return_value = [
        {
            "symbol": "BTCUSDT",
            "brackets": [
                {
                    "bracket": 1,
                    "initialLeverage": 10,
                    "notionalCap": 50000.0,
                    "notionalFloor": 0.0,
                }
            ],
        }
    ]

    # Preload clean empty transfers
    mock_client.fetch_income_paginated.return_value = []
    mock_client.fetch_income.return_value = []
    mock_client.fetch_mark_price.return_value = 50000.0
    mock_client.fetch_mark_price_data.side_effect = lambda sym=None: (50000.0, int(time.time() * 1000))

    orch = ExchangeOrchestrator(config, client=mock_client, journal=journal)
    return config, journal, mock_client, orch


def test_pre_flight_handshake_success(orchestrator_env):
    """Verify pre-flight handshake verifies all conditions."""
    _, journal, mock_client, orch = orchestrator_env

    mock_client.verify_clock_drift.return_value = 120
    mock_client.fetch_account_v2.return_value = {"canTrade": True, "totalMarginBalance": "10000.0"}
    mock_client.fetch_position_side_dual.return_value = {"dualSidePosition": False}
    mock_client.fetch_position_risk_v2.return_value = [
        {"symbol": "BTCUSDT", "positionAmt": "0.0", "positionSide": "BOTH", "leverage": "10"}
    ]
    mock_client.fetch_open_orders.return_value = []

    res = orch.pre_flight_handshake()
    assert res["clock_drift_ms"] == 120
    assert res["can_trade"] is True
    assert res["one_way_mode"] is True
    assert res["filters_loaded"] is True
    assert res["bracket_max_notional"] == 50000.0
    assert res["zero_external_positions"] is True
    assert res["zero_open_orders"] is True


def test_pre_flight_handshake_rejects_hedge_mode(orchestrator_env):
    """Verify pre-flight handshake rejects dual-side Hedge mode and triggers freeze."""
    _, journal, mock_client, orch = orchestrator_env

    mock_client.verify_clock_drift.return_value = 50
    mock_client.fetch_account_v2.return_value = {"canTrade": True}
    mock_client.fetch_position_side_dual.return_value = {"dualSidePosition": True}

    with pytest.raises(StateDiscrepancyError) as exc:
        orch.pre_flight_handshake()
    assert "Hedge Mode" in str(exc.value)
    assert journal.has_active_freeze("AMBIGUOUS_POSITION_STATE") is True


def test_execute_action_hold_produces_no_orders(orchestrator_env):
    """Verify Action.HOLD produces zero exchange orders."""
    _, _, mock_client, orch = orchestrator_env
    res = orch.execute_action(Action.HOLD)
    assert res is None
    mock_client.create_market_order.assert_not_called()


def test_execute_action_open_long_lifecycle(orchestrator_env):
    """Verify Action.OPEN_LONG commits intent CREATED before socket dispatch and updates state."""
    _, journal, mock_client, orch = orchestrator_env

    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    journal.record_daily_baseline(today_utc, 10000.0, "VERIFIED_UTC_BASELINE")

    mock_client.fetch_mark_price.return_value = 50000.0
    mock_client.fetch_account_v2.return_value = {
        "totalMarginBalance": "10000.00",
        "assets": [{"asset": "USDT", "availableBalance": "9000.00"}],
        "positions": [{"symbol": "BTCUSDT", "positionInitialMargin": "0.00"}],
    }
    mock_client.fetch_position_risk_v2.return_value = [
        {"symbol": "BTCUSDT", "positionAmt": "0.00", "positionSide": "BOTH", "leverage": "10"}
    ]
    mock_client.fetch_open_orders.return_value = []
    mock_client.create_market_order.return_value = {
        "id": "ex_long_123",
        "status": "FILLED",
        "filled": 0.2,
        "cost": 10000.0,
        "average": 50000.0,
    }

    order = orch.execute_action(Action.OPEN_LONG, mark_price=50000.0, mark_price_time_ms=int(time.time() * 1000))

    assert order is not None
    assert order.action_type == "OPEN_LONG"
    assert order.side == "BUY"
    assert order.state == "FILLED"
    assert order.reconciliation_state == "VERIFIED"
    assert order.filled_quantity == 0.2
    assert order.avg_price == 50000.0

    with journal.connection() as conn:
        transitions = conn.execute(
            "SELECT to_state FROM order_state_transitions WHERE client_order_id = ? ORDER BY id ASC;",
            (order.client_order_id,),
        ).fetchall()
        states = [t["to_state"] for t in transitions]
        assert states == ["CREATED", "DISPATCHING", "FILLED"]


def test_t02_network_failure_during_dispatch_transitions_to_unknown(orchestrator_env):
    """Test T02: Socket timeout transitions order to UNKNOWN and triggers freeze without resubmission."""
    _, journal, mock_client, orch = orchestrator_env

    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    journal.record_daily_baseline(today_utc, 10000.0, "VERIFIED_UTC_BASELINE")

    mock_client.fetch_mark_price.return_value = 50000.0
    mock_client.fetch_account_v2.return_value = {
        "totalMarginBalance": "10000.00",
        "assets": [{"asset": "USDT", "availableBalance": "9000.00"}],
        "positions": [{"symbol": "BTCUSDT", "positionInitialMargin": "0.00"}],
    }
    mock_client.fetch_position_risk_v2.return_value = [
        {"symbol": "BTCUSDT", "positionAmt": "0.00", "positionSide": "BOTH", "leverage": "10"}
    ]
    mock_client.fetch_open_orders.return_value = []
    mock_client.create_market_order.side_effect = Exception("ReadTimeout: Network dropped")

    with pytest.raises(ExchangeError) as exc:
        orch.execute_action(Action.OPEN_LONG, mark_price=50000.0, mark_price_time_ms=int(time.time() * 1000))

    assert "Order dispatch failed" in str(exc.value)
    assert journal.has_active_freeze("UNKNOWN_ORDER_OUTCOME") is True

    unresolved = journal.get_unresolved_orders()
    assert len(unresolved) == 1
    assert unresolved[0].state == "UNKNOWN"
    assert unresolved[0].reconciliation_state == "DISCREPANCY"


def test_execute_action_close_long_and_session_realized_pnl(orchestrator_env):
    """Verify CLOSE_LONG issues reduce-only order and tracks session realized PnL."""
    _, journal, mock_client, orch = orchestrator_env

    mock_client.fetch_account_v2.return_value = {
        "totalMarginBalance": "10200.00",
        "assets": [{"asset": "USDT", "availableBalance": "9200.00"}],
        "positions": [{"symbol": "BTCUSDT", "positionInitialMargin": "1000.00"}],
    }
    mock_client.fetch_position_risk_v2.return_value = [
        {
            "symbol": "BTCUSDT",
            "positionAmt": "0.200",
            "entryPrice": "50000.00",
            "unRealizedProfit": "200.00",
            "positionSide": "BOTH",
            "leverage": "10",
        }
    ]
    mock_client.create_market_order.return_value = {
        "id": "ex_close_long_1",
        "status": "FILLED",
        "filled": 0.2,
        "cost": 10200.0,
        "average": 51000.0,
    }

    assert orch.session_realized_pnl == 0.0

    order = orch.execute_action(Action.CLOSE_LONG, mark_price=51000.0, mark_price_time_ms=int(time.time() * 1000))

    assert order is not None
    assert order.action_type == "CLOSE_LONG"
    assert order.side == "SELL"
    assert order.reduce_only is True
    assert order.state == "FILLED"
    assert orch.session_realized_pnl == pytest.approx(200.0)


def test_execute_action_open_and_close_short(orchestrator_env):
    """Verify OPEN_SHORT and CLOSE_SHORT execution and realized PnL tracking."""
    _, journal, mock_client, orch = orchestrator_env

    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    journal.record_daily_baseline(today_utc, 10000.0, "VERIFIED_UTC_BASELINE")

    mock_client.fetch_mark_price.return_value = 50000.0
    mock_client.fetch_account_v2.return_value = {
        "totalMarginBalance": "10000.00",
        "assets": [{"asset": "USDT", "availableBalance": "9000.00"}],
        "positions": [{"symbol": "BTCUSDT", "positionInitialMargin": "0.00"}],
    }
    mock_client.fetch_position_risk_v2.return_value = [
        {"symbol": "BTCUSDT", "positionAmt": "0.00", "positionSide": "BOTH", "leverage": "10"}
    ]
    mock_client.fetch_open_orders.return_value = []
    mock_client.create_market_order.return_value = {
        "id": "ex_short_1",
        "status": "FILLED",
        "filled": 0.2,
        "cost": 10000.0,
        "average": 50000.0,
    }

    now_ts = int(time.time() * 1000)
    order_open = orch.execute_action(Action.OPEN_SHORT, mark_price=50000.0, mark_price_time_ms=now_ts)
    assert order_open.action_type == "OPEN_SHORT"
    assert order_open.side == "SELL"
    assert order_open.reduce_only is False

    mock_client.fetch_account_v2.return_value = {
        "totalMarginBalance": "10200.00",
        "assets": [{"asset": "USDT", "availableBalance": "9200.00"}],
        "positions": [{"symbol": "BTCUSDT", "positionInitialMargin": "1000.00"}],
    }
    mock_client.fetch_position_risk_v2.return_value = [
        {
            "symbol": "BTCUSDT",
            "positionAmt": "-0.200",
            "entryPrice": "50000.00",
            "unRealizedProfit": "200.00",
            "positionSide": "BOTH",
            "leverage": "10",
        }
    ]
    mock_client.create_market_order.return_value = {
        "id": "ex_close_short_1",
        "status": "FILLED",
        "filled": 0.2,
        "cost": 9800.0,
        "average": 49000.0,
    }

    order_close = orch.execute_action(Action.CLOSE_SHORT, mark_price=49000.0, mark_price_time_ms=now_ts)
    assert order_close.action_type == "CLOSE_SHORT"
    assert order_close.side == "BUY"
    assert order_close.reduce_only is True
    assert orch.session_realized_pnl == pytest.approx(200.0)


# =============================================================================
# DEFECT REGRESSION TESTS
# =============================================================================

def test_utc_midnight_baseline_clean_sampling(orchestrator_env):
    """Defect 1: Clean sampling within [00:00:00, 00:00:05] UTC registers VERIFIED_UTC_BASELINE."""
    _, journal, mock_client, orch = orchestrator_env

    mock_client.fetch_account_v2.return_value = {"totalMarginBalance": "10550.00"}

    # Mock datetime to 00:00:03 UTC
    mock_dt = datetime(2026, 11, 4, 0, 0, 3, tzinfo=timezone.utc)
    with patch("noone.exchange.orchestrator.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_dt
        mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

        baseline = orch.sample_utc_midnight_baseline()
        assert baseline is not None
        assert baseline.baseline_equity == 10550.0
        assert baseline.verified_status == "VERIFIED_UTC_BASELINE"


def test_baseline_sampling_outside_window_rejected(orchestrator_env):
    """Defect 1: Sampling outside [00:00:00, 00:00:05] must raise InvalidBaselineError and freeze."""
    _, journal, mock_client, orch = orchestrator_env

    # Mock datetime to 03:15:00 UTC (outside window)
    mock_dt = datetime(2026, 11, 4, 3, 15, 0, tzinfo=timezone.utc)
    with patch("noone.exchange.orchestrator.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_dt
        mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

        with pytest.raises(InvalidBaselineError) as exc:
            orch.sample_utc_midnight_baseline()
        assert "outside [00:00:00, 00:00:05] window" in str(exc.value)
        assert journal.has_active_freeze("DAILY_PNL_UNINITIALIZED") is True


def test_repeated_sampling_maintains_immutable_baseline(orchestrator_env):
    """Defect 1: Established baseline is immutable and cannot be overwritten."""
    _, journal, mock_client, orch = orchestrator_env

    mock_dt = datetime(2026, 11, 4, 0, 0, 2, tzinfo=timezone.utc)
    with patch("noone.exchange.orchestrator.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_dt
        mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

        mock_client.fetch_account_v2.return_value = {"totalMarginBalance": "10000.00"}
        b1 = orch.sample_utc_midnight_baseline()
        assert b1.baseline_equity == 10000.0

        # Attempt to sample again with different equity -> must return original 10000.0
        mock_client.fetch_account_v2.return_value = {"totalMarginBalance": "12000.00"}
        b2 = orch.sample_utc_midnight_baseline()
        assert b2.baseline_equity == 10000.0


def test_sampling_does_not_clear_active_daily_loss_freeze_same_day(orchestrator_env):
    """Defect 1: Sampling must never clear an active daily-loss freeze on the same day."""
    _, journal, mock_client, orch = orchestrator_env

    # Active freeze triggered today
    journal.trigger_safety_freeze("DAILY_LOSS_LIMIT", "Breached limit")
    assert journal.has_active_freeze("DAILY_LOSS_LIMIT") is True

    mock_dt = datetime(2026, 11, 4, 0, 0, 2, tzinfo=timezone.utc)
    with patch("noone.exchange.orchestrator.datetime") as mock_datetime:
        mock_datetime.now.return_value = mock_dt
        mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

        mock_client.fetch_account_v2.return_value = {"totalMarginBalance": "10000.00"}
        orch.sample_utc_midnight_baseline()

        # DAILY_LOSS_LIMIT MUST REMAIN ACTIVE!
        assert journal.has_active_freeze("DAILY_LOSS_LIMIT") is True


def test_residual_dust_exit_blocks_dispatch_and_halts(orchestrator_env):
    """Defect 4: Residual dust detection halts exit dispatch immediately without placing orders."""
    _, journal, mock_client, orch = orchestrator_env

    # Active dust position: 0.0005 BTC (below minQty 0.001)
    mock_client.fetch_account_v2.return_value = {
        "totalMarginBalance": "10000.00",
        "assets": [{"asset": "USDT", "availableBalance": "9000.00"}],
        "positions": [{"symbol": "BTCUSDT", "positionInitialMargin": "50.00"}],
    }
    mock_client.fetch_position_risk_v2.return_value = [
        {
            "symbol": "BTCUSDT",
            "positionAmt": "0.0005",
            "entryPrice": "50000.00",
            "unRealizedProfit": "0.00",
            "positionSide": "BOTH",
            "leverage": "10",
        }
    ]

    with pytest.raises(ResidualPositionError) as exc:
        orch.execute_action(Action.CLOSE_LONG, mark_price=50000.0, mark_price_time_ms=int(time.time() * 1000))

    assert "untradeable residual dust" in str(exc.value)
    # Order must NOT have been dispatched
    mock_client.create_market_order.assert_not_called()
    assert journal.has_active_freeze("POSITION_ACCOUNT_DISCREPANCY") is True


def test_leverage_mismatch_blocks_entry(orchestrator_env):
    """Defect 5: Effective exchange leverage not matching configured leverage blocks entry."""
    _, journal, mock_client, orch = orchestrator_env

    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    journal.record_daily_baseline(today_utc, 10000.0, "VERIFIED_UTC_BASELINE")

    mock_client.fetch_mark_price.return_value = 50000.0
    mock_client.fetch_account_v2.return_value = {
        "totalMarginBalance": "10000.00",
        "assets": [{"asset": "USDT", "availableBalance": "9000.00"}],
        "positions": [{"symbol": "BTCUSDT", "positionInitialMargin": "0.00"}],
    }
    # Exchange reports leverage = 20 (configured is 10)
    mock_client.fetch_position_risk_v2.return_value = [
        {"symbol": "BTCUSDT", "positionAmt": "0.00", "positionSide": "BOTH", "leverage": "20"}
    ]
    mock_client.fetch_open_orders.return_value = []

    with pytest.raises(RiskGateError) as exc:
        orch.execute_action(Action.OPEN_LONG, mark_price=50000.0, mark_price_time_ms=int(time.time() * 1000))
    assert "exchange leverage (20) does not match configured leverage (10)" in str(exc.value)


def test_partial_fill_handling_on_entry(orchestrator_env):
    """Defect 8: Partial fill on entry transitions to PARTIALLY_FILLED and triggers safety freeze."""
    _, journal, mock_client, orch = orchestrator_env

    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    journal.record_daily_baseline(today_utc, 10000.0, "VERIFIED_UTC_BASELINE")

    mock_client.fetch_mark_price.return_value = 50000.0
    mock_client.fetch_account_v2.return_value = {
        "totalMarginBalance": "10000.00",
        "assets": [{"asset": "USDT", "availableBalance": "9000.00"}],
        "positions": [{"symbol": "BTCUSDT", "positionInitialMargin": "0.00"}],
    }
    mock_client.fetch_position_risk_v2.return_value = [
        {"symbol": "BTCUSDT", "positionAmt": "0.00", "positionSide": "BOTH", "leverage": "10"}
    ]
    mock_client.fetch_open_orders.return_value = []

    # Exchange reports partial fill: 0.1 of 0.2
    mock_client.create_market_order.return_value = {
        "id": "ex_partial_entry",
        "status": "PARTIALLY_FILLED",
        "filled": 0.1,
        "cost": 5000.0,
        "average": 50000.0,
    }

    order = orch.execute_action(Action.OPEN_LONG, mark_price=50000.0, mark_price_time_ms=int(time.time() * 1000))
    assert order.state == "PARTIALLY_FILLED"
    assert order.filled_quantity == 0.1
    assert order.remaining_quantity == pytest.approx(0.1)
    assert journal.has_active_freeze("PARTIAL_FILL_UNRESOLVED") is True


def test_orchestrator_partial_fill_ccxt_response_blocks_subsequent_orders(orchestrator_env):
    """Item 3: Simulate partially filled CCXT order response, verify journal records correct quantities, and verify system blocks subsequent order."""
    _, journal, mock_client, orch = orchestrator_env

    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    journal.record_daily_baseline(today_utc, 10000.0, "VERIFIED_UTC_BASELINE")

    mock_client.fetch_mark_price.return_value = 50000.0
    mock_client.fetch_account_v2.return_value = {
        "totalMarginBalance": "10000.00",
        "assets": [{"asset": "USDT", "availableBalance": "9000.00"}],
        "positions": [{"symbol": "BTCUSDT", "positionInitialMargin": "0.00"}],
    }
    mock_client.fetch_position_risk_v2.return_value = [
        {"symbol": "BTCUSDT", "positionAmt": "0.00", "positionSide": "BOTH", "leverage": "10"}
    ]
    mock_client.fetch_open_orders.return_value = []

    # Actual CCXT unified order response for a partially filled market order
    # Total order amount was 0.20 BTC, only 0.08 filled, 0.12 remaining
    mock_client.create_market_order.return_value = {
        "id": "ccxt_order_part_001",
        "orderId": "ccxt_order_part_001",
        "clientOrderId": "noone_test_partial_client_id",
        "symbol": "BTC/USDT:USDT",
        "status": "PARTIALLY_FILLED",
        "amount": 0.20,
        "filled": 0.08,
        "remaining": 0.12,
        "cost": 4000.0,
        "average": 50000.0,
        "side": "buy",
        "type": "market",
    }

    now_ms = int(time.time() * 1000)
    order = orch.execute_action(Action.OPEN_LONG, mark_price=50000.0, mark_price_time_ms=now_ms)

    # Verify journal records the correct state and quantities
    assert order is not None
    assert order.state == "PARTIALLY_FILLED"
    assert order.filled_quantity == pytest.approx(0.08)
    assert order.remaining_quantity == pytest.approx(0.12)
    assert journal.has_active_freeze("PARTIAL_FILL_UNRESOLVED") is True

    # Verify that the order is also recorded as unresolved in the journal
    unresolved = journal.get_unresolved_orders()
    assert len(unresolved) == 1
    assert unresolved[0].client_order_id == order.client_order_id

    # Verify system does NOT blindly submit another order on subsequent action
    with pytest.raises(SafetyFreezeError) as exc_info:
        orch.execute_action(Action.OPEN_LONG, mark_price=50000.0, mark_price_time_ms=now_ms)
    assert "blocked by active safety freezes" in str(exc_info.value) or "unresolved" in str(exc_info.value)

    # Confirm create_market_order was called exactly once and was NOT called again
    assert mock_client.create_market_order.call_count == 1


def test_ambiguous_transfer_blocks_entry(orchestrator_env):
    """Defect 3: Ambiguous transfer in income history blocks entry and freezes daily PnL."""
    _, journal, mock_client, orch = orchestrator_env

    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    journal.record_daily_baseline(today_utc, 10000.0, "VERIFIED_UTC_BASELINE")

    # Income returns an ambiguous transfer (no isExternal flag)
    mock_client.fetch_income_paginated.return_value = [
        {"tranId": "ambig_001", "incomeType": "TRANSFER", "asset": "USDT", "income": "100.0", "time": 1730000000000}
    ]

    mock_client.fetch_mark_price.return_value = 50000.0
    mock_client.fetch_account_v2.return_value = {
        "totalMarginBalance": "10000.00",
        "assets": [{"asset": "USDT", "availableBalance": "9000.00"}],
        "positions": [{"symbol": "BTCUSDT", "positionInitialMargin": "0.00"}],
    }
    mock_client.fetch_position_risk_v2.return_value = [
        {"symbol": "BTCUSDT", "positionAmt": "0.00", "positionSide": "BOTH", "leverage": "10"}
    ]
    mock_client.fetch_open_orders.return_value = []

    with pytest.raises(DailyPnLUninitializedError) as exc:
        orch.execute_action(Action.OPEN_LONG, mark_price=50000.0, mark_price_time_ms=int(time.time() * 1000))
    assert "Ambiguous transfer origin detected" in str(exc.value)
    assert journal.has_active_freeze("DAILY_PNL_UNINITIALIZED") is True


def test_stale_pre_dispatch_data_blocks_entry(orchestrator_env):
    """Defect 7: Stale market data or snapshot blocks entry at the transport boundary."""
    _, journal, mock_client, orch = orchestrator_env

    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    journal.record_daily_baseline(today_utc, 10000.0, "VERIFIED_UTC_BASELINE")

    # Populate cache first
    orch._load_symbol_filters()
    # Simulate old cache for filters (> 1 hour) and failed refresh
    orch._filters_cached_at = time.time() - 4000.0
    mock_client.fetch_exchange_info.side_effect = Exception("exchangeInfo network refresh failed")

    mock_client.fetch_mark_price.return_value = 50000.0
    mock_client.fetch_account_v2.return_value = {
        "totalMarginBalance": "10000.00",
        "assets": [{"asset": "USDT", "availableBalance": "9000.00"}],
        "positions": [{"symbol": "BTCUSDT", "positionInitialMargin": "0.00"}],
    }
    mock_client.fetch_position_risk_v2.return_value = [
        {"symbol": "BTCUSDT", "positionAmt": "0.00", "positionSide": "BOTH", "leverage": "10"}
    ]
    mock_client.fetch_open_orders.return_value = []

    with pytest.raises(RiskGateError) as exc:
        orch.execute_action(Action.OPEN_LONG, mark_price=50000.0, mark_price_time_ms=int(time.time() * 1000))
    assert "filters data is stale" in str(exc.value)


def test_hard_production_lock_blocks_production_dispatch(tmp_path):
    """Defect 9: Hard runtime safety lock blocks production dispatch regardless of config."""
    # Attempting to initialize config with testnet=False must raise fatal ExchangeError
    with pytest.raises(ExchangeError) as exc:
        Component3Config(
            testnet=False,
            production_confirmed=True,
            database_path=tmp_path / "prod_test.db",
        )
    assert "PRODUCTION EXECUTION HARD LOCKED" in str(exc.value)


def test_caller_mark_price_missing_timestamp_rejected(orchestrator_env):
    """Defect 2 & 7: Caller-supplied mark_price without timestamp is rejected."""
    _, journal, mock_client, orch = orchestrator_env
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    journal.record_daily_baseline(today_utc, 10000.0, "VERIFIED_UTC_BASELINE")

    mock_client.fetch_account_v2.return_value = {
        "totalMarginBalance": "10000.00",
        "assets": [{"asset": "USDT", "availableBalance": "9000.00"}],
        "positions": [{"symbol": "BTCUSDT", "positionInitialMargin": "0.00"}],
    }
    mock_client.fetch_position_risk_v2.return_value = [
        {"symbol": "BTCUSDT", "positionAmt": "0.00", "positionSide": "BOTH", "leverage": "10"}
    ]
    mock_client.fetch_open_orders.return_value = []

    with pytest.raises(StaleMarketDataError) as exc:
        orch.execute_action(Action.OPEN_LONG, mark_price=50000.0, mark_price_time_ms=None)
    assert "requires explicit acquisition timestamp" in str(exc.value)


def test_caller_mark_price_stale_timestamp_rejected(orchestrator_env):
    """Defect 2 & 7: Caller-supplied mark_price with stale timestamp (> 2000ms) is rejected."""
    _, journal, mock_client, orch = orchestrator_env
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    journal.record_daily_baseline(today_utc, 10000.0, "VERIFIED_UTC_BASELINE")

    mock_client.fetch_account_v2.return_value = {
        "totalMarginBalance": "10000.00",
        "assets": [{"asset": "USDT", "availableBalance": "9000.00"}],
        "positions": [{"symbol": "BTCUSDT", "positionInitialMargin": "0.00"}],
    }
    mock_client.fetch_position_risk_v2.return_value = [
        {"symbol": "BTCUSDT", "positionAmt": "0.00", "positionSide": "BOTH", "leverage": "10"}
    ]
    mock_client.fetch_open_orders.return_value = []

    stale_ts = int((time.time() - 3.0) * 1000)
    with pytest.raises(StaleMarketDataError) as exc:
        orch.execute_action(Action.OPEN_LONG, mark_price=50000.0, mark_price_time_ms=stale_ts)
    assert "mark price data is stale" in str(exc.value)


def test_target_position_missing_fails_leverage_verification(orchestrator_env):
    """Defect 5 & 6: Missing target position in positionRisk endpoint fails closed."""
    _, journal, mock_client, orch = orchestrator_env
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    journal.record_daily_baseline(today_utc, 10000.0, "VERIFIED_UTC_BASELINE")

    mock_client.fetch_account_v2.return_value = {
        "totalMarginBalance": "10000.00",
        "assets": [{"asset": "USDT", "availableBalance": "9000.00"}],
        "positions": [{"symbol": "BTCUSDT", "positionInitialMargin": "0.00"}],
    }
    # First call (sync_account) has target; second call (fetch_position_risk_v2 in entry) is missing target
    mock_client.fetch_position_risk_v2.side_effect = [
        [{"symbol": "BTCUSDT", "positionAmt": "0.00", "positionSide": "BOTH", "leverage": "10"}],
        [{"symbol": "ETHUSDT", "positionAmt": "0.00", "positionSide": "BOTH", "leverage": "10"}],
    ]
    mock_client.fetch_open_orders.return_value = []

    now_ts = int(time.time() * 1000)
    with pytest.raises(RiskGateError) as exc:
        orch.execute_action(Action.OPEN_LONG, mark_price=50000.0, mark_price_time_ms=now_ts)
    assert "Target position record for BTCUSDT missing from exchange position risk response" in str(exc.value)
    assert journal.has_active_freeze("LEVERAGE_MISMATCH") is True


def test_unmatched_bracket_tier_raises_error(orchestrator_env):
    """Defect 5 & 6: Bracket query where symbol or tier cannot be matched fails closed without fallback."""
    _, journal, mock_client, orch = orchestrator_env
    mock_client.fetch_leverage_bracket.return_value = [
        {"symbol": "ETHUSDT", "brackets": [{"bracket": 1, "initialLeverage": 10, "maxNotional": 50000.0}]}
    ]
    with pytest.raises(RiskGateError) as exc:
        orch._load_leverage_bracket()
    assert "No leverage bracket found for symbol BTCUSDT" in str(exc.value)


def test_partial_fill_canceled_order_transitions_to_partially_filled_canceled(orchestrator_env):
    """Defect 3 & 4: Terminal CANCELED with filled > 0 transitions to PARTIALLY_FILLED_CANCELED and freezes."""
    _, journal, mock_client, orch = orchestrator_env
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    journal.record_daily_baseline(today_utc, 10000.0, "VERIFIED_UTC_BASELINE")

    mock_client.fetch_account_v2.return_value = {
        "totalMarginBalance": "10000.00",
        "assets": [{"asset": "USDT", "availableBalance": "9000.00"}],
        "positions": [{"symbol": "BTCUSDT", "positionInitialMargin": "0.00"}],
    }
    mock_client.fetch_position_risk_v2.return_value = [
        {"symbol": "BTCUSDT", "positionAmt": "0.00", "positionSide": "BOTH", "leverage": "10"}
    ]
    mock_client.fetch_open_orders.return_value = []

    # Exchange reports CANCELED with partial fill: 0.05 filled out of 0.2
    mock_client.create_market_order.return_value = {
        "id": "ex_pfc_1",
        "status": "CANCELED",
        "filled": 0.05,
        "cost": 2500.0,
        "average": 50000.0,
    }

    now_ts = int(time.time() * 1000)
    order = orch.execute_action(Action.OPEN_LONG, mark_price=50000.0, mark_price_time_ms=now_ts)
    assert order.state == "PARTIALLY_FILLED_CANCELED"
    assert order.filled_quantity == 0.05
    assert journal.has_active_freeze("PARTIAL_FILL_UNRESOLVED") is True


def test_partial_fill_expired_order_transitions_to_partially_filled_expired(orchestrator_env):
    """Defect 3 & 4: Terminal EXPIRED with filled > 0 transitions to PARTIALLY_FILLED_EXPIRED and freezes."""
    _, journal, mock_client, orch = orchestrator_env
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    journal.record_daily_baseline(today_utc, 10000.0, "VERIFIED_UTC_BASELINE")

    mock_client.fetch_account_v2.return_value = {
        "totalMarginBalance": "10000.00",
        "assets": [{"asset": "USDT", "availableBalance": "9000.00"}],
        "positions": [{"symbol": "BTCUSDT", "positionInitialMargin": "0.00"}],
    }
    mock_client.fetch_position_risk_v2.return_value = [
        {"symbol": "BTCUSDT", "positionAmt": "0.00", "positionSide": "BOTH", "leverage": "10"}
    ]
    mock_client.fetch_open_orders.return_value = []

    # Exchange reports EXPIRED with partial fill
    mock_client.create_market_order.return_value = {
        "id": "ex_pfe_1",
        "status": "EXPIRED",
        "filled": 0.08,
        "cost": 4000.0,
        "average": 50000.0,
    }

    now_ts = int(time.time() * 1000)
    order = orch.execute_action(Action.OPEN_LONG, mark_price=50000.0, mark_price_time_ms=now_ts)
    assert order.state == "PARTIALLY_FILLED_EXPIRED"
    assert order.filled_quantity == 0.08
    assert journal.has_active_freeze("PARTIAL_FILL_UNRESOLVED") is True


def test_zero_fill_canceled_order_transitions_to_canceled(orchestrator_env):
    """Defect 3: Terminal CANCELED with filled == 0 transitions to clean CANCELED without partial fill freeze."""
    _, journal, mock_client, orch = orchestrator_env
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    journal.record_daily_baseline(today_utc, 10000.0, "VERIFIED_UTC_BASELINE")

    mock_client.fetch_account_v2.return_value = {
        "totalMarginBalance": "10000.00",
        "assets": [{"asset": "USDT", "availableBalance": "9000.00"}],
        "positions": [{"symbol": "BTCUSDT", "positionInitialMargin": "0.00"}],
    }
    mock_client.fetch_position_risk_v2.return_value = [
        {"symbol": "BTCUSDT", "positionAmt": "0.00", "positionSide": "BOTH", "leverage": "10"}
    ]
    mock_client.fetch_open_orders.return_value = []

    # Exchange reports CANCELED with 0 fill
    mock_client.create_market_order.return_value = {
        "id": "ex_zero_canc",
        "status": "CANCELED",
        "filled": 0.0,
        "cost": 0.0,
        "average": 0.0,
    }

    now_ts = int(time.time() * 1000)
    order = orch.execute_action(Action.OPEN_LONG, mark_price=50000.0, mark_price_time_ms=now_ts)
    assert order.state == "CANCELED"
    assert order.filled_quantity == 0.0
    assert journal.has_active_freeze("PARTIAL_FILL_UNRESOLVED") is False


def test_transfer_history_authoritative_accounting(orchestrator_env):
    """Defect 3 & 5: Trading income is recognized as internal; verified external transfers calculate correctly."""
    _, journal, mock_client, orch = orchestrator_env
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    journal.record_daily_baseline(today_utc, 10000.0, "VERIFIED_UTC_BASELINE")

    mock_client.fetch_income_paginated.return_value = [
        {"tranId": "inc_001", "incomeType": "REALIZED_PNL", "asset": "USDT", "income": "250.0", "time": 1730000000000},
        {"tranId": "inc_002", "incomeType": "COMMISSION", "asset": "USDT", "income": "-5.0", "time": 1730000001000},
        {"tranId": "ext_deposit_1", "incomeType": "TRANSFER", "asset": "USDT", "income": "1000.0", "time": 1730000002000},
    ]

    orch.set_verified_external_transfers({"ext_deposit_1": True})

    ok, net_transfers, msg = orch.sync_transfer_history(1729990000000, 1730001000000)
    assert ok is True
    assert net_transfers == 1000.0
