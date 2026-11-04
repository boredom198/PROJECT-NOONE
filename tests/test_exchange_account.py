"""Tests for account and position synchronization (Component 3).

Verifies AccountSnapshot mapping, active Long fixture (10050.0 totalMarginBalance, T07),
missing balance rejection (T08-A), positionInitialMargin validation (T08-B),
duplicate position rejection (T10), hedge mode rejection, and session realized PnL preservation.
Conforms strictly to PROJECT NOONE Technical Design Specification V2.2.4 Sections 4 and 5.
"""

import pytest

from noone.exchange.account import (
    audit_account_positions,
    audit_open_orders,
    parse_account_snapshot,
)
from noone.exchange.errors import (
    AccountSyncError,
    DuplicatePositionRecordError,
    IncompatiblePositionSideError,
    MissingPositionRecordError,
)


@pytest.fixture
def active_long_fixtures():
    """Authoritative active Long account fixtures per V2.2.4 specification.

    totalWalletBalance: 10000.0
    totalUnrealizedProfit: 50.0
    totalMarginBalance: 10050.0 (wallet + unrealized)
    """
    account_data = {
        "totalWalletBalance": "10000.00000000",
        "totalUnrealizedProfit": "50.00000000",
        "totalMarginBalance": "10050.00000000",
        "totalRealizedProfit": "999999.00000000",  # Binance lifetime profit (must NOT be used!)
        "assets": [
            {
                "asset": "USDT",
                "walletBalance": "10000.00000000",
                "unrealizedProfit": "50.00000000",
                "marginBalance": "10050.00000000",
                "availableBalance": "9550.00000000",
                "initialMargin": "500.00000000",
            }
        ],
        "positions": [
            {
                "symbol": "BTCUSDT",
                "positionInitialMargin": "500.00000000",
                "unrealizedProfit": "50.00000000",
                "positionSide": "BOTH",
            }
        ],
    }

    position_risk_data = [
        {
            "symbol": "BTCUSDT",
            "positionAmt": "0.100",
            "entryPrice": "50000.00",
            "unRealizedProfit": "50.00000000",
            "positionSide": "BOTH",
            "leverage": "10",
        }
    ]

    return account_data, position_risk_data


def test_t07_account_snapshot_parsing_active_long(active_long_fixtures):
    """Test T07: Verify AccountSnapshot mapping for active Long with totalMarginBalance = 10050.0."""
    account_data, position_risk_data = active_long_fixtures

    snapshot = parse_account_snapshot(
        account_data=account_data,
        position_risk_data=position_risk_data,
        target_symbol="BTCUSDT",
        session_realized_pnl=42.50,
    )

    assert snapshot.equity == 10050.0
    assert snapshot.available_balance == 9550.0
    assert snapshot.position_side == 1
    assert snapshot.position_quantity == 0.1
    assert snapshot.entry_price == 50000.0
    assert snapshot.unrealized_pnl == 50.0
    assert snapshot.margin_used == 500.0
    assert snapshot.realized_pnl == 42.50  # Must preserve session realized PnL, not Binance lifetime!
    assert snapshot.progress == 0.0


def test_account_snapshot_parsing_active_short():
    """Verify AccountSnapshot mapping for active Short position (position_side = -1)."""
    account_data = {
        "totalMarginBalance": "5000.00000000",
        "assets": [{"asset": "USDT", "availableBalance": "4000.00000000"}],
        "positions": [{"symbol": "BTCUSDT", "positionInitialMargin": "800.00000000"}],
    }
    position_risk_data = [
        {
            "symbol": "BTCUSDT",
            "positionAmt": "-0.200",
            "entryPrice": "60000.00",
            "unRealizedProfit": "-120.00000000",
            "positionSide": "BOTH",
        }
    ]

    snapshot = parse_account_snapshot(
        account_data=account_data,
        position_risk_data=position_risk_data,
        target_symbol="BTCUSDT",
        session_realized_pnl=-50.0,
    )

    assert snapshot.equity == 5000.0
    assert snapshot.available_balance == 4000.0
    assert snapshot.position_side == -1
    assert snapshot.position_quantity == 0.2
    assert snapshot.entry_price == 60000.0
    assert snapshot.unrealized_pnl == -120.0
    assert snapshot.margin_used == 800.0
    assert snapshot.realized_pnl == -50.0


def test_account_snapshot_parsing_flat():
    """Verify AccountSnapshot mapping when position is flat."""
    account_data = {
        "totalMarginBalance": "10000.00000000",
        "assets": [{"asset": "USDT", "availableBalance": "10000.00000000"}],
        "positions": [{"symbol": "BTCUSDT", "positionInitialMargin": "0.00000000"}],
    }
    position_risk_data = [
        {
            "symbol": "BTCUSDT",
            "positionAmt": "0.000",
            "entryPrice": "0.00",
            "unRealizedProfit": "0.00000000",
            "positionSide": "BOTH",
        }
    ]

    snapshot = parse_account_snapshot(
        account_data=account_data,
        position_risk_data=position_risk_data,
        target_symbol="BTCUSDT",
        session_realized_pnl=0.0,
    )

    assert snapshot.equity == 10000.0
    assert snapshot.available_balance == 10000.0
    assert snapshot.position_side == 0
    assert snapshot.position_quantity == 0.0
    assert snapshot.entry_price == 0.0
    assert snapshot.unrealized_pnl == 0.0
    assert snapshot.margin_used == 0.0


def test_t08_a_missing_or_non_numeric_balance_rejection():
    """Test T08-A: Missing or non-numeric balance fields must raise AccountSyncError."""
    # 1. Missing totalMarginBalance
    with pytest.raises(AccountSyncError) as exc:
        parse_account_snapshot(
            account_data={"assets": [{"asset": "USDT", "availableBalance": "1000.0"}]},
            position_risk_data=[{"symbol": "BTCUSDT", "positionAmt": "0.0", "positionSide": "BOTH"}],
            target_symbol="BTCUSDT",
        )
    assert "Missing 'totalMarginBalance'" in str(exc.value)

    # 2. Non-numeric totalMarginBalance
    with pytest.raises(AccountSyncError) as exc:
        parse_account_snapshot(
            account_data={
                "totalMarginBalance": "INVALID_NUM",
                "assets": [{"asset": "USDT", "availableBalance": "1000.0"}],
            },
            position_risk_data=[{"symbol": "BTCUSDT", "positionAmt": "0.0", "positionSide": "BOTH"}],
            target_symbol="BTCUSDT",
        )
    assert "Malformed numeric value" in str(exc.value)

    # 3. Missing USDT asset in assets array
    with pytest.raises(AccountSyncError) as exc:
        parse_account_snapshot(
            account_data={
                "totalMarginBalance": "10000.0",
                "assets": [{"asset": "BUSD", "availableBalance": "1000.0"}],
            },
            position_risk_data=[{"symbol": "BTCUSDT", "positionAmt": "0.0", "positionSide": "BOTH"}],
            target_symbol="BTCUSDT",
        )
    assert "USDT asset record not found" in str(exc.value)


def test_t08_b_missing_or_invalid_position_initial_margin_rejection(active_long_fixtures):
    """Test T08-B: Missing, non-numeric, or <= 0.0 positionInitialMargin for active position raises AccountSyncError."""
    account_data, position_risk_data = active_long_fixtures

    # 1. Missing positionInitialMargin
    bad_account = dict(account_data)
    bad_account["positions"] = [{"symbol": "BTCUSDT"}]  # Missing positionInitialMargin
    with pytest.raises(AccountSyncError) as exc:
        parse_account_snapshot(bad_account, position_risk_data, "BTCUSDT")
    assert "Missing 'positionInitialMargin'" in str(exc.value)

    # 2. Zero positionInitialMargin for active position (0.1 BTC)
    bad_account2 = dict(account_data)
    bad_account2["positions"] = [{"symbol": "BTCUSDT", "positionInitialMargin": "0.00000000"}]
    with pytest.raises(AccountSyncError) as exc:
        parse_account_snapshot(bad_account2, position_risk_data, "BTCUSDT")
    assert "Invalid positionInitialMargin" in str(exc.value)

    # 3. Non-numeric positionInitialMargin
    bad_account3 = dict(account_data)
    bad_account3["positions"] = [{"symbol": "BTCUSDT", "positionInitialMargin": "NOT_A_NUMBER"}]
    with pytest.raises(AccountSyncError) as exc:
        parse_account_snapshot(bad_account3, position_risk_data, "BTCUSDT")
    assert "Malformed numeric value" in str(exc.value)


def test_t10_duplicate_position_record_rejection(active_long_fixtures):
    """Test T10: Duplicate position records for target symbol must raise DuplicatePositionRecordError."""
    account_data, position_risk_data = active_long_fixtures

    # Add duplicate BTCUSDT record
    duplicate_risk = list(position_risk_data) + [
        {
            "symbol": "BTCUSDT",
            "positionAmt": "0.100",
            "entryPrice": "50000.00",
            "unRealizedProfit": "50.00",
            "positionSide": "BOTH",
        }
    ]

    with pytest.raises(DuplicatePositionRecordError) as exc:
        parse_account_snapshot(account_data, duplicate_risk, "BTCUSDT")
    assert "Duplicate position risk record encountered" in str(exc.value)


def test_incompatible_position_side_dual_mode_rejection(active_long_fixtures):
    """Verify that hedge mode (LONG or SHORT positionSide) raises IncompatiblePositionSideError."""
    account_data, _ = active_long_fixtures

    hedge_risk = [
        {
            "symbol": "BTCUSDT",
            "positionAmt": "0.100",
            "entryPrice": "50000.00",
            "unRealizedProfit": "50.00",
            "positionSide": "LONG",  # Hedge mode!
        }
    ]

    with pytest.raises(IncompatiblePositionSideError) as exc:
        parse_account_snapshot(account_data, hedge_risk, "BTCUSDT")
    assert "One-Way Mode ('BOTH') is required" in str(exc.value)


def test_audit_open_orders_detection():
    """Verify audit_open_orders flags unexplained exchange open orders."""
    open_orders = [
        {"orderId": 1, "clientOrderId": "noone_known_001"},
        {"orderId": 2, "clientOrderId": "manual_limit_order_999"},
    ]
    known_cids = {"noone_known_001"}

    unexplained = audit_open_orders(open_orders, known_cids)
    assert len(unexplained) == 1
    assert unexplained[0]["clientOrderId"] == "manual_limit_order_999"

