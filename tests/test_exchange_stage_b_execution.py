"""Stage B — Execution Tests for Component 3.

Covers:
1. Quantity validation (stepSize, minQty, minNotional).
2. Leverage configuration (set_leverage, set_margin_mode).
3. Long order dispatch (create_market_order BUY).
4. Short order dispatch (create_market_order SELL).
5. Order-status retrieval (fetch_order by ID and origClientOrderId).
6. Partial-fill response handling.
7. Reduce-only position closure (reduceOnly=True).
8. Unknown order outcome handling (network timeout -> freeze, never blindly retry).

Mocked exchange responses only — zero live network calls.
"""

from decimal import Decimal
import time
from unittest.mock import MagicMock, patch
import ccxt
import pytest

from noone.exchange.client import BinanceFuturesClient
from noone.exchange.config import Component3Config
from noone.exchange.errors import ExchangeError, FilterValidationError
from noone.exchange.journal import ExecutionJournal
from noone.exchange.orchestrator import ExchangeOrchestrator
from noone.exchange.precision import (
    ParsedSymbolFilters,
    quantize_downward,
    check_residual_dust,
    parse_market_limits,
)
from noone.rl_engine.inference import Action


@pytest.fixture
def stage_b_config(tmp_path):
    return Component3Config(
        testnet=True,
        symbol="BTCUSDT",
        database_path=tmp_path / "stage_b.db",
    )


@pytest.fixture
def mock_exchange():
    ex = MagicMock(spec=ccxt.binanceusdm)
    ex.urls = {
        "api": {
            "fapiPublic": "https://testnet.binancefuture.com/fapi/v1",
            "fapiPrivate": "https://testnet.binancefuture.com/fapi/v1",
            "fapiPrivateV2": "https://testnet.binancefuture.com/fapi/v2",
        }
    }
    ex.hostname = "testnet.binancefuture.com"
    ex.markets = {
        "BTC/USDT:USDT": {
            "id": "BTCUSDT",
            "symbol": "BTC/USDT:USDT",
            "base": "BTC",
            "quote": "USDT",
            "contract": True,
            "type": "future",
            "limits": {
                "amount": {"min": 0.001, "max": 1000.0, "step": 0.001},
                "cost": {"min": 5.0, "max": 100000.0},
                "price": {"min": 0.1, "max": 1000000.0, "step": 0.1},
            },
            "precision": {"amount": 0.001, "price": 0.1},
        }
    }
    return ex


def test_stage_b_01_quantity_validation(stage_b_config, mock_exchange):
    """1. Verify downward quantization, filter validation, and basic order parameter checks."""
    step = Decimal("0.001")
    # Downward truncation: 0.1239 -> 0.123 (never rounds up to 0.124)
    assert quantize_downward(Decimal("0.1239"), step) == Decimal("0.123")
    assert quantize_downward(Decimal("0.1231"), step) == Decimal("0.123")
    assert quantize_downward(Decimal("0.1230"), step) == Decimal("0.123")

    filters = ParsedSymbolFilters(
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.001"),
        max_qty=Decimal("100.0"),
        min_notional=Decimal("5.0"),
    )

    # Valid order quantity and notional
    qty = quantize_downward(Decimal("0.010"), filters.step_size)
    mark_price = Decimal("60000.0")
    notional = qty * mark_price
    assert qty >= filters.min_qty
    assert notional >= filters.min_notional

    # Dust detection: quantity < minQty or notional < minNotional
    assert check_residual_dust(Decimal("0.00005"), mark_price, filters) is True
    assert check_residual_dust(Decimal("0.01"), mark_price, filters) is False

    # Basic parameter validation before dispatch (Item 4)
    client = BinanceFuturesClient(stage_b_config, exchange_instance=mock_exchange)

    # Reject zero / negative
    with pytest.raises(FilterValidationError):
        client.create_market_order("BTC/USDT:USDT", "buy", 0.0)
    with pytest.raises(FilterValidationError):
        client.create_market_order("BTC/USDT:USDT", "buy", -0.01)

    # Reject NaN / infinite
    with pytest.raises(FilterValidationError):
        client.create_market_order("BTC/USDT:USDT", "buy", float("nan"))
    with pytest.raises(FilterValidationError):
        client.create_market_order("BTC/USDT:USDT", "buy", float("inf"))

    # Reject invalid side
    with pytest.raises(FilterValidationError):
        client.create_market_order("BTC/USDT:USDT", "hold", 0.05)

    # Reject invalid type
    with pytest.raises(FilterValidationError):
        client.create_order("BTC/USDT:USDT", "trailing_stop", "buy", 0.05)


def test_stage_b_02_leverage_configuration(stage_b_config, mock_exchange):
    """2. Verify leverage and margin mode configuration dispatch."""
    mock_exchange.set_leverage.return_value = {"leverage": 10, "symbol": "BTCUSDT"}
    mock_exchange.set_margin_mode.return_value = {"code": 200, "msg": "success"}

    client = BinanceFuturesClient(stage_b_config, exchange_instance=mock_exchange)

    lev_res = client.set_leverage(10)
    mock_exchange.set_leverage.assert_called_with(10, "BTC/USDT:USDT")
    assert lev_res["leverage"] == 10

    margin_res = client.set_margin_mode("cross")
    mock_exchange.set_margin_mode.assert_called_with("cross", "BTC/USDT:USDT")
    assert margin_res["code"] == 200


def test_stage_b_03_long_order_dispatch(stage_b_config, mock_exchange):
    """3. Verify long order dispatch via create_market_order with BUY side."""
    mock_order = {
        "id": "111222333",
        "symbol": "BTC/USDT:USDT",
        "type": "market",
        "side": "buy",
        "amount": 0.05,
        "status": "closed",
        "filled": 0.05,
        "remaining": 0.0,
        "info": {"orderId": "111222333", "clientOrderId": "cid_long_01", "status": "FILLED"},
    }
    mock_exchange.create_order.return_value = mock_order
    client = BinanceFuturesClient(stage_b_config, exchange_instance=mock_exchange)

    res = client.create_market_order(
        symbol="BTC/USDT:USDT",
        side="buy",
        amount=0.05,
        client_order_id="cid_long_01",
        reduce_only=False,
    )

    mock_exchange.create_order.assert_called_once_with(
        symbol="BTC/USDT:USDT",
        type="market",
        side="buy",
        amount=0.05,
        params={"newClientOrderId": "cid_long_01"},
    )
    assert res["id"] == "111222333"
    assert res["side"] == "buy"


def test_stage_b_04_short_order_dispatch(stage_b_config, mock_exchange):
    """4. Verify short order dispatch via create_market_order with SELL side."""
    mock_order = {
        "id": "444555666",
        "symbol": "BTC/USDT:USDT",
        "type": "market",
        "side": "sell",
        "amount": 0.08,
        "status": "closed",
        "filled": 0.08,
        "remaining": 0.0,
        "info": {"orderId": "444555666", "clientOrderId": "cid_short_01", "status": "FILLED"},
    }
    mock_exchange.create_order.return_value = mock_order
    client = BinanceFuturesClient(stage_b_config, exchange_instance=mock_exchange)

    res = client.create_market_order(
        symbol="BTC/USDT:USDT",
        side="sell",
        amount=0.08,
        client_order_id="cid_short_01",
        reduce_only=False,
    )

    mock_exchange.create_order.assert_called_once_with(
        symbol="BTC/USDT:USDT",
        type="market",
        side="sell",
        amount=0.08,
        params={"newClientOrderId": "cid_short_01"},
    )
    assert res["id"] == "444555666"
    assert res["side"] == "sell"


def test_stage_b_05_order_status_retrieval(stage_b_config, mock_exchange):
    """5. Verify order status retrieval using orderId and origClientOrderId."""
    mock_response = {
        "id": "999888777",
        "clientOrderId": "cid_lookup_01",
        "symbol": "BTC/USDT:USDT",
        "status": "closed",
        "filled": 0.1,
        "average": 65000.0,
    }
    mock_exchange.fetch_order.return_value = mock_response
    client = BinanceFuturesClient(stage_b_config, exchange_instance=mock_exchange)

    # Retrieval by clientOrderId
    res = client.fetch_order_by_client_id("cid_lookup_01")
    mock_exchange.fetch_order.assert_called_with(
        id=None,
        symbol="BTC/USDT:USDT",
        params={"origClientOrderId": "cid_lookup_01"},
    )
    assert res["status"] == "closed"
    assert res["filled"] == 0.1


def test_stage_b_06_partial_fill_response_handling(stage_b_config, mock_exchange):
    """6. Verify partial fill response handling records partial quantity."""
    journal = ExecutionJournal(stage_b_config.database_path)
    client = BinanceFuturesClient(stage_b_config, exchange_instance=mock_exchange)

    cid = "cid_part_fill_01"
    journal.create_order_intent(
        client_order_id=cid,
        symbol="BTCUSDT",
        action_type="OPEN_LONG",
        side="BUY",
        quantity=0.10,
        price=60000.0,
        order_type="MARKET",
        reduce_only=False,
    )

    # Order returns PARTIALLY_FILLED
    journal.transition_order_state(
        client_order_id=cid,
        to_state="PARTIALLY_FILLED",
        filled_quantity=0.04,
        avg_price=60000.0,
        exchange_order_id="ex_part_1",
    )

    order_record = journal.get_order(cid)
    assert order_record is not None
    assert order_record.state == "PARTIALLY_FILLED"
    assert order_record.filled_quantity == pytest.approx(0.04)
    assert order_record.remaining_quantity == pytest.approx(0.06)


def test_stage_b_07_reduce_only_position_closure(stage_b_config, mock_exchange):
    """7. Verify closing an existing position passes reduceOnly=True."""
    mock_order = {
        "id": "777888999",
        "symbol": "BTC/USDT:USDT",
        "side": "sell",
        "status": "closed",
        "filled": 0.05,
    }
    mock_exchange.create_order.return_value = mock_order
    client = BinanceFuturesClient(stage_b_config, exchange_instance=mock_exchange)

    client.create_market_order(
        symbol="BTC/USDT:USDT",
        side="sell",
        amount=0.05,
        client_order_id="cid_close_01",
        reduce_only=True,
    )

    mock_exchange.create_order.assert_called_once_with(
        symbol="BTC/USDT:USDT",
        type="market",
        side="sell",
        amount=0.05,
        params={"newClientOrderId": "cid_close_01", "reduceOnly": True},
    )


def test_stage_b_08_unknown_order_outcome_handling(stage_b_config, mock_exchange):
    """8. Verify network timeout during dispatch triggers freeze and NEVER blindly retries."""
    journal = ExecutionJournal(stage_b_config.database_path)
    mock_client = MagicMock()
    mock_client.config = stage_b_config
    mock_client.verify_clock_drift.return_value = 10
    mock_client.fetch_account_v2.return_value = {
        "canTrade": True,
        "totalMarginBalance": "10000.0",
        "assets": [{"asset": "USDT", "availableBalance": "9000.0"}],
        "positions": [{"symbol": "BTCUSDT", "positionInitialMargin": "0.0"}],
    }
    mock_client.fetch_position_risk_v2.return_value = [
        {"symbol": "BTCUSDT", "positionAmt": "0.0", "positionSide": "BOTH", "entryPrice": "0.0", "unRealizedProfit": "0.0", "leverage": "10"}
    ]
    mock_client.fetch_position_side_dual.return_value = {"dualSidePosition": False}
    mock_client.fetch_open_orders.return_value = []
    mock_client.fetch_exchange_info.return_value = {
        "symbols": [{
            "symbol": "BTCUSDT",
            "filters": [
                {"filterType": "MARKET_LOT_SIZE", "minQty": "0.001", "maxQty": "100.0", "stepSize": "0.001"},
                {"filterType": "MIN_NOTIONAL", "notional": "5.0"},
            ]
        }]
    }
    mock_client.fetch_leverage_bracket.return_value = [
        {"symbol": "BTCUSDT", "brackets": [{"bracket": 1, "initialLeverage": 10, "notionalCap": 50000.0}]}
    ]

    # Pre-record daily baseline so risk checks pass
    today_utc = "2026-11-04"
    journal.record_daily_baseline(today_utc, 10000.0, "VERIFIED_UTC_BASELINE")

    # Simulate network failure during create_market_order dispatch
    mock_client.create_market_order.side_effect = ccxt.NetworkError("Connection reset by peer during order submission")

    orch = ExchangeOrchestrator(stage_b_config, client=mock_client, journal=journal)

    now_ms = int(time.time() * 1000)
    with pytest.raises(ExchangeError) as exc:
        orch.execute_action(Action.OPEN_LONG, mark_price=60000.0, mark_price_time_ms=now_ms)
    assert "Order dispatch failed" in str(exc.value) or "communication failure" in str(exc.value)

    # Verify order transitioned to UNKNOWN, NOT retried
    unresolved = journal.get_unresolved_orders()
    assert len(unresolved) == 1
    assert unresolved[0].state == "UNKNOWN"

    # Verify safety freeze was triggered
    freezes = journal.get_active_safety_freezes()
    reasons = [f.reason for f in freezes]
    assert "NETWORK_COMMUNICATION_FAILURE" in reasons or "UNKNOWN_ORDER_OUTCOME" in reasons

    # Subsequent entry attempt MUST be blocked by the active freeze (never blindly retry)
    with pytest.raises(Exception) as block_exc:
        orch.execute_action(Action.OPEN_LONG, mark_price=60000.0, mark_price_time_ms=now_ms)
    assert "freeze" in str(block_exc.value).lower() or "unresolved" in str(block_exc.value).lower()
