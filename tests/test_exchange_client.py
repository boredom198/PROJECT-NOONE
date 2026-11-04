"""Tests for BinanceFuturesClient (Component 3).

Verifies CCXT 4.4.52 method dispatch, pre-network transport boundary serialization (T01-A, T01-B),
HMAC-SHA256 signature generation, position mode probing (T09), and clock drift checks.
"""

import hashlib
import hmac
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import ccxt
import pytest

from noone.exchange.client import (
    BinanceFuturesClient,
    verify_non_production_exchange,
    validate_order_parameters,
)
from noone.exchange.config import Component3Config
from noone.exchange.errors import (
    ClockDriftError,
    ExchangeError,
    FilterValidationError,
    OrderRecoveryError,
)


@pytest.fixture
def client_config(tmp_path):
    return Component3Config(
        testnet=True,
        symbol="BTCUSDT",
        database_path=tmp_path / "test_exec.db",
    )


@pytest.fixture
def mock_exchange_instance():
    ex = ccxt.binanceusdm({
        "apiKey": "test_api_key_1234567890",
        "secret": "test_api_secret_abcdef1234567890abcdef",
        "enableRateLimit": False,
    })
    ex.set_sandbox_mode(True)
    # Preload mock market for offline tests
    ex.markets = {
        "BTC/USDT:USDT": {
            "id": "BTCUSDT",
            "symbol": "BTC/USDT:USDT",
            "base": "BTC",
            "quote": "USDT",
            "linear": True,
            "inverse": False,
            "option": False,
            "contract": True,
            "type": "future",
            "active": True,
        }
    }
    return ex


def test_client_init_and_testnet_mode(client_config, mock_exchange_instance):
    client = BinanceFuturesClient(
        client_config,
        exchange_instance=mock_exchange_instance,
    )
    assert client._exchange is mock_exchange_instance
    assert not hasattr(client, "exchange")
    assert client.config.testnet is True
    assert client.config.raw_symbol == "BTCUSDT"
    assert client.config.ccxt_symbol == "BTC/USDT:USDT"


def test_t01_a_endpoint_dispatch_and_parameter_forwarding(client_config, mock_exchange_instance):
    """Test T01-A: Verify fetch_order with origClientOrderId forwards arguments correctly."""
    client = BinanceFuturesClient(client_config, exchange_instance=mock_exchange_instance)
    mock_order_response = {
        "orderId": 987654321,
        "symbol": "BTCUSDT",
        "status": "FILLED",
        "clientOrderId": "noone_1730000000000_abcd1234",
        "price": "50000.0",
        "origQty": "0.100",
        "executedQty": "0.100",
        "cumQuote": "5000.0",
        "timeInForce": "GTC",
        "type": "MARKET",
        "side": "BUY",
        "updateTime": 1730000005000,
    }

    with patch.object(mock_exchange_instance, "fapiPrivateGetOrder", return_value=mock_order_response) as mock_get:
        with patch.object(mock_exchange_instance, "load_markets", return_value=mock_exchange_instance.markets):
            order = client.fetch_order_by_client_id("noone_1730000000000_abcd1234")

            # Check that CCXT forwarded symbol and origClientOrderId
            mock_get.assert_called_once()
            called_params = mock_get.call_args[0][0]
            assert called_params["symbol"] == "BTCUSDT"
            assert called_params["origClientOrderId"] == "noone_1730000000000_abcd1234"
            assert "orderId" not in called_params
            assert order["id"] == "987654321" or order["info"]["orderId"] == 987654321


def test_t01_b_pre_network_transport_boundary_serialization_and_hmac(client_config, mock_exchange_instance):
    """Test T01-B: Intercept CCXT fetch at transport boundary without network access.

    Verifies final serialized URL, query parameters, HTTP method, headers, and HMAC-SHA256 signature.
    """
    client = BinanceFuturesClient(client_config, exchange_instance=mock_exchange_instance)
    captured_requests = []

    def mock_fetch(url, method="GET", headers=None, body=None):
        captured_requests.append({
            "url": url,
            "method": method,
            "headers": headers or {},
            "body": body,
        })
        # Return valid mock response for CCXT to parse
        return {
            "orderId": 123456789,
            "symbol": "BTCUSDT",
            "status": "FILLED",
            "clientOrderId": "noone_transport_test_001",
            "price": "60000.0",
            "origQty": "0.050",
            "executedQty": "0.050",
            "cumQuote": "3000.0",
            "timeInForce": "GTC",
            "type": "MARKET",
            "side": "BUY",
            "updateTime": 1730000010000,
        }

    with patch.object(mock_exchange_instance, "fetch", side_effect=mock_fetch):
        with patch.object(mock_exchange_instance, "load_markets", return_value=mock_exchange_instance.markets):
            client.fetch_order_by_client_id("noone_transport_test_001")

    assert len(captured_requests) == 1
    req = captured_requests[0]

    # Verify HTTP Method
    assert req["method"] == "GET"

    # Verify Headers contain X-MBX-APIKEY
    assert req["headers"].get("X-MBX-APIKEY") == "test_api_key_1234567890"

    # Parse URL and Query parameters
    parsed_url = urlparse(req["url"])
    assert "/fapi/v1/order" in parsed_url.path
    query_params = parse_qs(parsed_url.query)

    assert query_params.get("symbol") == ["BTCUSDT"]
    assert query_params.get("origClientOrderId") == ["noone_transport_test_001"]
    assert "timestamp" in query_params
    assert "signature" in query_params

    # Verify HMAC-SHA256 signature calculation over the raw query string
    signature_on_wire = query_params["signature"][0]
    # The signature in Binance query string is computed over query string up to &signature=
    query_without_sig = parsed_url.query.split("&signature=")[0]
    expected_sig = hmac.new(
        "test_api_secret_abcdef1234567890abcdef".encode("utf-8"),
        query_without_sig.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    assert signature_on_wire == expected_sig


def test_t09_position_mode_probe_one_way_vs_hedge(client_config, mock_exchange_instance):
    """Test T09: Position mode probe must distinguish One-Way mode from Hedge mode."""
    client = BinanceFuturesClient(client_config, exchange_instance=mock_exchange_instance)

    # 1. One-Way Mode response
    with patch.object(mock_exchange_instance, "fapiPrivateGetPositionSideDual", return_value={"dualSidePosition": False}):
        res = client.fetch_position_side_dual()
        assert res["dualSidePosition"] is False

    # 2. Hedge Mode response
    with patch.object(mock_exchange_instance, "fapiPrivateGetPositionSideDual", return_value={"dualSidePosition": True}):
        res = client.fetch_position_side_dual()
        assert res["dualSidePosition"] is True


def test_clock_drift_verification(client_config, mock_exchange_instance):
    """Verify clock drift detection and threshold enforcement."""
    client = BinanceFuturesClient(client_config, exchange_instance=mock_exchange_instance)

    # Healthy drift (< 1000ms)
    with patch("time.time", return_value=1700000000.0):
        with patch.object(mock_exchange_instance, "fapiPublicGetTime", return_value={"serverTime": 1700000000100}):
            drift = client.verify_clock_drift()
            assert drift == 100

    # Excessive drift (>= 1000ms) -> ClockDriftError
    with patch("time.time", return_value=1700000000.0):
        with patch.object(mock_exchange_instance, "fapiPublicGetTime", return_value={"serverTime": 1700000002500}):
            with pytest.raises(ClockDriftError) as exc_info:
                client.verify_clock_drift()
            assert "Clock drift 2500ms exceeds threshold" in str(exc_info.value)


def test_injected_production_client_rejected(client_config):
    """Defect 1: Reject injected client configured with production Binance endpoint."""
    prod_ex = ccxt.binanceusdm({
        "apiKey": "test_key",
        "secret": "test_secret",
    })
    # Default without sandbox points to fapi.binance.com
    with pytest.raises(ExchangeError) as exc:
        BinanceFuturesClient(client_config, exchange_instance=prod_ex)
    assert "PRODUCTION EXECUTION HARD LOCKED" in str(exc.value)


def test_injected_production_hostname_rejected(client_config):
    """Defect 1: Reject injected exchange instance with production hostname."""
    fake_prod = MagicMock()
    fake_prod.urls = {"api": {"fapiPrivate": "https://fapi.binance.com/fapi/v1"}}
    fake_prod.hostname = "fapi.binance.com"
    with pytest.raises(ExchangeError) as exc:
        BinanceFuturesClient(client_config, exchange_instance=fake_prod)
    assert "PRODUCTION EXECUTION HARD LOCKED" in str(exc.value)


def test_raw_exchange_not_exposed(client_config, mock_exchange_instance):
    """Raw CCXT exchange must NOT be exposed through public exchange property."""
    client = BinanceFuturesClient(client_config, exchange_instance=mock_exchange_instance)
    assert not hasattr(client, "exchange")
    with pytest.raises(AttributeError):
        _ = client.exchange


def test_basic_order_parameter_validation(client_config, mock_exchange_instance):
    """Item 4: Reject zero, negative, NaN, Inf quantities, invalid sides, and invalid types."""
    client = BinanceFuturesClient(client_config, exchange_instance=mock_exchange_instance)

    # 1. Zero quantity
    with pytest.raises(FilterValidationError) as exc:
        client.create_market_order("BTC/USDT:USDT", "buy", 0.0)
    assert "positive" in str(exc.value)

    # 2. Negative quantity
    with pytest.raises(FilterValidationError) as exc:
        client.create_market_order("BTC/USDT:USDT", "buy", -0.05)
    assert "positive" in str(exc.value)

    # 3. NaN quantity
    with pytest.raises(FilterValidationError) as exc:
        client.create_market_order("BTC/USDT:USDT", "buy", float("nan"))
    assert "NaN or infinite" in str(exc.value)

    # 4. Infinite quantity
    with pytest.raises(FilterValidationError) as exc:
        client.create_market_order("BTC/USDT:USDT", "buy", float("inf"))
    assert "NaN or infinite" in str(exc.value)

    # 5. Invalid side
    with pytest.raises(FilterValidationError) as exc:
        client.create_market_order("BTC/USDT:USDT", "invalid_side", 0.05)
    assert "Invalid order side" in str(exc.value)

    # 6. Invalid type in create_order
    with pytest.raises(FilterValidationError) as exc:
        client.create_order("BTC/USDT:USDT", "stop_loss_limit", "buy", 0.05)
    assert "Invalid order type" in str(exc.value)


def test_mark_price_lacks_timestamp_rejected(client_config, mock_exchange_instance):
    """Item 5: Explicit error returned when exchange mark price response lacks source timestamp."""
    client = BinanceFuturesClient(client_config, exchange_instance=mock_exchange_instance)

    with patch.object(mock_exchange_instance, "fapiPublicGetPremiumIndex", return_value={"markPrice": "65000.0"}):
        with pytest.raises(ExchangeError) as exc:
            client.fetch_mark_price_data("BTCUSDT")
        assert "Exchange response lacks source timestamp 'time'" in str(exc.value)


def test_normal_testnet_configuration_succeeds(client_config, mock_exchange_instance):
    """Normal testnet configuration is verified and accepted."""
    client = BinanceFuturesClient(client_config, exchange_instance=mock_exchange_instance)
    assert client.config.testnet is True
    assert client.config.production_confirmed is False

