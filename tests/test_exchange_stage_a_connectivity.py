"""Stage A — Connectivity Tests for Component 3.

Covers:
1. Exchange initialization.
2. Testnet configuration (sandbox mode).
3. Market loading.
4. Balance retrieval.
5. Position retrieval.
6. Invalid credentials and connection failure handling.

Mocked exchange responses only — zero live network calls.
"""

from unittest.mock import MagicMock, patch
import ccxt
import pytest

from noone.exchange.client import BinanceFuturesClient
from noone.exchange.config import Component3Config
from noone.exchange.errors import ExchangeError


@pytest.fixture
def test_config(tmp_path):
    return Component3Config(
        testnet=True,
        symbol="BTCUSDT",
        database_path=tmp_path / "stage_a.db",
    )


@pytest.fixture
def mock_ccxt_exchange():
    """Mocked CCXT binanceusdm exchange in sandbox mode."""
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


def test_stage_a_01_exchange_initialization_and_production_lock(test_config, mock_ccxt_exchange):
    """1. Verify exchange initialization and production rejection."""
    # Successful testnet init
    client = BinanceFuturesClient(test_config, exchange_instance=mock_ccxt_exchange)
    assert client.config.testnet is True
    assert client.config.symbol == "BTCUSDT"

    # Reject production configuration (testnet=False)
    with pytest.raises(ExchangeError) as exc:
        Component3Config(testnet=False)
    assert "PRODUCTION EXECUTION HARD LOCKED" in str(exc.value)

    # Reject injected client pointing to production Binance
    prod_mock = MagicMock()
    prod_mock.urls = {"api": {"fapiPrivate": "https://fapi.binance.com/fapi/v1"}}
    prod_mock.hostname = "fapi.binance.com"
    with pytest.raises(ExchangeError) as exc_prod:
        BinanceFuturesClient(test_config, exchange_instance=prod_mock)
    assert "PRODUCTION EXECUTION HARD LOCKED" in str(exc_prod.value)


def test_stage_a_02_testnet_configuration_and_sandbox_mode(test_config):
    """2. Verify CCXT set_sandbox_mode(True) is invoked immediately on construction."""
    with patch("ccxt.binanceusdm") as mock_cls:
        instance = MagicMock()
        instance.urls = {
            "api": {"fapiPrivate": "https://testnet.binancefuture.com/fapi/v1"}
        }
        instance.hostname = "testnet.binancefuture.com"
        mock_cls.return_value = instance

        client = BinanceFuturesClient(test_config, api_key="dummy_key", api_secret="dummy_sec")
        instance.set_sandbox_mode.assert_called_once_with(True)
        assert client is not None


def test_stage_a_03_market_loading(test_config, mock_ccxt_exchange):
    """3. Verify market loading and symbol market metadata retrieval."""
    mock_ccxt_exchange.load_markets.return_value = mock_ccxt_exchange.markets
    client = BinanceFuturesClient(test_config, exchange_instance=mock_ccxt_exchange)

    markets = client.load_markets()
    mock_ccxt_exchange.load_markets.assert_called_once()
    assert "BTC/USDT:USDT" in markets

    mock_ccxt_exchange.market.return_value = mock_ccxt_exchange.markets["BTC/USDT:USDT"]
    market_info = client.get_market("BTC/USDT:USDT")
    assert market_info["symbol"] == "BTC/USDT:USDT"
    assert market_info["limits"]["amount"]["min"] == 0.001
    assert market_info["limits"]["cost"]["min"] == 5.0


def test_stage_a_04_balance_retrieval(test_config, mock_ccxt_exchange):
    """4. Verify balance and available margin retrieval via CCXT fetch_balance."""
    mock_balance = {
        "info": {"totalMarginBalance": "10000.0"},
        "USDT": {"free": 8500.0, "used": 1500.0, "total": 10000.0},
        "total": {"USDT": 10000.0},
        "free": {"USDT": 8500.0},
        "used": {"USDT": 1500.0},
    }
    mock_ccxt_exchange.fetch_balance.return_value = mock_balance
    client = BinanceFuturesClient(test_config, exchange_instance=mock_ccxt_exchange)

    raw_bal = client.fetch_balance()
    mock_ccxt_exchange.fetch_balance.assert_called_with({"type": "future"})
    assert raw_bal["total"]["USDT"] == 10000.0
    assert raw_bal["free"]["USDT"] == 8500.0

    summary = client.fetch_account_balance()
    assert summary["equity"] == 10000.0
    assert summary["available_balance"] == 8500.0
    assert summary["margin_used"] == 1500.0


def test_stage_a_05_position_retrieval(test_config, mock_ccxt_exchange):
    """5. Verify current open position retrieval via CCXT fetch_positions."""
    mock_positions = [
        {
            "symbol": "BTC/USDT:USDT",
            "contracts": 0.25,
            "contractSize": 1.0,
            "entryPrice": 62000.0,
            "unrealizedPnl": 150.0,
            "leverage": 10,
            "side": "long",
            "notional": 15500.0,
            "initialMargin": 1550.0,
            "info": {"symbol": "BTCUSDT", "positionAmt": "0.250"},
        }
    ]
    mock_ccxt_exchange.fetch_positions.return_value = mock_positions
    client = BinanceFuturesClient(test_config, exchange_instance=mock_ccxt_exchange)

    positions = client.fetch_positions()
    mock_ccxt_exchange.fetch_positions.assert_called_with(["BTC/USDT:USDT"], params={})
    assert len(positions) == 1
    assert positions[0]["contracts"] == 0.25
    assert positions[0]["side"] == "long"

    pos = client.fetch_position("BTC/USDT:USDT")
    assert pos is not None
    assert pos["entryPrice"] == 62000.0


def test_stage_a_06_invalid_credentials_and_connection_failure_handling(test_config, mock_ccxt_exchange):
    """6. Verify invalid credentials and network error handling wrap cleanly into ExchangeError."""
    client = BinanceFuturesClient(test_config, exchange_instance=mock_ccxt_exchange)

    # Authentication failure
    mock_ccxt_exchange.fetch_balance.side_effect = ccxt.AuthenticationError("API-key format invalid")
    with pytest.raises(ExchangeError) as auth_exc:
        client.fetch_balance()
    assert "Failed to fetch account balance" in str(auth_exc.value)

    # Network / connection failure
    mock_ccxt_exchange.fetch_positions.side_effect = ccxt.NetworkError("Connection timed out")
    with pytest.raises(ExchangeError) as net_exc:
        client.fetch_positions()
    assert "Failed to fetch positions" in str(net_exc.value)

