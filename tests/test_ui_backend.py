"""Component 6 backend tests (offline, mocked network boundaries)."""
from __future__ import annotations

import json
import urllib.request
from unittest.mock import MagicMock

import pytest

from noone.security.credentials import CredentialsNotFoundError
from noone.ui.services import (
    CredentialService,
    EngineService,
    MainnetAccountService,
    MarketService,
)
from noone.ui.services import TestnetAccountService as _TestnetAccountService
from noone.ui.state_store import UIStateStore


@pytest.fixture
def store(tmp_path):
    return UIStateStore(path=tmp_path / "ui_state.json")


def test_chart_uses_component5_embed():
    html = MarketService.__new__(MarketService).chart_html("ethusdt", "1h")
    assert "BINANCE:ETHUSDT.P" in html
    assert "s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" in html


def test_mainnet_facade_whitelist_reads_only(store):
    creds = CredentialService()
    svc = MainnetAccountService(creds)
    mock = MagicMock()
    mock.config.symbol = "BTC/USDT:USDT"
    mock.fetch_balance.return_value = {
        "equity": 1000.0,
        "available_balance": 900.0,
        "margin_used": 100.0,
    }
    mock.fetch_positions.return_value = [
        {"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.1,
         "entryPrice": 50000, "markPrice": 51000, "unrealizedPnl": 100,
         "leverage": 1, "marginMode": "isolated"}
    ]
    mock.fetch_open_orders.return_value = []
    mock.load_market_filters.return_value = {"min_qty": "0.001"}
    mock.fetch_mark_price_data.return_value = (51000.0, 123)
    # Dispatch methods must never be touched by the facade.
    mock.create_entry_market_order.side_effect = AssertionError("dispatch called")
    mock.create_exit_market_order.side_effect = AssertionError("dispatch called")
    mock.cancel_order.side_effect = AssertionError("cancel called")
    svc._clients["BTCUSDT"] = mock

    assert svc.balance("BTCUSDT")["equity"] == 1000.0
    assert svc.positions("BTCUSDT")[0]["side"] == "long"
    assert svc.open_orders("BTCUSDT") == []
    assert svc.rules("BTCUSDT")["read_only"] is True
    mock.create_entry_market_order.assert_not_called()
    mock.create_exit_market_order.assert_not_called()
    mock.cancel_order.assert_not_called()
    # Facade exposes no trading verbs.
    for name in ("create_order", "cancel_order", "execute", "place_order"):
        assert not hasattr(svc, name)


def test_engine_start_requires_testnet_credentials(store):
    creds = MagicMock(spec=CredentialService)
    creds.load.side_effect = CredentialsNotFoundError("none stored")
    engine = EngineService(creds, store, MagicMock(), MagicMock())
    with pytest.raises(CredentialsNotFoundError):
        engine.start("BTCUSDT")
    assert engine.status()["state"] == "STOPPED"


def test_daily_pnl_unavailable_without_verified_baseline(store, tmp_path):
    creds = CredentialService()
    svc = _TestnetAccountService(creds, store)
    journal = MagicMock()
    baseline = MagicMock()
    baseline.verified_status = "MISSING"
    journal.get_daily_baseline.return_value = None
    svc._clients["BTCUSDT"] = MagicMock()
    svc._journals["BTCUSDT"] = journal
    result = svc.daily_pnl("BTCUSDT")
    assert result["available"] is False
    assert result["value"] is None


def test_ui_state_roundtrip(tmp_path):
    path = tmp_path / "ui_state.json"
    s1 = UIStateStore(path=path)
    s1.update(watchlist=["SOLUSDT"], selected_symbol="SOLUSDT")
    s2 = UIStateStore(path=path)
    snap = s2.snapshot()
    assert snap["watchlist"] == ["SOLUSDT"]
    assert snap["selected_symbol"] == "SOLUSDT"
    # No credential material is ever part of UI state.
    assert "api_key" not in json.dumps(snap).lower()
    assert "secret" not in json.dumps(snap).lower()


def test_server_requires_token_and_serves_bootstrap():
    from noone.ui.server import AppContext, serve

    ctx = AppContext()
    server, port = serve(ctx, port=0)
    try:
        # No token -> 403
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/bootstrap", timeout=5)
            raise AssertionError("expected 403")
        except urllib.error.HTTPError as e:
            assert e.code == 403
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/bootstrap",
            headers={"X-NOONE-Token": ctx.token},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
        assert data["environment"] == "mainnet"
        assert data["selected_symbol"]
    finally:
        ctx.shutdown()
        server.shutdown()


def test_rules_endpoint_returns_safe_defaults_without_credentials(store):
    creds = MagicMock(spec=CredentialService)
    creds.status.return_value = {"mainnet": {"connected": False}, "testnet": {"connected": False}}
    market = MagicMock(spec=MarketService)
    market.public_filters.return_value = {
        "min_qty": "0.001",
        "max_qty": "1000.0",
        "step_size": "0.001",
        "min_notional": "5.0",
        "max_notional": "5000000.0",
        "tick_size": "0.1",
    }
    market.ticker.return_value = {"last_price": 78000.0}

    mainnet = MainnetAccountService(creds, market=market)
    mainnet_rules = mainnet.rules("BTCUSDT")
    assert mainnet_rules["read_only"] is True
    assert mainnet_rules["filters"]["min_qty"] == "0.001"
    assert mainnet_rules["take_profit_pct"] == 2.0
    assert mainnet_rules["stop_loss_pct"] == 1.0

    testnet = _TestnetAccountService(creds, store, market=market)
    testnet_rules = testnet.rules("BTCUSDT")
    assert testnet_rules["read_only"] is False
    assert testnet_rules["leverage"] == 10
    assert testnet_rules["take_profit_pct"] == 2.0
    assert testnet_rules["stop_loss_pct"] == 1.0

