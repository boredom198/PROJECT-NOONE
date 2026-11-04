"""Component 6 backend services.

Thin UI-layer adapters over existing NOONE components. No protected
component is modified. Mainnet access is a read-only whitelist facade
over MainnetDemoClient; Testnet trading goes exclusively through
Component 3's ExchangeOrchestrator.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from noone.chart.widget import generate_tradingview_embed_html
from noone.market_data import BinanceFuturesCollector
from noone.market_data.models import utcnow
from noone.security.credentials import (
    delete_credentials,
    has_credentials,
    load_credentials,
    store_credentials,
)
from noone.ui.paths import default_model_path, testnet_journal_path
from noone.ui.state_store import UIStateStore

MARKET_SOURCE_LABEL = "Binance USD\u24c8-M Futures (public market data)"


def _f(value: Any) -> Optional[float]:
    return float(value) if value is not None else None


def _compute_rsi(closes: list[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 2)


def _compute_macd(closes: list[float]) -> Optional[dict[str, float]]:
    if len(closes) < 26:
        return None

    def _ema(vals: list[float], p: int) -> list[float]:
        k = 2.0 / (p + 1.0)
        res = [sum(vals[:p]) / p]
        for v in vals[p:]:
            res.append(v * k + res[-1] * (1.0 - k))
        return res

    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    offset = len(ema12) - len(ema26)
    macd_line = [e12 - e26 for e12, e26 in zip(ema12[offset:], ema26)]
    if len(macd_line) < 9:
        return {"macd": round(macd_line[-1], 2), "signal": 0.0, "hist": 0.0}
    sig_line = _ema(macd_line, 9)
    return {
        "macd": round(macd_line[-1], 2),
        "signal": round(sig_line[-1], 2),
        "hist": round(macd_line[-1] - sig_line[-1], 2),
    }


def _compute_atr(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> tuple[Optional[float], Optional[float]]:
    if len(closes) < period + 1 or len(highs) != len(closes) or len(lows) != len(closes):
        return None, None
    trs = [
        max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        for i in range(1, len(closes))
    ]
    if len(trs) < period:
        return None, None
    atr = sum(trs[-period:]) / period
    curr_price = closes[-1]
    vol_pct = (atr / curr_price * 100.0) if curr_price > 0 else None
    return round(atr, 4), (round(vol_pct, 2) if vol_pct is not None else None)


# ---------------------------------------------------------------------------
# Market data (Component 4, public, keyless)
# ---------------------------------------------------------------------------

class MarketService:
    def __init__(self) -> None:
        self._collector = BinanceFuturesCollector()
        self._lock = threading.Lock()
        self._universe: list[dict[str, Any]] | None = None
        self._universe_at = 0.0
        self._ticker_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._sparkline_cache: dict[str, tuple[float, list[float]]] = {}
        self._detail_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._klines_cache: dict[str, tuple[float, list[Any]]] = {}
        self._filters_cache: dict[str, dict[str, Any]] = {}

    def universe(self, query: str = "") -> list[dict[str, Any]]:
        now = time.time()
        with self._lock:
            if self._universe is None or now - self._universe_at > 3600:
                symbols = self._collector.exchange_info()
                self._universe = [
                    {
                        "symbol": s.symbol,
                        "base_asset": s.base_asset,
                        "quote_asset": s.quote_asset,
                        "status": s.status,
                    }
                    for s in symbols
                    if s.contract_type == "PERPETUAL"
                    and s.status == "TRADING"
                    and (s.quote_asset or "").upper() == "USDT"
                ]
                self._universe.sort(key=lambda x: x["symbol"])
                self._universe_at = now
            items = self._universe
        q = query.strip().upper()
        if q:
            items = [i for i in items if q in i["symbol"] or q in (i["base_asset"] or "")]
        return items[:100]

    def public_filters(self, symbol: str) -> dict[str, Any]:
        sym = symbol.upper()
        now = time.time()
        with self._lock:
            if sym in self._filters_cache:
                return dict(self._filters_cache[sym])
        try:
            # Leverage exchange info from collector
            symbols = self._collector.exchange_info()
            for s in symbols:
                if s.symbol.upper() == sym:
                    raw_filters = s.raw.get("filters", []) if hasattr(s, "raw") and isinstance(s.raw, dict) else []
                    from noone.exchange.precision import parse_symbol_filters
                    pf = parse_symbol_filters(raw_filters)
                    res = {
                        "min_qty": str(pf.min_qty),
                        "max_qty": str(pf.max_qty),
                        "step_size": str(pf.step_size),
                        "min_notional": str(pf.min_notional),
                        "max_notional": str(pf.max_notional) if pf.max_notional else None,
                        "tick_size": str(pf.tick_size) if pf.tick_size else None,
                    }
                    with self._lock:
                        self._filters_cache[sym] = res
                    return res
        except Exception:
            pass
        fallback = {
            "min_qty": "0.001",
            "max_qty": "1000.0",
            "step_size": "0.001",
            "min_notional": "5.0",
            "max_notional": "5000000.0",
            "tick_size": "0.1",
        }
        return fallback

    def ticker(self, symbol: str) -> dict[str, Any]:
        sym = symbol.upper()
        now = time.time()
        with self._lock:
            cached = self._ticker_cache.get(sym)
            if cached and now - cached[0] < 1.5:
                return dict(cached[1])

        with self._lock:
            t = self._collector.ticker_24h(sym)
            # Fetch closes for sparkline with 60s cache
            closes: list[float] = []
            cached_spark = self._sparkline_cache.get(sym)
            if cached_spark and now - cached_spark[0] < 60.0:
                closes = list(cached_spark[1])
            else:
                try:
                    closes = [
                        float(k.close)
                        for k in self._collector.klines(sym, interval="1h", limit=24)
                        if k.close is not None
                    ]
                    self._sparkline_cache[sym] = (now, closes)
                except Exception:
                    closes = list(cached_spark[1]) if cached_spark else []

        result = {
            "symbol": sym,
            "last_price": _f(t.last_price),
            "price_change": _f(t.price_change),
            "price_change_percent": _f(t.price_change_percent),
            "high_price": _f(t.high_price),
            "low_price": _f(t.low_price),
            "volume": _f(t.volume),
            "quote_volume": _f(t.quote_volume),
            "sparkline": closes,
            "source": MARKET_SOURCE_LABEL,
        }
        with self._lock:
            self._ticker_cache[sym] = (now, result)
        return result

    def detail(self, symbol: str) -> dict[str, Any]:
        sym = symbol.upper()
        now = time.time()
        with self._lock:
            cached = self._detail_cache.get(sym)
            if cached and now - cached[0] < 1.5:
                return dict(cached[1])

        with self._lock:
            snap = self._collector.snapshot(sym, interval="15m")
            # Fetch recent 15m klines for technical indicators
            klines = []
            cached_kl = self._klines_cache.get(sym)
            if cached_kl and now - cached_kl[0] < 20.0:
                klines = cached_kl[1]
            else:
                try:
                    klines = self._collector.klines(sym, interval="15m", limit=35)
                    self._klines_cache[sym] = (now, klines)
                except Exception:
                    klines = cached_kl[1] if cached_kl else []

        t, m, oi = snap["ticker_24h"], snap["mark_price"], snap["open_interest"]
        funding = snap["funding_history"]

        closes = [float(k.close) for k in klines if getattr(k, "close", None) is not None]
        highs = [float(k.high) for k in klines if getattr(k, "high", None) is not None]
        lows = [float(k.low) for k in klines if getattr(k, "low", None) is not None]

        rsi = _compute_rsi(closes)
        macd = _compute_macd(closes)
        atr, vol_pct = _compute_atr(highs, lows, closes)

        # Estimate slippage/spread
        last_p = _f(t.last_price)
        mark_p = _f(m.mark_price)
        slippage_pct = 0.02
        if last_p and mark_p and mark_p > 0:
            diff = abs(mark_p - last_p) / mark_p * 100.0
            slippage_pct = max(round(diff, 3), 0.01)

        result = {
            "symbol": sym,
            "last_price": last_p,
            "price_change_percent": _f(t.price_change_percent),
            "high_price": _f(t.high_price),
            "low_price": _f(t.low_price),
            "volume_24h": _f(t.volume),
            "quote_volume_24h": _f(t.quote_volume),
            "mark_price": mark_p,
            "index_price": _f(m.index_price),
            "funding_rate": _f(m.last_funding_rate),
            "next_funding_time": m.next_funding_time.isoformat() if m.next_funding_time else None,
            "open_interest": _f(oi.open_interest),
            "rsi": rsi,
            "macd": macd,
            "atr": atr,
            "volatility_pct": vol_pct,
            "slippage_pct": slippage_pct,
            "last_funding_settlement": (
                {
                    "rate": _f(funding[-1].funding_rate),
                    "time": funding[-1].funding_time.isoformat() if funding[-1].funding_time else None,
                }
                if funding
                else None
            ),
            "source": MARKET_SOURCE_LABEL,
        }
        with self._lock:
            self._detail_cache[sym] = (now, result)
        return result

    def chart_html(self, symbol: str, interval: str) -> str:
        return generate_tradingview_embed_html(symbol=symbol.upper(), interval=interval)

    def close(self) -> None:
        try:
            self._collector.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Credentials (Component 7 DPAPI) — never exposes secrets
# ---------------------------------------------------------------------------

class CredentialService:
    def status(self) -> dict[str, Any]:
        return {
            "mainnet": {"connected": has_credentials("mainnet")},
            "testnet": {"connected": has_credentials("testnet")},
        }

    def connect(self, environment: str, api_key: str, api_secret: str) -> None:
        env = environment.strip().lower()
        store_credentials(api_key, api_secret, environment=env)

    def disconnect(self, environment: str) -> bool:
        return delete_credentials(environment.strip().lower())

    def load(self, environment: str) -> tuple[str, str]:
        # Explicit Component 7 rule: no environment-variable fallback.
        return load_credentials(environment.strip().lower(), allow_env_fallback=False)


# ---------------------------------------------------------------------------
# Mainnet read-only facade (approved whitelist over MainnetDemoClient)
# ---------------------------------------------------------------------------

class MainnetAccountService:
    """Exposes ONLY balance/positions/open-orders/mark/filters reads."""

    def __init__(self, credentials: CredentialService, market: MarketService | None = None) -> None:
        self._credentials = credentials
        self._market = market or MarketService()
        self._lock = threading.Lock()
        self._clients: dict[str, Any] = {}

    def _client(self, symbol: str):
        sym = symbol.upper()
        with self._lock:
            if sym in self._clients:
                return self._clients[sym]
            from noone.mainnet_demo.client import MainnetDemoClient
            from noone.mainnet_demo.config import MainnetDemoConfig

            api_key, api_secret = self._credentials.load("mainnet")
            base = sym[:-4] if sym.endswith("USDT") else sym
            config = MainnetDemoConfig(symbol=f"{base}/USDT:USDT", raw_symbol=sym)
            client = MainnetDemoClient(
                config, api_key=api_key, api_secret=api_secret, journal=None
            )
            self._clients[sym] = client
            return client

    def invalidate(self) -> None:
        with self._lock:
            self._clients.clear()

    def balance(self, symbol: str) -> dict[str, Any]:
        client = self._client(symbol)
        bal = client.fetch_balance()
        return {
            "equity": float(bal["equity"]),
            "available_balance": float(bal["available_balance"]),
            "margin_used": float(bal["margin_used"]),
            "environment": "mainnet",
        }

    def positions(self, symbol: str) -> list[dict[str, Any]]:
        client = self._client(symbol)
        rows = client.fetch_positions([client.config.symbol])
        out = []
        for p in rows:
            contracts = float(p.get("contracts") or 0.0)
            if abs(contracts) <= 1e-12 and float(p.get("positionAmt") or 0) == 0:
                continue
            out.append(
                {
                    "symbol": p.get("symbol") or symbol.upper(),
                    "side": p.get("side"),
                    "contracts": contracts,
                    "entry_price": _f(p.get("entryPrice")),
                    "mark_price": _f(p.get("markPrice")),
                    "unrealized_pnl": _f(p.get("unrealizedPnl")),
                    "leverage": p.get("leverage"),
                    "margin_mode": p.get("marginMode"),
                }
            )
        return out

    def open_orders(self, symbol: str) -> list[dict[str, Any]]:
        client = self._client(symbol)
        rows = client.fetch_open_orders(client.config.symbol)
        return [
            {
                "id": str(o.get("id") or o.get("orderId") or ""),
                "symbol": o.get("symbol") or symbol.upper(),
                "side": o.get("side"),
                "type": o.get("type"),
                "price": _f(o.get("price")),
                "amount": _f(o.get("amount")),
                "filled": _f(o.get("filled")),
                "status": o.get("status"),
                "datetime": o.get("datetime"),
            }
            for o in rows
        ]

    def rules(self, symbol: str) -> dict[str, Any]:
        sym = symbol.upper()
        # If client already mocked/cached in _clients, use it (preserves test_mainnet_facade_whitelist_reads_only)
        with self._lock:
            has_client = sym in self._clients
        if has_client:
            client = self._client(sym)
            filters = client.load_market_filters(client.config.symbol)
            mark, _ts = client.fetch_mark_price_data(sym)
            return {
                "symbol": sym,
                "environment": "mainnet",
                "read_only": True,
                "mark_price": float(mark),
                "filters": {k: str(v) for k, v in filters.items()},
                "leverage": 1,
                "margin_mode": "Isolated",
                "position_mode": "One-way",
                "risk_per_trade_pct": 0.5,
                "margin_allocation_fraction": 0.20,
                "max_position_notional": 50000.0,
                "take_profit_pct": 2.0,
                "stop_loss_pct": 1.0,
                "max_daily_loss_pct": 0.01,
                "max_daily_profit_pct": 0.02,
                "note": "Mainnet is strictly read-only. Displaying configured risk constraints and exchange filters.",
            }

        # If credentials exist, try loading via client
        if self._credentials.status()["mainnet"]["connected"]:
            try:
                client = self._client(sym)
                filters = client.load_market_filters(client.config.symbol)
                mark, _ts = client.fetch_mark_price_data(sym)
                return {
                    "symbol": sym,
                    "environment": "mainnet",
                    "read_only": True,
                    "mark_price": float(mark),
                    "filters": {k: str(v) for k, v in filters.items()},
                    "leverage": 1,
                    "margin_mode": "Isolated",
                    "position_mode": "One-way",
                    "risk_per_trade_pct": 0.5,
                    "margin_allocation_fraction": 0.20,
                    "max_position_notional": 50000.0,
                    "take_profit_pct": 2.0,
                    "stop_loss_pct": 1.0,
                    "max_daily_loss_pct": 0.01,
                    "max_daily_profit_pct": 0.02,
                    "note": "Mainnet is strictly read-only. Displaying configured risk constraints and exchange filters.",
                }
            except Exception:
                pass

        # Public fallback: retrieve real public filters without credentials
        filters = self._market.public_filters(sym)
        mark_price = None
        try:
            t = self._market.ticker(sym)
            mark_price = t.get("last_price")
        except Exception:
            pass
        return {
            "symbol": sym,
            "environment": "mainnet",
            "read_only": True,
            "mark_price": mark_price,
            "filters": filters,
            "leverage": 1,
            "margin_mode": "Isolated",
            "position_mode": "One-way",
            "risk_per_trade_pct": 0.5,
            "margin_allocation_fraction": 0.20,
            "max_position_notional": 50000.0,
            "take_profit_pct": 2.0,
            "stop_loss_pct": 1.0,
            "max_daily_loss_pct": 0.01,
            "max_daily_profit_pct": 0.02,
            "note": "Mainnet is strictly read-only. Displaying configured risk constraints and exchange filters.",
        }


# ---------------------------------------------------------------------------
# Testnet account (Component 3) — trading only via ExchangeOrchestrator
# ---------------------------------------------------------------------------

def _testnet_config(symbol: str, store: UIStateStore):
    rules = store.snapshot()["testnet_rules"]
    try:
        from noone.exchange.config import Component3Config

        return Component3Config(
            testnet=True,
            symbol=symbol.upper(),
            leverage=int(rules.get("leverage", 10)),
            margin_allocation_fraction=float(rules.get("margin_allocation_fraction", 0.10)),
            max_position_notional=float(rules.get("max_position_notional", 50000.0)),
            max_daily_loss_pct=float(rules.get("max_daily_loss_pct", 0.03)),
            database_path=testnet_journal_path(),
        )
    except Exception:
        from types import SimpleNamespace

        return SimpleNamespace(
            testnet=True,
            symbol=symbol.upper(),
            leverage=int(rules.get("leverage", 10)),
            margin_type="Cross",
            margin_allocation_fraction=float(rules.get("margin_allocation_fraction", 0.10)),
            max_position_notional=float(rules.get("max_position_notional", 50000.0)),
            max_daily_loss_pct=float(rules.get("max_daily_loss_pct", 0.03)),
            database_path=testnet_journal_path(),
        )


class TestnetAccountService:
    def __init__(self, credentials: CredentialService, store: UIStateStore, market: MarketService | None = None) -> None:
        self._credentials = credentials
        self._store = store
        self._market = market or MarketService()
        self._lock = threading.Lock()
        self._clients: dict[str, Any] = {}
        self._journals: dict[str, Any] = {}
        self.transfers_synced = False

    def _client(self, symbol: str):
        sym = symbol.upper()
        with self._lock:
            if sym in self._clients:
                return self._clients[sym]
            from noone.exchange.client import BinanceFuturesClient

            config = _testnet_config(sym, self._store)
            api_key, api_secret = self._credentials.load("testnet")
            client = BinanceFuturesClient(config, api_key=api_key, api_secret=api_secret)
            self._clients[sym] = client
            if sym not in self._journals:
                from noone.exchange.journal import ExecutionJournal
                self._journals[sym] = ExecutionJournal(config.database_path)
            return client

    def journal(self, symbol: str):
        sym = symbol.upper()
        with self._lock:
            if sym in self._journals:
                return self._journals[sym]
            from noone.exchange.journal import ExecutionJournal

            config = _testnet_config(sym, self._store)
            journal = ExecutionJournal(config.database_path)
            self._journals[sym] = journal
            return journal

    def invalidate(self) -> None:
        with self._lock:
            self._clients.clear()
            self._journals.clear()

    def balance(self, symbol: str) -> dict[str, Any]:
        client = self._client(symbol)
        bal = client.fetch_account_balance()
        return {**bal, "environment": "testnet"}

    def positions(self, symbol: str) -> list[dict[str, Any]]:
        client = self._client(symbol)
        rows = client.fetch_position_risk_v2(symbol.upper())
        out = []
        for p in rows:
            amt = float(p.get("positionAmt") or 0.0)
            if abs(amt) <= 1e-12:
                continue
            out.append(
                {
                    "symbol": p.get("symbol") or symbol.upper(),
                    "side": "long" if amt > 0 else "short",
                    "contracts": abs(amt),
                    "entry_price": _f(p.get("entryPrice")),
                    "mark_price": _f(p.get("markPrice")),
                    "unrealized_pnl": _f(p.get("unRealizedProfit")),
                    "leverage": p.get("leverage"),
                    "margin_mode": p.get("marginType"),
                }
            )
        return out

    def open_orders(self, symbol: str) -> list[dict[str, Any]]:
        client = self._client(symbol)
        rows = client.fetch_open_orders(symbol.upper())
        out = []
        for o in rows:
            info = o.get("info") or {}
            out.append(
                {
                    "id": str(o.get("id") or info.get("orderId") or ""),
                    "client_order_id": str(
                        o.get("clientOrderId") or info.get("clientOrderId") or ""
                    ),
                    "symbol": o.get("symbol") or symbol.upper(),
                    "side": o.get("side"),
                    "type": o.get("type"),
                    "price": _f(o.get("price") or info.get("price")),
                    "amount": _f(o.get("amount") or info.get("origQty")),
                    "filled": _f(o.get("filled") or info.get("executedQty")),
                    "status": o.get("status") or info.get("status"),
                    "datetime": o.get("datetime"),
                }
            )
        return out

    def cancel_order(self, symbol: str, order_id: str = "", client_order_id: str = "") -> dict[str, Any]:
        client = self._client(symbol)
        res = client.cancel_order(
            order_id=order_id or None,
            symbol=symbol.upper(),
            client_order_id=client_order_id or None,
        )
        return {"cancelled": True, "id": str(res.get("id") or order_id), "status": res.get("status")}

    def trades(self, symbol: str, limit: int = 100) -> list[dict[str, Any]]:
        client = self._client(symbol)
        rows = client.fetch_my_trades(symbol.upper(), limit=limit)
        out = []
        for t in rows:
            info = t.get("info") or {}
            out.append(
                {
                    "id": str(t.get("id") or info.get("tradeId") or ""),
                    "order_id": str(t.get("order") or info.get("orderId") or ""),
                    "symbol": t.get("symbol") or symbol.upper(),
                    "side": t.get("side"),
                    "price": _f(t.get("price")),
                    "amount": _f(t.get("amount")),
                    "cost": _f(t.get("cost")),
                    "realized_pnl": _f(info.get("realizedPnl")),
                    "commission": _f((t.get("fee") or {}).get("cost")),
                    "datetime": t.get("datetime"),
                    "timestamp": t.get("timestamp"),
                }
            )
        return out

    def rules(self, symbol: str) -> dict[str, Any]:
        sym = symbol.upper()
        cfg = _testnet_config(sym, self._store)
        store_snap = self._store.snapshot()
        testnet_rules = store_snap.get("testnet_rules", {})
        filters = {}

        with self._lock:
            has_client = sym in self._clients
        if has_client or self._credentials.status()["testnet"]["connected"]:
            try:
                client = self._client(sym)
                from noone.exchange.precision import parse_symbol_filters

                info = client.fetch_exchange_info(sym)
                target = None
                for s in info.get("symbols", []):
                    if str(s.get("symbol", "")).upper() == sym:
                        target = s
                        break
                if target:
                    f = parse_symbol_filters(target.get("filters", []))
                    filters = {
                        "min_qty": str(f.min_qty),
                        "max_qty": str(f.max_qty),
                        "step_size": str(f.step_size),
                        "min_notional": str(f.min_notional),
                        "max_notional": str(f.max_notional) if f.max_notional else None,
                        "tick_size": str(f.tick_size) if f.tick_size else None,
                    }
            except Exception:
                pass

        if not filters:
            filters = self._market.public_filters(sym)

        mark_price = None
        try:
            t = self._market.ticker(sym)
            mark_price = t.get("last_price")
        except Exception:
            pass

        return {
            "symbol": sym,
            "environment": "testnet",
            "read_only": False,
            "mark_price": mark_price,
            "filters": filters,
            "leverage": int(testnet_rules.get("leverage", cfg.leverage)),
            "margin_mode": str(testnet_rules.get("margin_mode", cfg.margin_type)),
            "position_mode": str(testnet_rules.get("position_mode", "One-way")),
            "risk_per_trade_pct": float(testnet_rules.get("risk_per_trade_pct", 0.5)),
            "margin_allocation_fraction": float(testnet_rules.get("margin_allocation_fraction", cfg.margin_allocation_fraction)),
            "max_position_notional": float(testnet_rules.get("max_position_notional", cfg.max_position_notional)),
            "take_profit_pct": float(testnet_rules.get("take_profit_pct", 2.0)),
            "stop_loss_pct": float(testnet_rules.get("stop_loss_pct", 1.0)),
            "max_daily_loss_pct": float(testnet_rules.get("max_daily_loss_pct", cfg.max_daily_loss_pct)),
            "max_daily_profit_pct": float(testnet_rules.get("max_daily_profit_pct", 0.02)),
            "note": "Applied values take effect at next Testnet engine start.",
        }

    def baseline_status(self, symbol: str) -> dict[str, Any]:
        journal = self.journal(symbol)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        baseline = journal.get_daily_baseline(today)
        freezes = journal.get_active_safety_freezes()
        return {
            "date_utc": today,
            "status": baseline.verified_status if baseline else "MISSING",
            "baseline_equity": baseline.baseline_equity if baseline else None,
            "active_freezes": [
                {"reason": fr.reason, "details": fr.details, "triggered_at": fr.triggered_at}
                for fr in freezes
            ],
        }

    def daily_pnl(self, symbol: str) -> dict[str, Any]:
        """Read-only PnL: equity - net_external_transfers - baseline.

        UNAVAILABLE unless baseline is VERIFIED and transfer sync succeeded.
        Never calls RiskGate.check_daily_pnl_limit (freeze side effects).
        """
        journal = self.journal(symbol)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        baseline = journal.get_daily_baseline(today)
        if baseline is None or baseline.verified_status != "VERIFIED_UTC_BASELINE":
            return {
                "available": False,
                "value": None,
                "reason": "Daily baseline is not VERIFIED_UTC_BASELINE",
                "baseline_status": baseline.verified_status if baseline else "MISSING",
            }
        if not self.transfers_synced:
            return {
                "available": False,
                "value": None,
                "reason": "External transfer synchronization has not succeeded this session",
                "baseline_status": baseline.verified_status,
            }
        from noone.exchange.orchestrator import ExchangeOrchestrator

        config = _testnet_config(symbol, self._store)
        orch = ExchangeOrchestrator(config, client=self._client(symbol), journal=journal)
        equity = orch.sync_account().equity
        midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        net = journal.get_net_external_transfers(int(midnight.timestamp() * 1000))
        return {
            "available": True,
            "value": equity - net - baseline.baseline_equity,
            "equity": equity,
            "net_external_transfers": net,
            "baseline_equity": baseline.baseline_equity,
            "baseline_status": baseline.verified_status,
        }


# ---------------------------------------------------------------------------
# Testnet execution engine (Component 1 inference -> Component 3 orchestrator)
# ---------------------------------------------------------------------------

class EngineService:
    """Single-worker Testnet loop. Mainnet construction is impossible here."""

    POLL_SECONDS = 20.0

    def __init__(self, credentials: CredentialService, store: UIStateStore,
                 testnet: TestnetAccountService, market: MarketService) -> None:
        self._credentials = credentials
        self._store = store
        self._testnet = testnet
        self._market = market
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._state = "STOPPED"  # STOPPED | RUNNING | HELD | ERROR
        self._symbol: str | None = None
        self._events: list[dict[str, Any]] = []
        self._last_error: str | None = None
        self._model_path = default_model_path()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "environment": "testnet",
                "symbol": self._symbol,
                "model_path": str(self._model_path),
                "model_available": self._model_path.exists(),
                "last_error": self._last_error,
                "events": list(self._events[-30:]),
            }

    def _log(self, kind: str, message: str, **extra: Any) -> None:
        with self._lock:
            self._events.append(
                {"ts": utcnow().isoformat(), "kind": kind, "message": message, **extra}
            )
            self._events = self._events[-200:]

    def start(self, symbol: str) -> dict[str, Any]:
        sym = symbol.upper()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise ValueError("Engine is already running; stop it before starting again")
        # Raises CredentialsNotFoundError when Testnet is not provisioned.
        self._credentials.load("testnet")
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self._model_path}")
        self.stop()
        self._stop.clear()
        with self._lock:
            self._symbol = sym
            self._state = "STARTING"
            self._last_error = None
        self._thread = threading.Thread(target=self._run, args=(sym,), daemon=True, name="noone-engine")
        self._thread.start()
        self._log("engine", f"Engine started for {sym} (Testnet)")
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self._state = "STOPPING"
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10)
        with self._lock:
            if self._state != "ERROR":
                self._state = "STOPPED"
            self._thread = None
        return self.status()

    def _run(self, symbol: str) -> None:
        from noone.exchange.journal import ExecutionJournal
        from noone.exchange.orchestrator import ExchangeOrchestrator
        from noone.market_data import BinanceFuturesCollector
        from noone.rl_engine.inference import ModelInference

        try:
            config = _testnet_config(symbol, self._store)
            client = self._testnet._client(symbol)
            journal = ExecutionJournal(config.database_path)
            orch = ExchangeOrchestrator(config, client=client, journal=journal)
            inference = ModelInference.load(self._model_path, allow_legacy_metadata=True)
            collector = BinanceFuturesCollector()
            self._log("engine", "Pre-flight handshake…")
            try:
                orch.pre_flight_handshake()
                self._log("engine", "Pre-flight handshake passed")
            except Exception as exc:  # handshake failure holds entries, loop stays observable
                self._log("held", f"Pre-flight held: {type(exc).__name__}: {exc}")
            baseline = journal.get_daily_baseline(
                datetime.now(timezone.utc).strftime("%Y-%m-%d")
            )
            self._log(
                "baseline",
                f"Daily baseline: {baseline.verified_status if baseline else 'MISSING'}",
            )
        except Exception as exc:
            with self._lock:
                self._state = "ERROR"
                self._last_error = f"{type(exc).__name__}: {exc}"
            self._log("error", self._last_error or str(exc))
            return

        while not self._stop.is_set():
            try:
                now = datetime.now(timezone.utc)
                # Preserve the existing 5-second UTC sampling behaviour.
                if now.hour == 0 and now.minute == 0 and now.second <= 5:
                    try:
                        orch.sample_utc_midnight_baseline()
                    except Exception:
                        pass
                history = collector.historical_klines(
                    symbol, interval="15m", total_limit=100, only_closed=True
                )
                account = orch.sync_account()
                pred = inference.predict(candles=list(history.candles), account=account)
                order = orch.execute_action(pred.action)
                # Only a completed entry path inside execute_action has
                # performed transfer sync (HOLD/exits do not).
                if order is not None and pred.action.name.startswith("OPEN"):
                    self._testnet.transfers_synced = True
                self._log(
                    "prediction",
                    f"{pred.action.name} (q_margin={pred.q_margin:.4f})",
                    action=pred.action.name,
                    order_state=(order.state if order is not None else None),
                )
                with self._lock:
                    self._state = "RUNNING"
            except Exception as exc:
                with self._lock:
                    self._state = "HELD"
                    self._last_error = f"{type(exc).__name__}: {exc}"
                self._log("held", self._last_error or str(exc))
            self._stop.wait(self.POLL_SECONDS)
        try:
            collector.close()
        except Exception:
            pass
        with self._lock:
            self._state = "STOPPED"
        self._log("engine", "Engine stopped")


