"""Localhost HTTP API + production static serving for Component 6."""
from __future__ import annotations

import json
import logging
import mimetypes
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from noone.ui.paths import frontend_dist_dir
from noone.ui.services import (
    MARKET_SOURCE_LABEL,
    CredentialService,
    EngineService,
    MainnetAccountService,
    MarketService,
    TestnetAccountService,
)
from noone.ui.state_store import UIStateStore

log = logging.getLogger("noone.ui")


class AppContext:
    def __init__(self, dev_mode: bool = False) -> None:
        self.dev_mode = dev_mode
        self.store = UIStateStore()
        self.market = MarketService()
        self.credentials = CredentialService()
        self.mainnet = MainnetAccountService(self.credentials, self.market)
        self.testnet = TestnetAccountService(self.credentials, self.store, self.market)
        self.engine = EngineService(self.credentials, self.store, self.testnet, self.market)
        self.token = secrets.token_urlsafe(24)
        snap = self.store.snapshot()
        self.environment = "mainnet"
        self.selected_symbol = str(snap["selected_symbol"]).upper()

    # -- aggregated views -------------------------------------------------
    def summary(self) -> dict[str, Any]:
        env, sym = self.environment, self.selected_symbol
        cred = self.credentials.status()
        out: dict[str, Any] = {
            "environment": env,
            "symbol": sym,
            "market_source": MARKET_SOURCE_LABEL,
            "credentials": cred,
            "pnl": {"available": False, "value": None, "reason": "Mainnet daily PnL is unavailable via existing interfaces"},
            "balance": None,
            "balance_state": "not_connected",
            "orders": [],
            "positions": [],
            "orders_state": "not_connected",
            "long_count": 0,
            "short_count": 0,
        }
        connected = cred[env]["connected"]
        if not connected:
            return out
        try:
            if env == "mainnet":
                out["balance"] = self.mainnet.balance(sym)
                out["positions"] = self.mainnet.positions(sym)
                out["orders"] = self.mainnet.open_orders(sym)
            else:
                out["balance"] = self.testnet.balance(sym)
                out["positions"] = self.testnet.positions(sym)
                out["orders"] = self.testnet.open_orders(sym)
                out["pnl"] = self.testnet.daily_pnl(sym)
            out["balance_state"] = "ok"
            out["orders_state"] = "ok"
            for p in out["positions"]:
                if str(p.get("side")).lower() == "long":
                    out["long_count"] += 1
                elif str(p.get("side")).lower() == "short":
                    out["short_count"] += 1
        except Exception as exc:
            out["balance_state"] = "error"
            out["orders_state"] = "error"
            out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    def shutdown(self) -> None:
        try:
            self.engine.stop()
        except Exception:
            pass
        self.market.close()


def _json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def make_handler(ctx: AppContext):
    dist = frontend_dist_dir()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            log.debug(fmt, *args)

        # -- helpers --
        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, exc: Exception, status: int = 400) -> None:
            self._send_json(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=status
            )

        def _check_token(self) -> bool:
            req_token = self.headers.get("X-NOONE-Token")
            if req_token and req_token == ctx.token:
                return True
            if ctx.dev_mode and req_token == "__NOONE_TOKEN_VALUE__":
                return True
            self._send_json({"ok": False, "error": "unauthorized"}, status=403)
            return False

        # -- static / SPA --
        def _serve_static(self, url_path: str) -> None:
            rel = url_path.lstrip("/") or "index.html"
            target = (dist / rel).resolve()
            if not str(target).startswith(str(dist.resolve())) or not target.is_file():
                target = dist / "index.html"
            if not target.is_file():
                body = (
                    b"<html><body style='font-family:sans-serif;background:#0b0e13;color:#e6e9ee'>"
                    b"<h2>PROJECT NOONE</h2><p>Frontend build not found. "
                    b"Run: npm --prefix frontend run build</p></body></html>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            data = target.read_bytes()
            if target.name == "index.html":
                html = data.decode("utf-8").replace("__NOONE_TOKEN_VALUE__", ctx.token)
                data = html.encode("utf-8")
            ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            if target.suffix == ".js":
                ctype = "text/javascript"
            elif target.suffix == ".css":
                ctype = "text/css"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header(
                "Cache-Control",
                "no-cache" if target.name == "index.html" else "public, max-age=31536000, immutable",
            )
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        # -- API --
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)
            if not path.startswith("/api/"):
                self._serve_static(path)
                return
            if not self._check_token():
                return
            sym = (qs.get("symbol", [ctx.selected_symbol])[0] or ctx.selected_symbol).upper()
            env = ctx.environment
            try:
                if path == "/api/bootstrap":
                    snap = ctx.store.snapshot()
                    self._send_json(
                        {
                            "ok": True,
                            "environment": env,
                            "selected_symbol": ctx.selected_symbol,
                            "interval": snap["interval"],
                            "watchlist": snap["watchlist"],
                            "credentials": ctx.credentials.status(),
                            "market_source": MARKET_SOURCE_LABEL,
                            "engine": ctx.engine.status(),
                        }
                    )
                elif path == "/api/market/universe":
                    self._send_json({"ok": True, "items": ctx.market.universe(qs.get("q", [""])[0])})
                elif path == "/api/market/ticker":
                    self._send_json({"ok": True, **ctx.market.ticker(sym)})
                elif path == "/api/market/detail":
                    self._send_json({"ok": True, **ctx.market.detail(sym)})
                elif path == "/api/chart":
                    interval = qs.get("interval", ["15m"])[0]
                    self._send_json(
                        {"ok": True, "html": ctx.market.chart_html(sym, interval), "symbol": sym, "interval": interval}
                    )
                elif path == "/api/summary":
                    self._send_json({"ok": True, **ctx.summary()})
                elif path == "/api/rules":
                    rules = (
                        ctx.mainnet.rules(sym) if env == "mainnet" else ctx.testnet.rules(sym)
                    )
                    self._send_json({"ok": True, **rules})
                elif path == "/api/baseline":
                    if env != "testnet":
                        self._send_json({"ok": True, "available": False, "status": "UNAVAILABLE", "active_freezes": [], "reason": "Baseline mechanism exists only in Component 3 (Testnet)"})
                    else:
                        self._send_json({"ok": True, **ctx.testnet.baseline_status(sym)})
                elif path == "/api/tradelogs":
                    if env == "mainnet":
                        self._send_json({"ok": True, "available": False, "items": [], "reason": "Reliable Mainnet trade history is unavailable via existing interfaces (demo client swallows fetch errors)"})
                    elif not ctx.credentials.status()["testnet"]["connected"]:
                        self._send_json({"ok": True, "available": False, "items": [], "reason": "Testnet credentials are not provisioned"})
                    else:
                        self._send_json({"ok": True, "available": True, "items": ctx.testnet.trades(sym)})
                elif path == "/api/engine/status":
                    self._send_json({"ok": True, **ctx.engine.status()})
                else:
                    self._send_json({"ok": False, "error": "not found"}, status=404)
            except Exception as exc:
                self._error(exc)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            if not path.startswith("/api/"):
                self._send_json({"ok": False, "error": "not found"}, status=404)
                return
            if not self._check_token():
                return
            body = _json_body(self)
            try:
                if path == "/api/env":
                    env = str(body.get("environment", "")).lower()
                    if env not in ("mainnet", "testnet"):
                        raise ValueError("environment must be mainnet or testnet")
                    ctx.engine.stop()
                    ctx.environment = env
                    ctx.testnet.invalidate()
                    ctx.mainnet.invalidate()
                    self._send_json({"ok": True, "environment": env})
                elif path == "/api/select":
                    sym = str(body.get("symbol", "")).upper()
                    if not sym:
                        raise ValueError("symbol is required")
                    if ctx.engine.status()["state"] in ("RUNNING", "HELD"):
                        ctx.engine.stop()
                    ctx.selected_symbol = sym
                    snap = ctx.store.snapshot()
                    watchlist = list(snap["watchlist"])
                    if sym not in watchlist and len(watchlist) < 5:
                        watchlist.append(sym)
                    ctx.store.update(selected_symbol=sym, watchlist=watchlist)
                    self._send_json({"ok": True, "selected_symbol": sym, "watchlist": watchlist})
                elif path == "/api/watchlist":
                    symbols = [str(s).upper() for s in body.get("symbols", [])][:5]
                    snap = ctx.store.update(watchlist=symbols)
                    self._send_json({"ok": True, "watchlist": snap["watchlist"]})
                elif path == "/api/interval":
                    interval = str(body.get("interval", "15m"))
                    ctx.store.update(interval=interval)
                    self._send_json({"ok": True, "interval": interval})
                elif path == "/api/connect":
                    env = str(body.get("environment", ctx.environment)).lower()
                    # Secret is used once for DPAPI storage and never returned/logged.
                    ctx.credentials.connect(env, str(body.get("api_key", "")), str(body.get("api_secret", "")))
                    ctx.mainnet.invalidate()
                    ctx.testnet.invalidate()
                    self._send_json({"ok": True, "credentials": ctx.credentials.status()})
                elif path == "/api/disconnect":
                    env = str(body.get("environment", ctx.environment)).lower()
                    if env == "testnet":
                        ctx.engine.stop()
                    deleted = ctx.credentials.disconnect(env)
                    ctx.mainnet.invalidate()
                    ctx.testnet.invalidate()
                    self._send_json({"ok": True, "deleted": deleted, "credentials": ctx.credentials.status()})
                elif path == "/api/engine/start":
                    if ctx.environment != "testnet":
                        raise ValueError("Engine execution is Testnet-only; Mainnet is read-only")
                    self._send_json({"ok": True, **ctx.engine.start(ctx.selected_symbol)})
                elif path == "/api/engine/stop":
                    self._send_json({"ok": True, **ctx.engine.stop()})
                elif path == "/api/orders/cancel":
                    if ctx.environment != "testnet":
                        raise ValueError("Order cancellation is disabled on Mainnet (read-only)")
                    res = ctx.testnet.cancel_order(
                        ctx.selected_symbol,
                        order_id=str(body.get("order_id", "")),
                        client_order_id=str(body.get("client_order_id", "")),
                    )
                    self._send_json({"ok": True, **res})
                elif path == "/api/rules/apply":
                    if ctx.environment != "testnet":
                        raise ValueError("Rules cannot be applied on Mainnet (read-only)")
                    current = ctx.store.snapshot()["testnet_rules"]
                    updated = dict(current)
                    for key in (
                        "leverage",
                        "margin_mode",
                        "position_mode",
                        "margin_allocation_fraction",
                        "risk_per_trade_pct",
                        "max_position_notional",
                        "take_profit_pct",
                        "stop_loss_pct",
                        "max_daily_loss_pct",
                        "max_daily_profit_pct",
                    ):
                        if key in body:
                            updated[key] = body[key]
                    ctx.store.update(testnet_rules=updated)
                    ctx.testnet.invalidate()
                    self._send_json({"ok": True, "testnet_rules": updated, "note": "Takes effect at next engine start"})
                else:
                    self._send_json({"ok": False, "error": "not found"}, status=404)
            except Exception as exc:
                self._error(exc)

    return Handler


def serve(ctx: AppContext, port: int = 0) -> tuple[ThreadingHTTPServer, int]:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(ctx))
    actual_port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="noone-http")
    thread.start()
    log.info("NOONE backend listening on 127.0.0.1:%s", actual_port)
    return server, actual_port
