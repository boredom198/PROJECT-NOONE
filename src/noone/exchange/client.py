"""CCXT Binance USD(S)-M Futures Client Wrapper for Component 3.

Encapsulates pinned CCXT 4.4.52 exchange operations for Binance Futures Testnet.
Directly implements the 6 core capabilities:
1. Connect to Binance USD(S)-M Futures Testnet (sandbox mode) & load markets.
2. Retrieve account balance and available margin.
3. Retrieve current open positions.
4. Set leverage and margin mode.
5. Open and close long/short positions (with reduceOnly).
6. Retrieve order status and executed quantities.
"""

from decimal import Decimal
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import ccxt

from noone.exchange.config import Component3Config
from noone.exchange.errors import (
    ClockDriftError,
    ExchangeError,
    FilterValidationError,
    OrderRecoveryError,
)

VALID_ORDER_SIDES = ("buy", "sell")
VALID_ORDER_TYPES = ("market", "limit")


def validate_order_parameters(
    side: str,
    amount: float,
    order_type: Optional[str] = None,
) -> None:
    """Validate basic order parameters before dispatch.

    Rejects:
    - Zero or negative quantities.
    - NaN and infinite quantities.
    - Invalid order sides.
    - Invalid order types where applicable.
    """
    if not isinstance(side, str) or side.lower() not in VALID_ORDER_SIDES:
        raise FilterValidationError(
            f"Invalid order side '{side}'. Must be one of {VALID_ORDER_SIDES}."
        )

    if order_type is not None:
        if not isinstance(order_type, str) or order_type.lower() not in VALID_ORDER_TYPES:
            raise FilterValidationError(
                f"Invalid order type '{order_type}'. Must be one of {VALID_ORDER_TYPES}."
            )

    if amount is None:
        raise FilterValidationError("Order quantity cannot be None.")

    try:
        amt = float(amount)
    except (TypeError, ValueError) as e:
        raise FilterValidationError(f"Invalid non-numeric order quantity: {amount}") from e

    if math.isnan(amt) or math.isinf(amt):
        raise FilterValidationError(f"Order quantity cannot be NaN or infinite: {amount}")

    if amt <= 0.0:
        raise FilterValidationError(f"Order quantity must be positive (> 0.0), got: {amount}")


PRODUCTION_BINANCE_HOSTS = (
    "fapi.binance.com",
    "dapi.binance.com",
    "api.binance.com",
    "sapi.binance.com",
    "eapi.binance.com",
    "papi.binance.com",
)


def verify_non_production_exchange(exchange_instance: Any) -> None:
    """Independently verify that an exchange instance is not pointing to production endpoints.

    Accepts the actual Binance USD(S)-M Futures sandbox endpoint (testnet.binancefuture.com)
    produced by CCXT 4.4.52 after set_sandbox_mode(True).
    Rejects any production endpoint.
    """
    if exchange_instance is None:
        return

    urls = getattr(exchange_instance, "urls", None)
    if isinstance(urls, dict):
        api_urls = urls.get("api", {})
        if isinstance(api_urls, dict):
            for endpoint_name, url_val in api_urls.items():
                if isinstance(url_val, str):
                    url_lower = url_val.lower()
                    for prod_host in PRODUCTION_BINANCE_HOSTS:
                        if prod_host in url_lower and "testnet" not in url_lower:
                            raise ExchangeError(
                                f"PRODUCTION EXECUTION HARD LOCKED: Exchange instance endpoint '{endpoint_name}' "
                                f"points to production Binance ({url_val})."
                            )

    hostname = getattr(exchange_instance, "hostname", "")
    if isinstance(hostname, str) and hostname:
        for prod_host in PRODUCTION_BINANCE_HOSTS:
            if prod_host in hostname.lower() and "testnet" not in hostname.lower():
                raise ExchangeError(
                    f"PRODUCTION EXECUTION HARD LOCKED: Exchange instance hostname points to production ({hostname})."
                )

    if isinstance(exchange_instance, ccxt.Exchange):
        fapi_private = None
        if isinstance(urls, dict) and isinstance(urls.get("api"), dict):
            fapi_private = urls["api"].get("fapiPrivate", "")
        if fapi_private and "testnet.binancefuture.com" not in fapi_private.lower():
            raise ExchangeError(
                f"PRODUCTION EXECUTION HARD LOCKED: Exchange instance fapiPrivate URL ({fapi_private}) "
                "is not a Binance Futures Testnet endpoint (testnet.binancefuture.com)."
            )


class BinanceFuturesClient:
    """Encapsulates CCXT Binance USD(S)-M Futures client for Testnet execution.

    Keeps CCXT exchange access strictly private inside the client; does not expose
    a raw exchange property.
    """

    def __init__(
        self,
        config: Component3Config,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        exchange_instance: Optional[Any] = None,
    ):
        self.config = config
        if not self.config.testnet or self.config.production_confirmed:
            raise ExchangeError(
                "PRODUCTION EXECUTION HARD LOCKED: Exchange client cannot operate in production mode."
            )

        if exchange_instance is not None:
            verify_non_production_exchange(exchange_instance)
            self._exchange = exchange_instance
            if hasattr(self._exchange, "set_sandbox_mode"):
                try:
                    self._exchange.set_sandbox_mode(True)
                except Exception:
                    pass
        else:
            self._exchange = self._create_exchange(api_key, api_secret)

    def _create_exchange(
        self, api_key: Optional[str], api_secret: Optional[str]
    ) -> ccxt.binanceusdm:
        if not self.config.testnet or self.config.production_confirmed:
            raise ExchangeError(
                "PRODUCTION EXECUTION HARD LOCKED: Exchange client cannot operate in production mode."
            )

        key = api_key or os.environ.get(self.config.api_key_env_var, "")
        secret = api_secret or os.environ.get(self.config.api_secret_env_var, "")

        exchange_params: Dict[str, Any] = {
            "apiKey": key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
            },
        }

        ex = ccxt.binanceusdm(exchange_params)
        ex.set_sandbox_mode(True)
        verify_non_production_exchange(ex)
        return ex

    # -------------------------------------------------------------------------
    # Core Capability 1: Connect to Testnet & Market Loading
    # -------------------------------------------------------------------------

    def load_markets(self, reload: bool = False) -> Dict[str, Any]:
        """Load and cache market structures from exchange."""
        try:
            return self._exchange.load_markets(reload)
        except Exception as e:
            raise ExchangeError(f"Failed to load markets from exchange: {e}") from e

    def get_market(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve unified market specification for target symbol."""
        sym = symbol or self.config.ccxt_symbol
        try:
            if not getattr(self._exchange, "markets", None):
                self.load_markets()
            return self._exchange.market(sym)
        except Exception as e:
            raise ExchangeError(f"Failed to get market metadata for {sym}: {e}") from e

    def fetch_time(self) -> int:
        """Fetch exchange server time in milliseconds."""
        try:
            if hasattr(self._exchange, "fetch_time"):
                return int(self._exchange.fetch_time())
            res = self._exchange.fapiPublicGetTime()
            return int(res.get("serverTime", 0))
        except Exception as e:
            raise ExchangeError(f"Failed to fetch exchange server time: {e}") from e

    def verify_clock_drift(self) -> int:
        """Verify clock drift between local system and exchange server."""
        local_before = int(time.time() * 1000)
        server_time = self.fetch_time()
        local_after = int(time.time() * 1000)
        local_midpoint = (local_before + local_after) // 2

        drift = abs(server_time - local_midpoint)
        if drift >= self.config.clock_drift_threshold_ms:
            raise ClockDriftError(
                f"Clock drift {drift}ms exceeds threshold {self.config.clock_drift_threshold_ms}ms "
                f"(server: {server_time}, local: {local_midpoint})"
            )
        return drift

    # -------------------------------------------------------------------------
    # Core Capability 2: Retrieve Balance & Available Margin
    # -------------------------------------------------------------------------

    def fetch_balance(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Retrieve unified account balance using standard CCXT fetch_balance."""
        try:
            p = {"type": "future"}
            if params:
                p.update(params)
            return self._exchange.fetch_balance(p)
        except Exception as e:
            raise ExchangeError(f"Failed to fetch account balance: {e}") from e

    def fetch_account_balance(self) -> Dict[str, float]:
        """Retrieve unified equity, available balance, and margin used."""
        bal = self.fetch_balance()
        total = bal.get("total", {}) or {}
        free = bal.get("free", {}) or {}
        used = bal.get("used", {}) or {}
        return {
            "equity": float(total.get("USDT", 0.0) or 0.0),
            "available_balance": float(free.get("USDT", 0.0) or 0.0),
            "margin_used": float(used.get("USDT", 0.0) or 0.0),
        }

    def fetch_account_v2(self) -> Dict[str, Any]:
        """Fetch private V2 account information (GET /fapi/v2/account)."""
        try:
            if hasattr(self._exchange, "fapiPrivateV2GetAccount"):
                return self._exchange.fapiPrivateV2GetAccount()
            return self.fetch_balance()
        except Exception as e:
            raise ExchangeError(f"Failed to fetch V2 account data: {e}") from e

    # -------------------------------------------------------------------------
    # Core Capability 3: Retrieve Open Positions
    # -------------------------------------------------------------------------

    def fetch_positions(
        self, symbols: Optional[List[str]] = None, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve open positions using standard CCXT fetch_positions."""
        try:
            syms = symbols if symbols is not None else [self.config.ccxt_symbol]
            return self._exchange.fetch_positions(syms, params=params or {})
        except Exception as e:
            raise ExchangeError(f"Failed to fetch positions: {e}") from e

    def fetch_position(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve target position dictionary for target symbol."""
        sym = symbol or self.config.ccxt_symbol
        positions = self.fetch_positions([sym])
        for p in positions:
            if p.get("symbol") == sym or p.get("info", {}).get("symbol") == self.config.raw_symbol:
                return p
        return positions[0] if positions else None

    def fetch_position_risk_v2(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch position risk records (GET /fapi/v2/positionRisk)."""
        try:
            if hasattr(self._exchange, "fapiPrivateV2GetPositionRisk"):
                params: Dict[str, Any] = {}
                if symbol is not None:
                    raw_sym = symbol.upper().replace("/", "").replace(":USDT", "").replace(":", "")
                    params["symbol"] = raw_sym
                res = self._exchange.fapiPrivateV2GetPositionRisk(params)
                if isinstance(res, list):
                    return res
                return [res] if isinstance(res, dict) else []
            return self.fetch_positions([symbol or self.config.ccxt_symbol])
        except Exception as e:
            raise ExchangeError(f"Failed to fetch V2 position risk data: {e}") from e

    def fetch_position_side_dual(self) -> Dict[str, Any]:
        """Check dual-side position mode (GET /fapi/v1/positionSide/dual)."""
        try:
            return self._exchange.fapiPrivateGetPositionSideDual()
        except Exception as e:
            raise ExchangeError(f"Failed to fetch position mode: {e}") from e

    # -------------------------------------------------------------------------
    # Core Capability 4: Set Leverage and Margin Mode
    # -------------------------------------------------------------------------

    def set_leverage(self, leverage: int, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Set leverage on the exchange using CCXT set_leverage."""
        sym = symbol or self.config.ccxt_symbol
        try:
            return self._exchange.set_leverage(leverage, sym)
        except Exception as e:
            raise ExchangeError(f"Failed to set leverage to {leverage} for {sym}: {e}") from e

    def set_margin_mode(self, margin_mode: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Set margin mode ('cross' or 'isolated') using CCXT set_margin_mode."""
        sym = symbol or self.config.ccxt_symbol
        try:
            return self._exchange.set_margin_mode(margin_mode.lower(), sym)
        except Exception as e:
            raise ExchangeError(f"Failed to set margin mode to {margin_mode} for {sym}: {e}") from e

    # -------------------------------------------------------------------------
    # Core Capability 5: Open & Close Positions
    # -------------------------------------------------------------------------

    def create_order(
        self,
        symbol: str,
        type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create order using CCXT unified create_order with parameter validation."""
        if not self.config.testnet or self.config.production_confirmed:
            raise ExchangeError("PRODUCTION EXECUTION HARD LOCKED: Order dispatch blocked.")
        verify_non_production_exchange(self._exchange)
        validate_order_parameters(side=side, amount=amount, order_type=type)

        p = params.copy() if params else {}
        sym = symbol or self.config.ccxt_symbol
        try:
            return self._exchange.create_order(
                symbol=sym,
                type=type.lower(),
                side=side.lower(),
                amount=amount,
                price=price,
                params=p,
            )
        except Exception as e:
            raise ExchangeError(f"Exchange create_order failed: {e}") from e

    def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        client_order_id: Optional[str] = None,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """Create a market order with deterministic clientOrderId and optional reduceOnly."""
        if not self.config.testnet or self.config.production_confirmed:
            raise ExchangeError("PRODUCTION EXECUTION HARD LOCKED: Order dispatch blocked.")
        verify_non_production_exchange(self._exchange)
        validate_order_parameters(side=side, amount=amount, order_type="market")

        params: Dict[str, Any] = {}
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        if reduce_only:
            params["reduceOnly"] = True

        sym = symbol or self.config.ccxt_symbol
        try:
            return self._exchange.create_order(
                symbol=sym,
                type="market",
                side=side.lower(),
                amount=amount,
                params=params,
            )
        except Exception as e:
            raise ExchangeError(f"Exchange create_order failed for {client_order_id or sym}: {e}") from e

    # -------------------------------------------------------------------------
    # Core Capability 6: Retrieve Order Status & Executed Quantities
    # -------------------------------------------------------------------------

    def fetch_order(
        self,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        client_order_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fetch order status using CCXT fetch_order with either orderId or origClientOrderId."""
        sym = symbol or self.config.ccxt_symbol
        p = params.copy() if params else {}
        if client_order_id:
            p["origClientOrderId"] = client_order_id

        try:
            return self._exchange.fetch_order(
                id=order_id,
                symbol=sym,
                params=p,
            )
        except ccxt.OrderNotFound as e:
            raise OrderRecoveryError(f"Order {order_id or client_order_id} not found on exchange.") from e
        except Exception as e:
            raise ExchangeError(f"Failed to fetch order {order_id or client_order_id}: {e}") from e

    def fetch_order_by_client_id(
        self, client_order_id: str, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """Convenience method to fetch order by origClientOrderId."""
        return self.fetch_order(client_order_id=client_order_id, symbol=symbol)

    def cancel_order(
        self,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        client_order_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Cancel an open order by ID or origClientOrderId."""
        sym = symbol or self.config.ccxt_symbol
        p = params.copy() if params else {}
        if client_order_id:
            p["origClientOrderId"] = client_order_id

        try:
            return self._exchange.cancel_order(
                id=order_id,
                symbol=sym,
                params=p,
            )
        except Exception as e:
            raise ExchangeError(f"Failed to cancel order {order_id or client_order_id}: {e}") from e

    def cancel_order_by_client_id(
        self, client_order_id: str, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """Convenience method to cancel an open order by origClientOrderId."""
        return self.cancel_order(client_order_id=client_order_id, symbol=symbol)

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch open resting orders across account or for a symbol."""
        sym = symbol or self.config.ccxt_symbol
        try:
            if hasattr(self._exchange, "fetch_open_orders"):
                return self._exchange.fetch_open_orders(sym)
            params: Dict[str, Any] = {}
            if symbol is not None:
                params["symbol"] = self.config.raw_symbol
            res = self._exchange.fapiPrivateGetOpenOrders(params)
            return res if isinstance(res, list) else []
        except Exception as e:
            try:
                params = {}
                if symbol is not None:
                    params["symbol"] = self.config.raw_symbol
                res = self._exchange.fapiPrivateGetOpenOrders(params)
                return res if isinstance(res, list) else []
            except Exception:
                raise ExchangeError(f"Failed to fetch open orders: {e}") from e

    def fetch_my_trades(
        self, symbol: Optional[str] = None, since: Optional[int] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch recent user trade executions."""
        sym = symbol or self.config.ccxt_symbol
        try:
            return self._exchange.fetch_my_trades(symbol=sym, since=since, limit=limit)
        except Exception as e:
            raise ExchangeError(f"Failed to fetch user trades: {e}") from e

    def fetch_my_trades_paginated(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        until: Optional[int] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Fetch trades with deduplication."""
        trades = self.fetch_my_trades(symbol=symbol, since=since, limit=limit)
        seen = set()
        deduped = []
        for t in trades:
            tid = str(t.get("id") or t.get("tradeId", ""))
            if tid not in seen:
                seen.add(tid)
                deduped.append(t)
        return deduped

    # -------------------------------------------------------------------------
    # Auxiliary Helpers
    # -------------------------------------------------------------------------

    def fetch_exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Fetch market trading rules and filters (GET /fapi/v1/exchangeInfo)."""
        try:
            params: Dict[str, Any] = {}
            if symbol is not None:
                raw_sym = symbol.upper().replace("/", "").replace(":USDT", "").replace(":", "")
                params["symbol"] = raw_sym
            return self._exchange.fapiPublicGetExchangeInfo(params)
        except Exception as e:
            raise ExchangeError(f"Failed to fetch exchange info: {e}") from e

    def fetch_leverage_bracket(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch symbol leverage brackets (GET /fapi/v1/leverageBracket)."""
        try:
            params: Dict[str, Any] = {}
            if symbol is not None:
                raw_sym = symbol.upper().replace("/", "").replace(":USDT", "").replace(":", "")
                params["symbol"] = raw_sym
            res = self._exchange.fapiPrivateGetLeverageBracket(params)
            return res if isinstance(res, list) else []
        except Exception as e:
            raise ExchangeError(f"Failed to fetch leverage bracket: {e}") from e

    def fetch_income(
        self,
        income_type: str = "TRANSFER",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Fetch income / transfer history (GET /fapi/v1/income)."""
        params: Dict[str, Any] = {
            "incomeType": income_type,
            "limit": limit,
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        try:
            res = self._exchange.fapiPrivateGetIncome(params)
            return res if isinstance(res, list) else []
        except Exception as e:
            raise ExchangeError(f"Failed to fetch income data: {e}") from e

    def fetch_income_paginated(
        self,
        income_type: str = "TRANSFER",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        page_size: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Simplified income pagination."""
        return self.fetch_income(income_type, start_time, end_time, page_size)

    def fetch_mark_price(self, symbol: Optional[str] = None) -> float:
        """Fetch current mark price."""
        price, _ = self.fetch_mark_price_data(symbol)
        return price

    def fetch_mark_price_data(self, symbol: Optional[str] = None) -> Tuple[float, int]:
        """Fetch current mark price and exchange source timestamp.

        Fails explicitly if the exchange response does not contain a source timestamp.
        Does not substitute local machine time.
        """
        try:
            raw_sym = (symbol or self.config.symbol).upper().replace("/", "").replace(":USDT", "").replace(":", "")
            res = self._exchange.fapiPublicGetPremiumIndex({"symbol": raw_sym})
            mark_price = res.get("markPrice")
            if mark_price is None:
                raise ExchangeError(f"markPrice missing in premiumIndex response for {raw_sym}")
            raw_time = res.get("time")
            if raw_time is None:
                raise ExchangeError(f"Exchange response lacks source timestamp 'time' for {raw_sym}")
            return float(mark_price), int(raw_time)
        except Exception as e:
            raise ExchangeError(f"Failed to fetch mark price: {e}") from e
