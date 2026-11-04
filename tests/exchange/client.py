"""CCXT Binance USD(S)-M Futures Client Wrapper for Component 3.

Encapsulates pinned CCXT 4.4.52 exchange operations, testnet isolation,
implicit method dispatches, error wrapping, and transport-level boundaries.
Conforms strictly to PROJECT NOONE Technical Design Specification V2.2.4 Sections 4, 7, and 12.
"""

from decimal import Decimal
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import ccxt

from noone.exchange.config import Component3Config
from noone.exchange.errors import (
    ClockDriftError,
    ExchangeError,
    OrderRecoveryError,
)


class BinanceFuturesClient:
    """Encapsulates CCXT Binance USD(S)-M Futures client and low-level API dispatches."""

    def __init__(
        self,
        config: Component3Config,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        exchange_instance: Optional[ccxt.binanceusdm] = None,
    ):
        self.config = config
        self._exchange = exchange_instance or self._create_exchange(api_key, api_secret)

    def _create_exchange(
        self, api_key: Optional[str], api_secret: Optional[str]
    ) -> ccxt.binanceusdm:
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
        if self.config.testnet:
            ex.set_sandbox_mode(True)

        return ex

    @property
    def exchange(self) -> ccxt.binanceusdm:
        return self._exchange

    # -------------------------------------------------------------------------
    # Server Time & Clock Drift
    # -------------------------------------------------------------------------

    def fetch_time(self) -> int:
        """Fetch exchange server time in milliseconds."""
        try:
            res = self._exchange.fapiPublicGetTime()
            return int(res.get("serverTime", 0))
        except Exception as e:
            raise ExchangeError(f"Failed to fetch exchange server time: {e}") from e

    def verify_clock_drift(self) -> int:
        """Verify clock drift between local system and exchange server.

        Raises ClockDriftError if |server_time - local_time| >= threshold_ms.
        Returns drift in milliseconds.
        """
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
    # Account & Position Endpoints (V2)
    # -------------------------------------------------------------------------

    def fetch_account_v2(self) -> Dict[str, Any]:
        """Fetch private V2 account information (GET /fapi/v2/account)."""
        try:
            return self._exchange.fapiPrivateV2GetAccount()
        except Exception as e:
            raise ExchangeError(f"Failed to fetch V2 account data: {e}") from e

    def fetch_position_risk_v2(
        self, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch private V2 position risk records (GET /fapi/v2/positionRisk)."""
        try:
            params: Dict[str, Any] = {}
            if symbol is not None:
                raw_sym = symbol.upper().replace("/", "").replace(":USDT", "").replace(":", "")
                params["symbol"] = raw_sym
            res = self._exchange.fapiPrivateV2GetPositionRisk(params)
            if isinstance(res, list):
                return res
            return [res] if isinstance(res, dict) else []
        except Exception as e:
            raise ExchangeError(f"Failed to fetch V2 position risk data: {e}") from e

    def fetch_position_side_dual(self) -> Dict[str, Any]:
        """Check dual-side position mode (GET /fapi/v1/positionSide/dual)."""
        try:
            return self._exchange.fapiPrivateGetPositionSideDual()
        except Exception as e:
            raise ExchangeError(f"Failed to fetch position mode: {e}") from e

    # -------------------------------------------------------------------------
    # Open Orders & Order Management
    # -------------------------------------------------------------------------

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch open resting orders across account or for a symbol (GET /fapi/v1/openOrders)."""
        try:
            params: Dict[str, Any] = {}
            if symbol is not None:
                raw_sym = symbol.upper().replace("/", "").replace(":USDT", "").replace(":", "")
                params["symbol"] = raw_sym
            res = self._exchange.fapiPrivateGetOpenOrders(params)
            return res if isinstance(res, list) else []
        except Exception as e:
            raise ExchangeError(f"Failed to fetch open orders: {e}") from e

    def fetch_order_by_client_id(
        self, client_order_id: str, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch order by clientOrderId using unified fetch_order method."""
        sym = symbol or self.config.ccxt_symbol
        try:
            return self._exchange.fetch_order(
                id=None,
                symbol=sym,
                params={"origClientOrderId": client_order_id},
            )
        except ccxt.OrderNotFound as e:
            raise OrderRecoveryError(f"Order {client_order_id} not found on exchange.") from e
        except Exception as e:
            raise ExchangeError(f"Failed to fetch order {client_order_id}: {e}") from e

    def create_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        client_order_id: str,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """Create a market order with deterministic clientOrderId."""
        params: Dict[str, Any] = {
            "newClientOrderId": client_order_id,
        }
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
            raise ExchangeError(f"Exchange create_order failed for {client_order_id}: {e}") from e

    def cancel_order_by_client_id(
        self, client_order_id: str, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """Cancel an open order by origClientOrderId."""
        sym = symbol or self.config.ccxt_symbol
        try:
            return self._exchange.cancel_order(
                id=None,
                symbol=sym,
                params={"origClientOrderId": client_order_id},
            )
        except Exception as e:
            raise ExchangeError(f"Failed to cancel order {client_order_id}: {e}") from e

    # -------------------------------------------------------------------------
    # User Trades & Income History
    # -------------------------------------------------------------------------

    def fetch_my_trades(
        self, symbol: Optional[str] = None, since: Optional[int] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch recent user trade executions."""
        sym = symbol or self.config.ccxt_symbol
        try:
            return self._exchange.fetch_my_trades(symbol=sym, since=since, limit=limit)
        except Exception as e:
            raise ExchangeError(f"Failed to fetch user trades: {e}") from e

    def fetch_income(
        self,
        income_type: str = "TRANSFER",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 100,
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

    # -------------------------------------------------------------------------
    # Market Metadata & Mark Price
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

    def fetch_mark_price(self, symbol: Optional[str] = None) -> float:
        """Fetch current mark price (GET /fapi/v1/premiumIndex)."""
        try:
            raw_sym = (symbol or self.config.symbol).upper().replace("/", "").replace(":USDT", "").replace(":", "")
            res = self._exchange.fapiPublicGetPremiumIndex({"symbol": raw_sym})
            mark_price = res.get("markPrice")
            if mark_price is None:
                raise ExchangeError(f"markPrice missing in premiumIndex response for {raw_sym}")
            return float(mark_price)
        except Exception as e:
            raise ExchangeError(f"Failed to fetch mark price: {e}") from e

