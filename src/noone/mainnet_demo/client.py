"""CCXT Binance USD(S)-M Futures Mainnet Client Wrapper for Mainnet Demo Mode.

Strictly encapsulates private exchange access, dynamic market filter resolution,
read-only preflight inspection, and validated order dispatch.
"""

from decimal import Decimal, ROUND_DOWN
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import ccxt

from noone.mainnet_demo.config import (
    MainnetDemoConfig,
    MainnetDemoDisarmedError,
    MainnetDemoError,
    MainnetDemoLockedError,
)


def quantize_down(val: Decimal, step: Decimal) -> Decimal:
    """Quantize a Decimal downward to a given step size."""
    if step <= Decimal("0"):
        return val
    multiples = (val / step).to_integral_value(rounding=ROUND_DOWN)
    quantized = multiples * step
    exp = step.as_tuple().exponent
    return quantized.quantize(Decimal(10) ** exp, rounding=ROUND_DOWN)


class MainnetDemoClient:
    """Mainnet execution client for Component Demo Mode."""

    def __init__(
        self,
        config: MainnetDemoConfig,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        exchange_instance: Optional[Any] = None,
        journal: Optional[Any] = None,
    ):
        self.config = config
        self.journal = journal
        if exchange_instance is not None:
            self._exchange = exchange_instance
        else:
            self._exchange = self._init_ccxt(api_key, api_secret)

    def _init_ccxt(self, api_key: Optional[str], api_secret: Optional[str]) -> ccxt.binanceusdm:
        key, secret = (api_key, api_secret) if api_key and api_secret else self.config.get_api_credentials()
        ex = ccxt.binanceusdm({
            "apiKey": key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
            },
        })
        # Explicitly ensure sandbox mode is NOT enabled; fail closed if unable to configure
        if hasattr(ex, "set_sandbox_mode"):
            try:
                ex.set_sandbox_mode(False)
            except Exception as e:
                raise MainnetDemoError(f"Failed to explicitly configure live sandbox mode: {e}") from e

        # Explicitly verify the exchange endpoint is Binance USD(S)-M Futures Mainnet
        fapi_url = ""
        if hasattr(ex, "urls") and isinstance(ex.urls, dict):
            api_info = ex.urls.get("api", {})
            if isinstance(api_info, dict):
                fapi_url = str(api_info.get("fapi", ""))
            elif isinstance(api_info, str):
                fapi_url = api_info

        if "testnet" in fapi_url.lower() or getattr(ex, "sandbox", False):
            raise MainnetDemoError(
                f"Exchange client is configured for testnet endpoint ({fapi_url}). "
                "Mainnet environment is strictly required. Aborting."
            )

        return ex

    def _check_dispatch_authorization(self, is_emergency: bool = False, reduce_only: bool = False) -> None:
        """Enforce authorization at the client order-dispatch boundary.

        Direct client calls must not bypass DISARMED or LOCKED state.
        Preserves a narrowly scoped emergency reduce-only exit mechanism.
        """
        if is_emergency:
            if not reduce_only:
                raise MainnetDemoError(
                    "Emergency order dispatch rejected: Emergency orders must be reduceOnly=True."
                )
            return

        if self.journal is not None and self.journal.is_locked():
            reason = self.journal.get_lock_reason() or "Lock reason unspecified"
            raise MainnetDemoLockedError(
                f"Order dispatch rejected at client boundary: Demo is permanently LOCKED ({reason})."
            )

        if not self.config.is_armed():
            raise MainnetDemoDisarmedError(
                "Order dispatch rejected at client boundary: Demo is DISARMED."
            )

    # -------------------------------------------------------------------------
    # Read-Only Preflight Inspection
    # -------------------------------------------------------------------------

    def inspect_preflight_account(self) -> Dict[str, Any]:
        """Perform strictly read-only audit of exchange account state.

        Fails closed without modifying any exchange settings if:
        - Account is in Hedge mode (must be One-Way mode).
        - Margin mode is not 'isolated'.
        - Leverage is not 1x.
        - Any open positions exist across account.
        - Any open orders exist across account.
        """
        results: Dict[str, Any] = {}

        # 1. Position Mode (One-Way mode check)
        try:
            dual_res = self._exchange.fapiPrivateGetPositionSideDual()
            dual_side = dual_res.get("dualSidePosition", True)
            if dual_side:
                raise MainnetDemoError(
                    "Account is configured in Hedge Mode (dualSidePosition=True). "
                    "One-Way Mode is strictly required. Aborting."
                )
            results["one_way_mode"] = True
        except Exception as e:
            if isinstance(e, MainnetDemoError):
                raise
            raise MainnetDemoError(f"Failed to inspect position mode: {e}") from e

        # 2. Positions & Margin/Leverage inspection
        raw_sym = self.config.raw_symbol
        try:
            positions_data = self._exchange.fapiPrivateV2GetPositionRisk()
            target_pos = None
            external_active = []

            for p in positions_data:
                sym = p.get("symbol", "").upper()
                amt = float(p.get("positionAmt", 0.0) or 0.0)
                if sym == raw_sym:
                    target_pos = p
                elif abs(amt) > 1e-8:
                    external_active.append(sym)

            if external_active:
                raise MainnetDemoError(
                    f"Conflicting active positions detected on non-target symbols: {external_active}. "
                    "Demo requires clean account with zero existing positions. Aborting."
                )

            if target_pos is None:
                raise MainnetDemoError(f"Target symbol {raw_sym} missing from position risk records.")

            curr_amt = float(target_pos.get("positionAmt", 0.0) or 0.0)
            if abs(curr_amt) > 1e-8:
                raise MainnetDemoError(
                    f"Existing open position on {raw_sym} detected (qty={curr_amt}). "
                    "Demo requires a flat account. Aborting."
                )

            # Leverage & Margin mode checks
            margin_type = str(target_pos.get("marginType", "")).lower()
            if margin_type != "isolated":
                raise MainnetDemoError(
                    f"Symbol {raw_sym} margin mode is '{margin_type}'. "
                    "Strictly 'isolated' margin mode is required. Aborting."
                )

            lev = int(target_pos.get("leverage", 0) or 0)
            if lev != 1:
                raise MainnetDemoError(
                    f"Symbol {raw_sym} leverage is {lev}x. Strictly 1x leverage is required. Aborting."
                )

            results["margin_mode"] = margin_type
            results["leverage"] = lev
            results["zero_positions"] = True
        except Exception as e:
            if isinstance(e, MainnetDemoError):
                raise
            raise MainnetDemoError(f"Failed to inspect positions/risk: {e}") from e

        # 3. Open Orders inspection
        try:
            open_orders = self.fetch_open_orders()
            if len(open_orders) > 0:
                raise MainnetDemoError(
                    f"Account has {len(open_orders)} open resting orders. "
                    "Demo requires zero open orders before execution. Aborting."
                )
            results["zero_open_orders"] = True
        except Exception as e:
            if isinstance(e, MainnetDemoError):
                raise
            raise MainnetDemoError(f"Failed to inspect open orders: {e}") from e

        return results

    # -------------------------------------------------------------------------
    # Dynamic Market Filters & Balance
    # -------------------------------------------------------------------------

    def load_market_filters(self, symbol: Optional[str] = None) -> Dict[str, Decimal]:
        """Dynamically retrieve and validate exchange trading rules and filters.

        Parses actual Binance LOT_SIZE, MARKET_LOT_SIZE, PRICE_FILTER and applicable
        notional filters without hardcoded fallbacks or interpreting decimal precision as step size.
        Aborts if reliable filters cannot be established.
        """
        sym = symbol or self.config.symbol
        try:
            if not getattr(self._exchange, "markets", None):
                self._exchange.load_markets()
            market = self._exchange.market(sym)

            min_qty: Optional[Decimal] = None
            max_qty: Optional[Decimal] = None
            step_size: Optional[Decimal] = None
            tick_size: Optional[Decimal] = None
            min_notional: Optional[Decimal] = None

            # 1. Parse from native Binance exchangeInfo filters if present
            raw_filters = market.get("info", {}).get("filters", [])
            if isinstance(raw_filters, list):
                for f in raw_filters:
                    ftype = str(f.get("filterType", "")).upper()
                    if ftype == "PRICE_FILTER":
                        ts = f.get("tickSize")
                        if ts is not None and float(ts) > 0:
                            tick_size = Decimal(str(ts))
                    elif ftype == "MARKET_LOT_SIZE":
                        mq = f.get("minQty")
                        mxq = f.get("maxQty")
                        ss = f.get("stepSize")
                        if mq is not None and float(mq) > 0:
                            min_qty = Decimal(str(mq))
                        if mxq is not None and float(mxq) > 0:
                            max_qty = Decimal(str(mxq))
                        if ss is not None and float(ss) > 0:
                            step_size = Decimal(str(ss))
                    elif ftype == "LOT_SIZE":
                        if min_qty is None:
                            mq = f.get("minQty")
                            if mq is not None and float(mq) > 0:
                                min_qty = Decimal(str(mq))
                        if max_qty is None:
                            mxq = f.get("maxQty")
                            if mxq is not None and float(mxq) > 0:
                                max_qty = Decimal(str(mxq))
                        if step_size is None:
                            ss = f.get("stepSize")
                            if ss is not None and float(ss) > 0:
                                step_size = Decimal(str(ss))
                    elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                        notional_val = f.get("notional") or f.get("minNotional")
                        if notional_val is not None and float(notional_val) > 0:
                            min_notional = Decimal(str(notional_val))

            # 2. Check CCXT limits if any required filter was not in native info
            limits = market.get("limits", {})
            amt_limits = limits.get("amount", {})
            cost_limits = limits.get("cost", {})
            price_limits = limits.get("price", {})

            if min_qty is None and amt_limits.get("min") is not None and float(amt_limits["min"]) > 0:
                min_qty = Decimal(str(amt_limits["min"]))
            if max_qty is None and amt_limits.get("max") is not None and float(amt_limits["max"]) > 0:
                max_qty = Decimal(str(amt_limits["max"]))
            if step_size is None and amt_limits.get("step") is not None and float(amt_limits["step"]) > 0:
                step_size = Decimal(str(amt_limits["step"]))
            if tick_size is None and price_limits.get("step") is not None and float(price_limits["step"]) > 0:
                tick_size = Decimal(str(price_limits["step"]))
            if min_notional is None and cost_limits.get("min") is not None and float(cost_limits["min"]) > 0:
                min_notional = Decimal(str(cost_limits["min"]))

            # Validate that every required filter has been reliably established
            missing = []
            if min_qty is None or min_qty <= Decimal("0"):
                missing.append("min_qty")
            if max_qty is None or max_qty <= Decimal("0"):
                missing.append("max_qty")
            if step_size is None or step_size <= Decimal("0"):
                missing.append("step_size")
            if tick_size is None or tick_size <= Decimal("0"):
                missing.append("tick_size")
            if min_notional is None or min_notional <= Decimal("0"):
                missing.append("min_notional")

            if missing:
                raise MainnetDemoError(
                    f"Failed to establish reliable dynamic exchange filters for {sym}. "
                    f"Missing or invalid filters: {missing}. Aborting."
                )

            return {
                "min_qty": min_qty,
                "max_qty": max_qty,
                "step_size": step_size,
                "min_notional": min_notional,
                "tick_size": tick_size,
            }
        except Exception as e:
            if isinstance(e, MainnetDemoError):
                raise
            raise MainnetDemoError(f"Failed to load dynamic market filters for {sym}: {e}") from e

    def fetch_balance(self) -> Dict[str, float]:
        """Fetch account balance and available margin."""
        try:
            bal = self._exchange.fetch_balance({"type": "future"})
            total = bal.get("total", {}) or {}
            free = bal.get("free", {}) or {}
            used = bal.get("used", {}) or {}
            return {
                "equity": float(total.get("USDT", 0.0) or 0.0),
                "available_balance": float(free.get("USDT", 0.0) or 0.0),
                "margin_used": float(used.get("USDT", 0.0) or 0.0),
            }
        except Exception as e:
            raise MainnetDemoError(f"Failed to fetch account balance: {e}") from e

    fetch_account_balance = fetch_balance

    def fetch_mark_price_data(self, symbol: Optional[str] = None) -> Tuple[float, int]:
        """Fetch mark price and source timestamp, failing if source timestamp is missing."""
        sym = symbol or self.config.raw_symbol
        try:
            res = self._exchange.fapiPublicGetPremiumIndex({"symbol": sym})
            price = res.get("markPrice")
            if price is None:
                raise MainnetDemoError(f"markPrice missing in response for {sym}")
            raw_time = res.get("time")
            if raw_time is None:
                raise MainnetDemoError(f"Exchange response lacks source timestamp 'time' for {sym}")
            return float(price), int(raw_time)
        except Exception as e:
            raise MainnetDemoError(f"Failed to fetch mark price: {e}") from e

    def fetch_positions(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Fetch current open positions."""
        syms = symbols or [self.config.symbol]
        try:
            return self._exchange.fetch_positions(syms)
        except Exception as e:
            raise MainnetDemoError(f"Failed to fetch positions: {e}") from e

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch active resting open orders."""
        sym = symbol or self.config.symbol
        try:
            return self._exchange.fetch_open_orders(sym)
        except Exception as e:
            raise MainnetDemoError(f"Failed to fetch open orders: {e}") from e

    # -------------------------------------------------------------------------
    # Validated Order Dispatch
    # -------------------------------------------------------------------------

    def create_entry_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        client_order_id: str,
    ) -> Dict[str, Any]:
        """Dispatch a validated market entry order. Authorization enforced at boundary."""
        self._check_dispatch_authorization(is_emergency=False, reduce_only=False)

        side_clean = side.lower()
        if side_clean not in ("buy", "sell"):
            raise ValueError(f"Invalid entry side: {side}")
        if amount is None or math.isnan(amount) or math.isinf(amount) or amount <= 0.0:
            raise ValueError(f"Invalid entry quantity: {amount}")

        params = {"newClientOrderId": client_order_id}
        try:
            res = self._exchange.create_order(
                symbol=symbol,
                type="market",
                side=side_clean,
                amount=amount,
                params=params,
            )
            return res
        except Exception as e:
            if isinstance(e, (MainnetDemoError, MainnetDemoDisarmedError, MainnetDemoLockedError)):
                raise
            raise MainnetDemoError(f"Market entry order dispatch failed for {client_order_id}: {e}") from e

    def create_protective_stop_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_price: float,
        entry_price: float,
        tick_size: Decimal,
        step_size: Decimal,
        client_order_id: str,
    ) -> Dict[str, Any]:
        """Dispatch a validated protective stop-loss order with reduceOnly=True.

        Enforces:
        - Dispatch authorization check at client boundary.
        - Valid trigger direction relative to entry price:
          - For long position (side="sell"): stop_price MUST be < entry_price.
          - For short position (side="buy"): stop_price MUST be > entry_price.
        - Strict price quantization to symbol tick_size.
        - Quantity quantization to step_size.
        - Mandatory reduceOnly=True.
        - Acceptance verification into active resting status ('open' or 'new').
        """
        self._check_dispatch_authorization(is_emergency=False, reduce_only=True)

        side_clean = side.lower()
        if side_clean not in ("buy", "sell"):
            raise ValueError(f"Invalid stop side: {side}")
        if amount is None or math.isnan(amount) or math.isinf(amount) or amount <= 0.0:
            raise ValueError(f"Invalid stop quantity: {amount}")

        # Validate trigger direction
        if side_clean == "sell":  # Protecting a LONG
            if stop_price >= entry_price:
                raise MainnetDemoError(
                    f"Invalid stop trigger for LONG position: stop_price ({stop_price}) >= entry_price ({entry_price})"
                )
        elif side_clean == "buy":  # Protecting a SHORT
            if stop_price <= entry_price:
                raise MainnetDemoError(
                    f"Invalid stop trigger for SHORT position: stop_price ({stop_price}) <= entry_price ({entry_price})"
                )

        # Quantize stop price and amount
        p_dec = quantize_down(Decimal(str(stop_price)), tick_size)
        q_dec = quantize_down(Decimal(str(amount)), step_size)

        quantized_stop_price = float(p_dec)
        quantized_amount = float(q_dec)

        if quantized_amount <= 0.0:
            raise MainnetDemoError("Quantized stop order quantity is zero.")

        params = {
            "newClientOrderId": client_order_id,
            "stopPrice": quantized_stop_price,
            "reduceOnly": True,
        }

        try:
            res = self._exchange.create_order(
                symbol=symbol,
                type="STOP_MARKET",
                side=side_clean,
                amount=quantized_amount,
                params=params,
            )
            # Verify order acceptance and active status
            oid = res.get("id") or res.get("orderId")
            if not oid:
                raise MainnetDemoError("Exchange response did not contain an order ID for protective stop.")

            status = str(res.get("status", "")).lower()
            if status not in ("open", "new"):
                # If status is ambiguous in creation response, query order state directly
                try:
                    fetched = self.fetch_order(symbol=symbol, order_id=str(oid), client_order_id=client_order_id)
                    status = str(fetched.get("status", "")).lower()
                except Exception as fe:
                    raise MainnetDemoError(
                        f"Protective stop order {oid} status could not be verified as active: {fe}"
                    ) from fe

            if status not in ("open", "new"):
                raise MainnetDemoError(
                    f"Protective stop order {oid} was rejected or not in active resting status. Status: '{status}'"
                )

            return res
        except Exception as e:
            if isinstance(e, (MainnetDemoError, MainnetDemoDisarmedError, MainnetDemoLockedError)):
                raise
            raise MainnetDemoError(f"Protective stop order dispatch failed for {client_order_id}: {e}") from e

    def create_exit_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        step_size: Decimal,
        client_order_id: str,
        is_emergency: bool = False,
    ) -> Dict[str, Any]:
        """Dispatch a market order to close position with reduceOnly=True.

        Authorization is enforced at the client boundary, permitting narrowly scoped
        emergency exits when is_emergency=True.
        """
        self._check_dispatch_authorization(is_emergency=is_emergency, reduce_only=True)

        side_clean = side.lower()
        if side_clean not in ("buy", "sell"):
            raise ValueError(f"Invalid exit side: {side}")

        q_dec = quantize_down(Decimal(str(amount)), step_size)
        quantized_amount = float(q_dec)
        if quantized_amount <= 0.0:
            raise MainnetDemoError(f"Quantized exit quantity evaluated to zero: {amount}")

        params = {
            "newClientOrderId": client_order_id,
            "reduceOnly": True,
        }

        try:
            return self._exchange.create_order(
                symbol=symbol,
                type="market",
                side=side_clean,
                amount=quantized_amount,
                params=params,
            )
        except Exception as e:
            if isinstance(e, (MainnetDemoError, MainnetDemoDisarmedError, MainnetDemoLockedError)):
                raise
            raise MainnetDemoError(f"Market exit order dispatch failed for {client_order_id}: {e}") from e

    def fetch_my_trades(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch user trade execution history for reconciliation."""
        sym = symbol or self.config.symbol
        try:
            if hasattr(self._exchange, "fetch_my_trades"):
                return self._exchange.fetch_my_trades(sym)
            return []
        except Exception:
            return []

    def cancel_order(
        self, symbol: str, order_id: Optional[str] = None, client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Cancel resting order."""
        params = {}
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        try:
            return self._exchange.cancel_order(
                id=order_id,
                symbol=symbol,
                params=params,
            )
        except Exception as e:
            raise MainnetDemoError(f"Failed to cancel order {order_id or client_order_id}: {e}") from e

    def fetch_order(
        self, symbol: str, order_id: Optional[str] = None, client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch order details by ID or client ID."""
        params = {}
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        try:
            return self._exchange.fetch_order(
                id=order_id,
                symbol=symbol,
                params=params,
            )
        except Exception as e:
            raise MainnetDemoError(f"Failed to fetch order {order_id or client_order_id}: {e}") from e
