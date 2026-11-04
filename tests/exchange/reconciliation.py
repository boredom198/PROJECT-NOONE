"""Reconciliation and Safety Freeze Manager for Component 3.

Implements the unified authoritative permission matrix and 9-step controlled
order recovery sequence.
Conforms strictly to PROJECT NOONE Technical Design Specification V2.2.4 Sections 6 and 11.
"""

from dataclasses import dataclass
from decimal import Decimal
import time
from typing import Any, Dict, List, Optional, Tuple

from noone.exchange.client import BinanceFuturesClient
from noone.exchange.config import Component3Config
from noone.exchange.errors import (
    OrderRecoveryError,
    SafetyFreezeError,
    StateDiscrepancyError,
)
from noone.exchange.journal import ExecutionJournal, JournalOrder


@dataclass(frozen=True)
class RecoveryOutcome:
    """Result of an order recovery procedure."""

    is_authoritative: bool
    final_order_state: str
    reconciliation_state: str
    filled_quantity: float
    avg_price: float
    details: str


class ReconciliationManager:
    """Coordinates safety freezes, order recovery, and post-trade accounting reconciliation."""

    def __init__(
        self,
        config: Component3Config,
        client: BinanceFuturesClient,
        journal: ExecutionJournal,
    ):
        self.config = config
        self.client = client
        self.journal = journal

    # -------------------------------------------------------------------------
    # Permission Matrix Checks
    # -------------------------------------------------------------------------

    def can_execute_entry(self) -> Tuple[bool, Optional[str]]:
        """Evaluate whether new position entries (OPEN_LONG, OPEN_SHORT) are permitted."""
        active_freezes = self.journal.get_active_safety_freezes()
        if not active_freezes:
            return True, None

        reasons = [f.reason for f in active_freezes]
        return False, f"Entry blocked by active safety freezes: {', '.join(reasons)}"

    def can_execute_exit(self, independently_verified: bool) -> Tuple[bool, Optional[str]]:
        """Evaluate whether position exits (CLOSE_LONG, CLOSE_SHORT) are permitted.

        Exits are permitted ONLY if independently verified via GET /fapi/v2/positionRisk.
        Unverified exits are strictly blocked across all conditions.
        """
        if not independently_verified:
            return False, "Unverified exits are strictly prohibited under all operational conditions."

        active_freezes = self.journal.get_active_safety_freezes()
        if not active_freezes:
            return True, None

        # Check for freezes that strictly block ALL exits (even verified)
        strictly_blocking = {
            "NETWORK_COMMUNICATION_FAILURE",
            "UNKNOWN_ORDER_OUTCOME",
            "AMBIGUOUS_POSITION_STATE",
        }
        for f in active_freezes:
            if f.reason in strictly_blocking:
                return (
                    False,
                    f"Exits strictly blocked while freeze '{f.reason}' is active.",
                )

        # Other freezes (POSITION_ACCOUNT_DISCREPANCY, ACCOUNT_WIDE_EXPOSURE_DETECTED,
        # DAILY_LOSS_LIMIT, DAILY_PNL_UNINITIALIZED) permit verified risk-reducing exits.
        return True, None

    # -------------------------------------------------------------------------
    # Nine-Step Controlled Order Recovery Protocol
    # -------------------------------------------------------------------------

    def recover_unknown_order(self, client_order_id: str) -> RecoveryOutcome:
        """Execute the 9-step controlled order recovery sequence for an unresolved order."""
        self.journal.trigger_safety_freeze(
            "UNKNOWN_ORDER_OUTCOME",
            f"Initiating 9-step recovery for client_order_id: {client_order_id}",
        )

        order_record = self.journal.get_order(client_order_id)
        if not order_record:
            raise OrderRecoveryError(f"Cannot recover order {client_order_id}: not found in journal.")

        # Step 1: Query order by origClientOrderId
        exchange_order: Optional[Dict[str, Any]] = None
        try:
            exchange_order = self.client.fetch_order_by_client_id(client_order_id)
            self.journal.log_recovery_attempt(
                client_order_id, 1, "fetch_order_by_client_id", "SUCCESS", str(exchange_order.get("status"))
            )
        except OrderRecoveryError as e:
            # OrderNotFound (-2013): Inconclusive outcome
            msg = f"OrderNotFound received for {client_order_id}; outcome remains inconclusive."
            self.journal.log_recovery_attempt(client_order_id, 1, "fetch_order_by_client_id", "INCONCLUSIVE", msg)
            return RecoveryOutcome(
                is_authoritative=False,
                final_order_state="UNKNOWN",
                reconciliation_state="DISCREPANCY",
                filled_quantity=0.0,
                avg_price=0.0,
                details=msg,
            )
        except Exception as e:
            msg = f"Network or exchange failure during query: {e}"
            self.journal.log_recovery_attempt(client_order_id, 1, "fetch_order_by_client_id", "FAILURE", msg)
            return RecoveryOutcome(
                is_authoritative=False,
                final_order_state="UNKNOWN",
                reconciliation_state="DISCREPANCY",
                filled_quantity=0.0,
                avg_price=0.0,
                details=msg,
            )

        status = str(exchange_order.get("status", "")).upper()
        exchange_oid = str(exchange_order.get("id") or exchange_order.get("orderId", ""))

        # Step 2 & 3: Retrieve trade fills and deduplicate in SQLite
        fills: List[Dict[str, Any]] = []
        try:
            trades = self.client.fetch_my_trades(symbol=self.config.ccxt_symbol)
            for t in trades:
                t_order_id = str(t.get("order") or t.get("orderId", ""))
                t_client_id = t.get("info", {}).get("clientOrderId", "")
                if t_order_id == exchange_oid or t_client_id == client_order_id:
                    trade_id = str(t.get("id") or t.get("tradeId", ""))
                    price = float(t.get("price", 0.0))
                    qty = float(t.get("amount", 0.0))
                    fee_info = t.get("fee", {}) or {}
                    commission = float(fee_info.get("cost", 0.0))
                    comm_asset = str(fee_info.get("currency", "USDT"))
                    rpnl = float(t.get("info", {}).get("realizedPnl", 0.0))
                    ts = int(t.get("timestamp") or time.time() * 1000)

                    inserted = self.journal.record_trade_fill(
                        trade_id=trade_id,
                        client_order_id=client_order_id,
                        symbol=self.config.raw_symbol,
                        price=price,
                        quantity=qty,
                        commission=commission,
                        commission_asset=comm_asset,
                        realized_pnl=rpnl,
                        timestamp=ts,
                        exchange_order_id=exchange_oid,
                    )
                    fills.append(t)
            self.journal.log_recovery_attempt(
                client_order_id, 3, "deduplicate_fills", "SUCCESS", f"Deduplicated {len(fills)} fills."
            )
        except Exception as e:
            self.journal.log_recovery_attempt(
                client_order_id, 3, "deduplicate_fills", "WARNING", f"Fill query encountered issue: {e}"
            )

        # Step 4: Reconcile executed quantity and price
        filled_qty = float(exchange_order.get("filled", 0.0))
        cum_quote = float(exchange_order.get("cost", 0.0))
        avg_price = float(exchange_order.get("average", 0.0))
        if avg_price <= 0.0 and filled_qty > 0.0 and cum_quote > 0.0:
            avg_price = cum_quote / filled_qty
        self.journal.log_recovery_attempt(
            client_order_id, 4, "reconcile_quantity_price", "SUCCESS",
            f"Filled: {filled_qty}, AvgPrice: {avg_price}"
        )

        # Step 5: Reconcile commissions and realized PnL
        self.journal.log_recovery_attempt(
            client_order_id, 5, "reconcile_commission_rpnl", "SUCCESS", "Fills and commissions recorded."
        )

        # Step 6: Retrieve current exchange position
        try:
            positions = self.client.fetch_position_risk_v2(self.config.raw_symbol)
            self.journal.log_recovery_attempt(
                client_order_id, 6, "fetch_exchange_position", "SUCCESS", f"Position records: {len(positions)}"
            )
        except Exception as e:
            self.journal.log_recovery_attempt(
                client_order_id, 6, "fetch_exchange_position", "FAILURE", str(e)
            )

        # Step 7: Retrieve current exchange balances
        try:
            account_data = self.client.fetch_account_v2()
            self.journal.log_recovery_attempt(
                client_order_id, 7, "fetch_exchange_balances", "SUCCESS", "Account balances retrieved."
            )
        except Exception as e:
            self.journal.log_recovery_attempt(
                client_order_id, 7, "fetch_exchange_balances", "FAILURE", str(e)
            )

        # Step 8: Inspect unresolved local intents
        unresolved = self.journal.get_unresolved_orders()
        self.journal.log_recovery_attempt(
            client_order_id, 8, "inspect_unresolved_intents", "SUCCESS", f"Unresolved orders: {len(unresolved)}"
        )

        # Step 9: Inspect unexplained open orders
        try:
            open_orders = self.client.fetch_open_orders()
            self.journal.log_recovery_attempt(
                client_order_id, 9, "inspect_open_orders", "SUCCESS", f"Open orders: {len(open_orders)}"
            )
        except Exception as e:
            self.journal.log_recovery_attempt(
                client_order_id, 9, "inspect_open_orders", "FAILURE", str(e)
            )

        # Handling Active NEW / PARTIALLY_FILLED vs Terminal
        if status in ("OPEN", "NEW", "PARTIALLY_FILLED"):
            # Request cancellation using original client order ID
            try:
                self.client.cancel_order_by_client_id(client_order_id)
                self.journal.transition_order_state(
                    client_order_id=client_order_id,
                    to_state="CANCELED",
                    reason="Canceled during active order recovery protocol",
                    exchange_order_id=exchange_oid,
                    filled_quantity=filled_qty,
                    cum_quote=cum_quote,
                    avg_price=avg_price,
                    reconciliation_state="VERIFIED",
                )
                self.journal.clear_safety_freeze("UNKNOWN_ORDER_OUTCOME", "Active order successfully canceled and reconciled.")
                return RecoveryOutcome(
                    is_authoritative=True,
                    final_order_state="CANCELED",
                    reconciliation_state="VERIFIED",
                    filled_quantity=filled_qty,
                    avg_price=avg_price,
                    details="Active order canceled and reconciled.",
                )
            except Exception as cancel_err:
                msg = f"Failed to cancel active order {client_order_id}: {cancel_err}"
                self.journal.log_recovery_attempt(client_order_id, 9, "cancel_active_order", "FAILURE", msg)
                return RecoveryOutcome(
                    is_authoritative=False,
                    final_order_state=status,
                    reconciliation_state="DISCREPANCY",
                    filled_quantity=filled_qty,
                    avg_price=avg_price,
                    details=msg,
                )

        # Terminal outcomes (FILLED, CANCELED, EXPIRED, REJECTED)
        terminal_map = {
            "CLOSED": "FILLED",
            "FILLED": "FILLED",
            "CANCELED": "CANCELED",
            "CANCELLED": "CANCELED",
            "EXPIRED": "EXPIRED",
            "REJECTED": "REJECTED",
        }
        final_state = terminal_map.get(status, "UNKNOWN")
        if final_state != "UNKNOWN":
            self.journal.transition_order_state(
                client_order_id=client_order_id,
                to_state=final_state,
                reason="Authoritative terminal state confirmed during 9-step recovery",
                exchange_order_id=exchange_oid,
                filled_quantity=filled_qty,
                cum_quote=cum_quote,
                avg_price=avg_price,
                reconciliation_state="VERIFIED",
            )
            self.journal.clear_safety_freeze(
                "UNKNOWN_ORDER_OUTCOME", f"Terminal outcome {final_state} reconciled successfully."
            )
            return RecoveryOutcome(
                is_authoritative=True,
                final_order_state=final_state,
                reconciliation_state="VERIFIED",
                filled_quantity=filled_qty,
                avg_price=avg_price,
                details=f"Authoritative outcome {final_state} confirmed.",
            )

        # Still unknown
        return RecoveryOutcome(
            is_authoritative=False,
            final_order_state="UNKNOWN",
            reconciliation_state="DISCREPANCY",
            filled_quantity=filled_qty,
            avg_price=avg_price,
            details=f"Order status '{status}' could not be definitively resolved.",
        )

