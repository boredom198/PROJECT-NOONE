"""Reconciliation and Safety Freeze Manager for Component 3.

Implements the unified authoritative permission matrix and fail-closed 9-step controlled
order recovery sequence.
Conforms strictly to PROJECT NOONE Technical Design Specification V2.2.4 Sections 6 and 11.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import time
from typing import Any, Dict, List, Optional, Tuple

from noone.exchange.account import audit_account_positions
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
        if active_freezes:
            reasons = [f.reason for f in active_freezes]
            return False, f"Entry blocked by active safety freezes: {', '.join(reasons)}"

        unresolved = self.journal.get_unresolved_orders()
        if unresolved:
            return False, f"Entry blocked: {len(unresolved)} unresolved orders exist in journal."

        return True, None

    def can_execute_exit(
        self, independently_verified: bool, is_dust: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """Evaluate whether position exits (CLOSE_LONG, CLOSE_SHORT) are permitted.

        Exits are permitted ONLY if independently verified via GET /fapi/v2/positionRisk.
        Unverified exits or residual dust exits are strictly blocked across all conditions.
        """
        if not independently_verified:
            return False, "Unverified exits are strictly prohibited under all operational conditions."

        if is_dust:
            return False, "Untradeable residual dust cannot be exited via market order; requires manual console action."

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
    # Fail-Closed Nine-Step Controlled Order Recovery Protocol (Defect 2)
    # -------------------------------------------------------------------------

    def recover_unknown_order(self, client_order_id: str) -> RecoveryOutcome:
        """Execute the fail-closed 9-step controlled order recovery sequence.

        Requires authoritative order identity/status, complete and deduplicated fill history,
        fees/PnL reconciliation, fresh position and balance snapshots, and checks for unresolved
        local intents or unexplained open orders before clearing the freeze.
        Stops safely and retains freeze if any check fails or evidence is contradictory.
        """
        self.journal.trigger_safety_freeze(
            "UNKNOWN_ORDER_OUTCOME",
            f"Initiating 9-step fail-closed recovery for client_order_id: {client_order_id}",
            client_order_id=client_order_id,
        )

        order_record = self.journal.get_order(client_order_id)
        if not order_record:
            msg = f"Cannot recover order {client_order_id}: not found in journal."
            self.journal.log_recovery_attempt(client_order_id, 0, "journal_lookup", "FAILURE", msg)
            raise OrderRecoveryError(msg)

        # Step 1: Query order by origClientOrderId
        exchange_order: Optional[Dict[str, Any]] = None
        try:
            exchange_order = self.client.fetch_order_by_client_id(client_order_id)
            self.journal.log_recovery_attempt(
                client_order_id, 1, "fetch_order_by_client_id", "SUCCESS", str(exchange_order.get("status"))
            )
        except OrderRecoveryError as e:
            # OrderNotFound (-2013): Inconclusive outcome -> Fail closed!
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

        # Step 2: Handle active NEW / OPEN / PARTIALLY_FILLED orders
        if status in ("OPEN", "NEW", "PARTIALLY_FILLED"):
            # Request cancellation using original client order ID
            try:
                self.client.cancel_order_by_client_id(client_order_id)
                self.journal.log_recovery_attempt(
                    client_order_id, 2, "cancel_active_order", "SUCCESS", "Cancel request dispatched."
                )
            except Exception as cancel_err:
                msg = f"Cancellation request failed for active order {client_order_id}: {cancel_err}"
                self.journal.log_recovery_attempt(client_order_id, 2, "cancel_active_order", "FAILURE", msg)
                return RecoveryOutcome(
                    is_authoritative=False,
                    final_order_state=status,
                    reconciliation_state="DISCREPANCY",
                    filled_quantity=float(exchange_order.get("filled", 0.0)),
                    avg_price=float(exchange_order.get("average", 0.0)),
                    details=msg,
                )

            # Re-query order on exchange to verify status is now terminal (CANCELED/EXPIRED)
            try:
                exchange_order = self.client.fetch_order_by_client_id(client_order_id)
                status = str(exchange_order.get("status", "")).upper()
                if status not in ("CANCELED", "CANCELLED", "EXPIRED", "FILLED", "CLOSED"):
                    msg = f"Order status after cancel acknowledgement is still active: '{status}'."
                    self.journal.log_recovery_attempt(client_order_id, 2, "verify_canceled_status", "FAILURE", msg)
                    return RecoveryOutcome(
                        is_authoritative=False,
                        final_order_state=status,
                        reconciliation_state="DISCREPANCY",
                        filled_quantity=float(exchange_order.get("filled", 0.0)),
                        avg_price=float(exchange_order.get("average", 0.0)),
                        details=msg,
                    )
            except Exception as requery_err:
                msg = f"Failed to re-query order after cancellation: {requery_err}"
                self.journal.log_recovery_attempt(client_order_id, 2, "verify_canceled_status", "FAILURE", msg)
                return RecoveryOutcome(
                    is_authoritative=False,
                    final_order_state="UNKNOWN",
                    reconciliation_state="DISCREPANCY",
                    filled_quantity=0.0,
                    avg_price=0.0,
                    details=msg,
                )

        # Step 3: Deduplicate trade fills in SQLite
        filled_qty = float(exchange_order.get("filled", 0.0))
        cum_quote = float(exchange_order.get("cost", 0.0))
        avg_price = float(exchange_order.get("average", 0.0))
        if avg_price <= 0.0 and filled_qty > 0.0 and cum_quote > 0.0:
            avg_price = cum_quote / filled_qty

        recorded_fills = []
        try:
            order_ts = None
            if order_record and order_record.created_at:
                try:
                    dt = datetime.fromisoformat(order_record.created_at.replace("Z", "+00:00"))
                    order_ts = int(dt.timestamp() * 1000) - 60_000
                except Exception:
                    pass

            # Paginate through trade history to ensure complete fill capture
            if hasattr(self.client, "fetch_my_trades_paginated"):
                trades = self.client.fetch_my_trades_paginated(
                    symbol=self.config.ccxt_symbol, since=order_ts
                )
            else:
                trades = self.client.fetch_my_trades(symbol=self.config.ccxt_symbol, since=order_ts)

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

                    self.journal.record_trade_fill(
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
                    recorded_fills.append(t)
            self.journal.log_recovery_attempt(
                client_order_id, 3, "deduplicate_fills", "SUCCESS", f"Deduplicated {len(recorded_fills)} fills."
            )
        except Exception as e:
            msg = f"Trade fills retrieval failed: {e}"
            self.journal.log_recovery_attempt(client_order_id, 3, "deduplicate_fills", "FAILURE", msg)
            return RecoveryOutcome(
                is_authoritative=False,
                final_order_state=status,
                reconciliation_state="DISCREPANCY",
                filled_quantity=filled_qty,
                avg_price=avg_price,
                details=msg,
            )

        # Step 4: Reconcile fill quantity and fee consistency
        if filled_qty > 0.0:
            db_fills = self.journal.get_order_fills(client_order_id)
            sum_fill_qty = sum(f.quantity for f in db_fills)
            if abs(Decimal(str(sum_fill_qty)) - Decimal(str(filled_qty))) > Decimal("1e-8"):
                msg = f"Fill quantity mismatch: fills sum {sum_fill_qty} != order filled {filled_qty}"
                self.journal.log_recovery_attempt(client_order_id, 4, "reconcile_fills", "FAILURE", msg)
                return RecoveryOutcome(
                    is_authoritative=False,
                    final_order_state=status,
                    reconciliation_state="DISCREPANCY",
                    filled_quantity=filled_qty,
                    avg_price=avg_price,
                    details=msg,
                )
            # Reconcile fee consistency
            total_comm = sum(f.commission for f in db_fills)
            if total_comm < 0.0:
                msg = f"Negative commission detected in fills for {client_order_id}: {total_comm}"
                self.journal.log_recovery_attempt(client_order_id, 4, "reconcile_fees", "FAILURE", msg)
                return RecoveryOutcome(
                    is_authoritative=False,
                    final_order_state=status,
                    reconciliation_state="DISCREPANCY",
                    filled_quantity=filled_qty,
                    avg_price=avg_price,
                    details=msg,
                )

        # Step 5: Retrieve and verify exchange position changes
        try:
            positions = self.client.fetch_position_risk_v2(self.config.raw_symbol)
            if not positions:
                msg = f"Position query returned empty for {self.config.raw_symbol}"
                self.journal.log_recovery_attempt(client_order_id, 5, "fetch_exchange_position", "FAILURE", msg)
                return RecoveryOutcome(
                    is_authoritative=False,
                    final_order_state=status,
                    reconciliation_state="DISCREPANCY",
                    filled_quantity=filled_qty,
                    avg_price=avg_price,
                    details=msg,
                )

            pos_audit = audit_account_positions(positions, self.config.raw_symbol)
            target_pos = pos_audit.target_position
            target_amt = Decimal(str(target_pos.get("positionAmt", "0"))) if target_pos else Decimal("0")

            if order_record:
                if order_record.action_type in ("OPEN_LONG", "OPEN_SHORT"):
                    expected_sign = Decimal("1") if order_record.side.upper() == "BUY" else Decimal("-1")
                    if filled_qty > 0 and (target_amt * expected_sign <= Decimal("0")):
                        msg = f"Position mismatch for entry order: target amount {target_amt} does not reflect fill {filled_qty} {order_record.side}"
                        self.journal.log_recovery_attempt(client_order_id, 5, "verify_position_delta", "FAILURE", msg)
                        return RecoveryOutcome(
                            is_authoritative=False,
                            final_order_state=status,
                            reconciliation_state="DISCREPANCY",
                            filled_quantity=filled_qty,
                            avg_price=avg_price,
                            details=msg,
                        )
                elif order_record.action_type in ("CLOSE_LONG", "CLOSE_SHORT"):
                    if filled_qty > 0:
                        req_qty_dec = Decimal(str(order_record.quantity))
                        filled_dec = Decimal(str(filled_qty))
                        rem_dec = req_qty_dec - filled_dec
                        if rem_dec <= Decimal("1e-8") and abs(target_amt) > Decimal("1e-8"):
                            msg = f"Position mismatch for fully closed exit: expected flat position but got {target_amt}"
                            self.journal.log_recovery_attempt(client_order_id, 5, "verify_position_delta", "FAILURE", msg)
                            return RecoveryOutcome(
                                is_authoritative=False,
                                final_order_state=status,
                                reconciliation_state="DISCREPANCY",
                                filled_quantity=filled_qty,
                                avg_price=avg_price,
                                details=msg,
                            )

            self.journal.log_recovery_attempt(
                client_order_id, 5, "fetch_exchange_position", "SUCCESS", f"Positions verified ({len(positions)})."
            )
        except Exception as e:
            msg = f"Exchange position risk query failed: {e}"
            self.journal.log_recovery_attempt(client_order_id, 5, "fetch_exchange_position", "FAILURE", msg)
            return RecoveryOutcome(
                is_authoritative=False,
                final_order_state=status,
                reconciliation_state="DISCREPANCY",
                filled_quantity=filled_qty,
                avg_price=avg_price,
                details=msg,
            )

        # Step 6: Retrieve and verify exchange balances
        try:
            account_data = self.client.fetch_account_v2()
            if not account_data or "totalMarginBalance" not in account_data:
                msg = "Exchange account query returned invalid or missing totalMarginBalance."
                self.journal.log_recovery_attempt(client_order_id, 6, "fetch_exchange_balances", "FAILURE", msg)
                return RecoveryOutcome(
                    is_authoritative=False,
                    final_order_state=status,
                    reconciliation_state="DISCREPANCY",
                    filled_quantity=filled_qty,
                    avg_price=avg_price,
                    details=msg,
                )
            self.journal.log_recovery_attempt(
                client_order_id, 6, "fetch_exchange_balances", "SUCCESS", "Account balance verified."
            )
        except Exception as e:
            msg = f"Account balance query failed: {e}"
            self.journal.log_recovery_attempt(client_order_id, 6, "fetch_exchange_balances", "FAILURE", msg)
            return RecoveryOutcome(
                is_authoritative=False,
                final_order_state=status,
                reconciliation_state="DISCREPANCY",
                filled_quantity=filled_qty,
                avg_price=avg_price,
                details=msg,
            )

        # Step 7: Inspect unexplained exchange open orders
        try:
            open_orders = self.client.fetch_open_orders()
            if open_orders:
                msg = f"Unexplained open orders detected during recovery: {len(open_orders)}"
                self.journal.log_recovery_attempt(client_order_id, 7, "inspect_open_orders", "FAILURE", msg)
                return RecoveryOutcome(
                    is_authoritative=False,
                    final_order_state=status,
                    reconciliation_state="DISCREPANCY",
                    filled_quantity=filled_qty,
                    avg_price=avg_price,
                    details=msg,
                )
            self.journal.log_recovery_attempt(
                client_order_id, 7, "inspect_open_orders", "SUCCESS", "Zero open orders confirmed."
            )
        except Exception as e:
            msg = f"Open orders query failed: {e}"
            self.journal.log_recovery_attempt(client_order_id, 7, "inspect_open_orders", "FAILURE", msg)
            return RecoveryOutcome(
                is_authoritative=False,
                final_order_state=status,
                reconciliation_state="DISCREPANCY",
                filled_quantity=filled_qty,
                avg_price=avg_price,
                details=msg,
            )

        # Step 8: Inspect local unresolved intents in SQLite
        unresolved = self.journal.get_unresolved_orders()
        other_unresolved = [uo for uo in unresolved if uo.client_order_id != client_order_id]
        if other_unresolved:
            self.journal.log_recovery_attempt(
                client_order_id,
                8,
                "inspect_unresolved_intents",
                "INFO",
                f"Note: {len(other_unresolved)} other unresolved intents remain in SQLite.",
            )
        else:
            self.journal.log_recovery_attempt(
                client_order_id, 8, "inspect_unresolved_intents", "SUCCESS", "No conflicting unresolved orders."
            )

        # Step 9: Reconcile terminal outcome authoritatively
        req_qty = order_record.quantity if order_record else float(exchange_order.get("amount", 0.0))
        eps = 1e-8
        is_full = (filled_qty >= req_qty - eps) and filled_qty > 0
        is_partial = (0 < filled_qty < req_qty - eps)
        is_zero = (filled_qty <= eps)

        if status in ("CLOSED", "FILLED"):
            if is_full:
                final_state = "FILLED"
            elif is_partial:
                final_state = "PARTIALLY_FILLED_CANCELED"
            else:
                final_state = "UNKNOWN"
        elif status in ("CANCELED", "CANCELLED"):
            if is_zero:
                final_state = "CANCELED"
            elif is_partial:
                final_state = "PARTIALLY_FILLED_CANCELED"
            else:
                final_state = "UNKNOWN"
        elif status in ("EXPIRED", "EXPIRED_IN_MATCH"):
            if is_zero:
                final_state = "EXPIRED"
            elif is_partial:
                final_state = "PARTIALLY_FILLED_EXPIRED"
            else:
                final_state = "UNKNOWN"
        elif status in ("REJECTED",):
            final_state = "REJECTED"
        else:
            final_state = "UNKNOWN"

        if final_state != "UNKNOWN":
            rec_state = "VERIFIED"
            self.journal.transition_order_state(
                client_order_id=client_order_id,
                to_state=final_state,
                reason="Authoritatively reconciled through 9-step fail-closed sequence",
                exchange_order_id=exchange_oid,
                filled_quantity=filled_qty,
                cum_quote=cum_quote,
                avg_price=avg_price,
                reconciliation_state=rec_state,
            )

            # Clear per-order freeze
            self.journal.clear_safety_freeze(
                "UNKNOWN_ORDER_OUTCOME",
                f"Terminal outcome {final_state} authoritatively verified for order {client_order_id}.",
                client_order_id=client_order_id,
            )

            # Trigger partial fill freeze if terminal with unexecuted contracts
            if final_state in ("PARTIALLY_FILLED_CANCELED", "PARTIALLY_FILLED_EXPIRED"):
                self.journal.trigger_safety_freeze(
                    "PARTIAL_FILL_UNRESOLVED",
                    f"Order {client_order_id} ended in {final_state} with partial fill {filled_qty}/{req_qty}.",
                    client_order_id=client_order_id,
                )

            # Clear general freeze only if no OTHER unresolved orders remain
            unresolved_after = self.journal.get_unresolved_orders()
            if not any(uo.client_order_id != client_order_id for uo in unresolved_after):
                self.journal.clear_safety_freeze(
                    "UNKNOWN_ORDER_OUTCOME",
                    "All uncertain orders resolved.",
                )

            return RecoveryOutcome(
                is_authoritative=True,
                final_order_state=final_state,
                reconciliation_state=rec_state,
                filled_quantity=filled_qty,
                avg_price=avg_price,
                details=f"Authoritative terminal outcome {final_state} confirmed with complete evidence.",
            )

        msg = f"Order status '{status}' is not a recognized terminal state."
        self.journal.log_recovery_attempt(client_order_id, 9, "final_reconciliation", "FAILURE", msg)
        return RecoveryOutcome(
            is_authoritative=False,
            final_order_state="UNKNOWN",
            reconciliation_state="DISCREPANCY",
            filled_quantity=filled_qty,
            avg_price=avg_price,
            details=msg,
        )

        msg = f"Order status '{status}' is not a recognized terminal state."
        self.journal.log_recovery_attempt(client_order_id, 9, "final_reconciliation", "FAILURE", msg)
        return RecoveryOutcome(
            is_authoritative=False,
            final_order_state="UNKNOWN",
            reconciliation_state="DISCREPANCY",
            filled_quantity=filled_qty,
            avg_price=avg_price,
            details=msg,
        )
