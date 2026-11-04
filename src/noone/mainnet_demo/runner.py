"""Mainnet Demo Runner.

Coordinates the complete 10-step lifecycle for a single real trade demonstration:
- Crash reconciliation & disarmed read-only preflight
- Dynamic market filter parsing & multi-factor risk budget calculation
- RL engine model inference & action verification
- Two-stage manual arming gate
- Market entry with partial fill tracking
- Immediate protective stop placement with trigger direction & tickSize validation
- Emergency fail-closed closure if stop fails
- Correct exit sequence: close position first -> verify flat -> cancel stop -> verify 0 open orders
- Persistent SQLite loss budget tracking and permanent demo lockout
"""

from decimal import Decimal, ROUND_UP
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

from noone.mainnet_demo.client import MainnetDemoClient, quantize_down
from noone.mainnet_demo.config import (
    DemoState,
    EmergencyExitError,
    MainnetDemoConfig,
    MainnetDemoDisarmedError,
    MainnetDemoError,
    MainnetDemoLockedError,
    RiskBudgetExceededError,
)
from noone.mainnet_demo.journal import MainnetDemoJournal
from noone.rl_engine.inference import (
    AccountSnapshot,
    Action,
    ModelInference,
    PredictionResult,
)


def quantize_up(val: Decimal, step: Decimal) -> Decimal:
    """Quantize a Decimal upward to a given step size."""
    if step <= Decimal("0"):
        return val
    multiples = (val / step).to_integral_value(rounding=ROUND_UP)
    quantized = multiples * step
    exp = step.as_tuple().exponent
    return quantized.quantize(Decimal(10) ** exp, rounding=ROUND_UP)


class MainnetDemoRunner:
    """Orchestrator for Mainnet Demo Execution Mode."""

    def __init__(
        self,
        config: MainnetDemoConfig,
        client: Optional[MainnetDemoClient] = None,
        journal: Optional[MainnetDemoJournal] = None,
        inference: Optional[ModelInference] = None,
    ):
        self.config = config
        self.journal = journal or MainnetDemoJournal(config.database_path)
        self.client = client or MainnetDemoClient(config, journal=self.journal)
        if getattr(self.client, "journal", None) is None:
            self.client.journal = self.journal
        self.inference = inference
        self.runner_id = f"runner_{int(time.time()*1000)}_{id(self)}"

    # -------------------------------------------------------------------------
    # 1. Crash Reconciliation
    # -------------------------------------------------------------------------

    def reconcile_on_startup(self) -> Dict[str, Any]:
        """Inspect exchange and local journal for interrupted states.

        Never blindly retries an entry. Fails closed and locks if unexpected exposure is found.
        """
        # 1. Check SQLite permanent lock
        if self.journal.is_locked():
            reason = self.journal.get_lock_reason() or "Unknown lock reason"
            raise MainnetDemoLockedError(
                f"Mainnet Demo is permanently LOCKED across restarts: {reason}. "
                "Order dispatch is forbidden."
            )

        reconciliation_report: Dict[str, Any] = {"clean": True}

        # 2. Check exchange positions with signed positionAmt and side awareness
        try:
            positions = self.client.fetch_positions([self.config.symbol])
            for pos in positions:
                raw_amt_str = str(pos.get("positionAmt") or pos.get("info", {}).get("positionAmt") or "0.0")
                try:
                    raw_amt = float(raw_amt_str)
                except Exception:
                    raw_amt = 0.0

                contracts = float(pos.get("contracts", 0.0) or 0.0)
                pos_side = str(pos.get("side", "")).lower()

                signed_amt = 0.0
                if abs(raw_amt) > 1e-8:
                    signed_amt = raw_amt
                elif contracts > 1e-8:
                    signed_amt = -contracts if pos_side in ("short", "sell") else contracts

                if abs(signed_amt) > 1e-8:
                    self.journal.trigger_permanent_lock(
                        f"CRASH_RECONCILIATION_POSITION_DETECTED: Active position of {signed_amt} found on exchange."
                    )
                    reconciliation_report["clean"] = False
                    reconciliation_report["active_position"] = pos

                    # SHORT position (signed_amt < 0) requires BUY to close; LONG requires SELL
                    exit_side = "buy" if signed_amt < 0 else "sell"
                    step_size = Decimal("0.001")
                    try:
                        filters = self.client.load_market_filters(self.config.symbol)
                        step_size = filters.get("step_size", step_size)
                    except Exception:
                        pass

                    try:
                        self.client.create_exit_market_order(
                            symbol=self.config.symbol,
                            side=exit_side,
                            amount=abs(signed_amt),
                            step_size=step_size,
                            client_order_id=f"crash_emerg_close_{int(time.time()*1000)}",
                            is_emergency=True,
                        )
                    except Exception:
                        pass

                    # Verify position is flat
                    try:
                        pos_check = self.client.fetch_positions([self.config.symbol])
                        flat = True
                        for p in pos_check:
                            p_amt = float(
                                p.get("positionAmt")
                                or p.get("info", {}).get("positionAmt")
                                or p.get("contracts")
                                or 0.0
                            )
                            if abs(p_amt) > 1e-8:
                                flat = False
                        if not flat:
                            self.journal.trigger_permanent_lock("CRASH_RECONCILIATION_EMERGENCY_FLAT_FAILED")
                            raise EmergencyExitError(
                                f"Leftover active position of {signed_amt} detected. "
                                "Emergency closure could not flatten exposure. Demo permanently LOCKED."
                            )
                    except EmergencyExitError:
                        raise
                    except Exception as ve:
                        self.journal.trigger_permanent_lock(f"CRASH_RECONCILIATION_VERIFICATION_UNCERTAIN: {ve}")
                        raise EmergencyExitError(
                            f"Position flattening verification failed or uncertain: {ve}. Demo permanently LOCKED."
                        ) from ve

                    raise EmergencyExitError(
                        f"Leftover active position of {signed_amt} detected on startup. "
                        "Emergency close dispatched, position verified flat, and demo permanently LOCKED."
                    )
        except EmergencyExitError:
            raise
        except Exception as e:
            if isinstance(e, MainnetDemoLockedError):
                raise
            self.journal.trigger_permanent_lock(f"CRASH_RECONCILIATION_QUERY_FAILED: {e}")
            raise MainnetDemoLockedError(f"Crash reconciliation exchange query failed: {e}. Demo LOCKED.") from e

        # 3. Check open orders on exchange on startup
        try:
            open_orders = self.client.fetch_open_orders(self.config.symbol)
            if len(open_orders) > 0:
                self.journal.trigger_permanent_lock(
                    f"CRASH_RECONCILIATION_OPEN_ORDERS_DETECTED: {len(open_orders)} open orders found on exchange."
                )
                for o in open_orders:
                    try:
                        self.client.cancel_order(self.config.symbol, order_id=o.get("id"))
                    except Exception:
                        pass
                rem_orders = self.client.fetch_open_orders(self.config.symbol)
                if len(rem_orders) > 0:
                    raise MainnetDemoLockedError(
                        f"Open orders detected on startup and could not be cancelled: {len(rem_orders)} remain. Demo LOCKED."
                    )
                raise MainnetDemoLockedError(
                    f"Found {len(open_orders)} open resting orders on startup. All cancelled and demo permanently LOCKED."
                )
        except MainnetDemoLockedError:
            raise
        except Exception as oe:
            self.journal.trigger_permanent_lock(f"CRASH_RECONCILIATION_ORDERS_QUERY_FAILED: {oe}")
            raise MainnetDemoLockedError(f"Failed to inspect open orders on startup: {oe}. Demo LOCKED.") from oe

        # 4. Check unresolved local orders in journal
        unresolved = self.journal.get_unresolved_orders()
        if unresolved:
            self.journal.trigger_permanent_lock(
                f"UNRESOLVED_LOCAL_ORDERS_DETECTED: {len(unresolved)} orders pending resolution."
            )
            raise MainnetDemoLockedError(
                f"Found {len(unresolved)} unresolved orders in journal. Demo permanently LOCKED."
            )

        return reconciliation_report

    # -------------------------------------------------------------------------
    # 2. Read-Only Preflight
    # -------------------------------------------------------------------------

    def run_preflight(self) -> Dict[str, Any]:
        """Perform strictly read-only preflight inspection."""
        self.reconcile_on_startup()
        return self.client.inspect_preflight_account()

    # -------------------------------------------------------------------------
    # 3. Dynamic Sizing & Multi-Factor Risk Budget Check
    # -------------------------------------------------------------------------

    def compute_safe_quantity_and_risk(
        self,
        mark_price: float,
        filters: Dict[str, Decimal],
    ) -> Tuple[float, float, float]:
        """Compute minimum valid quantity and evaluate total estimated risk against remaining budget.

        Formula:
        Total Risk = Stop Delta + Entry Fee + Exit Fee + Slippage Buffers + Funding Buffer + Emergency Buffer

        Never increases size beyond exchange minimums to force compatibility.
        Rejects if Total Risk exceeds remaining budget.
        """
        remaining_budget = self.journal.get_remaining_budget(self.config.max_loss_budget_usdt)
        if remaining_budget <= 0.0:
            raise RiskBudgetExceededError(
                f"Remaining loss budget is {remaining_budget:.4f} USDT. Demo budget exhausted."
            )

        min_qty = filters["min_qty"]
        step_size = filters["step_size"]
        min_notional = filters["min_notional"]

        p_dec = Decimal(str(mark_price))
        if p_dec <= Decimal("0"):
            raise ValueError(f"Invalid non-positive mark price: {mark_price}")

        # Compute dynamic minimum valid quantity
        # Must satisfy quantity >= min_qty AND quantity * mark_price >= min_notional
        notional_qty = quantize_up(min_notional / p_dec, step_size)
        candidate_qty_dec = max(min_qty, notional_qty)
        candidate_qty_dec = quantize_up(candidate_qty_dec, step_size)

        candidate_qty = float(candidate_qty_dec)
        notional = candidate_qty * mark_price

        # Multi-factor risk calculation
        stop_dist = mark_price * self.config.stop_loss_pct
        stop_price = mark_price - stop_dist  # for long, short is symmetric magnitude
        stop_delta = stop_dist * candidate_qty

        entry_fee = notional * self.config.taker_fee_rate
        exit_fee = (notional - stop_delta) * self.config.taker_fee_rate
        entry_slippage = notional * self.config.slippage_buffer_pct
        exit_slippage = notional * self.config.slippage_buffer_pct
        funding_buffer = notional * self.config.funding_buffer_pct
        emergency_buffer = notional * self.config.emergency_exposure_pct

        total_estimated_risk = (
            stop_delta
            + entry_fee
            + exit_fee
            + entry_slippage
            + exit_slippage
            + funding_buffer
            + emergency_buffer
        )

        if total_estimated_risk > remaining_budget:
            raise RiskBudgetExceededError(
                f"Total estimated trade risk ({total_estimated_risk:.4f} USDT) exceeds remaining budget "
                f"({remaining_budget:.4f} USDT) for minimum valid quantity {candidate_qty} BTC. "
                "Trade rejected. Never increasing size to satisfy exchange minimums."
            )

        return candidate_qty, total_estimated_risk, remaining_budget

    # -------------------------------------------------------------------------
    # 4. Model Inference
    # -------------------------------------------------------------------------

    def predict_model_action(
        self,
        candles: List[Any],
        account_balance: Dict[str, float],
    ) -> PredictionResult:
        """Run RL model inference on verified 15m closed candles.

        Aborts cleanly if model chooses HOLD.
        """
        if self.inference is None:
            if not self.config.model_path.exists():
                raise FileNotFoundError(f"Model file not found at {self.config.model_path}")
            self.inference = ModelInference.load(
                self.config.model_path, allow_legacy_metadata=True
            )

        account_snap = AccountSnapshot(
            equity=account_balance["equity"],
            available_balance=account_balance["available_balance"],
            position_side=0,
            position_quantity=0.0,
            entry_price=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            margin_used=0.0,
            progress=0.0,
        )

        pred = self.inference.predict(candles=candles, account=account_snap)
        return pred

    # -------------------------------------------------------------------------
    # 5. Single Demo Trade Execution Lifecycle
    # -------------------------------------------------------------------------

    def execute_demo_trade(self, candles: List[Any]) -> Dict[str, Any]:
        """Execute the full 10-step single demonstration trade lifecycle."""
        # 0. Atomic single-execution claiming
        self.journal.claim_execution(self.runner_id)

        # 1. Startup & crash reconciliation
        self.reconcile_on_startup()

        # 2. Read-only preflight
        preflight_results = self.client.inspect_preflight_account()

        # 3. Dynamic filters, balance & validated fresh mark price
        filters = self.client.load_market_filters(self.config.symbol)
        balance = self.client.fetch_account_balance()
        mark_price, mark_ts = self.client.fetch_mark_price_data(self.config.raw_symbol)

        # Validate mark price freshness (max 60 seconds staleness)
        now_ms = int(time.time() * 1000)
        staleness_ms = abs(now_ms - mark_ts)
        if staleness_ms > 60_000:
            self.journal.trigger_permanent_lock("STALE_MARK_PRICE_ABORT")
            raise MainnetDemoError(
                f"Mark price timestamp is stale ({staleness_ms}ms delay/skew, max allowed 60000ms). "
                "Execution aborted for safety."
            )

        # 4. Sizing & risk budget
        qty, est_risk, rem_budget = self.compute_safe_quantity_and_risk(mark_price, filters)

        # Verify margin requirement at 1x leverage
        if balance["available_balance"] < (qty * mark_price * 1.01):
            raise MainnetDemoError(
                f"Available balance ({balance['available_balance']:.2f} USDT) insufficient for 1x position "
                f"({qty * mark_price:.2f} USDT + fees)."
            )

        # 5. Model Inference
        pred = self.predict_model_action(candles, balance)
        if pred.action == Action.HOLD:
            self.journal.update_claim_status(self.runner_id, "COMPLETED")
            return {
                "status": "HOLD",
                "action": "HOLD",
                "orders_placed": 0,
                "message": "RL Engine decided HOLD at current market juncture. No orders dispatched.",
            }

        if pred.action not in (Action.OPEN_LONG, Action.OPEN_SHORT):
            raise MainnetDemoError(f"Unexpected entry action from model: {pred.action}")

        is_long = pred.action == Action.OPEN_LONG
        entry_side = "buy" if is_long else "sell"
        exit_side = "sell" if is_long else "buy"

        # 6. Arming Verification
        if not self.config.is_armed():
            raise MainnetDemoDisarmedError(
                "Execution is DISARMED. Explicit manual arming confirmation required before dispatch."
            )

        self.journal.set_state(DemoState.ARMED)
        self.journal.update_claim_status(self.runner_id, "EXECUTING")

        # 7. Entry Order Dispatch & Definitive Partial Fill Establishing
        entry_coid = f"demo_entry_{int(time.time()*1000)}"
        self.journal.record_order(
            client_order_id=entry_coid,
            order_type="MARKET",
            side=entry_side,
            requested_qty=qty,
            status="SUBMITTED",
        )

        entry_res = self.client.create_entry_market_order(
            symbol=self.config.symbol,
            side=entry_side,
            amount=qty,
            client_order_id=entry_coid,
        )

        entry_oid = str(entry_res.get("id") or entry_res.get("orderId", ""))
        filled_qty = float(entry_res.get("filled", 0.0) or 0.0)
        avg_entry_price = float(entry_res.get("average", 0.0) or mark_price)
        entry_status = str(entry_res.get("status", "")).lower()

        # If entry execution is not definitively closed or is partial, establish final executed quantity
        if entry_status in ("open", "partially_filled") or (0.0 < filled_qty < (qty - 1e-8)):
            # Cancel unfilled remainder to prevent subsequent unexpected exposure
            try:
                self.client.cancel_order(self.config.symbol, order_id=entry_oid, client_order_id=entry_coid)
            except Exception:
                pass
            # Query exchange to confirm exact terminal executed quantity
            try:
                updated_entry = self.client.fetch_order(
                    self.config.symbol, order_id=entry_oid, client_order_id=entry_coid
                )
                filled_qty = float(updated_entry.get("filled", filled_qty) or filled_qty)
                avg_entry_price = float(updated_entry.get("average", avg_entry_price) or avg_entry_price)
            except Exception:
                pass

        fee_info = entry_res.get("fee", {}) or {}
        entry_fee = float(fee_info.get("cost", 0.0) or (filled_qty * avg_entry_price * self.config.taker_fee_rate))

        self.journal.update_order(
            client_order_id=entry_coid,
            status="FILLED" if filled_qty >= (qty - 1e-8) else "PARTIALLY_FILLED",
            filled_qty=filled_qty,
            avg_price=avg_entry_price,
            fee=entry_fee,
            exchange_order_id=entry_oid,
        )

        if filled_qty <= 0.0:
            self.journal.trigger_permanent_lock("ENTRY_ZERO_FILL_FAILURE")
            raise MainnetDemoError("Entry order executed with zero fill quantity. Demo locked.")

        # Partial fill handling: Stop order amount MUST strictly match established executed quantity
        stop_amount = filled_qty

        # 8. Protective Stop Placement & Fallback
        if is_long:
            stop_price = avg_entry_price * (1.0 - self.config.stop_loss_pct)
        else:
            stop_price = avg_entry_price * (1.0 + self.config.stop_loss_pct)

        stop_coid = f"demo_stop_{int(time.time()*1000)}"
        self.journal.record_order(
            client_order_id=stop_coid,
            order_type="STOP_MARKET",
            side=exit_side,
            requested_qty=stop_amount,
            price=stop_price,
            status="SUBMITTED",
        )

        try:
            stop_res = self.client.create_protective_stop_order(
                symbol=self.config.symbol,
                side=exit_side,
                amount=stop_amount,
                stop_price=stop_price,
                entry_price=avg_entry_price,
                tick_size=filters["tick_size"],
                step_size=filters["step_size"],
                client_order_id=stop_coid,
            )
            stop_oid = str(stop_res.get("id") or stop_res.get("orderId", ""))
            self.journal.update_order(
                client_order_id=stop_coid,
                status="OPEN",
                exchange_order_id=stop_oid,
            )
        except Exception as stop_err:
            # Emergency Fail-Closed Fallback: Close position immediately via reduceOnly market order
            emerg_coid = f"demo_emerg_{int(time.time()*1000)}"
            try:
                self.client.create_exit_market_order(
                    symbol=self.config.symbol,
                    side=exit_side,
                    amount=stop_amount,
                    step_size=filters["step_size"],
                    client_order_id=emerg_coid,
                    is_emergency=True,
                )
            except Exception:
                pass

            # Verify position is flat
            try:
                pos_check = self.client.fetch_positions([self.config.symbol])
                flat = True
                for p in pos_check:
                    if abs(float(p.get("contracts", 0.0) or p.get("positionAmt", 0.0) or 0.0)) > 1e-8:
                        flat = False
                if not flat:
                    self.journal.trigger_permanent_lock("EMERGENCY_EXIT_UNRESOLVED_EXPOSURE")
            except Exception:
                pass

            self.journal.trigger_permanent_lock(f"STOP_PLACEMENT_FAILED: {stop_err}")
            raise EmergencyExitError(
                f"Protective stop placement failed ({stop_err}). "
                "Emergency reduce-only market exit dispatched. Demo permanently LOCKED."
            ) from stop_err

        # 9. Correct Exit Sequence
        # Step 9a: Close the position first via market reduce-only order
        exit_coid = f"demo_exit_{int(time.time()*1000)}"
        self.journal.record_order(
            client_order_id=exit_coid,
            order_type="MARKET",
            side=exit_side,
            requested_qty=stop_amount,
            status="SUBMITTED",
        )

        exit_res = self.client.create_exit_market_order(
            symbol=self.config.symbol,
            side=exit_side,
            amount=stop_amount,
            step_size=filters["step_size"],
            client_order_id=exit_coid,
            is_emergency=False,
        )

        exit_oid = str(exit_res.get("id") or exit_res.get("orderId", ""))
        # Never substitute requested quantity when exchange reports zero or missing filled
        raw_exit_filled = exit_res.get("filled")
        if raw_exit_filled is None or float(raw_exit_filled) <= 0.0:
            try:
                fetched_exit = self.client.fetch_order(
                    self.config.symbol, order_id=exit_oid, client_order_id=exit_coid
                )
                raw_exit_filled = fetched_exit.get("filled", 0.0)
            except Exception:
                raw_exit_filled = 0.0

        exit_filled = float(raw_exit_filled or 0.0)
        if exit_filled <= 0.0:
            self.journal.trigger_permanent_lock("EXIT_EXECUTION_UNCONFIRMED_ZERO_FILL")
            raise EmergencyExitError("Exit order executed with zero confirmed fill quantity. Demo locked.")

        avg_exit_price = float(exit_res.get("average", 0.0) or mark_price)
        exit_fee_info = exit_res.get("fee", {}) or {}
        exit_fee = float(exit_fee_info.get("cost", 0.0) or (exit_filled * avg_exit_price * self.config.taker_fee_rate))

        self.journal.update_order(
            client_order_id=exit_coid,
            status="FILLED" if exit_filled >= (stop_amount - 1e-8) else "PARTIALLY_FILLED",
            filled_qty=exit_filled,
            avg_price=avg_exit_price,
            fee=exit_fee,
            exchange_order_id=exit_oid,
        )

        # Step 9b: Verify position is flat
        pos_after_exit = self.client.fetch_positions([self.config.symbol])
        for p in pos_after_exit:
            rem_amt = abs(float(p.get("contracts", 0.0) or p.get("positionAmt", 0.0) or 0.0))
            if rem_amt > 1e-8:
                self.journal.trigger_permanent_lock(
                    f"EXIT_VERIFICATION_FAILED: Remaining position {rem_amt} after exit."
                )
                raise MainnetDemoError(
                    f"Position closure verification failed: {rem_amt} contracts remaining. Demo LOCKED."
                )

        # Step 9c: Cancel the remaining protective stop order
        try:
            self.client.cancel_order(
                symbol=self.config.symbol,
                client_order_id=stop_coid,
            )
            self.journal.update_order(client_order_id=stop_coid, status="CANCELED")
        except Exception:
            pass

        # Step 9d: Cancel resting open orders and verify zero remain
        open_orders_after = self.client.fetch_open_orders(self.config.symbol)
        if len(open_orders_after) > 0:
            for o in open_orders_after:
                try:
                    self.client.cancel_order(self.config.symbol, order_id=o.get("id"))
                except Exception:
                    pass

        # Mandatory second query: verify strictly 0 open orders remaining before declaring COMPLETED
        final_open_orders = self.client.fetch_open_orders(self.config.symbol)
        if len(final_open_orders) > 0:
            self.journal.trigger_permanent_lock(
                f"RESIDUAL_OPEN_ORDERS_DETECTED: {len(final_open_orders)} open orders remain on exchange."
            )
            raise MainnetDemoError(
                f"Residual open orders remain on exchange ({len(final_open_orders)}). Demo LOCKED."
            )

        # 10. Reconcile Realized PnL, Fees, Funding & Permanent Lockout
        if is_long:
            trade_pnl = (avg_exit_price - avg_entry_price) * exit_filled
        else:
            trade_pnl = (avg_entry_price - avg_exit_price) * exit_filled

        total_trade_fees = entry_fee + exit_fee

        # Attempt to reconcile against actual exchange trade fills if available
        try:
            user_trades = self.client.fetch_my_trades(self.config.symbol)
            if user_trades:
                trade_records = [
                    t for t in user_trades if str(t.get("order", "")) in (entry_oid, exit_oid)
                ]
                if trade_records:
                    matched_pnl = sum(float(t.get("realizedPnl", 0.0) or 0.0) for t in trade_records)
                    matched_fees = sum(float(t.get("fee", {}).get("cost", 0.0) or 0.0) for t in trade_records)
                    if abs(matched_pnl) > 1e-8:
                        trade_pnl = matched_pnl
                    if matched_fees > 0.0:
                        total_trade_fees = matched_fees
        except Exception:
            pass

        net_trade_result = trade_pnl - total_trade_fees
        if net_trade_result < 0:
            self.journal.record_realized_loss(abs(net_trade_result))

        self.journal.update_claim_status(self.runner_id, "COMPLETED")
        self.journal.trigger_permanent_lock("ONE_TRADE_DEMO_COMPLETED")

        return {
            "status": "COMPLETED",
            "action": "OPEN_LONG" if is_long else "OPEN_SHORT",
            "entry_order_id": entry_oid,
            "entry_price": avg_entry_price,
            "stop_order_id": stop_oid,
            "stop_price": stop_price,
            "exit_order_id": exit_oid,
            "exit_price": avg_exit_price,
            "executed_qty": exit_filled,
            "realized_pnl_usdt": trade_pnl,
            "total_fees_usdt": total_trade_fees,
            "net_result_usdt": net_trade_result,
            "remaining_loss_budget_usdt": self.journal.get_remaining_budget(self.config.max_loss_budget_usdt),
            "final_state": DemoState.LOCKED.value,
        }

