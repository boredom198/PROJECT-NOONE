"""Runtime Exchange Orchestrator for Component 3.

Coordinates Component 1 DQN actions (HOLD, OPEN_LONG, CLOSE_LONG, OPEN_SHORT, CLOSE_SHORT),
pre-flight handshake, account/position synchronization, risk checks, order execution,
and persistent journaling.
Conforms strictly to PROJECT NOONE Technical Design Specification V2.2.4.
"""

from datetime import datetime, timezone
from decimal import Decimal
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

from noone.rl_engine.inference import AccountSnapshot, Action
from noone.exchange.account import (
    audit_account_positions,
    audit_open_orders,
    parse_account_snapshot,
)
from noone.exchange.client import BinanceFuturesClient
from noone.exchange.config import Component3Config
from noone.exchange.errors import (
    AccountSyncError,
    DailyLossLimitError,
    DailyPnLUninitializedError,
    ExchangeError,
    FilterValidationError,
    ResidualPositionError,
    RiskGateError,
    SafetyFreezeError,
    StateDiscrepancyError,
)
from noone.exchange.journal import DailyBaselineRecord, ExecutionJournal, JournalOrder
from noone.exchange.precision import (
    check_residual_dust,
    parse_symbol_filters,
    ParsedSymbolFilters,
    quantize_downward,
)
from noone.exchange.reconciliation import ReconciliationManager
from noone.exchange.risk_gate import RiskGate


class ExchangeOrchestrator:
    """Central coordinator for Component 3 exchange execution."""

    def __init__(
        self,
        config: Component3Config,
        client: Optional[BinanceFuturesClient] = None,
        journal: Optional[ExecutionJournal] = None,
    ):
        self.config = config
        self.journal = journal or ExecutionJournal(config.database_path)
        self.client = client or BinanceFuturesClient(config)
        self.risk_gate = RiskGate(self.config, self.journal)
        self.reconciliation = ReconciliationManager(self.config, self.client, self.journal)

        self._symbol_filters: Optional[ParsedSymbolFilters] = None
        self._filters_cached_at: float = 0.0
        self._last_account_snapshot: Optional[AccountSnapshot] = None
        self._session_realized_pnl: float = 0.0

    @property
    def session_realized_pnl(self) -> float:
        """Cumulative realized PnL of closed trades in the current agent session."""
        return self._session_realized_pnl

    def set_session_realized_pnl(self, val: float) -> None:
        """Set session realized PnL (starts at 0.0 on launch)."""
        self._session_realized_pnl = float(val)

    # -------------------------------------------------------------------------
    # Pre-Flight Handshake (Section 12)
    # -------------------------------------------------------------------------

    def pre_flight_handshake(self) -> Dict[str, Any]:
        """Perform comprehensive pre-flight verification sequence.

        1. Clock drift check.
        2. Account trading capability probe.
        3. Position mode probe (One-Way mode).
        4. Symbol filter probe.
        5. Account-wide exposure audit (0 non-target positions).
        6. Account-wide open order audit (0 open orders).
        7. Inspect unresolved orders from prior sessions.
        """
        results: Dict[str, Any] = {}

        # 1. Clock drift probe
        drift_ms = self.client.verify_clock_drift()
        results["clock_drift_ms"] = drift_ms

        # 2. Account trading capability probe
        account_data = self.client.fetch_account_v2()
        can_trade = account_data.get("canTrade", False)
        if not can_trade:
            raise AccountSyncError("Exchange account trading is disabled (canTrade == False).")
        results["can_trade"] = True

        # 3. Position mode probe
        pos_mode_res = self.client.fetch_position_side_dual()
        dual_side = pos_mode_res.get("dualSidePosition", True)
        if dual_side:
            self.journal.trigger_safety_freeze(
                "AMBIGUOUS_POSITION_STATE",
                "Exchange account is configured in Hedge Mode (dualSidePosition == True).",
            )
            raise StateDiscrepancyError(
                "Exchange account is configured in Hedge Mode. One-Way Mode is required."
            )
        results["one_way_mode"] = True

        # 4. Symbol filter probe
        self._load_symbol_filters(force_refresh=True)
        results["filters_loaded"] = True

        # 5. Account-wide exposure audit
        pos_risk_data = self.client.fetch_position_risk_v2()
        pos_audit = audit_account_positions(pos_risk_data, self.config.raw_symbol)
        if pos_audit.has_external_positions:
            self.journal.trigger_safety_freeze(
                "ACCOUNT_WIDE_EXPOSURE_DETECTED",
                f"External positions detected on account: {pos_audit.external_positions}",
            )
            raise RiskGateError(
                f"Pre-flight failed: external positions detected: {pos_audit.external_positions}"
            )
        results["zero_external_positions"] = True

        # 6. Account-wide open order audit
        open_orders = self.client.fetch_open_orders()
        if open_orders:
            self.journal.trigger_safety_freeze(
                "ACCOUNT_WIDE_EXPOSURE_DETECTED",
                f"Open orders detected on account: {len(open_orders)}",
            )
            raise RiskGateError(
                f"Pre-flight failed: {len(open_orders)} open resting orders found on exchange."
            )
        results["zero_open_orders"] = True

        # 7. Check for unresolved startup orders in SQLite
        unresolved = self.journal.get_unresolved_orders()
        if unresolved:
            results["unresolved_startup_orders"] = len(unresolved)
            # Run recovery for each unresolved order
            for uo in unresolved:
                self.reconciliation.recover_unknown_order(uo.client_order_id)

        return results

    # -------------------------------------------------------------------------
    # Symbol Filters Cache
    # -------------------------------------------------------------------------

    def _load_symbol_filters(self, force_refresh: bool = False) -> ParsedSymbolFilters:
        now = time.time()
        if (
            not force_refresh
            and self._symbol_filters is not None
            and (now - self._filters_cached_at) < self.config.filter_cache_ttl_s
        ):
            return self._symbol_filters

        ex_info = self.client.fetch_exchange_info(self.config.raw_symbol)
        symbols = ex_info.get("symbols", [])
        raw_target = self.config.raw_symbol

        target_sym_info = None
        for s in symbols:
            if s.get("symbol", "").upper() == raw_target:
                target_sym_info = s
                break

        if not target_sym_info:
            raise FilterValidationError(f"Target symbol {raw_target} not found in exchangeInfo.")

        filters = target_sym_info.get("filters", [])
        parsed = parse_symbol_filters(filters)
        self._symbol_filters = parsed
        self._filters_cached_at = now
        return parsed

    # -------------------------------------------------------------------------
    # Account & Position Synchronization
    # -------------------------------------------------------------------------

    def sync_account(self) -> AccountSnapshot:
        """Synchronize exchange account and positions and produce verified AccountSnapshot."""
        account_data = self.client.fetch_account_v2()
        pos_risk_data = self.client.fetch_position_risk_v2()

        # Audit external exposure
        audit = audit_account_positions(pos_risk_data, self.config.raw_symbol)
        if audit.has_external_positions:
            self.journal.trigger_safety_freeze(
                "ACCOUNT_WIDE_EXPOSURE_DETECTED",
                f"External positions detected on {len(audit.external_positions)} symbols.",
            )

        snapshot = parse_account_snapshot(
            account_data=account_data,
            position_risk_data=pos_risk_data,
            target_symbol=self.config.raw_symbol,
            session_realized_pnl=self._session_realized_pnl,
        )
        self._last_account_snapshot = snapshot
        self.journal.log_sync_audit("account_sync", "SUCCESS", f"Equity: {snapshot.equity}")
        return snapshot

    # -------------------------------------------------------------------------
    # UTC Midnight Baseline Sampling (Section 10.3)
    # -------------------------------------------------------------------------

    def sample_utc_midnight_baseline(self) -> Optional[DailyBaselineRecord]:
        """Direct REST sampling of UTC midnight baseline within [00:00:00, 00:00:05] UTC.

        If offline at rollover or sampling fails, marks baseline as INVALID and triggers freeze.
        """
        now_dt = datetime.now(timezone.utc)
        date_str = now_dt.strftime("%Y-%m-%d")

        try:
            account_data = self.client.fetch_account_v2()
            raw_equity = account_data.get("totalMarginBalance")
            if raw_equity is None:
                raise AccountSyncError("totalMarginBalance missing in account data.")
            equity_val = float(raw_equity)
            if equity_val <= 0.0:
                raise AccountSyncError(f"Invalid non-positive equity sampled: {equity_val}")

            self.journal.record_daily_baseline(
                date_utc=date_str,
                baseline_equity=equity_val,
                verified_status="VERIFIED_UTC_BASELINE",
                note=f"Sampled at {now_dt.isoformat()}",
            )
            # If DAILY_PNL_UNINITIALIZED or DAILY_LOSS_LIMIT freeze was active, clear it
            self.journal.clear_safety_freeze(
                "DAILY_PNL_UNINITIALIZED", "Clean VERIFIED_UTC_BASELINE captured at midnight rollover."
            )
            self.journal.clear_safety_freeze(
                "DAILY_LOSS_LIMIT", "Daily loss limit reset at UTC midnight rollover."
            )
            return self.journal.get_daily_baseline(date_str)
        except Exception as e:
            self.journal.record_daily_baseline(
                date_utc=date_str,
                baseline_equity=0.0,
                verified_status="INVALID",
                note=f"Sampling failed: {e}",
            )
            self.journal.trigger_safety_freeze(
                "DAILY_PNL_UNINITIALIZED",
                f"UTC midnight baseline sampling failed for {date_str}: {e}",
            )
            return None

    # -------------------------------------------------------------------------
    # DQN Action Execution Pipeline
    # -------------------------------------------------------------------------

    def _generate_client_order_id(self) -> str:
        """Generate deterministic, collision-proof client order ID."""
        ts = int(time.time() * 1000)
        uid = uuid.uuid4().hex[:8]
        return f"noone_{ts}_{uid}"

    def execute_action(
        self, action: Action, mark_price: Optional[float] = None
    ) -> Optional[JournalOrder]:
        """Execute a discrete Action (0..4) emitted by Component 1 DQN."""
        if action == Action.HOLD:
            return None

        if action in (Action.OPEN_LONG, Action.OPEN_SHORT):
            return self._execute_entry_order(action, mark_price)
        elif action in (Action.CLOSE_LONG, Action.CLOSE_SHORT):
            return self._execute_exit_order(action, mark_price)
        else:
            raise ValueError(f"Unknown action: {action}")

    def _execute_entry_order(
        self, action: Action, mark_price: Optional[float]
    ) -> JournalOrder:
        # Check permissions
        can_enter, reason = self.reconciliation.can_execute_entry()
        if not can_enter:
            raise SafetyFreezeError(reason or "Entry blocked by safety freeze.")

        # Date & daily baseline check
        now_dt = datetime.now(timezone.utc)
        date_str = now_dt.strftime("%Y-%m-%d")
        baseline = self.journal.get_daily_baseline(date_str)
        if not baseline or baseline.verified_status != "VERIFIED_UTC_BASELINE":
            status = baseline.verified_status if baseline else "MISSING"
            self.journal.trigger_safety_freeze(
                "DAILY_PNL_UNINITIALIZED",
                f"Cannot execute entry: daily baseline status is {status}",
            )
            raise DailyPnLUninitializedError(
                f"Entry blocked: daily baseline status is {status}. Must be VERIFIED_UTC_BASELINE."
            )

        # Fresh data fetch
        fresh_snapshot = self.sync_account()
        filters = self._load_symbol_filters()

        if mark_price is None:
            mark_price = self.client.fetch_mark_price(self.config.raw_symbol)
        mark_price_dec = Decimal(str(mark_price))

        # Daily loss limit check
        midnight_dt = datetime(now_dt.year, now_dt.month, now_dt.day, tzinfo=timezone.utc)
        midnight_ts_ms = int(midnight_dt.timestamp() * 1000)
        net_transfers = Decimal(str(self.journal.get_net_external_transfers(midnight_ts_ms)))
        equity_dec = Decimal(str(fresh_snapshot.equity))
        baseline_dec = Decimal(str(baseline.baseline_equity))

        self.risk_gate.check_daily_pnl_limit(
            current_equity=equity_dec,
            baseline_equity=baseline_dec,
            net_external_transfers=net_transfers,
            verified_status=baseline.verified_status,
        )

        # Pre-dispatch fresh checks
        all_positions = self.client.fetch_position_risk_v2()
        open_orders = self.client.fetch_open_orders()
        pos_audit = audit_account_positions(all_positions, self.config.raw_symbol)
        target_amt = Decimal(str(pos_audit.target_position.get("positionAmt", "0"))) if pos_audit.target_position else Decimal("0")

        self.risk_gate.validate_pre_dispatch(
            account_snapshot=fresh_snapshot,
            mark_price=mark_price_dec,
            filters=filters,
            target_pos_amt=target_amt,
            all_positions=all_positions,
            open_orders=open_orders,
            is_entry=True,
        )

        # Sizing
        q_dec = self.risk_gate.compute_entry_sizing(
            account_snapshot=fresh_snapshot,
            mark_price=mark_price_dec,
            filters=filters,
        )
        qty_float = float(q_dec)

        side = "BUY" if action == Action.OPEN_LONG else "SELL"
        action_name = "OPEN_LONG" if action == Action.OPEN_LONG else "OPEN_SHORT"
        client_oid = self._generate_client_order_id()

        # Commit intent to SQLite before socket write
        self.journal.create_order_intent(
            client_order_id=client_oid,
            symbol=self.config.raw_symbol,
            action_type=action_name,
            side=side,
            quantity=qty_float,
            reduce_only=False,
            order_type="MARKET",
        )

        # Transition to DISPATCHING
        self.journal.transition_order_state(
            client_order_id=client_oid,
            to_state="DISPATCHING",
            reason="Dispatching market order to exchange",
        )

        # Socket call
        try:
            res = self.client.create_market_order(
                symbol=self.config.ccxt_symbol,
                side=side,
                amount=qty_float,
                client_order_id=client_oid,
                reduce_only=False,
            )
            # Success
            exchange_oid = str(res.get("id") or res.get("orderId", ""))
            status = str(res.get("status", "")).upper()
            filled_qty = float(res.get("filled", 0.0))
            cum_quote = float(res.get("cost", 0.0))
            avg_price = float(res.get("average", 0.0))
            if avg_price <= 0.0 and filled_qty > 0.0 and cum_quote > 0.0:
                avg_price = cum_quote / filled_qty

            final_state = "FILLED" if status in ("CLOSED", "FILLED") else "OPEN"
            self.journal.transition_order_state(
                client_order_id=client_oid,
                to_state=final_state,
                reason="Order successfully placed on exchange",
                exchange_order_id=exchange_oid,
                filled_quantity=filled_qty,
                cum_quote=cum_quote,
                avg_price=avg_price,
                reconciliation_state="VERIFIED",
            )
            return self.journal.get_order(client_oid)  # type: ignore
        except Exception as e:
            # Network timeout, socket error, or exchange error
            self.journal.transition_order_state(
                client_order_id=client_oid,
                to_state="UNKNOWN",
                reason=f"Order dispatch error: {e}",
                error_message=str(e),
                reconciliation_state="DISCREPANCY",
            )
            self.journal.trigger_safety_freeze(
                "UNKNOWN_ORDER_OUTCOME",
                f"Dispatch error on {client_oid}: {e}",
            )
            raise ExchangeError(f"Order dispatch failed for {client_oid}: {e}") from e

    def _execute_exit_order(
        self, action: Action, mark_price: Optional[float]
    ) -> JournalOrder:
        # Check permissions (exits permitted if independently verified)
        can_exit, reason = self.reconciliation.can_execute_exit(independently_verified=True)
        if not can_exit:
            raise SafetyFreezeError(reason or "Exit blocked by safety freeze.")

        # Fresh data fetch
        fresh_snapshot = self.sync_account()
        filters = self._load_symbol_filters()

        if mark_price is None:
            mark_price = self.client.fetch_mark_price(self.config.raw_symbol)
        mark_price_dec = Decimal(str(mark_price))

        # Target position verification
        all_positions = self.client.fetch_position_risk_v2()
        pos_audit = audit_account_positions(all_positions, self.config.raw_symbol)
        target_pos = pos_audit.target_position
        if not target_pos:
            raise StateDiscrepancyError("Cannot close position: target symbol position record missing.")

        pos_amt_dec = Decimal(str(target_pos.get("positionAmt", "0")))
        if action == Action.CLOSE_LONG and pos_amt_dec <= Decimal("1e-8"):
            raise StateDiscrepancyError(f"Cannot CLOSE_LONG: position amount is {pos_amt_dec}")
        if action == Action.CLOSE_SHORT and pos_amt_dec >= Decimal("-1e-8"):
            raise StateDiscrepancyError(f"Cannot CLOSE_SHORT: position amount is {pos_amt_dec}")

        # Quantize closing quantity strictly downward
        abs_qty = abs(pos_amt_dec)
        q_order = quantize_downward(abs_qty, filters.step_size)

        if q_order <= Decimal("0"):
            # Position amount is smaller than stepSize -> dust
            self.journal.trigger_safety_freeze(
                "POSITION_ACCOUNT_DISCREPANCY",
                f"Position quantity {abs_qty} is smaller than stepSize {filters.step_size} (dust).",
            )
            raise ResidualPositionError(
                f"Position quantity {abs_qty} cannot be closed: below stepSize {filters.step_size}."
            )

        # Check for unclosable residual after exit
        residual_after = abs_qty - q_order
        if check_residual_dust(residual_after, mark_price_dec, filters):
            self.journal.trigger_safety_freeze(
                "POSITION_ACCOUNT_DISCREPANCY",
                f"Exit order would leave unclosable residual dust: {residual_after}",
            )

        side = "SELL" if action == Action.CLOSE_LONG else "BUY"
        action_name = "CLOSE_LONG" if action == Action.CLOSE_LONG else "CLOSE_SHORT"
        client_oid = self._generate_client_order_id()
        qty_float = float(q_order)

        # Commit intent to SQLite before socket write
        self.journal.create_order_intent(
            client_order_id=client_oid,
            symbol=self.config.raw_symbol,
            action_type=action_name,
            side=side,
            quantity=qty_float,
            reduce_only=True,
            order_type="MARKET",
        )

        # Transition to DISPATCHING
        self.journal.transition_order_state(
            client_order_id=client_oid,
            to_state="DISPATCHING",
            reason="Dispatching reduce-only market close order to exchange",
        )

        # Socket call
        try:
            res = self.client.create_market_order(
                symbol=self.config.ccxt_symbol,
                side=side,
                amount=qty_float,
                client_order_id=client_oid,
                reduce_only=True,
            )
            exchange_oid = str(res.get("id") or res.get("orderId", ""))
            status = str(res.get("status", "")).upper()
            filled_qty = float(res.get("filled", 0.0))
            cum_quote = float(res.get("cost", 0.0))
            avg_price = float(res.get("average", 0.0))
            if avg_price <= 0.0 and filled_qty > 0.0 and cum_quote > 0.0:
                avg_price = cum_quote / filled_qty

            final_state = "FILLED" if status in ("CLOSED", "FILLED") else "OPEN"

            # Calculate and accumulate realized PnL
            # Entry price from position
            entry_p = float(target_pos.get("entryPrice", 0.0))
            if entry_p > 0.0 and avg_price > 0.0 and filled_qty > 0.0:
                if action == Action.CLOSE_LONG:
                    trade_rpnl = (avg_price - entry_p) * filled_qty
                else:
                    trade_rpnl = (entry_p - avg_price) * filled_qty
                self._session_realized_pnl += trade_rpnl

            self.journal.transition_order_state(
                client_order_id=client_oid,
                to_state=final_state,
                reason="Close order successfully executed",
                exchange_order_id=exchange_oid,
                filled_quantity=filled_qty,
                cum_quote=cum_quote,
                avg_price=avg_price,
                reconciliation_state="VERIFIED",
            )
            return self.journal.get_order(client_oid)  # type: ignore
        except Exception as e:
            self.journal.transition_order_state(
                client_order_id=client_oid,
                to_state="UNKNOWN",
                reason=f"Exit order dispatch error: {e}",
                error_message=str(e),
                reconciliation_state="DISCREPANCY",
            )
            self.journal.trigger_safety_freeze(
                "UNKNOWN_ORDER_OUTCOME",
                f"Dispatch error on exit order {client_oid}: {e}",
            )
            raise ExchangeError(f"Exit order dispatch failed for {client_oid}: {e}") from e
