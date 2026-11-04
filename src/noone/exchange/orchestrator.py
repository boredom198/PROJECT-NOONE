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
    InvalidBaselineError,
    ResidualPositionError,
    RiskGateError,
    SafetyFreezeError,
    StateDiscrepancyError,
    StaleMarketDataError,
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
        self._bracket_max_notional: Optional[Decimal] = None
        self._brackets_cached_at: float = 0.0
        self._last_account_snapshot: Optional[AccountSnapshot] = None
        self._session_realized_pnl: float = 0.0
        self.verified_external_transfer_ids: Optional[Set[str]] = None

    @property
    def session_realized_pnl(self) -> float:
        """Cumulative realized PnL of closed trades in the current agent session."""
        return self._session_realized_pnl

    def set_session_realized_pnl(self, val: float) -> None:
        """Set session realized PnL (starts at 0.0 on launch)."""
        self._session_realized_pnl = float(val)

    def set_verified_external_transfers(self, transfer_ids: Any) -> None:
        """Configure set of verified external transfer IDs for daily PnL accounting."""
        if isinstance(transfer_ids, dict):
            self.verified_external_transfer_ids = {str(k) for k, v in transfer_ids.items() if v}
        elif isinstance(transfer_ids, (list, set, tuple)):
            self.verified_external_transfer_ids = {str(x) for x in transfer_ids}
        else:
            self.verified_external_transfer_ids = None

    # -------------------------------------------------------------------------
    # Pre-Flight Handshake (Section 12)
    # -------------------------------------------------------------------------

    def pre_flight_handshake(self) -> Dict[str, Any]:
        """Perform comprehensive pre-flight verification sequence.

        1. Hard production lock check.
        2. Clock drift check.
        3. Account trading capability probe.
        4. Position mode probe (One-Way mode).
        5. Symbol filter probe.
        6. Leverage bracket probe.
        7. Account-wide exposure audit (0 non-target positions).
        8. Account-wide open order audit (0 open orders).
        9. Inspect unresolved orders from prior sessions.
        """
        # Defect 9: Hard production lock
        if not self.config.testnet or self.config.production_confirmed:
            raise ExchangeError(
                "PRODUCTION EXECUTION HARD LOCKED: Pre-flight handshake blocked by hard runtime lock."
            )

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

        # 5. Leverage bracket probe
        bracket_cap = self._load_leverage_bracket(force_refresh=True)
        results["bracket_max_notional"] = float(bracket_cap)

        # 6. Account-wide exposure audit
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

        # 7. Account-wide open order audit
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

        # 8. Check for unresolved startup orders in SQLite
        unresolved = self.journal.get_unresolved_orders()
        if unresolved:
            results["unresolved_startup_orders"] = len(unresolved)
            for uo in unresolved:
                self.reconciliation.recover_unknown_order(uo.client_order_id)

        return results

    # -------------------------------------------------------------------------
    # Symbol Filters & Leverage Bracket Cache (Defects 5 & 6)
    # -------------------------------------------------------------------------

    def _load_symbol_filters(self, force_refresh: bool = False) -> ParsedSymbolFilters:
        now = time.time()
        if (
            not force_refresh
            and self._symbol_filters is not None
            and (now - self._filters_cached_at) < self.config.filter_cache_ttl_s
        ):
            return self._symbol_filters

        try:
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
        except Exception:
            if self._symbol_filters is not None:
                return self._symbol_filters
            raise

    def _load_leverage_bracket(self, force_refresh: bool = False) -> Decimal:
        """Fetch and validate leverage brackets, returning bracket max notional."""
        now = time.time()
        if (
            not force_refresh
            and self._bracket_max_notional is not None
            and (now - self._brackets_cached_at) < self.config.filter_cache_ttl_s
        ):
            return self._bracket_max_notional

        try:
            brackets_res = self.client.fetch_leverage_bracket(self.config.raw_symbol)
            if not brackets_res:
                raise RiskGateError(f"Leverage bracket response empty for {self.config.raw_symbol}")

            target_bracket_item = None
            if isinstance(brackets_res, dict):
                brackets_res = [brackets_res]
            for b in brackets_res:
                if isinstance(b, dict):
                    sym = b.get("symbol")
                    if sym is not None:
                        if sym.upper() == self.config.raw_symbol:
                            target_bracket_item = b
                            break
                    elif "brackets" in b:
                        target_bracket_item = b
                        break

            if not target_bracket_item or "brackets" not in target_bracket_item:
                raise RiskGateError(f"No leverage bracket found for symbol {self.config.raw_symbol}")

            tier_list = target_bracket_item["brackets"]
            if not tier_list:
                raise RiskGateError(f"Empty bracket tiers for {self.config.raw_symbol}")

            cfg_lev = self.config.leverage
            matched_tier = None
            for tier in tier_list:
                init_lev = int(tier.get("initialLeverage", 0))
                if init_lev >= cfg_lev:
                    matched_tier = tier
                    break

            if matched_tier is None:
                raise RiskGateError(
                    f"No valid bracket tier matches configured leverage {cfg_lev} for {self.config.raw_symbol}."
                )

            notional_cap_val = matched_tier.get("notionalCap")
            if notional_cap_val is None:
                notional_cap_val = matched_tier.get("maxNotional")
            if notional_cap_val is None:
                raise RiskGateError(f"Missing notionalCap in leverage bracket tier for {self.config.raw_symbol}")

            notional_cap = Decimal(str(notional_cap_val))
            if notional_cap <= Decimal("0"):
                raise RiskGateError(f"Invalid non-positive notionalCap in leverage bracket: {notional_cap}")

            self._bracket_max_notional = notional_cap
            self._brackets_cached_at = now
            return notional_cap
        except Exception:
            if self._bracket_max_notional is not None:
                return self._bracket_max_notional
            raise
        self._brackets_cached_at = now
        return notional_cap

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
    # UTC Midnight Baseline Sampling (Defect 1)
    # -------------------------------------------------------------------------

    def sample_utc_midnight_baseline(self) -> Optional[DailyBaselineRecord]:
        """Direct REST sampling of UTC midnight baseline within [00:00:00, 00:00:05] UTC.

        Rules:
        - Sampling accepted ONLY within [00:00:00, 00:00:05] UTC.
        - Once established, baseline is completely immutable for that UTC day.
        - Out-of-window sampling must NOT establish or reset the baseline.
        - Sampling must NEVER clear an active daily-loss freeze on the same day.
        """
        now_dt = datetime.now(timezone.utc)
        date_str = now_dt.strftime("%Y-%m-%d")

        # Check existing baseline immutability
        existing = self.journal.get_daily_baseline(date_str)
        if existing and existing.verified_status == "VERIFIED_UTC_BASELINE":
            return existing

        # Window validation: strictly [00:00:00, 00:00:05] UTC
        is_in_window = (now_dt.hour == 0 and now_dt.minute == 0 and 0 <= now_dt.second <= 5)
        if not is_in_window:
            if not existing or existing.verified_status != "VERIFIED_UTC_BASELINE":
                self.journal.record_daily_baseline(
                    date_utc=date_str,
                    baseline_equity=0.0,
                    verified_status="INVALID",
                    note=f"Sample attempted outside valid [00:00:00, 00:00:05] window at {now_dt.time().isoformat()}",
                )
                self.journal.trigger_safety_freeze(
                    "DAILY_PNL_UNINITIALIZED",
                    f"UTC baseline sampling attempted outside valid [00:00:00, 00:00:05] window at {now_dt.time().isoformat()}",
                )
            raise InvalidBaselineError(
                f"UTC midnight baseline sampling rejected: outside [00:00:00, 00:00:05] window ({now_dt.time().isoformat()})."
            )

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
            self.journal.clear_safety_freeze(
                "DAILY_PNL_UNINITIALIZED", "Clean VERIFIED_UTC_BASELINE captured at midnight rollover."
            )

            # Defect 1: Never clear active daily loss freeze on the same day!
            active_freezes = self.journal.get_active_safety_freezes()
            for f in active_freezes:
                if f.reason == "DAILY_LOSS_LIMIT":
                    freeze_date = f.triggered_at[:10]
                    if freeze_date < date_str:
                        self.journal.clear_safety_freeze(
                            "DAILY_LOSS_LIMIT",
                            f"Daily loss limit from {freeze_date} reset at clean UTC rollover to {date_str}.",
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
    # Transfer History Synchronization (Defect 3)
    # -------------------------------------------------------------------------

    def sync_transfer_history(
        self, since_ms: int, until_ms: int
    ) -> Tuple[bool, Decimal, str]:
        """Fetch and reconcile income transfer history across [since_ms, until_ms].

        Paginates through GET /fapi/v1/income.
        Validates completeness and external classification.
        Fails closed (marks unverified and triggers freeze) if coverage is incomplete or ambiguous.
        """
        try:
            records = self.client.fetch_income_paginated(
                income_type="TRANSFER",
                start_time=since_ms,
                end_time=until_ms,
            )
        except Exception as e:
            self.journal.trigger_safety_freeze(
                "DAILY_PNL_UNINITIALIZED",
                f"Transfer history query failed: {e}",
            )
            return False, Decimal("0.0"), f"Query failed: {e}"

        net_external = Decimal("0.0")
        for r in records:
            itype = str(r.get("incomeType", ""))
            if itype != "TRANSFER":
                continue

            asset = str(r.get("asset", "")).upper()
            if asset != "USDT":
                tran_id = str(r.get("tranId", ""))
                inc = float(r.get("income", 0.0))
                ts = int(r.get("time", 0))
                self.journal.record_external_transfer(
                    tran_id=tran_id,
                    income_type=itype,
                    asset=asset,
                    income=inc,
                    timestamp=ts,
                    is_external=False,
                    note=f"Non-USDT transfer excluded from USDT margin baseline: {asset}",
                )
                continue

            tran_id = str(r.get("tranId", ""))
            inc = float(r.get("income", 0.0))
            ts = int(r.get("time", 0))

            # External vs internal classification
            # Binance USD(S)-M Futures GET /fapi/v1/income does not include an 'isExternal' field.
            # To be defensible, we check against verified external transfer IDs if provided.
            if self.verified_external_transfer_ids is not None:
                is_external_bool = str(tran_id) in self.verified_external_transfer_ids
            else:
                # Without authoritative external proof, transfer provenance is ambiguous.
                msg = (
                    f"Ambiguous transfer origin detected for tranId {tran_id}: "
                    "Binance USD(S)-M Futures /fapi/v1/income does not return external deposit/withdrawal "
                    "provenance ('isExternal' does not exist in exchange response). "
                    "Marking Daily PnL UNVERIFIED and blocking entry dispatch."
                )
                self.journal.trigger_safety_freeze("DAILY_PNL_UNINITIALIZED", msg)
                return False, Decimal("0.0"), msg

            self.journal.record_external_transfer(
                tran_id=tran_id,
                income_type=itype,
                asset=asset,
                income=inc,
                timestamp=ts,
                is_external=is_external_bool,
                note=f"Authoritatively verified external={is_external_bool}",
            )
            if is_external_bool:
                net_external += Decimal(str(inc))

        total_net = Decimal(str(self.journal.get_net_external_transfers(since_ms)))
        return True, total_net, "Success"

    # -------------------------------------------------------------------------
    # DQN Action Execution Pipeline
    # -------------------------------------------------------------------------

    def _generate_client_order_id(self) -> str:
        """Generate deterministic, collision-proof client order ID."""
        ts = int(time.time() * 1000)
        uid = uuid.uuid4().hex[:8]
        return f"noone_{ts}_{uid}"

    def execute_action(
        self,
        action: Action,
        mark_price: Optional[float] = None,
        mark_price_time_ms: Optional[int] = None,
    ) -> Optional[JournalOrder]:
        """Execute a discrete Action (0..4) emitted by Component 1 DQN."""
        if action == Action.HOLD:
            return None

        if action in (Action.OPEN_LONG, Action.OPEN_SHORT):
            return self._execute_entry_order(action, mark_price, mark_price_time_ms)
        elif action in (Action.CLOSE_LONG, Action.CLOSE_SHORT):
            return self._execute_exit_order(action, mark_price, mark_price_time_ms)
        else:
            raise ValueError(f"Unknown action: {action}")

    def _execute_entry_order(
        self,
        action: Action,
        mark_price: Optional[float],
        mark_price_time_ms: Optional[int] = None,
    ) -> JournalOrder:
        # Defect 9: Hard production lock
        if not self.config.testnet or self.config.production_confirmed:
            raise ExchangeError(
                "PRODUCTION EXECUTION HARD LOCKED: Entry order dispatch blocked by hard runtime lock."
            )

        # Check permissions
        can_enter, reason = self.reconciliation.can_execute_entry()
        if not can_enter:
            raise SafetyFreezeError(reason or "Entry blocked by safety freeze.")

        # Date & daily baseline check
        now_dt = datetime.now(timezone.utc)
        now_ms = int(time.time() * 1000)
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

        # Defect 3: Synchronize transfer history from UTC midnight to now
        midnight_dt = datetime(now_dt.year, now_dt.month, now_dt.day, tzinfo=timezone.utc)
        midnight_ts_ms = int(midnight_dt.timestamp() * 1000)
        transfers_ok, net_transfers, transfer_msg = self.sync_transfer_history(midnight_ts_ms, now_ms)
        if not transfers_ok:
            raise DailyPnLUninitializedError(f"Entry blocked: transfer sync failed or ambiguous: {transfer_msg}")

        # Fresh queries with timestamps captured strictly AFTER retrieval (Defect 2 & 7)
        fresh_snapshot = self.sync_account()
        acc_ts = int(time.time() * 1000)

        all_positions = self.client.fetch_position_risk_v2()
        pos_risk_ts = int(time.time() * 1000)

        open_orders = self.client.fetch_open_orders()
        open_orders_ts = int(time.time() * 1000)

        if mark_price is not None:
            if mark_price_time_ms is None:
                raise StaleMarketDataError(
                    "Caller-supplied mark_price requires explicit acquisition timestamp (mark_price_time_ms)."
                )
            mark_ts = int(mark_price_time_ms)
        else:
            mark_price, mark_ts = self.client.fetch_mark_price_data(self.config.raw_symbol)
        mark_price_dec = Decimal(str(mark_price))

        filters = self._load_symbol_filters()
        bracket_cap = self._load_leverage_bracket()

        pos_audit = audit_account_positions(all_positions, self.config.raw_symbol)
        target_pos = pos_audit.target_position
        if target_pos is None:
            self.journal.trigger_safety_freeze(
                "LEVERAGE_MISMATCH",
                f"Target position record for {self.config.raw_symbol} missing from exchange position risk response.",
            )
            raise RiskGateError(
                f"Target position record for {self.config.raw_symbol} missing from exchange position risk response; "
                "cannot verify effective leverage."
            )

        target_amt = Decimal(str(target_pos.get("positionAmt", "0")))
        actual_lev = int(target_pos.get("leverage", 0))

        # Daily loss limit check
        equity_dec = Decimal(str(fresh_snapshot.equity))
        baseline_dec = Decimal(str(baseline.baseline_equity))
        self.risk_gate.check_daily_pnl_limit(
            current_equity=equity_dec,
            baseline_equity=baseline_dec,
            net_external_transfers=net_transfers,
            verified_status=baseline.verified_status,
        )

        # Pre-dispatch fresh checks with timestamps (Defect 5 & 7)
        timestamps = {
            "mark_price": mark_ts,
            "target_position": pos_risk_ts,
            "account_snapshot": acc_ts,
            "all_positions": pos_risk_ts,
            "open_orders": open_orders_ts,
            "leverage": pos_risk_ts,
            "filters": int(self._filters_cached_at * 1000),
            "brackets": int(self._brackets_cached_at * 1000),
        }

        self.risk_gate.validate_pre_dispatch(
            account_snapshot=fresh_snapshot,
            mark_price=mark_price_dec,
            filters=filters,
            target_pos_amt=target_amt,
            all_positions=all_positions,
            open_orders=open_orders,
            is_entry=True,
            actual_leverage=actual_lev,
            bracket_max_notional=bracket_cap,
            timestamps=timestamps,
            current_time_ms=int(time.time() * 1000),
        )

        # Sizing (Defect 5 & 6: enforced bracket and maxNotional)
        q_dec = self.risk_gate.compute_entry_sizing(
            account_snapshot=fresh_snapshot,
            mark_price=mark_price_dec,
            filters=filters,
            bracket_max_notional=bracket_cap,
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

        self.journal.transition_order_state(
            client_order_id=client_oid,
            to_state="DISPATCHING",
            reason="Dispatching market order to exchange",
        )

        # Final pre-dispatch validation immediately before socket write
        self.risk_gate.validate_pre_dispatch(
            account_snapshot=fresh_snapshot,
            mark_price=mark_price_dec,
            filters=filters,
            target_pos_amt=target_amt,
            all_positions=all_positions,
            open_orders=open_orders,
            is_entry=True,
            actual_leverage=actual_lev,
            bracket_max_notional=bracket_cap,
            timestamps=timestamps,
            current_time_ms=int(time.time() * 1000),
            current_client_order_id=client_oid,
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
            exchange_oid = str(res.get("id") or res.get("orderId", ""))
            status = str(res.get("status", "")).upper()
            filled_qty = float(res.get("filled", 0.0))
            cum_quote = float(res.get("cost", 0.0))
            avg_price = float(res.get("average", 0.0))
            if avg_price <= 0.0 and filled_qty > 0.0 and cum_quote > 0.0:
                avg_price = cum_quote / filled_qty

            eps = 1e-8
            is_full = (filled_qty >= qty_float - eps) and filled_qty > 0
            is_partial = (0 < filled_qty < qty_float - eps)
            is_zero = (filled_qty <= eps)

            if status in ("CLOSED", "FILLED"):
                if is_full:
                    final_state = "FILLED"
                    rec_state = "VERIFIED"
                elif is_partial:
                    final_state = "PARTIALLY_FILLED_CANCELED"
                    rec_state = "DISCREPANCY"
                else:
                    final_state = "UNKNOWN"
                    rec_state = "DISCREPANCY"
            elif status in ("PARTIALLY_FILLED", "PARTIAL"):
                final_state = "PARTIALLY_FILLED"
                rec_state = "PENDING"
            elif status in ("CANCELED", "CANCELLED"):
                if is_zero:
                    final_state = "CANCELED"
                    rec_state = "VERIFIED"
                elif is_partial:
                    final_state = "PARTIALLY_FILLED_CANCELED"
                    rec_state = "DISCREPANCY"
                else:
                    final_state = "UNKNOWN"
                    rec_state = "DISCREPANCY"
            elif status in ("EXPIRED", "EXPIRED_IN_MATCH"):
                if is_zero:
                    final_state = "EXPIRED"
                    rec_state = "VERIFIED"
                elif is_partial:
                    final_state = "PARTIALLY_FILLED_EXPIRED"
                    rec_state = "DISCREPANCY"
                else:
                    final_state = "UNKNOWN"
                    rec_state = "DISCREPANCY"
            elif status in ("NEW", "OPEN"):
                if is_zero:
                    final_state = "OPEN"
                    rec_state = "PENDING"
                else:
                    final_state = "PARTIALLY_FILLED"
                    rec_state = "PENDING"
            else:
                final_state = "UNKNOWN"
                rec_state = "DISCREPANCY"

            self.journal.transition_order_state(
                client_order_id=client_oid,
                to_state=final_state,
                reason=f"Order executed with exchange status {status} (filled {filled_qty}/{qty_float})",
                exchange_order_id=exchange_oid,
                filled_quantity=filled_qty,
                cum_quote=cum_quote,
                avg_price=avg_price,
                reconciliation_state=rec_state,
            )

            if final_state in ("PARTIALLY_FILLED", "PARTIALLY_FILLED_CANCELED", "PARTIALLY_FILLED_EXPIRED"):
                self.journal.trigger_safety_freeze(
                    "PARTIAL_FILL_UNRESOLVED",
                    f"Entry order {client_oid} ended with partial fill ({filled_qty}/{qty_float}).",
                    client_order_id=client_oid,
                )
            elif final_state == "UNKNOWN":
                self.journal.trigger_safety_freeze(
                    "UNKNOWN_ORDER_OUTCOME",
                    f"Entry order {client_oid} in contradictory outcome: status={status}, filled={filled_qty}.",
                    client_order_id=client_oid,
                )

            return self.journal.get_order(client_oid)  # type: ignore
        except Exception as e:
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
                client_order_id=client_oid,
            )
            raise ExchangeError(f"Order dispatch failed for {client_oid}: {e}") from e

    def _execute_exit_order(
        self,
        action: Action,
        mark_price: Optional[float],
        mark_price_time_ms: Optional[int] = None,
    ) -> JournalOrder:
        # Defect 9: Hard production lock
        if not self.config.testnet or self.config.production_confirmed:
            raise ExchangeError(
                "PRODUCTION EXECUTION HARD LOCKED: Exit order dispatch blocked by hard runtime lock."
            )

        # Check permissions
        can_exit, reason = self.reconciliation.can_execute_exit(independently_verified=True)
        if not can_exit:
            raise SafetyFreezeError(reason or "Exit blocked by safety freeze.")

        fresh_snapshot = self.sync_account()
        acc_ts = int(time.time() * 1000)

        filters = self._load_symbol_filters()

        if mark_price is not None:
            if mark_price_time_ms is None:
                raise StaleMarketDataError(
                    "Caller-supplied mark_price requires explicit acquisition timestamp (mark_price_time_ms)."
                )
            mark_ts = int(mark_price_time_ms)
        else:
            mark_price, mark_ts = self.client.fetch_mark_price_data(self.config.raw_symbol)
        mark_price_dec = Decimal(str(mark_price))

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

        abs_qty = abs(pos_amt_dec)

        # Defect 4: Residual dust safety check on current position
        if abs_qty < filters.min_qty or (abs_qty * mark_price_dec) < filters.min_notional:
            self.journal.trigger_safety_freeze(
                "POSITION_ACCOUNT_DISCREPANCY",
                f"Position quantity {abs_qty} is untradeable residual dust (below minQty {filters.min_qty} or minNotional {filters.min_notional}).",
            )
            raise ResidualPositionError(
                f"Position quantity {abs_qty} is untradeable residual dust. Exit dispatch halted."
            )

        q_order = quantize_downward(abs_qty, filters.step_size)
        if q_order <= Decimal("0"):
            self.journal.trigger_safety_freeze(
                "POSITION_ACCOUNT_DISCREPANCY",
                f"Position quantity {abs_qty} is smaller than stepSize {filters.step_size} (dust).",
            )
            raise ResidualPositionError(
                f"Position quantity {abs_qty} cannot be closed: below stepSize {filters.step_size}. Exit dispatch halted."
            )

        # Defect 4: Fail closed if residual dust would remain after exit
        residual_after = abs_qty - q_order
        if check_residual_dust(residual_after, mark_price_dec, filters):
            self.journal.trigger_safety_freeze(
                "POSITION_ACCOUNT_DISCREPANCY",
                f"Exit order would leave unclosable residual dust: {residual_after}",
            )
            raise ResidualPositionError(
                f"Exit order of {q_order} would leave unclosable residual dust: {residual_after}. Exit dispatch halted."
            )

        side = "SELL" if action == Action.CLOSE_LONG else "BUY"
        action_name = "CLOSE_LONG" if action == Action.CLOSE_LONG else "CLOSE_SHORT"
        client_oid = self._generate_client_order_id()
        qty_float = float(q_order)

        self.journal.create_order_intent(
            client_order_id=client_oid,
            symbol=self.config.raw_symbol,
            action_type=action_name,
            side=side,
            quantity=qty_float,
            reduce_only=True,
            order_type="MARKET",
        )

        self.journal.transition_order_state(
            client_order_id=client_oid,
            to_state="DISPATCHING",
            reason="Dispatching reduce-only market close order to exchange",
        )

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

            entry_p = float(target_pos.get("entryPrice", 0.0))
            if entry_p > 0.0 and avg_price > 0.0 and filled_qty > 0.0:
                if action == Action.CLOSE_LONG:
                    trade_rpnl = (avg_price - entry_p) * filled_qty
                else:
                    trade_rpnl = (entry_p - avg_price) * filled_qty
                self._session_realized_pnl += trade_rpnl

            eps = 1e-8
            is_full = (filled_qty >= qty_float - eps) and filled_qty > 0
            is_partial = (0 < filled_qty < qty_float - eps)
            is_zero = (filled_qty <= eps)

            if status in ("CLOSED", "FILLED"):
                if is_full:
                    final_state = "FILLED"
                    rec_state = "VERIFIED"
                elif is_partial:
                    final_state = "PARTIALLY_FILLED_CANCELED"
                    rec_state = "DISCREPANCY"
                else:
                    final_state = "UNKNOWN"
                    rec_state = "DISCREPANCY"
            elif status in ("PARTIALLY_FILLED", "PARTIAL"):
                final_state = "PARTIALLY_FILLED"
                rec_state = "PENDING"
            elif status in ("CANCELED", "CANCELLED"):
                if is_zero:
                    final_state = "CANCELED"
                    rec_state = "VERIFIED"
                elif is_partial:
                    final_state = "PARTIALLY_FILLED_CANCELED"
                    rec_state = "DISCREPANCY"
                else:
                    final_state = "UNKNOWN"
                    rec_state = "DISCREPANCY"
            elif status in ("EXPIRED", "EXPIRED_IN_MATCH"):
                if is_zero:
                    final_state = "EXPIRED"
                    rec_state = "VERIFIED"
                elif is_partial:
                    final_state = "PARTIALLY_FILLED_EXPIRED"
                    rec_state = "DISCREPANCY"
                else:
                    final_state = "UNKNOWN"
                    rec_state = "DISCREPANCY"
            elif status in ("NEW", "OPEN"):
                if is_zero:
                    final_state = "OPEN"
                    rec_state = "PENDING"
                else:
                    final_state = "PARTIALLY_FILLED"
                    rec_state = "PENDING"
            else:
                final_state = "UNKNOWN"
                rec_state = "DISCREPANCY"

            self.journal.transition_order_state(
                client_order_id=client_oid,
                to_state=final_state,
                reason=f"Exit order executed with exchange status {status} (filled {filled_qty}/{qty_float})",
                exchange_order_id=exchange_oid,
                filled_quantity=filled_qty,
                cum_quote=cum_quote,
                avg_price=avg_price,
                reconciliation_state=rec_state,
            )

            if final_state in ("PARTIALLY_FILLED", "PARTIALLY_FILLED_CANCELED", "PARTIALLY_FILLED_EXPIRED"):
                self.journal.trigger_safety_freeze(
                    "PARTIAL_FILL_UNRESOLVED",
                    f"Exit order {client_oid} ended with partial fill ({filled_qty}/{qty_float}).",
                    client_order_id=client_oid,
                )
            elif final_state == "UNKNOWN":
                self.journal.trigger_safety_freeze(
                    "UNKNOWN_ORDER_OUTCOME",
                    f"Exit order {client_oid} in contradictory outcome: status={status}, filled={filled_qty}.",
                    client_order_id=client_oid,
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
                client_order_id=client_oid,
            )
            raise ExchangeError(f"Exit order dispatch failed for {client_oid}: {e}") from e
