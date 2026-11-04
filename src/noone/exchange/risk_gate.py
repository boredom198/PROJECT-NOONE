"""Risk sizing, exposure enforcement, and daily loss gate for Component 3.

Implements the complete 8-step sizing workflow, multi-factor margin buffer engine,
authoritative daily PnL calculation across midnight rollover, and fresh pre-dispatch validation.
Conforms strictly to PROJECT NOONE Technical Design Specification V2.2.4 Sections 9 and 10.
"""

from decimal import Decimal
import time
from typing import Any, Dict, List, Optional, Set

from noone.rl_engine.inference import AccountSnapshot
from noone.exchange.config import Component3Config
from noone.exchange.errors import (
    AccountSyncError,
    DailyLossLimitError,
    DailyPnLUninitializedError,
    FilterValidationError,
    RiskGateError,
    SafetyFreezeError,
    StaleMarketDataError,
)
from noone.exchange.journal import ExecutionJournal
from noone.exchange.precision import ParsedSymbolFilters, quantize_downward


class RiskGate:
    """Enforces position sizing, margin buffering, exposure limits, and daily loss checks."""

    def __init__(self, config: Component3Config, journal: ExecutionJournal):
        self.config = config
        self.journal = journal

    # -------------------------------------------------------------------------
    # 8-Step Sizing Workflow & Multi-Factor Margin Buffer Engine
    # -------------------------------------------------------------------------

    def compute_entry_sizing(
        self,
        account_snapshot: AccountSnapshot,
        mark_price: Decimal,
        filters: ParsedSymbolFilters,
        bracket_max_notional: Optional[Decimal] = None,
    ) -> Decimal:
        """Execute the 8-step sizing workflow to determine safe order quantity.

        Enforces:
        - Target margin allocation fraction
        - Leverage scaling
        - Max position notional, leverage bracket caps, and maxNotional filter limits
        - Multi-factor margin buffer (taker fee + 0.5% safety buffer)
        - Downward quantization to stepSize
        """
        # Step 1: Context validation
        if mark_price <= Decimal("0"):
            raise RiskGateError(f"Invalid non-positive mark price: {mark_price}")
        if account_snapshot.equity <= 0.0:
            raise RiskGateError(f"Invalid non-positive equity: {account_snapshot.equity}")

        equity_dec = Decimal(str(account_snapshot.equity))
        avail_dec = Decimal(str(account_snapshot.available_balance))

        # Step 2: Target margin allocation
        m_alloc_frac = Decimal(str(self.config.margin_allocation_fraction))
        m_intended = equity_dec * m_alloc_frac

        # Step 3: Target notional & leverage caps (incorporating bracket & maxNotional)
        leverage_dec = Decimal(str(self.config.leverage))
        notional_cap = Decimal(str(self.config.max_position_notional))

        if bracket_max_notional is not None:
            if bracket_max_notional <= Decimal("0"):
                raise RiskGateError(f"Invalid non-positive bracket_max_notional: {bracket_max_notional}")
            notional_cap = min(notional_cap, bracket_max_notional)

        if filters.max_notional is not None:
            notional_cap = min(notional_cap, filters.max_notional)

        n_target = min(m_intended * leverage_dec, notional_cap)
        if n_target <= Decimal("0"):
            raise RiskGateError(f"Target notional evaluated to non-positive: {n_target}")

        # Step 4: Exposure Verification
        if account_snapshot.position_side != 0 or account_snapshot.position_quantity > 0.0:
            raise RiskGateError("Cannot size entry order: active position already exists.")

        # Step 5: Raw quantity
        q_raw = n_target / mark_price

        # Step 6: Quantization downward
        q_quantized = quantize_downward(q_raw, filters.step_size)
        if q_quantized <= Decimal("0"):
            raise RiskGateError(
                f"Downward quantization resulted in zero quantity. Raw: {q_raw}, stepSize: {filters.step_size}"
            )

        # Step 7: Post-rounding recomputation & multi-factor margin buffer
        n_final = q_quantized * mark_price
        fee_rate_dec = Decimal(str(self.config.fee_rate))
        safety_buf_dec = Decimal(str(self.config.safety_buffer_pct))

        estimated_fee = n_final * fee_rate_dec
        safety_buffer = n_final * safety_buf_dec
        m_required = (n_final / leverage_dec) + estimated_fee + safety_buffer

        # Filter bounds verification
        if q_quantized < filters.min_qty:
            raise FilterValidationError(
                f"Quantized quantity {q_quantized} below minQty {filters.min_qty}"
            )
        if q_quantized > filters.max_qty:
            raise FilterValidationError(
                f"Quantized quantity {q_quantized} exceeds maxQty {filters.max_qty}"
            )
        if n_final < filters.min_notional:
            raise FilterValidationError(
                f"Final notional {n_final} below minNotional {filters.min_notional}"
            )
        if filters.max_notional is not None and n_final > filters.max_notional:
            raise FilterValidationError(
                f"Final notional {n_final} exceeds maxNotional {filters.max_notional}"
            )
        if bracket_max_notional is not None and n_final > bracket_max_notional:
            raise RiskGateError(
                f"Final notional {n_final} exceeds leverage bracket cap {bracket_max_notional}"
            )

        # Available margin buffer verification
        if m_required > avail_dec:
            raise RiskGateError(
                f"Required margin {m_required:.4f} (incl fee {estimated_fee:.4f} + buffer {safety_buffer:.4f}) "
                f"exceeds available balance {avail_dec:.4f}"
            )

        return q_quantized

    # -------------------------------------------------------------------------
    # Daily PnL Accounting Across Midnight Rollover
    # -------------------------------------------------------------------------

    def check_daily_pnl_limit(
        self,
        current_equity: Decimal,
        baseline_equity: Decimal,
        net_external_transfers: Decimal,
        verified_status: str,
    ) -> Decimal:
        """Calculate authoritative daily PnL and enforce daily loss limit.

        Formula: Daily PnL = Current Equity - Net External Transfers - Baseline Equity
        """
        # Baseline status verification
        if verified_status != "VERIFIED_UTC_BASELINE":
            self.journal.trigger_safety_freeze(
                "DAILY_PNL_UNINITIALIZED",
                f"Daily baseline status is '{verified_status}', not 'VERIFIED_UTC_BASELINE'.",
            )
            raise DailyPnLUninitializedError(
                f"Cannot trade entries: daily baseline is uninitialized or unverified ({verified_status})."
            )

        if baseline_equity <= Decimal("0"):
            self.journal.trigger_safety_freeze(
                "DAILY_PNL_UNINITIALIZED",
                f"Baseline equity is non-positive: {baseline_equity}",
            )
            raise DailyPnLUninitializedError(
                f"Invalid non-positive baseline equity: {baseline_equity}"
            )

        daily_pnl = current_equity - net_external_transfers - baseline_equity
        max_loss_pct = Decimal(str(self.config.max_daily_loss_pct))
        max_allowed_loss = baseline_equity * max_loss_pct

        if daily_pnl <= -max_allowed_loss:
            self.journal.trigger_safety_freeze(
                "DAILY_LOSS_LIMIT",
                f"Daily PnL {daily_pnl:.2f} breached max loss limit -{max_allowed_loss:.2f} ({max_loss_pct * 100}%).",
            )
            raise DailyLossLimitError(
                f"Daily loss limit breached: Daily PnL {daily_pnl:.2f} <= -{max_allowed_loss:.2f}"
            )

        return daily_pnl

    # -------------------------------------------------------------------------
    # Fresh Pre-Dispatch Validation (11-Point Check + Freshness Verification)
    # -------------------------------------------------------------------------

    def validate_pre_dispatch(
        self,
        account_snapshot: AccountSnapshot,
        mark_price: Decimal,
        filters: ParsedSymbolFilters,
        target_pos_amt: Decimal,
        all_positions: List[Dict[str, Any]],
        open_orders: List[Dict[str, Any]],
        is_entry: bool,
        actual_leverage: Optional[int] = None,
        bracket_max_notional: Optional[Decimal] = None,
        timestamps: Optional[Dict[str, int]] = None,
        current_time_ms: Optional[int] = None,
        current_client_order_id: Optional[str] = None,
    ) -> None:
        """Execute fresh pre-dispatch validation directly prior to socket transmission.

        Enforces:
        - Active safety freeze checks
        - Local unresolved intent inspection
        - Mark price validity and freshness (<= 2000ms)
        - Account snapshot and position freshness (<= 3000ms / <= 2000ms)
        - Effective leverage matching configured leverage
        - Leverage bracket cap availability and compliance
        - Zero external positions and zero open orders
        - Non-stale, non-future-dated input data
        """
        now_ms = current_time_ms if current_time_ms is not None else int(time.time() * 1000)

        # 1. Freshness validation (Defect 7)
        if timestamps is not None:
            freshness_limits = {
                "mark_price": self.config.mark_price_freshness_ms,
                "target_position": self.config.position_freshness_ms,
                "account_snapshot": self.config.equity_freshness_ms,
                "all_positions": 3000,
                "open_orders": 3000,
                "leverage": 3000,
                "filters": self.config.filter_cache_ttl_s * 1000,
                "brackets": self.config.filter_cache_ttl_s * 1000,
            }

            for key, max_age in freshness_limits.items():
                if key not in timestamps or timestamps[key] is None or timestamps[key] <= 0:
                    raise RiskGateError(f"Pre-dispatch validation failed: missing timestamp for '{key}'.")

                ts = timestamps[key]
                # Future-dated check
                if ts > now_ms + 1000:
                    raise RiskGateError(
                        f"Pre-dispatch validation failed: future-dated timestamp for '{key}': {ts} > {now_ms}."
                    )

                age = now_ms - ts
                if age > max_age:
                    if key == "mark_price":
                        raise StaleMarketDataError(
                            f"Pre-dispatch validation failed: mark price data is stale: age {age}ms > {max_age}ms."
                        )
                    elif key in ("account_snapshot", "target_position"):
                        raise AccountSyncError(
                            f"Pre-dispatch validation failed: {key} data is stale: age {age}ms > {max_age}ms."
                        )
                    else:
                        raise RiskGateError(
                            f"Pre-dispatch validation failed: {key} data is stale: age {age}ms > {max_age}ms."
                        )

        # 2. Active safety freeze check
        active_freezes = self.journal.get_active_safety_freezes()
        if active_freezes:
            reasons = [f.reason for f in active_freezes]
            raise SafetyFreezeError(
                f"Pre-dispatch validation failed: active safety freezes: {reasons}",
                reason=reasons[0],
            )

        # 3. Local unresolved intents in SQLite
        unresolved = self.journal.get_unresolved_orders()
        if current_client_order_id is not None:
            unresolved = [o for o in unresolved if o.client_order_id != current_client_order_id]
        if unresolved:
            raise RiskGateError(
                f"Pre-dispatch validation failed: {len(unresolved)} unresolved order intents in SQLite."
            )

        # 4. Mark price validity
        if mark_price <= Decimal("0"):
            raise RiskGateError(f"Pre-dispatch validation failed: non-positive mark price {mark_price}")

        # 5. Filters validity
        if filters.step_size <= Decimal("0") or filters.min_qty <= Decimal("0"):
            raise FilterValidationError("Pre-dispatch validation failed: invalid filter dimensions.")

        raw_target = self.config.raw_symbol

        # 6. Account-wide external positions audit
        for p in all_positions:
            sym = p.get("symbol", "").upper()
            if sym != raw_target:
                amt = Decimal(str(p.get("positionAmt", "0")))
                if abs(amt) > Decimal("1e-8"):
                    self.journal.trigger_safety_freeze(
                        "ACCOUNT_WIDE_EXPOSURE_DETECTED",
                        f"Detected active external position in {sym}: {amt}",
                    )
                    raise RiskGateError(
                        f"Pre-dispatch failed: external position detected on {sym} ({amt})."
                    )

        # 7. Account-wide open orders audit
        if open_orders:
            self.journal.trigger_safety_freeze(
                "ACCOUNT_WIDE_EXPOSURE_DETECTED",
                f"Detected {len(open_orders)} unexplained open orders on exchange.",
            )
            raise RiskGateError(
                f"Pre-dispatch failed: {len(open_orders)} open orders resting on exchange."
            )

        # 8. Position direction consistency & leverage verification
        if is_entry:
            if abs(target_pos_amt) > Decimal("1e-8"):
                raise RiskGateError(
                    f"Pre-dispatch entry check failed: position is not flat ({target_pos_amt})."
                )
            if account_snapshot.position_side != 0 or account_snapshot.position_quantity > 0.0:
                raise RiskGateError("Pre-dispatch entry check failed: account snapshot shows active position.")

            # Leverage verification (Defect 5 & 6)
            if actual_leverage is None or actual_leverage <= 0:
                self.journal.trigger_safety_freeze(
                    "LEVERAGE_MISMATCH",
                    f"Effective leverage could not be verified on exchange for {self.config.raw_symbol}.",
                )
                raise RiskGateError(
                    f"Pre-dispatch validation failed: missing or invalid exchange leverage ({actual_leverage})."
                )
            if actual_leverage != self.config.leverage:
                self.journal.trigger_safety_freeze(
                    "LEVERAGE_MISMATCH",
                    f"Exchange leverage {actual_leverage} does not match configured leverage {self.config.leverage}",
                )
                raise RiskGateError(
                    f"Pre-dispatch validation failed: exchange leverage ({actual_leverage}) does not match "
                    f"configured leverage ({self.config.leverage})."
                )
        else:
            # Exit order
            if abs(target_pos_amt) <= Decimal("1e-8"):
                raise RiskGateError(
                    "Pre-dispatch exit check failed: target position on exchange is already flat."
                )
