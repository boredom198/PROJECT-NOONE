"""Perpetual-futures trading environment (Component 8).

Simulation assumptions (deliberately simplified, documented, and
NOT exchange-exact):

- Decisions are made on a closed candle; actions execute at the
  *next* candle's open (no lookahead), adjusted adversely by
  ``slippage_rate``. Positions are marked at each candle's close.
- PnL: long ``qty*(exit-entry)``, short ``qty*(entry-exit)``.
  ``equity = balance + unrealized_pnl``; ``balance`` is the wallet
  (initial + realized PnL - fees - funding). Margin is reserved
  from balance: ``available = balance - margin_used``.
- Sizing: ``margin = available * capital_allocation_fraction``,
  ``notional = margin * leverage`` (capped by
  ``max_position_notional``), ``qty = notional / execution_price``.
- Fees: ``fee_rate * notional`` on every open and close, deducted
  from balance. Slippage is the adverse price adjustment above;
  a configured slippage above ``max_slippage_rate`` is rejected by
  config validation and re-checked at execution.
- Funding (constant mode, default): at the configured recurring
  UTC times (``funding_times_utc``, default 00:00/08:00/16:00),
  longs pay ``qty * close * funding_rate`` when the rate is
  positive (shorts receive); signs reverse with the rate/direction.
  Events are derived from candle timestamps (window
  ``(cursor, close_time]``, each event once, regardless of dataset
  start or missing candles). This uses the candle close as a
  mark-price approximation and a configured rate — it is not
  Binance's exact funding schedule or index calculation.
  Historical mode (``funding_mode="historical"``) instead requires
  the full Component 4 ``FundingHistory``, strictly validates it at
  construction (Stage 1), and applies its actual settlements with
  their own rates and settlement marks (Stage 2): settlements at
  or before a candle open apply to the pre-execution position,
  settlements in ``(open, close]`` apply after action and risk
  checks to the remaining position. Each applied settlement is
  recorded in ``funding_ledger`` and reconciled against
  ``total_funding``. Risk checks precede post-open funding because
  candles cannot establish exact intrabar ordering; Binance
  settlement timing may deviate from nominal timestamps, so this
  is a simulation convention, not exact exchange timing.
- Liquidation (simplified isolated-margin approximation, not
  Binance's exact tiers): long liquidation price
  ``entry*(1 - 1/leverage + maintenance_margin_rate)``, short
  mirrored. Intra-candle checks use the adverse extreme (low for
  longs, high for shorts); liquidation is checked first, then
  stop-loss, then take-profit (pessimistic ordering when several
  levels fall inside one candle).
- Stop-loss/take-profit: percentage distance from entry, enforced
  by the environment regardless of the agent's action.
- Daily limits: per UTC day, measured against the equity at the
  day's first processed candle. Breach force-closes any position at
  that candle's close (fees+slippage apply) and ends the episode.
- Reward (separate from accounting, see ``_reward``):
  ``(equity_after - equity_before) / initial_balance * reward_scale``
  per step — the fractional equity change, fees/funding included via
  equity. No bonuses for trading frequency.

Invalid actions (e.g. opening while a position is open, closing a
position that does not exist) are no-ops equivalent to Hold and are
flagged as ``info["invalid_action"]``; exactly one position may be
open at a time and positions cannot be flipped in a single step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import IntEnum
from typing import Any

from noone.market_data.models import (
    FundingHistory,
    FundingSettlement,
    Kline,
    KlineGap,
    KlineHistory,
)
from noone.trading_env.config import TradingConfig


class Action(IntEnum):
    HOLD = 0
    OPEN_LONG = 1
    CLOSE_LONG = 2
    OPEN_SHORT = 3
    CLOSE_SHORT = 4


@dataclass
class Position:
    side: int  # +1 long, -1 short
    quantity: float
    entry_price: float
    margin: float
    opened_index: int
    opened_at: datetime | None = None


@dataclass(frozen=True)
class FundingLedgerEntry:
    """One applied historical funding settlement (Stage 2 audit record)."""

    symbol: str
    funding_time: datetime
    funding_rate: Decimal
    settlement_mark_price: Decimal
    rate_type: str | None
    position_side: int  # +1 long, -1 short
    position_quantity: Decimal
    settlement_notional: Decimal
    funding_payment: Decimal  # signed cash received (+ credit)
    balance_adjustment: Decimal


@dataclass(frozen=True)
class StepResult:
    observation: dict[str, float]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


def _ohlc_error(candle: Kline) -> str | None:
    """B2 validation: reason string if the candle is malformed."""
    import math

    values = {"open": candle.open, "high": candle.high,
              "low": candle.low, "close": candle.close}
    for name, value in values.items():
        if value is None:
            return f"missing {name}"
        if not math.isfinite(value):
            return f"non-finite {name}"
        if value <= 0:
            return f"non-positive {name}"
    assert candle.open is not None and candle.high is not None
    assert candle.low is not None and candle.close is not None
    if candle.high < max(candle.open, candle.close, candle.low):
        return "high below open/close/low"
    if candle.low > min(candle.open, candle.close, candle.high):
        return "low above open/close/high"
    return None


def _level_fill(side: int, level: float, candle: Kline, *, take_profit: bool = False) -> float:
    """B1 gap-through fill for a price level.

    Stop/liquidation: if the candle opened beyond the level, fill at
    the worse open; otherwise fill at the level. Take-profit mirrors
    favourably (gap beyond TP fills at the better open).
    """
    open_price = float(candle.open or level)
    if take_profit:
        return max(open_price, level) if side == 1 else min(open_price, level)
    return min(open_price, level) if side == 1 else max(open_price, level)


class PerpetualFuturesEnv:
    """Historical-candle perpetual-futures simulation."""

    def __init__(
        self, candles: list[Kline], config: TradingConfig | None = None,
        *, gaps: tuple[KlineGap, ...] = (),
        news_archive: Any = None, news_asset: str | None = None,
        funding_history: FundingHistory | None = None,
    ):
        if len(candles) < 2:
            raise ValueError("At least 2 candles are required")
        if any(c.open_time is None for c in candles):
            raise ValueError("All candles need open_time for ordering")
        # B2 policy: malformed candles are SKIPPED and recorded in
        # `rejected_candles` (never zero-filled, never fabricated).
        # Missing data therefore cannot trigger false risk events.
        self.rejected_candles: list[tuple[int, str]] = []
        valid: list[Kline] = []
        for idx, candle in enumerate(sorted(candles, key=lambda c: c.open_time)):
            reason = _ohlc_error(candle)
            if reason is None:
                valid.append(candle)
            else:
                self.rejected_candles.append((idx, reason))
        self._all_candles = valid
        # Closed candles only; the forming candle is never exposed.
        self.candles = [c for c in self._all_candles if c.is_closed]
        self.dropped_unclosed = len(self._all_candles) - len(self.candles)
        if len(self.candles) < 2:
            raise ValueError("At least 2 closed candles are required")
        self.gaps = gaps
        self.config = config or TradingConfig()
        self.news_archive = news_archive
        self.news_asset = news_asset.upper() if news_asset else None
        self.funding_history = funding_history
        self._validate_funding_history()
        self._historical_settlements: tuple[FundingSettlement, ...] = ()
        if self.config.funding_mode == "historical" and funding_history is not None:
            self._historical_settlements = tuple(sorted(
                funding_history.settlements,
                key=lambda s: (s.funding_time, s.rate_type or "")))
        self._reset_state()

    @classmethod
    def from_history(
        cls, history: KlineHistory, config: TradingConfig | None = None,
        *, funding_history: FundingHistory | None = None,
        **kwargs: Any,
    ) -> "PerpetualFuturesEnv":
        """Build from Component 4's validated ``KlineHistory``.

        Gap information is carried onto the env (``env.gaps``) and
        surfaced in every step's info; missing candles are never
        fabricated or filled. In historical funding mode the full
        ``FundingHistory`` is required and strictly validated.
        """
        if funding_history is not None and "funding_history" in kwargs:
            raise TypeError("funding_history supplied twice")
        return cls(list(history.candles), config, gaps=history.gaps,
                   funding_history=funding_history, **kwargs)

    # -- historical funding gate (Stage 1) --------------------------
    @property
    def _funding_period(self) -> tuple[datetime, datetime]:
        first, last = self.candles[0], self.candles[-1]
        start = first.open_time
        end = last.close_time or last.open_time
        assert start is not None and end is not None
        return start, end

    def _validate_funding_history(self) -> None:
        """Strict historical-funding gate; constant mode unchanged.

        Historical mode is explicit and strict: it requires the full
        Component 4 ``FundingHistory`` (never a bare settlement list),
        collected with ``context_limit=3``, whose requested range and
        coverage span the candle period, with complete pagination,
        sufficient before/after context, no before/after scoped blind
        spots, valid settlement marks, and
        ``completeness_status_for_period(start, end) == "complete"``.
        """
        mode = self.config.funding_mode
        history = self.funding_history
        if mode == "constant":
            if history is not None:
                raise ValueError(
                    "funding_history supplied but funding_mode is "
                    "'constant': set funding_mode='historical' "
                    "explicitly to use historical funding")
            return
        if history is None:
            raise ValueError(
                "funding_mode='historical' requires a full "
                "FundingHistory object via funding_history=...; "
                "a bare list of funding events is not accepted")
        if not isinstance(history, FundingHistory):
            raise TypeError(
                "funding_history must be a FundingHistory object, "
                f"not {type(history).__name__}: a bare list of "
                "funding events is not accepted")
        for candle in self.candles:
            if candle.close_time is None or (
                    candle.open_time is not None
                    and candle.close_time < candle.open_time):
                raise ValueError(
                    "Historical funding requires a valid candle "
                    f"close_time for every candle; candle at "
                    f"{candle.open_time} has close_time="
                    f"{candle.close_time}")
        start, end = self._funding_period
        candle_symbols = {c.symbol for c in self.candles}
        if len(candle_symbols) != 1 or history.symbol not in candle_symbols:
            raise ValueError(
                f"Funding history symbol {history.symbol!r} does not "
                f"match candle symbol {sorted(candle_symbols)!r}")
        if not history.pagination_complete:
            raise ValueError(
                "Funding history pagination is incomplete "
                f"(pagination_termination={history.pagination_termination!r}); "
                "strict historical evaluation requires complete pagination")
        if (history.requested_start_time is None
                or history.requested_end_time is None
                or history.requested_start_time > start
                or history.requested_end_time < end):
            raise ValueError(
                "Funding history requested range "
                f"({history.requested_start_time}.."
                f"{history.requested_end_time}) does not cover the "
                f"requested candle period ({start}..{end})")
        if not history.covers_period(start, end):
            raise ValueError(
                "Funding history coverage does not cover the candle "
                f"period ({start}..{end}); coverage is "
                f"{history.coverage_start_time}..{history.coverage_end_time}")
        for direction, context in (("before", history.before_context),
                                   ("after", history.after_context)):
            if context is None:
                raise ValueError(
                    "Historical funding requires before_context and "
                    "after_context context collected with "
                    f"context_limit=3; {direction}_context is missing")
            if context.direction != direction:
                raise ValueError(
                    f"{direction} context has mismatched direction "
                    f"{context.direction!r}")
            if context.requested_count != 3:
                raise ValueError(
                    f"{direction} context was not collected with "
                    f"context_limit=3 (requested_count="
                    f"{context.requested_count}); a history without "
                    "context does not prove boundary validation")
            if context.status != "sufficient":
                raise ValueError(
                    f"{direction} context status is {context.status!r}, "
                    "not 'sufficient'; strict historical evaluation "
                    "requires sufficient context on both sides")
            if context.received_count < 3:
                raise ValueError(
                    f"{direction} context received_count="
                    f"{context.received_count} < 3; strict historical "
                    "evaluation requires at least 3 context settlements")
        for spot in history.scoped_blind_spots:
            if spot.boundary in ("before", "after"):
                raise ValueError(
                    f"Historical funding has a {spot.boundary} scoped "
                    f"blind spot: {spot.reason}")
        seen_identities: set = set()
        for settlement in history.settlements:
            if settlement.identity in seen_identities:
                raise ValueError(
                    "Duplicate funding settlement identity "
                    f"{settlement.identity!r}: settlement records are "
                    "identified by (symbol, funding_time, rate_type) "
                    "and each identity may appear only once in a "
                    "FundingHistory")
            seen_identities.add(settlement.identity)
            if settlement.symbol != history.symbol:
                raise ValueError(
                    f"Funding settlement symbol {settlement.symbol!r} "
                    f"does not match history symbol {history.symbol!r}")
            if settlement.funding_time is None:
                raise ValueError(
                    "Funding settlement without a funding_time cannot "
                    "be used for strict historical evaluation")
            if settlement.funding_rate is None or (
                    hasattr(settlement.funding_rate, "is_finite")
                    and not settlement.funding_rate.is_finite()):
                raise ValueError(
                    "Funding settlement at "
                    f"{settlement.funding_time} has no valid funding "
                    "rate; strict historical evaluation requires a "
                    "valid rate")
            mark = settlement.mark_price
            if mark is None or (
                    hasattr(mark, "is_finite") and not mark.is_finite()
                    ) or mark <= 0:
                raise ValueError(
                    "Invalid settlement mark price for funding "
                    f"settlement at {settlement.funding_time}: "
                    f"mark_price={mark!r}; strict historical "
                    "evaluation requires the positive historical "
                    "Binance settlement mark")
        status = history.completeness_status_for_period(start, end)
        if status != "complete":
            raise ValueError(
                "Historical funding is not complete for the candle "
                f"period ({start}..{end}): "
                f"completeness_status_for_period={status!r}; strict "
                "historical evaluation requires 'complete'")

    def funding_metadata(self) -> dict[str, Any]:
        """Auditable funding metadata for run metadata (Stage 1).

        Constant mode reports its simulated configuration. Historical
        mode embeds the full Component 4 ``FundingHistory.metadata()``
        plus the period completeness assessed at construction.
        """
        history = self.funding_history
        if self.config.funding_mode == "constant" or history is None:
            return {
                "funding_mode": self.config.funding_mode,
                "historical_accounting_implemented": False,
                "funding_strict": False,
                "funding_rate": self.config.funding_rate,
                "funding_times_utc": list(self.config.funding_times_utc),
                "funding_history": None,
                "completeness_status_for_period": None,
                "acknowledged_anomaly_count": 0,
                "unacknowledged_anomaly_count": 0,
            }
        start, end = self._funding_period
        return {
            "funding_mode": self.config.funding_mode,
            "historical_accounting_implemented": True,
            "funding_strict": True,
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
            "completeness_status_for_period":
                history.completeness_status_for_period(start, end),
            "acknowledged_anomaly_count": len(history.acknowledged_anomalies),
            "unacknowledged_anomaly_count":
                len(history.unacknowledged_anomalies),
            "funding_history": history.metadata(),
        }

    # -- state -----------------------------------------------------
    def _reset_state(self) -> None:
        cfg = self.config
        self._index = 0
        self.balance = cfg.initial_balance
        self.position: Position | None = None
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.total_funding = 0.0  # signed cash received (+ credit)
        self._terminated = False
        self._truncated = False
        self._termination_reason: str | None = None
        self._day_key: Any = None
        self._day_start_equity = cfg.initial_balance
        self._last_equity = cfg.initial_balance
        first = self.candles[0]
        self._funding_cursor = first.close_time or first.open_time
        self.funding_ledger: list[FundingLedgerEntry] = []
        self._historical_funding_cursor = 0
        self._funding_balance_adjustments = 0.0

    def reset(self) -> dict[str, float]:
        """Reset the episode; returns the first observation."""
        self._reset_state()
        return self._observation()

    # -- accounting helpers ----------------------------------------
    @property
    def current_candle(self) -> Kline:
        return self.candles[self._index]

    @property
    def current_price(self) -> float:
        return float(self.current_candle.close or 0.0)

    @property
    def margin_used(self) -> float:
        return self.position.margin if self.position else 0.0

    @property
    def available_balance(self) -> float:
        return self.balance - self.margin_used

    def unrealized_pnl(self, price: float | None = None) -> float:
        if self.position is None:
            return 0.0
        mark = self.current_price if price is None else price
        return self.position.side * self.position.quantity * (
            mark - self.position.entry_price)

    @property
    def equity(self) -> float:
        return self.balance + self.unrealized_pnl()

    def _execution_price(self, side: int, action: str, raw: float) -> float:
        slip = self.config.slippage_rate
        if slip > self.config.max_slippage_rate:
            raise ValueError("Configured slippage exceeds maximum permitted")
        # Buys pay up, sells receive less (adverse to the trader).
        is_buy = (action == "open" and side == 1) or (
            action == "close" and side == -1)
        return raw * (1 + slip) if is_buy else raw * (1 - slip)

    def _close_position(self, raw_price: float, reason: str) -> dict[str, Any]:
        pos = self.position
        assert pos is not None
        exit_price = self._execution_price(pos.side, "close", raw_price)
        pnl = pos.side * pos.quantity * (exit_price - pos.entry_price)
        fee = pos.quantity * exit_price * self.config.fee_rate
        self.balance += pnl - fee
        self.realized_pnl += pnl
        self.total_fees += fee
        self.position = None
        return {"event": reason, "exit_price": exit_price,
                "realized_pnl": pnl, "fee": fee}

    def _open_position(self, side: int, candle: Kline) -> dict[str, Any]:
        import math

        cfg = self.config
        raw = candle.open
        # B4: never open on a missing/zero/negative/non-finite price.
        # No position, no fee, no balance/margin change.
        if raw is None or not math.isfinite(raw) or raw <= 0:
            return {"event": "open_rejected",
                    "invalid_reason": "invalid_execution_price"}
        exec_price = self._execution_price(side, "open", float(raw))
        if not math.isfinite(exec_price) or exec_price <= 0:
            return {"event": "open_rejected",
                    "invalid_reason": "invalid_execution_price"}
        margin = self.available_balance * cfg.capital_allocation_fraction
        notional = margin * cfg.leverage
        if cfg.max_position_notional is not None:
            notional = min(notional, cfg.max_position_notional)
            margin = notional / cfg.leverage
        quantity = notional / exec_price if exec_price > 0 else 0.0
        fee = notional * cfg.fee_rate
        self.balance -= fee
        self.total_fees += fee
        self.position = Position(
            side=side, quantity=quantity, entry_price=exec_price,
            margin=margin, opened_index=self._index,
            opened_at=candle.open_time)
        return {"event": "open_long" if side == 1 else "open_short",
                "entry_price": exec_price, "quantity": quantity,
                "margin": margin, "notional": notional, "fee": fee}

    def _risk_checks(self, candle: Kline) -> dict[str, Any] | None:
        """Liquidation -> stop-loss -> take-profit (in that order)."""
        pos = self.position
        if pos is None:
            return None
        cfg = self.config
        entry = pos.entry_price
        # Candles reaching here passed B2 validation (OHLC present,
        # positive, consistent), so extremes are real values.
        if pos.side == 1:
            liq = entry * (1 - 1 / cfg.leverage + cfg.maintenance_margin_rate)
            if float(candle.low) <= liq:  # type: ignore[arg-type]
                return self._close_position(
                    _level_fill(1, liq, candle), "liquidation")
            if cfg.stop_loss_pct is not None and float(candle.low) <= entry * (1 - cfg.stop_loss_pct):  # type: ignore[arg-type]
                level = entry * (1 - cfg.stop_loss_pct)
                return self._close_position(
                    _level_fill(1, level, candle), "stop_loss")
            if cfg.take_profit_pct is not None and float(candle.high) >= entry * (1 + cfg.take_profit_pct):  # type: ignore[arg-type]
                level = entry * (1 + cfg.take_profit_pct)
                return self._close_position(
                    _level_fill(1, level, candle, take_profit=True),
                    "take_profit")
        else:
            liq = entry * (1 + 1 / cfg.leverage - cfg.maintenance_margin_rate)
            if float(candle.high) >= liq:  # type: ignore[arg-type]
                return self._close_position(
                    _level_fill(-1, liq, candle), "liquidation")
            if cfg.stop_loss_pct is not None and float(candle.high) >= entry * (1 + cfg.stop_loss_pct):  # type: ignore[arg-type]
                level = entry * (1 + cfg.stop_loss_pct)
                return self._close_position(
                    _level_fill(-1, level, candle), "stop_loss")
            if cfg.take_profit_pct is not None and float(candle.low) <= entry * (1 - cfg.take_profit_pct):  # type: ignore[arg-type]
                level = entry * (1 - cfg.take_profit_pct)
                return self._close_position(
                    _level_fill(-1, level, candle, take_profit=True),
                    "take_profit")
        return None

    def _funding_events_between(self, start, end) -> list:
        """Funding timestamps in (start, end] from the UTC schedule."""
        from datetime import datetime, timedelta, timezone

        events = []
        day = start.date()
        while day <= end.date():
            for value in self.config.funding_times_utc:
                hour, minute = (int(p) for p in value.split(":"))
                event = datetime(day.year, day.month, day.day, hour,
                                 minute, tzinfo=timezone.utc)
                if start < event <= end:
                    events.append(event)
            day += timedelta(days=1)
        return sorted(events)

    def _apply_funding(self, candle: Kline) -> float:
        """Timestamp-based funding (B3).

        Events are the configured UTC times falling in
        ``(cursor, candle.close_time]``; the cursor advances every
        step, so each event applies exactly once regardless of
        dataset start or missing candles. Only a position held
        through the window is charged (positions opened at this
        candle's open precede any event inside the candle; events
        before the open belong to earlier windows). Long pays when
        the rate is positive. Basis: candle close (mark proxy) at a
        configured simulation rate — not exchange-exact.
        """
        cfg = self.config
        end = candle.close_time or candle.open_time
        if end is None:
            return 0.0
        cursor = self._funding_cursor or end
        events = self._funding_events_between(cursor, end)
        self._funding_cursor = end
        if self.position is None or not events:
            return 0.0
        mark = float(candle.close or 0.0)
        payment = (-self.position.side * self.position.quantity
                   * mark * cfg.funding_rate * len(events))
        self.balance += payment
        self.total_funding += payment
        return payment

    # -- historical funding accounting (Stage 2) ---------------------
    def _consume_historical_settlements(
        self, upper_bound: datetime,
    ) -> list[FundingLedgerEntry]:
        """Consume settlements with ``funding_time <= upper_bound``.

        Consumption is by monotonic index into the sorted settlement
        list, never by timestamp alone, so distinct records sharing a
        timestamp (different ``rate_type``) are each processed once
        and a consumed record can never be applied again. Settlements
        consumed while flat produce no ledger entry and no payment.
        """
        entries: list[FundingLedgerEntry] = []
        settlements = self._historical_settlements
        while self._historical_funding_cursor < len(settlements):
            settlement = settlements[self._historical_funding_cursor]
            if settlement.funding_time is None:
                raise ValueError(
                    "Funding settlement without a funding_time cannot "
                    "be applied in historical mode")
            if settlement.funding_time > upper_bound:
                break
            # Advance only after successful validation/application,
            # so a failed settlement is never silently consumed.
            entry = self._apply_historical_settlement(settlement)
            self._historical_funding_cursor += 1
            if entry is not None:
                entries.append(entry)
        if entries:
            self._reconcile_historical_funding()
        return entries

    def _apply_historical_settlement(
        self, settlement: FundingSettlement,
    ) -> FundingLedgerEntry | None:
        if self.position is None:
            return None
        rate = settlement.funding_rate
        mark = settlement.mark_price
        if rate is None or (hasattr(rate, "is_finite")
                            and not rate.is_finite()):
            raise ValueError(
                "Invalid funding rate for historical settlement at "
                f"{settlement.funding_time}: {rate!r}")
        if mark is None or (hasattr(mark, "is_finite")
                            and not mark.is_finite()) or mark <= 0:
            raise ValueError(
                "Invalid settlement mark price for historical "
                f"settlement at {settlement.funding_time}: {mark!r}")
        rate_dec = rate if isinstance(rate, Decimal) else Decimal(str(rate))
        mark_dec = mark if isinstance(mark, Decimal) else Decimal(str(mark))
        quantity = Decimal(str(self.position.quantity))
        notional = quantity * mark_dec
        payment = (Decimal(-self.position.side) * quantity
                   * mark_dec * rate_dec)
        payment_float = float(payment)
        self.balance += payment_float
        self.total_funding += payment_float
        self._funding_balance_adjustments += payment_float
        entry = FundingLedgerEntry(
            symbol=settlement.symbol, funding_time=settlement.funding_time,  # type: ignore[arg-type]
            funding_rate=rate_dec, settlement_mark_price=mark_dec,
            rate_type=settlement.rate_type, position_side=self.position.side,
            position_quantity=quantity, settlement_notional=notional,
            funding_payment=payment, balance_adjustment=payment)
        self.funding_ledger.append(entry)
        return entry

    def _reconcile_historical_funding(self) -> None:
        ledger_total = sum(
            (entry.funding_payment for entry in self.funding_ledger),
            Decimal("0"))
        tolerance = max(1e-6, 1e-9 * abs(self.total_funding))
        if abs(float(ledger_total) - self.total_funding) > tolerance:
            raise RuntimeError(
                "Historical funding reconciliation failed: ledger "
                f"total {ledger_total} != total_funding "
                f"{self.total_funding} (tolerance {tolerance})")
        if abs(self._funding_balance_adjustments
               - self.total_funding) > tolerance:
            raise RuntimeError(
                "Historical funding reconciliation failed: funding "
                f"balance adjustments {self._funding_balance_adjustments} "
                f"!= total_funding {self.total_funding} "
                f"(tolerance {tolerance})")

    def reconcile_historical_funding(self) -> bool:
        """Verify ledger, ``total_funding``, and balance adjustments agree."""
        self._reconcile_historical_funding()
        return True

    @staticmethod
    def _ledger_entry_dict(entry: FundingLedgerEntry) -> dict[str, Any]:
        return {
            "symbol": entry.symbol,
            "funding_time": entry.funding_time.isoformat(),
            "funding_rate": str(entry.funding_rate),
            "settlement_mark_price": str(entry.settlement_mark_price),
            "rate_type": entry.rate_type,
            "position_side": entry.position_side,
            "position_quantity": str(entry.position_quantity),
            "settlement_notional": str(entry.settlement_notional),
            "funding_payment": str(entry.funding_payment),
            "balance_adjustment": str(entry.balance_adjustment),
        }

    def _reward(self, prev_equity: float, new_equity: float) -> float:
        """Fractional equity change per step (unit: fraction of the
        initial balance). Kept separate from account accounting."""
        return (new_equity - prev_equity) / self.config.initial_balance * self.config.reward_scale

    def _observation(self) -> dict[str, float]:
        candle = self.current_candle
        prev_close = (self.candles[self._index - 1].close
                      if self._index > 0 else candle.open)
        close = float(candle.close or 0.0)
        obs: dict[str, float] = {
            "index": float(self._index),
            "progress": self._index / max(1, len(self.candles) - 1),
            "open": float(candle.open or 0.0),
            "high": float(candle.high or 0.0),
            "low": float(candle.low or 0.0),
            "close": close,
            "volume": float(candle.volume or 0.0),
            "quote_volume": float(candle.quote_volume or 0.0),
            "price_change": (close / float(prev_close) - 1.0)
            if prev_close else 0.0,
            "position_side": float(self.position.side if self.position else 0),
            "position_quantity": self.position.quantity if self.position else 0.0,
            "entry_price": self.position.entry_price if self.position else 0.0,
            "unrealized_pnl": self.unrealized_pnl(),
            "realized_pnl": self.realized_pnl,
            "balance": self.balance,
            "available_balance": self.available_balance,
            "equity": self.equity,
            "margin_used": self.margin_used,
            "leverage": self.config.leverage,
            "total_fees": self.total_fees,
            "total_funding": self.total_funding,
        }
        obs.update(self._news_features(candle))
        return obs

    def _news_features(self, candle: Kline) -> dict[str, float]:
        """Point-in-time news features (never future articles)."""
        out = {"news_count": 0.0, "news_compound": 0.0,
               "news_confidence": 0.0, "news_age_seconds": -1.0}
        if self.news_archive is None or not self.news_asset:
            return out
        decision_time = candle.close_time or candle.open_time
        if decision_time is None:
            return out
        articles = self.news_archive.available_at(
            decision_time, asset=self.news_asset)
        if not articles:
            return out
        compounds, confidences = [], []
        latest = articles[-1]
        for article in articles[-20:]:
            result = self.news_archive.classification_for(article.article_id)
            if result:
                compounds.append(float(result["compound"]))
                confidences.append(float(result["confidence_proxy"]))
        out["news_count"] = float(len(articles))
        if compounds:
            out["news_compound"] = sum(compounds) / len(compounds)
            out["news_confidence"] = sum(confidences) / len(confidences)
        if latest.published_at is not None:
            out["news_age_seconds"] = (
                decision_time - latest.published_at).total_seconds()
        return out

    # -- main interaction -------------------------------------------
    def step(self, action: Action | int) -> StepResult:
        if self._terminated or self._truncated:
            raise RuntimeError("Episode finished; call reset()")
        action = Action(action)
        prev_equity = self.equity
        historical = self.config.funding_mode == "historical"
        info: dict[str, Any] = {"gaps_in_history": len(self.gaps),
                                "rejected_candles": len(self.rejected_candles),
                                "funding_mode": self.config.funding_mode,
                                "historical_accounting_implemented": historical}

        if self._index + 1 >= len(self.candles):
            self._truncated = True
            self._termination_reason = "end_of_data"
            if historical:
                info["funding_payment"] = 0.0
                info["funding_ledger_entries"] = []
                info["funding_ledger_count"] = len(self.funding_ledger)
                info["total_funding"] = self.total_funding
            info["reason"] = self._termination_reason
            return StepResult(self._observation(), 0.0, False, True, info)

        self._index += 1
        candle = self.current_candle

        # Daily-limit bookkeeping anchor.
        day_key = candle.open_time.date() if candle.open_time else None
        if day_key != self._day_key:
            self._day_key = day_key
            self._day_start_equity = prev_equity

        # Historical Stage 2, bucket A: settlements at or before the
        # candle open apply to the pre-execution position.
        pre_entries: list[FundingLedgerEntry] = []
        if historical:
            assert candle.open_time is not None
            pre_entries = self._consume_historical_settlements(
                candle.open_time)

        # Validate the action against the current position state.
        valid = (
            action == Action.HOLD
            or (action == Action.OPEN_LONG and self.position is None)
            or (action == Action.OPEN_SHORT and self.position is None)
            or (action == Action.CLOSE_LONG and self.position is not None
                and self.position.side == 1)
            or (action == Action.CLOSE_SHORT and self.position is not None
                and self.position.side == -1))
        info["invalid_action"] = not valid
        effective = action if valid else Action.HOLD

        event: dict[str, Any] | None = None
        if effective == Action.OPEN_LONG:
            event = self._open_position(1, candle)
        elif effective == Action.OPEN_SHORT:
            event = self._open_position(-1, candle)
        if event is not None and event.get("event") == "open_rejected":
            info["invalid_action"] = True
            info["invalid_reason"] = event["invalid_reason"]
            info["event"] = "open_rejected"
            event = None
        elif effective in (Action.CLOSE_LONG, Action.CLOSE_SHORT):
            event = self._close_position(
                float(candle.open or 0.0), "close")

        if self.position is not None and event is None or (
                self.position is not None and effective in (
                    Action.OPEN_LONG, Action.OPEN_SHORT, Action.HOLD)):
            risk_event = self._risk_checks(candle)
            if risk_event is not None:
                event = risk_event
                if risk_event["event"] == "liquidation":
                    self._terminated = True
                    self._termination_reason = "liquidation"

        if historical:
            # Bucket D: settlements in (open, close] apply to the
            # position remaining after action and risk checks. Risk
            # checks run first because candles cannot establish exact
            # intrabar ordering; Binance settlement timing may deviate
            # from nominal timestamps, so this is a simulation
            # convention, not exact exchange timing.
            if candle.close_time is None:
                raise ValueError(
                    "Historical funding requires a valid candle "
                    f"close_time; candle at {candle.open_time} has "
                    "close_time=None")
            post_entries = self._consume_historical_settlements(
                candle.close_time)
            step_entries = pre_entries + post_entries
            info["funding_payment"] = float(sum(
                (entry.funding_payment for entry in step_entries),
                Decimal("0")))
            info["funding_ledger_entries"] = [
                self._ledger_entry_dict(entry) for entry in step_entries]
            info["funding_ledger_count"] = len(self.funding_ledger)
            info["total_funding"] = self.total_funding
        else:
            funding = self._apply_funding(candle)
            if funding:
                info["funding_payment"] = funding
        if event:
            info.update(event)

        # Daily profit/loss limits force-close and end the episode.
        cfg = self.config
        daily_pnl = self.equity - self._day_start_equity
        info["daily_pnl"] = daily_pnl
        if ((cfg.daily_loss_limit is not None and daily_pnl <= -cfg.daily_loss_limit)
                or (cfg.daily_profit_limit is not None
                    and daily_pnl >= cfg.daily_profit_limit)):
            if self.position is not None:
                info.update(self._close_position(
                    float(candle.close or 0.0), "daily_limit_close"))
            self._terminated = True
            self._termination_reason = "daily_limit"

        if self.equity <= 0:
            self._terminated = True
            self._termination_reason = "bankrupt"

        if self._index >= len(self.candles) - 1 and not self._terminated:
            self._truncated = True
            self._termination_reason = "end_of_data"

        info["reason"] = self._termination_reason
        reward = self._reward(prev_equity, self.equity)
        return StepResult(self._observation(), reward,
                          self._terminated, self._truncated, info)
