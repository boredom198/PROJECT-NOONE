"""Configurable trading/risk parameters for Component 8.

All limits are enforced by the environment, independently of the
agent's chosen action. ``None`` disables an optional limit.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradingConfig:
    initial_balance: float = 10_000.0  # quote currency (e.g. USDT)
    leverage: float = 5.0
    max_leverage: float = 20.0
    # Fraction of available balance reserved as margin per position.
    capital_allocation_fraction: float = 0.10
    max_capital_allocation_fraction: float = 0.50
    max_position_notional: float | None = None  # quote currency cap
    fee_rate: float = 0.0005  # taker fee, charged on open and close
    slippage_rate: float = 0.0002  # adverse price adjustment per fill
    max_slippage_rate: float = 0.001  # fills worse than this are rejected
    maintenance_margin_rate: float = 0.005  # simplified, not exchange-exact
    # Funding mode: explicit. "constant" (default) uses the
    # configured simulation rate below; "historical" requires a full
    # Component 4 FundingHistory and is strict at construction
    # (see PerpetualFuturesEnv). Historical accounting itself is
    # applied from the history's settlements, never synthesised.
    funding_mode: str = "constant"
    # Funding: configured simulation rate (not exchange-exact),
    # applied at recurring UTC funding times (Binance-style 8h
    # schedule by default). Longs pay when positive, shorts receive.
    # Constant mode only; ignored as a rate source in historical mode.
    funding_rate: float = 0.0001
    funding_times_utc: tuple[str, ...] = ("00:00", "08:00", "16:00")
    # Deprecated: formerly the candle-count funding interval. Kept
    # for interface compatibility; funding is timestamp-based now.
    funding_interval_candles: int = 480
    stop_loss_pct: float | None = None  # e.g. 0.02 = 2% adverse from entry
    take_profit_pct: float | None = None
    daily_loss_limit: float | None = None  # quote currency per UTC day
    daily_profit_limit: float | None = None
    reward_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.funding_mode not in ("constant", "historical"):
            raise ValueError(
                f"Invalid funding_mode {self.funding_mode!r} "
                "(want 'constant' or 'historical')")
        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be positive")
        if not 0 < self.leverage <= self.max_leverage:
            raise ValueError("leverage must be in (0, max_leverage]")
        if not 0 < self.capital_allocation_fraction:
            raise ValueError("capital_allocation_fraction must be positive")
        if self.capital_allocation_fraction > self.max_capital_allocation_fraction:
            raise ValueError(
                "capital_allocation_fraction exceeds "
                "max_capital_allocation_fraction")
        if self.slippage_rate > self.max_slippage_rate:
            raise ValueError("slippage_rate exceeds max_slippage_rate")
        if self.fee_rate < 0 or self.slippage_rate < 0:
            raise ValueError("fee/slippage rates cannot be negative")
        if self.funding_interval_candles < 1:
            raise ValueError("funding_interval_candles must be >= 1")
        for value in self.funding_times_utc:
            parts = value.split(":")
            if len(parts) != 2 or not all(p.isdigit() for p in parts):
                raise ValueError(f"Invalid funding time {value!r} (want HH:MM)")
            hour, minute = int(parts[0]), int(parts[1])
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError(f"Invalid funding time {value!r} (want HH:MM)")
