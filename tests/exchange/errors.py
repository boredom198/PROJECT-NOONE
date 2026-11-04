"""Exception hierarchy for Component 3 (CCXT Exchange Connectivity / Account).

Defined strictly in accordance with PROJECT NOONE Technical Design Specification V2.2.4.
"""

from typing import Optional


class ExchangeError(Exception):
    """Base exception for all Component 3 exchange-related errors."""
    pass


class AccountSyncError(ExchangeError):
    """Raised when account balance or position synchronization fails."""
    pass


class RiskGateError(ExchangeError):
    """Raised when risk check, sizing, exposure, or margin buffer validation fails."""
    pass


class DailyLossLimitError(RiskGateError):
    """Raised when daily loss limit is breached or active."""
    pass


class DailyPnLUninitializedError(RiskGateError):
    """Raised when daily baseline is uninitialized, invalid, or unverified."""
    pass


class FilterValidationError(ExchangeError):
    """Raised when symbol trading filters (lot size, notional) are invalid or violated."""
    pass


class SafetyFreezeError(ExchangeError):
    """Raised when an operation is prohibited due to an active Safety Freeze."""

    def __init__(self, message: str, reason: Optional[str] = None):
        super().__init__(message)
        self.reason = reason


class OrderRecoveryError(ExchangeError):
    """Raised when order recovery fails or outcome cannot be authoritatively established."""
    pass


class StateDiscrepancyError(ExchangeError):
    """Raised when local journal state diverges from exchange source of truth."""
    pass


class IncompatiblePositionSideError(ExchangeError):
    """Raised when account position mode is dual-side (hedge) or position records are incompatible."""
    pass


class MissingPositionRecordError(ExchangeError):
    """Raised when expected position record is absent from exchange response."""
    pass


class DuplicatePositionRecordError(ExchangeError):
    """Raised when unexpected duplicate position records are returned for a symbol."""
    pass


class StaleMarketDataError(ExchangeError):
    """Raised when market data or mark price exceeds freshness TTL."""
    pass


class ClockDriftError(ExchangeError):
    """Raised when local system clock drifts excessively from exchange server time."""
    pass


class InvalidBaselineError(ExchangeError):
    """Raised when UTC midnight baseline is missing, invalid, or corrupt."""
    pass


class UnverifiedTransferError(ExchangeError):
    """Raised when external transfer classification is ambiguous or cannot be verified."""
    pass


class ResidualPositionError(ExchangeError):
    """Raised when residual dust or position size below minimum filter remains active."""
    pass

