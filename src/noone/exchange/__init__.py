"""Component 3 — CCXT Exchange Connectivity / Account.

Provides secure, auditable, and resilient execution integration with Binance USD(S)-M Futures.
Strictly adheres to PROJECT NOONE Technical Design Specification V2.2.4.
"""

from noone.exchange.config import Component3Config
from noone.exchange.client import BinanceFuturesClient
from noone.exchange.journal import (
    ExecutionJournal,
    JournalOrder,
    JournalTradeFill,
    DailyBaselineRecord,
    SafetyFreezeRecord,
)
from noone.exchange.precision import (
    ParsedSymbolFilters,
    parse_symbol_filters,
    parse_market_limits,
    quantize_downward,
    check_residual_dust,
)
from noone.exchange.account import (
    audit_account_positions,
    audit_open_orders,
    parse_account_snapshot,
    PositionAuditResult,
)
from noone.exchange.risk_gate import RiskGate
from noone.exchange.reconciliation import ReconciliationManager, RecoveryOutcome
from noone.exchange.orchestrator import ExchangeOrchestrator
from noone.exchange.errors import (
    ExchangeError,
    AccountSyncError,
    RiskGateError,
    DailyLossLimitError,
    DailyPnLUninitializedError,
    FilterValidationError,
    SafetyFreezeError,
    OrderRecoveryError,
    StateDiscrepancyError,
    IncompatiblePositionSideError,
    MissingPositionRecordError,
    DuplicatePositionRecordError,
    StaleMarketDataError,
    ClockDriftError,
    InvalidBaselineError,
    UnverifiedTransferError,
    ResidualPositionError,
)

__all__ = [
    "Component3Config",
    "BinanceFuturesClient",
    "ExecutionJournal",
    "JournalOrder",
    "JournalTradeFill",
    "DailyBaselineRecord",
    "SafetyFreezeRecord",
    "ParsedSymbolFilters",
    "parse_symbol_filters",
    "parse_market_limits",
    "quantize_downward",
    "check_residual_dust",
    "audit_account_positions",
    "audit_open_orders",
    "parse_account_snapshot",
    "PositionAuditResult",
    "RiskGate",
    "ReconciliationManager",
    "RecoveryOutcome",
    "ExchangeOrchestrator",
    "ExchangeError",
    "AccountSyncError",
    "RiskGateError",
    "DailyLossLimitError",
    "DailyPnLUninitializedError",
    "FilterValidationError",
    "SafetyFreezeError",
    "OrderRecoveryError",
    "StateDiscrepancyError",
    "IncompatiblePositionSideError",
    "MissingPositionRecordError",
    "DuplicatePositionRecordError",
    "StaleMarketDataError",
    "ClockDriftError",
    "InvalidBaselineError",
    "UnverifiedTransferError",
    "ResidualPositionError",
]

