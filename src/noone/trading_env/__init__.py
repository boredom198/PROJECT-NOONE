"""Component 8: Trading Environment.

Historical perpetual-futures simulation for the future DQN + RL
Engine (Component 1, not implemented here). The environment executes
agent-supplied actions; it contains no trading strategy, no
indicators, and no neural-network/agent code.
"""
from noone.trading_env.config import TradingConfig
from noone.trading_env.environment import (
    Action,
    FundingLedgerEntry,
    PerpetualFuturesEnv,
    Position,
    StepResult,
)

__all__ = [
    "Action",
    "FundingLedgerEntry",
    "PerpetualFuturesEnv",
    "Position",
    "StepResult",
    "TradingConfig",
]
