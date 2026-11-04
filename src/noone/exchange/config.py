"""Configuration invariants for Component 3 (CCXT Exchange Connectivity / Account).

Defined strictly in accordance with PROJECT NOONE Technical Design Specification V2.2.4.
Enforces hard runtime safety locks to prevent production trading.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional, Tuple

from noone.exchange.errors import ExchangeError

# Hard runtime safety lock: Production execution is permanently disabled in this release.
PRODUCTION_HARD_LOCKED: bool = True


@dataclass(frozen=True)
class Component3Config:
    """Component 3 runtime configuration parameters and safety guards."""

    testnet: bool = True
    production_confirmed: bool = False
    symbol: str = "BTCUSDT"
    api_key_env_var: str = "BINANCE_TESTNET_API_KEY"
    api_secret_env_var: str = "BINANCE_TESTNET_API_SECRET"
    leverage: int = 10
    margin_type: str = "cross"
    max_position_notional: float = 50_000.0
    margin_allocation_fraction: float = 0.10
    max_daily_loss_pct: float = 0.03
    database_path: Path = field(default_factory=lambda: Path("data/execution_journal.db"))
    fee_rate: float = 0.0005
    safety_buffer_pct: float = 0.005
    clock_drift_threshold_ms: int = 1000
    equity_freshness_ms: int = 3000
    position_freshness_ms: int = 2000
    mark_price_freshness_ms: int = 2000
    filter_cache_ttl_s: int = 3600

    def __post_init__(self) -> None:
        # Defect 9: Hard runtime safety lock. No configuration flag or boolean may enable production.
        if not self.testnet or self.production_confirmed:
            raise ExchangeError(
                "PRODUCTION EXECUTION HARD LOCKED: Component 3 production trading is disabled by hard runtime lock. "
                "Neither testnet=False nor production_confirmed=True is permitted."
            )

        if self.leverage < 1 or self.leverage > 125:
            raise ValueError(f"Invalid leverage setting: {self.leverage}. Must be in [1, 125].")
        if self.margin_allocation_fraction <= 0.0 or self.margin_allocation_fraction > 1.0:
            raise ValueError(
                f"Invalid margin_allocation_fraction: {self.margin_allocation_fraction}. Must be in (0.0, 1.0]."
            )
        if self.max_position_notional <= 0.0:
            raise ValueError(
                f"Invalid max_position_notional: {self.max_position_notional}. Must be > 0.0."
            )
        if self.max_daily_loss_pct <= 0.0 or self.max_daily_loss_pct >= 1.0:
            raise ValueError(
                f"Invalid max_daily_loss_pct: {self.max_daily_loss_pct}. Must be in (0.0, 1.0)."
            )
        if self.margin_type.lower() != "cross":
            raise ValueError(
                f"Unsupported margin_type: {self.margin_type}. Only 'cross' margin is currently supported."
            )

    @property
    def raw_symbol(self) -> str:
        """Returns raw exchange symbol format, e.g. 'BTCUSDT'."""
        s = self.symbol.replace("/", "").replace(":USDT", "").replace(":", "")
        return s.upper()

    @property
    def ccxt_symbol(self) -> str:
        """Returns unified CCXT linear futures market symbol, e.g. 'BTC/USDT:USDT'."""
        raw = self.raw_symbol
        if raw.endswith("USDT"):
            base = raw[:-4]
            return f"{base}/USDT:USDT"
        return self.symbol

    def get_api_credentials(self) -> Tuple[str, str]:
        """Retrieve API key and secret from environment variables."""
        api_key = os.environ.get(self.api_key_env_var, "")
        api_secret = os.environ.get(self.api_secret_env_var, "")
        if not api_key or not api_secret:
            raise ValueError(
                f"Missing exchange API credentials in environment variables: "
                f"{self.api_key_env_var}, {self.api_secret_env_var}"
            )
        return api_key, api_secret
