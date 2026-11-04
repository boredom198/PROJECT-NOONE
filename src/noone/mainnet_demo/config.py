"""Configuration for Mainnet Demo Execution Mode."""
from dataclasses import dataclass, field
from enum import Enum
import math
import os
from pathlib import Path
from typing import Tuple

_REQUIRED_ARM_PHRASE = "ARM_MAINNET_DEMO_TRADE"


class DemoState(str, Enum):
    DISARMED = "DISARMED"
    ARMED = "ARMED"
    LOCKED = "LOCKED"


class MainnetDemoError(Exception):
    pass


class MainnetDemoDisarmedError(MainnetDemoError):
    pass


class MainnetDemoLockedError(MainnetDemoError):
    pass


class RiskBudgetExceededError(MainnetDemoError):
    pass


class EmergencyExitError(MainnetDemoError):
    pass


@dataclass(frozen=True)
class MainnetDemoConfig:
    symbol: str = "BTC/USDT:USDT"
    raw_symbol: str = "BTCUSDT"
    leverage: int = 1
    margin_mode: str = "isolated"
    max_loss_budget_usdt: float = 5.0
    stop_loss_pct: float = 0.015
    taker_fee_rate: float = 0.0005
    slippage_buffer_pct: float = 0.0010
    funding_buffer_pct: float = 0.0003
    emergency_exposure_pct: float = 0.0010
    model_path: Path = field(default_factory=lambda: Path("runs/stage2-realdata/final_model.zip"))
    database_path: Path = field(default_factory=lambda: Path("data/mainnet_demo.db"))
    armed: bool = False
    confirmation_phrase: str = ""
    # Kept for caller compatibility; __post_init__ prevents overriding it.
    required_arm_phrase: str = _REQUIRED_ARM_PHRASE
    confirm_futures_trading_enabled: bool = False
    confirm_withdrawals_disabled: bool = False
    confirm_ip_whitelist_configured: bool = False
    api_key_env_var: str = "BINANCE_MAINNET_API_KEY"
    api_secret_env_var: str = "BINANCE_MAINNET_API_SECRET"

    def __post_init__(self) -> None:
        float_fields = {
            "max_loss_budget_usdt": (0.0, 10.0),
            "stop_loss_pct": (0.0, 0.10),
            "taker_fee_rate": (0.0, 0.01),
            "slippage_buffer_pct": (0.0, 0.05),
            "funding_buffer_pct": (0.0, 0.05),
            "emergency_exposure_pct": (0.0, 0.05),
        }
        for name, (lower, upper) in float_fields.items():
            value = getattr(self, name)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value)):
                raise ValueError(f"Configuration parameter '{name}' must be finite numeric, got {value!r}")
            if value < lower or value > upper:
                raise ValueError(f"Configuration parameter '{name}' ({value}) outside [{lower}, {upper}]")

        if type(self.leverage) is not int or self.leverage != 1:
            raise ValueError(f"Mainnet demo requires integer 1x leverage, got {self.leverage!r}")
        if not isinstance(self.margin_mode, str):
            raise ValueError("margin_mode must be a string")
        if self.margin_mode.lower() != "isolated":
            raise ValueError(f"Mainnet demo requires isolated margin, got {self.margin_mode!r}")
        if self.max_loss_budget_usdt <= 0.0:
            raise ValueError("max_loss_budget_usdt must be greater than zero")
        if self.stop_loss_pct <= 0.0:
            raise ValueError("stop_loss_pct must be greater than zero")
        if self.required_arm_phrase != _REQUIRED_ARM_PHRASE:
            raise ValueError("required_arm_phrase is fixed and cannot be overridden")

    def is_armed(self) -> bool:
        """These booleans are operator attestations, not exchange-side verification."""
        return (
            self.armed is True
            and self.confirmation_phrase == _REQUIRED_ARM_PHRASE
            and self.confirm_futures_trading_enabled is True
            and self.confirm_withdrawals_disabled is True
            and self.confirm_ip_whitelist_configured is True
        )

    def get_api_credentials(self) -> Tuple[str, str]:
        key = os.environ.get(self.api_key_env_var, "").strip()
        secret = os.environ.get(self.api_secret_env_var, "").strip()
        if not key or not secret:
            raise ValueError(
                "Missing Mainnet exchange credentials in environment variables: "
                f"{self.api_key_env_var}, {self.api_secret_env_var}"
            )
        return key, secret
