"""Account and position parsing for Component 3.

Maps Binance V2 endpoints (GET /fapi/v2/account and GET /fapi/v2/positionRisk) to
Component 1's AccountSnapshot contract while strictly enforcing positionInitialMargin
validity, One-Way mode verification, and account-wide exposure detection.
Conforms strictly to PROJECT NOONE Technical Design Specification V2.2.4 Sections 4 and 5.
"""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Set

from noone.rl_engine.inference import AccountSnapshot
from noone.exchange.errors import (
    AccountSyncError,
    DuplicatePositionRecordError,
    IncompatiblePositionSideError,
    MissingPositionRecordError,
)

DECIMAL_ZERO_TOLERANCE = Decimal("1e-8")


@dataclass(frozen=True)
class PositionAuditResult:
    """Result of an account-wide position risk audit."""

    target_position: Optional[Dict[str, Any]]
    external_positions: List[Dict[str, Any]]
    has_external_positions: bool
    is_dual_side: bool


def _to_decimal(val: Any, field_name: str) -> Decimal:
    if val is None:
        raise AccountSyncError(f"Missing required numeric field: {field_name}")
    try:
        d = Decimal(str(val))
        if d.is_nan() or d.is_infinite():
            raise AccountSyncError(f"Field {field_name} is NaN or Infinite: {val}")
        return d
    except (InvalidOperation, ValueError, TypeError) as e:
        raise AccountSyncError(f"Malformed numeric value for {field_name}: {val}") from e


def audit_account_positions(
    position_risk_data: List[Dict[str, Any]], target_symbol: str
) -> PositionAuditResult:
    """Audit all position risk records to detect non-target positions or hedge mode."""
    target_pos: Optional[Dict[str, Any]] = None
    external_positions: List[Dict[str, Any]] = []
    is_dual_side = False

    raw_target = target_symbol.upper().replace("/", "").replace(":USDT", "").replace(":", "")

    for p in position_risk_data:
        sym = p.get("symbol", "").upper()
        pos_side = p.get("positionSide", "BOTH")
        if pos_side in ("LONG", "SHORT"):
            is_dual_side = True

        amt = _to_decimal(p.get("positionAmt", "0"), f"positionAmt for {sym}")
        is_active = abs(amt) > DECIMAL_ZERO_TOLERANCE

        if sym == raw_target:
            if target_pos is not None:
                raise DuplicatePositionRecordError(
                    f"Duplicate position risk record encountered for target symbol {target_symbol}"
                )
            target_pos = p
        else:
            if is_active:
                external_positions.append(p)

    return PositionAuditResult(
        target_position=target_pos,
        external_positions=external_positions,
        has_external_positions=len(external_positions) > 0,
        is_dual_side=is_dual_side,
    )


def audit_open_orders(
    open_orders_data: List[Dict[str, Any]], known_client_order_ids: Set[str]
) -> List[Dict[str, Any]]:
    """Identify unexplained open orders across the exchange."""
    unexplained = []
    for order in open_orders_data:
        coid = order.get("clientOrderId")
        if coid not in known_client_order_ids:
            unexplained.append(order)
    return unexplained


def parse_account_snapshot(
    account_data: Dict[str, Any],
    position_risk_data: List[Dict[str, Any]],
    target_symbol: str,
    session_realized_pnl: float = 0.0,
) -> AccountSnapshot:
    """Parse GET /fapi/v2/account and GET /fapi/v2/positionRisk into AccountSnapshot.

    Enforces:
    - Equity from totalMarginBalance > 0.0.
    - Available balance from assets[asset=='USDT'].availableBalance >= 0.0.
    - positionInitialMargin strictly extracted for active positions (> 0.0).
    - Prohibits calculated/fallback estimates or silent zeros.
    - Realized PnL strictly from session tracker, NOT Binance lifetime totalRealizedProfit.
    """
    if not isinstance(account_data, dict):
        raise AccountSyncError(f"account_data must be a dict, got {type(account_data)}")
    if not isinstance(position_risk_data, list):
        raise AccountSyncError(f"position_risk_data must be a list, got {type(position_risk_data)}")

    raw_target = target_symbol.upper().replace("/", "").replace(":USDT", "").replace(":", "")

    # 1. Equity
    raw_equity = account_data.get("totalMarginBalance")
    if raw_equity is None:
        raise AccountSyncError("Missing 'totalMarginBalance' in account data response.")
    equity_dec = _to_decimal(raw_equity, "totalMarginBalance")
    if equity_dec <= Decimal("0"):
        raise AccountSyncError(f"Invalid account equity: {equity_dec}. Must be > 0.0.")
    equity = float(equity_dec)

    # 2. Available balance (USDT)
    assets = account_data.get("assets", [])
    if not isinstance(assets, list):
        raise AccountSyncError(f"Expected 'assets' list in account data, got {type(assets)}")

    usdt_asset = None
    for a in assets:
        if isinstance(a, dict) and a.get("asset") == "USDT":
            usdt_asset = a
            break

    if usdt_asset is None:
        raise AccountSyncError("USDT asset record not found in account assets list.")

    avail_raw = usdt_asset.get("availableBalance")
    if avail_raw is None:
        raise AccountSyncError("Missing 'availableBalance' for USDT asset in account data.")
    avail_dec = _to_decimal(avail_raw, "USDT.availableBalance")
    if avail_dec < Decimal("0"):
        raise AccountSyncError(f"Invalid negative available balance: {avail_dec}")
    available_balance = float(avail_dec)

    # 3. Position Risk audit & target position extraction
    audit = audit_account_positions(position_risk_data, raw_target)
    if audit.is_dual_side:
        raise IncompatiblePositionSideError(
            "Account or symbol is configured in Hedge Mode (dual-side). One-Way Mode ('BOTH') is required."
        )

    target_pos_risk = audit.target_position
    if target_pos_risk is None:
        raise MissingPositionRecordError(
            f"Target symbol {raw_target} position record missing from position risk response."
        )

    pos_side_str = target_pos_risk.get("positionSide", "BOTH")
    if pos_side_str != "BOTH":
        raise IncompatiblePositionSideError(
            f"Expected positionSide == 'BOTH' for One-Way mode, got '{pos_side_str}'"
        )

    amt_dec = _to_decimal(target_pos_risk.get("positionAmt", "0"), f"positionAmt for {raw_target}")
    is_active = abs(amt_dec) > DECIMAL_ZERO_TOLERANCE

    # 4. Position Initial Margin from /fapi/v2/account positions array
    account_positions = account_data.get("positions", [])
    if not isinstance(account_positions, list):
        raise AccountSyncError(
            f"Expected 'positions' list in account data, got {type(account_positions)}"
        )

    target_acc_pos = None
    for ap in account_positions:
        if isinstance(ap, dict) and ap.get("symbol", "").upper() == raw_target:
            target_acc_pos = ap
            break

    if target_acc_pos is None:
        raise MissingPositionRecordError(
            f"Target symbol {raw_target} missing from account positions list."
        )

    raw_pim = target_acc_pos.get("positionInitialMargin")

    if is_active:
        if amt_dec > Decimal("0"):
            pos_side = 1
        else:
            pos_side = -1
        pos_qty = float(abs(amt_dec))

        entry_price_dec = _to_decimal(target_pos_risk.get("entryPrice", "0"), "entryPrice")
        unrealized_pnl_dec = _to_decimal(target_pos_risk.get("unRealizedProfit", "0"), "unRealizedProfit")

        entry_price = float(entry_price_dec)
        unrealized_pnl = float(unrealized_pnl_dec)

        if raw_pim is None:
            raise AccountSyncError(
                f"Missing 'positionInitialMargin' for active position {raw_target}."
            )
        pim_dec = _to_decimal(raw_pim, "positionInitialMargin")
        if pim_dec <= Decimal("0"):
            raise AccountSyncError(
                f"Invalid positionInitialMargin ({pim_dec}) for active position with quantity {pos_qty}. Must be > 0.0."
            )
        margin_used = float(pim_dec)
    else:
        pos_side = 0
        pos_qty = 0.0
        entry_price = 0.0
        unrealized_pnl = 0.0
        margin_used = 0.0

    return AccountSnapshot(
        equity=equity,
        available_balance=available_balance,
        position_side=pos_side,
        position_quantity=pos_qty,
        entry_price=entry_price,
        unrealized_pnl=unrealized_pnl,
        realized_pnl=float(session_realized_pnl),
        margin_used=margin_used,
        progress=0.0,
    )

