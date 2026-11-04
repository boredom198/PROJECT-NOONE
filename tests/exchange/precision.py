"""Precision handling and Binance market filter parser for Component 3.

Implements strict Decimal downward quantization, filter hierarchy parsing
(MARKET_LOT_SIZE vs LOT_SIZE, MIN_NOTIONAL vs NOTIONAL), and residual dust detection.
Conforms strictly to PROJECT NOONE Technical Design Specification V2.2.4 Sections 8 and 14.
"""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any, Dict, List, Optional

from noone.exchange.errors import FilterValidationError, ResidualPositionError


@dataclass(frozen=True)
class ParsedSymbolFilters:
    """Validated symbol constraints parsed from Binance exchangeInfo filters."""

    step_size: Decimal
    min_qty: Decimal
    max_qty: Decimal
    min_notional: Decimal
    tick_size: Optional[Decimal] = None


def _to_decimal(val: Any, field_name: str) -> Decimal:
    if val is None:
        raise FilterValidationError(f"Missing required filter field: {field_name}")
    try:
        d = Decimal(str(val))
        if d.is_nan() or d.is_infinite():
            raise FilterValidationError(f"Filter field {field_name} is NaN or Infinite: {val}")
        return d
    except (InvalidOperation, ValueError, TypeError) as e:
        raise FilterValidationError(f"Malformed numeric filter value for {field_name}: {val}") from e


def parse_symbol_filters(filters: List[Dict[str, Any]]) -> ParsedSymbolFilters:
    """Parse Binance exchangeInfo filter list following strict precedence rules.

    Precedence:
    1. Quantity: MARKET_LOT_SIZE preferred; if absent, LOT_SIZE (if applyToMarket is True or omitted).
    2. Notional: Maximum minNotional among applicable MIN_NOTIONAL and NOTIONAL filters.
    3. Price: PRICE_FILTER tickSize if present.
    """
    if not filters:
        raise FilterValidationError("Empty filters list provided for symbol.")

    market_lot_filter: Optional[Dict[str, Any]] = None
    lot_size_filter: Optional[Dict[str, Any]] = None
    notional_filters: List[Dict[str, Any]] = []
    price_filter: Optional[Dict[str, Any]] = None

    seen_types = set()
    for f in filters:
        ftype = f.get("filterType")
        if not ftype:
            raise FilterValidationError(f"Filter record missing filterType: {f}")

        # Check for duplicate conflicting filter types where duplicate is invalid
        if ftype in ("MARKET_LOT_SIZE", "LOT_SIZE", "PRICE_FILTER"):
            if ftype in seen_types:
                raise FilterValidationError(f"Duplicate filter type detected: {ftype}")
            seen_types.add(ftype)

        if ftype == "MARKET_LOT_SIZE":
            market_lot_filter = f
        elif ftype == "LOT_SIZE":
            lot_size_filter = f
        elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
            notional_filters.append(f)
        elif ftype == "PRICE_FILTER":
            price_filter = f

    # 1. Parse Quantity filter
    step_size: Optional[Decimal] = None
    min_qty: Optional[Decimal] = None
    max_qty: Optional[Decimal] = None

    if market_lot_filter is not None:
        s = _to_decimal(market_lot_filter.get("stepSize"), "MARKET_LOT_SIZE.stepSize")
        mn = _to_decimal(market_lot_filter.get("minQty"), "MARKET_LOT_SIZE.minQty")
        mx = _to_decimal(market_lot_filter.get("maxQty"), "MARKET_LOT_SIZE.maxQty")
        if s > 0 and mn > 0 and mx >= mn:
            step_size, min_qty, max_qty = s, mn, mx
        else:
            raise FilterValidationError(
                f"Invalid MARKET_LOT_SIZE bounds: stepSize={s}, minQty={mn}, maxQty={mx}"
            )
    elif lot_size_filter is not None:
        apply_to_market = lot_size_filter.get("applyToMarket", True)
        if apply_to_market:
            s = _to_decimal(lot_size_filter.get("stepSize"), "LOT_SIZE.stepSize")
            mn = _to_decimal(lot_size_filter.get("minQty"), "LOT_SIZE.minQty")
            mx = _to_decimal(lot_size_filter.get("maxQty"), "LOT_SIZE.maxQty")
            if s > 0 and mn > 0 and mx >= mn:
                step_size, min_qty, max_qty = s, mn, mx
            else:
                raise FilterValidationError(
                    f"Invalid LOT_SIZE bounds: stepSize={s}, minQty={mn}, maxQty={mx}"
                )
        else:
            raise FilterValidationError("LOT_SIZE filter explicitly does not apply to market orders.")
    else:
        raise FilterValidationError("Neither MARKET_LOT_SIZE nor LOT_SIZE filter found for symbol.")

    if step_size is None or min_qty is None or max_qty is None:
        raise FilterValidationError("Unable to resolve quantity constraints from filters.")

    # 2. Parse Notional filter (enforce strictest applicable)
    applicable_min_notionals: List[Decimal] = []
    for nf in notional_filters:
        ftype = nf.get("filterType")
        # Check market applicability
        applies = True
        if ftype == "MIN_NOTIONAL":
            applies = nf.get("applyToMarket", True)
        elif ftype == "NOTIONAL":
            applies = nf.get("applyMinToMarket", True)

        if applies:
            # Some versions use minNotional, some notional
            val = nf.get("notional") if nf.get("notional") is not None else nf.get("minNotional")
            if val is not None:
                d = _to_decimal(val, f"{ftype}.minNotional")
                if d > 0:
                    applicable_min_notionals.append(d)

    if not applicable_min_notionals:
        raise FilterValidationError("No applicable minimum notional filter resolved.")

    strictest_min_notional = max(applicable_min_notionals)

    # 3. Parse Price filter (tickSize)
    tick_size: Optional[Decimal] = None
    if price_filter is not None:
        ts = _to_decimal(price_filter.get("tickSize"), "PRICE_FILTER.tickSize")
        if ts > 0:
            tick_size = ts

    return ParsedSymbolFilters(
        step_size=step_size,
        min_qty=min_qty,
        max_qty=max_qty,
        min_notional=strictest_min_notional,
        tick_size=tick_size,
    )


def quantize_downward(quantity: Decimal, step_size: Decimal) -> Decimal:
    """Truncate quantity strictly downward to the nearest multiple of step_size.

    Prohibition: Rounding upward is strictly forbidden.
    """
    if step_size <= Decimal("0"):
        raise FilterValidationError(f"Invalid step_size: {step_size}")
    if quantity <= Decimal("0"):
        return Decimal("0")

    # (quantity // step_size) * step_size
    multiples = (quantity / step_size).to_integral_value(rounding=ROUND_DOWN)
    quantized = multiples * step_size

    # Format to match exponent of step_size
    step_exp = step_size.as_tuple().exponent
    quantized = quantized.quantize(Decimal(10) ** step_exp, rounding=ROUND_DOWN)
    return quantized


def check_residual_dust(
    remaining_qty: Decimal, mark_price: Decimal, filters: ParsedSymbolFilters
) -> bool:
    """Determine whether an active position constitutes an unclosable dust/residual."""
    abs_qty = abs(remaining_qty)
    if abs_qty <= Decimal("1e-8"):
        return False  # Flat within reconciliation tolerance

    notional = abs_qty * mark_price
    if abs_qty < filters.min_qty or notional < filters.min_notional:
        return True
    return False

