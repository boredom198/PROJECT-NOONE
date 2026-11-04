"""Tests for precision handling and Binance market filter parser (Component 3).

Verifies filter precedence (MARKET_LOT_SIZE vs LOT_SIZE, MIN_NOTIONAL vs NOTIONAL),
malformed filter rejection, Decimal downward quantization, and residual dust detection (T12).
Conforms to PROJECT NOONE Technical Design Specification V2.2.4 Sections 8 and 14.
"""

from decimal import Decimal
import pytest

from noone.exchange.errors import FilterValidationError
from noone.exchange.precision import (
    check_residual_dust,
    parse_symbol_filters,
    ParsedSymbolFilters,
    quantize_downward,
)


def test_market_lot_size_precedence_over_lot_size():
    """Verify MARKET_LOT_SIZE takes precedence over LOT_SIZE when both are present."""
    raw_filters = [
        {
            "filterType": "LOT_SIZE",
            "minQty": "0.001",
            "maxQty": "1000.0",
            "stepSize": "0.001",
        },
        {
            "filterType": "MARKET_LOT_SIZE",
            "minQty": "0.005",
            "maxQty": "100.0",
            "stepSize": "0.005",
        },
        {
            "filterType": "MIN_NOTIONAL",
            "notional": "5.0",
            "applyToMarket": True,
        },
    ]

    parsed = parse_symbol_filters(raw_filters)
    assert parsed.step_size == Decimal("0.005")
    assert parsed.min_qty == Decimal("0.005")
    assert parsed.max_qty == Decimal("100.0")
    assert parsed.min_notional == Decimal("5.0")


def test_lot_size_fallback_when_market_lot_size_absent():
    """Verify LOT_SIZE is adopted when MARKET_LOT_SIZE is absent."""
    raw_filters = [
        {
            "filterType": "LOT_SIZE",
            "minQty": "0.001",
            "maxQty": "500.0",
            "stepSize": "0.001",
            "applyToMarket": True,
        },
        {
            "filterType": "NOTIONAL",
            "minNotional": "10.0",
            "applyMinToMarket": True,
        },
    ]

    parsed = parse_symbol_filters(raw_filters)
    assert parsed.step_size == Decimal("0.001")
    assert parsed.min_qty == Decimal("0.001")
    assert parsed.max_qty == Decimal("500.0")
    assert parsed.min_notional == Decimal("10.0")


def test_lot_size_rejected_if_not_applying_to_market():
    """Verify LOT_SIZE is rejected if applyToMarket is False."""
    raw_filters = [
        {
            "filterType": "LOT_SIZE",
            "minQty": "0.001",
            "maxQty": "500.0",
            "stepSize": "0.001",
            "applyToMarket": False,
        },
        {
            "filterType": "MIN_NOTIONAL",
            "notional": "5.0",
        },
    ]

    with pytest.raises(FilterValidationError) as exc:
        parse_symbol_filters(raw_filters)
    assert "does not apply to market orders" in str(exc.value)


def test_strictest_notional_filter_enforcement():
    """Verify that when both MIN_NOTIONAL and NOTIONAL apply, the strictest (highest) is enforced."""
    raw_filters = [
        {
            "filterType": "MARKET_LOT_SIZE",
            "minQty": "0.001",
            "maxQty": "100.0",
            "stepSize": "0.001",
        },
        {
            "filterType": "MIN_NOTIONAL",
            "notional": "5.0",
            "applyToMarket": True,
        },
        {
            "filterType": "NOTIONAL",
            "minNotional": "10.0",
            "applyMinToMarket": True,
        },
    ]

    parsed = parse_symbol_filters(raw_filters)
    assert parsed.min_notional == Decimal("10.0")


def test_malformed_and_duplicate_filter_rejection():
    """Verify non-numeric values, zero stepSize, or duplicate filter types raise FilterValidationError."""
    # 1. Non-numeric stepSize
    with pytest.raises(FilterValidationError):
        parse_symbol_filters([
            {"filterType": "MARKET_LOT_SIZE", "stepSize": "INVALID", "minQty": "0.001", "maxQty": "100.0"},
            {"filterType": "MIN_NOTIONAL", "notional": "5.0"},
        ])

    # 2. Zero stepSize
    with pytest.raises(FilterValidationError):
        parse_symbol_filters([
            {"filterType": "MARKET_LOT_SIZE", "stepSize": "0", "minQty": "0.001", "maxQty": "100.0"},
            {"filterType": "MIN_NOTIONAL", "notional": "5.0"},
        ])

    # 3. Duplicate MARKET_LOT_SIZE
    with pytest.raises(FilterValidationError):
        parse_symbol_filters([
            {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "100.0"},
            {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.002", "minQty": "0.002", "maxQty": "100.0"},
            {"filterType": "MIN_NOTIONAL", "notional": "5.0"},
        ])


def test_downward_quantization_never_rounds_up():
    """Verify that quantize_downward strictly truncates downward and never rounds upward."""
    step_size = Decimal("0.001")

    # 0.0059 must become 0.005, NOT 0.006
    assert quantize_downward(Decimal("0.0059"), step_size) == Decimal("0.005")

    # 0.0051 must become 0.005
    assert quantize_downward(Decimal("0.0051"), step_size) == Decimal("0.005")

    # 0.0050 must stay 0.005
    assert quantize_downward(Decimal("0.0050"), step_size) == Decimal("0.005")

    # 0.0009 must become 0.000
    assert quantize_downward(Decimal("0.0009"), step_size) == Decimal("0.000")

    # Step size 0.01
    step_01 = Decimal("0.01")
    assert quantize_downward(Decimal("1.2399"), step_01) == Decimal("1.23")


def test_t12_residual_dust_detection():
    """Test T12: Verify detection of unclosable residual dust below minQty or minNotional."""
    filters = ParsedSymbolFilters(
        step_size=Decimal("0.001"),
        min_qty=Decimal("0.002"),
        max_qty=Decimal("100.0"),
        min_notional=Decimal("10.0"),
    )
    mark_price = Decimal("50000.0")

    # 1. Flat position (< 1e-8) is NOT residual dust
    assert check_residual_dust(Decimal("0.000000001"), mark_price, filters) is False
    assert check_residual_dust(Decimal("0.0"), mark_price, filters) is False

    # 2. Quantity below minQty (0.001 < 0.002) is residual dust
    assert check_residual_dust(Decimal("0.001"), mark_price, filters) is True

    # 3. Quantity meets minQty but notional below minNotional (e.g. mark_price = 1000, 0.002 * 1000 = 2.0 < 10.0)
    assert check_residual_dust(Decimal("0.002"), Decimal("1000.0"), filters) is True

    # 4. Valid closable position (0.002 * 50000 = 100.0 >= 10.0 and 0.002 >= 0.002) is NOT residual dust
    assert check_residual_dust(Decimal("0.002"), mark_price, filters) is False

