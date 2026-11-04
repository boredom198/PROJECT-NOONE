"""Unit tests for Component 5 — TradingView Charts embed widget generator.

Tests symbol mapping, interval mapping, input validation, and generated
pre-made TradingView Advanced Chart widget HTML configuration.
"""
from __future__ import annotations

import json
import re
import pytest

from noone.chart.widget import (
    DEFAULT_INTERVAL,
    DEFAULT_SYMBOL,
    DEFAULT_THEME,
    SUPPORTED_INTERVAL_MAP,
    SUPPORTED_THEMES,
    generate_tradingview_embed_html,
    to_tradingview_interval,
    to_tradingview_symbol,
)


class TestTradingViewSymbolMapping:
    """Test symbol conversion from NOONE formats to TradingView Binance Futures tickers."""

    @pytest.mark.parametrize(
        "input_symbol,expected",
        [
            ("BTCUSDT", "BINANCE:BTCUSDT.P"),
            ("btcusdt", "BINANCE:BTCUSDT.P"),
            ("BTC/USDT", "BINANCE:BTCUSDT.P"),
            ("btc/usdt", "BINANCE:BTCUSDT.P"),
            ("BTC-USDT", "BINANCE:BTCUSDT.P"),
            ("BTC_USDT", "BINANCE:BTCUSDT.P"),
            ("BTC/USDT:USDT", "BINANCE:BTCUSDT.P"),
            ("ETHUSDT", "BINANCE:ETHUSDT.P"),
            ("ethusdt", "BINANCE:ETHUSDT.P"),
            ("ETH/USDT", "BINANCE:ETHUSDT.P"),
            ("SOLUSDT", "BINANCE:SOLUSDT.P"),
            ("DOGEUSDT", "BINANCE:DOGEUSDT.P"),
            ("1000PEPEUSDT", "BINANCE:1000PEPEUSDT.P"),
            ("BINANCE:BTCUSDT.P", "BINANCE:BTCUSDT.P"),
            ("BINANCE:BTCUSDT.PERP", "BINANCE:BTCUSDT.P"),
        ],
    )
    def test_valid_symbol_conversion(self, input_symbol: str, expected: str):
        assert to_tradingview_symbol(input_symbol) == expected

    @pytest.mark.parametrize(
        "invalid_symbol",
        [
            "",
            "   ",
            "BTC",
            "USDT",
            "BTCBUSD",
            "ETHBTC",
            "BTC/EUR",
            "BTC$",
            "!!!",
            "123",
        ],
    )
    def test_invalid_symbol_rejected(self, invalid_symbol: str):
        with pytest.raises(ValueError, match="Unsupported symbol|Invalid cryptocurrency symbol|Symbol cannot be empty"):
            to_tradingview_symbol(invalid_symbol)

    @pytest.mark.parametrize("non_string", [None, 123, 45.6, [], {}])
    def test_non_string_symbol_rejected(self, non_string):
        with pytest.raises(ValueError, match="Symbol must be a string"):
            to_tradingview_symbol(non_string)  # type: ignore[arg-type]


class TestTradingViewIntervalMapping:
    """Test timeframe conversion from NOONE intervals to TradingView intervals."""

    @pytest.mark.parametrize(
        "noone_interval,expected_tv_interval",
        [
            ("1m", "1"),
            ("3m", "3"),
            ("5m", "5"),
            ("15m", "15"),
            ("30m", "30"),
            ("1h", "60"),
            ("2h", "120"),
            ("4h", "240"),
            ("6h", "360"),
            ("8h", "480"),
            ("12h", "720"),
            ("1d", "D"),
            ("3d", "3D"),
            ("1w", "W"),
        ],
    )
    def test_supported_interval_mapping(self, noone_interval: str, expected_tv_interval: str):
        assert to_tradingview_interval(noone_interval) == expected_tv_interval

    def test_case_and_whitespace_insensitivity(self):
        assert to_tradingview_interval(" 15M ") == "15"
        assert to_tradingview_interval("1H") == "60"
        assert to_tradingview_interval("1D") == "D"

    @pytest.mark.parametrize("unsupported_interval", ["2m", "10m", "45m", "2d", "1M", "invalid", ""])
    def test_unsupported_interval_rejected(self, unsupported_interval: str):
        with pytest.raises(ValueError, match="Unsupported interval"):
            to_tradingview_interval(unsupported_interval)

    @pytest.mark.parametrize("non_string", [None, 15, 60, [], {}])
    def test_non_string_interval_rejected(self, non_string):
        with pytest.raises(ValueError, match="Interval must be a string"):
            to_tradingview_interval(non_string)  # type: ignore[arg-type]


class TestTradingViewEmbedGeneration:
    """Test generated pre-made TradingView embed HTML and JSON configuration."""

    def test_default_embed_html(self):
        html = generate_tradingview_embed_html()
        assert "<!-- TradingView Widget BEGIN -->" in html
        assert '<div class="tradingview-widget-container"' in html
        assert "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" in html

        # Extract and parse the embedded JSON config
        match = re.search(r"<script[^>]*>\s*(\{.*?\})\s*</script>", html, re.DOTALL)
        assert match is not None, "Failed to locate JSON config script block"
        config = json.loads(match.group(1))

        assert config["symbol"] == "BINANCE:BTCUSDT.P"
        assert config["interval"] == "15"
        assert config["theme"] == "dark"
        assert config["timezone"] == "Etc/UTC"
        assert config["style"] == "1"
        assert config["autosize"] is True
        assert config["allow_symbol_change"] is True
        assert config["enable_publishing"] is False
        assert config["support_host"] == "https://www.tradingview.com"

    def test_custom_embed_configuration(self):
        html = generate_tradingview_embed_html(
            symbol="ETH/USDT",
            interval="1h",
            theme="light",
            autosize=False,
            timezone="America/New_York",
            allow_symbol_change=False,
        )
        match = re.search(r"<script[^>]*>\s*(\{.*?\})\s*</script>", html, re.DOTALL)
        assert match is not None
        config = json.loads(match.group(1))

        assert config["symbol"] == "BINANCE:ETHUSDT.P"
        assert config["interval"] == "60"
        assert config["theme"] == "light"
        assert config["autosize"] is False
        assert config["timezone"] == "America/New_York"
        assert config["allow_symbol_change"] is False

    @pytest.mark.parametrize("invalid_theme", ["blue", "neon", "solarized", ""])
    def test_invalid_theme_rejected(self, invalid_theme: str):
        with pytest.raises(ValueError, match="Invalid theme"):
            generate_tradingview_embed_html(theme=invalid_theme)

    @pytest.mark.parametrize("invalid_timezone", ["", "   "])
    def test_invalid_timezone_rejected(self, invalid_timezone: str):
        with pytest.raises(ValueError, match="Timezone must be a non-empty string"):
            generate_tradingview_embed_html(timezone=invalid_timezone)

    @pytest.mark.parametrize("invalid_locale", ["", "   "])
    def test_invalid_locale_rejected(self, invalid_locale: str):
        with pytest.raises(ValueError, match="Locale must be a non-empty string"):
            generate_tradingview_embed_html(locale=invalid_locale)

    @pytest.mark.parametrize("invalid_style", ["", "   "])
    def test_invalid_style_rejected(self, invalid_style: str):
        with pytest.raises(ValueError, match="Style must be a non-empty string"):
            generate_tradingview_embed_html(style=invalid_style)


class TestDependencyFreeGuarantee:
    """Verify that Component 5 has zero non-standard-library runtime dependencies."""

    def test_noone_chart_widget_imports(self):
        import sys
        import noone.chart.widget as widget_module

        # Inspect module globals and imported modules
        # widget.py must only import from json, re, typing, __future__
        allowed_stdlib = {"json", "re", "typing", "__future__", "annotations"}
        for name, val in widget_module.__dict__.items():
            if hasattr(val, "__name__") and hasattr(val, "__file__"):
                mod_name = val.__name__.split(".")[0]
                if mod_name not in allowed_stdlib and mod_name != "noone":
                    pytest.fail(f"Unauthorized external dependency imported: {mod_name}")

