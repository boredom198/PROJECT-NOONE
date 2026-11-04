"""PROJECT NOONE — Component 5 (TradingView Charts).

Provides official pre-made TradingView Advanced Chart embed widget generation
for Binance USDⓈ-M Futures without custom chart UI, renderer, or dependencies.
"""
from __future__ import annotations

import json
import re
from typing import Any

# Supported intervals aligned with Binance Futures and TradingView notation
SUPPORTED_INTERVAL_MAP: dict[str, str] = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "6h": "360",
    "8h": "480",
    "12h": "720",
    "1d": "D",
    "3d": "3D",
    "1w": "W",
}

SUPPORTED_THEMES: frozenset[str] = frozenset({"dark", "light"})

DEFAULT_SYMBOL: str = "BTCUSDT"
DEFAULT_INTERVAL: str = "15m"
DEFAULT_THEME: str = "dark"
DEFAULT_TIMEZONE: str = "Etc/UTC"

# Regex for base cryptocurrency ticker (e.g. BTC, ETH, SOL, DOGE, 1000PEPE)
_BASE_REGEX = re.compile(r"^[A-Z0-9]{2,15}$")


def to_tradingview_symbol(symbol: str) -> str:
    """Convert an internal NOONE Binance Futures pair to TradingView ticker format.

    Examples:
        'BTCUSDT' -> 'BINANCE:BTCUSDT.P'
        'btcusdt' -> 'BINANCE:BTCUSDT.P'
        'BTC/USDT' -> 'BINANCE:BTCUSDT.P'
        'BTC/USDT:USDT' -> 'BINANCE:BTCUSDT.P'
        'BINANCE:BTCUSDT.P' -> 'BINANCE:BTCUSDT.P'

    Raises:
        ValueError: If symbol is empty, not a string, or not a valid USDT perpetual pair.
    """
    if not isinstance(symbol, str):
        raise ValueError(f"Symbol must be a string, got {type(symbol).__name__}")

    raw = symbol.strip().upper()
    if not raw:
        raise ValueError("Symbol cannot be empty.")

    # Idempotent handling if already fully qualified TradingView ticker
    if raw.startswith("BINANCE:") and (raw.endswith(".P") or raw.endswith(".PERP")):
        return raw if raw.endswith(".P") else raw[:-5] + ".P"

    # Normalize common delimiter forms: 'BTC/USDT:USDT' -> 'BTCUSDT', 'BTC/USDT' -> 'BTCUSDT'
    clean = raw.split(":")[0]  # drop CCXT settlement suffix like ':USDT'
    clean = clean.replace("/", "").replace("-", "").replace("_", "")

    if not clean.endswith("USDT"):
        raise ValueError(
            f"Unsupported symbol {symbol!r}: only USDT-margined perpetuals are supported."
        )

    base = clean[:-4]
    if not base or not _BASE_REGEX.match(base):
        raise ValueError(f"Invalid cryptocurrency symbol {symbol!r}: malformed base asset.")

    return f"BINANCE:{base}USDT.P"


def to_tradingview_interval(interval: str) -> str:
    """Map internal NOONE timeframe string to TradingView widget interval string.

    Examples:
        '15m' -> '15'
        '1h'  -> '60'
        '1d'  -> 'D'

    Raises:
        ValueError: If the interval is not supported.
    """
    if not isinstance(interval, str):
        raise ValueError(f"Interval must be a string, got {type(interval).__name__}")

    raw = interval.strip()
    # In crypto APIs, '1M' denotes 1 Month (calendar interval), which is
    # excluded in NOONE. Distinguish '1M' from '1m' (1 Minute).
    if raw == "1M":
        raise ValueError(
            f"Unsupported interval '1M': month calendar interval is not supported. "
            f"Supported intervals: {sorted(SUPPORTED_INTERVAL_MAP.keys())}"
        )

    key = raw.lower()
    if key not in SUPPORTED_INTERVAL_MAP:
        raise ValueError(
            f"Unsupported interval {interval!r}. Supported intervals: "
            f"{sorted(SUPPORTED_INTERVAL_MAP.keys())}"
        )

    return SUPPORTED_INTERVAL_MAP[key]


def generate_tradingview_embed_html(
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    theme: str = DEFAULT_THEME,
    autosize: bool = True,
    timezone: str = DEFAULT_TIMEZONE,
    allow_symbol_change: bool = True,
    style: str = "1",
    locale: str = "en",
) -> str:
    """Generate official pre-made TradingView Advanced Chart embed HTML.

    Args:
        symbol: Market symbol (e.g. 'BTCUSDT', 'BTC/USDT').
        interval: Timeframe (e.g. '15m', '1h', '1d').
        theme: UI theme ('dark' or 'light').
        autosize: Responsive sizing flag.
        timezone: Chart timezone (default 'Etc/UTC').
        allow_symbol_change: Whether user can switch symbols in the widget.
        style: Chart style ('1' = Candlesticks).
        locale: Language locale (default 'en').

    Returns:
        String containing complete TradingView widget container and embed script.

    Raises:
        ValueError: If any parameter fails validation.
    """
    tv_symbol = to_tradingview_symbol(symbol)
    tv_interval = to_tradingview_interval(interval)

    if not isinstance(theme, str):
        raise ValueError(f"Theme must be a string, got {type(theme).__name__}")
    theme_clean = theme.strip().lower()
    if theme_clean not in SUPPORTED_THEMES:
        raise ValueError(
            f"Invalid theme {theme!r}. Supported themes: {sorted(SUPPORTED_THEMES)}"
        )

    if not isinstance(timezone, str) or not timezone.strip():
        raise ValueError("Timezone must be a non-empty string.")

    if not isinstance(locale, str) or not locale.strip():
        raise ValueError("Locale must be a non-empty string.")

    if not isinstance(style, str) or not style.strip():
        raise ValueError("Style must be a non-empty string.")

    widget_config: dict[str, Any] = {
        "autosize": bool(autosize),
        "symbol": tv_symbol,
        "interval": tv_interval,
        "timezone": timezone.strip(),
        "theme": theme_clean,
        "style": style.strip(),
        "locale": locale.strip(),
        "enable_publishing": False,
        "allow_symbol_change": bool(allow_symbol_change),
        "calendar": False,
        "support_host": "https://www.tradingview.com",
    }

    config_json = json.dumps(widget_config, indent=2)

    return (
        '<!-- TradingView Widget BEGIN -->\n'
        '<div class="tradingview-widget-container" style="height:100%;width:100%">\n'
        '  <div class="tradingview-widget-container__widget" style="height:calc(100% - 32px);width:100%"></div>\n'
        '  <div class="tradingview-widget-copyright"><a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank"><span class="blue-text">Track all markets on TradingView</span></a></div>\n'
        '  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>\n'
        f'{config_json}\n'
        '  </script>\n'
        '</div>\n'
        '<!-- TradingView Widget END -->'
    )
