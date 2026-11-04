"""Persisted UI preferences (watchlist, selection, Testnet rule overrides).

Stored as plain JSON under %LOCALAPPDATA%/NOONE/ui_state.json. Contains no
credentials and no trading state — only presentation preferences and the
Testnet engine rule overrides applied at next engine start.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from noone.ui.paths import ui_state_path

_DEFAULTS: dict[str, Any] = {
    "watchlist": ["BTCUSDT", "ETHUSDT", "PEPEUSDT"],
    "selected_symbol": "BTCUSDT",
    "interval": "15m",
    "testnet_rules": {
        "leverage": 10,
        "margin_mode": "Cross",
        "position_mode": "One-way",
        "margin_allocation_fraction": 0.10,
        "risk_per_trade_pct": 0.5,
        "max_position_notional": 50000.0,
        "take_profit_pct": 2.0,
        "stop_loss_pct": 1.0,
        "max_daily_loss_pct": 0.03,
        "max_daily_profit_pct": 0.02,
    },
}


class UIStateStore:
    def __init__(self, path: Path | None = None):
        self._path = path or ui_state_path()
        self._lock = threading.Lock()
        self._data: dict[str, Any] = dict(_DEFAULTS)
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                merged = dict(_DEFAULTS)
                merged.update({k: raw[k] for k in _DEFAULTS if k in raw})
                if isinstance(raw.get("testnet_rules"), dict):
                    rules = dict(_DEFAULTS["testnet_rules"])
                    rules.update(raw["testnet_rules"])
                    merged["testnet_rules"] = rules
                self._data = merged
        except Exception:
            pass  # defaults are the honest fallback for UI preferences

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._data))

    def update(self, **changes: Any) -> dict[str, Any]:
        with self._lock:
            for key, value in changes.items():
                if key in _DEFAULTS:
                    self._data[key] = value
            self._save()
            return json.loads(json.dumps(self._data))

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        except Exception:
            pass
