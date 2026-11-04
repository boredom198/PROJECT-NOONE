"""Filesystem paths for Component 6 (UI layer only).

Writable state lives under %LOCALAPPDATA%/NOONE (the same root Component 7
uses for DPAPI credential files). Bundled resources resolve through
sys._MEIPASS when frozen with PyInstaller.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def app_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home() / "AppData" / "Local"
    path = root / "NOONE"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ui_state_path() -> Path:
    return app_data_dir() / "ui_state.json"


def testnet_journal_path() -> Path:
    return app_data_dir() / "execution_journal_testnet.db"


def log_dir() -> Path:
    path = app_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resource_path(*parts: str) -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidate = Path(meipass, *parts)
        if candidate.exists():
            return candidate
    here = Path(__file__).resolve()
    candidates = [
        here.parent.joinpath(*parts),
        here.parents[3].joinpath(*parts),
        here.parents[2].joinpath(*parts),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def frontend_dist_dir() -> Path:
    bundled = resource_path("static_dist")
    if (bundled / "index.html").exists():
        return bundled
    return Path(__file__).resolve().parents[3] / "frontend" / "dist"


def default_model_path() -> Path:
    repo = Path(__file__).resolve().parents[3]
    candidate = repo / "runs" / "stage2-realdata" / "final_model.zip"
    if candidate.exists():
        return candidate
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        bundled = Path(meipass) / "models" / "final_model.zip"
        if bundled.exists():
            return bundled
    return candidate
