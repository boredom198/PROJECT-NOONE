# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import os
from PyInstaller.utils.hooks import collect_all

block_cipher = None

repo_root = Path.cwd()
frontend_dist = repo_root / "frontend" / "dist"
model_file = repo_root / "runs" / "stage2-realdata" / "final_model.zip"

datas = [
    (str(frontend_dist), "static_dist"),
]
if model_file.exists():
    datas.append((str(model_file), "models"))

wv_datas, wv_binaries, wv_hidden = collect_all("webview")
pn_datas, pn_binaries, pn_hidden = collect_all("pythonnet")

datas += wv_datas + pn_datas

a = Analysis(
    [str(repo_root / "src" / "noone" / "ui" / "launcher.py")],
    pathex=[str(repo_root / "src")],
    binaries=wv_binaries + pn_binaries,
    datas=datas,
    hiddenimports=[
        "webview",
        "clr",
        "pythonnet",
        "noone",
        "noone.ui",
        "noone.ui.launcher",
        "noone.ui.server",
        "noone.ui.services",
        "noone.ui.state_store",
        "noone.ui.paths",
        "noone.security",
        "noone.security.dpapi",
        "noone.security.credentials",
        "noone.chart",
        "noone.chart.widget",
        "noone.market_data",
        "noone.market_data.binance_futures",
        "noone.market_data.models",
    ] + wv_hidden + pn_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NOONE",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="NOONE",
)

