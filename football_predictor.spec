# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_all

# Gyűjtsd össze a necesszáris modulokat
datas = []
binaries = []
hiddenimports = []

# Scikit-learn
hiddenimports += ['sklearn.neighbors.typedefs']

# XGBoost
hiddenimports += ['xgboost']

# Torch
hiddenimports += ['torch']

# Pandas és NumPy szükséges importok
hiddenimports += ['pandas', 'numpy']

# Flask és CORS importok
hiddenimports += ['flask', 'flask_cors']

# Egyéni modulok
hiddenimports += ['predict']

# Modellek és adatok
datas += [('models/', 'models/')]
datas += [('data/', 'data/')]
datas += [('predict/', 'predict/')]

a = Analysis(
    ['football_predictor_backend.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='football_predictor_backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    name='football_predictor_backend',
)
