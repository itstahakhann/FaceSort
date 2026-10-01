# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build for the Python backend of the FaceSort app.

Produces ``backend.exe`` — the loopback API server that the Electron shell
spawns (see ``electron/main.js`` and ``src/api/server.py``).  Streamlit is
*not* part of this build: the desktop UI is Electron, so the bundle only
carries the engine (insightface/ONNX Runtime/OpenCV/scikit-learn/FastAPI).

Build it with the wrapper instead of calling PyInstaller by hand::

    python build.py backend                 # dist/backend/backend.exe (onedir)
    python build.py backend --onefile       # dist/backend.exe
    python build.py backend --embed-models  # also bundle the ONNX weights

Models
------
The two ONNX graphs the pipeline actually runs (``det_10g.onnx`` and
``w600k_r50.onnx``, staged by ``python build.py models``) are added to
``models/buffalo_l/`` **inside** the bundle, so ``backend.exe`` is fully
self-contained and never downloads anything. ``src.face_model`` resolves
``sys._MEIPASS`` first in a frozen build, so the bundled weights are found
automatically. Set ``FACEORG_EMBED_MODELS=0`` to leave them out (the engine
then needs a model root, e.g. ``FACEORG_MODEL_ROOT``).

This is the spec-file equivalent of the command-line form below. The only
difference is that ``datas`` uses tuples and therefore needs no OS-specific
separator, while ``--add-data`` uses ``;`` on Windows and ``:`` elsewhere::

    # Windows
    pyinstaller --add-data "python_build\\models\\buffalo_l;models/buffalo_l"
    # macOS / Linux
    pyinstaller --add-data "python_build/models/buffalo_l:models/buffalo_l"

``build.py backend`` runs exactly this spec with ``--distpath
python_build/dist --workpath python_build/build``. The ONEDIR layout is the
default because electron-builder copies that folder's *contents* straight
into the app's ``resources/`` folder, which is where ``electron/main.js``
expects to find ``resources/backend.exe``.

ONE- vs ONEDIR
--------------
``FACEORG_ONEFILE=1`` produces a single self-extracting exe. It is only useful
for standalone/scripted use: it unpacks ~360 MB to ``%TEMP%`` on every launch,
so the desktop app always uses the onedir build.
"""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files

# PyInstaller sets SPECPATH to the directory holding this file (repo root).
ROOT = Path(os.path.abspath(SPECPATH))
if not (ROOT / "src" / "api" / "server.py").is_file():
    raise SystemExit(
        f"backend.spec expects the repository layout (src/api/server.py) "
        f"next to it, but looked in {ROOT}."
    )

# Models are bundled by default; FACEORG_EMBED_MODELS=0 opts out.
EMBED_MODELS = os.environ.get("FACEORG_EMBED_MODELS", "1") == "1"
ONEFILE = os.environ.get("FACEORG_ONEFILE", "0") == "1"
MODEL_STAGE = ROOT / "python_build" / "models" / "buffalo_l"

# --------------------------------------------------------------------------
# data files
# --------------------------------------------------------------------------
datas = [(str(ROOT / "config.yaml"), ".")]

if EMBED_MODELS:
    if not MODEL_STAGE.is_dir():
        raise SystemExit(
            f"No staged weights in {MODEL_STAGE}.\n"
            "Run 'python build.py models' first (it copies det_10g.onnx and "
            "w600k_r50.onnx out of the local insightface cache), or set "
            "FACEORG_EMBED_MODELS=0 to build without them."
        )
    onnx = sorted(str(path) for path in MODEL_STAGE.glob("*.onnx"))
    if not onnx:
        raise SystemExit(f"No .onnx files found in {MODEL_STAGE}.")
    datas += [(path, "models/buffalo_l") for path in onnx]
    print(f"[backend.spec] bundling {len(onnx)} model file(s) from {MODEL_STAGE}")

# --------------------------------------------------------------------------
# hidden imports — third-party packages whose modules are imported lazily or
# by string, which the static analyser cannot see.
# --------------------------------------------------------------------------
datas += collect_data_files("insightface", include_py_files=False)
datas += collect_data_files("PIL", include_py_files=False)

hiddenimports = [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "anyio._backends._asyncio",
    "insightface.app",
    "insightface.app.common",
    "insightface.model_zoo",
    "insightface.utils.storage",
    "insightface.utils.face_align",
    "sklearn.cluster",
    "sklearn.cluster._dbscan",
    "sklearn.metrics.pairwise",
    "scipy.spatial.distance",
    "PIL.Image",
    "PIL.ImageOps",
    "yaml",
    "multipart",
]

for package in ("uvicorn", "starlette", "fastapi", "onnxruntime", "sklearn"):
    try:
        for module in collect_all(package)[2]:
            if module not in hiddenimports:
                hiddenimports.append(module)
    except Exception:  # pragma: no cover - optional sub-dependency
        pass

# The Electron shell is the UI: nothing from the legacy Streamlit app (or the
# other big optional desktop stacks) belongs in the engine bundle.
excludes = [
    "streamlit", "altair", "pyarrow", "pandas", "matplotlib", "tkinter",
    "PyQt5", "PyQt6", "PySide2", "PySide6", "IPython", "pytest", "_pytest",
    "notebook", "sphinx",
]

block_cipher = None

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if ONEFILE:
    exe = EXE(
        pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [],
        name="backend",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        runtime_tmpdir=None,
        console=True,          # the app shows the engine log; stdout is the port
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
else:
    exe = EXE(
        pyz, a.scripts, [],
        exclude_binaries=True,
        name="backend",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe, a.binaries, a.zipfiles, a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name="backend",
    )
