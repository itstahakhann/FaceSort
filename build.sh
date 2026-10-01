#!/usr/bin/env bash
# ===========================================================================
#  FaceSort - master build script (macOS / Linux)
#
#  Runs the two build stages in order, failing fast with a clear message:
#    STEP 1  PyInstaller      -> python_build/dist/backend/backend
#                              (with the InsightFace ONNX models bundled)
#    STEP 2  electron-builder -> electron/release/*
#
#  Usage:   ./build.sh              full build
#           ./build.sh --dir        also build the unpacked app folder
#           ./build.sh --clean      discard PyInstaller's cache first
# ===========================================================================
set -euo pipefail

SEP=":"
PROJECT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT"

CLEAN=""
ELECTRON_SCRIPT="dist"
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN="--clean" ;;
    --dir)   ELECTRON_SCRIPT="dist:dir" ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

# --- pick the interpreter that owns our dependencies -------------------------
PY=""
if [ -x "$PROJECT/.venv/bin/python" ]; then
  PY="$PROJECT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  echo "[FATAL] No Python found. Create .venv and run: pip install -r requirements.txt" >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "[FATAL] npm not found. Install Node.js 20+ (https://nodejs.org)" >&2
  exit 1
fi

echo "=========================================================================="
echo " FaceSort build"
echo " project : $PROJECT"
echo " python  : $PY"
echo " node    : $(node --version)  npm $(npm --version)"
echo "=========================================================================="

# --- STEP 0: stage the ONNX weights the engine bundles ----------------------
echo
echo "[STEP 0/2] Staging InsightFace ONNX models -> python_build/models/buffalo_l"
"$PY" build.py models
echo "[OK] STEP 0/2 models staged"

# --- STEP 1: PyInstaller ----------------------------------------------------
# The spec bundles the weights itself; this is the equivalent command line:
#   pyinstaller --noconfirm --distpath python_build/dist \
#               --workpath python_build/build \
#               --add-data "python_build/models/buffalo_l:models/buffalo_l" \
#               backend.spec
echo
echo "[STEP 1/2] PyInstaller - building the Python backend"
if [ -n "$CLEAN" ]; then
  "$PY" build.py backend --clean
else
  "$PY" build.py backend
fi

ENGINE="$PROJECT/python_build/dist/backend/backend"
if [ ! -x "$ENGINE" ]; then
  echo "[FAIL] STEP 1 - expected $ENGINE but it is missing." >&2
  exit 1
fi
echo "[OK] STEP 1/2 engine ready ($ENGINE, $(du -sh "$(dirname "$ENGINE")" | cut -f1))"

# --- STEP 2: electron-builder -----------------------------------------------
echo
echo "[STEP 2/2] electron-builder - packaging the desktop app"
if [ ! -d "$PROJECT/electron/node_modules" ]; then
  if [ -f "$PROJECT/electron/package-lock.json" ]; then
    (cd "$PROJECT/electron" && npm ci)
  else
    (cd "$PROJECT/electron" && npm install)
  fi
fi
(cd "$PROJECT/electron" && npm run "$ELECTRON_SCRIPT")

echo
echo "=========================================================================="
echo " [DONE] Build finished"
echo "   engine : python_build/dist/backend/backend"
echo "   output : electron/release/"
ls -lh "$PROJECT"/electron/release/*.exe "$PROJECT"/electron/release/*.dmg \
       "$PROJECT"/electron/release/*.AppImage 2>/dev/null || true
echo "=========================================================================="
