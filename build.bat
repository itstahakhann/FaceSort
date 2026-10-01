@echo off
REM ==========================================================================
REM  FaceSort - master build script (Windows)
REM
REM  Runs the two build stages in order, failing fast with a clear message:
REM    STEP 1  PyInstaller   -> python_build\dist\backend\backend.exe
REM                             (with the InsightFace ONNX models bundled)
REM    STEP 2  electron-builder -> electron\release\*.exe
REM
REM  Usage:   build.bat            full build
REM           build.bat --dir      also build the unpacked app folder
REM           build.bat --clean    discard PyInstaller's cache first
REM ==========================================================================
setlocal EnableDelayedExpansion

cd /d "%~dp0"
set "SEP=;"
set "PROJECT=%CD%"

REM --- pick the interpreter that owns our dependencies -------------------------
set "PY="
if exist "%PROJECT%\.venv\Scripts\python.exe" set "PY=%PROJECT%\.venv\Scripts\python.exe"
if not defined PY (
  for /f "delims=" %%i in ('where python 2^>nul') do if not defined PY set "PY=%%i"
)
if not defined PY (
  echo [FATAL] No Python found. Create .venv and run: pip install -r requirements.txt
  exit /b 1
)

echo ==========================================================================
echo  FaceSort build
echo  project : %PROJECT%
echo  python   : %PY%
echo ==========================================================================

REM --- STEP 0: stage the ONNX weights the engine bundles ----------------------
echo.
echo [STEP 0/2] Staging InsightFace ONNX models
echo            python_build\models\buffalo_l
"%PY%" build.py models
if errorlevel 1 (
  echo [FAIL] STEP 0 - could not stage the models.
  echo        Run once with: %PY% -m src.main --fetch-models
  exit /b 1
)
echo [OK] STEP 0/2 models staged

REM --- STEP 1: PyInstaller ----------------------------------------------------
REM The spec bundles the weights itself; this is the equivalent command line:
REM   pyinstaller --noconfirm --distpath python_build\dist ^
REM               --workpath python_build\build ^
REM               --add-data "python_build\models\buffalo_l%SEP%models/buffalo_l" ^
REM               backend.spec
echo.
echo [STEP 1/2] PyInstaller - building the Python backend
if "%~1"=="--clean" (
  "%PY%" build.py backend --clean
) else if "%~1"=="--dir" (
  "%PY%" build.py backend
) else (
  "%PY%" build.py backend
)
if errorlevel 1 (
  echo [FAIL] STEP 1 - PyInstaller did not finish. See the log above.
  exit /b 1
)
if not exist "%PROJECT%\python_build\dist\backend\backend.exe" (
  echo [FAIL] STEP 1 - expected %PROJECT%\python_build\dist\backend\backend.exe
  echo        but it is missing.
  exit /b 1
)
for %%A in ("%PROJECT%\python_build\dist\backend\backend.exe") do echo [OK] STEP 1/2 backend.exe ready (%%~zA bytes)

REM --- STEP 2: electron-builder ------------------------------------------------
echo.
echo [STEP 2/2] electron-builder - packaging the desktop app
if "%~1"=="--dir" (
  call npm --prefix "%PROJECT%\electron" run dist:dir
) else (
  call npm --prefix "%PROJECT%\electron" run dist
)
if errorlevel 1 (
  echo [FAIL] STEP 2 - electron-builder failed. Common causes:
  echo        - no node_modules:  cd electron ^&^& npm install
  echo        - backend missing: run STEP 1 again
  exit /b 1
)

echo.
echo ==========================================================================
echo  [DONE] Build finished
echo    engine  : python_build\dist\backend\backend.exe
echo    output  : electron\release\
dir /b "%PROJECT%\electron\release\*.exe" 2>nul
echo ==========================================================================
endlocal
exit /b 0
