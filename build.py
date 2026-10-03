"""Build script for the FaceSort desktop app.

Two artifacts, one command::

    python build.py models     # stage the ONNX weights for packaging
    python build.py backend    # PyInstaller -> dist/backend/backend.exe
    python build.py electron   # electron-builder -> electron/release/*.exe
python build.py release    # copy the installers to release/ with checksums
    python build.py all        # everything above, in order
    python build.py clean      # remove build/ dist/ electron/release/

Why the weights are staged separately
-------------------------------------
The pipeline only ever needs the SCRFD detector and the ArcFace recogniser
out of ``buffalo_l`` (``det_10g.onnx`` + ``w600k_r50.onnx``, ~182 MB); the
landmark/gender models add ~140 MB and are never executed.  ``models``
copies exactly those two files into ``build/models/buffalo_l/`` so the same
staging directory feeds both packaging routes:

* ``backend --embed-models``  -> weights go *inside* the PyInstaller bundle,
  making ``backend.exe`` fully standalone.
* the Electron build          -> ``extraResources`` ships ``build/models``
  next to the backend and ``electron/main.js`` points the backend at it via
  ``FACEORG_MODEL_ROOT``, so the bundle stays ~180 MB smaller.

Neither route ever downloads anything at run time: the engine only reads
ONNX files that are already on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
ELECTRON_DIR = ROOT / "electron"

#: Everything PyInstaller-related lives under ``python_build/`` so the
#: layout electron-builder copies from is predictable:
#:
#:     python_build/models/buffalo_l/*.onnx   <- staged weights (build input)
#:     python_build/dist/backend/backend.exe  <- the engine, copied to resources/
#:     python_build/dist/backend/_internal/   <- its libs + the bundled weights
PYI_ROOT = ROOT / "python_build"
MODEL_STAGE = PYI_ROOT / "models" / "buffalo_l"
PYI_DIST = PYI_ROOT / "dist"
PYI_WORK = PYI_ROOT / "build"
BACKEND_DIR = PYI_DIST / "backend"          # onedir layout
BACKEND_EXE = BACKEND_DIR / "backend.exe"
BACKEND_SPEC = ROOT / "backend.spec"

#: Only these two buffalo_l models are used by the pipeline (SPEC §14).
REQUIRED_MODELS = ("det_10g.onnx", "w600k_r50.onnx")

IS_WINDOWS = os.name == "nt"


def python() -> str:
    """Interpreter that owns the virtualenv with our dependencies."""
    if VENV_PYTHON.is_file():
        return str(VENV_PYTHON)
    if shutil.which("python"):
        return "python"
    raise SystemExit("No Python interpreter found (expected .venv).")


def npm() -> str:
    """Locate npm: PATH first, then the standard Windows install location."""
    found = shutil.which("npm")
    if found:
        return found
    for candidate in (r"C:\Program Files\nodejs\npm.cmd",
                      r"C:\Program Files (x86)\nodejs\npm.cmd"):
        if Path(candidate).is_file():
            return candidate
    raise SystemExit(
        "npm was not found. Install Node.js 20+ (https://nodejs.org) so the "
        "Electron desktop app can be built."
    )


def run(command: Sequence[str], cwd: Optional[Path] = None) -> int:
    """Run a subprocess, streaming its output, and return its exit code."""
    printable = " ".join(str(part) for part in command)
    print(f"\n$ {printable}\n", flush=True)
    completed = subprocess.run([str(part) for part in command],
                               cwd=str(cwd or ROOT))
    return completed.returncode


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------
def stage_models() -> int:
    """Copy the ONNX weights the pipeline needs into ``build/models``."""
    sys.path.insert(0, str(ROOT))
    try:
        from src.face_model import model_dir, models_installed
    except ImportError as exc:  # pragma: no cover
        print(f"Cannot import src.face_model: {exc}", file=sys.stderr)
        return 2

    if not models_installed():
        print(f"No buffalo_l weights found at {model_dir()}.\n"
              "Fetch them once with:  python -m src.main --fetch-models",
              file=sys.stderr)
        return 2

    source = model_dir()
    MODEL_STAGE.mkdir(parents=True, exist_ok=True)
    missing = []
    for name in REQUIRED_MODELS:
        origin = source / name
        if not origin.is_file():
            missing.append(name)
            continue
        target = MODEL_STAGE / name
        if not target.is_file() or target.stat().st_size != origin.stat().st_size:
            shutil.copy2(origin, target)
        print(f"  {name:<16} {origin.stat().st_size / 1e6:7.1f} MB")
    if missing:
        print(f"Missing from {source}: {', '.join(missing)}", file=sys.stderr)
        return 2

    total = sum(path.stat().st_size for path in MODEL_STAGE.glob("*.onnx"))
    print(f"Staged {len(list(MODEL_STAGE.glob('*.onnx')))} model(s), "
          f"{total / 1e6:.1f} MB in {MODEL_STAGE}")
    return 0


def build_backend(onefile: bool = False, embed_models: bool = True,
                  clean: bool = False) -> int:
    """Run PyInstaller on ``backend.spec`` into ``python_build/dist``.

    The command is the spec plus the two global path options::

        pyinstaller --noconfirm \\
            --distpath python_build/dist --workpath python_build/build \\
            backend.spec

    The spec adds the staged ONNX weights itself (its tuple form of
    ``--add-data``, which needs no ``;``/``:`` separator difference).
    """
    if not VENV_PYTHON.is_file():
        print("PyInstaller lives in .venv — create it and "
              "`pip install -r requirements.txt` first.", file=sys.stderr)
        return 2

    env = dict(os.environ)
    env["FACEORG_ONEFILE"] = "1" if onefile else "0"
    env["FACEORG_EMBED_MODELS"] = "1" if embed_models else "0"
    if embed_models and not any(MODEL_STAGE.glob("*.onnx")):
        print("No staged weights found; running the models step first.")
        code = stage_models()
        if code:
            return code

    command: List[str] = [python(), "-m", "PyInstaller", "--noconfirm",
                          "--distpath", str(PYI_DIST),
                          "--workpath", str(PYI_WORK)]
    if clean:
        command.append("--clean")
    command.append(str(BACKEND_SPEC))

    # Show the equivalent --add-data form (the separator differs per OS) so
    # it is obvious where the weights come from when reading build logs.
    separator = ";" if IS_WINDOWS else ":"
    print("=" * 70)
    print(f"STEP 1/2  PyInstaller  (onefile={onefile}, models={embed_models})")
    print(f"  models  : {MODEL_STAGE}  ->  models/buffalo_l  "
          f"(equivalent: --add-data \"{MODEL_STAGE}{separator}models/buffalo_l\")")
    print(f"  distpath: {PYI_DIST}")
    print(f"  workpath: {PYI_WORK}")
    print("=" * 70)
    completed = subprocess.run(command, cwd=str(ROOT), env=env)
    if completed.returncode != 0:
        print(f"\nPyInstaller FAILED (exit {completed.returncode}). "
              f"Fix the errors above and re-run.", file=sys.stderr)
        return completed.returncode

    produced = PYI_DIST / "backend.exe" if onefile else BACKEND_EXE
    if not produced.is_file():
        print(f"\nPyInstaller reported success but {produced} is missing.",
              file=sys.stderr)
        return 1
    print(f"\nBackend built: {produced} "
          f"({_tree_size(produced if onefile else BACKEND_DIR) / 1e6:.0f} MB)")
    return 0


def build_electron(installer: bool = True, skip_install: bool = False) -> int:
    """Install npm dependencies and produce the distributable."""
    print("=" * 70)
    print("STEP 2/2  electron-builder")
    print("=" * 70)
    if not (ELECTRON_DIR / "package.json").is_file():
        print(f"No Electron project at {ELECTRON_DIR}.", file=sys.stderr)
        return 2
    if not BACKEND_EXE.is_file():
        print(f"Python backend missing: {BACKEND_EXE}\n"
              "Run the PyInstaller step first:  python build.py backend",
              file=sys.stderr)
        return 2
    print(f"  extraResources source: {BACKEND_DIR}  ->  <app>/resources/")
    print(f"  expected packaged path: <app>/resources/backend.exe")

    if not skip_install:
        if not (ELECTRON_DIR / "node_modules").is_dir():
            if (ELECTRON_DIR / "package-lock.json").is_file():
                code = run([npm(), "ci"], cwd=ELECTRON_DIR)
            else:
                code = run([npm(), "install"], cwd=ELECTRON_DIR)
            if code:
                print("\nnpm install FAILED — the Electron app needs its "
                      "dependencies before it can be packaged.", file=sys.stderr)
                return code

    command = [npm(), "run", "dist" if installer else "dist:dir"]
    code = run(command, cwd=ELECTRON_DIR)
    if code:
        print(f"\nelectron-builder FAILED (exit {code}).", file=sys.stderr)
    return code


def stage_release() -> int:
    """Collect the built artifacts into ``release/`` at the repo root.

    electron-builder writes to ``electron/release/``; this gathers the two
    things a person actually wants to test-install — the NSIS installer and the
    portable exe — plus checksums and a note, into one obvious folder.

    It copies rather than moves, so ``electron/release/`` stays as
    electron-builder expects and a rebuild does not lose anything.
    """
    print("=" * 70)
    print("STAGE  release/")
    print("=" * 70)

    source = ELECTRON_DIR / "release"
    if not source.is_dir():
        print(f"Nothing to collect: {source} does not exist.\n"
              "Run 'python build.py electron' first.", file=sys.stderr)
        return 2

    wanted = sorted(source.glob("*.exe"))
    if not wanted:
        print(f"No .exe artifacts in {source}.\n"
              "Run 'python build.py electron' first.", file=sys.stderr)
        return 2

    target = ROOT / "release"
    target.mkdir(parents=True, exist_ok=True)
    copied: List[tuple] = []
    for artifact in wanted:
        # Skip electron-builder's own scratch output: it is an unpacked tree,
        # not something you install.
        if artifact.name.endswith(".__uninstaller.exe"):
            continue
        destination = target / artifact.name
        print(f"  copying {artifact.name} ({artifact.stat().st_size / 1e6:.0f} MB)")
        shutil.copy2(artifact, destination)
        copied.append(destination)

    if not copied:
        print("Only the uninstaller stub was found; the build may have failed.",
              file=sys.stderr)
        return 2

    # Checksums, so a downloaded copy can be verified against the original.
    lines = []
    for path in copied:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
        print(f"  sha256 {path.name}: {digest[:16]}...")
    (target / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")

    (target / "README.txt").write_text(_release_readme(copied), encoding="utf-8")

    print()
    print(f"Release staged in {target}")
    for path in copied:
        print(f"  {path.name:<38} {path.stat().st_size / 1e6:>7.0f} MB")
    print("  SHA256SUMS.txt")
    print("  README.txt")
    return 0


def _release_readme(artifacts: List[Path]) -> str:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    names = "\n".join(f"  - {path.name}  ({path.stat().st_size / 1e6:.0f} MB)"
                      for path in artifacts)
    # Name the real installer in the checksum hint. Hardcoding a filename here
    # is how this file came to tell people to verify a FaceSort exe after the
    # product was renamed.
    installer = next((p.name for p in artifacts if "Setup" in p.name),
                     "FaceFlow-Setup-1.0.0.exe")
    return f"""FaceFlow - release build
Generated {generated}
{'=' * 60}

WHAT IS HERE
{names}
  - SHA256SUMS.txt   checksums, to confirm a copy is intact
  - README.txt       this file

HOW TO CHECK THAT INSTALLATION WORKS

1. The installer  (FaceFlow-Setup-*.exe)
   Double-click it and click through. It installs per-user, so no admin
   rights are needed and nothing goes to Program Files.
   After it finishes, launch FaceFlow from the Start menu.

   EXPECTED: a blue progress line in the terminal window it opened, ending
   with something like:
       [bridge] engine is ready

   IMPORTANT - these builds are NOT code-signed. Windows SmartScreen will
   show "Windows protected your PC" because it cannot verify an unknown
   publisher. Click "More info" then "Run anyway". This is expected for any
   self-built release, not a sign that the download is bad.

   Verify the copy is intact:
       certutil -hashfile "{installer}" SHA256
   and compare with the same line in SHA256SUMS.txt.

2. The portable exe  (FaceFlow-Portable-*.exe)
   Copy it anywhere and double-click it. No install, no registry, no
   uninstaller - delete the file when you are done.
   Use this to confirm the app itself works before trusting the installer.
   It unpacks itself to a temporary folder on first run, so give it a
   moment before concluding it has failed to start.

3. Where things go
   Two folders, for two different reasons:

   Profile   %APPDATA%\\FaceFlow         (the window's own settings -
                theme, recent folders. This path follows the app's name, so a
                profile from an earlier build is left behind unused.)
   Names DB  %LOCALAPPDATA%\\FaceSort    (the people you have named.
                Deliberately unchanged - renaming this path would orphan
                every remembered name already on disk.)

   Photos go wherever you point them. The app never modifies your originals
   unless you choose Move instead of Copy.

4. Uninstalling
   Settings > Apps > Installed apps > FaceFlow > Uninstall.
   The portable exe needs no uninstalling. To forget every remembered name,
   delete facesort_names.db from %LOCALAPPDATA%\\FaceSort.

IF SOMETHING GOES WRONG

  "The engine never became ready"
      The bundled Python engine failed to start. Re-run the app from a
      terminal to see its output, and check that antivirus has not quarantined
      resources\\_internal (a 500 MB folder of ONNX models looks unusual).

  "Face model is not installed"
      The ONNX weights did not unpack. Re-extract, or build from source with
      'python build.py all'.

  The app opens but the window is blank
      Press Ctrl+Shift+I for developer tools and read the console. A CSP or
      script error would appear there.

BUILDING THIS YOURSELF

    python build.py all          # models -> engine -> installer
    python build.py release      # collect the artifacts above
"""


def clean_everything() -> int:
    """Remove all build output (keeps node_modules, which is a big re-download)."""
    targets = [PYI_ROOT, ROOT / "build", ROOT / "dist", ELECTRON_DIR / "release",
               ELECTRON_DIR / "node_modules" / ".cache"]
    for target in targets:
        if target.exists():
            print(f"  removing {target.relative_to(ROOT)}")
            shutil.rmtree(target, ignore_errors=True)
    return 0


def _tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build.py",
        description="Build the FaceSort desktop app "
                    "(Electron shell + PyInstaller Python backend).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("target", choices=["models", "backend", "electron",
                                           "release", "all", "clean"],
                        help="which build step to run")
    parser.add_argument("--onefile", action="store_true",
                        help="backend: single self-extracting exe (slow start)")
    parser.add_argument("--no-models", dest="embed_models", action="store_false",
                        help="backend: do NOT bundle the ONNX weights "
                             "(they are included by default)")
    parser.set_defaults(embed_models=True)
    parser.add_argument("--clean", action="store_true",
                        help="backend: discard PyInstaller's cache first")
    parser.add_argument("--dir", dest="directory", action="store_true",
                        help="electron: unpacked app directory instead of an "
                             "NSIS installer")
    parser.add_argument("--skip-install", action="store_true",
                        help="electron: reuse the existing node_modules")
    args = parser.parse_args(argv)

    if args.target == "models":
        return stage_models()
    if args.target == "backend":
        return build_backend(args.onefile, args.embed_models, args.clean)
    if args.target == "electron":
        return build_electron(installer=not args.directory,
                              skip_install=args.skip_install)
    if args.target == "release":
        return stage_release()
    if args.target == "clean":
        return clean_everything()

    # all: models -> PyInstaller -> electron-builder
    for step in (stage_models,):
        code = step()
        if code:
            return code
    code = build_backend(args.onefile, embed_models=args.embed_models,
                         clean=args.clean)
    if code:
        return code
    return build_electron(installer=not args.directory,
                          skip_install=args.skip_install)


if __name__ == "__main__":
    sys.exit(main())
