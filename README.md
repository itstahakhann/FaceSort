# Face Grouping & Auto-Organizer

A local, **offline** desktop app that scans a folder of photos, detects every
face, groups photos of the same person together, lets you name each group, and
sorts the photos into named folders. Nothing ever leaves your machine.

The shipped desktop app is an **Electron shell around a Python engine**:

```
┌─────────────────────────────┐        ┌──────────────────────────────────┐
│  Electron (renderer)        │  HTTP  │  backend.exe  (PyInstaller)      │
│  sidebar · cluster grid     │ ─────► │  FastAPI on 127.0.0.1:<free>    │
│  main.js: spawn + lifecycle │  fetch │  scanner · detector · clusterer  │
│  preload.js: narrow bridge  │ ◄───── │  organizer · SQLite name DB     │
└─────────────────────────────┘  JSON  └──────────────────────────────────┘
        │  stdout: "PORT:8000"
        └────────────────────────► (main.js parses it, then polls /status)
```

A source checkout also keeps the interactive **CLI** (`python -m src.main`) as
the reference front-end, and the legacy Streamlit UI under `src/ui/`.

See [SPEC.md.md](SPEC.md.md) for the original specification.

## Features

Implemented (M1 + M2 + M3 + M4 + M5 + M6):

- Scans `jpg/jpeg/png/bmp/webp` and skips corrupt files without crashing
- Detects 0..N faces per photo and computes a 512-D embedding for each one
- Clusters embeddings (DBSCAN, cosine distance) so all photos of the same
  person land in the same group
- **Thumbnail preview (M2):** before each question a grid of cropped faces
  is rendered as a temp JPEG and opened in your default image viewer, so
  you see who the cluster is — each tile numbered for the commands below
- **Cluster review (§10):** `merge` two clusters, `split` one apart (by
  face number), `rename`, `list`, `show`, or `abort` — all typed at the
  same prompt, before anything is copied or moved
- **Persistent name database (SQLite):** people you have named before are
  recognised automatically on later runs — their clusters are labelled
  without a question, and only genuinely unknown faces are prompted
- Prints each cluster's photo paths and asks `Who is this? (name/skip)`
- Copies (or moves) photos into one folder per person; a photo with several
  named people appears in **all** of those folders; skipped groups go to
  `_unknown/`; clashing file names get an `_1`, `_2`, … suffix
- Summary report: photos scanned, faces detected, clusters, folders
  created, auto-labelled vs prompted, merges/splits, known people in the DB
- **Streamlit GUI (M4):** sidebar for input/output folders, tolerance and
  copy/move; a progress bar while scanning; one card per cluster with its
  face montage and a name input pre-filled from the database; a single
  **Run Organizer** button that does the sorting — all offline, and the
  scan is cached in the session so interacting with the page never
  re-runs the face pipeline
- **Electron desktop app (M5):** the shipped Windows app — a native-feeling
  window with a config sidebar, a grid of cluster cards with face thumbnails
  and per-group name inputs, a live progress bar, and one **Sort into
  folders** action; the Python engine runs as a hidden child process and
  the whole thing installs as a single `.exe` — see
  [Desktop app](#desktop-app-electron--python)
- **Local REST API (M5):** the engine behind the UI exposes `POST /scan`,
  `GET /clusters`, `POST /name_cluster`, `POST /organize` and `GET /status`
  on a free loopback port, announcing it with a `PORT:<port>` line on stdout
- **Multiprocess scanning (M6):** folders with 24+ photos are processed by
  a pool of worker processes (auto-sized to the CPU, `--workers` /
  `workers:` to override), each with a tightly bounded number of inference
  threads — see [Performance](#performance-m6)

## Requirements

**To use the app:** nothing. The Windows installer ships the Electron shell,
the Python engine and the ONNX weights; no Python, Node or model download is
needed on the target machine.

**To build or run from source:**

- Python 3.9+ (Windows, macOS, or Linux)
- Node.js 20+ and npm — only for the Electron app (`npm start`, `npm run dist`)
- CPU only — no GPU required

## Installation

```bash
git clone <repo-url> FaceSort
cd FaceSort

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

> **One-time setup:** download the InsightFace model weights (~300 MB) with
> `python -m src.main --fetch-models`. This is the only step that touches the
> network — once the weights are cached locally, all inference runs fully
> offline.

## Configuration

Defaults live in [`config.yaml`](config.yaml):

| Key | Default | Description |
|-----|---------|-------------|
| `input_folder` | `./input_photos` | Source of images |
| `output_folder` | `./grouped_photos` | Destination |
| `tolerance` | `0.5` | Clustering strictness |
| `min_faces_per_cluster` | `2` | Ignore tiny clusters |
| `mode` | `copy` | `copy` or `move` |
| `detector_backend` | `retinaface` | `retinaface` / `dlib_hog` / `dlib_cnn` |
| `unknown_folder` | `_unknown` | Folder for skipped clusters |
| `names_db` | `./facesort_names.db` | Remembered people (SQLite); empty disables auto-labelling. The default spelling resolves to `%LOCALAPPDATA%\FaceSort\facesort_names.db` (`~/Library/Application Support/…`, `$XDG_DATA_HOME/…`), **not** the working directory; give an explicit path to override, or set `FACEORG_DATA_DIR` |
| `workers` | `0` | Worker processes for the vision stage: `0` = auto (`cpu_count // 2`, max 4), `1` = single process |
| `use_eye_regions` | `true` | Extract a second periocular fingerprint per face (see [Age invariance](#age-invariance-childhood-adult-photos)); `false` restores whole-face-only clustering |
| `weight_full_face` | `0.4` | Whole-face share of the fused distance |
| `weight_eye_region` | `0.6` | Eye-region share of the fused distance (should sum to 1 with the above) |

Supported image formats: `jpg`, `jpeg`, `png`, `bmp`, `webp` (plus `heic`
optionally).

Environment variables (all optional):

| Variable | Purpose |
|----------|---------|
| `FACEORG_MODEL_ROOT` | Directory that contains `models/buffalo_l` (insightface cache layout). Only needed when running the engine from source against a custom model directory — a frozen `backend.exe` finds the weights it ships with on its own |
| `FACEORG_CONFIG` | Path to `config.yaml` for the engine/CLI (defaults to the bundled copy, then the repo's) |
| `INSIGHTFACE_HOME` | insightface's own model-root override (honoured after `FACEORG_MODEL_ROOT`) |

## Running the CLI

```bash
python -m src.main
```

The tool scans the input folder, clusters the faces, then walks you through
each group — it prints the photo paths and asks `Who is this? (name/skip)`:

```
[1/3] Cluster #0: 2 face(s) in 2 photo(s)
    ./input_photos/holiday_01.jpg
    ./input_photos/holiday_07.jpg
Who is this? (name/skip) Alice
```

Type a name to create `grouped_photos/<name>/`, or press Enter (or type
`skip` / `unknown`) to send the group to `grouped_photos/_unknown/`. Photos
containing more than one named person are copied into each of those folders.

### Remembering names (M3)

Names you type are stored in a local SQLite file (`names_db` in
`config.yaml`) together with the average face embedding and a few sample
photo paths. On the next run every cluster centroid is compared against the
database with the same cosine `tolerance` used for clustering, so familiar
people are labelled automatically and **only unknown faces are prompted**:

```
[auto] Cluster #0: Alice (distance 0.312 <= 0.50) e.g. ./photos/holiday_01.jpg

2 cluster(s) matched the name database and need no answer.

[1/1] Cluster #3: 4 face(s) in 4 photo(s)
Who is this? (name/skip)
```

- Skipped (`skip`/empty) answers are **never** saved to the database.
- Closest match wins when several known people are within `tolerance`
  (a warning is logged).
- Use `--no-db` to ignore the database for one run, or set
  `names_db:` to an empty value in `config.yaml` to disable it entirely.
- The database is a plain local file — delete it to forget everyone.
- **Where it lives:** the default (`./facesort_names.db`) is resolved to a
  per-user, always-writable folder rather than the working directory, because
  the engine is frozen *inside* the install directory — which is read-only on
  a normal Windows install:

  | Platform | Path |
  |----------|------|
  | Windows | `%LOCALAPPDATA%\FaceSort\facesort_names.db` |
  | macOS | `~/Library/Application Support/FaceSort/facesort_names.db` |
  | Linux | `$XDG_DATA_HOME/FaceSort/facesort_names.db` (else `~/.local/share/…`) |

  `FACEORG_DATA_DIR` overrides the folder, and any explicit `names_db:` path
  is used verbatim.

### Preview & cluster review (M2)

Every cluster is shown with a **thumbnail montage** first — a grid of the
cropped faces, numbered, saved as a temp JPEG and opened in your default
image viewer (F5). Use `--no-preview` if you would rather not open
windows; the prompt also prints the montage path so you can open it
yourself.

The same prompt accepts review commands, so a wrong DBSCAN result can be
fixed *before* anything is copied (§10, §12):

```
[1/3] Cluster #0: 4 face(s) in 4 photo(s)
    preview: C:\...\faceorg_preview_x1\cluster_0_1.jpg (opened in the image viewer)
    (1) ./input_photos/holiday_01.jpg
    (2) ./input_photos/holiday_07.jpg
    ...
Who is this? (name/skip) split 1,2
```

| Command | Effect |
|---------|--------|
| `merge <cluster-id>` | merge this cluster with another; if the other already has a name, that name is inherited and no question is asked |
| `split <n,n,...>` | move those numbered faces into a new cluster, reviewed next |
| `split <cluster-id> <n,n,...>` | split another cluster (e.g. one the DB auto-labelled) the same way |
| `show <cluster-id>` | re-display that cluster with its photos and preview |
| `list` | list every cluster, its size, and its current state |
| `rename <cluster-id> <name>` | name another cluster without being prompted for it |
| `help` | show the command list |
| `abort` | quit without changing any files |

Plain answers still work exactly as before: a name names the cluster,
Enter/`skip`/`unknown` sends it to the unknown folder, and each named
cluster is saved to the name DB immediately (so Ctrl+C never loses it).
`help`, `list`, `merge`, `split`, `show`, `rename` and `abort` are
reserved, so they cannot be used as person names.

### Flags (each one overrides config.yaml)

| Flag | Overrides |
|------|-----------|
| `--config PATH` | which config file to read |
| `--input FOLDER` | `input_folder` |
| `--output FOLDER` | `output_folder` |
| `--tolerance EPS` | `tolerance` (DBSCAN eps, cosine distance) |
| `--min-faces N` | `min_faces_per_cluster` |
| `--mode copy\|move` | `mode` |
| `--unknown-folder NAME` | `unknown_folder` |
| `--workers N` | `workers` (worker processes; `0` = auto, `1` = off) |
| `--gui` | start the Streamlit web UI instead of the CLI |
| `--cli` | force the interactive CLI (the executable opens the GUI on a bare double-click) |
| `--no-db` | disables the name DB for this run (no auto-labelling) |
| `--no-preview` | skips rendering/opening the thumbnail montage |
| `--fetch-models` | download weights once, then exit |
| `-v` / `-vv` | more logging |

```bash
python -m src.main --config path/to/config.yaml
python -m src.main --input "D:\Vacation" --output "D:\Vacation\sorted" --mode move
python -m src.main --tolerance 0.45 --min-faces 3 -v
```

`python -m src.cli` is an alias for the same command. Run
`python -m src.main --help` for the full list of options.

## Age invariance (childhood ↔ adult photos)

A whole-face ArcFace embedding encodes face *shape* as much as identity, and
shape changes as a child grows — the jaw widens and lengthens, the cheeks
round out, the nose lengthens. Cosine distance between a childhood photo and
the same person's adult photo is therefore often larger than the clustering
tolerance, which splits one person into a "child" group and an "adult" group.

FaceSort fixes this with a **second fingerprint per face** and a blended
distance.

### The periocular ("eye region") embedding

`src/eye_embedder.py` extracts an additional embedding from the region that
changes least with age: the brows, both eyes and the bridge of the nose
(the "periocular" region).

Rather than cropping pixels and resizing them, it **re-frames the five
landmarks** and lets the existing ArcFace recogniser do what it already does.
The transform is driven by where the landmarks sit, so pulling the nose and
mouth points *towards* the eye line (`PERIOCULAR_SPREAD = 0.55`) forces a
tighter zoom: the eyes land on ArcFace's template positions and the mouth is
pushed off the bottom edge. The model normalises the region itself rather
than our own resize inventing a second geometry.

Rendering the warp at spreads 0.3 / 0.45 / 0.6 / 0.8 / 1.0 confirms the
direction: 1.0 reproduces the whole face, 0.55 is eyes-and-brows, 0.3 is eyes
only. Below ~0.35 the inter-eye distance distorts too much to mean anything.

Cost: one extra forward pass of the recognition net on a 112×112 input.
Faces whose eye region is under 24px are skipped (too small or too blurred to
carry signal) and fall back to the whole-face vector — which is exactly the
low-resolution childhood photo case the manual merge exists for.

### The fused distance

`src/fusion.py` blends the two cosine similarities:

```
fused_similarity = 0.4 · cos(full_a, full_b) + 0.6 · cos(eye_a, eye_b)
fused_distance   = 1 − fused_similarity
```

The eye region is weighted higher because it is the age-stable part, but the
whole-face term is never dropped — periocular alone is a weak discriminator
between siblings. Both similarities are cosine distances in `[0, 2]`, so the
blend shares that range and the existing `tolerance` keeps its meaning.

When either side has no eye vector the blend **renormalises over the
evidence that exists** rather than dropping the comparison. Clustering uses a
precomputed distance matrix (not a Python callable per pair) because the
pipeline is CPU-bound; with mixed availability the clusterer runs both
modalities separately and reconciles them in `relabel_ages()`.

### Manual merge — the safety net

No method is 100% across a wide age gap, so the review screen has a
**Same person?** mode: pick two groups, and they become one. Supplying a
name also writes the link to the name database, and `merge_clusters_by_memory()`
fuses groups the database already knows are one person — so a merge confirmed
once survives every later scan even when the fused distance did not group them
on its own.

| Setting | Default | Meaning |
|---------|---------|---------|
| `use_eye_regions` | `true` | Extract the second fingerprint (set `false` for the old behaviour) |
| `weight_full_face` | `0.4` | Whole-face share of the fused distance |
| `weight_eye_region` | `0.6` | Eye-region share (the two should sum to 1) |

`GET /clusters` reports `eye_coverage` per group; the UI shows a note when a
group leaned on the whole face alone, which is the hint to use the merge.

## Desktop app (Electron + Python)

The desktop app is two pieces that meet on a loopback socket:

| Piece | Lives in | Role |
|-------|----------|------|
| **Electron main** (`electron/main.js`) | `main.js` | Creates the window, spawns the engine, parses `PORT:<port>` from its stdout, polls `/status` until it answers, forwards engine log lines, and kills the whole engine process tree on quit |
| **Preload bridge** (`electron/preload.js`) | `preload.js` | The only path from renderer to Node: the REST helpers, the log/ready events, the native folder picker, dropped-folder resolution, and `openPath` |
| **Renderer** (`electron/renderer/`) | `index.html`, `styles.css`, `app.js` | The UI: no framework, no webfont, no build step — icons are an inline SVG sprite so the app stays fully offline |
| **Engine** (`src/api/server.py`) | `python_build/dist/backend/backend.exe` (PyInstaller) | FastAPI app wrapping scanner → detector → clusterer → organizer + the SQLite name DB |

### The interface

Four views, driven by one small state machine:

1. **Configure** — pick (or drag-and-drop) the input folder, choose the
   destination, tune tolerance / min faces / worker processes and copy-vs-move.
2. **Scan** — gradient progress bar with a live percentage and the current
   file, a **live preview of the image being processed** (`GET /thumb`),
   shimmer skeletons and an animated orb.
3. **Review** — one card per group with cropped face thumbnails; hovering
   lifts the card and zooms the grid, clicking a face opens it full-screen
   (`/photo`), *Details* flips the card in 3D to show the destination folder
   and every file, naming opens a slide-up dialog with suggestions, and a
   toast confirms each name. Cards are titled by the person's name once set,
   and the filter box narrows them as you type. **Same person?** links two
   groups that should be one (see [Age invariance](#age-invariance-childhood-adult-photos)).
4. **Finish** — animated checkmark, per-folder counts, "open output folder"
   and an optional confetti celebration.

Extras: dark/light theme (persisted, follows the OS on first run), toasts, a
scroll-to-top button, a search filter for groups, an engine log panel, and
`prefers-reduced-motion` support.

Performance notes: animation is restricted to `transform`/`opacity` (no layout
thrash), staggered card reveals are capped at 12 cards, and `will-change` is
deliberately avoided.

`main.js` resolves the engine from the mode it is running in, and logs the
exact path before spawning it (this is the first thing to check when startup
fails):

```js
// packaged:  <app>/resources/backend.exe
// dev:       <repo>/python_build/dist/backend/backend.exe
log(`mode    : ${IS_PACKAGED ? 'packaged' : 'development'}`);
log(`engine  : ${executable}`);
```

### Running it from a source checkout

```bash
python -m src.main --fetch-models      # once: ~180 MB of ONNX weights
build.bat                              # or ./build.sh  (or the steps below)
```

To run the app straight from a source checkout without packaging it:

```bash
python build.py models                 # stage det_10g + w600k_r50 for the engine
python build.py backend                # -> python_build/dist/backend/backend.exe
npm install                            # once, at the project root (delegates to electron/)
npm start                              # from the project root or from electron/
```

`npm start` works from either directory — the root `package.json` just forwards
to `electron/`. In development `main.js` spawns the **PyInstaller output**
(`python_build/dist/backend/backend.exe`), so the engine must be built first;
editing `src/*.py` needs another `python build.py backend` (~6 min) to take
effect.

The bridge narrates startup in the terminal, which is the first place to look
when the window does not appear:

```
[bridge] mode    : development
[bridge] engine  : …\python_build\dist\backend\backend.exe
[bridge] cwd     : …\FaceSort
[bridge] spawned … (pid 4720)
[bridge] renderer ready: {"title":"FaceSort","bridge":"object","cards":0,…}
[bridge] engine reported port 52459
[bridge] engine is ready
```

Renderer errors, failed page loads, preload problems and GPU crashes are
mirrored into the same log (e.g. `renderer ERROR: …`, `renderer FAILED to
load: …`).

**If it does not start**

| Symptom | Cause / fix |
|---------|-------------|
| `npm error enoent Could not read package.json` | You ran npm in a folder without a manifest — use the project root or `electron/` |
| Dialog: *The Python engine was not found* | The path in the log does not exist yet → `python build.py backend` |
| Console prints the bridge lines but no window | The window is behind another window, or it was closed while a second instance focused the first one (single-instance lock) |
| Window opens blank/white | GPU driver problem → `npx electron . --disable-gpu` |
| `FACEORG_SMOKE=1 npm start` prints `SMOKE_OK {…}` and exits | Not a failure — that is the automated self-check passing |

`npm start` opens the window, spawns the engine, and the app is usable within
a second or two (the first scan additionally loads the models).

For automated checks (CI, or verifying a packaging change) the bridge has a
self-terminating smoke mode — it boots, verifies the handshake, prints one
machine-readable line and exits:

```bash
FACEORG_SMOKE=1 npm start
# [bridge] engine  : .../python_build/dist/backend/backend.exe
# [bridge] engine reported port 65063
# [bridge] engine is ready
# SMOKE_OK {"packaged":false,"port":65063,"models":{...,"installed":true},...}
```

Useful while developing:

| Goal | Command |
|---|---|
| Auto-verify the bridge, then exit | `FACEORG_SMOKE=1 npm start` |
| Open DevTools | `Ctrl+Shift+I` in the window (or View → Toggle DevTools) |
| Skip GPU (old drivers, blank window) | `npx electron . --disable-gpu` |
| Inspect the real window over HTTP | `npx electron . --remote-debugging-port=9223`, then open `http://127.0.0.1:9223` |

The engine can also be driven on its own — it prints the port it bound to:

```bash
python -m src.api.server -v
# PORT:52448
# {"service": "facesort", "state": "idle", ...}   <- GET /status
```

### Building the Windows executable

One command does everything — it runs the two stages in order and stops with
a clear message at the first failure:

```bat
build.bat                :: Windows
```
```bash
./build.sh               :: macOS / Linux
```

If you prefer to drive packaging from npm (the Python engine must already be
built — see step 1 below):

```cmd
npm install              :: once - installs the Electron toolchain
npm run electron:build   :: packages the app -> electron\release\*.exe
```

| npm script | What it does |
|------------|--------------|
| `npm start` | Launch the app in development mode |
| `npm run electron:start` | Same as `npm start` (explicit name) |
| `npm run electron:build` | Package with electron-builder (NSIS installer) |
| `npm run dist` | Same, with the Windows target spelled out |
| `npm run dist:portable` | Portable single-file exe (no installer) |
| `npm run dist:dir` | Unpacked app folder in `release\win-unpacked` |

What the master script does, step by step:

| Step | Command | Produces |
|------|---------|----------|
| 0 | `python build.py models` | stages the two ONNX weights in `python_build\models\buffalo_l` |
| 1 | `python build.py backend` → `pyinstaller --noconfirm --distpath python_build\dist --workpath python_build\build backend.spec` | the engine at `python_build\dist\backend\backend.exe` (+ `_internal\`, models included) |
| 2 | `npm run electron:build` (electron-builder) | `electron\release\FaceSort-Setup-1.0.0.exe` |

Step 1 is the Python half and npm cannot run it, so after changing anything
under `src\`, run `build.bat` (or `python build.py backend`) before
`npm run electron:build` — otherwise the packaged app would ship the
previous engine build.

The build produces three ready-to-run `.exe` files:

| File | What it is | How to run it |
|------|------------|---------------|
| `electron/release/FaceSort-Setup-1.0.0.exe` | **NSIS installer** — per-user, lets the user pick the folder, creates desktop + start-menu shortcuts | Double-click, install, launch from the shortcut |
| `electron/release/FaceSort-Portable-1.0.0.exe` | **Portable single file** (no installer) — the same app, unpacked to `%TEMP%` on each launch (`npm run dist:portable`) | Copy it anywhere and double-click |
| `python_build/dist/backend/backend.exe` | **Standalone engine** (onefile variant: `python build.py backend --onefile`) — the API server with the ONNX weights inside, no UI | Prints `PORT:<port>`; for scripted/headless use or another front-end |

### Where the weights come from

`backend.spec` adds the staged ONNX graphs to `models/buffalo_l/` **inside**
the engine bundle, so `backend.exe` is self-contained and never downloads
anything. `datas` entries are tuples and therefore need no OS-specific
separator; the equivalent command line is:

```bat
:: Windows  (semicolon)
pyinstaller --add-data "python_build\models\buffalo_l;models/buffalo_l" ...
```
```bash
# macOS / Linux  (colon)
pyinstaller --add-data "python_build/models/buffalo_l:models/buffalo_l" ...
```

Note: the weights come from the **insightface cache** (`~/.insightface/models/buffalo_l`),
*not* from `site-packages/insightface/models` — insightface downloads them to
the user profile on first use, so `python build.py models` stages them from
there and `python -m src.main --fetch-models` is the only networked step.

`python build.py backend --onefile` produces a single self-extracting engine
(slow: it unpacks ~360 MB to `%TEMP%` per launch) — the desktop app always
uses the onedir build.

`python build.py backend --embed-models` puts the weights inside the engine
binary (self-contained, ~180 MB duplicate, and a onefile build extracts to
`%TEMP%` on every launch — which is why the Electron route ships the weights
as a resource instead).

### What ships where

`extraResources` in `electron/package.json` copies the PyInstaller output into
the app's `resources/` folder:

| Config | From | Lands as |
|--------|------|----------|
| `{ "from": "../python_build/dist/backend", "to": "." }` | `python_build/dist/backend/` (PyInstaller onedir, models included) | `resources/backend.exe` + `resources/_internal/` |

`to: "."` means *the contents* of the folder go straight into `resources/`,
which is what makes the packaged path exactly
`path.join(process.resourcesPath, 'backend.exe')` in `main.js`. The ONNX
weights ride along inside `resources/_internal/models/buffalo_l/`, so there is
no second copy and no environment variable to set.

```
FaceSort/
└── resources/
    ├── app.asar                    # Electron shell (main.js, preload.js, renderer/)
    ├── backend.exe                 # the Python engine (spawned by main.js)
    └── _internal/
        ├── models/buffalo_l/       # det_10g.onnx, w600k_r50.onnx
        └── ...                     # numpy, cv2, onnxruntime, sklearn, ...
```

`asar: true` stays on for the JavaScript. The engine does **not** need
`asarUnpack`: `extraResources` files are copied next to `app.asar`, never
inside it, so `backend.exe` and the ONNX files are always plain files on disk
and can be executed/read by a child process.

Only the two ONNX models the pipeline actually executes are shipped
(`det_10g.onnx` 17 MB + `w600k_r50.onnx` 174 MB); buffalo_l's landmark and
gender-age graphs are never called, so they are left out.

### Port handshake & process lifecycle

1. `main.js` resolves the engine path from the mode it runs in and logs it:
   packaged → `path.join(process.resourcesPath, 'backend.exe')`, development →
   `<repo>/python_build/dist/backend/backend.exe`. It is started with the
   user's data folder as its working directory, so relative paths in
   `config.yaml` (the name DB) never write into the install folder. No model
   path is passed: the weights live inside the engine bundle and
   `src.face_model` resolves `sys._MEIPASS` itself.
2. The engine binds a **free** port itself (`bind(0)` → `listen()`), then
   prints exactly one line to stdout: `PORT:<port>`. Nothing else is ever
   written to stdout — all logging goes to stderr, because stdout is the
   protocol channel.
3. `main.js` matches `/^PORT:(\d+)$/`, polls `GET /status` until it answers
   (the socket is already listening, so there is no port race), and sends
   `{port, status}` to the renderer.
4. `preload.js` turns the port into a base URL; the renderer only ever calls
   `http://127.0.0.1:<port>`.
5. On quit, `main.js` runs `taskkill /pid <pid> /T /F` (Windows) so the
   engine *and* its multiprocessing pool workers die together.

### API reference

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/status` | State machine (`idle`/`scanning`/`ready`/`organizing`/`done`/`error`), progress (`processed`, `total`, `current`), scan stats, effective config (plus `input_exists`/`output_exists` so the sidebar can skip dead defaults), model status, port/pid. Doubles as the readiness probe |
| `POST` | `/scan` | Validates the input folder, then scans on a background thread. Body: `input_folder`, `output_folder`, `tolerance`, `min_faces_per_cluster`, `mode`, `unknown_folder`, `workers`, `use_db` |
| `GET` | `/clusters` | Every cluster with its faces: photo name, bbox, detection score and a cropped JPEG **data URL** thumbnail |
| `POST` | `/name_cluster` | Name a group (`name_cluster: {cluster_id, name}`). Empty name → unknown folder; non-empty names are remembered in the SQLite DB for future scans |
| `POST` | `/merge_clusters` | Link two groups as one person — the manual age bridge. Body `{cluster_a, cluster_b, name?}`; `name` is optional (omit it to link for this run only). Groups the database already knows are one person are re-fused on every scan |
| `POST` | `/organize` | Copies/moves every photo into `output_folder/<person>/` **on a worker thread** — returns `{"started": true, "total": n}`; follow `/status` for per-file progress and read the report from `status.results` |
| `GET` | `/thumb?name=` | Small (420 px) JPEG of one photo **inside the scanned folder** — the live scan preview. Scoped by design: only a *basename* is accepted and resolved inside the input folder, so the endpoint can never be used to read arbitrary files |
| `GET` | `/photo?name=` | The same photo at 1600 px, for the review lightbox. Identical scoping; renders are memoised on `(path, mtime, size)` in a 64-entry cache |

Interactive docs are available while the engine runs: `http://127.0.0.1:<port>/docs`.

### Security model

- The renderer has `contextIsolation: true` and `nodeIntegration: false`; it
  cannot touch the filesystem or spawn processes.
- The bridge in `preload.js` only ever talks to the hard-coded
  `http://127.0.0.1:<port>` base URL — no caller can redirect a request.
- The renderer ships a strict CSP (`default-src 'none'`, images `data:`,
  `connect-src http://127.0.0.1:*`).
- The engine binds `127.0.0.1` only and is single-user; its CORS policy is
  permissive *because* it is loopback-only (the renderer is `file://` in dev).
- Nothing in the engine reaches the network at run time: ONNX files are only
  ever read from disk, and the models ship with the app.

## Legacy Streamlit UI (M4, optional)

Kept for reference and for headless/server setups; the Electron app is what
ships now.

```bash
streamlit run src/ui/app.py
# or, equivalently (starts the same server and opens the browser):
python -m src.main --gui
```

The browser opens automatically (or visit the URL printed in the
terminal). The GUI is a thin front-end over the same pipeline as the CLI:

1. **Sidebar → Configuration** — input/output folders, tolerance, and
   copy/move mode; the values start from `config.yaml`. *Re-scan photos*
   throws away the cached scan.
2. **Scan photos & detect faces** — runs detection and embedding with a
   live progress bar. The results are stored in `st.session_state`, so
   typing names, moving sliders, or any other interaction re-renders the
   page **without** re-running the pipeline (changing the input folder
   asks you to re-scan instead of doing it silently).
3. **Review the clusters** — each cluster shows its numbered face
   montage, its photo list, and a `Who is this? (name/skip)` input
   pre-filled with the name the database recognised (highlighted as
   auto-detected).
4. **Run Organizer** — copies (or moves) every photo into
   `output_folder/<person>/`; names you typed are saved to the name DB
   for the next run, and unnamed clusters go to `_unknown/`.

Everything runs locally — no photo, face, or name ever leaves the
machine. Streamlit's optional usage telemetry is disabled in
[`.streamlit/config.toml`](.streamlit/config.toml).

## Performance (M6)

The vision stage (load → detect → embed) runs on a pool of **worker
processes** whenever that can help:

- `workers: 0` (the default) = auto: `cpu_count // 2` workers (max 4).
  Each worker gets `cpu_count // workers` ONNX Runtime threads — the
  buffalo_l graphs benchmark best around two threads, and independent
  processes scale further than one graph with a big thread pool.
- `workers: 1` disables the pool entirely; any other value forces that
  many workers (clamped to the CPU count).
- Folders with fewer than 24 photos always stay in-process: the per-worker
  model start-up (several seconds) would cost more than the pool saves.
- Workers pin OpenCV/BLAS to a single thread each so N workers never fight
  over cores, and a bad photo becomes a counted "failed/unreadable" result
  instead of killing the pool.

### Measuring throughput

`benchmarks/bench_pipeline.py` builds 1080p JPEGs from any folder of
photos and times the pipeline against the SPEC §6 target:

```bash
python benchmarks/bench_pipeline.py --source ./photos --count 60   # measure
python benchmarks/bench_pipeline.py --source ./photos --check      # exit 1 if < 5 img/s
python benchmarks/bench_pipeline.py --source ./photos --workers 1  # force sequential
```

Measured on the reference machine for this repository (a 2013 laptop,
Intel i5-3337U — 2 cores / 4 threads, CPU only, 30–60 × 1080p photos):

| Configuration | Throughput |
|---|---|
| single process (`--workers 1`) | 0.45 img/s |
| auto pool (`--workers 0` → 2 workers × 2 threads) | 0.47 img/s incl. pool start-up — ≈ **0.55 img/s** steady state |

**Why this machine does not reach 5 img/s:** the default buffalo_l
pipeline costs ≈ 2.7 CPU-seconds of compute per 1080p photo here (SCRFD
detection + ArcFace-R50 embedding). 5 img/s would mean ≈ 13.5 CPU-seconds
of compute per second — about 7× what two 2013 cores can deliver, so no
software parallelism can close that gap; multiprocessing removes the
software-side limit, and what remains on this hardware is physics. The
SPEC §6 target presumes a modern multi-core CPU — run the `--check`
command above on your machine to verify it there. The optional GPU path
mentioned in SPEC §11 is not implemented: this machine has no CUDA GPU
and an untestable code path would be worse than none.

## Tests

```bash
python tests/test_pipeline.py    # image loader, detector, embedder
python tests/test_clusterer.py   # clustering, de-duplication, config mapping
python tests/test_age_invariance.py  # periocular landmarks, fused distance, age pairing
python tests/test_organizer.py   # folders, copy/move, name collisions
python tests/test_names_db.py    # persistent name DB, matching, blending
python tests/test_preview.py     # thumbnail montage, temp files, viewer
python tests/test_main.py        # CLI flags, prompts, M3 flow, merge/split
python tests/test_app.py         # legacy Streamlit GUI: session state, cards
python tests/test_parallel.py    # multiprocess stage: workers, statuses, pool
python tests/test_api.py         # REST API: scan/clusters/name/organize, PORT
python tests/test_api.py --real  # same, plus a real server + real-model scan
```

The scripts are plain Python (no test runner needed) and do not require the
model weights (only `--real` does). They are also collectable by `pytest tests/`.

## Project layout

```
FaceSort/
├── src/
│   ├── main.py          # CLI entry point (M1 pipeline)
│   ├── cli.py           # alias for main.py
│   ├── api/
│   │   └── server.py    # FastAPI backend for the Electron app (M5)
│   ├── image_loader.py  # scans the input folder, skips corrupt files
│   ├── face_model.py    # shared buffalo_l / onnxruntime CPU model (offline)
│   ├── detector.py      # face detection (+ multiprocess pool, M6)
│   ├── embedder.py      # face embeddings (whole-face + periocular)
│   ├── eye_embedder.py  # periocular landmarks & age-stable fingerprint
│   ├── fusion.py        # age-invariant fused distance (pure maths)
│   ├── clusterer.py     # clustering of embeddings
│   ├── organizer.py     # copies/moves photos into named folders
│   ├── names_db.py      # persistent name database, SQLite (M3)
│   └── ui/
│       ├── preview.py   # thumbnail montage / face crops (M2, reused by the API)
│       └── app.py       # legacy Streamlit front-end (M4, optional)
├── electron/            # desktop app (M5)
│   ├── package.json     # app manifest + electron-builder config
│   ├── main.js          # window, engine spawn, PORT handshake, shutdown
│   ├── preload.js       # the only renderer↔Node bridge (REST + dialogs)
│   ├── build/icon.ico   # app icon (regenerate: python tools/make_icon.py)
│   └── renderer/
│       ├── index.html   # sidebar + cluster grid
│       ├── styles.css   # theme (no framework, no build step)
│       └── app.js       # UI logic against the five endpoints
├── tools/
│   └── make_icon.py     # regenerates electron/build/icon.ico
├── build.bat            # master build script (Windows): PyInstaller -> npm run build
├── build.sh             # master build script (macOS / Linux)
├── backend.spec         # PyInstaller build of the API engine (M5)
├── build.py             # build orchestrator: models / backend / electron / clean
├── python_build/        # PyInstaller output: models/, build/, dist/backend/ (git-ignored)
├── electron/release/    # installers produced by electron-builder (git-ignored)
├── tests/
│   ├── test_pipeline.py
│   ├── test_clusterer.py
│   ├── test_age_invariance.py
│   ├── test_organizer.py
│   ├── test_names_db.py
│   ├── test_preview.py
│   ├── test_main.py
│   ├── test_app.py
│   ├── test_parallel.py
│   └── test_api.py
├── benchmarks/
│   └── bench_pipeline.py   # throughput vs the SPEC §6 target (M6)
├── packaging/
│   └── entry.py            # PyInstaller entry point (freeze_support first)
├── .streamlit/config.toml  # legacy GUI settings (usage telemetry off)
├── backend.spec            # PyInstaller build of the API engine (M5)
├── build.py               # build orchestrator: models / backend / electron
├── python_build/          # PyInstaller output: models/, build/, dist/backend/ (git-ignored)
├── config.yaml
├── requirements.txt
├── SPEC.md.md
└── README.md
```

## Development status

| Milestone | Status |
|-----------|--------|
| M1 – Core CLI (scan → detect → cluster → prompt → sort) | ✅ done |
| M2 – Preview UI (montage + merge/split review) | ✅ done |
| M3 – Persistent DB | ✅ done |
| M4 – GUI (Streamlit, now legacy/optional) | ✅ done — superseded by the Electron app |
| M5 – Packaging (Electron app + PyInstaller engine) | ✅ done — see [Desktop app](#desktop-app-electron--python) |
| M6 – Performance (multiprocessing) | ✅ done — GPU path not applicable (CPU-only machine), see [Performance](#performance-m6) |

## License

Free and open-source — all dependencies are free (see SPEC.md §14).
