# Frontend redesign: notes and discrepancies

This records what the FaceFlow redesign changed, and — more importantly —
every place where the brief and the actual codebase disagreed. The brief
instructed that where the architecture differs from `APP_STATUS.md`, the code
is the source of truth and the discrepancy should be documented rather than
worked around silently.

## `APP_STATUS.md` does not exist

The brief names `APP_STATUS.md` as the authoritative architecture document and
says to read it in full before changing anything. There is no such file in the
repository. The actual architecture documentation is:

- `README.md` (~50 KB) — architecture, endpoints, build, data models
- `SPEC.md.md` — the original milestone specification
- `CHANGELOG.md` — release history

Everything below was established by reading those and the code itself.

## Endpoints named in the brief that do not exist

Enumerating `src/api/server.py` gives the real surface:

| Route | Brief listed it? |
|---|---|
| `GET /status` | yes |
| `POST /scan` | yes |
| `GET /clusters` | yes |
| `POST /name_cluster` | yes |
| `POST /merge_clusters` | yes |
| `GET /people_list` | yes |
| `GET /intersection` | yes |
| `GET /co_occurrence_matrix` | yes |
| `POST /export_gallery` | yes |
| **`POST /split_cluster`** | **listed — does not exist** |
| **`GET /config`** | **listed — does not exist** |
| **`POST /config`** | **listed — does not exist** |
| `GET /gallery_status` | *not listed — exists* |
| `POST /export_intersection` | *not listed — exists* |
| `GET /thumb` | *not listed — exists* |
| `GET /photo` | *not listed — exists* |

No endpoint was renamed, removed or changed. Three were added earlier for the
gallery feature and the review lightbox.

### `POST /split_cluster` — cannot be built without backend work

There is no split operation anywhere in `src/`. Splitting a cluster means
re-running recognition on a subset of a cluster's photos and storing the new
grouping — clusterer, organizer and the schema would all be involved.

The brief forbids rewriting the backend and forbids fabricating data, so the
Split control **ships present, disabled, and explains itself**. Its tooltip
reads:

> Splitting a group is not supported by the local engine yet. Use Merge to join
> two groups the engine split apart, and Skip to send a group to the unknown
> folder.

This is listed as a **future backend enhancement**, not a frontend gap.

### `GET /config` and `POST /config` — the config surface is different

Configuration is not its own endpoint. It works like this:

- `GET /status` returns a `config` object with every key the UI can read:
  `input_folder`, `output_folder`, `tolerance`, `min_faces_per_cluster`, `mode`,
  `unknown_folder`, `workers`, `names_db`, plus `input_exists` / `output_exists`
  so the renderer can tell "unset" from "set to a path that is missing".
- `POST /scan` takes a body that **overrides `config.yaml` for that run**. The
  docstring says so: *"every field overrides `config.yaml`"*.

Consequences, and how the UI handles them:

1. **Settings are held in the renderer and applied on the next scan.** The
   Settings page says exactly this: *"These values are sent to the engine with
   your next scan."* It does not claim to persist anything, because it does not.
2. **There is no Save button**, because there is nothing to save. A fake one
   would be worse than none.
3. Only keys the engine actually reads are exposed. Nothing was invented.

`appInfo` (version, engine path, data-directory path) comes from the existing
`app:info` IPC handler, not from a new endpoint.

## Product naming

The brief says the app is currently called "Face Grouping & Auto-Organizer".
That name appears in `README.md` and as a comment in `config.yaml`, but the
Electron `productName` was `FaceSort`, and the engine's service string is
separate again. The redesign adopts **FaceFlow** in:

- `index.html` (`<title>`) and the sidebar brand
- `electron/main.js` - the window title, and the titles on the two native
  error dialogs
- `electron/package.json` — `productName`, `appId`, `artifactName`,
  `nsis.shortcutName`

That renames the installers from `FaceSort-Setup-*.exe` to
`FaceFlow-Setup-*.exe`. `build.py release` globs `*.exe`, so it keeps working,
but any published download link and `release_notes.md` need updating.

The tag line, "Private photo organization, powered locally", is the sidebar
subtitle and the About copy.

## Architectural constraints that shaped the code

Both were found by inspecting, not by assumption.

**`file://` blocks ES modules.** `main.js` calls `mainWindow.loadFile(...)`, so
the page origin is `null` and a `<script type="module">` fails CORS. The
renderer is therefore a set of classic scripts attached to one `window.FF`
namespace, loaded in dependency order from `index.html`. This is why the
structure is `js/core.js`, `js/ui.js`, `js/views/*.js` rather than ES modules —
it is a platform constraint, not a style preference.

**CSP forbids inline styles.** The preload-maintained policy is
`style-src 'self'`. There are no `style="..."` attributes anywhere in the
markup; every inline value is set through the CSSOM from JavaScript, which the
policy permits. This is why there are no `<style>` blocks either.

## What is preserved from the previous UI

Everything that worked:

| Capability | Where it lives now |
|---|---|
| Folder pick + drag-and-drop | Scan view, plus a window-wide drop veil |
| Scan with live percentage, preview, stats | Scan view |
| Cluster cards with face montages | Photos view |
| Naming, rename, skip to `_unknown` | Photos view, dialog |
| Manual merge ("same person?") | Photos view, two-step pick |
| Copy vs Move, tolerance, min faces, workers | Scan + Settings |
| Filmstrip of recently read photos | Scan view |
| Live progress while sorting | Scan view, organising state |
| Organize, folder tally, done screen | Photos view |
| People list, person detail, photo grid | People view |
| Pairwise intersection, co-occurrence, export | Relationships view |
| Gallery export with thumbnails, password, progress, success | Gallery view |
| Theme toggle, toasts, dialogs, lightbox, keyboard support | `js/ui.js` |

## Not yet wired

Two things exist in the UI and behave honestly, but are inert because the
engine does not support them:

- **Split** — disabled, with an explanation. See above.
- **Rename from a person** — calls `POST /name_cluster`, which names a
  *cluster*. A person is an aggregate across clusters, so the rename applies to
  the clusters behind that name on the next scan. The code comment says so; the
  toast does not overclaim.

## Verification

Because the real detector cannot find faces in synthetic drawings, the screens
that need clusters and face crops (review, naming, merge, person detail,
relationship results) were driven against a **stubbed preload bridge** in a
headless browser. Everything on screen in those screenshots is produced by the
shipping `index.html` / `js/` / `styles/`; only the engine payloads are
fabricated. Startup, the engine handshake and every section were additionally
verified against the **real** engine, and the packaged `FaceFlow.exe` was
smoke-tested to confirm the renderer ships inside `app.asar`.

## The menu bar was removed, not renamed

The File/Edit/View/Window/Help bar took a row of window height, and every
item on it was already reachable elsewhere:

- *Open input folder* only jumped to the screen where a folder is chosen,
  which the sidebar reaches directly.
- *View* and *Window* offered reload, zoom and fullscreen, all of which the
  title bar buttons and the maximised window already cover.
- *Help > About* said **less** than Settings > About, which shows the
  version, the engine path, the name-database path and an offline callout.
  Reinstating a dialog that says less would be a downgrade, so it was left
  out rather than re-homed.

Removing the bar also removes its accelerators, so the three worth keeping
are re-registered in `installWindowShortcuts` through `before-input-event`:
`Ctrl+Shift+I`/`F12` (the release README tells people to press this when the
window comes up blank, so it is the only diagnostic a user has), `Ctrl+O`,
and `Ctrl+R`/`F5`.

`globalShortcut` would have been the wrong tool here: it registers with the
operating system and would take these keys from every other program on the
machine.

### Known trade-off

Fullscreen (View > Toggle Fullscreen) is gone. Maximise still works, and
Ctrl+Shift+I can open the standalone DevTools window if a true full-screen
view is ever needed for debugging.
