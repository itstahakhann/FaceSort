# Changelog

All notable changes to FaceFlow are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-10-03

First public release, shipped as **FaceFlow** — private photo organization,
powered locally. An Electron desktop app drives a CPU-only Python engine
(InsightFace RetinaFace + ArcFace) over a loopback socket. No photo, face or
name ever leaves the machine, and the app makes no network requests at all.

### Added

**Age-invariant grouping** — a whole-face embedding encodes face shape as much
as identity, and shape changes as a child grows, so one person's childhood and
adult photos routinely land in separate clusters. Each face now gets a second
fingerprint from the periocular region (brows, eyes, nose bridge), which barely
changes with age, and clustering blends both signals weighted 0.4 whole-face /
0.6 eye region. Faces whose eye region is too small or blurred to fingerprint
fall back to the whole-face vector rather than being dropped.

**Manual age bridge** — a *Merge* action in the review screen links two groups
the model could not join. Supplying a name writes the link to the database, and
groups the database already knows are one person are re-fused on every later
scan.

**Relationships** — pick two or more named people and see only the photos where
every one of them appears, which is how you find the family trip rather than one
person's pictures of it. A pairwise co-occurrence ranking answers the same
question at the group level. Both draw on the names you assign, so they grow as
you use the app, and both export to a folder.

**Gallery export** — produces a folder containing one HTML page, a stylesheet, a
script and the photos: a people index, per-person grids, a full-screen viewer
with keyboard and touch support, a name filter and a dark/light toggle. It opens
from `file://` with no server and no network, so it survives being zipped,
emailed, put on a USB stick, or opened with no signal. Optionally restricted to
chosen people and password-gated.

**The FaceFlow interface** — a fixed sidebar (Overview, People, Photos,
Relationships, Gallery Export) beside a content pane that swaps views without the
window reloading. One card per group with a face montage, naming with
suggestions, a search box, and drag-and-drop folder selection. Organising shows
the destination folders as they are created and files each photo with live
progress. Recent Activity records your own actions rather than raw log lines, and
a permanent offline indicator is wired to the real network state.

**Dark and light themes**, following the OS on first run and persisted after.

**Per-user data locations** — the name database resolves to
`%LOCALAPPDATA%\FaceSort\` (Application Support on macOS, XDG on Linux) rather
than the working directory, which is read-only under a normal Windows install.
Overridable with `FACEORG_DATA_DIR`.

**Organizing with progress** — copying runs on a worker thread and reports
per-file progress, so long copy jobs show real movement instead of freezing.

**Installer and portable build** — a per-machine installer and a single-file
portable executable, both Windows x64, both offline from first launch.

### Known limitations

- **Split is unavailable.** The review screen's *Split* control is present but
  disabled, and says why: the engine has no split operation. Use *Merge* to
  join two groups the engine pulled apart, and *Skip* to set a group aside.
  Splitting properly means changing the recognition and storage layers, so it is
  not faked in the interface.
- **The gallery password is a gate, not encryption.** The SHA-256 digest ships
  inside the HTML file, so anyone who opens the source can read it, and a weak
  password falls to an offline dictionary. It stops casual browsing, not a
  determined reader — and since a browser must read a JPEG to display it,
  anyone who can open the page can open the image files directly. The app says
  this next to the setting.
- Windows binaries are **unsigned**. SmartScreen will warn on first run; click
  *More info → Run anyway*.

### Notes

- Inference is fully offline. The only code path that touches the network is
  `python build.py models`, which fetches the ONNX weights once during a build.
- Polling starts only while a job is running and stops the moment it settles, so
  an idle app makes no requests at all.
- Measured on an i5-3337U: roughly 0.3 images/second with 2 worker processes.
  Detection dominates; the age-invariance pass adds about 21% per image.
- The repository, the on-disk data directory and the package namespace remain
  `FaceSort`. Renaming the data path would orphan every existing user's
  remembered people.

[1.0.0]: https://github.com/itstahakhann/FaceSort/releases/tag/v1.0.0
