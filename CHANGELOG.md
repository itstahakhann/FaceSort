# Changelog

All notable changes to FaceSort are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-10-02

First public release. Offline face grouping and photo auto-organizer: an
Electron desktop app driving a CPU-only Python engine (InsightFace
RetinaFace + ArcFace) over a loopback socket. No photo, face or name ever
leaves the machine.

### Added

**Age-invariant grouping** — a whole-face embedding encodes face shape as much
as identity, and shape changes as a child grows, so one person's childhood and
adult photos routinely land in separate clusters. Each face now gets a second
fingerprint from the periocular region (brows, eyes, nose bridge), which barely
changes with age, and clustering blends both signals weighted 0.4 whole-face /
0.6 eye region. Faces whose eye region is too small or blurred to fingerprint
fall back to the whole-face vector rather than being dropped.

**Manual age bridge** — a "Same person?" mode in the review screen links two
groups the model could not join. Supplying a name writes the link to the
database, and groups the database already knows are one person are re-fused on
every later scan.

**Four-view interface** — configure, scan, review, finish, with a stepper, a
live preview of the photo being processed, animated progress, 3D flip cards, a
full-photo lightbox, drag-and-drop folder selection and toasts.

**Dark and light themes**, following the OS on first run and persisted after.

**Per-user data locations** — the name database resolves to
`%LOCALAPPDATA%\FaceSort\` (Application Support on macOS, XDG on Linux) rather
than the working directory, which is read-only under a normal Windows install.
Overridable with `FACEORG_DATA_DIR`.

**Organizing with progress** — copying runs on a worker thread and reports
per-file progress, so long copy jobs show real movement instead of freezing.

### Notes

- Windows binaries are **unsigned**. SmartScreen will warn on first run; click
  *More info → Run anyway*.
- Inference is fully offline. The only code path that touches the network is
  `python build.py models`, which fetches the ONNX weights once.
- Measured on an i5-3337U: roughly 0.3 images/second with 2 worker processes.
  Detection dominates; the age-invariance pass adds about 21% per image.

[1.0.0]: https://github.com/itstahakhann/FaceSort/releases/tag/v1.0.0