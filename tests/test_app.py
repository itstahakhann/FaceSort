"""Unit tests for the M4 Streamlit GUI (SPEC §11).

Run with::

    python tests/test_app.py
    pytest tests/

Drives the real app through Streamlit's own head-less ``AppTest`` harness,
with the face pipeline stubbed out (the model pipeline itself is covered
by ``test_pipeline``).  Verifies that

* the sidebar shows the config.yaml defaults,
* scanning happens exactly once — widget interaction never re-runs it,
* cluster cards appear with a name input pre-filled from the name DB,
* **Run Organizer** really copies the photos into person folders,
* changing the input folder asks for a re-scan instead of doing one.

No model weights and no network access are required.
"""

import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from src.clusterer import FaceRecord
from src.names_db import NamesDB

_failures = []
APP_PATH = Path(__file__).resolve().parent.parent / "src" / "ui" / "app.py"


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}: {label} {detail}")
    if not condition:
        _failures.append(label)


# -------------------------------------------------------------- fixtures
def make_photos(folder: Path, count: int = 4):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        path = folder / f"photo_{i}.jpg"
        Image.new("RGB", (240, 240), (30 + 50 * i, 80, 140)).save(path)
        paths.append(path)
    return paths


def fake_records(paths, split=None):
    """Two "people": first ``split`` photos embed to +1, rest to -1."""
    half = len(paths) // 2 if split is None else split
    records = []
    for index, path in enumerate(paths):
        sign = 1.0 if index < half else -1.0
        embedding = np.full(4, sign, dtype=np.float32)
        embedding /= np.linalg.norm(embedding)
        records.append(FaceRecord(image_path=str(path),
                                  bbox=(10.0, 10.0, 60.0, 60.0),
                                  embedding=embedding))
    return records


@contextmanager
def patched(config=None, records=None):
    """Stub ``src.main.run_pipeline`` + ``load_config``; yield a call log."""
    from src import main as cli

    original_pipeline = cli.run_pipeline
    original_load = cli.load_config
    calls = {"pipeline": 0, "progress": []}

    def fake_pipeline(cfg, progress_cb=None):
        calls["pipeline"] += 1
        stats = cli.PipelineStats()
        stats.records = list(records or [])
        stats.images_scanned = len(stats.records)
        stats.faces_detected = len(stats.records)
        stats.faces_embedded = len(stats.records)
        stats.seconds = 0.05
        if progress_cb is not None:
            for index, record in enumerate(stats.records, 1):
                calls["progress"].append(index)
                progress_cb(index, len(stats.records), record.image_path)
        return stats

    def fake_load(_path):
        base = {
            "input_folder": "./input_photos",
            "output_folder": "./grouped_photos",
            "tolerance": 0.5,
            "mode": "copy",
            "min_faces_per_cluster": 2,
            "unknown_folder": "_unknown",
            "names_db": "",
        }
        base.update(config or {})
        return base

    cli.run_pipeline = fake_pipeline
    cli.load_config = fake_load
    try:
        yield calls
    finally:
        cli.run_pipeline = original_pipeline
        cli.load_config = original_load


def open_app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    return at


def widget_by_label(widgets, label):
    for widget in widgets:
        if getattr(widget, "label", None) == label:
            return widget
    return None


def name_inputs(at):
    return [w for w in at.text_input
            if (w.label or "").startswith("Who is this")]


def press(at, label):
    """Click the button carrying ``label`` and re-run the app."""
    for button in at.button:
        if button.label == label:
            return button.click().run()
    raise AssertionError(f"button {label!r} not found")


# ----------------------------------------------------------------- tests
def test_initial_render():
    print()
    at = open_app()
    check("app renders without exception", not at.exception,
          str(at.exception))

    inp = widget_by_label(at.text_input, "Input folder")
    out = widget_by_label(at.text_input, "Output folder")
    check("sidebar: input folder default from config.yaml",
          inp is not None and inp.value == "./input_photos",
          repr(getattr(inp, "value", None)))
    check("sidebar: output folder default from config.yaml",
          out is not None and out.value == "./grouped_photos",
          repr(getattr(out, "value", None)))
    check("sidebar: tolerance default is 0.5",
          len(at.number_input) == 1
          and abs(float(at.number_input[0].value) - 0.5) < 1e-9,
          str([w.value for w in at.number_input]))
    check("sidebar: mode defaults to copy",
          len(at.radio) == 1 and at.radio[0].value == "copy",
          str([w.value for w in at.radio]))
    check("scan button offered", any(
        b.label == "Scan photos & detect faces" for b in at.button))
    check("re-scan button offered",
          any(b.label == "Re-scan photos" for b in at.button))
    check("organizer button hidden before a scan",
          not any(b.label == "Run Organizer" for b in at.button))
    check("start hint shown",
          any("Scan photos" in (i.value or "") for i in at.info),
          str([i.value for i in at.info]))


def test_scan_organize_and_caching():
    print()
    root = Path(tempfile.mkdtemp(prefix="faceorg_app_"))
    try:
        photos = make_photos(root / "photos")
        out_dir = root / "grouped"
        records = fake_records(photos)
        with patched(config={"input_folder": str(root / "photos"),
                             "output_folder": str(out_dir)},
                     records=records) as calls:
            at = open_app()
            check("pre-scan render clean", not at.exception,
                  str(at.exception))

            at = press(at, "Scan photos & detect faces")
            check("post-scan render clean", not at.exception,
                  str(at.exception))
            check("pipeline ran exactly once",
                  calls["pipeline"] == 1, f"calls={calls['pipeline']}")
            check("progress callback fired once per photo",
                  calls["progress"] == [1, 2, 3, 4], str(calls["progress"]))
            check("one name input per cluster (2 clusters)",
                  len(name_inputs(at)) == 2, f"got {len(name_inputs(at))}")
            check("metrics row rendered", len(at.metric) == 4,
                  f"got {len(at.metric)}")
            check("cluster montages rendered", len(at.image) >= 2,
                  f"got {len(at.image)}")

            # typing a name must not re-scan
            at = name_inputs(at)[0].set_value("Alice").run()
            check("typing a name keeps the cached scan",
                  calls["pipeline"] == 1, f"calls={calls['pipeline']}")

            # moving a slider must not re-scan either
            at = at.number_input[0].set_value(0.7).run()
            check("changing tolerance does not re-scan",
                  calls["pipeline"] == 1, f"calls={calls['pipeline']}")
            values = sorted((w.value or "") for w in name_inputs(at))
            check("typed name survives later reruns",
                  values == ["", "Alice"], str(values))
            check("tolerance widget keeps its new value",
                  abs(float(at.number_input[0].value) - 0.7) < 1e-9)

            # name the second cluster, then organize
            at = name_inputs(at)[1].set_value("Bob").run()
            check("run organizer button present",
                  any(b.label == "Run Organizer" for b in at.button))
            at = press(at, "Run Organizer")
            check("organize run clean", not at.exception,
                  str(at.exception))
            check("summary announced",
                  any("Placed" in s.value for s in at.success),
                  str([s.value for s in at.success]))
            folders = sorted(p.name for p in out_dir.iterdir()) \
                if out_dir.exists() else []
            check("person folders created",
                  folders == ["Alice", "Bob"], str(folders))
            placed = list(out_dir.glob("*/*")) if out_dir.exists() else []
            check("every photo placed", len(placed) == 4,
                  f"got {len(placed)}")
            check("organizing does not re-scan",
                  calls["pipeline"] == 1, f"calls={calls['pipeline']}")
            check("no errors reported", not at.error,
                  str([e.value for e in at.error]))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_auto_label_prefills_from_db():
    print()
    root = Path(tempfile.mkdtemp(prefix="faceorg_applab_"))
    try:
        photos = make_photos(root / "photos", count=2)
        records = fake_records(photos, split=2)   # both +1 -> one cluster
        db_path = str(root / "names.db")
        db = NamesDB(db_path)
        db.upsert_person("Alice", np.ones(4, dtype=np.float32),
                         sample_image_paths=[str(photos[0])])
        db.close()

        with patched(config={"input_folder": str(root / "photos"),
                             "output_folder": str(root / "out"),
                             "names_db": db_path},
                     records=records):
            at = open_app()
            at = press(at, "Scan photos & detect faces")
            check("auto-label scan clean", not at.exception,
                  str(at.exception))
            boxes = [s.value for s in at.success]
            check("auto-detection announced",
                  any("auto-detected" in b and "Alice" in b for b in boxes),
                  str(boxes))
            values = [(w.value or "") for w in name_inputs(at)]
            check("name input pre-filled from the DB",
                  values == ["Alice"], str(values))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_folder_change_requests_rescan():
    print()
    root = Path(tempfile.mkdtemp(prefix="faceorg_appchg_"))
    try:
        photos = make_photos(root / "photos")
        other = root / "other_folder"
        records = fake_records(photos)
        with patched(config={"input_folder": str(root / "photos"),
                             "output_folder": str(root / "out")},
                     records=records) as calls:
            at = open_app()
            at = press(at, "Scan photos & detect faces")
            check("first scan ran", calls["pipeline"] == 1,
                  f"calls={calls['pipeline']}")

            at = widget_by_label(at.text_input, "Input folder") \
                .set_value(str(other)).run()
            check("folder change render clean", not at.exception,
                  str(at.exception))
            check("re-scan offered after folder change", any(
                b.label == "Scan photos & detect faces" for b in at.button))
            check("folder change does not scan by itself",
                  calls["pipeline"] == 1, f"calls={calls['pipeline']}")
            check("stale clusters are hidden",
                  len(name_inputs(at)) == 0,
                  f"got {len(name_inputs(at))}")
            check("change is signalled", any(
                "input folder changed" in (w.value or "").lower()
                for w in at.warning),
                str([w.value for w in at.warning]))

            at = press(at, "Re-scan photos")
            check("re-scan button render clean", not at.exception,
                  str(at.exception))
            check("cached scan cleared",
                  "scan_stats" not in at.session_state)
            check("clearing the cache does not scan",
                  calls["pipeline"] == 1, f"calls={calls['pipeline']}")

            at = press(at, "Scan photos & detect faces")
            check("manual re-scan runs the pipeline again",
                  calls["pipeline"] == 2, f"calls={calls['pipeline']}")
            check("clusters visible again", len(name_inputs(at)) == 2,
                  f"got {len(name_inputs(at))}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_no_stray_files():
    print()
    stray = Path("facesort_names.db")
    check("no stray name DB left in the repo", not stray.exists(),
          str(stray))


def main():
    test_initial_render()
    test_scan_organize_and_caching()
    test_auto_label_prefills_from_db()
    test_folder_change_requests_rescan()
    test_no_stray_files()
    print()
    if _failures:
        print("FAILURES:", _failures)
        return 1
    print("All GUI (app) tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
