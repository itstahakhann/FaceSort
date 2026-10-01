"""Unit tests for the multiprocess vision stage (SPEC M6).

Run with::

    python tests/test_parallel.py     # plain runner, no extra dependencies
    pytest tests/                     # also collectable by pytest

The real ``buffalo_l`` weights are not required: detection/embedding are
injected as fakes, and the one test that really spawns worker processes
uses unreadable paths (workers exit before any model load) and skips
itself when the weights are absent.
"""

import logging
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

import src.detector as detector_mod
import src.embedder as embedder_mod
import src.main as main_mod
from src.clusterer import make_face_records
from src.detector import (
    DEFAULT_MAX_WORKERS,
    MIN_PARALLEL_IMAGES,
    DetectedFace,
    ProcessedImage,
    process_image,
    process_images,
    resolve_workers,
)
from src.embedder import FaceEmbedding
from src.face_model import ModelsNotInstalledError, models_installed
from src.main import apply_overrides, build_parser, run_pipeline

logging.basicConfig(level=logging.CRITICAL)  # keep the expected warnings quiet
try:  # cv2 prints "can't open/read file" straight to stderr; keep it quiet
    import cv2.utils.logging as _cv2_log

    _cv2_log.setLogLevel(_cv2_log.LOG_LEVEL_SILENT)
except Exception:  # pragma: no cover - older OpenCV builds
    pass

_failures = []


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}: {label} {detail}")
    if not condition:
        _failures.append(label)


KPS = np.array(
    [[10, 10], [30, 10], [30, 30], [10, 30], [20, 20]], dtype=np.float32
)


# ------------------------------------------------------------------ fixtures
def _photo_folder():
    root = Path(tempfile.mkdtemp(prefix="faceorg_parallel_"))
    Image.new("RGB", (24, 24), (10, 20, 30)).save(root / "good.jpg")
    (root / "corrupt.jpg").write_bytes(b"definitely not a jpeg")
    return root


class _FakeDetector:
    """Stands in for FaceDetector — one face, or an error on demand."""

    explode = False

    def __init__(self, *args, **kwargs):
        pass

    def detect(self, image):
        if type(self).explode:
            raise RuntimeError("detector exploded")
        return [DetectedFace(bbox=(1, 1, 9, 9), score=0.99, kps=KPS)]


class _FakeEmbedder:
    """Stands in for FaceEmbedder — constant 512-D fingerprints."""

    explode = False

    def __init__(self, *args, **kwargs):
        pass

    def embed(self, image, detections):
        if type(self).explode:
            raise RuntimeError("embedder exploded")
        return [
            FaceEmbedding(detection=d,
                          embedding=np.ones(512, dtype=np.float32))
            for d in detections
        ]


class _fake_face_model:
    """Patches the per-image vision stage; use with ``with``."""

    def __init__(self, detector=_FakeDetector, embedder=_FakeEmbedder):
        self.detector = detector
        self.embedder = embedder
        self._saved = None

    def __enter__(self):
        self._saved = (detector_mod.FaceDetector, embedder_mod.FaceEmbedder)
        detector_mod.FaceDetector = self.detector
        embedder_mod.FaceEmbedder = self.embedder
        return self

    def __exit__(self, *exc):
        detector_mod.FaceDetector, embedder_mod.FaceEmbedder = self._saved
        return False


# ---------------------------------------------------------------------- tests
def test_resolve_workers():
    import os

    cpus = os.cpu_count() or 1
    expected_auto = max(1, min(cpus // 2, DEFAULT_MAX_WORKERS))
    check("auto = min(cpu//2, DEFAULT_MAX_WORKERS)",
          resolve_workers(0) == expected_auto,
          f"(got {resolve_workers(0)}, cpus={cpus})")
    check("None is also auto", resolve_workers(None) == resolve_workers(0))
    check("auto is always at least 1", resolve_workers(0) >= 1)
    check("explicit 1 stays 1", resolve_workers(1) == 1)
    check("explicit 2 stays 2", resolve_workers(2) == min(2, cpus))
    check("explicit 99 clamps to cpu count", resolve_workers(99) == cpus)
    check("garbage falls back to auto", resolve_workers("x") == resolve_workers(0))
    check("negative falls back to auto", resolve_workers(-3) == resolve_workers(0))
    check("threshold is sane", 1 <= MIN_PARALLEL_IMAGES <= 100)


def test_cli_workers_flag():
    parser = build_parser()
    args = parser.parse_args(["--workers", "2"])
    config = {"workers": 0}
    apply_overrides(config, args)
    check("--workers overrides config", config["workers"] == 2)
    config = {"workers": 0}
    apply_overrides(config, parser.parse_args([]))
    check("no --workers leaves config alone", config["workers"] == 0)


def test_process_image_ok_and_pickleable():
    root = _photo_folder()
    with _fake_face_model():
        result = process_image(str(root / "good.jpg"))
    check("good photo -> status ok", result.status == "ok", result.status)
    check("faces detected", result.faces_detected == 1)
    check("one embedding", len(result.embeddings) == 1)
    check("error is empty", result.error == "")

    records = make_face_records(result.path, result.embeddings)
    check("records glue still works", len(records) == 1
          and records[0].bbox == (1, 1, 9, 9))

    # spawn workers pickle every result back to the parent
    try:
        roundtrip = pickle.loads(pickle.dumps(result))
        check("ProcessedImage survives pickling",
              roundtrip.path == result.path
              and roundtrip.faces_detected == result.faces_detected
              and len(roundtrip.embeddings) == 1)
    except Exception as exc:
        check("ProcessedImage survives pickling", False, str(exc))


def test_process_image_unreadable():
    root = _photo_folder()
    corrupt = process_image(str(root / "corrupt.jpg"))
    missing = process_image(str(root / "no_such.jpg"))
    check("corrupt file -> unreadable", corrupt.status == "unreadable")
    check("missing file -> unreadable", missing.status == "unreadable")
    check("unreadable carries no faces",
          corrupt.faces_detected == 0 and not corrupt.embeddings)


def test_process_image_never_raises():
    root = _photo_folder()
    with _fake_face_model(detector=type(
            "Boom", (_FakeDetector,),
            {"explode": True, "__init__": lambda self, *a, **k: None})):
        result = process_image(str(root / "good.jpg"))
    check("detector error -> status failed", result.status == "failed")
    check("failure keeps the message", "detector exploded" in result.error)
    check("failed result has no faces", result.faces_detected == 0)

    boom_embed = type("BoomEmbed", (_FakeEmbedder,),
                       {"explode": True,
                        "__init__": lambda self, *a, **k: None})
    with _fake_face_model(embedder=boom_embed):
        result = process_image(str(root / "good.jpg"))
    check("embed error -> status failed", result.status == "failed")
    check("embed failure counted as detected-then-failed",
          result.status == "failed" and result.faces_detected == 0)


def test_inline_mode_keeps_order():
    root = _photo_folder()
    paths = [str(root / "good.jpg"), str(root / "corrupt.jpg"),
             str(root / "missing.jpg")]
    with _fake_face_model():
        results = list(process_images(paths, workers=1))
    check("workers=1 yields one result per path", len(results) == 3)
    check("workers=1 keeps input order",
          [r.path for r in results] == paths)
    check("workers=1 statuses inline",
          [r.status for r in results] == ["ok", "unreadable", "unreadable"])


def test_small_scan_stays_inline():
    # 3 paths < MIN_PARALLEL_IMAGES: auto workers must not spawn a pool,
    # which an inline run proves by preserving the input order.
    root = _photo_folder()
    paths = [str(root / "corrupt.jpg"), str(root / "nope1.jpg"),
             str(root / "nope2.jpg")]
    results = list(process_images(paths, workers=0))
    check("below threshold -> one result per path", len(results) == 3)
    check("below threshold stays ordered (no pool)",
          [r.path for r in results] == paths)
    check("below threshold statuses",
          all(r.status == "unreadable" for r in results))


def test_empty_input():
    check("no paths -> no results", list(process_images([], workers=0)) == [])


def test_missing_models_fail_fast():
    saved = detector_mod.require_models

    def _raise():
        raise ModelsNotInstalledError("model gone")

    detector_mod.require_models = _raise
    try:
        paths = [f"missing_{i}.jpg" for i in range(MIN_PARALLEL_IMAGES + 6)]
        try:
            list(process_images(paths, workers=0))
            raised = False
        except ModelsNotInstalledError:
            raised = True
        check("pool branch raises ModelsNotInstalledError before spawning",
              raised)
    finally:
        detector_mod.require_models = saved


def test_real_spawn_pool():
    """The actual M6 path: spawn workers and collect results."""
    if not models_installed():
        print("SKIP: buffalo_l weights not installed; pool test skipped")
        return
    paths = [f"definitely_missing_{i}.jpg"
             for i in range(MIN_PARALLEL_IMAGES + 6)]
    results = list(process_images(paths, workers=0))  # auto -> a real pool
    check("pool yields one result per path", len(results) == len(paths))
    check("pool results all unreadable",
          all(r.status == "unreadable" for r in results))
    check("pool covers every path", {r.path for r in results} == set(paths))


def test_run_pipeline_stats_mapping():
    """run_pipeline must fold ProcessImageResults into the F10 counters."""
    root = _photo_folder()
    good = str(root / "good.jpg")

    def fake_scan(_folder):
        return [Path(good), Path(root / "missing.jpg"),
                Path(root / "odd.jpg"), Path(root / "blank.jpg")]

    def fake_process(paths, workers=None):
        two = [
            FaceEmbedding(detection=DetectedFace(
                bbox=(1, 1, 9, 9), score=0.9, kps=KPS),
                embedding=np.ones(512, dtype=np.float32)),
            FaceEmbedding(detection=DetectedFace(
                bbox=(2, 2, 8, 8), score=0.8, kps=KPS),
                embedding=np.ones(512, dtype=np.float32)),
        ]
        yield ProcessedImage(path=good, status="ok",
                             faces_detected=2, embeddings=two)
        yield ProcessedImage(path=str(root / "missing.jpg"),
                             status="unreadable")
        yield ProcessedImage(path=str(root / "odd.jpg"), status="failed",
                             error="boom")
        yield ProcessedImage(path=str(root / "blank.jpg"), status="ok",
                             faces_detected=0)

    saved = (main_mod.scan_images, main_mod.process_images,
             main_mod.require_models)
    main_mod.scan_images = fake_scan
    main_mod.process_images = fake_process
    main_mod.require_models = lambda: None
    seen = []
    try:
        stats = run_pipeline({"input_folder": str(root), "workers": 1},
                             progress_cb=lambda d, t, p: seen.append((d, t, p)))
    finally:
        (main_mod.scan_images, main_mod.process_images,
         main_mod.require_models) = saved

    check("scanned every yielded path", stats.images_scanned == 4)
    check("unreadable counted", stats.images_unreadable == 1)
    check("failed counted", stats.images_failed == 1)
    # Parity with the old sequential loop: a failed image is "no usable
    # faces" too, an unreadable one is not scanned at all.
    check("without_faces counts failed + empty ok",
          stats.images_without_faces == 2, stats.images_without_faces)
    check("faces detected", stats.faces_detected == 2)
    check("faces embedded", stats.faces_embedded == 2)
    check("records built from ok results", len(stats.records) == 2)
    check("progress callback per photo",
          seen == [(1, 4, good), (2, 4, str(root / "missing.jpg")),
                   (3, 4, str(root / "odd.jpg")),
                   (4, 4, str(root / "blank.jpg"))], seen)
    check("seconds recorded", stats.seconds >= 0.0)


def main() -> int:
    test_resolve_workers()
    test_cli_workers_flag()
    test_process_image_ok_and_pickleable()
    test_process_image_unreadable()
    test_process_image_never_raises()
    test_inline_mode_keeps_order()
    test_small_scan_stays_inline()
    test_empty_input()
    test_missing_models_fail_fast()
    test_real_spawn_pool()
    test_run_pipeline_stats_mapping()
    print()
    if _failures:
        print("FAILURES:", _failures)
        return 1
    print("All parallel (M6) tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
