"""Unit tests for the vision pipeline (SPEC §7) — no model weights needed.

Run with::

    python tests/test_pipeline.py      # plain runner, no extra dependencies
    pytest tests/                      # also collectable by pytest

The real ``buffalo_l`` weights are not required: the insightface model is
injected as a small fake so the detection/embedding logic can be checked
in milliseconds.
"""

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from src.detector import DetectedFace, FaceDetector, deduplicate, iou
from src.embedder import FaceEmbedder, FaceEmbeddingError
from src.image_loader import load_image, scan_images

logging.basicConfig(level=logging.CRITICAL)  # keep the expected warnings quiet

_failures = []


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}: {label} {detail}")
    if not condition:
        _failures.append(label)


# ------------------------------------------------------------------ fixtures
def _make_input_folder():
    root = Path(tempfile.mkdtemp(prefix="faceorg_test_"))
    (root / "sub").mkdir()
    Image.new("RGB", (32, 32), (255, 0, 0)).save(root / "a.jpg")
    Image.new("RGB", (16, 16), (0, 255, 0)).save(root / "b.png")
    Image.new("RGB", (8, 8)).save(root / "sub" / "c.webp")
    (root / "broken.jpg").write_bytes(b"not an image at all")
    (root / "notes.txt").write_text("ignore me")
    return root


class FakeDetModel:
    def __init__(self, bboxes, kpss):
        self.bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 5)
        self.kpss = kpss

    def detect(self, image, max_num=0, metric="default"):
        assert max_num == 0
        return self.bboxes, self.kpss


class FakeAnalysis:
    """Mimics insightface's FaceAnalysis for detection-only tests."""

    def __init__(self, bboxes, kpss=None):
        self.det_model = FakeDetModel(bboxes, kpss)
        self.models = {}


class FakeRecognition:
    def __init__(self, fail=False):
        self.fail = fail

    def get(self, image, face):
        if self.fail:
            raise RuntimeError("boom")
        face.embedding = np.full(512, 3.0, dtype=np.float32)
        return face.embedding


class FakeRecognitionAnalysis:
    def __init__(self, fail=False):
        self.models = {"recognition": FakeRecognition(fail)}


KPS = np.array(
    [[10, 10], [30, 10], [30, 30], [10, 30], [20, 20]], dtype=np.float32
)
FRAME = np.zeros((100, 100, 3), dtype=np.uint8)


# --------------------------------------------------------------- image_loader
def test_image_loader():
    root = _make_input_folder()
    try:
        paths = list(scan_images(root))
        names = [p.name for p in paths]
        check(
            "scan yields only readable images",
            names == ["a.jpg", "b.png", "c.webp"],
            str(names),
        )
        check("corrupt file skipped without raising", "broken.jpg" not in names)
        check("unsupported extension ignored", "notes.txt" not in names)
        check("recursive by default", any(p.parent.name == "sub" for p in paths))

        flat = [p.name for p in scan_images(root, recursive=False)]
        check("recursive=False stays at top level", sorted(flat) == ["a.jpg", "b.png"])

        img = load_image(root / "a.jpg")
        check(
            "load_image decodes to a BGR array",
            isinstance(img, np.ndarray) and img.shape == (32, 32, 3),
        )
        check("load_image returns None for a corrupt file",
              load_image(root / "broken.jpg") is None)

        try:
            list(scan_images(root / "missing"))
            check("missing folder raises FileNotFoundError", False)
        except FileNotFoundError:
            check("missing folder raises FileNotFoundError", True)
    finally:
        import shutil

        shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------------------------- detector
def test_detector():
    analysis = FakeAnalysis(
        [
            [5, 5, 45, 45, 0.98],      # face A (best box)
            [7, 6, 46, 44, 0.71],      # face A again -> duplicate
            [60, 60, 95, 95, 0.99],    # face B
        ],
        [KPS, KPS, KPS + 55],
    )
    faces = FaceDetector(analysis=analysis).detect(FRAME)
    check("detects and de-duplicates to 2 faces", len(faces) == 2,
          str([f.bbox for f in faces]))
    check("keeps the strongest duplicate",
          abs(faces[0].score - 0.98) < 1e-5)
    check("bbox parsed as (x1, y1, x2, y2)", faces[0].bbox == (5, 5, 45, 45))
    check(
        "SPEC §8 aliases map correctly",
        (faces[0].left, faces[0].top, faces[0].right, faces[0].bottom)
        == (5, 5, 45, 45),
    )
    check("landmarks kept", faces[0].kps is not None and faces[0].kps.shape == (5, 2))

    check("image with no faces -> empty list",
          FaceDetector(analysis=FakeAnalysis([])).detect(FRAME) == [])
    check("unreadable image -> empty list, no crash",
          FaceDetector(analysis=analysis).detect(None) == [])

    clipped = FaceDetector(
        analysis=FakeAnalysis(
            [
                [-20, -20, 10, 10, 0.9],   # partially outside -> clipped
                [500, 500, 600, 600, 0.9],  # fully outside -> dropped
                [90, 90, 95, 95, 0.9],
            ]
        )
    ).detect(FRAME)
    check(
        "out-of-frame boxes clipped or dropped",
        [f.bbox for f in clipped] == [(0, 0, 10, 10), (90, 90, 95, 95)],
        str([f.bbox for f in clipped]),
    )

    a = DetectedFace(bbox=(0, 0, 10, 10), score=0.9)
    b = DetectedFace(bbox=(5, 0, 15, 10), score=0.8)  # 33% overlap: other face
    d = DetectedFace(bbox=(1, 1, 11, 11), score=0.7)  # 69% overlap: duplicate
    c = DetectedFace(bbox=(50, 50, 60, 60), score=0.6)
    check("iou for overlapping boxes", abs(iou(a, b) - 50 / 150) < 1e-9)
    check("iou for disjoint boxes", iou(a, c) == 0.0)
    check("dedupe drops only true duplicates and keeps order",
          deduplicate([a, b, d, c]) == [a, b, c])


# ------------------------------------------------------------------- embedder
def test_embedder():
    detector = FaceDetector(
        analysis=FakeAnalysis([[5, 5, 45, 45, 0.98], [60, 60, 95, 95, 0.99]],
                              [KPS, KPS + 55])
    )
    detected = detector.detect(FRAME)
    embedder = FaceEmbedder(analysis=FakeRecognitionAnalysis())
    vectors = embedder.embed(FRAME, detected)

    check("one embedding per face", len(vectors) == len(detected))
    check("embedding is 512-D float32",
          vectors[0].embedding.shape == (512,)
          and vectors[0].embedding.dtype == np.float32)
    check("embedding is L2-normalised",
          abs(float(np.linalg.norm(vectors[0].embedding)) - 1.0) < 1e-5)
    check("detection travels with its embedding",
          vectors[0].detection is detected[0])
    check("no faces -> no embeddings", embedder.embed(FRAME, []) == [])
    check("missing image -> no embeddings", embedder.embed(None, detected) == [])

    no_kps = DetectedFace(bbox=(0, 0, 10, 10), score=0.9, kps=None)
    partial = embedder.embed(FRAME, [no_kps, detected[0]])
    check("face without landmarks is skipped, not fatal",
          len(partial) == 1 and partial[0].detection is detected[0])

    failing = FaceEmbedder(analysis=FakeRecognitionAnalysis(fail=True))
    check("recognition failure is logged and skipped",
          failing.embed(FRAME, detected) == [])

    try:
        embedder.embed_one(FRAME, no_kps)
        check("embed_one raises FaceEmbeddingError", False)
    except FaceEmbeddingError:
        check("embed_one raises FaceEmbeddingError", True)


def main():
    test_image_loader()
    test_detector()
    test_embedder()
    print()
    if _failures:
        print("FAILURES:", _failures)
        return 1
    print("All pipeline tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
