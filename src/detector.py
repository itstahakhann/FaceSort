"""Face detection — find 0..N faces per image (insightface RetinaFace, CPU).

An image that contains no face is not an error: :meth:`FaceDetector.detect`
simply returns an empty list.  Near-identical detections of the same face
(overlapping boxes) are merged by score, per SPEC §10.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .face_model import (
    DEFAULT_DET_SIZE,
    DEFAULT_DET_THRESH,
    get_face_analysis,
    require_models,
    set_intra_op_threads,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DetectedFace",
    "FaceDetector",
    "ProcessedImage",
    "DEFAULT_MAX_WORKERS",
    "MIN_PARALLEL_IMAGES",
    "resolve_workers",
    "process_image",
    "process_images",
    "iou",
    "deduplicate",
]


@dataclass(frozen=True)
class DetectedFace:
    """A single detected face.

    ``bbox`` uses the OpenCV ``(x1, y1, x2, y2)`` ordering (left, top,
    right, bottom in pixels); the ``left``/``top``/``right``/``bottom``
    properties expose the same values in the SPEC §8 naming.
    """

    bbox: Tuple[int, int, int, int]
    score: float
    kps: Optional[np.ndarray] = None  # 5x2 landmarks, when available

    @property
    def left(self) -> int:
        return self.bbox[0]

    @property
    def top(self) -> int:
        return self.bbox[1]

    @property
    def right(self) -> int:
        return self.bbox[2]

    @property
    def bottom(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_xyxy(self) -> np.ndarray:
        return np.asarray(self.bbox, dtype=np.int32)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def iou(a: DetectedFace, b: DetectedFace) -> float:
    """Intersection-over-union of two detections (0.0 when disjoint)."""
    return box_iou(a.bbox, b.bbox)


def deduplicate(
    faces: Sequence[DetectedFace], iou_threshold: float = 0.5
) -> List[DetectedFace]:
    """Drop duplicate detections of the same face (SPEC §10).

    The highest-scoring box wins; the original detection order is kept.
    """
    indexed = list(enumerate(faces))
    kept: List[int] = []
    for idx, face in sorted(indexed, key=lambda item: item[1].score, reverse=True):
        if all(iou(face, faces[j]) < iou_threshold for j in kept):
            kept.append(idx)
    kept.sort()
    return [faces[i] for i in kept]


class FaceDetector:
    """Detects faces with the buffalo_l RetinaFace/SCRFD model on the CPU."""

    def __init__(
        self,
        det_thresh: float = DEFAULT_DET_THRESH,
        det_size: Tuple[int, int] = DEFAULT_DET_SIZE,
        iou_threshold: float = 0.5,
        analysis: Any = None,
    ) -> None:
        self.det_thresh = det_thresh
        self.det_size = det_size
        self.iou_threshold = iou_threshold
        self._analysis = analysis

    @property
    def analysis(self) -> Any:
        """The shared insightface ``FaceAnalysis`` instance (built lazily)."""
        if self._analysis is None:
            self._analysis = get_face_analysis(
                det_thresh=self.det_thresh, det_size=self.det_size
            )
        return self._analysis

    def detect(self, image: Optional[np.ndarray]) -> List[DetectedFace]:
        """Return 0..N faces found in ``image`` (BGR, ``uint8``).

        Images without faces, unreadable input, or empty frames yield an
        empty list rather than raising.
        """
        if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim < 2:
            logger.warning("detect() called without a usable image; returning no faces.")
            return []

        bboxes, kpss = self.analysis.det_model.detect(image, max_num=0)
        if bboxes is None or len(bboxes) == 0:
            logger.debug("No faces found in image of shape %s", image.shape)
            return []

        height, width = image.shape[:2]
        faces: List[DetectedFace] = []
        for index, row in enumerate(bboxes):
            x1, y1, x2, y2 = (int(round(float(v))) for v in row[:4])
            score = float(row[4]) if len(row) > 4 else 1.0
            if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
                logger.debug("Dropping out-of-frame box %s", (x1, y1, x2, y2))
                continue
            # Keep the box inside the frame so later crops cannot go out of
            # bounds.
            x1 = min(max(x1, 0), width - 1)
            x2 = min(max(x2, 0), width)
            y1 = min(max(y1, 0), height - 1)
            y2 = min(max(y2, 0), height)
            if x2 <= x1 or y2 <= y1:
                logger.debug("Dropping degenerate box %s", (x1, y1, x2, y2))
                continue

            kps = None
            if kpss is not None and len(kpss) > index:
                kps = np.asarray(kpss[index], dtype=np.float32)

            faces.append(
                DetectedFace(bbox=(x1, y1, x2, y2), score=score, kps=kps)
            )

        deduped = deduplicate(faces, self.iou_threshold)
        if len(deduped) != len(faces):
            logger.debug(
                "Merged duplicate detections: %d -> %d", len(faces), len(deduped)
            )
        return deduped


# ---------------------------------------------------------------------------
# Multiprocessing (M6 — SPEC §10: ">10k images → batch processing")
# ---------------------------------------------------------------------------

#: Auto never exceeds this many workers (SPEC M6 says "large folders"; each
#: worker holds its own copy of the ONNX models, so more just wastes RAM).
DEFAULT_MAX_WORKERS = 4

#: Below this many photos, the per-worker model start-up (~5-10 s each)
#: costs more than the pool saves, so small scans stay single-process.
MIN_PARALLEL_IMAGES = 24


def resolve_workers(requested: Optional[int] = None) -> int:
    """Number of worker processes for the vision stage (0 = auto).

    Auto picks ``min(cpu_count // 2, DEFAULT_MAX_WORKERS)``: the ONNX
    graphs benchmark best with ~2 intra-op threads each (more adds sync
    overhead, less leaves cores idle), and ``process_images`` gives every
    worker ``cpu_count // workers`` threads — so halving the CPU count
    lands each worker on that sweet spot.  Explicit values are clamped to
    ``1..cpu_count``; ``1`` disables the pool entirely.
    """
    cpus = os.cpu_count() or 1
    try:
        wanted = int(requested) if requested is not None else 0
    except (TypeError, ValueError):
        wanted = 0
    if wanted <= 0:
        return max(1, min(cpus // 2, DEFAULT_MAX_WORKERS))
    return max(1, min(wanted, cpus))


@dataclass
class ProcessedImage:
    """One photo after the per-image vision stage (load → detect → embed).

    ``status`` is ``"ok"``, ``"unreadable"`` (the loader refused the file)
    or ``"failed"`` (an unexpected error, described in ``error``).  The
    stage never raises — one bad photo becomes a failed result, exactly
    like the sequential pipeline's per-image ``try/except``.
    """

    path: str
    status: str = "ok"
    faces_detected: int = 0
    #: ``List[FaceEmbedding]`` — typed loosely because src.embedder imports
    #: this module (deferred import inside process_image breaks the cycle).
    embeddings: List[Any] = field(default_factory=list)
    error: str = ""


def process_image(path: str) -> ProcessedImage:
    """Load, detect and embed a single photo.  Never raises.

    This is the unit of work handed to pool workers; it also runs inline
    when the scan is too small (or ``workers == 1``) to be worth a pool.
    """
    from .embedder import FaceEmbedder  # deferred: src.embedder imports us
    from .image_loader import load_image

    try:
        image = load_image(path)
        if image is None:
            return ProcessedImage(path=path, status="unreadable")

        detections = FaceDetector().detect(image)
        embeddings = FaceEmbedder().embed(image, detections)
        return ProcessedImage(
            path=path,
            faces_detected=len(detections),
            embeddings=list(embeddings),
        )
    except Exception as exc:  # one odd photo must not abort the run
        logger.warning("Skipping %s: %s", path, exc)
        return ProcessedImage(path=path, status="failed", error=str(exc))


def _init_worker(threads: int, level: int) -> None:
    """Run once per spawned worker (M6).

    Caps every thread pool a worker would otherwise size to *all* cores:
    ONNX Runtime gets ``threads`` intra-op threads (N workers x all-cores
    would oversubscribe the CPU), and OpenCV/BLAS are pinned to one thread
    each — their parallel regions are tiny (decodes, warps, norms) and
    their per-worker thread pools would just fight between workers.
    """
    logging.basicConfig(
        level=level,
        format="%(levelname)-7s %(name)s: %(message)s",
    )
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:  # pragma: no cover - cv2 always present, but be safe
        logger.debug("could not limit cv2 threads", exc_info=True)
    set_intra_op_threads(threads)


def process_images(
    paths: Sequence[Any], workers: Optional[int] = None
) -> Iterator[ProcessedImage]:
    """Yield :func:`process_image` results for every path, in parallel
    when it pays off (SPEC M6).

    * fewer than :data:`MIN_PARALLEL_IMAGES` photos, or ``workers == 1``:
      results stream through this process — no pool, no extra model loads;
    * otherwise a ``spawn``-based pool of ``resolve_workers(workers)``
      processes runs the photos concurrently, yielding each result as it
      completes (unordered).

    ``spawn`` (rather than ``fork``) is what Windows uses anyway, and it is
    the safe choice everywhere with ONNX Runtime's thread pools and with
    PyInstaller-frozen executables.
    """
    path_list = [str(p) for p in paths]
    if not path_list:
        return

    count = resolve_workers(workers)
    if count <= 1 or len(path_list) < MIN_PARALLEL_IMAGES:
        for path in path_list:
            yield process_image(path)
        return

    # Fail fast with the one clear "model missing" message instead of one
    # warning per photo from every worker.
    require_models()

    threads = max(1, (os.cpu_count() or 1) // count)
    level = logging.getLogger().level or logging.WARNING
    logger.info(
        "Processing %d photos with %d worker processes (%d ORT thread(s) "
        "each) ...",
        len(path_list),
        count,
        threads,
    )

    import multiprocessing as mp  # deferred: single-process runs skip it

    # Spawned children inherit this process's environment the moment they
    # start, i.e. before they import numpy/OpenBLAS — pin those pools to
    # one thread as well (each worker's vector work is tiny; full-core BLAS
    # pools per worker would only fight each other).  Restored in `finally`
    # so the parent's later stages (DBSCAN) run untouched.
    env_names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    saved_env = {name: os.environ.get(name) for name in env_names}
    for name in env_names:
        os.environ[name] = "1"

    try:
        context = mp.get_context("spawn")
        with context.Pool(
            count, initializer=_init_worker, initargs=(threads, level)
        ) as pool:
            for result in pool.imap_unordered(
                process_image, path_list, chunksize=1
            ):
                yield result
    finally:
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
