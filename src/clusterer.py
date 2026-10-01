"""Clustering engine — group the faces of the same person together (SPEC §7).

``FaceClusterer`` takes the embeddings produced by :mod:`src.embedder`
(plus the image path and bounding box of each face) and groups them with
DBSCAN using **cosine** distance.  ``tolerance`` from ``config.yaml`` is
DBSCAN's ``eps``; ``min_faces_per_cluster`` drops the tiny groups that are
almost certainly noise (SPEC §9).

Duplicate detections of the same face inside one photo are merged before
clustering (SPEC §10) — by IoU first, then by how close the box centres are,
which also catches one box nested inside another.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.cluster import DBSCAN

from .detector import box_iou

logger = logging.getLogger(__name__)

DEFAULT_TOLERANCE = 0.5          # config.yaml: tolerance (DBSCAN eps)
DEFAULT_MIN_FACES = 2            # config.yaml: min_faces_per_cluster
DEFAULT_IOU_THRESHOLD = 0.5      # duplicate boxes must overlap this much...
DEFAULT_PROXIMITY_RATIO = 0.35   # ...or have centres this close (of the smaller face)

NOISE_LABEL = -1                 # DBSCAN's "not clustered" label

__all__ = [
    "FaceRecord",
    "FaceCluster",
    "FaceClusterer",
    "cluster_embeddings",
    "deduplicate_records",
    "make_face_records",
]


@dataclass
class FaceRecord:
    """One face of one image (SPEC §8 data model).

    ``bbox`` is ``(x1, y1, x2, y2)`` in pixels; ``embedding`` is the
    L2-normalised fingerprint coming from :class:`src.embedder.FaceEmbedder`.
    ``cluster_id`` / ``person_name`` are filled in later by the clustering
    and naming steps.
    """

    image_path: Union[str, Path]
    bbox: Tuple[int, int, int, int]
    embedding: np.ndarray
    score: float = 0.0
    cluster_id: Optional[int] = None
    person_name: Optional[str] = None

    def __post_init__(self) -> None:
        self.image_path = Path(self.image_path)
        self.bbox = tuple(int(v) for v in self.bbox)
        self.embedding = np.asarray(self.embedding, dtype=np.float32).reshape(-1)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def diagonal(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return math.hypot(max(0, x2 - x1), max(0, y2 - y1))


@dataclass
class FaceCluster:
    """A group of faces that belong to the same person."""

    cluster_id: int
    faces: List[FaceRecord] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.faces)

    @property
    def image_paths(self) -> List[Path]:
        """Unique image paths in this cluster, first-seen order."""
        seen: Dict[Path, None] = {}
        for face in self.faces:
            seen.setdefault(face.image_path, None)
        return list(seen)

    @property
    def bboxes(self) -> List[Tuple[int, int, int, int]]:
        return [face.bbox for face in self.faces]

    @property
    def embeddings(self) -> np.ndarray:
        """Stacked embeddings, shape ``(size, dim)``."""
        if not self.faces:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([face.embedding for face in self.faces])

    @property
    def centroid(self) -> Optional[np.ndarray]:
        """Mean embedding of the cluster (re-normalised)."""
        if not self.faces:
            return None
        mean = self.embeddings.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm > 0 else mean


def _is_same_face(
    a: FaceRecord,
    b: FaceRecord,
    iou_threshold: float,
    proximity_ratio: float,
) -> bool:
    """True when two boxes of the *same image* describe the same face."""
    if box_iou(a.bbox, b.bbox) >= iou_threshold:
        return True
    # Nested boxes (a tight box inside a loose one) can have a low IoU
    # while still being the same face — compare centre distance instead.
    diagonal = min(a.diagonal, b.diagonal)
    if diagonal <= 0:
        return False
    ax, ay = a.center
    bx, by = b.center
    return math.hypot(ax - bx, ay - by) <= proximity_ratio * diagonal


def deduplicate_records(
    records: Sequence[FaceRecord],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    proximity_ratio: float = DEFAULT_PROXIMITY_RATIO,
) -> List[FaceRecord]:
    """Drop repeated detections of the same face within one image (SPEC §10).

    The highest-scoring box wins; the original input order is preserved.
    Faces in *different* images are never merged here.
    """
    by_image: Dict[Path, List[int]] = {}
    for index, record in enumerate(records):
        by_image.setdefault(record.image_path, []).append(index)

    kept: set = set()
    for path, indices in by_image.items():
        chosen: List[int] = []
        # Best score first, so the strongest detection survives the merge.
        for index in sorted(indices, key=lambda i: records[i].score, reverse=True):
            record = records[index]
            if all(
                not _is_same_face(record, records[other], iou_threshold, proximity_ratio)
                for other in chosen
            ):
                chosen.append(index)
                kept.add(index)
        dropped = len(indices) - len(chosen)
        if dropped:
            logger.debug("Merged %d duplicate detection(s) in %s", dropped, path)

    return [record for index, record in enumerate(records) if index in kept]


def _embedding_ok(embedding: np.ndarray) -> bool:
    return (
        embedding.ndim == 1
        and embedding.size > 0
        and bool(np.all(np.isfinite(embedding)))
        and float(np.linalg.norm(embedding)) > 0.0
    )


def cluster_embeddings(
    embeddings: Union[np.ndarray, Sequence[np.ndarray]],
    tolerance: float = DEFAULT_TOLERANCE,
    min_samples: int = 1,
) -> np.ndarray:
    """Run DBSCAN (cosine metric, ``eps = tolerance``) and return the labels.

    ``min_samples=1`` lets DBSCAN chain every face that fits inside
    ``tolerance``; the "too small to be a real person" filtering happens
    afterwards via ``min_faces_per_cluster``, which is what SPEC §9 asks for.
    """
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.size == 0:
        return np.empty(0, dtype=np.int64)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D embedding matrix, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("embeddings contain NaN or infinite values")
    if not (0.0 < float(tolerance) <= 2.0):
        raise ValueError(
            f"tolerance (DBSCAN eps) must be in (0, 2], got {tolerance}"
        )

    model = DBSCAN(
        eps=float(tolerance),
        min_samples=max(1, int(min_samples)),
        metric="cosine",
        n_jobs=-1,
    )
    return model.fit_predict(matrix)


class FaceClusterer:
    """Groups face records into clusters (one cluster per person)."""

    def __init__(
        self,
        tolerance: float = DEFAULT_TOLERANCE,
        min_faces_per_cluster: int = DEFAULT_MIN_FACES,
        min_samples: int = 1,
        deduplicate: bool = True,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        proximity_ratio: float = DEFAULT_PROXIMITY_RATIO,
    ) -> None:
        self.tolerance = float(tolerance)
        self.min_faces_per_cluster = max(1, int(min_faces_per_cluster))
        self.min_samples = int(min_samples)
        self.deduplicate = deduplicate
        self.iou_threshold = iou_threshold
        self.proximity_ratio = proximity_ratio

    @classmethod
    def from_config(cls, config: Mapping) -> "FaceClusterer":
        """Build a clusterer from the ``config.yaml`` mapping."""
        return cls(
            tolerance=config.get("tolerance", DEFAULT_TOLERANCE),
            min_faces_per_cluster=config.get(
                "min_faces_per_cluster", DEFAULT_MIN_FACES
            ),
        )

    def cluster(self, records: Sequence[FaceRecord]) -> List[FaceCluster]:
        """Cluster ``records`` and return the clusters worth keeping.

        Clusters with fewer than ``min_faces_per_cluster`` faces are ignored,
        as are faces whose embedding is unusable (NaN/zero) or was a
        duplicate detection inside its own image.
        """
        records = list(records)
        if not records:
            logger.info("Nothing to cluster: no faces were provided.")
            return []

        if self.deduplicate:
            records = deduplicate_records(
                records, self.iou_threshold, self.proximity_ratio
            )

        usable = [record for record in records if _embedding_ok(record.embedding)]
        if len(usable) != len(records):
            logger.warning(
                "Ignoring %d face(s) with an unusable embedding.",
                len(records) - len(usable),
            )
        if not usable:
            logger.info("Nothing to cluster: no usable face embeddings.")
            return []

        labels = cluster_embeddings(
            np.stack([record.embedding for record in usable]),
            tolerance=self.tolerance,
            min_samples=self.min_samples,
        )

        # Group by label, keeping the order in which each group first appears.
        groups: Dict[int, List[int]] = {}
        for index, label in enumerate(labels):
            groups.setdefault(int(label), []).append(index)
        if NOISE_LABEL in groups:
            # DBSCAN noise points become singletons; the size filter below
            # then decides whether they are worth keeping.
            for index in groups.pop(NOISE_LABEL):
                groups.setdefault(NOISE_LABEL - index, []).append(index)
        ordered = sorted(groups.values(), key=lambda members: members[0])

        clusters: List[FaceCluster] = []
        ignored_faces = 0
        for members in ordered:
            faces = [usable[index] for index in members]
            if len(faces) < self.min_faces_per_cluster:
                ignored_faces += len(faces)
                continue
            cluster_id = len(clusters)
            for face in faces:
                face.cluster_id = cluster_id
            clusters.append(FaceCluster(cluster_id=cluster_id, faces=faces))

        logger.info(
            "Clustered %d face(s) -> %d cluster(s) kept, %d face(s) ignored "
            "(tolerance=%.2f cosine, min_faces_per_cluster=%d).",
            len(usable),
            len(clusters),
            ignored_faces,
            self.tolerance,
            self.min_faces_per_cluster,
        )
        return clusters


def make_face_records(
    image_path: Union[str, Path],
    face_embeddings: Sequence,
) -> List[FaceRecord]:
    """Convert one image's ``FaceEmbedding`` items into ``FaceRecord``s.

    Convenience glue between :meth:`src.embedder.FaceEmbedder.embed` and
    :meth:`FaceClusterer.cluster`.
    """
    return [
        FaceRecord(
            image_path=image_path,
            bbox=item.detection.bbox,
            embedding=item.embedding,
            score=item.detection.score,
        )
        for item in face_embeddings
    ]
