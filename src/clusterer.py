"""Clustering engine — group the faces of the same person together (SPEC §7).

``FaceClusterer`` takes the embeddings produced by :mod:`src.embedder`
(plus the image path and bounding box of each face) and groups them with
DBSCAN.  ``tolerance`` from ``config.yaml`` is DBSCAN's ``eps``;
``min_faces_per_cluster`` drops the tiny groups that are almost certainly
noise (SPEC §9).

Age invariance
--------------
The distance is a **fused** one: a blend of whole-face and periocular
similarity, weighted toward the eye region because that part of the face
changes least between a childhood photo and an adult one (see
:mod:`src.fusion` for the reasoning and the fallback rules).  Without this,
one person's childhood and adult photos routinely land in separate clusters.

Because the fused distance is only defined for pairs that *both* have an eye
embedding, the two modalities are clustered separately and the results are
then reconciled by group:

* the periocular pass (when enough faces have eye embeddings) proposes its
  own clusters;
* the whole-face pass proposes its own;
* the two label sets are aligned by their mutual agreement, so a person the
  eye region already agrees on wins, and the whole-face pass only breaks
  ties the eye region is unsure about.

This keeps both signals available — a pair with two missing eye embeddings is
still comparable via the fused fallback (whole-face only), which is what stops
low-resolution childhood photos from being dropped.

Duplicate detections of the same face inside one photo are merged before
clustering (SPEC §10) — by IoU first, then by how close the box centres are,
which also catches one box nested inside another.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.cluster import DBSCAN

from .detector import box_iou
from .fusion import (
    DEFAULT_WEIGHT_EYE,
    DEFAULT_WEIGHT_FULL,
    FusedMetric,
    validate_weights,
)

logger = logging.getLogger(__name__)

DEFAULT_TOLERANCE = 0.5          # config.yaml: tolerance (DBSCAN eps)
DEFAULT_MIN_FACES = 2            # config.yaml: min_faces_per_cluster
DEFAULT_IOU_THRESHOLD = 0.5      # duplicate boxes must overlap this much...
DEFAULT_PROXIMITY_RATIO = 0.35   # ...or have centres this close (of the smaller face)

#: Below this many faces carrying an eye embedding, the periocular pass is
#: skipped: a handful of vectors cannot establish a reliable grouping, so the
#: whole-face pass alone is more trustworthy.
MIN_EYE_FACES = 2

NOISE_LABEL = -1                 # DBSCAN's "not clustered" label

__all__ = [
    "FaceRecord",
    "FaceCluster",
    "FaceClusterer",
    "cluster_embeddings",
    "deduplicate_records",
    "make_face_records",
    "relabel_ages",
]


@dataclass
class FaceRecord:
    """One face of one image (SPEC §8 data model).

    ``bbox`` is ``(x1, y1, x2, y2)`` in pixels; ``embedding`` is the
    L2-normalised whole-face fingerprint coming from
    :class:`src.embedder.FaceEmbedder`.  ``eye_embedding`` is the periocular
    vector for the same face, or ``None`` when the region was too small or
    too blurred to fingerprint — a normal outcome for small childhood photos,
    and never an error.
    ``cluster_id`` / ``person_name`` are filled in later by the clustering
    and naming steps.
    """

    image_path: Union[str, Path]
    bbox: Tuple[int, int, int, int]
    embedding: np.ndarray
    eye_embedding: Optional[np.ndarray] = None
    score: float = 0.0
    cluster_id: Optional[int] = None
    person_name: Optional[str] = None

    def __post_init__(self) -> None:
        self.image_path = Path(self.image_path)
        self.bbox = tuple(int(v) for v in self.bbox)
        self.embedding = np.asarray(self.embedding, dtype=np.float32).reshape(-1)
        if self.eye_embedding is not None:
            self.eye_embedding = np.asarray(
                self.eye_embedding, dtype=np.float32
            ).reshape(-1)

    @property
    def has_eye_embedding(self) -> bool:
        """True when a usable periocular vector is attached."""
        return _embedding_ok(self.eye_embedding) if self.eye_embedding is not None else False

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
        """Stacked whole-face embeddings, shape ``(size, dim)``."""
        if not self.faces:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([face.embedding for face in self.faces])

    @property
    def eye_embeddings(self) -> np.ndarray:
        """Stacked periocular embeddings of the faces that have one.

        Shape ``(size, dim)`` when every face has an eye embedding;
        otherwise ``(n, dim)`` for just those that do, which callers check
        with :attr:`eye_coverage`.
        """
        vectors = [
            face.eye_embedding
            for face in self.faces
            if _embedding_ok(face.eye_embedding) if face.eye_embedding is not None
        ]
        if not vectors:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack(vectors)

    @property
    def eye_coverage(self) -> float:
        """Fraction of this cluster's faces that carry an eye embedding."""
        if not self.faces:
            return 0.0
        with_eyes = sum(
            1
            for face in self.faces
            if face.eye_embedding is not None and _embedding_ok(face.eye_embedding)
        )
        return with_eyes / len(self.faces)

    @property
    def centroid(self) -> Optional[np.ndarray]:
        """Mean whole-face embedding of the cluster (re-normalised)."""
        if not self.faces:
            return None
        mean = self.embeddings.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm > 0 else mean

    @property
    def eye_centroid(self) -> Optional[np.ndarray]:
        """Mean periocular embedding of the cluster (re-normalised).

        ``None`` when no face in the cluster has a usable eye embedding.
        """
        vectors = self.eye_embeddings
        if vectors.size == 0:
            return None
        mean = vectors.mean(axis=0)
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


def _embedding_ok(embedding) -> bool:
    if embedding is None:
        return False
    array = np.asarray(embedding, dtype=np.float32).reshape(-1)
    return (
        array.ndim == 1
        and array.size > 0
        and bool(np.all(np.isfinite(array)))
        and float(np.linalg.norm(array)) > 0.0
    )


def _validate_matrix(embeddings, tolerance: float) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.size == 0:
        return matrix
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D embedding matrix, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("embeddings contain NaN or infinite values")
    if not (0.0 < float(tolerance) <= 2.0):
        raise ValueError(
            f"tolerance (DBSCAN eps) must be in (0, 2], got {tolerance}"
        )
    return matrix


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
    matrix = _validate_matrix(embeddings, tolerance)
    if matrix.size == 0:
        return np.empty(0, dtype=np.int64)

    model = DBSCAN(
        eps=float(tolerance),
        min_samples=max(1, int(min_samples)),
        metric="cosine",
        n_jobs=-1,
    )
    return model.fit_predict(matrix)


def cluster_fused(
    full_embeddings: np.ndarray,
    eye_embeddings: Sequence[Optional[np.ndarray]],
    tolerance: float = DEFAULT_TOLERANCE,
    min_samples: int = 1,
    metric: Optional[FusedMetric] = None,
) -> np.ndarray:
    """DBSCAN over the fused (whole-face + periocular) distance.

    A precomputed distance matrix is used rather than a Python callable per
    pair: the pipeline is CPU-bound and this keeps clustering from becoming
    the bottleneck.
    """
    matrix = _validate_matrix(full_embeddings, tolerance)
    if matrix.size == 0:
        return np.empty(0, dtype=np.int64)

    metric = metric or FusedMetric()
    distances = metric.matrix(matrix, eye_embeddings)
    model = DBSCAN(
        eps=float(tolerance),
        min_samples=max(1, int(min_samples)),
        metric="precomputed",
        n_jobs=-1,
    )
    return model.fit_predict(distances)


def _split_groups(labels: Sequence[int]) -> List[List[int]]:
    """Group indices by label, noise as singletons, first-seen order."""
    groups: Dict[int, List[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    if NOISE_LABEL in groups:
        for index in groups.pop(NOISE_LABEL):
            groups.setdefault(NOISE_LABEL - index, []).append(index)
    return sorted(groups.values(), key=lambda members: members[0])


def relabel_ages(
    eye_groups: Sequence[Sequence[int]],
    full_labels: Sequence[int],
    eye_coverage: Sequence[float],
) -> np.ndarray:
    """Reconcile the periocular and whole-face clusterings into one label set.

    Rationale: the periocular region is the age-stable signal, so where it
    forms a confident group that group wins; the whole-face pass only breaks
    ties among faces the eye region could not place.

    Rules, in order:

    1. An eye group is *kept* when it is corroborated by the whole-face pass
       (its members mostly share a whole-face label) or when it is large and
       the eye signal is unanimous. A group the whole-face pass completely
       contradicts is dropped rather than merged, so a periocular false
       positive (two similar-looking strangers) does not silently merge them.
    2. Remaining faces are grouped by their whole-face label.

    The result is a plain ``int`` label per face; the exact grouping is
    decided by the input, so this function is pure and easy to test.
    """
    count = len(full_labels)
    labels = np.empty(count, dtype=np.int64)
    if count == 0:
        return labels
    if not eye_groups:
        labels[:] = np.asarray(full_labels, dtype=np.int64)
        return labels

    full = np.asarray(full_labels, dtype=np.int64)

    # Faces already agreeing under the whole-face pass are "settled"; they are
    # the baseline the eye groups are measured against.
    full_counts: Dict[int, int] = {}
    for label in full:
        full_counts[int(label)] = full_counts.get(int(label), 0) + 1

    used = np.zeros(count, dtype=bool)
    next_label = 0
    assignments: List[Tuple[float, List[int]]] = []

    for group in eye_groups:
        members = [index for index in group if 0 <= index < count]
        if not members:
            continue
        inside = [
            int(full[index]) for index in members
            if int(full[index]) != NOISE_LABEL
        ]
        if not inside:
            continue  # whole-face pass has nothing to corroborate
        dominant, hits = Counter(inside).most_common(1)[0]
        agreement = hits / len(members)
        coverage = float(np.mean([eye_coverage[i] for i in members]))
        # unanimous eye grouping + solid whole-face agreement, or an eye
        # region that was available for every member
        confident = agreement >= 0.6 or (coverage >= 0.999 and agreement >= 0.4)
        if confident:
            assignments.append((agreement, members))
            for index in members:
                used[index] = True

    # Strongest agreement first, so cluster ids read as most-confident first.
    assignments.sort(key=lambda item: item[0], reverse=True)
    for _, members in assignments:
        labels[members] = next_label
        next_label += 1

    # Everything else keeps its whole-face grouping (noise stays noise).
    for label in sorted(full_counts):
        if label == NOISE_LABEL:
            continue
        members = [i for i in range(count) if not used[i] and int(full[i]) == label]
        if not members:
            continue
        labels[members] = next_label
        next_label += 1
        for index in members:
            used[index] = True

    leftover = [i for i in range(count) if not used[i]]
    for index in leftover:
        labels[index] = next_label
        next_label += 1

    return labels


class FaceClusterer:
    """Groups face records into clusters (one cluster per person).

    Age invariance comes from the fused distance; see the module docstring.
    Set ``weight_eye = 0`` to fall back to plain whole-face clustering, which
    is useful for A/B-ing the behaviour against the previous behaviour.
    """

    def __init__(
        self,
        tolerance: float = DEFAULT_TOLERANCE,
        min_faces_per_cluster: int = DEFAULT_MIN_FACES,
        min_samples: int = 1,
        deduplicate: bool = True,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        proximity_ratio: float = DEFAULT_PROXIMITY_RATIO,
        weight_full: float = DEFAULT_WEIGHT_FULL,
        weight_eye: float = DEFAULT_WEIGHT_EYE,
        use_eye_regions: bool = True,
        min_eye_faces: int = MIN_EYE_FACES,
    ) -> None:
        self.tolerance = float(tolerance)
        self.min_faces_per_cluster = max(1, int(min_faces_per_cluster))
        self.min_samples = int(min_samples)
        self.deduplicate = deduplicate
        self.iou_threshold = iou_threshold
        self.proximity_ratio = proximity_ratio
        self.weight_full, self.weight_eye = validate_weights(weight_full, weight_eye)
        self.use_eye_regions = bool(use_eye_regions) and self.weight_eye > 0.0
        self.min_eye_faces = max(1, int(min_eye_faces))
        self.metric = FusedMetric(self.weight_full, self.weight_eye)

    @classmethod
    def from_config(cls, config: Mapping) -> "FaceClusterer":
        """Build a clusterer from the ``config.yaml`` mapping."""
        return cls(
            tolerance=config.get("tolerance", DEFAULT_TOLERANCE),
            min_faces_per_cluster=config.get(
                "min_faces_per_cluster", DEFAULT_MIN_FACES
            ),
            weight_full=config.get("weight_full_face", DEFAULT_WEIGHT_FULL),
            weight_eye=config.get("weight_eye_region", DEFAULT_WEIGHT_EYE),
            use_eye_regions=config.get("use_eye_regions", True),
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

        full_matrix = np.stack([record.embedding for record in usable])
        eye_vectors: List[Optional[np.ndarray]] = [
            record.eye_embedding for record in usable
        ]
        eye_coverage = [
            1.0 if _embedding_ok(vector) else 0.0 for vector in eye_vectors
        ]
        eye_count = int(sum(eye_coverage))

        labels = self._label_faces(full_matrix, eye_vectors, eye_coverage)

        ordered = _split_groups(labels)

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

        if eye_count:
            logger.info(
                "%d of %d face(s) carried a periocular embedding; clustering "
                "used %s at tolerance=%.2f.",
                eye_count,
                len(usable),
                self.metric.describe() if self.use_eye_regions else "whole-face only",
                self.tolerance,
            )
        else:
            logger.info(
                "No usable periocular embeddings; clustered on whole-face "
                "vectors at tolerance=%.2f.",
                self.tolerance,
            )
        logger.info(
            "Clustered %d face(s) -> %d cluster(s) kept, %d face(s) ignored "
            "(tolerance=%.2f, min_faces_per_cluster=%d).",
            len(usable),
            len(clusters),
            ignored_faces,
            self.tolerance,
            self.min_faces_per_cluster,
        )
        return clusters

    def _label_faces(
        self,
        full_matrix: np.ndarray,
        eye_vectors: Sequence[Optional[np.ndarray]],
        eye_coverage: Sequence[float],
    ) -> np.ndarray:
        """Produce one label per face, fusing both modalities when possible."""
        count = full_matrix.shape[0]
        eye_count = int(sum(eye_coverage))

        if not self.use_eye_regions or eye_count < self.min_eye_faces:
            # Not enough eye evidence to trust it: whole-face only.
            return cluster_embeddings(
                full_matrix, tolerance=self.tolerance, min_samples=self.min_samples
            )

        if eye_count == count:
            # Every face has an eye vector, so one fused pass is the right
            # answer — no need to reconcile two separate clusterings.
            return cluster_fused(
                full_matrix,
                eye_vectors,
                tolerance=self.tolerance,
                min_samples=self.min_samples,
                metric=self.metric,
            )

        # Mixed availability: cluster each modality over the faces that have
        # it, then reconcile.
        eye_index = [i for i in range(count) if eye_coverage[i] > 0]
        eye_groups: List[List[int]] = []
        if len(eye_index) >= self.min_eye_faces:
            eye_labels = cluster_fused(
                full_matrix[eye_index],
                [eye_vectors[i] for i in eye_index],
                tolerance=self.tolerance,
                min_samples=self.min_samples,
                metric=self.metric,
            )
            eye_groups = [
                [eye_index[position] for position in members]
                for members in _split_groups(eye_labels)
            ]
        full_labels = cluster_embeddings(
            full_matrix, tolerance=self.tolerance, min_samples=self.min_samples
        )
        return relabel_ages(eye_groups, full_labels, eye_coverage)


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
            eye_embedding=getattr(item, "eye_embedding", None),
            score=item.detection.score,
        )
        for item in face_embeddings
    ]
