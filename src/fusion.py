"""Age-invariant fused distance between two faces.

Why fusion is needed
--------------------
A whole-face ArcFace embedding encodes face *shape* as much as identity, and
shape changes as a child grows: the jaw widens and lengthens, the cheeks
round out, the nose lengthens, the forehead-to-chin ratio shifts.  Cosine
distance between a child's photo and the same person's adult photo is
therefore noticeably larger than between two adult photos of that person —
often larger than the clustering tolerance, which is how one person ends up
split across a "child" cluster and an "adult" cluster.

The periocular region (brows, eyes, nose bridge — see :mod:`src.eye_embedder`)
is the most age-stable part of the face: the orbital bone structure and the
inter-eye geometry are largely fixed after early childhood.  Blending the
whole-face similarity with the periocular similarity therefore gives a score
that is *stable across the age gap* while still carrying the discrimination
of the full face.

The score
---------
For two faces with L2-normalised vectors::

    fused_similarity = w_full * cos(full_a, full_b)
                      + w_eye  * cos(eye_a, eye_b)      # only when both exist
    fused_distance   = 1 - fused_similarity

Weights default to ``0.4`` full / ``0.6`` eye, so the age-stable region
dominates — but the full-face term is never dropped, because the periocular
region alone is a weak discriminator between siblings.

This module is pure maths on vectors: no model, no I/O, no state.  That
makes the behaviour testable in isolation and keeps the clusterer readable.

Missing eye embeddings
----------------------
Not every face has a usable periocular vector — a 20px face, a heavy blur,
a profile shot with closed eyes.  Rather than dropping those faces (which
would lose exactly the low-quality childhood photos this is meant to fix),
the blend **renormalises over the evidence that exists**: a pair with no eye
vectors scores on the full face alone, a pair with both scores on the blend.
:meth:`FusedMetric.weights_for` reports which blend a given pair used, so
the clusterer can log it and the UI can flag it.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Default blend. The eye region is weighted higher because it is the
#: age-stable part; see the module docstring.
DEFAULT_WEIGHT_FULL = 0.4
DEFAULT_WEIGHT_EYE = 0.6

#: ``tolerance`` is a cosine distance in (0, 2] and the fused distance is a
#: convex combination of cosine distances, so it shares that range: 0 means
#: identical direction, 2 means opposite. Keeping one scale means the
#: existing ``tolerance`` config value keeps its meaning.
DISTANCE_RANGE = (0.0, 2.0)

__all__ = [
    "DEFAULT_WEIGHT_FULL",
    "DEFAULT_WEIGHT_EYE",
    "FusedMetric",
    "cosine_similarity",
    "fused_similarity",
    "fused_distance",
    "validate_weights",
]


def validate_weights(weight_full: float, weight_eye: float) -> Tuple[float, float]:
    """Check a weight pair and return it normalised to sum to 1.

    Normalising here means a caller can pass ``(2, 3)`` or ``(0.5, 0.5)`` and
    get a valid blend, while ``(1, 0)`` (eye region disabled) still works.
    """
    try:
        full = float(weight_full)
        eye = float(weight_eye)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"weights must be numbers: {exc}") from exc
    if not np.isfinite(full) or not np.isfinite(eye):
        raise ValueError(f"weights must be finite, got {full} and {eye}")
    if full < 0.0 or eye < 0.0:
        raise ValueError(f"weights must be non-negative, got {full} and {eye}")
    total = full + eye
    if total <= 0.0:
        raise ValueError("at least one of the weights must be positive")
    return full / total, eye / total


def cosine_similarity(a, b) -> Optional[float]:
    """Cosine similarity of two vectors, or ``None`` if unusable.

    Both vectors are normalised internally.  Size mismatch, empty input, or
    non-finite values yield ``None`` — the caller decides whether that means
    "fall back to the other signal" or "these cannot be compared".
    """
    left = np.asarray(a, dtype=np.float32).reshape(-1)
    right = np.asarray(b, dtype=np.float32).reshape(-1)
    if left.size == 0 or right.size == 0 or left.size != right.size:
        return None
    if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
        return None
    norm = float(np.linalg.norm(left)) * float(np.linalg.norm(right))
    if not np.isfinite(norm) or norm <= 0.0:
        return None
    return float(np.clip(np.dot(left, right) / norm, -1.0, 1.0))


def fused_similarity(
    full_a,
    full_b,
    eye_a=None,
    eye_b=None,
    weight_full: float = DEFAULT_WEIGHT_FULL,
    weight_eye: float = DEFAULT_WEIGHT_EYE,
) -> Optional[float]:
    """Blend the two cosine similarities into one score.

    Returns ``None`` only when *neither* signal is usable — no caller should
    then treat the pair as "distance 0", which would wrongly merge strangers.
    """
    full = cosine_similarity(full_a, full_b)
    eye = cosine_similarity(eye_a, eye_b)

    if full is None:
        # No whole-face signal: the eye region alone can still carry the
        # comparison. This is the reverse of the usual fallback.
        return eye
    if eye is None:
        return full

    w_full, w_eye = validate_weights(weight_full, weight_eye)
    return w_full * full + w_eye * eye


def fused_distance(
    full_a,
    full_b,
    eye_a=None,
    eye_b=None,
    weight_full: float = DEFAULT_WEIGHT_FULL,
    weight_eye: float = DEFAULT_WEIGHT_EYE,
) -> Optional[float]:
    """Age-invariant distance in ``[0, 2]``; ``None`` if uncomputable."""
    similarity = fused_similarity(
        full_a, full_b, eye_a, eye_b, weight_full, weight_eye
    )
    if similarity is None:
        return None
    return float(1.0 - similarity)


class FusedMetric:
    """Callable fused distance, ready for scikit-learn's ``metric=``.

    Built once with the weights and a flag, then reused for every pair::

        metric = FusedMetric()
        metric(full_a, full_b, eye_a, eye_b)   # -> float or None

    ``None`` is a problem for scikit-learn, which expects a number, so
    :meth:`as_sklearn_metric` returns a wrapper that substitutes the maximum
    distance (2.0) for uncomputable pairs: "as far apart as possible" is the
    safe answer when we cannot measure.
    """

    def __init__(
        self,
        weight_full: float = DEFAULT_WEIGHT_FULL,
        weight_eye: float = DEFAULT_WEIGHT_EYE,
    ) -> None:
        self.weight_full, self.weight_eye = validate_weights(
            weight_full, weight_eye
        )

    def weights_for(self, eye_a, eye_b) -> Tuple[float, float]:
        """The blend actually used for a pair, after fallback.

        Lets the clusterer log *why* a comparison was made and the UI explain
        that low-resolution photos leaned on the whole-face vector.
        """
        eye = cosine_similarity(eye_a, eye_b)
        return (1.0, 0.0) if eye is None else (self.weight_full, self.weight_eye)

    def __call__(
        self, full_a, full_b, eye_a=None, eye_b=None
    ) -> Optional[float]:
        return fused_distance(
            full_a, full_b, eye_a, eye_b, self.weight_full, self.weight_eye
        )

    def matrix(
        self, full: np.ndarray, eyes: Sequence[Optional[np.ndarray]]
    ) -> np.ndarray:
        """Dense ``(n, n)`` fused-distance matrix for ``metric="precomputed"``.

        Passing a precomputed matrix is much faster than a Python callable for
        every pair, which matters because the pipeline is otherwise CPU-bound.
        Rows with no eye vector fall back to full-face distance against
        everyone, and pairs that cannot be compared at all get 2.0.
        """
        full = np.asarray(full, dtype=np.float32)
        if full.ndim != 2:
            raise ValueError(f"expected a 2-D matrix, got shape {full.shape}")
        count = full.shape[0]
        if len(eyes) != count:
            raise ValueError(
                f"got {len(eyes)} eye vectors for {count} full embeddings"
            )

        # Full-face similarity for every pair, via one matrix product.
        norms = np.linalg.norm(full, axis=1, keepdims=True)
        safe = np.where(norms > 0, norms, 1.0)
        unit = full / safe
        sim = np.clip(unit @ unit.T, -1.0, 1.0)

        # Per-row: the normalised eye vector, or None when this face has none.
        rows: List[Optional[np.ndarray]] = [None] * count
        eye_dim = 0
        for index, vector in enumerate(eyes):
            if vector is None:
                continue
            array = np.asarray(vector, dtype=np.float32).reshape(-1)
            if array.size == 0 or not np.all(np.isfinite(array)):
                continue
            length = float(np.linalg.norm(array))
            if not np.isfinite(length) or length <= 0.0:
                continue
            if eye_dim == 0:
                eye_dim = array.size
            if array.size != eye_dim:
                # Inconsistent dimension: treat as missing rather than
                # comparing a 256-D vector against a 512-D one.
                logger.debug("Eye embedding %d has dim %d, expected %d; ignoring.",
                             index, array.size, eye_dim)
                continue
            rows[index] = array / length

        indices = [i for i, row in enumerate(rows) if row is not None]
        if not indices:
            # Nothing to blend: this is exactly the plain cosine distance.
            distance = 1.0 - sim
            np.fill_diagonal(distance, 0.0)
            return np.clip(distance, *DISTANCE_RANGE).astype(np.float32)

        # eye_sim[i, j] is the periocular similarity when *both* sides have a
        # usable vector; NaN marks every pair that must fall back.
        eye_sim = np.full((count, count), np.nan, dtype=np.float32)
        stacked = np.stack([rows[i] for i in indices])
        block = np.clip(stacked @ stacked.T, -1.0, 1.0)
        eye_sim[np.ix_(indices, indices)] = block

        both = np.isfinite(eye_sim)
        similarity = np.where(
            both, self.weight_full * sim + self.weight_eye * np.nan_to_num(eye_sim),
            sim,
        )
        distance = 1.0 - similarity
        np.fill_diagonal(distance, 0.0)
        return np.clip(distance, *DISTANCE_RANGE).astype(np.float32)

    def as_sklearn_metric(self):
        """A ``(a, b) -> float`` wrapper for scikit-learn.

        scikit-learn passes each vector as a 1-D array, so this expects the
        *row index* semantics of :meth:`matrix` only when the caller has
        pre-resolved them; for the general case it treats the two arguments
        as (full_a, full_b) with no eye data, i.e. whole-face only.
        """
        metric = self

        def callable_metric(a, b) -> float:
            value = metric(a, b)
            return DISTANCE_RANGE[1] if value is None else value

        return callable_metric

    def describe(self) -> str:
        return (
            f"fused(full={self.weight_full:.2f}, eye={self.weight_eye:.2f})"
        )