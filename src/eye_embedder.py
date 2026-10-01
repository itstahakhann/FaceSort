"""Periocular ("eye region") embeddings — an age-stable second fingerprint.

The problem this solves
-----------------------
ArcFace embeddings of a *whole* face drift noticeably between a child's
photo and the same person's adult photo: the jaw fills out, the cheeks
round, the nose lengthens, and the soft biometrics the model leans on
(bone structure, overall face shape) all move.  A clusterer that only sees
that vector splits one person into a "child" cluster and an "adult" cluster.

The periocular region — eyebrows, both eyes, and the bridge of the nose —
is the part of the face that changes *least* between ages: the orbital
bone structure and the inter-eye geometry are largely fixed after early
childhood.  Embedding that region separately and leaning on it gives the
matcher a signal that survives the age gap.

How the region is framed
------------------------
Rather than cropping pixels and resizing them (which invents its own
normalisation and throws away the recogniser's own alignment), this module
*re-frames the landmarks* and lets the existing recognition model do exactly
what it already does: solve for the similarity transform that maps the five
face landmarks onto ArcFace's 112x112 template, then warp.

The trick is that the transform is driven by where the five points *are*.
Pushing the nose and mouth points further from the eyes (see
:func:`expand_landmarks`) forces a tighter zoom, so the 112x112 canvas ends
up containing the eyes at their template positions with the mouth pushed off
the bottom edge — the periocular region, normalised by the model itself
rather than by our own resize.

This costs one extra forward pass of the recognition net on a 112x112 input
(``w600k_r50``). Detection on the full frame dominates the per-image cost,
so this is a modest addition rather than a second full pass; see the timing
note in :mod:`src.embedder`.

Landmark order (insightface 5-point convention)::

    0 left eye   1 right eye   2 nose tip   3 left mouth   4 right mouth
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from .detector import DetectedFace
from .face_model import make_face

logger = logging.getLogger(__name__)

__all__ = [
    "LEFT_EYE",
    "RIGHT_EYE",
    "NOSE_TIP",
    "LEFT_MOUTH",
    "RIGHT_MOUTH",
    "PERIOCULAR_SPREAD",
    "EyeEmbeddingError",
    "expand_landmarks",
    "periocular_bbox",
    "embed_periocular",
]

#: Indices into insightface's 5-point landmark array.
LEFT_EYE, RIGHT_EYE, NOSE_TIP, LEFT_MOUTH, RIGHT_MOUTH = range(5)

#: How far the nose/mouth points are pulled *towards* the eye line.
#:
#: 1.0 is the real face geometry, which reproduces the whole face. Values
#: below 1.0 compress the mouth upwards, so the similarity transform has to
#: shrink the vertical extent of the source frame to hit ArcFace's fixed
#: template — magnifying the eye region and pushing the mouth off the bottom
#: edge. (Verified by rendering the warp at 0.3 / 0.45 / 0.6 / 0.8 / 1.0:
#: 1.0 shows the whole face, 0.3 is eyes only.)
#:
#: 0.55 keeps the brows, both eyes and a strip of nose bridge — the region the
#: literature calls periocular. Below ~0.35 the inter-eye distance gets
#: distorted enough that the fingerprint stops meaning much; above ~0.8 the
#: mouth starts creeping back in and the signal becomes whole-face again.
PERIOCULAR_SPREAD = 0.55

#: A periocular region needs enough pixels for the eye/brow texture to carry
#: any signal.  Below this the embedding is noise, and we prefer to store no
#: eye vector at all rather than a misleading one.
MIN_EYE_REGION_PX = 24


class EyeEmbeddingError(RuntimeError):
    """A periocular region could not be embedded."""


def _validate(kps: np.ndarray) -> np.ndarray:
    points = np.asarray(kps, dtype=np.float32)
    if points.shape != (5, 2):
        raise EyeEmbeddingError(
            f"expected 5x2 landmarks, got shape {points.shape}"
        )
    if not bool(np.all(np.isfinite(points))):
        raise EyeEmbeddingError("landmarks contain NaN or infinite values")
    return points


def eye_span(kps: np.ndarray) -> float:
    """Inter-eye distance in pixels — the unit the crop is expressed in."""
    points = _validate(kps)
    return float(np.linalg.norm(points[LEFT_EYE] - points[RIGHT_EYE]))


#: ArcFace's 112x112 template: eye-gap in template pixels, and the distance
#: from the eye midpoint to the top and bottom edges.
TEMPLATE_EDGE = 112
TEMPLATE_EYE_GAP = 35.2
TEMPLATE_EYE_TOP = 51.6
TEMPLATE_EYE_BOTTOM = 60.4


def expand_landmarks(
    kps: np.ndarray, spread: float = PERIOCULAR_SPREAD
) -> np.ndarray:
    """Return landmarks that frame the periocular region when warped.

    The two eye points are left exactly where they are (they must land on
    ArcFace's template eye positions for the alignment to be meaningful);
    the nose and mouth points are scaled away from the eye midpoint by
    ``spread``.
    """
    points = _validate(kps)
    if spread <= 0:
        raise ValueError(f"spread must be positive, got {spread}")

    eye_mid = (points[LEFT_EYE] + points[RIGHT_EYE]) / 2.0
    out = points.copy()
    for index in (NOSE_TIP, LEFT_MOUTH, RIGHT_MOUTH):
        out[index] = eye_mid + (points[index] - eye_mid) * float(spread)
    return out





def periocular_bbox(
    kps: np.ndarray, spread: float = PERIOCULAR_SPREAD
) -> Optional[Tuple[int, int, int, int]]:
    """Approximate pixel bounds of the periocular region.

    This is the geometric equivalent of the warp above, returned as an
    OpenCV ``(x1, y1, x2, y2)`` box.  It is used for thumbnails, for the
    minimum-size guard, and by the tests; the embedding itself is always
    produced by the model, never by cropping with this box.
    """
    try:
        points = _validate(kps)
    except EyeEmbeddingError:
        return None

    span = eye_span(points)
    if span <= 0:
        return None

    eye_mid = (points[LEFT_EYE] + points[RIGHT_EYE]) / 2.0

    # The box mirrors what the model actually warps, so it accounts for both
    # the template's proportions and the `spread` squeeze on the non-eye
    # landmarks (which shrinks the source area the warp samples from).
    scale = float(spread) if spread > 0 else 1.0
    per_template = (span * scale) / TEMPLATE_EYE_GAP
    half = (TEMPLATE_EDGE / 2.0) * per_template
    top = eye_mid[1] - TEMPLATE_EYE_TOP * per_template
    bottom = eye_mid[1] + TEMPLATE_EYE_BOTTOM * per_template
    left = eye_mid[0] - half
    right = eye_mid[0] + half

    if not all(np.isfinite(v) for v in (top, bottom, left, right)):
        return None

    return (
        int(np.floor(left)),
        int(np.floor(top)),
        int(np.ceil(right)),
        int(np.ceil(bottom)),
    )


def _too_small(bbox: Tuple[int, int, int, int], min_pixels: int) -> bool:
    x1, y1, x2, y2 = bbox
    return min(x2 - x1, y2 - y1) < min_pixels


def embed_periocular(
    image: np.ndarray,
    detection: DetectedFace,
    recognition_model: Any,
    spread: float = PERIOCULAR_SPREAD,
    min_pixels: int = MIN_EYE_REGION_PX,
) -> np.ndarray:
    """Return the L2-normalised periocular fingerprint of one face.

    Raises :class:`EyeEmbeddingError` when the region cannot be embedded —
    no landmarks, too few pixels to carry signal, or non-finite output.
    Callers treat that as "this face has no eye vector", never as a failure.
    """
    if image is None:
        raise EyeEmbeddingError("no image to embed the eye region from")
    if detection.kps is None:
        raise EyeEmbeddingError(
            f"face at {detection.bbox} has no landmarks to frame the eye region"
        )

    region = periocular_bbox(detection.kps, spread)
    if region is None:
        raise EyeEmbeddingError(f"degenerate eye region for face at {detection.bbox}")
    if _too_small(region, min_pixels):
        raise EyeEmbeddingError(
            f"eye region for face at {detection.bbox} is smaller than "
            f"{min_pixels}px ({region[2] - region[0]}x{region[3] - region[1]}); "
            "too low-resolution to fingerprint"
        )

    warped_kps = expand_landmarks(detection.kps, spread)
    # The recogniser normalises on the landmarks; the bbox only has to be a
    # sane container for the transform, so the face's own box is fine.
    face = make_face(detection.bbox, warped_kps, detection.score)
    try:
        result = recognition_model.get(image, face)
    except Exception as exc:  # a single bad crop must not kill the run
        raise EyeEmbeddingError(
            f"periocular recognition failed for face at {detection.bbox}: {exc}"
        ) from exc

    embedding = face.embedding
    if embedding is None and result is not None:
        embedding = result
    if embedding is None:
        raise EyeEmbeddingError(
            f"recognition model returned no embedding for the eye region of "
            f"face at {detection.bbox}"
        )

    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or norm <= 0.0:
        raise EyeEmbeddingError(
            f"non-finite periocular embedding for face at {detection.bbox}"
        )
    return embedding / norm


def usable(kps: Optional[np.ndarray]) -> bool:
    """True when ``kps`` can produce a meaningful periocular region."""
    if kps is None:
        return False
    region = periocular_bbox(kps)
    return region is not None and not _too_small(region, MIN_EYE_REGION_PX)


def normalise_points(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Coerce a sequence of 2-D points into a validated ``5x2`` array."""
    return _validate(np.asarray(points, dtype=np.float32))