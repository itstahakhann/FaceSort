"""Face embeddings — turn each detected face into a numeric fingerprint.

Uses the ArcFace recognition head of the same buffalo_l model on the CPU
(one shared ``FaceAnalysis`` instance, see :mod:`src.face_model`).  The
returned vector is L2-normalised, so cosine distance and Euclidean distance
on it are interchangeable for the clustering step (SPEC §7).

Two fingerprints per face
-------------------------
Alongside the whole-face vector each face also gets a **periocular**
("eye region") vector — see :mod:`src.eye_embedder`.  Whole-face embeddings
drift as a child grows into an adult, so the clusterer blends the two
(:mod:`src.clusterer`); the eye vector is what keeps a childhood photo and
an adult photo of the same person in one group.

Cost: one extra forward pass of the recognition net on a 112x112 input per
face.  Detection over the full frame dominates, so this adds a modest
fraction to the per-image time rather than doubling it, and both passes
reuse the one shared model instance.

No network call is ever made here: the weights come from the local cache.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np

from .detector import DetectedFace
from .eye_embedder import EyeEmbeddingError, embed_periocular
from .face_model import MODEL_NAME, make_face

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512  # buffalo_l (w600k_r50) produces 512-D vectors

__all__ = [
    "FaceEmbedding",
    "FaceEmbedder",
    "FaceEmbeddingError",
    "EMBEDDING_DIM",
]


class FaceEmbeddingError(RuntimeError):
    """A single face could not be turned into an embedding."""


@dataclass(frozen=True)
class FaceEmbedding:
    """A detected face together with its two fingerprints.

    ``embedding`` is the whole-face vector and ``eye_embedding`` the
    periocular one.  The latter is ``None`` when the region was too small or
    too low-resolution to fingerprint — every consumer must treat that as
    "no eye signal for this face", never as an error.
    """

    detection: DetectedFace
    embedding: np.ndarray  # float32, L2-normalised, shape (EMBEDDING_DIM,)
    eye_embedding: Optional[np.ndarray] = None  # same shape, or None


class FaceEmbedder:
    """Extracts whole-face and periocular embeddings using the ArcFace model."""

    def __init__(self, analysis: Any = None, include_eye_regions: bool = True) -> None:
        self._analysis = analysis
        self.include_eye_regions = bool(include_eye_regions)

    @property
    def analysis(self) -> Any:
        """The shared insightface ``FaceAnalysis`` instance (built lazily)."""
        if self._analysis is None:
            from .face_model import get_face_analysis

            self._analysis = get_face_analysis()
        return self._analysis

    @property
    def recognition_model(self) -> Any:
        """The ArcFace model that maps an aligned face to a vector."""
        models = getattr(self.analysis, "models", None) or {}
        model = models.get("recognition")
        if model is None:
            raise RuntimeError(
                "No recognition model found in the loaded models "
                f"({sorted(models)}); expected the {MODEL_NAME} bundle."
            )
        return model

    def embed_one(self, image: np.ndarray, detection: DetectedFace) -> np.ndarray:
        """Return the L2-normalised fingerprint of one face.

        Raises :class:`FaceEmbeddingError` when the face cannot be embedded
        (no landmarks, non-finite output, ...).
        """
        if image is None:
            raise FaceEmbeddingError("no image to embed the face from")
        if detection.kps is None:
            raise FaceEmbeddingError(
                f"face at {detection.bbox} has no landmarks to align on"
            )

        face = make_face(detection.bbox, detection.kps, detection.score)
        try:
            result = self.recognition_model.get(image, face)
        except Exception as exc:  # a single bad crop must not kill the run
            raise FaceEmbeddingError(
                f"recognition failed for face at {detection.bbox}: {exc}"
            ) from exc

        embedding = face.embedding
        if embedding is None and result is not None:
            embedding = result
        if embedding is None:
            raise FaceEmbeddingError(
                f"recognition model returned no embedding for face at {detection.bbox}"
            )

        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm <= 0.0:
            raise FaceEmbeddingError(
                f"non-finite embedding for face at {detection.bbox}"
            )
        return embedding / norm

    def eye_embed_one(
        self, image: np.ndarray, detection: DetectedFace
    ) -> np.ndarray:
        """Return the periocular fingerprint of one face (see eye_embedder)."""
        return embed_periocular(image, detection, self.recognition_model)

    def embed(
        self, image: Optional[np.ndarray], detections: Sequence[DetectedFace]
    ) -> List[FaceEmbedding]:
        """Embed every face of one image.

        Faces that cannot be embedded are logged and skipped, so an image
        with no faces simply yields an empty list.  A face whose *eye region*
        fails is still returned — with ``eye_embedding=None`` — because the
        whole-face vector alone is still worth clustering.
        """
        if image is None:
            logger.warning("embed() called without an image; returning no embeddings.")
            return []

        results: List[FaceEmbedding] = []
        missing_eyes = 0
        for detection in detections:
            try:
                embedding = self.embed_one(image, detection)
            except FaceEmbeddingError as exc:
                logger.warning("Skipping face: %s", exc)
                continue

            eye_embedding = None
            if self.include_eye_regions:
                try:
                    eye_embedding = self.eye_embed_one(image, detection)
                except EyeEmbeddingError as exc:
                    # Expected for small or blurred faces; not worth a warning
                    # per photo, but it is worth counting for the log summary.
                    logger.debug("No periocular embedding: %s", exc)
                    missing_eyes += 1
            results.append(
                FaceEmbedding(
                    detection=detection,
                    embedding=embedding,
                    eye_embedding=eye_embedding,
                )
            )
        if missing_eyes:
            logger.info(
                "%d of %d face(s) had no usable eye region (too small or "
                "blurred); they still cluster on the whole-face vector.",
                missing_eyes, len(results),
            )
        return results
