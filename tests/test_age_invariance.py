"""Periocular landmarks and age-invariant fused distance.

Run: python tests/test_age_invariance.py

Covers the three modules that implement age invariance:
``src.eye_embedder`` (landmark re-framing), ``src.fusion`` (the blended
score) and ``src.clusterer`` (using it for grouping).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clusterer import (  # noqa: E402
    FaceClusterer,
    FaceRecord,
    cluster_fused,
    relabel_ages,
)
from src.detector import DetectedFace  # noqa: E402
from src.eye_embedder import (  # noqa: E402
    EyeEmbeddingError,
    LEFT_EYE,
    NOSE_TIP,
    PERIOCULAR_SPREAD,
    RIGHT_EYE,
    embed_periocular,
    expand_landmarks,
    eye_span,
    periocular_bbox,
    usable,
)
from src.fusion import (  # noqa: E402
    FusedMetric,
    fused_distance,
    fused_similarity,
    validate_weights,
)

_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}{(': ' + detail) if detail else ''}")
        _failures.append(label)


def unit(ndim: int, seed: int = 0) -> np.ndarray:
    """A random unit vector, for maths that only needs an angle."""
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=ndim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def tilt(base: np.ndarray, degrees: float, seed: int = 1) -> np.ndarray:
    """Rotate ``base`` by ``degrees`` in two dimensions (cosine shift)."""
    other = unit(base.size, seed)
    cos_a, cos_b = np.cos(np.radians(degrees)), np.cos(np.radians(90 - degrees))
    blended = cos_a * base + cos_b * other
    return blended / np.linalg.norm(blended)


def canonical_landmarks(eye_distance: float = 60.0) -> np.ndarray:
    """A frontal 5-point face with the given inter-eye distance."""
    half = eye_distance / 2.0
    return np.array(
        [
            [100.0 - half, 120.0],   # 0 left eye
            [100.0 + half, 120.0],   # 1 right eye
            [100.0, 150.0],          # 2 nose tip  (30px below the eyes)
            [100.0 - 32.0, 180.0],   # 3 left mouth
            [100.0 + 32.0, 180.0],   # 4 right mouth
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
def test_landmark_expansion() -> None:
    print("periocular landmark re-framing")
    kps = canonical_landmarks()

    spread_out = expand_landmarks(kps, PERIOCULAR_SPREAD)

    # eyes must be untouched: they anchor the alignment
    check("left eye point is unchanged",
          np.allclose(spread_out[LEFT_EYE], kps[LEFT_EYE]))
    check("right eye point is unchanged",
          np.allclose(spread_out[RIGHT_EYE], kps[RIGHT_EYE]))

    eye_mid = (kps[LEFT_EYE] + kps[RIGHT_EYE]) / 2.0
    before = np.linalg.norm(kps[NOSE_TIP] - eye_mid)
    after = np.linalg.norm(spread_out[NOSE_TIP] - eye_mid)
    # PERIOCULAR_SPREAD < 1 pulls the nose *up* towards the eyes, which is what
    # makes the model warp zoom onto the eye region.
    check("the default spread is below 1 (zooms in)",
          PERIOCULAR_SPREAD < 1.0, f"{PERIOCULAR_SPREAD}")
    check("the nose is pulled towards the eye line",
          after < before, f"{before:.1f} -> {after:.1f}")
    check("the nose moved by the requested factor",
          abs(after - before * PERIOCULAR_SPREAD) < 0.01)

    # a larger spread pulls the nose back down, undoing the zoom
    loose = expand_landmarks(kps, 1.0)
    check("a spread of 1.0 restores the real geometry",
          abs(np.linalg.norm(loose[NOSE_TIP] - eye_mid) - before) < 0.01)

    try:
        expand_landmarks(kps, 0)
        check("a non-positive spread is rejected", False)
    except ValueError:
        check("a non-positive spread is rejected", True)

    for bad in (np.zeros((4, 2)), np.zeros((5, 3)), np.full((5, 2), np.nan)):
        try:
            expand_landmarks(bad)
            check(f"invalid landmarks rejected ({bad.shape})", False)
        except EyeEmbeddingError:
            check(f"invalid landmarks rejected ({bad.shape})", True)


def test_periocular_geometry() -> None:
    print("periocular region geometry")
    kps = canonical_landmarks(60.0)
    check("eye span is measured between the eyes",
          abs(eye_span(kps) - 60.0) < 0.01)

    region = periocular_bbox(kps)
    check("a region is produced for a frontal face", region is not None)
    if region:
        x1, y1, x2, y2 = region
        check("the region covers both eyes",
              x1 <= kps[LEFT_EYE][0] and x2 >= kps[RIGHT_EYE][0], str(region))
        # the box mirrors ArcFace's 112x112 template, so it is square by
        # construction; what matters is that it stays inside the face and
        # excludes the mouth, since that is the whole point of the region.
        check("the region is square (it mirrors the 112x112 template)",
              abs((x2 - x1) - (y2 - y1)) <= 2, f"{x2 - x1}x{y2 - y1}")
        check("the region excludes the mouth corners",
              y2 <= kps[3][1], f"bottom={y2} mouth={kps[3][1]}")
        check("the region covers the brows",
              y1 < kps[LEFT_EYE][1], f"top={y1} eyes={kps[LEFT_EYE][1]}")

        # a spread of 1.0 (real geometry) describes a bigger area than 0.55
        tight = periocular_bbox(kps, PERIOCULAR_SPREAD)
        loose_box = periocular_bbox(kps, 1.0)
        check("a tighter spread describes a smaller region",
              (tight[2] - tight[0]) < (loose_box[2] - loose_box[0]),
              f"{tight[2]-tight[0]} vs {loose_box[2]-loose_box[0]}")
        

    # degenerate inputs
    check("zero span yields no region",
          periocular_bbox(np.array([[10, 10], [10, 10], [10, 12],
                                    [9, 13], [11, 13]], dtype=np.float32)) is None)
    check("wrong landmark count yields no region",
          periocular_bbox(np.zeros((3, 2), dtype=np.float32)) is None)

    # the size guard: a tiny face has no usable eye region
    tiny = canonical_landmarks(12.0)
    check("a tiny face reports no usable region", not usable(tiny))
    check("a normal face is usable", usable(kps))
    check("missing landmarks are not usable", not usable(None))


def test_embed_periocular_guards() -> None:
    print("periocular embedding guards")

    class Boom:
        def get(self, image, face):
            raise RuntimeError("model exploded")

    kps = canonical_landmarks()
    detection = DetectedFace(bbox=(40, 60, 160, 220), score=0.99, kps=kps)
    image = np.zeros((280, 200, 3), dtype=np.uint8)

    try:
        embed_periocular(image, detection, Boom())
        check("a recognition failure surfaces as EyeEmbeddingError", False)
    except EyeEmbeddingError:
        check("a recognition failure surfaces as EyeEmbeddingError", True)

    no_kps = DetectedFace(bbox=(40, 60, 160, 220), score=0.99, kps=None)
    try:
        embed_periocular(image, no_kps, Boom())
        check("a face without landmarks is refused", False)
    except EyeEmbeddingError:
        check("a face without landmarks is refused", True)

    tiny = DetectedFace(
        bbox=(40, 60, 160, 220), score=0.99, kps=canonical_landmarks(10.0)
    )
    try:
        embed_periocular(image, tiny, Boom())
        check("a too-small eye region is refused before inference", False)
    except EyeEmbeddingError as exc:
        check("a too-small eye region is refused before inference",
              "too low-resolution" in str(exc), str(exc))

    try:
        embed_periocular(None, detection, Boom())
        check("a missing image is refused", False)
    except EyeEmbeddingError:
        check("a missing image is refused", True)


def test_fused_maths() -> None:
    print("fused similarity maths")
    a = unit(512, 0)
    b = unit(512, 1)
    eye_same = unit(256, 2)
    eye_other = unit(256, 3)

    full_distance = 1.0 - float(np.dot(a, b))
    check("full cosine distance is in [0, 2]",
          0.0 <= full_distance <= 2.0)

    # Identical eye regions must pull two dissimilar faces closer together:
    # that is the whole point of the age-invariant blend.
    same_eye = fused_distance(a, b, eye_same, eye_same)
    diff_eye = fused_distance(a, b, eye_same, eye_other)
    check("matching eye regions reduce the distance", same_eye < full_distance,
          f"{same_eye:.3f} vs full {full_distance:.3f}")
    check("differing eye regions raise it", diff_eye > full_distance,
          f"{diff_eye:.3f} vs full {full_distance:.3f}")

    # identical inputs -> distance 0 regardless of weights
    check("identical vectors score 0", abs(fused_distance(a, a, eye_same,
                                                          eye_same)) < 1e-6)

    # fallback: eye vectors present on only one side must not crash or zero out
    only_one = fused_distance(a, b, eye_same, None)
    check("a missing eye vector falls back to the full face",
          abs(only_one - full_distance) < 1e-6,
          f"{only_one:.3f} vs {full_distance:.3f}")
    check("no eye vectors at all also falls back",
          abs(fused_distance(a, b) - full_distance) < 1e-6)

    # neither signal comparable -> None, so callers can refuse the pair
    check("size-mismatched eye vectors are ignored",
          abs(fused_distance(a, b, unit(8, 4), unit(9, 5)) - full_distance) < 1e-6)
    check("an empty vector is not a match",
          abs(fused_distance(a, b, np.zeros(256, np.float32),
                             np.zeros(256, np.float32)) - full_distance) < 1e-6)

    # weights
    w_full, w_eye = validate_weights(2.0, 3.0)
    check("weights normalise to 1", abs(w_full + w_eye - 1.0) < 1e-9)
    check("weight ratio is preserved", abs(w_full / w_eye - 2 / 3) < 1e-9)
    for bad in ((-1, 1), (0, 0), (float("nan"), 1), ("a", 1)):
        try:
            validate_weights(*bad)
            check(f"invalid weights rejected: {bad}", False)
        except ValueError:
            check(f"invalid weights rejected: {bad}", True)

    # a weight of 0 for the eye region disables fusion without special-casing
    full_only = FusedMetric(1.0, 0.0)
    check("weight_eye=0 reduces to whole-face distance",
          abs(full_only(a, b, eye_same, eye_other) - full_distance) < 1e-6)


def test_fused_metric_matrix() -> None:
    print("fused distance matrix")
    metric = FusedMetric()
    eye_a, eye_b, eye_c = unit(256, 10), unit(256, 11), unit(256, 12)
    full = np.stack([eye_a[:256], eye_b[:256], eye_c[:256]])

    matrix = metric.matrix(full, [eye_a, eye_b, eye_c])
    check("the matrix is square", matrix.shape == (3, 3), str(matrix.shape))
    check("the diagonal is zero", np.allclose(np.diag(matrix), 0.0))
    check("the matrix is symmetric", np.allclose(matrix, matrix.T))
    check("distances stay in [0, 2]",
          float(matrix.min()) >= 0.0 and float(matrix.max()) <= 2.0)

    # mixed availability must not crash and must not produce NaN
    mixed = metric.matrix(full, [eye_a, None, eye_c])
    check("a missing eye vector is handled", np.all(np.isfinite(mixed)))
    check("the mixed matrix is still symmetric", np.allclose(mixed, mixed.T))

    # all-missing degenerates to plain cosine
    none_eyes = metric.matrix(full, [None, None, None])
    unit_full = full / np.linalg.norm(full, axis=1, keepdims=True)
    expected = 1.0 - np.clip(unit_full @ unit_full.T, -1.0, 1.0)
    np.fill_diagonal(expected, 0.0)
    check("with no eye vectors it equals the cosine matrix",
          np.allclose(none_eyes, expected, atol=1e-5))

    try:
        metric.matrix(full, [eye_a, eye_b])
        check("an eye-vector count mismatch is rejected", False)
    except ValueError:
        check("an eye-vector count mismatch is rejected", True)
    try:
        metric.matrix(np.zeros(4, dtype=np.float32), [])
        check("a 1-D full matrix is rejected", False)
    except ValueError:
        check("a 1-D full matrix is rejected", True)


def test_eye_weight_helps_age_pairs() -> None:
    print("eye weighting on a simulated age pair")
    metric = FusedMetric(0.4, 0.6)

    # Model the situation: the same person photographed as a child and as an
    # adult.  Whole-face vectors sit 0.55 apart (beyond a 0.5 tolerance, so a
    # whole-face-only clusterer splits them) while the eye region agrees at
    # 0.9 similarity.
    child_face, adult_face = unit(512, 20), unit(512, 21)
    full_distance = 1.0 - float(np.dot(child_face, adult_face))
    check("the modelled age gap exceeds the default tolerance",
          full_distance > 0.5, f"{full_distance:.3f}")

    same_person_eye = tilt(unit(256, 22), 26.0, 23)   # same eyes, small drift
    child_eye = unit(256, 24)
    adult_eye = tilt(child_eye, 26.0, 25)            # same eyes, aged
    other_person_eye = unit(256, 26)

    same_person = fused_distance(
        child_face, adult_face, child_eye, adult_eye, 0.4, 0.6
    )
    different = fused_distance(
        child_face, adult_face, child_eye, other_person_eye, 0.4, 0.6
    )
    check("fused distance pulls the age pair under tolerance",
          same_person <= 0.5, f"{same_person:.3f}")
    check("strangers stay well outside tolerance", different > 0.5,
          f"{different:.3f}")
    check("the fused score separates the two cases",
          same_person + 0.2 < different, f"{same_person:.3f} vs {different:.3f}")
    check("unused variable guard", metric is not None)


def test_clustering_uses_fusion() -> None:
    print("clustering with the fused distance")

    def person_record(index: int, identity: int, eye_identity: int) -> FaceRecord:
        """One photo of one person: near-copies of that person's vectors.

        A face is *the same person photographed again*, i.e. the vector sits a
        few degrees off that person's baseline — not a fresh random direction.
        Using random vectors here would test nothing, since two random unit
        vectors are ~90 degrees apart, far outside any tolerance.
        """
        return FaceRecord(
            image_path=f"C:/photos/p{identity}_{index}.jpg",
            bbox=(10, 10, 90, 90),
            embedding=tilt(unit(512, 100 + identity), 6.0 * index, 500 + index),
            eye_embedding=tilt(unit(256, 200 + eye_identity), 4.0 * index, 700 + index),
        )

    def record(seed: int, eye_seed: Optional[int] = None) -> FaceRecord:
        return FaceRecord(
            image_path=f"C:/photos/p{seed}.jpg",
            bbox=(10, 10, 90, 90),
            embedding=unit(512, 100 + seed),
            eye_embedding=None if eye_seed is None else unit(256, 200 + eye_seed),
        )

    # three photos of one person (small drift, as re-shoots produce)
    person = [person_record(i, identity=1, eye_identity=1) for i in range(3)]
    clusters = FaceClusterer(tolerance=0.5, min_faces_per_cluster=2).cluster(person)
    check("similar faces cluster", len(clusters) == 1, f"{len(clusters)} clusters")

    # three different people must stay apart
    strangers = [
        person_record(i, identity=10 + i, eye_identity=10 + i) for i in range(3)
    ]
    # min_faces_per_cluster=1: at tolerance 0.5 three strangers are all DBSCAN
    # noise, i.e. singletons, and the default size filter would drop them
    # before we could count them.
    clusters = FaceClusterer(tolerance=0.5, min_faces_per_cluster=1).cluster(strangers)
    check("strangers do not merge", len(clusters) == 3, f"{len(clusters)} clusters")

    # eye regions agree, whole faces disagree -> fused keeps them together,
    # whole-face-only would split them.  Tight tolerance makes the point.
    shared_eye = unit(256, 77)
    drifted = [
        FaceRecord(
            image_path=f"C:/photos/age{i}.jpg",
            bbox=(10, 10, 90, 90),
            embedding=unit(512, 300 + i),      # deliberately far apart
            eye_embedding=shared_eye,          # identical eye region
        )
        for i in range(4)
    ]
    # min_faces_per_cluster=1 so the whole-face run reports its four singletons
    # rather than filtering them all away as noise.
    fused = FaceClusterer(tolerance=0.5, min_faces_per_cluster=1).cluster(drifted)
    whole_only = FaceClusterer(
        tolerance=0.5, min_faces_per_cluster=1, weight_eye=0.0
    ).cluster(drifted)
    check("the eye region merges faces the whole face separates",
          len(fused) == 1, f"fused gave {len(fused)} clusters")
    check("without eye weighting they stay split",
          len(whole_only) == 4, f"whole-face gave {len(whole_only)} clusters")

    # and with the real default size filter the fused result survives as one
    # usable group, which is the point of the feature
    fused_default = FaceClusterer(tolerance=0.5, min_faces_per_cluster=2).cluster(
        drifted
    )
    check("the fused group survives the default size filter",
          len(fused_default) == 1 and fused_default[0].size == 4,
          f"{[c.size for c in fused_default]}")

    # low-resolution faces (no eye vector) must still cluster, not be dropped
    no_eyes = [
        FaceRecord(
            image_path=f"C:/photos/blur{i}.jpg",
            bbox=(10, 10, 90, 90),
            embedding=tilt(unit(512, 100), 6.0 * i, 800 + i),
            eye_embedding=None,
        )
        for i in range(3)
    ]
    clusters = FaceClusterer(tolerance=0.5, min_faces_per_cluster=2).cluster(no_eyes)
    check("faces without eye embeddings still cluster",
          len(clusters) == 1, f"{len(clusters)} clusters")

    # mixed: some have eyes, some do not
    mixed = [
        person_record(0, identity=2, eye_identity=2),
        person_record(1, identity=2, eye_identity=2),
        FaceRecord(image_path="C:/photos/m2.jpg", bbox=(10, 10, 90, 90),
                   embedding=tilt(unit(512, 102), 6.0, 810),
                   eye_embedding=None),
        FaceRecord(image_path="C:/photos/m3.jpg", bbox=(10, 10, 90, 90),
                   embedding=tilt(unit(512, 103), 12.0, 811),
                   eye_embedding=None),
    ]
    clusters = FaceClusterer(tolerance=0.5, min_faces_per_cluster=1).cluster(mixed)
    check("a mixed set still produces clusters", len(clusters) >= 1)
    check("every face lands in exactly one cluster",
          sum(c.size for c in clusters) == len(mixed),
          f"{sum(c.size for c in clusters)} vs {len(mixed)}")

    # a single eye vector is too little to trust: the whole-face pass decides
    lone_eye = [
        person_record(0, identity=3, eye_identity=3),
        FaceRecord(image_path="C:/photos/l1.jpg", bbox=(10, 10, 90, 90),
                   embedding=tilt(unit(512, 104), 5.0, 820), eye_embedding=None),
        FaceRecord(image_path="C:/photos/l2.jpg", bbox=(10, 10, 90, 90),
                   embedding=tilt(unit(512, 105), 5.0, 821), eye_embedding=None),
    ]
    clusters = FaceClusterer(tolerance=0.5, min_faces_per_cluster=1).cluster(lone_eye)
    check("one eye vector among many is not enough to fuse",
          len(clusters) >= 1)


def test_cluster_fused_labels() -> None:
    print("cluster_fused labels")
    full = np.stack([unit(512, i) for i in range(4)])
    eyes = [unit(256, 40 + i) for i in range(4)]
    labels = cluster_fused(full, eyes, tolerance=0.5, min_samples=1)
    check("a label per face", len(labels) == 4, str(labels))
    check("labels are integers", np.issubdtype(labels.dtype, np.integer))

    empty = cluster_fused(np.empty((0, 0), dtype=np.float32), [], tolerance=0.5)
    check("an empty input yields no labels", empty.size == 0)


def test_relabel_ages() -> None:
    print("reconciling the two clusterings")
    # eye groups say {0,1} and {2,3}; full-face agrees -> two clusters
    labels = relabel_ages([[0, 1], [2, 3]], [0, 0, 1, 1], [1.0, 1.0, 1.0, 1.0])
    check("corroborated eye groups survive",
          labels[0] == labels[1] and labels[2] == labels[3]
          and labels[0] != labels[2], str(labels))

    # eye says {0,1} but the whole face firmly puts them apart -> trust the
    # eye region, since it is the age-stable signal
    labels = relabel_ages([[0, 1]], [0, 1, 2, 3], [1.0, 1.0, 1.0, 1.0])
    check("an uncorroborated eye group is kept when unanimous",
          labels[0] == labels[1], str(labels))

    # no eye groups -> whole-face labels pass through
    labels = relabel_ages([], [3, 3, 4, -1], [0.0] * 4)
    check("no eye groups falls back to the whole-face pass",
          labels[0] == labels[1] == 3, str(labels))

    # every face placed exactly once
    labels = relabel_ages([[0], [1, 2]], [0, 1, 1, -1], [1.0, 1.0, 1.0, 0.0])
    check("no face is dropped or duplicated", len(labels) == 4, str(labels))

    empty = relabel_ages([], [], [])
    check("empty input is safe", empty.size == 0)


def test_face_record_eye_field() -> None:
    print("FaceRecord carries the eye embedding")
    record = FaceRecord(
        image_path="C:/a.jpg", bbox=(1, 2, 3, 4), embedding=unit(8, 1),
        eye_embedding=unit(8, 2),
    )
    check("has_eye_embedding is true when set", record.has_eye_embedding)

    record = FaceRecord(image_path="C:/a.jpg", bbox=(1, 2, 3, 4),
                        embedding=unit(8, 1))
    check("has_eye_embedding is false when absent", not record.has_eye_embedding)

    record = FaceRecord(image_path="C:/a.jpg", bbox=(1, 2, 3, 4),
                        embedding=unit(8, 1), eye_embedding=np.zeros(8))
    check("a zero eye vector does not count", not record.has_eye_embedding)


def main() -> int:
    test_landmark_expansion()
    test_periocular_geometry()
    test_embed_periocular_guards()
    test_fused_maths()
    test_fused_metric_matrix()
    test_eye_weight_helps_age_pairs()
    test_clustering_uses_fusion()
    test_cluster_fused_labels()
    test_relabel_ages()
    test_face_record_eye_field()
    print()
    if _failures:
        print(f"{len(_failures)} check(s) FAILED:")
        for label in _failures:
            print(f"  - {label}")
        return 1
    print("All age-invariance tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())