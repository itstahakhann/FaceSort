"""Unit tests for the clustering engine (SPEC §7, §9, §10).

Run with::

    python tests/test_clusterer.py      # plain runner, no extra dependencies
    pytest tests/                       # also collectable by pytest

No model weights and no photos are needed: embeddings are synthesised so
that the DBSCAN behaviour itself is what gets checked.
"""

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clusterer import (
    FaceClusterer,
    FaceRecord,
    cluster_embeddings,
    deduplicate_records,
    make_face_records,
)
from src.detector import DetectedFace
from src.embedder import FaceEmbedding

logging.basicConfig(level=logging.CRITICAL)

_failures = []


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}: {label} {detail}")
    if not condition:
        _failures.append(label)


DIM = 64


def unit(seed: int, jitter: float = 0.01) -> np.ndarray:
    """A deterministic pseudo-face embedding around axis ``seed``."""
    rng = np.random.default_rng(seed)
    vector = np.zeros(DIM, dtype=np.float32)
    vector[seed % DIM] = 1.0
    vector += rng.normal(scale=jitter, size=DIM).astype(np.float32)
    return vector / np.linalg.norm(vector)


def record(image: str, embedding: np.ndarray, box=(0, 0, 10, 10), score=0.9):
    return FaceRecord(image_path=image, bbox=box, embedding=embedding, score=score)


def test_empty_and_trivial():
    clusterer = FaceClusterer(tolerance=0.5, min_faces_per_cluster=2)
    check("no records -> no clusters", clusterer.cluster([]) == [])
    check(
        "a single face is ignored when min_faces_per_cluster=2",
        clusterer.cluster([record("a.jpg", unit(1))]) == [],
    )
    check(
        "a single face is kept when min_faces_per_cluster=1",
        len(FaceClusterer(tolerance=0.5, min_faces_per_cluster=1).cluster(
            [record("a.jpg", unit(1))]
        )) == 1,
    )


def test_grouping_by_cosine_distance():
    # Person A in 4 photos, person B in 3 photos, one stranger each.
    records = (
        [record(f"a{i}.jpg", unit(0)) for i in range(4)]
        + [record(f"b{i}.jpg", unit(7)) for i in range(3)]
        + [record("lonely.jpg", unit(21))]
    )
    clusters = FaceClusterer(tolerance=0.5, min_faces_per_cluster=2).cluster(records)

    check("two people -> two clusters", len(clusters) == 2, str(len(clusters)))
    sizes = sorted(c.size for c in clusters)
    check("cluster sizes are 4 and 3 (singleton ignored)", sizes == [3, 4], str(sizes))

    groups = {c.cluster_id: {f.image_path.name for f in c.faces} for c in clusters}
    check(
        "people are not mixed",
        groups[0] == {f"a{i}.jpg" for i in range(4)}
        or groups[1] == {f"a{i}.jpg" for i in range(4)},
        str(groups),
    )
    check(
        "cluster_ids are 0..n-1",
        sorted(c.cluster_id for c in clusters) == [0, 1],
    )


def test_tolerance_comes_from_config():
    import yaml

    config = yaml.safe_load(
        Path(__file__).resolve().parent.parent.joinpath("config.yaml").read_text(
            encoding="utf-8"
        )
    )
    check("config.yaml carries tolerance=0.5", config["tolerance"] == 0.5)
    check("config.yaml carries min_faces_per_cluster=2",
          config["min_faces_per_cluster"] == 2)

    clusterer = FaceClusterer.from_config(config)
    check("from_config reads tolerance", clusterer.tolerance == 0.5)
    check("from_config reads min_faces_per_cluster",
          clusterer.min_faces_per_cluster == 2)

    # Two vectors with cosine distance ~0.6: inside the default eps of 0.5? no.
    v1 = np.zeros(DIM, dtype=np.float32)
    v2 = np.zeros(DIM, dtype=np.float32)
    v1[0] = 1.0
    v2[0] = np.cos(1.0)  # ~0.54 rad apart -> cosine distance ~0.13
    v2[1] = np.sin(1.0)
    close = [record("x.jpg", v1), record("y.jpg", v2)]
    v3 = np.zeros(DIM, dtype=np.float32)
    v3[0] = 1.0
    v4 = np.zeros(DIM, dtype=np.float32)
    v4[0] = 0.3
    v4[1] = 0.95  # cosine distance ~0.4+ -> borderline, controlled by eps
    wide = [record("x.jpg", v3), record("y.jpg", v4)]

    loose = FaceClusterer(tolerance=0.5, min_faces_per_cluster=2).cluster(close)
    strict = FaceClusterer(tolerance=0.05, min_faces_per_cluster=2).cluster(close)
    check("eps=0.5 keeps a close pair together", len(loose) == 1, str(len(loose)))
    check("eps=0.05 splits the same pair", len(strict) == 0, str(len(strict)))
    check(
        "cosine distance puts orthogonal faces apart",
        len(FaceClusterer(tolerance=0.5, min_faces_per_cluster=2).cluster(wide)) <= 1,
    )


def test_same_face_twice_in_one_image():
    embedding = unit(3)
    records = [
        # same photo, same face, two overlapping boxes
        record("p.jpg", embedding, box=(10, 10, 60, 60), score=0.98),
        record("p.jpg", unit(3, jitter=0.02), box=(14, 12, 62, 58), score=0.70),
        # a genuinely different face elsewhere in the photo
        record("p.jpg", unit(9), box=(200, 10, 250, 60), score=0.95),
    ]
    deduped = deduplicate_records(records)
    check("duplicate boxes in one photo collapse to one",
          len(deduped) == 2, str([r.bbox for r in deduped]))
    check("the strongest duplicate survives",
          deduped[0].score == 0.98 and deduped[0].bbox == (10, 10, 60, 60))

    # nested box (low IoU but the centres coincide) still counts as a duplicate
    nested = [
        record("q.jpg", embedding, box=(0, 0, 100, 100), score=0.6),
        record("q.jpg", embedding, box=(35, 35, 65, 65), score=0.9),
    ]
    check("nested box in the same photo collapses too",
          len(deduplicate_records(nested)) == 1)

    # the same boxes in *different* photos are two different observations
    other = [
        record("r1.jpg", embedding, box=(10, 10, 60, 60)),
        record("r2.jpg", embedding, box=(10, 10, 60, 60)),
    ]
    check("identical boxes in different photos are both kept",
          len(deduplicate_records(other)) == 2)

    # clustering after the merge must not double-count that face: the same
    # two people again in a second photo -> two clusters of two faces each
    records = records + [
        record("q.jpg", unit(3, jitter=0.03), box=(10, 10, 60, 60), score=0.97),
        record("q.jpg", unit(9, jitter=0.03), box=(200, 10, 250, 60), score=0.94),
    ]
    clusters = FaceClusterer(tolerance=0.5, min_faces_per_cluster=2).cluster(records)
    check("two people form two clusters after the merge", len(clusters) == 2,
          str([(c.cluster_id, c.size) for c in clusters]))
    check("the duplicate face is counted exactly once",
          sum(c.size for c in clusters) == 4,
          str(sum(c.size for c in clusters)))


def test_cluster_payload():
    records = [record(f"p{i}.jpg", unit(0)) for i in range(3)]
    records += [record(f"q{i}.jpg", unit(5)) for i in range(3)]
    clusters = FaceClusterer(tolerance=0.5, min_faces_per_cluster=2).cluster(records)

    cluster = clusters[0]
    check("cluster exposes image paths", len(cluster.image_paths) == 3,
          str(cluster.image_paths))
    check("cluster exposes bounding boxes",
          cluster.bboxes == [(0, 0, 10, 10)] * 3)
    check("cluster exposes a stacked embedding matrix",
          cluster.embeddings.shape == (3, DIM)
          and cluster.embeddings.dtype == np.float32)
    check("centroid is unit length",
          abs(float(np.linalg.norm(cluster.centroid)) - 1.0) < 1e-5)
    check("records are stamped with their cluster_id",
          all(face.cluster_id == cluster.cluster_id for face in cluster.faces))
    check("image_paths are unique",
          len(set(cluster.image_paths)) == len(cluster.image_paths))


def test_unusable_embeddings_are_dropped():
    ok = [record(f"a{i}.jpg", unit(0)) for i in range(3)]
    broken = [
        record("nan.jpg", np.full(DIM, np.nan, dtype=np.float32)),
        record("zero.jpg", np.zeros(DIM, dtype=np.float32)),
    ]
    clusters = FaceClusterer(tolerance=0.5, min_faces_per_cluster=2).cluster(
        ok + broken
    )
    check("NaN/zero embeddings never reach DBSCAN",
          len(clusters) == 1 and clusters[0].size == 3,
          str([(c.cluster_id, c.size) for c in clusters]))
    check("broken embeddings are not assigned to a cluster",
          all(face.cluster_id is None for face in broken))


def test_raw_labels_helper():
    matrix = np.stack([unit(0), unit(0, 0.001), unit(9)])
    labels = cluster_embeddings(matrix, tolerance=0.5)
    check("cluster_embeddings returns one label per face", len(labels) == 3)
    check("similar faces share a label", labels[0] == labels[1])
    check("different face gets a different label", labels[2] != labels[0])

    check("empty input -> empty labels", len(cluster_embeddings([])) == 0)
    try:
        cluster_embeddings([unit(0), np.full(DIM, np.inf)])
        check("non-finite input raises ValueError", False)
    except ValueError:
        check("non-finite input raises ValueError", True)
    try:
        cluster_embeddings([unit(0)], tolerance=5.0)
        check("invalid tolerance raises ValueError", False)
    except ValueError:
        check("invalid tolerance raises ValueError", True)


def test_glue_from_embedder_output():
    embedding = unit(2)
    items = [
        FaceEmbedding(
            detection=DetectedFace(bbox=(1, 2, 30, 40), score=0.9, kps=None),
            embedding=embedding,
        )
    ]
    records = make_face_records("photo.jpg", items)
    check("make_face_records maps path, box, embedding",
          len(records) == 1
          and records[0].image_path == Path("photo.jpg")
          and records[0].bbox == (1, 2, 30, 40)
          and records[0].score == 0.9
          and np.allclose(records[0].embedding, embedding))


def main():
    test_empty_and_trivial()
    test_grouping_by_cosine_distance()
    test_tolerance_comes_from_config()
    test_same_face_twice_in_one_image()
    test_cluster_payload()
    test_unusable_embeddings_are_dropped()
    test_raw_labels_helper()
    test_glue_from_embedder_output()
    print()
    if _failures:
        print("FAILURES:", _failures)
        return 1
    print("All clusterer tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
