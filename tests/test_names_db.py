"""Unit tests for the persistent name database (M3, SPEC F11).

Run with::

    python tests/test_names_db.py
    pytest tests/

Uses temporary files and ``:memory:`` SQLite — no model weights, no prompts.
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.names_db import (
    DB_FILENAME,
    DEFAULT_DB_PATH,
    MAX_SAMPLE_PATHS,
    NamesDB,
    cosine_distance,
    open_names_db,
)

_failures = []


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}: {label} {detail}")
    if not condition:
        _failures.append(label)


def unit(values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


ALICE = unit([1.0, 0.0, 0.0, 0.0])
BOB = unit([0.0, 1.0, 0.0, 0.0])
OPPOSITE = unit([-1.0, 0.0, 0.0, 0.0])


def test_cosine_distance():
    check("identical vectors -> 0", cosine_distance(ALICE, ALICE) < 1e-6)
    check("orthogonal vectors -> 1", abs(cosine_distance(ALICE, BOB) - 1.0) < 1e-6)
    check("opposite vectors -> 2", abs(cosine_distance(ALICE, OPPOSITE) - 2.0) < 1e-6)
    check("raw (unnormalised) input matches",
          abs(cosine_distance([2.0, 0.0], [0.0, 3.0]) - 1.0) < 1e-6)
    check("zero vector -> maximal distance", cosine_distance(ALICE, np.zeros(4)) == 2.0)
    try:
        cosine_distance(ALICE, BOB[:3])
        check("mismatched dims raise ValueError", False)
    except ValueError:
        check("mismatched dims raise ValueError", True)
    check("empty vectors -> maximal distance",
          cosine_distance(np.zeros(0), np.zeros(0)) == 2.0)


def test_upsert_and_match():
    with NamesDB(":memory:") as db:
        check("empty db matches nothing", db.find_match(ALICE, 0.5) is None)
        check("person count starts at 0", db.person_count() == 0)

        person = db.upsert_person(
            "Alice", ALICE, ["/photos/a1.jpg", "/photos/a2.jpg"], faces_seen=3
        )
        check("person is stored", person.person_name == "Alice")
        check("sample paths stored",
              person.sample_image_paths == ["/photos/a1.jpg", "/photos/a2.jpg"],
              str(person.sample_image_paths))
        check("faces_seen stored", person.faces_seen == 3)
        check("centroid is unit length",
              abs(float(np.linalg.norm(person.embedding_centroid)) - 1.0) < 1e-5)
        check("one known person", db.person_count() == 1)

        match = db.find_match(ALICE, tolerance=0.5)
        check("identical centroid matches", match is not None
              and match.person_name == "Alice")
        check("match distance is ~0", match is not None and match.distance < 1e-5)
        check("match carries samples", match is not None
              and match.sample_image_paths[0] == "/photos/a1.jpg")

        noisy = unit([0.95, 0.31, 0.0, 0.0])  # close to ALICE
        near = db.find_match(noisy, tolerance=0.5)
        check("nearby centroid matches", near is not None
              and near.person_name == "Alice" and near.distance < 0.5)
        check("strict tolerance rejects it",
              db.find_match(noisy, tolerance=0.01) is None)
        check("unrelated centroid is not labelled",
              db.find_match(OPPOSITE, tolerance=0.5) is None)
        check("unknown person -> None", db.find_match(unit([0, 0, 1, 0.0]),
                                                      0.5) is None)


def test_closest_wins():
    with NamesDB(":memory:") as db:
        db.upsert_person("Alice", ALICE)
        db.upsert_person("Bob", BOB)
        halfway = unit([1.0, 0.4, 0.0, 0.0])  # closer to Alice
        match = db.find_match(halfway, tolerance=1.5)
        check("closest known person wins", match is not None
              and match.person_name == "Alice", str(match))
        check("distance reported", match is not None and 0 < match.distance < 1)


def test_update_blends_centroid():
    with NamesDB(":memory:") as db:
        db.upsert_person("Alice", ALICE, ["/a.jpg"], faces_seen=1)
        shifted = unit([1.0, 0.5, 0.0, 0.0])
        person = db.upsert_person("Alice", shifted, ["/b.jpg"], faces_seen=1)

        check("faces_seen accumulates", person.faces_seen == 2,
              str(person.faces_seen))
        check("samples merge", person.sample_image_paths == ["/a.jpg", "/b.jpg"],
              str(person.sample_image_paths))
        centroid = person.embedding_centroid
        check("blended centroid sits between the two inputs",
              cosine_distance(centroid, ALICE) < cosine_distance(ALICE, shifted)
              and cosine_distance(centroid, shifted) < cosine_distance(ALICE, shifted),
              f"d_alice={cosine_distance(centroid, ALICE):.3f} "
              f"d_shifted={cosine_distance(centroid, shifted):.3f}")
        check("still one row per person", db.person_count() == 1)


def test_case_insensitive_identity():
    with NamesDB(":memory:") as db:
        db.upsert_person("Alice", ALICE)
        db.upsert_person("ALICE", BOB, faces_seen=5)
        check("same person regardless of case -> one row", db.person_count() == 1)
        person = db.get_person("alice")
        check("lookup ignores case", person is not None)
        check("latest spelling is kept", person is not None
              and person.person_name == "ALICE")
        check("new evidence moves the stored centroid",
              person is not None
              and cosine_distance(person.embedding_centroid, BOB)
              < cosine_distance(person.embedding_centroid, ALICE),
              "" if person is None else
              f"d_bob={cosine_distance(person.embedding_centroid, BOB):.3f} "
              f"d_alice={cosine_distance(person.embedding_centroid, ALICE):.3f}")
        check("delete works", db.delete_person("aLiCe")
              and db.person_count() == 0)
        check("deleting a stranger is harmless", not db.delete_person("nobody"))


def test_sample_capping_and_dims():
    with NamesDB(":memory:") as db:
        many = [f"/photos/img_{i}.jpg" for i in range(MAX_SAMPLE_PATHS + 5)]
        person = db.upsert_person("Alice", ALICE, many)
        check("samples capped", len(person.sample_image_paths) == MAX_SAMPLE_PATHS,
              str(len(person.sample_image_paths)))

        other_dims = unit([1.0, 0.0, 0.0])
        db.upsert_person("Bob", other_dims)
        check("dims stored per person",
              db.get_person("Bob").embedding_centroid.size == 3)
        check("different embedding sizes are skipped, not crashed",
              db.find_match(ALICE, tolerance=1.0) is not None
              and db.get_person("Alice").dims == 4)
        check("match still finds the compatible person",
              db.find_match(ALICE, tolerance=0.5).person_name == "Alice")


def test_persistence_across_reopen():
    root = Path(tempfile.mkdtemp(prefix="faceorg_db_"))
    path = root / "sub" / "names.db"  # parent dir must be created
    try:
        with NamesDB(path) as db:
            db.upsert_person("Alice", ALICE, ["/photos/a1.jpg"])
        check("database file created", path.is_file())

        with NamesDB(path) as db:
            person = db.get_person("Alice")
            check("rows survive a reopen", person is not None)
            check("centroid survives a reopen", person is not None
                  and cosine_distance(person.embedding_centroid, ALICE) < 1e-5)
            check("samples survive a reopen", person is not None
                  and person.sample_image_paths == ["/photos/a1.jpg"])
            check("person count survives", db.person_count() == 1)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_validation_and_open_names_db():
    with NamesDB(":memory:") as db:
        try:
            db.upsert_person("   ", ALICE)
            check("blank name rejected", False)
        except ValueError:
            check("blank name rejected", True)
        try:
            db.upsert_person("Nobody", np.zeros(4, dtype=np.float32))
            check("zero-length centroid rejected", False)
        except ValueError:
            check("zero-length centroid rejected", True)

    check("disabled config -> no db", open_names_db({"names_db": ""}) is None)
    check("names_db: none -> no db", open_names_db({"names_db": "none"}) is None)

    # With no names_db key (or the legacy default name) the DB must land in the
    # per-user data dir - NOT the working directory, which may be read-only or a
    # developer's source tree.
    import os

    root = Path(tempfile.mkdtemp(prefix="faceorg_default_"))
    sandbox = Path(tempfile.mkdtemp(prefix="faceorg_datadir_"))
    previous = Path.cwd()
    previous_override = os.environ.get("FACEORG_DATA_DIR")
    try:
        os.chdir(root)                      # cwd must not influence anything
        os.environ["FACEORG_DATA_DIR"] = str(sandbox)
        db = open_names_db({})
        expected = str(sandbox / DB_FILENAME)
        check("missing key falls back to the per-user default path",
              db is not None and db.path == expected,
              "" if db is None else db.path)
        if db is not None:
            db.close()
        check("default db file lands in the data dir, not the cwd",
              (sandbox / DB_FILENAME).is_file()
              and not (root / DB_FILENAME).exists())

        legacy = open_names_db({"names_db": "./facesort_names.db"})
        if legacy is not None:
            legacy.close()
        check("legacy './facesort_names.db' resolves to the same file",
              (sandbox / DB_FILENAME).is_file()
              and not (root / DB_FILENAME).exists())
    finally:
        if previous_override is None:
            os.environ.pop("FACEORG_DATA_DIR", None)
        else:
            os.environ["FACEORG_DATA_DIR"] = previous_override
        os.chdir(previous)
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(sandbox, ignore_errors=True)

    # opening a directory as the DB path must degrade, not raise
    directory = tempfile.mkdtemp()
    try:
        check("unusable path -> warning, not exception",
              open_names_db({"names_db": directory}) is None)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def main():
    test_cosine_distance()
    test_upsert_and_match()
    test_closest_wins()
    test_update_blends_centroid()
    test_case_insensitive_identity()
    test_sample_capping_and_dims()
    test_persistence_across_reopen()
    test_validation_and_open_names_db()
    print()
    if _failures:
        print("FAILURES:", _failures)
        return 1
    print("All name DB tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
