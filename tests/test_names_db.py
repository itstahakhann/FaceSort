"""Unit tests for the persistent name database (M3, SPEC F11).

Run with::

    python tests/test_names_db.py
    pytest tests/

Uses temporary files and ``:memory:`` SQLite — no model weights, no prompts.
"""

import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.fusion import FusedMetric
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


def test_eye_embedding_storage() -> None:
    """Periocular centroids: stored, blended, and optional."""
    print("\neye embeddings (age invariance)")
    from src.names_db import Person

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "eyes.db"
        with NamesDB(path) as db:
            full_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            full_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            eye = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

            person = db.upsert_person(
                "Kid", full_a, ["a.jpg"], faces_seen=3,
                eye_embedding_centroid=eye, eye_faces_seen=3,
            )
            check("eye centroid is stored", person.has_eye_embedding)
            check("eye face count is stored", person.eye_faces_seen == 3)
            check("stored eye vector round-trips",
                  np.allclose(person.eye_embedding_centroid, eye))

            # a person with no eye vector at all is fine
            plain = db.upsert_person("Plain", full_b, ["b.jpg"], faces_seen=2)
            check("a person without eye embeddings is valid",
                  not plain.has_eye_embedding)
            check("their eye face count is zero", plain.eye_faces_seen == 0)

            # re-upserting WITHOUT an eye vector must not erase what is stored
            db.upsert_person("Kid", full_b, ["c.jpg"], faces_seen=1)
            again = db.get_person("Kid")
            check("a run without eye vectors keeps the stored centroid",
                  again.has_eye_embedding, "erased")
            check("the stored vector is unchanged",
                  np.allclose(again.eye_embedding_centroid, eye))

            # and a run WITH one blends
            blended = db.upsert_person(
                "Kid", full_b, ["d.jpg"], faces_seen=3,
                eye_embedding_centroid=eye, eye_faces_seen=3,
            )
            check("a second eye vector blends in", blended.eye_faces_seen == 6)
            check("blending identical vectors is a no-op",
                  np.allclose(blended.eye_embedding_centroid, eye, atol=1e-5))

            # an unusable eye vector is discarded, not raised
            db.upsert_person("Broken", full_a, [], faces_seen=1,
                             eye_embedding_centroid=np.zeros(4))
            check("a zero eye vector is discarded",
                  not db.get_person("Broken").has_eye_embedding)

            # merging two people folds both centroids together
            merged = db.merge_persons("Kid", "Plain")
            check("merge returns the surviving person", merged is not None)
            check("merge removes the second name",
                  db.get_person("Plain") is None)
            check("merge combines the face counts",
                  merged.faces_seen == 9, str(merged.faces_seen))
            check("merge keeps the eye centroid",
                  merged.has_eye_embedding)
            check("merge keeps both photo samples",
                  len(merged.sample_image_paths) == 4,
                  str(merged.sample_image_paths))

            # merging an unknown name is a no-op, not a crash
            check("merging an unknown name returns None",
                  db.merge_persons("Kid", "Nobody") is None)
            for bad in (("", "Kid"), ("Kid", "Kid")):
                try:
                    db.merge_persons(*bad)
                    check(f"invalid merge rejected: {bad}", False)
                except ValueError:
                    check(f"invalid merge rejected: {bad}", True)

    # matching uses the fused distance, and says which signal it leaned on
    print("\nfused matching against the name database")
    with tempfile.TemporaryDirectory() as tmp:
        with NamesDB(Path(tmp) / "match.db") as db:
            kid_full = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            kid_eye = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
            plain_full = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            db.upsert_person("Kid", kid_full, [], faces_seen=2,
                             eye_embedding_centroid=kid_eye, eye_faces_seen=2)
            db.upsert_person("Plain", plain_full, [], faces_seen=2)

            # a query matching Kid's whole face AND eye region
            match = db.find_match(kid_full, tolerance=2.0,
                                  eye_embedding_centroid=kid_eye)
            check("a match is returned", match is not None)
            if match:
                check("the closest person wins", match.person_name == "Kid",
                      match.person_name)
                check("the match reports that it used the eye region",
                      match.used_eye is True)
                check("both component distances are reported",
                      match.full_distance is not None
                      and match.eye_distance is not None)
                check("an exact match scores ~0", match.distance < 1e-6,
                      f"{match.distance:.4f}")

            # querying for Plain: it has no eye vector, so the fallback fires
            plain_match = db.find_match(plain_full, tolerance=2.0,
                                        eye_embedding_centroid=kid_eye)
            check("matching still works for eye-less people",
                  plain_match is not None)
            if plain_match:
                check("the eye-less person is chosen on the whole face",
                      plain_match.person_name == "Plain",
                      plain_match.person_name)
                check("the fallback is reported as whole-face only",
                      plain_match.used_eye is False)

            # a query with no eye vector still matches on the whole face
            no_eye_query = db.find_match(kid_full, tolerance=2.0)
            check("a query without eye data matches on the whole face",
                  no_eye_query is not None
                  and no_eye_query.person_name == "Kid")
            if no_eye_query:
                check("that match is reported as whole-face only",
                      no_eye_query.used_eye is False)

            # The point of the fused score: when the whole face leans one way and the
            # eye region the other, the eye region decides.  Both people need
            # an eye vector, otherwise the whole-face fallback would always win.
            other = db.get_person("Plain")
            db.upsert_person("Plain", plain_full, [], faces_seen=1,
                             eye_embedding_centroid=np.array([0.0, 0.0, 0.0, 1.0],
                                                             dtype=np.float32),
                             eye_faces_seen=1)
            kid_eye_exact = db.get_person("Kid").eye_embedding_centroid
            # query: whole face points at Plain (cos 0.954), eyes at Kid (cos 1.0)
            ambiguous_full = np.array([0.3, 0.954, 0.0, 0.0], dtype=np.float32)
            ambiguous_full /= np.linalg.norm(ambiguous_full)
            fused_kid = db.find_match(ambiguous_full, tolerance=2.0,
                                      eye_embedding_centroid=kid_eye_exact,
                                      metric=FusedMetric(0.4, 0.6))
            whole_only = db.find_match(ambiguous_full, tolerance=2.0,
                                       eye_embedding_centroid=kid_eye_exact,
                                       metric=FusedMetric(1.0, 0.0))
            check("fused matching prefers the eye-region match",
                  fused_kid is not None and fused_kid.person_name == "Kid",
                  f"fused chose {fused_kid.person_name if fused_kid else None}")
            check("whole-face-only matching would choose the other person",
                  whole_only is not None and whole_only.person_name == "Plain",
                  f"whole chose {whole_only.person_name if whole_only else None}")

    # backward compatibility: a database written before the eye columns
    print("\nbackward compatibility with an older database")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "old.db"
        conn = sqlite3.connect(path)
        conn.execute(
            """
            CREATE TABLE persons (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name        TEXT    NOT NULL COLLATE NOCASE UNIQUE,
                embedding          BLOB    NOT NULL,
                dims               INTEGER NOT NULL,
                sample_image_paths TEXT    NOT NULL DEFAULT '[]',
                faces_seen         INTEGER NOT NULL DEFAULT 1,
                created_at         TEXT    NOT NULL,
                updated_at         TEXT    NOT NULL
            )
            """
        )
        vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        conn.execute(
            "INSERT INTO persons (person_name, embedding, dims, "
            "sample_image_paths, faces_seen, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("Legacy", vector.tobytes(), 3, '["old.jpg"]', 4,
             "2024-01-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"),
        )
        conn.commit()
        conn.close()

        with NamesDB(path) as db:
            person = db.get_person("Legacy")
            check("an old row opens without error", person is not None)
            check("the old centroid is intact",
                  np.allclose(person.embedding_centroid, vector))
            check("the eye centroid reads as absent",
                  not person.has_eye_embedding)
            check("the eye face count defaults to zero",
                  person.eye_faces_seen == 0)

            match = db.find_match(vector, tolerance=0.5,
                                  eye_embedding_centroid=np.array([0.0, 1.0, 0.0],
                                                                   dtype=np.float32))
            check("an old row still matches on the whole face",
                  match is not None and match.person_name == "Legacy")
            if match:
                check("the legacy match reports the whole-face fallback",
                      match.used_eye is False)

            # upgrading it stores eye embeddings from then on
            upgraded = db.upsert_person(
                "Legacy", vector, [], faces_seen=1,
                eye_embedding_centroid=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            )
            check("an upgraded row gains an eye centroid",
                  upgraded.has_eye_embedding)

        # reopening after the migration must not try to add columns again
        with NamesDB(path) as db:
            check("reopening a migrated database is idempotent",
                  db.get_person("Legacy") is not None)


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
    test_eye_embedding_storage()
    test_validation_and_open_names_db()
    print()
    if _failures:
        print("FAILURES:", _failures)
        return 1
    print("All name DB tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
