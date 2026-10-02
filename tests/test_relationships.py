"""Relationship intersection: "which photos contain these people together".

Run: python tests/test_relationships.py

Covers the occurrence table in :mod:`src.names_db` (recording, intersection,
people list, co-occurrence, merge) and the four HTTP endpoints in
:mod:`src.api.server`.
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api.server import create_app  # noqa: E402
from src.names_db import NamesDB  # noqa: E402

_failures: list[str] = []
_checks = [0]


def check(label: str, condition: bool, detail: str = "") -> None:
    _checks[0] += 1
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}{(': ' + detail) if detail else ''}")
        _failures.append(label)


def make_db() -> tuple[NamesDB, Path]:
    tmp = Path(tempfile.mkdtemp(prefix="faceorg_rel_"))
    db = NamesDB(tmp / "rel.db")
    return db, tmp


def seed(db: NamesDB) -> None:
    """A small household: John and Mary share photos, John and Alice do too.

    John: a b c
    Mary: b c d
    Alice: c e
    Nobody: f
    """
    # A row in `persons` per person: that is what merge_persons folds, and the
    # occurrence table is what the queries read.
    for index, name in enumerate(("John", "Mary", "Alice", "Nobody")):
        vector = np.zeros(4, np.float32)
        vector[index] = 1.0
        db.upsert_person(name, vector, [], faces_seen=1)
    db.record_occurrences("John", ["a.jpg", "b.jpg", "c.jpg"], faces=2)
    db.record_occurrences("Mary", ["b.jpg", "c.jpg", "d.jpg"], faces=1)
    db.record_occurrences("Alice", ["c.jpg", "e.jpg"], faces=1)
    db.record_occurrences("Nobody", ["f.jpg"], faces=1)


# ---------------------------------------------------------------------------
def test_schema() -> None:
    print("occurrence table and indexes")
    db, tmp = make_db()
    try:
        tables = {
            row[0] for row in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")
        }
        check("face_occurrences exists", "face_occurrences" in tables, str(tables))
        indexes = {
            row[0] for row in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'")
        }
        for needed in ("idx_occurrences_image", "idx_occurrences_pair",
                       "idx_occurrences_person_rollup"):
            check(f"index {needed} exists", needed in indexes, str(sorted(indexes)))

        # opening an old database must add the table, not fail
        legacy = tmp / "legacy.db"
        conn = sqlite3.connect(legacy)
        conn.execute(
            """
            CREATE TABLE persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                embedding BLOB NOT NULL, dims INTEGER NOT NULL,
                sample_image_paths TEXT NOT NULL DEFAULT '[]',
                faces_seen INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO persons (person_name, embedding, dims, "
            "sample_image_paths, faces_seen, created_at, updated_at) "
            "VALUES ('Old', ?, 4, '[]', 2, '2024-01-01', '2024-01-01')",
            (np.array([1.0, 0, 0, 0], np.float32).tobytes(),),
        )
        conn.commit()
        conn.close()
        with NamesDB(legacy) as upgraded:
            tables = {
                row[0] for row in upgraded._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")
            }
            check("a legacy database gains the table on open",
                  "face_occurrences" in tables)
            check("its rows still read back",
                  upgraded.get_person("Old") is not None)
    finally:
        db.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_recording() -> None:
    print("recording occurrences")
    db, tmp = make_db()
    try:
        written = db.record_occurrences("John", ["a.jpg", "b.jpg"], faces=2)
        check("record_occurrences reports how many pairs it wrote",
              written == 2, str(written))
        check("the photos come back", db.photos_for_person("John") == ["a.jpg", "b.jpg"])

        # idempotent: the same pair must not duplicate
        db.record_occurrences("John", ["a.jpg", "b.jpg"], faces=1)
        rows = db._conn.execute(
            "SELECT COUNT(*) FROM face_occurrences WHERE person_name='John'"
        ).fetchone()[0]
        check("re-recording does not duplicate rows", rows == 2, str(rows))
        faces = db._conn.execute(
            "SELECT SUM(faces) FROM face_occurrences WHERE person_name='John'"
        ).fetchone()[0]
        check("re-recording accumulates the face count", faces == 6, str(faces))

        check("an empty name is ignored", db.record_occurrences("", ["a.jpg"]) == 0)
        check("no paths is ignored", db.record_occurrences("John", []) == 0)
        check("blank paths are skipped",
              db.record_occurrences("Ghost", ["", "  "]) == 0)
        check("forgetting removes them", db.forget_occurrences("John") >= 0)
    finally:
        db.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_intersection() -> None:
    print("intersection")
    db, tmp = make_db()
    try:
        seed(db)

        check("John AND Mary -> the shared photos",
              db.get_intersection(["John", "Mary"]) == ["b.jpg", "c.jpg"])
        check("John AND Mary AND Alice -> only c",
              db.get_intersection(["John", "Mary", "Alice"]) == ["c.jpg"])
        check("Alice AND Nobody -> nothing",
              db.get_intersection(["Alice", "Nobody"]) == [])
        check("one name -> all their photos",
              db.get_intersection(["Mary"]) == ["b.jpg", "c.jpg", "d.jpg"])
        check("no names -> nothing", db.get_intersection([]) == [])
        check("an unknown name -> nothing",
              db.get_intersection(["John", "Ghost"]) == [])
        check("matching is case-insensitive",
              db.get_intersection(["john", "MARY"]) == ["b.jpg", "c.jpg"])
        check("a duplicated name is collapsed",
              db.get_intersection(["John", "John", "Mary"]) == ["b.jpg", "c.jpg"])
        check("whitespace is trimmed",
              db.get_intersection([" John ", "Mary"]) == ["b.jpg", "c.jpg"])

        check("resolve_names separates matched from unknown",
              db.resolve_names(["john", "Ghost", "MARY"])
              == (["John", "Mary"], ["Ghost"]))
        check("people_in_photo lists everyone in one photo",
              db.people_in_photo("c.jpg") == ["Alice", "John", "Mary"],
              str(db.people_in_photo("c.jpg")))

        people = db.get_all_people()
        names = [entry["name"] for entry in people]
        check("every person is listed",
              names == ["Alice", "John", "Mary", "Nobody"], str(names))
        john = next(p for p in people if p["name"] == "John")
        check("photo counts are distinct photos", john["photos"] == 3,
              str(john["photos"]))
        check("face counts are summed", john["faces"] == 6, str(john["faces"]))
    finally:
        db.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_co_occurrence() -> None:
    print("co-occurrence")
    db, tmp = make_db()
    try:
        seed(db)
        pairs = {(p["a"], p["b"]): p["photos"] for p in db.get_co_occurrence()}
        check("John & Mary share 2 photos", pairs.get(("John", "Mary")) == 2,
              str(pairs))
        check("John & Alice share 1 photo", pairs.get(("Alice", "John")) == 1,
              str(pairs))
        check("pairs are alphabetical",
              all(p["a"].lower() < p["b"].lower() for p in db.get_co_occurrence()))
        check("ordered by descending count",
              [p["photos"] for p in db.get_co_occurrence()]
              == sorted((p["photos"] for p in db.get_co_occurrence()), reverse=True))
        check("a person with no shared photo is absent from pairs",
              not any("Nobody" in (p["a"], p["b"]) for p in db.get_co_occurrence()))
        check("a person with no shared photo is still in get_all_people",
              any(p["name"] == "Nobody" for p in db.get_all_people()))

        empty, tmp2 = make_db()
        try:
            check("no data yields no pairs", empty.get_co_occurrence() == [])
        finally:
            empty.close()
            shutil.rmtree(tmp2, ignore_errors=True)
    finally:
        db.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_merge_moves_occurrences() -> None:
    print("merging moves recorded photos")
    db, tmp = make_db()
    try:
        seed(db)
        merged = db.merge_persons("John", "Mary")
        check("the merge succeeds", merged is not None)
        photos = db.photos_for_person("John")
        check("John now has Mary's photos too",
              photos == ["a.jpg", "b.jpg", "c.jpg", "d.jpg"], str(photos))
        check("Mary has no recorded photos left",
              db.photos_for_person("Mary") == [])
        check("a photo both shared is not double-counted",
              photos.count("b.jpg") == 1 and photos.count("c.jpg") == 1)
        check("the union intersects with Alice correctly",
              # John is now a,b,c,d and Alice is c,e — so only c has both.
              db.get_intersection(["John", "Alice"]) == ["c.jpg"],
              str(db.get_intersection(["John", "Alice"])))
        check("Alice's own photos are untouched",
              db.photos_for_person("Alice") == ["c.jpg", "e.jpg"])
    finally:
        db.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_merge_occurrences_directly() -> None:
    print("merge_occurrences edge cases")
    db, tmp = make_db()
    try:
        db.record_occurrences("A", ["x.jpg", "y.jpg"])
        db.record_occurrences("B", ["y.jpg", "z.jpg"])
        moved = db.merge_occurrences("A", "B")
        check("moving occurrences reports how many", moved == 2, str(moved))
        check("photos move to the kept name",
              db.photos_for_person("A") == ["x.jpg", "y.jpg", "z.jpg"],
              str(db.photos_for_person("A")))
        check("the merged name keeps nothing",
              db.photos_for_person("B") == [])
        rows = db._conn.execute(
            "SELECT COUNT(*) FROM face_occurrences WHERE image_path='y.jpg'"
        ).fetchone()[0]
        check("the shared photo has exactly one row", rows == 1, str(rows))
        shared = db._conn.execute(
            "SELECT faces FROM face_occurrences WHERE image_path='y.jpg'"
        ).fetchone()[0]
        check("the shared photo keeps both faces' counts", shared == 2, str(shared))
        check("merging into itself is a no-op", db.merge_occurrences("A", "A") == 0)
        check("empty names are refused", db.merge_occurrences("", "A") == 0)
        check("a name with no photos is a no-op",
              db.merge_occurrences("A", "Ghost") == 0)
    finally:
        db.close()
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
def test_endpoints() -> None:
    print("HTTP endpoints")
    tmp = Path(tempfile.mkdtemp(prefix="faceorg_rel_api_"))
    try:
        photos = tmp / "photos"
        photos.mkdir()
        for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg", "f.jpg"):
            (photos / name).write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 64)

        db_path = tmp / "rel.db"
        db = NamesDB(db_path)
        seed(db)
        db.close()

        config = tmp / "config.yaml"
        config.write_text(
            f"input_folder: {photos.as_posix()}\n"
            f"output_folder: {(tmp / 'out').as_posix()}\n"
            f"names_db: {db_path.as_posix()}\n",
            encoding="utf-8",
        )

        app = create_app(config)
        with TestClient(app) as client:
            # Occurrences are recorded with absolute paths during a real scan;
            # seed them that way here.
            recorded = NamesDB(db_path)
            recorded._conn.execute("DELETE FROM face_occurrences")
            recorded.record_occurrences("John", [photos / n for n in ("a.jpg", "b.jpg", "c.jpg")])
            recorded.record_occurrences("Mary", [photos / n for n in ("b.jpg", "c.jpg", "d.jpg")])
            recorded.record_occurrences("Alice", [photos / n for n in ("c.jpg", "e.jpg")])
            recorded.record_occurrences("Nobody", [photos / "f.jpg"])
            recorded.close()

            listing = client.get("/people_list").json()
            check("GET /people_list works", listing["count"] == 4,
                  str(listing["count"]))
            check("it says the DB is available", listing["db"] is True)

            both = client.get("/intersection", params={"names": "John,Mary"}).json()
            check("GET /intersection returns the shared photos",
                  [Path(p).name for p in both["photos"]] == ["b.jpg", "c.jpg"],
                  str(both["photos"]))
            check("and a count", both["count"] == 2)
            check("and the resolved names", both["names"] == ["John", "Mary"])
            check("no unknown names", both["unknown"] == [])
            check("per-person counts are reported",
                  both["per_person"] == {"John": 3, "Mary": 3},
                  str(both["per_person"]))

            three = client.get(
                "/intersection", params={"names": "John,Mary,Alice"}).json()
            check("three names intersect to one photo",
                  [Path(p).name for p in three["photos"]] == ["c.jpg"],
                  str(three["photos"]))

            one = client.get("/intersection", params={"names": "Mary"}).json()
            check("a single name returns all their photos", one["count"] == 3)

            nothing = client.get(
                "/intersection", params={"names": "Alice,Nobody"}).json()
            check("no overlap returns nothing", nothing["count"] == 0)

            typo = client.get(
                "/intersection", params={"names": "John,Jonn"}).json()
            check("an unknown name is reported, not silently dropped",
                  typo["unknown"] == ["Jonn"], str(typo["unknown"]))
            check("the known name still matches",
                  typo["count"] == 3, str(typo["count"]))

            matrix = client.get("/co_occurrence_matrix").json()
            check("GET /co_occurrence_matrix works",
                  sorted(matrix["names"]) == ["Alice", "John", "Mary", "Nobody"],
                  str(matrix["names"]))
            check("it reports the maximum cell", matrix["max"] == 2,
                  str(matrix["max"]))
            check("and the pairs", len(matrix["pairs"]) == 3,
                  str(len(matrix["pairs"])))

            # export
            destination = tmp / "exported"
            destination.mkdir()
            bad = client.post("/export_intersection", json={
                "names": ["John", "Mary"], "output_folder": str(tmp / "nope"),
            })
            check("exporting to a missing folder is a 400", bad.status_code == 400)

            exported = client.post("/export_intersection", json={
                "names": ["John", "Mary"], "output_folder": str(destination),
            })
            check("export succeeds", exported.status_code == 200,
                  str(exported.status_code))
            body = exported.json()
            check("both photos copied", body["copied"] == 2, str(body["copied"]))
            check("the files are really there",
                  sorted(p.name for p in destination.iterdir()) == ["b.jpg", "c.jpg"],
                  str(sorted(p.name for p in destination.iterdir())))

            again = client.post("/export_intersection", json={
                "names": ["John", "Mary"], "output_folder": str(destination),
            })
            check("a repeated export does not overwrite",
                  len(list(destination.iterdir())) == 4,
                  str(len(list(destination.iterdir()))))

            no_match = client.post("/export_intersection", json={
                "names": ["Alice", "Nobody"], "output_folder": str(destination),
            })
            check("exporting an empty result is a 404",
                  no_match.status_code == 404)

            # the originals must be untouched
            check("the source photos are still in place",
                  len(list(photos.iterdir())) == 6)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_endpoints_without_db() -> None:
    print("endpoints with the database disabled")
    tmp = Path(tempfile.mkdtemp(prefix="faceorg_rel_nodb_"))
    try:
        photos = tmp / "photos"
        photos.mkdir()
        (photos / "a.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 64)
        config = tmp / "config.yaml"
        config.write_text(
            f"input_folder: {photos.as_posix()}\n"
            f"output_folder: {(tmp / 'out').as_posix()}\n"
            "names_db: ''\n",
            encoding="utf-8",
        )
        app = create_app(config)
        with TestClient(app) as client:
            # the API session config drives use_db; with no scan yet it is {}.
            # Force the disabled path explicitly.
            app.state.session.config["use_db"] = False
            listing = client.get("/people_list").json()
            check("people_list degrades gracefully",
                  listing["people"] == [] and listing["db"] is False)
            result = client.get("/intersection", params={"names": "John"}).json()
            check("intersection degrades gracefully",
                  result["count"] == 0 and result["db"] is False)
            check("and explains itself",
                  "disabled" in (result.get("error") or ""), str(result.get("error")))
            matrix = client.get("/co_occurrence_matrix").json()
            check("the matrix degrades gracefully",
                  matrix["pairs"] == [] and matrix["db"] is False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_scale() -> None:
    print("query cost at 50k+ rows")
    import random

    db, tmp = make_db()
    try:
        random.seed(11)
        people = [f"P{i:02d}" for i in range(40)]
        rows = [
            (random.choice(people), f"photo_{i % 8000:05d}.jpg", 1,
             "2026-10-02T00:00:00+00:00")
            for i in range(50000)
        ]
        db._conn.executemany(
            "INSERT OR IGNORE INTO face_occurrences "
            "(person_name, image_path, faces, seen_at) VALUES (?,?,?,?)",
            rows,
        )
        db._conn.commit()
        total = db._conn.execute(
            "SELECT COUNT(*) FROM face_occurrences").fetchone()[0]
        check(f"the table holds {total} rows (>= 50k)", total >= 45000, str(total))

        def timed(label, fn, budget_ms, repeat=10):
            fn()
            start = time.perf_counter()
            for _ in range(repeat):
                out = fn()
            ms = (time.perf_counter() - start) / repeat * 1000
            check(f"{label} under {budget_ms} ms (was {ms:.1f} ms)",
                  ms < budget_ms, f"{ms:.1f} ms")
            return out

        timed("a 2-name intersection", lambda: db.get_intersection(["P01", "P02"]), 60)
        timed("a 3-name intersection",
              lambda: db.get_intersection(["P01", "P02", "P03"]), 60)
        timed("one person's photos", lambda: db.photos_for_person("P01"), 60)
        timed("everyone in one photo",
              lambda: db.people_in_photo("photo_00042.jpg"), 30)
        timed("the people list", lambda: db.get_all_people(), 400)

        # the intersection query must use the covering index, not scan
        plan = db._conn.execute(
            "EXPLAIN QUERY PLAN SELECT image_path FROM face_occurrences "
            "WHERE person_name IN (?,?) GROUP BY image_path "
            "HAVING COUNT(DISTINCT person_name) = 2",
            ("P01", "P02"),
        ).fetchall()
        detail = " ".join(str(row[-1]) for row in plan)
        check("the intersection uses a covering index",
              "COVERING INDEX" in detail, detail)
        check("and does not full-scan the table",
              "SCAN face_occurrences" not in detail, detail)
    finally:
        db.close()
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    test_schema()
    test_recording()
    test_intersection()
    test_co_occurrence()
    test_merge_moves_occurrences()
    test_merge_occurrences_directly()
    test_endpoints()
    test_endpoints_without_db()
    test_scale()
    print()
    if _failures:
        print(f"{len(_failures)} of {_checks[0]} checks FAILED:")
        for label in _failures:
            print(f"  - {label}")
        return 1
    print(f"All relationship tests passed ({_checks[0]} checks).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())