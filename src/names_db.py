"""Persistent name database (M3, SPEC F11) — SQLite, fully local.

Remembers who each person is across runs: one row per person holding the
name, the L2-normalised **embedding centroid** of every face ever seen for
them, and a handful of **sample image paths** (human-readable proof of who
is who).

The CLI compares each new cluster centroid against the database with the
same cosine ``tolerance`` used for clustering, so a cluster whose centroid
is close enough to a known person is auto-labelled instead of prompting.
Only names the user actually typed are written back.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .fusion import FusedMetric

logger = logging.getLogger(__name__)

#: Filename inside the per-user data directory.
DB_FILENAME = "facesort_names.db"
#: Folder created inside the platform's per-user data location.
APP_DIR_NAME = "FaceSort"

#: Historical spelling: treated as "use the per-user default" so existing
#: ``config.yaml`` files (which say ``./facesort_names.db``) keep working
#: without writing into whatever directory the app happens to run from.
LEGACY_DEFAULT_NAMES = ("./" + DB_FILENAME, DB_FILENAME)

DEFAULT_TOLERANCE = 0.5
MAX_SAMPLE_PATHS = 8


def user_data_dir() -> Path:
    """Writable per-user data directory — never the install folder.

    The engine is frozen next to the app, so the process working directory can
    be read-only (Program Files) or a developer's source tree.  The database
    has to live somewhere the user owns.
    """
    override = os.environ.get("FACEORG_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME")
                    or (Path.home() / ".local" / "share"))
    return root / APP_DIR_NAME


def default_db_path() -> Path:
    """Absolute path of the name database inside :func:`user_data_dir`."""
    return user_data_dir() / DB_FILENAME


def resolve_db_path(raw: object) -> Path:
    """Turn a configured ``names_db`` value into a concrete path.

    An explicit path (absolute, or relative like ``./data/names.db``) is
    honoured as given; the legacy default name resolves to the per-user
    location instead of the current working directory.
    """
    text = str(raw).strip()
    if text in LEGACY_DEFAULT_NAMES:
        return default_db_path()
    return Path(text).expanduser()


DEFAULT_DB_PATH = str(default_db_path())

__all__ = [
    "Person",
    "PersonMatch",
    "NamesDB",
    "open_names_db",
    "cosine_distance",
    "resolve_db_path",
    "user_data_dir",
    "default_db_path",
    "DEFAULT_DB_PATH",
]

#: Schema columns added after the first release, applied on open.  Kept as a
#: module constant so the migration is documented in one place and testable.
MIGRATED_COLUMNS: Mapping[str, Mapping[str, str]] = {
    "persons": {
        "eye_embedding": "BLOB",
        "eye_dims": "INTEGER",
        "eye_faces_seen": "INTEGER NOT NULL DEFAULT 0",
    },
}


def cosine_distance(a, b) -> float:
    """Cosine distance (0 = identical direction, 2 = opposite).

    Vectors are normalised internally, so raw or pre-normalised embeddings
    behave the same.  Missing/zero-length vectors count as maximally
    different instead of raising.
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size != b.size:
        raise ValueError(f"embedding sizes differ: {a.size} vs {b.size}")
    if a.size == 0:
        return 2.0
    norm = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if not np.isfinite(norm) or norm <= 0.0:
        return 2.0
    similarity = float(np.dot(a, b)) / norm
    return 1.0 - float(np.clip(similarity, -1.0, 1.0))


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.size == 0 or not bool(np.all(np.isfinite(vector))):
        raise ValueError("embedding is empty or not finite")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("embedding has zero length")
    return vector / norm


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Person:
    """One remembered person (SPEC §8 ``Person``).

    ``eye_embedding_centroid`` is the periocular mean used for age-invariant
    matching (see :mod:`src.fusion`).  It is ``None`` for rows written before
    this column existed, and for people whose photos were too small or blurred
    to fingerprint — both are read back without error, and matching falls back
    to the whole-face vector.
    """

    person_name: str
    embedding_centroid: np.ndarray
    eye_embedding_centroid: Optional[np.ndarray] = None
    sample_image_paths: List[str] = field(default_factory=list)
    faces_seen: int = 1
    eye_faces_seen: int = 0
    created_at: str = ""
    updated_at: str = ""

    @property
    def dims(self) -> int:
        return int(self.embedding_centroid.size)

    @property
    def has_eye_embedding(self) -> bool:
        """True when a usable periocular centroid is stored."""
        vector = self.eye_embedding_centroid
        return vector is not None and int(vector.size) > 0


@dataclass
class PersonMatch:
    """The known person a cluster centroid was matched against."""

    person_name: str
    distance: float
    sample_image_paths: List[str] = field(default_factory=list)
    #: Which signals produced ``distance`` — lets the UI explain that a match
    #: leaned on the eye region (age-invariant) or the whole face (fallback).
    used_eye: bool = False
    full_distance: Optional[float] = None
    eye_distance: Optional[float] = None


class NamesDB:
    """SQLite-backed store of known people. One row per person."""

    def __init__(self, path: Union[str, Path] = DEFAULT_DB_PATH) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    # ------------------------------------------------------------- lifecycle
    def _create_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS persons (
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
        # Periocular ("eye region") centroid, added after the first release.
        # Nullable on purpose: rows written by an older build have no eye
        # embedding, and a person remembered from low-resolution photos may
        # have none either. Matching falls back to the whole-face vector in
        # both cases rather than failing.
        self._ensure_columns("persons", MIGRATED_COLUMNS["persons"])
        self._create_occurrence_schema()
        self._conn.commit()

    def _create_occurrence_schema(self) -> None:
        """Create the per-(person, photo) table used by "who is in this photo".

        The ``persons`` table cannot answer "which photos contain both Alex and
        Sam": it stores one centroid per person plus a *capped* list of sample
        paths (see :data:`MAX_SAMPLE_PATHS`) for display.  An intersection over
        those samples would silently miss most of a person's photos, so
        occurrences get their own table — one row per (person, photo) pair.

        ``UNIQUE(person_name, image_path)`` makes re-recording idempotent, so
        re-naming or re-running a scan converges instead of duplicating.
        """
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS face_occurrences (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT NOT NULL COLLATE NOCASE,
                image_path  TEXT NOT NULL,
                faces       INTEGER NOT NULL DEFAULT 1,
                seen_at     TEXT NOT NULL,
                UNIQUE(person_name, image_path)
            )
            """
        )
        # "Which people are in this photo" — the reverse lookup.
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_occurrences_image
                ON face_occurrences(image_path)
            """
        )
        # Covering index for the intersection: GROUP BY image_path while
        # filtering on person_name reads person_name from the index itself.
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_occurrences_pair
                ON face_occurrences(person_name, image_path)
            """
        )
        # Per-person rollup (the picker list and the "of N photos" total).
        # The collation is spelled out on the index so it matches the
        # `GROUP BY person_name COLLATE NOCASE` in get_all_people: with a
        # matching collation SQLite walks the index in order instead of
        # building a temp b-tree, which is the difference between ~15ms and
        # ~200ms on a 50k-row table. It also covers the query, so the table
        # itself is never read.
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_occurrences_person_rollup
                ON face_occurrences(person_name COLLATE NOCASE, faces, seen_at)
            """
        )
        # Older databases (and the first release of this table) have the
        # single-column variant; it is now redundant.
        self._conn.execute("DROP INDEX IF EXISTS idx_occurrences_person")

    def _ensure_columns(
        self, table: str, columns: Mapping[str, str]
    ) -> None:
        """Add any missing columns to ``table`` (idempotent migration).

        SQLite has no ``ADD COLUMN IF NOT EXISTS``, so the existing columns
        are read back and only the genuinely new ones are added. This is what
        lets an old database file open without a manual migration step.
        """
        present = {
            str(row["name"])
            for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for name, definition in columns.items():
            if name in present:
                continue
            self._conn.execute(
                f"ALTER TABLE {table} ADD COLUMN {name} {definition}"
            )
            logger.info("Migrated %s: added column %r", table, name)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "NamesDB":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ----------------------------------------------------------------- write
    def upsert_person(
        self,
        person_name: str,
        embedding_centroid,
        sample_image_paths: Sequence[Union[str, Path]] = (),
        faces_seen: int = 1,
        eye_embedding_centroid=None,
        eye_faces_seen: Optional[int] = None,
    ) -> Person:
        """Remember (or refresh) a person.

        Re-naming an existing person blends the new centroid into the old
        one weighted by how many faces each side contributed, so the stored
        centroid slowly converges on the person's true average.  Sample
        paths are merged and capped.

        ``eye_embedding_centroid`` is the periocular mean used for
        age-invariant matching.  It is blended by the same rule as the
        whole-face centroid and is optional: passing ``None`` (or zero
        ``eye_faces_seen``) leaves any previously stored eye centroid intact
        rather than discarding it, so a run over low-resolution photos cannot
        erase what a good run learned.
        """
        name = str(person_name or "").strip()
        if not name:
            raise ValueError("person_name must not be empty")
        centroid = _unit(embedding_centroid)
        faces_seen = max(1, int(faces_seen))

        try:
            eye_centroid = None if eye_embedding_centroid is None else _unit(
                eye_embedding_centroid
            )
        except ValueError:
            logger.warning(
                "Discarding an unusable eye centroid for %r; the whole-face "
                "vector will be used for matching.", name,
            )
            eye_centroid = None
        eye_faces = (
            0 if eye_centroid is None
            else max(1, int(eye_faces_seen if eye_faces_seen is not None else 1))
        )

        samples = self._merge_samples([], _clean_samples(sample_image_paths))
        row = self._get_row(name)
        if row is None:
            now = _now()
            self._conn.execute(
                """
                INSERT INTO persons (person_name, embedding, dims,
                                     sample_image_paths, faces_seen,
                                     eye_embedding, eye_dims, eye_faces_seen,
                                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    centroid.tobytes(),
                    centroid.size,
                    json.dumps(samples),
                    faces_seen,
                    None if eye_centroid is None else eye_centroid.tobytes(),
                    None if eye_centroid is None else eye_centroid.size,
                    eye_faces,
                    now,
                    now,
                ),
            )
        else:
            previous = np.frombuffer(row["embedding"], dtype=np.float32)
            previous_faces = max(1, int(row["faces_seen"]))
            if previous.size == centroid.size:
                blended = previous * previous_faces + centroid * faces_seen
                blended = _unit(blended)
            else:  # dimension change (different model) -> start fresh
                logger.warning(
                    "Embedding size for %r changed (%d -> %d); replacing it.",
                    name, previous.size, centroid.size,
                )
                blended = centroid
            samples = self._merge_samples(_decode_samples(row["sample_image_paths"]),
                                          samples)

            previous_eye = self._stored_eye_vector(row)
            previous_eye_faces = max(0, int(row["eye_faces_seen"] or 0))
            if eye_centroid is None:
                # Nothing new to learn: keep whatever was stored before.
                eye_blob = previous_eye
                eye_faces_stored = previous_eye_faces
            elif previous_eye is not None and previous_eye.size == eye_centroid.size:
                total = previous_eye_faces + eye_faces
                blended_eye = _unit(
                    previous_eye * previous_eye_faces + eye_centroid * eye_faces
                )
                eye_blob = blended_eye
                eye_faces_stored = total
            else:
                if previous_eye is not None:
                    logger.warning(
                        "Eye embedding size for %r changed (%d -> %d); "
                        "replacing it.", name, previous_eye.size, eye_centroid.size,
                    )
                eye_blob = eye_centroid
                eye_faces_stored = eye_faces

            self._conn.execute(
                """
                UPDATE persons
                   SET person_name = ?, embedding = ?, dims = ?,
                       sample_image_paths = ?, faces_seen = ?,
                       eye_embedding = ?, eye_dims = ?, eye_faces_seen = ?,
                       updated_at = ?
                 WHERE id = ?
                """,
                (
                    name,          # keep the spelling the user just typed
                    blended.tobytes(),
                    blended.size,
                    json.dumps(samples),
                    previous_faces + faces_seen,
                    None if eye_blob is None else eye_blob.tobytes(),
                    None if eye_blob is None else int(eye_blob.size),
                    eye_faces_stored,
                    _now(),
                    row["id"],
                ),
            )
        self._conn.commit()

        person = self.get_person(name)
        if person is None:  # pragma: no cover - concurrency is not a thing here
            raise sqlite3.OperationalError("person disappeared after upsert")
        logger.debug(
            "Remembered %r (%d face(s), %d eye region(s), %d sample photo(s)).",
            name, person.faces_seen, person.eye_faces_seen,
            len(person.sample_image_paths),
        )
        return person

    def delete_person(self, person_name: str) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM persons WHERE person_name = ? COLLATE NOCASE",
            (str(person_name),),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------ read
    def get_person(self, person_name: str) -> Optional[Person]:
        row = self._get_row(person_name)
        return self._row_to_person(row) if row is not None else None

    def list_persons(self) -> List[Person]:
        rows = self._conn.execute(
            "SELECT * FROM persons ORDER BY person_name COLLATE NOCASE"
        ).fetchall()
        return [self._row_to_person(row) for row in rows]

    def person_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM persons").fetchone()
        return int(row["n"])

    def find_match(
        self,
        embedding_centroid,
        tolerance: float = DEFAULT_TOLERANCE,
        eye_embedding_centroid=None,
        metric: Optional[FusedMetric] = None,
    ) -> Optional[PersonMatch]:
        """Closest known person whose centroid is within ``tolerance``.

        Distances are the same fused distance the clusterer uses (see
        :mod:`src.fusion`), so ``tolerance`` means exactly what it means for
        clustering — and a person's remembered childhood photos match their
        adult photos through the eye region.

        When a person has no stored eye centroid (an older database row, or
        photos that were too small to fingerprint) the comparison falls back
        to the whole face, and the returned :class:`PersonMatch` says so via
        ``used_eye=False``.
        """
        if embedding_centroid is None:
            return None
        try:
            centroid = _unit(embedding_centroid)
        except ValueError as exc:
            logger.warning("Cannot match against the name DB: %s", exc)
            return None

        metric = metric or FusedMetric()
        eye = None
        if eye_embedding_centroid is not None:
            try:
                eye = _unit(eye_embedding_centroid)
            except ValueError:
                eye = None

        matches: List[PersonMatch] = []
        for person in self.list_persons():
            if person.dims != centroid.size:
                logger.debug(
                    "Skipping %r: embedding size %d != %d",
                    person.person_name, person.dims, centroid.size,
                )
                continue
            person_eye = person.eye_embedding_centroid
            usable_eye = (
                eye is not None
                and person_eye is not None
                and person_eye.size == eye.size
            )
            distance = metric(
                centroid, person.embedding_centroid,
                eye if usable_eye else None,
                person_eye if usable_eye else None,
            )
            if distance is None or distance > tolerance:
                continue
            matches.append(
                PersonMatch(
                    person_name=person.person_name,
                    distance=float(distance),
                    sample_image_paths=list(person.sample_image_paths),
                    used_eye=usable_eye,
                    full_distance=cosine_distance(
                        centroid, person.embedding_centroid
                    ),
                    eye_distance=(
                        cosine_distance(eye, person_eye) if usable_eye else None
                    ),
                )
            )
        if not matches:
            return None

        matches.sort(key=lambda match: match.distance)
        best = matches[0]
        if len(matches) > 1:
            runner_up = matches[1]
            logger.warning(
                "Centroid is within tolerance of %d known people; using the "
                "closest one (%r at %.3f, next %r at %.3f).",
                len(matches), best.person_name, best.distance,
                runner_up.person_name, runner_up.distance,
            )
        return best

    def merge_persons(
        self,
        keep_name: str,
        other_name: str,
    ) -> Optional[Person]:
        """Fold ``other_name`` into ``keep_name`` and delete it.

        This is the "same person, different age" safety net: the user states
        the ground truth, and both centroids are combined (weighted by how
        many faces each contributed) so future runs match the merged person
        through either signal.

        Recorded photo occurrences are re-pointed too, so "which photos contain
        both of them and Mary" keeps working for the photos that were only ever
        filed under the discarded name.

        The name that survives is ``keep_name`` — pass the one the user wants
        to keep.  Returns the merged person, or ``None`` if either name was
        unknown or they are the same row.
        """
        keep = str(keep_name or "").strip()
        other = str(other_name or "").strip()
        if not keep or not other:
            raise ValueError("both names must not be empty")
        if keep.lower() == other.lower():
            raise ValueError("cannot merge a person into themselves")

        keep_row = self._get_row(keep)
        other_row = self._get_row(other)
        if keep_row is None or other_row is None:
            missing = keep if keep_row is None else other
            logger.warning("Cannot merge: %r is not in the database.", missing)
            return None

        keep_faces = max(1, int(keep_row["faces_seen"]))
        other_faces = max(1, int(other_row["faces_seen"]))
        keep_vec = np.frombuffer(keep_row["embedding"], dtype=np.float32)
        other_vec = np.frombuffer(other_row["embedding"], dtype=np.float32)
        if keep_vec.size != other_vec.size:
            raise ValueError(
                "Cannot merge: embedding sizes differ "
                f"({keep_vec.size} vs {other_vec.size})."
            )
        merged = _unit(keep_vec * keep_faces + other_vec * other_faces)

        merged_eye = self._merge_eye_vectors(keep_row, other_row, keep, other)

        samples = self._merge_samples(
            _decode_samples(keep_row["sample_image_paths"]),
            _decode_samples(other_row["sample_image_paths"]),
        )
        self._conn.execute(
            """
            UPDATE persons
               SET embedding = ?, dims = ?, sample_image_paths = ?,
                   faces_seen = ?, eye_embedding = ?, eye_dims = ?,
                   eye_faces_seen = ?, updated_at = ?
             WHERE id = ?
            """,
            (
                merged.tobytes(),
                merged.size,
                json.dumps(samples),
                keep_faces + other_faces,
                None if merged_eye is None else merged_eye.tobytes(),
                None if merged_eye is None else int(merged_eye.size),
                keep_faces + other_faces if merged_eye is not None else 0,
                _now(),
                keep_row["id"],
            ),
        )
        self._conn.execute(
            "DELETE FROM persons WHERE id = ?", (other_row["id"],)
        )
        # Move this person's recorded photos onto the surviving name, so the
        # intersection queries keep seeing them.
        try:
            self.merge_occurrences(keep, other)
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            logger.warning(
                "Merged %r into %r but could not move photo occurrences: %s",
                other, keep, exc,
            )
        self._conn.commit()
        logger.info(
            "Merged %r into %r (%d + %d faces).", other, keep, other_faces, keep_faces,
        )
        return self.get_person(keep)

    def _merge_eye_vectors(
        self,
        keep_row: sqlite3.Row,
        other_row: sqlite3.Row,
        keep: str,
        other: str,
    ) -> Optional[np.ndarray]:
        """Combine two stored eye centroids, tolerating either being absent."""
        keep_eye = self._stored_eye_vector(keep_row)
        other_eye = self._stored_eye_vector(other_row)
        if keep_eye is None:
            return other_eye
        if other_eye is None:
            return keep_eye
        if keep_eye.size != other_eye.size:
            logger.warning(
                "Eye embedding sizes differ for %r (%d) and %r (%d); keeping "
                "the first one's eye centroid.",
                keep, keep_eye.size, other, other_eye.size,
            )
            return keep_eye
        keep_faces = max(1, int(keep_row["eye_faces_seen"] or 1))
        other_faces = max(1, int(other_row["eye_faces_seen"] or 1))
        return _unit(keep_eye * keep_faces + other_eye * other_faces)

    @staticmethod
    def _stored_eye_vector(row: sqlite3.Row) -> Optional[np.ndarray]:
        """Read a row's eye centroid, tolerating older rows without one."""
        try:
            blob = row["eye_embedding"]
        except (IndexError, KeyError):
            return None  # pragma: no cover - column always exists post-migration
        if blob is None:
            return None
        vector = np.frombuffer(blob, dtype=np.float32)
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            return None
        return vector

    # ------------------------------------------------- photo occurrences
    # "Who is in this photo, and who appears with whom" needs one row per
    # (person, photo). See _create_occurrence_schema for why the persons table
    # cannot serve this.

    def record_occurrences(
        self,
        person_name: str,
        image_paths: Sequence[Union[str, Path]],
        faces: int = 1,
    ) -> int:
        """Note that ``person_name`` appears in each of ``image_paths``.

        Idempotent: re-recording an existing pair updates the face count rather
        than adding a duplicate, so repeated scans of the same folder converge.
        Returns the number of pairs written.
        """
        name = str(person_name or "").strip()
        if not name:
            return 0
        paths = [str(path) for path in image_paths or () if str(path).strip()]
        if not paths:
            return 0

        now = _now()
        faces = max(1, int(faces))
        cursor = self._conn.executemany(
            """
            INSERT INTO face_occurrences (person_name, image_path, faces, seen_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(person_name, image_path) DO UPDATE SET
                faces   = faces + excluded.faces,
                seen_at = excluded.seen_at
            """,
            [(name, path, faces, now) for path in paths],
        )
        self._conn.commit()
        return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else len(paths)

    def forget_occurrences(self, person_name: str) -> int:
        """Drop every recorded photo for one person (they were re-named)."""
        cursor = self._conn.execute(
            "DELETE FROM face_occurrences WHERE person_name = ? COLLATE NOCASE",
            (str(person_name),),
        )
        self._conn.commit()
        return cursor.rowcount

    def merge_occurrences(self, keep_name: str, other_name: str) -> int:
        """Fold ``other_name``'s photos into ``keep_name``.

        Called when two remembered people are merged: without this the merged
        person would keep only the photos recorded under the surviving name, and
        "find photos with them and X" would go quiet for exactly the photos the
        user just told us belong to them.

        Photos both people were in collapse into a single row with their face
        counts summed, so no occurrence is lost. Written as read-then-insert
        rather than a plain ``UPDATE`` because the table is
        ``UNIQUE(person_name, image_path)``: an in-place update trips that
        constraint on every shared photo, and SQLite then applies only part of
        the change -- which is why the first attempt of this method appeared to
        merge nothing at all.
        """
        keep = str(keep_name or "").strip()
        other = str(other_name or "").strip()
        if not keep or not other or keep.lower() == other.lower():
            return 0

        # One spelling per person: if both "John" and "john" have rows, resolve
        # to a single name or resolve_names would report the person twice.
        canonical = self._conn.execute(
            "SELECT MIN(person_name) FROM face_occurrences "
            "WHERE person_name = ? COLLATE NOCASE",
            (keep,),
        ).fetchone()[0]
        keep = str(canonical) if canonical else keep

        moving = self._conn.execute(
            "SELECT image_path, faces FROM face_occurrences "
            "WHERE person_name = ? COLLATE NOCASE",
            (other,),
        ).fetchall()
        if not moving:
            return 0

        existing = {
            str(row["image_path"]): int(row["faces"] or 1)
            for row in self._conn.execute(
                "SELECT image_path, faces FROM face_occurrences "
                "WHERE person_name = ? COLLATE NOCASE",
                (keep,),
            )
        }
        now = _now()
        for row in moving:
            path = str(row["image_path"])
            faces = int(row["faces"] or 1) + existing.get(path, 0)
            if path in existing:
                self._conn.execute(
                    "UPDATE face_occurrences SET faces = ?, seen_at = ? "
                    "WHERE person_name = ? COLLATE NOCASE AND image_path = ?",
                    (faces, now, keep, path),
                )
            else:
                self._conn.execute(
                    "INSERT INTO face_occurrences "
                    "(person_name, image_path, faces, seen_at) VALUES (?,?,?,?)",
                    (keep, path, faces, now),
                )
        self._conn.execute(
            "DELETE FROM face_occurrences WHERE person_name = ? COLLATE NOCASE",
            (other,),
        )
        self._conn.commit()
        return len(moving)

    def known_names(self) -> List[str]:
        """Every person with at least one recorded photo, alphabetically."""
        rows = self._conn.execute(
            """
            SELECT person_name, MIN(person_name) AS key
              FROM face_occurrences
             GROUP BY person_name COLLATE NOCASE
             ORDER BY key COLLATE NOCASE
            """
        ).fetchall()
        return [str(row["key"]) for row in rows]

    def get_all_people(self) -> List[Dict[str, Any]]:
        """Each known person with how many photos they appear in.

        ``photos`` counts distinct photos; ``faces`` is the summed face count,
        so a photo with two photos of the same person counts once in ``photos``
        but twice in ``faces``.
        """
        # COUNT(*) rather than COUNT(DISTINCT image_path): the table's
        # UNIQUE(person_name, image_path) already guarantees one row per pair,
        # so the two agree - and dropping DISTINCT lets SQLite answer from the
        # covering index instead of building a temp b-tree (231ms -> ~15ms at
        # 50k rows on this machine).
        rows = self._conn.execute(
            """
            SELECT person_name,
                   COUNT(*)   AS photos,
                   SUM(faces) AS faces,
                   MAX(seen_at) AS last_seen
              FROM face_occurrences
             GROUP BY person_name COLLATE NOCASE
             ORDER BY MIN(person_name) COLLATE NOCASE
            """
        ).fetchall()
        return [
            {
                "name": str(row["person_name"]),
                "photos": int(row["photos"]),
                "faces": int(row["faces"] or 0),
                "last_seen": str(row["last_seen"] or ""),
            }
            for row in rows
        ]

    def photos_for_person(
        self, person_name: str, limit: Optional[int] = None
    ) -> List[str]:
        """Distinct photos one person appears in."""
        sql = (
            "SELECT DISTINCT image_path FROM face_occurrences "
            "WHERE person_name = ? COLLATE NOCASE ORDER BY image_path"
        )
        params: List[Any] = [str(person_name)]
        if limit is not None and int(limit) > 0:
            sql += " LIMIT ?"
            params.append(int(limit))
        return [str(row["image_path"]) for row in self._conn.execute(sql, params)]

    def people_in_photo(self, image_path: str) -> List[str]:
        """Everyone recorded as appearing in one photo."""
        rows = self._conn.execute(
            """
            SELECT person_name FROM face_occurrences
             WHERE image_path = ?
             GROUP BY person_name COLLATE NOCASE
             ORDER BY MIN(person_name) COLLATE NOCASE
            """,
            (str(image_path),),
        ).fetchall()
        return [str(row["person_name"]) for row in rows]

    def resolve_names(self, names: Sequence[str]) -> Tuple[List[str], List[str]]:
        """Split requested names into (matched, unknown), case-insensitively.

        Names are matched the way a person would expect ("alex" finds "Alex"),
        and anything unmatched comes back as `unknown` so the UI can say which
        name was wrong instead of silently returning fewer photos.
        """
        matched: List[str] = []
        unknown: List[str] = []
        for raw in names or ():
            text = str(raw or "").strip()
            if not text:
                continue
            row = self._conn.execute(
                "SELECT person_name FROM face_occurrences "
                "WHERE person_name = ? COLLATE NOCASE LIMIT 1",
                (text,),
            ).fetchone()
            if row is None:
                row = self._conn.execute(
                    """
                    SELECT person_name FROM face_occurrences
                     WHERE person_name LIKE ? COLLATE NOCASE LIMIT 1
                    """,
                    (f"{text}%",),
                ).fetchone()
            if row is None:
                unknown.append(text)
            else:
                canonical = str(row["person_name"])
                if canonical.lower() not in {m.lower() for m in matched}:
                    matched.append(canonical)
        return matched, unknown

    def get_intersection(self, names: Sequence[str]) -> List[str]:
        """Photos containing **every** one of ``names``.

        A single grouped pass rather than N set intersections in Python:
        ``GROUP BY image_path HAVING COUNT(DISTINCT person_name) = N`` is one
        indexed scan, which is what keeps this fast at 50k+ rows.

        Duplicates in ``names`` are collapsed, so asking twice for one person
        does not require two occurrences in a photo.
        """
        wanted = [str(name).strip() for name in names or () if str(name).strip()]
        if not wanted:
            return []
        unique: List[str] = []
        seen = set()
        for name in wanted:
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(name)

        placeholders = ",".join("?" for _ in unique)
        rows = self._conn.execute(
            f"""
            SELECT image_path
              FROM face_occurrences
             WHERE person_name IN ({placeholders}) COLLATE NOCASE
             GROUP BY image_path
            HAVING COUNT(DISTINCT person_name) = ?
             ORDER BY image_path
            """,
            (*unique, len(unique)),
        ).fetchall()
        return [str(row["image_path"]) for row in rows]

    def get_co_occurrence(self) -> List[Dict[str, Any]]:
        """Pairwise photo counts: how often each two people appear together.

        Returns one entry per *observed* pair (never pairs with a count of
        zero), so the heatmap has a sparse but honest matrix to draw. ``names``
        lists everyone known, including those who appear in no shared photo.
        """
        people = self.get_all_people()
        names = [entry["name"] for entry in people]
        if len(names) < 2:
            return []

        rows = self._conn.execute(
            """
            SELECT a.person_name AS a, b.person_name AS b,
                   COUNT(*) AS together
              FROM face_occurrences AS a
              JOIN face_occurrences AS b
                ON a.image_path = b.image_path
               AND a.person_name < b.person_name COLLATE NOCASE
             GROUP BY a.person_name COLLATE NOCASE, b.person_name COLLATE NOCASE
             ORDER BY together DESC
            """
        ).fetchall()
        # SQLite applies COLLATE NOCASE to `<` only on an explicit operand, and
        # a person recorded as both "Alex" and "alex" must not pair with itself
        # — so canonicalise both sides by their lowercase key here rather than
        # trusting the comparison operator.
        lookup = {name.lower(): name for name in names}
        pairs: List[Dict[str, Any]] = []
        for row in rows:
            left = lookup.get(str(row["a"]).lower())
            right = lookup.get(str(row["b"]).lower())
            if left is None or right is None or left == right:
                continue
            first, second = sorted((left, right), key=str.lower)
            pairs.append(
                {"a": first, "b": second, "photos": int(row["together"])}
            )
        pairs.sort(key=lambda item: (-item["photos"], item["a"], item["b"]))
        return pairs

    # --------------------------------------------------------------- helpers
    def _get_row(self, person_name: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM persons WHERE person_name = ? COLLATE NOCASE",
            (str(person_name),),
        ).fetchone()

    @staticmethod
    def _row_to_person(row: sqlite3.Row) -> Person:
        eye = NamesDB._stored_eye_vector(row)
        return Person(
            person_name=row["person_name"],
            embedding_centroid=np.frombuffer(row["embedding"], dtype=np.float32),
            eye_embedding_centroid=eye,
            sample_image_paths=_decode_samples(row["sample_image_paths"]),
            faces_seen=int(row["faces_seen"]),
            eye_faces_seen=int(row["eye_faces_seen"] or 0),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _merge_samples(existing: Iterable[str], new: Iterable[str]) -> List[str]:
        merged: List[str] = list(existing)
        seen = {path.lower() for path in merged}
        for path in new:
            key = path.lower()
            if key in seen:
                continue
            merged.append(path)
            seen.add(key)
            if len(merged) >= MAX_SAMPLE_PATHS:
                break
        return merged[:MAX_SAMPLE_PATHS]


def _decode_samples(raw: str) -> List[str]:
    try:
        data = json.loads(raw or "[]")
    except (TypeError, ValueError):  # pragma: no cover - corrupt row
        return []
    return [str(item) for item in data] if isinstance(data, list) else []


def _clean_samples(paths: Sequence[Union[str, Path]]) -> List[str]:
    cleaned: List[str] = []
    for path in paths or ():
        text = str(path)
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def open_names_db(config: Optional[Dict] = None) -> Optional[NamesDB]:
    """Open the database named by ``config['names_db']``.

    Returns ``None`` (with a warning) when the database is disabled in the
    config or cannot be opened — auto-labelling is an enhancement, so a
    broken DB must never stop the run.
    """
    config = config or {}
    raw = config.get("names_db", DB_FILENAME)
    if raw is None or str(raw).strip().lower() in ("", "none", "false", "off"):
        logger.info("Name database disabled (names_db is empty).")
        return None
    # DB_FILENAME / "./DB_FILENAME" are resolved lazily by resolve_db_path, so a
    # changed FACEORG_DATA_DIR is honoured even in a long-lived process
    path = resolve_db_path(raw)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        db = NamesDB(str(path))
        logger.info("Name database: %s", path)
        return db
    except (OSError, sqlite3.Error) as exc:
        logger.warning(
            "Name database %r is unavailable (%s); continuing without "
            "auto-labelling.", path, exc,
        )
        return None
