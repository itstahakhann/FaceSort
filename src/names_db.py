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
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

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
    """One remembered person (SPEC §8 ``Person``)."""

    person_name: str
    embedding_centroid: np.ndarray
    sample_image_paths: List[str] = field(default_factory=list)
    faces_seen: int = 1
    created_at: str = ""
    updated_at: str = ""

    @property
    def dims(self) -> int:
        return int(self.embedding_centroid.size)


@dataclass
class PersonMatch:
    """The known person a cluster centroid was matched against."""

    person_name: str
    distance: float
    sample_image_paths: List[str] = field(default_factory=list)


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
        self._conn.commit()

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
    ) -> Person:
        """Remember (or refresh) a person.

        Re-naming an existing person blends the new centroid into the old
        one weighted by how many faces each side contributed, so the stored
        centroid slowly converges on the person's true average.  Sample
        paths are merged and capped.
        """
        name = str(person_name or "").strip()
        if not name:
            raise ValueError("person_name must not be empty")
        centroid = _unit(embedding_centroid)
        faces_seen = max(1, int(faces_seen))

        samples = self._merge_samples([], _clean_samples(sample_image_paths))
        row = self._get_row(name)
        if row is None:
            now = _now()
            self._conn.execute(
                """
                INSERT INTO persons (person_name, embedding, dims,
                                     sample_image_paths, faces_seen,
                                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    centroid.tobytes(),
                    centroid.size,
                    json.dumps(samples),
                    faces_seen,
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
            self._conn.execute(
                """
                UPDATE persons
                   SET person_name = ?, embedding = ?, dims = ?,
                       sample_image_paths = ?, faces_seen = ?, updated_at = ?
                 WHERE id = ?
                """,
                (
                    name,          # keep the spelling the user just typed
                    blended.tobytes(),
                    blended.size,
                    json.dumps(samples),
                    previous_faces + faces_seen,
                    _now(),
                    row["id"],
                ),
            )
        self._conn.commit()

        person = self.get_person(name)
        if person is None:  # pragma: no cover - concurrency is not a thing here
            raise sqlite3.OperationalError("person disappeared after upsert")
        logger.debug(
            "Remembered %r (%d face(s), %d sample photo(s)).",
            name, person.faces_seen, len(person.sample_image_paths),
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
        self, embedding_centroid, tolerance: float = DEFAULT_TOLERANCE
    ) -> Optional[PersonMatch]:
        """Closest known person whose centroid is within ``tolerance``.

        Distances are cosine distances — the same metric the clusterer
        uses — so ``tolerance`` means exactly what it means for
        clustering (SPEC F11 / F4).
        """
        if embedding_centroid is None:
            return None
        try:
            centroid = _unit(embedding_centroid)
        except ValueError as exc:
            logger.warning("Cannot match against the name DB: %s", exc)
            return None

        matches: List[PersonMatch] = []
        for person in self.list_persons():
            if person.dims != centroid.size:
                logger.debug(
                    "Skipping %r: embedding size %d != %d",
                    person.person_name, person.dims, centroid.size,
                )
                continue
            distance = cosine_distance(centroid, person.embedding_centroid)
            if distance <= tolerance:
                matches.append(
                    PersonMatch(
                        person_name=person.person_name,
                        distance=distance,
                        sample_image_paths=list(person.sample_image_paths),
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

    # --------------------------------------------------------------- helpers
    def _get_row(self, person_name: str) -> Optional[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM persons WHERE person_name = ? COLLATE NOCASE",
            (str(person_name),),
        ).fetchone()

    @staticmethod
    def _row_to_person(row: sqlite3.Row) -> Person:
        return Person(
            person_name=row["person_name"],
            embedding_centroid=np.frombuffer(row["embedding"], dtype=np.float32),
            sample_image_paths=_decode_samples(row["sample_image_paths"]),
            faces_seen=int(row["faces_seen"]),
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
