"""Place clustered photos into one folder per person (SPEC §4, F7–F12).

* one folder per name under ``output_folder`` (existing folders are reused,
  SPEC §10)
* a photo that contains several named people is written into *every*
  matching folder (SPEC §4.7) — in ``move`` mode the original is deleted
  only once, and only after all copies succeeded
* unnamed / skipped clusters land in ``_unknown`` (SPEC §4.8)
* name clashes with an existing file get an ``_1``, ``_2``, ... suffix
  (SPEC F12), and names are sanitised so they are valid on every OS
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .clusterer import FaceCluster

logger = logging.getLogger(__name__)

DEFAULT_MODE = "copy"
DEFAULT_UNKNOWN_FOLDER = "_unknown"
MAX_NAME_LENGTH = 64

#: words that mean "do not name this cluster"
SKIP_TOKENS = frozenset({"", "skip", "unknown"})

_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

__all__ = [
    "OrganizeSummary",
    "PhotoOrganizer",
    "normalize_person_name",
    "DEFAULT_UNKNOWN_FOLDER",
]


def normalize_person_name(raw: Optional[str]) -> Optional[str]:
    """Turn user input into a safe folder name.

    ``None`` means "no name" (skip / empty / illegal-only input) and sends
    the photo to the unknown folder.  Characters that are invalid in file
    names become ``_``, trailing dots/spaces are trimmed and Windows
    reserved device names are prefixed, so the same input produces the same
    folder on Windows, macOS and Linux.
    """
    if raw is None:
        return None
    name = str(raw).strip()
    if name.lower() in SKIP_TOKENS:
        return None
    name = _ILLEGAL_CHARS.sub("_", name)
    name = name[:MAX_NAME_LENGTH].rstrip(". ")
    if not name:
        return None
    if name.upper() in _RESERVED_NAMES:
        name = f"_{name}"
    return name


@dataclass
class OrganizeSummary:
    """What one organise run did (feeds the final report, SPEC F10)."""

    mode: str
    output_folder: Path
    folders: Dict[str, int] = field(default_factory=dict)  # folder -> photos
    folders_created: List[Path] = field(default_factory=list)
    files_placed: int = 0
    copies_written: int = 0
    sources_removed: int = 0
    errors: List[Tuple[Path, str]] = field(default_factory=list)

    @property
    def unknown_placed(self) -> int:
        return self.folders.get(DEFAULT_UNKNOWN_FOLDER, 0)

    @property
    def ok(self) -> bool:
        return not self.errors

    def format(self) -> str:
        """Human-readable report (SPEC F10)."""
        lines = [
            f"Output folder : {self.output_folder}",
            f"Mode          : {self.mode}",
            f"Folders       : {len(self.folders)} ({len(self.folders_created)} new)",
        ]
        for name in sorted(self.folders, key=str.lower):
            lines.append(f"  {name:<24} {self.folders[name]:>4} photo(s)")
        lines.append(f"Photos placed : {self.files_placed}")
        lines.append(f"Copies written: {self.copies_written}")
        if self.mode == "move":
            lines.append(f"Originals removed: {self.sources_removed}")
        lines.append(f"Errors        : {len(self.errors)}")
        for path, reason in self.errors:
            lines.append(f"  ! {path}: {reason}")
        return "\n".join(lines)


def _unique_destination(folder: Path, filename: str, taken: set) -> Path:
    """Pick a free ``filename`` inside ``folder`` (existing files count too)."""
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    candidate = folder / filename
    counter = 0
    while candidate.name.lower() in taken or candidate.exists():
        counter += 1
        candidate = folder / f"{stem}_{counter}{suffix}"
        if counter > 9999:  # pathological, but never loop forever
            raise OSError(f"could not find a free name for {filename} in {folder}")
    taken.add(candidate.name.lower())
    return candidate


class PhotoOrganizer:
    """Copies (or moves) photos into ``output_folder/<person>/`` folders."""

    def __init__(
        self,
        output_folder: Union[str, Path],
        mode: str = DEFAULT_MODE,
        unknown_folder: str = DEFAULT_UNKNOWN_FOLDER,
    ) -> None:
        mode = str(mode).lower()
        if mode not in ("copy", "move"):
            raise ValueError(f"mode must be 'copy' or 'move', got {mode!r}")
        unknown = normalize_person_name(unknown_folder) or DEFAULT_UNKNOWN_FOLDER
        self.output_folder = Path(output_folder).expanduser()
        self.mode = mode
        self.unknown_folder = unknown

    @classmethod
    def from_config(cls, config: Dict) -> "PhotoOrganizer":
        """Build an organizer from the (already merged) ``config.yaml`` dict."""
        return cls(
            output_folder=config.get("output_folder", "./grouped_photos"),
            mode=config.get("mode", DEFAULT_MODE),
            unknown_folder=config.get("unknown_folder", DEFAULT_UNKNOWN_FOLDER),
        )

    def organize(
        self,
        named_clusters: Sequence[Tuple[Optional[str], FaceCluster]],
        progress_cb: Optional[Callable[[int, int, Path], None]] = None,
    ) -> OrganizeSummary:
        """Place every photo of every cluster.

        ``named_clusters`` is a sequence of ``(name_or_None, cluster)``
        tuples, where ``name`` is whatever the user typed (raw input is
        normalised here).

        ``progress_cb`` is optional and called as ``cb(done, total, source)``
        after each source photo has been placed — the desktop UI drives its
        progress bar with it.  A failing callback never aborts the run.
        """
        assignments: List[Tuple[Optional[str], Path]] = []
        for name, cluster in named_clusters:
            person = normalize_person_name(name)
            for path in cluster.image_paths:
                assignments.append((person, path))
        return self.organize_files(assignments, progress_cb=progress_cb)

    def organize_files(
        self,
        assignments: Iterable[Tuple[Optional[str], Union[str, Path]]],
        progress_cb: Optional[Callable[[int, int, Path], None]] = None,
    ) -> OrganizeSummary:
        """Place ``(name_or_None, photo_path)`` pairs.

        Grouping happens per *source photo*: a photo with two named people
        is copied into both folders, and in move mode its original is
        removed exactly once — and only if every copy succeeded.
        """
        summary = OrganizeSummary(mode=self.mode, output_folder=self.output_folder)
        # per-folder set of names already chosen during this run, so two
        # different folders can both receive "photo.jpg"
        taken_by_dir: Dict[Path, set] = {}

        by_source: Dict[Path, List[str]] = {}
        for name, path in assignments:
            folder_name = normalize_person_name(name) or self.unknown_folder
            by_source.setdefault(Path(path), []).append(folder_name)

        if not by_source:
            logger.info("No photos to place.")
            return summary

        placed = 0
        total_sources = len(by_source)
        for source, wanted in by_source.items():
            folder_names = list(dict.fromkeys(wanted))  # one copy per folder
            if not source.is_file():
                reason = "source missing or not a file"
                logger.warning("Skipping %s: %s", source, reason)
                summary.errors.append((source, reason))
                continue

            written = 0
            for folder_name in folder_names:
                destination_dir = self.output_folder / folder_name
                existed = destination_dir.exists()
                try:
                    destination_dir.mkdir(parents=True, exist_ok=True)
                    taken = taken_by_dir.setdefault(destination_dir, set())
                    destination = _unique_destination(
                        destination_dir, source.name, taken
                    )
                    shutil.copy2(source, destination)
                except OSError as exc:
                    logger.warning("Could not place %s into %s: %s",
                                   source, destination_dir, exc)
                    summary.errors.append((destination_dir, str(exc)))
                    continue

                if not existed:
                    summary.folders_created.append(destination_dir)
                summary.folders[folder_name] = summary.folders.get(folder_name, 0) + 1
                summary.copies_written += 1
                written += 1

            if written == 0:
                continue
            summary.files_placed += 1
            placed += 1

            if progress_cb is not None:
                # progress is a UI nicety: never let it break the run
                try:
                    progress_cb(placed, total_sources, source)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("organize progress callback failed: %s", exc)

            if self.mode == "move":
                if written != len(folder_names):
                    logger.warning(
                        "Kept original %s: only %d of %d copies succeeded.",
                        source, written, len(folder_names),
                    )
                    summary.errors.append(
                        (source, "original kept because some copies failed")
                    )
                    continue
                try:
                    source.unlink()
                    summary.sources_removed += 1
                except OSError as exc:
                    logger.warning("Could not remove %s: %s", source, exc)
                    summary.errors.append((source, str(exc)))

        logger.info(
            "Placed %d photo(s) into %d folder(s) (%s mode).",
            summary.files_placed, len(summary.folders), summary.mode,
        )
        return summary
