"""Local HTML gallery export — a self-contained, offline photo website.

Produces a folder that opens in any browser with no server, no network and no
build step: one ``index.html`` plus local CSS/JS/images. Everything is inlined
or copied next to the HTML, so the folder can be zipped, put on a USB stick or
opened on a phone years later and still work.

Why a generator rather than a template the user runs themselves: the photo
paths live in the name database as absolute paths from whatever machine scanned
them, so the copy has to happen where those files exist.

Optional password protection is **client-side only** — see
:func:`password_digest` for what that does and does not buy. It is a privacy
curtain against someone who opens the file casually, not a security control.

Layout produced::

    <output>/
      index.html
      assets/css/gallery.css
      assets/js/gallery.js
      assets/images/<person>/photo.jpg
      assets/thumbnails/<person>/photo.jpg
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "GalleryOptions",
    "GalleryResult",
    "GalleryError",
    "THUMB_EDGE",
    "THUMB_QUALITY",
    "build_gallery",
    "folder_name_for",
    "password_digest",
    "list_template_dirs",
]

#: Thumbnails are capped on their long edge, not their width: a portrait photo
#: scaled to 300px *wide* would end up 400px tall and slow the grid down.
THUMB_EDGE = 300
THUMB_QUALITY = 80

#: Characters Windows refuses in a folder name, plus the separators we treat as
#: path characters regardless of platform.
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

#: Reserved device names on Windows: a folder called CON breaks Explorer even
#: though it is a legal name on Linux.
_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

#: Extensions we will not put in a gallery, even if the occurrence table names
#: them. Browsers do not render these reliably and they are not photographs.
_SKIP_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".json", ".html", ".htm", ".xml"}


class GalleryError(RuntimeError):
    """The gallery could not be built."""


def folder_name_for(name: str, taken: Optional[set] = None) -> str:
    """A safe, unique folder name for a person.

    Handles the awkward real-world cases rather than assuming names are clean:
    slashes and quotes, Windows' reserved device names, names that are only
    dots (which resolve to the parent directory), and two people whose names
    sanitise to the same thing.
    """
    cleaned = _UNSAFE.sub("_", str(name or "").strip())
    # A trailing dot or space is silently dropped by Windows, so "John." and
    # "John" would collide on disk.
    cleaned = cleaned.rstrip(". ")
    if not cleaned or set(cleaned) <= {".", " "}:
        cleaned = "unnamed"
    if cleaned.upper() in _RESERVED:
        cleaned = f"{cleaned}_"
    # Keep room for " (1)" and for a 255-character path limit.
    cleaned = cleaned[:80]

    if taken is None:
        return cleaned
    if cleaned not in taken:
        return cleaned
    stem = cleaned[:76]
    index = 2
    while f"{stem} ({index})" in taken:
        index += 1
    return f"{stem} ({index})"


def password_digest(password: str) -> str:
    """SHA-256 of a password, hex — the value embedded in the page.

    .. warning::
       This is **not** encryption. The digest ships inside the HTML, so anyone
       who reads the file can recover it, and an offline dictionary attack on a
       short password is trivial. It stops a family member or a guest from
       casually browsing; it does not protect the photos from anyone who means
       it. Anything stronger needs a real server doing the checking.
    """
    return hashlib.sha256(str(password or "").encode("utf-8")).hexdigest()


@dataclass
class GalleryOptions:
    """What to include and where to put it."""

    output_path: str
    selected_people: Sequence[str] = ()
    include_thumbnails: bool = True
    password: str = ""
    title: str = "FaceSort Gallery"
    #: Overwrite an existing gallery folder. The caller is expected to have
    #: confirmed this with the user.
    overwrite: bool = False
    #: Largest photo edge kept in the gallery. Full-resolution phone photos are
    #: 4000px+ and make a gallery impossible to open; capping keeps it usable.
    max_image_edge: int = 2400
    image_quality: int = 88


@dataclass
class GalleryResult:
    """What was built."""

    output_path: str
    total_photos: int = 0
    people: List[str] = field(default_factory=list)
    thumbnails: int = 0
    bytes_written: int = 0
    skipped: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    took_seconds: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "output_path": self.output_path,
            "total_photos": self.total_photos,
            "people": list(self.people),
            "thumbnails": self.thumbnails,
            "bytes_written": self.bytes_written,
            "skipped": len(self.skipped),
            "missing": len(self.missing),
            "skipped_names": self.skipped[:20],
            "missing_names": self.missing[:20],
            "took_seconds": round(self.took_seconds, 2),
        }


def list_template_dirs() -> List[Path]:
    """Candidate directories holding ``gallery.html``, most specific first.

    ``_MEIPASS`` comes first because the frozen engine reads the template out
    of the bundle rather than beside the executable.
    """
    candidates: List[Path] = []
    meipass = getattr(__import__("sys"), "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "templates")
    here = Path(__file__).resolve().parent
    candidates.append(here / "templates")          # running from source
    candidates.append(here.parent / "src" / "templates")
    return candidates


def _find_template() -> Path:
    for directory in list_template_dirs():
        candidate = directory / "gallery.html"
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(d) for d in list_template_dirs())
    raise GalleryError(
        f"The gallery template (gallery.html) was not found. Looked in: {searched}"
    )


def _environment() -> "object":
    """A Jinja environment with autoescaping on — names reach the page."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    template_path = _find_template().parent
    return Environment(
        loader=FileSystemLoader(str(template_path)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _write_thumbnail(source: Path, target: Path) -> bool:
    """Write a JPEG thumbnail. Returns False for anything unreadable.

    A corrupt or truncated original is skipped rather than aborting the export:
    losing one photo out of ten thousand is a much better outcome than losing
    the gallery.
    """
    try:
        from PIL import Image, ImageOps
    except ImportError:  # pragma: no cover - Pillow is a hard dependency
        logger.error("Pillow is not available; cannot make thumbnails")
        return False

    try:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image) or image
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            elif image.mode == "L":
                image = image.convert("RGB")
            image.thumbnail((THUMB_EDGE, THUMB_EDGE))
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target, format="JPEG", quality=THUMB_QUALITY,
                       optimize=True, progressive=True)
        return True
    except Exception as exc:  # noqa: BLE001 - one bad photo is not fatal
        logger.warning("Skipping thumbnail for %s: %s", source.name, exc)
        return False


def _write_image(source: Path, target: Path, max_edge: int, quality: int) -> int:
    """Copy a photo into the gallery, downscaling if it is enormous.

    Returns the bytes written, or 0 when the file could not be processed.
    """
    try:
        from PIL import Image, ImageOps
    except ImportError:  # pragma: no cover
        shutil.copy2(source, target)
        return target.stat().st_size

    try:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image) or image
            if image.mode != "RGB":
                image = image.convert("RGB")
            largest = max(image.size)
            if largest > max_edge:
                image.thumbnail((max_edge, max_edge))
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target, format="JPEG", quality=quality,
                       optimize=True, progressive=True)
            return target.stat().st_size
    except Exception as exc:  # noqa: BLE001 - one bad photo is not fatal
        logger.warning("Skipping %s: %s", source.name, exc)
        return 0


def _resolve_source(path_text: str, scan_root: Path) -> Optional[Path]:
    """Turn a recorded path into a real file, tolerating relative ones."""
    candidate = Path(path_text)
    if not candidate.is_absolute():
        candidate = scan_root / candidate
    try:
        return candidate if candidate.is_file() else None
    except OSError:
        return None


def build_gallery(
    options: GalleryOptions,
    photos_by_person: Dict[str, List[str]],
    progress: Optional[Callable[[int, int, str], None]] = None,
    scan_root: Optional[Path] = None,
) -> GalleryResult:
    """Build the gallery from ``{person: [photo paths]}``.

    ``photos_by_person`` is the selection, already grouped. ``progress`` is
    called as ``cb(done, total, current_name)`` so a long export can report
    movement; it is optional and a failing callback never aborts the build.
    """
    import time

    started = time.perf_counter()
    output = Path(options.output_path).expanduser()
    scan_root = scan_root or Path(".")
    result = GalleryResult(output_path=str(output))

    if not photos_by_person:
        raise GalleryError("There are no photos to put in the gallery.")

    if output.exists() and any(output.iterdir()):
        if not options.overwrite:
            raise GalleryError(
                f"{output} already exists and is not empty. "
                "Confirm to replace it."
            )
        logger.info("Replacing the existing gallery at %s", output)
        shutil.rmtree(output, ignore_errors=True)

    total = sum(len(paths) for paths in photos_by_person.values())
    done = 0

    # Reserve the folder names first so two people who sanitise to the same
    # string cannot overwrite each other's folders mid-build.
    taken: set = set()
    folders: Dict[str, str] = {}
    for name in photos_by_person:
        folder = folder_name_for(name, taken)
        taken.add(folder)
        folders[name] = folder

    entries: List[Dict[str, object]] = []
    for name, paths in photos_by_person.items():
        folder = folders[name]
        items: List[Dict[str, str]] = []
        for path_text in paths:
            done += 1
            if progress is not None:
                try:
                    progress(done, total, name)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("progress callback failed: %s", exc)

            source = _resolve_source(path_text, scan_root)
            if source is None:
                result.missing.append(path_text)
                continue
            if source.suffix.lower() in _SKIP_SUFFIXES:
                result.skipped.append(source.name)
                continue

            target_name = f"{source.stem}.jpg"
            target = output / "assets" / "images" / folder / target_name
            written = _write_image(
                source, target, options.max_image_edge, options.image_quality
            )
            if not written:
                result.skipped.append(source.name)
                continue
            result.bytes_written += written
            result.total_photos += 1

            record = {
                "src": f"assets/images/{folder}/{target_name}",
                "name": source.name,
            }
            if options.include_thumbnails:
                thumb = output / "assets" / "thumbnails" / folder / target_name
                if _write_thumbnail(source, thumb):
                    record["thumb"] = f"assets/thumbnails/{folder}/{target_name}"
                    result.thumbnails += 1
            items.append(record)

        if items:
            entries.append({
                "name": name,
                "folder": folder,
                "count": len(items),
                "photos": items,
                "cover": items[0].get("thumb") or items[0]["src"],
            })
            result.people.append(name)

    if not entries:
        raise GalleryError(
            "None of the photos could be read, so there is nothing to show. "
            "They may have been moved or deleted since they were scanned."
        )

    # CSS and JS ship next to the HTML rather than being inlined: both are
    # sizeable, and a separate file keeps the markup readable. Both are still
    # fully local, so the folder still works with no network.
    css_dir = output / "assets" / "css"
    js_dir = output / "assets" / "js"
    css_dir.mkdir(parents=True, exist_ok=True)
    js_dir.mkdir(parents=True, exist_ok=True)

    template_dir = _find_template().parent
    for asset in ("gallery.css", "gallery.js"):
        source = template_dir / asset
        if not source.is_file():
            raise GalleryError(
                f"The gallery asset {asset} is missing from {template_dir}. "
                "The generator ships these alongside gallery.html."
            )
        shutil.copy2(source, (css_dir if asset.endswith(".css") else js_dir) / asset)

    environment = _environment()
    template = environment.get_template("gallery.html")
    page = template.render(
        people=entries,
        title=html.escape(options.title),
        subtitle=f"{result.total_photos} photo(s) of {len(entries)} people",
        generated=time.strftime("%Y-%m-%d %H:%M"),
        password_hash=password_digest(options.password) if options.password else "",
        has_password=bool(options.password),
        total_photos=result.total_photos,
        total_bytes=result.bytes_written,
    )
    (output / "index.html").write_text(page, encoding="utf-8")

    result.took_seconds = time.perf_counter() - started
    logger.info(
        "Gallery written to %s: %d photo(s), %d people, %d thumbnail(s), "
        "%.1f MB in %.1fs.",
        output, result.total_photos, len(result.people), result.thumbnails,
        result.bytes_written / 1_048_576, result.took_seconds,
    )
    return result


def photos_for_selection(
    db,
    mode: str,
    selected: Sequence[str],
) -> Dict[str, List[str]]:
    """Gather ``{person: [paths]}`` for a mode and selection.

    ``mode="all"`` takes everyone with recorded photos; ``mode="people"`` takes
    the named ones and reports which were not found rather than silently
    exporting fewer people than the user asked for.
    """
    if mode == "people":
        matched, unknown = db.resolve_names(selected)
        if unknown:
            raise GalleryError(
                "Not in the name database: " + ", ".join(unknown)
            )
        return {name: db.photos_for_person(name) for name in matched}

    people = db.get_all_people()
    if not people:
        raise GalleryError("Nobody has been named yet.")
    return {
        entry["name"]: db.photos_for_person(entry["name"])
        for entry in people
    }