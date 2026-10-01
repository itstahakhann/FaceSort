"""Scan a folder for images and hand the pipeline only readable files.

Corrupt or unreadable files are logged and skipped — they never abort the
scan (SPEC §6 "Robustness").  HEIC/RAW are intentionally out of scope for
now (SPEC §F1 marks HEIC as optional).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Union

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "is_supported",
    "is_readable",
    "scan_images",
    "load_image",
]

PathLike = Union[str, Path]


def is_supported(path: PathLike) -> bool:
    """True when the file extension is one we can process."""
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def is_readable(path: PathLike) -> bool:
    """True when the file can be opened and verified as an image.

    Logs and returns ``False`` for corrupt/truncated/permission-denied files
    instead of raising.
    """
    path = Path(path)
    try:
        with Image.open(path) as image:
            image.verify()
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        logger.warning("Skipping unreadable image %s: %s", path, exc)
        return False
    except Exception as exc:  # never let an odd file kill the scan
        logger.warning("Skipping image %s: unexpected error: %s", path, exc)
        return False
    return True


def _iter_files(root: Path, recursive: bool) -> Iterator[Path]:
    """Yield files below ``root``, tolerating unreadable sub-directories."""
    walker = root.rglob("*") if recursive else root.iterdir()
    while True:
        try:
            entry = next(walker)
        except StopIteration:
            return
        except OSError as exc:
            logger.warning("Cannot read some entries under %s: %s", root, exc)
            return
        try:
            if entry.is_file():
                yield entry
        except OSError as exc:
            logger.warning("Cannot stat %s: %s", entry, exc)


def scan_images(
    input_folder: PathLike, recursive: bool = True, validate: bool = True
) -> Iterator[Path]:
    """Yield the paths of the images in ``input_folder`` (sorted, stable).

    Files with unsupported extensions are ignored quietly; corrupt files are
    logged and skipped when ``validate`` is true.  Raises only when the
    input folder itself does not exist or is not a directory.
    """
    root = Path(input_folder).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Input folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")

    candidates: List[Path] = [
        path for path in _iter_files(root, recursive) if is_supported(path)
    ]
    candidates.sort(key=lambda path: str(path).lower())

    if not candidates:
        logger.warning(
            "No images with extension %s found in %s",
            ", ".join(sorted(SUPPORTED_EXTENSIONS)),
            root,
        )

    for path in candidates:
        if validate and not is_readable(path):
            continue
        yield path


def load_image(path: PathLike) -> Optional[np.ndarray]:
    """Decode an image into a BGR ``uint8`` array, or ``None`` when unreadable."""
    path = Path(path)
    try:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        logger.warning("Cannot decode %s: %s", path, exc)
        return None
    if image is None:
        logger.warning("Cannot decode %s (corrupt or unsupported file)", path)
        return None
    return image
