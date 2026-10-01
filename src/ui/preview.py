"""Thumbnail montage per cluster (M2, SPEC F5 / §11).

Renders a grid of cropped faces for one cluster, saves it as a temporary
JPEG, and (optionally) opens it in the default OS image viewer so the user
can verify a group *before* naming it.

Everything here is local: crops come straight from the photos already on
disk, and the montage files live in a per-run temp folder that
:class:`PreviewBoard` deletes again when the run ends.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

from PIL import Image, ImageDraw, ImageFont

from ..image_loader import load_image

logger = logging.getLogger(__name__)

__all__ = [
    "CELL_SIZE",
    "MAX_FACES",
    "PreviewBoard",
    "PreviewResult",
    "build_montage",
    "face_crop",
    "open_in_viewer",
    "save_montage",
]

CELL_SIZE = 128          # edge length of one face tile, in pixels
MAX_FACES = 36           # faces drawn before the montage says "+N more"
GAP = 4                  # space between tiles
MARGIN = 8               # border around the whole grid
CAPTION_HEIGHT = 26      # header strip with the cluster info
INDEX_BOX = 18           # size of the numbered badge in each tile
CROP_MARGIN = 0.12       # bbox expansion, as a fraction of the face size

BACKGROUND = (244, 244, 244)
TILE_BACKGROUND = (255, 255, 255)
PLACEHOLDER = (198, 198, 198)
CAPTION_BG = (40, 40, 40)
CAPTION_FG = (255, 255, 255)
BADGE_BG = (0, 0, 0)
BADGE_FG = (255, 220, 60)

PathLike = Union[str, Path]


@dataclass
class PreviewResult:
    """Where a montage was written and whether the viewer took it."""

    path: Path
    opened: bool


def _crop_box(bbox, width: int, height: int):
    """Expand a face box by :data:`CROP_MARGIN` and clamp it to the image."""
    x1, y1, x2, y2 = (float(value) for value in bbox)
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    dx = box_width * CROP_MARGIN
    dy = box_height * CROP_MARGIN
    left = max(0, int(math.floor(x1 - dx)))
    top = max(0, int(math.floor(y1 - dy)))
    right = min(width - 1, int(math.ceil(x2 + dx)))
    bottom = min(height - 1, int(math.ceil(y2 + dy)))
    if right <= left or bottom <= top:  # degenerate box -> single pixel region
        left, top, right, bottom = 0, 0, max(1, width), max(1, height)
    return left, top, right, bottom


def face_crop(image_bgr, bbox, cell_size: int = CELL_SIZE) -> Image.Image:
    """Crop one face out of a BGR ``uint8`` array and fit it to a tile."""
    import cv2

    height, width = image_bgr.shape[:2]
    left, top, right, bottom = _crop_box(bbox, width, height)
    crop = image_bgr[top:bottom, left:right]
    if crop.size == 0:  # pragma: no cover - clamped boxes prevent this
        raise ValueError("empty crop")
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image.thumbnail((cell_size, cell_size), Image.LANCZOS)
    return image


def _fit_tile(face: Image.Image, cell_size: int) -> Image.Image:
    """Paste a face onto a square white tile of ``cell_size``."""
    tile = Image.new("RGB", (cell_size, cell_size), TILE_BACKGROUND)
    tile.paste(face, ((cell_size - face.width) // 2,
                      (cell_size - face.height) // 2))
    return tile


def _placeholder_tile(cell_size: int, label: str) -> Image.Image:
    tile = Image.new("RGB", (cell_size, cell_size), PLACEHOLDER)
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), label, font=font)
    draw.text(
        ((cell_size - (box[2] - box[0])) // 2,
         (cell_size - (box[3] - box[1])) // 2),
        label, fill=(90, 90, 90), font=font,
    )
    return tile


def _grid_size(count: int) -> int:
    """Columns for a compact grid (1 -> 1, 2-4 -> 2, 5-9 -> 3, capped at 5)."""
    return max(1, min(5, math.ceil(math.sqrt(max(1, count)))))


def build_montage(
    cluster,
    cell_size: int = CELL_SIZE,
    max_faces: int = MAX_FACES,
    note: Optional[str] = None,
) -> Image.Image:
    """Render ``cluster``'s faces as a numbered grid of thumbnails.

    Each tile carries the 1-based face index used by the CLI's ``split``
    command, so the picture and the text list stay in sync.  Faces whose
    photo cannot be read become grey placeholders instead of raising.
    """
    faces = list(cluster.faces)[:max_faces]
    hidden = len(cluster.faces) - len(faces)
    columns = _grid_size(len(faces))
    rows = max(1, math.ceil(len(faces) / columns)) if faces else 1

    width = MARGIN * 2 + columns * cell_size + (columns - 1) * GAP
    height = (MARGIN * 2 + CAPTION_HEIGHT
              + rows * cell_size + (rows - 1) * GAP)
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    photo_count = len({face.image_path for face in cluster.faces})
    caption = (
        f"Cluster #{cluster.cluster_id} - "
        f"{len(cluster.faces)} face(s) / {photo_count} photo(s)"
    )
    if hidden:
        caption += f"  (+{hidden} more)"
    if note:
        caption += f"  - {note}"
    draw.rectangle([0, 0, width, CAPTION_HEIGHT], fill=CAPTION_BG)
    draw.text((MARGIN, 8), caption, fill=CAPTION_FG, font=font)

    for position, face in enumerate(faces):
        row, column = divmod(position, columns)
        left = MARGIN + column * (cell_size + GAP)
        top = MARGIN * 2 + CAPTION_HEIGHT + row * (cell_size + GAP)

        try:
            image = load_image(face.image_path)
            if image is None:
                raise ValueError("unreadable image")
            tile = _fit_tile(face_crop(image, face.bbox, cell_size), cell_size)
        except Exception as exc:  # never fail a run over a preview
            logger.debug("Preview tile for %s failed: %s", face.image_path, exc)
            tile = _placeholder_tile(cell_size, "n/a")

        canvas.paste(tile, (left, top))

        badge = str(position + 1)
        box = draw.textbbox((0, 0), badge, font=font)
        badge_width = (box[2] - box[0]) + 6
        badge_height = (box[3] - box[1]) + 4
        draw.rectangle([left, top, left + badge_width, top + badge_height],
                       fill=BADGE_BG)
        draw.text((left + 3, top + 1), badge, fill=BADGE_FG, font=font)

    return canvas


def save_montage(
    cluster,
    directory: PathLike,
    filename: Optional[str] = None,
    cell_size: int = CELL_SIZE,
    max_faces: int = MAX_FACES,
    note: Optional[str] = None,
    quality: int = 85,
) -> Path:
    """Build the montage and write it as a JPEG inside ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = (f"cluster_{cluster.cluster_id}_"
                    f"{uuid.uuid4().hex[:8]}.jpg")
    path = directory / filename
    image = build_montage(cluster, cell_size=cell_size,
                          max_faces=max_faces, note=note)
    image.save(path, "JPEG", quality=quality)
    return path


def open_in_viewer(path: PathLike, launcher: Optional[Callable] = None) -> bool:
    """Open ``path`` in the OS default image viewer.

    ``launcher`` is injectable for tests; by default it picks
    ``os.startfile`` on Windows and ``open`` / ``xdg-open`` elsewhere.
    Never raises — a headless machine simply returns ``False``.
    """
    path = Path(path)
    try:
        if launcher is not None:
            launcher(path)
            return True
        if os.name == "nt":  # pragma: no cover - Windows only
            os.startfile(str(path))  # noqa: S606
        elif sys.platform == "darwin":  # pragma: no cover - macOS only
            subprocess.Popen(["open", str(path)],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        else:  # pragma: no cover - POSIX desktop only
            subprocess.Popen(["xdg-open", str(path)],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        return True
    except Exception as exc:  # headless / no handler installed
        logger.warning("Could not open %s in an image viewer: %s", path, exc)
        return False


class PreviewBoard:
    """Owns the temp folder of montages for one run (and cleans it up)."""

    def __init__(
        self,
        enabled: bool = True,
        open_viewer: bool = True,
        directory: Optional[PathLike] = None,
        cell_size: int = CELL_SIZE,
        max_faces: int = MAX_FACES,
        opener: Optional[Callable] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.open_viewer = bool(open_viewer)
        self.cell_size = cell_size
        self.max_faces = max_faces
        #: injectable viewer launcher (defaults to the OS image viewer)
        self.opener = opener or open_in_viewer
        self._directory = Path(directory) if directory is not None else None
        self._owns_directory = directory is None
        self._counter = 0

    # ------------------------------------------------------------- rendering
    @property
    def directory(self) -> Path:
        """Lazily create the run's temp folder."""
        if self._directory is None:
            self._directory = Path(tempfile.mkdtemp(prefix="faceorg_preview_"))
        return self._directory

    def show(self, cluster, note: Optional[str] = None,
             open_viewer: Optional[bool] = None) -> Optional[PreviewResult]:
        """Save a montage for ``cluster``; ``None`` when previews are off."""
        if not self.enabled:
            return None
        self._counter += 1
        try:
            path = save_montage(
                cluster,
                self.directory,
                filename=f"cluster_{cluster.cluster_id}_{self._counter}.jpg",
                cell_size=self.cell_size,
                max_faces=self.max_faces,
                note=note,
            )
        except Exception as exc:  # a preview must never abort the run
            logger.warning("Could not build preview for cluster #%s: %s",
                           cluster.cluster_id, exc)
            return None

        should_open = self.open_viewer if open_viewer is None else open_viewer
        opened = bool(self.opener(path)) if should_open else False
        logger.debug("Preview for cluster #%s: %s (opened=%s)",
                     cluster.cluster_id, path, opened)
        return PreviewResult(path=path, opened=opened)

    # ------------------------------------------------------------ lifecycle
    def cleanup(self) -> None:
        """Delete the temp folder (best effort — viewers may hold files)."""
        if self._directory is None or not self._owns_directory:
            return
        try:
            shutil.rmtree(self._directory, ignore_errors=False)
        except OSError as exc:
            logger.debug("Keeping preview folder %s: %s", self._directory, exc)
        finally:
            self._directory = None

    def __enter__(self) -> "PreviewBoard":
        return self

    def __exit__(self, *exc_info) -> None:
        self.cleanup()
