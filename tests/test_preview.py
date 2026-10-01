"""Unit tests for the M2 thumbnail montage (SPEC F5, §11).

Run with::

    python tests/test_preview.py
    pytest tests/

Builds small throwaway photos, so no model weights are required.
"""

import math
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from src.clusterer import FaceCluster, FaceRecord
from src.ui.preview import (
    CAPTION_HEIGHT,
    CELL_SIZE,
    GAP,
    MARGIN,
    MAX_FACES,
    PreviewBoard,
    build_montage,
    face_crop,
    open_in_viewer,
    save_montage,
)

_failures = []


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}: {label} {detail}")
    if not condition:
        _failures.append(label)


def make_workspace():
    root = Path(tempfile.mkdtemp(prefix="faceorg_preview_"))
    (root / "photos").mkdir()
    (root / "out").mkdir()
    return root


def add_photo(root, name, size=(400, 300), color=(180, 90, 40)):
    path = root / "photos" / name
    Image.new("RGB", size, color).save(path)
    return path


def face(path, box=(10, 10, 80, 90)):
    return FaceRecord(image_path=path, bbox=box,
                      embedding=np.ones(4, dtype=np.float32))


def expected_size(columns, rows, cell=CELL_SIZE):
    width = MARGIN * 2 + columns * cell + (columns - 1) * GAP
    height = MARGIN * 2 + CAPTION_HEIGHT + rows * cell + (rows - 1) * GAP
    return width, height


def test_build_montage_grid():
    root = make_workspace()
    try:
        paths = [add_photo(root, f"p{i}.jpg") for i in range(4)]
        cluster = FaceCluster(cluster_id=0,
                              faces=[face(path) for path in paths])
        image = build_montage(cluster)

        check("montage is an RGB image",
              image.mode == "RGB", image.mode)
        check("4 faces -> 2x2 grid",
              image.size == expected_size(2, 2), str(image.size))
        check("montage is bigger than one cell", image.width > CELL_SIZE)

        # a single face makes a 1x1 montage
        one = FaceCluster(cluster_id=1, faces=[face(paths[0])])
        check("1 face -> 1x1 grid",
              build_montage(one).size == expected_size(1, 1))

        # 9 faces -> 3x3
        nine = FaceCluster(
            cluster_id=2,
            faces=[face(paths[i % 4]) for i in range(9)],
        )
        check("9 faces -> 3x3 grid",
              build_montage(nine).size == expected_size(3, 3),
              str(build_montage(nine).size))

        # empty cluster must not crash
        empty = FaceCluster(cluster_id=3, faces=[])
        check("empty cluster still renders",
              build_montage(empty).size[0] > 0)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_truncation_and_placeholders():
    root = make_workspace()
    try:
        good = add_photo(root, "good.jpg")
        missing = root / "photos" / "gone.jpg"  # never created
        broken = root / "photos" / "broken.jpg"
        broken.write_bytes(b"this is not an image")

        faces = [face(missing), face(broken), face(good)]
        cluster = FaceCluster(cluster_id=0, faces=faces)
        image = build_montage(cluster)
        check("missing/broken photos become placeholders, not errors",
              image.size == expected_size(2, 2), str(image.size))

        many = FaceCluster(
            cluster_id=1,
            faces=[face(good) for _ in range(MAX_FACES + 3)],
        )
        image = build_montage(many)
        columns = 5
        rows = math.ceil(MAX_FACES / columns)
        check("very large clusters are capped at max_faces",
              image.size == expected_size(columns, rows), str(image.size))

        small = build_montage(many, max_faces=4, cell_size=64)
        check("custom max_faces and cell_size are honoured",
              small.size == expected_size(2, 2, cell=64), str(small.size))
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_face_crop_clamps_to_image():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    inside = face_crop(image, (10, 10, 50, 60), cell_size=32)
    check("ordinary crop is smaller than the cell",
          inside.width <= 32 and inside.height <= 32, str(inside.size))

    clamped = face_crop(image, (-50, -50, 400, 400), cell_size=32)
    check("box outside the image is clamped to the frame",
          0 < clamped.width <= 32 and 0 < clamped.height <= 32,
          str(clamped.size))

    degenerate = face_crop(image, (60, 60, 60, 60), cell_size=32)
    check("zero-sized box still produces a tile",
          degenerate.width > 0 and degenerate.height > 0,
          str(degenerate.size))


def test_save_montage(tmp_path=None):
    root = make_workspace()
    try:
        path = add_photo(root, "a.jpg")
        cluster = FaceCluster(cluster_id=7, faces=[face(path)])
        out = save_montage(cluster, root / "out", note="test note")

        check("montage written as a file", out.is_file(), str(out))
        with Image.open(out) as saved:
            check("file is a real JPEG",
                  saved.format == "JPEG", str(saved.format))
            check("saved size matches the rendered image",
                  saved.size == build_montage(cluster).size, str(saved.size))
        check("name carries the cluster id", "cluster_7" in out.name, out.name)

        again = save_montage(cluster, root / "out")
        check("two saves never overwrite each other", again != out,
              f"{out.name} vs {again.name}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_open_in_viewer():
    calls = []

    def fake_launcher(path):
        calls.append(Path(path))

    check("injected launcher is used",
          open_in_viewer("some file.jpg", launcher=fake_launcher) is True
          and calls == [Path("some file.jpg")], str(calls))

    def broken_launcher(path):
        raise OSError("no viewer on this machine")

    check("viewer failure returns False instead of raising",
          open_in_viewer("some file.jpg", launcher=broken_launcher) is False)


def test_preview_board():
    root = make_workspace()
    try:
        path = add_photo(root, "a.jpg")
        cluster = FaceCluster(cluster_id=0, faces=[face(path)])

        disabled = PreviewBoard(enabled=False, directory=root / "out")
        check("disabled board shows nothing", disabled.show(cluster) is None)

        launched = []
        board = PreviewBoard(open_viewer=False, directory=root / "out",
                             opener=lambda p: launched.append(Path(p)) or True)
        first = board.show(cluster)
        check("show returns the montage path",
              first is not None and first.path.is_file(),
              "" if first is None else str(first.path))
        check("viewer not opened when previews say no",
              first is not None and first.opened is False and launched == [],
              str(launched))

        forced = board.show(cluster, open_viewer=True)
        check("explicit open_viewer=True goes through the opener",
              forced is not None and forced.opened is True
              and len(launched) == 1, str(launched))

        opener_board = PreviewBoard(open_viewer=True, directory=root / "out",
                                    opener=lambda p: True)
        automatic = opener_board.show(cluster)
        check("board default opens the viewer",
              automatic is not None and automatic.opened is True)

        second = board.show(cluster)
        check("each show writes a fresh file",
              second is not None and second.path != first.path)
        check("previews live in the board's folder",
              second is not None
              and second.path.parent == root / "out")

        # a note ends up in the filename-independent caption, and cleanup
        # only removes folders the board created itself
        board.cleanup()
        check("cleanup keeps a caller-provided folder",
              (root / "out").is_dir() and first.path.parent.is_dir())

        owned = PreviewBoard(open_viewer=False)
        result = owned.show(cluster)
        owned_dir = owned.directory
        check("board creates its own temp folder",
              result is not None and owned_dir.is_dir())
        owned.cleanup()
        check("cleanup removes the temp folder", not owned_dir.exists())
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_preview_note_renders():
    root = make_workspace()
    try:
        paths = [add_photo(root, f"p{i}.jpg") for i in range(4)]
        cluster = FaceCluster(
            cluster_id=2,
            faces=[face(paths[i % 4]) for i in range(9)],  # wide 3x3 grid
        )
        plain = build_montage(cluster)
        noted = build_montage(cluster, note="after merge")
        check("caption note does not change the grid size",
              plain.size == noted.size, str(plain.size))
        check("caption note changes the rendered pixels",
              plain.tobytes() != noted.tobytes())
        box = (0, 0, plain.width, CAPTION_HEIGHT)
        check("difference sits in the caption strip",
              plain.crop(box).tobytes() != noted.crop(box).tobytes())
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    test_build_montage_grid()
    test_truncation_and_placeholders()
    test_face_crop_clamps_to_image()
    test_save_montage()
    test_open_in_viewer()
    test_preview_board()
    test_preview_note_renders()
    print()
    if _failures:
        print("FAILURES:", _failures)
        return 1
    print("All preview tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
