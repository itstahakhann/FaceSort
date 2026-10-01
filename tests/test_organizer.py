"""Unit tests for the photo organizer (SPEC F7–F12, §10).

Run with::

    python tests/test_organizer.py      # plain runner, no extra dependencies
    pytest tests/                       # also collectable by pytest

Everything runs against temporary folders — no model weights, no prompts.
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clusterer import FaceCluster, FaceRecord
from src.organizer import PhotoOrganizer, normalize_person_name

_failures = []


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}: {label} {detail}")
    if not condition:
        _failures.append(label)


def make_workdir():
    root = Path(tempfile.mkdtemp(prefix="faceorg_org_"))
    (root / "photos").mkdir()
    for name in ("alice1.jpg", "alice2.jpg", "bob1.jpg", "both.jpg", "stray.jpg"):
        (root / "photos" / name).write_bytes(f"content of {name}".encode())
    (root / "out").mkdir()
    return root


def face(path, box=(0, 0, 10, 10)):
    import numpy as np

    return FaceRecord(image_path=path, bbox=box, embedding=np.ones(8, dtype=np.float32))


def cluster(cluster_id, *paths):
    return FaceCluster(cluster_id=cluster_id,
                       faces=[face(p) for p in paths])


def test_normalize_person_name():
    check("plain name passes through", normalize_person_name("Alice") == "Alice")
    check("surrounding spaces are stripped", normalize_person_name("  Bob  ") == "Bob")
    check("skip token -> None", normalize_person_name("skip") is None)
    check("SKIP token -> None", normalize_person_name("SKIP") is None)
    check("unknown token -> None", normalize_person_name("Unknown") is None)
    check("empty input -> None", normalize_person_name("   ") is None)
    check("None -> None", normalize_person_name(None) is None)
    check("path separators are neutralised",
          normalize_person_name("../evil") == ".._evil"
          or normalize_person_name("../evil").find("/") == -1,
          str(normalize_person_name("../evil")))
    check("windows-illegal characters are replaced",
          all(c not in normalize_person_name('a<b>c:"d|e?f*g')
              for c in '<>:"|?*'))
    check("trailing dot is trimmed", normalize_person_name("Ann.") == "Ann")
    check("reserved device name is escaped",
          normalize_person_name("CON") == "_CON")
    check("over-long name is capped",
          len(normalize_person_name("x" * 500)) <= 64)


def test_copy_mode():
    root = make_workdir()
    try:
        organizer = PhotoOrganizer(root / "out", mode="copy")
        summary = organizer.organize([
            ("Alice", cluster(0, root / "photos" / "alice1.jpg",
                              root / "photos" / "alice2.jpg")),
            ("Bob", cluster(1, root / "photos" / "bob1.jpg")),
            (None, cluster(2, root / "photos" / "stray.jpg")),
        ])

        alice = root / "out" / "Alice"
        check("folder per name is created", (alice / "alice1.jpg").is_file()
              and (alice / "alice2.jpg").is_file())
        check("originals survive in copy mode",
              (root / "photos" / "alice1.jpg").is_file())
        check("counts per folder", summary.folders == {"Alice": 2, "Bob": 1,
                                                       "_unknown": 1},
              str(summary.folders))
        check("summary counts placed photos", summary.files_placed == 4)
        check("no errors", summary.ok and summary.errors == [])
        check("folders_created lists new folders",
              len(summary.folders_created) == 3)
        check("report renders", "Alice" in summary.format()
              and "copy" in summary.format())
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_multiple_named_people():
    root = make_workdir()
    try:
        both = root / "photos" / "both.jpg"
        PhotoOrganizer(root / "out", mode="copy").organize([
            ("Alice", cluster(0, both)),
            ("Bob", cluster(1, both)),
        ])
        check("photo with two named people lands in both folders",
              (root / "out" / "Alice" / "both.jpg").is_file()
              and (root / "out" / "Bob" / "both.jpg").is_file())

        # move mode: the original must disappear exactly once
        out2 = root / "out2"
        out2.mkdir()
        PhotoOrganizer(out2, mode="move").organize([
            ("Alice", cluster(0, both)),
            ("Bob", cluster(1, both)),
        ])
        check("move mode: copies exist in both folders",
              (out2 / "Alice" / "both.jpg").is_file()
              and (out2 / "Bob" / "both.jpg").is_file())
        check("move mode: original removed exactly once",
              not both.exists())
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_move_mode():
    root = make_workdir()
    try:
        source = root / "photos" / "bob1.jpg"
        summary = PhotoOrganizer(root / "out", mode="move").organize(
            [("Bob", cluster(0, source))]
        )
        check("move: file appears in the person folder",
              (root / "out" / "Bob" / "bob1.jpg").is_file())
        check("move: original is gone", not source.exists())
        check("move: counters", summary.sources_removed == 1
              and summary.files_placed == 1)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_duplicate_filenames():
    root = make_workdir()
    try:
        # two different photos with the same file name, same person
        (root / "photos" / "holiday").mkdir()
        (root / "photos" / "work").mkdir()
        (root / "photos" / "holiday" / "img.jpg").write_bytes(b"holiday")
        (root / "photos" / "work" / "img.jpg").write_bytes(b"work")

        organizer = PhotoOrganizer(root / "out", mode="copy")
        organizer.organize([
            ("Alice", cluster(0, root / "photos" / "holiday" / "img.jpg",
                              root / "photos" / "work" / "img.jpg")),
        ])
        folder = root / "out" / "Alice"
        names = sorted(p.name for p in folder.iterdir())
        check("clashing names get _1 suffix", names == ["img.jpg", "img_1.jpg"],
              str(names))
        contents = {p.name: p.read_bytes() for p in folder.iterdir()}
        check("both files kept intact",
              sorted(contents.values()) == [b"holiday", b"work"])

        # a second run must not overwrite what the first run wrote
        organizer.organize([
            ("Alice", cluster(0, root / "photos" / "holiday" / "img.jpg")),
        ])
        names = sorted(p.name for p in folder.iterdir())
        check("existing files from an earlier run are not overwritten",
              names == ["img.jpg", "img_1.jpg", "img_2.jpg"], str(names))

        # the same file name in two different folders must NOT be suffixed
        PhotoOrganizer(root / "out", mode="copy").organize([
            ("Bob", cluster(0, root / "photos" / "holiday" / "img.jpg")),
        ])
        check("same name in another folder stays clean",
              (root / "out" / "Bob" / "img.jpg").is_file())
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_unknown_and_edge_cases():
    root = make_workdir()
    try:
        stray = root / "photos" / "stray.jpg"
        organizer = PhotoOrganizer(root / "out", mode="copy")
        summary = organizer.organize_files([
            ("skip", stray),
            ("", stray),            # same photo, same destination -> one copy
            (None, stray),
        ])
        check("skipped clusters go to _unknown",
              (root / "out" / "_unknown" / "stray.jpg").is_file())
        check("one copy per folder, not per assignment",
              summary.copies_written == 1 and summary.folders["_unknown"] == 1,
              str(summary.folders))

        missing = root / "photos" / "nope.jpg"
        summary = organizer.organize_files([("Alice", missing)])
        check("missing source is reported, not raised",
              not summary.ok and summary.errors[0][0] == missing,
              str(summary.errors))
        check("nothing was placed for the missing source",
              summary.files_placed == 0)

        check("empty run is fine",
              organizer.organize([]).files_placed == 0)

        custom_organizer = PhotoOrganizer(root / "out", mode="copy",
                                          unknown_folder="unsorted")
        custom = custom_organizer.organize_files([("skip", stray)])
        check("unknown folder is configurable",
              custom_organizer.unknown_folder == "unsorted"
              and (root / "out" / "unsorted" / "stray.jpg").is_file())
        check("bad mode is rejected", _bad_mode())
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _bad_mode():
    try:
        PhotoOrganizer("somewhere", mode="link")
    except ValueError:
        return True
    return False


def test_from_config():
    import yaml

    config = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "config.yaml").read_text(
            encoding="utf-8"
        )
    )
    organizer = PhotoOrganizer.from_config(config)
    check("from_config reads output_folder",
          organizer.output_folder == Path("./grouped_photos"))
    check("from_config reads mode", organizer.mode == "copy")
    check("from_config reads unknown_folder",
          organizer.unknown_folder == "_unknown")


def main():
    test_normalize_person_name()
    test_copy_mode()
    test_multiple_named_people()
    test_move_mode()
    test_duplicate_filenames()
    test_unknown_and_edge_cases()
    test_from_config()
    print()
    if _failures:
        print("FAILURES:", _failures)
        return 1
    print("All organizer tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
