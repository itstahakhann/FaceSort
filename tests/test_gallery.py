"""Offline HTML gallery export.

Run: python tests/test_gallery.py

Covers folder-name sanitisation, the password digest, thumbnail generation,
the self-contained output (no remote references), overwrite protection and the
HTTP endpoint including progress reporting.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from src.api.server import create_app  # noqa: E402
from src.gallery import (  # noqa: E402
    THUMB_EDGE,
    GalleryError,
    GalleryOptions,
    build_gallery,
    folder_name_for,
    password_digest,
    photos_for_selection,
)
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


def make_photo(path: Path, width: int = 900, height: int = 600,
               colour=(180, 90, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), colour).save(path, format="JPEG",
                                                  quality=90)


def seed_library(root: Path) -> tuple[Path, list[Path]]:
    """A small library: three photos each for John and Mary, one shared."""
    photos = root / "photos"
    shared = photos / "trip.jpg"
    make_photo(shared, 1200, 800, (40, 90, 160))
    john = [shared]
    mary = [shared]
    for index in range(3):
        john.append(photos / f"john_{index}.jpg")
        mary.append(photos / f"mary_{index}.jpg")
        make_photo(john[-1], 800 + index * 100, 600)
        make_photo(mary[-1], 700, 900, (60, 140, 90))

    db_path = root / "gallery.db"
    db = NamesDB(db_path)
    for index, name in enumerate(("John", "Mary")):
        vector = np.zeros(4, np.float32)
        vector[index] = 1.0
        db.upsert_person(name, vector, [], faces_seen=1)
    db.record_occurrences("John", john, faces=1)
    db.record_occurrences("Mary", mary, faces=1)
    db.close()
    return db_path, [shared] + john[1:] + mary[1:]


# ---------------------------------------------------------------------------
def test_folder_names() -> None:
    print("folder-name sanitisation")
    check("a plain name is untouched", folder_name_for("John") == "John")
    check("path separators are replaced",
          "/" not in folder_name_for("a/b") and "\\" not in folder_name_for("a\\b"),
          folder_name_for("a/b"))
    check("Windows-illegal characters are replaced",
          not re.search(r'[<>:"|?*]', folder_name_for('a<b>c:d"e|f?g*h')),
          folder_name_for('a<b>c:d"e|f?g*h'))
    check("control characters are removed",
          folder_name_for("a\x00b\x1fc") == "a_b_c",
          folder_name_for("a\x00b\x1fc"))
    check("a trailing dot is trimmed (Windows drops it anyway)",
          folder_name_for("John. ") == "John", folder_name_for("John. "))
    check("a name of only dots cannot escape the gallery",
          folder_name_for("...") == "unnamed", folder_name_for("..."))
    check("an empty name has a fallback", folder_name_for("") == "unnamed")
    for reserved in ("CON", "prn", "NUL", "COM1", "LPT9"):
        check(f"reserved device name {reserved} is escaped",
              folder_name_for(reserved) != reserved.lower(),
              folder_name_for(reserved))
    check("very long names are truncated",
          len(folder_name_for("x" * 300)) <= 80)

    taken = {"John"}
    check("a colliding name gets a suffix",
          folder_name_for("John", taken) == "John (2)",
          folder_name_for("John", taken))
    taken.add("John (2)")
    check("a second collision increments",
          folder_name_for("John", taken) == "John (3)",
          folder_name_for("John", taken))


def test_password_digest() -> None:
    print("password digest")
    digest = password_digest("hunter2")
    check("it is 64 hex characters (SHA-256)",
          len(digest) == 64 and re.fullmatch(r"[0-9a-f]{64}", digest) is not None)
    check("it is stable", password_digest("hunter2") == digest)
    check("a different password gives a different digest",
          password_digest("hunter3") != digest)
    check("an empty password still hashes", len(password_digest("")) == 64)


def test_build_gallery() -> None:
    print("building a gallery")
    tmp = Path(tempfile.mkdtemp(prefix="faceorg_gal_"))
    try:
        _db, photos = seed_library(tmp)
        out = tmp / "site"
        result = build_gallery(
            GalleryOptions(output_path=str(out)),
            {"John": [str(p) for p in photos if "john" in p.name or "trip" in p.name],
             "Mary": [str(p) for p in photos if "mary" in p.name or "trip" in p.name]},
        )

        check("index.html is written", (out / "index.html").is_file())
        check("the stylesheet is copied",
              (out / "assets" / "css" / "gallery.css").is_file())
        check("the script is copied",
              (out / "assets" / "js" / "gallery.js").is_file())
        check("photos are copied per person",
              (out / "assets" / "images" / "John").is_dir()
              and (out / "assets" / "images" / "Mary").is_dir())
        check("thumbnails are generated",
              (out / "assets" / "thumbnails" / "John" / "john_0.jpg").is_file())
        check("the summary counts photos", result.total_photos == 8,
              # 4 for John + 4 for Mary; trip.jpg is in both folders, which is
              # the whole point of a relationship gallery.
              str(result.total_photos))
        check("the summary lists people", sorted(result.people) == ["John", "Mary"])
        check("it reports the thumbnail count", result.thumbnails == 8,
              str(result.thumbnails))
        check("it reports bytes written", result.bytes_written > 0)

        # thumbnails must actually be small and keep their aspect ratio
        with Image.open(out / "assets" / "thumbnails" / "John" / "john_0.jpg") as thumb:
            check(f"a thumbnail is capped at {THUMB_EDGE}px",
                  max(thumb.size) <= THUMB_EDGE, str(thumb.size))
            check("a thumbnail keeps its aspect ratio",
                  abs((thumb.size[0] / thumb.size[1]) - (800 / 600)) < 0.05,
                  str(thumb.size))
        with Image.open(out / "assets" / "images" / "Mary" / "mary_0.jpg") as full:
            check("the gallery keeps the full frame", max(full.size) > THUMB_EDGE)

        # offline: nothing may point at a remote resource
        html = (out / "index.html").read_text(encoding="utf-8")
        remote = re.findall(r'(?:src|href)\s*=\s*["\'](https?:)?//', html)
        check("the HTML references no remote resources", not remote, str(remote))
        css = (out / "assets" / "css" / "gallery.css").read_text(encoding="utf-8")
        check("the CSS has no @import or url() fetch",
              "@import" not in css and "url(http" not in css)
        js = (out / "assets" / "js" / "gallery.js").read_text(encoding="utf-8")
        check("the JS references no remote resources",
              "http://" not in js and "https://" not in js)
        check("the JS has no CDN loader", "cdn" not in js.lower())

        # the lightbox and password machinery must be present
        for needed in ('id="lightbox"', 'id="lb-prev"', 'id="lb-next"',
                       'id="filter"', 'id="theme-toggle"', 'ArrowLeft',
                       'Escape'):
            check(f"the viewer has {needed}", needed in html or needed in js)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_thumbnails() -> None:
    print("thumbnails disabled")
    tmp = Path(tempfile.mkdtemp(prefix="faceorg_gal2_"))
    try:
        _db, photos = seed_library(tmp)
        out = tmp / "site"
        build_gallery(
            GalleryOptions(output_path=str(out), include_thumbnails=False),
            {"John": [str(p) for p in photos if "john" in p.name]},
        )
        check("no thumbnails directory is created",
              not (out / "assets" / "thumbnails").exists())
        html = (out / "index.html").read_text(encoding="utf-8")
        check("the markup does not reference a missing thumbnail",
              "assets/thumbnails" not in html)
        check("photos are still copied",
              len(list((out / "assets" / "images" / "John").glob("*.jpg"))) == 3)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_password_gate() -> None:
    print("password-protected gallery")
    tmp = Path(tempfile.mkdtemp(prefix="faceorg_gal3_"))
    try:
        _db, photos = seed_library(tmp)
        out = tmp / "site"
        build_gallery(
            GalleryOptions(output_path=str(out), password="hunter2"),
            {"John": [str(p) for p in photos if "john" in p.name]},
        )
        html = (out / "index.html").read_text(encoding="utf-8")
        check("a password gate is rendered", 'id="gate"' in html)
        check("the gallery starts hidden", 'id="app"' in html and 'hidden' in html)
        check("the digest is embedded",
              password_digest("hunter2") in html, "digest missing")
        check("the plain password is NOT embedded", "hunter2" not in html)
        check("the digest is compared with SHA-256", "SHA-256" in html)
        check("the client-side limitation is documented",
              "NOT protection" in html or "not protection" in html.lower())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_overwrite_and_failures() -> None:
    print("overwrite protection and bad input")
    tmp = Path(tempfile.mkdtemp(prefix="faceorg_gal4_"))
    try:
        _db, photos = seed_library(tmp)
        out = tmp / "site"
        options = GalleryOptions(output_path=str(out))
        selection = {"John": [str(p) for p in photos if "john" in p.name]}
        build_gallery(options, selection)
        check("the first build succeeded", (out / "index.html").is_file())

        try:
            build_gallery(options, selection)
            check("a second build refuses to clobber", False)
        except GalleryError as exc:
            check("a second build refuses to clobber",
                  "already exists" in str(exc), str(exc))
        check("the existing gallery is untouched",
              (out / "index.html").is_file())

        build_gallery(GalleryOptions(output_path=str(out), overwrite=True),
                      selection)
        check("overwrite=True replaces it", (out / "index.html").is_file())

        try:
            build_gallery(GalleryOptions(output_path=str(tmp / "empty-site")), {})
            check("an empty selection is refused", False)
        except GalleryError:
            check("an empty selection is refused", True)

        # missing and corrupt files must be skipped, not fatal
        broken = tmp / "broken.jpg"
        broken.write_bytes(b"this is not a JPEG at all")
        gone = tmp / "gone.jpg"
        out2 = tmp / "site2"
        result = build_gallery(
            GalleryOptions(output_path=str(out2)),
            {"John": [str(gone), str(broken),
                      str([p for p in photos if "john" in p.name][0])]},
        )
        check("a corrupt file is skipped, not fatal", result.total_photos == 1,
              str(result.total_photos))
        check("the missing file is reported", len(result.missing) == 1,
              str(result.missing))
        check("the corrupt file is reported", len(result.skipped) == 1,
              str(result.skipped))
        check("the gallery is still written", (out2 / "index.html").is_file())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_selection_helper() -> None:
    print("selection helper")
    tmp = Path(tempfile.mkdtemp(prefix="faceorg_gal5_"))
    try:
        db_path, _photos = seed_library(tmp)
        with NamesDB(db_path) as db:
            every = photos_for_selection(db, "all", [])
            check("mode=all returns everyone", sorted(every) == ["John", "Mary"],
                  str(sorted(every)))
            check("each person has their photos", len(every["John"]) == 4)

            picked = photos_for_selection(db, "people", ["john"])
            check("mode=people is case-insensitive", list(picked) == ["John"],
                  str(list(picked)))

            try:
                photos_for_selection(db, "people", ["Ghost"])
                check("an unknown person is refused", False)
            except GalleryError as exc:
                check("an unknown person is refused", "Not in the name database" in str(exc),
                      str(exc))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_progress_callback() -> None:
    print("progress reporting")
    tmp = Path(tempfile.mkdtemp(prefix="faceorg_gal6_"))
    try:
        _db, photos = seed_library(tmp)
        seen: list[tuple[int, int, str]] = []
        build_gallery(
            GalleryOptions(output_path=str(tmp / "site")),
            {"John": [str(p) for p in photos if "john" in p.name]},
            progress=lambda done, total, who: seen.append((done, total, who)),
        )
        check("the callback fired once per photo", len(seen) == 3, str(len(seen)))
        check("the total is reported", all(total == 3 for _, total, _ in seen))
        check("the count advances",
              [d for d, _, _ in seen] == [1, 2, 3],
              str([d for d, _, _ in seen]))

        # a failing callback must not abort the build
        def boom(done, total, who):
            raise RuntimeError("callback exploded")
        result = build_gallery(
            GalleryOptions(output_path=str(tmp / "site2")),
            {"John": [str(p) for p in photos if "john" in p.name]},
            progress=boom,
        )
        check("a failing progress callback does not abort the build",
              result.total_photos == 3)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_endpoint() -> None:
    print("HTTP endpoint")
    tmp = Path(tempfile.mkdtemp(prefix="faceorg_gal7_"))
    try:
        db_path, _photos = seed_library(tmp)
        config = tmp / "config.yaml"
        config.write_text(
            f"input_folder: {(tmp / 'photos').as_posix()}\n"
            f"output_folder: {(tmp / 'out').as_posix()}\n"
            f"names_db: {db_path.as_posix()}\n",
            encoding="utf-8",
        )
        app = create_app(config)
        with TestClient(app) as client:
            bad = client.post("/export_gallery", json={
                "output_path": str(tmp / "g1"), "export_mode": "people",
                "selected_people": [],
            })
            check("people mode with no selection is a 400", bad.status_code == 400)

            ghost = client.post("/export_gallery", json={
                "output_path": str(tmp / "g1"), "export_mode": "people",
                "selected_people": ["Ghost"], "wait": True,
            })
            check("an unknown person is a 404", ghost.status_code == 404,
                  str(ghost.status_code))

            built = client.post("/export_gallery", json={
                "output_path": str(tmp / "site"), "export_mode": "all",
                "include_thumbnails": True, "wait": True,
            })
            check("a synchronous build succeeds", built.status_code == 200,
                  str(built.status_code))
            body = built.json()
            check("it returns the gallery path",
                  body["output_path"].endswith("site"), str(body.get("output_path")))
            check("it returns a summary", body["total_photos"] == 8,
                  str(body.get("total_photos")))
            check("index.html exists on disk", (tmp / "site" / "index.html").is_file())

            status = client.get("/gallery_status").json()
            check("the finished report is available",
                  status["result"] is not None and status["building"] is False)

            # overwrite guard
            again = client.post("/export_gallery", json={
                "output_path": str(tmp / "site"), "wait": True,
            })
            check("rebuilding without overwrite is a 409", again.status_code == 409,
                  str(again.status_code))
            forced = client.post("/export_gallery", json={
                "output_path": str(tmp / "site"), "wait": True, "overwrite": True,
            })
            check("overwrite=True rebuilds", forced.status_code == 200)

            # background build reports progress
            async_result = client.post("/export_gallery", json={
                "output_path": str(tmp / "site3"), "wait": False,
            })
            check("a background build starts", async_result.json()["started"] is True)
            deadline = time.time() + 60
            while time.time() < deadline:
                if not client.get("/gallery_status").json()["building"]:
                    break
                time.sleep(0.1)
            final = client.get("/gallery_status").json()
            check("the background build finishes",
                  final["result"] is not None, str(final))
            check("and wrote the gallery", (tmp / "site3" / "index.html").is_file())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    test_folder_names()
    test_password_digest()
    test_build_gallery()
    test_no_thumbnails()
    test_password_gate()
    test_overwrite_and_failures()
    test_selection_helper()
    test_progress_callback()
    test_endpoint()
    print()
    if _failures:
        print(f"{len(_failures)} of {_checks[0]} checks FAILED:")
        for label in _failures:
            print(f"  - {label}")
        return 1
    print(f"All gallery tests passed ({_checks[0]} checks).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())