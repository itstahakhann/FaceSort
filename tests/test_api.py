"""Tests for the local API backend (``src/api/server.py``).

Run with::

    python tests/test_api.py         # fast: no ONNX models, stubbed pipeline
    python tests/test_api.py --real  # also drive a real scan + real organize

The default run replaces :func:`src.api.server.run_pipeline` with a stub so
the whole REST surface (progress, cluster serialization, thumbnails, naming,
organizing, error paths) is covered in under a second and without the
300 MB of weights.  ``--real`` additionally starts the actual server as a
subprocess, checks the ``PORT:<port>`` handshake Electron depends on, and
runs a genuine three-photo scan through the real models.
"""

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():  # pragma: no cover - odd checkout
    PYTHON = Path(sys.executable)

_checks = 0
_failures = []


def check(condition, message):
    global _checks
    _checks += 1
    if not condition:
        _failures.append(message)
        print(f"  FAIL: {message}")


def make_photo(path: Path, seed: int = 0) -> Path:
    """Write a small deterministic JPEG with a face-ish bright square."""
    rng = np.random.default_rng(seed)
    image = (rng.integers(40, 90, size=(120, 120, 3))).astype(np.uint8)
    image[30:80, 35:85] = (230, 200, 190)  # a "face" rectangle
    try:
        import cv2

        cv2.imwrite(str(path), image)
    except ImportError:  # pragma: no cover
        from PIL import Image

        Image.fromarray(image[:, :, ::-1]).save(path)
    return path


def build_client(tmp: Path, stub_pipeline, config_path: Path = None):
    """A TestClient over a fresh app whose pipeline is stubbed out."""
    from fastapi.testclient import TestClient

    import src.api.server as server_mod

    server_mod.run_pipeline = stub_pipeline  # type: ignore[assignment]
    app = server_mod.create_app(
        config_path=config_path or (REPO_ROOT / "config.yaml"))
    app.state.session.config = {}
    return TestClient(app), app


def write_config(path: Path, names_db: Path) -> Path:
    """A config.yaml pointing the name DB somewhere disposable."""
    path.write_text(
        "input_folder: ./input_photos\n"
        "output_folder: ./grouped_photos\n"
        "tolerance: 0.5\n"
        "min_faces_per_cluster: 1\n"
        "mode: copy\n"
        "unknown_folder: _unknown\n"
        f"names_db: {names_db.as_posix()}\n",
        encoding="utf-8",
    )
    return path


def fake_pipeline(records_for, seconds=0.0):
    """Build a stand-in for :func:`src.main.run_pipeline`."""

    def run(config, progress_cb=None):
        from src.main import PipelineStats

        stats = PipelineStats()
        paths = sorted(Path(config["input_folder"]).iterdir())
        for index, path in enumerate(paths, 1):
            if progress_cb is not None:
                progress_cb(index, len(paths), path)
            stats.images_scanned += 1
            stats.faces_detected += len(records_for(path))
            stats.faces_embedded += len(records_for(path))
            stats.records.extend(records_for(path))
        stats.seconds = seconds
        return stats

    return run


def embedding_for(name: str) -> np.ndarray:
    """Stable pseudo-embedding; same name -> same vector (so it clusters)."""
    rng = np.random.default_rng(abs(hash(name)) % (2 ** 32))
    vector = rng.normal(size=512).astype(np.float32)
    return vector / np.linalg.norm(vector)


def wait_for_state(client, wanted, timeout=20.0):
    deadline = time.time() + timeout
    body = {}
    while time.time() < deadline:
        body = client.get("/status").json()
        if body["state"] in wanted:
            return body
        time.sleep(0.05)
    return body


def run_organize(client, mode=None, timeout=60.0):
    """POST /organize (async) and wait for the worker thread to finish."""
    started = client.post("/organize", json={} if not mode else {"mode": mode})
    if started.status_code != 200:
        return started.status_code, started.json(), {}
    body = wait_for_state(client, {"done", "ready", "error"}, timeout)
    return 200, body.get("results") or {}, body


def test_handshake_helpers():
    print("port handshake helpers")
    import src.api.server as server_mod

    check(server_mod.format_port_line(8123) == "PORT:8123",
          "format_port_line must be exactly 'PORT:8123'")
    sock = server_mod.bind_local_socket()
    try:
        host, port = sock.getsockname()
        check(host == "127.0.0.1", "server must bind 127.0.0.1 only")
        check(port > 0, "an OS-assigned free port must be reported")
        check(server_mod.format_port_line(port).startswith("PORT:"),
              "port line prefix")
    finally:
        sock.close()


def test_api_flow():
    print("REST flow (stubbed pipeline)")
    tmp = Path(tempfile.mkdtemp(prefix="faceorg_api_"))
    try:
        photos = tmp / "input_photos"
        photos.mkdir()
        for index, name in enumerate(("alice_1.jpg", "alice_2.jpg",
                                      "bob_1.jpg")):
            make_photo(photos / name, seed=index)

        faces = {
            "alice_1.jpg": ["Alice"],
            "alice_2.jpg": ["Alice"],
            "bob_1.jpg": ["Bob"],
        }

        def records_for(path):
            from src.clusterer import FaceRecord

            return [
                FaceRecord(image_path=path, bbox=(30, 30, 80, 80),
                           embedding=embedding_for(who), score=0.9)
                for who in faces[path.name]
            ]

        client, app = build_client(tmp, fake_pipeline(records_for))

        # -- /status before anything happened ---------------------------
        body = client.get("/status").json()
        check(body["state"] == "idle", "initial state is idle")
        check(body["clusters"] == 0 and body["scanning"] is False,
              "no clusters before a scan")
        check("config" in body and "tolerance" in body["config"],
              "status exposes the config defaults for the sidebar")
        check(body["config"]["input_exists"] is False,
              "a non-existent default folder is flagged (sidebar stays empty)")
        check(body["config"]["output_exists"] is False,
              "same for the output folder")
        check(body["models"]["name"] == "buffalo_l",
              "status reports the model name")
        check(client.get("/clusters").status_code == 409,
              "/clusters before a scan is a 409")

        # -- validation ---------------------------------------------------
        bad = client.post("/scan", json={"input_folder": str(tmp / "nope"),
                                         "output_folder": str(tmp / "out")})
        check(bad.status_code == 400, "missing input folder -> 400")
        check("does not exist" in bad.json()["detail"],
              "the 400 explains why")

        # -- scan ---------------------------------------------------------
        response = client.post("/scan", json={
            "input_folder": str(photos),
            "output_folder": str(tmp / "grouped_photos"),
            "tolerance": 0.5,
            "min_faces_per_cluster": 1,   # keep Bob's single face as a group
            "mode": "copy",
            "workers": 1,
            "use_db": False,
        })
        check(response.status_code == 200, "POST /scan accepted")
        check(response.json()["total"] == 3, "scan reports the image count")

        body = wait_for_state(client, {"ready", "error"})
        check(body["state"] == "ready",
              f"scan reaches 'ready' (got {body['state']}: {body.get('error')})")
        check(body["stats"]["photos"] == 3, "three photos scanned")
        check(body["stats"]["faces"] == 3, "three faces detected")
        check(body["processed"] == 3 and body["total"] == 3,
              "progress counters end at 3/3")

        # -- /clusters ----------------------------------------------------
        payload = client.get("/clusters").json()
        check(payload["count"] == 2, f"two clusters, got {payload['count']}")
        clusters = sorted(payload["clusters"], key=lambda c: c["id"])
        sizes = sorted(c["size"] for c in clusters)
        check(sizes == [1, 2], f"cluster sizes 1+2, got {sizes}")
        check(all(c["name"] == "" and not c["auto"] for c in clusters),
              "nothing is named yet")
        thumbs = [face["thumb"] for c in clusters for face in c["faces"]]
        check(all(t and t.startswith("data:image/jpeg;base64,") for t in thumbs),
              "every face carries an inline JPEG thumbnail")
        raw = base64.b64decode(thumbs[0].split(",", 1)[1])
        check(raw[:2] == b"\xff\xd8", "thumbnail really is a JPEG")
        check(all("photo" in f and f["photo"] for c in clusters
                  for f in c["faces"]), "faces carry their photo name")

        # -- naming --------------------------------------------------------
        alice = next(c for c in clusters if c["size"] == 2)
        bob = next(c for c in clusters if c["size"] == 1)
        check(client.post("/name_cluster",
                          json={"cluster_id": 999, "name": "Ghost"}
                          ).status_code == 404,
              "naming an unknown cluster -> 404")

        named = client.post("/name_cluster",
                            json={"cluster_id": alice["id"],
                                  "name": "  Alice Smith "}).json()
        check(named["name"] == "Alice Smith", "name is normalised")
        check(named["remembered"] is False,
              "use_db=False means the name is NOT written to the DB")
        check(not (REPO_ROOT / "facesort_names.db").exists(),
              "use_db=False leaves no name DB behind in the repo")
        check(client.post("/name_cluster",
                          json={"cluster_id": bob["id"], "name": ""}
                          ).json()["name"] == "",
              "empty name = skip (unknown folder)")

        after = {c["id"]: c for c in client.get("/clusters").json()["clusters"]}
        check(after[alice["id"]]["name"] == "Alice Smith",
              "the name shows up in /clusters")
        status = client.get("/status").json()
        check(status["named"] == 1 and status["unknown"] == 1,
              "status counts named/unknown clusters")

        # -- organize ------------------------------------------------------
        code, result, org_status = run_organize(client)
        check(code == 200, "organize accepted")
        check(org_status.get("organizing") is False,
              "the organizing flag is cleared when it finishes")
        check(org_status["organize"]["total"] == 3,
              f"organize counted 3 source photos ({org_status['organize']})")
        check(result.get("ok") is True, "organize succeeded")
        check(result["folders"].get("Alice Smith") == 2,
              f"Alice got both photos, got {result['folders']}")
        check(result["folders"].get("_unknown") == 1,
              "the skipped cluster went to _unknown")
        out = tmp / "grouped_photos"
        check(len(list((out / "Alice Smith").glob("*.jpg"))) == 2,
              "two files on disk for Alice")
        check(len(list((out / "_unknown").glob("*.jpg"))) == 1,
              "one file on disk for the unknown folder")
        check((out / "Alice Smith" / "alice_1.jpg").is_file(),
              "original file names are preserved")
        check(len(list((photos).glob("*.jpg"))) == 3,
              "copy mode leaves the sources alone")
        check(client.get("/status").json()["state"] == "done",
              "state becomes 'done' after organizing")

        # re-organizing must not clobber (dedup suffixes)
        _code, again, _s = run_organize(client, mode="copy")
        check(len(list((out / "Alice Smith").glob("*.jpg"))) == 4,
              "a second run writes _1 duplicates instead of overwriting")
        check(again["copies_written"] == 3, "second run placed 3 more files")

        # -- error propagation ---------------------------------------------
        failure_app_client, _ = build_client(tmp, fake_pipeline(records_for))
        import src.api.server as server_mod
        server_mod.run_pipeline = _explode  # type: ignore[assignment]
        failure_app_client.post("/scan", json={
            "input_folder": str(photos), "output_folder": str(tmp / "o2"),
            "use_db": False,
        })
        body = wait_for_state(failure_app_client, {"error"})
        check(body["state"] == "error", "a failing scan ends in 'error'")
        check("RuntimeError" in body["error"],
              f"the error message reaches the UI, got {body['error']!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_name_db_round_trip():
    """use_db=True: names are remembered, and a later scan auto-labels them.

    This is the API's version of M3 — the second scan of the same folder must
    come back with the name already filled in.
    """
    print("name DB: remember + auto-label on the next scan")
    tmp = Path(tempfile.mkdtemp(prefix="faceorg_api_db_"))
    try:
        photos = tmp / "input_photos"
        photos.mkdir()
        for index, name in enumerate(("ann_1.jpg", "ann_2.jpg", "bo_1.jpg")):
            make_photo(photos / name, seed=10 + index)
        faces = {"ann_1.jpg": ["Ann"], "ann_2.jpg": ["Ann"], "bo_1.jpg": ["Bo"]}

        def records_for(path):
            from src.clusterer import FaceRecord

            return [FaceRecord(image_path=path, bbox=(30, 30, 80, 80),
                               embedding=embedding_for(who), score=0.9)
                    for who in faces[path.name]]

        db_path = tmp / "names.db"
        config = write_config(tmp / "config.yaml", db_path)
        client, _ = build_client(tmp, fake_pipeline(records_for), config_path=config)

        def scan():
            client.post("/scan", json={
                "input_folder": str(photos), "output_folder": str(tmp / "out"),
                "min_faces_per_cluster": 1, "use_db": True,
            })
            body = wait_for_state(client, {"ready", "error"})
            check(body["state"] == "ready", f"scan ready ({body.get('error')})")
            return {c["id"]: c for c in client.get("/clusters").json()["clusters"]}

        first = scan()
        ann = max(first.values(), key=lambda c: c["size"])
        result = client.post("/name_cluster",
                             json={"cluster_id": ann["id"], "name": "Ann Lee"}
                             ).json()
        check(result["remembered"] is True, "use_db=True persists the name")
        check(db_path.exists(), f"the name DB was created at {db_path}")

        second = scan()
        labelled = [c for c in second.values() if c["auto"]]
        check(len(labelled) == 1,
              f"exactly one group is auto-labelled on the second scan ({len(labelled)})")
        if labelled:
            check(labelled[0]["name"] == "Ann Lee",
                  f"the remembered name comes back ({labelled[0]['name']!r})")
        check(any(not c["name"] for c in second.values()),
              "the unknown person is still unnamed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_merge_clusters():
    """POST /merge_clusters links two groups as one person at different ages."""
    print("manual merge (same person, different age)")
    tmp = Path(tempfile.mkdtemp(prefix="faceorg_api_merge_"))
    try:
        photos = tmp / "input_photos"
        photos.mkdir()
        # two groups of one person: "kid" (3 photos) and "adult" (2 photos)
        plan = {
            "kid_1.jpg": "kid", "kid_2.jpg": "kid", "kid_3.jpg": "kid",
            "adult_1.jpg": "adult", "adult_2.jpg": "adult",
        }
        for index, name in enumerate(plan):
            make_photo(photos / name, seed=40 + index)

        def records_for(path):
            from src.clusterer import FaceRecord

            return [FaceRecord(
                image_path=path, bbox=(30, 30, 80, 80),
                embedding=embedding_for(plan[path.name]), score=0.9,
            )]

        db_path = tmp / "names.db"
        config = write_config(tmp / "config.yaml", db_path)
        client, _ = build_client(tmp, fake_pipeline(records_for), config_path=config)

        check(client.post("/merge_clusters",
                          json={"cluster_a": 0, "cluster_b": 0}).status_code == 409,
              "merging before a scan is a 409")

        client.post("/scan", json={
            "input_folder": str(photos), "output_folder": str(tmp / "out"),
            "min_faces_per_cluster": 1, "use_db": True,
        })
        wait_for_state(client, {"ready", "error"})
        before = {c["id"]: c for c in client.get("/clusters").json()["clusters"]}
        check(len(before) == 2, f"two groups to start with ({len(before)})")
        sizes = sorted(c["size"] for c in before.values())
        check(sizes == [2, 3], f"group sizes are 3 and 2 ({sizes})")

        kid = max(before.values(), key=lambda c: c["size"])["id"]
        adult = min(before.values(), key=lambda c: c["size"])["id"]

        check(client.post("/merge_clusters",
                          json={"cluster_a": kid, "cluster_b": kid}).status_code
              == 400, "merging a group with itself is a 400")
        check(client.post("/merge_clusters",
                          json={"cluster_a": kid, "cluster_b": 99}).status_code
              == 404, "an unknown group id is a 404")

        merged = client.post("/merge_clusters", json={
            "cluster_a": adult, "cluster_b": kid, "name": "Sam",
        })
        check(merged.status_code == 200, f"merge accepted ({merged.status_code})")
        result = merged.json()
        check(result["ok"] is True, "merge reports success")
        check(result["size"] == 5,
              f"the merged group has all 5 faces ({result.get('size')})")
        check(result["remembered"] is True,
              "a named merge is written to the name database")

        after = {c["id"]: c for c in client.get("/clusters").json()["clusters"]}
        check(len(after) == 1, f"one group remains ({len(after)})")
        if after:
            only = list(after.values())[0]
            check(only["size"] == 5, f"it holds every face ({only['size']})")
            check(only["photos"] == 5, f"and every photo ({only['photos']})")
            check(only["name"] == "Sam", f"it carries the name ({only['name']!r})")

        # ids must be renumbered without gaps, since the UI keys off them
        check(sorted(after) == list(range(len(after))),
              f"cluster ids are contiguous ({sorted(after)})")

        # the merged person is now remembered: the organizer sees one group
        _code, report, _s = run_organize(client)
        check(report.get("folders", {}).get("Sam") == 5,
              f"all 5 photos land in Sam's folder ({report.get('folders')})")

        # a re-scan should auto-label the combined group from the database
        client.post("/scan", json={
            "input_folder": str(photos), "output_folder": str(tmp / "out"),
            "min_faces_per_cluster": 1, "use_db": True,
        })
        wait_for_state(client, {"ready", "error"})
        rescanned = client.get("/clusters").json()["clusters"]
        check(len(rescanned) == 1, f"still one group after a re-scan ({len(rescanned)})")
        if rescanned:
            check(rescanned[0]["auto"] is True,
                  "the remembered name auto-labels the merged group")
            check(rescanned[0]["name"] == "Sam",
                  f"with the right name ({rescanned[0]['name']!r})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_db_path_resolution():
    """The name DB lives in the per-user data dir, never beside the exe."""
    print("name DB path resolution")
    from src.names_db import (APP_DIR_NAME, DB_FILENAME, default_db_path,
                              open_names_db, resolve_db_path, user_data_dir)

    data_dir = user_data_dir()
    check(data_dir.name == APP_DIR_NAME,
          f"data dir is named {APP_DIR_NAME} (got {data_dir.name})")
    check(data_dir.is_absolute(), "the data dir is an absolute path")
    check(default_db_path().name == DB_FILENAME,
          f"the DB file is {DB_FILENAME}")
    check(default_db_path().parent == data_dir,
          "the DB sits directly inside the data dir")

    # the legacy spelling must NOT resolve to the working directory any more
    for legacy in ("./facesort_names.db", "facesort_names.db"):
        check(resolve_db_path(legacy) == default_db_path(),
              f"{legacy!r} resolves to the per-user DB")
    check(resolve_db_path("./data/custom.db") == Path("./data/custom.db"),
          "an explicit relative path is honoured as given")
    check(resolve_db_path("/tmp/facesort.db") == Path("/tmp/facesort.db"),
          "an explicit absolute path is honoured")

    with tempfile.TemporaryDirectory() as sandbox:
        old = os.environ.get("FACEORG_DATA_DIR")
        os.environ["FACEORG_DATA_DIR"] = sandbox
        try:
            check(user_data_dir() == Path(sandbox),
                  "FACEORG_DATA_DIR overrides the location")
            check(default_db_path() == Path(sandbox) / DB_FILENAME,
                  "the override drives the DB path too")
            db = open_names_db({"names_db": "./facesort_names.db"})
            check(db is not None, "the DB opens under the override")
            if db is not None:
                db.close()
            check((Path(sandbox) / DB_FILENAME).exists(),
                  "the DB file was created inside the override directory")
        finally:
            if old is None:
                os.environ.pop("FACEORG_DATA_DIR", None)
            else:
                os.environ["FACEORG_DATA_DIR"] = old

    check(open_names_db({"names_db": ""}) is None,
          "an empty names_db still disables the database")
    check(open_names_db({"names_db": "none"}) is None,
          "names_db: none still disables the database")


def test_thumb_endpoint():
    """GET /thumb + /photo stream previews of photos inside the scanned folder."""
    print("live preview endpoints (/thumb, /photo)")
    tmp = Path(tempfile.mkdtemp(prefix="faceorg_api_thumb_"))
    try:
        photos = tmp / "input_photos"
        photos.mkdir()
        make_photo(photos / "shot_a.jpg", seed=3)
        make_photo(photos / "shot_b.jpg", seed=4)
        (tmp / "secret.txt").write_text("not an image", encoding="utf-8")

        client, _ = build_client(tmp, fake_pipeline(lambda path: []))
        client.post("/scan", json={
            "input_folder": str(photos), "output_folder": str(tmp / "out"),
            "use_db": False,
        })
        wait_for_state(client, {"ready", "error"})

        response = client.get("/thumb", params={"name": "shot_a.jpg"})
        check(response.status_code == 200, f"thumb 200 ({response.status_code})")
        check(response.headers["content-type"] == "image/jpeg",
              "thumb is served as image/jpeg")
        check(response.content[:2] == b"\xff\xd8", "thumb really is a JPEG")
        check(len(response.content) < 200_000, "the preview is downscaled")

        check(client.get("/thumb", params={"name": "missing.jpg"}).status_code
              == 404, "a missing photo is a 404")
        # the reported config must show where the DB really is, not the
        # legacy "./facesort_names.db" spelling from config.yaml
        reported = client.get("/status").json()["config"]["names_db"]
        check(Path(reported).is_absolute(),
              f"status reports an absolute names_db path (got {reported})")
        check(Path(reported).name == "facesort_names.db",
              "status points at facesort_names.db")
        check(client.get("/thumb", params={"name": "secret.txt"}).status_code
              == 404, "non-images are refused")
        check(client.get("/thumb",
                         params={"name": "../secret.txt"}).status_code == 404,
              "path traversal is refused")
        outside = client.get("/thumb", params={"name": str(tmp / "secret.txt")})
        check(outside.status_code in (403, 404),
              "absolute paths outside the folder are refused")

        again = client.get("/thumb", params={"name": "shot_b.jpg"})
        check(again.status_code == 200, "a second photo also renders")

        # /photo returns the whole frame for the review lightbox
        full = client.get("/photo", params={"name": "shot_a.jpg"})
        check(full.status_code == 200, f"photo 200 ({full.status_code})")
        check(full.headers["content-type"] == "image/jpeg",
              "photo is served as image/jpeg")
        check(full.content[:2] == b"\xff\xd8", "photo really is a JPEG")
        check(len(full.content) >= len(response.content),
              "the whole photo is not smaller than the scan preview")
        check(client.get("/photo", params={"name": "../secret.txt"}).status_code
              == 404, "/photo refuses path traversal too")
        check(client.get("/photo", params={"name": "missing.jpg"}).status_code
              == 404, "/photo 404s on a missing file")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_scanning_state():
    """The session must report 'scanning' while a scan is in flight.

    Regression guard: an instant stub finishes before any poll can observe
    the intermediate state, and the UI's progress bar depends on it.
    """
    print("scanning state is observable")
    import threading

    tmp = Path(tempfile.mkdtemp(prefix="faceorg_api_state_"))
    try:
        photos = tmp / "input_photos"
        photos.mkdir()
        make_photo(photos / "a.jpg", seed=1)
        make_photo(photos / "b.jpg", seed=2)

        release = threading.Event()

        def slow_pipeline(config, progress_cb=None):
            from src.main import PipelineStats

            if progress_cb is not None:
                progress_cb(1, 2, photos / "a.jpg")
            release.wait(20)
            stats = PipelineStats()
            stats.images_scanned = 2
            return stats

        client, _ = build_client(tmp, slow_pipeline)
        client.post("/scan", json={
            "input_folder": str(photos), "output_folder": str(tmp / "out"),
            "use_db": False,
        })

        deadline = time.time() + 10
        body = {}
        while time.time() < deadline:
            body = client.get("/status").json()
            if body["state"] == "scanning" or body["state"] == "ready":
                break
            time.sleep(0.02)
        check(body["state"] == "scanning",
              f"a running scan reports 'scanning' (got {body['state']})")
        check(body["scanning"] is True, "the scanning flag is set")
        check(body["processed"] == 1 and body["total"] == 2,
              f"progress is exposed mid-scan ({body['processed']}/{body['total']})")

        release.set()
        final = wait_for_state(client, {"ready"})
        check(final["state"] == "ready", "the scan still finishes afterwards")
    finally:
        release.set()
        shutil.rmtree(tmp, ignore_errors=True)


def _explode(config, progress_cb=None):
    raise RuntimeError("detector exploded")


def test_models_and_default_config():
    print("model root + config resolution")
    import src.api.server as server_mod
    import src.face_model as face_model

    original = os.environ.pop("FACEORG_MODEL_ROOT", None)
    try:
        os.environ["FACEORG_MODEL_ROOT"] = str(REPO_ROOT / "build" / "models")
        check(face_model.model_root() == REPO_ROOT / "build" / "models",
              "FACEORG_MODEL_ROOT wins (Electron packaged mode)")
        check(Path(face_model.model_dir()).parts[-2:] == ("models", "buffalo_l"),
              "model_dir appends models/buffalo_l to the root")
    finally:
        os.environ.pop("FACEORG_MODEL_ROOT", None)
        if original is not None:
            os.environ["FACEORG_MODEL_ROOT"] = original

    check(server_mod.default_config_path().name == "config.yaml",
          "dev default config path lives in the repo")
    check(server_mod.PORT_PREFIX == "PORT:",
          "Electron handshake prefix is stable")


# --------------------------------------------------------------------------
# optional real end-to-end run (needs the ONNX weights)
# --------------------------------------------------------------------------
def http_json(url, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def collect_real_photos(target: Path, limit: int = 3) -> int:
    """Copy a few real photos into ``target`` for the real end-to-end run.

    Random noise is not detectable by SCRFD, so the real run needs genuine
    photographs.  Looks in the usual user folders (and the Windows wallpaper
    set as a fallback) rather than shipping binaries in the repository.
    """
    search = [Path(arg.split("=", 1)[1]) for arg in sys.argv
              if arg.startswith("--photos=")]
    search += [Path.home() / "Pictures", Path.home() / "Desktop",
               Path("C:/Windows/Web/Wallpaper")]
    for folder in search:
        if not folder.is_dir():
            continue
        photos = sorted(
            path for path in folder.glob("*.jpg")
            if path.is_file() and path.stat().st_size > 60_000
        )
        for source in photos[:limit]:
            shutil.copy2(source, target / source.name)
        if list(target.glob("*.jpg")):
            break
    return len(list(target.glob("*.jpg")))


def test_real_server():
    print("real server: PORT handshake + real scan (needs models)")
    import src.face_model as face_model

    if not face_model.models_installed():
        print("  SKIP: buffalo_l weights are not installed")
        return

    tmp = Path(tempfile.mkdtemp(prefix="faceorg_api_real_"))
    process = None
    try:
        photos = tmp / "input_photos"
        photos.mkdir()
        count = collect_real_photos(photos)
        if not count:
            print("  SKIP: no real photographs found (use --photos=<dir>)")
            return

        env = dict(os.environ, PYTHONUNBUFFERED="1")
        process = subprocess.Popen(
            [str(PYTHON), "-m", "src.api.server", "-v"],
            cwd=str(REPO_ROOT), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

        port = None
        deadline = time.time() + 60
        while time.time() < deadline:
            line = process.stdout.readline()
            if not line:
                break
            if line.startswith("PORT:"):
                port = int(line.split(":", 1)[1].strip())
                break
        check(port is not None, "server prints PORT:<port> on stdout")
        if port is None:
            return

        base = f"http://127.0.0.1:{port}"
        status = http_json(f"{base}/status")
        check(status["service"] == "facesort", "GET /status answers")
        check(status["models"]["installed"] is True, "models detected")
        check(status["state"] == "idle", "fresh server is idle")

        started = http_json(f"{base}/scan", {
            "input_folder": str(photos), "output_folder": str(tmp / "out"),
            "min_faces_per_cluster": 1, "workers": 1, "use_db": False,
        })
        check(started["started"] is True, "POST /scan starts")

        deadline = time.time() + 300
        while time.time() < deadline:
            status = http_json(f"{base}/status")
            if status["state"] in ("ready", "error"):
                break
            time.sleep(0.5)
        check(status["state"] == "ready",
              f"real scan finished (got {status['state']}: {status['error']})")
        check(status["stats"]["photos"] == count,
              f"real scan counted {count} photo(s)")

        if status["stats"]["faces"] == 0:
            print("  NOTE: no faces in the sample photos; skipping the "
                  "cluster/organize assertions")
            return

        payload = http_json(f"{base}/clusters")
        check(payload["count"] >= 1,
              "the real pipeline produced at least one cluster")
        check(all(face["thumb"].startswith("data:image/jpeg;base64,")
                  for cluster in payload["clusters"]
                  for face in cluster["faces"]),
              "real faces render as inline thumbnails")
        first = payload["clusters"][0]
        named = http_json(f"{base}/name_cluster",
                          {"cluster_id": first["id"], "name": "Test Person"})
        check(named["ok"] is True, "POST /name_cluster accepted")

        # /organize is asynchronous: it returns {"started": true, ...}
        started = http_json(f"{base}/organize", {})
        check(started.get("started") is True, "POST /organize starts")
        check(started.get("total", 0) >= 1,
              f"organize counted the photos ({started})")
        deadline = time.time() + 300
        while time.time() < deadline:
            status = http_json(f"{base}/status")
            if status["state"] in ("done", "error"):
                break
            time.sleep(0.5)
        check(status["state"] == "done",
              f"organize finished (got {status['state']}: {status['error']})")
        result = status["results"] or {}
        check(result.get("ok") is True, "real organize succeeded")
        check(result["folders"].get("Test Person", 0) >= 1,
              f"the named folder received photos ({result.get('folders')})")
        placed = sum(result["folders"].values())
        on_disk = sum(1 for _ in (tmp / "out").rglob("*.jpg"))
        check(on_disk == placed,
              f"every placed photo exists on disk ({on_disk} vs {placed})")
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover
                process.kill()
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("Running API tests...")
    test_handshake_helpers()
    test_models_and_default_config()
    test_scanning_state()
    test_db_path_resolution()
    test_thumb_endpoint()
    test_api_flow()
    test_name_db_round_trip()
    test_merge_clusters()
    if "--real" in sys.argv:
        test_real_server()

    if _failures:
        print(f"\n{len(_failures)} of {_checks} checks FAILED:")
        for failure in _failures:
            print(f"  - {failure}")
        return 1
    print(f"\nAll {_checks} API checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
