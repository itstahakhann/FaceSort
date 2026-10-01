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
        result = client.post("/organize", json={}).json()
        check(result["ok"] is True, "organize succeeded")
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
        again = client.post("/organize", json={"mode": "copy"}).json()
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
        result = http_json(f"{base}/organize", {})
        check(result["ok"] is True, "real organize succeeded")
        check(result["folders"].get("Test Person", 0) >= 1,
              f"the named folder received photos ({result['folders']})")
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
    test_api_flow()
    test_name_db_round_trip()
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
