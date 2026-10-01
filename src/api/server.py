"""Loopback REST API that exposes the pipeline to the Electron frontend.

Design constraints
------------------
**stdout is a protocol channel.**  The Electron main process spawns this
server (as ``backend.exe`` in packaged builds) and waits for a single line
``PORT:<port>`` to learn which free port the OS handed out.  Therefore
*nothing* else is ever written to stdout: all logging goes to stderr.

**Single user, single session.**  This is a desktop app, not a service, so
the current scan (records, clusters, names, progress) lives in one
in-memory :class:`ScanSession` guarded by a lock instead of a database.
The SQLite name DB (``src.names_db``) is still the persistent memory of
"who is who" (M3), and is opened per operation so no connection is shared
between threads.

**Offline.**  Inference only ever reads ONNX files that are already on disk
(``src.face_model``); the models ship inside the PyInstaller bundle (or
next to it, see ``FACEORG_MODEL_ROOT``) and nothing here reaches the
network.

Endpoints
---------
``GET  /status``        progress + configuration + model/port info (also the
                        readiness probe the frontend polls while booting)
``POST /scan``          start a background scan (never blocks the event loop)
``GET  /clusters``      clusters with per-face JPEG thumbnails (data URLs)
``POST /name_cluster``  name (or skip) one cluster, remembering it in the DB
``POST /organize``      copy/move every photo into ``output/<name>/``
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..clusterer import FaceCluster, FaceClusterer
from ..face_model import (
    MODEL_NAME,
    ModelsNotInstalledError,
    model_dir,
    models_installed,
)
from ..image_loader import load_image, scan_images
from ..main import DEFAULT_CONFIG, load_config, run_pipeline, save_named_cluster
from ..names_db import open_names_db
from ..organizer import PhotoOrganizer, normalize_person_name
from ..ui.preview import face_crop

logger = logging.getLogger("facesort.api")

#: Marker the Electron main process greps for on the child's stdout.
PORT_PREFIX = "PORT:"

#: Rendered face thumbnail edge length in pixels (JPEG data URL).
THUMB_CELL = 192
THUMB_QUALITY = 80

SERVICE = "facesort"
API_VERSION = "1.0"

#: Session states, in the order a user moves through them.
STATE_IDLE = "idle"
STATE_SCANNING = "scanning"
STATE_READY = "ready"          # clusters waiting to be named
STATE_ORGANIZING = "organizing"
STATE_DONE = "done"            # photos have been sorted
STATE_ERROR = "error"

__all__ = [
    "create_app",
    "serve",
    "main",
    "bind_local_socket",
    "format_port_line",
    "default_config_path",
    "ScanSession",
]


# --------------------------------------------------------------------------
# configuration / small helpers
# --------------------------------------------------------------------------
def default_config_path() -> Path:
    """Where ``config.yaml`` lives for this process.

    Order: ``FACEORG_CONFIG`` env var, then the copy bundled into the frozen
    executable, then the repository's ``config.yaml`` (dev checkout).
    """
    env = os.environ.get("FACEORG_CONFIG")
    if env:
        return Path(env).expanduser()
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        return base / "config.yaml"
    return DEFAULT_CONFIG


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_dir(value: Any) -> bool:
    """True when a configured folder actually exists (used by the sidebar)."""
    if not value:
        return False
    try:
        return Path(str(value)).expanduser().is_dir()
    except OSError:  # pragma: no cover - weird path characters
        return False


def _thumb_data_url(path: Path, bbox: Tuple[int, int, int, int]) -> Optional[str]:
    """Render one face crop as a ``data:image/jpeg;base64,...`` URL.

    Data URLs (rather than file paths or a static route) keep the renderer
    free of ``file://``/CSP problems and need no temp-file bookkeeping.  One
    cropped 192 px JPEG is ~5 kB, so even a few hundred faces are cheap.
    """
    image = load_image(path)
    if image is None:
        return None
    try:
        face = face_crop(image, bbox, cell_size=THUMB_CELL)
    except Exception as exc:  # noqa: BLE001 - a bad crop must not break the UI
        logger.debug("crop failed for %s: %s", path, exc)
        return None
    buffer = io.BytesIO()
    face.save(buffer, format="JPEG", quality=THUMB_QUALITY)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


# --------------------------------------------------------------------------
# request models
# --------------------------------------------------------------------------
class ScanRequest(BaseModel):
    """Body of ``POST /scan``; every field overrides ``config.yaml``."""

    input_folder: str = Field(..., min_length=1)
    output_folder: str = Field(..., min_length=1)
    tolerance: float = Field(0.5, ge=0.0, le=2.0)
    min_faces_per_cluster: int = Field(2, ge=1, le=1000)
    mode: str = Field("copy", pattern="^(copy|move)$")
    unknown_folder: str = Field("_unknown", min_length=1)
    workers: int = Field(0, ge=0, le=64)
    use_db: bool = True


class NameRequest(BaseModel):
    """Body of ``POST /name_cluster`` (empty name = unknown folder)."""

    cluster_id: int
    name: str = ""


class OrganizeRequest(BaseModel):
    """Body of ``POST /organize``."""

    mode: Optional[str] = Field(None, pattern="^(copy|move)$")


# --------------------------------------------------------------------------
# session state
# --------------------------------------------------------------------------
class ScanSession:
    """Everything about the current run, safe to touch from many threads."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thumbs: Dict[Tuple[str, Tuple[int, int, int, int]], Optional[str]] = {}
        self.config: Dict[str, Any] = {}
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        with self._lock:
            self.state = STATE_IDLE
            self.phase = ""
            self.processed = 0
            self.total = 0
            self.current = ""
            self.error = ""
            self.clusters: List[FaceCluster] = []
            self.names: Dict[int, Optional[str]] = {}
            self.auto_labels: Dict[int, str] = {}
            self.scan_stats: Dict[str, Any] = {}
            self.results: Optional[Dict[str, Any]] = None
            self.scan_seconds = 0.0
            self._thumbs.clear()
        self._thread: Optional[threading.Thread] = None
        self._started_at = 0.0

    # -- scan bookkeeping --------------------------------------------------
    @property
    def scanning(self) -> bool:
        with self._lock:
            return self.state == STATE_SCANNING

    def begin(self, config: Dict[str, Any], total: int) -> None:
        with self._lock:
            self.config = dict(config)
            self.total = total
            self.processed = 0
            self.current = ""
            self.phase = "scanning photos"
            self.error = ""
            self.state = STATE_SCANNING
            self._started_at = time.perf_counter()

    def progress(self, done: int, total: int, path: Any) -> None:
        with self._lock:
            self.processed = done
            if total:
                self.total = total
            self.current = Path(path).name if path else ""

    def finish(self, clusters: List[FaceCluster], stats: Dict[str, Any],
               names: Dict[int, Optional[str]], auto: Dict[int, str]) -> None:
        with self._lock:
            self.clusters = clusters
            self.names = names
            self.auto_labels = auto
            self.scan_stats = stats
            self.scan_seconds = float(stats.get("seconds", 0.0))
            self.current = ""
            self.phase = "review the groups"
            self.state = STATE_READY

    def fail(self, message: str) -> None:
        with self._lock:
            self.state = STATE_ERROR
            self.error = message
            self.phase = ""
            self.current = ""

    def cluster(self, cluster_id: int) -> FaceCluster:
        with self._lock:
            for cluster in self.clusters:
                if cluster.cluster_id == cluster_id:
                    return cluster
        raise KeyError(cluster_id)

    def thumbs_for(self, cluster: FaceCluster) -> List[Dict[str, Any]]:
        """Serialize one cluster, rendering (and caching) face thumbnails."""
        faces: List[Dict[str, Any]] = []
        for record in cluster.faces:
            key = (str(record.image_path), tuple(record.bbox))
            if key not in self._thumbs:
                self._thumbs[key] = _thumb_data_url(record.image_path, record.bbox)
            faces.append({
                "photo": record.image_path.name,
                "path": str(record.image_path),
                "bbox": list(record.bbox),
                "score": round(float(record.score), 4),
                "thumb": self._thumbs[key],
            })
        return faces


# --------------------------------------------------------------------------
# application
# --------------------------------------------------------------------------
def create_app(config_path: Optional[Path] = None) -> FastAPI:
    """Build the FastAPI application and its single :class:`ScanSession`."""
    path = Path(config_path) if config_path else default_config_path()
    try:
        base_config = load_config(path)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.warning("Could not read %s (%s); falling back to defaults.", path, exc)
        base_config = {}

    app = FastAPI(
        title="FaceSort API",
        version=API_VERSION,
        description="Loopback-only API between the Electron UI and the "
                    "Python face pipeline. Never exposed to a network.",
    )
    # The renderer is loaded from file:// (origin "null") in dev and from the
    # app:// scheme when packaged, so allow any origin.  The server only ever
    # binds 127.0.0.1 and carries no authentication, which is safe here and
    # keeps the frontend free of CORS special-casing.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    session = ScanSession()
    app.state.session = session
    app.state.config_path = path

    # -- helpers -----------------------------------------------------------
    def public_config() -> Dict[str, Any]:
        config = dict(base_config)
        with session._lock:  # noqa: SLF001 - same module, single owner
            config.update(session.config)
        return config

    def default_scan_config(config_request: ScanRequest) -> Dict[str, Any]:
        return {
            "input_folder": config_request.input_folder,
            "output_folder": config_request.output_folder,
            "tolerance": float(config_request.tolerance),
            "min_faces_per_cluster": int(config_request.min_faces_per_cluster),
            "mode": config_request.mode,
            "unknown_folder": config_request.unknown_folder,
            "workers": int(config_request.workers),
            # Both are load-bearing: without use_db the session would fall
            # back to the default (True) and silently write the name DB.
            "use_db": bool(config_request.use_db),
            "names_db": base_config.get("names_db", "./facesort_names.db"),
        }

    # -- the scan thread ---------------------------------------------------
    def run_scan(config: Dict[str, Any]) -> None:
        """Worker thread body: detect, embed, cluster, auto-label."""
        try:
            stats = run_pipeline(config, progress_cb=session.progress)
            clusters = FaceClusterer.from_config(config).cluster(stats.records)

            tolerance = _as_float(config.get("tolerance", 0.5), 0.5)
            auto_labels: Dict[int, str] = {}
            use_db = bool(config.get("use_db", True))
            db = open_names_db(config) if use_db else None
            try:
                for cluster in clusters:
                    if cluster.centroid is None:
                        continue
                    match = db.find_match(cluster.centroid, tolerance) if db else None
                    if match is not None:
                        auto_labels[cluster.cluster_id] = match.person_name
            finally:
                if db is not None:
                    db.close()

            names = {
                cluster.cluster_id: auto_labels.get(cluster.cluster_id)
                for cluster in clusters
            }
            rate = (stats.images_scanned / stats.seconds) if stats.seconds else 0.0
            payload = {
                "photos": stats.images_scanned,
                "unreadable": stats.images_unreadable,
                "failed": stats.images_failed,
                "without_faces": stats.images_without_faces,
                "faces": stats.faces_detected,
                "embedded": stats.faces_embedded,
                "seconds": round(stats.seconds, 2),
                "img_per_s": round(rate, 3),
            }
            session.finish(clusters, payload, names, auto_labels)
            logger.info("Scan finished: %s photos, %s faces, %s cluster(s) "
                        "in %.1fs", stats.images_scanned, stats.faces_detected,
                        len(clusters), stats.seconds)
        except ModelsNotInstalledError as exc:
            logger.error("Models missing: %s", exc)
            session.fail(str(exc))
        except Exception as exc:  # noqa: BLE001 - report to the UI, never crash
            logger.error("Scan failed: %s\n%s", exc, traceback.format_exc())
            session.fail(f"{type(exc).__name__}: {exc}")

    # -- endpoints ---------------------------------------------------------
    @app.get("/status")
    def get_status() -> Dict[str, Any]:
        """Progress, configuration and environment for the whole UI."""
        with session._lock:
            named = sum(1 for value in session.names.values() if value)
            state = session.state
            body = {
                "service": SERVICE,
                "api_version": API_VERSION,
                "state": state,
                "phase": session.phase,
                "processed": session.processed,
                "total": session.total,
                "current": session.current,
                "error": session.error,
                "clusters": len(session.clusters),
                "named": named,
                "unknown": max(0, len(session.names) - named),
                "stats": dict(session.scan_stats),
                "results": session.results,
                "scanning": state == STATE_SCANNING,
            }
        elapsed = (
            time.perf_counter() - session._started_at
            if state == STATE_SCANNING and session._started_at else 0.0
        )
        body["elapsed"] = round(elapsed, 1)
        config = public_config()
        body["config"] = {
            "input_folder": config.get("input_folder", "./input_photos"),
            "output_folder": config.get("output_folder", "./grouped_photos"),
            "tolerance": _as_float(config.get("tolerance", 0.5), 0.5),
            "min_faces_per_cluster": _as_int(
                config.get("min_faces_per_cluster", 2), 2),
            "mode": config.get("mode", "copy"),
            "unknown_folder": config.get("unknown_folder", "_unknown"),
            "workers": _as_int(config.get("workers", 0), 0),
            "names_db": config.get("names_db", "./facesort_names.db"),
            # The renderer cannot stat a path itself, and on a fresh install
            # the configured defaults usually do not exist yet — so tell it
            # whether to prefill the fields or leave them empty.
            "input_exists": _is_dir(config.get("input_folder")),
            "output_exists": _is_dir(config.get("output_folder")),
        }
        body["models"] = {
            "name": MODEL_NAME,
            "installed": models_installed(),
            "dir": str(model_dir()),
        }
        body["runtime"] = {
            "frozen": bool(getattr(sys, "frozen", False)),
            "pid": os.getpid(),
            "python": sys.version.split()[0],
            "config_path": str(app.state.config_path),
        }
        return body

    @app.post("/scan")
    def start_scan(request: ScanRequest) -> Dict[str, Any]:
        """Validate the settings, then scan in the background."""
        if session.scanning:
            raise HTTPException(status_code=409,
                                detail="A scan is already running.")

        folder = Path(request.input_folder).expanduser()
        if not folder.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Input folder does not exist: {folder}")
        if not models_installed():
            raise HTTPException(
                status_code=400,
                detail=f"Face model {MODEL_NAME!r} is not available at "
                       f"{model_dir()}. Reinstall the app or run "
                       f"'python -m src.main --fetch-models'.")

        config = default_scan_config(request)
        total = len(list(scan_images(folder)))
        session.reset()
        session.begin(config, total)
        thread = threading.Thread(target=run_scan, args=(config,),
                                  name="faceorg-scan", daemon=True)
        thread.start()
        return {"started": True, "total": total}

    @app.get("/clusters")
    def get_clusters() -> Dict[str, Any]:
        """Every cluster with its faces and inline thumbnails."""
        with session._lock:
            if session.state in (STATE_IDLE, STATE_SCANNING):
                raise HTTPException(
                    status_code=409,
                    detail="No scan results yet — POST /scan first.")
            clusters = list(session.clusters)
            names = dict(session.names)
            auto = dict(session.auto_labels)

        payload: List[Dict[str, Any]] = []
        for cluster in clusters:
            faces = session.thumbs_for(cluster)
            name = names.get(cluster.cluster_id)
            payload.append({
                "id": cluster.cluster_id,
                "size": cluster.size,
                "photos": len(cluster.image_paths),
                "name": name or "",
                "auto": cluster.cluster_id in auto,
                "faces": faces,
            })
        return {"clusters": payload, "count": len(payload)}

    @app.post("/name_cluster")
    def name_cluster(request: NameRequest) -> Dict[str, Any]:
        """Name one cluster; an empty name sends it to the unknown folder."""
        try:
            cluster = session.cluster(request.cluster_id)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"No cluster #{request.cluster_id} in this scan.")

        name = normalize_person_name(request.name)
        with session._lock:
            session.names[cluster.cluster_id] = name
            session.auto_labels.pop(cluster.cluster_id, None)
            if name is None and session.state == STATE_DONE:
                session.state = STATE_READY  # allow re-running the organizer

        persisted = False
        if name and session.config.get("use_db", True):
            db = open_names_db(session.config)
            try:
                persisted = save_named_cluster(db, cluster, name)
            finally:
                if db is not None:
                    db.close()
        return {"ok": True, "cluster_id": cluster.cluster_id,
                "name": name or "", "remembered": persisted}

    @app.post("/organize")
    def organize(request: OrganizeRequest) -> Dict[str, Any]:
        """Sort every photo into ``output_folder/<person>/``."""
        with session._lock:
            if session.state in (STATE_IDLE, STATE_SCANNING):
                raise HTTPException(
                    status_code=409,
                    detail="Nothing to organize yet — POST /scan first.")
            config = dict(session.config)
            clusters = list(session.clusters)
            names = dict(session.names)
            state = session.state
        session.state = STATE_ORGANIZING

        if request.mode:
            config["mode"] = request.mode
        try:
            organizer = PhotoOrganizer.from_config(config)
            summary = organizer.organize(
                [(names.get(cluster.cluster_id), cluster)
                 for cluster in clusters])
        except (ValueError, OSError) as exc:
            session.state = state
            raise HTTPException(status_code=400,
                                detail=f"Could not organize: {exc}")
        except Exception as exc:  # noqa: BLE001
            session.state = state
            logger.error("Organize failed: %s\n%s", exc, traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

        results = {
            "ok": summary.ok,
            "mode": summary.mode,
            "output_folder": str(summary.output_folder),
            "folders": dict(summary.folders),
            "folders_created": [str(path) for path in summary.folders_created],
            "files_placed": summary.files_placed,
            "copies_written": summary.copies_written,
            "sources_removed": summary.sources_removed,
            "unknown_placed": summary.unknown_placed,
            "errors": [[str(path), reason] for path, reason in summary.errors],
        }
        session.state = STATE_DONE
        session.results = results
        session.phase = "finished"
        return results

    return app


# --------------------------------------------------------------------------
# serving: free port, PORT: handshake, uvicorn
# --------------------------------------------------------------------------
def format_port_line(port: int) -> str:
    """The one line of stdout the Electron main process waits for."""
    return f"{PORT_PREFIX}{port}"


def bind_local_socket(host: str = "127.0.0.1", port: int = 0) -> socket.socket:
    """Bind *and listen* on a free port.

    The socket is handed to uvicorn as-is, so there is no window in which
    another process could grab the port between choosing it and serving on
    it, and connections that arrive before the event loop starts simply sit
    in the accept backlog.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(128)
    return sock


def serve(host: str = "127.0.0.1", port: int = 0,
          config_path: Optional[Path] = None) -> int:
    """Run the API until interrupted. Returns a process exit code."""
    import uvicorn

    app = create_app(config_path)
    sock = bind_local_socket(host, port)
    bound_port = sock.getsockname()[1]

    # The handshake.  Nothing else may ever reach stdout.
    print(format_port_line(bound_port), flush=True)

    uvicorn_config = uvicorn.Config(
        app,
        host=host,
        port=bound_port,
        log_config=None,       # keep our stderr logging in charge
        access_log=False,
        log_level="info",
    )
    server = uvicorn.Server(uvicorn_config)
    try:
        server.run(sockets=[sock])
    except KeyboardInterrupt:  # pragma: no cover - interactive path
        pass
    finally:
        try:
            sock.close()
        except OSError:
            pass
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: ``backend.exe [--host H] [--port P] [--config C]``."""
    parser = argparse.ArgumentParser(
        prog="facesort-backend",
        description="Local API server for the FaceSort desktop app. "
                    "Binds 127.0.0.1 on a free port and prints "
                    "'PORT:<port>' on stdout.",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to bind (default: %(default)s)")
    parser.add_argument("--port", type=int, default=0,
                        help="port to bind; 0 = let the OS pick a free one")
    parser.add_argument("--config", type=Path, default=None,
                        help="path to config.yaml (default: bundled/repo)")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v for info, -vv for debug logging (stderr)")
    args = parser.parse_args(argv)

    level = logging.WARNING if args.verbose == 0 else (
        logging.INFO if args.verbose == 1 else logging.DEBUG)
    # stream=sys.stderr keeps stdout reserved for the PORT: handshake.
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning("Binding %s exposes the API beyond this machine; "
                       "127.0.0.1 is the supported configuration.", args.host)

    return serve(host=args.host, port=args.port, config_path=args.config)


if __name__ == "__main__":  # pragma: no cover - module entry
    import multiprocessing

    # Must precede everything: in a frozen build every pool worker re-launches
    # this executable and has to exit here instead of starting the server.
    multiprocessing.freeze_support()
    sys.exit(main(sys.argv[1:]))
