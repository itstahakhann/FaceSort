"""Shared insightface (buffalo_l) model management — CPU only.

Offline guarantee
-----------------
``get_face_analysis()`` only ever builds a model from files that are
**already on disk**.  If the local model directory is missing it raises
:class:`ModelsNotInstalledError` instead of letting insightface download
anything behind the user's back.  :func:`download_models()` is the single,
explicitly called function that may touch the network (one-time weight
fetch), so running the pipeline itself never performs a network call.

``src.detector`` and ``src.embedder`` share one lazily created
``FaceAnalysis`` instance — loading the ONNX models twice would double both
the memory footprint and the start-up time.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "buffalo_l"
CPU_PROVIDERS = ["CPUExecutionProvider"]
DEFAULT_DET_THRESH = 0.5
DEFAULT_DET_SIZE = (640, 640)

_MANIFEST_FILE = "manifest.json"

__all__ = [
    "MODEL_NAME",
    "CPU_PROVIDERS",
    "ModelsNotInstalledError",
    "model_root",
    "model_dir",
    "models_installed",
    "require_models",
    "download_models",
    "get_face_analysis",
    "set_intra_op_threads",
    "reset_face_analysis",
    "make_face",
]


class ModelsNotInstalledError(RuntimeError):
    """The buffalo_l weights are not present in the local model cache."""


def model_root() -> Path:
    """insightface-style root that holds ``models/buffalo_l/``.

    Resolution order:

    1. ``FACEORG_MODEL_ROOT`` — set by the Electron main process so a
       packaged app points the backend at ``<resources>/models`` (the ONNX
       files shipped through electron-builder's ``extraResources``).
    2. ``INSIGHTFACE_HOME`` — insightface's own override, unchanged.
    3. ``sys._MEIPASS`` in a frozen build, where the spec bundles the
       weights under ``models/buffalo_l`` so ``backend.exe`` is standalone.
    4. ``~/.insightface`` — the normal dev-checkout location.
    """
    env = os.environ.get("FACEORG_MODEL_ROOT")
    if env:
        return Path(env).expanduser()
    env = os.environ.get("INSIGHTFACE_HOME")
    if env:
        return Path(env).expanduser()
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:  # frozen executable: weights sit next to the bundled code
        return Path(meipass)
    return Path("~").expanduser() / ".insightface"


def model_dir() -> Path:
    """Directory that holds the buffalo_l model files."""
    return model_root() / "models" / MODEL_NAME


def models_installed() -> bool:
    """True when the buffalo_l weights are already on disk."""
    directory = model_dir()
    if not directory.is_dir():
        return False
    # buffalo_l ships raw .onnx files; newer insightface releases may ship a
    # manifest.json package instead.  Non-recursive, mirroring the way
    # FaceAnalysis itself discovers the models.
    return any(directory.glob("*.onnx")) or (directory / _MANIFEST_FILE).is_file()


def require_models() -> None:
    """Raise :class:`ModelsNotInstalledError` unless buffalo_l is on disk.

    The parallel (multiprocess) pipeline calls this *before* spawning any
    worker so a missing model fails fast with one clear message instead of
    one warning per photo.
    """
    if not models_installed():
        raise ModelsNotInstalledError(
            f"Face model {MODEL_NAME!r} is not installed at {model_dir()}.\n"
            "Download it once with:  python -m src.main --fetch-models\n"
            "(packaged builds ship the weights; point FACEORG_MODEL_ROOT at\n"
            "the folder that contains models/buffalo_l if they were moved.)\n"
            "Inference itself never downloads anything (fully offline)."
        )


def download_models(force: bool = False) -> Path:
    """Fetch the buffalo_l weights into the local cache.

    This is the only function in this package that performs a network call.
    Call it once up front (``python -m src.cli --fetch-models``) so that
    subsequent runs are guaranteed to be fully offline.
    """
    try:
        from insightface.utils.storage import download, ensure_available
    except ImportError as exc:  # pragma: no cover - missing dependency
        raise RuntimeError(
            "insightface is not installed. Run: pip install -r requirements.txt"
        ) from exc

    logger.info("Fetching %r weights into %s ...", MODEL_NAME, model_root())
    if force:
        path = download("models", MODEL_NAME, force=True, root=str(model_root()))
    else:
        # No-op when the model is already cached locally.
        path = ensure_available("models", MODEL_NAME, root=str(model_root()))
    if not models_installed():
        raise RuntimeError(
            f"Download finished but {model_dir()} does not look like a valid "
            f"{MODEL_NAME} model directory."
        )
    logger.info("Model ready: %s", path)
    return Path(path)


_analysis: Optional[Any] = None
_build_signature: Optional[Tuple[Any, ...]] = None
_intra_op_threads: Optional[int] = None


def set_intra_op_threads(threads: Optional[int]) -> None:
    """Cap ONNX Runtime's intra-op thread pool for this process.

    The default (``None``) lets each session use every core, which is right
    for single-process runs.  Worker processes set this to a small number
    (e.g. ``cpu_count // workers``) *before* the first
    :func:`get_face_analysis` call so N pool workers do not oversubscribe
    the CPU.  Has no effect once the models are loaded.
    """
    global _intra_op_threads
    if _analysis is not None:
        logger.debug(
            "FaceAnalysis already loaded; intra-op threads stay at their "
            "current setting."
        )
        return
    _intra_op_threads = None if threads is None else max(1, int(threads))


def get_face_analysis(
    det_thresh: float = DEFAULT_DET_THRESH,
    det_size: Tuple[int, int] = DEFAULT_DET_SIZE,
) -> Any:
    """Return the process-wide CPU-only ``FaceAnalysis`` instance.

    The first call loads the ONNX models from the local cache; later calls
    return the cached instance (any changed detection parameters are ignored
    with a debug log, since the models are already prepared).
    """
    global _analysis, _build_signature

    signature = (float(det_thresh), tuple(int(v) for v in det_size))
    if _analysis is not None:
        if signature != _build_signature:
            logger.debug(
                "FaceAnalysis already initialised with %s; ignoring %s.",
                _build_signature,
                signature,
            )
        return _analysis

    if not models_installed():
        require_models()

    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:  # pragma: no cover - missing dependency
        raise RuntimeError(
            "insightface is not installed. Run: pip install -r requirements.txt"
        ) from exc

    sess_options = None
    if _intra_op_threads is not None:
        # insightface forwards ``sess_options`` to every ONNX session it
        # creates (det + recognition), and SCRFD's derived resolution
        # sessions inherit them from the reference session.
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = int(_intra_op_threads)

    logger.info(
        "Loading %s models (CPU) from %s%s ...",
        MODEL_NAME,
        model_dir(),
        f" [intra-op threads: {_intra_op_threads}]"
        if _intra_op_threads is not None else "",
    )
    analysis = FaceAnalysis(
        # Passing an existing directory bypasses insightface's implicit
        # download path entirely — no code branch can reach the network here.
        name=str(model_dir()),
        root=str(model_root()),
        providers=list(CPU_PROVIDERS),
        sess_options=sess_options,
    )
    # ctx_id=-1 selects the CPU in insightface; providers are pinned above.
    analysis.prepare(
        ctx_id=-1,
        det_thresh=float(det_thresh),
        det_size=tuple(int(v) for v in det_size),
    )

    _analysis = analysis
    _build_signature = signature
    return analysis


def reset_face_analysis() -> None:
    """Drop the cached instance (used by tests and long-lived processes)."""
    global _analysis, _build_signature
    _analysis = None
    _build_signature = None


def make_face(bbox, kps, det_score: float) -> Any:
    """Build an insightface ``Face`` from our :class:`DetectedFace` data."""
    from insightface.app.common import Face

    return Face(
        bbox=np.asarray(bbox, dtype=np.float32),
        kps=None if kps is None else np.asarray(kps, dtype=np.float32),
        det_score=float(det_score),
    )
