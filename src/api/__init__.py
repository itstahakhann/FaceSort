"""Local HTTP API layer for the desktop frontend (Electron).

The shipped desktop app is ``Electron (UI) + this package (engine)``:

* :mod:`src.api.server` exposes the pipeline as a loopback-only REST API and
  speaks ``PORT:<port>`` on stdout so the Electron main process can find it.
* Everything visual (window, styling, cluster grid) lives in ``electron/``.

The CLI (``python -m src.main``) keeps working unchanged and remains the
reference implementation of the pipeline.
"""

__all__ = ["server"]
