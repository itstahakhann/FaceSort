"""PyInstaller entry point for the Python backend (``backend.exe``).

Electron's main process spawns this executable and waits for a
``PORT:<port>`` line on stdout (see :mod:`src.api.server`), so the entry
point stays deliberately tiny.

``multiprocessing.freeze_support()`` must be the *first* statement: in a
frozen build every pool worker (M6 multiprocessing) re-launches the
executable with ``--multiprocessing-fork`` on the command line.  Calling
``freeze_support()`` first makes those re-launches hand control to the
worker instead of starting a second server — and it is why the import of
``src.api.server`` below can never execute inside a worker.
"""

import multiprocessing
import sys

multiprocessing.freeze_support()

from src.api.server import main  # noqa: E402  (must follow freeze_support)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
