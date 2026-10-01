"""Backwards-compatible alias for :mod:`src.main`.

The real command-line entry point lives in ``src/main.py``; both work::

    python -m src.main
    python -m src.cli
"""

from .main import DEFAULT_CONFIG, PROMPT, build_parser, load_config, main

__all__ = ["DEFAULT_CONFIG", "PROMPT", "build_parser", "load_config", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
