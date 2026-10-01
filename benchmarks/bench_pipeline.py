"""Micro-benchmark: pipeline throughput in images/sec (SPEC §6: >= 5 img/s CPU).

Builds a scratch folder of 1080p JPEGs (SPEC §6 wording) from any source
folder of real photos, then times :func:`src.main.run_pipeline` and reports
the rate — sequentially (``--workers 1``) or with multiprocessing.

Usage::

    python benchmarks/bench_pipeline.py --source ./photos --count 60
    python benchmarks/bench_pipeline.py --source ./photos --count 300 --workers 4
    python benchmarks/bench_pipeline.py --check          # exit 1 if < target

The scratch folder lives in the system temp dir and is reused between runs
(rebuilt when the requested count or source changes).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

TARGET_IMG_PER_SEC = 5.0  # SPEC §6
CANVAS = (1920, 1080)     # "1080p photos"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_sources(source: Path):
    paths = [p for p in sorted(source.rglob("*"))
             if p.is_file() and p.suffix.lower() in EXTENSIONS
             and p.stat().st_size > 0]
    return paths


def build_bench_folder(source: Path, folder: Path, count: int) -> int:
    """Create ``count`` 1080p JPEGs (real faces, centred on a grey canvas)."""
    marker = folder / f".bench_{source.name}_{count}"
    if folder.is_dir() and marker.exists():
        return len(list(folder.glob("bench_*.jpg")))

    sources = collect_sources(source)
    if not sources:
        raise SystemExit(f"no usable photos found in {source}")

    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)

    canvases = []
    for src in sources:
        try:
            with Image.open(src) as im:
                im = im.convert("RGB")
                im.thumbnail(CANVAS)  # keep aspect, fit inside 1920x1080
                canvas = Image.new("RGB", CANVAS, (128, 128, 128))
                canvas.paste(im, ((CANVAS[0] - im.width) // 2,
                                  (CANVAS[1] - im.height) // 2))
                canvases.append(canvas)
        except Exception as exc:  # a bad source photo must not kill the bench
            print(f"  (skipping {src.name}: {exc})")
    if not canvases:
        raise SystemExit(f"could not render any usable photo from {source}")

    for index in range(count):
        canvases[index % len(canvases)].save(
            folder / f"bench_{index:05d}.jpg", "JPEG", quality=85
        )
    marker.touch()
    return count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True,
                        help="folder of real photos to build the bench set from")
    parser.add_argument("--count", type=int, default=60,
                        help="number of 1080p photos (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=1,
                        help="pipeline workers; 0 = auto (default: %(default)s)")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 when the rate is below the SPEC target")
    parser.add_argument("--keep", action="store_true",
                        help="keep the scratch folder afterwards")
    args = parser.parse_args(argv)

    from src.main import run_pipeline

    folder = Path(tempfile.gettempdir()) / "faceorg_bench" / "photos"
    built = build_bench_folder(args.source.expanduser().resolve(), folder,
                               args.count)
    print(f"bench folder: {folder} ({built} photos of {CANVAS[0]}x{CANVAS[1]})")

    config = {
        "input_folder": str(folder),
        "workers": int(args.workers),
        "tolerance": 0.5,
        "min_faces_per_cluster": 2,
    }
    started = time.perf_counter()
    stats = run_pipeline(config)
    wall = time.perf_counter() - started

    rate = stats.images_scanned / stats.seconds if stats.seconds else 0.0
    wall_rate = stats.images_scanned / wall if wall else 0.0
    print(f"images      : {stats.images_scanned} scanned, "
          f"{stats.images_unreadable} unreadable, {stats.images_failed} failed")
    print(f"faces       : {stats.faces_detected} detected, "
          f"{stats.faces_embedded} embedded")
    print(f"pipeline    : {stats.seconds:.1f}s  -> {rate:.2f} img/s "
          f"(wall {wall:.1f}s incl. scan/startup -> {wall_rate:.2f} img/s)")
    ok = rate >= TARGET_IMG_PER_SEC
    print(f"SPEC §6 target ({TARGET_IMG_PER_SEC:.0f} img/s): "
          f"{'PASS' if ok else 'FAIL'}")

    if not args.keep:
        shutil.rmtree(folder, ignore_errors=True)
    if args.check and not ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
