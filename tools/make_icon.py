"""Generate electron/build/icon.ico — the desktop app icon (brand mark).

Kept as a script so the asset is reproducible instead of a mystery binary in
the repository. Run from the repository root:

    python -m tools.make_icon          # or: python tools/make_icon.py
"""
from pathlib import Path

from PIL import Image, ImageDraw

SIZES = [16, 24, 32, 48, 64, 128, 256]
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "electron" / "build" / "icon.ico"

BG_TOP = (79, 140, 255)      # --accent
BG_BOTTOM = (122, 92, 255)   # the violet used in the sidebar brand mark
FACE = (255, 255, 255)


def rounded_gradient(size: int) -> Image.Image:
    """Diagonal gradient tile with rounded corners (transparent outside)."""
    base = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    pixels = tile.load()
    for y in range(size):
        for x in range(size):
            t = (x + y) / (2.0 * (size - 1) or 1.0)
            pixels[x, y] = (
                int(BG_TOP[0] + (BG_BOTTOM[0] - BG_TOP[0]) * t),
                int(BG_TOP[1] + (BG_BOTTOM[1] - BG_TOP[1]) * t),
                int(BG_TOP[2] + (BG_BOTTOM[2] - BG_TOP[2]) * t),
                255,
            )
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size - 1, size - 1), radius=int(size * 0.22), fill=255)
    base.paste(tile, (0, 0), mask)
    return base


def draw_face(image: Image.Image) -> Image.Image:
    """Head + shoulders, matching the SVG mark used in the UI."""
    size = image.size[0]
    draw = ImageDraw.Draw(image)
    cx = size / 2.0
    head_r = size * 0.155
    head_cy = size * 0.40
    # shoulders: a wide arc
    shoulder_w = size * 0.34
    draw.ellipse(
        (cx - shoulder_w, size * 0.66, cx + shoulder_w, size * 1.30),
        fill=FACE,
    )
    draw.ellipse(
        (cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r),
        fill=FACE,
    )
    return image


def main() -> int:
    images = [draw_face(rounded_gradient(s)) for s in SIZES]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    images[-1].save(OUT, format="ICO",
                    sizes=[(s, s) for s in SIZES], append_images=images[:-1])
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} kB, sizes={SIZES})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
