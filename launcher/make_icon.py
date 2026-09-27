"""Draw the QuantPulse Terminal icons (original artwork: a market "pulse" line on a dark tile).

    .venv\\Scripts\\python.exe launcher\\make_icon.py

writes ``assets/quantpulse.ico`` (the app), ``assets/quantpulse-trading.ico`` (Trading Control: an amber
PAPER badge), ``assets/quantpulse-stop.ico`` (Stop: a red badge) and ``assets/quantpulse.png``. Every icon
holds 16-256 px images; the small ones are drawn with fewer, thicker strokes so they stay crisp.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)
SCALE = 8  # draw large, then downsample (antialiasing)

NAVY_TOP = (16, 38, 72)
NAVY_BOTTOM = (8, 20, 40)
GRID = (255, 255, 255, 34)
PULSE = (46, 230, 166)
GLOW = (46, 230, 166, 55)
CANDLE_UP = (46, 230, 166, 150)
CANDLE_DOWN = (255, 107, 107, 150)
AMBER = (255, 176, 32)
RED = (230, 57, 70)
WHITE = (255, 255, 255)

# the pulse: flat, a heartbeat-like spike, then a climb (x, y in 0..1, y downwards)
PULSE_POINTS = [
    (0.14, 0.64),
    (0.30, 0.64),
    (0.37, 0.48),
    (0.44, 0.78),
    (0.52, 0.34),
    (0.60, 0.56),
    (0.70, 0.44),
    (0.86, 0.22),
]
CANDLES = [  # x, open, close, high, low (0..1)
    (0.24, 0.56, 0.48, 0.44, 0.60),
    (0.40, 0.62, 0.54, 0.50, 0.66),
    (0.56, 0.52, 0.60, 0.46, 0.64),
    (0.72, 0.46, 0.36, 0.32, 0.50),
]


def _gradient(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size))
    draw = ImageDraw.Draw(img)
    for y in range(size):
        t = y / max(size - 1, 1)
        color = tuple(round(a + (b - a) * t) for a, b in zip(NAVY_TOP, NAVY_BOTTOM, strict=True))
        draw.line([(0, y), (size, y)], fill=(*color, 255))
    return img


def _tile(px: int, detail: bool) -> Image.Image:
    """One icon image of ``px`` pixels."""
    size = px * SCALE
    radius = round(size * (0.22 if px >= 32 else 0.18))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    tile.paste(_gradient(size), (0, 0), mask)

    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if detail:  # faint grid and candlesticks behind the pulse
        for y in (0.30, 0.50, 0.70):
            draw.line(
                [(size * 0.12, size * y), (size * 0.88, size * y)], fill=GRID, width=max(SCALE, size // 160)
            )
        body = size * 0.045
        for x, o, c, hi, lo in CANDLES:
            color = CANDLE_UP if c < o else CANDLE_DOWN
            cx = size * x
            draw.line([(cx, size * hi), (cx, size * lo)], fill=color, width=max(SCALE, size // 110))
            top, bottom = sorted((size * o, size * c))
            draw.rectangle([cx - body / 2, top, cx + body / 2, bottom], fill=color)
    points = [
        (size * x, size * y) for x, y in (PULSE_POINTS if detail else [*PULSE_POINTS[::2], PULSE_POINTS[-1]])
    ]
    width = round(size * (0.055 if detail else 0.10))
    draw.line(points, fill=GLOW, width=round(width * 2.2), joint="curve")
    draw.line(points, fill=PULSE, width=width, joint="curve")
    end_x, end_y = points[-1]
    dot = width * (1.1 if detail else 0.9)
    draw.ellipse([end_x - dot, end_y - dot, end_x + dot, end_y + dot], fill=WHITE)
    tile = Image.alpha_composite(tile, layer)
    return tile


def _badge(img: Image.Image, color: tuple[int, int, int], kind: str) -> Image.Image:
    size = img.width
    out = img.copy()
    draw = ImageDraw.Draw(out)
    r = size * 0.21
    cx, cy = size - r - size * 0.02, size - r - size * 0.02
    draw.ellipse(
        [cx - r - size * 0.025, cy - r - size * 0.025, cx + r + size * 0.025, cy + r + size * 0.025],
        fill=NAVY_BOTTOM,
    )
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    if kind == "stop":
        s = r * 0.42
        draw.rectangle([cx - s, cy - s, cx + s, cy + s], fill=WHITE)
    else:  # a shield: controls, no automation
        w, h = r * 0.55, r * 0.68
        draw.polygon(
            [
                (cx - w, cy - h * 0.7),
                (cx, cy - h),
                (cx + w, cy - h * 0.7),
                (cx + w * 0.8, cy + h * 0.3),
                (cx, cy + h),
                (cx - w * 0.8, cy + h * 0.3),
            ],
            fill=WHITE,
        )
    return out


def render(badge: tuple[tuple[int, int, int], str] | None = None) -> list[Image.Image]:
    images = []
    for px in SIZES:
        big = _tile(px, detail=px >= 48)
        if badge is not None:
            big = _badge(big, *badge)
        images.append(big.resize((px, px), Image.Resampling.LANCZOS))
    return images


def save_ico(path: Path, images: list[Image.Image]) -> None:
    largest = images[-1]
    largest.save(path, format="ICO", sizes=[im.size for im in images], append_images=images[:-1])


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    app = render()
    save_ico(ASSETS / "quantpulse.ico", app)
    app[-1].save(ASSETS / "quantpulse.png")
    save_ico(ASSETS / "quantpulse-trading.ico", render((AMBER, "shield")))
    save_ico(ASSETS / "quantpulse-stop.ico", render((RED, "stop")))
    print(f"icons written to {ASSETS}")


if __name__ == "__main__":
    main()
