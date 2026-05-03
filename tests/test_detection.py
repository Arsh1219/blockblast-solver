"""Synthetic Block Blast screenshot generator + detection tests.

Run from repo root:
    python3 tests/test_detection.py

Generates synthetic screenshots in multiple themes (blue, brown, pink, dark)
and asserts that solver.detect_grid_bounds, solver.read_grid_state, and
solver.detect_pieces produce the expected output. This guards against
theme-specific regressions like the blue-only color masks that previously
failed on brown screenshots.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import solver  # noqa: E402


# A handful of typical brick colors used across Block Blast themes.
BRICK_PALETTE = [
    (235, 95, 55),    # orange
    (240, 200, 60),   # yellow
    (90, 180, 80),    # green
    (160, 80, 200),   # purple
    (220, 80, 80),    # red
    (60, 160, 220),   # sky blue
    (90, 210, 220),   # cyan
    (235, 110, 170),  # pink
]


def _draw_brick(d: ImageDraw.ImageDraw, x0: int, y0: int, cs: int,
                color: tuple[int, int, int]) -> None:
    inset = max(2, cs // 14)
    d.rounded_rectangle(
        [x0 + inset, y0 + inset, x0 + cs - inset, y0 + cs - inset],
        radius=max(3, cs // 8), fill=color, outline=(255, 255, 255, 80), width=2,
    )


def make_screenshot(
    theme: str,
    board: list[list[int]],
    pieces: list[list[list[int]]],
    cell: int = 80,
    seed: int = 0,
) -> tuple[Image.Image, dict]:
    """Render a synthetic Block Blast screenshot for the given theme."""
    themes = {
        "blue":  {"page": (35, 90, 165),  "grid": (25, 55, 100)},
        "brown": {"page": (190, 170, 145), "grid": (75, 55, 40)},
        "pink":  {"page": (240, 180, 200), "grid": (110, 40, 80)},
        "dark":  {"page": (30, 30, 35),    "grid": (60, 60, 70)},
    }
    if theme not in themes:
        raise ValueError(f"Unknown theme: {theme}")
    page = themes[theme]["page"]
    grid = themes[theme]["grid"]

    rng = np.random.default_rng(seed)
    n_rows = n_cols = len(board)

    pad = cell  # outer page padding
    grid_pad = cell // 4  # padding between grid edge and cells
    grid_w = n_cols * cell + 2 * grid_pad
    grid_h = n_rows * cell + 2 * grid_pad

    # Real Block Blast renders pieces smaller than board cells.
    piece_cell = max(20, int(cell * 0.55))
    status_h = max(20, int(cell * 0.4))
    # Layout: status, grid, gap, pieces row, ad gap, ad strip, bottom pad
    piece_band = int(cell * 2.0)
    ad_band = int(cell * 0.6)
    ad_gap = int(cell * 0.8)  # clear vertical gap between pieces and ad
    img_w = grid_w + 2 * pad
    img_h = (status_h + pad + grid_h + cell // 2 + piece_band
             + ad_gap + ad_band + pad)

    img = Image.new("RGB", (img_w, img_h), page)
    d = ImageDraw.Draw(img)

    # Status-bar / score noise at the top so corners-only sampling would fail.
    d.rectangle([0, 0, img_w, status_h],
                fill=(int(page[0] * 0.7), int(page[1] * 0.7), int(page[2] * 0.7)))
    d.text((pad, max(4, int(cell * 0.1))), "12345", fill=(255, 255, 255))

    # An "ad" banner near the bottom in a vivid color (challenges piece detection).
    ad_y = img_h - pad - ad_band
    d.rectangle([pad // 2, ad_y, img_w - pad // 2, ad_y + ad_band],
                fill=(245, 130, 30))

    # Grid rectangle (rounded)
    grid_left = pad
    grid_top = status_h + pad
    d.rounded_rectangle(
        [grid_left, grid_top, grid_left + grid_w, grid_top + grid_h],
        radius=cell // 4, fill=grid,
    )

    # Cells
    for r in range(n_rows):
        for c in range(n_cols):
            if board[r][c]:
                color = BRICK_PALETTE[(r * 7 + c * 3 + rng.integers(0, 8)) % len(BRICK_PALETTE)]
                x0 = grid_left + grid_pad + c * cell
                y0 = grid_top + grid_pad + r * cell
                _draw_brick(d, x0, y0, cell, color)

    # Pieces row (using piece_cell, smaller than board cell)
    piece_y = grid_top + grid_h + cell // 2
    n_pieces = len(pieces)
    band_w = img_w - 2 * pad
    slot_w = band_w // n_pieces
    piece_layouts = []
    for idx, p in enumerate(pieces):
        ph = len(p) * piece_cell
        pw = max(len(row) for row in p) * piece_cell
        slot_x = pad + idx * slot_w
        ox = slot_x + (slot_w - pw) // 2
        oy = piece_y + (piece_band - ph) // 2
        for r, row in enumerate(p):
            for c, v in enumerate(row):
                if v:
                    color = BRICK_PALETTE[(idx * 5 + r * 3 + c) % len(BRICK_PALETTE)]
                    _draw_brick(d, ox + c * piece_cell, oy + r * piece_cell,
                                piece_cell, color)
        piece_layouts.append((ox, oy, pw, ph))

    return img, {"page": page, "grid": grid, "cell": cell, "piece_cell": piece_cell,
                 "grid_bbox": (grid_left, grid_top, grid_w, grid_h),
                 "piece_layouts": piece_layouts}


def assert_eq(name, actual, expected):
    if actual != expected:
        raise AssertionError(f"{name}\n  expected: {expected}\n  actual:   {actual}")


def run_theme(theme: str) -> None:
    print(f"\n--- theme: {theme} ---")
    # 8x8 board with a varied pattern: each row has a different fill count.
    board = [
        [0, 0, 1, 1, 1, 0, 1, 1],
        [0, 0, 1, 1, 1, 0, 1, 1],
        [0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 1, 0],
    ]
    # P1: L (2x2 with corner missing), P2: 1x4 horizontal, P3: 3x3 square
    pieces_input = [
        [[1, 0],
         [1, 0],
         [1, 1]],                # 3x2 L
        [[1, 1, 1, 1]],           # 1x4
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],             # 3x3
    ]

    img, meta = make_screenshot(theme, board, pieces_input)
    img.save(f"/tmp/synth_{theme}.png")
    arr = np.array(img)

    bounds = solver.detect_grid_bounds(arr, grid_size=8)
    print(f"  bounds: top={bounds.top} bottom={bounds.bottom} "
          f"left={bounds.left} right={bounds.right} "
          f"cell≈{bounds.cell_w:.1f}x{bounds.cell_h:.1f}")

    # Bounds should roughly match the rendered grid rectangle (allow some tolerance).
    gx, gy, gw, gh = meta["grid_bbox"]
    assert abs(bounds.left - gx) < gw * 0.1, f"left off: {bounds.left} vs {gx}"
    assert abs(bounds.top - gy) < gh * 0.1, f"top off: {bounds.top} vs {gy}"
    assert abs(bounds.right - (gx + gw)) < gw * 0.1, f"right off: {bounds.right} vs {gx+gw}"
    assert abs(bounds.bottom - (gy + gh)) < gh * 0.1, f"bottom off: {bounds.bottom} vs {gy+gh}"

    detected_board = solver.read_grid_state(arr, bounds)
    if detected_board != board:
        for r in range(8):
            print(f"    row {r}: got {detected_board[r]} want {board[r]}")
        raise AssertionError(f"[{theme}] board state mismatch")
    print("  board: OK")

    pieces = solver.detect_pieces(arr, bounds)
    assert_eq(f"[{theme}] piece count", len(pieces), 3)

    # Check each detected piece's shape (rows x cols and cell list).
    expected_shapes = [
        # (rows, cols, sorted cells)
        (3, 2, sorted([(0, 0), (1, 0), (2, 0), (2, 1)])),
        (1, 4, sorted([(0, 0), (0, 1), (0, 2), (0, 3)])),
        (3, 3, sorted([(r, c) for r in range(3) for c in range(3)])),
    ]
    for i, (p, (er, ec, ecells)) in enumerate(zip(pieces, expected_shapes)):
        got_cells = sorted(p.cells)
        if (p.height, p.width, got_cells) != (er, ec, ecells):
            raise AssertionError(
                f"[{theme}] piece {i+1} mismatch:\n"
                f"  expected: {er}x{ec} {ecells}\n"
                f"  actual:   {p.height}x{p.width} {got_cells}"
            )
    print(f"  pieces: OK ({[f'{p.height}x{p.width}' for p in pieces]})")


def run_density(theme: str, fill_pct: float) -> None:
    print(f"\n--- {theme}, fill={fill_pct:.0%} ---")
    rng = np.random.default_rng(int(fill_pct * 100))
    board = [[1 if rng.random() < fill_pct else 0 for _ in range(8)] for _ in range(8)]
    pieces_input = [
        [[1, 1], [1, 0]],
        [[1, 1, 1]],
        [[1], [1], [1]],
    ]
    img, _ = make_screenshot(theme, board, pieces_input)
    arr = np.array(img)
    bounds = solver.detect_grid_bounds(arr, grid_size=8)
    detected_board = solver.read_grid_state(arr, bounds)
    if detected_board != board:
        bad = sum(1 for r in range(8) for c in range(8)
                  if detected_board[r][c] != board[r][c])
        raise AssertionError(f"density {fill_pct:.0%} on {theme}: {bad} cells wrong")
    pieces = solver.detect_pieces(arr, bounds)
    if len(pieces) != 3:
        raise AssertionError(f"density {fill_pct:.0%} on {theme}: piece count {len(pieces)}")
    print(f"  density {fill_pct:.0%}: OK")


def run_grid_size(theme: str, n: int) -> None:
    print(f"\n--- {theme}, grid={n}x{n} ---")
    board = [[(r + c) % 3 == 0 for c in range(n)] for r in range(n)]
    pieces_input = [
        [[1, 1, 1]],
        [[1, 0], [1, 1]],
        [[1], [1]],
    ]
    img, _ = make_screenshot(theme, board, pieces_input, cell=70)
    arr = np.array(img)
    bounds = solver.detect_grid_bounds(arr, grid_size=n)
    detected_board = solver.read_grid_state(arr, bounds)
    if detected_board != board:
        raise AssertionError(f"grid {n}x{n} on {theme}: board mismatch")
    pieces = solver.detect_pieces(arr, bounds)
    if len(pieces) != 3:
        raise AssertionError(f"grid {n}x{n} on {theme}: piece count {len(pieces)}")
    print(f"  grid {n}x{n}: OK")


def main() -> int:
    failures = []

    # Theme sweep
    for theme in ("blue", "brown", "pink", "dark"):
        try:
            run_theme(theme)
        except AssertionError as e:
            failures.append((f"theme:{theme}", str(e)))
            print(f"  FAILED: {e}")

    # Density sweep
    for theme in ("blue", "brown"):
        for pct in (0.1, 0.3, 0.5, 0.7):
            try:
                run_density(theme, pct)
            except AssertionError as e:
                failures.append((f"density:{theme}:{pct}", str(e)))
                print(f"  FAILED: {e}")

    # Grid-size sweep
    for theme in ("blue", "brown"):
        for n in (8, 9, 10):
            try:
                run_grid_size(theme, n)
            except AssertionError as e:
                failures.append((f"grid:{theme}:{n}", str(e)))
                print(f"  FAILED: {e}")

    print()
    if failures:
        print(f"{len(failures)} test(s) failed:")
        for tag, msg in failures:
            print(f"  - {tag}: {msg.splitlines()[0]}")
        return 1
    print(f"All detection tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
