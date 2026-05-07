"""
Block Blast solver core.
- Auto-detects grid bounds, grid state, and the 3 pieces from a screenshot.
- Brute-force searches piece placements (with line-clear mechanic) for valid solutions.
- Renders an annotated step-by-step image of the chosen solution.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from itertools import permutations
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


# ---------- Color masks ----------

def _mode_color(pixels: np.ndarray, q: int = 16) -> np.ndarray:
    """Quantize pixels to bins of size `q` per channel and return the modal color."""
    if pixels.size == 0:
        return np.array([0, 0, 0], dtype=np.int32)
    p = pixels.astype(np.int32)
    qp = (p // q) * q
    keys = qp[:, 0] * (256 * 256) + qp[:, 1] * 256 + qp[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    mode_key = vals[int(np.argmax(counts))]
    r = (mode_key // (256 * 256)) & 0xFF
    g = (mode_key // 256) & 0xFF
    b = mode_key & 0xFF
    # Refine: take the mean of the pixels in the modal bin.
    in_bin = (qp[:, 0] == r) & (qp[:, 1] == g) & (qp[:, 2] == b)
    if in_bin.any():
        return p[in_bin].mean(axis=0).astype(np.int32)
    return np.array([r, g, b], dtype=np.int32)


def estimate_page_bg(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """Estimate the page background color (and a per-image tolerance) from image edges.

    Block Blast themes vary (blue, brown, pink, dark, ...). The play area is
    centered, so the *left and right edges* of the screenshot are almost
    always pure page background. We sample narrow vertical strips, take the
    modal color, and derive a tolerance from how much the sampled pixels
    deviate from that mode (small for solid backgrounds, larger for noisy
    or gradient backgrounds).
    """
    H, W = arr.shape[:2]
    edge_w = max(15, int(W * 0.04))
    y0 = max(20, int(H * 0.05))
    y1 = H - max(20, int(H * 0.08))
    if y1 <= y0:
        y0, y1 = 0, H
    left = arr[y0:y1, :edge_w]
    right = arr[y0:y1, W - edge_w:]
    pixels = np.concatenate([left.reshape(-1, 3), right.reshape(-1, 3)], axis=0)
    bg = _mode_color(pixels, q=16)
    # Tolerance: large enough to absorb edge antialiasing / JPEG noise,
    # small enough to flag a subtly-different grid background (e.g., dark
    # themes where grid and page differ by only ~30 in value).
    diffs = np.abs(pixels.astype(np.int32) - bg).max(axis=-1)
    tol = int(np.percentile(diffs, 98)) + 12
    tol = max(18, min(tol, 45))
    return bg, tol


def page_bg_mask(arr: np.ndarray, page_bg: np.ndarray, tol: int = 38) -> np.ndarray:
    """True where pixels are close to the sampled page background color."""
    diff = np.abs(arr.astype(np.int32) - page_bg).max(axis=-1)
    return diff < tol


def brick_mask(arr: np.ndarray) -> np.ndarray:
    """Bright + saturated colored brick (kept for back-compat / fallback uses)."""
    arr_f = arr.astype(np.float32) / 255.0
    r, g, b = arr_f[..., 0], arr_f[..., 1], arr_f[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    s = np.where(maxc > 1e-6, (maxc - minc) / np.maximum(maxc, 1e-5), 0.0)
    return (v > 0.55) & (s > 0.32)


def grid_bg_mask(arr: np.ndarray) -> np.ndarray:
    """Legacy blue grid-bg mask. Kept for back-compat; new pipeline doesn't rely on it."""
    r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
    return (r < 100) & (g < 120) & (b > 55) & (b < 150) & (b > r)


# ---------- Grid detection ----------

@dataclass
class GridBounds:
    top: int
    bottom: int
    left: int
    right: int
    rows: int
    cols: int

    @property
    def cell_h(self) -> float:
        return (self.bottom - self.top) / self.rows

    @property
    def cell_w(self) -> float:
        return (self.right - self.left) / self.cols


def detect_grid_bounds(arr: np.ndarray, grid_size: int = 8) -> GridBounds:
    """Find the playing grid's bounding box.

    Theme-agnostic strategy:
      1. Estimate the page background color from the image corners.
      2. Mark every pixel that's *not* the page background (this includes the
         grid's interior background and every brick).
      3. The grid is the largest connected blob of "not page background" in
         the upper portion of the image.
      4. Take that blob's bounding box.

    This works whether the page is blue, brown, pink, dark, etc.
    """
    H, W = arr.shape[:2]
    page_bg, tol = estimate_page_bg(arr)

    not_page = ~page_bg_mask(arr, page_bg, tol=tol)
    # Trim a status-bar strip at the very top and the lower portion (pieces / ads).
    not_page[:max(8, int(H * 0.025)), :] = False
    not_page[int(H * 0.78):, :] = False

    # Dilate so brick pixels merge with their surrounding grid background and
    # form a single big blob roughly equal to the grid rectangle.
    dilated = ndimage.binary_dilation(not_page, iterations=4)

    labeled, n = ndimage.label(dilated)
    if n == 0:
        raise ValueError("Could not detect grid in image.")

    sizes = ndimage.sum(dilated, labeled, range(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1

    coords = np.argwhere(labeled == biggest)
    ymin, xmin = coords.min(axis=0)
    ymax, xmax = coords.max(axis=0)

    # Sanity check: the grid blob should be roughly square-ish and big.
    bw, bh = xmax - xmin, ymax - ymin
    if bw < W * 0.4 or bh < H * 0.2:
        raise ValueError("Detected grid blob is too small; image may be cropped.")

    return GridBounds(top=int(ymin), bottom=int(ymax), left=int(xmin), right=int(xmax),
                      rows=grid_size, cols=grid_size)


def _estimate_grid_bg(arr: np.ndarray, gb: GridBounds) -> np.ndarray:
    """Estimate the empty-cell background color from inside the grid rectangle.

    The grid interior contains exactly two kinds of pixels: grid background
    (a single flat dark color) and bricks (various saturated colors). Even
    on densely-packed boards, the grid-bg color dominates the per-bin
    histogram because it's one color while bricks are split across many.
    We take the modal quantized color.
    """
    pad_y = max(2, int(gb.cell_h * 0.05))
    pad_x = max(2, int(gb.cell_w * 0.05))
    interior = arr[max(0, gb.top + pad_y):gb.bottom - pad_y + 1,
                   max(0, gb.left + pad_x):gb.right - pad_x + 1]
    if interior.size == 0:
        return np.array([0, 0, 0], dtype=np.int32)
    return _mode_color(interior.reshape(-1, 3), q=16)


def read_grid_state(arr: np.ndarray, gb: GridBounds) -> List[List[int]]:
    """Sample each cell center; mark filled if it differs from the grid background."""
    grid_bg = _estimate_grid_bg(arr, gb)
    grid: List[List[int]] = []
    sample = max(6, int(min(gb.cell_h, gb.cell_w) * 0.18))
    arr_i = arr.astype(np.int32)
    for r in range(gb.rows):
        row: List[int] = []
        for c in range(gb.cols):
            cy = int(gb.top + (r + 0.5) * gb.cell_h)
            cx = int(gb.left + (c + 0.5) * gb.cell_w)
            patch = arr_i[max(0, cy - sample):cy + sample,
                          max(0, cx - sample):cx + sample]
            if patch.size == 0:
                row.append(0); continue
            mean_color = patch.reshape(-1, 3).mean(axis=0)
            diff = float(np.max(np.abs(mean_color - grid_bg)))
            row.append(1 if diff > 35 else 0)
        grid.append(row)
    return grid


# ---------- Piece detection ----------

@dataclass
class Piece:
    name: str                         # "P1", "P2", "P3"
    cells: List[Tuple[int, int]]      # (row, col) offsets from top-left of bbox
    height: int
    width: int


def detect_pieces(arr: np.ndarray, gb: GridBounds) -> List[Piece]:
    """Find 1-3 piece bounding boxes below the grid, then read each piece's cell pattern.

    Block Blast hands you up to three pieces at a time. After you place one,
    the tray shows fewer until a new triple is generated, so detection must
    accept any count from 1 to 3 (not exactly 3).
    """
    H, W = arr.shape[:2]
    page_bg, tol = estimate_page_bg(arr)
    # "Brick" anywhere on the page = anything that isn't page background.
    # In the piece tray (no grid background to confuse us), this cleanly isolates
    # the colored shapes regardless of theme.
    not_page_full = ~page_bg_mask(arr, page_bg, tol=tol)

    # Restrict to the band immediately below the board. Pieces are at most
    # ~3 cells tall and sit close to the grid; capping by cell-size keeps ads,
    # banners, and home-bar UI out of the search.
    region_top = gb.bottom + max(15, int(gb.cell_h * 0.25))
    region_bot = min(
        H,
        gb.bottom + int(gb.cell_h * 5.5),
        gb.bottom + int((H - gb.bottom) * 0.7),
    )
    if region_bot <= region_top + 10:
        raise ValueError("No room below the board to detect pieces.")
    region = not_page_full[region_top:region_bot, :].copy()

    # Moderate dilation to merge bricks within each piece (don't merge across pieces).
    dilated = ndimage.binary_dilation(region, iterations=8)
    labeled, n = ndimage.label(dilated)

    min_blob = max(400, int((gb.cell_h * 0.5) ** 2))
    comps = []
    for i in range(1, n + 1):
        mask = (labeled == i)
        if mask.sum() < min_blob:
            continue
        ys, xs = np.where(mask)
        comps.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))

    if not comps:
        raise ValueError("No pieces found below the board.")

    # The piece tray sits closer to the grid than any ad / banner. Cluster
    # components by vertical position and keep only the topmost cluster.
    top_y = min((c[1] + c[3]) / 2 for c in comps)
    band_tol = max(gb.cell_h * 1.5, 60)
    piece_band = [c for c in comps if (c[1] + c[3]) / 2 - top_y < band_tol]

    # Keep at most 3 pieces. If we somehow detected more (rare; usually a
    # piece's halo split into two blobs), prefer the largest by area.
    if len(piece_band) > 3:
        piece_band.sort(key=lambda c: (c[2] - c[0]) * (c[3] - c[1]), reverse=True)
        piece_band = piece_band[:3]
    piece_band.sort(key=lambda c: c[0])  # left to right

    # Cell-size search: try a range bracketed by the grid's own cell size
    # (pieces in the tray are typically 0.4-0.85x the grid cell size) and pick
    # the size that makes every piece's bbox closest to an integer cell count.
    raw_dims = [(x1 - x0 + 1, y1 - y0 + 1) for (x0, y0, x1, y1) in piece_band]
    raw_dims = [(max(1, w - 14), max(1, h - 14)) for (w, h) in raw_dims]
    cs_min = max(12, int(gb.cell_h * 0.30))
    cs_max = max(cs_min + 1, int(gb.cell_h * 0.95))

    best_cs, best_err = None, float("inf")
    for cs in range(cs_min, cs_max + 1):
        err = 0.0
        ok = True
        for (w, h) in raw_dims:
            cw = w / cs
            ch = h / cs
            if cw < 0.7 or ch < 0.7 or cw > 5.5 or ch > 5.5:
                ok = False; break
            err += (cw - round(cw)) ** 2 + (ch - round(ch)) ** 2
        if ok and err < best_err:
            best_err, best_cs = err, cs
    if best_cs is None:
        raise ValueError("Could not determine piece cell size.")

    pieces: List[Piece] = []
    for idx, (x0, y0, x1, y1) in enumerate(piece_band):
        # Use original (un-dilated) not-page pixels inside this piece's bbox.
        ay0 = region_top + y0
        ay1 = region_top + y1
        sub = not_page_full[ay0:ay1 + 1, x0:x1 + 1]

        # Tight bbox around the true (un-dilated) piece pixels.
        ys, xs = np.where(sub)
        if len(ys) == 0:
            continue
        py0, py1 = ys.min(), ys.max()
        px0, px1 = xs.min(), xs.max()
        sub = sub[py0:py1 + 1, px0:px1 + 1]
        ph, pw = sub.shape

        rows = max(1, round(ph / best_cs))
        cols = max(1, round(pw / best_cs))

        cell_h_p = ph / rows
        cell_w_p = pw / cols

        cells: List[Tuple[int, int]] = []
        for r in range(rows):
            for c in range(cols):
                cy0 = int(r * cell_h_p + cell_h_p * 0.25)
                cy1 = int(r * cell_h_p + cell_h_p * 0.75)
                cx0 = int(c * cell_w_p + cell_w_p * 0.25)
                cx1 = int(c * cell_w_p + cell_w_p * 0.75)
                patch = sub[cy0:cy1 + 1, cx0:cx1 + 1]
                if patch.size and patch.mean() > 0.4:
                    cells.append((r, c))

        if not cells:
            continue

        pieces.append(Piece(name=f"P{len(pieces) + 1}", cells=cells,
                            height=rows, width=cols))

    if not pieces:
        raise ValueError("Detected piece bounds but couldn't read any cells.")

    return pieces


# ---------- Solver ----------

@dataclass
class Move:
    piece: Piece
    row: int
    col: int


@dataclass
class Solution:
    moves: List[Move]                     # in placement order (may be < len(pieces))
    line_clears: int
    final_board: List[List[int]]
    unplaced: List[Piece]                 # pieces that couldn't fit anywhere

    @property
    def all_placed(self) -> bool:
        return not self.unplaced


def _can_place(board: List[List[int]], piece: Piece, ar: int, ac: int) -> bool:
    n_rows, n_cols = len(board), len(board[0])
    for dr, dc in piece.cells:
        r, c = ar + dr, ac + dc
        if r < 0 or r >= n_rows or c < 0 or c >= n_cols:
            return False
        if board[r][c] == 1:
            return False
    return True


def _place(board: List[List[int]], piece: Piece, ar: int, ac: int) -> List[List[int]]:
    nb = [row[:] for row in board]
    for dr, dc in piece.cells:
        nb[ar + dr][ac + dc] = 1
    return nb


def _clear_lines(board: List[List[int]]) -> Tuple[List[List[int]], int]:
    rows, cols = len(board), len(board[0])
    full_r = [r for r in range(rows) if all(board[r][c] for c in range(cols))]
    full_c = [c for c in range(cols) if all(board[r][c] for r in range(rows))]
    if not full_r and not full_c:
        return board, 0
    nb = [row[:] for row in board]
    for r in full_r:
        for c in range(cols):
            nb[r][c] = 0
    for c in full_c:
        for r in range(rows):
            nb[r][c] = 0
    return nb, len(full_r) + len(full_c)


def solve(board: List[List[int]], pieces: List[Piece]) -> Optional[Solution]:
    """Return the best placement of as many pieces as possible.

    Tries every order and every placement, applies line clears between moves,
    and tracks the (most-pieces-placed, most-line-clears) maximum. Returns a
    Solution with `unplaced` listing any piece that couldn't fit. Returns None
    only when not even a single piece fits anywhere on the starting board.
    """
    if not pieces:
        return None
    rows, cols = len(board), len(board[0])

    best = {
        "placed": -1,
        "clears": -1,
        "moves": [],          # type: List[Move]
        "board": board,
        "unplaced": list(pieces),
    }

    def update_best(n_placed: int, total_clears: int, moves: List[Move],
                    curr_board: List[List[int]], remaining: List[Piece]) -> None:
        if (n_placed > best["placed"] or
                (n_placed == best["placed"] and total_clears > best["clears"])):
            best["placed"] = n_placed
            best["clears"] = total_clears
            best["moves"] = list(moves)
            best["board"] = curr_board
            best["unplaced"] = list(remaining)

    def recurse(curr_board: List[List[int]], remaining: List[Piece],
                placed_moves: List[Move], total_clears: int) -> None:
        update_best(len(placed_moves), total_clears, placed_moves, curr_board, remaining)
        if not remaining:
            return
        for i, piece in enumerate(remaining):
            new_remaining = remaining[:i] + remaining[i + 1:]
            for r in range(rows - piece.height + 1):
                for c in range(cols - piece.width + 1):
                    if not _can_place(curr_board, piece, r, c):
                        continue
                    nb = _place(curr_board, piece, r, c)
                    nb, clears = _clear_lines(nb)
                    placed_moves.append(Move(piece, r, c))
                    recurse(nb, new_remaining, placed_moves, total_clears + clears)
                    placed_moves.pop()

    recurse(board, list(pieces), [], 0)

    if best["placed"] <= 0:
        return None
    return Solution(
        moves=best["moves"],
        line_clears=best["clears"],
        final_board=best["board"],
        unplaced=best["unplaced"],
    )


# ---------- Rendering ----------

CELL_PX = 70
PAD = 8
GRID_BG = (45, 70, 100)
EMPTY_CELL = (60, 90, 125)
BRICK_RGB = (235, 95, 55)
HIGHLIGHT_COLORS = [
    (50, 180, 70),    # P1 green
    (240, 200, 50),   # P2 yellow
    (160, 80, 200),   # P3 purple
]
CLEAR_FLASH = (255, 230, 100)


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/Library/Fonts/Arial Bold.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_panel(board: List[List[int]],
                  highlights=None,
                  clear_cols: Optional[List[int]] = None,
                  clear_rows: Optional[List[int]] = None,
                  title: str = "") -> Image.Image:
    highlights = highlights or {}
    clear_cols = clear_cols or []
    clear_rows = clear_rows or []
    rows, cols = len(board), len(board[0])

    title_h = 48 if title else 0
    W = cols * CELL_PX + 2 * PAD
    H = rows * CELL_PX + 2 * PAD + title_h
    img = Image.new("RGB", (W, H), (200, 225, 240))
    d = ImageDraw.Draw(img)
    if title:
        d.text((PAD, 12), title, fill=(20, 20, 60), font=_font(22))

    d.rectangle([PAD, title_h + PAD, W - PAD, H - PAD], fill=GRID_BG)

    for r in range(rows):
        for c in range(cols):
            x0 = PAD + c * CELL_PX + 3
            y0 = title_h + PAD + r * CELL_PX + 3
            x1 = PAD + (c + 1) * CELL_PX - 3
            y1 = title_h + PAD + (r + 1) * CELL_PX - 3
            if c in clear_cols or r in clear_rows:
                d.rectangle([x0, y0, x1, y1], fill=CLEAR_FLASH,
                            outline=(255, 200, 0), width=3)
            elif (r, c) in highlights:
                d.rectangle([x0, y0, x1, y1], fill=highlights[(r, c)],
                            outline=(0, 0, 0), width=3)
            elif board[r][c]:
                d.rectangle([x0, y0, x1, y1], fill=BRICK_RGB,
                            outline=(180, 60, 30), width=2)
            else:
                d.rectangle([x0, y0, x1, y1], fill=EMPTY_CELL,
                            outline=(35, 55, 80), width=1)
    return img


def render_solution(initial_board: List[List[int]], solution: Solution) -> Image.Image:
    """Render a 2x3 grid of panels showing the solution step by step."""
    panels: List[Image.Image] = []
    panels.append(_render_panel(initial_board, title="Initial board"))

    board = [row[:] for row in initial_board]
    for idx, mv in enumerate(solution.moves):
        color = HIGHLIGHT_COLORS[idx % len(HIGHLIGHT_COLORS)]
        hl = {(mv.row + dr, mv.col + dc): color for dr, dc in mv.piece.cells}
        panels.append(_render_panel(
            board, highlights=hl,
            title=f"Step {idx + 1}: place {mv.piece.name} at row {mv.row}, col {mv.col}"
        ))
        board = _place(board, mv.piece, mv.row, mv.col)

        # Detect lines that will clear after this placement and show the flash.
        rows, cols = len(board), len(board[0])
        full_r = [r for r in range(rows) if all(board[r][c] for c in range(cols))]
        full_c = [c for c in range(cols) if all(board[r][c] for r in range(rows))]
        if full_r or full_c:
            label_parts = []
            if full_r: label_parts.append(f"row(s) {full_r}")
            if full_c: label_parts.append(f"col(s) {full_c}")
            panels.append(_render_panel(
                board, clear_rows=full_r, clear_cols=full_c,
                title=f"  ↳ {' & '.join(label_parts)} clear!"
            ))
            board, _ = _clear_lines(board)

    panels.append(_render_panel(board, title="Final board"))

    # Lay out panels in a grid with 3 columns.
    pw, ph = panels[0].size
    n_cols = 3
    n_rows = (len(panels) + n_cols - 1) // n_cols
    gap = 12
    out = Image.new("RGB",
                    (pw * n_cols + gap * (n_cols + 1),
                     ph * n_rows + gap * (n_rows + 1)),
                    (200, 225, 240))
    for i, panel in enumerate(panels):
        r, c = divmod(i, n_cols)
        out.paste(panel, (gap + c * (pw + gap), gap + r * (ph + gap)))
    return out


# ---------- Top-level convenience ----------

@dataclass
class Analysis:
    bounds: GridBounds
    board: List[List[int]]
    pieces: List[Piece]
    solution: Optional[Solution]


def analyze_image(pil_image: Image.Image, grid_size: int = 8) -> Analysis:
    arr = np.array(pil_image.convert("RGB"))
    bounds = detect_grid_bounds(arr, grid_size=grid_size)
    board = read_grid_state(arr, bounds)
    pieces = detect_pieces(arr, bounds)
    solution = solve(board, pieces)
    return Analysis(bounds=bounds, board=board, pieces=pieces, solution=solution)
