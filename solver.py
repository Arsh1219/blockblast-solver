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

def brick_mask(arr: np.ndarray) -> np.ndarray:
    """Orange/red bricks: high R, mid G, low B."""
    r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
    return (r > 170) & (g > 40) & (g < 190) & (b < 110) & (r > b + 60)


def grid_bg_mask(arr: np.ndarray) -> np.ndarray:
    """Dark blue-gray grid background."""
    r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
    return (r < 90) & (g < 110) & (b > 60) & (b < 140) & (b > r)


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
    Strategy: union of grid-bg and brick pixels, take the largest connected component
    in the upper portion of the image, take its bounding box.
    """
    H, W = arr.shape[:2]
    combined = grid_bg_mask(arr) | brick_mask(arr)

    # Restrict to upper 75% to avoid the pieces region.
    upper = combined.copy()
    upper[int(H * 0.75):, :] = False

    # Dilate to merge small gaps.
    upper = ndimage.binary_dilation(upper, iterations=3)

    labeled, n = ndimage.label(upper)
    if n == 0:
        raise ValueError("Could not detect grid in image.")

    # Pick the largest component by area.
    sizes = ndimage.sum(upper, labeled, range(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1

    coords = np.argwhere(labeled == biggest)
    ymin, xmin = coords.min(axis=0)
    ymax, xmax = coords.max(axis=0)

    return GridBounds(top=int(ymin), bottom=int(ymax), left=int(xmin), right=int(xmax),
                      rows=grid_size, cols=grid_size)


def read_grid_state(arr: np.ndarray, gb: GridBounds) -> List[List[int]]:
    """Sample each cell center, decide filled if enough brick pixels."""
    bm = brick_mask(arr)
    grid: List[List[int]] = []
    sample = max(8, int(min(gb.cell_h, gb.cell_w) * 0.25))
    for r in range(gb.rows):
        row: List[int] = []
        for c in range(gb.cols):
            cy = int(gb.top + (r + 0.5) * gb.cell_h)
            cx = int(gb.left + (c + 0.5) * gb.cell_w)
            patch = bm[max(0, cy - sample):cy + sample, max(0, cx - sample):cx + sample]
            ratio = patch.mean() if patch.size else 0.0
            row.append(1 if ratio > 0.25 else 0)
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
    """Find 3 piece bounding boxes below the grid, then read each piece's cell pattern."""
    H, W = arr.shape[:2]
    bm = brick_mask(arr)

    # Restrict to region below the board.
    region_top = gb.bottom + 20
    region_bot = min(H, gb.bottom + int((H - gb.bottom) * 0.85))
    region = bm[region_top:region_bot, :].copy()

    # Heavy dilation to merge bricks within each piece.
    dilated = ndimage.binary_dilation(region, iterations=10)
    labeled, n = ndimage.label(dilated)

    comps = []
    for i in range(1, n + 1):
        mask = (labeled == i)
        if mask.sum() < 500:        # ignore noise
            continue
        ys, xs = np.where(mask)
        comps.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))

    if len(comps) < 3:
        raise ValueError(f"Expected 3 pieces below the board, found {len(comps)}.")

    # Sort left to right; if more than 3, keep the 3 largest by area.
    comps.sort(key=lambda c: (c[2] - c[0]) * (c[3] - c[1]), reverse=True)
    comps = comps[:3]
    comps.sort(key=lambda c: c[0])  # left to right

    # Find a single cell size that fits all three pieces' bboxes well.
    # Cell sizes are tested as fractions of the smaller piece dimensions.
    raw_dims = [(x1 - x0 + 1, y1 - y0 + 1) for (x0, y0, x1, y1) in comps]
    # Strip dilation padding (we expanded by ~10 px each side).
    raw_dims = [(max(1, w - 16), max(1, h - 16)) for (w, h) in raw_dims]

    best_cs, best_err = None, float("inf")
    for cs in range(20, 70):
        err = 0.0
        ok = True
        for (w, h) in raw_dims:
            cw = w / cs
            ch = h / cs
            if cw < 0.7 or ch < 0.7 or cw > 5 or ch > 5:
                ok = False; break
            err += (cw - round(cw)) ** 2 + (ch - round(ch)) ** 2
        if ok and err < best_err:
            best_err, best_cs = err, cs
    if best_cs is None:
        raise ValueError("Could not determine piece cell size.")

    pieces: List[Piece] = []
    for idx, (x0, y0, x1, y1) in enumerate(comps):
        # Use original (un-dilated) brick pixels inside this piece.
        ay0 = region_top + y0
        ay1 = region_top + y1
        sub = bm[ay0:ay1 + 1, x0:x1 + 1]

        # Strip outer dilation buffer from bbox to recover true bounds.
        # Find tightest bbox of true brick pixels inside.
        ys, xs = np.where(sub)
        if len(ys) == 0:
            continue
        py0, py1 = ys.min(), ys.max()
        px0, px1 = xs.min(), xs.max()
        sub = sub[py0:py1 + 1, px0:px1 + 1]
        ph, pw = sub.shape

        rows = max(1, round(ph / best_cs))
        cols = max(1, round(pw / best_cs))

        cell_h = ph / rows
        cell_w = pw / cols

        cells: List[Tuple[int, int]] = []
        for r in range(rows):
            for c in range(cols):
                cy0 = int(r * cell_h + cell_h * 0.25)
                cy1 = int(r * cell_h + cell_h * 0.75)
                cx0 = int(c * cell_w + cell_w * 0.25)
                cx1 = int(c * cell_w + cell_w * 0.75)
                patch = sub[cy0:cy1 + 1, cx0:cx1 + 1]
                if patch.size and patch.mean() > 0.4:
                    cells.append((r, c))

        if not cells:
            raise ValueError(f"Piece {idx + 1} appears empty.")

        pieces.append(Piece(name=f"P{idx + 1}", cells=cells, height=rows, width=cols))

    return pieces


# ---------- Solver ----------

@dataclass
class Move:
    piece: Piece
    row: int
    col: int


@dataclass
class Solution:
    moves: List[Move]                     # in placement order
    line_clears: int
    final_board: List[List[int]]


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
    """Return best solution by line-clears, or None if no valid placement exists."""
    rows, cols = len(board), len(board[0])
    best: Optional[Solution] = None

    for order in permutations(range(len(pieces))):
        seq = [pieces[i] for i in order]
        p1, p2, p3 = seq

        for r1 in range(rows - p1.height + 1):
            for c1 in range(cols - p1.width + 1):
                if not _can_place(board, p1, r1, c1):
                    continue
                b1 = _place(board, p1, r1, c1)
                b1, k1 = _clear_lines(b1)

                for r2 in range(rows - p2.height + 1):
                    for c2 in range(cols - p2.width + 1):
                        if not _can_place(b1, p2, r2, c2):
                            continue
                        b2 = _place(b1, p2, r2, c2)
                        b2, k2 = _clear_lines(b2)

                        for r3 in range(rows - p3.height + 1):
                            for c3 in range(cols - p3.width + 1):
                                if not _can_place(b2, p3, r3, c3):
                                    continue
                                b3 = _place(b2, p3, r3, c3)
                                b3, k3 = _clear_lines(b3)

                                clears = k1 + k2 + k3
                                if best is None or clears > best.line_clears:
                                    best = Solution(
                                        moves=[
                                            Move(p1, r1, c1),
                                            Move(p2, r2, c2),
                                            Move(p3, r3, c3),
                                        ],
                                        line_clears=clears,
                                        final_board=b3,
                                    )
    return best


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
