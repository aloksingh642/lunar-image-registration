"""Force correspondences to cover the overlap instead of one crater.

Selecting only the smallest descriptor distances often piles hundreds of points
on the strongest texture. Registration then looks precise and is still free to
drift everywhere else. This selector keeps a limited number of high-quality
inliers in each grid cell of the overlapping region.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def select_distributed(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    distances: np.ndarray,
    inlier_mask: np.ndarray,
    image_shape: tuple[int, int],
    overlap_mask: np.ndarray | None,
    grid_rows: int = 6,
    grid_cols: int = 6,
    per_cell: int = 3,
    min_spacing_px: float = 6.0,
) -> dict[str, Any]:
    height, width = image_shape[:2]
    grid_rows = int(np.clip(grid_rows, 2, 24))
    grid_cols = int(np.clip(grid_cols, 2, 24))
    per_cell = int(np.clip(per_cell, 1, 20))
    inlier_mask = np.asarray(inlier_mask).astype(bool)
    eligible = np.where(inlier_mask)[0]
    cells: dict[tuple[int, int], list[int]] = {}
    cell_rc = np.full((len(ref_pts), 2), -1, dtype=int)
    for index in eligible:
        row, col = _cell_of(ref_pts[index, 0], ref_pts[index, 1], width, height, grid_rows, grid_cols)
        cell_rc[index] = (row, col)
        cells.setdefault((row, col), []).append(index)

    selected: list[int] = []
    for key, indices in cells.items():
        ranked = sorted(indices, key=lambda item: float(distances[item]))
        chosen: list[int] = []
        for index in ranked:
            if len(chosen) >= per_cell:
                break
            if min_spacing_px > 0 and _too_close(ref_pts, index, chosen, min_spacing_px):
                continue
            chosen.append(index)
        selected.extend(chosen)

    selected_idx = np.array(sorted(selected), dtype=int) if selected else np.zeros((0,), dtype=int)
    overlap_cells = _overlap_cells(overlap_mask, width, height, grid_rows, grid_cols)
    selected_cells = {(int(cell_rc[i, 0]), int(cell_rc[i, 1])) for i in selected_idx}
    occupied = selected_cells & set(overlap_cells) if overlap_cells else selected_cells
    coverage_denom = len(overlap_cells) if overlap_cells else grid_rows * grid_cols
    coverage = len(occupied) / coverage_denom if coverage_denom else 0.0
    uniformity = _uniformity(selected_idx, cell_rc, overlap_cells or _all_cells(grid_rows, grid_cols))
    return {
        "selected_index": selected_idx,
        "cell_rc": cell_rc,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "n_selected": int(len(selected_idx)),
        "n_cells_with_inliers": len(cells),
        "n_overlap_cells": int(coverage_denom),
        "n_occupied_cells": int(len(occupied)),
        "spatial_coverage": float(coverage),
        "uniformity": float(uniformity),
        "overlap_cells": overlap_cells,
        "occupied_cells": occupied,
    }


def coverage_from_points(
    ref_pts: np.ndarray,
    image_shape: tuple[int, int],
    overlap_mask: np.ndarray | None,
    grid_rows: int,
    grid_cols: int,
) -> float:
    if len(ref_pts) == 0:
        return 0.0
    height, width = image_shape[:2]
    cells = {_cell_of(x, y, width, height, grid_rows, grid_cols) for x, y in ref_pts}
    overlap_cells = _overlap_cells(overlap_mask, width, height, grid_rows, grid_cols)
    denom = overlap_cells if overlap_cells else _all_cells(grid_rows, grid_cols)
    return len(set(cells) & set(denom)) / max(1, len(denom))


def _cell_of(x: float, y: float, width: int, height: int, rows: int, cols: int) -> tuple[int, int]:
    col = int(np.clip(np.floor(x * cols / max(width, 1)), 0, cols - 1))
    row = int(np.clip(np.floor(y * rows / max(height, 1)), 0, rows - 1))
    return row, col


def _too_close(ref_pts: np.ndarray, index: int, chosen: list[int], min_spacing: float) -> bool:
    if not chosen:
        return False
    delta = ref_pts[chosen] - ref_pts[index]
    return bool(np.any(np.linalg.norm(delta, axis=1) < min_spacing))


def _overlap_cells(mask: np.ndarray | None, width: int, height: int, rows: int, cols: int) -> list[tuple[int, int]]:
    if mask is None:
        return []
    binary = mask > 0
    cells = []
    for row in range(rows):
        y0 = int(round(row * height / rows))
        y1 = int(round((row + 1) * height / rows))
        for col in range(cols):
            x0 = int(round(col * width / cols))
            x1 = int(round((col + 1) * width / cols))
            tile = binary[y0:y1, x0:x1]
            if tile.size and np.any(tile):
                cells.append((row, col))
    return cells


def _all_cells(rows: int, cols: int) -> list[tuple[int, int]]:
    return [(row, col) for row in range(rows) for col in range(cols)]


def _uniformity(selected_idx: np.ndarray, cell_rc: np.ndarray, cells: list[tuple[int, int]]) -> float:
    if len(cells) <= 1 or len(selected_idx) == 0:
        return 0.0
    counts = {cell: 0 for cell in cells}
    for index in selected_idx:
        cell = (int(cell_rc[index, 0]), int(cell_rc[index, 1]))
        if cell in counts:
            counts[cell] += 1
    values = np.array(list(counts.values()), dtype=np.float64)
    total = values.sum()
    if total <= 0:
        return 0.0
    probability = values[values > 0] / total
    entropy = -np.sum(probability * np.log(probability))
    return float(entropy / np.log(len(cells)))
