"""Registration figures. No external assets, so they preview anywhere."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


GOLD = (224, 177, 90)
BLUE = (126, 182, 255)
GREEN = (110, 207, 154)
RED = (214, 112, 112)
INK = (12, 18, 32)
GRID = (70, 92, 128)


def build_visuals(
    source: np.ndarray,
    reference: np.ndarray,
    registered: np.ndarray,
    mask: np.ndarray,
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    inlier_mask: np.ndarray,
    selected_index: np.ndarray,
    spatial: dict[str, Any],
    max_draw: int = 220,
) -> dict[str, np.ndarray]:
    selected = np.zeros(len(src_pts), dtype=bool)
    if len(selected_index):
        selected[selected_index] = True
    return {
        "matches_initial": draw_matches(source, reference, src_pts, ref_pts, None, max_draw, "Candidate matches"),
        "matches_inliers": draw_matches(source, reference, src_pts, ref_pts, inlier_mask, max_draw, "Inliers and outliers"),
        "matches_spatial": draw_matches(
            source, reference, src_pts[selected], ref_pts[selected], None, max_draw, "Spatially selected correspondences"
        ),
        "spatial_distribution": draw_spatial(reference, ref_pts, selected, spatial),
        "overlay": blend(registered, reference, mask, 0.5),
        "checkerboard": checkerboard(registered, reference, mask, tiles=8),
        "difference": difference_map(registered, reference, mask),
        "registered": to_rgb(registered),
        "reference": to_rgb(reference),
        "source": to_rgb(source),
    }


def draw_matches(
    source: np.ndarray,
    reference: np.ndarray,
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    inlier_mask: np.ndarray | None,
    max_draw: int,
    title: str,
) -> np.ndarray:
    left = to_rgb(source)
    right = to_rgb(reference)
    height = max(left.shape[0], right.shape[0])
    gap = 18
    canvas = np.full((height + 28, left.shape[1] + right.shape[1] + gap, 3), INK, np.uint8)
    canvas[28 : 28 + left.shape[0], : left.shape[1]] = left
    x_off = left.shape[1] + gap
    canvas[28 : 28 + right.shape[0], x_off : x_off + right.shape[1]] = right
    _label(canvas, title, 8, 18)
    if len(src_pts) == 0:
        _label(canvas, "No correspondences", 8, height + 12)
        return canvas
    indices = _choose_indices(len(src_pts), inlier_mask, max_draw)
    for index in indices:
        p1 = (int(round(src_pts[index, 0])), int(round(src_pts[index, 1])) + 28)
        p2 = (int(round(ref_pts[index, 0])) + x_off, int(round(ref_pts[index, 1])) + 28)
        good = True if inlier_mask is None else bool(inlier_mask[index])
        color = GOLD if good else RED
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, BLUE, -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 2, GREEN, -1, cv2.LINE_AA)
    shown = len(indices)
    _label(canvas, f"showing {shown} of {len(src_pts)}", canvas.shape[1] - 210, 18)
    return canvas


def draw_spatial(
    reference: np.ndarray,
    ref_pts: np.ndarray,
    selected_mask: np.ndarray,
    spatial: dict[str, Any],
) -> np.ndarray:
    canvas = to_rgb(reference)
    height, width = canvas.shape[:2]
    rows, cols = int(spatial["grid_rows"]), int(spatial["grid_cols"])
    overlay = canvas.copy()
    overlap = set(map(tuple, spatial.get("overlap_cells") or []))
    occupied = set(map(tuple, spatial.get("occupied_cells") or []))
    for row in range(rows):
        y0 = int(round(row * height / rows))
        y1 = int(round((row + 1) * height / rows))
        for col in range(cols):
            x0 = int(round(col * width / cols))
            x1 = int(round((col + 1) * width / cols))
            cell = (row, col)
            if overlap and cell not in overlap:
                color = (28, 34, 48)
            elif cell in occupied:
                color = (28, 78, 62)
            elif overlap:
                color = (92, 48, 48)
            else:
                color = (40, 48, 64)
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), color, -1)
            cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), GRID, 1, cv2.LINE_AA)
    canvas = cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0)
    if len(ref_pts):
        rejected = np.where(~selected_mask)[0] if selected_mask is not None else []
        for index in rejected[:400]:
            center = (int(round(ref_pts[index, 0])), int(round(ref_pts[index, 1])))
            cv2.circle(canvas, center, 2, RED, -1, cv2.LINE_AA)
        chosen = np.where(selected_mask)[0] if selected_mask is not None else range(len(ref_pts))
        for index in chosen:
            center = (int(round(ref_pts[index, 0])), int(round(ref_pts[index, 1])))
            cv2.circle(canvas, center, 4, GOLD, -1, cv2.LINE_AA)
            cv2.circle(canvas, center, 5, (255, 244, 214), 1, cv2.LINE_AA)
    coverage = 100.0 * float(spatial.get("spatial_coverage", 0.0))
    _label(canvas, f"coverage {coverage:.1f}%   gold = selected   red = not selected", 8, 18)
    return canvas


def blend(registered: np.ndarray, reference: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    reg = to_rgb(registered)
    ref = to_rgb(reference)
    reg, ref, mask = _match_size(reg, ref, mask)
    blended = cv2.addWeighted(reg, alpha, ref, 1.0 - alpha, 0)
    out = ref.copy()
    valid = mask > 0
    out[valid] = blended[valid]
    return out


def checkerboard(registered: np.ndarray, reference: np.ndarray, mask: np.ndarray, tiles: int = 8) -> np.ndarray:
    reg = to_rgb(registered)
    ref = to_rgb(reference)
    reg, ref, mask = _match_size(reg, ref, mask)
    height, width = ref.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width]
    tile_h = max(1, height / tiles)
    tile_w = max(1, width / tiles)
    board = ((ys // tile_h + xs // tile_w) % 2) == 0
    out = ref.copy()
    valid = (mask > 0) & board
    out[valid] = reg[valid]
    return out


def difference_map(registered: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> np.ndarray:
    reg = to_gray(registered)
    ref = to_gray(reference)
    if reg.shape != ref.shape:
        reg = cv2.resize(reg, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = cv2.absdiff(reg, ref)
    colored = cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    valid = mask > 0
    if valid.shape != colored.shape[:2]:
        valid = cv2.resize(valid.astype(np.uint8), (colored.shape[1], colored.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    reference_rgb = to_rgb(reference)
    if reference_rgb.shape[:2] != colored.shape[:2]:
        reference_rgb = cv2.resize(reference_rgb, (colored.shape[1], colored.shape[0]))
    colored[~valid] = reference_rgb[~valid]
    return colored


def to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image[:, :, :3].copy()


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(to_rgb(image), cv2.COLOR_RGB2GRAY)


def _match_size(reg: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if reg.shape[:2] != ref.shape[:2]:
        reg = cv2.resize(reg, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR)
    if mask.shape[:2] != ref.shape[:2]:
        mask = cv2.resize(mask, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST)
    return reg, ref, mask


def _choose_indices(count: int, inlier_mask: np.ndarray | None, max_draw: int) -> np.ndarray:
    if count <= max_draw:
        return np.arange(count)
    if inlier_mask is None:
        return np.linspace(0, count - 1, max_draw).astype(int)
    inliers = np.where(inlier_mask)[0]
    outliers = np.where(~inlier_mask)[0]
    take_in = inliers[: max_draw // 2]
    take_out = outliers[: max_draw - len(take_in)]
    return np.concatenate([take_in, take_out])


def _label(canvas: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (232, 238, 248), 1, cv2.LINE_AA)
