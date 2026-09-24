"""Local correspondence refinement.

Two methods are implemented:

* ``ncc`` — normalized cross-correlation of a reference patch inside a small
  source window, with quadratic interpolation of the correlation peak.
  The peak NCC is a quality indicator. A decimal coordinate is not.
* ``cornersubpix`` — OpenCV ``cornerSubPix`` on each image independently,
  followed by the same NCC score so the pair still has a quality number.
  Independent corner snapping can leave a crater rim and is not treated as
  proof of sub-pixel registration accuracy.

Neither method claims mission sub-pixel accuracy. On synthetic data the
pipeline can compare refined points with the known transform. On real data
it reports NCC, shift, and the change in reprojection RMSE only.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def refine_correspondences(
    source: np.ndarray,
    reference: np.ndarray,
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    method: str = "ncc",
    template_radius: int = 8,
    search_radius: int = 4,
    ncc_accept: float = 0.40,
) -> dict[str, Any]:
    src_gray = _float_gray(source)
    ref_gray = _float_gray(reference)
    refined_src = np.array(src_pts, dtype=np.float64, copy=True)
    refined_ref = np.array(ref_pts, dtype=np.float64, copy=True)
    ncc = np.full(len(src_pts), np.nan, dtype=np.float64)
    shift = np.full(len(src_pts), np.nan, dtype=np.float64)
    accepted = np.zeros(len(src_pts), dtype=bool)
    sharpness = np.full(len(src_pts), np.nan, dtype=np.float64)

    if method == "cornersubpix":
        refined_src = _corner_subpix(source, refined_src, template_radius)
        refined_ref = _corner_subpix(reference, refined_ref, template_radius)

    radius = int(template_radius)
    search = int(search_radius)
    for index, (src_pt, ref_pt) in enumerate(zip(refined_src, refined_ref)):
        # Template stays on the original reference coordinate so cornerSubPix
        # cannot silently match a different crater rim without an NCC check.
        template = _patch(ref_gray, ref_pts[index], radius)
        window_center = src_pts[index] if method == "cornersubpix" else src_pt
        window = _patch(src_gray, window_center, radius + search)
        if template is None or window is None:
            continue
        if window.shape[0] < template.shape[0] or window.shape[1] < template.shape[1]:
            continue
        response = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
        _, peak, _, loc = cv2.minMaxLoc(response)
        dx, dy, curve = _quadratic_offset(response, loc)
        origin_x = window_center[0] - (radius + search)
        origin_y = window_center[1] - (radius + search)
        center_x = origin_x + loc[0] + radius + dx
        center_y = origin_y + loc[1] + radius + dy
        displacement = float(np.hypot(center_x - src_pts[index, 0], center_y - src_pts[index, 1]))
        ncc[index] = float(peak)
        shift[index] = displacement
        sharpness[index] = curve
        if peak >= ncc_accept and displacement <= search + 1.0:
            refined_src[index] = (center_x, center_y)
            refined_ref[index] = ref_pts[index]
            accepted[index] = True

    finite_ncc = ncc[np.isfinite(ncc)]
    finite_shift = shift[np.isfinite(shift)]
    return {
        "method": method,
        "refined_src": refined_src,
        "refined_ref": refined_ref,
        "ncc": ncc,
        "shift_px": shift,
        "sharpness": sharpness,
        "accepted": accepted,
        "n_attempted": int(len(src_pts)),
        "n_accepted": int(accepted.sum()),
        "mean_ncc": float(np.mean(finite_ncc)) if finite_ncc.size else float("nan"),
        "median_ncc": float(np.median(finite_ncc)) if finite_ncc.size else float("nan"),
        "mean_shift_px": float(np.mean(finite_shift)) if finite_shift.size else float("nan"),
        "mean_sharpness": float(np.nanmean(sharpness)) if np.isfinite(sharpness).any() else float("nan"),
        "quality_note": (
            "Refinement quality is the mean normalized cross-correlation of accepted local peaks. "
            "Mean shift is how far points moved, not an error. "
            "Sub-pixel accuracy is reported only when a ground-truth transform exists."
        ),
    }


def _corner_subpix(image: np.ndarray, points: np.ndarray, radius: int) -> np.ndarray:
    if len(points) == 0:
        return points.copy()
    gray = _float_gray(image)
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    height, width = gray_u8.shape
    usable = (
        (points[:, 0] > radius + 2)
        & (points[:, 1] > radius + 2)
        & (points[:, 0] < width - radius - 3)
        & (points[:, 1] < height - radius - 3)
    )
    refined = points.copy()
    if not np.any(usable):
        return refined
    corners = points[usable].reshape(-1, 1, 2).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
    try:
        updated = cv2.cornerSubPix(gray_u8, corners, (radius, radius), (-1, -1), criteria)
        refined[usable] = updated.reshape(-1, 2)
    except cv2.error:
        return refined
    return refined


def _quadratic_offset(response: np.ndarray, loc: tuple[int, int]) -> tuple[float, float, float]:
    x, y = loc
    dx = dy = 0.0
    curve = 0.0
    if 0 < x < response.shape[1] - 1:
        left, center, right = response[y, x - 1], response[y, x], response[y, x + 1]
        denom = left - 2 * center + right
        if abs(denom) > 1e-8:
            dx = float(np.clip(0.5 * (left - right) / denom, -0.75, 0.75))
            curve += float(-denom)
    if 0 < y < response.shape[0] - 1:
        up, center, down = response[y - 1, x], response[y, x], response[y + 1, x]
        denom = up - 2 * center + down
        if abs(denom) > 1e-8:
            dy = float(np.clip(0.5 * (up - down) / denom, -0.75, 0.75))
            curve += float(-denom)
    return dx, dy, curve


def _patch(image: np.ndarray, point: np.ndarray, radius: int) -> np.ndarray | None:
    x, y = float(point[0]), float(point[1])
    xi, yi = int(round(x)), int(round(y))
    if xi - radius < 1 or yi - radius < 1 or xi + radius >= image.shape[1] - 1 or yi + radius >= image.shape[0] - 1:
        return None
    patch = image[yi - radius : yi + radius + 1, xi - radius : xi + radius + 1]
    if patch.shape != (2 * radius + 1, 2 * radius + 1):
        return None
    if float(np.std(patch)) < 1e-3:
        return None
    return patch


def _float_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image.astype(np.float32)
