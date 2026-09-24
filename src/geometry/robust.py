"""MAGSAC++ / RANSAC estimation with a bias toward the simpler model."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.exceptions import InsufficientMatchesError, RegistrationError
from src.geometry.transforms import apply_transform, reprojection_errors, rmse


_MIN_POINTS = {"similarity": 3, "affine": 3, "homography": 4}


def estimator_flag(name: str, model: str) -> tuple[int, str]:
    """Return an OpenCV method flag and the method that will actually run.

    OpenCV 4.11 accepts USAC_MAGSAC in ``estimateAffine2D`` and ``findHomography``,
    but not in ``estimateAffinePartial2D``. Similarity therefore uses RANSAC.
    """
    key = (name or "magsac").lower()
    wants_magsac = key in {"magsac", "magsac++", "usac_magsac"}
    if model == "similarity":
        return cv2.RANSAC, "ransac"
    if wants_magsac and hasattr(cv2, "USAC_MAGSAC"):
        return cv2.USAC_MAGSAC, "magsac"
    return cv2.RANSAC, "ransac"


def estimate_transform(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    model: str = "similarity",
    estimator: str = "magsac",
    threshold: float = 3.0,
    confidence: float = 0.999,
    max_iters: int = 4000,
    seed: int = 42,
) -> dict[str, Any]:
    model = model.lower()
    if model == "auto":
        return estimate_auto(src_pts, ref_pts, estimator, threshold, confidence, max_iters, seed)
    _require_points(src_pts, model)
    cv2.setRNGSeed(int(seed))
    matrix, mask, used = _fit(src_pts, ref_pts, model, estimator, threshold, confidence, max_iters)
    packed = _pack(src_pts, ref_pts, matrix, mask, model, used, threshold)
    if used != (estimator or "").lower():
        packed["estimator_note"] = (
            f"Requested '{estimator}', ran '{used}'. "
            "OpenCV's similarity estimator does not implement MAGSAC; affine and homography do."
        )
    return packed


def estimate_auto(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    estimator: str = "magsac",
    threshold: float = 3.0,
    confidence: float = 0.999,
    max_iters: int = 4000,
    seed: int = 42,
) -> dict[str, Any]:
    """Prefer similarity unless a freer model clearly explains the same points better.

    A homography can absorb viewpoint, but it can also bend a sparse lunar match
    set. Auto mode keeps the extra degrees of freedom only when RMSE falls by
    at least 15% and at least 80% of the simpler inliers remain.
    """
    attempts = []
    for model in ("similarity", "affine", "homography"):
        if len(src_pts) < _MIN_POINTS[model]:
            continue
        try:
            attempts.append(
                estimate_transform(src_pts, ref_pts, model, estimator, threshold, confidence, max_iters, seed)
            )
        except (InsufficientMatchesError, RegistrationError):
            continue
    if not attempts:
        raise RegistrationError(
            "No geometric model could be estimated. Try a larger reprojection threshold or a different detector."
        )
    chosen = attempts[0]
    for candidate in attempts[1:]:
        if _prefer_complex(chosen, candidate):
            chosen = candidate
    chosen["auto_attempts"] = [
        {"model": item["model"], "inliers": item["n_inliers"], "rmse_px": item["rmse_px"]} for item in attempts
    ]
    chosen["auto_note"] = (
        f"Auto model selection kept '{chosen['model']}'. "
        "A more flexible model is used only when it reduces reprojection RMSE by at least 15% "
        "without discarding more than 20% of the simpler model's inliers."
    )
    return chosen


def quick_inlier_count(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    model: str = "similarity",
    estimator: str = "magsac",
    threshold: float = 3.5,
    seed: int = 42,
) -> int:
    model = "similarity" if model == "auto" else model
    if len(src_pts) < _MIN_POINTS.get(model, 4):
        return 0
    try:
        result = estimate_transform(src_pts, ref_pts, model, estimator, threshold, 0.99, 1500, seed)
    except (InsufficientMatchesError, RegistrationError, cv2.error, ValueError):
        return 0
    return int(result["n_inliers"])


def refit_selected(
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    model: str,
    estimator: str,
    threshold: float,
    confidence: float,
    max_iters: int,
    seed: int,
) -> dict[str, Any]:
    """Robust-fit spatially selected points, falling back to least squares if too many are rejected."""
    if model == "auto":
        model = "similarity"
    if len(src_pts) < _MIN_POINTS.get(model, 4):
        raise InsufficientMatchesError(
            f"Only {len(src_pts)} correspondences remain after outlier rejection and spatial selection; "
            f"a {model} model needs at least {_MIN_POINTS.get(model, 4)}. "
            "The images may not overlap, the sun-angle change may be too severe for this prototype, "
            "or a different detector/illumination mode is needed."
        )
    robust = estimate_transform(src_pts, ref_pts, model, estimator, threshold, confidence, max_iters, seed)
    if robust["n_inliers"] >= max(_MIN_POINTS[model], int(0.6 * len(src_pts))):
        robust["fit_mode"] = "robust_on_spatial_selection"
        return robust
    matrix = _least_squares(src_pts, ref_pts, model)
    mask = np.ones(len(src_pts), dtype=bool)
    packed = _pack(src_pts, ref_pts, matrix, mask.reshape(-1, 1), model, estimator, threshold)
    packed["fit_mode"] = "least_squares_fallback"
    packed["warning"] = (
        "Robust refitting rejected too many of the spatially selected points, "
        "so the adopted transform is a least-squares fit of those selected points."
    )
    return packed


def _fit(src_pts, ref_pts, model, estimator, threshold, confidence, max_iters):
    src = np.ascontiguousarray(src_pts, dtype=np.float32)
    ref = np.ascontiguousarray(ref_pts, dtype=np.float32)
    flag, used = estimator_flag(estimator, model)
    try:
        matrix, mask = _fit_once(src, ref, model, flag, threshold, confidence, max_iters)
    except cv2.error:
        if flag == cv2.RANSAC:
            raise
        matrix, mask = _fit_once(src, ref, model, cv2.RANSAC, threshold, confidence, max_iters)
        used = "ransac"
    if matrix is None or mask is None:
        raise RegistrationError(
            f"The {used} estimator did not find a {model} model. "
            "The images may not overlap, or the threshold may be tighter than the residual."
        )
    return np.asarray(matrix, dtype=np.float64), mask, used


def _fit_once(src, ref, model, flag, threshold, confidence, max_iters):
    if model == "similarity":
        return cv2.estimateAffinePartial2D(
            src, ref, method=flag, ransacReprojThreshold=threshold, confidence=confidence, maxIters=max_iters
        )
    if model == "affine":
        return cv2.estimateAffine2D(
            src, ref, method=flag, ransacReprojThreshold=threshold, confidence=confidence, maxIters=max_iters
        )
    if model == "homography":
        return cv2.findHomography(
            src, ref, method=flag, ransacReprojThreshold=threshold, confidence=confidence, maxIters=max_iters
        )
    raise ValueError(f"Unknown transform model '{model}'. Use similarity, affine, homography, or auto.")


def _pack(src_pts, ref_pts, matrix, mask, model, estimator, threshold) -> dict[str, Any]:
    inliers = np.asarray(mask).ravel().astype(bool)
    errors = reprojection_errors(src_pts, ref_pts, matrix, model)
    inlier_errors = errors[inliers]
    return {
        "matrix": matrix,
        "model": model,
        "estimator": estimator,
        "threshold_px": float(threshold),
        "inlier_mask": inliers,
        "n_inliers": int(inliers.sum()),
        "n_outliers": int((~inliers).sum()),
        "rmse_px": rmse(inlier_errors),
        "median_px": float(np.median(inlier_errors)) if inlier_errors.size else float("nan"),
        "max_px": float(np.max(inlier_errors)) if inlier_errors.size else float("nan"),
        "errors_px": errors,
    }


def _prefer_complex(simple: dict[str, Any], complex_result: dict[str, Any]) -> bool:
    if simple["n_inliers"] == 0 or not np.isfinite(simple["rmse_px"]):
        return True
    keeps_points = complex_result["n_inliers"] >= 0.8 * simple["n_inliers"]
    improves = complex_result["rmse_px"] <= 0.85 * simple["rmse_px"]
    return bool(keeps_points and improves and complex_result["n_inliers"] >= _MIN_POINTS[complex_result["model"]])


def _require_points(src_pts: np.ndarray, model: str) -> None:
    needed = _MIN_POINTS.get(model, 4)
    if len(src_pts) < needed:
        raise InsufficientMatchesError(
            f"Only {len(src_pts)} correspondences are available; a {model} model needs at least {needed}. "
            "Try SIFT, enable scale search, or relax the ratio test."
        )


def _least_squares(src_pts: np.ndarray, ref_pts: np.ndarray, model: str) -> np.ndarray:
    if model == "homography":
        matrix, _ = cv2.findHomography(src_pts.astype(np.float32), ref_pts.astype(np.float32), 0)
        if matrix is None:
            raise RegistrationError("Least-squares homography failed.")
        return np.asarray(matrix, dtype=np.float64)
    design = np.concatenate([src_pts, np.ones((len(src_pts), 1))], axis=1)
    x_coef, _, _, _ = np.linalg.lstsq(design, ref_pts[:, 0], rcond=None)
    y_coef, _, _, _ = np.linalg.lstsq(design, ref_pts[:, 1], rcond=None)
    matrix = np.vstack([x_coef, y_coef])
    if model == "similarity":
        # Project the affine least-squares result back toward a similarity by polar factorization.
        linear = matrix[:, :2]
        u, singular, vt = np.linalg.svd(linear)
        scale = float(np.mean(np.abs(singular)))
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            rotation = u @ np.diag([1.0, -1.0]) @ vt
        projected = np.zeros((2, 3), dtype=np.float64)
        projected[:, :2] = scale * rotation
        projected[:, 2] = matrix[:, 2]
        return projected
    return matrix


def transform_points_preview(src_pts: np.ndarray, matrix: np.ndarray, model: str) -> np.ndarray:
    return apply_transform(src_pts, matrix, model)
