"""Explicit scale search on top of the detector's own scale space.

SIFT and AKAZE already search octaves inside one image. They can still miss a
pair when OHRC (~0.3 m) is matched to TMC-2 (5 m) or SELENE (10 m), because
the images presented to the detector have very different pixel extents.
This module resamples the source and keeps the scale that produces the most
geometrically consistent matches.
"""

from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np

from src.features.extractors import extract_features
from src.matching.matcher import match_descriptors, points_from_matches


InlierCounter = Callable[[np.ndarray, np.ndarray], int]


def search_scales(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    detector: str,
    max_features: int,
    rootsift: bool,
    ratio: float,
    scales: tuple[float, ...] | list[float],
    source_mask: np.ndarray | None,
    count_inliers: InlierCounter,
    prior_scale: float | None = None,
) -> dict[str, Any]:
    candidates = _candidate_scales(scales, prior_scale, source.shape, reference.shape)
    kp_ref, desc_ref = extract_features(reference, detector, max_features, None, rootsift)
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for scale in candidates:
        trial = _match_one_scale(
            source,
            reference,
            scale=scale,
            detector=detector,
            max_features=max_features,
            rootsift=rootsift,
            ratio=ratio,
            source_mask=source_mask,
            count_inliers=count_inliers,
            reference_features=(kp_ref, desc_ref),
        )
        trials.append({key: trial[key] for key in ("scale", "n_keypoints_source", "n_keypoints_reference", "n_matches", "n_inliers")})
        if best is None or _better(trial, best):
            best = trial
    assert best is not None
    best["trials"] = trials
    best["prior_scale"] = prior_scale
    return best


def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-3:
        return image
    height, width = image.shape[:2]
    new_size = (max(16, int(round(width * scale))), max(16, int(round(height * scale))))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, new_size, interpolation=interp)


def _match_one_scale(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    scale: float,
    detector: str,
    max_features: int,
    rootsift: bool,
    ratio: float,
    source_mask: np.ndarray | None,
    count_inliers: InlierCounter,
    reference_features: tuple | None = None,
) -> dict[str, Any]:
    source_scaled = resize_image(source, scale)
    mask_scaled = None if source_mask is None else resize_image(source_mask, scale)
    kp_src, desc_src = extract_features(source_scaled, detector, max_features, mask_scaled, rootsift)
    if reference_features is None:
        kp_ref, desc_ref = extract_features(reference, detector, max_features, None, rootsift)
    else:
        kp_ref, desc_ref = reference_features
    matches = match_descriptors(desc_src, desc_ref, ratio=ratio)
    src_pts, ref_pts, distances = points_from_matches(kp_src, kp_ref, matches, source_scale=scale, reference_scale=1.0)
    n_inliers = int(count_inliers(src_pts, ref_pts)) if len(src_pts) else 0
    return {
        "scale": float(scale),
        "n_keypoints_source": len(kp_src),
        "n_keypoints_reference": len(kp_ref),
        "n_matches": len(matches),
        "n_inliers": n_inliers,
        "src_pts": src_pts,
        "ref_pts": ref_pts,
        "distances": distances,
    }


def _candidate_scales(
    scales: tuple[float, ...] | list[float],
    prior: float | None,
    source_shape: tuple[int, ...],
    reference_shape: tuple[int, ...],
) -> list[float]:
    values = [float(item) for item in scales if item > 0.05]
    if prior is not None and prior > 0.05:
        values.extend([prior * factor for factor in (0.8, 1.0, 1.25)])
    # Also try to bring the source short side near the reference short side.
    src_min = min(source_shape[:2])
    ref_min = min(reference_shape[:2])
    if src_min > 0:
        values.append(ref_min / src_min)
    unique = sorted({round(value, 3) for value in values if 0.15 <= value <= 6.0})
    return unique or [1.0]


def _better(trial: dict[str, Any], best: dict[str, Any]) -> bool:
    """Prefer more inliers, but do not chase a tiny gain away from native scale."""
    if best["n_inliers"] == 0:
        return trial["n_inliers"] > 0 or trial["n_matches"] > best["n_matches"]
    if trial["n_inliers"] > best["n_inliers"] * 1.08:
        return True
    if best["n_inliers"] > trial["n_inliers"] * 1.08:
        return False
    return abs(np.log(max(trial["scale"], 1e-3))) < abs(np.log(max(best["scale"], 1e-3)))
