"""Ratio-test matching. Descriptor distance is a candidate score, not a geometry."""

from __future__ import annotations

import cv2
import numpy as np


def match_descriptors(
    desc_src: np.ndarray,
    desc_ref: np.ndarray,
    ratio: float = 0.75,
) -> list:
    if desc_src is None or desc_ref is None or len(desc_src) < 2 or len(desc_ref) < 2:
        return []
    norm = cv2.NORM_HAMMING if desc_src.dtype == np.uint8 else cv2.NORM_L2
    knn = _knn(desc_src, desc_ref, norm)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            good.append(best)
    good.sort(key=lambda item: item.distance)
    return good


def points_from_matches(
    kp_src: list,
    kp_ref: list,
    matches: list,
    source_scale: float = 1.0,
    reference_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return points in the coordinate frame of the images that were detected on.

    ``source_scale`` is the factor applied to the original working image before
    detection. Dividing by it maps detections back to that working image.
    """
    if not matches:
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty.copy(), np.zeros((0,), dtype=np.float64)
    src = np.array([kp_src[m.queryIdx].pt for m in matches], dtype=np.float64) / source_scale
    ref = np.array([kp_ref[m.trainIdx].pt for m in matches], dtype=np.float64) / reference_scale
    distances = np.array([m.distance for m in matches], dtype=np.float64)
    return src, ref, distances


def _knn(desc_src: np.ndarray, desc_ref: np.ndarray, norm: int) -> list:
    if norm == cv2.NORM_L2 and len(desc_src) >= 800 and len(desc_ref) >= 800:
        index_params = {"algorithm": 1, "trees": 5}
        search_params = {"checks": 64}
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        try:
            return matcher.knnMatch(
                np.ascontiguousarray(desc_src, dtype=np.float32),
                np.ascontiguousarray(desc_ref, dtype=np.float32),
                k=2,
            )
        except cv2.error:
            pass
    matcher = cv2.BFMatcher(norm)
    return matcher.knnMatch(desc_src, desc_ref, k=2)
