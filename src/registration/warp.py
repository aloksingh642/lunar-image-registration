"""Geometric resampling. This is not radiometric calibration or orthorectification."""

from __future__ import annotations

import cv2
import numpy as np


def warp_source_to_reference(
    source: np.ndarray,
    reference_shape: tuple[int, ...],
    matrix: np.ndarray,
    model: str,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = reference_shape[:2]
    matrix = np.asarray(matrix, dtype=np.float64)
    if model in {"similarity", "affine"} or matrix.shape == (2, 3):
        registered = cv2.warpAffine(
            source,
            matrix[:2, :3].astype(np.float32),
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid = cv2.warpAffine(
            np.full(source.shape[:2], 255, np.uint8),
            matrix[:2, :3].astype(np.float32),
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        registered = cv2.warpPerspective(
            source,
            matrix.astype(np.float32),
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid = cv2.warpPerspective(
            np.full(source.shape[:2], 255, np.uint8),
            matrix.astype(np.float32),
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return registered, valid
