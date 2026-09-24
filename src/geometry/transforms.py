"""Transform application and parameter reporting."""

from __future__ import annotations

import numpy as np


def apply_transform(points: np.ndarray, matrix: np.ndarray, model: str) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if model in {"similarity", "affine"} or matrix.shape == (2, 3):
        hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
        return hom @ matrix[:2, :3].T
    if matrix.shape != (3, 3):
        raise ValueError(f"Unexpected matrix shape {matrix.shape} for model '{model}'.")
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    projected = hom @ matrix.T
    return projected[:, :2] / np.clip(projected[:, 2:3], 1e-12, None)


def reprojection_errors(src_pts: np.ndarray, ref_pts: np.ndarray, matrix: np.ndarray, model: str) -> np.ndarray:
    predicted = apply_transform(src_pts, matrix, model)
    return np.linalg.norm(predicted - ref_pts, axis=1)


def rmse(errors: np.ndarray) -> float:
    if len(errors) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(errors))))


def decompose_transform(matrix: np.ndarray, model: str, image_shape: tuple[int, int] | None = None) -> dict:
    matrix = np.asarray(matrix, dtype=np.float64)
    if model in {"similarity", "affine"} or matrix.shape == (2, 3):
        linear = matrix[:2, :2]
        translation = matrix[:2, 2]
        singular, rotation = _polar(linear)
        return {
            "model": model,
            "scale_x": float(singular[0]),
            "scale_y": float(singular[1]),
            "scale": float(np.sqrt(abs(singular[0] * singular[1]))),
            "rotation_deg": float(rotation),
            "translation_px": [float(translation[0]), float(translation[1])],
            "matrix": matrix.tolist(),
        }
    center = np.array([[0.0, 0.0]])
    if image_shape is not None:
        center = np.array([[image_shape[1] / 2.0, image_shape[0] / 2.0]])
    scale = _homography_scale(matrix, center[0])
    return {
        "model": "homography",
        "scale": scale,
        "rotation_deg": None,
        "translation_px": None,
        "note": "Homography scale is the local scale at the image center, not a global similarity scale.",
        "matrix": matrix.tolist(),
    }


def similarity_matrix(scale: float, rotation_deg: float, tx: float, ty: float) -> np.ndarray:
    theta = np.deg2rad(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [scale * cos_t, -scale * sin_t, tx],
            [scale * sin_t, scale * cos_t, ty],
        ],
        dtype=np.float64,
    )


def working_transform(matrix: np.ndarray, model: str, scale_src: float, scale_ref: float) -> np.ndarray:
    """Map a full-resolution transform into a frame where both images were resized.

    ``scale_src`` is working_pixels / original_pixels for the source.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if model in {"similarity", "affine"} or matrix.shape == (2, 3):
        out = np.zeros((2, 3), dtype=np.float64)
        out[:, :2] = (scale_ref / scale_src) * matrix[:2, :2]
        out[:, 2] = scale_ref * matrix[:2, 2]
        return out
    scale_ref_m = np.diag([scale_ref, scale_ref, 1.0])
    scale_src_inv = np.diag([1.0 / scale_src, 1.0 / scale_src, 1.0])
    return scale_ref_m @ matrix @ scale_src_inv


def full_transform(matrix: np.ndarray, model: str, scale_src: float, scale_ref: float) -> np.ndarray:
    """Inverse of ``working_transform``."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if abs(scale_src - 1.0) < 1e-9 and abs(scale_ref - 1.0) < 1e-9:
        return matrix.copy()
    if model in {"similarity", "affine"} or matrix.shape == (2, 3):
        out = np.zeros((2, 3), dtype=np.float64)
        out[:, :2] = (scale_src / scale_ref) * matrix[:2, :2]
        out[:, 2] = matrix[:2, 2] / scale_ref
        return out
    scale_ref_inv = np.diag([1.0 / scale_ref, 1.0 / scale_ref, 1.0])
    scale_src_m = np.diag([scale_src, scale_src, 1.0])
    return scale_ref_inv @ matrix @ scale_src_m


def _polar(linear: np.ndarray) -> tuple[np.ndarray, float]:
    u, singular, vt = np.linalg.svd(linear)
    rotation_m = u @ vt
    if np.linalg.det(rotation_m) < 0:
        u = u.copy()
        u[:, -1] *= -1
        rotation_m = u @ vt
        singular = singular.copy()
        singular[-1] *= -1
    rotation = np.degrees(np.arctan2(rotation_m[1, 0], rotation_m[0, 0]))
    return singular, float(rotation)


def _homography_scale(matrix: np.ndarray, center: np.ndarray) -> float:
    p0 = apply_transform(center.reshape(1, 2), matrix, "homography")[0]
    px = apply_transform((center + np.array([1.0, 0.0])).reshape(1, 2), matrix, "homography")[0]
    py = apply_transform((center + np.array([0.0, 1.0])).reshape(1, 2), matrix, "homography")[0]
    return float(0.5 * (np.hypot(*(px - p0)) + np.hypot(*(py - p0))))
