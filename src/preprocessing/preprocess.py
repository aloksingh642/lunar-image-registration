"""Preprocessing that does not replace the original image.

Lunar correspondence fails if it trusts raw brightness. Shadowed crater walls
and sun-facing rims swap appearance when the sun moves. These operators build
a feature image while the original array is kept for the warped product.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.models import ImageBundle, PipelineConfig


def preprocess_pair(
    source: ImageBundle,
    reference: ImageBundle,
    config: PipelineConfig,
    sun_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    illumination = resolve_illumination(config.illumination, source, reference, sun_analysis)
    src_gray = to_structural_gray(source.image, source.sensor)
    ref_gray = to_structural_gray(reference.image, reference.sensor)
    src_feat, src_info = preprocess_gray(src_gray, config, illumination)
    ref_feat, ref_info = preprocess_gray(ref_gray, config, illumination)
    if config.histogram_match:
        src_feat = match_histogram(src_feat, ref_feat)
        src_info["histogram_match"] = True
    else:
        src_info["histogram_match"] = False
    return {
        "source_gray": src_gray,
        "reference_gray": ref_gray,
        "source_features": src_feat,
        "reference_features": ref_feat,
        "illumination_used": illumination,
        "source_info": src_info,
        "reference_info": ref_info,
        "source_pyramid": gaussian_pyramid(src_feat),
        "reference_pyramid": gaussian_pyramid(ref_feat),
    }


def resolve_illumination(
    mode: str,
    source: ImageBundle,
    reference: ImageBundle,
    sun_analysis: dict[str, Any] | None,
) -> str:
    mode = (mode or "auto").lower()
    if mode != "auto":
        return mode
    # Retinex is the default structural representation. Gradient magnitude is
    # available, but on these lunar-like pairs it removed the albedo texture
    # SIFT still needs. IIRS cubes are reduced to a structural image first;
    # that is not spectral matching.
    if sun_analysis and sun_analysis.get("large_difference"):
        return "retinex"
    return "retinex"


def to_gray(image: np.ndarray) -> np.ndarray:
    return to_structural_gray(image, sensor="Other")


def to_structural_gray(image: np.ndarray, sensor: str = "Other") -> np.ndarray:
    """Collapse color or spectral bands to one structural image.

    IIRS-like cubes are averaged after per-band percentile stretch. That keeps
    spatial structure and discards spectral identity on purpose.
    """
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        bands = []
        for index in range(image.shape[2]):
            bands.append(_percentile_norm(image[:, :, index].astype(np.float32)))
        gray = np.mean(bands, axis=0)
        gray = (gray * 255).astype(np.uint8)
        return gray
    if gray.dtype != np.uint8:
        gray = _to_uint8(gray)
    if sensor == "IIRS":
        gray = cv2.GaussianBlur(gray, (0, 0), 1.2)
    return gray


def preprocess_gray(gray: np.ndarray, config: PipelineConfig, illumination: str) -> tuple[np.ndarray, dict[str, Any]]:
    info: dict[str, Any] = {"illumination": illumination, "denoise": config.denoise, "contrast": config.contrast}
    out = gray
    if config.denoise == "gaussian":
        out = cv2.GaussianBlur(out, (0, 0), 0.8)
    elif config.denoise == "bilateral":
        out = cv2.bilateralFilter(out, 5, 28, 5)
    elif config.denoise == "median":
        out = cv2.medianBlur(out, 3)
    elif config.denoise != "none":
        raise ValueError(f"Unknown denoise mode '{config.denoise}'.")

    if config.normalize == "minmax":
        out = _to_uint8(out.astype(np.float32))
    elif config.normalize == "percentile":
        out = _to_uint8(_percentile_norm(out.astype(np.float32)))
    elif config.normalize == "zscore":
        values = out.astype(np.float32)
        values = (values - values.mean()) / (values.std() + 1e-6)
        out = _to_uint8(values)
    elif config.normalize != "none":
        raise ValueError(f"Unknown normalize mode '{config.normalize}'.")

    if config.contrast == "clahe":
        clip = float(config.clahe_clip)
        grid = int(config.clahe_grid)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        out = clahe.apply(out)
    elif config.contrast == "histogram_eq":
        out = cv2.equalizeHist(out)
    elif config.contrast != "none":
        raise ValueError(f"Unknown contrast mode '{config.contrast}'.")

    if illumination == "retinex":
        out = single_scale_retinex(out, sigma=30)
    elif illumination == "gradient":
        out = gradient_magnitude(out)
    elif illumination == "dog":
        out = difference_of_gaussians(out)
    elif illumination != "none":
        raise ValueError(f"Unknown illumination mode '{illumination}'.")
    info["output_shape"] = list(out.shape)
    return out, info


def single_scale_retinex(gray: np.ndarray, sigma: float = 30.0) -> np.ndarray:
    """High-pass the illumination field. Removes slow sun-driven brightness, keeps rims and texture."""
    values = gray.astype(np.float32)
    low = cv2.GaussianBlur(values, (0, 0), sigma)
    residual = values - low
    return _to_uint8(_percentile_norm(residual))


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    values = gray.astype(np.float32)
    gx = cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx * gx + gy * gy)
    return _to_uint8(_percentile_norm(magnitude))


def difference_of_gaussians(gray: np.ndarray, sigma1: float = 1.0, sigma2: float = 3.0) -> np.ndarray:
    values = gray.astype(np.float32)
    dog = cv2.GaussianBlur(values, (0, 0), sigma1) - cv2.GaussianBlur(values, (0, 0), sigma2)
    return _to_uint8(_percentile_norm(dog))


def match_histogram(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    try:
        from skimage.exposure import match_histograms
    except ImportError:
        return source
    matched = match_histograms(source, reference)
    return matched.astype(np.uint8)


def gaussian_pyramid(image: np.ndarray, levels: int = 4) -> list[np.ndarray]:
    pyramid = [image]
    current = image
    for _ in range(levels - 1):
        if min(current.shape[:2]) < 64:
            break
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid


def _percentile_norm(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((values - lo) / (hi - lo), 0, 1)


def _to_uint8(values: np.ndarray) -> np.ndarray:
    array = values.astype(np.float32)
    if array.max() <= 1.01 and array.min() >= -0.01:
        array = np.clip(array, 0, 1) * 255.0
        return array.astype(np.uint8)
    return _percentile_norm(array).__mul__(255).astype(np.uint8)
