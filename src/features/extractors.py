"""Classical detectors used for scale, rotation, and illumination variation.

SIFT is the default because its scale space directly targets the altitude and
resolution differences between OHRC, TMC-2, IIRS, LRO NAC, and SELENE.
RootSIFT rescales SIFT descriptors so illumination changes affect them less.

A learned backend is intentionally not bundled. Registering one is possible
through ``DETECTORS``; this prototype does not call it "AI".
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np


DetectorFn = Callable[[np.ndarray, np.ndarray | None, int], tuple[list, np.ndarray]]
DETECTORS: dict[str, DetectorFn] = {}


def register_detector(name: str) -> Callable[[DetectorFn], DetectorFn]:
    def decorator(func: DetectorFn) -> DetectorFn:
        DETECTORS[name.lower()] = func
        return func

    return decorator


def extract_features(
    image: np.ndarray,
    method: str = "sift",
    max_features: int = 5000,
    mask: np.ndarray | None = None,
    rootsift: bool = True,
) -> tuple[list, np.ndarray]:
    key = method.lower()
    if key not in DETECTORS:
        known = ", ".join(sorted(DETECTORS))
        raise ValueError(f"Unknown detector '{method}'. Available: {known}.")
    keypoints, descriptors = DETECTORS[key](image, mask, int(max_features))
    if descriptors is None or len(keypoints) == 0:
        return [], np.zeros((0, 128), dtype=np.float32)
    if rootsift and key == "sift":
        descriptors = to_rootsift(descriptors)
    return keypoints, np.ascontiguousarray(descriptors)


@register_detector("sift")
def _sift(image: np.ndarray, mask: np.ndarray | None, max_features: int) -> tuple[list, np.ndarray]:
    detector = cv2.SIFT_create(
        nfeatures=max_features,
        nOctaveLayers=4,
        contrastThreshold=0.02,
        edgeThreshold=12,
        sigma=1.2,
    )
    return detector.detectAndCompute(_gray_uint8(image), _mask_uint8(mask))


@register_detector("orb")
def _orb(image: np.ndarray, mask: np.ndarray | None, max_features: int) -> tuple[list, np.ndarray]:
    detector = cv2.ORB_create(nfeatures=max_features, scaleFactor=1.2, nlevels=10, fastThreshold=8)
    return detector.detectAndCompute(_gray_uint8(image), _mask_uint8(mask))


@register_detector("akaze")
def _akaze(image: np.ndarray, mask: np.ndarray | None, max_features: int) -> tuple[list, np.ndarray]:
    detector = cv2.AKAZE_create()
    keypoints, descriptors = detector.detectAndCompute(_gray_uint8(image), _mask_uint8(mask))
    if keypoints and len(keypoints) > max_features:
        order = np.argsort([-kp.response for kp in keypoints])[:max_features]
        keypoints = [keypoints[i] for i in order]
        descriptors = descriptors[order]
    return keypoints, descriptors


def to_rootsift(descriptors: np.ndarray) -> np.ndarray:
    """Arandjelović & Zisserman RootSIFT: L1-normalize, then square-root."""
    desc = descriptors.astype(np.float32)
    desc /= np.sum(np.abs(desc), axis=1, keepdims=True) + 1e-8
    return np.sqrt(desc)


def keypoints_to_array(keypoints: list) -> np.ndarray:
    if not keypoints:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array([kp.pt for kp in keypoints], dtype=np.float64)


def _gray_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def _mask_uint8(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    out = mask.astype(np.uint8)
    if out.max() == 1:
        out = out * 255
    return out
