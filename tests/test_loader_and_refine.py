"""Loader failures, synthetic labeling, and a controlled sub-pixel shift."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.demo.synthetic import make_pair
from src.exceptions import ImageLoadError
from src.io.loader import load_image
from src.refinement.subpixel import refine_correspondences


def test_missing_file_is_a_clear_error():
    with pytest.raises(ImageLoadError, match="File not found"):
        load_image("/home/user/lunar-registration/data/raw/this-file-does-not-exist.png")


def test_invalid_extension():
    path = Path("/tmp/not-an-image.txt")
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(ImageLoadError, match="Unsupported"):
        load_image(path)


def test_png_and_sidecar(tmp_path: Path):
    image = np.full((40, 48), 80, np.uint8)
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), image)
    (tmp_path / "frame.json").write_text(
        '{"sensor": "OHRC", "gsd_m": 0.3, "sun_elevation_deg": 20, "product_id": null}',
        encoding="utf-8",
    )
    bundle = load_image(path, role="source")
    assert bundle.sensor == "OHRC"
    assert bundle.metadata["gsd_m"] == 0.3
    assert bundle.origin == "user"
    assert "product_id" not in bundle.metadata


def test_synthetic_pair_is_labeled_and_not_a_mission_product():
    pair = make_pair("baseline", seed=3, size=128)
    assert pair["source"].origin == "synthetic"
    assert pair["reference"].origin == "synthetic"
    assert "Not Chandrayaan-2" in pair["disclaimer"]
    assert pair["source"].metadata["metadata_kind"] == "simulation_parameter"
    assert pair["source"].metadata.get("product_id") in (None, "")
    assert pair["ground_truth_matrix"].shape == (2, 3)


def test_ncc_refinement_recovers_a_half_pixel_shift():
    rng = np.random.default_rng(1)
    reference = rng.integers(30, 220, (96, 96), np.uint8)
    reference = cv2.GaussianBlur(reference, (0, 0), 1.1)
    cv2.circle(reference, (48, 48), 7, 235, -1)
    cv2.rectangle(reference, (20, 60), (36, 70), 40, -1)
    matrix = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, -0.25]], np.float32)
    source = cv2.warpAffine(reference, matrix, (96, 96))
    result = refine_correspondences(
        source,
        reference,
        np.array([[48.0, 48.0]]),
        np.array([[48.0, 48.0]]),
        method="ncc",
        template_radius=8,
        search_radius=3,
        ncc_accept=0.3,
    )
    refined = result["refined_src"][0]
    error_before = float(np.hypot(0.5, 0.25))
    error_after = float(np.hypot(refined[0] - 48.5, refined[1] - 47.75))
    assert result["ncc"][0] > 0.5
    assert result["accepted"][0]
    assert error_after < error_before
    assert error_after < 0.25
