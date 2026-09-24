"""Geometry, metrics, and spatial-selection tests that do not need a full match."""

import numpy as np
import pytest

from src.evaluation.metrics import build_metrics
from src.geometry.spatial import select_distributed
from src.geometry.transforms import apply_transform, decompose_transform, rmse, similarity_matrix
from src.registration.warp import warp_source_to_reference


def test_similarity_roundtrip():
    matrix = similarity_matrix(1.35, -12.5, 20.0, -7.0)
    params = decompose_transform(matrix, "similarity")
    assert params["scale"] == pytest.approx(1.35, abs=1e-6)
    assert params["rotation_deg"] == pytest.approx(-12.5, abs=1e-6)
    assert params["translation_px"][0] == pytest.approx(20.0, abs=1e-6)
    assert params["translation_px"][1] == pytest.approx(-7.0, abs=1e-6)


def test_apply_transform_translation():
    matrix = similarity_matrix(1.0, 0.0, 4.0, -2.0)
    points = np.array([[10.0, 8.0], [0.0, 0.0]])
    moved = apply_transform(points, matrix, "similarity")
    assert moved[0, 0] == pytest.approx(14.0)
    assert moved[0, 1] == pytest.approx(6.0)


def test_rmse_of_perfect_fit_is_zero():
    src = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [8.0, 6.0]])
    matrix = similarity_matrix(1.2, 15.0, 3.0, -1.0)
    ref = apply_transform(src, matrix, "similarity")
    errors = np.linalg.norm(apply_transform(src, matrix, "similarity") - ref, axis=1)
    assert rmse(errors) == pytest.approx(0.0, abs=1e-9)


def test_warp_shifts_a_block():
    import cv2

    source = np.zeros((80, 90), np.uint8)
    source[20:30, 15:25] = 255
    matrix = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 5.0]], np.float64)
    warped, mask = warp_source_to_reference(source, (80, 90), matrix, "similarity")
    assert warped[25:35, 25:35].mean() > 200
    assert mask[25, 25] > 0
    assert warped[20, 15] == 0


def test_spatial_selection_limits_a_cluster():
    width, height = 100, 100
    dense = np.column_stack(
        [
            np.linspace(2, 16, 30),
            np.linspace(2, 16, 30),
        ]
    )
    others = np.array(
        [
            [30.0, 10.0],
            [50.0, 10.0],
            [70.0, 50.0],
            [20.0, 70.0],
            [80.0, 80.0],
        ]
    )
    ref = np.vstack([dense, others])
    src = ref.copy()
    distances = np.concatenate([np.full(30, 0.1), np.full(len(others), 0.4)])
    mask = np.ones(len(ref), dtype=bool)
    overlap = np.ones((height, width), np.uint8) * 255
    selected = select_distributed(
        src, ref, distances, mask, (height, width), overlap, grid_rows=5, grid_cols=5, per_cell=2, min_spacing_px=0
    )
    chosen = ref[selected["selected_index"]]
    in_dense_cell = np.sum((chosen[:, 0] < 20) & (chosen[:, 1] < 20))
    assert in_dense_cell <= 2
    assert selected["n_occupied_cells"] >= 6
    assert selected["spatial_coverage"] >= 6 / 25


def test_metric_definitions_are_present():
    src = np.array([[0.0, 0.0], [1.0, 1.0]])
    ref = src.copy()
    matrix = similarity_matrix(1, 0, 0, 0)
    metrics = build_metrics(
        n_candidates=10,
        n_inliers=4,
        n_spatial=2,
        n_final=2,
        final_src=src,
        final_ref=ref,
        matrix=matrix,
        model="similarity",
        spatial={"spatial_coverage": 0.5, "uniformity": 0.8, "n_overlap_cells": 4, "n_occupied_cells": 2},
        overlap_fraction=0.4,
        refinement=None,
        transform_params={"scale": 1.0, "rotation_deg": 0.0, "translation_px": [0.0, 0.0]},
        evaluation_mode="no_ground_truth",
        mean_distance=12.0,
    )
    assert metrics["inlier_ratio"] == pytest.approx(0.4)
    assert "not an independent" in metrics["definitions"]["rmse"]
    assert metrics["outliers"] == 6
