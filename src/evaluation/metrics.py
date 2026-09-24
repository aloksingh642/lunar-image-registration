"""Evaluation that does not invent ground truth.

Two modes:

* ``synthetic_benchmark`` / ``user_supplied_ground_truth`` — a known matrix
  exists, so geometric error can be compared with that matrix.
* ``no_ground_truth`` — only internal consistency is reported. Reprojection
  RMSE of the inliers is optimistic because those inliers were selected to
  fit the model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.geometry.transforms import apply_transform, decompose_transform, reprojection_errors, rmse


RMSE_DEFINITION = (
    "RMSE is the root-mean-square Euclidean distance, in pixels of the working image, "
    "between transformed source points and their corresponding reference points. "
    "The reported registration RMSE uses the final model points. "
    "Those points already survived outlier rejection, so this number is a consistency residual, "
    "not an independent accuracy against lunar ground control."
)

INLIER_RATIO_DEFINITION = (
    "Inlier ratio = geometrically consistent matches / descriptor candidate matches. "
    "Candidates are the pairs that passed the Lowe ratio test. "
    "Inliers are the candidates accepted by MAGSAC++ or RANSAC within the reprojection threshold."
)

COVERAGE_DEFINITION = (
    "Spatial coverage = overlap grid cells that contain at least one spatially selected correspondence "
    "/ overlap grid cells. The overlap is the valid footprint of the preliminary warp. "
    "Uniformity is the Shannon entropy of selected-point counts over those overlap cells, divided by "
    "log(number of overlap cells). 1 is uniform; 0 means every selected point sits in one cell."
)


def assess_reliability(metrics: dict[str, Any], gt: dict[str, Any] | None = None) -> dict[str, Any]:
    """A prototype flag. It is not a certification and not a map-accuracy grade."""
    reasons: list[str] = []
    inliers = int(metrics.get("inliers") or 0)
    coverage = float(metrics.get("spatial_coverage") or 0.0)
    scale = metrics.get("estimated_scale")
    if inliers < 12:
        reasons.append("Fewer than 12 geometric inliers.")
    if coverage < 0.15:
        reasons.append("Selected correspondences cover under 15% of the overlap grid.")
    if scale is not None and (float(scale) < 0.15 or float(scale) > 6.0):
        reasons.append("Estimated scale is outside 0.15–6 and may be degenerate.")
    if gt and gt.get("corner_rmse_px") is not None and float(gt["corner_rmse_px"]) > 8.0:
        reasons.append("Ground-truth corner error is above 8 px on this pair.")
    if reasons:
        level = "unreliable"
    elif inliers >= 40 and coverage >= 0.50:
        level = "consistent_on_this_pair"
    else:
        level = "marginal"
    return {
        "level": level,
        "reasons": reasons,
        "note": (
            "Reliability here means the match set is large and spread out enough for this prototype to trust the fit. "
            "It is not ISRO certification, not sub-pixel validation, and not evidence the method works on all Chandrayaan-2 data."
        ),
    }


def overlap_intensity_correlation(registered: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float | None:
    """Pearson correlation inside the overlap. Descriptive only — not an accuracy score."""
    import cv2

    if registered.ndim == 3:
        registered = cv2.cvtColor(registered, cv2.COLOR_RGB2GRAY)
    if reference.ndim == 3:
        reference = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    valid = mask > 0
    if int(valid.sum()) < 64:
        return None
    left = registered[valid].astype(np.float64)
    right = reference[valid].astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    denom = np.sqrt(np.sum(left * left) * np.sum(right * right)) + 1e-8
    return float(np.sum(left * right) / denom)


def build_metrics(
    *,
    n_candidates: int,
    n_inliers: int,
    n_spatial: int,
    n_final: int,
    final_src: np.ndarray,
    final_ref: np.ndarray,
    matrix: np.ndarray,
    model: str,
    spatial: dict[str, Any],
    overlap_fraction: float,
    refinement: dict[str, Any] | None,
    transform_params: dict[str, Any],
    evaluation_mode: str,
    mean_distance: float | None,
) -> dict[str, Any]:
    errors = reprojection_errors(final_src, final_ref, matrix, model) if len(final_src) else np.zeros((0,))
    metrics = {
        "evaluation_mode": evaluation_mode,
        "total_matches": int(n_candidates),
        "inliers": int(n_inliers),
        "outliers": int(max(0, n_candidates - n_inliers)),
        "inlier_ratio": float(n_inliers / n_candidates) if n_candidates else 0.0,
        "n_spatial_selected": int(n_spatial),
        "n_final_model_points": int(n_final),
        "rmse_px": rmse(errors),
        "median_reprojection_px": float(np.median(errors)) if errors.size else float("nan"),
        "max_reprojection_px": float(np.max(errors)) if errors.size else float("nan"),
        "spatial_coverage": float(spatial.get("spatial_coverage", 0.0)),
        "spatial_coverage_percent": float(100.0 * spatial.get("spatial_coverage", 0.0)),
        "uniformity": float(spatial.get("uniformity", 0.0)),
        "n_overlap_cells": int(spatial.get("n_overlap_cells", 0)),
        "n_occupied_cells": int(spatial.get("n_occupied_cells", 0)),
        "overlap_percent": float(100.0 * overlap_fraction),
        "mean_descriptor_distance": None if mean_distance is None else float(mean_distance),
        "estimated_scale": transform_params.get("scale"),
        "estimated_rotation_deg": transform_params.get("rotation_deg"),
        "estimated_translation_px": transform_params.get("translation_px"),
        "transform_model": model,
        "definitions": {
            "rmse": RMSE_DEFINITION,
            "inlier_ratio": INLIER_RATIO_DEFINITION,
            "spatial_coverage": COVERAGE_DEFINITION,
        },
    }
    if refinement is not None:
        metrics.update(
            {
                "refinement_method": refinement["method"],
                "refinement_mean_ncc": refinement["mean_ncc"],
                "refinement_median_ncc": refinement["median_ncc"],
                "refinement_mean_shift_px": refinement["mean_shift_px"],
                "refinement_acceptance_ratio": (
                    refinement["n_accepted"] / refinement["n_attempted"] if refinement["n_attempted"] else 0.0
                ),
                "refinement_quality_note": refinement["quality_note"],
            }
        )
    else:
        metrics.update(
            {
                "refinement_method": "disabled",
                "refinement_mean_ncc": None,
                "refinement_median_ncc": None,
                "refinement_mean_shift_px": None,
                "refinement_acceptance_ratio": None,
                "refinement_quality_note": "Sub-pixel refinement was turned off.",
            }
        )
    return metrics


def evaluate_ground_truth(
    estimated: np.ndarray,
    estimated_model: str,
    truth: np.ndarray,
    truth_model: str,
    source_shape: tuple[int, int],
    src_points_initial: np.ndarray | None = None,
    ref_points_initial: np.ndarray | None = None,
    src_points_refined: np.ndarray | None = None,
    ref_points_refined: np.ndarray | None = None,
) -> dict[str, Any]:
    height, width = source_shape[:2]
    corners = np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float64,
    )
    predicted = apply_transform(corners, estimated, estimated_model)
    actual = apply_transform(corners, truth, truth_model)
    corner_err = np.linalg.norm(predicted - actual, axis=1)
    result: dict[str, Any] = {
        "available": True,
        "label": "Compared with the known transform. This is not lunar survey accuracy.",
        "corner_rmse_px": rmse(corner_err),
        "corner_median_px": float(np.median(corner_err)),
        "corner_max_px": float(np.max(corner_err)),
        "corner_errors_px": corner_err.tolist(),
    }
    est_params = decompose_transform(estimated, estimated_model, source_shape)
    gt_params = decompose_transform(truth, truth_model, source_shape)
    if est_params.get("scale") is not None and gt_params.get("scale") is not None:
        result["scale_error_abs"] = abs(float(est_params["scale"]) - float(gt_params["scale"]))
        result["scale_error_percent"] = result["scale_error_abs"] / max(abs(float(gt_params["scale"])), 1e-8) * 100.0
    if est_params.get("rotation_deg") is not None and gt_params.get("rotation_deg") is not None:
        delta = float(est_params["rotation_deg"]) - float(gt_params["rotation_deg"])
        result["rotation_error_deg"] = abs((delta + 180.0) % 360.0 - 180.0)
    if est_params.get("translation_px") and gt_params.get("translation_px"):
        result["translation_error_px"] = float(
            np.hypot(
                est_params["translation_px"][0] - gt_params["translation_px"][0],
                est_params["translation_px"][1] - gt_params["translation_px"][1],
            )
        )
    result["estimated_parameters"] = est_params
    result["ground_truth_parameters"] = gt_params
    if src_points_initial is not None and ref_points_initial is not None and len(src_points_initial):
        result["localization_rmse_initial_px"] = _localization_rmse(src_points_initial, ref_points_initial, truth, truth_model)
    if src_points_refined is not None and ref_points_refined is not None and len(src_points_refined):
        result["localization_rmse_refined_px"] = _localization_rmse(src_points_refined, ref_points_refined, truth, truth_model)
    return result


def _localization_rmse(src_pts: np.ndarray, ref_pts: np.ndarray, truth: np.ndarray, model: str) -> float:
    projected = apply_transform(src_pts, truth, model)
    return rmse(np.linalg.norm(projected - ref_pts, axis=1))
