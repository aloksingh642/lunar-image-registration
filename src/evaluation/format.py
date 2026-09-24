"""One formatter for the dashboard, HTML report, and JSON export."""

from __future__ import annotations

from typing import Any


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):.2f}%"


def _num(value: float | None, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:  # NaN
        return "—"
    return f"{number:.{digits}f}{suffix}"


def metric_rows(metrics: dict[str, Any], gt: dict[str, Any] | None = None) -> list[dict[str, str]]:
    rows = [
        {"label": "Total matches", "value": str(metrics.get("total_matches", 0)), "note": "Descriptor candidates after the ratio test."},
        {"label": "Inliers", "value": str(metrics.get("inliers", 0)), "note": "Geometrically consistent candidates."},
        {"label": "Outliers", "value": str(metrics.get("outliers", 0)), "note": "Candidates rejected by the robust estimator."},
        {"label": "Inlier ratio", "value": _pct(metrics.get("inlier_ratio")), "note": metrics["definitions"]["inlier_ratio"]},
        {"label": "RMSE", "value": _num(metrics.get("rmse_px"), 3, " px"), "note": metrics["definitions"]["rmse"]},
        {"label": "Median residual", "value": _num(metrics.get("median_reprojection_px"), 3, " px"), "note": "Median reprojection error of the final model points."},
        {"label": "Max residual", "value": _num(metrics.get("max_reprojection_px"), 3, " px"), "note": "Largest reprojection error among the final model points."},
        {"label": "Spatial coverage", "value": _num(metrics.get("spatial_coverage_percent"), 1, "%"), "note": metrics["definitions"]["spatial_coverage"]},
        {"label": "Uniformity", "value": _num(metrics.get("uniformity"), 2), "note": "1 = selected points spread evenly over overlap cells; 0 = concentrated in one cell."},
        {"label": "Overlap", "value": _num(metrics.get("overlap_percent"), 1, "%"), "note": "Fraction of reference pixels covered by the warped source."},
        {"label": "Spatially selected", "value": str(metrics.get("n_spatial_selected", 0)), "note": "Correspondences kept by the grid policy."},
        {"label": "Final model points", "value": str(metrics.get("n_final_model_points", 0)), "note": "Points used by the adopted transform after refitting."},
        {"label": "Estimated scale", "value": _num(metrics.get("estimated_scale"), 4), "note": "Geometric scale of the adopted model, source pixels to reference pixels. Not the internal resampling factor."},
        {"label": "Match resample", "value": _num(metrics.get("matching_resample_factor"), 3), "note": "Source was resized by this factor before detection. Keypoints were mapped back, so RMSE stays in working-image pixels."},
        {"label": "Overlap correlation", "value": _num(metrics.get("overlap_intensity_correlation"), 3), "note": "Pearson correlation of intensities in the overlap. Descriptive only. Low correlation is expected when the sun angle changes; it is not an error."},
        {"label": "Reliability", "value": str((metrics.get("reliability") or {}).get("level", "—")), "note": (metrics.get("reliability") or {}).get("note", "")},
        {"label": "Estimated rotation", "value": _num(metrics.get("estimated_rotation_deg"), 2, "°"), "note": "Degrees, image coordinates. Empty for a homography."},
        {
            "label": "Refinement quality",
            "value": _num(metrics.get("refinement_mean_ncc"), 3),
            "note": metrics.get("refinement_quality_note") or "Mean NCC of local refinement. Not a sub-pixel accuracy claim.",
        },
        {
            "label": "Refinement shift",
            "value": _num(metrics.get("refinement_mean_shift_px"), 3, " px"),
            "note": "Mean distance points moved during refinement. This is not an error.",
        },
    ]
    if gt and gt.get("available"):
        rows.extend(
            [
                {"label": "GT corner RMSE", "value": _num(gt.get("corner_rmse_px"), 3, " px"), "note": "Known-transform error at the four source corners. Synthetic or user-supplied only."},
                {"label": "GT scale error", "value": _num(gt.get("scale_error_percent"), 2, "%"), "note": "Absolute relative scale error against the known similarity/affine scale."},
                {"label": "GT rotation error", "value": _num(gt.get("rotation_error_deg"), 3, "°"), "note": "Absolute rotation error against the known transform."},
                {"label": "GT translation error", "value": _num(gt.get("translation_error_px"), 3, " px"), "note": "Translation-vector error. Meaningful mainly for similarity/affine."},
                {"label": "Localization RMSE initial", "value": _num(gt.get("localization_rmse_initial_px"), 3, " px"), "note": "Match-point error against the known transform, before refinement."},
                {"label": "Localization RMSE refined", "value": _num(gt.get("localization_rmse_refined_px"), 3, " px"), "note": "Match-point error against the known transform, after refinement."},
            ]
        )
    return rows
