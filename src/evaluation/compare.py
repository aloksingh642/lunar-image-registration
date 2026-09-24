"""Compare classical detectors on the same pair.

This is a matching-stage comparison. It turns scale search and refinement off
so the table is about the detector, not about a second full registration.
"""

from __future__ import annotations

from dataclasses import replace

from src.models import ImageBundle, PipelineConfig
from src.pipeline import run_registration


def compare_detectors(
    source: ImageBundle,
    reference: ImageBundle,
    config: PipelineConfig,
    detectors: tuple[str, ...] = ("sift", "orb", "akaze"),
) -> list[dict]:
    rows = []
    for name in detectors:
        trial = replace(
            config,
            detector=name,
            rootsift=name == "sift",
            scale_search=False,
            refine=False,
            auto_retry=False,
            make_visuals=False,
        )
        try:
            result = run_registration(source, reference, trial)
            metrics = result.metrics
            gt = result.gt_metrics or {}
            rows.append(
                {
                    "detector": name,
                    "matches": metrics["total_matches"],
                    "inliers": metrics["inliers"],
                    "inlier_ratio": round(metrics["inlier_ratio"], 4),
                    "rmse_px": None if metrics["rmse_px"] != metrics["rmse_px"] else round(metrics["rmse_px"], 3),
                    "spatial_coverage": round(metrics["spatial_coverage"], 3),
                    "estimated_scale": None if metrics.get("estimated_scale") is None else round(metrics["estimated_scale"], 4),
                    "gt_corner_rmse_px": None if gt.get("corner_rmse_px") is None else round(gt["corner_rmse_px"], 3),
                    "reliability": metrics.get("reliability", {}).get("level"),
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001 — one detector must not hide the others
            rows.append({"detector": name, "error": str(exc)})
    return rows
