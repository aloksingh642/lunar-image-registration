"""End-to-end lunar correspondence and registration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from src.evaluation.metrics import (
    assess_reliability,
    build_metrics,
    evaluate_ground_truth,
    overlap_intensity_correlation,
)
from src.exceptions import InsufficientMatchesError, RegistrationError
from src.geometry.robust import estimate_transform, quick_inlier_count, refit_selected
from src.geometry.spatial import select_distributed
from src.geometry.transforms import decompose_transform, full_transform
from src.io.metadata import scale_prior_from_gsd, sun_angle_analysis
from src.logging_config import get_logger
from src.matching.multiscale import search_scales
from src.models import ImageBundle, PipelineConfig, image_summary
from src.preprocessing.preprocess import preprocess_pair, resolve_illumination
from src.refinement.subpixel import refine_correspondences
from src.registration.warp import warp_source_to_reference
from src.visualization.plots import build_visuals, to_rgb

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    config: dict[str, Any]
    evaluation_mode: str
    data_origin: str
    source_info: dict[str, Any]
    reference_info: dict[str, Any]
    metrics: dict[str, Any]
    gt_metrics: dict[str, Any] | None
    transform_working: np.ndarray
    transform_full: np.ndarray
    transform_model: str
    transform_params: dict[str, Any]
    src_points: np.ndarray
    ref_points: np.ndarray
    distances: np.ndarray
    inlier_mask: np.ndarray
    selected_index: np.ndarray
    final_src: np.ndarray
    final_ref: np.ndarray
    refined_src: np.ndarray
    refined_ref: np.ndarray
    refinement: dict[str, Any] | None
    registered: np.ndarray
    registered_mask: np.ndarray
    feature_source: np.ndarray
    feature_reference: np.ndarray
    visuals: dict[str, np.ndarray] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    scale_analysis: dict[str, Any] = field(default_factory=dict)
    sun_analysis: dict[str, Any] = field(default_factory=dict)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    spatial: dict[str, Any] = field(default_factory=dict)
    working_scale_source: float = 1.0
    working_scale_reference: float = 1.0

    @property
    def disclaimer(self) -> str:
        if self.data_origin == "synthetic":
            return (
                "Synthetic demo. These images are procedural crater fields, not Chandrayaan-2, LRO, or SELENE data. "
                "Ground-truth errors measure the simulator, not mission accuracy."
            )
        if self.evaluation_mode == "user_supplied_ground_truth":
            return "A user-supplied ground-truth matrix was found. It is not independent lunar survey verification."
        return (
            "No ground truth is available. RMSE is the reprojection residual of the estimated correspondences, "
            "not a verified lunar registration accuracy."
        )


def run_registration(
    source: ImageBundle,
    reference: ImageBundle,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    config = config or PipelineConfig()
    started = time.perf_counter()
    log: list[str] = []
    warnings: list[str] = []

    def stage(message: str) -> None:
        line = f"{time.perf_counter() - started:6.2f}s  {message}"
        log.append(line)
        logger.info(message)

    if source.origin == "synthetic" or reference.origin == "synthetic":
        warnings.append("This run uses synthetic/demo imagery. It is not a Chandrayaan-2 result.")
    if source.origin != reference.origin and "synthetic" in {source.origin, reference.origin}:
        warnings.append("Source and reference origins differ. Do not mix synthetic and real images in one scientific claim.")

    sun = sun_angle_analysis(source.metadata, reference.metadata)
    if not sun.get("available"):
        warnings.append("Sun-angle metadata is missing. Illumination handling uses the image only.")
    elif sun.get("metadata_kind") == "simulation_parameter":
        warnings.append("Sun angles on this run are simulation parameters, not spacecraft ephemeris.")
    gsd = scale_prior_from_gsd(source.metadata, reference.metadata)
    stage(f"Loaded source {source.width}×{source.height} and reference {reference.width}×{reference.height}.")

    work_src, scale_src, mask_src = _working_image(source.image, source.mask, config.max_working_side)
    work_ref, scale_ref, _ = _working_image(reference.image, None, config.max_working_side)
    if scale_src != 1 or scale_ref != 1:
        warnings.append(
            f"Feature matching used a working copy (scales {scale_src:.3f}, {scale_ref:.3f} relative to the loaded images)."
        )
    stage("Built working images.")

    work_source = ImageBundle(
        image=work_src,
        origin=source.origin,
        sensor=source.sensor,
        role="source",
        metadata=source.metadata,
        mask=mask_src,
    )
    work_reference = ImageBundle(
        image=work_ref,
        origin=reference.origin,
        sensor=reference.sensor,
        role="reference",
        metadata=reference.metadata,
        reference_catalog=reference.reference_catalog,
    )
    prepared = preprocess_pair(work_source, work_reference, config, sun)
    stage(f"Preprocessed with illumination mode '{prepared['illumination_used']}'.")

    match = _match_with_retries(prepared, work_source, work_reference, config, sun, gsd, stage)
    if match["n_matches"] < 6:
        raise InsufficientMatchesError(
            f"Only {match['n_matches']} descriptor matches were found. "
            "Confirm that the images overlap, try SIFT with scale search, or use retinex/gradient preprocessing."
        )
    stage(
        f"Selected scale {match['scale']:.3f} with {match['n_matches']} matches "
        f"and {match['n_inliers']} quick inliers."
    )

    geometry_model = "similarity" if config.model == "auto" else config.model
    preliminary = estimate_transform(
        match["src_pts"],
        match["ref_pts"],
        config.model,
        config.estimator,
        config.ransac_threshold_px,
        config.confidence,
        config.max_iters,
        config.rng_seed,
    )
    stage(
        f"Robust {preliminary['model']} fit: {preliminary['n_inliers']} inliers, "
        f"RMSE {preliminary['rmse_px']:.3f} px."
    )
    if preliminary.get("auto_note"):
        warnings.append(preliminary["auto_note"])
    if preliminary.get("estimator_note"):
        warnings.append(preliminary["estimator_note"])
    min_needed = 4 if preliminary["model"] == "homography" else 3
    if preliminary["n_inliers"] < min_needed:
        raise InsufficientMatchesError(
            f"Only {preliminary['n_inliers']} geometrically consistent matches were found "
            f"from {match['n_matches']} descriptor candidates. "
            "Check the overlap, try SIFT with scale search, or use retinex/gradient preprocessing. "
            "A very large sun-angle change can still defeat this prototype; that is a limitation, not a hidden success."
        )

    overlap_mask = _overlap_from_matrix(
        preliminary["matrix"], preliminary["model"], work_src.shape, work_ref.shape
    )
    spatial = select_distributed(
        match["src_pts"],
        match["ref_pts"],
        match["distances"],
        preliminary["inlier_mask"],
        work_ref.shape,
        overlap_mask,
        config.grid_rows,
        config.grid_cols,
        config.per_cell,
        config.min_spacing_px,
    )
    selected = spatial["selected_index"]
    min_needed = 4 if preliminary["model"] == "homography" else 3
    if len(selected) < min_needed:
        warnings.append(
            "Spatial selection kept too few points, so all robust inliers were used. "
            "Coverage then describes the inlier cloud, which may still be concentrated."
        )
        selected = np.where(preliminary["inlier_mask"])[0]
        spatial["selected_index"] = selected
        spatial["n_selected"] = int(len(selected))
    stage(f"Spatial selection kept {len(selected)} correspondences across {spatial['n_occupied_cells']} cells.")

    adopted = refit_selected(
        match["src_pts"][selected],
        match["ref_pts"][selected],
        preliminary["model"],
        config.estimator,
        config.ransac_threshold_px,
        config.confidence,
        config.max_iters,
        config.rng_seed,
    )
    if adopted.get("warning"):
        warnings.append(adopted["warning"])
    final_src = match["src_pts"][selected]
    final_ref = match["ref_pts"][selected]
    if adopted.get("fit_mode") == "robust_on_spatial_selection":
        keep = adopted["inlier_mask"]
        final_src = final_src[keep]
        final_ref = final_ref[keep]
        selected_final = selected[keep]
    else:
        selected_final = selected

    refinement = None
    refined_src = final_src.copy()
    refined_ref = final_ref.copy()
    if config.refine and len(final_src) >= min_needed:
        refinement = refine_correspondences(
            prepared["source_features"],
            prepared["reference_features"],
            final_src,
            final_ref,
            config.refine_method,
            config.template_radius,
            config.search_radius,
            config.ncc_accept,
        )
        stage(
            f"Refinement accepted {refinement['n_accepted']}/{refinement['n_attempted']} "
            f"points, mean NCC {refinement['mean_ncc']:.3f}."
        )
        if refinement["n_accepted"] >= min_needed and (
            refinement["mean_ncc"] == refinement["mean_ncc"] and refinement["mean_ncc"] >= config.ncc_accept
        ):
            use_src = final_src.copy()
            accepted = refinement["accepted"]
            use_src[accepted] = refinement["refined_src"][accepted]
            use_ref = final_ref
            try:
                refit = refit_selected(
                    use_src,
                    use_ref,
                    adopted["model"],
                    config.estimator,
                    config.ransac_threshold_px,
                    config.confidence,
                    config.max_iters,
                    config.rng_seed,
                )
                if refit["n_inliers"] >= min_needed and refit["rmse_px"] <= max(adopted["rmse_px"] * 1.25, adopted["rmse_px"] + 0.5):
                    adopted = refit
                    refined_src = use_src
                    refined_ref = use_ref
                    if refit.get("fit_mode") == "robust_on_spatial_selection":
                        refined_src = use_src[refit["inlier_mask"]]
                        refined_ref = use_ref[refit["inlier_mask"]]
                        final_src = final_src[refit["inlier_mask"]]
                        final_ref = final_ref[refit["inlier_mask"]]
                else:
                    warnings.append(
                        "Refined points did not improve the geometric fit, so the transform keeps the pre-refinement points. "
                        "Refined coordinates are still exported."
                    )
                    refined_src = use_src
                    refined_ref = use_ref
            except (InsufficientMatchesError, RegistrationError) as exc:
                warnings.append(f"Refined refit failed ({exc}). The pre-refinement transform was kept.")
        else:
            warnings.append(
                "Local refinement quality was low, so decimal shifts were not used in the adopted transform. "
                "A decimal coordinate by itself is not sub-pixel accuracy."
            )
    elif config.refine:
        warnings.append("Too few points to refine.")

    model_src = refined_src if refinement is not None and len(refined_src) == adopted["matrix"].shape[0] else final_src
    # The adopted matrix was fit to whichever point set produced it. Use that matrix as-is.
    registered, valid = warp_source_to_reference(work_src, work_ref.shape, adopted["matrix"], adopted["model"])
    overlap_fraction = float(np.mean(valid > 0))
    if overlap_fraction < 0.15:
        warnings.append("The warped source covers under 15% of the reference. The overlap may be too small to trust.")
    stage("Warped the source into the reference frame.")

    params = decompose_transform(adopted["matrix"], adopted["model"], work_src.shape[:2])
    gt_matrix, gt_model, evaluation_mode = _ground_truth(source, reference, scale_src, scale_ref)
    if evaluation_mode == "no_ground_truth":
        warnings.append("No ground truth. RMSE is an internal residual, not verified lunar accuracy.")
    mean_distance = float(np.mean(match["distances"][preliminary["inlier_mask"]])) if preliminary["n_inliers"] else None
    metrics = build_metrics(
        n_candidates=int(match["n_matches"]),
        n_inliers=int(preliminary["n_inliers"]),
        n_spatial=int(spatial["n_selected"]),
        n_final=int(len(model_src) if len(final_src) else 0),
        final_src=refined_src if len(refined_src) else final_src,
        final_ref=refined_ref if len(refined_ref) else final_ref,
        matrix=adopted["matrix"],
        model=adopted["model"],
        spatial=spatial,
        overlap_fraction=overlap_fraction,
        refinement=refinement,
        transform_params=params,
        evaluation_mode=evaluation_mode,
        mean_distance=mean_distance,
    )
    # Final point count should match the points whose residual is reported.
    report_src = refined_src if refinement is not None else final_src
    report_ref = refined_ref if refinement is not None else final_ref
    if len(report_src) == 0:
        report_src, report_ref = final_src, final_ref
    metrics = build_metrics(
        n_candidates=int(match["n_matches"]),
        n_inliers=int(preliminary["n_inliers"]),
        n_spatial=int(spatial["n_selected"]),
        n_final=int(len(report_src)),
        final_src=report_src,
        final_ref=report_ref,
        matrix=adopted["matrix"],
        model=adopted["model"],
        spatial=spatial,
        overlap_fraction=overlap_fraction,
        refinement=refinement,
        transform_params=params,
        evaluation_mode=evaluation_mode,
        mean_distance=mean_distance,
    )
    gt_metrics = None
    if gt_matrix is not None and gt_model is not None:
        gt_metrics = evaluate_ground_truth(
            adopted["matrix"],
            adopted["model"],
            gt_matrix,
            gt_model,
            work_src.shape[:2],
            final_src,
            final_ref,
            refined_src,
            refined_ref,
        )
        stage(
            f"Ground-truth corner RMSE {gt_metrics['corner_rmse_px']:.3f} px "
            f"({evaluation_mode})."
        )
        initial_loc = gt_metrics.get("localization_rmse_initial_px")
        refined_loc = gt_metrics.get("localization_rmse_refined_px")
        if (
            initial_loc is not None
            and refined_loc is not None
            and refined_loc > initial_loc + 0.05
        ):
            warnings.append(
                "On this pair, sub-pixel refinement did not reduce ground-truth localization error. "
                "NCC is a quality indicator. Decimal coordinates are not accuracy."
            )
    metrics["matching_resample_factor"] = float(match["scale"])
    metrics["overlap_intensity_correlation"] = overlap_intensity_correlation(registered, work_ref, valid)
    metrics["reliability"] = assess_reliability(metrics, gt_metrics)
    if metrics["reliability"]["level"] == "unreliable":
        warnings.append(
            "Reliability flag: unreliable. " + " ".join(metrics["reliability"]["reasons"])
        )

    visuals = {}
    if config.make_visuals:
        visuals = build_visuals(
            work_src,
            work_ref,
            registered,
            valid,
            match["src_pts"],
            match["ref_pts"],
            preliminary["inlier_mask"],
            spatial["selected_index"],
            spatial,
            config.max_draw,
        )
        visuals["feature_source"] = to_rgb(prepared["source_features"])
        visuals["feature_reference"] = to_rgb(prepared["reference_features"])
        stage("Built visualizations.")

    full_matrix = full_transform(adopted["matrix"], adopted["model"], scale_src, scale_ref)
    origin = "synthetic" if "synthetic" in {source.origin, reference.origin} else source.origin
    result = PipelineResult(
        config=config.to_dict(),
        evaluation_mode=evaluation_mode,
        data_origin=origin,
        source_info=image_summary(source),
        reference_info=image_summary(reference),
        metrics=metrics,
        gt_metrics=gt_metrics,
        transform_working=adopted["matrix"],
        transform_full=full_matrix,
        transform_model=adopted["model"],
        transform_params=params,
        src_points=match["src_pts"],
        ref_points=match["ref_pts"],
        distances=match["distances"],
        inlier_mask=preliminary["inlier_mask"],
        selected_index=spatial["selected_index"],
        final_src=final_src,
        final_ref=final_ref,
        refined_src=refined_src,
        refined_ref=refined_ref,
        refinement=refinement,
        registered=registered,
        registered_mask=valid,
        feature_source=prepared["source_features"],
        feature_reference=prepared["reference_features"],
        visuals=visuals,
        warnings=warnings,
        log=log,
        scale_analysis={
            "selected_working_scale": match["scale"],
            "trials": match.get("trials", []),
            "gsd_prior": gsd,
            "estimated": params,
            "pyramid_levels_source": [list(level.shape[:2]) for level in prepared["source_pyramid"]],
            "pyramid_levels_reference": [list(level.shape[:2]) for level in prepared["reference_pyramid"]],
        },
        sun_analysis=sun,
        attempts=match.get("attempt_log", []),
        spatial={key: value for key, value in spatial.items() if key not in {"cell_rc"}},
        working_scale_source=scale_src,
        working_scale_reference=scale_ref,
    )
    # cell_rc is useful for export; keep it out of JSON-only copies but available here.
    result.spatial["cell_rc"] = spatial.get("cell_rc")
    stage("Finished.")
    return result


def _match_with_retries(prepared, source, reference, config: PipelineConfig, sun, gsd, stage) -> dict[str, Any]:
    illumination_plan = [prepared["illumination_used"]]
    if config.auto_retry:
        for mode in ("retinex", "gradient", "dog"):
            if mode not in illumination_plan:
                illumination_plan.append(mode)
    detector_plan = [config.detector]
    if config.auto_retry and config.detector != "sift":
        detector_plan.append("sift")

    prior = gsd.get("expected_scale_source_to_reference") if gsd.get("available") else None
    attempt_log: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    feature_cache = {
        prepared["illumination_used"]: (prepared["source_features"], prepared["reference_features"])
    }

    for detector in detector_plan:
        for illumination in illumination_plan:
            features = feature_cache.get(illumination)
            if features is None:
                from src.preprocessing.preprocess import preprocess_gray

                src_feat, _ = preprocess_gray(prepared["source_gray"], config, illumination)
                ref_feat, _ = preprocess_gray(prepared["reference_gray"], config, illumination)
                features = (src_feat, ref_feat)
                feature_cache[illumination] = features
            scales = config.scale_candidates if config.scale_search and best is None else (1.0,)
            if best is not None and config.scale_search:
                scales = (best["scale"], 1.0)
            trial = search_scales(
                features[0],
                features[1],
                detector=detector,
                max_features=config.max_features,
                rootsift=config.rootsift and detector == "sift",
                ratio=config.ratio,
                scales=scales,
                source_mask=source.mask,
                count_inliers=lambda s, r: quick_inlier_count(
                    s, r, "similarity", config.estimator, max(config.ransac_threshold_px, 3.0), config.rng_seed
                ),
                prior_scale=prior,
            )
            summary = {
                "detector": detector,
                "illumination": illumination,
                "scale": trial["scale"],
                "n_matches": trial["n_matches"],
                "n_inliers": trial["n_inliers"],
            }
            attempt_log.append(summary)
            stage(
                f"Attempt {detector}/{illumination}: scale {trial['scale']:.3f}, "
                f"{trial['n_matches']} matches, {trial['n_inliers']} quick inliers."
            )
            trial["attempt_log"] = attempt_log
            trial["illumination"] = illumination
            trial["detector"] = detector
            trial["feature_source"] = features[0]
            trial["feature_reference"] = features[1]
            if best is None or trial["n_inliers"] > best["n_inliers"]:
                best = trial
            if best["n_inliers"] >= 30 and best["n_matches"] >= 40:
                break
        if best is not None and best["n_inliers"] >= 30:
            break
    assert best is not None
    prepared["source_features"] = best.pop("feature_source")
    prepared["reference_features"] = best.pop("feature_reference")
    prepared["illumination_used"] = best["illumination"]
    return best


def _working_image(image: np.ndarray, mask: np.ndarray | None, max_side: int) -> tuple[np.ndarray, float, np.ndarray | None]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image, 1.0, mask
    scale = max_side / longest
    size = (max(16, int(round(width * scale))), max(16, int(round(height * scale))))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    resized_mask = None if mask is None else cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return resized, scale, resized_mask


def _overlap_from_matrix(matrix: np.ndarray, model: str, source_shape: tuple[int, ...], reference_shape: tuple[int, ...]) -> np.ndarray:
    dummy = np.full(source_shape[:2], 255, np.uint8)
    _, valid = warp_source_to_reference(dummy, reference_shape, matrix, model)
    return valid


def _ground_truth(
    source: ImageBundle,
    reference: ImageBundle,
    scale_src: float,
    scale_ref: float,
) -> tuple[np.ndarray | None, str | None, str]:
    matrix = source.ground_truth_matrix
    model = source.ground_truth_model
    if matrix is None:
        return None, None, "no_ground_truth"
    from src.geometry.transforms import working_transform

    working = working_transform(np.asarray(matrix, dtype=np.float64), model or "similarity", scale_src, scale_ref)
    if source.origin == "synthetic":
        mode = "synthetic_benchmark"
    else:
        mode = "user_supplied_ground_truth"
    return working, model or "similarity", mode
