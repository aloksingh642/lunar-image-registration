"""Scientific dashboard for SIH problem statement 26166.

Run from the project root:

    streamlit run app/ui/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.demo.synthetic import PRESETS, make_pair
from src.evaluation.compare import compare_detectors
from src.evaluation.format import metric_rows
from src.exceptions import LunarRegError
from src.export.exporter import export_result
from src.io.loader import load_image
from src.models import REFERENCE_NOTES, SENSOR_NOTES, PipelineConfig, load_default_config
from src.paths import OUTPUT_DIR
from src.pipeline import run_registration

st.set_page_config(
    page_title="Lunar correspondence · PS 26166",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@460;560;650&family=Source+Sans+3:wght@400;560;640&display=swap');
  html, body, [class*="css"] { font-family: "Source Sans 3", "Segoe UI", sans-serif; }
  h1, h2, h3, .block-title { font-family: "Space Grotesk", "Segoe UI", sans-serif; letter-spacing: -0.03em; }
  .stApp { background: radial-gradient(1100px 520px at 8% -10%, #1b2a4a 0%, #0c1220 42%, #090e18 100%); }
  .block-container { padding-top: 1.1rem; max-width: 1180px; }
  .hero { display: flex; gap: 16px; align-items: flex-start; margin-bottom: 8px; }
  .mark { width: 46px; height: 46px; flex: none; }
  .kicker { color: #e0b15a; font-family: "Space Grotesk", sans-serif; letter-spacing: 0.14em; font-size: 12px; text-transform: uppercase; margin: 0; }
  .hero h1 { margin: 2px 0 4px; font-size: 32px; font-weight: 560; color: #f4f7fb; }
  .sub { color: #9aabc6; margin: 0; max-width: 760px; }
  .banner { border-radius: 12px; padding: 12px 14px; margin: 12px 0 16px; border: 1px solid; }
  .banner strong { font-family: "Space Grotesk", sans-serif; }
  .synthetic { background: #2a2112; border-color: #e0b15a; color: #f6e4bf; }
  .user { background: #122433; border-color: #7eb6ff; color: #d7e7ff; }
  .bad { background: #2c171b; border-color: #e07a7a; color: #ffd5d5; }
  .ok { background: #13261c; border-color: #7dcea0; color: #d9f5e6; }
  .tiles { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin: 8px 0 16px; }
  .tile { background: #141c2e; border: 1px solid #2a3a58; border-radius: 12px; padding: 10px 12px; }
  .tile span { display: block; color: #93a4c2; font-size: 12px; letter-spacing: 0.04em; text-transform: uppercase; }
  .tile b { display: block; color: #f4f7fb; font-family: "Space Grotesk", sans-serif; font-size: 22px; font-weight: 560; margin-top: 2px; }
  .note { color: #93a4c2; font-size: 14px; }
  div[data-testid="stSidebar"] { background: #10182a; border-right: 1px solid #243049; }
  .stTabs [data-baseweb="tab-list"] { gap: 6px; }
  .stTabs [data-baseweb="tab"] { background: #141c2e; border-radius: 8px 8px 0 0; }
  @media (max-width: 900px) { .tiles { grid-template-columns: repeat(2, 1fr); } }
</style>
""",
    unsafe_allow_html=True,
)


def _header() -> None:
    st.markdown(
        """
<div class="hero">
  <svg class="mark" viewBox="0 0 64 64" aria-hidden="true">
    <circle cx="32" cy="32" r="30" fill="#141c2e" stroke="#e0b15a" stroke-width="2"/>
    <path d="M32 6 a26 26 0 1 0 0 52 a18 18 0 1 1 0-52z" fill="#d7deea"/>
    <circle cx="24" cy="28" r="4" fill="#0c1220" opacity="0.35"/>
    <circle cx="40" cy="40" r="6" fill="#0c1220" opacity="0.28"/>
  </svg>
  <div>
    <p class="kicker">ISRO · Smart India Hackathon · PS 26166</p>
    <h1>Lunar correspondence</h1>
    <p class="sub">Multi-modal, sun-angle and scale-aware registration for Chandrayaan-2 class imagery against an LRO or SELENE-style reference. Research prototype. Not an operational or certified product.</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _banner(kind: str, text: str) -> None:
    st.markdown(f'<div class="banner {kind}">{text}</div>', unsafe_allow_html=True)


def _tiles(items: list[tuple[str, str]]) -> None:
    cells = "".join(f"<div class='tile'><span>{label}</span><b>{value}</b></div>" for label, value in items)
    st.markdown(f"<div class='tiles'>{cells}</div>", unsafe_allow_html=True)


def _encode_png(image: np.ndarray) -> bytes:
    array = image
    if array.ndim == 2:
        ok, encoded = cv2.imencode(".png", array)
    else:
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    if not ok:
        raise LunarRegError("Could not encode a PNG for download.")
    return encoded.tobytes()


def _save_upload(uploaded, role: str) -> Path:
    folder = OUTPUT_DIR / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{role}_{uploaded.name}"
    path.write_bytes(uploaded.getvalue())
    return path


def _build_config(base: PipelineConfig) -> PipelineConfig:
    st.sidebar.markdown("### Pipeline")
    detector = st.sidebar.selectbox("Detector", ["sift", "orb", "akaze"], index=["sift", "orb", "akaze"].index(base.detector))
    with st.sidebar.expander("Preprocessing", expanded=False):
        illumination = st.selectbox("Illumination representation", ["auto", "retinex", "gradient", "dog", "none"], index=["auto", "retinex", "gradient", "dog", "none"].index(base.illumination))
        denoise = st.selectbox("Denoise", ["none", "gaussian", "bilateral", "median"], index=["none", "gaussian", "bilateral", "median"].index(base.denoise))
        contrast = st.selectbox("Contrast", ["none", "clahe", "histogram_eq"], index=["none", "clahe", "histogram_eq"].index(base.contrast))
        normalize = st.selectbox("Normalize", ["none", "minmax", "percentile", "zscore"], index=["none", "minmax", "percentile", "zscore"].index(base.normalize))
        histogram_match = st.checkbox("Histogram-match source to reference", value=base.histogram_match)
    with st.sidebar.expander("Matching and scale", expanded=False):
        ratio = st.slider("Lowe ratio", 0.60, 0.95, float(base.ratio), 0.01)
        scale_search = st.checkbox("Explicit scale search", value=base.scale_search)
        auto_retry = st.checkbox("Retry weaker settings if inliers are scarce", value=base.auto_retry)
        max_features = st.slider("Max features per image", 500, 8000, int(base.max_features), 100)
    with st.sidebar.expander("Geometry", expanded=False):
        model = st.selectbox("Transform model", ["auto", "similarity", "affine", "homography"], index=["auto", "similarity", "affine", "homography"].index(base.model))
        estimator = st.selectbox("Robust estimator", ["magsac", "ransac"], index=["magsac", "ransac"].index(base.estimator))
        threshold = st.slider("Reprojection threshold (px)", 0.8, 8.0, float(base.ransac_threshold_px), 0.1)
    with st.sidebar.expander("Spatial distribution", expanded=False):
        grid_rows = st.slider("Grid rows", 2, 12, int(base.grid_rows))
        grid_cols = st.slider("Grid columns", 2, 12, int(base.grid_cols))
        per_cell = st.slider("Matches kept per cell", 1, 8, int(base.per_cell))
        spacing = st.slider("Minimum spacing (px)", 0.0, 30.0, float(base.min_spacing_px), 1.0)
    with st.sidebar.expander("Sub-pixel refinement", expanded=False):
        refine = st.checkbox("Refine correspondences", value=base.refine)
        refine_method = st.selectbox("Method", ["ncc", "cornersubpix"], index=0 if base.refine_method == "ncc" else 1)
        ncc_accept = st.slider("Minimum NCC to accept a shift", 0.2, 0.95, float(base.ncc_accept), 0.05)
    return PipelineConfig(
        detector=detector,
        rootsift=detector == "sift",
        max_features=max_features,
        ratio=ratio,
        scale_search=scale_search,
        scale_candidates=base.scale_candidates,
        max_working_side=base.max_working_side,
        model=model,
        estimator=estimator,
        ransac_threshold_px=threshold,
        confidence=base.confidence,
        max_iters=base.max_iters,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        per_cell=per_cell,
        min_spacing_px=spacing,
        refine=refine,
        refine_method=refine_method,
        template_radius=base.template_radius,
        search_radius=base.search_radius,
        ncc_accept=ncc_accept,
        auto_retry=auto_retry,
        denoise=denoise,
        contrast=contrast,
        normalize=normalize,
        illumination=illumination,
        histogram_match=histogram_match,
        clahe_clip=base.clahe_clip,
        clahe_grid=base.clahe_grid,
        make_visuals=True,
        max_draw=base.max_draw,
        rng_seed=base.rng_seed,
    )


def _load_bundles(mode: str):
    if mode == "Synthetic demo":
        preset = st.session_state["preset"]
        pair = make_pair(preset, seed=int(st.session_state["seed"]), size=int(st.session_state["size"]))
        return pair["source"], pair["reference"], pair
    src_up = st.session_state.get("src_up")
    ref_up = st.session_state.get("ref_up")
    if src_up is None or ref_up is None:
        raise LunarRegError("Upload both a source image and a reference image, or switch to the synthetic demo.")
    src_path = _save_upload(src_up, "source")
    ref_path = _save_upload(ref_up, "reference")
    src_meta = st.session_state.get("src_meta")
    ref_meta = st.session_state.get("ref_meta")
    source = load_image(
        src_path,
        sensor=st.session_state["source_sensor"],
        role="source",
        metadata_path=_save_upload(src_meta, "source_meta") if src_meta is not None else None,
    )
    reference = load_image(
        ref_path,
        sensor=st.session_state["reference_catalog"],
        role="reference",
        reference_catalog=st.session_state["reference_catalog"],
        metadata_path=_save_upload(ref_meta, "reference_meta") if ref_meta is not None else None,
    )
    return source, reference, None


def _run(config: PipelineConfig) -> None:
    source, reference, pair = _load_bundles(st.session_state["mode"])
    result = run_registration(source, reference, config)
    st.session_state["result"] = result
    st.session_state["pair"] = pair
    st.session_state["source_bundle"] = source
    st.session_state["reference_bundle"] = reference


def _fmt(value, digits=2, suffix=""):
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "—"
    return f"{number:.{digits}f}{suffix}"


def _show_result(result) -> None:
    if result.data_origin == "synthetic":
        _banner(
            "synthetic",
            "<strong>Synthetic / demo data.</strong> Procedural crater field. Not Chandrayaan-2, LRO, or SELENE imagery. "
            "Ground-truth errors measure the simulator only.",
        )
    else:
        _banner(
            "user",
            "<strong>User-supplied imagery.</strong> No mission ground truth is assumed. "
            "RMSE is the reprojection residual of the estimated correspondences, not verified lunar accuracy.",
        )
    level = result.metrics.get("reliability", {}).get("level", "unknown")
    if level == "unreliable":
        _banner("bad", "<strong>Reliability: unreliable.</strong> " + " ".join(result.metrics["reliability"].get("reasons", [])))
    elif level == "marginal":
        _banner("synthetic", "<strong>Reliability: marginal.</strong> The fit ran, but the match set is thin or uneven. Do not treat it as a map product.")
    else:
        _banner("ok", "<strong>Reliability: consistent on this pair.</strong> Enough spread-out inliers for this prototype. This is not a certification.")

    ratio = result.metrics["inlier_ratio"]
    _tiles(
        [
            ("Total matches", str(result.metrics["total_matches"])),
            ("Inliers", str(result.metrics["inliers"])),
            ("Inlier ratio", f"{100 * ratio:.2f}%"),
            ("RMSE", _fmt(result.metrics["rmse_px"], 3, " px")),
            ("Spatial coverage", _fmt(result.metrics["spatial_coverage_percent"], 1, "%")),
            ("Refinement NCC", _fmt(result.metrics.get("refinement_mean_ncc"), 3)),
        ]
    )
    if result.warnings:
        with st.expander("Warnings", expanded=level == "unreliable"):
            for warning in result.warnings:
                st.write(warning)

    tabs = st.tabs(
        ["Inputs", "Preprocessing", "Initial matches", "Filtered matches", "Spatial distribution", "Registration", "Metrics", "Export"]
    )
    visuals = result.visuals
    with tabs[0]:
        left, right = st.columns(2)
        left.image(visuals.get("source", result.feature_source), caption="Source", width="stretch")
        right.image(visuals.get("reference", result.feature_reference), caption="Reference", width="stretch")
        c1, c2 = st.columns(2)
        c1.json(result.source_info)
        c2.json(result.reference_info)
        if result.sun_analysis.get("available"):
            kind = result.sun_analysis.get("metadata_kind", "user_supplied")
            st.info(
                f"Sun-angle difference: Δazimuth {result.sun_analysis['delta_azimuth_deg']:.1f}°, "
                f"Δelevation {result.sun_analysis['delta_elevation_deg']:.1f}°. "
                f"Metadata kind: {kind}. {result.sun_analysis['recommendation']}"
            )
        else:
            st.caption(result.sun_analysis.get("reason", "No sun-angle metadata."))
    with tabs[1]:
        st.caption("Feature images. The originals are kept and are what gets warped. Brightness is not the matcher.")
        left, right = st.columns(2)
        left.image(visuals.get("feature_source", result.feature_source), caption="Source feature image", width="stretch")
        right.image(visuals.get("feature_reference", result.feature_reference), caption="Reference feature image", width="stretch")
        st.write(
            f"Pyramid levels (height × width), source: {result.scale_analysis.get('pyramid_levels_source')}. "
            f"Reference: {result.scale_analysis.get('pyramid_levels_reference')}."
        )
        st.write(f"Scale-search trials: {result.scale_analysis.get('trials')}")
        st.caption("The resampling factor is not the geometric scale. Geometric scale is in Metrics.")
    with tabs[2]:
        st.image(visuals["matches_initial"], caption="Descriptor candidates after the ratio test. Not yet geometrically filtered.", width="stretch")
    with tabs[3]:
        st.image(visuals["matches_inliers"], caption="Gold lines passed the robust estimator. Red lines were rejected.", width="stretch")
        st.image(visuals["matches_spatial"], caption="Spatially selected correspondences used to keep the fit from collapsing onto one crater.", width="stretch")
    with tabs[4]:
        st.image(visuals["spatial_distribution"], caption="Gold points were kept. Empty red cells inside the overlap are uncovered.", width="stretch")
        st.write(
            f"Occupied cells {result.metrics['n_occupied_cells']} / overlap cells {result.metrics['n_overlap_cells']}. "
            f"Uniformity {result.metrics['uniformity']:.2f} (1 = even, 0 = one cell)."
        )
    with tabs[5]:
        st.image(visuals["registered"], caption="Source warped into the reference frame. Geometric resample only.", width="stretch")
        left, right = st.columns(2)
        left.image(visuals["overlay"], caption="Alpha blend inside the valid footprint", width="stretch")
        right.image(visuals["checkerboard"], caption="Checkerboard", width="stretch")
        st.image(visuals["difference"], caption="Absolute difference. Dark is closer. Illumination change also creates difference, so this is not a pure error map.", width="stretch")
    with tabs[6]:
        rows = metric_rows(result.metrics, result.gt_metrics)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.markdown("**Adopted transform** `" + result.transform_model + "`")
        st.code(np.array2string(result.transform_working, precision=5))
        st.json(result.transform_params)
        if result.gt_metrics:
            st.markdown("**Ground truth comparison** — simulator or user file, not a lunar survey.")
            st.json({key: value for key, value in result.gt_metrics.items() if key not in {"estimated_parameters", "ground_truth_parameters"}})
        with st.expander("How the numbers are calculated"):
            for key, text in result.metrics["definitions"].items():
                st.markdown(f"**{key}.** {text}")
        with st.expander("Run log"):
            st.code("\n".join(result.log))
    with tabs[7]:
        st.caption("CSV and JSON are the scientific exports. PNG figures are for inspection.")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if st.button("Write experiment folder", type="primary"):
            folder = OUTPUT_DIR / f"experiment_{stamp}"
            written = export_result(result, folder)
            st.session_state["export_dir"] = str(folder)
            st.success(f"Wrote {folder}")
            st.json(written)
        corr = _correspondence_csv(result)
        st.download_button("Download correspondences.csv", corr, file_name="correspondences.csv", mime="text/csv")
        payload = {
            "evaluation_mode": result.evaluation_mode,
            "disclaimer": result.disclaimer,
            "metrics": result.metrics,
            "ground_truth": result.gt_metrics,
            "transform_model": result.transform_model,
            "transform_working": result.transform_working.tolist(),
            "config": result.config,
            "warnings": result.warnings,
        }
        st.download_button(
            "Download metrics.json",
            json.dumps(payload, indent=2, default=_json_default),
            file_name="metrics.json",
            mime="application/json",
        )
        c1, c2, c3 = st.columns(3)
        c1.download_button("Registered PNG", _encode_png(result.registered), file_name="registered.png")
        c2.download_button("Overlay PNG", _encode_png(visuals["overlay"]), file_name="overlay.png")
        c3.download_button("Checkerboard PNG", _encode_png(visuals["checkerboard"]), file_name="checkerboard.png")


def _correspondence_csv(result) -> str:
    n = len(result.refined_src)
    frame = pd.DataFrame(
        {
            "source_x": result.refined_src[:, 0] if n else [],
            "source_y": result.refined_src[:, 1] if n else [],
            "reference_x": result.refined_ref[:, 0] if n else [],
            "reference_y": result.refined_ref[:, 1] if n else [],
        }
    )
    return frame.to_csv(index=False)


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _landing() -> None:
    left, right = st.columns([1.2, 0.8])
    with left:
        st.markdown("### What this prototype does")
        st.markdown(
            """
Correspondence means a pair of pixel locations, one in the source and one in the reference, that are believed to see the same lunar point.
Registration means estimating a transform from those pairs and resampling the source into the reference frame.

Lunar pairs break ordinary template matching because:

- the Sun moves, so crater walls swap between bright and dark
- the spacecraft altitude and the camera GSD change the scale
- OHRC, TMC-2, IIRS, LRO NAC, and SELENE do not share a resolution or a spectral response
- the strongest matches often pile up on one crisp crater and leave the rest of the overlap unconstrained
"""
        )
        st.markdown("### What it does not claim")
        st.markdown(
            "It is not ISRO-certified, not a DEM orthorectifier, not a spectral matcher for IIRS, and not evidence of sub-pixel accuracy on real Chandrayaan-2 data. A decimal coordinate is not accuracy. RMSE of the inliers is a residual, because those inliers were chosen to fit the model."
        )
    with right:
        st.markdown("### Instruments, briefly")
        st.markdown(f"**OHRC.** {SENSOR_NOTES['OHRC']}")
        st.markdown(f"**TMC-2.** {SENSOR_NOTES['TMC-2']}")
        st.markdown(f"**IIRS.** {SENSOR_NOTES['IIRS']}")
        st.markdown(f"**LRO NAC.** {REFERENCE_NOTES['LRO NAC']}")
        st.markdown(f"**SELENE.** {REFERENCE_NOTES['SELENE']}")
    st.caption("Start with the synthetic baseline. Real files can be loaded from the sidebar once you have them.")


def main() -> None:
    _header()
    base = load_default_config()
    st.sidebar.markdown("### Data")
    mode = st.sidebar.radio("Experiment", ["Synthetic demo", "Upload images"], key="mode")
    if mode == "Synthetic demo":
        _banner("synthetic", "<strong>Current experiment: synthetic / demo.</strong> Nothing on this setting is a mission product.")
        st.sidebar.selectbox("Scenario", list(PRESETS), key="preset", format_func=lambda key: PRESETS[key]["title"])
        st.sidebar.caption(PRESETS[st.session_state["preset"]]["description"])
        st.sidebar.number_input("Seed", 0, 9999, 7, key="seed")
        st.sidebar.slider("Image size", 320, 768, 560, 32, key="size")
        st.sidebar.selectbox("Label source as", ["OHRC", "TMC-2", "IIRS", "Other"], key="source_sensor", index=0, disabled=True)
        st.sidebar.caption("Sensor labels on the synthetic path come from the scenario, not from a spacecraft.")
    else:
        _banner("user", "<strong>Current experiment: user imagery.</strong> Metadata is used only when a file actually contains it.")
        st.session_state["src_up"] = st.sidebar.file_uploader("Source image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
        st.session_state["ref_up"] = st.sidebar.file_uploader("Reference image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
        st.sidebar.selectbox("Source sensor", ["OHRC", "TMC-2", "IIRS", "Other"], key="source_sensor")
        st.sidebar.selectbox("Reference catalog", ["LRO NAC", "SELENE", "Other"], key="reference_catalog")
        st.sidebar.caption(SENSOR_NOTES[st.session_state["source_sensor"]])
        st.sidebar.caption(REFERENCE_NOTES[st.session_state["reference_catalog"]])
        st.session_state["src_meta"] = st.sidebar.file_uploader("Optional source metadata JSON", type=["json", "yaml", "yml"])
        st.session_state["ref_meta"] = st.sidebar.file_uploader("Optional reference metadata JSON", type=["json", "yaml", "yml"])
    config = _build_config(base)
    run = st.sidebar.button("Run registration", type="primary", width="stretch")
    compare = st.sidebar.button("Compare SIFT / ORB / AKAZE", width="stretch")
    if run:
        with st.spinner("Matching, rejecting outliers, distributing points, refining, and warping…"):
            try:
                _run(config)
            except LunarRegError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"The pipeline failed: {exc}")
                st.exception(exc)
    if compare:
        try:
            source, reference, _pair = _load_bundles(mode)
            with st.spinner("Running three detectors without scale search or refinement…"):
                rows = compare_detectors(source, reference, config)
            st.session_state["comparison"] = rows
        except LunarRegError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
    if st.session_state.get("comparison"):
        st.markdown("### Detector comparison")
        st.caption("Scale search and refinement are off in this table so the detectors are compared on the same images. A blank ground-truth cell means no known transform was loaded.")
        st.dataframe(pd.DataFrame(st.session_state["comparison"]), width="stretch", hide_index=True)
    result = st.session_state.get("result")
    if result is None:
        _landing()
    else:
        _show_result(result)


if __name__ == "__main__":
    main()
else:
    main()
