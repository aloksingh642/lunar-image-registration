"""Write a self-contained experiment folder. Numbers come from the run, never from examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.format import metric_rows
from src.io.loader import save_image
from src.pipeline import PipelineResult


def export_result(result: PipelineResult, out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    def put_image(name: str, image: np.ndarray) -> None:
        path = out / name
        save_image(path, image)
        written[name] = str(path)

    put_image("registered.png", result.registered)
    put_image("registered_mask.png", result.registered_mask)
    for key, image in result.visuals.items():
        put_image(f"{key}.png", image)

    correspondences = _correspondence_frame(result)
    csv_path = out / "correspondences.csv"
    correspondences.to_csv(csv_path, index=False)
    written["correspondences.csv"] = str(csv_path)

    candidates = _candidate_frame(result)
    all_path = out / "matches_all.csv"
    candidates.to_csv(all_path, index=False)
    written["matches_all.csv"] = str(all_path)

    payload = {
        "evaluation_mode": result.evaluation_mode,
        "data_origin": result.data_origin,
        "disclaimer": result.disclaimer,
        "metrics": _jsonable(result.metrics),
        "ground_truth": _jsonable(result.gt_metrics),
        "transform_model": result.transform_model,
        "transform_working": result.transform_working.tolist(),
        "transform_full_loaded_pixels": result.transform_full.tolist(),
        "transform_parameters": _jsonable(result.transform_params),
        "working_scale_source": result.working_scale_source,
        "working_scale_reference": result.working_scale_reference,
        "source": _jsonable(result.source_info),
        "reference": _jsonable(result.reference_info),
        "sun_analysis": _jsonable(result.sun_analysis),
        "scale_analysis": _jsonable(result.scale_analysis),
        "warnings": result.warnings,
        "log": result.log,
        "attempts": result.attempts,
        "config": result.config,
        "spatial_summary": _jsonable({k: v for k, v in result.spatial.items() if k != "cell_rc"}),
    }
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    written["metrics.json"] = str(metrics_path)

    config_path = out / "config.json"
    config_path.write_text(json.dumps(result.config, indent=2), encoding="utf-8")
    written["config.json"] = str(config_path)

    report = _markdown_report(result)
    report_path = out / "report.md"
    report_path.write_text(report, encoding="utf-8")
    written["report.md"] = str(report_path)

    html_path = out / "report.html"
    html_path.write_text(_html_report(result, out), encoding="utf-8")
    written["report.html"] = str(html_path)
    return written


def _correspondence_frame(result: PipelineResult) -> pd.DataFrame:
    n = len(result.refined_src) if len(result.refined_src) else len(result.final_src)
    src0 = result.final_src
    ref0 = result.final_ref
    if len(src0) != n:
        src0 = np.resize(src0, (n, 2)) if len(src0) else np.full((n, 2), np.nan)
        ref0 = np.resize(ref0, (n, 2)) if len(ref0) else np.full((n, 2), np.nan)
    # Prefer exact alignment: if refinement changed the count, initial columns may differ.
    if len(result.final_src) == n:
        src0, ref0 = result.final_src, result.final_ref
    else:
        src0 = np.full((n, 2), np.nan)
        ref0 = np.full((n, 2), np.nan)
    ncc = np.full(n, np.nan)
    shift = np.full(n, np.nan)
    accepted = np.zeros(n, dtype=bool)
    if result.refinement is not None and len(result.refinement["ncc"]) == len(result.final_src) and len(result.final_src) == n:
        ncc = result.refinement["ncc"]
        shift = result.refinement["shift_px"]
        accepted = result.refinement["accepted"]
    return pd.DataFrame(
        {
            "source_x": result.refined_src[:, 0] if n else [],
            "source_y": result.refined_src[:, 1] if n else [],
            "reference_x": result.refined_ref[:, 0] if n else [],
            "reference_y": result.refined_ref[:, 1] if n else [],
            "source_x_before_refinement": src0[:, 0] if n else [],
            "source_y_before_refinement": src0[:, 1] if n else [],
            "reference_x_before_refinement": ref0[:, 0] if n else [],
            "reference_y_before_refinement": ref0[:, 1] if n else [],
            "ncc": ncc,
            "refinement_shift_px": shift,
            "refinement_accepted": accepted.astype(int),
            "coordinate_frame": "working_image_pixels",
        }
    )


def _candidate_frame(result: PipelineResult) -> pd.DataFrame:
    n = len(result.src_points)
    selected = np.zeros(n, dtype=bool)
    if len(result.selected_index):
        selected[result.selected_index] = True
    cell_rc = result.spatial.get("cell_rc")
    if cell_rc is None or len(cell_rc) != n:
        rows = np.full(n, -1)
        cols = np.full(n, -1)
    else:
        rows = cell_rc[:, 0]
        cols = cell_rc[:, 1]
    return pd.DataFrame(
        {
            "source_x": result.src_points[:, 0] if n else [],
            "source_y": result.src_points[:, 1] if n else [],
            "reference_x": result.ref_points[:, 0] if n else [],
            "reference_y": result.ref_points[:, 1] if n else [],
            "descriptor_distance": result.distances if n else [],
            "inlier": result.inlier_mask.astype(int) if n else [],
            "spatially_selected": selected.astype(int),
            "grid_row": rows,
            "grid_col": cols,
        }
    )


def _markdown_report(result: PipelineResult) -> str:
    lines = [
        "# Lunar correspondence experiment",
        "",
        f"**Evaluation mode:** `{result.evaluation_mode}`",
        "",
        f"> {result.disclaimer}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for row in metric_rows(result.metrics, result.gt_metrics):
        lines.append(f"| {row['label']} | {row['value']} |")
    lines.extend(["", "## How to read the numbers", ""])
    for key, text in result.metrics.get("definitions", {}).items():
        lines.append(f"- **{key}:** {text}")
    lines.extend(["", "## Warnings", ""])
    if result.warnings:
        lines.extend(f"- {item}" for item in result.warnings)
    else:
        lines.append("- None.")
    lines.extend(["", "## Log", "", "```", *result.log, "```", ""])
    lines.append("The warped image is a geometric resample, not a radiometrically calibrated or orthorectified product.")
    return "\n".join(lines) + "\n"


def _html_report(result: PipelineResult, out: Path) -> str:
    import base64

    def img(name: str) -> str:
        path = out / f"{name}.png"
        if not path.exists():
            return ""
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<figure><img src="data:image/png;base64,{encoded}" alt="{name}"><figcaption>{name.replace("_", " ")}</figcaption></figure>'

    rows = "".join(
        f"<tr><th>{row['label']}</th><td>{row['value']}</td><td>{row['note']}</td></tr>"
        for row in metric_rows(result.metrics, result.gt_metrics)
    )
    warnings = "".join(f"<li>{item}</li>" for item in result.warnings) or "<li>None.</li>"
    figures = "".join(
        img(name)
        for name in (
            "source",
            "reference",
            "matches_initial",
            "matches_inliers",
            "spatial_distribution",
            "registered",
            "overlay",
            "checkerboard",
            "difference",
        )
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Lunar correspondence report</title>
<style>
  body {{ margin: 0; background: #0c1220; color: #e7eef8; font-family: "Segoe UI", sans-serif; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 64px; }}
  h1 {{ font-weight: 560; letter-spacing: -0.03em; }}
  .banner {{ border: 1px solid #e0b15a; background: #2a2112; color: #f3dfb4; padding: 12px 14px; border-radius: 10px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 18px 0 28px; }}
  th, td {{ text-align: left; vertical-align: top; padding: 8px 10px; border-bottom: 1px solid #243049; }}
  th {{ width: 220px; color: #e0b15a; font-weight: 560; }}
  figure {{ margin: 16px 0; }}
  img {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid #243049; }}
  figcaption {{ color: #93a0b8; font-size: 13px; margin-top: 6px; }}
  .mode {{ color: #7eb6ff; font-family: ui-monospace, monospace; }}
</style>
</head>
<body>
<main>
  <p class="mode">PS 26166 · {result.evaluation_mode} · {result.data_origin}</p>
  <h1>Lunar correspondence report</h1>
  <p class="banner">{result.disclaimer}</p>
  <h2>Metrics</h2>
  <table>{rows}</table>
  <h2>Warnings</h2>
  <ul>{warnings}</ul>
  <h2>Figures</h2>
  {figures}
  <p>Geometric resample only. Not radiometric calibration, orthorectification, or an ISRO product.</p>
</main>
</body>
</html>
"""


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
