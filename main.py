"""Command-line entry point for the lunar correspondence prototype.

Examples
--------
python main.py demo --preset baseline --out outputs/demo_run
python main.py benchmark --out outputs/benchmark
python main.py register --source data/demo/baseline/synthetic_source.png \\
    --reference data/demo/baseline/synthetic_reference.png --out outputs/user_run
python main.py generate-demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.demo.synthetic import PRESETS, make_pair
from src.evaluation.format import metric_rows
from src.exceptions import LunarRegError
from src.export.exporter import export_result
from src.io.loader import load_image, save_image
from src.models import PipelineConfig, load_default_config
from src.paths import DEMO_DIR, ensure_output_dir
from src.pipeline import run_registration


def main() -> int:
    parser = argparse.ArgumentParser(description="Lunar image correspondence prototype (SIH PS 26166).")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Run one synthetic scenario and export it.")
    demo.add_argument("--preset", default="baseline", choices=sorted(PRESETS))
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--size", type=int, default=640)
    demo.add_argument("--out", default="outputs/demo_run")
    demo.add_argument("--model", default=None, help="Override the transform model.")

    bench = sub.add_parser("benchmark", help="Run the synthetic scenarios and write a CSV summary.")
    bench.add_argument("--out", default="outputs/benchmark")
    bench.add_argument("--size", type=int, default=512)
    bench.add_argument("--seeds", default="7,11")

    reg = sub.add_parser("register", help="Register two image files.")
    reg.add_argument("--source", required=True)
    reg.add_argument("--reference", required=True)
    reg.add_argument("--source-sensor", default="Other")
    reg.add_argument("--reference-catalog", default="Other")
    reg.add_argument("--source-metadata", default=None)
    reg.add_argument("--reference-metadata", default=None)
    reg.add_argument("--out", default="outputs/register_run")
    reg.add_argument("--model", default=None)

    gen = sub.add_parser("generate-demo", help="Write labeled synthetic pairs under data/demo.")
    gen.add_argument("--size", type=int, default=640)
    gen.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()
    try:
        if args.command == "demo":
            return run_demo(args)
        if args.command == "benchmark":
            return run_benchmark(args)
        if args.command == "register":
            return run_register(args)
        if args.command == "generate-demo":
            return generate_demo(args.size, args.seed)
    except LunarRegError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def run_demo(args: argparse.Namespace) -> int:
    pair = make_pair(args.preset, seed=args.seed, size=args.size)
    config = load_default_config()
    if args.model:
        config.model = args.model
    elif pair["ground_truth_model"] == "affine":
        config.model = "affine"
    result = run_registration(pair["source"], pair["reference"], config)
    written = export_result(result, args.out)
    _print_summary(result)
    print(f"exported: {written['report.html']}")
    return 0 if result.metrics.get("reliability", {}).get("level") != "unreliable" else 1


def run_register(args: argparse.Namespace) -> int:
    source = load_image(
        args.source,
        sensor=args.source_sensor,
        role="source",
        metadata_path=args.source_metadata,
    )
    reference = load_image(
        args.reference,
        sensor=args.reference_catalog,
        role="reference",
        reference_catalog=args.reference_catalog,
        metadata_path=args.reference_metadata,
    )
    config = load_default_config()
    if args.model:
        config.model = args.model
    result = run_registration(source, reference, config)
    written = export_result(result, args.out)
    _print_summary(result)
    print(f"exported: {written['report.html']}")
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    import pandas as pd

    out = ensure_output_dir(Path(args.out))
    seeds = [int(item) for item in str(args.seeds).split(",") if item]
    rows = []
    for seed in seeds:
        for preset, spec in PRESETS.items():
            pair = make_pair(preset, seed=seed, size=args.size)
            config = load_default_config()
            config.make_visuals = False
            config.max_working_side = args.size
            if spec["model"] == "affine":
                config.model = "affine"
            try:
                result = run_registration(pair["source"], pair["reference"], config)
                metrics = result.metrics
                gt = result.gt_metrics or {}
                rows.append(
                    {
                        "preset": preset,
                        "seed": seed,
                        "evaluation_mode": result.evaluation_mode,
                        "matches": metrics["total_matches"],
                        "inliers": metrics["inliers"],
                        "inlier_ratio": metrics["inlier_ratio"],
                        "rmse_px": metrics["rmse_px"],
                        "spatial_coverage": metrics["spatial_coverage"],
                        "uniformity": metrics["uniformity"],
                        "estimated_scale": metrics.get("estimated_scale"),
                        "gt_corner_rmse_px": gt.get("corner_rmse_px"),
                        "gt_scale_error_percent": gt.get("scale_error_percent"),
                        "gt_rotation_error_deg": gt.get("rotation_error_deg"),
                        "gt_translation_error_px": gt.get("translation_error_px"),
                        "localization_initial_px": gt.get("localization_rmse_initial_px"),
                        "localization_refined_px": gt.get("localization_rmse_refined_px"),
                        "refinement_mean_ncc": metrics.get("refinement_mean_ncc"),
                        "overlap_correlation": metrics.get("overlap_intensity_correlation"),
                        "reliability": metrics.get("reliability", {}).get("level"),
                        "model": result.transform_model,
                        "error": "",
                    }
                )
                print(
                    f"{preset:18} seed {seed}  inliers {metrics['inliers']:4}  "
                    f"corner {gt.get('corner_rmse_px', float('nan')):7.3f} px  "
                    f"{metrics.get('reliability', {}).get('level')}"
                )
            except LunarRegError as exc:
                rows.append({"preset": preset, "seed": seed, "error": str(exc), "reliability": "failed"})
                print(f"{preset:18} seed {seed}  FAILED  {exc}")
    frame = pd.DataFrame(rows)
    csv_path = out / "benchmark.csv"
    frame.to_csv(csv_path, index=False)
    (out / "benchmark.json").write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")
    note = {
        "label": "synthetic_benchmark",
        "disclaimer": "Procedural crater fields. Not Chandrayaan-2, LRO, or SELENE results.",
        "size": args.size,
        "seeds": seeds,
        "config": load_default_config().to_dict(),
    }
    (out / "benchmark_note.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    print(f"wrote {csv_path}")
    return 0


def generate_demo(size: int, seed: int) -> int:
    for preset in ("baseline", "sun_angle", "large_scale"):
        pair = make_pair(preset, seed=seed, size=size)
        folder = DEMO_DIR / preset
        folder.mkdir(parents=True, exist_ok=True)
        save_image(folder / "synthetic_source.png", pair["source"].image)
        save_image(folder / "synthetic_reference.png", pair["reference"].image)
        _write_json(folder / "synthetic_source.json", pair["source"].metadata)
        _write_json(folder / "synthetic_reference.json", pair["reference"].metadata)
        _write_json(
            folder / "ground_truth.json",
            {
                "model": pair["ground_truth_model"],
                "matrix": pair["ground_truth_matrix"].tolist(),
                "disclaimer": pair["disclaimer"],
                "parameters": pair["parameters"],
            },
        )
        print(f"wrote {folder}")
    return 0


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_summary(result) -> None:
    print(f"mode: {result.evaluation_mode}   origin: {result.data_origin}")
    print(result.disclaimer)
    for row in metric_rows(result.metrics, result.gt_metrics):
        print(f"  {row['label']:<28} {row['value']}")
    for warning in result.warnings:
        print(f"  warning: {warning}")


if __name__ == "__main__":
    raise SystemExit(main())
