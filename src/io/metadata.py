"""Optional metadata. Missing fields are normal and must not stop registration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.exceptions import ImageLoadError


_SUN_KEYS = ("sun_azimuth_deg", "sun_elevation_deg")
_OPTIONAL_KEYS = (
    "origin",
    "sensor",
    "reference_catalog",
    "gsd_m",
    "sun_azimuth_deg",
    "sun_elevation_deg",
    "incidence_angle_deg",
    "emission_angle_deg",
    "phase_angle_deg",
    "acquisition_time_utc",
    "product_id",
    "notes",
    "metadata_kind",
    "width",
    "height",
    "dtype",
    "offset_bytes",
    "byte_order",
    "simulation",
    "ground_truth_model",
)


def load_metadata_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ImageLoadError(f"Metadata file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ImageLoadError(f"Could not parse metadata file {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ImageLoadError(f"Metadata file {path.name} must contain a JSON/YAML object.")
    return data


def discover_sidecar(image_path: Path) -> Path | None:
    candidates = [
        image_path.with_suffix(".json"),
        image_path.with_suffix(".yaml"),
        image_path.with_suffix(".yml"),
        image_path.parent / f"{image_path.stem}_metadata.json",
        image_path.parent / "metadata.json",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate != image_path:
            return candidate
    return None


def discover_ground_truth(image_path: Path) -> Path | None:
    candidates = [
        image_path.parent / "ground_truth.json",
        image_path.parent / f"{image_path.stem}_ground_truth.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def sanitize_metadata(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Keep known keys and drop nulls. Extra keys are preserved under ``extra``."""
    if not raw:
        return {}
    clean: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None or value == "":
            continue
        if key in _OPTIONAL_KEYS or key in {"georeference", "ground_truth_matrix"}:
            clean[key] = value
        else:
            extra[key] = value
    if extra:
        clean["extra"] = extra
    return clean


def sun_angle_analysis(source_meta: dict[str, Any], reference_meta: dict[str, Any]) -> dict[str, Any]:
    """Compare sun angles when both sides actually provide them.

    A large difference is a reason to prefer a gradient/retinex representation.
    It is not used as a Hapke photometric correction: that would need a DEM.
    """
    if not all(key in source_meta and key in reference_meta for key in _SUN_KEYS):
        return {
            "available": False,
            "reason": "Sun azimuth/elevation metadata is missing on at least one image.",
        }
    try:
        src_az = float(source_meta["sun_azimuth_deg"])
        ref_az = float(reference_meta["sun_azimuth_deg"])
        src_el = float(source_meta["sun_elevation_deg"])
        ref_el = float(reference_meta["sun_elevation_deg"])
    except (TypeError, ValueError):
        return {"available": False, "reason": "Sun-angle metadata is not numeric."}

    delta_az = abs((src_az - ref_az + 180.0) % 360.0 - 180.0)
    delta_el = abs(src_el - ref_el)
    large = delta_az >= 35.0 or delta_el >= 12.0
    return {
        "available": True,
        "source_azimuth_deg": src_az,
        "source_elevation_deg": src_el,
        "reference_azimuth_deg": ref_az,
        "reference_elevation_deg": ref_el,
        "delta_azimuth_deg": delta_az,
        "delta_elevation_deg": delta_el,
        "large_difference": large,
        "metadata_kind": source_meta.get("metadata_kind", "user_supplied"),
        "recommendation": (
            "Large simulated or reported sun-angle difference. "
            "Brightness matching alone is not used. A retinex or gradient representation is preferred."
            if large
            else "Sun-angle difference is modest. CLAHE plus a gradient descriptor is still used; brightness is not the matcher."
        ),
    }


def scale_prior_from_gsd(source_meta: dict[str, Any], reference_meta: dict[str, Any]) -> dict[str, Any]:
    """Expected similarity scale from ground sample distance, if both are known.

    A transform maps source pixels onto reference pixels. One source pixel covers
    ``gsd_source`` metres, which is ``gsd_source / gsd_reference`` reference pixels.
    """
    if source_meta.get("gsd_m") in (None, "") or reference_meta.get("gsd_m") in (None, ""):
        return {"available": False, "reason": "GSD / pixel resolution metadata is missing."}
    try:
        gsd_s = float(source_meta["gsd_m"])
        gsd_r = float(reference_meta["gsd_m"])
    except (TypeError, ValueError):
        return {"available": False, "reason": "GSD metadata is not numeric."}
    if gsd_s <= 0 or gsd_r <= 0:
        return {"available": False, "reason": "GSD must be positive."}
    return {
        "available": True,
        "gsd_source_m": gsd_s,
        "gsd_reference_m": gsd_r,
        "expected_scale_source_to_reference": gsd_s / gsd_r,
        "note": "Used only to center the scale search. It is not assumed to be exact.",
    }
