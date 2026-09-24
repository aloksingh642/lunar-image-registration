"""Shared data types for images, configuration, and pipeline results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np


SENSOR_NOTES = {
    "OHRC": (
        "Orbiter High Resolution Camera: panchromatic, about 0.25 m GSD at nadir "
        "(ISRO also quotes 0.32 m) and a roughly 3 km swath from 100 km. "
        "A large scale difference versus TMC-2, IIRS, SELENE, or even LRO NAC is expected."
    ),
    "TMC-2": (
        "Terrain Mapping Camera-2: panchromatic (0.5–0.8 µm), 5 m resolution, "
        "20 km swath, fore/nadir/aft views near ±25°. Fore and aft are not the same geometry as nadir."
    ),
    "IIRS": (
        "Imaging Infrared Spectrometer: about 80 m spatial resolution, 20 km swath, "
        "0.8–5 µm in ~256 bands at ~20 nm. This prototype reduces a cube to a structural "
        "grayscale image and then matches that image. It does not match spectra, and a "
        "multi-band file is not evidence of spectral correspondence."
    ),
    "Other": "Generic or unspecified source imagery.",
}

REFERENCE_NOTES = {
    "LRO NAC": "Lunar Reconnaissance Orbiter Narrow Angle Camera: about 0.5 m/pixel, ~5 km swath, panchromatic.",
    "SELENE": "SELENE (Kaguya) Terrain Camera: about 10 m/pixel stereo pushbroom imagery.",
    "Other": "Generic or unspecified reference imagery.",
}


@dataclass
class ImageBundle:
    """One loaded image plus whatever metadata actually exists.

    ``origin`` is ``synthetic``, ``user``, or ``unknown``. Synthetic bundles must
    never be described as Chandrayaan-2 or LRO products.
    """

    image: np.ndarray
    origin: str
    sensor: str
    role: str
    metadata: dict[str, Any] = field(default_factory=dict)
    path: str | None = None
    reference_catalog: str | None = None
    mask: np.ndarray | None = None
    ground_truth_matrix: np.ndarray | None = None
    ground_truth_model: str | None = None
    native_image: np.ndarray | None = None

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def channels(self) -> int:
        return 1 if self.image.ndim == 2 else int(self.image.shape[2])


@dataclass
class PipelineConfig:
    detector: str = "sift"
    rootsift: bool = True
    max_features: int = 5000
    ratio: float = 0.80
    scale_search: bool = True
    scale_candidates: tuple[float, ...] = (0.5, 0.75, 1.0, 1.33, 1.75, 2.0, 2.5)
    max_working_side: int = 960
    model: str = "auto"
    estimator: str = "magsac"
    ransac_threshold_px: float = 3.0
    confidence: float = 0.999
    max_iters: int = 4000
    grid_rows: int = 6
    grid_cols: int = 6
    per_cell: int = 3
    min_spacing_px: float = 6.0
    refine: bool = True
    refine_method: str = "ncc"
    template_radius: int = 8
    search_radius: int = 4
    ncc_accept: float = 0.40
    auto_retry: bool = True
    denoise: str = "bilateral"
    contrast: str = "clahe"
    normalize: str = "percentile"
    illumination: str = "auto"
    histogram_match: bool = False
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    make_visuals: bool = True
    max_draw: int = 220
    rng_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["scale_candidates"] = list(self.scale_candidates)
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "PipelineConfig":
        raw = raw or {}
        flat = _flatten_config(raw)
        known = {item.name for item in fields(cls)}
        kwargs = {key: value for key, value in flat.items() if key in known and value is not None}
        if "scale_candidates" in kwargs:
            kwargs["scale_candidates"] = tuple(float(v) for v in kwargs["scale_candidates"])
        return cls(**kwargs)


def _flatten_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept either a flat dict or the nested YAML layout."""
    flat = {key: value for key, value in raw.items() if not isinstance(value, dict)}
    pre = raw.get("preprocess") or {}
    geo = raw.get("geometry") or {}
    spatial = raw.get("spatial") or {}
    ref = raw.get("refinement") or {}
    mapping = {
        "denoise": pre.get("denoise"),
        "contrast": pre.get("contrast"),
        "normalize": pre.get("normalize"),
        "illumination": pre.get("illumination"),
        "histogram_match": pre.get("histogram_match"),
        "clahe_clip": pre.get("clahe_clip"),
        "clahe_grid": pre.get("clahe_grid"),
        "model": geo.get("model"),
        "estimator": geo.get("estimator"),
        "ransac_threshold_px": geo.get("ransac_threshold_px"),
        "confidence": geo.get("confidence"),
        "max_iters": geo.get("max_iters"),
        "grid_rows": spatial.get("grid_rows"),
        "grid_cols": spatial.get("grid_cols"),
        "per_cell": spatial.get("per_cell"),
        "min_spacing_px": spatial.get("min_spacing_px"),
        "refine": ref.get("enabled"),
        "refine_method": ref.get("method"),
        "template_radius": ref.get("template_radius"),
        "search_radius": ref.get("search_radius"),
        "ncc_accept": ref.get("ncc_accept"),
    }
    for key, value in mapping.items():
        if value is not None and key not in flat:
            flat[key] = value
    return flat


def load_default_config() -> PipelineConfig:
    import yaml

    from src.paths import DEFAULT_CONFIG

    if not DEFAULT_CONFIG.exists():
        return PipelineConfig()
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")) or {}
    return PipelineConfig.from_dict(raw)


def image_summary(bundle: ImageBundle) -> dict[str, Any]:
    meta = bundle.metadata or {}
    return {
        "width": bundle.width,
        "height": bundle.height,
        "channels": bundle.channels,
        "dtype": str(bundle.image.dtype),
        "origin": bundle.origin,
        "sensor": bundle.sensor,
        "role": bundle.role,
        "reference_catalog": bundle.reference_catalog,
        "path": bundle.path,
        "gsd_m": meta.get("gsd_m"),
        "sun_azimuth_deg": meta.get("sun_azimuth_deg"),
        "sun_elevation_deg": meta.get("sun_elevation_deg"),
        "incidence_angle_deg": meta.get("incidence_angle_deg"),
        "emission_angle_deg": meta.get("emission_angle_deg"),
        "phase_angle_deg": meta.get("phase_angle_deg"),
        "product_id": meta.get("product_id"),
        "acquisition_time_utc": meta.get("acquisition_time_utc"),
        "georeference": meta.get("georeference"),
        "metadata_kind": meta.get("metadata_kind", "absent" if not meta else "user_supplied"),
        "notes": meta.get("notes"),
        "simulation": meta.get("simulation"),
        "n_bands_original": meta.get("n_bands_original"),
    }
