"""Procedural cratered terrain with a known geometric transform.

The images are shaded relief plus albedo. They are not Chandrayaan-2, LRO, or
SELENE products, and the metadata is labeled as simulation parameters.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.geometry.transforms import similarity_matrix
from src.models import ImageBundle


PRESETS: dict[str, dict[str, Any]] = {
    "baseline": {
        "title": "Baseline similarity",
        "description": "Moderate scale, rotation, and a small sun-angle change.",
        "scale": 1.18,
        "rotation_deg": 8.0,
        "translation": (18.0, -12.0),
        "sun_reference": (38.0, 28.0),
        "sun_source": (58.0, 22.0),
        "noise_sigma": 3.0,
        "model": "similarity",
        "cross_sensor": False,
        "source_sensor": "OHRC",
        "reference_catalog": "LRO NAC",
    },
    "sun_angle": {
        "title": "Strong sun-angle change",
        "description": "Same kind of terrain, with a large azimuth and elevation difference.",
        "scale": 1.12,
        "rotation_deg": 6.0,
        "translation": (14.0, 16.0),
        "sun_reference": (30.0, 33.0),
        "sun_source": (84.0, 18.0),
        "noise_sigma": 4.0,
        "model": "similarity",
        "cross_sensor": False,
        "source_sensor": "TMC-2",
        "reference_catalog": "LRO NAC",
    },
    "large_scale": {
        "title": "Large scale difference",
        "description": "Source is sampled much more coarsely than the reference.",
        "scale": 0.48,
        "rotation_deg": 11.0,
        "translation": (10.0, -8.0),
        "sun_reference": (40.0, 30.0),
        "sun_source": (55.0, 24.0),
        "noise_sigma": 3.0,
        "model": "similarity",
        "cross_sensor": False,
        "source_sensor": "TMC-2",
        "reference_catalog": "SELENE",
    },
    "viewpoint_affine": {
        "title": "Viewpoint-like affine",
        "description": "Shear and unequal axis scale, a 2D stand-in for a mild viewpoint change. Not a camera model.",
        "scale": 1.15,
        "rotation_deg": 9.0,
        "translation": (12.0, 6.0),
        "shear": 0.08,
        "axis_ratio": 0.94,
        "sun_reference": (48.0, 26.0),
        "sun_source": (78.0, 18.0),
        "noise_sigma": 3.5,
        "model": "affine",
        "cross_sensor": False,
        "source_sensor": "OHRC",
        "reference_catalog": "LRO NAC",
    },
    "cross_sensor": {
        "title": "Cross-sensor simulation",
        "description": "Blur, gamma, and pushbroom-like striping. A structural stand-in for a modality gap, not an IIRS cube.",
        "scale": 0.72,
        "rotation_deg": 5.0,
        "translation": (8.0, 11.0),
        "sun_reference": (36.0, 32.0),
        "sun_source": (70.0, 16.0),
        "noise_sigma": 5.0,
        "model": "similarity",
        "cross_sensor": True,
        "source_sensor": "IIRS",
        "reference_catalog": "LRO NAC",
    },
}


def make_pair(
    preset: str = "baseline",
    seed: int = 7,
    size: int = 640,
    **overrides: Any,
) -> dict[str, Any]:
    if preset not in PRESETS:
        known = ", ".join(PRESETS)
        raise ValueError(f"Unknown synthetic preset '{preset}'. Choose one of: {known}.")
    spec = {**PRESETS[preset], **overrides}
    spec["preset"] = preset
    rng = np.random.default_rng(seed)
    canvas = int(size * 1.7)
    dem = render_dem(canvas, canvas, seed)
    albedo = render_albedo(canvas, canvas, seed + 19, dem)
    ref_sun = spec["sun_reference"]
    src_sun = spec["sun_source"]
    shaded_ref = shade_dem(dem, albedo, ref_sun[0], ref_sun[1])
    shaded_src = shade_dem(dem, albedo, src_sun[0], src_sun[1])

    origin = ((canvas - size) // 2, (canvas - size) // 2)
    reference = shaded_ref[origin[1] : origin[1] + size, origin[0] : origin[0] + size]
    matrix = _ground_truth_matrix(spec, size)
    source, valid = _sample_source(shaded_src, matrix, origin, size)
    if spec.get("cross_sensor"):
        source = simulate_cross_sensor(source, rng)
    source = _add_noise(source, rng, float(spec.get("noise_sigma", 0.0)))
    valid = cv2.erode(valid, np.ones((5, 5), np.uint8))

    disclaimer = (
        "Synthetic lunar-like terrain for software development. "
        "Not Chandrayaan-2, LRO, SELENE, or any other mission product."
    )
    simulation = {
        "preset": preset,
        "seed": seed,
        "size": size,
        "scale": spec["scale"],
        "rotation_deg": spec["rotation_deg"],
        "translation_px": list(spec["translation"]),
        "sun_source": list(src_sun),
        "sun_reference": list(ref_sun),
        "cross_sensor_simulation": bool(spec.get("cross_sensor")),
        "disclaimer": disclaimer,
    }
    source_meta = {
        "origin": "synthetic",
        "metadata_kind": "simulation_parameter",
        "sensor": spec["source_sensor"],
        "sun_azimuth_deg": src_sun[0],
        "sun_elevation_deg": src_sun[1],
        "notes": disclaimer,
        "simulation": simulation,
        "product_id": None,
    }
    reference_meta = {
        "origin": "synthetic",
        "metadata_kind": "simulation_parameter",
        "sensor": spec["reference_catalog"],
        "reference_catalog": spec["reference_catalog"],
        "sun_azimuth_deg": ref_sun[0],
        "sun_elevation_deg": ref_sun[1],
        "notes": disclaimer,
        "simulation": simulation,
        "product_id": None,
    }
    source_bundle = ImageBundle(
        image=source,
        origin="synthetic",
        sensor=spec["source_sensor"],
        role="source",
        metadata=source_meta,
        reference_catalog=None,
        mask=valid,
        ground_truth_matrix=matrix,
        ground_truth_model=spec["model"],
    )
    reference_bundle = ImageBundle(
        image=reference,
        origin="synthetic",
        sensor=spec["reference_catalog"],
        role="reference",
        metadata=reference_meta,
        reference_catalog=spec["reference_catalog"],
    )
    return {
        "source": source_bundle,
        "reference": reference_bundle,
        "ground_truth_matrix": matrix,
        "ground_truth_model": spec["model"],
        "parameters": simulation,
        "description": spec["description"],
        "title": spec["title"],
        "disclaimer": disclaimer,
    }


def render_dem(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Broad mare-like undulation stays smaller than the craters, or shading
    # turns into blobs and the rims disappear at high sun elevation.
    dem = fbm(height, width, rng, octaves=4, cell0=140, persistence=0.55) * 0.12
    dem += fbm(height, width, np.random.default_rng(seed + 3), octaves=3, cell0=16, persistence=0.5) * 0.035
    yy, xx = np.mgrid[0:height, 0:width]
    for _ in range(4):
        x0, y0 = rng.uniform(0, width), rng.uniform(0, height)
        angle = rng.uniform(0, np.pi)
        length = rng.uniform(width * 0.3, width * 0.75)
        x1, y1 = x0 + np.cos(angle) * length, y0 + np.sin(angle) * length
        dist = _segment_distance(xx, yy, x0, y0, x1, y1)
        width_px = rng.uniform(8, 18)
        dem += rng.uniform(0.04, 0.10) * np.exp(-0.5 * (dist / width_px) ** 2)

    n_craters = max(36, (height * width) // 11000)
    for _ in range(n_craters):
        cx = rng.uniform(-0.04 * width, 1.04 * width)
        cy = rng.uniform(-0.04 * height, 1.04 * height)
        radius = float(10 ** rng.uniform(np.log10(5), np.log10(max(16, width * 0.075))))
        depth = 0.55 * (radius / 28.0) ** 0.85
        dist = np.hypot(xx - cx, yy - cy)
        floor = dist < radius * 0.78
        bowl = np.zeros_like(dem)
        bowl[floor] = -depth * (1.0 - (dist[floor] / (radius * 0.78)) ** 2) ** 1.25
        rim_sigma = max(1.2, 0.07 * radius)
        rim = 0.42 * depth * np.exp(-0.5 * ((dist - radius * 0.92) / rim_sigma) ** 2)
        dem += bowl + rim
    # A population of small craters gives scale-stable texture.
    for _ in range(n_craters * 3):
        cx = rng.uniform(0, width)
        cy = rng.uniform(0, height)
        radius = float(rng.uniform(2.2, 6.5))
        depth = 0.22 * radius / 4.0
        y0, y1 = max(0, int(cy - radius * 2)), min(height, int(cy + radius * 2) + 1)
        x0, x1 = max(0, int(cx - radius * 2)), min(width, int(cx + radius * 2) + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        local_y, local_x = np.ogrid[y0:y1, x0:x1]
        dist = np.hypot(local_x - cx, local_y - cy)
        local = np.zeros((y1 - y0, x1 - x0), np.float32)
        floor = dist < radius * 0.8
        local[floor] = -depth * (1.0 - (dist[floor] / (radius * 0.8)) ** 2)
        local += 0.4 * depth * np.exp(-0.5 * ((dist - radius) / max(0.8, 0.18 * radius)) ** 2)
        dem[y0:y1, x0:x1] += local
    return dem.astype(np.float32)


def render_albedo(height: int, width: int, seed: int, dem: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Fine mottling is stable across sun angles. Large albedo blobs were
    # competing with crater rims and are intentionally not used.
    albedo = 0.56 + 0.08 * fbm(height, width, rng, octaves=4, cell0=28, persistence=0.55)
    albedo += rng.normal(0, 0.012, size=(height, width)).astype(np.float32)
    return np.clip(albedo, 0.32, 0.84).astype(np.float32)


def add_grains(albedo: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    out = albedo.copy()
    height, width = out.shape
    for _ in range(n):
        x = int(rng.integers(1, width - 1))
        y = int(rng.integers(1, height - 1))
        radius = int(rng.integers(1, 3))
        value = float(rng.uniform(-0.08, 0.10))
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
        out[y0:y1, x0:x1][disk] += value
    return out


def shade_dem(dem: np.ndarray, albedo: np.ndarray, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Lambertian shade. Azimuth 0 means the light source is toward the top of the image."""
    gy, gx = np.gradient(dem.astype(np.float32))
    exaggeration = 4.5
    nx, ny, nz = -gx * exaggeration, -gy * exaggeration, np.ones_like(gx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(max(8.0, elevation_deg))
    sx = np.cos(elevation) * np.sin(azimuth)
    sy = -np.cos(elevation) * np.cos(azimuth)
    sz = np.sin(elevation)
    lit = np.clip(nx * sx + ny * sy + nz * sz, 0, 1)
    ambient = 0.22
    image = albedo * (ambient + (1.0 - ambient) * lit)
    image = np.clip(image, 0, 1) ** 0.9
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def simulate_cross_sensor(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Appearance change only. Geometry is unchanged. Not a real spectrometer."""
    values = image.astype(np.float32)
    values = cv2.GaussianBlur(values, (0, 0), 1.15)
    values = 255.0 * np.clip(values / 255.0, 0, 1) ** 1.15
    low = cv2.GaussianBlur(values, (0, 0), 11)
    values = np.clip(values + 0.30 * (low - low.mean()), 0, 255)
    stripes = rng.normal(0, 2.2, size=(1, values.shape[1])).astype(np.float32)
    values = values + stripes
    return np.clip(values, 0, 255).astype(np.uint8)


def fbm(height: int, width: int, rng: np.random.Generator, octaves: int, cell0: float, persistence: float) -> np.ndarray:
    acc = np.zeros((height, width), np.float32)
    norm = 0.0
    amp = 1.0
    cell = float(cell0)
    for _ in range(octaves):
        gh = max(2, int(np.ceil(height / cell)) + 2)
        gw = max(2, int(np.ceil(width / cell)) + 2)
        grid = rng.random((gh, gw)).astype(np.float32)
        up = cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC)
        acc += amp * up
        norm += amp
        amp *= persistence
        cell = max(2.0, cell * 0.5)
    return acc / max(norm, 1e-6)


def _ground_truth_matrix(spec: dict[str, Any], size: int) -> np.ndarray:
    tx, ty = spec["translation"]
    # Keep the transformed source roughly over the reference center.
    center = (size - 1) / 2.0
    linear = similarity_matrix(float(spec["scale"]), float(spec["rotation_deg"]), 0.0, 0.0)
    if spec.get("model") == "affine":
        shear = float(spec.get("shear", 0.0))
        axis = float(spec.get("axis_ratio", 1.0))
        extra = np.array([[1.0, shear], [0.0, axis]], dtype=np.float64)
        linear[:2, :2] = linear[:2, :2] @ extra
    mapped_center = linear[:2, :2] @ np.array([center, center])
    linear[0, 2] = center + tx - mapped_center[0]
    linear[1, 2] = center + ty - mapped_center[1]
    return linear


def _sample_source(
    shaded: np.ndarray,
    matrix: np.ndarray,
    origin: tuple[int, int],
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    x_ref = matrix[0, 0] * xs + matrix[0, 1] * ys + matrix[0, 2]
    y_ref = matrix[1, 0] * xs + matrix[1, 1] * ys + matrix[1, 2]
    map_x = (x_ref + origin[0]).astype(np.float32)
    map_y = (y_ref + origin[1]).astype(np.float32)
    sampled = cv2.remap(shaded, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    height, width = shaded.shape[:2]
    valid = (map_x >= 2) & (map_y >= 2) & (map_x < width - 3) & (map_y < height - 3)
    sampled[~valid] = 0
    return sampled, valid.astype(np.uint8) * 255


def _add_noise(image: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image
    noisy = image.astype(np.float32) + rng.normal(0, sigma, size=image.shape).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _segment_distance(xx, yy, x0, y0, x1, y1) -> np.ndarray:
    vx, vy = x1 - x0, y1 - y0
    length2 = vx * vx + vy * vy + 1e-8
    t = np.clip(((xx - x0) * vx + (yy - y0) * vy) / length2, 0, 1)
    px, py = x0 + t * vx, y0 + t * vy
    return np.hypot(xx - px, yy - py)
