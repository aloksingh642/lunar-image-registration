"""Load common rasters, optional GeoTIFF tags, and simple raw sidecars.

Real Chandrayaan-2 products are often PDS or GeoTIFF. This loader reads the
formats that can be handled without a network service. Missing georeferencing
is reported; it is not invented.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.exceptions import ImageLoadError
from src.io.metadata import (
    discover_ground_truth,
    discover_sidecar,
    load_metadata_file,
    sanitize_metadata,
)
from src.models import ImageBundle

_MAX_BYTES = 512 * 1024 * 1024
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".img", ".raw"}


def load_metadata(path: str | Path) -> dict[str, Any]:
    return sanitize_metadata(load_metadata_file(path))


def load_image(
    path: str | Path,
    *,
    sensor: str = "Other",
    role: str = "source",
    origin: str | None = None,
    reference_catalog: str | None = None,
    metadata_path: str | Path | None = None,
) -> ImageBundle:
    path = Path(path)
    if not path.exists():
        raise ImageLoadError(f"File not found: {path}")
    if path.suffix.lower() not in _IMAGE_SUFFIXES:
        raise ImageLoadError(
            f"Unsupported file type '{path.suffix}'. Use PNG, JPEG, TIFF/GeoTIFF, BMP, or a raw .img with a JSON sidecar."
        )

    meta: dict[str, Any] = {}
    sidecar = Path(metadata_path) if metadata_path else discover_sidecar(path)
    if sidecar is not None:
        meta.update(sanitize_metadata(load_metadata_file(sidecar)))

    if path.suffix.lower() in {".img", ".raw"}:
        image, native = _load_raw(path, meta)
    elif path.suffix.lower() in {".tif", ".tiff"}:
        image, native, tiff_meta = _load_tiff(path)
        meta = {**tiff_meta, **meta}  # sidecar wins over guessed tags
    else:
        image, native = _load_common(path), None

    image = _standardize_layout(image)
    if native is not None:
        native = _standardize_layout(native)
    _validate_array(image, path)

    resolved_origin = origin or str(meta.get("origin") or "user")
    if resolved_origin not in {"synthetic", "user", "unknown"}:
        resolved_origin = "user"
    resolved_sensor = str(meta.get("sensor") or sensor or "Other")
    resolved_catalog = meta.get("reference_catalog", reference_catalog)
    if "metadata_kind" not in meta:
        meta["metadata_kind"] = "user_supplied" if sidecar is not None or meta.get("georeference") else "absent"

    gt_matrix = None
    gt_model = None
    gt_path = discover_ground_truth(path)
    if gt_path is not None:
        gt = load_metadata_file(gt_path)
        if "matrix" in gt:
            gt_matrix = np.asarray(gt["matrix"], dtype=np.float64)
            gt_model = str(gt.get("model", "similarity"))
            meta["ground_truth_file"] = str(gt_path)
            meta["ground_truth_note"] = (
                "A ground-truth matrix file was found next to the image. "
                "It is used only for evaluation and is not treated as mission-certified accuracy."
            )

    if meta.get("origin") == "synthetic":
        resolved_origin = "synthetic"

    return ImageBundle(
        image=_as_display_ready(image),
        native_image=native,
        origin=resolved_origin,
        sensor=resolved_sensor,
        role=role,
        metadata=meta,
        path=str(path),
        reference_catalog=None if resolved_catalog in (None, "") else str(resolved_catalog),
        ground_truth_matrix=gt_matrix,
        ground_truth_model=gt_model,
    )


def save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        try:
            import tifffile
        except ImportError as exc:
            raise ImageLoadError("tifffile is required to write TIFF files.") from exc
        tifffile.imwrite(path, image)
        return
    import cv2

    to_write = image
    if image.ndim == 3 and image.shape[2] == 3:
        to_write = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        to_write = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    if not cv2.imwrite(str(path), to_write):
        raise ImageLoadError(f"Could not write image: {path}")


def _load_common(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImageLoadError(f"Could not decode {path.name}.") from exc
        image = np.asarray(Image.open(path))
    else:
        image = _bgr_to_rgb(image)
    return image


def _load_tiff(path: Path) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    meta: dict[str, Any] = {}
    try:
        import rasterio

        with rasterio.open(path) as dataset:
            array = dataset.read()
            array = np.moveaxis(array, 0, -1) if array.shape[0] not in (1,) else array[0]
            if array.ndim == 3 and array.shape[2] == 1:
                array = array[:, :, 0]
            transform = dataset.transform
            meta["georeference"] = {
                "reader": "rasterio",
                "crs": str(dataset.crs) if dataset.crs else None,
                "transform": [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
                "bounds": list(dataset.bounds),
                "resolution": list(dataset.res),
                "count": dataset.count,
            }
            if dataset.res and dataset.res[0]:
                meta.setdefault("gsd_m", float(dataset.res[0]))
                meta["gsd_note"] = (
                    "GSD was taken from the GeoTIFF pixel scale. "
                    "Confirm the CRS units before treating this as metres."
                )
        return array, array.copy(), meta
    except ImportError:
        meta["georeference_note"] = (
            "rasterio is not installed. Pixel values were read, but a full CRS decode was not performed."
        )
    except Exception as exc:  # noqa: BLE001 — fall back rather than fail a readable TIFF
        meta["georeference_note"] = f"rasterio could not open this TIFF ({exc}). Fell back to tifffile."

    try:
        import tifffile
    except ImportError as exc:
        raise ImageLoadError(
            "Could not read TIFF. Install tifffile, or convert the product to PNG."
        ) from exc
    try:
        with tifffile.TiffFile(path) as tif:
            array = tif.asarray()
            meta.update(_partial_geotiff_tags(tif))
    except Exception as exc:  # noqa: BLE001
        raise ImageLoadError(f"Could not decode TIFF {path.name}: {exc}") from exc
    return array, array.copy(), meta


def _partial_geotiff_tags(tif: Any) -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        page = tif.pages[0]
        tags = page.tags
        scale = tags[33550].value if 33550 in tags else None
        tie = tags[33922].value if 33922 in tags else None
        if scale is not None:
            info["georeference"] = {
                "reader": "tifffile-partial",
                "model_pixel_scale": list(scale),
                "model_tiepoint": list(tie) if tie is not None else None,
                "warning": "Partial GeoTIFF tag parse only. CRS is not fully decoded without rasterio.",
            }
            info["gsd_m"] = float(scale[0])
            info["gsd_note"] = "Pixel scale tag was found. Units depend on the CRS and were not verified."
    except Exception:
        return {}
    return info


def _load_raw(path: Path, meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    width = meta.get("width")
    height = meta.get("height")
    dtype = str(meta.get("dtype", "uint16"))
    offset = int(meta.get("offset_bytes", 0))
    if not width or not height:
        raise ImageLoadError(
            "Raw .img/.raw files need a sidecar with width, height, and dtype. "
            "A recommended path for PDS products is to convert them to GeoTIFF with GDAL or ISIS, then load the TIFF."
        )
    try:
        np_dtype = np.dtype(dtype)
    except TypeError as exc:
        raise ImageLoadError(f"Unsupported raw dtype '{dtype}'.") from exc
    if str(meta.get("byte_order", "")).lower() in {"big", ">", "msb"}:
        np_dtype = np_dtype.newbyteorder(">")
    count = int(width) * int(height)
    bands = int(meta.get("bands", 1))
    with path.open("rb") as handle:
        handle.seek(offset)
        buffer = handle.read(count * bands * np_dtype.itemsize)
    expected = count * bands * np_dtype.itemsize
    if len(buffer) < expected:
        raise ImageLoadError(
            f"Raw file is shorter than width × height × bands × dtype ({expected} bytes expected after offset)."
        )
    array = np.frombuffer(buffer, dtype=np_dtype, count=count * bands)
    if bands == 1:
        array = array.reshape((int(height), int(width)))
    else:
        array = array.reshape((bands, int(height), int(width)))
    return array, array.copy()


def _standardize_layout(array: np.ndarray) -> np.ndarray:
    array = np.squeeze(array)
    if array.ndim == 2:
        return array
    if array.ndim != 3:
        raise ImageLoadError(f"Unsupported image shape {array.shape}. Expected a 2D image or a band stack.")
    # Bands-first scientific cubes: (bands, H, W) with bands smaller than both spatial axes.
    if array.shape[0] <= 512 and array.shape[0] < array.shape[1] and array.shape[0] < array.shape[2]:
        array = np.moveaxis(array, 0, -1)
    return array


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    import cv2

    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return image


def _validate_array(image: np.ndarray, path: Path) -> None:
    if image.size == 0:
        raise ImageLoadError(f"{path.name} decoded to an empty array.")
    if image.nbytes > _MAX_BYTES:
        raise ImageLoadError(
            f"{path.name} is {image.nbytes / 1e6:.0f} MB in memory, above the 512 MB prototype limit. "
            "Subset the product or select fewer bands before loading."
        )
    if min(image.shape[:2]) < 32:
        raise ImageLoadError(f"{path.name} is smaller than 32 pixels on one side, which is too small to match.")


def _as_display_ready(image: np.ndarray) -> np.ndarray:
    """Return uint8 for the interactive path. Native values stay on ``native_image``."""
    if image.dtype == np.uint8:
        return image
    return stretch_to_uint8(image)


def stretch_to_uint8(image: np.ndarray) -> np.ndarray:
    array = image.astype(np.float32)
    if array.ndim == 2:
        return _stretch_band(array)
    bands = [_stretch_band(array[:, :, i]) for i in range(array.shape[2])]
    return np.stack(bands, axis=-1)


def _stretch_band(band: np.ndarray) -> np.ndarray:
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros(band.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((band - lo) / (hi - lo), 0, 1)
    scaled = np.nan_to_num(scaled, nan=0.0)
    return (scaled * 255).astype(np.uint8)


def matrix_from_metadata(meta: dict[str, Any]) -> np.ndarray | None:
    raw = meta.get("ground_truth_matrix")
    if raw is None:
        return None
    return np.asarray(raw, dtype=np.float64)


def dumps_metadata(meta: dict[str, Any]) -> str:
    return json.dumps(meta, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)
