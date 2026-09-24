"""Project paths resolved from this file, never from a hard-coded machine path."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEMO_DIR = DATA_DIR / "demo"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs"
CONFIG_DIR = ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


def ensure_output_dir(path: Path | None = None) -> Path:
    out = path or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out
