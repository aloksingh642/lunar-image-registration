"""Run the synthetic benchmark. Numbers are simulator results, not mission accuracy."""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.argv = [str(ROOT / "main.py"), "benchmark", *sys.argv[1:]]
runpy.run_path(str(ROOT / "main.py"), run_name="__main__")
