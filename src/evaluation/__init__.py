"""Metrics, ground-truth comparison, and display formatting."""

from src.evaluation.metrics import build_metrics, evaluate_ground_truth
from src.evaluation.format import metric_rows

__all__ = ["build_metrics", "evaluate_ground_truth", "metric_rows"]
