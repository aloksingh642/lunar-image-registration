"""Integration test: recover a known similarity on the synthetic baseline."""

import numpy as np

from src.demo.synthetic import make_pair
from src.models import PipelineConfig
from src.pipeline import run_registration


def test_baseline_recovers_known_similarity():
    pair = make_pair("baseline", seed=7, size=360)
    config = PipelineConfig(
        max_features=2500,
        auto_retry=False,
        make_visuals=False,
        max_working_side=360,
        model="similarity",
        scale_search=True,
        refine=True,
        ratio=0.8,
    )
    result = run_registration(pair["source"], pair["reference"], config)
    assert result.evaluation_mode == "synthetic_benchmark"
    assert result.data_origin == "synthetic"
    assert result.metrics["inliers"] >= 30
    assert result.metrics["inlier_ratio"] >= 0.4
    assert result.metrics["spatial_coverage"] >= 0.4
    gt = result.gt_metrics
    assert gt["corner_rmse_px"] < 3.0
    assert gt["scale_error_percent"] < 3.0
    assert gt["rotation_error_deg"] < 1.5
    assert "not Chandrayaan-2" in result.disclaimer or "not a Chandrayaan-2" in " ".join(result.warnings)
    # The reported RMSE is a residual, and the definition must say so.
    assert "not an independent" in result.metrics["definitions"]["rmse"]
    assert result.metrics["reliability"]["level"] != "unreliable"
    assert np.isfinite(result.metrics["rmse_px"])
