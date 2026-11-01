from __future__ import annotations

import json
from pathlib import Path

import joblib

from breast_cancer_project.modeling import build_model_specs
from breast_cancer_project.training import run_training


EXPECTED_FILES = {
    "model_bundle.joblib",
    "metrics.json",
    "model_comparison.csv",
    "feature_summary.csv",
    "sample_input.csv",
    "figures/class_distribution.png",
    "figures/model_comparison.png",
    "figures/confusion_matrix.png",
    "figures/roc_pr_curves.png",
    "figures/feature_importance.png",
}


def _one_combination_per_model():
    specs = build_model_specs()
    for spec in specs.values():
        spec["param_grid"] = {
            name: [values[0]] for name, values in spec["param_grid"].items()
        }
    return specs


def _assert_artifact_contract(output_dir: Path):
    for relative in EXPECTED_FILES:
        path = output_dir / relative
        assert path.is_file(), relative
        assert path.stat().st_size > 0, relative
    assert not list(output_dir.glob("*.tmp"))

    bundle = joblib.load(output_dir / "model_bundle.joblib")
    assert set(bundle) == {
        "model",
        "model_name",
        "features",
        "feature_metadata",
        "target_mapping",
        "test_metrics",
    }
    assert len(bundle["features"]) == 10
    assert set(bundle["feature_metadata"]) == set(bundle["features"])
    for stats in bundle["feature_metadata"].values():
        assert stats["min"] <= stats["median"] <= stats["max"]

    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert metrics["selected_model"] == bundle["model_name"]
    assert metrics["test_metrics"] == bundle["test_metrics"]
    assert len(metrics["cv_comparison"]) == 3
    return metrics


def test_run_training_creates_complete_reproducible_artifacts(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = run_training(
        first_dir,
        cv_splits=2,
        model_specs=_one_combination_per_model(),
    )
    second = run_training(
        second_dir,
        cv_splits=2,
        model_specs=_one_combination_per_model(),
    )

    first_metrics = _assert_artifact_contract(first_dir)
    second_metrics = _assert_artifact_contract(second_dir)
    assert first == first_metrics
    assert second == second_metrics
    assert first["selected_model"] == second["selected_model"]
    assert first["test_metrics"] == second["test_metrics"]
    assert first["cv_comparison"] == second["cv_comparison"]

