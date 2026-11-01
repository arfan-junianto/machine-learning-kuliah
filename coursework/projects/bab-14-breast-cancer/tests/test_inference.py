from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from breast_cancer_project.inference import predict_one, validate_input
from breast_cancer_project.modeling import build_model_specs
from breast_cancer_project.training import run_training


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _one_combination_per_model():
    specs = build_model_specs()
    for spec in specs.values():
        spec["param_grid"] = {
            name: [values[0]] for name, values in spec["param_grid"].items()
        }
    return specs


@pytest.fixture(scope="module")
def trained_bundle(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("inference-artifacts")
    run_training(output_dir, cv_splits=2, model_specs=_one_combination_per_model())
    return output_dir, joblib.load(output_dir / "model_bundle.joblib")


def _median_values(bundle):
    return {
        feature: bundle["feature_metadata"][feature]["median"]
        for feature in reversed(bundle["features"])
    }


def test_validate_input_orders_features_and_predicts_probability(trained_bundle):
    _, bundle = trained_bundle
    frame = validate_input(_median_values(bundle), bundle["feature_metadata"])
    result = predict_one(bundle, _median_values(bundle))

    assert list(frame.columns) == bundle["features"]
    assert frame.shape == (1, 10)
    assert result["class_id"] in {0, 1}
    assert result["label"] in {"jinak", "ganas"}
    assert 0.0 <= result["probability_malignant"] <= 1.0
    assert result["model_name"] == bundle["model_name"]


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda values, feature, stats: values.pop(feature), "Kolom input tidak sesuai"),
        (lambda values, feature, stats: values.update({"fitur asing": 1}), "Kolom input tidak sesuai"),
        (lambda values, feature, stats: values.update({feature: "abc"}), "harus numerik"),
        (lambda values, feature, stats: values.update({feature: np.nan}), "harus finite"),
        (
            lambda values, feature, stats: values.update(
                {feature: stats[feature]["max"] + 1}
            ),
            "di luar rentang",
        ),
    ],
)
def test_validate_input_rejects_unsafe_values(trained_bundle, mutation, message):
    _, bundle = trained_bundle
    values = _median_values(bundle)
    feature = bundle["features"][0]
    mutation(values, feature, bundle["feature_metadata"])

    with pytest.raises(ValueError, match=message):
        validate_input(values, bundle["feature_metadata"])


def test_streamlit_app_renders_inputs_and_medical_disclaimer(
    trained_bundle, monkeypatch
):
    output_dir, _ = trained_bundle
    monkeypatch.setenv("MODEL_BUNDLE_PATH", str(output_dir / "model_bundle.joblib"))

    app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=20)

    assert not app.exception
    assert len(app.number_input) == 10
    assert any("bukan diagnosis medis" in item.value.lower() for item in app.warning)
    assert app.title[0].value == "Skrining Risiko Kanker Payudara"
