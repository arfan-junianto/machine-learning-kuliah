from __future__ import annotations

import warnings

import pandas as pd
import pytest

from breast_cancer_project import modeling
from breast_cancer_project.data import load_project_data, split_data
from breast_cancer_project.modeling import (
    SCORING,
    build_model_specs,
    evaluate_model,
    select_best_model,
    tune_models,
)


def test_model_specs_define_three_leakage_safe_pipelines():
    specs = build_model_specs()

    assert set(specs) == {"Logistic Regression", "SVM-RBF", "Random Forest"}
    assert set(SCORING) == {"f2", "recall", "f1", "precision", "roc_auc", "pr_auc"}
    assert "scaler" in specs["Logistic Regression"]["estimator"].named_steps
    assert "scaler" in specs["SVM-RBF"]["estimator"].named_steps
    assert "scaler" not in specs["Random Forest"]["estimator"].named_steps
    for spec in specs.values():
        assert "model" in spec["estimator"].named_steps
        assert spec["param_grid"]
        assert all(name.startswith("model__") for name in spec["param_grid"])


def test_svm_tuning_does_not_use_deprecated_probability_mode():
    X, y, _ = load_project_data()
    X_train, _, y_train, _ = split_data(X, y)
    svm_only = {"SVM-RBF": _one_combination_per_model()["SVM-RBF"]}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tune_models(X_train, y_train, cv_splits=2, model_specs=svm_only)

    assert not [item for item in caught if issubclass(item.category, FutureWarning)]


def _one_combination_per_model():
    tiny = build_model_specs()
    for spec in tiny.values():
        spec["param_grid"] = {
            name: [values[0]] for name, values in spec["param_grid"].items()
        }
    return tiny


def test_tuning_compares_all_models_and_selects_highest_f2_deterministically():
    X, y, _ = load_project_data()
    X_train, _, y_train, _ = split_data(X, y)

    results, estimators = tune_models(
        X_train,
        y_train,
        cv_splits=2,
        model_specs=_one_combination_per_model(),
    )

    assert set(results["model"]) == {
        "Logistic Regression",
        "SVM-RBF",
        "Random Forest",
    }
    assert set(estimators) == set(results["model"])
    assert {
        "cv_f2_mean",
        "cv_f2_std",
        "cv_recall_mean",
        "cv_f1_mean",
        "cv_precision_mean",
        "cv_roc_auc_mean",
        "cv_pr_auc_mean",
        "best_params",
    }.issubset(results.columns)

    ordered = results.sort_values(
        ["cv_f2_mean", "cv_recall_mean", "cv_pr_auc_mean", "model"],
        ascending=[False, False, False, True],
    )
    assert select_best_model(results) == ordered.iloc[0]["model"]


def test_evaluate_model_reports_screening_metrics_and_normalizes_column_order():
    X, y, _ = load_project_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    results, estimators = tune_models(
        X_train,
        y_train,
        cv_splits=2,
        model_specs=_one_combination_per_model(),
    )
    model = estimators[select_best_model(results)]

    metrics = evaluate_model(model, X_test, y_test)
    reordered = evaluate_model(model, X_test.loc[:, reversed(X_test.columns)], y_test)

    assert metrics == reordered
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == 114
    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f2",
        "roc_auc",
        "pr_auc",
        "confusion_matrix",
    }
    for name, value in metrics.items():
        if name != "confusion_matrix":
            assert 0.0 <= value <= 1.0


def test_evaluate_model_rejects_missing_feature():
    X, y, _ = load_project_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    _, estimators = tune_models(
        X_train,
        y_train,
        cv_splits=2,
        model_specs=_one_combination_per_model(),
    )
    model = estimators["Logistic Regression"]

    with pytest.raises(ValueError, match="Kolom input tidak sesuai"):
        evaluate_model(model, X_test.drop(columns=[X_test.columns[0]]), y_test)


def test_bootstrap_metric_intervals_are_deterministic_and_bounded():
    X, y, _ = load_project_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    results, estimators = tune_models(
        X_train,
        y_train,
        cv_splits=2,
        model_specs=_one_combination_per_model(),
    )
    model = estimators[select_best_model(results)]

    first = modeling.bootstrap_metric_intervals(
        model, X_test, y_test, n_bootstraps=100, random_state=42
    )
    second = modeling.bootstrap_metric_intervals(
        model, X_test, y_test, n_bootstraps=100, random_state=42
    )

    assert first == second
    assert set(first) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f2",
        "roc_auc",
        "pr_auc",
    }
    for interval in first.values():
        assert set(interval) == {"lower", "upper"}
        assert 0.0 <= interval["lower"] <= interval["upper"] <= 1.0
