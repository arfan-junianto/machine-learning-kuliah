"""Orkestrasi training, evaluasi holdout, dan penyimpanan artefak."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay

from .data import FEATURES, IQB_THRESHOLDS, load_project_data, split_by_country
from .modeling import evaluate_model, select_best_model, tune_models


def _json_default(value: Any):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Tipe tidak dapat diserialisasi: {type(value)!r}")


def _save_plots(output_dir: Path, results: pd.DataFrame, model, X_test, y_test) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    ordered = results.sort_values("cv_f2_mean")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(ordered["model"], ordered["cv_f2_mean"], xerr=ordered["cv_f2_std"])
    ax.set(xlabel="F2 cross-validation", title="Perbandingan Model")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(figures / "model-comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=["Normal", "Bermasalah"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Negara Holdout")
    fig.tight_layout()
    fig.savefig(figures / "confusion-matrix.png", dpi=160)
    plt.close(fig)

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1",
        n_repeats=20,
        random_state=42,
    )
    order = np.argsort(importance.importances_mean)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(
        np.asarray(FEATURES)[order],
        importance.importances_mean[order],
        xerr=importance.importances_std[order],
    )
    ax.set(xlabel="Penurunan F1", title="Permutation Importance")
    fig.tight_layout()
    fig.savefig(figures / "feature-importance.png", dpi=160)
    plt.close(fig)


def run_training(
    raw_dir: Path,
    output_dir: Path,
    cv_splits: int = 5,
    random_state: int = 42,
    model_specs: dict | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y, groups, frame = load_project_data(raw_dir)
    X_train, X_test, y_train, y_test, groups_train, groups_test = split_by_country(
        X, y, groups, random_state=random_state
    )
    overlap = set(groups_train) & set(groups_test)
    if overlap:
        raise RuntimeError(f"Data leakage negara terdeteksi: {sorted(overlap)[:5]}")

    results, estimators = tune_models(
        X_train,
        y_train,
        groups_train,
        cv_splits=cv_splits,
        random_state=random_state,
        model_specs=model_specs,
    )
    selected_name = select_best_model(results)
    selected_model = estimators[selected_name]
    metrics = evaluate_model(selected_model, X_test, y_test)

    feature_ranges = {
        feature: {
            "min": float(X_train[feature].min()),
            "max": float(X_train[feature].max()),
            "median": float(X_train[feature].median()),
        }
        for feature in FEATURES
    }
    bundle = {
        "model": selected_model,
        "model_name": selected_name,
        "features": list(FEATURES),
        "feature_ranges": feature_ranges,
        "iqb_thresholds": IQB_THRESHOLDS,
        "metrics": metrics,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "training_countries": int(groups_train.nunique()),
        "test_countries": int(groups_test.nunique()),
        "target_definition": (
            "Gangguan berat jika download_p75 < 25 Mbps, latency_p75 > 50 ms, "
            "atau loss_p75 > 0,5%."
        ),
    }
    joblib.dump(bundle, output_dir / "model_bundle.joblib")

    export_results = results.copy()
    export_results["best_params"] = export_results["best_params"].map(
        lambda value: json.dumps(value, sort_keys=True, default=_json_default)
    )
    export_results.to_csv(output_dir / "model-comparison.csv", index=False)
    metadata = {
        "selected_model": selected_name,
        "test_metrics": metrics,
        "rows_after_cleaning": int(len(frame)),
        "class_distribution": {
            str(key): int(value) for key, value in y.value_counts().sort_index().items()
        },
        "train_test": {
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "training_countries": int(groups_train.nunique()),
            "test_countries": int(groups_test.nunique()),
            "country_overlap": 0,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8"
    )
    _save_plots(output_dir, results, selected_model, X_test, y_test)
    return metadata
