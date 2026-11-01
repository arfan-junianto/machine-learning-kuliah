"""Orkestrasi training end-to-end yang reproducible."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import (
    build_feature_metadata,
    save_bundle,
    save_json,
    save_plots,
)
from .data import load_project_data, split_data
from .modeling import evaluate_model, select_best_model, tune_models


def run_training(
    output_dir: Path,
    cv_splits: int = 5,
    random_state: int = 42,
    model_specs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Latih, pilih, evaluasi, dan serialisasi seluruh artefak proyek."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, source_metadata = load_project_data()
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
    )
    cv_results, estimators = tune_models(
        X_train,
        y_train,
        cv_splits=cv_splits,
        random_state=random_state,
        model_specs=model_specs,
    )
    selected_name = select_best_model(cv_results)
    selected_model = estimators[selected_name]
    test_metrics = evaluate_model(selected_model, X_test, y_test)
    feature_metadata = build_feature_metadata(X_train)

    bundle = {
        "model": selected_model,
        "model_name": selected_name,
        "features": list(X.columns),
        "feature_metadata": feature_metadata,
        "target_mapping": source_metadata["target_mapping"],
        "test_metrics": test_metrics,
    }
    save_bundle(output_dir / "model_bundle.joblib", bundle)

    export_results = cv_results.copy()
    export_results["best_params"] = export_results["best_params"].map(str)
    export_results.to_csv(output_dir / "model_comparison.csv", index=False)

    summary = X.describe().T.loc[:, ["min", "max", "mean", "50%", "std"]]
    summary = summary.rename(columns={"50%": "median"})
    summary.index.name = "feature"
    summary.to_csv(output_dir / "feature_summary.csv")
    X.median().to_frame().T.to_csv(output_dir / "sample_input.csv", index=False)

    save_plots(
        output_dir / "figures",
        X,
        y,
        cv_results,
        selected_model,
        X_test,
        y_test,
        test_metrics,
        random_state=random_state,
    )

    metrics = {
        "selected_model": selected_name,
        "selection_metric": "F2",
        "random_state": random_state,
        "cv_splits": cv_splits,
        "dataset": {
            **source_metadata,
            "selected_features": len(X.columns),
            "class_counts": {
                "jinak": int((y == 0).sum()),
                "ganas": int((y == 1).sum()),
            },
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        },
        "cv_comparison": cv_results.to_dict(orient="records"),
        "test_metrics": test_metrics,
    }
    return save_json(output_dir / "metrics.json", metrics)

