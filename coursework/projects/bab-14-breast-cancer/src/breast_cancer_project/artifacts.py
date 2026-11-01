"""Pembuatan tabel, grafik, dan bundle deployment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    make_scorer,
    precision_recall_curve,
    roc_curve,
    fbeta_score,
)
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def build_feature_metadata(X: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Ringkas rentang data training untuk form dan validasi inferensi."""
    return {
        feature: {
            "min": float(X[feature].min()),
            "max": float(X[feature].max()),
            "median": float(X[feature].median()),
            "mean": float(X[feature].mean()),
        }
        for feature in X.columns
    }


def _positive_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X.loc[:, model.feature_names_in_])
    return probabilities[:, list(model.classes_).index(1)]


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def save_plots(
    figures_dir: Path,
    X: pd.DataFrame,
    y: pd.Series,
    cv_results: pd.DataFrame,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_metrics: dict[str, Any],
    random_state: int = 42,
) -> None:
    """Simpan lima visual utama EDA dan evaluasi."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    counts = y.value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    bars = plt.bar(["Jinak", "Ganas"], counts.values, color=["#3b82f6", "#ef4444"])
    plt.title("Distribusi Kelas Diagnosis")
    plt.ylabel("Jumlah sampel")
    plt.bar_label(bars)
    _save_figure(figures_dir / "class_distribution.png")

    comparison = cv_results.sort_values("cv_f2_mean", ascending=True)
    positions = np.arange(len(comparison))
    width = 0.25
    plt.figure(figsize=(8, 4.8))
    plt.barh(positions - width, comparison["cv_f2_mean"], width, label="F2")
    plt.barh(positions, comparison["cv_recall_mean"], width, label="Recall")
    plt.barh(positions + width, comparison["cv_pr_auc_mean"], width, label="PR-AUC")
    plt.yticks(positions, comparison["model"])
    plt.xlim(0, 1.05)
    plt.xlabel("Rata-rata Stratified CV")
    plt.title("Perbandingan Model pada Training Set")
    plt.legend()
    _save_figure(figures_dir / "model_comparison.png")

    matrix = np.asarray(test_metrics["confusion_matrix"])
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks([0, 1], ["Jinak", "Ganas"])
    plt.yticks([0, 1], ["Jinak", "Ganas"])
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title("Confusion Matrix Test Set")
    for row in range(2):
        for column in range(2):
            plt.text(column, row, int(matrix[row, column]), ha="center", va="center")
    _save_figure(figures_dir / "confusion_matrix.png")

    scores = _positive_scores(model, X_test)
    fpr, tpr, _ = roc_curve(y_test, scores)
    precision, recall, _ = precision_recall_curve(y_test, scores)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, color="#2563eb")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(title="ROC Curve", xlabel="False Positive Rate", ylabel="Recall")
    axes[1].plot(recall, precision, color="#dc2626")
    axes[1].axhline(y_test.mean(), linestyle="--", color="gray")
    axes[1].set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    figure.suptitle("Kinerja Model Terpilih pada Test Set")
    _save_figure(figures_dir / "roc_pr_curves.png")

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=make_scorer(fbeta_score, beta=2),
        n_repeats=10,
        random_state=random_state,
        n_jobs=1,
    )
    order = np.argsort(importance.importances_mean)
    plt.figure(figsize=(8, 5.5))
    plt.barh(
        np.asarray(X_test.columns)[order],
        importance.importances_mean[order],
        xerr=importance.importances_std[order],
        color="#7c3aed",
    )
    plt.xlabel("Penurunan F2 setelah permutasi")
    plt.title("Permutation Feature Importance")
    _save_figure(figures_dir / "feature_importance.png")


def save_bundle(path: Path, bundle: dict[str, Any]) -> None:
    """Tulis bundle secara atomik agar aplikasi tak membaca file parsial."""
    temporary = path.with_suffix(".tmp")
    joblib.dump(bundle, temporary)
    temporary.replace(path)


def save_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    """Normalisasi tipe NumPy lalu simpan JSON deterministik."""
    normalized = json.loads(
        json.dumps(
            value,
            default=lambda item: item.item() if hasattr(item, "item") else str(item),
            sort_keys=True,
        )
    )
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n")
    return normalized

