"""Model, cross-validation berbasis negara, dan evaluasi."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


SCORING = {
    "f2": make_scorer(fbeta_score, beta=2),
    "recall": "recall",
    "f1": "f1",
    "precision": "precision",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def build_model_specs(random_state: int = 42) -> dict[str, dict[str, Any]]:
    return {
        "Logistic Regression": {
            "estimator": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=3000, random_state=random_state)),
                ]
            ),
            "param_grid": {
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        },
        "Decision Tree": {
            "estimator": Pipeline(
                [("model", DecisionTreeClassifier(random_state=random_state))]
            ),
            "param_grid": {
                "model__max_depth": [3, 5, 8, None],
                "model__min_samples_leaf": [1, 5, 15],
                "model__class_weight": [None, "balanced"],
            },
        },
        "Random Forest": {
            "estimator": Pipeline(
                [("model", RandomForestClassifier(random_state=random_state, n_jobs=1))]
            ),
            "param_grid": {
                "model__n_estimators": [150, 300],
                "model__max_depth": [5, 10, None],
                "model__min_samples_leaf": [1, 5],
                "model__class_weight": [None, "balanced"],
            },
        },
    }


def tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    cv_splits: int = 5,
    random_state: int = 42,
    model_specs: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    specs = model_specs or build_model_specs(random_state)
    cv = StratifiedGroupKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )
    rows: list[dict[str, Any]] = []
    estimators: dict[str, Pipeline] = {}
    for name, spec in specs.items():
        search = GridSearchCV(
            spec["estimator"],
            spec["param_grid"],
            scoring=SCORING,
            refit="f2",
            cv=cv,
            n_jobs=1,
        )
        search.fit(X_train, y_train, groups=groups_train)
        index = search.best_index_
        rows.append(
            {
                "model": name,
                "cv_f2_mean": float(search.cv_results_["mean_test_f2"][index]),
                "cv_f2_std": float(search.cv_results_["std_test_f2"][index]),
                "cv_recall_mean": float(search.cv_results_["mean_test_recall"][index]),
                "cv_f1_mean": float(search.cv_results_["mean_test_f1"][index]),
                "cv_precision_mean": float(search.cv_results_["mean_test_precision"][index]),
                "cv_roc_auc_mean": float(search.cv_results_["mean_test_roc_auc"][index]),
                "cv_pr_auc_mean": float(search.cv_results_["mean_test_pr_auc"][index]),
                "best_params": search.best_params_,
            }
        )
        estimators[name] = search.best_estimator_
    return pd.DataFrame(rows), estimators


def select_best_model(results: pd.DataFrame) -> str:
    ordered = results.sort_values(
        ["cv_f2_mean", "cv_recall_mean", "cv_pr_auc_mean", "model"],
        ascending=[False, False, False, True],
    )
    return str(ordered.iloc[0]["model"])


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_test, predictions, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "pr_auc": float(average_precision_score(y_test, scores)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
