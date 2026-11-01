"""Pemuatan dan pembagian dataset proyek."""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


FEATURES = (
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
)


def load_project_data() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Muat dataset dan ubah kelas ganas menjadi label positif (1)."""
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.loc[:, list(FEATURES)].copy()
    y = (dataset.target == 0).astype(int).rename("diagnosis_malignant")
    metadata = {
        "target_mapping": {0: "jinak", 1: "ganas"},
        "source_samples": int(dataset.data.shape[0]),
        "source_features": int(dataset.data.shape[1]),
    }
    return X, y, metadata


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Bagi data secara terstratifikasi dan deterministik."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

