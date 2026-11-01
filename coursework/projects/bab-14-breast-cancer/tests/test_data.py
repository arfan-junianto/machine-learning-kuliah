from __future__ import annotations

import pandas as pd

from breast_cancer_project.data import FEATURES, load_project_data, split_data


def test_load_project_data_exposes_ten_mean_features_and_malignant_as_positive():
    X, y, metadata = load_project_data()

    assert X.shape == (569, 10)
    assert tuple(X.columns) == FEATURES
    assert y.value_counts().to_dict() == {0: 357, 1: 212}
    assert metadata["target_mapping"] == {0: "jinak", 1: "ganas"}
    assert metadata["source_samples"] == 569
    assert metadata["source_features"] == 30
    assert X.isna().sum().sum() == 0


def test_split_data_is_stratified_and_deterministic():
    X, y, _ = load_project_data()

    first = split_data(X, y)
    second = split_data(X, y)
    X_train, X_test, y_train, y_test = first

    assert (len(X_train), len(X_test)) == (455, 114)
    assert y_train.value_counts().to_dict() == {0: 285, 1: 170}
    assert y_test.value_counts().to_dict() == {0: 72, 1: 42}
    for left, right in zip(first, second, strict=True):
        if isinstance(left, pd.DataFrame):
            pd.testing.assert_frame_equal(left, right)
        else:
            pd.testing.assert_series_equal(left, right)
    assert set(X_train.index).isdisjoint(X_test.index)
