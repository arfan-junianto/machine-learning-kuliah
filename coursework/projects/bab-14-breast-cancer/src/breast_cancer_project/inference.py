"""Validasi dan inferensi satu sampel untuk deployment."""

from __future__ import annotations

import math
from typing import Any, Mapping

import pandas as pd


def validate_input(
    values: Mapping[str, Any],
    feature_metadata: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Validasi schema, tipe, nilai finite, dan rentang training."""
    expected = list(feature_metadata)
    if set(values) != set(expected) or len(values) != len(expected):
        missing = sorted(set(expected) - set(values))
        extra = sorted(set(values) - set(expected))
        raise ValueError(
            f"Kolom input tidak sesuai; kurang={missing}, berlebih={extra}"
        )

    normalized: dict[str, float] = {}
    for feature in expected:
        raw_value = values[feature]
        if isinstance(raw_value, bool):
            raise ValueError(f"{feature} harus numerik")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{feature} harus numerik") from exc
        if not math.isfinite(value):
            raise ValueError(f"{feature} harus finite")
        minimum = float(feature_metadata[feature]["min"])
        maximum = float(feature_metadata[feature]["max"])
        if not minimum <= value <= maximum:
            raise ValueError(
                f"{feature} di luar rentang training [{minimum}, {maximum}]"
            )
        normalized[feature] = value

    return pd.DataFrame([normalized], columns=expected)


def predict_one(bundle: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    """Prediksi satu observasi dan kembalikan probabilitas kelas ganas."""
    frame = validate_input(values, bundle["feature_metadata"])
    model = bundle["model"]
    class_id = int(model.predict(frame)[0])
    probabilities = model.predict_proba(frame)[0]
    positive_index = list(model.classes_).index(1)
    return {
        "class_id": class_id,
        "label": str(bundle["target_mapping"][class_id]),
        "probability_malignant": float(probabilities[positive_index]),
        "model_name": str(bundle["model_name"]),
    }

