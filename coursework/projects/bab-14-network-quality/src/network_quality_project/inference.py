"""Validasi input dan inferensi model kualitas jaringan."""

from __future__ import annotations

import math
from typing import Mapping

import pandas as pd

from .data import FEATURES


def validate_input(values: Mapping[str, float], feature_ranges: Mapping[str, dict]) -> pd.DataFrame:
    if set(values) != set(FEATURES):
        raise ValueError("Input harus memuat bandwidth, latency, dan packet loss")
    ordered: dict[str, float] = {}
    for feature in FEATURES:
        value = float(values[feature])
        if not math.isfinite(value):
            raise ValueError(f"{feature} harus bernilai finite")
        minimum = float(feature_ranges[feature]["min"])
        maximum = float(feature_ranges[feature]["max"])
        if not minimum <= value <= maximum:
            raise ValueError(f"{feature} harus berada pada rentang {minimum:.4g}–{maximum:.4g}")
        ordered[feature] = value
    return pd.DataFrame([ordered], columns=FEATURES)


def predict_quality(bundle: Mapping, values: Mapping[str, float]) -> dict:
    frame = validate_input(values, bundle["feature_ranges"])
    model = bundle["model"]
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0, list(model.classes_).index(1)])
    return {
        "label": "BERMASALAH" if prediction == 1 else "NORMAL",
        "probability_problematic": probability,
    }
