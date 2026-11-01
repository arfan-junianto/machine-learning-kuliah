"""Aplikasi Streamlit untuk demonstrasi model terpilih."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from breast_cancer_project.inference import predict_one  # noqa: E402


FEATURE_LABELS = {
    "mean radius": "Rata-rata radius",
    "mean texture": "Rata-rata tekstur",
    "mean perimeter": "Rata-rata perimeter",
    "mean area": "Rata-rata luas",
    "mean smoothness": "Rata-rata kehalusan",
    "mean compactness": "Rata-rata kekompakan",
    "mean concavity": "Rata-rata konkavitas",
    "mean concave points": "Rata-rata titik konkaf",
    "mean symmetry": "Rata-rata simetri",
    "mean fractal dimension": "Rata-rata dimensi fraktal",
}


@st.cache_resource
def load_bundle(path: str):
    return joblib.load(path)


def main() -> None:
    st.set_page_config(page_title="Skrining Kanker Payudara", page_icon="🔬")
    st.title("Skrining Risiko Kanker Payudara")
    st.warning(
        "Aplikasi ini hanya demonstrasi akademik dan bukan diagnosis medis. "
        "Keputusan klinis wajib dilakukan oleh tenaga kesehatan."
    )

    bundle_path = Path(
        os.environ.get("MODEL_BUNDLE_PATH", PROJECT_ROOT / "artifacts/model_bundle.joblib")
    )
    if not bundle_path.is_file():
        st.error("Model belum tersedia. Jalankan `python train.py` terlebih dahulu.")
        return

    bundle = load_bundle(str(bundle_path))
    st.caption(f"Model terpilih: **{bundle['model_name']}** · label positif: ganas")
    st.write(
        "Masukkan karakteristik inti sel hasil citra FNA. Rentang input dibatasi "
        "pada rentang data training untuk mencegah ekstrapolasi diam-diam."
    )

    values: dict[str, float] = {}
    with st.form("prediction-form"):
        columns = st.columns(2)
        for index, feature in enumerate(bundle["features"]):
            stats = bundle["feature_metadata"][feature]
            span = float(stats["max"] - stats["min"])
            with columns[index % 2]:
                values[feature] = st.number_input(
                    FEATURE_LABELS.get(feature, feature),
                    min_value=float(stats["min"]),
                    max_value=float(stats["max"]),
                    value=float(stats["median"]),
                    step=max(span / 1000, 0.00001),
                    format="%.5f",
                    key=feature,
                )
        submitted = st.form_submit_button("Prediksi", type="primary")

    if submitted:
        try:
            result = predict_one(bundle, values)
        except ValueError as exc:
            st.error(str(exc))
            return
        probability = result["probability_malignant"]
        if result["class_id"] == 1:
            st.error(f"Prediksi model: GANAS · probabilitas ganas {probability:.1%}")
        else:
            st.success(f"Prediksi model: JINAK · probabilitas ganas {probability:.1%}")
        st.info("Hasil harus dikonfirmasi melalui pemeriksaan klinis profesional.")


if __name__ == "__main__":
    main()

