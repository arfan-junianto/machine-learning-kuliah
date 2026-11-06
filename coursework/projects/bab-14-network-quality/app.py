"""Aplikasi Streamlit prediksi kualitas jaringan berbasis agregat M-Lab."""

from pathlib import Path
import sys

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from network_quality_project.data import FEATURES  # noqa: E402
from network_quality_project.inference import predict_quality  # noqa: E402

LABELS = {
    "download_mbps": "Kecepatan download (Mbps)",
    "latency_ms": "Latensi (ms)",
    "packet_loss_pct": "Packet loss (%)",
}

st.set_page_config(page_title="Prediksi Gangguan Jaringan", page_icon="🌐")
st.title("🌐 Prediksi Gangguan Jaringan Internet")
st.warning(
    "Demonstrasi akademik berbasis agregat negara-bulan. Hasil bukan diagnosis "
    "koneksi individual dan tetap perlu dikonfirmasi dengan pengukuran langsung."
)

artifact = ROOT / "artifacts" / "model_bundle.joblib"
if not artifact.exists():
    st.error("Model belum tersedia. Jalankan `python download_data.py` lalu `python train.py`.")
    st.stop()

bundle = joblib.load(artifact)
st.caption(
    f"Model: {bundle['model_name']} · F2 holdout: {bundle['metrics']['f2']:.3f} · "
    f"Recall: {bundle['metrics']['recall']:.3f}"
)
st.write(
    "Masukkan hasil pengukuran jaringan. Rentang input mengikuti data training "
    "agar model tidak mengekstrapolasi terlalu jauh."
)

values = {}
columns = st.columns(3)
for column, feature in zip(columns, FEATURES):
    limits = bundle["feature_ranges"][feature]
    with column:
        values[feature] = st.number_input(
            LABELS[feature],
            min_value=float(limits["min"]),
            max_value=float(limits["max"]),
            value=float(limits["median"]),
            format="%.4f",
        )

if st.button("Prediksi", type="primary"):
    result = predict_quality(bundle, values)
    probability = result["probability_problematic"]
    if result["label"] == "BERMASALAH":
        st.error(f"Prediksi: BERMASALAH · probabilitas {probability:.1%}")
    else:
        st.success(f"Prediksi: NORMAL · risiko bermasalah {probability:.1%}")
    st.info(
        "Target merupakan weak label gangguan berat: bahkan p75 gagal jika download "
        "< 25 Mbps, latensi > 50 ms, atau packet loss > 0,5%."
    )
