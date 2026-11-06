# Mini Project Bab 14 v2 — Gangguan Jaringan Internet

Proyek klasifikasi **Normal/Bermasalah** menggunakan tiga input yang mudah dipahami:

- kecepatan download (Mbps);
- latensi (ms);
- packet loss (%).

Workflow mengikuti mini project Bab 14: pemahaman masalah, persiapan data, sedikitnya tiga model, cross-validation dan tuning yang adil, evaluasi holdout, lalu deployment. [ML-MODUL-01, hlm. 89–91]

## Sumber dan desain data

Data berasal dari [M-Lab Monthly Stats](https://www.measurementlab.net/data/monthly-stats/) tahun 2025, agregasi negara-bulan dari pengukuran NDT. Dataset ini berlisensi CC0. Proyek menggunakan median (`p50`) sebagai fitur.

Target adalah **weak label gangguan berat**, bukan kebenaran diagnosis individual. Negara-bulan ditandai bermasalah bila bahkan kuartil performa yang lebih baik (`p75`; percentile M-Lab dinormalisasi agar angka lebih tinggi berarti performa lebih baik) masih melanggar salah satu ambang video conference M-Lab IQB:

- download kurang dari 25 Mbps;
- latensi lebih dari 50 ms;
- packet loss lebih dari 0,5%.

Pemisahan train/test dilakukan berdasarkan negara. Dengan demikian negara di test tidak pernah muncul di training, sehingga kebocoran akibat pengamatan bulanan berulang dapat dicegah.

## Menjalankan lokal

```bash
cd coursework/projects/bab-14-network-quality
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_data.py
python train.py
streamlit run app.py
```

Buka alamat yang ditampilkan Streamlit, biasanya `http://localhost:8501`.

## Menjalankan di Google Colab

Setelah folder ini tersedia di GitHub, buka [`Implementasi-Bab-14-v2-Jaringan.ipynb`](Implementasi-Bab-14-v2-Jaringan.ipynb), klik tombol **Open in Colab**, kemudian pilih **Runtime → Run all**. Notebook akan memasang dependensi, mengunduh data kecil M-Lab, melatih tiga model, dan menampilkan evaluasi.

## Struktur

```text
bab-14-network-quality/
├── app.py                 # UI Streamlit
├── download_data.py       # unduh 12 Parquet bulanan
├── train.py               # CV, tuning, evaluasi, artefak
├── Implementasi-...ipynb  # alur Google Colab
├── src/                   # kode reusable
├── tests/                 # tes data leakage dan validasi input
├── data/raw/              # data M-Lab yang diunduh
└── artifacts/             # model, metrik, tabel, dan gambar
```

## Artefak hasil training

- `artifacts/model_bundle.joblib` — model siap dipakai aplikasi;
- `artifacts/metrics.json` — metrik holdout dan audit pemisahan negara;
- `artifacts/model-comparison.csv` — hasil CV tiga model;
- `artifacts/figures/` — perbandingan model, confusion matrix, feature importance.

## Batasan

Data merupakan agregat negara-bulan, bukan pengukuran router atau pelanggan tertentu. Label dibuat dari aturan threshold sehingga hasil hanya cocok untuk demonstrasi akademik. Untuk diagnosis nyata, ukur juga upload, jitter, rute, Wi-Fi, dan kondisi perangkat secara langsung.
