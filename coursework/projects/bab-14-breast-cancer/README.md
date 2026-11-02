# Mini Project Bab 14 — Skrining Kanker Payudara

Proyek ini menyelesaikan alur machine learning end-to-end: memahami masalah, mengeksplorasi dan menyiapkan data, membandingkan tiga model, melakukan hyperparameter tuning, mengevaluasi model terpilih pada test set, lalu menyajikannya melalui aplikasi Streamlit. Struktur tersebut mengikuti tugas utama Bab 14 [ML-MODUL-01, hlm. 89–94].

> **Peringatan:** aplikasi dan model ini hanya demonstrasi akademik, bukan diagnosis medis dan bukan pengganti tenaga kesehatan.

## Live Demo

[![Buka aplikasi Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://machine-learning-kuliah-edg2ydcbvp2yjsu7weh5dx.streamlit.app/)

Demo online dapat dibuka melalui:

**https://machine-learning-kuliah-edg2ydcbvp2yjsu7weh5dx.streamlit.app/**

### Cara mencoba aplikasi

1. Buka tautan **Live Demo** di atas.
2. Tunggu sampai formulir **Skrining Risiko Kanker Payudara** tampil.
3. Gunakan nilai bawaan untuk percobaan pertama, atau ubah sepuluh karakteristik inti sel sesuai data yang ingin diuji.
4. Klik tombol **Prediksi**.
5. Aplikasi menampilkan kelas prediksi **JINAK** atau **GANAS** beserta probabilitas ganas.

Nilai input dibatasi pada rentang data training. Hasil aplikasi tetap merupakan demonstrasi model dan tidak boleh digunakan sebagai diagnosis medis.

### Jika aplikasi meminta login

Pemilik deployment perlu menjadikannya publik melalui aplikasi Streamlit:

1. Klik **Share** di kanan atas aplikasi.
2. Pilih **Make this app public**; atau buka **Manage app → Settings → Sharing**.
3. Pada **Who can view this app**, pilih **This app is public and searchable**, lalu simpan.
4. Uji tautan kembali melalui jendela incognito tanpa login.

## Deploy dari GitHub ke Streamlit Community Cloud

1. Login ke [Streamlit Community Cloud](https://share.streamlit.io/) menggunakan akun GitHub.
2. Klik **Create app**, lalu pilih repository `arfan-junianto/machine-learning-kuliah`.
3. Gunakan branch `main`.
4. Isi **Main file path** dengan:

   ```text
   coursework/projects/bab-14-breast-cancer/app.py
   ```

5. Klik **Deploy** dan tunggu proses instalasi dependency selesai. Streamlit membaca `requirements.txt` yang berada di folder proyek ini.
6. Setelah aplikasi aktif, atur akses publik melalui menu **Share** agar tautannya dapat dibuka orang lain.

## Hasil utama

- Dataset: Breast Cancer Wisconsin (Diagnostic), 569 observasi dan 30 fitur asli; proyek memakai 10 fitur `mean`.
- Model yang dibandingkan: Logistic Regression, SVM-RBF, dan Random Forest.
- Model terpilih: **Random Forest** berdasarkan F2 rata-rata CV tertinggi, `0,934 ± 0,021`.
- Test set: accuracy `94,74%`, recall ganas `92,86%`, F1 `92,86%`, ROC-AUC `0,9848`, dan PR-AUC `0,9740`.
- Confusion matrix: TN=69, FP=3, FN=3, TP=39.

Angka lengkap ada di [`artifacts/metrics.json`](artifacts/metrics.json) dan pembahasannya di [`report.md`](report.md).

## Struktur

```text
.
├── app.py                         # aplikasi Streamlit
├── train.py                       # entry point training penuh
├── report.md                      # laporan ilmiah
├── presentation.md                # deck dan catatan presentasi 10–15 menit
├── artifacts/                     # model, metrik, tabel, dan visual hasil training
├── src/breast_cancer_project/
│   ├── data.py                    # data dan split terstratifikasi
│   ├── modeling.py                # pipeline, tuning, evaluasi
│   ├── artifacts.py               # serialisasi dan visualisasi
│   ├── training.py                # orkestrasi end-to-end
│   └── inference.py               # validasi input dan prediksi
└── tests/                         # tes data, modeling, training, inferensi, dan UI
```

## Menjalankan dari awal

Jalankan dari folder proyek ini:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
MPLCONFIGDIR=/tmp/ml-mpl .venv/bin/python train.py
MPLCONFIGDIR=/tmp/ml-mpl .venv/bin/pytest -q
.venv/bin/streamlit run app.py
```

Atau gunakan `make setup`, `make train`, `make test`, dan `make app`.

## Reproducibility

- Seed: `42`.
- Split: 80:20 dengan stratifikasi.
- Tuning: `GridSearchCV`, Stratified 5-Fold, `refit="f2"`.
- Semua scaling berada di dalam `Pipeline`, sehingga di-fit hanya pada fold training.
- Test set tidak dipakai dalam tuning atau pemilihan model; test dibuka sekali setelah pemenang CV ditetapkan.
- Bundle deployment berisi model, urutan fitur, rentang input, mapping target, dan metrik test.

## Sumber

- Materi utama: `[ML-MODUL-01, hlm. 89–94]`, `[ML-P02, hlm. 6–12]`, dan `[ML-P07, hlm. 19–24]`.
- **Sumber luar:** UCI Machine Learning Repository, *Breast Cancer Wisconsin (Diagnostic)*, DOI `10.24432/C5DW2B`.
- **Sumber luar:** dokumentasi scikit-learn `load_breast_cancer`.
