# Bab 12 — Neural Network: MLP dengan Keras/TensorFlow

Status: **selesai**

Implementasi utama dan latihan terdapat pada [ML-MODUL-01, hlm. 80–82].

## File

[Buka notebook Bab 12](implementasi-bab-12-mlp.ipynb)

Notebook mencakup:

- setup NumPy, Matplotlib, TensorFlow/Keras, dan scikit-learn;
- dataset `load_digits`, split terstratifikasi, dan standardisasi anti-leakage;
- MLP 64–Dropout–32–10;
- accuracy, classification report, confusion matrix, serta kurva loss/accuracy;
- eksperimen tiga arsitektur; dan
- early stopping dengan pengembalian bobot terbaik.

## Menjalankan di Google Colab

Setelah repository di-push ke GitHub, gunakan template **Open in Colab** pada README utama. Di Colab pilih **Runtime → Run all** dan tunggu seluruh cell selesai. GPU tidak wajib karena dataset kecil.

## Status Verifikasi

JSON notebook, sintaks seluruh cell, urutan preprocessing, penggunaan seed, serta smoke test dataset telah diuji lokal. Training TensorFlow belum dijalankan penuh pada environment lokal karena paket TensorFlow tidak tersedia untuk environment Python yang aktif; jalankan **Runtime → Run all** di Colab sebagai verifikasi runtime akhir.

Kode yang terpotong oleh tata letak PDF sudah disusun menjadi Python valid. Penggunaan layer `Input`, seed eksplisit, classification report, dan confusion matrix merupakan adaptasi/pelengkap yang ditandai di dalam notebook.
