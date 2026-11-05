# Desain Audit Kualitas Notebook Machine Learning

## Tujuan

Memastikan notebook implementasi modul tidak memilih model berulang kali memakai test set, memenuhi latihan Bab 4, 7, 8, dan 10, serta memiliki tes yang mendeteksi pola data leakage secara semantik—bukan hanya melalui pencarian string.

## Batasan

- Pertahankan notebook sebagai materi belajar yang mudah dijalankan di Google Colab.
- Pertahankan `scripts/build_remaining_notebooks.py` sebagai sumber deterministik notebook Bab 1–11 dan 13.
- Gunakan materi modul sebagai sumber utama dan pertahankan sitasi `[ML-MODUL-01, hlm. N]`.
- Jangan mengubah PDF sumber atau memperluas pekerjaan ke notebook TensorFlow Bab 12–13.
- Test set hanya digunakan sekali untuk evaluasi final setelah pemilihan model/hiperparameter dari training set atau cross-validation.

## Kondisi Saat Ini

- Pemilihan model Bab 1, 5, 6, 8, 9, dan 11 sudah memakai cross-validation, validation split, atau OOB pada data training.
- Latihan inti Bab 7, 8, dan 10 sudah tercakup.
- Bab 4 sudah membandingkan tiga metode feature selection, tetapi belum mengimplementasikan latihan Titanic dengan `ColumnTransformer` sesuai modul [ML-MODUL-01, hlm. 34].
- Tes saat ini memakai marker nama variabel dan satu forbidden substring, sehingga perubahan bentuk kode dapat membuat leakage lolos tanpa terdeteksi.

## Pendekatan

1. **Marker teks saja.** Paling murah, tetapi rapuh dan tidak membuktikan aliran data. Tidak dipilih.
2. **Analisis AST + uji mutasi + smoke execution.** Memeriksa pemanggilan seleksi model secara struktural, membuktikan detektor menolak contoh leakage sintetis, dan menjalankan notebook klasik. **Dipilih** karena seimbang antara ketelitian dan biaya eksekusi.
3. **Eksekusi notebook penuh dengan inspeksi runtime/fold.** Paling mendalam, tetapi jauh lebih lambat dan tidak diperlukan untuk kurikulum lokal ini.

## Desain Perubahan

### Kontrak anti-leakage

Tes mengekstrak seluruh code cell menjadi AST dan memeriksa pemanggilan yang digunakan untuk pemilihan model. Argumen data pada `cross_val_score`, `GridSearchCV.fit`, serta fitting kandidat harus berasal dari variabel training atau mekanisme validasi/OOB. Pemanggilan prediksi/score terhadap test set tetap diizinkan untuk evaluasi final, tetapi hasilnya tidak boleh dipakai untuk memilih kandidat.

Detektor juga diuji dengan potongan kode sintetis:

- contoh aman berbasis `X_train` harus diterima;
- tuning kandidat berdasarkan `accuracy_score(y_test, ...)` harus ditolak;
- `GridSearchCV.fit(X_test, y_test)` harus ditolak;
- evaluasi final satu kali pada test set harus tetap diterima.

### Kelengkapan latihan

- Bab 4 ditambah latihan kedua memakai Titanic, `ColumnTransformer`, pipeline numerik/kategorikal, dan evaluasi akurasi sesuai [ML-MODUL-01, hlm. 34]. Dataset disediakan secara deterministik dari URL publik dengan fallback data kecil agar smoke test tidak bergantung pada jaringan.
- Bab 7 harus tetap memuat dua model, confusion matrix, precision/recall/F1, ROC, PR, dan perbandingan tiga model dengan Stratified k-Fold [ML-MODUL-01, hlm. 54].
- Bab 8 harus tetap memuat tiga kernel dan decision boundary beberapa konfigurasi [ML-MODUL-01, hlm. 59].
- Bab 10 harus tetap memuat Elbow + Silhouette untuk k=2–10 serta perbandingan K-Means, Hierarchical, dan DBSCAN [ML-MODUL-01, hlm. 70].

### Verifikasi

Regenerasi notebook, validasi JSON/sintaks/struktur, jalankan tes anti-leakage dan kelengkapan, lalu smoke-run Bab 1–11 tanpa menyimpan output ke notebook.

## Non-goal

- Mengoptimalkan skor model untuk mendapatkan angka tertinggi.
- Mengganti dataset atau arsitektur Bab 12–14.
- Menjalankan training TensorFlow penuh secara lokal.
