# Machine Learning Coursework

[![Tests](https://github.com/arfan-junianto/machine-learning-kuliah/actions/workflows/tests.yml/badge.svg)](https://github.com/arfan-junianto/machine-learning-kuliah/actions/workflows/tests.yml)

Kumpulan tugas, notebook implementasi, dan mini project mata kuliah Machine Learning. Repository ini dirancang agar setiap pekerjaan dapat dibaca di GitHub dan dijalankan ulang secara reproducible.

## Akses Cepat

| Bagian | Isi | Link |
|---|---|---|
| Implementasi Bab 1–13 | Notebook mandiri, dokumentasi, dan tombol Google Colab | [View Index](coursework/projects/implementasi-modul-machine-learning/README.md) |
| Mini Project Bab 14 | Training, evaluasi, laporan, presentasi, pengujian, dan aplikasi Streamlit | [View Project](coursework/projects/bab-14-breast-cancer/README.md) |
| Live Demo Bab 14 | Aplikasi skrining risiko kanker payudara untuk demonstrasi akademik | [Open Streamlit](https://machine-learning-kuliah-edg2ydcbvp2yjsu7weh5dx.streamlit.app/) |
| Latihan Kisi-kisi UTS | Soal dan jawaban Pertemuan 6 dalam Markdown, HTML, dan PDF | [View Assignment](https://arfan-junianto.github.io/machine-learning-kuliah/Jawaban-Latihan-Kisi-kisi-UTS.html) |

## Struktur Publik

```text
coursework/
├── assignments/
│   └── pertemuan-06/
└── projects/
    ├── implementasi-modul-machine-learning/
    └── bab-14-breast-cancer/
```

## Menjalankan Notebook

Pilih tombol **Buka** pada tabel di indeks Bab 1–13, kemudian gunakan **Runtime → Run all** di Google Colab. Bab 12–13 sebaiknya menggunakan runtime GPU. Notebook canonical disimpan tanpa output agar riwayat Git tetap ringkas; hasil eksekusi dapat disimpan sebagai salinan di Google Drive.

## Menjalankan Mini Project

```bash
python3 -m venv .venv
.venv/bin/pip install -r coursework/projects/bab-14-breast-cancer/requirements.txt
.venv/bin/pytest coursework/projects/bab-14-breast-cancer/tests -q
.venv/bin/streamlit run coursework/projects/bab-14-breast-cancer/app.py
```

## Sumber dan Batas Penggunaan

Implementasi mengikuti materi kuliah, terutama Modul Praktikum Machine Learning `[ML-MODUL-01]`. Sitasi halaman tersedia pada setiap notebook dan laporan. Mini project memakai dataset publik Breast Cancer Wisconsin (Diagnostic).

> Seluruh prediksi kesehatan dalam repository ini hanya demonstrasi akademik, bukan diagnosis medis dan bukan pengganti tenaga kesehatan.

Knowledge base PDF mentah dipertahankan secara lokal dan tidak dipublikasikan di repository ini.
