# Desain Presentasi Repository Machine Learning

## Intent

Membuat presentasi untuk dosen yang menjelaskan repository GitHub `arfan-junianto/machine-learning-kuliah` sebagai pusat dokumentasi, implementasi, pengujian, dan deployment tugas mata kuliah Machine Learning. Presentasi harus dapat dibawakan fleksibel sekitar 12–15 menit dan tetap dapat dilanjutkan jika demo internet gagal.

## Goals

- Menjelaskan alasan repository dibuat dan manfaat struktur yang reproducible.
- Menunjukkan implementasi Bab 1–13 yang dapat dibuka melalui Google Colab.
- Menggunakan Bab 1 sebagai demo notebook singkat.
- Menggunakan Bab 14 sebagai bukti workflow machine learning end-to-end sampai aplikasi Streamlit.
- Menunjukkan bukti kualitas berupa test otomatis, anti-data-leakage, sitasi materi, dan pemisahan source/output.
- Menyediakan satu file HTML mandiri untuk presentasi layar dan penggunaan offline.
- Menanam catatan pembicara di dalam HTML agar tidak memerlukan dokumen pendamping.

## Non-goals

- Tidak membahas algoritma dan seluruh cell kode pada setiap Bab 1–13.
- Tidak mengklaim mini project Bab 14 sebagai perangkat diagnosis medis.
- Tidak memasukkan PDF knowledge base mentah ke GitHub atau deck publik.
- Tidak menjadikan presentasi sebagai laporan teknis Bab 14 kedua; Bab 14 hanya bagian utama dari cerita repository.

## Constraints

- Bahasa utama: Bahasa Indonesia.
- Target: 10 slide, rasio 16:9, durasi inti 12–15 menit.
- Nama pada sampul: Arfan Junianto; NIM ditulis sebagai placeholder `[NIM]` sampai pengguna menggantinya.
- Semua tautan publik memakai repository, Colab, GitHub Pages, dan Streamlit yang sudah tersedia.
- Klaim teknis materi kuliah mencantumkan referensi `[ID, hlm. N]` sesuai aturan repository.
- Visual antarmuka tidak boleh berupa halaman penuh di dalam halaman; hanya bagian relevan yang ditampilkan.
- Semua CSS, JavaScript, ikon, diagram, visual fallback, dan catatan pembicara harus tertanam di satu file HTML tanpa dependency lokal atau CDN.

## Approaches

### 1. Tur GitHub lengkap

Membuka setiap folder dan bab satu per satu. Pendekatan ini lengkap, tetapi mudah menjadi panjang dan kehilangan cerita utama.

### 2. Presentasi teknis Bab 14

Memusatkan pembahasan pada model, evaluasi, dan Streamlit. Pendekatan ini kuat secara teknis, tetapi tidak menjelaskan nilai repository sebagai kumpulan seluruh tugas.

### 3. Portfolio + live demo (dipilih)

Menjelaskan struktur dan reproducibility repository, mendemokan satu notebook di Colab, lalu memakai Bab 14 sebagai showcase end-to-end. Pendekatan ini dipilih karena menyeimbangkan organisasi tugas, pemahaman materi, implementasi, pengujian, dan hasil deployment.

## Storyline

1. Judul dan pesan utama repository.
2. Masalah yang diselesaikan: materi/tugas tersebar dan sulit direproduksi.
3. Peta struktur repository dan fungsi tiap bagian publik.
4. Indeks implementasi Bab 1–13 dan pola notebook mandiri.
5. Demo Bab 1 di Google Colab.
6. Bab 14 sebagai workflow end-to-end.
7. Hasil model dan evaluasi jujur Bab 14.
8. Demo deployment Streamlit.
9. Quality assurance: test, anti-leakage, sitasi, dan reproducibility.
10. Kesimpulan, keterbatasan, dan tautan repository.

## Deliverables

Hasil utama disimpan sebagai satu file:

- `coursework/presentations/repository-machine-learning/Presentasi-Repository-Machine-Learning.html`: deck 10 slide, visual offline, catatan pembicara, cue demo, fallback, dan bantuan kontrol yang semuanya tertanam.

File pengujian boleh berada di folder `tests/`, tetapi pengguna hanya perlu membawa satu file HTML untuk presentasi.

## Demo Flow

- Buka halaman utama GitHub dan tunjukkan tabel akses cepat.
- Buka indeks Bab 1–13, lalu klik tombol Colab Bab 1.
- Jelaskan input–training–evaluasi tanpa menunggu semua cell bila waktu terbatas.
- Kembali ke README Bab 14 dan buka live Streamlit.
- Klik prediksi menggunakan nilai bawaan, lalu jelaskan disclaimer akademik.
- Jika internet gagal, lanjutkan memakai visual fallback Colab dan Streamlit yang tertanam di HTML.

## Acceptance Criteria

- Deck memiliki tepat 10 slide dengan hierarki visual yang konsisten.
- HTML dapat dinavigasi dengan tombol panah, Home/End, kontrol layar, dan tombol fullscreen.
- Catatan pembicara dapat ditampilkan/disembunyikan tanpa muncul pada layar utama secara default.
- File tetap berfungsi saat offline dan tidak memuat stylesheet, script, font, iframe, atau gambar eksternal.
- Semua tautan demo benar dan tidak menggunakan path lokal komputer.
- Catatan mencakup pembukaan, transisi, cue demo, fallback, penutup, dan pertanyaan dosen yang mungkin muncul.
- Isi teknis Bab 14 konsisten dengan metrik dan disclaimer pada repository.
- Root README memiliki tautan menuju folder presentasi.
