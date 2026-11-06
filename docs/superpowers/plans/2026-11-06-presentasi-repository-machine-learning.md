# Presentasi Repository Machine Learning HTML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Membuat satu file HTML mandiri berisi deck 10 slide dan catatan presenter 12–15 menit untuk menjelaskan repository Machine Learning kepada dosen.

**Architecture:** Seluruh struktur slide, style, navigasi, inline SVG, visual fallback, tautan demo, dan catatan pembicara berada di satu dokumen HTML. JavaScript hanya mengatur perpindahan slide, fullscreen, bantuan, dan catatan; tidak ada dependency lokal, CDN, iframe, atau fetch jaringan sehingga deck tetap berfungsi offline.

**Tech Stack:** HTML5, CSS, JavaScript vanilla, inline SVG, pytest, BeautifulSoup, Selenium/Chromium.

**Spec:** `docs/superpowers/specs/2026-11-06-presentasi-repository-machine-learning-design.md`

## Global Constraints

- Bahasa utama Bahasa Indonesia; tepat 10 slide dengan rasio 16:9.
- Durasi inti 12–15 menit dan seluruh cerita tetap dapat dibawakan offline.
- Nama sampul `Arfan Junianto`; NIM tetap `[NIM]` sampai pengguna menggantinya.
- Output presentasi hanya satu file `Presentasi-Repository-Machine-Learning.html`.
- Tidak boleh ada dependency lokal/eksternal, CDN, iframe, fetch, atau path absolut komputer.
- Catatan presenter tertanam dan tersembunyi secara default.
- Klaim materi kuliah wajib memakai referensi `[ID, hlm. N]`; sumber umum/luar ditandai.
- Jangan mengubah file di `library/sources/pdf/`.

## Review Focus

- Offline: slide, visual, dan catatan tetap bekerja tanpa request jaringan.
- Streamlit tidur atau Colab lambat: visual fallback dan ucapan pengganti tersedia di slide/catatan.
- Proyektor 1366×768 atau layar 1920×1080: konten tidak terpotong dan teks tetap terbaca.
- Keyboard/mouse: navigasi tersedia lewat keduanya, dan tombol tidak menutupi konten.
- Identitas belum lengkap: `[NIM]` terlihat sebagai placeholder yang harus diganti.

---

### Task 1: Kontrak Konten dan Struktur HTML

**Files:**
- Create: `coursework/presentations/repository-machine-learning/tests/test_presentation.py`
- Create: `coursework/presentations/repository-machine-learning/Presentasi-Repository-Machine-Learning.html`

**Interfaces:**
- Consumes: root README, indeks notebook Bab 1–13, README/metrik Bab 14, dan storyline spec.
- Produces: dokumen HTML dengan tepat 10 `.slide`, isi akademik, catatan per slide, tautan publik, dan visual inline.

- [ ] **Step 1: Tulis test kontrak konten yang gagal**

Test memastikan file HTML tersedia; berbahasa Indonesia; mempunyai tepat 10 `.slide`; memuat GitHub, Colab Bab 1, Streamlit, `[NIM]`, disclaimer medis, referensi `[ML-MODUL-01, hlm. 89–94]`; serta tidak memuat `/home/`, iframe, dependency eksternal, gambar eksternal, atau fetch.

- [ ] **Step 2: Jalankan test untuk memastikan gagal karena HTML belum tersedia**

Run: `/home/arfan/Downloads/project/kuliah/ML/.venv/bin/pytest -p no:cacheprovider coursework/presentations/repository-machine-learning/tests/test_presentation.py -q`

Expected: FAIL karena file HTML belum tersedia.

- [ ] **Step 3: Implementasikan struktur dan isi minimum HTML**

Buat 10 slide sesuai storyline spec. Gunakan inline SVG/CSS untuk peta repository, tampilan Colab, workflow Bab 14, hasil model, Streamlit, quality gates, serta kartu tautan tanpa mengandalkan file eksternal.

- [ ] **Step 4: Tambahkan catatan presenter ke setiap slide**

Setiap `.slide` memiliki `.speaker-notes` dengan ucapan 45–90 detik, transisi, cue demo, dan fallback. Slide terakhir juga memuat delapan jawaban singkat pertanyaan dosen dalam panel catatan.

- [ ] **Step 5: Jalankan test kontrak konten**

Run: `/home/arfan/Downloads/project/kuliah/ML/.venv/bin/pytest -p no:cacheprovider coursework/presentations/repository-machine-learning/tests/test_presentation.py -q`

Expected: seluruh test Task 1 PASS.

- [ ] **Step 6: Commit**

```bash
git add coursework/presentations/repository-machine-learning docs/superpowers/specs/2026-11-06-presentasi-repository-machine-learning-design.md docs/superpowers/plans/2026-11-06-presentasi-repository-machine-learning.md
git commit -m "feat: add standalone repository presentation"
```

### Task 2: Navigasi, Tampilan, dan Perilaku Offline

**Files:**
- Modify: `coursework/presentations/repository-machine-learning/tests/test_presentation.py`
- Modify: `coursework/presentations/repository-machine-learning/Presentasi-Repository-Machine-Learning.html`

**Interfaces:**
- Consumes: sepuluh slide dan catatan Task 1.
- Produces: navigasi ArrowLeft/ArrowRight, Home/End, tombol mouse, fullscreen `F`, catatan `N`, bantuan `?`, indikator progres, dan desain responsif 16:9.

- [ ] **Step 1: Tambahkan test perilaku dan layout yang gagal**

Test statis memastikan kontrol dan keyboard handler tersedia. Test Selenium pada 1366×768 dan 1920×1080 memastikan hanya satu slide aktif, perpindahan slide berjalan, catatan tersembunyi default, tombol `N` membuka catatan, dan tidak ada overflow pada tiap slide.

- [ ] **Step 2: Jalankan test untuk memastikan gagal pada perilaku yang belum tersedia**

Run: `/home/arfan/Downloads/project/kuliah/ML/.venv/bin/pytest -p no:cacheprovider coursework/presentations/repository-machine-learning/tests/test_presentation.py -q`

Expected: FAIL pada kontrol/navigasi/layout yang belum diimplementasikan.

- [ ] **Step 3: Implementasikan desain dan navigasi**

Gunakan tema gelap GitHub dengan aksen biru/teal, judul minimum 36px, isi minimum 20px, safe area proyektor, focus state aksesibel, serta JavaScript kecil tanpa library.

- [ ] **Step 4: Jalankan seluruh test presentasi**

Run: `/home/arfan/Downloads/project/kuliah/ML/.venv/bin/pytest -p no:cacheprovider coursework/presentations/repository-machine-learning/tests/test_presentation.py -q`

Expected: seluruh test konten, offline, navigasi, dan layout PASS.

- [ ] **Step 5: Commit**

```bash
git add coursework/presentations/repository-machine-learning
git commit -m "feat: add offline presentation controls"
```

### Task 3: Integrasi Repository dan Verifikasi Akhir

**Files:**
- Modify: `README.md`
- Modify: `coursework/presentations/repository-machine-learning/tests/test_presentation.py`

**Interfaces:**
- Consumes: HTML final Task 2.
- Produces: tautan presentasi yang mudah ditemukan dari halaman utama repository serta verifikasi tanpa regression.

- [ ] **Step 1: Tambahkan test integrasi README yang gagal**

Test memastikan root README menautkan `coursework/presentations/repository-machine-learning/Presentasi-Repository-Machine-Learning.html` dan menyebut presentasi offline.

- [ ] **Step 2: Jalankan test untuk memastikan gagal**

Run: `/home/arfan/Downloads/project/kuliah/ML/.venv/bin/pytest -p no:cacheprovider coursework/presentations/repository-machine-learning/tests/test_presentation.py -q`

Expected: FAIL karena tautan root README belum tersedia.

- [ ] **Step 3: Tambahkan tautan akses cepat ke root README**

Tambahkan satu baris pada tabel Akses Cepat untuk presentasi repository HTML offline.

- [ ] **Step 4: Jalankan verifikasi lengkap**

Run: `/home/arfan/Downloads/project/kuliah/ML/.venv/bin/pytest -p no:cacheprovider coursework/presentations/repository-machine-learning/tests/test_presentation.py coursework/projects/implementasi-modul-machine-learning/tests coursework/projects/bab-14-breast-cancer/tests coursework/assignments/pertemuan-06/tests/test_screen_layout.py -q`

Expected: seluruh test PASS tanpa regression.

- [ ] **Step 5: Periksa file HTML secara offline**

Blokir request HTTP/HTTPS pada browser test; buka `file://.../Presentasi-Repository-Machine-Learning.html`; navigasikan 10 slide dan pastikan tidak ada resource hilang.

- [ ] **Step 6: Commit**

```bash
git add README.md coursework/presentations/repository-machine-learning
git commit -m "docs: link standalone course presentation"
```

## Self-review

- Spec coverage: 10 slide, satu file HTML, offline, catatan presenter, demo fallback, navigasi, dan integrasi README dipetakan ke Task 1–3.
- Step scan: setiap step menghasilkan satu kontrak, implementasi, atau bukti verifikasi.
- Interface consistency: Task 2 dan 3 hanya mengonsumsi file HTML stabil dari Task 1.
- Review focus: offline, demo lambat, viewport, keyboard/mouse, dan placeholder NIM semuanya mempunyai test atau pengecekan eksplisit.
- Proportion: tiga task memisahkan konten, perilaku presentasi, dan integrasi repository.
