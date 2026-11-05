# Notebook Quality Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Memperkuat kelengkapan latihan dan verifikasi anti-leakage notebook implementasi Machine Learning.

**Architecture:** Generator notebook tetap menjadi sumber kebenaran. Tes baru memisahkan pemeriksaan AST anti-leakage dari kontrak kelengkapan latihan, lalu smoke runner membuktikan notebook klasik tetap dapat dieksekusi berurutan.

**Tech Stack:** Python, nbformat, AST standar Python, pytest, scikit-learn, pandas, matplotlib.

**Spec:** `docs/superpowers/specs/2026-11-05-notebook-quality-audit-design.md`

## Global Constraints

- Pertahankan sitasi materi berbentuk `[ML-MODUL-01, hlm. N]`.
- Test set hanya digunakan untuk evaluasi final, bukan pemilihan model.
- Regenerasi notebook melalui `scripts/build_remaining_notebooks.py`.
- Notebook Bab 1–11 harus lolos smoke execution tanpa menyimpan output.

## Review Focus

- Variasi nama variabel test set tetap terdeteksi melalui asal data yang ditentukan kontrak tes.
- `GridSearchCV.fit` pada test set ditolak.
- Evaluasi final pada test set tidak menghasilkan false positive.
- Latihan Titanic tetap dapat dieksekusi ketika jaringan tidak tersedia.
- Perubahan generator dan file `.ipynb` tidak menyimpang satu sama lain.

---

### Task 1: Detektor anti-leakage semantik

**Files:**
- Create: `coursework/projects/implementasi-modul-machine-learning/tests/notebook_quality.py`
- Create: `coursework/projects/implementasi-modul-machine-learning/tests/test_notebook_quality.py`
- Modify: `coursework/projects/implementasi-modul-machine-learning/tests/test_all_notebooks.py:69`

**Interfaces:**
- Produces: `find_model_selection_leaks(source: str) -> list[str]`
- Produces: `assert_no_model_selection_leakage(source: str) -> None`

- [ ] **Step 1: Tulis tes sintetis yang menerima CV pada training set dan evaluasi final test set.**
- [ ] **Step 2: Tulis tes sintetis yang menolak tuning berbasis `y_test` dan `GridSearchCV.fit(X_test, y_test)`.**
- [ ] **Step 3: Jalankan tes dan pastikan gagal karena helper belum ada.**
- [ ] **Step 4: Implementasikan visitor AST minimal dan pesan pelanggaran yang menyebut baris serta pola masalah.**
- [ ] **Step 5: Terapkan helper pada notebook bab yang melakukan model selection.**
- [ ] **Step 6: Jalankan tes unit dan pastikan seluruh kasus lulus.**

### Task 2: Lengkapi kontrak latihan Bab 4, 7, 8, dan 10

**Files:**
- Modify: `coursework/projects/implementasi-modul-machine-learning/tests/test_all_notebooks.py:93`
- Modify: `coursework/projects/implementasi-modul-machine-learning/scripts/build_remaining_notebooks.py:482`
- Regenerate: `coursework/projects/implementasi-modul-machine-learning/notebooks/bab-04-feature-engineering-pipeline/implementasi-bab-04-feature-engineering.ipynb`

**Interfaces:**
- Consumes: `build_notebook(...)` dan daftar `NOTEBOOK_SPECS` yang sudah ada.
- Produces: latihan Titanic Bab 4 dan kontrak struktur latihan empat bab.

- [ ] **Step 1: Perluas tes Bab 4 agar meminta Titanic, `ColumnTransformer`, pipeline numerik/kategorikal, dan metrik akurasi; perkuat marker hasil Bab 7, 8, dan 10.**
- [ ] **Step 2: Jalankan tes target dan pastikan Bab 4 gagal karena latihan Titanic belum tersedia.**
- [ ] **Step 3: Tambahkan latihan Titanic deterministik dengan fallback offline ke generator Bab 4.**
- [ ] **Step 4: Regenerasi notebook dan jalankan kembali tes target sampai lulus.**

### Task 3: Verifikasi end-to-end dan dokumentasi status

**Files:**
- Modify: `coursework/projects/implementasi-modul-machine-learning/README.md`

**Interfaces:**
- Consumes: seluruh notebook hasil generator dan suite pytest.
- Produces: bukti bahwa notebook klasik berjalan dan dokumentasi quality gate.

- [ ] **Step 1: Jalankan seluruh tes notebook.**
- [ ] **Step 2: Jalankan `smoke_run_classic_notebooks.py` untuk Bab 1–11.**
- [ ] **Step 3: Regenerasi notebook sekali lagi dan pastikan `git diff` tidak berubah (deterministik).**
- [ ] **Step 4: Dokumentasikan tes anti-leakage semantik dan perintah verifikasi di README.**
- [ ] **Step 5: Jalankan suite final dan periksa hanya file dalam scope yang berubah.**
