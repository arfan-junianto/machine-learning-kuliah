from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_ROOT = PROJECT_ROOT / "notebooks"

CHAPTERS = {
    "bab-01-konsep-dasar-machine-learning": ("Konsep Dasar Machine Learning", "14–15", "selesai"),
    "bab-02-workflow-crisp-dm-bias-variance": ("Workflow Machine Learning, CRISP-DM, dan Diagnosis Bias-Variance", "20–21", "selesai"),
    "bab-03-data-preparation-lanjutan": ("Data Preparation Lanjutan", "26–27", "selesai"),
    "bab-04-feature-engineering-pipeline": ("Feature Engineering dan Pipeline", "32–34", "selesai"),
    "bab-05-decision-tree-pruning": ("Decision Tree dan Pruning", "39–40", "selesai"),
    "bab-06-knn-naive-bayes": ("KNN dan Naive Bayes", "46–47", "selesai"),
    "bab-07-evaluasi-model": ("Evaluasi Model", "53–54", "selesai"),
    "bab-08-support-vector-machine": ("Support Vector Machine", "58–59", "selesai"),
    "bab-09-ensemble-learning": ("Ensemble Learning", "63–64", "selesai"),
    "bab-10-clustering": ("Clustering", "69–70", "selesai"),
    "bab-11-dimensionality-reduction": ("Dimensionality Reduction", "74–75", "selesai"),
    "bab-12-neural-network-mlp": ("Neural Network: MLP dengan Keras/TensorFlow", "80–82", "selesai"),
    "bab-13-cnn-transfer-learning": ("CNN dan Transfer Learning", "86–88", "selesai"),
    "bab-14-mini-project-end-to-end": ("Mini Project End-to-End", "92–94", "tersedia terpisah"),
}


def test_project_has_index_and_requirements():
    assert (PROJECT_ROOT / "README.md").is_file()
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    for dependency in (
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "imbalanced-learn",
        "scipy",
        "tensorflow",
    ):
        assert dependency in requirements.lower()

    development_requirements = (
        PROJECT_ROOT / "requirements-dev.txt"
    ).read_text(encoding="utf-8").lower()
    for dependency in ("-r requirements.txt", "notebook", "nbformat", "pytest"):
        assert dependency in development_requirements


def test_all_fourteen_chapters_have_honest_status_and_source_page():
    folders = {path.name for path in NOTEBOOKS_ROOT.iterdir() if path.is_dir()}
    assert folders == set(CHAPTERS)

    for folder, (title, page, status) in CHAPTERS.items():
        readme = (NOTEBOOKS_ROOT / folder / "README.md").read_text(encoding="utf-8")
        assert title in readme
        assert f"[ML-MODUL-01, hlm. {page}]" in readme
        assert f"Status: **{status}**" in readme


def test_completed_chapters_one_through_thirteen_have_notebooks():
    notebooks = sorted(NOTEBOOKS_ROOT.glob("*/implementasi-*.ipynb"))
    assert len(notebooks) == 13
    assert {path.parent.name for path in notebooks} == {
        folder for folder, (_, _, status) in CHAPTERS.items() if status == "selesai"
    }


def test_root_readme_explains_github_colab_and_local_usage():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    notebook_path = (
        "notebooks/bab-12-neural-network-mlp/implementasi-bab-12-mlp.ipynb"
    )

    assert notebook_path in readme
    assert "Open in Colab" in readme
    assert (
        "colab.research.google.com/github/arfan-junianto/"
        "machine-learning-kuliah/blob/main/"
    ) in readme
    assert "<USERNAME>/<REPOSITORY>" not in readme
    assert "python -m pip install -r coursework/projects/implementasi-modul-machine-learning/requirements-dev.txt" in readme
    assert "python -m venv .venv" in readme
    assert "Python 3.11 atau 3.12" in readme
    assert ".venv/bin/pytest" in readme
    assert "pytest" in readme


def test_bab12_readme_has_run_all_and_verification_limits():
    readme = (
        NOTEBOOKS_ROOT / "bab-12-neural-network-mlp" / "README.md"
    ).read_text(encoding="utf-8")

    assert "Runtime → Run all" in readme
    assert "TensorFlow" in readme
    assert "belum dijalankan penuh" in readme
    assert "[ML-MODUL-01, hlm. 80–82]" in readme
