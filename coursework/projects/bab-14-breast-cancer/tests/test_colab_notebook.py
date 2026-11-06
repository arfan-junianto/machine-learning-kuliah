import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "mini-project-bab-14-breast-cancer.ipynb"


def test_colab_notebook_is_clean_and_covers_end_to_end_workflow():
    assert NOTEBOOK.is_file()
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["colab"]["name"] == NOTEBOOK.name
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])
    assert all(cell.get("execution_count") is None for cell in notebook["cells"])

    content = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    for required in (
        "[ML-MODUL-01, hlm. 89–91]",
        "load_project_data",
        "split_data",
        "tune_models",
        "evaluate_model",
        "confusion_matrix",
        "roc_curve",
        "permutation_importance",
        "joblib.dump",
        "Streamlit",
    ):
        assert required in content


def test_project_readme_links_directly_to_colab_notebook():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    expected = (
        "https://colab.research.google.com/github/arfan-junianto/"
        "machine-learning-kuliah/blob/main/coursework/projects/"
        "bab-14-breast-cancer/notebooks/mini-project-bab-14-breast-cancer.ipynb"
    )

    assert expected in readme
    assert "Buka Notebook di Google Colab" in readme
