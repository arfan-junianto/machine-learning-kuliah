import ast
from pathlib import Path

import nbformat
import pytest

from notebook_quality import assert_no_model_selection_leakage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_ROOT = PROJECT_ROOT / "notebooks"

CHAPTERS = {
    1: ("bab-01-konsep-dasar-machine-learning", "implementasi-bab-01-konsep-dasar.ipynb", "14–15", ("load_iris", "KNeighborsClassifier")),
    2: ("bab-02-workflow-crisp-dm-bias-variance", "implementasi-bab-02-workflow-bias-variance.ipynb", "20–21", ("make_classification", "learning_curve", "CRISP-DM")),
    3: ("bab-03-data-preparation-lanjutan", "implementasi-bab-03-data-preparation.ipynb", "26–27", ("KNNImputer", "IsolationForest", "SMOTE", "RandomUnderSampler")),
    4: ("bab-04-feature-engineering-pipeline", "implementasi-bab-04-feature-engineering.ipynb", "32–34", ("ColumnTransformer", "Pipeline", "SelectKBest", "OneHotEncoder")),
    5: ("bab-05-decision-tree-pruning", "implementasi-bab-05-decision-tree.ipynb", "39–40", ("DecisionTreeClassifier", "cost_complexity_pruning_path", "plot_tree")),
    6: ("bab-06-knn-naive-bayes", "implementasi-bab-06-knn-naive-bayes.ipynb", "46–47", ("KNeighborsClassifier", "GaussianNB", "MultinomialNB")),
    7: ("bab-07-evaluasi-model", "implementasi-bab-07-evaluasi-model.ipynb", "53–54", ("confusion_matrix", "roc_curve", "precision_recall_curve", "cross_val_score")),
    8: ("bab-08-support-vector-machine", "implementasi-bab-08-svm.ipynb", "58–59", ("SVC", "GridSearchCV", "decision_function")),
    9: ("bab-09-ensemble-learning", "implementasi-bab-09-ensemble.ipynb", "63–64", ("RandomForestClassifier", "AdaBoostClassifier", "GradientBoostingClassifier")),
    10: ("bab-10-clustering", "implementasi-bab-10-clustering.ipynb", "69–70", ("KMeans", "AgglomerativeClustering", "DBSCAN", "silhouette_score")),
    11: ("bab-11-dimensionality-reduction", "implementasi-bab-11-dimensionality-reduction.ipynb", "74–75", ("PCA", "TSNE", "LinearDiscriminantAnalysis")),
    12: ("bab-12-neural-network-mlp", "implementasi-bab-12-mlp.ipynb", "80–82", ("keras.Sequential", "EarlyStopping")),
    13: ("bab-13-cnn-transfer-learning", "implementasi-bab-13-cnn-transfer-learning.ipynb", "86–88", ("keras.datasets.cifar10", "MobileNetV2", "Conv2D", "preprocess_input")),
}


@pytest.mark.parametrize("chapter", CHAPTERS)
def test_each_chapter_has_valid_notebook_with_core_material(chapter):
    folder, filename, pages, markers = CHAPTERS[chapter]
    path = NOTEBOOKS_ROOT / folder / filename
    assert path.is_file(), f"Notebook Bab {chapter} belum ada"

    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert notebook.metadata.kernelspec.language == "python"
    assert notebook.metadata.colab.name == filename
    assert len(notebook.cells) >= 8

    ids = [cell.id for cell in notebook.cells]
    assert len(ids) == len(set(ids))
    assert all(ids)

    combined = "\n".join(cell.source for cell in notebook.cells)
    assert f"[ML-MODUL-01, hlm. {pages}]" in combined
    assert "SEED = 42" in combined
    assert "/home/" not in combined
    assert "Downloads/project" not in combined
    for marker in markers:
        assert marker in combined

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        ast.parse(cell.source, filename=f"bab-{chapter}-cell-{index}")
        assert cell.execution_count is None
        assert cell.outputs == []


def test_exactly_chapters_one_through_thirteen_have_notebooks():
    notebooks = sorted(NOTEBOOKS_ROOT.glob("*/implementasi-*.ipynb"))
    assert len(notebooks) == 13
    assert {path.parent.name for path in notebooks} == {
        values[0] for values in CHAPTERS.values()
    }


def notebook_source(chapter):
    folder, filename, _, _ = CHAPTERS[chapter]
    notebook = nbformat.read(NOTEBOOKS_ROOT / folder / filename, as_version=4)
    return "\n".join(cell.source for cell in notebook.cells)


def notebook_code_source(chapter):
    folder, filename, _, _ = CHAPTERS[chapter]
    notebook = nbformat.read(NOTEBOOKS_ROOT / folder / filename, as_version=4)
    return "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )


@pytest.mark.parametrize("chapter", CHAPTERS)
def test_notebook_model_selection_is_semantically_free_from_holdout_leakage(chapter):
    assert_no_model_selection_leakage(notebook_code_source(chapter))


@pytest.mark.parametrize(
    ("chapter", "required_marker", "forbidden_marker"),
    [
        (1, "k_cv_scores", "k_scores.append(accuracy_score(y_test"),
        (5, "validation_accuracy", "score = accuracy_score(y_test, candidate.predict(X_test))"),
        (6, "k_cv_scores", "k_scores.append(accuracy_score(y_test"),
        (8, "kernel_cv_scores", "kernel_scores[kernel] = accuracy_score(y_test"),
        (9, "selected_n_estimators", "candidate.oob_score_,\n                    accuracy_score(y_test"),
        (11, "compression_cv_rows", "accuracy = accuracy_score(y_test, classifier.predict(X_test_reduced))"),
    ],
)
def test_model_selection_does_not_tune_on_test_set(chapter, required_marker, forbidden_marker):
    source = notebook_source(chapter)
    assert required_marker in source
    assert forbidden_marker not in source


@pytest.mark.parametrize(
    ("chapter", "markers"),
    [
        (4, ("selected_feature_names", "filter_features", "rfe_features", "embedded_features")),
        (7, ("exercise_axes", "exercise_roc", "exercise_pr")),
        (8, ("boundary_configs", "kernel_cv_scores")),
        (10, ("dbscan_silhouette", "selected_business_model", "cluster_summary")),
    ],
)
def test_practical_exercises_include_required_comparison_outputs(chapter, markers):
    source = notebook_source(chapter)
    for marker in markers:
        assert marker in source
