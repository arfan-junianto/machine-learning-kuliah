import ast
import json
from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "bab-12-neural-network-mlp"
    / "implementasi-bab-12-mlp.ipynb"
)


def load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def code_source(notebook):
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def markdown_source(notebook):
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )


def test_notebook_is_valid_colab_compatible_json():
    notebook = load_notebook()
    nbformat.validate(nbformat.from_dict(notebook))
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    assert notebook["metadata"]["kernelspec"]["language"] == "python"
    assert notebook["metadata"]["colab"]["name"] == "implementasi-bab-12-mlp.ipynb"
    assert len(notebook["cells"]) >= 12
    cell_ids = [cell.get("id") for cell in notebook["cells"]]
    assert all(cell_ids)
    assert len(cell_ids) == len(set(cell_ids))


def test_every_python_cell_compiles_and_starts_without_saved_outputs():
    notebook = load_notebook()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        ast.parse(source, filename=f"cell-{index}")
        assert cell["execution_count"] is None
        assert cell["outputs"] == []


def test_notebook_covers_bab12_workflow_without_data_leakage():
    notebook = load_notebook()
    source = code_source(notebook)

    required_fragments = (
        "from sklearn.datasets import load_digits",
        "train_test_split(",
        "X_train_full",
        "X_val",
        "test_size=0.2",
        "random_state=SEED",
        "stratify=y",
        "scaler.fit_transform(X_train)",
        "scaler.transform(X_val)",
        "scaler.transform(X_test)",
        "keras.layers.Input(shape=(X_train.shape[1],))",
        "keras.layers.Dropout(0.3)",
        'loss="sparse_categorical_crossentropy"',
        "classification_report(",
        "ConfusionMatrixDisplay.from_predictions(",
        "arsitektur_list",
        "keras.callbacks.EarlyStopping(",
        'monitor="val_loss"',
        "restore_best_weights=True",
    )
    for fragment in required_fragments:
        assert fragment in source

    assert source.index("train_test_split(") < source.index("scaler.fit_transform(X_train)")
    assert source.index("X_val") < source.index("scaler.fit_transform(X_train)")
    assert "fit_transform(X_test)" not in source
    assert "fit_transform(X_val)" not in source
    assert "validation_split=" not in source


def test_cells_define_symbols_before_they_are_used():
    notebook = load_notebook()
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]

    def index_of(fragment):
        return next(index for index, source in enumerate(code_cells) if fragment in source)

    assert index_of("import tensorflow as tf") < index_of("digits = load_digits()")
    assert index_of("digits = load_digits()") < index_of("X_train_full")
    assert index_of("X_train_full") <= index_of("scaler = StandardScaler()")
    assert index_of("scaler = StandardScaler()") < index_of("model = keras.Sequential(")
    assert index_of("model = keras.Sequential(") < index_of("history = model.fit(")
    assert index_of("history = model.fit(") < index_of("model.evaluate(")
    assert index_of("def build_experiment_model") <= index_of("for arsitektur in arsitektur_list")
    all_code = "\n".join(code_cells)
    assert all_code.index("def build_experiment_model") < all_code.index(
        "for arsitektur in arsitektur_list"
    )
    assert index_of("for arsitektur in arsitektur_list") < index_of("architecture_names =")
    assert index_of("def build_experiment_model") < index_of("early_stopping_model =")


def test_notebook_is_reproducible_and_explains_source_adaptations():
    notebook = load_notebook()
    source = code_source(notebook)
    markdown = markdown_source(notebook)

    assert "SEED = 42" in source
    assert "tf.keras.utils.set_random_seed(SEED)" in source
    assert "[ML-MODUL-01, hlm. 80–82]" in markdown
    assert "adaptasi" in markdown.lower()
    assert "Input" in markdown
    assert "/home/" not in source
    assert "Downloads/project" not in source
