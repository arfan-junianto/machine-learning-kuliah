"""Jalankan seluruh code cell notebook Bab 1–11 tanpa mengubah file ipynb."""

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ml-course-matplotlib")

import matplotlib.pyplot as plt
import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_ROOT = PROJECT_ROOT / "notebooks"


def execute_notebook(path):
    notebook = nbformat.read(path, as_version=4)
    namespace = {
        "__name__": "__main__",
        "display": lambda value: None,
    }
    for index, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type != "code":
            continue
        try:
            exec(compile(cell.source, f"{path.name}:cell-{index}", "exec"), namespace)
        except Exception as error:
            raise RuntimeError(f"{path.name} gagal pada cell {index}") from error
        finally:
            plt.close("all")


def main():
    for chapter in range(1, 12):
        matches = sorted(NOTEBOOKS_ROOT.glob(f"bab-{chapter:02d}-*/implementasi-*.ipynb"))
        if len(matches) != 1:
            raise RuntimeError(f"Bab {chapter} memiliki {len(matches)} notebook")
        execute_notebook(matches[0])
        print(f"PASS Bab {chapter}: {matches[0].name}")


if __name__ == "__main__":
    main()
