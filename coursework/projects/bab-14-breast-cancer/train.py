"""CLI untuk menghasilkan seluruh artefak mini project."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from breast_cancer_project.training import run_training  # noqa: E402


if __name__ == "__main__":
    result = run_training(PROJECT_ROOT / "artifacts")
    print(f"Model terpilih: {result['selected_model']}")
    print(f"Artefak tersimpan di: {PROJECT_ROOT / 'artifacts'}")

