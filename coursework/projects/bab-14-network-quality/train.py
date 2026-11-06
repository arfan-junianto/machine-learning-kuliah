"""Latih tiga model dan simpan model terbaik beserta laporan evaluasinya."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from network_quality_project.training import run_training  # noqa: E402


if __name__ == "__main__":
    summary = run_training(ROOT / "data" / "raw", ROOT / "artifacts")
    print(json.dumps(summary, indent=2))
