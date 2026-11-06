"""Unduh M-Lab Monthly Stats 2025 berukuran kecil."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from network_quality_project.data import download_monthly_stats  # noqa: E402


if __name__ == "__main__":
    paths = download_monthly_stats(ROOT / "data" / "raw")
    print(f"Selesai: {len(paths)} file tersimpan di {ROOT / 'data' / 'raw'}")
