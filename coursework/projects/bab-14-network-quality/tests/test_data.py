from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from network_quality_project.data import _prepare_month, split_by_country  # noqa: E402


def test_weak_label_uses_p75_not_median(tmp_path: Path):
    path = tmp_path / "mlab-downloads-country-2025-01.parquet"
    pd.DataFrame(
        {
            "country_code": ["AA", "BB"],
            "sample_count": [1000, 1000],
            "download_p50": [50.0, 50.0],
            "latency_p50": [20.0, 20.0],
            "loss_p50": [0.001, 0.001],
            "download_p75": [10.0, 40.0],
            "latency_p75": [20.0, 20.0],
            "loss_p75": [0.001, 0.001],
        }
    ).to_parquet(path)
    result = _prepare_month(path)
    assert result["problematic"].tolist() == [1, 0]
    assert result["download_mbps"].tolist() == [50.0, 50.0]


def test_group_split_has_no_country_overlap():
    X = pd.DataFrame({"a": range(20)})
    y = pd.Series([0, 1] * 10)
    groups = pd.Series([f"C{i // 2}" for i in range(20)])
    _, _, _, _, train_groups, test_groups = split_by_country(X, y, groups)
    assert set(train_groups).isdisjoint(set(test_groups))
