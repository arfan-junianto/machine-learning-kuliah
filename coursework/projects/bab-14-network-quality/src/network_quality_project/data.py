"""Unduh dan siapkan M-Lab Monthly Stats untuk klasifikasi agregat."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


FEATURES = ("download_mbps", "latency_ms", "packet_loss_pct")
IQB_THRESHOLDS = {
    "download_mbps": 25.0,
    "latency_ms": 50.0,
    "packet_loss_pct": 0.5,
}
MONTHS_2025 = tuple(f"2025-{month:02d}" for month in range(1, 13))
URL_TEMPLATE = (
    "https://storage.googleapis.com/mlab-sandbox-iqb-us-central1/cache/v1/"
    "{start}01T000000Z/{end}01T000000Z/downloads_by_country/data.parquet"
)


def _next_month(month: str) -> str:
    timestamp = pd.Timestamp(f"{month}-01") + pd.offsets.MonthBegin(1)
    return timestamp.strftime("%Y%m")


def monthly_url(month: str) -> str:
    """Bangun URL resmi M-Lab Monthly Stats untuk satu bulan."""
    start = month.replace("-", "")
    return URL_TEMPLATE.format(start=start, end=_next_month(month))


def download_monthly_stats(
    raw_dir: Path,
    months: tuple[str, ...] = MONTHS_2025,
) -> list[Path]:
    """Unduh file kecil Parquet; file yang sudah ada tidak diunduh ulang."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for month in months:
        destination = raw_dir / f"mlab-downloads-country-{month}.parquet"
        paths.append(destination)
        if destination.exists():
            continue
        with urlopen(monthly_url(month), timeout=60) as response:
            payload = response.read()
        temporary = destination.with_suffix(".parquet.part")
        temporary.write_bytes(payload)
        temporary.replace(destination)
    return paths


def _prepare_month(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    month = path.stem.rsplit("-", 2)[-2] + "-" + path.stem.rsplit("-", 1)[-1]
    required = {
        "country_code",
        "sample_count",
        "download_p50",
        "latency_p50",
        "loss_p50",
        "download_p75",
        "latency_p75",
        "loss_p75",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Kolom M-Lab tidak lengkap: {missing}")

    result = frame.loc[:, sorted(required)].copy()
    result["month"] = month
    result = result.rename(
        columns={
            "download_p50": "download_mbps",
            "latency_p50": "latency_ms",
            "loss_p50": "packet_loss_fraction",
        }
    )
    result["packet_loss_pct"] = result["packet_loss_fraction"] * 100.0
    result["problematic"] = (
        (result["download_p75"] < IQB_THRESHOLDS["download_mbps"])
        | (result["latency_p75"] > IQB_THRESHOLDS["latency_ms"])
        | (result["loss_p75"] * 100.0 > IQB_THRESHOLDS["packet_loss_pct"])
    ).astype(int)
    return result


def load_project_data(
    raw_dir: Path,
    minimum_samples: int = 100,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Muat median sebagai fitur dan weak label gangguan berat dari p75."""
    paths = sorted(Path(raw_dir).glob("mlab-downloads-country-*.parquet"))
    if not paths:
        raise FileNotFoundError("Data belum ada; jalankan download_data.py")
    frame = pd.concat([_prepare_month(path) for path in paths], ignore_index=True)
    frame = frame.loc[frame["sample_count"] >= minimum_samples].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=[*FEATURES, "problematic", "country_code"])
    frame = frame.loc[
        (frame["download_mbps"] > 0)
        & (frame["latency_ms"] >= 0)
        & (frame["packet_loss_pct"] >= 0)
    ].reset_index(drop=True)
    X = frame.loc[:, list(FEATURES)].astype(float)
    y = frame["problematic"].astype(int)
    groups = frame["country_code"].astype(str)
    return X, y, groups, frame


def split_by_country(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Pisahkan negara agar negara test tidak muncul di training."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_index, test_index = next(splitter.split(X, y, groups))
    return (
        X.iloc[train_index].copy(),
        X.iloc[test_index].copy(),
        y.iloc[train_index].copy(),
        y.iloc[test_index].copy(),
        groups.iloc[train_index].copy(),
        groups.iloc[test_index].copy(),
    )
