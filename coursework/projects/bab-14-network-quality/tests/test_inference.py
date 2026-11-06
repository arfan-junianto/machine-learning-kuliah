from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from network_quality_project.inference import validate_input  # noqa: E402


RANGES = {
    "download_mbps": {"min": 1, "max": 100},
    "latency_ms": {"min": 1, "max": 200},
    "packet_loss_pct": {"min": 0, "max": 10},
}


def test_input_order_is_canonical():
    frame = validate_input(
        {"packet_loss_pct": 1, "latency_ms": 20, "download_mbps": 50}, RANGES
    )
    assert frame.columns.tolist() == ["download_mbps", "latency_ms", "packet_loss_pct"]


def test_out_of_training_range_is_rejected():
    with pytest.raises(ValueError, match="rentang"):
        validate_input(
            {"download_mbps": 500, "latency_ms": 20, "packet_loss_pct": 1}, RANGES
        )
