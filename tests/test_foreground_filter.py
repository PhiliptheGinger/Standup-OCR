import numpy as np
import pytest

from src.foreground_filter import (
    ForegroundFilterConfig,
    analyze_foreground,
    compute_foreground_stats,
)


def test_compute_foreground_stats_detects_dark_pixels() -> None:
    sample = np.full((32, 128), 255, dtype=np.uint8)
    sample[:, 48:80] = 0

    stats = compute_foreground_stats(sample)

    assert stats["ink_ratio"] == pytest.approx(32 * 32 / (32 * 128), rel=1e-2)
    assert stats["edge_density"] > 0.0
    assert stats["contrast"] > 0


def test_analyze_foreground_borderline_detection() -> None:
    cfg = ForegroundFilterConfig(
        ink_ratio_threshold=0.2,
        edge_density_threshold=0.05,
        contrast_threshold=10.0,
        ink_margin=0.05,
        edge_margin=0.02,
        contrast_margin=2.0,
    )
    stats = {"ink_ratio": 0.18, "edge_density": 0.045, "contrast": 9.5}

    keep, borderline = analyze_foreground(stats, cfg)

    assert keep is False
    assert borderline is True


def test_analyze_foreground_accepts_strong_signal() -> None:
    cfg = ForegroundFilterConfig(
        ink_ratio_threshold=0.2,
        edge_density_threshold=0.05,
        contrast_threshold=10.0,
    )
    stats = {"ink_ratio": 0.4, "edge_density": 0.2, "contrast": 30.0}

    keep, borderline = analyze_foreground(stats, cfg)

    assert keep is True
    assert borderline is False
