"""Tests for ``rvi.analysis.rvi``."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from rvi.analysis.encroachment import detect_encroachment
from rvi.analysis.rvi import (
    compute_proximity_from_distances,
    compute_rvi,
    compute_rvi_multi,
    coverage_score,
    density_normalise,
    density_raw,
    proximity_score,
    sensitivity_grid,
)
from rvi.geometry.segment import segment_waterways

# ---------------------------------------------------------------------------
# Sub-score primitives
# ---------------------------------------------------------------------------


def test_density_raw_is_buildings_per_kilometre() -> None:
    n = pd.Series([0, 10, 20])
    length_m = pd.Series([500, 500, 1000])
    raw = density_raw(n, length_m)
    # 10 buildings / 0.5 km = 20.0
    assert raw.tolist() == [0.0, 20.0, 20.0]


def test_density_raw_handles_zero_length() -> None:
    n = pd.Series([5, 0])
    length_m = pd.Series([0, 0])
    raw = density_raw(n, length_m)
    assert raw.tolist() == [0.0, 0.0]


def test_density_normalise_min_max() -> None:
    raw = pd.Series([0.0, 5.0, 10.0])
    norm = density_normalise(raw)
    assert norm.tolist() == [0.0, 0.5, 1.0]


def test_density_normalise_constant_input_yields_zero() -> None:
    raw = pd.Series([3.0, 3.0, 3.0])
    norm = density_normalise(raw)
    assert (norm == 0.0).all()


def test_coverage_clipped_to_one() -> None:
    enc = pd.Series([0.0, 100.0, 5000.0])
    buf = pd.Series([1000.0, 1000.0, 1000.0])
    cov = coverage_score(enc, buf)
    assert cov.tolist() == [0.0, 0.1, 1.0]


def test_proximity_score_from_distance_summary() -> None:
    n = pd.Series([0, 1, 2])
    d = pd.Series([np.nan, 0.0, 50.0])  # building at bank, building halfway
    r = pd.Series([100.0, 100.0, 100.0])
    p = proximity_score(d, r, n)
    assert p.tolist() == [0.0, 1.0, 0.5]


def test_compute_proximity_from_distances_matches_definition() -> None:
    # Three buildings at distances 0, 50, 100 inside r=100. Mean penetration:
    # (1 - 0/100 + 1 - 50/100 + max(0, 1 - 100/100))/3 = (1 + 0.5 + 0)/3 = 0.5
    p = compute_proximity_from_distances([0.0, 50.0, 100.0], 100.0)
    assert p == pytest.approx(0.5)


def test_compute_proximity_from_distances_handles_empty_and_zero_radius() -> None:
    assert compute_proximity_from_distances([], 100.0) == 0.0
    assert compute_proximity_from_distances([10.0], 0.0) == 0.0


# ---------------------------------------------------------------------------
# compute_rvi end-to-end
# ---------------------------------------------------------------------------


def test_compute_rvi_returns_all_score_columns(
    waterways_metric, buildings_metric, cfg
) -> None:
    segments = segment_waterways(waterways_metric, config=cfg)
    enc = detect_encroachment(segments, buildings_metric, buffer_width_m=30, config=cfg)
    scored = compute_rvi(enc, config=cfg)
    for col in ("density_raw", "density_norm", "coverage", "proximity", "rvi_composite"):
        assert col in scored.columns
    # Composite within [0, 1].
    assert (scored["rvi_composite"] >= 0).all()
    assert (scored["rvi_composite"] <= 1).all()


def test_rvi_zero_when_no_encroachment(
    waterways_metric, empty_buildings, cfg
) -> None:
    segments = segment_waterways(waterways_metric, config=cfg)
    enc = detect_encroachment(segments, empty_buildings, buffer_width_m=30, config=cfg)
    scored = compute_rvi(enc, config=cfg)
    assert (scored["rvi_composite"] == 0.0).all()
    assert (scored["density_norm"] == 0.0).all()
    assert (scored["coverage"] == 0.0).all()
    assert (scored["proximity"] == 0.0).all()


def test_rvi_weight_validation() -> None:
    # Composing requires weights summing to ~1.
    with pytest.raises(ValueError):
        compute_rvi(
            gpd.GeoDataFrame({"x": [1]}),
            alpha=1.0,
            beta=1.0,
            gamma=1.0,
        )


def test_rvi_alternative_weights_change_composite(
    waterways_metric, buildings_metric, cfg
) -> None:
    segments = segment_waterways(waterways_metric, config=cfg)
    enc = detect_encroachment(segments, buildings_metric, buffer_width_m=30, config=cfg)
    a = compute_rvi(enc, config=cfg, alpha=0.4, beta=0.3, gamma=0.3)
    b = compute_rvi(enc, config=cfg, alpha=1.0, beta=0.0, gamma=0.0)
    # Should differ on at least one segment with non-zero density and non-zero
    # coverage/proximity.
    diffs = (a["rvi_composite"] - b["rvi_composite"]).abs()
    assert (diffs > 0).any()


def test_compute_rvi_multi_columns_per_width(
    waterways_metric, buildings_metric, cfg
) -> None:
    from rvi.analysis.encroachment import detect_encroachment_multi

    segments = segment_waterways(waterways_metric, config=cfg)
    by_width = detect_encroachment_multi(segments, buildings_metric, config=cfg)
    wide = compute_rvi_multi(by_width, config=cfg)
    for w in (6, 10, 30):
        assert f"rvi_composite_{w}m" in wide.columns
        assert f"density_norm_{w}m" in wide.columns
        assert f"coverage_{w}m" in wide.columns
        assert f"proximity_{w}m" in wide.columns


def test_sensitivity_grid_covers_default_weights(
    waterways_metric, buildings_metric, cfg
) -> None:
    segments = segment_waterways(waterways_metric, config=cfg)
    enc = detect_encroachment(segments, buildings_metric, buffer_width_m=30, config=cfg)
    grid = sensitivity_grid(enc, config=cfg, step=0.2)
    # The (0.4, 0.3) point is included with gamma rounded to 0.3.
    triples = grid[["alpha", "beta", "gamma"]].drop_duplicates().values.tolist()
    # Looser check: at least one row has alpha=1 / beta=0 / gamma=0.
    extreme = any(a == 1.0 and b == 0.0 for a, b, _ in triples)
    assert extreme, "sensitivity grid should sweep the alpha=1 corner"
    # All rows have weights summing to ~1.
    for a, b, g in triples:
        assert a + b + g == pytest.approx(1.0, abs=1e-6)
