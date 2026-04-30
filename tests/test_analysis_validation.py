"""Tests for ``rvi.analysis.validation``."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from rvi.analysis.encroachment import detect_encroachment
from rvi.analysis.rvi import compute_rvi
from rvi.analysis.validation import (
    SpearmanResult,
    aggregate_upstream_catchment,
    aggregate_upstream_euclidean,
    correlate_upstream_rvi_to_severity,
    spearman_with_ci,
    stratified_correlation,
)
from rvi.geometry.segment import segment_waterways

# ---------------------------------------------------------------------------
# Spearman
# ---------------------------------------------------------------------------


def test_spearman_perfect_positive() -> None:
    res = spearman_with_ci(
        x=[1, 2, 3, 4, 5, 6, 7, 8],
        y=[1, 2, 3, 4, 5, 6, 7, 8],
        n_bootstrap=200,
        seed=1,
    )
    assert res.rho == pytest.approx(1.0)
    assert res.n == 8


def test_spearman_perfect_negative() -> None:
    res = spearman_with_ci(
        x=[1, 2, 3, 4, 5, 6, 7, 8],
        y=[8, 7, 6, 5, 4, 3, 2, 1],
        n_bootstrap=200,
        seed=2,
    )
    assert res.rho == pytest.approx(-1.0)


def test_spearman_with_too_few_points_returns_nan() -> None:
    res = spearman_with_ci(x=[1, 2], y=[1, 2], n_bootstrap=10)
    assert np.isnan(res.rho)
    assert res.n == 2


def test_spearman_handles_nans_pairwise() -> None:
    res = spearman_with_ci(
        x=[1, 2, np.nan, 4, 5, 6],
        y=[1, 2, 3, np.nan, 5, 6],
        n_bootstrap=100,
        seed=3,
    )
    assert res.n == 4
    assert res.rho == pytest.approx(1.0)


def test_spearman_ci_envelopes_point_estimate() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=80)
    y = x + rng.normal(scale=0.5, size=80)
    res = spearman_with_ci(x, y, n_bootstrap=300, seed=4)
    assert res.ci_low <= res.rho <= res.ci_high


def test_spearman_result_as_dict() -> None:
    res = SpearmanResult(rho=0.5, pvalue=0.01, ci_low=0.2, ci_high=0.8, n=42)
    d = res.as_dict()
    assert d["method"] == "spearman"
    assert d["n"] == 42


# ---------------------------------------------------------------------------
# aggregate_upstream_euclidean
# ---------------------------------------------------------------------------


def test_aggregate_upstream_euclidean_returns_one_row_per_gauge(
    waterways_metric, buildings_metric, gauges_metric, cfg
) -> None:
    segments = segment_waterways(waterways_metric, config=cfg)
    enc = detect_encroachment(segments, buildings_metric, buffer_width_m=30, config=cfg)
    scored = compute_rvi(enc, config=cfg)
    upstream = aggregate_upstream_euclidean(scored, gauges_metric, config=cfg)
    assert set(upstream["gauge_id"]) == {"g1", "g2"}
    # Gauge g1 (close) sees segments; g2 is far away — zero count.
    g1 = upstream[upstream["gauge_id"] == "g1"].iloc[0]
    g2 = upstream[upstream["gauge_id"] == "g2"].iloc[0]
    assert int(g1["n_upstream_segments"]) > 0
    assert int(g2["n_upstream_segments"]) == 0


def test_aggregate_upstream_euclidean_carries_severity_columns(
    waterways_metric, buildings_metric, gauges_metric, cfg
) -> None:
    segments = segment_waterways(waterways_metric, config=cfg)
    enc = detect_encroachment(segments, buildings_metric, buffer_width_m=30, config=cfg)
    scored = compute_rvi(enc, config=cfg)
    out = aggregate_upstream_euclidean(scored, gauges_metric, config=cfg)
    for col in ("severity", "severity_int", "quality_verified"):
        assert col in out.columns


def test_aggregate_upstream_euclidean_radius_controls_inclusion(
    waterways_metric, buildings_metric, gauges_metric, cfg
) -> None:
    segments = segment_waterways(waterways_metric, config=cfg)
    enc = detect_encroachment(segments, buildings_metric, buffer_width_m=30, config=cfg)
    scored = compute_rvi(enc, config=cfg)
    tight = aggregate_upstream_euclidean(scored, gauges_metric, config=cfg, radius_m=10.0)
    loose = aggregate_upstream_euclidean(scored, gauges_metric, config=cfg, radius_m=100_000.0)
    assert tight["n_upstream_segments"].sum() <= loose["n_upstream_segments"].sum()


# ---------------------------------------------------------------------------
# aggregate_upstream_catchment
# ---------------------------------------------------------------------------


def test_aggregate_upstream_catchment_uses_polygons(
    waterways_metric, buildings_metric, gauges_metric, cfg
) -> None:
    from shapely.geometry import box

    segments = segment_waterways(waterways_metric, config=cfg)
    enc = detect_encroachment(segments, buildings_metric, buffer_width_m=30, config=cfg)
    scored = compute_rvi(enc, config=cfg)

    bounds = scored.total_bounds
    catchments = gpd.GeoDataFrame(
        {
            "gauge_id": ["g1", "g2"],
            "geometry": [
                box(*bounds).buffer(100),
                box(bounds[0] + 100_000, bounds[1] + 100_000, bounds[0] + 110_000, bounds[1] + 110_000),
            ],
        },
        geometry="geometry",
        crs=cfg.crs_metric,
    )
    out = aggregate_upstream_catchment(scored, gauges_metric, catchments=catchments, config=cfg)
    g1 = out[out["gauge_id"] == "g1"].iloc[0]
    g2 = out[out["gauge_id"] == "g2"].iloc[0]
    assert int(g1["n_upstream_segments"]) > 0
    assert int(g2["n_upstream_segments"]) == 0


# ---------------------------------------------------------------------------
# Higher-level wrappers
# ---------------------------------------------------------------------------


def test_correlate_upstream_rvi_to_severity_returns_result() -> None:
    df = pd.DataFrame(
        {
            "upstream_rvi_p75": [0.1, 0.3, 0.5, 0.7, 0.9, 0.0, 0.2, 0.4],
            "severity_int": [1, 2, 3, 3, 4, 1, 2, 2],
        }
    )
    res = correlate_upstream_rvi_to_severity(df)
    assert res.n == 8
    assert res.rho > 0  # monotonic association by construction.


def test_correlate_returns_nan_when_too_few_points() -> None:
    df = pd.DataFrame(
        {"upstream_rvi_p75": [0.1, 0.5], "severity_int": [1, 3]}
    )
    res = correlate_upstream_rvi_to_severity(df, min_pairs=5)
    assert np.isnan(res.rho)


def test_stratified_correlation_separates_quality_tiers() -> None:
    df = pd.DataFrame(
        {
            "upstream_rvi_p75": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "severity_int": [1, 2, 3, 4, 4, 1, 2, 2, 3, 3],
            "quality_verified": [True, True, True, True, True, False, False, False, False, False],
        }
    )
    res = stratified_correlation(df, rvi_field="upstream_rvi_p75", severity_field="severity_int")
    assert set(res.keys()) == {"all", "quality_verified", "non_quality_verified"}
    assert res["quality_verified"].n == 5
    assert res["non_quality_verified"].n == 5
