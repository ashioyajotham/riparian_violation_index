"""Tests for ``rvi.viz.choropleth`` (smoke + structural assertions only)."""

from __future__ import annotations

import pandas as pd

from rvi.analysis.encroachment import detect_encroachment
from rvi.analysis.rvi import compute_rvi
from rvi.analysis.validation import SpearmanResult
from rvi.config import Config
from rvi.geometry.segment import segment_waterways
from rvi.viz.choropleth import (
    aggregate_rvi_by_county,
    county_choropleth,
    rvi_severity_scatter,
    segment_map,
    sensitivity_heatmap,
)


def test_segment_map_returns_folium_object(
    waterways_metric, buildings_metric, cfg: Config, tmp_path
) -> None:
    import folium

    segments = segment_waterways(waterways_metric, config=cfg)
    enc = detect_encroachment(segments, buildings_metric, buffer_width_m=30, config=cfg)
    scored = compute_rvi(enc, config=cfg)
    fmap = segment_map(scored, config=cfg, title="Test")
    assert isinstance(fmap, folium.Map)
    out = tmp_path / "map.html"
    fmap.save(str(out))
    html = out.read_text(encoding="utf-8")
    assert "RVI" in html


def test_county_choropleth_smoke(
    waterways_metric, buildings_metric, counties_metric, cfg: Config
) -> None:
    import folium

    segments = segment_waterways(waterways_metric, config=cfg)
    enc = detect_encroachment(segments, buildings_metric, buffer_width_m=30, config=cfg)
    scored = compute_rvi(enc, config=cfg)
    agg = aggregate_rvi_by_county(scored, counties_metric, config=cfg)
    fmap = county_choropleth(counties_metric, agg, config=cfg)
    assert isinstance(fmap, folium.Map)


def test_aggregate_rvi_by_county_returns_one_row_per_assigned_county(
    waterways_metric, buildings_metric, counties_metric, cfg: Config
) -> None:
    segments = segment_waterways(waterways_metric, config=cfg)
    enc = detect_encroachment(segments, buildings_metric, buffer_width_m=30, config=cfg)
    scored = compute_rvi(enc, config=cfg)
    agg = aggregate_rvi_by_county(scored, counties_metric, config=cfg)
    assert "rvi_mean" in agg.columns
    assert "n_segments" in agg.columns
    assert (agg["n_segments"] > 0).all()


def test_rvi_severity_scatter_with_result_annotation(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "upstream_rvi_p75": [0.1, 0.3, 0.5, 0.7, 0.9],
            "severity_int": [1, 2, 3, 3, 4],
        }
    )
    res = SpearmanResult(rho=0.92, pvalue=0.01, ci_low=0.5, ci_high=1.0, n=5)
    out = tmp_path / "scatter.png"
    fig = rvi_severity_scatter(df, result=res, save_path=out)
    assert out.exists()
    assert out.stat().st_size > 0
    fig.clear()


def test_rvi_severity_scatter_handles_empty(tmp_path) -> None:
    out = tmp_path / "scatter_empty.png"
    fig = rvi_severity_scatter(pd.DataFrame({"upstream_rvi_p75": [], "severity_int": []}), save_path=out)
    assert out.exists()
    fig.clear()


def test_sensitivity_heatmap_smoke(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "alpha": [0.0, 0.0, 0.5, 0.5, 1.0],
            "beta": [0.0, 1.0, 0.0, 0.5, 0.0],
            "gamma": [1.0, 0.0, 0.5, 0.0, 0.0],
            "rvi_composite": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    out = tmp_path / "sens.png"
    fig = sensitivity_heatmap(df, save_path=out)
    assert out.exists()
    fig.clear()
