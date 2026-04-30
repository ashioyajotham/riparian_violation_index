"""Tests for ``rvi.analysis.encroachment``."""

from __future__ import annotations

import geopandas as gpd
import pytest

from rvi.analysis.encroachment import (
    detect_encroachment,
    detect_encroachment_multi,
)
from rvi.config import Config
from rvi.geometry.segment import segment_waterways


def test_detects_encroaching_buildings_inside_buffer(
    waterways_metric: gpd.GeoDataFrame,
    buildings_metric: gpd.GeoDataFrame,
    cfg: Config,
) -> None:
    segments = segment_waterways(waterways_metric, config=cfg)
    enc = detect_encroachment(segments, buildings_metric, buffer_width_m=30, config=cfg)
    # At least one segment must register encroachment.
    assert (enc["n_buildings"] > 0).any()
    # All segments retained (left-join) including those with zero buildings.
    assert len(enc) == len(segments)
    # Non-zero footprint for encroaching segments.
    encroached = enc[enc["n_buildings"] > 0]
    assert (encroached["total_footprint_m2"] > 0).all()


def test_zero_buildings_yields_zero_stats(
    waterways_metric: gpd.GeoDataFrame,
    empty_buildings: gpd.GeoDataFrame,
    cfg: Config,
) -> None:
    segments = segment_waterways(waterways_metric, config=cfg)
    enc = detect_encroachment(segments, empty_buildings, buffer_width_m=30, config=cfg)
    assert (enc["n_buildings"] == 0).all()
    assert (enc["total_footprint_m2"] == 0).all()


def test_buffer_widths_change_detection_count(
    waterways_metric: gpd.GeoDataFrame,
    buildings_metric: gpd.GeoDataFrame,
    cfg: Config,
) -> None:
    segments = segment_waterways(waterways_metric, config=cfg)
    enc6 = detect_encroachment(segments, buildings_metric, buffer_width_m=6, config=cfg)
    enc30 = detect_encroachment(segments, buildings_metric, buffer_width_m=30, config=cfg)
    # 30m buffer must catch ≥ as many buildings as 6m.
    assert int(enc30["n_buildings"].sum()) >= int(enc6["n_buildings"].sum())


def test_detect_encroachment_multi_keys_match_widths(
    waterways_metric: gpd.GeoDataFrame,
    buildings_metric: gpd.GeoDataFrame,
    cfg: Config,
) -> None:
    segments = segment_waterways(waterways_metric, config=cfg)
    out = detect_encroachment_multi(segments, buildings_metric, config=cfg)
    assert set(out.keys()) == {6.0, 10.0, 30.0}
    for w, gdf in out.items():
        assert (gdf["buffer_width_m"] == w).all()


def test_proximity_distance_metrics_are_finite_when_present(
    waterways_metric: gpd.GeoDataFrame,
    buildings_metric: gpd.GeoDataFrame,
    cfg: Config,
) -> None:
    segments = segment_waterways(waterways_metric, config=cfg)
    enc = detect_encroachment(segments, buildings_metric, buffer_width_m=30, config=cfg)
    encroached = enc[enc["n_buildings"] > 0]
    if len(encroached) > 0:
        assert encroached["mean_dist_m"].notna().all()
        assert (encroached["mean_dist_m"] >= 0).all()


def test_encroachment_requires_strahler_column(
    waterways_metric: gpd.GeoDataFrame,
    buildings_metric: gpd.GeoDataFrame,
    cfg: Config,
) -> None:
    segments = segment_waterways(waterways_metric, config=cfg)
    bad = segments.drop(columns=["strahler"])
    with pytest.raises(ValueError, match="strahler"):
        detect_encroachment(bad, buildings_metric, buffer_width_m=30, config=cfg)


def test_empty_segments_returns_empty(
    cfg: Config, buildings_metric: gpd.GeoDataFrame
) -> None:
    empty = gpd.GeoDataFrame(
        {"segment_id": [], "strahler": [], "segment_length_m": []},
        geometry=gpd.GeoSeries([], crs=cfg.crs_metric),
        crs=cfg.crs_metric,
    )
    enc = detect_encroachment(empty, buildings_metric, buffer_width_m=30, config=cfg)
    assert enc.empty
