"""Tests for ``rvi.geometry.buffer``."""

from __future__ import annotations

import geopandas as gpd
import pytest

from rvi.config import Config
from rvi.geometry.buffer import (
    buffer_radius,
    buffer_waterways,
    buffer_waterways_multi,
    union_buffer,
)


def test_buffer_radius_adds_strahler_offset(cfg: Config) -> None:
    # River (strahler=4) at 30m legal buffer → 30 + 20 = 50.
    assert buffer_radius(legal_buffer_m=30, strahler=4, config=cfg) == 50.0
    # Stream (strahler=2) at 6m legal buffer → 6 + 3 = 9.
    assert buffer_radius(legal_buffer_m=6, strahler=2, config=cfg) == 9.0


def test_buffer_waterways_returns_metric_polygons(
    waterways_metric: gpd.GeoDataFrame, cfg: Config
) -> None:
    out = buffer_waterways(waterways_metric, width_m=30, config=cfg)
    assert str(out.crs) == cfg.crs_metric
    assert (out.geometry.geom_type.isin({"Polygon", "MultiPolygon"})).all()
    # Each row carries the buffer metadata columns.
    for col in ("buffer_width_m", "half_width_m", "buffer_radius_m", "buffer_area_m2"):
        assert col in out.columns
    # Areas are positive and roughly proportional to length × 2r.
    river = out[out["osm_id"] == "w1"].iloc[0]
    expected_min = 2000.0 * 2.0 * 50.0 * 0.95  # length * 2r * tolerance
    assert river["buffer_area_m2"] >= expected_min


def test_buffer_radius_per_row_uses_strahler(
    waterways_metric: gpd.GeoDataFrame, cfg: Config
) -> None:
    out = buffer_waterways(waterways_metric, width_m=30, config=cfg)
    river = out[out["osm_id"] == "w1"].iloc[0]
    stream = out[out["osm_id"] == "w2"].iloc[0]
    assert river["buffer_radius_m"] == 30 + cfg.half_width_for_strahler(4)
    assert stream["buffer_radius_m"] == 30 + cfg.half_width_for_strahler(2)
    # half_width column exposes the offset alone.
    assert river["half_width_m"] == cfg.half_width_for_strahler(4)


def test_buffer_widths_grow_monotonically(
    waterways_metric: gpd.GeoDataFrame, cfg: Config
) -> None:
    a6 = buffer_waterways(waterways_metric, width_m=6, config=cfg)
    a10 = buffer_waterways(waterways_metric, width_m=10, config=cfg)
    a30 = buffer_waterways(waterways_metric, width_m=30, config=cfg)
    # Larger buffer ⇒ larger area for every feature.
    for osm_id in waterways_metric["osm_id"]:
        v6 = float(a6[a6["osm_id"] == osm_id]["buffer_area_m2"].iloc[0])
        v10 = float(a10[a10["osm_id"] == osm_id]["buffer_area_m2"].iloc[0])
        v30 = float(a30[a30["osm_id"] == osm_id]["buffer_area_m2"].iloc[0])
        assert v6 < v10 < v30


def test_buffer_waterways_multi_returns_all_widths(
    waterways_metric: gpd.GeoDataFrame, cfg: Config
) -> None:
    out = buffer_waterways_multi(waterways_metric, config=cfg)
    assert set(out.keys()) == {6.0, 10.0, 30.0}
    for w, gdf in out.items():
        assert (gdf["buffer_width_m"] == w).all()


def test_buffer_waterways_reprojects_geographic_input(
    waterways_metric: gpd.GeoDataFrame, cfg: Config
) -> None:
    geo = waterways_metric.to_crs("EPSG:4326")
    out = buffer_waterways(geo, width_m=10, config=cfg)
    assert str(out.crs) == cfg.crs_metric


def test_empty_input_returns_empty_buffer_gdf(cfg: Config) -> None:
    empty = gpd.GeoDataFrame(
        {"strahler": []},
        geometry=gpd.GeoSeries([], crs=cfg.crs_metric),
        crs=cfg.crs_metric,
    )
    out = buffer_waterways(empty, width_m=30, config=cfg)
    assert out.empty


def test_missing_strahler_column_raises(
    waterways_metric: gpd.GeoDataFrame, cfg: Config
) -> None:
    bad = waterways_metric.drop(columns=["strahler"])
    with pytest.raises(ValueError, match="strahler"):
        buffer_waterways(bad, width_m=10, config=cfg)


def test_union_buffer_dissolves_overlapping_buffers(
    waterways_metric: gpd.GeoDataFrame, cfg: Config
) -> None:
    out = buffer_waterways(waterways_metric, width_m=30, config=cfg)
    union = union_buffer(out)
    assert union.is_valid
    # The union area is at most the sum of inputs (overlap dissolved).
    assert union.area <= float(out["buffer_area_m2"].sum()) + 1e-6
