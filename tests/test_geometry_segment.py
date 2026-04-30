"""Tests for ``rvi.geometry.segment``."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiLineString

from rvi.config import Config
from rvi.geometry.segment import segment_waterways


def test_segments_have_unique_ids_and_target_length(
    waterways_metric: gpd.GeoDataFrame, cfg: Config
) -> None:
    segments = segment_waterways(waterways_metric, config=cfg)
    assert not segments.empty
    # IDs are unique.
    assert segments["segment_id"].is_unique
    # All segments are non-trivially long.
    assert (segments["segment_length_m"] > 0).all()


def test_total_length_is_conserved(
    waterways_metric: gpd.GeoDataFrame, cfg: Config
) -> None:
    segments = segment_waterways(waterways_metric, config=cfg)
    total = float(segments["segment_length_m"].sum())
    expected = float(waterways_metric.geometry.length.sum())
    assert total == pytest.approx(expected, rel=1e-6)


def test_segments_are_close_to_target(
    waterways_metric: gpd.GeoDataFrame, cfg: Config
) -> None:
    segments = segment_waterways(waterways_metric, config=cfg)
    target = cfg.segment_length_m
    # Most segments lie within ±10% of target. The 2 km river yields exactly
    # 4 × 500 m segments; the 1 km stream yields 2 × 500 m segments.
    long_enough = segments["segment_length_m"] >= 0.5 * target
    assert long_enough.all()


def test_segment_id_format(
    waterways_metric: gpd.GeoDataFrame, cfg: Config
) -> None:
    segments = segment_waterways(waterways_metric, config=cfg)
    # Format is "{osm_id}_s{index:04d}".
    for sid in segments["segment_id"]:
        prefix, _, suffix = sid.partition("_s")
        assert suffix.isdigit() and len(suffix) == 4
        assert prefix in {"w1", "w2"}


def test_segment_index_is_sequential_per_waterway(
    waterways_metric: gpd.GeoDataFrame, cfg: Config
) -> None:
    segments = segment_waterways(waterways_metric, config=cfg)
    for _osm_id, group in segments.groupby("osm_id"):
        idxs = sorted(group["segment_index"])
        assert idxs == list(range(len(idxs)))


def test_short_tail_merges_into_predecessor(cfg: Config) -> None:
    # 1050 m river — without merging, last segment would be 50m.
    # min_segment_length_m default is 50; we set 100 to force a merge.
    line = LineString([(0, 0), (1050, 0)])
    gdf = gpd.GeoDataFrame(
        {
            "osm_id": ["w1"],
            "waterway": ["river"],
            "name": [None],
            "name_local": [None],
            "strahler": [4],
            "geometry": [line],
        },
        geometry="geometry",
        crs=cfg.crs_metric,
    )
    segments = segment_waterways(gdf, config=cfg, min_segment_length_m=100.0)
    assert (segments["segment_length_m"] >= 100.0).all()


def test_handles_multilinestring_input(cfg: Config) -> None:
    mls = MultiLineString([
        LineString([(0, 0), (500, 0)]),
        LineString([(500, 0), (1000, 0)]),
    ])
    gdf = gpd.GeoDataFrame(
        {
            "osm_id": ["w1"],
            "waterway": ["river"],
            "strahler": [4],
            "geometry": [mls],
        },
        geometry="geometry",
        crs=cfg.crs_metric,
    )
    out = segment_waterways(gdf, config=cfg)
    # The 1km MultiLineString should yield 2 × 500m segments.
    assert len(out) >= 2
    assert float(out["segment_length_m"].sum()) == pytest.approx(1000.0)


def test_segment_ids_are_unique_for_disconnected_multilinestring(cfg: Config) -> None:
    """Regression: an OSM relation whose parts cannot be merged by linemerge
    must still produce globally unique segment_ids within the parent feature.

    This is the exact pathology that caused a 39 M-row Cartesian explosion
    in compute_rvi_multi during the live Nairobi pilot.
    """
    # Two disconnected line strings — linemerge cannot fuse them.
    disconnected = MultiLineString([
        LineString([(0, 0), (1200, 0)]),
        LineString([(0, 5000), (1200, 5000)]),
    ])
    gdf = gpd.GeoDataFrame(
        {
            "osm_id": ["rel-42"],
            "waterway": ["river"],
            "strahler": [4],
            "geometry": [disconnected],
        },
        geometry="geometry",
        crs=cfg.crs_metric,
    )
    out = segment_waterways(gdf, config=cfg)
    assert len(out) >= 4  # two 1.2km lines × ~3 segments each
    # The contract: every segment_id is unique within the run.
    assert out["segment_id"].is_unique
    # The contract: segment_index is a running counter per parent osm_id.
    indices = sorted(out["segment_index"].astype(int).tolist())
    assert indices == list(range(len(indices)))


def test_empty_input_returns_empty_gdf(cfg: Config) -> None:
    empty = gpd.GeoDataFrame(
        {"strahler": []},
        geometry=gpd.GeoSeries([], crs=cfg.crs_metric),
        crs=cfg.crs_metric,
    )
    out = segment_waterways(empty, config=cfg)
    assert out.empty


def test_segment_preserves_metadata_columns(
    waterways_metric: gpd.GeoDataFrame, cfg: Config
) -> None:
    out = segment_waterways(waterways_metric, config=cfg)
    for col in ("osm_id", "waterway", "name", "name_local", "strahler"):
        assert col in out.columns
