"""Encroachment detection (§5.4 of the proposal).

For each 500-metre segment, we count the building footprints whose geometry
intersects the riparian buffer of that segment, measure the area they occupy
inside the buffer, and compute the per-building distance from the river
centreline. The result is the per-segment statistics block consumed by
:mod:`rvi.analysis.rvi`.

Inputs (all in :data:`Config.crs_metric` — UTM 37S):

* ``segments``  — output of :func:`rvi.geometry.segment.segment_waterways`
  (LineString centrelines).
* ``buildings`` — output of :mod:`rvi.ingestion.buildings` (Polygons).
* ``buffer_width_m`` — one of the legal Kenyan setbacks (6 / 10 / 30 m).

Output schema (one row per segment):

* ``segment_id``                       (str)
* ``buffer_width_m``                   (float, e.g. 30.0)
* ``segment_length_m``                 (float)
* ``buffer_radius_m``                  (float, B + h)
* ``buffer_area_m2``                   (float)
* ``n_buildings``                      (int)
* ``total_footprint_m2``               (float)
* ``mean_dist_m``                      (float, NaN if n_buildings==0)
* ``min_dist_m``                       (float, NaN if n_buildings==0)
* ``max_dist_m``                       (float, NaN if n_buildings==0)
* ``geometry``                         (segment LineString, EPSG:32737)
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry

from rvi.config import Config, get_config
from rvi.geometry.buffer import buffer_radius

logger = logging.getLogger(__name__)


ENCROACHMENT_COLUMNS: tuple[str, ...] = (
    "segment_id",
    "buffer_width_m",
    "segment_length_m",
    "buffer_radius_m",
    "buffer_area_m2",
    "n_buildings",
    "total_footprint_m2",
    "mean_dist_m",
    "min_dist_m",
    "max_dist_m",
)


def detect_encroachment(
    segments: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    *,
    buffer_width_m: float,
    config: Config | None = None,
) -> gpd.GeoDataFrame:
    """Compute per-segment encroachment statistics for one buffer width.

    The function constructs the per-segment buffer locally (so segments and
    buffers always stay aligned at this stage), spatially joins building
    footprints into the buffer, and aggregates per segment.
    """
    cfg = config or get_config()
    if "segment_id" not in segments.columns:
        raise ValueError("segments GeoDataFrame must contain a 'segment_id' column")
    if "strahler" not in segments.columns:
        raise ValueError("segments GeoDataFrame must contain a 'strahler' column")

    if segments.empty:
        return _empty_encroachment_gdf(cfg.crs_metric)

    seg_metric = _to_metric(segments, cfg)
    if buildings is None or buildings.empty:
        # All segments default to zero encroachment.
        return _zero_encroachment(seg_metric, buffer_width_m=float(buffer_width_m), cfg=cfg)

    bld_metric = _to_metric(buildings, cfg)

    # Build per-segment buffer with Strahler-corrected radius.
    radii = pd.Series(
        [
            buffer_radius(legal_buffer_m=float(buffer_width_m), strahler=int(s), config=cfg)
            for s in seg_metric["strahler"].fillna(1).astype(int)
        ],
        index=seg_metric.index,
        dtype=float,
    )
    buffers_geo = [
        _safe_buffer(g, r) for g, r in zip(seg_metric.geometry, radii, strict=True)
    ]
    buffers = gpd.GeoDataFrame(
        {
            "segment_id": seg_metric["segment_id"].values,
            "buffer_radius_m": radii.values,
            "buffer_area_m2": [
                float(b.area) if (b is not None and not b.is_empty) else 0.0
                for b in buffers_geo
            ],
            "geometry": buffers_geo,
        },
        geometry="geometry",
        crs=cfg.crs_metric,
    )

    if bld_metric.empty:
        return _attach_zero_stats(seg_metric, buffers, float(buffer_width_m), cfg)

    # Ensure buildings have an id column we can group on.
    if "building_id" not in bld_metric.columns:
        bld_metric = bld_metric.copy()
        bld_metric["building_id"] = pd.RangeIndex(start=1, stop=len(bld_metric) + 1)
    if "footprint_area_m2" not in bld_metric.columns:
        bld_metric = bld_metric.copy()
        bld_metric["footprint_area_m2"] = bld_metric.geometry.area.astype(float)

    # Spatial join — every (segment, building) pair where they intersect.
    bld_subset = bld_metric[["building_id", "footprint_area_m2", "geometry"]].copy()
    joined = gpd.sjoin(
        buffers[["segment_id", "buffer_radius_m", "buffer_area_m2", "geometry"]],
        bld_subset,
        how="inner",
        predicate="intersects",
    )

    if joined.empty:
        return _attach_zero_stats(seg_metric, buffers, float(buffer_width_m), cfg)

    # Centroid → centreline distance per (segment, building).
    seg_lookup = seg_metric.set_index("segment_id").geometry.to_dict()
    bld_lookup_geom = bld_subset.set_index("building_id").geometry.to_dict()
    bld_lookup_area = bld_subset.set_index("building_id")["footprint_area_m2"].to_dict()

    distances = np.empty(len(joined), dtype=float)
    footprint_in_buf = np.empty(len(joined), dtype=float)
    for idx, (sid, bid) in enumerate(
        zip(joined["segment_id"].values, joined["building_id"].values, strict=True)
    ):
        seg_geom = seg_lookup.get(sid)
        bld_geom = bld_lookup_geom.get(bid)
        if seg_geom is None or bld_geom is None:
            distances[idx] = np.nan
            footprint_in_buf[idx] = 0.0
            continue
        centroid = bld_geom.representative_point()
        distances[idx] = float(centroid.distance(seg_geom))
        footprint_in_buf[idx] = float(bld_lookup_area.get(bid, bld_geom.area))

    joined = joined.assign(dist_m=distances, footprint_m2=footprint_in_buf)

    grouped = (
        joined.groupby("segment_id", sort=False)
        .agg(
            n_buildings=("building_id", "nunique"),
            total_footprint_m2=("footprint_m2", "sum"),
            mean_dist_m=("dist_m", "mean"),
            min_dist_m=("dist_m", "min"),
            max_dist_m=("dist_m", "max"),
        )
        .reset_index()
    )

    out = seg_metric.merge(
        buffers[["segment_id", "buffer_radius_m", "buffer_area_m2"]],
        on="segment_id",
        how="left",
    )
    out = out.merge(grouped, on="segment_id", how="left")
    out["buffer_width_m"] = float(buffer_width_m)
    out["n_buildings"] = out["n_buildings"].fillna(0).astype(int)
    out["total_footprint_m2"] = out["total_footprint_m2"].fillna(0.0).astype(float)
    for col in ("mean_dist_m", "min_dist_m", "max_dist_m"):
        out[col] = out[col].astype(float)

    return _select_encroachment_columns(out, cfg)


def detect_encroachment_multi(
    segments: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    *,
    config: Config | None = None,
    widths_m: Iterable[float] | None = None,
) -> dict[float, gpd.GeoDataFrame]:
    """Run :func:`detect_encroachment` for every legal buffer width."""
    cfg = config or get_config()
    widths = tuple(float(w) for w in (widths_m or cfg.buffer_widths_m))
    return {
        w: detect_encroachment(segments, buildings, buffer_width_m=w, config=cfg)
        for w in widths
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _to_metric(gdf: gpd.GeoDataFrame, cfg: Config) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        return gdf.set_crs(cfg.crs_metric)
    if str(gdf.crs).upper() == cfg.crs_metric.upper() or gdf.crs == cfg.crs_metric:
        return gdf
    return gdf.to_crs(cfg.crs_metric)


def _safe_buffer(geom: BaseGeometry | None, radius: float) -> BaseGeometry:
    if geom is None or geom.is_empty:
        from shapely.geometry import Polygon

        return Polygon()
    return geom.buffer(distance=radius, cap_style=2, join_style=2, mitre_limit=5.0)


def _zero_encroachment(
    seg_metric: gpd.GeoDataFrame, *, buffer_width_m: float, cfg: Config
) -> gpd.GeoDataFrame:
    radii = pd.Series(
        [
            buffer_radius(legal_buffer_m=buffer_width_m, strahler=int(s), config=cfg)
            for s in seg_metric["strahler"].fillna(1).astype(int)
        ],
        index=seg_metric.index,
        dtype=float,
    )
    out = seg_metric.copy()
    out["buffer_width_m"] = float(buffer_width_m)
    out["buffer_radius_m"] = radii.values
    # Approximate area (segment_length × 2r) — exact value isn't needed because
    # zero buildings means D=C=P=0 anyway.
    out["buffer_area_m2"] = (out["segment_length_m"].fillna(0.0) * 2.0 * radii).values
    out["n_buildings"] = 0
    out["total_footprint_m2"] = 0.0
    out["mean_dist_m"] = np.nan
    out["min_dist_m"] = np.nan
    out["max_dist_m"] = np.nan
    return _select_encroachment_columns(out, cfg)


def _attach_zero_stats(
    seg_metric: gpd.GeoDataFrame,
    buffers: gpd.GeoDataFrame,
    width: float,
    cfg: Config,
) -> gpd.GeoDataFrame:
    out = seg_metric.merge(
        buffers[["segment_id", "buffer_radius_m", "buffer_area_m2"]],
        on="segment_id",
        how="left",
    )
    out["buffer_width_m"] = float(width)
    out["n_buildings"] = 0
    out["total_footprint_m2"] = 0.0
    out["mean_dist_m"] = np.nan
    out["min_dist_m"] = np.nan
    out["max_dist_m"] = np.nan
    return _select_encroachment_columns(out, cfg)


def _select_encroachment_columns(
    out: gpd.GeoDataFrame, cfg: Config
) -> gpd.GeoDataFrame:
    keep = [
        "segment_id",
        "osm_id",
        "waterway",
        "name",
        "name_local",
        "strahler",
        "buffer_width_m",
        "segment_length_m",
        "buffer_radius_m",
        "buffer_area_m2",
        "n_buildings",
        "total_footprint_m2",
        "mean_dist_m",
        "min_dist_m",
        "max_dist_m",
        "geometry",
    ]
    keep = [c for c in keep if c in out.columns or c == "geometry"]
    out = out[keep]
    return gpd.GeoDataFrame(out, geometry="geometry", crs=cfg.crs_metric)


def _empty_encroachment_gdf(crs: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "segment_id": pd.Series(dtype=str),
            "buffer_width_m": pd.Series(dtype=float),
            "segment_length_m": pd.Series(dtype=float),
            "buffer_radius_m": pd.Series(dtype=float),
            "buffer_area_m2": pd.Series(dtype=float),
            "n_buildings": pd.Series(dtype="int64"),
            "total_footprint_m2": pd.Series(dtype=float),
            "mean_dist_m": pd.Series(dtype=float),
            "min_dist_m": pd.Series(dtype=float),
            "max_dist_m": pd.Series(dtype=float),
        },
        geometry=gpd.GeoSeries([], crs=crs),
        crs=crs,
    )


__all__ = [
    "ENCROACHMENT_COLUMNS",
    "detect_encroachment",
    "detect_encroachment_multi",
]
