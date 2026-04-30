"""Riparian buffer generation (§5.3 of the proposal).

The legal Kenyan setback is measured **from the highest water mark on the
bank**, not from the centreline that OSM ships. We therefore buffer at the
*total* radius

.. math::

    r_s = B_s + h_s

where :math:`B_s` is the legal setback (6 / 10 / 30 m) and :math:`h_s` is the
Strahler-order half-width offset that approximates the centreline-to-bank
distance for waterway segment *s*.

Buffer polygons use:

* ``cap_style=2`` (flat) — produces clean rectangular corridors instead of
  rounded ends that would otherwise overlap at confluences.
* ``join_style=2`` (mitre) — sharp corners at sinuous bends.

All operations happen in :data:`Config.crs_metric` (UTM 37N). If the input is
in geographic coordinates the function reprojects, buffers in metres, and
either keeps the result in metric CRS (the default) or reprojects back.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from rvi.config import Config, get_config

logger = logging.getLogger(__name__)


BUFFER_COLUMNS_EXTRA: tuple[str, ...] = (
    "buffer_width_m",
    "half_width_m",
    "buffer_radius_m",
    "buffer_area_m2",
)


def buffer_radius(
    *,
    legal_buffer_m: float,
    strahler: int,
    config: Config | None = None,
) -> float:
    """Return :math:`r_s = B_s + h_s` for one waterway feature."""
    cfg = config or get_config()
    return float(legal_buffer_m) + cfg.half_width_for_strahler(int(strahler))


def buffer_waterways(
    waterways: gpd.GeoDataFrame,
    width_m: float,
    *,
    config: Config | None = None,
    keep_metric: bool = True,
    cap_style: int = 2,
    join_style: int = 2,
    mitre_limit: float = 5.0,
) -> gpd.GeoDataFrame:
    """Buffer each waterway feature at *width_m* (metres) plus its Strahler offset.

    Parameters
    ----------
    waterways
        GeoDataFrame produced by :mod:`rvi.ingestion.osm`. Must have a
        ``strahler`` integer column.
    width_m
        Legal setback in metres (6, 10, or 30).
    keep_metric
        If True (the default), the output stays in :data:`Config.crs_metric`,
        which is what the segmentation and encroachment stages expect. Pass
        ``False`` if you need the output in EPSG:4326 for plotting.
    """
    cfg = config or get_config()
    if waterways.empty:
        return _empty_buffer_gdf(waterways.crs or cfg.crs_metric)

    if "strahler" not in waterways.columns:
        raise ValueError("waterways GeoDataFrame must contain a 'strahler' column")

    metric = (
        waterways.to_crs(cfg.crs_metric) if waterways.crs != cfg.crs_metric else waterways.copy()
    )
    if metric.crs is None:
        metric = metric.set_crs(cfg.crs_metric)

    radii = pd.Series(
        [
            buffer_radius(legal_buffer_m=width_m, strahler=int(s), config=cfg)
            for s in metric["strahler"].fillna(1).astype(int)
        ],
        index=metric.index,
        dtype=float,
    )

    half_widths = radii - float(width_m)

    geoms = [
        _safe_buffer(
            geom,
            distance=float(r),
            cap_style=cap_style,
            join_style=join_style,
            mitre_limit=mitre_limit,
        )
        for geom, r in zip(metric.geometry, radii, strict=True)
    ]

    out = metric.copy()
    out["geometry"] = geoms
    out["buffer_width_m"] = float(width_m)
    out["half_width_m"] = half_widths.values
    out["buffer_radius_m"] = radii.values
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=cfg.crs_metric)
    out = out[~out.geometry.is_empty & out.geometry.notna()].copy()
    out["buffer_area_m2"] = out.geometry.area.astype(float)

    if not keep_metric:
        out = out.to_crs(cfg.crs_geographic)
    return out


def buffer_waterways_multi(
    waterways: gpd.GeoDataFrame,
    *,
    config: Config | None = None,
    widths_m: Sequence[float] | None = None,
) -> dict[float, gpd.GeoDataFrame]:
    """Generate one buffer GeoDataFrame per legal width.

    Returns a dict keyed by width in metres so downstream code can pick the
    one for the analysis it is performing.
    """
    cfg = config or get_config()
    widths = tuple(float(w) for w in (widths_m or cfg.buffer_widths_m))
    return {w: buffer_waterways(waterways, width_m=w, config=cfg) for w in widths}


def union_buffer(buffers: gpd.GeoDataFrame) -> BaseGeometry:
    """Return the dissolved union of all buffer polygons.

    Used by the buildings DuckDB stream to filter footprints in a single
    spatial predicate.
    """
    if buffers.empty:
        return Polygon()
    # Shapely 2 / geopandas 1: prefer the explicit union_all() method.
    geom = buffers.geometry
    if hasattr(geom, "union_all"):
        return geom.union_all()
    return geom.unary_union  # pragma: no cover - geopandas < 1.0


def _safe_buffer(
    geom: BaseGeometry | None,
    *,
    distance: float,
    cap_style: int,
    join_style: int,
    mitre_limit: float,
) -> BaseGeometry:
    if geom is None or geom.is_empty:
        return Polygon()
    try:
        result = geom.buffer(
            distance=distance,
            cap_style=cap_style,
            join_style=join_style,
            mitre_limit=mitre_limit,
        )
    except Exception as exc:  # pragma: no cover - shapely seldom raises
        logger.warning("buffer failed (distance=%.1f): %s", distance, exc)
        return Polygon()
    if isinstance(result, (Polygon, MultiPolygon)):
        return result
    return Polygon()


def _empty_buffer_gdf(crs: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "buffer_width_m": pd.Series(dtype=float),
            "half_width_m": pd.Series(dtype=float),
            "buffer_radius_m": pd.Series(dtype=float),
            "buffer_area_m2": pd.Series(dtype=float),
        },
        geometry=gpd.GeoSeries([], crs=crs),
        crs=crs,
    )


__all__ = [
    "BUFFER_COLUMNS_EXTRA",
    "buffer_radius",
    "buffer_waterways",
    "buffer_waterways_multi",
    "union_buffer",
]
