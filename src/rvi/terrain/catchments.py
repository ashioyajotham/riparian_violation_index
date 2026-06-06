"""DEM-driven gauge catchment delineation for Phase 2 validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from shapely.geometry import shape
from shapely.ops import unary_union

try:  # pragma: no cover - exercised indirectly in environments with extras
    import rasterio
    from pysheds.grid import Grid
    from rasterio import features
except ImportError:  # pragma: no cover
    Grid = None  # type: ignore[assignment]
    features = None  # type: ignore[assignment]
    rasterio = None  # type: ignore[assignment]

from rvi.config import Config, get_config

logger = logging.getLogger(__name__)


def delineate_catchments_from_dem(
    gauges: gpd.GeoDataFrame,
    dem_path: Path | str,
    *,
    gauge_id_column: str = "gauge_id",
    output_crs: str | None = None,
    config: Config | None = None,
) -> gpd.GeoDataFrame:
    """Delineate one catchment polygon per gauge from a DEM raster.

    Gauges are reprojected into the DEM CRS, the DEM is routed with
    :mod:`pysheds`, and each catchment mask is polygonized with
    :func:`rasterio.features.shapes`.
    """
    cfg = config or get_config()
    if gauges.empty:
        return gpd.GeoDataFrame(
            {gauge_id_column: []},
            geometry=gpd.GeoSeries([], crs=output_crs or cfg.crs_geographic),
            crs=output_crs or cfg.crs_geographic,
        )
    if Grid is None or rasterio is None or features is None:
        raise ImportError(
            'DEM catchment delineation requires the "phase2" extra: '
            'pip install -e ".[phase2]"'
        )

    dem_path = Path(dem_path)
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs or cfg.crs_geographic

    gauge_frame = gauges
    if gauge_frame.crs is None:
        gauge_frame = gauge_frame.set_crs(cfg.crs_geographic)
    if dem_crs is not None and gauge_frame.crs != dem_crs:
        gauge_frame = gauge_frame.to_crs(dem_crs)

    grid = Grid.from_raster(str(dem_path))
    dem = grid.read_raster(str(dem_path))
    fdir = grid.flowdir(dem)

    rows: list[dict[str, Any]] = []
    for _, row in gauge_frame.iterrows():
        geom = row.geometry
        gauge_id = row.get(gauge_id_column)
        if geom is None or geom.is_empty or gauge_id is None:
            continue
        try:
            catchment = grid.catchment(
                x=float(geom.x),
                y=float(geom.y),
                fdir=fdir,
                xytype="coordinate",
                snap="corner",
            )
        except Exception as exc:
            logger.warning("Catchment delineation failed for gauge %s: %s", gauge_id, exc)
            continue
        polygon = _polygonize_catchment(catchment)
        if polygon is None or polygon.is_empty:
            continue
        rows.append({gauge_id_column: str(gauge_id), "geometry": polygon})

    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=dem_crs or cfg.crs_geographic)
    if output_crs is not None and not out.empty and out.crs != output_crs:
        out = out.to_crs(output_crs)
    return out


def _polygonize_catchment(catchment: Any):
    mask = np.asarray(catchment).astype(bool)
    if mask.size == 0 or not mask.any():
        return None
    affine = getattr(catchment, "affine", None)
    if affine is None:
        raise ValueError("catchment raster is missing an affine transform")
    polygons = [
        shape(geom)
        for geom, val in features.shapes(
            mask.astype("uint8"),
            mask=mask,
            transform=affine,
        )
        if val == 1
    ]
    if not polygons:
        return None
    return unary_union(polygons)


__all__ = ["delineate_catchments_from_dem"]
