"""Linear-referencing waterway segmentation (§5.3 of the proposal).

Each waterway centreline is cut into roughly :data:`Config.segment_length_m`
chunks (default 500 m) using :func:`shapely.ops.substring`. Segment ids take
the form ``{osm_id}_s{index:04d}`` per the proposal.

Conservation of total river length is preserved to within floating-point
precision: the union of all segment lengths equals the input centreline
length.

Tail segments shorter than :data:`Config.min_segment_length_m` are merged
back into their previous segment so that all output segments are at least
``min_segment_length_m`` long.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, substring

from rvi.config import Config, get_config

logger = logging.getLogger(__name__)


SEGMENT_COLUMNS_EXTRA: tuple[str, ...] = (
    "segment_id",
    "segment_index",
    "segment_length_m",
)


def segment_waterways(
    waterways: gpd.GeoDataFrame,
    *,
    config: Config | None = None,
    segment_length_m: float | None = None,
    min_segment_length_m: float | None = None,
) -> gpd.GeoDataFrame:
    """Cut a waterway GeoDataFrame into ~uniform-length segments.

    Returns a GeoDataFrame with one row per segment, in
    :data:`Config.crs_metric` (UTM 37S), preserving the parent's metadata
    columns (``waterway``, ``name``, ``name_local``, ``strahler``, ``osm_id``)
    plus the segmentation columns:

    * ``segment_id`` (str) — ``{osm_id}_s{index:04d}``
    * ``segment_index`` (int)
    * ``segment_length_m`` (float)
    """
    cfg = config or get_config()
    target = float(segment_length_m or cfg.segment_length_m)
    min_len = float(min_segment_length_m or cfg.min_segment_length_m)
    if target <= 0:
        raise ValueError("segment_length_m must be positive")
    if min_len < 0:
        raise ValueError("min_segment_length_m must be non-negative")

    if waterways.empty:
        return _empty_segments_gdf(waterways.crs or cfg.crs_metric)

    metric = (
        waterways.to_crs(cfg.crs_metric)
        if waterways.crs != cfg.crs_metric
        else waterways.copy()
    )
    if metric.crs is None:
        metric = metric.set_crs(cfg.crs_metric)

    rows: list[dict[str, object]] = []
    for _, row in metric.iterrows():
        geom: BaseGeometry | None = row.geometry
        if geom is None or geom.is_empty:
            continue
        parent_meta = {k: row.get(k) for k in row.index if k != "geometry"}
        osm_id = str(parent_meta.get("osm_id", "noid"))
        # Use a single counter per parent feature so segment_ids stay unique
        # even when the parent's geometry is a disconnected MultiLineString
        # whose parts ``linemerge`` cannot fuse into one LineString.
        running_idx = 0
        for line in _iter_simple_lines(geom):
            for sub in _cut_line(line, target=target, min_len=min_len):
                if sub.is_empty or sub.length <= 0:
                    continue
                rows.append(
                    {
                        **parent_meta,
                        "geometry": sub,
                        "segment_index": running_idx,
                        "segment_id": f"{osm_id}_s{running_idx:04d}",
                        "segment_length_m": float(sub.length),
                    }
                )
                running_idx += 1

    if not rows:
        return _empty_segments_gdf(cfg.crs_metric)

    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=cfg.crs_metric)
    return _stable_segment_columns(out)


def _iter_simple_lines(geom: BaseGeometry) -> Iterable[LineString]:
    """Yield LineStrings from any LineString / MultiLineString input."""
    if isinstance(geom, LineString):
        yield geom
        return
    if isinstance(geom, MultiLineString):
        merged = linemerge(geom)
        if isinstance(merged, LineString):
            yield merged
        elif isinstance(merged, MultiLineString):
            yield from merged.geoms
        else:  # pragma: no cover - defensive
            for g in geom.geoms:
                if isinstance(g, LineString):
                    yield g
        return
    return


def _cut_line(
    line: LineString, *, target: float, min_len: float
) -> list[LineString]:
    """Slice one LineString into segments of length ~`target` metres."""
    total = line.length
    if total <= 0:
        return []
    n_segments = max(1, math.ceil(total / target))
    breakpoints = np.linspace(0.0, total, n_segments + 1).tolist()

    pieces: list[LineString] = []
    for a, b in itertools.pairwise(breakpoints):
        if b - a <= 0:
            continue
        sub = substring(line, a, b)
        if isinstance(sub, LineString) and not sub.is_empty:
            pieces.append(sub)

    # Merge any too-short tail back into its predecessor.
    if min_len > 0 and len(pieces) >= 2 and pieces[-1].length < min_len:
        last = pieces.pop()
        prev = pieces.pop()
        merged = LineString(list(prev.coords) + list(last.coords)[1:])
        pieces.append(merged)

    return pieces


def _stable_segment_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Order columns deterministically (helps when persisting GeoPackages)."""
    leading = [
        "segment_id",
        "segment_index",
        "osm_id",
        "waterway",
        "name",
        "name_local",
        "strahler",
        "segment_length_m",
    ]
    cols = [c for c in leading if c in gdf.columns]
    rest = [c for c in gdf.columns if c not in cols and c != "geometry"]
    return gdf[[*cols, *rest, "geometry"]]


def _empty_segments_gdf(crs: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "segment_id": pd.Series(dtype=str),
            "segment_index": pd.Series(dtype="int64"),
            "segment_length_m": pd.Series(dtype=float),
        },
        geometry=gpd.GeoSeries([], crs=crs),
        crs=crs,
    )


__all__ = ["SEGMENT_COLUMNS_EXTRA", "segment_waterways"]
