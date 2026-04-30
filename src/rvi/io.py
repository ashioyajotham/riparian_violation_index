"""I/O helpers — defensive GeoPackage writers.

Pandas 3.0 made pyarrow-backed strings the default extension for object
columns. After multiple ``DataFrame.merge`` / ``concat`` operations, the
resulting :class:`pandas.arrays.ArrowExtensionArray` can be split into many
small chunks. ``pyogrio.write_dataframe`` calls ``np.asarray(values)`` on
each column, which forces a single contiguous allocation of the *entire*
underlying string buffer — for medium-large frames (a few thousand rows ×
many string columns) this routinely OOMs even though the on-disk file would
be small.

The fix: demote pyarrow / extension dtypes to plain ``object``/``str``
before handing the frame to pyogrio. The resulting GPKG is identical; only
the in-memory representation changes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)


def _demote_pyarrow_strings(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a copy of *gdf* with extension/pyarrow columns coerced to plain types.

    The geometry column is left untouched.
    """
    geom_col = gdf.geometry.name
    safe = gdf.copy()
    for col in list(safe.columns):
        if col == geom_col:
            continue
        ser = safe[col]
        dtype_str = str(ser.dtype).lower()
        is_extension = pd.api.types.is_extension_array_dtype(ser.dtype)
        if not (is_extension or "[pyarrow]" in dtype_str):
            continue
        # Numeric extension types: convert to plain numpy via to_numpy with NaN.
        if pd.api.types.is_integer_dtype(ser.dtype) or pd.api.types.is_float_dtype(
            ser.dtype
        ):
            safe[col] = ser.to_numpy(dtype="float64", na_value=float("nan"))
        elif pd.api.types.is_bool_dtype(ser.dtype):
            safe[col] = ser.to_numpy(dtype="bool", na_value=False)
        else:
            # String / object-like: materialise as Python strings.
            safe[col] = ser.astype("object").where(ser.notna(), None)
    return safe


def write_geopackage(
    gdf: gpd.GeoDataFrame,
    path: Path,
    *,
    layer: str | None = None,
    driver: str = "GPKG",
) -> Path:
    """Write *gdf* to a GeoPackage, neutralising pyarrow-string OOMs.

    Returns the path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = _demote_pyarrow_strings(gdf)
    if layer is None:
        safe.to_file(path, driver=driver)
    else:
        safe.to_file(path, layer=layer, driver=driver)
    return path


__all__ = ["write_geopackage"]
