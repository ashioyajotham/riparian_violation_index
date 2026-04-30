"""Tests for ``rvi.io`` — defensive GeoPackage writes against pyarrow OOMs."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from rvi.io import _demote_pyarrow_strings, write_geopackage


def _make_pyarrow_gdf() -> gpd.GeoDataFrame:
    """Build a GeoDataFrame whose object columns use pyarrow string dtype."""
    df = pd.DataFrame(
        {
            "name": pd.array(["alpha", "beta", "gamma"], dtype="string[pyarrow]"),
            "score": pd.array([0.1, 0.5, 0.9], dtype="float64[pyarrow]"),
            "ok": pd.array([True, False, True], dtype="bool[pyarrow]"),
        }
    )
    geoms = [
        LineString([(0, 0), (1, 1)]),
        LineString([(1, 1), (2, 0)]),
        LineString([(2, 0), (3, 3)]),
    ]
    return gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")


def test_demote_pyarrow_strings_returns_plain_dtypes() -> None:
    gdf = _make_pyarrow_gdf()
    safe = _demote_pyarrow_strings(gdf)
    # The original frame is left untouched (extension dtype preserved).
    assert pd.api.types.is_extension_array_dtype(gdf["name"].dtype)
    # The demoted frame uses plain numpy/object dtypes that pyogrio understands.
    assert not pd.api.types.is_extension_array_dtype(safe["name"].dtype)
    assert not pd.api.types.is_extension_array_dtype(safe["score"].dtype)
    assert not pd.api.types.is_extension_array_dtype(safe["ok"].dtype)
    assert str(safe["score"].dtype) == "float64"
    assert str(safe["ok"].dtype) == "bool"
    # Geometry column survives untouched.
    assert safe.geometry.name == gdf.geometry.name


def test_write_geopackage_roundtrip(tmp_path) -> None:
    gdf = _make_pyarrow_gdf()
    path = tmp_path / "out.gpkg"
    write_geopackage(gdf, path)
    assert path.exists() and path.stat().st_size > 0
    loaded = gpd.read_file(path)
    assert len(loaded) == len(gdf)
    assert set(loaded["name"]) == {"alpha", "beta", "gamma"}


def test_write_geopackage_handles_plain_frame(tmp_path) -> None:
    """Plain numpy/object dtypes pass through unchanged."""
    gdf = gpd.GeoDataFrame(
        {"a": [1, 2, 3], "b": ["x", "y", "z"]},
        geometry=[LineString([(0, 0), (1, 1)])] * 3,
        crs="EPSG:4326",
    )
    path = tmp_path / "plain.gpkg"
    write_geopackage(gdf, path)
    loaded = gpd.read_file(path)
    assert list(loaded["b"]) == ["x", "y", "z"]


def test_write_geopackage_with_layer_name(tmp_path) -> None:
    gdf = gpd.GeoDataFrame(
        {"x": [1, 2]},
        geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
        crs="EPSG:4326",
    )
    path = tmp_path / "multi.gpkg"
    write_geopackage(gdf, path, layer="custom_layer")
    loaded = gpd.read_file(path, layer="custom_layer")
    assert len(loaded) == 2


@pytest.mark.parametrize(
    "dtype, raw, expected",
    [
        ("Int64", [1, pd.NA, 3], [1.0, float("nan"), 3.0]),
        ("Float64", [0.1, pd.NA, 0.9], [0.1, float("nan"), 0.9]),
    ],
)
def test_demote_handles_nullable_numeric(dtype, raw, expected) -> None:
    gdf = gpd.GeoDataFrame(
        {"v": pd.array(raw, dtype=dtype)},
        geometry=[LineString([(0, 0), (1, 1)])] * 3,
        crs="EPSG:4326",
    )
    safe = _demote_pyarrow_strings(gdf)
    out = list(safe["v"])
    for got, want in zip(out, expected, strict=True):
        if want != want:  # NaN check
            assert got != got
        else:
            assert got == pytest.approx(want)
