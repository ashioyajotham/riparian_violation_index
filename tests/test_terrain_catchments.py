"""Tests for DEM-driven catchment delineation helpers."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import Point

from rvi.terrain import catchments as catchments_mod


class _FakeRaster:
    def __init__(self, array, affine=None):
        self._array = np.asarray(array)
        self.affine = affine or Affine.translation(0, 2) * Affine.scale(1, -1)

    def __array__(self, dtype=None):
        if dtype is None:
            return self._array
        return self._array.astype(dtype)


class _FakeGrid:
    @classmethod
    def from_raster(cls, _path: str):
        return cls()

    def read_raster(self, _path: str):
        return _FakeRaster([[5, 4], [3, 2]])

    def flowdir(self, dem):
        return dem

    def catchment(self, *, x: float, y: float, fdir, xytype: str, snap: str):
        assert xytype == "coordinate"
        assert snap == "corner"
        assert isinstance(x, float)
        assert isinstance(y, float)
        return _FakeRaster([[1, 1], [0, 0]], affine=fdir.affine)


class _FakeRasterio:
    class _Src:
        crs = "EPSG:4326"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    @staticmethod
    def open(_path: Path | str):
        return _FakeRasterio._Src()


class _FakeFeatures:
    @staticmethod
    def shapes(data, *, mask, transform):
        arr = np.asarray(data)
        assert arr.shape == (2, 2)
        assert mask.any()
        assert transform == Affine.translation(0, 2) * Affine.scale(1, -1)
        yield (
            {
                "type": "Polygon",
                "coordinates": [[(0, 2), (2, 2), (2, 1), (0, 1), (0, 2)]],
            },
            1,
        )


def test_delineate_catchments_from_dem_with_mocks(monkeypatch) -> None:
    monkeypatch.setattr(catchments_mod, "Grid", _FakeGrid)
    monkeypatch.setattr(catchments_mod, "rasterio", _FakeRasterio)
    monkeypatch.setattr(catchments_mod, "features", _FakeFeatures)

    gauges = gpd.GeoDataFrame(
        {"gauge_id": ["g1"]},
        geometry=[Point(1.5, 1.5)],
        crs="EPSG:4326",
    )
    out = catchments_mod.delineate_catchments_from_dem(gauges, Path("fake_dem.tif"))
    assert list(out["gauge_id"]) == ["g1"]
    assert str(out.crs) == "EPSG:4326"
    assert out.geometry.iloc[0].area > 0

