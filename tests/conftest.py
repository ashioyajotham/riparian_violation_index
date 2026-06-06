"""Shared pytest fixtures for the RVI-Kenya test suite.

All fixtures here are offline / synthetic: every test in the repo runs without
network access. Tests that *would* hit the Flood Hub API mock the underlying
``requests.Session`` instead.
"""

from __future__ import annotations

# Force matplotlib to a non-interactive backend before any other module
# imports it. Some Windows Python installs ship a broken Tk; Agg works
# everywhere and never opens a display.
import os
from pathlib import Path
from itertools import count

import matplotlib

matplotlib.use("Agg", force=True)

import _pytest.pathlib as _pytest_pathlib
import _pytest.tmpdir as _pytest_tmpdir
from typing import Any

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon, box

from rvi.config import Config

# Some Windows sandbox environments mark pytest's temporary root unreadable
# during session teardown, even after the tests themselves have completed.
# Disabling only the dead-symlink cleanup keeps tmp_path usable without
# masking any test failures.
_pytest_pathlib.cleanup_dead_symlinks = lambda _root: None
_pytest_tmpdir.cleanup_dead_symlinks = lambda _root: None
_TMP_COUNTER = count()

# ---------------------------------------------------------------------------
# Coordinates: the synthetic toy basin lives in central Nairobi (UTM 37S).
# Origin is chosen so the EPSG:4326 round-trip stays inside [-1.30°, -1.27°],
# [36.80°, 36.84°], an area smaller than the pilot bbox.
# ---------------------------------------------------------------------------

# UTM 37S coordinates near Nairobi CBD.
ORIGIN_X = 250_000.0
ORIGIN_Y = 9_855_000.0


@pytest.fixture()
def cfg() -> Config:
    """A baseline Config that does not consult the environment."""
    return Config()


@pytest.fixture()
def tmp_path() -> Path:
    """Project-local temp dir that avoids tempfile/pytest tmpdir ACL issues."""
    base = Path("outputs") / f"pytest-tmp-{os.getpid()}"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"case-{next(_TMP_COUNTER):04d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture()
def small_bbox() -> tuple[float, float, float, float]:
    """A small synthetic bbox covering ~3 × 3 km of central Nairobi."""
    return (36.80, -1.30, 36.84, -1.27)


# ---------------------------------------------------------------------------
# Synthetic waterways
# ---------------------------------------------------------------------------


def _line_in_metric(*coords: tuple[float, float]) -> LineString:
    return LineString([(ORIGIN_X + dx, ORIGIN_Y + dy) for dx, dy in coords])


@pytest.fixture()
def waterways_metric() -> gpd.GeoDataFrame:
    """Two synthetic waterways in EPSG:32737 (UTM 37S).

    * a 2-km river running west-to-east at y=0  (strahler 4)
    * a 1-km stream running south-to-north at x=500 (strahler 2)
    """
    river = _line_in_metric((0, 0), (2000, 0))
    stream = _line_in_metric((500, -200), (500, 800))
    return gpd.GeoDataFrame(
        {
            "osm_id": ["w1", "w2"],
            "waterway": ["river", "stream"],
            "name": ["Test River", "Test Stream"],
            "name_local": ["Mto", None],
            "strahler": [4, 2],
            "length_m": [2000.0, 1000.0],
            "geometry": [river, stream],
        },
        geometry="geometry",
        crs="EPSG:32737",
    )


@pytest.fixture()
def waterways_geo(waterways_metric: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return waterways_metric.to_crs("EPSG:4326")


# ---------------------------------------------------------------------------
# Synthetic buildings
# ---------------------------------------------------------------------------


def _square(cx: float, cy: float, side: float = 10.0) -> Polygon:
    half = side / 2.0
    return Polygon(
        [
            (ORIGIN_X + cx - half, ORIGIN_Y + cy - half),
            (ORIGIN_X + cx + half, ORIGIN_Y + cy - half),
            (ORIGIN_X + cx + half, ORIGIN_Y + cy + half),
            (ORIGIN_X + cx - half, ORIGIN_Y + cy + half),
        ]
    )


@pytest.fixture()
def buildings_metric() -> gpd.GeoDataFrame:
    """A handful of synthetic building footprints near the river / stream.

    Layout (offset from origin, all squares of 10 m side):
      * b1 — at (50, 5):    inside the 6 m + half-width buffer of the river.
      * b2 — at (50, 25):   inside the 30 m buffer but outside 6 m / 10 m.
      * b3 — at (1500, 1):  very close to the river (heavy density)
      * b4 — at (1500, 50): just beyond the 30 m buffer
      * b5 — at (510, 0):   inside the stream's narrow buffer
    """
    geoms = [
        _square(50, 5),
        _square(50, 25),
        _square(1500, 1),
        _square(1500, 50),
        _square(510, 0),
    ]
    return gpd.GeoDataFrame(
        {
            "building_id": [1, 2, 3, 4, 5],
            "country": ["Kenya"] * 5,
            "quadkey": ["fake0"] * 5,
            "footprint_area_m2": [100.0] * 5,
            "geometry": geoms,
        },
        geometry="geometry",
        crs="EPSG:32737",
    )


@pytest.fixture()
def buildings_geo(buildings_metric: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return buildings_metric.to_crs("EPSG:4326")


@pytest.fixture()
def empty_buildings() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "building_id": pd.Series(dtype="int64"),
            "country": pd.Series(dtype=str),
            "quadkey": pd.Series(dtype=str),
            "footprint_area_m2": pd.Series(dtype=float),
        },
        geometry=gpd.GeoSeries([], crs="EPSG:32737"),
        crs="EPSG:32737",
    )


# ---------------------------------------------------------------------------
# Synthetic gauges
# ---------------------------------------------------------------------------


@pytest.fixture()
def gauges_metric() -> gpd.GeoDataFrame:
    """Two synthetic Flood Hub gauges near the test waterways."""
    pts = [
        Point(ORIGIN_X + 1500, ORIGIN_Y + 0),  # gauge near the encroached zone
        Point(ORIGIN_X + 50_000, ORIGIN_Y + 50_000),  # remote gauge: zero upstream
    ]
    return gpd.GeoDataFrame(
        {
            "gauge_id": ["g1", "g2"],
            "site_name": ["Nairobi-Stage", "Remote-Hill"],
            "river": ["Test River", "Other"],
            "country_code": ["KE", "KE"],
            "quality_verified": [True, False],
            "source": ["test"] * 2,
            "latitude": [-1.28, -0.9],
            "longitude": [36.83, 37.4],
            "severity": ["SEVERE", "NO_FLOODING"],
            "severity_int": [3, 1],
        },
        geometry=pts,
        crs="EPSG:32737",
    )


# ---------------------------------------------------------------------------
# Overpass JSON fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def overpass_response() -> dict[str, Any]:
    """Minimal but realistic Overpass JSON with one river and one stream."""
    return {
        "version": 0.6,
        "elements": [
            {
                "type": "way",
                "id": 11111,
                "tags": {"waterway": "river", "name": "Nairobi", "name:sw": "Nairobi"},
                "geometry": [
                    {"lat": -1.28, "lon": 36.80},
                    {"lat": -1.28, "lon": 36.82},
                    {"lat": -1.28, "lon": 36.84},
                ],
            },
            {
                "type": "way",
                "id": 22222,
                "tags": {"waterway": "stream", "name": "Mathare"},
                "geometry": [
                    {"lat": -1.29, "lon": 36.82},
                    {"lat": -1.27, "lon": 36.82},
                ],
            },
            {
                # An infrastructure tag we want filtered out.
                "type": "way",
                "id": 33333,
                "tags": {"waterway": "weir", "name": "Some Weir"},
                "geometry": [
                    {"lat": -1.28, "lon": 36.81},
                    {"lat": -1.28, "lon": 36.815},
                ],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Counties polygon fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def counties_metric() -> gpd.GeoDataFrame:
    """Two adjacent counties covering the test waterways."""
    nairobi = box(ORIGIN_X - 100, ORIGIN_Y - 100, ORIGIN_X + 1000, ORIGIN_Y + 1000)
    kiambu = box(ORIGIN_X + 1000, ORIGIN_Y - 100, ORIGIN_X + 3000, ORIGIN_Y + 1000)
    return gpd.GeoDataFrame(
        {
            "county": ["Nairobi", "Kiambu"],
            "geometry": [nairobi, kiambu],
        },
        geometry="geometry",
        crs="EPSG:32737",
    )
