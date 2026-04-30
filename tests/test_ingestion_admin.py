"""Tests for ``rvi.ingestion.admin`` — Kenya county boundaries (GADM)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, mapping

from rvi.config import Config
from rvi.ingestion.admin import (
    COUNTY_COLUMNS,
    DEFAULT_KENYA_COUNTIES_URL,
    _normalise_counties,
    download_kenya_counties,
    load_kenya_counties,
)


def _gadm_style_geojson(n: int = 2) -> bytes:
    """Build a tiny GADM-format GeoJSON FeatureCollection."""
    polys = [
        Polygon([(36.7, -1.4), (36.9, -1.4), (36.9, -1.2), (36.7, -1.2), (36.7, -1.4)]),
        Polygon([(37.0, -0.5), (37.5, -0.5), (37.5, -0.0), (37.0, -0.0), (37.0, -0.5)]),
    ][:n]
    names = ["Nairobi", "Kiambu"][:n]
    features = [
        {
            "type": "Feature",
            "properties": {
                "GID_0": "KEN",
                "GID_1": f"KEN.{i + 1}_1",
                "NAME_1": names[i],
                "HASC_1": f"KE.{names[i][:2].upper()}",
            },
            "geometry": mapping(polys[i]),
        }
        for i in range(n)
    ]
    fc = {"type": "FeatureCollection", "features": features}
    return json.dumps(fc).encode("utf-8")


def test_default_url_is_gadm_kenya() -> None:
    assert "gadm" in DEFAULT_KENYA_COUNTIES_URL
    assert "KEN" in DEFAULT_KENYA_COUNTIES_URL
    assert DEFAULT_KENYA_COUNTIES_URL.endswith(".json")


def test_download_kenya_counties_caches_response(cfg: Config, tmp_path) -> None:
    target = tmp_path / "kenya_counties.geojson"
    response = MagicMock()
    response.status_code = 200
    response.content = _gadm_style_geojson()
    response.raise_for_status.return_value = None
    session = MagicMock()
    session.get.return_value = response

    custom_cfg = Config(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        interim_dir=tmp_path / "data" / "interim",
        processed_dir=tmp_path / "data" / "processed",
        cache_dir=tmp_path / "data" / "cache",
        outputs_dir=tmp_path / "outputs",
    )

    gdf = download_kenya_counties(
        config=custom_cfg, target_path=target, session=session
    )
    assert not gdf.empty
    assert list(gdf.columns) == list(COUNTY_COLUMNS)
    assert set(gdf["county"]) == {"Nairobi", "Kiambu"}
    assert target.exists()

    # Second call must NOT re-download.
    session.get.reset_mock()
    gdf2 = download_kenya_counties(
        config=custom_cfg, target_path=target, session=session
    )
    session.get.assert_not_called()
    assert len(gdf2) == len(gdf)


def test_load_kenya_counties_reads_local_file(cfg: Config, tmp_path) -> None:
    path = tmp_path / "kenya_counties.geojson"
    path.write_bytes(_gadm_style_geojson())
    gdf = load_kenya_counties(path, config=cfg)
    assert set(gdf["county"]) == {"Nairobi", "Kiambu"}
    assert (gdf.geometry.is_valid).all()


def test_normalise_counties_requires_name_column(cfg: Config) -> None:
    bogus = gpd.GeoDataFrame(
        {"random_field": ["x", "y"]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 2)]),
        ],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="name column"):
        _normalise_counties(bogus, cfg)


def test_normalise_counties_drops_empty_geometries(cfg: Config) -> None:
    df = gpd.GeoDataFrame(
        {
            "GID_1": ["KEN.1_1", "KEN.2_1"],
            "NAME_1": ["Real", "Empty"],
        },
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
            Polygon(),
        ],
        crs="EPSG:4326",
    )
    out = _normalise_counties(df, cfg)
    assert len(out) == 1
    assert out["county"].iloc[0] == "Real"
