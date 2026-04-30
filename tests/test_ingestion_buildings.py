"""Tests for ``rvi.ingestion.buildings`` (offline)."""

from __future__ import annotations

import gzip
from unittest.mock import MagicMock

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from rvi.config import Config
from rvi.ingestion import buildings as bld_mod
from rvi.ingestion.buildings import (
    load_buildings_for_bbox,
    lonlat_to_tile,
    quadkeys_for_bbox,
    tile_to_quadkey,
)

# ---------------------------------------------------------------------------
# Quadkey arithmetic
# ---------------------------------------------------------------------------


def test_lonlat_to_tile_known_values() -> None:
    # (0, 0) at zoom 1 → tile (1, 1)
    assert lonlat_to_tile(0.0, 0.0, 1) == (1, 1)
    # Top-left of the world is (0, 0).
    assert lonlat_to_tile(-179.99, 85.0, 1) == (0, 0)


def test_tile_to_quadkey_known_values() -> None:
    # Bing-Maps reference: (3, 5) at zoom 3 → "213".
    assert tile_to_quadkey(3, 5, 3) == "213"


def test_quadkeys_for_bbox_returns_unique_strings() -> None:
    keys = quadkeys_for_bbox((36.80, -1.30, 36.84, -1.27), zoom=9)
    assert all(isinstance(k, str) for k in keys)
    assert len(keys) == len(set(keys))
    # The Nairobi pilot bbox is small enough to land in 1–4 tiles at zoom 9.
    assert 1 <= len(keys) <= 6


# ---------------------------------------------------------------------------
# Loader pipeline (mocked HTTP)
# ---------------------------------------------------------------------------


def _fake_index_csv(quadkey: str) -> bytes:
    csv = (
        "Location,QuadKey,Url,Size\n"
        f"Kenya,{quadkey},https://example/fake-tile.csv.gz,123\n"
        "OtherCountry,zzz,https://example/other.csv.gz,123\n"
    )
    return csv.encode("utf-8")


def _fake_tile_csv() -> bytes:
    # One footprint just south of the river, big enough to register area.
    polygon = Polygon([(36.81, -1.281), (36.81, -1.279), (36.812, -1.279), (36.812, -1.281)])
    # WKT contains commas, so the column must be CSV-quoted.
    csv = 'QuadKey,Location,geometry_wkt\n' + f'qk,Kenya,"{polygon.wkt}"\n'
    return gzip.compress(csv.encode("utf-8"))


def _fake_tile_geojsonl() -> bytes:
    """Microsoft 2026-format tile: one GeoJSON Feature per line."""
    import json

    coords = [
        (36.81, -1.281), (36.81, -1.279), (36.812, -1.279), (36.812, -1.281),
        (36.81, -1.281),
    ]
    feature = {
        "type": "Feature",
        "properties": {"height": -1.0, "confidence": -1.0},
        "geometry": {"type": "Polygon", "coordinates": [list(map(list, coords))]},
    }
    return gzip.compress((json.dumps(feature) + "\n").encode("utf-8"))


def test_load_buildings_for_bbox_assembles_tiles(
    cfg: Config, small_bbox, tmp_path
) -> None:
    # Pick whatever quadkey the bbox actually covers and inject our fake tile.
    quadkeys = quadkeys_for_bbox(small_bbox, zoom=9)
    qk = quadkeys[0]

    session = MagicMock()
    index_response = MagicMock()
    index_response.status_code = 200
    index_response.content = _fake_index_csv(qk)
    index_response.raise_for_status.return_value = None

    tile_response = MagicMock()
    tile_response.status_code = 200
    tile_response.content = _fake_tile_csv()
    tile_response.raise_for_status.return_value = None
    tile_response.__enter__ = lambda self: self
    tile_response.__exit__ = lambda self, *a, **kw: None

    def get_side(url, **_kwargs):
        if "dataset-links" in url or url.endswith(".csv"):
            return index_response
        return tile_response

    session.get.side_effect = get_side

    custom_cfg = Config(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        interim_dir=tmp_path / "data" / "interim",
        processed_dir=tmp_path / "data" / "processed",
        cache_dir=tmp_path / "data" / "cache",
        outputs_dir=tmp_path / "outputs",
    )

    gdf = load_buildings_for_bbox(
        small_bbox,
        config=custom_cfg,
        session=session,
        progress=False,
    )
    assert not gdf.empty
    assert "footprint_area_m2" in gdf.columns
    assert (gdf["footprint_area_m2"] > 0).all()


def test_load_buildings_for_bbox_uses_cache(cfg: Config, small_bbox, tmp_path) -> None:
    custom_cfg = Config(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        interim_dir=tmp_path / "data" / "interim",
        processed_dir=tmp_path / "data" / "processed",
        cache_dir=tmp_path / "data" / "cache",
        outputs_dir=tmp_path / "outputs",
    )
    custom_cfg.ensure_dirs()

    # Pre-populate the per-bbox cache with a tiny GeoPackage.
    cache_path = (
        custom_cfg.cache_dir / f"ms_buildings_bbox_{bld_mod._bbox_slug(small_bbox)}.gpkg"
    )
    fake = bld_mod._empty_buildings_gdf(custom_cfg)
    fake.to_file(cache_path, driver="GPKG")

    session = MagicMock()  # must NOT be called
    out = load_buildings_for_bbox(
        small_bbox, config=custom_cfg, session=session, progress=False
    )
    assert "geometry" in out.columns
    session.get.assert_not_called()


def test_no_matching_country_raises(cfg: Config, small_bbox, tmp_path) -> None:
    custom_cfg = Config(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        interim_dir=tmp_path / "data" / "interim",
        processed_dir=tmp_path / "data" / "processed",
        cache_dir=tmp_path / "data" / "cache",
        outputs_dir=tmp_path / "outputs",
    )
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.content = b"Location,QuadKey,Url\nMars,xxx,https://example/x.csv.gz\n"
    response.raise_for_status.return_value = None
    session.get.return_value = response

    with pytest.raises(ValueError, match="country"):
        load_buildings_for_bbox(
            small_bbox, config=custom_cfg, session=session, progress=False
        )


# ---------------------------------------------------------------------------
# Geometry parser
# ---------------------------------------------------------------------------


def test_rows_to_geodataframe_parses_wkt(cfg: Config) -> None:
    df = pd.DataFrame(
        {
            "QuadKey": ["qk"],
            "Location": ["Kenya"],
            "geometry_wkt": ["POLYGON((36.8 -1.28, 36.81 -1.28, 36.81 -1.27, 36.8 -1.27, 36.8 -1.28))"],
        }
    )
    gdf = bld_mod._rows_to_geodataframe(df, cfg)
    assert not gdf.empty
    assert gdf.geometry.iloc[0].is_valid


def test_load_buildings_for_bbox_handles_geojsonl_format(
    cfg: Config, small_bbox, tmp_path
) -> None:
    """Microsoft's 2026 tiles ship as GeoJSONL despite the .csv.gz extension."""
    quadkeys = quadkeys_for_bbox(small_bbox, zoom=9)
    qk = quadkeys[0]

    session = MagicMock()
    index_response = MagicMock()
    index_response.status_code = 200
    index_response.content = _fake_index_csv(qk)
    index_response.raise_for_status.return_value = None

    tile_response = MagicMock()
    tile_response.status_code = 200
    tile_response.content = _fake_tile_geojsonl()
    tile_response.raise_for_status.return_value = None

    def get_side(url, **_kwargs):
        if "dataset-links" in url or url.endswith(".csv"):
            return index_response
        return tile_response

    session.get.side_effect = get_side

    custom_cfg = Config(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        interim_dir=tmp_path / "data" / "interim",
        processed_dir=tmp_path / "data" / "processed",
        cache_dir=tmp_path / "data" / "cache",
        outputs_dir=tmp_path / "outputs",
    )

    gdf = load_buildings_for_bbox(
        small_bbox, config=custom_cfg, session=session, progress=False
    )
    assert not gdf.empty, "GeoJSONL tile should yield at least one footprint"
    assert (gdf["footprint_area_m2"] > 0).all()


def test_parse_tile_text_dispatches_on_first_byte(cfg: Config) -> None:
    """The format sniff: '{' -> JSONL; otherwise -> CSV."""
    csv_text = (
        'QuadKey,Location,geometry_wkt\n'
        'qk,Kenya,"POLYGON((36.8 -1.28, 36.81 -1.28, 36.81 -1.27, 36.8 -1.27, 36.8 -1.28))"\n'
    )
    csv_gdf = bld_mod._parse_tile_text(csv_text, cfg)
    assert not csv_gdf.empty

    jsonl_text = (
        '{"type":"Feature","properties":{},"geometry":'
        '{"type":"Polygon","coordinates":'
        '[[[36.8,-1.28],[36.81,-1.28],[36.81,-1.27],[36.8,-1.27],[36.8,-1.28]]]}}\n'
    )
    jsonl_gdf = bld_mod._parse_tile_text(jsonl_text, cfg)
    assert not jsonl_gdf.empty
    assert jsonl_gdf.geometry.iloc[0].is_valid


def test_parse_jsonl_tile_skips_malformed_lines(cfg: Config) -> None:
    text = (
        '{"type":"Feature","geometry":{"type":"Polygon","coordinates":'
        '[[[36.8,-1.28],[36.81,-1.28],[36.81,-1.27],[36.8,-1.28]]]}}\n'
        'not json at all\n'
        '\n'
        '{"type":"Feature","geometry":{"type":"Point","coordinates":[36.8,-1.28]}}\n'
    )
    gdf = bld_mod._parse_jsonl_tile(text, cfg)
    # Only the first line is a valid Polygon; the Point and garbage are skipped.
    assert len(gdf) == 1


def test_load_buildings_for_country_streams_and_filters(
    cfg: Config, tmp_path
) -> None:
    """The country-scale loader keeps only footprints inside the buffer union."""
    from rvi.ingestion.buildings import load_buildings_for_country

    # Buffer covering the south-western corner of our two test tiles.
    buffer_poly = Polygon(
        [(36.80, -1.290), (36.83, -1.290), (36.83, -1.270), (36.80, -1.270)]
    )
    buffers_geo = gpd.GeoDataFrame(
        {"_id": [1]}, geometry=[buffer_poly], crs="EPSG:4326"
    )

    # Two tiles: one inside the buffer, one outside.
    inside = Polygon(
        [(36.81, -1.281), (36.81, -1.279), (36.812, -1.279), (36.812, -1.281)]
    )
    outside = Polygon(
        [(40.00, -3.00), (40.01, -3.00), (40.01, -2.99), (40.00, -2.99)]
    )
    inside_csv = (
        'QuadKey,Location,geometry_wkt\n' + f'qk1,Kenya,"{inside.wkt}"\n'
    )
    outside_csv = (
        'QuadKey,Location,geometry_wkt\n' + f'qk2,Kenya,"{outside.wkt}"\n'
    )
    inside_tile = gzip.compress(inside_csv.encode("utf-8"))
    outside_tile = gzip.compress(outside_csv.encode("utf-8"))

    custom_cfg = Config(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        interim_dir=tmp_path / "data" / "interim",
        processed_dir=tmp_path / "data" / "processed",
        cache_dir=tmp_path / "data" / "cache",
        outputs_dir=tmp_path / "outputs",
    )

    # Index advertises both quadkeys for Kenya.
    index_csv = (
        "Location,QuadKey,Url\n"
        "Kenya,qk1,https://example/inside.csv.gz\n"
        "Kenya,qk2,https://example/outside.csv.gz\n"
    )
    index_response = MagicMock()
    index_response.status_code = 200
    index_response.content = index_csv.encode("utf-8")
    index_response.raise_for_status.return_value = None

    inside_response = MagicMock()
    inside_response.status_code = 200
    inside_response.content = inside_tile
    inside_response.raise_for_status.return_value = None

    outside_response = MagicMock()
    outside_response.status_code = 200
    outside_response.content = outside_tile
    outside_response.raise_for_status.return_value = None

    def get_side(url, **_kwargs):
        if "dataset-links" in url or url.endswith(".csv"):
            return index_response
        if "inside" in url:
            return inside_response
        return outside_response

    session = MagicMock()
    session.get.side_effect = get_side

    out = load_buildings_for_country(
        buffers_geo,
        config=custom_cfg,
        country="Kenya",
        session=session,
        progress=False,
    )

    # Only the inside tile's footprint is retained.
    assert len(out) == 1
    assert (out["country"] == "Kenya").all()
    # Cache file was written.
    assert (custom_cfg.cache_dir / "ms_buildings_country_kenya.gpkg").exists()


def test_rows_to_geodataframe_parses_geojson_string(cfg: Config) -> None:
    df = pd.DataFrame(
        {
            "QuadKey": ["qk"],
            "Location": ["Kenya"],
            "geometry": [
                '{"type":"Polygon","coordinates":[[[36.8,-1.28],[36.81,-1.28],[36.81,-1.27],[36.8,-1.27],[36.8,-1.28]]]}'
            ],
        }
    )
    gdf = bld_mod._rows_to_geodataframe(df, cfg)
    assert not gdf.empty
    assert gdf.geometry.iloc[0].geom_type == "Polygon"
