"""Tests for ``rvi.ingestion.osm`` (offline only).

The Overpass HTTP path is exercised via a mocked ``requests.Session``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from shapely.geometry import LineString

from rvi.config import Config
from rvi.ingestion.osm import (
    build_overpass_query,
    fetch_waterways_overpass,
    filter_waterway_types,
    load_waterways,
    parse_overpass_response,
    save_waterways,
)

# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------


def test_build_overpass_query_has_required_clauses() -> None:
    q = build_overpass_query(
        (36.80, -1.30, 36.84, -1.27), ("river", "stream", "canal", "drain", "ditch")
    )
    # bbox order is south,west,north,east per Overpass conventions.
    assert "(-1.3,36.8,-1.27,36.84)" in q
    assert '"waterway"' in q
    assert "river|stream|canal|drain|ditch" in q
    assert "out geom" in q
    assert q.startswith("[out:json]")


def test_build_overpass_query_rejects_bad_bbox() -> None:
    with pytest.raises(ValueError, match="ordering"):
        build_overpass_query((36.84, -1.30, 36.80, -1.27), ("river",))
    with pytest.raises(ValueError, match="non-empty"):
        build_overpass_query((36.80, -1.30, 36.84, -1.27), ())
    with pytest.raises(ValueError):
        build_overpass_query((1, 2, 3), ("river",))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def test_parse_overpass_response_filters_to_known_waterway_types(
    overpass_response: dict[str, Any], cfg: Config
) -> None:
    gdf = parse_overpass_response(overpass_response, config=cfg)
    # The "weir" element must be dropped.
    assert set(gdf["waterway"]) == {"river", "stream"}
    assert (gdf["osm_id"] == "33333").sum() == 0


def test_parse_overpass_response_assigns_strahler(
    overpass_response: dict[str, Any], cfg: Config
) -> None:
    gdf = parse_overpass_response(overpass_response, config=cfg)
    river = gdf[gdf["waterway"] == "river"].iloc[0]
    stream = gdf[gdf["waterway"] == "stream"].iloc[0]
    assert int(river["strahler"]) == cfg.strahler_for_waterway("river")
    assert int(stream["strahler"]) == cfg.strahler_for_waterway("stream")


def test_parse_overpass_response_preserves_swahili_name(
    overpass_response: dict[str, Any], cfg: Config
) -> None:
    gdf = parse_overpass_response(overpass_response, config=cfg)
    river = gdf[gdf["osm_id"] == "11111"].iloc[0]
    # name:sw is preserved as name_local.
    assert river["name_local"] == "Nairobi"


def test_parse_overpass_response_lengths_are_metric_positive(
    overpass_response: dict[str, Any], cfg: Config
) -> None:
    gdf = parse_overpass_response(overpass_response, config=cfg)
    # All retained features are LineStrings with positive length.
    assert (gdf["length_m"] > 0).all()
    assert all(isinstance(g, LineString) for g in gdf.geometry)


def test_parse_overpass_empty_payload_returns_empty_gdf(cfg: Config) -> None:
    gdf = parse_overpass_response({"elements": []}, config=cfg)
    assert gdf.empty
    assert "geometry" in gdf.columns
    assert str(gdf.crs) == "EPSG:4326"


# ---------------------------------------------------------------------------
# fetch_waterways_overpass with a mocked session
# ---------------------------------------------------------------------------


def test_fetch_waterways_overpass_uses_mocked_session(
    overpass_response: dict[str, Any], cfg: Config, tmp_path
) -> None:
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = overpass_response
    response.raise_for_status.return_value = None
    session.post.return_value = response

    cache_path = tmp_path / "overpass_cache.json"
    gdf = fetch_waterways_overpass(
        bbox=(36.80, -1.30, 36.84, -1.27),
        config=cfg,
        cache_path=cache_path,
        session=session,
    )
    assert not gdf.empty
    assert cache_path.exists()
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cached["elements"][0]["id"] == 11111

    # Second call uses the cache and should NOT post again.
    session.post.reset_mock()
    gdf2 = fetch_waterways_overpass(
        bbox=(36.80, -1.30, 36.84, -1.27),
        config=cfg,
        cache_path=cache_path,
        session=session,
    )
    assert not gdf2.empty
    session.post.assert_not_called()


def test_fetch_waterways_overpass_retries_on_429(cfg: Config) -> None:
    session = MagicMock()
    bad = MagicMock()
    bad.status_code = 429
    bad.text = "rate limited"
    bad.raise_for_status.side_effect = AssertionError("should not be called")
    good = MagicMock()
    good.status_code = 200
    good.json.return_value = {"elements": []}
    good.raise_for_status.return_value = None
    session.post.side_effect = [bad, good]

    gdf = fetch_waterways_overpass(
        bbox=(36.80, -1.30, 36.84, -1.27),
        config=cfg,
        session=session,
        max_retries=2,
        retry_backoff_s=0.0,
    )
    assert gdf.empty
    assert session.post.call_count == 2


# ---------------------------------------------------------------------------
# I/O round-trip
# ---------------------------------------------------------------------------


def test_save_and_load_waterways_roundtrip(
    overpass_response: dict[str, Any], cfg: Config, tmp_path
) -> None:
    gdf = parse_overpass_response(overpass_response, config=cfg)
    path = tmp_path / "waterways.gpkg"
    save_waterways(gdf, path)
    loaded = load_waterways(path)
    assert len(loaded) == len(gdf)
    assert set(loaded["waterway"]) == set(gdf["waterway"])


def test_filter_waterway_types_keeps_only_listed(
    overpass_response: dict[str, Any], cfg: Config
) -> None:
    gdf = parse_overpass_response(overpass_response, config=cfg)
    only_rivers = filter_waterway_types(gdf, {"river"})
    assert set(only_rivers["waterway"]) == {"river"}
