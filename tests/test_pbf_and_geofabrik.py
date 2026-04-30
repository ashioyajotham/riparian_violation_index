"""Tests for the osmium-backed PBF reader and the Geofabrik downloader."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rvi.config import Config
from rvi.ingestion.osm import (
    download_geofabrik_kenya_pbf,
    fetch_waterways_pbf,
)


def _build_synthetic_pbf(target_dir):
    """Build a tiny .osm.pbf containing one river and one stream."""
    osmium = pytest.importorskip("osmium")
    path = target_dir / "synthetic.osm.pbf"

    writer = osmium.SimpleWriter(str(path))
    try:
        # Two nodes for the river.
        writer.add_node(osmium.osm.mutable.Node(
            id=1, location=(36.80, -1.28), tags={}
        ))
        writer.add_node(osmium.osm.mutable.Node(
            id=2, location=(36.84, -1.28), tags={}
        ))
        # Two more for the stream.
        writer.add_node(osmium.osm.mutable.Node(
            id=3, location=(36.82, -1.30), tags={}
        ))
        writer.add_node(osmium.osm.mutable.Node(
            id=4, location=(36.82, -1.27), tags={}
        ))
        # River way.
        writer.add_way(osmium.osm.mutable.Way(
            id=11111,
            nodes=[1, 2],
            tags={"waterway": "river", "name": "Test River", "name:sw": "Mto"},
        ))
        # Stream way.
        writer.add_way(osmium.osm.mutable.Way(
            id=22222,
            nodes=[3, 4],
            tags={"waterway": "stream", "name": "Test Stream"},
        ))
        # An infrastructure tag we want filtered out.
        writer.add_node(osmium.osm.mutable.Node(
            id=5, location=(36.81, -1.28), tags={}
        ))
        writer.add_node(osmium.osm.mutable.Node(
            id=6, location=(36.815, -1.28), tags={}
        ))
        writer.add_way(osmium.osm.mutable.Way(
            id=33333,
            nodes=[5, 6],
            tags={"waterway": "weir", "name": "Some Weir"},
        ))
    finally:
        writer.close()
    return path


@pytest.fixture()
def synthetic_pbf(tmp_path):
    pytest.importorskip("osmium")
    return _build_synthetic_pbf(tmp_path)


def test_fetch_waterways_pbf_reads_synthetic_file(
    synthetic_pbf, cfg: Config
) -> None:
    gdf = fetch_waterways_pbf(synthetic_pbf, config=cfg)
    assert not gdf.empty
    # The "weir" feature is filtered out.
    assert set(gdf["waterway"]) == {"river", "stream"}
    # Both retained features have positive length.
    assert (gdf["length_m"] > 0).all()
    # The Swahili name is preserved.
    river = gdf[gdf["osm_id"] == "11111"].iloc[0]
    assert river["name_local"] == "Mto"
    # Strahler ordering applied.
    stream = gdf[gdf["osm_id"] == "22222"].iloc[0]
    assert int(river["strahler"]) == cfg.strahler_for_waterway("river")
    assert int(stream["strahler"]) == cfg.strahler_for_waterway("stream")


def test_fetch_waterways_pbf_bbox_filters_post_parse(
    synthetic_pbf, cfg: Config
) -> None:
    # Bbox far from the synthetic features → empty result.
    gdf = fetch_waterways_pbf(
        synthetic_pbf, config=cfg, bbox=(50.0, -5.0, 51.0, -4.0)
    )
    assert gdf.empty
    # Bbox covering the features → both retained.
    gdf = fetch_waterways_pbf(
        synthetic_pbf, config=cfg, bbox=(36.79, -1.31, 36.85, -1.26)
    )
    assert len(gdf) == 2


def test_fetch_waterways_pbf_missing_file_raises(cfg: Config, tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        fetch_waterways_pbf(tmp_path / "does_not_exist.pbf", config=cfg)


def test_download_geofabrik_kenya_pbf_streams_and_caches(
    cfg: Config, tmp_path
) -> None:
    fake_body = b"binary-pbf-bytes" * 1024  # 16 KB of fake data
    response = MagicMock()
    response.status_code = 200
    response.headers = {"Content-Length": str(len(fake_body))}
    response.iter_content = lambda chunk_size: iter(
        [fake_body[i : i + chunk_size] for i in range(0, len(fake_body), chunk_size)]
    )
    response.raise_for_status.return_value = None
    response.__enter__ = lambda self: self
    response.__exit__ = lambda self, *a: False

    session = MagicMock()
    session.get.return_value = response

    target = tmp_path / "kenya-latest.osm.pbf"
    out = download_geofabrik_kenya_pbf(
        config=cfg, target_path=target, session=session, progress=False
    )
    assert out == target
    assert target.exists()
    assert target.read_bytes() == fake_body
    # No partial file left behind.
    assert not (tmp_path / "kenya-latest.osm.pbf.part").exists()

    # Second call hits the cache, no GET issued.
    session.get.reset_mock()
    download_geofabrik_kenya_pbf(
        config=cfg, target_path=target, session=session, progress=False
    )
    session.get.assert_not_called()
