"""End-to-end integration test for ``rvi.pipeline.run_pilot`` (offline)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import geopandas as gpd

from rvi.config import Config
from rvi.pipeline import run_pilot


def _patched_overpass(payload: dict[str, Any]):
    def _mock(*_args, **_kwargs) -> dict[str, Any]:
        return payload

    return _mock


def test_run_pilot_smoke_with_skip_floodhub_and_synthetic_buildings(
    overpass_response: dict[str, Any],
    buildings_metric: gpd.GeoDataFrame,
    tmp_path,
) -> None:
    """Pipeline runs end-to-end with mocked Overpass + injected buildings."""
    cfg = Config(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        interim_dir=tmp_path / "data" / "interim",
        processed_dir=tmp_path / "data" / "processed",
        cache_dir=tmp_path / "data" / "cache",
        outputs_dir=tmp_path / "outputs",
        bootstrap_iterations=50,
    )
    bbox = (36.80, -1.30, 36.84, -1.27)

    with patch(
        "rvi.pipeline.fetch_waterways_overpass",
        side_effect=_patched_overpass(overpass_response),
    ):
        # Need Overpass to actually return a parsed GeoDataFrame —
        # patch returns the payload, but in pipeline we expect a GeoDataFrame.
        # So patch differently: patch fetch_waterways_overpass itself
        # to return a GeoDataFrame derived from the payload.
        from rvi.ingestion.osm import parse_overpass_response

        wgdf = parse_overpass_response(overpass_response, config=cfg)

    with patch("rvi.pipeline.fetch_waterways_overpass", return_value=wgdf):
        result = run_pilot(
            bbox=bbox,
            config=cfg,
            run_name="smoke",
            skip_buildings=False,
            skip_floodhub=True,
            buildings=buildings_metric,
        )

    assert result.run_dir.exists()
    assert result.manifest_path.exists()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_name"] == "smoke"
    assert manifest["skip_floodhub"] is True
    # Waterways and segments artefacts should exist (we got 2 valid waterways).
    assert (tmp_path / "outputs" / "smoke" / "waterways.gpkg").exists()
    assert (tmp_path / "outputs" / "smoke" / "segments.gpkg").exists()
    assert (tmp_path / "outputs" / "smoke" / "rvi_segments.gpkg").exists()
    # No correlation because Flood Hub was skipped.
    assert result.correlation == {}


def test_run_pilot_with_skip_buildings_zero_rvi(
    overpass_response: dict[str, Any], tmp_path
) -> None:
    cfg = Config(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        interim_dir=tmp_path / "data" / "interim",
        processed_dir=tmp_path / "data" / "processed",
        cache_dir=tmp_path / "data" / "cache",
        outputs_dir=tmp_path / "outputs",
    )
    bbox = (36.80, -1.30, 36.84, -1.27)

    from rvi.ingestion.osm import parse_overpass_response

    wgdf = parse_overpass_response(overpass_response, config=cfg)

    with patch("rvi.pipeline.fetch_waterways_overpass", return_value=wgdf):
        result = run_pilot(
            bbox=bbox,
            config=cfg,
            run_name="zero",
            skip_buildings=True,
            skip_floodhub=True,
        )

    assert result.rvi_segments_path.exists()
    rvi_gdf = gpd.read_file(result.rvi_segments_path)
    composite_cols = [c for c in rvi_gdf.columns if c.startswith("rvi_composite_")]
    assert composite_cols, "expected at least one rvi_composite_*m column"
    for col in composite_cols:
        assert (rvi_gdf[col].fillna(0) == 0).all()
