"""End-to-end integration tests for ``rvi.pipeline`` (offline).

Both ``run_pilot`` and ``run_national`` are exercised here; network calls
(Overpass, Flood Hub, GADM, Microsoft tiles) are patched so the suite stays
hermetic.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import geopandas as gpd
import pandas as pd

from rvi.ingestion.floodhub import FloodStatus, Gauge, Severity, gauges_to_geodataframe, statuses_to_dataframe
from rvi.config import Config
from rvi.pipeline import run_national, run_pilot


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
    assert result.sensitivity_path is not None and result.sensitivity_path.exists()
    assert (
        result.sensitivity_heatmap_path is not None
        and result.sensitivity_heatmap_path.exists()
    )
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
    assert result.sensitivity_path is not None and result.sensitivity_path.exists()
    assert (
        result.sensitivity_heatmap_path is not None
        and result.sensitivity_heatmap_path.exists()
    )
    rvi_gdf = gpd.read_file(result.rvi_segments_path)
    composite_cols = [c for c in rvi_gdf.columns if c.startswith("rvi_composite_")]
    assert composite_cols, "expected at least one rvi_composite_*m column"
    for col in composite_cols:
        assert (rvi_gdf[col].fillna(0) == 0).all()


# ---------------------------------------------------------------------------
# run_national
# ---------------------------------------------------------------------------


def _synthetic_counties_geo() -> gpd.GeoDataFrame:
    """Two synthetic counties roughly covering the Nairobi-pilot bbox."""
    from shapely.geometry import box

    return gpd.GeoDataFrame(
        {
            "county_id": ["KEN.1_1", "KEN.2_1"],
            "county": ["Nairobi", "Kiambu"],
            "iso_code": ["KE.NR", "KE.KI"],
        },
        geometry=[
            box(36.65, -1.45, 36.90, -1.15),
            box(36.90, -1.45, 37.10, -1.15),
        ],
        crs="EPSG:4326",
    )


def _synthetic_gauges_and_statuses() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    gauges_geo = gauges_to_geodataframe(
        [
            Gauge(
                gauge_id="g1",
                latitude=-1.28,
                longitude=36.83,
                quality_verified=True,
            ),
            Gauge(
                gauge_id="g2",
                latitude=-0.50,
                longitude=37.00,
                quality_verified=False,
            ),
        ]
    )
    statuses_df = statuses_to_dataframe(
        [
            FloodStatus(gauge_id="g1", severity=Severity.SEVERE),
            FloodStatus(gauge_id="g2", severity=Severity.NO_FLOODING),
        ]
    )
    return gauges_geo, statuses_df


def _synthetic_catchments_geo() -> gpd.GeoDataFrame:
    from shapely.geometry import box

    return gpd.GeoDataFrame(
        {
            "gauge_id": ["g1", "g2"],
            "geometry": [
                box(36.79, -1.31, 36.85, -1.26),
                box(37.30, -0.60, 37.50, -0.40),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def test_run_national_smoke_with_injected_inputs(
    waterways_metric: gpd.GeoDataFrame,
    buildings_metric: gpd.GeoDataFrame,
    tmp_path,
) -> None:
    """run_national must work end-to-end when waterways/buildings/counties are
    injected (no PBF, no Microsoft download, no Flood Hub network)."""
    cfg = Config(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        interim_dir=tmp_path / "data" / "interim",
        processed_dir=tmp_path / "data" / "processed",
        cache_dir=tmp_path / "data" / "cache",
        outputs_dir=tmp_path / "outputs",
        bootstrap_iterations=10,
    )

    waterways_geo = waterways_metric.to_crs("EPSG:4326")
    buildings_geo = buildings_metric.to_crs("EPSG:4326")
    counties_geo = _synthetic_counties_geo()

    with patch(
        "rvi.pipeline.download_kenya_counties", return_value=counties_geo
    ):
        result = run_national(
            config=cfg,
            run_name="nat_smoke",
            waterways=waterways_geo,
            buildings=buildings_geo,
            skip_floodhub=True,
        )

    assert result.run_dir.exists()
    assert result.manifest_path.exists()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["phase"] == "1-national"
    assert manifest["waterways_count"] == len(waterways_geo)
    assert manifest["segments_count"] > 0
    assert manifest["buildings_count"] == len(buildings_geo)
    assert manifest["correlation"] == {}

    # Core artefacts exist.
    assert result.waterways_path.exists()
    assert result.segments_path.exists()
    assert result.rvi_segments_path.exists()
    assert result.sensitivity_path is not None and result.sensitivity_path.exists()
    assert (
        result.sensitivity_heatmap_path is not None
        and result.sensitivity_heatmap_path.exists()
    )
    # County choropleth + counties GPKG written.
    assert result.choropleth_path is not None and result.choropleth_path.exists()
    assert result.counties_path is not None and result.counties_path.exists()


def test_run_national_skip_counties(
    waterways_metric: gpd.GeoDataFrame,
    buildings_metric: gpd.GeoDataFrame,
    tmp_path,
) -> None:
    """--skip-counties suppresses GADM download and choropleth output."""
    cfg = Config(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        interim_dir=tmp_path / "data" / "interim",
        processed_dir=tmp_path / "data" / "processed",
        cache_dir=tmp_path / "data" / "cache",
        outputs_dir=tmp_path / "outputs",
    )

    with patch("rvi.pipeline.download_kenya_counties") as gadm_mock:
        result = run_national(
            config=cfg,
            run_name="nat_no_counties",
            waterways=waterways_metric.to_crs("EPSG:4326"),
            buildings=buildings_metric.to_crs("EPSG:4326"),
            skip_floodhub=True,
            skip_counties=True,
        )

    gadm_mock.assert_not_called()
    assert result.choropleth_path is None
    assert result.counties_path is None
    assert result.rvi_segments_path.exists()
    assert result.sensitivity_path is not None and result.sensitivity_path.exists()
    assert (
        result.sensitivity_heatmap_path is not None
        and result.sensitivity_heatmap_path.exists()
    )


def test_run_national_skip_buildings_yields_zero_rvi(
    waterways_metric: gpd.GeoDataFrame, tmp_path
) -> None:
    cfg = Config(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        interim_dir=tmp_path / "data" / "interim",
        processed_dir=tmp_path / "data" / "processed",
        cache_dir=tmp_path / "data" / "cache",
        outputs_dir=tmp_path / "outputs",
    )

    result = run_national(
        config=cfg,
        run_name="nat_no_bld",
        waterways=waterways_metric.to_crs("EPSG:4326"),
        skip_buildings=True,
        skip_floodhub=True,
        skip_counties=True,
    )
    assert result.sensitivity_path is not None and result.sensitivity_path.exists()
    assert (
        result.sensitivity_heatmap_path is not None
        and result.sensitivity_heatmap_path.exists()
    )
    rvi_gdf = gpd.read_file(result.rvi_segments_path)
    composite_cols = [c for c in rvi_gdf.columns if c.startswith("rvi_composite_")]
    for col in composite_cols:
        assert (rvi_gdf[col].fillna(0) == 0).all()


def test_run_national_prefers_duckdb_buildings_loader(
    waterways_metric: gpd.GeoDataFrame,
    buildings_metric: gpd.GeoDataFrame,
    tmp_path,
) -> None:
    cfg = Config(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        interim_dir=tmp_path / "data" / "interim",
        processed_dir=tmp_path / "data" / "processed",
        cache_dir=tmp_path / "data" / "cache",
        outputs_dir=tmp_path / "outputs",
    )

    with (
        patch("rvi.ingestion.buildings.stream_buildings_duckdb", return_value=buildings_metric.to_crs("EPSG:4326")) as duckdb_mock,
        patch("rvi.ingestion.buildings.load_buildings_for_country") as fallback_mock,
    ):
        result = run_national(
            config=cfg,
            run_name="nat_duckdb",
            waterways=waterways_metric.to_crs("EPSG:4326"),
            skip_floodhub=True,
            skip_counties=True,
        )

    duckdb_mock.assert_called_once()
    fallback_mock.assert_not_called()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["buildings_loader"] == "duckdb"


def test_run_national_falls_back_when_duckdb_loader_fails(
    waterways_metric: gpd.GeoDataFrame,
    buildings_metric: gpd.GeoDataFrame,
    tmp_path,
) -> None:
    cfg = Config(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        interim_dir=tmp_path / "data" / "interim",
        processed_dir=tmp_path / "data" / "processed",
        cache_dir=tmp_path / "data" / "cache",
        outputs_dir=tmp_path / "outputs",
    )

    with (
        patch(
            "rvi.ingestion.buildings.stream_buildings_duckdb",
            side_effect=RuntimeError("spatial extension unavailable"),
        ) as duckdb_mock,
        patch(
            "rvi.ingestion.buildings.load_buildings_for_country",
            return_value=buildings_metric.to_crs("EPSG:4326"),
        ) as fallback_mock,
    ):
        result = run_national(
            config=cfg,
            run_name="nat_duckdb_fallback",
            waterways=waterways_metric.to_crs("EPSG:4326"),
            skip_floodhub=True,
            skip_counties=True,
        )

    duckdb_mock.assert_called_once()
    fallback_mock.assert_called_once()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["buildings_loader"] == "geopandas_fallback"


def test_run_national_with_bbox_uses_bbox_buildings_loader(
    waterways_metric: gpd.GeoDataFrame,
    buildings_metric: gpd.GeoDataFrame,
    tmp_path,
) -> None:
    cfg = Config(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        interim_dir=tmp_path / "data" / "interim",
        processed_dir=tmp_path / "data" / "processed",
        cache_dir=tmp_path / "data" / "cache",
        outputs_dir=tmp_path / "outputs",
    )

    bbox = (36.65, -1.45, 37.10, -1.15)
    with (
        patch(
            "rvi.ingestion.buildings.load_buildings_for_bbox",
            return_value=buildings_metric.to_crs("EPSG:4326"),
        ) as bbox_mock,
        patch("rvi.ingestion.buildings.stream_buildings_duckdb") as duckdb_mock,
        patch("rvi.ingestion.buildings.load_buildings_for_country") as fallback_mock,
    ):
        result = run_national(
            config=cfg,
            run_name="nat_bbox_loader",
            bbox=bbox,
            waterways=waterways_metric.to_crs("EPSG:4326"),
            skip_floodhub=True,
            skip_counties=True,
        )

    bbox_mock.assert_called_once()
    duckdb_mock.assert_not_called()
    fallback_mock.assert_not_called()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["buildings_loader"] == "bbox"


def test_run_national_prefers_country_cache_when_present(
    waterways_metric: gpd.GeoDataFrame,
    buildings_metric: gpd.GeoDataFrame,
    tmp_path,
) -> None:
    cfg = Config(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        interim_dir=tmp_path / "data" / "interim",
        processed_dir=tmp_path / "data" / "processed",
        cache_dir=tmp_path / "data" / "cache",
        outputs_dir=tmp_path / "outputs",
    )
    cfg.ensure_dirs()
    (cfg.cache_dir / "ms_buildings_country_kenya.gpkg").write_text("cached")

    with (
        patch(
            "rvi.ingestion.buildings.load_buildings_for_country",
            return_value=buildings_metric.to_crs("EPSG:4326"),
        ) as country_mock,
        patch("rvi.ingestion.buildings.stream_buildings_duckdb") as duckdb_mock,
    ):
        result = run_national(
            config=cfg,
            run_name="nat_country_cache",
            waterways=waterways_metric.to_crs("EPSG:4326"),
            skip_floodhub=True,
            skip_counties=True,
        )

    country_mock.assert_called_once()
    duckdb_mock.assert_not_called()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["buildings_loader"] == "country_cache"


def test_run_pilot_uses_catchment_validation_when_supplied(
    overpass_response: dict[str, Any],
    buildings_metric: gpd.GeoDataFrame,
    tmp_path,
) -> None:
    cfg = Config(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        interim_dir=tmp_path / "data" / "interim",
        processed_dir=tmp_path / "data" / "processed",
        cache_dir=tmp_path / "data" / "cache",
        outputs_dir=tmp_path / "outputs",
        bootstrap_iterations=10,
    )
    from rvi.ingestion.osm import parse_overpass_response

    wgdf = parse_overpass_response(overpass_response, config=cfg)
    gauges_geo, statuses_df = _synthetic_gauges_and_statuses()
    catchments_geo = _synthetic_catchments_geo()

    mock_client = MagicMock()
    mock_client.search_gauges_by_area.return_value = [
        Gauge(gauge_id="g1", latitude=-1.28, longitude=36.83, quality_verified=True),
        Gauge(gauge_id="g2", latitude=-0.50, longitude=37.00, quality_verified=False),
    ]
    mock_client.search_latest_flood_status_by_area.return_value = [
        FloodStatus(gauge_id="g1", severity=Severity.SEVERE),
        FloodStatus(gauge_id="g2", severity=Severity.NO_FLOODING),
    ]

    with (
        patch("rvi.pipeline.fetch_waterways_overpass", return_value=wgdf),
        patch("rvi.pipeline.FloodHubClient", return_value=mock_client),
    ):
        result = run_pilot(
            config=cfg,
            run_name="catchment_smoke",
            buildings=buildings_metric,
            catchments=catchments_geo,
        )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["validation_mode"] == "catchment"
    assert result.correlation
    upstream_30 = result.upstream_paths[30.0]
    upstream = pd.read_csv(upstream_30)
    assert int(upstream.loc[upstream["gauge_id"] == "g1", "n_upstream_segments"].iloc[0]) > 0


def test_run_national_uses_catchment_validation_when_supplied(
    waterways_metric: gpd.GeoDataFrame,
    buildings_metric: gpd.GeoDataFrame,
    tmp_path,
) -> None:
    cfg = Config(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        interim_dir=tmp_path / "data" / "interim",
        processed_dir=tmp_path / "data" / "processed",
        cache_dir=tmp_path / "data" / "cache",
        outputs_dir=tmp_path / "outputs",
        bootstrap_iterations=10,
    )
    counties_geo = _synthetic_counties_geo()
    catchments_geo = _synthetic_catchments_geo()

    mock_client = MagicMock()
    mock_client.search_gauges_by_area.return_value = [
        Gauge(gauge_id="g1", latitude=-1.28, longitude=36.83, quality_verified=True),
        Gauge(gauge_id="g2", latitude=-0.50, longitude=37.00, quality_verified=False),
    ]
    mock_client.search_latest_flood_status_by_area.return_value = [
        FloodStatus(gauge_id="g1", severity=Severity.SEVERE),
        FloodStatus(gauge_id="g2", severity=Severity.NO_FLOODING),
    ]

    with (
        patch("rvi.pipeline.download_kenya_counties", return_value=counties_geo),
        patch("rvi.pipeline.FloodHubClient", return_value=mock_client),
    ):
        result = run_national(
            config=cfg,
            run_name="nat_catchment_smoke",
            waterways=waterways_metric.to_crs("EPSG:4326"),
            buildings=buildings_metric.to_crs("EPSG:4326"),
            catchments=catchments_geo,
        )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["validation_mode"] == "catchment"
    assert result.correlation


def test_run_pilot_uses_dem_to_generate_catchments(
    overpass_response: dict[str, Any],
    buildings_metric: gpd.GeoDataFrame,
    tmp_path,
) -> None:
    cfg = Config(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        interim_dir=tmp_path / "data" / "interim",
        processed_dir=tmp_path / "data" / "processed",
        cache_dir=tmp_path / "data" / "cache",
        outputs_dir=tmp_path / "outputs",
        bootstrap_iterations=10,
    )
    from rvi.ingestion.osm import parse_overpass_response

    wgdf = parse_overpass_response(overpass_response, config=cfg)
    catchments_geo = _synthetic_catchments_geo()

    mock_client = MagicMock()
    mock_client.search_gauges_by_area.return_value = [
        Gauge(gauge_id="g1", latitude=-1.28, longitude=36.83, quality_verified=True),
        Gauge(gauge_id="g2", latitude=-0.50, longitude=37.00, quality_verified=False),
    ]
    mock_client.search_latest_flood_status_by_area.return_value = [
        FloodStatus(gauge_id="g1", severity=Severity.SEVERE),
        FloodStatus(gauge_id="g2", severity=Severity.NO_FLOODING),
    ]

    with (
        patch("rvi.pipeline.fetch_waterways_overpass", return_value=wgdf),
        patch("rvi.pipeline.FloodHubClient", return_value=mock_client),
        patch("rvi.pipeline.delineate_catchments_from_dem", return_value=catchments_geo) as dem_mock,
    ):
        result = run_pilot(
            config=cfg,
            run_name="dem_catchment_smoke",
            buildings=buildings_metric,
            dem_path=tmp_path / "fake_dem.tif",
        )

    dem_mock.assert_called_once()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["validation_mode"] == "dem_catchment"
    assert result.correlation
