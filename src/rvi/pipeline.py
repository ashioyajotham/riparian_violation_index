"""High-level pipeline orchestration (§5.1 of the proposal).

This module composes the per-stage implementations into a single
``run_pilot()`` function so the CLI and notebooks can both call into the
same code path. Outputs land under ``cfg.outputs_dir / run_name``:

::

    outputs/<run_name>/
    ├── manifest.json                          # parameters used for the run
    ├── waterways.gpkg
    ├── segments.gpkg
    ├── buildings.gpkg          (optional; large)
    ├── encroachment_<width>m.gpkg
    ├── rvi_segments.gpkg       (per-segment, all widths joined)
    ├── upstream_<width>m.csv   (per-gauge upstream stats)
    ├── correlation.json        (Spearman ρ, CI, p-value)
    ├── rvi_segment_map.html    (Folium leaflet)
    └── rvi_severity_scatter.png
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from rvi.analysis.encroachment import detect_encroachment_multi
from rvi.analysis.rvi import compute_rvi, compute_rvi_multi
from rvi.analysis.validation import (
    SpearmanResult,
    aggregate_upstream_euclidean,
    correlate_upstream_rvi_to_severity,
    stratified_correlation,
)
from rvi.config import Config, get_config
from rvi.geometry.segment import segment_waterways
from rvi.ingestion.floodhub import (
    FloodHubClient,
    FloodHubError,
    gauges_to_geodataframe,
    join_gauges_with_status,
    statuses_to_dataframe,
)
from rvi.ingestion.osm import fetch_waterways_overpass
from rvi.io import write_geopackage
from rvi.viz.choropleth import rvi_severity_scatter, segment_map

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public result container
# ---------------------------------------------------------------------------


@dataclass
class PilotResult:
    """Bundle of artefacts produced by :func:`run_pilot`."""

    run_dir: Path
    waterways_path: Path
    segments_path: Path
    rvi_segments_path: Path
    map_path: Path | None
    correlation: dict[str, SpearmanResult]
    upstream_paths: dict[float, Path]
    manifest_path: Path

    def correlation_summary(self) -> dict[str, dict[str, float | int | str]]:
        return {k: v.as_dict() for k, v in self.correlation.items()}


# ---------------------------------------------------------------------------
# Pilot driver
# ---------------------------------------------------------------------------


def run_pilot(
    bbox: tuple[float, float, float, float] | None = None,
    *,
    config: Config | None = None,
    run_name: str = "nairobi_pilot",
    skip_buildings: bool = False,
    skip_floodhub: bool = False,
    buildings: gpd.GeoDataFrame | None = None,
) -> PilotResult:
    """Run the full pilot pipeline end-to-end.

    Parameters
    ----------
    bbox
        Pilot bounding box (defaults to ``cfg.nairobi_bbox``).
    skip_buildings
        Convenience for smoke-testing without the Microsoft download. The
        pipeline still produces RVI scores (all zero) so the validation
        stage's plumbing exercises end-to-end.
    buildings
        Inject a pre-loaded buildings GeoDataFrame (used by tests and notebooks
        that already have one). Bypasses the Microsoft download.
    skip_floodhub
        Skip the Flood Hub network calls (useful when the pilot key has not
        been issued yet). The correlation step is then skipped.
    """
    cfg = config or get_config()
    cfg.ensure_dirs()
    bbox = bbox or cfg.nairobi_bbox
    run_dir = cfg.outputs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "run_name": run_name,
        "bbox": list(bbox),
        "buffer_widths_m": list(cfg.buffer_widths_m),
        "segment_length_m": cfg.segment_length_m,
        "rvi_weights": {"alpha": cfg.rvi_alpha, "beta": cfg.rvi_beta, "gamma": cfg.rvi_gamma},
        "upstream_radius_m": cfg.upstream_radius_m,
        "skip_buildings": skip_buildings,
        "skip_floodhub": skip_floodhub,
    }

    # 1) Waterways ----------------------------------------------------------
    logger.info("Stage 1/5: fetching waterways from Overpass")
    waterways = fetch_waterways_overpass(
        bbox,
        config=cfg,
        cache_path=cfg.cache_dir / f"overpass_{run_name}.json",
    )
    waterways_path = run_dir / "waterways.gpkg"
    if not waterways.empty:
        write_geopackage(waterways, waterways_path)
    manifest["waterways_count"] = len(waterways)

    # 2) Segments -----------------------------------------------------------
    logger.info("Stage 2/5: segmenting waterways")
    segments = segment_waterways(waterways, config=cfg)
    segments_path = run_dir / "segments.gpkg"
    if not segments.empty:
        write_geopackage(segments, segments_path)
    manifest["segments_count"] = len(segments)

    # 3) Buildings ----------------------------------------------------------
    bld: gpd.GeoDataFrame | None = buildings
    if bld is None and not skip_buildings:
        logger.info("Stage 3/5: loading Microsoft footprints (this is the slow step)")
        # Heavy import deferred so smoke tests don't pay the cost.
        from rvi.ingestion.buildings import load_buildings_for_bbox

        bld = load_buildings_for_bbox(bbox, config=cfg, progress=False)
    if bld is None:
        logger.warning("Skipping buildings download — RVI will be all zeros.")
        # Synthesise an empty buildings GeoDataFrame.
        bld = gpd.GeoDataFrame(
            {
                "building_id": pd.Series(dtype="int64"),
                "country": pd.Series(dtype=str),
                "quadkey": pd.Series(dtype=str),
                "footprint_area_m2": pd.Series(dtype=float),
            },
            geometry=gpd.GeoSeries([], crs=cfg.crs_geographic),
            crs=cfg.crs_geographic,
        )
    manifest["buildings_count"] = len(bld)

    # 4) Encroachment + RVI -------------------------------------------------
    logger.info("Stage 4/5: computing encroachment and RVI for all buffer widths")
    enc_by_width = detect_encroachment_multi(segments, bld, config=cfg)
    for width, enc in enc_by_width.items():
        path = run_dir / f"encroachment_{int(width)}m.gpkg"
        if not enc.empty:
            write_geopackage(enc, path)
    rvi_segments = compute_rvi_multi(enc_by_width, config=cfg)
    rvi_path = run_dir / "rvi_segments.gpkg"
    if not rvi_segments.empty:
        write_geopackage(rvi_segments, rvi_path)

    # 5) Validation ---------------------------------------------------------
    logger.info("Stage 5/5: validation oracle")
    correlation: dict[str, SpearmanResult] = {}
    upstream_paths: dict[float, Path] = {}

    gauges_geo: gpd.GeoDataFrame | None = None
    statuses_df: pd.DataFrame | None = None

    if not skip_floodhub:
        try:
            client = FloodHubClient(config=cfg)
            gauges = client.search_gauges_by_area(region_code="KE")
            statuses = client.search_latest_flood_status_by_area(region_code="KE")
            gauges_geo = gauges_to_geodataframe(gauges)
            statuses_df = statuses_to_dataframe(statuses)
            (run_dir / "gauges.gpkg").parent.mkdir(parents=True, exist_ok=True)
            if not gauges_geo.empty:
                write_geopackage(gauges_geo, run_dir / "gauges.gpkg")
                statuses_df.to_csv(run_dir / "gauge_statuses.csv", index=False)
        except FloodHubError as exc:
            logger.warning("Flood Hub unavailable (%s); skipping validation.", exc)

    if gauges_geo is not None and not gauges_geo.empty and statuses_df is not None:
        gauges_with_status = join_gauges_with_status(gauges_geo, statuses_df)
        for width in cfg.buffer_widths_m:
            scored = compute_rvi(enc_by_width[float(width)], config=cfg)
            upstream = aggregate_upstream_euclidean(
                scored, gauges_with_status, config=cfg
            )
            up_path = run_dir / f"upstream_{int(width)}m.csv"
            upstream.to_csv(up_path, index=False)
            upstream_paths[float(width)] = up_path
            res = correlate_upstream_rvi_to_severity(upstream, config=cfg)
            correlation[f"{int(width)}m_p75"] = res
            stratified = stratified_correlation(upstream, config=cfg)
            for tier_name, tier_res in stratified.items():
                correlation[f"{int(width)}m_p75_{tier_name}"] = tier_res

    # Map + scatter ---------------------------------------------------------
    map_path: Path | None = None
    if not rvi_segments.empty:
        col = (
            f"rvi_composite_{int(cfg.primary_buffer_m)}m"
            if f"rvi_composite_{int(cfg.primary_buffer_m)}m" in rvi_segments.columns
            else "rvi_composite_30m"
        )
        if col in rvi_segments.columns:
            fmap = segment_map(
                rvi_segments,
                config=cfg,
                rvi_column=col,
                title=f"{run_name} — RVI segment map",
            )
            map_path = run_dir / "rvi_segment_map.html"
            fmap.save(str(map_path))

    scatter_path = run_dir / "rvi_severity_scatter.png"
    primary = correlation.get(f"{int(cfg.primary_buffer_m)}m_p75")
    primary_upstream_path = upstream_paths.get(float(cfg.primary_buffer_m))
    if primary is not None and primary_upstream_path is not None:
        upstream_df = pd.read_csv(primary_upstream_path)
        rvi_severity_scatter(
            upstream_df,
            result=primary,
            title=f"{run_name} — Spearman ρ = {primary.rho:+.3f}",
            save_path=scatter_path,
        )

    # Manifest --------------------------------------------------------------
    manifest["correlation"] = {k: v.as_dict() for k, v in correlation.items()}
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, default=str, indent=2),
        encoding="utf-8",
    )

    return PilotResult(
        run_dir=run_dir,
        waterways_path=waterways_path,
        segments_path=segments_path,
        rvi_segments_path=rvi_path,
        map_path=map_path,
        correlation=correlation,
        upstream_paths=upstream_paths,
        manifest_path=manifest_path,
    )


__all__ = ["PilotResult", "run_pilot"]
