"""Administrative boundaries — Kenya counties (GADM level 1).

The proposal's national run produces a county-level RVI choropleth (§5.6).
GADM (https://gadm.org) ships free, well-maintained country-by-country
administrative boundaries. For Kenya, level 1 is *County* (47 counties under
the 2010 Constitution).

The download is small (~1 MB GeoJSON), licence-free for non-commercial use,
and reasonably stable — we cache it once under ``cfg.cache_dir`` and reuse.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from rvi.config import Config, get_config

logger = logging.getLogger(__name__)


# GADM 4.1 Kenya, level-1 (county) boundaries as GeoJSON.
DEFAULT_KENYA_COUNTIES_URL = (
    "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_KEN_1.json"
)


COUNTY_COLUMNS: tuple[str, ...] = (
    "county_id",
    "county",
    "iso_code",
    "geometry",
)


def download_kenya_counties(
    *,
    config: Config | None = None,
    target_path: Path | None = None,
    overwrite: bool = False,
    session: requests.Session | None = None,
    url: str | None = None,
) -> gpd.GeoDataFrame:
    """Download (and cache) the GADM Kenya level-1 county boundaries.

    Returns a GeoDataFrame in the canonical schema:

    ============== ======================================================
    column         meaning
    -------------- ------------------------------------------------------
    ``county_id``  GADM stable id (e.g. ``KEN.1_1``)
    ``county``     human-readable county name (e.g. ``Nairobi``, ``Kiambu``)
    ``iso_code``   ISO HASC code if present, else ``None``
    ``geometry``   MultiPolygon in EPSG:4326
    ============== ======================================================

    Implementation notes
    --------------------
    GADM 4.1 GeoJSON uses ``GID_1`` for the county id and ``NAME_1`` for
    the human-readable name. We rename to the project's canonical schema.
    """
    cfg = config or get_config()
    cfg.ensure_dirs()
    target_path = (
        Path(target_path) if target_path else cfg.cache_dir / "kenya_counties.geojson"
    )

    if not target_path.exists() or overwrite:
        sess = session or requests.Session()
        sess.headers.setdefault("User-Agent", cfg.user_agent)
        download_url = url or DEFAULT_KENYA_COUNTIES_URL
        logger.info("Downloading Kenya counties: %s", download_url)
        resp = sess.get(download_url, timeout=cfg.request_timeout_s)
        resp.raise_for_status()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(resp.content)

    gdf = gpd.read_file(target_path)
    if gdf.empty:
        return _empty_counties_gdf()

    return _normalise_counties(gdf, cfg)


def load_kenya_counties(
    path: Path,
    *,
    config: Config | None = None,
) -> gpd.GeoDataFrame:
    """Load county boundaries from a local file (any GeoPandas-readable format)."""
    cfg = config or get_config()
    gdf = gpd.read_file(path)
    return _normalise_counties(gdf, cfg)


def _normalise_counties(gdf: gpd.GeoDataFrame, cfg: Config) -> gpd.GeoDataFrame:
    """Coerce a GADM-style frame into the canonical RVI schema."""
    df = gdf.copy()

    # GADM 4.1 columns: GID_1 (id), NAME_1 (county), HASC_1 (ISO sub-code).
    name_col = next(
        (c for c in ("NAME_1", "name_1", "ADM1_NAME", "county", "name") if c in df.columns),
        None,
    )
    id_col = next(
        (c for c in ("GID_1", "gid_1", "ADM1_GID", "county_id") if c in df.columns),
        None,
    )
    iso_col = next(
        (c for c in ("HASC_1", "hasc_1", "iso_code", "ISO") if c in df.columns), None
    )

    if name_col is None:
        raise ValueError(
            f"county boundary file is missing a name column; got {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "county_id": df[id_col].astype(str) if id_col else df.index.astype(str),
            "county": df[name_col].astype(str),
            "iso_code": df[iso_col].astype(str) if iso_col else None,
        }
    )

    out = gpd.GeoDataFrame(
        out, geometry=df.geometry.values, crs=df.crs or cfg.crs_geographic
    )

    # Drop any feature outside the canonical schema (e.g. a country-level row)
    # by requiring a non-empty, non-null geometry. In geopandas >= 1.0
    # ``notna()`` no longer returns False for empty geometries, so we
    # explicitly combine both predicates (this is the project-blessed
    # idiom and silenced below).
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "GeoSeries.notna", UserWarning)
        keep = ~out.geometry.is_empty & out.geometry.notna()
    out = out[keep].copy()
    return out[list(COUNTY_COLUMNS)]


def _empty_counties_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "county_id": pd.Series(dtype=str),
            "county": pd.Series(dtype=str),
            "iso_code": pd.Series(dtype=str),
        },
        geometry=gpd.GeoSeries([], crs="EPSG:4326"),
        crs="EPSG:4326",
    )


__all__ = [
    "COUNTY_COLUMNS",
    "DEFAULT_KENYA_COUNTIES_URL",
    "download_kenya_counties",
    "load_kenya_counties",
]
