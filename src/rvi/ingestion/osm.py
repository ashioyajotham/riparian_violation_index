"""OSM waterway ingestion (§3.1, §5.2 of the proposal).

Two complementary backends:

* :func:`fetch_waterways_overpass` — direct HTTP call to the Overpass API for
  small bounded queries (the Nairobi pilot).
* :func:`fetch_waterways_pbf`     — read a Geofabrik ``.osm.pbf`` extract via
  ``pyrosm`` for the national-scale run.

Both produce a uniform :class:`geopandas.GeoDataFrame` with the columns:

==============  ====================================================
column          meaning
--------------  ----------------------------------------------------
``osm_id``      stable OSM way id (string)
``waterway``    OSM ``waterway=*`` tag (river / stream / canal / ...)
``name``        OSM ``name`` tag (English/admin name) or ``None``
``name_local``  OSM ``name:sw`` (Swahili) when available
``strahler``    integer order from :data:`config.DEFAULT_WATERWAY_STRAHLER`
``length_m``    great-circle length, computed in :data:`Config.crs_metric`
``geometry``    LineString in :data:`Config.crs_geographic` (EPSG:4326)
==============  ====================================================

Filtering matches the proposal: only waterways that carry riparian obligations
under Kenyan law are kept (``river``, ``stream``, ``canal``, ``drain``,
``ditch``); infrastructure tags such as ``weir`` are dropped.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge

from rvi.config import Config, get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_waterways_overpass(
    bbox: tuple[float, float, float, float],
    *,
    config: Config | None = None,
    waterway_types: Sequence[str] | None = None,
    cache_path: Path | None = None,
    overwrite: bool = False,
    session: requests.Session | None = None,
    max_retries: int = 3,
    retry_backoff_s: float = 5.0,
) -> gpd.GeoDataFrame:
    """Fetch waterways from the Overpass API for a bounding box.

    Parameters
    ----------
    bbox
        ``(west, south, east, north)`` in EPSG:4326.
    waterway_types
        Override the default waterway types from config.
    cache_path
        If set, the raw Overpass JSON is persisted there and reused on
        subsequent calls (unless ``overwrite=True``).

    Notes
    -----
    The Overpass query unions ``way`` and ``relation`` waterways and uses
    ``out geom`` so node coordinates are returned inline — no second roundtrip
    needed.
    """
    cfg = config or get_config()
    types = tuple(waterway_types or cfg.waterway_types)

    cache: dict[str, Any] | None = None
    if cache_path is not None and cache_path.exists() and not overwrite:
        import json

        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        logger.info("Using cached Overpass response: %s", cache_path)

    if cache is None:
        query = build_overpass_query(bbox, types)
        cache = _post_overpass(
            query=query,
            url=cfg.overpass_url,
            timeout=cfg.request_timeout_s,
            user_agent=cfg.user_agent,
            session=session,
            max_retries=max_retries,
            backoff_s=retry_backoff_s,
        )
        if cache_path is not None:
            import json

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache), encoding="utf-8")

    return parse_overpass_response(cache, config=cfg)


def fetch_waterways_pbf(
    pbf_path: Path,
    *,
    config: Config | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    waterway_types: Sequence[str] | None = None,
) -> gpd.GeoDataFrame:
    """Read waterways from a Geofabrik ``.osm.pbf`` extract using ``pyrosm``.

    ``pyrosm`` is an optional, heavy dependency (installed via the
    ``national`` extra). It is imported lazily so the pilot pipeline does
    not require it.
    """
    cfg = config or get_config()
    types = tuple(waterway_types or cfg.waterway_types)

    try:
        from pyrosm import OSM  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fetch_waterways_pbf requires the optional 'national' extra: "
            "pip install -e \".[national]\""
        ) from exc

    logger.info("Reading PBF: %s", pbf_path)
    osm = OSM(str(pbf_path), bounding_box=list(bbox) if bbox else None)
    raw = osm.get_data_by_custom_criteria(
        custom_filter={"waterway": list(types)},
        filter_type="keep",
        keep_nodes=False,
        keep_ways=True,
        keep_relations=True,
    )
    if raw is None or raw.empty:
        return _empty_waterways_gdf(cfg)
    return _normalise_waterways(raw, cfg)


# ---------------------------------------------------------------------------
# Overpass query helpers
# ---------------------------------------------------------------------------


def build_overpass_query(
    bbox: tuple[float, float, float, float],
    waterway_types: Sequence[str],
) -> str:
    """Construct the Overpass QL string for a bounded waterway query."""
    if len(bbox) != 4:
        raise ValueError("bbox must be (west, south, east, north)")
    west, south, east, north = bbox
    if not (west < east and south < north):
        raise ValueError(
            f"invalid bbox ordering: west<east and south<north required, got {bbox}"
        )
    if not waterway_types:
        raise ValueError("waterway_types must be non-empty")

    # Overpass bbox order is (south, west, north, east).
    bbox_str = f"{south},{west},{north},{east}"
    type_alt = "|".join(waterway_types)

    return (
        '[out:json][timeout:90];\n'
        '(\n'
        f'  way["waterway"~"^({type_alt})$"]({bbox_str});\n'
        f'  relation["waterway"~"^({type_alt})$"]({bbox_str});\n'
        ');\n'
        'out geom;\n'
    )


def _post_overpass(
    *,
    query: str,
    url: str,
    timeout: float,
    user_agent: str,
    session: requests.Session | None,
    max_retries: int,
    backoff_s: float,
) -> dict[str, Any]:
    sess = session or requests.Session()
    headers = {"User-Agent": user_agent}
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("POST Overpass (attempt %d/%d)", attempt, max_retries)
            resp = sess.post(
                url,
                data={"data": query},
                headers=headers,
                timeout=timeout,
            )
            if resp.status_code in (429, 502, 503, 504):
                # Soft errors → backoff and retry.
                raise requests.HTTPError(f"Overpass returned HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            wait = backoff_s * attempt
            logger.warning("Overpass attempt %d failed: %s. Retrying in %.1fs", attempt, exc, wait)
            if attempt < max_retries:
                time.sleep(wait)
    assert last_exc is not None
    raise RuntimeError(f"Overpass failed after {max_retries} attempts") from last_exc


def parse_overpass_response(
    payload: dict[str, Any],
    *,
    config: Config | None = None,
) -> gpd.GeoDataFrame:
    """Convert a parsed Overpass JSON response into a uniform GeoDataFrame."""
    cfg = config or get_config()
    elements: list[dict[str, Any]] = payload.get("elements", []) or []

    rows: list[dict[str, Any]] = []
    for el in elements:
        geom = _overpass_element_to_geom(el)
        if geom is None or geom.is_empty:
            continue
        tags: dict[str, Any] = el.get("tags", {}) or {}
        waterway = tags.get("waterway")
        if waterway not in cfg.waterway_types:
            continue
        rows.append(
            {
                "osm_id": str(el.get("id")),
                "waterway": waterway,
                "name": tags.get("name"),
                "name_local": tags.get("name:sw") or tags.get("name:en") or None,
                "strahler": cfg.strahler_for_waterway(waterway),
                "geometry": geom,
            }
        )

    if not rows:
        return _empty_waterways_gdf(cfg)

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=cfg.crs_geographic)
    return _annotate_lengths(gdf, cfg)


def _overpass_element_to_geom(element: dict[str, Any]) -> LineString | MultiLineString | None:
    """Turn a single Overpass ``way``/``relation`` element into a LineString.

    Overpass ``way`` elements have a flat ``geometry: [{lat, lon}, ...]`` list.
    ``relation`` elements have ``members`` each with their own ``geometry``.
    """
    el_type = element.get("type")
    if el_type == "way":
        coords = [
            (float(p["lon"]), float(p["lat"]))
            for p in element.get("geometry", []) or []
            if "lat" in p and "lon" in p
        ]
        return LineString(coords) if len(coords) >= 2 else None

    if el_type == "relation":
        parts: list[LineString] = []
        for member in element.get("members", []) or []:
            geom = member.get("geometry") or []
            coords = [
                (float(p["lon"]), float(p["lat"]))
                for p in geom
                if "lat" in p and "lon" in p
            ]
            if len(coords) >= 2:
                parts.append(LineString(coords))
        if not parts:
            return None
        merged = linemerge(MultiLineString(parts))
        return merged if not merged.is_empty else None
    return None


# ---------------------------------------------------------------------------
# Normalisation shared between Overpass and PBF backends
# ---------------------------------------------------------------------------


WATERWAY_COLUMNS: tuple[str, ...] = (
    "osm_id",
    "waterway",
    "name",
    "name_local",
    "strahler",
    "length_m",
    "geometry",
)


def _normalise_waterways(raw: gpd.GeoDataFrame, cfg: Config) -> gpd.GeoDataFrame:
    """Coerce a pyrosm-derived GeoDataFrame into the canonical RVI schema."""
    df = raw.copy()
    if "id" in df.columns and "osm_id" not in df.columns:
        df["osm_id"] = df["id"]
    if "osm_id" not in df.columns:
        df["osm_id"] = pd.Series(range(len(df)), index=df.index).astype(str)

    df["osm_id"] = df["osm_id"].astype(str)

    if "tags" in df.columns:
        # pyrosm packs less-common tags into a JSON-like 'tags' dict column.
        df["name_local"] = df.apply(_extract_name_local, axis=1)
    else:
        df["name_local"] = df.get("name:sw")

    if "waterway" not in df.columns:
        raise ValueError("PBF result is missing the 'waterway' column")

    df["strahler"] = df["waterway"].map(cfg.strahler_for_waterway).fillna(1).astype(int)

    df = df.rename(columns={"geom": "geometry"} if "geom" in df.columns else {})
    if df.geometry.crs is None:
        df = df.set_crs(cfg.crs_geographic)

    df = df[df.geometry.notna()].copy()
    df = df[df["waterway"].isin(cfg.waterway_types)].copy()

    keep = [c for c in ("osm_id", "waterway", "name", "name_local", "strahler") if c in df.columns]
    df = df[[*keep, "geometry"]]
    if "name" not in df.columns:
        df["name"] = None
    if "name_local" not in df.columns:
        df["name_local"] = None

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=df.crs or cfg.crs_geographic)
    return _annotate_lengths(gdf, cfg)


def _extract_name_local(row: pd.Series) -> Any:
    tags = row.get("tags")
    if isinstance(tags, dict):
        return tags.get("name:sw") or tags.get("name:en")
    if isinstance(tags, str):
        # Pyrosm sometimes returns tags as a "k=>v;k=>v" string. Best-effort parse.
        for chunk in tags.split(","):
            if "name:sw" in chunk:
                _, _, v = chunk.partition("=>")
                return v.strip().strip('"') or None
    return None


def _annotate_lengths(gdf: gpd.GeoDataFrame, cfg: Config) -> gpd.GeoDataFrame:
    """Add an accurate ``length_m`` column computed in metric CRS."""
    if gdf.empty:
        gdf["length_m"] = pd.Series(dtype=float)
        return gdf[list(WATERWAY_COLUMNS)]
    metric = gdf.to_crs(cfg.crs_metric)
    gdf = gdf.copy()
    gdf["length_m"] = metric.geometry.length.astype(float)
    return gdf[list(WATERWAY_COLUMNS)]


def _empty_waterways_gdf(cfg: Config) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {col: pd.Series(dtype="object") for col in WATERWAY_COLUMNS if col != "geometry"},
        geometry=gpd.GeoSeries([], crs=cfg.crs_geographic),
        crs=cfg.crs_geographic,
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def save_waterways(gdf: gpd.GeoDataFrame, path: Path, *, layer: str = "waterways") -> Path:
    """Persist a waterway GeoDataFrame as a GeoPackage layer."""
    from rvi.io import write_geopackage

    return write_geopackage(gdf, path, layer=layer)


def load_waterways(path: Path, *, layer: str = "waterways") -> gpd.GeoDataFrame:
    return gpd.read_file(path, layer=layer)


__all__ = [
    "WATERWAY_COLUMNS",
    "build_overpass_query",
    "fetch_waterways_overpass",
    "fetch_waterways_pbf",
    "load_waterways",
    "parse_overpass_response",
    "save_waterways",
]


def filter_waterway_types(
    gdf: gpd.GeoDataFrame, types: Iterable[str]
) -> gpd.GeoDataFrame:
    """Return only rows whose ``waterway`` value is in *types*."""
    return gdf[gdf["waterway"].isin(set(types))].copy()
