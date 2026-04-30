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
    """Read waterways from a Geofabrik ``.osm.pbf`` extract using ``osmium``.

    ``osmium`` (libosmium's Python bindings, installed via the ``national``
    extra) is imported lazily so the pilot pipeline does not require it.
    The function reads the PBF in a single streaming pass, materialises each
    matching way's geometry via ``osmium.geom.WKTFactory``, and returns a
    GeoDataFrame in the canonical schema documented at the top of this
    module.

    Parameters
    ----------
    pbf_path
        Path to a ``.osm.pbf`` file (e.g. the Geofabrik Kenya extract).
    bbox
        Optional ``(west, south, east, north)`` filter applied after parsing.
        We intentionally don't push the bbox into osmium itself — for a
        country-sized PBF the speed-up is negligible and the post-filter is
        easier to reason about.
    """
    cfg = config or get_config()
    types = tuple(waterway_types or cfg.waterway_types)
    pbf_path = Path(pbf_path)
    if not pbf_path.exists():
        raise FileNotFoundError(f"PBF not found: {pbf_path}")

    try:
        import osmium  # type: ignore[import-not-found]
        from osmium.geom import WKTFactory  # type: ignore[import-not-found]
        from shapely import wkt as _wkt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            'fetch_waterways_pbf requires the optional "national" extra: '
            'pip install -e ".[national]"'
        ) from exc

    logger.info("Reading PBF: %s (%.1f MB)", pbf_path, pbf_path.stat().st_size / 1e6)
    fact = WKTFactory()
    type_set = set(types)

    fp = (
        osmium.FileProcessor(str(pbf_path))
        .with_locations()
        .with_filter(osmium.filter.KeyFilter("waterway"))
    )

    rows: list[dict[str, Any]] = []
    way_count = 0
    for obj in fp:
        if not obj.is_way():
            continue
        way_count += 1
        ww = obj.tags.get("waterway")
        if ww not in type_set:
            continue
        try:
            wkt_str = fact.create_linestring(obj)
        except Exception as exc:  # incomplete way / closed ring / etc.
            logger.debug("skip way %s: %s", obj.id, exc)
            continue
        if not wkt_str:
            continue
        try:
            geom = _wkt.loads(wkt_str)
        except Exception:
            continue
        if geom.is_empty:
            continue
        tags = obj.tags
        rows.append(
            {
                "osm_id": str(obj.id),
                "waterway": ww,
                "name": tags.get("name"),
                "name_local": tags.get("name:sw") or tags.get("name:en") or None,
                "strahler": cfg.strahler_for_waterway(ww),
                "geometry": geom,
            }
        )

    logger.info(
        "PBF parsed: %d ways inspected, %d retained (%s)",
        way_count,
        len(rows),
        ", ".join(types),
    )
    if not rows:
        return _empty_waterways_gdf(cfg)

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=cfg.crs_geographic)
    if bbox is not None:
        from shapely.geometry import box

        bbox_geom = box(*bbox)
        gdf = gdf[gdf.geometry.intersects(bbox_geom)].copy()

    return _annotate_lengths(gdf, cfg)


def download_geofabrik_kenya_pbf(
    *,
    config: Config | None = None,
    target_path: Path | None = None,
    overwrite: bool = False,
    session: requests.Session | None = None,
    chunk_size: int = 1 << 20,  # 1 MiB
    progress: bool = True,
) -> Path:
    """Download the Geofabrik Kenya PBF (~317 MB), cached on disk.

    The file is streamed to disk so it never has to fit in memory. By
    default the destination is ``cfg.cache_dir / "kenya-latest.osm.pbf"``;
    pass ``target_path=`` to override.
    """
    cfg = config or get_config()
    cfg.ensure_dirs()
    target_path = Path(target_path) if target_path else cfg.cache_dir / "kenya-latest.osm.pbf"

    if target_path.exists() and not overwrite:
        logger.info(
            "Geofabrik Kenya PBF already cached: %s (%.1f MB)",
            target_path,
            target_path.stat().st_size / 1e6,
        )
        return target_path

    sess = session or requests.Session()
    sess.headers.setdefault("User-Agent", cfg.user_agent)

    url = cfg.geofabrik_kenya_pbf_url
    logger.info("Downloading Geofabrik Kenya PBF: %s -> %s", url, target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = target_path.with_suffix(target_path.suffix + ".part")

    with sess.get(url, stream=True, timeout=cfg.request_timeout_s) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", "0")) or None
        bar = None
        if progress:
            try:
                from tqdm import tqdm

                bar = tqdm(
                    total=total, unit="B", unit_scale=True, desc="Geofabrik Kenya PBF"
                )
            except ImportError:  # pragma: no cover
                bar = None
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                fh.write(chunk)
                if bar is not None:
                    bar.update(len(chunk))
        if bar is not None:
            bar.close()

    tmp.replace(target_path)
    logger.info("Saved %s (%.1f MB)", target_path, target_path.stat().st_size / 1e6)
    return target_path


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
    "download_geofabrik_kenya_pbf",
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
