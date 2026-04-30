"""Microsoft Global ML Building Footprints ingestion (§3.2 of the proposal).

Two scales of access:

* :func:`load_buildings_for_bbox` — for the Nairobi pilot. Downloads the few
  Microsoft tiles whose quadkeys cover the bounding box and assembles them into
  one :class:`geopandas.GeoDataFrame`. The output is cached under
  ``cache_dir`` keyed by the bbox so re-runs are cheap.
* :func:`stream_buildings_duckdb` — for the national-scale run. Uses DuckDB
  with the ``spatial`` extension to stream and spatially-filter the full
  ~15-million-record dataset without loading it into memory.

The Microsoft dataset is published as a CSV index (``dataset-links.csv``)
listing one ``.csv.gz`` URL per quadkey tile. Each row of a tile file has:

    QuadKey, Location, geometry_wkt   (or "geometry" or "WKT")

where ``geometry`` is a WKT polygon in EPSG:4326. We parse both the modern
``geometry_wkt`` column and the older ``geometry`` GeoJSON-string column.

References
----------
https://github.com/microsoft/GlobalMLBuildingFootprints
"""

from __future__ import annotations

import gzip
import io
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from shapely import wkt as _wkt
from shapely.geometry import Polygon, box

from rvi.config import Config, get_config

logger = logging.getLogger(__name__)


BUILDINGS_COLUMNS: tuple[str, ...] = (
    "building_id",
    "country",
    "quadkey",
    "footprint_area_m2",
    "geometry",
)


# ---------------------------------------------------------------------------
# Quadkey arithmetic
#
# The Microsoft dataset is partitioned by Bing Maps quadkey strings (one
# character per zoom level). For pilot-scale queries we only need to enumerate
# the quadkeys that intersect a bounding box at a given zoom level. Microsoft's
# tiles vary in zoom (often 9), so we compute several plausible zooms and the
# loader gracefully skips empty tiles.
# ---------------------------------------------------------------------------


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """Convert ``(lon, lat)`` to Bing-Maps tile (x, y) at *zoom*."""
    import math

    sin_lat = math.sin(math.radians(lat))
    sin_lat = max(min(sin_lat, 0.9999), -0.9999)
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int(
        (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * n
    )
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return x, y


def tile_to_quadkey(x: int, y: int, zoom: int) -> str:
    """Bing-Maps tile (x, y, z) → quadkey string."""
    quadkey: list[str] = []
    for i in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if x & mask:
            digit += 1
        if y & mask:
            digit += 2
        quadkey.append(str(digit))
    return "".join(quadkey)


def quadkeys_for_bbox(
    bbox: tuple[float, float, float, float], zoom: int = 9
) -> list[str]:
    """Enumerate every quadkey at *zoom* that intersects the bbox."""
    west, south, east, north = bbox
    x0, y0 = lonlat_to_tile(west, north, zoom)
    x1, y1 = lonlat_to_tile(east, south, zoom)
    keys: list[str] = []
    for x in range(min(x0, x1), max(x0, x1) + 1):
        for y in range(min(y0, y1), max(y0, y1) + 1):
            keys.append(tile_to_quadkey(x, y, zoom))
    return sorted(set(keys))


# ---------------------------------------------------------------------------
# Pilot loader
# ---------------------------------------------------------------------------


def load_buildings_for_bbox(
    bbox: tuple[float, float, float, float],
    *,
    config: Config | None = None,
    country: str = "Kenya",
    zoom: int = 9,
    index_url: str | None = None,
    cache_path: Path | None = None,
    overwrite: bool = False,
    session: requests.Session | None = None,
    progress: bool = True,
) -> gpd.GeoDataFrame:
    """Return all Microsoft footprints whose centroid falls inside *bbox*.

    Implementation outline:

    1. Read the Microsoft index CSV (one row per quadkey tile, with a URL).
    2. Restrict to the requested ``country``.
    3. Compute the quadkeys covering ``bbox`` at zoom 9 (Microsoft's default).
    4. Download each matching tile, parse the WKT/GeoJSON geometry column.
    5. Filter to centroids inside the bbox to drop fully-outside tile rows.
    6. Compute footprint areas in metric CRS.
    """
    cfg = config or get_config()
    cache_path = (
        Path(cache_path)
        if cache_path is not None
        else cfg.cache_dir / f"ms_buildings_bbox_{_bbox_slug(bbox)}.gpkg"
    )
    if cache_path.exists() and not overwrite:
        logger.info("Using cached MS buildings: %s", cache_path)
        return gpd.read_file(cache_path)

    cfg.ensure_dirs()
    sess = session or requests.Session()
    sess.headers.setdefault("User-Agent", cfg.user_agent)

    index_df = _fetch_buildings_index(
        cfg=cfg, sess=sess, override_url=index_url
    )
    index_df = index_df[index_df["Location"].astype(str).str.lower() == country.lower()].copy()
    if index_df.empty:
        raise ValueError(
            f"Microsoft buildings index has no rows for country={country!r}"
        )

    target_quadkeys = set(quadkeys_for_bbox(bbox, zoom=zoom))
    if not target_quadkeys:
        return _empty_buildings_gdf(cfg)

    qk_col = _find_index_quadkey_column(index_df)
    selected = index_df[index_df[qk_col].astype(str).isin(target_quadkeys)]
    if selected.empty:
        # Try string-prefix match (Microsoft sometimes uses a coarser zoom).
        selected = index_df[
            index_df[qk_col].astype(str).apply(
                lambda qk: any(qk.startswith(t) or t.startswith(qk) for t in target_quadkeys)
            )
        ]

    if selected.empty:
        logger.warning(
            "No Microsoft tiles match bbox %s (zoom=%d) for country=%s",
            bbox,
            zoom,
            country,
        )
        return _empty_buildings_gdf(cfg)

    iterator: Iterable[tuple[Any, pd.Series]] = selected.iterrows()
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(
                iterator,
                total=len(selected),
                desc="MS tiles",
                unit="tile",
            )
        except ImportError:  # pragma: no cover - tqdm is required
            pass

    frames: list[gpd.GeoDataFrame] = []
    bbox_geom = box(*bbox)
    for _, row in iterator:
        url = str(row["Url"])
        gdf = _download_and_parse_tile(url, sess=sess, cfg=cfg)
        if gdf.empty:
            continue
        gdf["country"] = country
        gdf["quadkey"] = str(row[qk_col])
        # Centroid-in-bbox filter using fast, vectorised shapely 2 ops.
        centroids = gdf.geometry.representative_point()
        gdf = gdf[centroids.within(bbox_geom)].copy()
        if not gdf.empty:
            frames.append(gdf)

    if not frames:
        return _empty_buildings_gdf(cfg)

    out = pd.concat(frames, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=cfg.crs_geographic)
    out = _annotate_areas(out, cfg)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    from rvi.io import write_geopackage  # avoid circular import on package init

    write_geopackage(out, cache_path)
    return out


# ---------------------------------------------------------------------------
# National loader (DuckDB)
# ---------------------------------------------------------------------------


def load_buildings_for_country(
    waterway_buffers: gpd.GeoDataFrame,
    *,
    config: Config | None = None,
    country: str = "Kenya",
    index_url: str | None = None,
    cache_path: Path | None = None,
    overwrite: bool = False,
    session: requests.Session | None = None,
    progress: bool = True,
) -> gpd.GeoDataFrame:
    """Stream every Microsoft tile for a country and keep only footprints
    intersecting the waterway-buffer union.

    This is the path used by the national-scale run. The function downloads
    each tile in turn, parses it (CSV or GeoJSONL — see :func:`_parse_tile_text`),
    and spatially joins the tile against the buffer GeoDataFrame *immediately*
    so we never hold more than one tile's worth of buildings in memory.

    Filtering at the tile boundary takes the dataset from ~15 M Kenyan
    footprints down to ~1–2 M (anything within the 30 m + Strahler buffer
    of any waterway in Kenya), which fits comfortably in memory.
    """
    cfg = config or get_config()
    cfg.ensure_dirs()
    cache_path = (
        Path(cache_path)
        if cache_path is not None
        else cfg.cache_dir / f"ms_buildings_country_{country.lower()}.gpkg"
    )
    if cache_path.exists() and not overwrite:
        logger.info("Using cached country-scale buildings: %s", cache_path)
        return gpd.read_file(cache_path)

    if waterway_buffers.empty:
        logger.warning("waterway_buffers is empty; nothing to filter against")
        return _empty_buildings_gdf(cfg)

    sess = session or requests.Session()
    sess.headers.setdefault("User-Agent", cfg.user_agent)

    index_df = _fetch_buildings_index(cfg=cfg, sess=sess, override_url=index_url)
    country_rows = index_df[
        index_df["Location"].astype(str).str.lower() == country.lower()
    ].copy()
    if country_rows.empty:
        raise ValueError(
            f"Microsoft buildings index has no rows for country={country!r}"
        )

    # Reproject buffers to EPSG:4326 once so the per-tile sjoin is cheap.
    buffers_geo = (
        waterway_buffers.to_crs(cfg.crs_geographic)
        if waterway_buffers.crs and waterway_buffers.crs != cfg.crs_geographic
        else waterway_buffers
    )
    buffers_geo = gpd.GeoDataFrame(
        {"_buffer_idx": range(len(buffers_geo))},
        geometry=buffers_geo.geometry.values,
        crs=cfg.crs_geographic,
    )

    iterator: Iterable[tuple[Any, pd.Series]] = country_rows.iterrows()
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(
                iterator,
                total=len(country_rows),
                desc=f"MS tiles ({country})",
                unit="tile",
            )
        except ImportError:  # pragma: no cover
            pass

    qk_col = _find_index_quadkey_column(country_rows)
    frames: list[gpd.GeoDataFrame] = []
    total_inspected = 0
    total_kept = 0
    for _, row in iterator:
        url = str(row["Url"])
        tile = _download_and_parse_tile(url, sess=sess, cfg=cfg)
        if tile.empty:
            continue
        total_inspected += len(tile)
        # Per-tile spatial filter against the buffer union.
        joined = gpd.sjoin(
            tile[["geometry"]], buffers_geo, how="inner", predicate="intersects"
        )
        if joined.empty:
            continue
        kept = tile.loc[joined.index.unique()].copy()
        kept["country"] = country
        kept["quadkey"] = str(row[qk_col])
        total_kept += len(kept)
        frames.append(kept)

    logger.info(
        "Country-scale build: inspected %d footprints, kept %d after buffer filter",
        total_inspected,
        total_kept,
    )

    if not frames:
        return _empty_buildings_gdf(cfg)

    out = pd.concat(frames, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=cfg.crs_geographic)
    # Re-id globally; per-tile building_id collides across tiles.
    out["building_id"] = pd.RangeIndex(start=1, stop=len(out) + 1)
    out = _annotate_areas(out, cfg)

    from rvi.io import write_geopackage

    write_geopackage(out, cache_path)
    return out


def stream_buildings_duckdb(
    waterway_buffers: gpd.GeoDataFrame,
    *,
    config: Config | None = None,
    country: str = "Kenya",
    index_url: str | None = None,
    parquet_dir: Path | None = None,
) -> gpd.GeoDataFrame:
    """Stream Microsoft footprints filtered to the union of waterway buffers.

    This is the path used by the national-scale run. The function:

    1. Reads (or builds) a Parquet copy of the Microsoft tiles for *country*.
    2. Loads DuckDB's ``spatial`` extension and registers the buffer geometry.
    3. Performs a single ``ST_Intersects`` join.

    Building the Parquet copy is expensive on a fresh machine but yields
    constant-time queries thereafter; ``parquet_dir`` defaults to
    ``cfg.cache_dir / "ms_buildings_parquet"``.

    Parameters
    ----------
    waterway_buffers
        A GeoDataFrame whose ``geometry`` is a (Multi)Polygon coverage of all
        riparian buffer corridors. The function returns building footprints
        intersecting this union.
    """
    import duckdb

    cfg = config or get_config()
    parquet_dir = Path(parquet_dir) if parquet_dir else cfg.cache_dir / "ms_buildings_parquet"

    if waterway_buffers.empty:
        return _empty_buildings_gdf(cfg)

    if parquet_dir.exists() and any(parquet_dir.glob("*.parquet")):
        logger.info("Reusing parquet at %s", parquet_dir)
    else:
        _materialise_country_parquet(
            cfg=cfg,
            country=country,
            parquet_dir=parquet_dir,
            override_index_url=index_url,
        )

    con = duckdb.connect()
    con.install_extension("spatial")
    con.load_extension("spatial")

    # Push the buffer union into DuckDB as a single WKT scalar.
    if waterway_buffers.crs is None:
        waterway_buffers = waterway_buffers.set_crs(cfg.crs_geographic)
    buffers_geo = waterway_buffers.to_crs(cfg.crs_geographic)
    union_geom = (
        buffers_geo.geometry.union_all()
        if hasattr(buffers_geo.geometry, "union_all")
        else buffers_geo.geometry.unary_union
    )
    union_wkt = union_geom.wkt

    con.execute("CREATE OR REPLACE TEMP VIEW buf_view AS SELECT ST_GeomFromText(?) AS geom", [union_wkt])

    parquet_glob = str(parquet_dir / "*.parquet")
    sql = f"""
        SELECT
            row_number() OVER () AS building_id,
            country,
            quadkey,
            ST_AsText(geom) AS wkt
        FROM (
            SELECT
                country,
                quadkey,
                ST_GeomFromText(geometry_wkt) AS geom
            FROM read_parquet('{parquet_glob}')
        ) src
        JOIN buf_view ON ST_Intersects(src.geom, buf_view.geom)
    """
    df = con.execute(sql).fetch_df()
    con.close()

    if df.empty:
        return _empty_buildings_gdf(cfg)

    df["geometry"] = df["wkt"].apply(_wkt.loads)
    df = df.drop(columns=["wkt"])
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=cfg.crs_geographic)
    return _annotate_areas(gdf, cfg)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_buildings_index(
    *, cfg: Config, sess: requests.Session, override_url: str | None
) -> pd.DataFrame:
    url = override_url or cfg.ms_buildings_index_url
    cache_path = cfg.cache_dir / "ms_buildings_index.csv"
    cfg.ensure_dirs()
    if cache_path.exists():
        logger.info("Using cached MS index: %s", cache_path)
        return pd.read_csv(cache_path)
    logger.info("Fetching MS index: %s", url)
    resp = sess.get(url, timeout=cfg.request_timeout_s)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return pd.read_csv(io.BytesIO(resp.content))


def _find_index_quadkey_column(df: pd.DataFrame) -> str:
    for cand in ("QuadKey", "Quadkey", "quadkey"):
        if cand in df.columns:
            return cand
    raise KeyError("MS buildings index has no QuadKey column")


def _download_and_parse_tile(
    url: str, *, sess: requests.Session, cfg: Config
) -> gpd.GeoDataFrame:
    """Download one Microsoft tile and parse its WKT geometries.

    The CSV is small enough (single-tile) that streaming gives no benefit;
    we read the whole body and decompress in-memory.
    """
    try:
        resp = sess.get(url, timeout=cfg.request_timeout_s)
        resp.raise_for_status()
        body = resp.content
    except requests.RequestException as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return _empty_buildings_gdf(cfg)
    try:
        text = (
            gzip.decompress(body).decode("utf-8")
            if url.endswith(".gz")
            else body.decode("utf-8")
        )
    except OSError as exc:
        logger.warning("Failed to gunzip %s: %s", url, exc)
        return _empty_buildings_gdf(cfg)

    if not text.strip():
        return _empty_buildings_gdf(cfg)

    return _parse_tile_text(text, cfg)


def _parse_tile_text(text: str, cfg: Config) -> gpd.GeoDataFrame:
    """Parse a Microsoft tile body as either GeoJSONL or legacy CSV.

    Microsoft's "Global ML Building Footprints" tiles have shipped in two
    formats over time:

    * **Legacy CSV.gz** — flat CSV with a ``geometry_wkt`` (or ``geometry``
      GeoJSON-string) column.
    * **GeoJSONL.gz** — one GeoJSON ``Feature`` per line, with
      ``geometry.type == "Polygon"``. The 2026-02 Kenya tiles use this.

    We sniff the first non-whitespace character: ``{`` ⇒ JSONL, else CSV.
    """
    head = text.lstrip()[:1]
    if head == "{":
        return _parse_jsonl_tile(text, cfg)

    try:
        df = pd.read_csv(io.StringIO(text))
    except pd.errors.ParserError as exc:
        # Some legacy tiles ship un-quoted commas in WKT; fall back to the
        # tolerant Python engine.
        logger.warning("CSV parse failed (%s); retrying with python engine", exc)
        try:
            df = pd.read_csv(
                io.StringIO(text), engine="python", on_bad_lines="skip"
            )
        except Exception as exc2:  # pragma: no cover - defensive
            logger.warning("Tile CSV unparseable, skipping: %s", exc2)
            return _empty_buildings_gdf(cfg)
    return _rows_to_geodataframe(df, cfg)


def _parse_jsonl_tile(text: str, cfg: Config) -> gpd.GeoDataFrame:
    """Parse a GeoJSONL tile (one ``Feature`` per line) into a GeoDataFrame."""
    import json as _json

    geoms: list[Polygon] = []
    heights: list[float] = []
    confidences: list[float] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        geom = obj.get("geometry") or {}
        if geom.get("type") != "Polygon":
            continue
        coords = geom.get("coordinates") or []
        if not coords or not coords[0]:
            continue
        try:
            poly = Polygon(coords[0], holes=coords[1:] or None)
        except (TypeError, ValueError):
            continue
        if poly.is_empty or not poly.is_valid:
            continue
        geoms.append(poly)
        props = obj.get("properties") or {}
        heights.append(_to_float(props.get("height")))
        confidences.append(_to_float(props.get("confidence")))

    if not geoms:
        return _empty_buildings_gdf(cfg)

    out = gpd.GeoDataFrame(
        {
            "country": [""] * len(geoms),
            "quadkey": [""] * len(geoms),
            "height": heights,
            "confidence": confidences,
            "geometry": geoms,
        },
        geometry="geometry",
        crs=cfg.crs_geographic,
    )
    out["building_id"] = pd.RangeIndex(start=1, stop=len(out) + 1)
    return out[["building_id", "country", "quadkey", "geometry"]]


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _rows_to_geodataframe(df: pd.DataFrame, cfg: Config) -> gpd.GeoDataFrame:
    """Convert a Microsoft tile DataFrame into a GeoDataFrame.

    Microsoft has used three geometry encodings over time:

    * ``geometry_wkt``  — preferred, plain WKT.
    * ``geometry``      — GeoJSON string (older tiles).
    * ``WKT``           — same as ``geometry_wkt`` but capitalised.

    We probe in that order.
    """
    if df.empty:
        return _empty_buildings_gdf(cfg)

    wkt_col = next((c for c in ("geometry_wkt", "WKT", "wkt") if c in df.columns), None)
    if wkt_col is not None:
        geoms = df[wkt_col].astype(str).apply(_safe_wkt)
    elif "geometry" in df.columns:
        geoms = df["geometry"].astype(str).apply(_safe_geojson_to_polygon)
    else:
        logger.warning("MS tile lacks geometry column; columns=%s", list(df.columns))
        return _empty_buildings_gdf(cfg)

    out = gpd.GeoDataFrame(
        {
            "country": df.get("Location", df.get("country", "")).astype(str)
            if "Location" in df.columns or "country" in df.columns
            else "",
            "quadkey": df.get("QuadKey", df.get("quadkey", "")).astype(str)
            if "QuadKey" in df.columns or "quadkey" in df.columns
            else "",
            "geometry": geoms,
        },
        geometry="geometry",
        crs=cfg.crs_geographic,
    )
    out = out[out.geometry.notna()].copy()
    out["building_id"] = pd.RangeIndex(start=1, stop=len(out) + 1)
    return out[["building_id", "country", "quadkey", "geometry"]]


def _safe_wkt(s: str) -> Polygon | None:
    try:
        return _wkt.loads(s)
    except Exception:
        return None


def _safe_geojson_to_polygon(s: str) -> Polygon | None:
    import json

    try:
        obj = json.loads(s)
    except Exception:
        return None
    coords = obj.get("coordinates")
    geom_type = obj.get("type")
    try:
        if geom_type == "Polygon" and coords:
            return Polygon(coords[0], holes=coords[1:] or None)
    except Exception:
        return None
    return None


def _annotate_areas(gdf: gpd.GeoDataFrame, cfg: Config) -> gpd.GeoDataFrame:
    if gdf.empty:
        gdf["footprint_area_m2"] = pd.Series(dtype=float)
        cols = [c for c in BUILDINGS_COLUMNS if c in gdf.columns or c == "geometry"]
        return gdf.reindex(columns=[*cols, "geometry"]).drop_duplicates(
            subset=["geometry"], keep="first"
        )
    metric = gdf.to_crs(cfg.crs_metric)
    gdf = gdf.copy()
    gdf["footprint_area_m2"] = metric.geometry.area.astype(float)
    if "building_id" not in gdf.columns:
        gdf["building_id"] = pd.RangeIndex(start=1, stop=len(gdf) + 1)
    if "country" not in gdf.columns:
        gdf["country"] = ""
    if "quadkey" not in gdf.columns:
        gdf["quadkey"] = ""
    return gdf[list(BUILDINGS_COLUMNS)]


def _empty_buildings_gdf(cfg: Config) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "building_id": pd.Series(dtype="int64"),
            "country": pd.Series(dtype=str),
            "quadkey": pd.Series(dtype=str),
            "footprint_area_m2": pd.Series(dtype=float),
        },
        geometry=gpd.GeoSeries([], crs=cfg.crs_geographic),
        crs=cfg.crs_geographic,
    )


def _bbox_slug(bbox: tuple[float, float, float, float]) -> str:
    w, s, e, n = bbox
    return f"{w:.4f}_{s:.4f}_{e:.4f}_{n:.4f}".replace("-", "m").replace(".", "p")


def _materialise_country_parquet(
    *,
    cfg: Config,
    country: str,
    parquet_dir: Path,
    override_index_url: str | None,
) -> None:  # pragma: no cover - heavy national-scale path
    """Convert all of *country*'s Microsoft tiles into a single parquet folder.

    Skipped from the test suite (it would download many GB), but exercised
    by ``rvi national``.
    """
    import duckdb

    sess = requests.Session()
    sess.headers["User-Agent"] = cfg.user_agent
    index = _fetch_buildings_index(cfg=cfg, sess=sess, override_url=override_index_url)
    index = index[index["Location"].astype(str).str.lower() == country.lower()]
    parquet_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.install_extension("spatial")
    con.load_extension("spatial")

    for _, row in index.iterrows():
        url = str(row["Url"])
        qk = str(row[_find_index_quadkey_column(index)])
        target = parquet_dir / f"{country}_{qk}.parquet"
        if target.exists():
            continue
        gdf = _download_and_parse_tile(url, sess=sess, cfg=cfg)
        if gdf.empty:
            continue
        df = pd.DataFrame(
            {
                "country": gdf["country"],
                "quadkey": gdf["quadkey"],
                "geometry_wkt": gdf.geometry.apply(lambda g: g.wkt),
            }
        )
        con.from_df(df).to_parquet(str(target))
    con.close()


def buildings_iter(
    gdf: gpd.GeoDataFrame, batch_size: int = 50_000
) -> Iterator[gpd.GeoDataFrame]:
    """Iterate a buildings GeoDataFrame in chunks (preserves CRS)."""
    if gdf.empty:
        return
    for start in range(0, len(gdf), batch_size):
        yield gdf.iloc[start : start + batch_size].copy()


__all__ = [
    "BUILDINGS_COLUMNS",
    "buildings_iter",
    "load_buildings_for_bbox",
    "load_buildings_for_country",
    "lonlat_to_tile",
    "quadkeys_for_bbox",
    "stream_buildings_duckdb",
    "tile_to_quadkey",
]
