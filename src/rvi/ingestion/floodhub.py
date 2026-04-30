"""Google Flood Forecasting API client (§2.4, §3.3 of the proposal).

This module is a thin REST wrapper over the six Flood Hub endpoints listed in
the proposal:

================================================  ======  =======================================
Endpoint                                          Method  Purpose
------------------------------------------------  ------  ---------------------------------------
``gauges:searchGaugesByArea``                     POST    discover gauge ids in a region/polygon
``floodStatus:searchLatestFloodStatusByArea``     POST    current severity in a region
``floodStatus:queryLatestFloodStatusByGaugeIds``  GET     current severity for specific gauges
``gauges:queryGaugeForecasts``                    GET     7-day forecast for one gauge
``gaugeModels:batchGet``                          GET     warning / danger / extreme thresholds
``gauges:batchGet``                               GET     gauge metadata
================================================  ======  =======================================

The :class:`Severity` IntEnum exposes the integer ordinal encoding described
in §2.4.1 (``UNKNOWN=0``, ``NO_FLOODING=1``, ``ABOVE_NORMAL=2``, ``SEVERE=3``,
``EXTREME=4``). Using IntEnum preserves the ordinal relationship through every
downstream statistical operation (``int(Severity.SEVERE) == 3``) without any
intermediate mapping step.
"""

from __future__ import annotations

import enum
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from rvi.config import Config, get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity ordinal (§2.4.1)
# ---------------------------------------------------------------------------


class Severity(enum.IntEnum):
    """Integer-ordinal encoding of Flood Hub severity values.

    The IntEnum design is deliberate: the proposal's Spearman correlation in
    §5.5 is rank-based and consumes integers directly. Because IntEnum subclasses
    ``int``, ``Severity.SEVERE`` participates as ``3`` in any pandas / scipy
    operation without an extra mapping step.
    """

    UNKNOWN = 0
    NO_FLOODING = 1
    ABOVE_NORMAL = 2
    SEVERE = 3
    EXTREME = 4

    @classmethod
    def from_api(cls, value: str | int | None) -> Severity:
        """Coerce a raw API value (string or already-int) into the enum."""
        if value is None:
            return cls.UNKNOWN
        if isinstance(value, int):
            try:
                return cls(value)
            except ValueError:
                return cls.UNKNOWN
        # API returns e.g. "SEVERE", "SEVERITY_UNSPECIFIED", "EXTREME".
        token = str(value).upper().strip()
        mapping = {
            "NO_FLOODING": cls.NO_FLOODING,
            "ABOVE_NORMAL": cls.ABOVE_NORMAL,
            "SEVERE": cls.SEVERE,
            "EXTREME": cls.EXTREME,
            "SEVERITY_UNSPECIFIED": cls.UNKNOWN,
            "UNKNOWN": cls.UNKNOWN,
            "": cls.UNKNOWN,
        }
        return mapping.get(token, cls.UNKNOWN)


# ---------------------------------------------------------------------------
# Typed data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gauge:
    """A Flood Hub gauge location."""

    gauge_id: str
    latitude: float
    longitude: float
    site_name: str | None = None
    river: str | None = None
    country_code: str | None = None
    quality_verified: bool = False
    source: str | None = None

    def to_point(self) -> Point:
        return Point(self.longitude, self.latitude)


@dataclass(frozen=True)
class FloodStatus:
    """Current flood severity reading for a single gauge."""

    gauge_id: str
    severity: Severity
    issued_time: str | None = None  # RFC-3339 timestamp string
    forecast_change: str | None = None
    raw: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class FloodHubError(RuntimeError):
    """Raised on irrecoverable Flood Hub API errors."""


class FloodHubClient:
    """Thin REST client for the Flood Forecasting API.

    The client adds the API key as a ``key=...`` query parameter on every
    request (the standard Google API auth pattern for keyed endpoints) and
    implements paged retrieval transparently for the two area-search endpoints,
    which return ``nextPageToken`` for large regions.
    """

    REGION_KENYA = "KE"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        config: Config | None = None,
        session: requests.Session | None = None,
        max_retries: int = 3,
        retry_backoff_s: float = 2.0,
    ) -> None:
        cfg = config or get_config()
        self._cfg = cfg
        self._base_url = (base_url or cfg.floodhub_base_url).rstrip("/")
        self._api_key = api_key or cfg.floodhub_api_key
        self._session = session or requests.Session()
        self._max_retries = max_retries
        self._backoff = retry_backoff_s
        self._headers = {
            "User-Agent": cfg.user_agent,
            "Accept": "application/json",
        }

    # -- low-level HTTP -----------------------------------------------------

    def _require_key(self) -> str:
        if not self._api_key:
            raise FloodHubError(
                "FLOODHUB_API_KEY is not set; cannot call Flood Hub. "
                "Populate it in .env or pass api_key= explicitly."
            )
        return self._api_key

    def _params(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"key": self._require_key()}
        if extra:
            for k, v in extra.items():
                if v is None:
                    continue
                if isinstance(v, bool):
                    params[k] = "true" if v else "false"
                else:
                    params[k] = v
        return params

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}/{path.lstrip('/')}"
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                logger.debug("Flood Hub %s %s (attempt %d)", method, path, attempt)
                resp = self._session.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    headers=self._headers,
                    timeout=self._cfg.request_timeout_s,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"Flood Hub HTTP {resp.status_code}: {resp.text[:200]}")
                if resp.status_code >= 400:
                    raise FloodHubError(
                        f"Flood Hub HTTP {resp.status_code}: {resp.text[:500]}"
                    )
                return resp.json() if resp.text else {}
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                wait = self._backoff * attempt
                logger.warning(
                    "Flood Hub call failed (attempt %d/%d): %s. Sleeping %.1fs",
                    attempt,
                    self._max_retries,
                    exc,
                    wait,
                )
                if attempt < self._max_retries:
                    time.sleep(wait)
        assert last_exc is not None
        raise FloodHubError(f"Flood Hub failed after {self._max_retries} attempts") from last_exc

    # -- public endpoints ---------------------------------------------------

    def search_gauges_by_area(
        self,
        *,
        region_code: str | None = REGION_KENYA,
        polygon: list[tuple[float, float]] | None = None,
        include_non_quality_verified: bool = True,
        page_size: int = 500,
    ) -> list[Gauge]:
        """Discover gauges by region code or polygon.

        Either ``region_code`` (e.g. ``"KE"``) or a ``polygon`` of
        ``(lng, lat)`` vertices must be supplied. Pages are concatenated
        transparently.
        """
        if region_code is None and not polygon:
            raise ValueError("provide region_code or polygon")

        body: dict[str, Any] = {
            "includeNonQualityVerified": include_non_quality_verified,
            "pageSize": page_size,
        }
        if region_code:
            body["regionCode"] = region_code
        if polygon:
            body["polygon"] = {
                "vertices": [{"longitude": x, "latitude": y} for (x, y) in polygon]
            }

        gauges: list[Gauge] = []
        next_token: str | None = None
        while True:
            payload = dict(body)
            if next_token:
                payload["pageToken"] = next_token
            resp = self._request(
                "POST",
                "gauges:searchGaugesByArea",
                params=self._params(),
                json=payload,
            )
            for raw in resp.get("gauges", []) or []:
                g = _parse_gauge(raw)
                if g is not None:
                    gauges.append(g)
            next_token = resp.get("nextPageToken") or None
            if not next_token:
                break
        return gauges

    def search_latest_flood_status_by_area(
        self,
        *,
        region_code: str | None = REGION_KENYA,
        polygon: list[tuple[float, float]] | None = None,
        include_non_quality_verified: bool = True,
        page_size: int = 500,
    ) -> list[FloodStatus]:
        """Current severity for all gauges in a region (paginated)."""
        if region_code is None and not polygon:
            raise ValueError("provide region_code or polygon")
        body: dict[str, Any] = {
            "includeNonQualityVerified": include_non_quality_verified,
            "pageSize": page_size,
        }
        if region_code:
            body["regionCode"] = region_code
        if polygon:
            body["polygon"] = {
                "vertices": [{"longitude": x, "latitude": y} for (x, y) in polygon]
            }

        statuses: list[FloodStatus] = []
        next_token: str | None = None
        while True:
            payload = dict(body)
            if next_token:
                payload["pageToken"] = next_token
            resp = self._request(
                "POST",
                "floodStatus:searchLatestFloodStatusByArea",
                params=self._params(),
                json=payload,
            )
            for raw in resp.get("floodStatuses", []) or []:
                fs = _parse_status(raw)
                if fs is not None:
                    statuses.append(fs)
            next_token = resp.get("nextPageToken") or None
            if not next_token:
                break
        return statuses

    def query_latest_flood_status_by_gauge_ids(
        self, gauge_ids: Iterable[str]
    ) -> list[FloodStatus]:
        ids = list(dict.fromkeys(str(g) for g in gauge_ids))
        if not ids:
            return []
        params = self._params({"gaugeIds": ids})
        resp = self._request(
            "GET",
            "floodStatus:queryLatestFloodStatusByGaugeIds",
            params=params,
        )
        out: list[FloodStatus] = []
        for raw in resp.get("floodStatuses", []) or []:
            fs = _parse_status(raw)
            if fs is not None:
                out.append(fs)
        return out

    def query_gauge_forecasts(
        self,
        gauge_ids: Iterable[str],
        *,
        issued_time: str | None = None,
    ) -> dict[str, Any]:
        """Return the raw forecast JSON for the supplied gauges."""
        ids = list(dict.fromkeys(str(g) for g in gauge_ids))
        if not ids:
            return {"forecasts": []}
        params = self._params({"gaugeIds": ids, "issuedTime": issued_time})
        return self._request("GET", "gauges:queryGaugeForecasts", params=params)

    def batch_get_gauge_models(self, gauge_ids: Iterable[str]) -> dict[str, Any]:
        ids = list(dict.fromkeys(str(g) for g in gauge_ids))
        if not ids:
            return {"gaugeModels": []}
        params = self._params({"gaugeIds": ids})
        return self._request("GET", "gaugeModels:batchGet", params=params)

    def batch_get_gauges(self, gauge_ids: Iterable[str]) -> list[Gauge]:
        ids = list(dict.fromkeys(str(g) for g in gauge_ids))
        if not ids:
            return []
        params = self._params({"gaugeIds": ids})
        resp = self._request("GET", "gauges:batchGet", params=params)
        return [g for g in (_parse_gauge(r) for r in resp.get("gauges", []) or []) if g]


# ---------------------------------------------------------------------------
# Validation oracle: pandas/geopandas adapters
# ---------------------------------------------------------------------------


def gauges_to_geodataframe(
    gauges: Iterable[Gauge], *, crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """Convert a sequence of :class:`Gauge` objects into a GeoDataFrame."""
    rows: list[dict[str, Any]] = []
    geoms: list[Point] = []
    for g in gauges:
        rows.append(
            {
                "gauge_id": g.gauge_id,
                "site_name": g.site_name,
                "river": g.river,
                "country_code": g.country_code,
                "quality_verified": g.quality_verified,
                "source": g.source,
                "latitude": g.latitude,
                "longitude": g.longitude,
            }
        )
        geoms.append(g.to_point())
    if not rows:
        return gpd.GeoDataFrame(
            {
                "gauge_id": pd.Series(dtype=str),
                "site_name": pd.Series(dtype=object),
                "river": pd.Series(dtype=object),
                "country_code": pd.Series(dtype=object),
                "quality_verified": pd.Series(dtype=bool),
                "source": pd.Series(dtype=object),
                "latitude": pd.Series(dtype=float),
                "longitude": pd.Series(dtype=float),
            },
            geometry=gpd.GeoSeries([], crs=crs),
            crs=crs,
        )
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=crs)


def statuses_to_dataframe(statuses: Iterable[FloodStatus]) -> pd.DataFrame:
    """Tabular view of flood statuses with the integer-ordinal column."""
    rows = [
        {
            "gauge_id": s.gauge_id,
            "severity": s.severity.name,
            "severity_int": int(s.severity),
            "issued_time": s.issued_time,
            "forecast_change": s.forecast_change,
        }
        for s in statuses
    ]
    df = pd.DataFrame(
        rows,
        columns=["gauge_id", "severity", "severity_int", "issued_time", "forecast_change"],
    )
    if df.empty:
        df["severity_int"] = df["severity_int"].astype(int)
    return df


def join_gauges_with_status(
    gauges: gpd.GeoDataFrame, statuses: pd.DataFrame
) -> gpd.GeoDataFrame:
    """Left-join gauges with their latest severity row.

    Gauges with no status row receive ``severity='UNKNOWN'``, ``severity_int=0``.
    """
    if statuses.empty:
        out = gauges.copy()
        out["severity"] = "UNKNOWN"
        out["severity_int"] = 0
        out["issued_time"] = None
        out["forecast_change"] = None
        return out
    merged = gauges.merge(statuses, on="gauge_id", how="left")
    merged["severity"] = merged["severity"].fillna("UNKNOWN")
    merged["severity_int"] = merged["severity_int"].fillna(0).astype(int)
    return gpd.GeoDataFrame(merged, geometry=gauges.geometry.name, crs=gauges.crs)


# ---------------------------------------------------------------------------
# Internal: response parsers
# ---------------------------------------------------------------------------


def _parse_gauge(raw: dict[str, Any]) -> Gauge | None:
    """Coerce a single Flood Hub gauge payload into a :class:`Gauge`."""
    gid = raw.get("gaugeId") or raw.get("id")
    location = raw.get("location") or {}
    lat = _safe_float(raw.get("latitude") or location.get("latitude"))
    lon = _safe_float(raw.get("longitude") or location.get("longitude"))
    if gid is None or lat is None or lon is None:
        return None
    return Gauge(
        gauge_id=str(gid),
        latitude=lat,
        longitude=lon,
        site_name=raw.get("siteName") or raw.get("name"),
        river=raw.get("river"),
        country_code=raw.get("countryCode") or raw.get("country"),
        quality_verified=bool(raw.get("qualityVerified", False)),
        source=raw.get("source"),
    )


def _parse_status(raw: dict[str, Any]) -> FloodStatus | None:
    gid = raw.get("gaugeId") or raw.get("id")
    if gid is None:
        return None
    severity_raw = raw.get("severity") or raw.get("severityLevel")
    return FloodStatus(
        gauge_id=str(gid),
        severity=Severity.from_api(severity_raw),
        issued_time=raw.get("issuedTime") or raw.get("forecastTimestamp"),
        forecast_change=raw.get("forecastChange") or raw.get("forecastTrend"),
        raw=raw,
    )


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "FloodHubClient",
    "FloodHubError",
    "FloodStatus",
    "Gauge",
    "Severity",
    "gauges_to_geodataframe",
    "join_gauges_with_status",
    "statuses_to_dataframe",
]
