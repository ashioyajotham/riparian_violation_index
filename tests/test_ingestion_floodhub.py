"""Tests for ``rvi.ingestion.floodhub`` (HTTP fully mocked)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from rvi.config import Config
from rvi.ingestion.floodhub import (
    FloodHubClient,
    FloodHubError,
    FloodStatus,
    Gauge,
    Severity,
    gauges_to_geodataframe,
    join_gauges_with_status,
    statuses_to_dataframe,
)

# ---------------------------------------------------------------------------
# Severity IntEnum
# ---------------------------------------------------------------------------


def test_severity_intenum_ordinals_match_proposal() -> None:
    assert int(Severity.UNKNOWN) == 0
    assert int(Severity.NO_FLOODING) == 1
    assert int(Severity.ABOVE_NORMAL) == 2
    assert int(Severity.SEVERE) == 3
    assert int(Severity.EXTREME) == 4


def test_severity_from_api_handles_strings_ints_and_none() -> None:
    assert Severity.from_api("EXTREME") is Severity.EXTREME
    assert Severity.from_api("severe") is Severity.SEVERE
    assert Severity.from_api("SEVERITY_UNSPECIFIED") is Severity.UNKNOWN
    assert Severity.from_api("not-a-real-value") is Severity.UNKNOWN
    assert Severity.from_api(None) is Severity.UNKNOWN
    assert Severity.from_api(3) is Severity.SEVERE
    assert Severity.from_api(99) is Severity.UNKNOWN


def test_severity_supports_arithmetic_ordering() -> None:
    # IntEnum semantics: each member is a real int.
    assert Severity.SEVERE > Severity.NO_FLOODING
    assert int(Severity.EXTREME) == Severity.SEVERE + 1
    assert sum([Severity.NO_FLOODING, Severity.EXTREME]) == 5


# ---------------------------------------------------------------------------
# FloodHubClient — mocked transport
# ---------------------------------------------------------------------------


def _mock_response(payload: dict[str, Any], status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = "ok"
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _client(cfg: Config, session: MagicMock) -> FloodHubClient:
    cfg = Config(**{**cfg.__dict__, "floodhub_api_key": "test-key"})
    return FloodHubClient(api_key="test-key", config=cfg, session=session, max_retries=2, retry_backoff_s=0.0)


def test_search_gauges_by_area_paginates_transparently(cfg: Config) -> None:
    page1 = {
        "gauges": [
            {"gaugeId": "g1", "latitude": -1.28, "longitude": 36.83, "qualityVerified": True},
        ],
        "nextPageToken": "abc",
    }
    page2 = {
        "gauges": [
            {"gaugeId": "g2", "latitude": -0.5, "longitude": 37.0, "qualityVerified": False},
        ],
    }
    session = MagicMock()
    session.request.side_effect = [_mock_response(page1), _mock_response(page2)]
    cli = _client(cfg, session)

    gauges = cli.search_gauges_by_area(region_code="KE")
    assert {g.gauge_id for g in gauges} == {"g1", "g2"}
    assert session.request.call_count == 2

    # The pageToken must reach the second call body.
    second_call_kwargs = session.request.call_args_list[1].kwargs
    assert second_call_kwargs["json"]["pageToken"] == "abc"


def test_search_latest_flood_status_by_area_parses_severity(cfg: Config) -> None:
    payload = {
        "floodStatuses": [
            {"gaugeId": "g1", "severity": "SEVERE", "issuedTime": "2026-04-30T08:00:00Z"},
            {"gaugeId": "g2", "severity": "NO_FLOODING"},
        ]
    }
    session = MagicMock()
    session.request.return_value = _mock_response(payload)
    cli = _client(cfg, session)

    statuses = cli.search_latest_flood_status_by_area(region_code="KE")
    assert {s.gauge_id for s in statuses} == {"g1", "g2"}
    sev = {s.gauge_id: s.severity for s in statuses}
    assert sev["g1"] is Severity.SEVERE
    assert sev["g2"] is Severity.NO_FLOODING


def test_query_latest_flood_status_by_gauge_ids_dedupes(cfg: Config) -> None:
    session = MagicMock()
    session.request.return_value = _mock_response(
        {"floodStatuses": [{"gaugeId": "g1", "severity": "EXTREME"}]}
    )
    cli = _client(cfg, session)
    out = cli.query_latest_flood_status_by_gauge_ids(["g1", "g1", "g1"])
    assert len(out) == 1
    params = session.request.call_args.kwargs["params"]
    assert params["gaugeIds"] == ["g1"]


def test_no_api_key_raises_helpful_error(cfg: Config) -> None:
    cli = FloodHubClient(api_key=None, config=cfg, session=MagicMock())
    with pytest.raises(FloodHubError, match="FLOODHUB_API_KEY"):
        cli.search_gauges_by_area(region_code="KE")


def test_search_requires_region_or_polygon(cfg: Config) -> None:
    cli = _client(cfg, MagicMock())
    with pytest.raises(ValueError):
        cli.search_gauges_by_area(region_code=None, polygon=None)


def test_500_then_200_succeeds_via_retry(cfg: Config) -> None:
    bad = MagicMock()
    bad.status_code = 503
    bad.text = "unavailable"
    good = _mock_response({"gauges": []})
    session = MagicMock()
    session.request.side_effect = [bad, good]
    cli = _client(cfg, session)
    out = cli.search_gauges_by_area(region_code="KE")
    assert out == []
    assert session.request.call_count == 2


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def test_gauges_to_geodataframe_round_trip() -> None:
    gauges = [
        Gauge(gauge_id="g1", latitude=-1.28, longitude=36.83, quality_verified=True),
        Gauge(gauge_id="g2", latitude=-0.5, longitude=37.0, quality_verified=False),
    ]
    gdf = gauges_to_geodataframe(gauges)
    assert list(gdf["gauge_id"]) == ["g1", "g2"]
    assert gdf.geometry.iloc[0].x == pytest.approx(36.83)
    assert gdf.geometry.iloc[0].y == pytest.approx(-1.28)
    assert str(gdf.crs) == "EPSG:4326"


def test_statuses_to_dataframe_includes_int_column() -> None:
    statuses = [
        FloodStatus(gauge_id="g1", severity=Severity.SEVERE),
        FloodStatus(gauge_id="g2", severity=Severity.NO_FLOODING),
    ]
    df = statuses_to_dataframe(statuses)
    assert list(df["severity_int"]) == [3, 1]
    assert list(df["severity"]) == ["SEVERE", "NO_FLOODING"]


def test_join_gauges_with_status_handles_unknown_gauges() -> None:
    gauges = gauges_to_geodataframe(
        [
            Gauge(gauge_id="g1", latitude=-1.28, longitude=36.83),
            Gauge(gauge_id="g2", latitude=-0.5, longitude=37.0),
            Gauge(gauge_id="g3", latitude=0.5, longitude=37.5),  # no status
        ]
    )
    statuses = statuses_to_dataframe(
        [
            FloodStatus(gauge_id="g1", severity=Severity.SEVERE),
            FloodStatus(gauge_id="g2", severity=Severity.NO_FLOODING),
        ]
    )
    merged = join_gauges_with_status(gauges, statuses)
    g3 = merged[merged["gauge_id"] == "g3"].iloc[0]
    assert g3["severity"] == "UNKNOWN"
    assert int(g3["severity_int"]) == 0
