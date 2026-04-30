"""Validation oracle: pairing upstream RVI with Flood Hub severity (§5.5).

For each Flood Hub gauge, this module:

1. Identifies all RVI-scored river segments within a 50-km Euclidean radius
   (Phase 1 approximation; see §5.5). Phase 2 — true catchment polygons via
   ``pysheds`` — has a separate function with the same signature so callers
   can swap implementations transparently.
2. Aggregates the segments' RVI scores into ``mean``, ``max``, and 75th
   percentile statistics.
3. Pairs those statistics with the gauge's :class:`Severity` integer.
4. Computes Spearman's rank correlation with a non-parametric bootstrap
   confidence interval.

The bootstrap is implemented locally rather than via ``scipy.stats.bootstrap``
to keep the dependency surface predictable across SciPy versions.

Statistical outputs
-------------------
:class:`SpearmanResult` is the structured return of :func:`spearman_with_ci`;
its ``rho``, ``pvalue``, ``ci_low``, ``ci_high``, ``n`` fields are sufficient
for citation.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats

from rvi.config import Config, get_config

logger = logging.getLogger(__name__)


UPSTREAM_COLUMNS: tuple[str, ...] = (
    "gauge_id",
    "n_upstream_segments",
    "upstream_rvi_mean",
    "upstream_rvi_max",
    "upstream_rvi_p75",
    "severity",
    "severity_int",
    "quality_verified",
)


# ---------------------------------------------------------------------------
# Phase 1: Euclidean upstream
# ---------------------------------------------------------------------------


def aggregate_upstream_euclidean(
    segments_with_rvi: gpd.GeoDataFrame,
    gauges: gpd.GeoDataFrame,
    *,
    config: Config | None = None,
    radius_m: float | None = None,
    rvi_column: str = "rvi_composite",
) -> pd.DataFrame:
    """For each gauge, aggregate the RVI of segments within *radius_m* metres.

    Both inputs are reprojected to :data:`Config.crs_metric` internally.

    Parameters
    ----------
    segments_with_rvi
        Output of :func:`rvi.analysis.rvi.compute_rvi`. Must carry
        ``rvi_column`` (default ``rvi_composite``) plus a geometry column.
    gauges
        Output of :func:`rvi.ingestion.floodhub.gauges_to_geodataframe` joined
        with status (i.e., a ``severity_int`` column is expected).

    Returns
    -------
    DataFrame with the columns listed in :data:`UPSTREAM_COLUMNS`. Gauges with
    zero upstream segments are still represented; their RVI aggregates are
    ``NaN``.
    """
    cfg = config or get_config()
    radius = float(radius_m or cfg.upstream_radius_m)

    if segments_with_rvi.empty or gauges.empty:
        return pd.DataFrame(columns=list(UPSTREAM_COLUMNS))

    if rvi_column not in segments_with_rvi.columns:
        raise ValueError(
            f"segments_with_rvi must contain '{rvi_column}'; got {list(segments_with_rvi.columns)}"
        )

    seg = (
        segments_with_rvi.to_crs(cfg.crs_metric)
        if segments_with_rvi.crs and segments_with_rvi.crs != cfg.crs_metric
        else segments_with_rvi.copy()
    )
    if seg.crs is None:
        seg = seg.set_crs(cfg.crs_metric)

    gau = (
        gauges.to_crs(cfg.crs_metric)
        if gauges.crs and gauges.crs != cfg.crs_metric
        else gauges.copy()
    )
    if gau.crs is None:
        gau = gau.set_crs(cfg.crs_metric)

    # Buffer each gauge by `radius` and spatially join segments into them.
    gau = gau.copy()
    gau["_gauge_buffer"] = gau.geometry.buffer(radius)
    gau_buffers = gpd.GeoDataFrame(
        gau.drop(columns=["geometry"]).rename(columns={"_gauge_buffer": "geometry"}),
        geometry="geometry",
        crs=cfg.crs_metric,
    )

    # Use segment representative_points for the join — fast & deterministic.
    seg_points = seg.copy()
    seg_points["geometry"] = seg.geometry.representative_point()
    seg_points = gpd.GeoDataFrame(seg_points, geometry="geometry", crs=cfg.crs_metric)

    joined = gpd.sjoin(
        seg_points[["segment_id", rvi_column, "geometry"]],
        gau_buffers[["gauge_id", "geometry"]],
        how="inner",
        predicate="within",
    )

    if joined.empty:
        agg = pd.DataFrame(
            columns=[
                "gauge_id",
                "n_upstream_segments",
                "upstream_rvi_mean",
                "upstream_rvi_max",
                "upstream_rvi_p75",
            ]
        )
    else:
        agg = (
            joined.groupby("gauge_id", sort=False)[rvi_column]
            .agg(
                n_upstream_segments="count",
                upstream_rvi_mean="mean",
                upstream_rvi_max="max",
                upstream_rvi_p75=lambda s: float(np.nanpercentile(s, 75)),
            )
            .reset_index()
        )

    base = gau[
        [c for c in ("gauge_id", "severity", "severity_int", "quality_verified") if c in gau.columns]
    ].copy()
    if "severity_int" not in base.columns:
        base["severity_int"] = 0
    if "severity" not in base.columns:
        base["severity"] = "UNKNOWN"
    if "quality_verified" not in base.columns:
        base["quality_verified"] = False

    out = base.merge(agg, on="gauge_id", how="left")
    out["n_upstream_segments"] = out["n_upstream_segments"].fillna(0).astype(int)
    for c in ("upstream_rvi_mean", "upstream_rvi_max", "upstream_rvi_p75"):
        if c not in out.columns:
            out[c] = np.nan
    return out[list(UPSTREAM_COLUMNS)]


# ---------------------------------------------------------------------------
# Phase 2: hydrologic catchment (DEM-driven)
# ---------------------------------------------------------------------------


def aggregate_upstream_catchment(
    segments_with_rvi: gpd.GeoDataFrame,
    gauges: gpd.GeoDataFrame,
    *,
    catchments: gpd.GeoDataFrame,
    rvi_column: str = "rvi_composite",
    config: Config | None = None,
) -> pd.DataFrame:
    """Phase-2 upstream aggregation using pre-computed catchment polygons.

    Parameters
    ----------
    catchments
        GeoDataFrame with one Polygon row per gauge. Must have ``gauge_id``.
        Production code obtains this from :mod:`pysheds` flow-direction +
        flow-accumulation rasters; this function is decoupled from that step
        so the rest of the pipeline can be unit-tested with a synthetic
        catchment fixture.
    """
    cfg = config or get_config()
    if catchments.empty or segments_with_rvi.empty:
        return pd.DataFrame(columns=list(UPSTREAM_COLUMNS))

    seg = segments_with_rvi.to_crs(cfg.crs_metric)
    cat = catchments.to_crs(cfg.crs_metric)

    seg_points = seg.copy()
    seg_points["geometry"] = seg.geometry.representative_point()
    seg_points = gpd.GeoDataFrame(seg_points, geometry="geometry", crs=cfg.crs_metric)

    joined = gpd.sjoin(
        seg_points[["segment_id", rvi_column, "geometry"]],
        cat[["gauge_id", "geometry"]],
        how="inner",
        predicate="within",
    )

    if joined.empty:
        agg = pd.DataFrame(
            columns=[
                "gauge_id",
                "n_upstream_segments",
                "upstream_rvi_mean",
                "upstream_rvi_max",
                "upstream_rvi_p75",
            ]
        )
    else:
        agg = (
            joined.groupby("gauge_id", sort=False)[rvi_column]
            .agg(
                n_upstream_segments="count",
                upstream_rvi_mean="mean",
                upstream_rvi_max="max",
                upstream_rvi_p75=lambda s: float(np.nanpercentile(s, 75)),
            )
            .reset_index()
        )

    base_cols = [
        c
        for c in ("gauge_id", "severity", "severity_int", "quality_verified")
        if c in gauges.columns
    ]
    base = gauges[base_cols].copy()
    if "severity_int" not in base.columns:
        base["severity_int"] = 0
    if "severity" not in base.columns:
        base["severity"] = "UNKNOWN"
    if "quality_verified" not in base.columns:
        base["quality_verified"] = False

    out = base.merge(agg, on="gauge_id", how="left")
    out["n_upstream_segments"] = out["n_upstream_segments"].fillna(0).astype(int)
    for c in ("upstream_rvi_mean", "upstream_rvi_max", "upstream_rvi_p75"):
        if c not in out.columns:
            out[c] = np.nan
    return out[list(UPSTREAM_COLUMNS)]


# ---------------------------------------------------------------------------
# Spearman + bootstrap CI
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpearmanResult:
    """Statistical result for one (RVI, severity) pairing."""

    rho: float
    pvalue: float
    ci_low: float
    ci_high: float
    n: int
    method: str = "spearman"

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "rho": self.rho,
            "pvalue": self.pvalue,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "method": self.method,
        }


def spearman_with_ci(
    x: Iterable[float],
    y: Iterable[float],
    *,
    n_bootstrap: int | None = None,
    ci: float | None = None,
    seed: int | None = 12345,
    config: Config | None = None,
) -> SpearmanResult:
    """Spearman rank correlation with non-parametric bootstrap CI.

    NaNs in either array are dropped pairwise before correlation.
    Returns ``rho=nan`` if fewer than 3 valid pairs remain.
    """
    cfg = config or get_config()
    iters = int(n_bootstrap if n_bootstrap is not None else cfg.bootstrap_iterations)
    conf = float(ci if ci is not None else cfg.bootstrap_ci)

    arr_x = np.asarray(list(x), dtype=float)
    arr_y = np.asarray(list(y), dtype=float)
    if arr_x.shape != arr_y.shape:
        raise ValueError(f"x and y shapes differ: {arr_x.shape} vs {arr_y.shape}")
    mask = np.isfinite(arr_x) & np.isfinite(arr_y)
    arr_x, arr_y = arr_x[mask], arr_y[mask]
    n = arr_x.size

    if n < 3:
        return SpearmanResult(
            rho=float("nan"),
            pvalue=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=n,
        )

    rho, pvalue = _scipy_stats.spearmanr(arr_x, arr_y)
    rho = float(rho)
    pvalue = float(pvalue) if pvalue is not None else float("nan")

    rng = np.random.default_rng(seed)
    boots = np.empty(iters, dtype=float)
    indices = np.arange(n)
    for i in range(iters):
        sample = rng.choice(indices, size=n, replace=True)
        bx = arr_x[sample]
        by = arr_y[sample]
        if np.allclose(bx, bx[0]) or np.allclose(by, by[0]):
            boots[i] = np.nan
            continue
        r, _ = _scipy_stats.spearmanr(bx, by)
        boots[i] = float(r) if r is not None else np.nan

    valid = boots[np.isfinite(boots)]
    if valid.size < 10:
        ci_low = ci_high = float("nan")
    else:
        alpha = 1.0 - conf
        ci_low = float(np.quantile(valid, alpha / 2.0))
        ci_high = float(np.quantile(valid, 1.0 - alpha / 2.0))

    return SpearmanResult(
        rho=rho, pvalue=pvalue, ci_low=ci_low, ci_high=ci_high, n=n
    )


def correlate_upstream_rvi_to_severity(
    upstream: pd.DataFrame,
    *,
    rvi_field: str = "upstream_rvi_p75",
    severity_field: str = "severity_int",
    min_pairs: int = 5,
    config: Config | None = None,
) -> SpearmanResult:
    """Convenience wrapper: Spearman ρ between an upstream RVI agg and severity."""
    if rvi_field not in upstream.columns or severity_field not in upstream.columns:
        raise ValueError(
            f"upstream table missing column(s); need {rvi_field!r} and {severity_field!r}"
        )

    df = upstream[[rvi_field, severity_field]].dropna()
    if len(df) < max(3, int(min_pairs)):
        return SpearmanResult(
            rho=float("nan"),
            pvalue=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=len(df),
        )
    return spearman_with_ci(df[rvi_field], df[severity_field], config=config)


def stratified_correlation(
    upstream: pd.DataFrame,
    *,
    rvi_field: str = "upstream_rvi_p75",
    severity_field: str = "severity_int",
    config: Config | None = None,
) -> dict[str, SpearmanResult]:
    """Run :func:`correlate_upstream_rvi_to_severity` separately by gauge tier.

    Returns ``{"all": ..., "quality_verified": ..., "non_quality_verified": ...}``
    matching Research Question 3 in §4.
    """
    out: dict[str, SpearmanResult] = {}
    out["all"] = correlate_upstream_rvi_to_severity(
        upstream, rvi_field=rvi_field, severity_field=severity_field, config=config
    )

    if "quality_verified" not in upstream.columns:
        return out

    qv = upstream[upstream["quality_verified"].astype(bool)]
    nqv = upstream[~upstream["quality_verified"].astype(bool)]
    out["quality_verified"] = correlate_upstream_rvi_to_severity(
        qv, rvi_field=rvi_field, severity_field=severity_field, config=config
    )
    out["non_quality_verified"] = correlate_upstream_rvi_to_severity(
        nqv, rvi_field=rvi_field, severity_field=severity_field, config=config
    )
    return out


__all__ = [
    "UPSTREAM_COLUMNS",
    "SpearmanResult",
    "aggregate_upstream_catchment",
    "aggregate_upstream_euclidean",
    "correlate_upstream_rvi_to_severity",
    "spearman_with_ci",
    "stratified_correlation",
]
