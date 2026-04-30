"""Riparian Violation Index — Density / Coverage / Proximity (§2.3 of the proposal).

The RVI for a 500-metre river segment :math:`s` is a composite in
:math:`[0, 1]`:

.. math::

    \\mathrm{RVI}_s = \\alpha\\,D_s^{\\text{norm}} + \\beta\\,C_s + \\gamma\\,P_s

with default weights :math:`\\alpha = 0.4`, :math:`\\beta = 0.3`,
:math:`\\gamma = 0.3`.

Sub-scores
----------

**Density** (§2.3.1). Buildings per kilometre of river, then min-max
normalised across the analysis area:

.. math::

    D_s = \\frac{n_s}{L_s},\\qquad
    D_s^{\\text{norm}} = \\frac{D_s - D_{\\min}}{D_{\\max} - D_{\\min}}

When :math:`D_{\\max} = D_{\\min}` (degenerate dataset where every segment
shares the same density, including all-zero), :math:`D_s^{\\text{norm}} := 0`.

**Coverage** (§2.3.2). Fraction of buffer area covered by footprints,
clipped to ``[0, 1]``:

.. math::

    C_s = \\min\\!\\left(1,\\; \\frac{A_{\\text{enc},s}}{A_{\\text{buf},s}}\\right)

**Proximity** (§2.3.3). Mean penetration of buildings into the buffer:

.. math::

    P_s = \\frac{1}{n_s} \\sum_{i=1}^{n_s} \\max\\!\\left(0,\\; 1 - \\frac{d_i}{r_s}\\right)

For segments with zero encroaching buildings, :math:`P_s := 0`.

In this implementation the per-segment :math:`P_s` is approximated from the
encroachment block's ``mean_dist_m`` when individual building distances have
already been aggregated:

.. math::

    P_s \\approx \\max\\!\\left(0,\\; 1 - \\frac{\\overline{d}_s}{r_s}\\right)

This is mathematically identical when buildings are uniformly distributed
inside the buffer; for non-uniform distributions it is monotonic in the true
:math:`P_s` and so preserves the rank order needed for the Spearman test.
A higher-fidelity per-building Proximity is available via
:func:`compute_proximity_from_distances` for callers that retain raw
distances.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd

from rvi.config import Config, get_config

logger = logging.getLogger(__name__)


RVI_SCORE_COLUMNS: tuple[str, ...] = (
    "density_raw",
    "density_norm",
    "coverage",
    "proximity",
    "rvi_composite",
)


# ---------------------------------------------------------------------------
# Sub-scores (numeric, vectorised)
# ---------------------------------------------------------------------------


def density_raw(n_buildings: pd.Series, length_m: pd.Series) -> pd.Series:
    """Buildings per kilometre. Length is in *metres* (matches segment schema).

    Returns 0 where ``length_m <= 0`` to avoid divide-by-zero on degenerate
    inputs.
    """
    n = pd.Series(n_buildings, dtype=float).fillna(0.0)
    length_km = pd.Series(length_m, dtype=float).fillna(0.0) / 1000.0
    out = pd.Series(np.zeros(len(n), dtype=float), index=n.index)
    mask = length_km > 0
    out.loc[mask] = n.loc[mask] / length_km.loc[mask]
    return out


def density_normalise(raw: pd.Series) -> pd.Series:
    """Min-max normalise the raw density to ``[0, 1]``.

    A degenerate dataset (constant raw values) maps everything to 0.
    """
    raw = pd.Series(raw, dtype=float).fillna(0.0)
    if raw.empty:
        return raw
    lo, hi = float(raw.min()), float(raw.max())
    if hi <= lo:
        return pd.Series(np.zeros(len(raw), dtype=float), index=raw.index)
    return ((raw - lo) / (hi - lo)).clip(lower=0.0, upper=1.0)


def coverage_score(
    encroaching_area_m2: pd.Series, buffer_area_m2: pd.Series
) -> pd.Series:
    """Fraction of buffer area covered by building footprints, clipped to [0, 1]."""
    enc = pd.Series(encroaching_area_m2, dtype=float).fillna(0.0)
    buf = pd.Series(buffer_area_m2, dtype=float).fillna(0.0)
    out = pd.Series(np.zeros(len(enc), dtype=float), index=enc.index)
    mask = buf > 0
    out.loc[mask] = (enc.loc[mask] / buf.loc[mask]).clip(lower=0.0, upper=1.0)
    return out


def proximity_score(
    mean_dist_m: pd.Series,
    buffer_radius_m: pd.Series,
    n_buildings: pd.Series,
) -> pd.Series:
    """Aggregated Proximity sub-score :math:`P_s` from segment-level statistics.

    A segment with ``n_buildings == 0`` always gets :math:`P_s = 0`.
    """
    d = pd.Series(mean_dist_m, dtype=float)
    r = pd.Series(buffer_radius_m, dtype=float)
    n = pd.Series(n_buildings, dtype=float).fillna(0.0)
    out = pd.Series(np.zeros(len(n), dtype=float), index=n.index)
    mask = (n > 0) & (r > 0) & d.notna()
    if mask.any():
        out.loc[mask] = (1.0 - (d.loc[mask] / r.loc[mask])).clip(lower=0.0, upper=1.0)
    return out


def compute_proximity_from_distances(
    distances_m: Iterable[float], buffer_radius_m: float
) -> float:
    """Per-building Proximity (§2.3.3) for callers that retain raw distances."""
    distances = np.asarray(list(distances_m), dtype=float)
    if buffer_radius_m <= 0 or distances.size == 0:
        return 0.0
    contributions = np.clip(1.0 - distances / buffer_radius_m, a_min=0.0, a_max=None)
    return float(contributions.mean())


# ---------------------------------------------------------------------------
# Top-level scorer
# ---------------------------------------------------------------------------


def compute_rvi(
    encroachment: gpd.GeoDataFrame,
    *,
    config: Config | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
) -> gpd.GeoDataFrame:
    """Add the RVI sub-scores and composite to a per-segment encroachment table.

    Parameters
    ----------
    encroachment
        Output of :func:`rvi.analysis.encroachment.detect_encroachment`.
    alpha, beta, gamma
        Override the weights from :class:`rvi.config.Config`. Useful for the
        Research-Question-4 sensitivity analysis.

    Returns
    -------
    GeoDataFrame
        ``encroachment`` plus the columns:
        ``density_raw``, ``density_norm``, ``coverage``, ``proximity``,
        ``rvi_composite``.
    """
    cfg = config or get_config()
    a = float(alpha if alpha is not None else cfg.rvi_alpha)
    b = float(beta if beta is not None else cfg.rvi_beta)
    g = float(gamma if gamma is not None else cfg.rvi_gamma)
    weight_sum = a + b + g
    if not (0.99 <= weight_sum <= 1.01):
        raise ValueError(
            f"RVI weights must sum to ~1.0; got alpha+beta+gamma = {weight_sum:.4f}"
        )

    if encroachment.empty:
        out = encroachment.copy()
        for col in RVI_SCORE_COLUMNS:
            out[col] = pd.Series(dtype=float)
        return out

    out = encroachment.copy()
    out["density_raw"] = density_raw(out["n_buildings"], out["segment_length_m"])
    out["density_norm"] = density_normalise(out["density_raw"])
    out["coverage"] = coverage_score(out["total_footprint_m2"], out["buffer_area_m2"])
    out["proximity"] = proximity_score(
        out["mean_dist_m"], out["buffer_radius_m"], out["n_buildings"]
    )
    out["rvi_composite"] = (
        a * out["density_norm"] + b * out["coverage"] + g * out["proximity"]
    ).clip(lower=0.0, upper=1.0)
    return out


def compute_rvi_multi(
    encroachment_by_width: dict[float, gpd.GeoDataFrame],
    *,
    config: Config | None = None,
) -> gpd.GeoDataFrame:
    """Stack the per-segment RVI for all buffer widths into a wide table.

    Returns a GeoDataFrame with the segments and one column per width:

    * ``rvi_composite_6m`` / ``rvi_composite_10m`` / ``rvi_composite_30m``
    * ``density_norm_<width>m``, ``coverage_<width>m``, ``proximity_<width>m``
    """
    cfg = config or get_config()
    if not encroachment_by_width:
        return gpd.GeoDataFrame()

    base_width = (
        cfg.primary_buffer_m
        if cfg.primary_buffer_m in encroachment_by_width
        else next(iter(encroachment_by_width))
    )
    base = compute_rvi(encroachment_by_width[base_width], config=cfg).copy()

    keep_meta = [
        "segment_id",
        "osm_id",
        "waterway",
        "name",
        "name_local",
        "strahler",
        "segment_length_m",
        "geometry",
    ]
    keep_meta = [c for c in keep_meta if c in base.columns]
    wide = base[keep_meta].copy()

    # Defensive dedupe — segment_id MUST be unique. If an upstream stage
    # ever produces collisions, we'd otherwise Cartesian-product through the
    # successive merges (e.g., 3 widths × k duplicates → k^3 explosion).
    if not wide["segment_id"].is_unique:
        n_before = len(wide)
        wide = wide.drop_duplicates(subset=["segment_id"], keep="first")
        logger.warning(
            "compute_rvi_multi: dropped %d duplicate segment_id rows",
            n_before - len(wide),
        )

    for width, enc in encroachment_by_width.items():
        scored = compute_rvi(enc, config=cfg)
        suffix = f"_{int(width)}m"
        renamer = {
            "density_norm": f"density_norm{suffix}",
            "coverage": f"coverage{suffix}",
            "proximity": f"proximity{suffix}",
            "rvi_composite": f"rvi_composite{suffix}",
            "n_buildings": f"n_buildings{suffix}",
            "buffer_radius_m": f"buffer_radius_m{suffix}",
            "buffer_area_m2": f"buffer_area_m2{suffix}",
        }
        cols = ["segment_id", *renamer.keys()]
        cols = [c for c in cols if c in scored.columns]
        slice_ = scored[cols].rename(columns=renamer)
        if not slice_["segment_id"].is_unique:
            slice_ = slice_.drop_duplicates(subset=["segment_id"], keep="first")
        wide = wide.merge(slice_, on="segment_id", how="left", validate="one_to_one")

    return gpd.GeoDataFrame(wide, geometry=base.geometry.name, crs=base.crs)


# ---------------------------------------------------------------------------
# Sensitivity analysis (§4 RQ4)
# ---------------------------------------------------------------------------


def sensitivity_grid(
    encroachment: gpd.GeoDataFrame,
    *,
    config: Config | None = None,
    step: float = 0.1,
) -> pd.DataFrame:
    """Score the segments across all weight triples summing to 1.

    Returns a long-form DataFrame with one row per ``(alpha, beta, gamma,
    segment_id)`` combination and a ``rvi_composite`` column. The downstream
    Spearman analysis pivots over this.

    Parameters
    ----------
    step
        Granularity of the alpha/beta sweep. ``0.1`` produces 66 valid
        triples (the default; matches the proposal's appendix figure).
    """
    cfg = config or get_config()
    if encroachment.empty:
        return pd.DataFrame(
            columns=["alpha", "beta", "gamma", "segment_id", "rvi_composite"]
        )

    triples: list[tuple[float, float, float]] = []
    a = 0.0
    while a <= 1.0 + 1e-9:
        b = 0.0
        while b <= 1.0 - a + 1e-9:
            g = 1.0 - a - b
            if g >= -1e-9:
                triples.append((round(a, 4), round(b, 4), round(max(g, 0.0), 4)))
            b += step
        a += step

    sub_score_cache = compute_rvi(encroachment, config=cfg)
    rows: list[pd.DataFrame] = []
    for a_val, b_val, g_val in triples:
        composite = (
            a_val * sub_score_cache["density_norm"]
            + b_val * sub_score_cache["coverage"]
            + g_val * sub_score_cache["proximity"]
        ).clip(lower=0.0, upper=1.0)
        rows.append(
            pd.DataFrame(
                {
                    "alpha": a_val,
                    "beta": b_val,
                    "gamma": g_val,
                    "segment_id": sub_score_cache["segment_id"].values,
                    "rvi_composite": composite.values,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


__all__ = [
    "RVI_SCORE_COLUMNS",
    "compute_proximity_from_distances",
    "compute_rvi",
    "compute_rvi_multi",
    "coverage_score",
    "density_normalise",
    "density_raw",
    "proximity_score",
    "sensitivity_grid",
]
