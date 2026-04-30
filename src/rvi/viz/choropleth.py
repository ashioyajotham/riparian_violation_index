"""Visualisation outputs (§5.6 of the proposal).

Three deliverables:

* :func:`segment_map` — Folium leaflet showing every RVI-scored segment as a
  coloured LineString, with a popup of the segment's sub-scores. Used for the
  Nairobi pilot detail map.
* :func:`county_choropleth` — Folium choropleth of mean RVI by county /
  sub-county polygon. Used for the national run.
* :func:`rvi_severity_scatter` — matplotlib scatter of upstream RVI vs
  Flood Hub severity, with annotated Spearman ρ and bootstrap CI.

All Folium outputs return a :class:`folium.Map`; saving is the caller's
responsibility (``m.save("path.html")``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from rvi.analysis.validation import SpearmanResult
from rvi.config import Config, get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Folium maps
# ---------------------------------------------------------------------------


def segment_map(
    segments_with_rvi: gpd.GeoDataFrame,
    *,
    config: Config | None = None,
    rvi_column: str = "rvi_composite",
    tiles: str = "CartoDB positron",
    zoom_start: int = 12,
    line_weight: int = 4,
    title: str | None = None,
) -> Any:
    """Plot per-segment RVI on an interactive Folium map.

    Segments are coloured along a 5-class viridis-like ramp from low (light)
    to high (dark) RVI. The map's centre and zoom default to the bounds of
    the input segments.
    """
    import folium  # imported lazily so config import stays cheap

    cfg = config or get_config()
    if segments_with_rvi.empty:
        return folium.Map(location=[-1.28, 36.82], zoom_start=zoom_start, tiles=tiles)

    geo = (
        segments_with_rvi.to_crs(cfg.crs_geographic)
        if segments_with_rvi.crs and segments_with_rvi.crs != cfg.crs_geographic
        else segments_with_rvi
    )
    bounds = geo.total_bounds
    centre = [(bounds[1] + bounds[3]) / 2.0, (bounds[0] + bounds[2]) / 2.0]
    fmap = folium.Map(location=centre, zoom_start=zoom_start, tiles=tiles, control_scale=True)

    if title:
        title_html = f"""
            <div style="position: fixed; top: 12px; left: 50px; z-index: 9999;
                        background: white; padding: 6px 10px; border: 1px solid #888;
                        border-radius: 4px; font-family: sans-serif; font-size: 14px;">
                <b>{title}</b>
            </div>
        """
        fmap.get_root().html.add_child(folium.Element(title_html))

    bins = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    colour_ramp = ["#fde725", "#7ad151", "#22a884", "#2a788e", "#414487"]

    def _classify(score: float) -> str:
        if not np.isfinite(score):
            return "#bbbbbb"
        for upper, colour in zip(bins[1:], colour_ramp, strict=True):
            if score <= upper:
                return colour
        return colour_ramp[-1]

    for _, row in geo.iterrows():
        score = float(row.get(rvi_column, np.nan))
        colour = _classify(score)
        coords = _line_to_latlon(row.geometry)
        if not coords:
            continue
        popup_html = _popup_html(row, rvi_column)
        folium.PolyLine(
            locations=coords,
            color=colour,
            weight=line_weight,
            opacity=0.85,
            tooltip=f"RVI = {score:.3f}",
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(fmap)

    legend = _build_legend(bins, colour_ramp, rvi_column)
    fmap.get_root().html.add_child(folium.Element(legend))
    return fmap


def county_choropleth(
    counties: gpd.GeoDataFrame,
    aggregated_rvi: pd.DataFrame,
    *,
    config: Config | None = None,
    county_id_column: str = "county",
    rvi_column: str = "rvi_mean",
    tiles: str = "CartoDB positron",
    zoom_start: int = 6,
    legend_name: str = "Mean RVI by county",
) -> Any:
    """Folium choropleth of an aggregated RVI value per administrative polygon.

    Parameters
    ----------
    counties
        GeoDataFrame of polygons keyed by ``county_id_column``.
    aggregated_rvi
        Tabular per-county aggregate, with the same ``county_id_column`` and
        a numeric ``rvi_column``.
    """
    import folium

    cfg = config or get_config()
    if counties.empty:
        return folium.Map(location=[0.5, 37.9], zoom_start=zoom_start, tiles=tiles)

    geo = (
        counties.to_crs(cfg.crs_geographic)
        if counties.crs and counties.crs != cfg.crs_geographic
        else counties
    )
    bounds = geo.total_bounds
    centre = [(bounds[1] + bounds[3]) / 2.0, (bounds[0] + bounds[2]) / 2.0]
    fmap = folium.Map(
        location=centre, zoom_start=zoom_start, tiles=tiles, control_scale=True
    )

    merged = geo.merge(aggregated_rvi, on=county_id_column, how="left")
    folium.Choropleth(
        geo_data=merged.to_json(),
        data=merged,
        columns=[county_id_column, rvi_column],
        key_on=f"feature.properties.{county_id_column}",
        fill_color="YlOrRd",
        fill_opacity=0.75,
        line_opacity=0.4,
        nan_fill_color="#dddddd",
        legend_name=legend_name,
    ).add_to(fmap)

    folium.GeoJson(
        merged.to_json(),
        style_function=lambda _: {"fillOpacity": 0, "color": "#444", "weight": 0.5},
        tooltip=folium.GeoJsonTooltip(
            fields=[county_id_column, rvi_column],
            aliases=["County", legend_name],
            localize=True,
        ),
    ).add_to(fmap)

    return fmap


# ---------------------------------------------------------------------------
# Aggregation helper for the choropleth
# ---------------------------------------------------------------------------


def aggregate_rvi_by_county(
    segments_with_rvi: gpd.GeoDataFrame,
    counties: gpd.GeoDataFrame,
    *,
    config: Config | None = None,
    county_id_column: str = "county",
    rvi_column: str = "rvi_composite",
) -> pd.DataFrame:
    """Mean / max / count of RVI scores per county polygon.

    Uses representative_point of each segment so a segment that crosses a
    county boundary is assigned to the dominant county.
    """
    cfg = config or get_config()
    if segments_with_rvi.empty or counties.empty:
        return pd.DataFrame(
            columns=[county_id_column, "rvi_mean", "rvi_max", "rvi_p75", "n_segments"]
        )
    seg = segments_with_rvi.to_crs(cfg.crs_metric)
    cty = counties.to_crs(cfg.crs_metric)
    if county_id_column not in cty.columns:
        raise ValueError(f"counties is missing column {county_id_column!r}")

    seg_pts = seg.copy()
    seg_pts["geometry"] = seg.geometry.representative_point()
    seg_pts = gpd.GeoDataFrame(seg_pts, geometry="geometry", crs=cfg.crs_metric)

    joined = gpd.sjoin(
        seg_pts[["segment_id", rvi_column, "geometry"]],
        cty[[county_id_column, "geometry"]],
        how="inner",
        predicate="within",
    )
    if joined.empty:
        return pd.DataFrame(
            columns=[county_id_column, "rvi_mean", "rvi_max", "rvi_p75", "n_segments"]
        )

    return (
        joined.groupby(county_id_column, sort=True)[rvi_column]
        .agg(
            rvi_mean="mean",
            rvi_max="max",
            rvi_p75=lambda s: float(np.nanpercentile(s, 75)),
            n_segments="count",
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Matplotlib outputs
# ---------------------------------------------------------------------------


def rvi_severity_scatter(
    upstream: pd.DataFrame,
    *,
    rvi_field: str = "upstream_rvi_p75",
    severity_field: str = "severity_int",
    result: SpearmanResult | None = None,
    title: str | None = None,
    save_path: Path | str | None = None,
) -> Any:
    """Scatter of upstream RVI vs Flood Hub severity with Spearman annotation.

    Returns the matplotlib ``Figure`` so the caller can further customise or
    save it themselves; if ``save_path`` is provided the figure is also
    written to disk.
    """
    import matplotlib.pyplot as plt

    df = upstream[[rvi_field, severity_field]].dropna()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    if df.empty:
        ax.text(0.5, 0.5, "No paired observations", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig

    jittered = df[severity_field] + np.random.default_rng(7).uniform(
        -0.12, 0.12, size=len(df)
    )
    ax.scatter(df[rvi_field], jittered, alpha=0.7, edgecolor="k", linewidths=0.4)
    ax.set_xlabel(f"Upstream RVI ({rvi_field})")
    ax.set_ylabel("Flood Hub severity (ordinal)")
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["UNKNOWN", "NO_FLOODING", "ABOVE_NORMAL", "SEVERE", "EXTREME"])
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)

    if result is not None and np.isfinite(result.rho):
        annotation = (
            f"Spearman ρ = {result.rho:+.3f}\n"
            f"95% CI = [{result.ci_low:+.3f}, {result.ci_high:+.3f}]\n"
            f"p = {result.pvalue:.3g}, n = {result.n}"
        )
        ax.text(
            0.02,
            0.98,
            annotation,
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            bbox={"facecolor": "white", "edgecolor": "0.5", "alpha": 0.9},
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def sensitivity_heatmap(
    sensitivity_df: pd.DataFrame,
    *,
    save_path: Path | str | None = None,
) -> Any:
    """Heatmap of mean RVI across the (alpha, beta) grid (gamma = 1-α-β).

    Consumes the long-form output of :func:`rvi.analysis.rvi.sensitivity_grid`.
    """
    import matplotlib.pyplot as plt

    if sensitivity_df.empty:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "Empty sensitivity grid", ha="center", va="center")
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
        return fig

    pivot = (
        sensitivity_df.groupby(["alpha", "beta"])["rvi_composite"]
        .mean()
        .reset_index()
        .pivot(index="alpha", columns="beta", values="rvi_composite")
        .sort_index(ascending=False)
        .sort_index(axis=1)
    )

    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{b:.2f}" for b in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{a:.2f}" for a in pivot.index])
    ax.set_xlabel(r"$\beta$ (Coverage weight)")
    ax.set_ylabel(r"$\alpha$ (Density weight)")
    ax.set_title("Mean RVI across (α, β) weight grid")
    fig.colorbar(im, ax=ax, label="Mean RVI")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _line_to_latlon(geom: Any) -> list[list[float]]:
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [[y, x] for x, y in geom.coords]
    if geom.geom_type == "MultiLineString":
        out: list[list[float]] = []
        for part in geom.geoms:
            out.extend([y, x] for x, y in part.coords)
        return out
    return []


def _popup_html(row: pd.Series, rvi_column: str) -> str:
    fields = [
        ("Segment", row.get("segment_id")),
        ("Waterway", row.get("waterway")),
        ("Name", row.get("name") or row.get("name_local") or "—"),
        ("Length (m)", _fmt_num(row.get("segment_length_m"))),
        ("Buildings", row.get("n_buildings")),
        ("Density (norm)", _fmt_num(row.get("density_norm"))),
        ("Coverage", _fmt_num(row.get("coverage"))),
        ("Proximity", _fmt_num(row.get("proximity"))),
        (f"RVI ({rvi_column})", _fmt_num(row.get(rvi_column))),
    ]
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in fields)
    return (
        '<table style="font-family: sans-serif; font-size: 12px; '
        'border-collapse: collapse;">'
        f"{rows}</table>"
    )


def _fmt_num(v: Any) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "—"
    return f"{f:.3f}" if abs(f) < 1000 else f"{f:.0f}"


def _build_legend(
    bins: list[float], colours: list[str], label: str
) -> str:
    rows = "".join(
        f"<tr><td><span style='display:inline-block;width:18px;height:12px;"
        f"background:{c};margin-right:6px;'></span></td>"
        f"<td>{lo:.2f} – {hi:.2f}</td></tr>"
        for lo, hi, c in zip(bins[:-1], bins[1:], colours, strict=True)
    )
    return (
        '<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; '
        'background: white; padding: 8px 12px; border: 1px solid #888; '
        'border-radius: 4px; font-family: sans-serif; font-size: 12px;">'
        f"<b>{label}</b><table>{rows}</table></div>"
    )


__all__ = [
    "aggregate_rvi_by_county",
    "county_choropleth",
    "rvi_severity_scatter",
    "segment_map",
    "sensitivity_heatmap",
]
