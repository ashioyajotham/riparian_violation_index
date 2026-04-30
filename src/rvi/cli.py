"""Command-line entry-point: ``rvi pilot`` and ``rvi national``.

Examples
--------

::

    # Pilot run (Nairobi basin)
    rvi pilot

    # Pilot run, but skip the heavy Microsoft download (smoke test)
    rvi pilot --skip-buildings

    # Pilot run with no Flood Hub key set
    rvi pilot --skip-floodhub

    # National run (requires the `national` extra installed)
    rvi national
"""

from __future__ import annotations

import json
import logging

import click

from rvi.config import get_config

logger = logging.getLogger("rvi.cli")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-v", "--verbose", is_flag=True, help="Verbose (DEBUG) logging.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """RVI-Kenya — Riparian Violation Index pipeline."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@main.command()
@click.option(
    "--area",
    default="nairobi",
    show_default=True,
    help="Pilot area name (defaults to nairobi). Used to name the output dir.",
)
@click.option(
    "--bbox",
    type=str,
    default=None,
    help='Override bbox: "west,south,east,north" in EPSG:4326.',
)
@click.option(
    "--skip-buildings",
    is_flag=True,
    help="Do not download Microsoft footprints (smoke test only).",
)
@click.option(
    "--skip-floodhub",
    is_flag=True,
    help="Do not call Google Flood Hub (validation step is skipped).",
)
def pilot(area: str, bbox: str | None, skip_buildings: bool, skip_floodhub: bool) -> None:
    """Run the Nairobi pilot pipeline (or any small bbox)."""
    from rvi.pipeline import run_pilot

    cfg = get_config()
    box: tuple[float, float, float, float] | None = None
    if bbox:
        try:
            west, south, east, north = (float(x) for x in bbox.split(","))
            box = (west, south, east, north)
        except (TypeError, ValueError) as exc:
            raise click.UsageError(
                'bbox must be "west,south,east,north" floats'
            ) from exc

    result = run_pilot(
        bbox=box,
        config=cfg,
        run_name=f"{area}_pilot",
        skip_buildings=skip_buildings,
        skip_floodhub=skip_floodhub,
    )

    click.echo(f"Run directory: {result.run_dir}")
    if result.map_path is not None:
        click.echo(f"Interactive map: {result.map_path}")
    if result.correlation:
        click.echo("\nCorrelation (Spearman ρ — upstream RVI vs Flood Hub severity):")
        for k, v in result.correlation.items():
            click.echo(
                f"  {k:>32s}: ρ={v.rho:+.3f}, n={v.n}, "
                f"95%CI=[{v.ci_low:+.3f}, {v.ci_high:+.3f}], p={v.pvalue:.3g}"
            )
    else:
        click.echo("Validation skipped or no Flood Hub data available.")
    click.echo(f"\nManifest: {result.manifest_path}")


@main.command()
@click.option(
    "--pbf",
    "pbf_path",
    type=click.Path(exists=False),
    default=None,
    help="Path to a Geofabrik Kenya PBF (downloaded if absent).",
)
@click.option(
    "--bbox",
    type=str,
    default=None,
    help='Restrict the PBF parse to a bbox: "west,south,east,north" (EPSG:4326).',
)
@click.option(
    "--run-name",
    default="national",
    show_default=True,
    help="Output sub-directory under outputs/.",
)
@click.option(
    "--skip-buildings",
    is_flag=True,
    help="Do not stream Microsoft footprints (smoke test only).",
)
@click.option(
    "--skip-floodhub",
    is_flag=True,
    help="Do not call Google Flood Hub (validation step is skipped).",
)
@click.option(
    "--skip-counties",
    is_flag=True,
    help="Do not download GADM counties or build the choropleth.",
)
def national(
    pbf_path: str | None,
    bbox: str | None,
    run_name: str,
    skip_buildings: bool,
    skip_floodhub: bool,
    skip_counties: bool,
) -> None:
    """Run the country-scale pipeline (proposal \u00a78 Phase 1)."""
    try:
        import osmium  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise click.ClickException(
            'national run requires the "national" extra: '
            'pip install -e ".[national]"'
        ) from exc

    from pathlib import Path as _P

    from rvi.pipeline import run_national

    cfg = get_config()

    box: tuple[float, float, float, float] | None = None
    if bbox:
        try:
            west, south, east, north = (float(x) for x in bbox.split(","))
            box = (west, south, east, north)
        except (TypeError, ValueError) as exc:
            raise click.UsageError(
                'bbox must be "west,south,east,north" floats'
            ) from exc

    result = run_national(
        config=cfg,
        pbf_path=_P(pbf_path) if pbf_path else None,
        run_name=run_name,
        bbox=box,
        skip_buildings=skip_buildings,
        skip_floodhub=skip_floodhub,
        skip_counties=skip_counties,
    )

    click.echo(f"Run directory: {result.run_dir}")
    if result.choropleth_path is not None:
        click.echo(f"County choropleth: {result.choropleth_path}")
    if result.map_path is not None:
        click.echo(f"Segment map: {result.map_path}")
    if result.correlation:
        click.echo(
            "\nCorrelation (Spearman \u03c1 \u2014 upstream RVI vs Flood Hub severity):"
        )
        for k, v in result.correlation.items():
            click.echo(
                f"  {k:>32s}: \u03c1={v.rho:+.3f}, n={v.n}, "
                f"95%CI=[{v.ci_low:+.3f}, {v.ci_high:+.3f}], p={v.pvalue:.3g}"
            )
    else:
        click.echo("Validation skipped or no Flood Hub data available.")
    click.echo(f"\nManifest: {result.manifest_path}")


@main.command()
def show_config() -> None:
    """Print the resolved Config (env-driven) as JSON."""
    cfg = get_config()
    payload = {
        k: (str(v) if hasattr(v, "as_posix") else v)
        for k, v in cfg.__dict__.items()
        if not k.startswith("_")
    }
    click.echo(json.dumps(payload, default=str, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
