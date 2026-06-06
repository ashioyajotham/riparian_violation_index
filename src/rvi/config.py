"""Project configuration — single source of truth for all RVI-Kenya parameters.

Every numeric constant referenced anywhere in the pipeline is defined here. This
makes every research parameter (buffer widths, RVI weights, segment length,
upstream radius, ...) inspectable, overridable, and reproducible.

Override mechanism
------------------
1. Defaults are class-level attributes.
2. Environment variables (loaded from ``.env`` if present) override defaults.
3. A caller may construct a ``Config(...)`` with explicit kwargs to override
   anything programmatically (typical for tests and notebooks).

The module exposes two things:

* :class:`Config` — a frozen dataclass holding every parameter.
* :func:`get_config` — returns a process-wide, lazily-instantiated default.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env into os.environ once, on import. Calling code can still override
# anything by passing kwargs to Config(...).
load_dotenv(override=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - guard against misconfig
        raise ValueError(f"Environment variable {key}={raw!r} is not a float") from exc


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover
        raise ValueError(f"Environment variable {key}={raw!r} is not an int") from exc


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key) or default


def _env_path(key: str, default: Path) -> Path:
    raw = os.environ.get(key)
    return Path(raw).expanduser() if raw else default


# ---------------------------------------------------------------------------
# Strahler stream-order half-width offsets (§5.3)
# ---------------------------------------------------------------------------
#
# OSM waterways are mapped as centrelines. The Kenyan legal setback is measured
# from the highest water mark on the *bank*. We approximate the bank position
# by adding an estimated half-width before applying the legal buffer.
#
# Values follow the proposal table (§5.3) and are intentionally conservative.
DEFAULT_STRAHLER_HALF_WIDTHS_M: Mapping[int, float] = {
    1: 1.0,   # drain / ditch
    2: 3.0,   # stream
    3: 8.0,   # canal / minor river
    4: 20.0,  # major river
}


# Mapping from OSM `waterway` tag to a representative Strahler stream order.
# Used at ingestion time to label waterways for the buffer offset.
DEFAULT_WATERWAY_STRAHLER: Mapping[str, int] = {
    "river": 4,
    "canal": 3,
    "stream": 2,
    "drain": 1,
    "ditch": 1,
}


# Severity oracle (§2.4.1). IntEnum-style mapping preserved here so non-enum
# code paths can still produce the integers used in the Spearman test.
DEFAULT_SEVERITY_INT: Mapping[str, int] = {
    "UNKNOWN": 0,
    "SEVERITY_UNSPECIFIED": 0,
    "NO_FLOODING": 1,
    "ABOVE_NORMAL": 2,
    "SEVERE": 3,
    "EXTREME": 4,
}


# Nairobi pilot bounding box: west, south, east, north (EPSG:4326).
# Encloses Nairobi River, Mathare, Ngong, Motoine, Ruiru basins.
DEFAULT_NAIROBI_BBOX: tuple[float, float, float, float] = (
    36.65,  # west
    -1.45,  # south
    37.10,  # east
    -1.15,  # north
)


# ---------------------------------------------------------------------------
# The Config dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Frozen, fully-typed parameter container for the entire pipeline.

    Defaults match the values stated in :file:`RVI_Kenya_Proposal.md`. Any
    field can be overridden at construction time, and a handful are picked up
    from environment variables when ``Config()`` is called with no args.
    """

    # --- Coordinate reference systems --------------------------------------
    crs_geographic: str = "EPSG:4326"
    """Geographic CRS used for ingestion (lat/lon)."""

    crs_metric: str = "EPSG:32737"
    """UTM zone 37S — appropriate for Nairobi and most Kenyan study areas."""

    # --- Riparian buffer widths (metres) -----------------------------------
    buffer_widths_m: tuple[int, int, int] = (6, 10, 30)
    """Three legal Kenyan setbacks (Water Act, PLUPA/EMCA, Survey Regs)."""

    primary_buffer_m: int = 30
    """Width used when reporting a single 'headline' RVI."""

    strahler_half_widths_m: Mapping[int, float] = field(
        default_factory=lambda: dict(DEFAULT_STRAHLER_HALF_WIDTHS_M)
    )

    waterway_strahler: Mapping[str, int] = field(
        default_factory=lambda: dict(DEFAULT_WATERWAY_STRAHLER)
    )

    waterway_types: tuple[str, ...] = ("river", "stream", "canal", "drain", "ditch")
    """OSM `waterway=*` values that carry riparian obligations under Kenyan law."""

    # --- Segmentation ------------------------------------------------------
    segment_length_m: float = 500.0
    """Target along-river length of each RVI segment (§2.3, §5.3)."""

    min_segment_length_m: float = 50.0
    """Tail segments shorter than this are merged back into the previous segment."""

    # --- RVI weights (§2.3.4) ---------------------------------------------
    rvi_alpha: float = 0.4
    """Weight on the (normalised) Density sub-score."""

    rvi_beta: float = 0.3
    """Weight on the Coverage sub-score."""

    rvi_gamma: float = 0.3
    """Weight on the Proximity sub-score."""

    # --- Validation oracle -------------------------------------------------
    upstream_radius_m: float = 50_000.0
    """Phase-1 Euclidean radius around each gauge (§5.5)."""

    upstream_aggregations: tuple[str, ...] = ("mean", "max", "p75")
    """Statistics produced from the upstream segment set."""

    bootstrap_iterations: int = 1_000
    """Resamples for the Spearman bootstrap CI."""

    bootstrap_ci: float = 0.95
    """Confidence level for the bootstrap interval."""

    severity_int: Mapping[str, int] = field(
        default_factory=lambda: dict(DEFAULT_SEVERITY_INT)
    )

    # --- Pilot scope -------------------------------------------------------
    nairobi_bbox: tuple[float, float, float, float] = DEFAULT_NAIROBI_BBOX

    # --- HTTP / API --------------------------------------------------------
    user_agent: str = "rvi-kenya/0.1 (+https://github.com/ashioyajotham/rvi-kenya)"
    request_timeout_s: float = 120.0
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    floodhub_base_url: str = "https://floodforecasting.googleapis.com/v1"
    floodhub_api_key: str | None = None
    geofabrik_kenya_pbf_url: str = (
        "https://download.geofabrik.de/africa/kenya-latest.osm.pbf"
    )
    ms_buildings_index_url: str = (
        "https://minedbuildings.z5.web.core.windows.net/global-buildings/"
        "dataset-links.csv"
    )

    # --- Paths -------------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path("data"))
    raw_dir: Path = field(default_factory=lambda: Path("data") / "raw")
    interim_dir: Path = field(default_factory=lambda: Path("data") / "interim")
    processed_dir: Path = field(default_factory=lambda: Path("data") / "processed")
    cache_dir: Path = field(default_factory=lambda: Path("data") / "cache")
    outputs_dir: Path = field(default_factory=lambda: Path("outputs"))

    # =====================================================================
    # Helpers
    # =====================================================================

    def __post_init__(self) -> None:
        # Ruff B009: dataclass(frozen=True) requires object.__setattr__ for any
        # post-init normalisation.
        weight_sum = self.rvi_alpha + self.rvi_beta + self.rvi_gamma
        if not (0.99 <= weight_sum <= 1.01):
            raise ValueError(
                f"RVI weights must sum to ~1.0; got "
                f"alpha+beta+gamma = {weight_sum:.4f}"
            )
        for w in self.buffer_widths_m:
            if w <= 0:
                raise ValueError(f"buffer width must be positive; got {w}")
        if self.segment_length_m <= 0:
            raise ValueError("segment_length_m must be positive")
        if not (0 < self.bootstrap_ci < 1):
            raise ValueError("bootstrap_ci must be in (0, 1)")
        if self.upstream_radius_m <= 0:
            raise ValueError("upstream_radius_m must be positive")
        for agg in self.upstream_aggregations:
            if agg not in {"mean", "max", "min", "p75", "p90"}:
                raise ValueError(f"unsupported upstream aggregation: {agg!r}")

    def half_width_for_strahler(self, order: int) -> float:
        """Return the centreline-to-bank half-width offset for a Strahler order."""
        if order in self.strahler_half_widths_m:
            return float(self.strahler_half_widths_m[order])
        # Clamp out-of-range orders to the nearest defined order.
        known = sorted(self.strahler_half_widths_m)
        order = max(known[0], min(known[-1], order))
        return float(self.strahler_half_widths_m[order])

    def strahler_for_waterway(self, waterway: str) -> int:
        return int(self.waterway_strahler.get(waterway, 1))

    def severity_to_int(self, severity: str | None) -> int:
        if severity is None:
            return self.severity_int["UNKNOWN"]
        return self.severity_int.get(str(severity).upper(), self.severity_int["UNKNOWN"])

    def ensure_dirs(self) -> None:
        """Create all data/output directories. Idempotent."""
        for p in (
            self.data_dir,
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.cache_dir,
            self.outputs_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

    # ---- Constructors -----------------------------------------------------

    @classmethod
    def from_env(cls, **overrides: object) -> Config:
        """Construct a :class:`Config` honouring environment variables.

        Env vars consulted (all optional; defaults apply when absent):

        * ``FLOODHUB_API_KEY``         — Google Flood Hub API key
        * ``FLOODHUB_BASE_URL``        — override the API host
        * ``RVI_USER_AGENT``           — polite User-Agent string
        * ``RVI_REQUEST_TIMEOUT``      — HTTP timeout in seconds
        * ``RVI_DATA_DIR``             — override ``./data``
        * ``RVI_OUTPUTS_DIR``          — override ``./outputs``
        * ``GEOFABRIK_KENYA_PBF_URL``  — alternate PBF mirror
        * ``MS_BUILDINGS_INDEX_URL``   — alternate footprint index
        """
        data_dir = _env_path("RVI_DATA_DIR", Path("data"))
        outputs_dir = _env_path("RVI_OUTPUTS_DIR", Path("outputs"))
        kwargs: dict[str, object] = {
            "user_agent": _env_str(
                "RVI_USER_AGENT",
                "rvi-kenya/0.1 (+https://github.com/ashioyajotham/rvi-kenya)",
            ),
            "request_timeout_s": _env_float("RVI_REQUEST_TIMEOUT", 120.0),
            "floodhub_api_key": os.environ.get("FLOODHUB_API_KEY") or None,
            "floodhub_base_url": _env_str(
                "FLOODHUB_BASE_URL", "https://floodforecasting.googleapis.com/v1"
            ),
            "geofabrik_kenya_pbf_url": _env_str(
                "GEOFABRIK_KENYA_PBF_URL",
                "https://download.geofabrik.de/africa/kenya-latest.osm.pbf",
            ),
            "ms_buildings_index_url": _env_str(
                "MS_BUILDINGS_INDEX_URL",
                "https://minedbuildings.z5.web.core.windows.net/global-buildings/"
                "dataset-links.csv",
            ),
            "data_dir": data_dir,
            "raw_dir": data_dir / "raw",
            "interim_dir": data_dir / "interim",
            "processed_dir": data_dir / "processed",
            "cache_dir": data_dir / "cache",
            "outputs_dir": outputs_dir,
        }
        kwargs.update(overrides)
        return cls(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Process-wide default
# ---------------------------------------------------------------------------

_DEFAULT: Config | None = None


def get_config() -> Config:
    """Return a lazily-instantiated, env-driven default :class:`Config`."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Config.from_env()
    return _DEFAULT


def set_config(cfg: Config) -> None:
    """Replace the process-wide default. Useful for tests."""
    global _DEFAULT
    _DEFAULT = cfg


def reset_config() -> None:
    """Drop the cached default so the next ``get_config()`` re-reads env vars."""
    global _DEFAULT
    _DEFAULT = None


__all__ = [
    "DEFAULT_NAIROBI_BBOX",
    "DEFAULT_SEVERITY_INT",
    "DEFAULT_STRAHLER_HALF_WIDTHS_M",
    "DEFAULT_WATERWAY_STRAHLER",
    "Config",
    "get_config",
    "reset_config",
    "set_config",
]
