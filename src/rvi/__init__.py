"""RVI-Kenya: computational Riparian Violation Index for Kenya.

Public API
----------
``rvi.config``               — global, immutable parameter container.
``rvi.ingestion``            — OSM, Microsoft footprints, Google Flood Hub.
``rvi.geometry``             — buffers and segmentation.
``rvi.analysis``             — encroachment statistics, RVI scoring, validation.
``rvi.viz``                  — Folium / matplotlib outputs.

Importing this top-level package is intentionally cheap: heavy geospatial libraries
are imported lazily inside the submodules that need them.
"""

from __future__ import annotations

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("rvi-kenya")
except _metadata.PackageNotFoundError:  # editable install before metadata is built
    __version__ = "0.1.0+dev"

__all__ = ["__version__"]
