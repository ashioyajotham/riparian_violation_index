# RVI-Kenya

**Riparian Violation Index for Kenya** — a fully open-source computational pipeline that
quantifies riparian-zone encroachment along Kenya's surface drainage network and validates
it against live flood-severity data from the Google Flood Forecasting API.

The full proposal lives in [`RVI_Kenya_Proposal.md`](./RVI_Kenya_Proposal.md). This README
is the operator manual.

---

## What it does

For every 500-metre river segment in the analysis area:

1. **Buffers** the OSM waterway centreline at the three legal Kenyan setbacks (6 m, 10 m,
   30 m), correcting for centreline-to-bank offset using a Strahler-order half-width.
2. **Spatially joins** Microsoft Global ML Building Footprints into each buffer.
3. **Computes** three sub-scores per segment:
   - **Density** — encroaching buildings per kilometre of river,
   - **Coverage** — fraction of buffer area covered by footprints,
   - **Proximity** — mean penetration of buildings into the buffer.
4. **Combines** them into a composite **RVI** score in `[0, 1]`:
   `RVI = α·D + β·C + γ·P` with default weights `α=0.4, β=0.3, γ=0.3`.
5. **Validates** the index by Spearman-correlating per-gauge upstream RVI against
   real-time flood severity from Google Flood Hub.

The same pipeline produces an interactive Folium choropleth, a ranked list of the most
encroached segments, and a publishable scatter plot of `upstream_rvi_p75` vs
`floodhub_severity_int`, plus a weight-sensitivity heatmap for the primary buffer width.

---

## Repository layout

```
rvi-kenya/
├── pyproject.toml              # build config + dependencies
├── README.md                   # this file
├── METHODOLOGY.md              # rigorous formula documentation
├── RVI_Kenya_Proposal.md       # full research proposal
├── .env.example                # copy to .env and fill in API keys
├── src/rvi/
│   ├── config.py               # single source of truth for all parameters
│   ├── ingestion/
│   │   ├── osm.py              # OSM waterways (Overpass + Geofabrik PBF)
│   │   ├── floodhub.py         # Google Flood Forecasting REST client
│   │   └── buildings.py        # Microsoft footprints via tiled GeoDataFrame ingestion
│   ├── geometry/
│   │   ├── buffer.py           # multi-width riparian buffers
│   │   └── segment.py          # 500 m linear segmentation
│   ├── analysis/
│   │   ├── encroachment.py     # spatial join + per-segment statistics
│   │   ├── rvi.py              # D, C, P sub-scores + composite
│   │   └── validation.py       # Spearman ρ + bootstrap CI vs Flood Hub
│   ├── viz/
│   │   └── choropleth.py       # Folium maps + matplotlib scatter
│   └── cli.py                  # `rvi pilot` / `rvi national` entry-points
├── tests/                      # pytest suite (offline, mocks for network)
└── notebooks/
    ├── 01_nairobi_pilot.ipynb
    └── 03_floodhub_validation.ipynb
```

---

## Getting started

### 1. Install

The pilot pipeline runs lean. The `national` extra adds the `osmium` PBF
reader for country-wide waterway ingestion; the `phase2` extra adds the
DEM / catchment-delineation stack (`pysheds`, `rasterio`, `rioxarray`)
used for workflows that replace the Phase-1 Euclidean upstream radius with
hydrologically correct catchment polygons. The current CLI accepts
precomputed catchment polygons via `--catchments` and can derive them from
a DEM via `--dem`.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# bash / zsh
source .venv/bin/activate

pip install -e ".[dev]"                       # pilot + dev tools
pip install -e ".[dev,national]"              # + Geofabrik PBF support
pip install -e ".[dev,national,phase2]"       # + DEM catchment delineation
```

### 2. Configure

```bash
cp .env.example .env
# edit .env and set FLOODHUB_API_KEY
```

### 3. Run the test suite (no network required)

```bash
pytest -m "not network"
```

### 4. Run the Nairobi pilot

```bash
rvi pilot --area nairobi
rvi pilot --area nairobi --catchments data/processed/gauge_catchments.gpkg
rvi pilot --area nairobi --dem data/raw/copernicus_dem.tif
```

This:

- fetches Nairobi-basin waterways from Overpass,
- downloads the building footprint tiles covering the basin,
- queries the Flood Hub status for Kenyan gauges,
- writes `outputs/nairobi_pilot/rvi_segments.gpkg`, `outputs/nairobi_pilot/rvi_segment_map.html`,
  and `outputs/nairobi_pilot/rvi_sensitivity_analysis.png`.

### 5. Run the national pipeline

```bash
rvi national
rvi national --catchments data/processed/gauge_catchments.gpkg
rvi national --dem data/raw/copernicus_dem.tif
```

Requires `pip install -e ".[national]"`. Streams the Geofabrik Kenya PBF and
filters Microsoft footprint tiles against the national riparian corridor.

---

## Reproducibility

- All numeric parameters live in `src/rvi/config.py`. Override per run via `--config` or
  environment variables.
- Every intermediate stage persists to disk as a GeoPackage; re-running with cached inputs
  skips refetch.
- Overpass, Geofabrik, GADM, Microsoft footprint artifacts, and Flood Hub responses are cached on disk.
- The exact Geofabrik PBF date and Microsoft footprint tile manifest used for any run are
  written into `outputs/<run_id>/manifest.json`.

---

## Licence

MIT — see [`LICENSE`](./LICENSE). Source datasets retain their original licences (OSM ODbL,
Microsoft ODbL, Google Flood Hub pilot terms).
