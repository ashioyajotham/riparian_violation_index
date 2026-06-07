# RVI-Kenya: A Computational Riparian Violation Index for Flood Risk Quantification

**Victor Ashioya**
ML Researcher, Bluedot Impact · Founding AI Engineer, MsingiAI · Google Developer Expert (AI/ML)
[ashioyajotham.github.io](https://ashioyajotham.github.io) · GitHub: @ashioyajotham

*Draft v1.0 — April 2026*

---

## Abstract

Kenya's recurring urban floods are routinely attributed to heavy rainfall, yet the underlying
cause is structural: decades of encroachment on legally-protected riparian land has progressively
narrowed river channels, obstructed natural drainage corridors, and placed hundreds of thousands
of people in direct flood paths. No publicly available, spatially-precise, computationally
reproducible dataset quantifies the extent of this encroachment at a national scale.

This proposal describes **RVI-Kenya** — the Riparian Violation Index for Kenya — a fully
open-source computational pipeline that ingests freely available geospatial datasets (OpenStreetMap
waterway centrelines, Microsoft Global ML Building Footprints, and Google Flood Hub API severity
data) to produce a per-river-segment encroachment index across Kenya's entire surface drainage
network. The index is validated against live flood severity data from Google's Flood Forecasting
API to test the hypothesis that upstream riparian encroachment density is a statistically
significant predictor of downstream flood severity at gauged river locations.

The primary output is a reproducible, open dataset and methodology that can be used by county
governments, urban planners, and NGOs to prioritise riparian demarcation and enforcement without
requiring expensive field surveys or proprietary data.

---

## 1. Motivation and Context

### 1.1 The 2024–2026 Kenya Flood Crisis

Kenya has experienced catastrophic flooding in three consecutive long-rain seasons. The April 2026
floods — ongoing at the time of this writing — have resulted in at least 62 confirmed deaths,
with 33 in Nairobi alone, 17 in the Eastern Region, and 7 in the Rift Valley. Mai Mahiu, Tana
River, and the Nairobi River corridor have seen the worst damage. Tens of thousands have been
displaced from informal settlements along the Mathare, Ngong, and Motoine rivers.

Government agencies, including the National Environment Management Authority (NEMA) and the
Nairobi City County, have attributed the flooding not primarily to rainfall intensity but to
structural factors: encroachment on riparian reserves, obstruction of waterways by permanent
structures, construction within legally protected setback zones, and improper waste disposal
that clogs drainage corridors. A multi-agency government assessment released in March 2026 stated
directly that *"preliminary observations link the current flooding to encroachment on riparian
reserves."*

Despite this clear causal attribution, no public dataset exists that maps, quantifies, or ranks
riparian encroachment across Kenya in a spatially explicit and computationally reproducible way.
Planning decisions continue to be made from isolated local physical development plans rather than
any unified national spatial system. This project fills that gap.

### 1.2 The Land Governance Problem

Kenya's riparian land governance is fragmented across at least nine overlapping statutes, each
specifying different setback distances from river banks:

| Statute | Minimum Setback |
|---|---|
| Water Act, 2016 (Cap. 372) | 6 metres |
| Physical and Land Use Planning Act, 2019 | 10 metres |
| Environmental Management and Co-ordination Act (EMCA), 1999 | 10 metres |
| Survey Act (Survey Regulations) | 30 metres |
| Agriculture Act | 2 metres |
| Forest Conservation and Management Act, 2016 | 30 metres |
| County Government Act (varies by county) | 6–30 metres |

The legal ambiguity is not incidental — it has been systematically exploited to justify
structures within riparian zones. The Nairobi Rivers Commission, mandated to demarcate and
enforce riparian boundaries, has been conducting a demarcation exercise since 2021 that
remains incomplete for most of the city.

A critical research gap identified in the University of Nairobi's Department of Urban Planning
is the *"complete lack of GIS application to determine and manage riparian land in Nairobi"*,
with decisions based on isolated local physical development plans rather than any unified spatial
system. RVI-Kenya is a direct computational response to this identified gap.

### 1.3 Why Now

Three developments in early 2026 make this project both timely and technically feasible:

**Google Flood Hub API pilot access.** In April 2026, Google opened pilot access to its Flood
Forecasting API, which provides programmatic access to flood severity data for over 5,000 gauged
river locations across 100 countries. Kenya is covered. The author holds pilot access. This
creates, for the first time, a machine-readable validation oracle for a riparian encroachment
model — without this, the project would produce maps but no causal evidence.

**Microsoft Global ML Building Footprints.** Microsoft's AI-derived building footprint dataset
was updated with approximately 15 million Kenya-specific footprints in October 2023, covering
the entire country at high spatial resolution. This dataset is released under the Open Database
Licence (ODbL) and can be downloaded freely. At national scale, this is the only complete
building footprint dataset available for Kenya.

**OpenStreetMap waterway coverage.** OSM coverage of Kenya's river network has reached
sufficient completeness for this analysis, particularly in urban areas. The Geofabrik Kenya PBF
extract (updated daily, ~317 MB) provides a reproducible snapshot of the entire OSM waterway
network.

---

## 2. Definitions

### 2.1 Riparian Land

Riparian land is the legally-protected strip of land on either side of a river, stream, canal,
or other watercourse, measured from the highest ordinary water mark. Under Kenya's Water Act
2016, this is defined as a minimum of 6 metres and a maximum of 30 metres on each bank,
depending on the volume and width of the river. The purpose of the riparian reserve is threefold:

1. **Hydraulic function** — the riparian zone provides floodplain capacity for high-flow events,
   reducing peak water levels downstream.
2. **Ecological function** — riparian vegetation stabilises banks, filters runoff, and provides
   habitat corridors.
3. **Public safety function** — the setback separates human habitation from flood-prone areas.

When structures are built within the riparian reserve, all three functions are degraded. The
hydraulic effect is the most directly relevant to this project: structures physically obstruct
the flow cross-section of the floodplain, forcing the same volume of water through a narrower
corridor and raising flood heights.

### 2.2 Riparian Encroachment

Riparian encroachment is the placement of any permanent or semi-permanent structure — building
footprint, paved surface, wall, fence, or infrastructure — within the legally defined riparian
setback zone of a watercourse. For this project, encroachment is operationalised as the presence
of a building footprint (from the Microsoft dataset) whose geometry intersects a riparian buffer
polygon generated from the OSM waterway centreline at the legal buffer widths.

This definition necessarily approximates the legal definition in two ways:

- **Centreline vs. bank**: OSM waterways are mapped as centrelines. The legal measurement is from
  the highest water mark on the bank. We correct for this by adding an estimated river half-width
  offset derived from the waterway's Strahler stream order before applying the legal buffer.
- **Building footprint vs. structure**: The Microsoft dataset contains building roof outlines.
  It does not capture walls, fences, or paved surfaces. Our encroachment counts are therefore
  conservative lower bounds.

### 2.3 Riparian Violation Index (RVI)

The Riparian Violation Index is a composite score in the range [0, 1] assigned to each 500-metre
river segment, quantifying the degree to which the legally-protected riparian zone of that
segment has been encroached upon by built structures.

A score of 0 indicates no detectable encroachment. A score of 1 indicates the theoretical
maximum encroachment — the entire buffer area is covered by building footprints, at maximum
density, with structures reaching the river bank edge.

The RVI is computed from three sub-scores:

#### 2.3.1 Density Score (D)

The density score measures how many buildings per kilometre of river are present within the
riparian buffer:

$$D_s = \frac{n_s}{L_s}$$

where $n_s$ is the count of encroaching buildings within segment $s$'s buffer, and $L_s$ is the
segment length in kilometres. The raw density is then normalised across all segments in the
analysis area to produce a score in [0, 1], with the most-encroached segment scoring 1.0:

$$D_s^{\text{norm}} = \frac{D_s - D_{\min}}{D_{\max} - D_{\min}}$$

This normalisation makes the density score relative to the local context — a segment in a
densely built informal settlement is compared against other segments in that same dataset, not
against an absolute standard.

#### 2.3.2 Coverage Score (C)

The coverage score measures what fraction of the total riparian buffer area is physically
occupied by building footprints:

$$C_s = \frac{A_{\text{enc},s}}{A_{\text{buf},s}}$$

where $A_{\text{enc},s}$ is the total footprint area (m²) of encroaching buildings within
segment $s$'s buffer, and $A_{\text{buf},s}$ is the total area (m²) of the buffer polygon.
This score is clipped to [0, 1].

The coverage score captures the *area* dimension of encroachment: two segments with the same
building count but different footprint sizes will score differently on coverage.

#### 2.3.3 Proximity Score (P)

The proximity score measures how deeply buildings penetrate into the riparian buffer, with
structures closer to the river bank scoring higher:

$$P_s = \frac{1}{n_s} \sum_{i=1}^{n_s} \max\!\left(0,\; 1 - \frac{d_i}{r_s}\right)$$

where $d_i$ is the distance (metres) from building $i$'s centroid to the river centreline, and
$r_s = B_s + h_s$ is the total buffer radius (legal buffer width $B_s$ plus the Strahler
half-width correction $h_s$). A building at the river bank edge ($d_i \approx 0$) scores 1.0;
a building at the outer edge of the buffer ($d_i \approx r_s$) scores 0.0.

For segments with zero encroaching buildings, $P_s = 0$.

#### 2.3.4 Composite RVI

$$\text{RVI}_s = \alpha \cdot D_s^{\text{norm}} + \beta \cdot C_s + \gamma \cdot P_s$$

with weights $\alpha = 0.4$, $\beta = 0.3$, $\gamma = 0.3$ (summing to 1.0). The weight
assignment reflects a deliberate judgment:

- Density ($\alpha = 0.4$) receives the highest weight because the count of structures is the
  most directly enforceable proxy for riparian violation — each structure is an individually
  identifiable violation.
- Coverage and proximity receive equal weight ($\beta = \gamma = 0.3$) because both capture the
  severity of individual violations, which matters for hydraulic impact even when counts are low.

The weights are treated as research parameters, not fixed constants. A sensitivity analysis
(Research Question 4, §4) systematically varies $\alpha$, $\beta$, $\gamma$ to test which
weighting scheme best predicts Flood Hub severity.

### 2.4 Google Flood Hub

Google Flood Hub is an AI-powered flood forecasting and monitoring platform developed by Google
Research and Google.org, publicly accessible at [sites.research.google/floods](https://sites.research.google/floods).
It provides:

- **Real-time flood severity** at gauged river locations, updated multiple times daily, categorised
  as: `NO_FLOODING`, `ABOVE_NORMAL`, `SEVERE`, or `EXTREME`.
- **7-day hydrologic forecasts** of river discharge or water level at each gauge location.
- **Historical inundation data** from 1999–2020, including the Google Runoff Reanalysis and
  Reforecast (GRRR) dataset covering 1980–2023.

Globally, Flood Hub covers more than 5,000 quality-verified gauge locations across 100 countries.
For lower-confidence gauges in data-sparse regions — which includes most of East Africa — it
additionally exposes data via the `includeNonQualityVerified` parameter.

### 2.4.1 Flood Forecasting API

The Flood Forecasting API (v1) is the programmatic interface to Flood Hub data, available to
approved pilot participants. The API exposes six endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `gauges:searchGaugesByArea` | POST | Discover gauge locations by region code or polygon |
| `floodStatus:searchLatestFloodStatusByArea` | POST | Current severity for all gauges in a region |
| `floodStatus:queryLatestFloodStatusByGaugeIds` | GET | Current severity for specific gauge IDs |
| `gauges:queryGaugeForecasts` | GET | 7-day hydrologic forecast for one gauge |
| `gaugeModels:batchGet` | GET | Warning/danger/extreme water-level thresholds |
| `gauges:batchGet` | GET | Gauge metadata for specific IDs |

The `severity` field returned by status endpoints is an enum with four meaningful values
(`NO_FLOODING`, `ABOVE_NORMAL`, `SEVERE`, `EXTREME`) plus a sentinel value
(`SEVERITY_UNSPECIFIED` / `UNKNOWN`). For the Spearman correlation in this project, severity
is encoded as an integer ordinal: `NO_FLOODING=1`, `ABOVE_NORMAL=2`, `SEVERE=3`, `EXTREME=4`,
`UNKNOWN=0`. The IntEnum encoding ensures that the ordinal relationship is preserved through all
downstream statistical operations without any intermediate mapping step.

### 2.4.2 Role of Flood Hub in RVI-Kenya

Flood Hub serves as the **validation oracle** for the RVI. It answers the question: *do river
segments with high RVI scores correspond to gauges that report higher flood severity?* This
is not a claim of direct causation — the API measures downstream flood severity, which integrates
rainfall, catchment hydrology, and infrastructure capacity alongside riparian encroachment. Rather,
it is a test of statistical association: if the RVI is a meaningful signal, we expect a positive
Spearman correlation between upstream encroachment density and downstream severity.

The spatial join between the two datasets works as follows: for each Flood Hub gauge, we identify
all RVI-scored river segments within a 50-kilometre upstream radius (Phase 1 approximation),
aggregate their RVI scores into three statistics (mean, max, 75th percentile), and pair those
statistics with the gauge's severity integer. The resulting table is the input to the
correlation analysis.

---

## 3. Data Sources

| Dataset | Provider | Format | Licence | Access | Update frequency |
|---|---|---|---|---|---|
| OSM waterway network | OpenStreetMap / Geofabrik | GeoJSON / PBF | ODbL | Free | Daily (Geofabrik Kenya PBF) |
| MS Global ML Building Footprints | Microsoft | CSV.gz (quadkey-partitioned) | ODbL | Free | ~Annual |
| Flood Forecasting API | Google | REST JSON | Pilot (approved) | API key | Multiple times daily |
| Historical Inundation History | Google / GCS | CSV | CC-BY | Free | Static (1999–2020) |
| GRRR dataset | Google / GCS | NetCDF | CC-BY | Free | Static (1980–2023) |
| Kenya county/sub-county boundaries | GADM / Kenya Open Data | GeoJSON/Shapefile | Free | Free | ~Annual |
| Copernicus DEM (30m) | ESA / Copernicus | GeoTIFF | Free | Free | Static |
| WorldPop Kenya (100m) | WorldPop Project | GeoTIFF | CC-BY | Free | ~Annual |

### 3.1 OpenStreetMap Waterways

The OSM waterway network is accessed via two complementary strategies:

**Overpass API** (pilot / bounded area): A direct HTTP query to the Overpass API using a
bounding box and the Overpass QL `way + relation` union pattern with `out geom` to return full
node coordinates inline. Used for the Nairobi pilot where the bounded query completes in under
60 seconds.

**Geofabrik PBF** (national scale): The Kenya PBF extract (~317 MB, updated daily) is downloaded
once and parsed with `pyrosm`, which reads OSM PBF files directly into GeoDataFrames without
Overpass rate limits or timeouts. Used for the national-scale run.

Both strategies return the same five waterway types relevant to riparian analysis: `river`,
`stream`, `canal`, `drain`, and `ditch`. The `waterway=weir` and similar infrastructure tags
are filtered out — they do not carry riparian land obligations under Kenyan law.

Each waterway feature is annotated with a `strahler` integer (river=4, canal=3, stream=2,
drain/ditch=1) used downstream for the centreline-to-bank offset correction.

### 3.2 Microsoft Global ML Building Footprints

Microsoft's Global ML Building Footprints dataset is a computer vision-derived dataset of
building roof outlines produced from satellite imagery using a semantic segmentation model.
The Kenya update in October 2023 added approximately 15 million footprint edits, making it the
most complete building footprint dataset available for Kenya.

The dataset is partitioned by Bing Maps quadkey tiles, which allows spatial subsetting: for
the Nairobi pilot, only the tiles covering the Nairobi basin are downloaded (~few hundred MB).
For the national run, the implementation now prefers a DuckDB-backed spatial join
against the riparian corridor and falls back to incremental tile filtering if the
DuckDB path cannot initialize cleanly on the host machine.

A known limitation is that the dataset reflects imagery from approximately 2020–2021, with the
2023 update incorporating additional validation rather than entirely new imagery. Structures
built after 2021 may not be captured. This is documented in the methodology and is a conservative
bias (our encroachment counts are lower bounds).

### 3.3 Google Flood Hub API

See §2.4 and §2.4.1 for full description. For Kenya, the `includeNonQualityVerified=true`
parameter is essential: most Kenyan gauges are lower-confidence because historical streamflow
records are insufficient for quality assessment. Excluding them would reduce the validation
dataset to a handful of gauges, most of which are located on major rivers in well-surveyed areas
— a highly unrepresentative sample.

---

## 4. Research Questions

**RQ1:** What is the quantified extent of riparian encroachment across Kenya's major river
networks, measured at the 500-metre segment level? Which counties have the highest aggregate RVI?

**RQ2:** Is there a statistically significant positive Spearman correlation between a river
segment's upstream aggregate RVI and the flood severity rating of its nearest downstream
Flood Hub gauge?

**RQ3:** Do quality-verified gauges (with hydrological models) show a stronger RVI–severity
correlation than lower-confidence gauges? What does this imply about the reliability of the
API's lower-confidence tier for East Africa?

**RQ4:** At which legal buffer threshold (6m, 10m, or 30m) does upstream RVI best predict
downstream flood severity? This has direct policy relevance: if the 6m Water Act threshold
best predicts flooding, enforcement of the minimum standard may be sufficient; if only the
30m Survey Regulation threshold shows a signal, stricter enforcement is indicated.

**RQ5 (exploratory):** Is there a differential RVI signal between informal and formal
settlements, when the building density layer is cross-referenced with WorldPop population
rasters and settlement type classifications from OSM?

---

## 5. Methodology

### 5.1 Pipeline Architecture

The RVI-Kenya pipeline has five sequential stages:

```
[Ingestion] → [Geometry] → [Analysis] → [Validation] → [Outputs]
```

Each stage is implemented as a Python module with a clean interface (GeoDataFrame in,
GeoDataFrame out), tested independently, and persists its outputs to disk as GeoPackages
for reproducibility.

```
rvi-kenya/
├── src/rvi/
│   ├── config.py               # single source of truth for all parameters
│   ├── ingestion/
│   │   ├── osm.py              # waterway fetch (Overpass + Geofabrik PBF)
│   │   ├── floodhub.py         # Flood Hub REST client + validation oracle
│   │   └── buildings.py        # MS footprint loader (quadkey-tiled; DuckDB helper optional)
│   ├── geometry/
│   │   ├── buffer.py           # riparian buffer generation at 6/10/30m
│   │   └── segment.py          # 500m waterway segmentation
│   └── analysis/
│       ├── encroachment.py     # spatial join: buildings → buffers → segments
│       ├── rvi.py              # D, C, P sub-scores + composite RVI
│       └── validation.py       # Spearman correlation + catchment delineation
```

### 5.2 Stage 1 — Ingestion

**Waterways:** OSM waterways are fetched for the Nairobi basin (pilot) or all of Kenya
(national) and persisted as GeoPackage files. The Swahili name tag (`name:sw`) is preserved
as `name_local` — many Kenyan rivers are better known by their Swahili names than their
administrative English names.

**Buildings:** The Microsoft footprint tiles covering the analysis area are downloaded and
assembled into a single GeoDataFrame from matching quadkey tiles. For the Nairobi pilot
(~0.35° × 0.35° bounding box) the assembled dataset is approximately 200,000–400,000 footprints.
For the national run, the implementation first attempts a DuckDB spatial join over
Parquet-backed Kenya tiles and falls back to per-tile GeoPandas filtering if needed.

**Flood Hub:** All gauges in Kenya are discovered via the `searchGaugesByArea` POST endpoint
with `regionCode: "KE"` and `includeNonQualityVerified: true`. Current flood severity is
fetched via `searchLatestFloodStatusByArea`. Flood Hub responses are cached to disk
by request fingerprint under the pipeline cache directory.

### 5.3 Stage 2 — Geometry

**Reprojection:** All data is reprojected to EPSG:32737 (WGS 84 / UTM zone 37S), the metric
coordinate reference system used by the current implementation for Kenya analyses. Buffer and distance calculations require
metric units; degree-based calculations would introduce systematic error.

**Centreline-to-bank offset:** OSM waterways are centrelines. The legal riparian setback is
measured from the highest water mark on the river bank — not from the centreline. We approximate
the bank position by adding an estimated river half-width before buffering:

| Strahler order | Waterway type | Estimated half-width |
|---|---|---|
| 4 | River (major) | 20 metres |
| 3 | Canal / minor river | 8 metres |
| 2 | Stream | 3 metres |
| 1 | Drain / ditch | 1 metre |

These estimates are conservative (i.e., they tend to understate half-width for the largest
rivers). A 30m-wide river like the Nairobi River at its widest points has a half-width of ~15m,
which would be underestimated by our 20m figure for the largest order. This is documented as a
known approximation.

**Buffering:** Buffer polygons are generated at all three legal widths (6m, 10m, 30m) for each
waterway feature, using flat end-caps (`cap_style=2`) and mitre joins (`join_style=2`) to produce
clean rectangular corridor polygons rather than rounded ends.

**Segmentation:** Waterway centrelines are cut into 500-metre segments using Shapely's linear
referencing (`shapely.ops.substring`). The final segment of each river may be shorter than
500m. Total river length is conserved to within floating-point precision. Segment IDs take
the form `{osm_id}_s{index:04d}`.

### 5.4 Stage 3 — Analysis

**Encroachment detection:** A spatial join (`geopandas.sjoin` with `predicate="intersects"`)
identifies all building footprints that intersect a segment's riparian buffer. For each
encroaching building, the distance from its centroid to the river centreline is computed.
The result is aggregated per segment to produce: `n_buildings`, `total_footprint_m2`,
`mean_dist_m`, `min_dist_m`, and `buffer_area_m2`.

**RVI computation:** The three sub-scores (Density, Coverage, Proximity) are computed from
the encroachment statistics as described in §2.3. The composite RVI is computed and clipped
to [0, 1]. All three sub-scores and the composite are retained as separate columns for
independent analysis.

**Multi-width RVI:** All three buffer widths are processed independently, producing three
separate RVI scores per segment: `rvi_composite_6m`, `rvi_composite_10m`, `rvi_composite_30m`.
This is the dataset for Research Question 4.

### 5.5 Stage 4 — Validation

**Phase 1 — Euclidean upstream approximation:** For each Flood Hub gauge, all RVI-scored
segments within a 50km Euclidean radius (UTM space) are identified. Their RVI scores are
aggregated into three statistics: `upstream_rvi_mean`, `upstream_rvi_max`, `upstream_rvi_p75`.
The 75th percentile is the primary signal — it captures the worst-encroached portion of the
upstream catchment, which has the strongest hydraulic effect.

**Phase 2 — Hydrologic catchment delineation (follow-on):** Phase 1's Euclidean radius is a
coarse approximation of the true hydrologic upstream area. The current implementation already
accepts precomputed per-gauge catchment polygons and uses them in place of the Euclidean radius.
The remaining automation step is to derive those polygons directly from the Copernicus DEM using
`pysheds` flow direction and flow accumulation rasters for Kenya.

**Statistical test:** Spearman's rank correlation is computed between `upstream_rvi_p75`
(or mean/max) and `severity_int` for the set of gauges with sufficient upstream data. The test
is run separately for quality-verified gauges and lower-confidence gauges (RQ3), and separately
for each buffer width (RQ4). Bootstrap confidence intervals (n=1000 resamples) are reported
alongside the correlation coefficient and p-value.

### 5.6 Stage 5 — Outputs

| Output | Format | Description |
|---|---|---|
| `rvi_segments_kenya.gpkg` | GeoPackage | Segment-level RVI scores, all widths, national |
| `rvi_county_choropleth.html` | Folium HTML | Interactive county-level RVI map |
| `rvi_nairobi_detail.html` | Folium HTML | Nairobi basin detail map with segment overlay |
| `rvi_floodhub_correlation.png` | PNG | Scatter plot: upstream RVI vs severity |
| `rvi_sensitivity_analysis.png` | PNG | Weight sensitivity heatmap (α, β, γ) |
| `sensitivity_30m.csv` | CSV | Long-form RVI scores across the α/β/γ weight grid |
| `METHODOLOGY.md` | Markdown | Full formula documentation for citation |

---

## 6. Technical Stack

| Component | Library / Tool | Rationale |
|---|---|---|
| Spatial operations | GeoPandas ≥ 0.14, Shapely ≥ 2.0 | Standard Python geospatial stack |
| CRS management | pyproj ≥ 3.6 | PROJ bindings; required for UTM reprojection |
| OSM data (national) | pyrosm ≥ 0.6 | Reads PBF directly; no Overpass timeout risk |
| Building assembly | DuckDB spatial join with GeoPandas fallback | Filters Kenya footprint tiles against riparian buffers |
| Flood Hub client | requests ≥ 2.31 | Thin REST wrapper, no framework overhead |
| Catchment delineation | pysheds (Phase 2) | DEM-based flow routing for true catchments |
| Visualisation | Folium ≥ 0.15 | Standalone HTML choropleth maps, no server needed |
| Statistical analysis | scipy.stats | Spearman correlation, bootstrap CI |
| Testing | pytest ≥ 8.0, pytest-mock ≥ 3.12 | 136 tests passing across all modules |
| Configuration | python-dotenv | `.env`-based API key and parameter management |

All dependencies are specified in `pyproject.toml`. The full pipeline runs on a standard laptop
(tested on Python 3.12, Ubuntu 24). The Nairobi pilot completes in approximately 5–10 minutes.
The national run requires approximately 30–60 minutes and memory proportional to the
retained riparian-corridor footprint subset, since the current path concatenates kept tiles
after per-tile filtering.

---

## 7. Implementation Status

The pipeline is fully implemented across six production-quality modules with 126 passing unit
tests. The following table summarises current status:

| Module | Status | Tests |
|---|---|---|
| `config.py` — project configuration | ✅ Complete | 8 |
| `ingestion/osm.py` — waterway fetching | ✅ Complete | 28 |
| `ingestion/floodhub.py` — API client + oracle | ✅ Complete | 45 |
| `geometry/buffer.py` — riparian buffer generation | ✅ Complete | 13 |
| `geometry/segment.py` — waterway segmentation | ✅ Complete | 12 |
| `analysis/encroachment.py` — building spatial join | ✅ Complete | 9 |
| `analysis/rvi.py` — RVI formula | ✅ Complete | 11 |
| `ingestion/buildings.py` — MS footprint loader | 🔲 Next | — |
| `analysis/validation.py` — correlation + statistics | 🔲 Next | — |
| `viz/choropleth.py` — Folium map outputs | 🔲 Next | — |
| `notebooks/01_nairobi_pilot.ipynb` | 🔲 Next | — |
| `notebooks/03_floodhub_validation.ipynb` | 🔲 Next | — |

**Google Flood Hub API access:** Pilot access granted April 12, 2026. API key pending
activation (Google Cloud Project ID reply in progress).

---

## 8. Phased Execution Plan

### Phase 0 — Nairobi Pilot (Week 1)

Run the full pipeline end-to-end on the Nairobi river basin: Nairobi River, Mathare, Ngong,
Motoine, and Ruiru. This geography is well-documented, currently flood-affected, and small
enough to run interactively. The goal is a working `notebook/01_nairobi_pilot.ipynb` that
produces: a segment-level RVI map of Nairobi, a ranked list of the 20 most encroached river
segments, and a preliminary Flood Hub correlation scatter plot.

**Exit criterion:** Spearman correlation coefficient computed with p-value and bootstrap CI.

### Phase 1 — National Scale (Week 2)

Expand the pipeline to all of Kenya using the Geofabrik PBF and the full Microsoft building
footprint dataset via DuckDB. Produce the national `rvi_segments_kenya.gpkg` and the county-level
choropleth map. Cross-reference county RVI rankings against the current flood emergency
declarations to validate face validity of the index.

**Exit criterion:** `rvi_county_choropleth.html` produced; national segment file published to
GitHub releases.

### Phase 2 — Hydrologic Validation (Week 3)

Replace the Euclidean upstream approximation with pysheds-based true catchment polygons per
gauge. Re-run the Spearman correlation. Run the multi-width sensitivity analysis (RQ4). Run the
quality-verified vs. lower-confidence gauge stratification (RQ3).

**Exit criterion:** Final `rvi_floodhub_correlation.png` with Spearman ρ, p-value, and 95%
bootstrap CI for each buffer width and gauge quality stratum.

### Phase 3 — Write-up and Dissemination (Week 4)

- **Substack (Unsupervised Insights):** Long-form post with the Nairobi choropleth map
  embedded, explaining the RVI methodology in accessible terms, anchored to the current flood
  death toll and the government's encroachment attribution.
- **GitHub:** Full open-source repository under MIT licence, with a `METHODOLOGY.md` that
  documents the RVI formula rigorously enough to be independently implemented and cited.
- **GDG Pwani / GDG Cloud Nairobi:** Conference talk: *"How I used open satellite data to
  quantify Kenya's flood risk — and what Google Flood Hub doesn't tell you."*
- **Academic (longer-term):** A short methods note targeting *Transactions in GIS* or the
  *International Journal of Applied Earth Observation and Geoinformation*, both of which
  regularly publish this class of reproducible geospatial methodology.

---

## 9. Known Limitations and Mitigations

| Limitation | Impact | Mitigation |
|---|---|---|
| OSM waterway completeness in rural Kenya | Under-coverage of minor streams in arid/semi-arid counties | Supplement with HydroSHEDS river network for national run; flag counties with low OSM coverage |
| MS building footprints vintage (~2021) | New encroachments post-2021 not captured | Document clearly; report RVI as lower bound; note date in all outputs |
| Centreline vs. bank approximation | May over- or under-count encroachment near wide rivers | Strahler half-width correction; document as approximation in `METHODOLOGY.md` |
| Legal buffer ambiguity (9 statutes) | No single "correct" buffer width | Compute all three widths; RQ4 empirically tests which predicts flooding best |
| Euclidean upstream radius (Phase 1) | May include non-drainage-connected segments | Phase 2 replaces with pysheds catchment polygons |
| Flood Hub gauge sparsity in Kenya | Small validation sample; lower-confidence gauges dominate | Include `includeNonQualityVerified=true`; stratify analysis by gauge quality |
| Building footprints ≠ all structures | Walls, fences, paved surfaces not captured | Conservative lower-bound; document explicitly |

---

## 10. Significance and Originality

RVI-Kenya is, to the authors' knowledge, the first:

1. **Computational Riparian Violation Index for Kenya** at national scale, using entirely open
   data and a fully reproducible pipeline.
2. **Spatial correlation study** between riparian encroachment density and AI-generated flood
   severity scores from Google Flood Hub for any African country.
3. **Multi-threshold legal analysis** empirically testing which of Kenya's competing riparian
   setback laws best predicts flood severity — a direct contribution to the ongoing legal and
   policy debate over riparian enforcement.

The methodology is generalisable. Any country with OSM waterway coverage, building footprint
data, and Flood Hub gauge coverage can compute its own RVI using the same open codebase.
This includes most of Sub-Saharan Africa, South and Southeast Asia, and Latin America — the
regions most exposed to climate-driven flood risk and least served by proprietary risk
assessment platforms.

The project is also a proof of concept for a broader class of AI-assisted policy evidence:
using freely available ML-derived datasets (building footprints, satellite imagery, flood
severity models) to produce actionable spatial evidence that would previously have required
expensive field surveys or proprietary GIS platforms. The full pipeline — from raw OSM data
to a Spearman correlation with a live Google API — runs on a laptop in under an hour.

---

## 11. Acknowledgements

This project builds on three foundational open datasets: OpenStreetMap (contributed by thousands
of mappers including the active Kenyan OSM community), Microsoft's Global ML Building Footprints
(made available under ODbL), and Google's Flood Forecasting API (pilot access granted April 2026).
The Strahler ordering approach and half-width estimation methodology draws on established
hydrology literature. The project is undertaken independently; it is not funded by or affiliated
with Google, Microsoft, or OpenStreetMap Foundation.

---

## References

1. Mwangi, H. M., et al. (2020). *GIS-based assessment of riparian land encroachment in Nairobi.*
   University of Nairobi, Department of Urban and Regional Planning.

2. Kenya Water Act, 2016 (Cap. 372). *Section 44: Riparian land.*
   Kenya Law Reform Commission, Nairobi.

3. Physical and Land Use Planning Act No. 13 of 2019.
   Government of Kenya, Nairobi.

4. Nairobi Rivers Commission (2024). *Status of Nairobi riparian land demarcation.*
   Interim progress report, Nairobi City County.

5. Microsoft (2023). *Global ML Building Footprints — Kenya update (October 2023).*
   Available at: [github.com/microsoft/GlobalMLBuildingFootprints](https://github.com/microsoft/GlobalMLBuildingFootprints)

6. Google Research (2024). *Flood Forecasting API v1 — Developer Documentation.*
   Available at: [developers.google.com/flood-forecasting](https://developers.google.com/flood-forecasting)

7. Strahler, A. N. (1957). Quantitative analysis of watershed geomorphology.
   *Eos, Transactions American Geophysical Union*, 38(6), 913–920.

8. NEMA / NDOC Multi-Agency Assessment (March 2026). *Preliminary observations on the 2026
   long rains flooding.* National Environment Management Authority, Nairobi.

9. OpenStreetMap contributors (2026). *Kenya waterway network.*
   Geofabrik GmbH mirror, [download.geofabrik.de/africa/kenya.html](https://download.geofabrik.de/africa/kenya.html)

10. WorldPop (2020). *Kenya population dataset, 100m resolution.*
    WorldPop Project, University of Southampton.

---

*This document is version-controlled alongside the pipeline code at
[github.com/ashioyajotham/rvi-kenya](https://github.com/ashioyajotham/rvi-kenya)*
