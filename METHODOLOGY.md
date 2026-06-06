# RVI-Kenya — Methodology

**Companion document to the implementation in [`src/rvi/`](./src/rvi/) and the
research proposal in [`RVI_Kenya_Proposal.md`](./RVI_Kenya_Proposal.md).**

This file defines the Riparian Violation Index (RVI) formally enough to be
re-implemented and cited independently. Notation matches the proposal where
possible; deviations are noted.

---

## 1. Inputs

For every analysis we begin with three data products, all in the geographic
CRS **EPSG:4326** at ingestion time:

* **Waterways** \( \mathcal{W} \): a set of OSM `waterway` features, each
  with a centreline `LineString` and a Strahler order
  \( o(w) \in \{1, 2, 3, 4\} \). Mapped to `(river|stream|canal|drain|ditch)`
  via [`Config.waterway_strahler`](./src/rvi/config.py).
* **Buildings** \( \mathcal{B} \): polygon footprints from Microsoft's
  Global ML Building Footprints (October 2023 Kenya update).
* **Gauges** \( \mathcal{G} \): Google Flood Hub gauges retrieved via
  `gauges:searchGaugesByArea` with `regionCode="KE"` and
  `includeNonQualityVerified=true`, paired with their current severity from
  `floodStatus:searchLatestFloodStatusByArea`.

All three are reprojected to the metric CRS **EPSG:32737** (UTM 37S) before
any spatial computation. Buffering, distance, and area in EPSG:4326 give
systematically wrong metric results in Kenya.

---

## 2. Geometry

### 2.1 Centreline-to-bank correction

OSM waterways are *centrelines*; the legal Kenyan setback is measured from
the **highest water mark on the bank**. We approximate the bank position by
adding a Strahler-dependent half-width \( h(o) \) to the centreline before
buffering:

| Strahler order | Waterway type | Half-width \( h(o) \) |
|----------------|---------------|------------------------|
| 4              | Major river   | **20 m**               |
| 3              | Canal         | **8 m**                |
| 2              | Stream        | **3 m**                |
| 1              | Drain / ditch | **1 m**                |

These figures are conservative — they slightly under-state half-widths for
the largest Kenyan rivers. The per-row offset is applied via
`Config.half_width_for_strahler(o)`.

### 2.2 Riparian buffers

For waterway feature \( w \) with Strahler order \( o(w) \), and legal
setback \( B \in \{6, 10, 30\} \) metres (Kenyan Water Act / PLUPA / Survey
Regulations), the riparian buffer polygon is

\[
\Pi(w; B) \;=\; \mathrm{buffer}\!\bigl(\mathrm{geom}(w);\; r(w; B)\bigr),\qquad
r(w; B) \;=\; B + h\!\bigl(o(w)\bigr)
\]

with `cap_style=2` (flat ends) and `join_style=2` (mitre) so we get clean
rectangular corridor polygons rather than rounded ends.

### 2.3 Segmentation

Each waterway centreline is cut into segments of target length
\( L_0 = 500 \) m using Shapely's linear-referencing
(`shapely.ops.substring`):

* For \( w \) of length \( L(w) \), let \( N = \lceil L(w) / L_0 \rceil \).
* Cut at the \( N+1 \) breakpoints \( 0,\, L(w)/N,\, \dots,\, L(w) \).
* The final segment whose length is below
  `Config.min_segment_length_m` (default 50 m) is merged into its
  predecessor, so all retained segments are at least 50 m long.

Total length is conserved up to floating-point precision:
\( \sum_s L_s = \sum_w L(w) \). Segment ids take the form
`{osm_id}_s{index:04d}`.

---

## 3. Encroachment statistics

For each segment \( s \) and legal width \( B \):

* \( \Pi_s(B) \) — the segment's individual riparian buffer at total radius
  \( r_s = B + h(o_s) \).
* \( \mathcal{B}_s \) — the set of buildings whose geometry intersects
  \( \Pi_s(B) \) (joined via `geopandas.sjoin(predicate="intersects")`).
* \( n_s = |\mathcal{B}_s| \) — encroaching building count.
* \( A_{\text{enc},s} = \sum_{b \in \mathcal{B}_s} \mathrm{area}(b) \) —
  total footprint area in metres squared.
* \( A_{\text{buf},s} = \mathrm{area}(\Pi_s(B)) \) — buffer area.
* \( d_i \) — distance, in metres, from building \( b_i \)'s
  representative point to \( s \)'s centreline.

These six per-segment quantities are persisted in
`encroachment_<width>m.gpkg`.

---

## 4. RVI sub-scores

### 4.1 Density (D)

\[
D_s \;=\; \frac{n_s}{L_s/1000},\qquad
D_s^{\mathrm{norm}} \;=\;
\begin{cases}
0 & \text{if } D_{\max} = D_{\min} \\
\dfrac{D_s - D_{\min}}{D_{\max} - D_{\min}} & \text{otherwise}
\end{cases}
\]

with \( L_s \) in metres. The min-max normalisation makes the score relative
to the analysis area: a segment in a dense informal settlement is compared
against other segments in *that* dataset, not against an absolute benchmark.

A degenerate dataset (all segments share the same \( D_s \), e.g. all-zero)
maps everything to \( D_s^{\mathrm{norm}} = 0 \).

### 4.2 Coverage (C)

\[
C_s \;=\; \min\!\Bigl(1,\; \tfrac{A_{\text{enc},s}}{A_{\text{buf},s}}\Bigr)
\]

Clipped to \([0, 1]\). Captures the area dimension of encroachment: two
segments with the same building count but different footprint sizes will
score differently.

### 4.3 Proximity (P)

The proposal definition (§2.3.3) is

\[
P_s \;=\; \frac{1}{n_s} \sum_{i=1}^{n_s}
\max\!\Bigl(0,\; 1 - \tfrac{d_i}{r_s}\Bigr)
\]

with \( r_s = B + h(o_s) \) and \( P_s = 0 \) when \( n_s = 0 \).

In this implementation, we use a per-segment summary form when raw
distances have already been aggregated:

\[
\hat P_s \;=\; \max\!\Bigl(0,\; 1 - \tfrac{\bar d_s}{r_s}\Bigr)
\]

where \( \bar d_s \) is the mean of \( \{d_i\} \). The two are identical
when buildings are uniformly distributed inside the buffer, and \( \hat P_s \)
is monotonic in \( P_s \) for non-uniform distributions, which is sufficient
for the rank-based Spearman test downstream. Callers that retain raw
per-building distances can use
[`compute_proximity_from_distances`](./src/rvi/analysis/rvi.py) for the exact
form.

### 4.4 Composite RVI

\[
\mathrm{RVI}_s \;=\; \alpha\, D_s^{\mathrm{norm}}
                  + \beta\, C_s
                  + \gamma\, P_s,
\qquad \alpha + \beta + \gamma = 1
\]

with default weights \( \alpha = 0.4 \), \( \beta = 0.3 \), \( \gamma = 0.3 \).
The result is clipped to \([0, 1]\).

The weights are research parameters, not fixed constants. The
[`sensitivity_grid`](./src/rvi/analysis/rvi.py) routine sweeps the simplex
in steps of \( \Delta = 0.1 \) (66 valid triples by default) and recomputes
the composite without re-running the upstream encroachment join, supporting
Research Question 4 of the proposal.

---

## 5. Validation oracle (Phase 1 — Euclidean)

For each Flood Hub gauge \( g \) at metric position \( (x_g, y_g) \) with
severity ordinal \( s_g \in \{0, 1, 2, 3, 4\} \) (see §6 below):

\[
\mathcal{S}_g \;=\; \bigl\{ s : \mathrm{representative\_point}(s) \in
B_2(g; R) \bigr\}
\]

with \( R = 50{,}000 \) m (Euclidean disc, Phase 1 approximation).

We aggregate \( \{\mathrm{RVI}_s : s \in \mathcal{S}_g\} \) into three
statistics:

* `upstream_rvi_mean` \(= \overline{\mathrm{RVI}_s}\)
* `upstream_rvi_max`  \(= \max \mathrm{RVI}_s\)
* `upstream_rvi_p75`  \(= Q_{0.75}(\mathrm{RVI}_s)\)

The 75th percentile is the **primary signal**: it captures the worst-encroached
portion of the upstream area, which has the strongest hydraulic effect.

### 5.1 Phase 2 — true catchment

[`aggregate_upstream_catchment`](./src/rvi/analysis/validation.py) accepts a
pre-computed catchment polygon GeoDataFrame (one polygon per gauge) and uses
that instead of the Euclidean disc. Production pipelines obtain the polygons
from the Copernicus DEM via `pysheds` flow-direction + flow-accumulation
rasters; the function is decoupled from that step so the rest of the pipeline
can be unit-tested with synthetic catchments.

---

## 6. Severity ordinal encoding

Per §2.4.1 of the proposal, the integer encoding is exposed as the IntEnum
[`rvi.ingestion.floodhub.Severity`](./src/rvi/ingestion/floodhub.py):

| Flood Hub `severity`        | Integer |
|------------------------------|---------|
| `SEVERITY_UNSPECIFIED` / unknown | **0** |
| `NO_FLOODING`                | **1** |
| `ABOVE_NORMAL`               | **2** |
| `SEVERE`                     | **3** |
| `EXTREME`                    | **4** |

Because IntEnum subclasses `int`, every member is a real integer in any
arithmetic or pandas operation — no intermediate mapping step is needed.

---

## 7. Spearman ρ + bootstrap CI

We compute Spearman's rank correlation between the upstream RVI aggregate
(default `upstream_rvi_p75`) and the gauge severity ordinal:

\[
\hat\rho \;=\; \mathrm{spearmanr}\!\bigl(
  \{u_g\},\; \{s_g\}\bigr),\qquad
u_g = Q_{0.75}(\mathrm{RVI}_s : s \in \mathcal{S}_g)
\]

We pair NaNs out before correlation. The 95% confidence interval comes from
a non-parametric bootstrap with `Config.bootstrap_iterations` resamples
(default 1000) drawn with replacement from the gauge-level pairs:

\[
\hat\rho^{(b)} \;=\; \mathrm{spearmanr}\!\bigl(
  \{u_{g_i}\}_{i=1}^n,\; \{s_{g_i}\}_{i=1}^n\bigr)
\]

CI bounds are the empirical quantiles of \( \{\hat\rho^{(b)}\}_b \) at
\( \alpha/2 \) and \( 1 - \alpha/2 \) with \( \alpha = 0.05 \).

For Research Question 3 (proposal §4), we additionally stratify gauges by
their `quality_verified` flag and report \( \hat\rho \) separately for each
tier.

---

## 8. Outputs

All artefacts of a run land under `outputs/<run_name>/`:

```
outputs/<run_name>/
├── manifest.json                      # parameters used for the run
├── waterways.gpkg
├── segments.gpkg
├── encroachment_6m.gpkg
├── encroachment_10m.gpkg
├── encroachment_30m.gpkg
├── rvi_segments.gpkg                  # per-segment, all widths joined
├── upstream_6m.csv
├── upstream_10m.csv
├── upstream_30m.csv
├── gauges.gpkg                        # Flood Hub gauges + severity
├── gauge_statuses.csv                 # severity / severity_int / issued_time
├── rvi_segment_map.html               # Folium leaflet
└── rvi_severity_scatter.png           # Spearman annotated scatter
```

Each run's `manifest.json` records the bbox, all parameters, dataset row
counts, and the full correlation table — sufficient to reproduce the run
deterministically given the same input data.

---

## 9. Known approximations and conservative biases

| Approximation | Direction of bias |
|---------------|--------------------|
| OSM `waterway` completeness in arid counties | Under-counts segments (RVI is a *lower bound* in those areas). |
| Microsoft footprints reflect ~2021 imagery   | New post-2021 encroachments are missed. RVI is a lower bound. |
| Centreline → bank approximation              | Under-states bank position for the largest rivers. |
| Building footprint ≠ all structures (no walls / paving) | RVI is a lower bound on physical encroachment. |
| Phase-1 Euclidean upstream                   | Includes some non-drainage-connected segments. Replaced by Phase 2 catchments. |
| Bootstrap CI for ties / small *n*            | Standard Spearman caveats apply. |

These biases are documented in the `manifest.json` of every run and in §9
of the research proposal.

---

## 10. Citation

If you use RVI-Kenya in academic or policy work, please cite:

> Ashioya, V. (2026). *RVI-Kenya: A computational Riparian Violation Index
> for Flood Risk Quantification.* Open-source release at
> [github.com/ashioyajotham/rvi-kenya](https://github.com/ashioyajotham/rvi-kenya).
