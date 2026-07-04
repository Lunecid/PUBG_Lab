# Terrain Experience Feature Design

Date: 2026-07-02

## Purpose

The Caldera 3D world is not a visualization endpoint. It is a measurement
surface for converting a player trajectory into an interpretable tactical
experience:

```text
3D terrain / buildings
-> player path in x/y/z/t
-> terrain, visibility, cover, and movement-pressure features
-> trajectory-experience figures and, where labels exist, phase/window models
```

This note translates military terrain-analysis concepts and spatial-visibility
literature into feature families that are computable from the current Caldera
OpenUSD mesh and trajectory database.

Current Caldera boundary:

```text
Caldera supports terrain-affordance measurement and visualization now.
Caldera does not currently support phase-wise survival or final-rank modeling.
```

Companion lineage note:

```text
docs/feature_engineering/TERRAIN_EXPERIENCE_RESEARCH_LINEAGE_2026-07-02.md
```

Use this document for the feature contract and implementation plan. Use the
lineage note for doctrine roots, paper families, and manuscript reading order.

## Literature Anchors

### Military Terrain Framing: OAKOC / OCOKA

Military terrain analysis often decomposes terrain into:

- observation and fields of fire;
- avenues of approach;
- key terrain;
- obstacles;
- cover and concealment.

The definition root is the terrain-analysis / intelligence-preparation
doctrine lineage: FM 5-33, FM 34-130, and ATP 2-01.3. These are doctrine
sources, not peer-reviewed method claims. The better computational bridge is
automated IPB work such as Grindle et al. (2004), which turns intelligence
preparation of the battlefield into algorithmic terrain overlays.

For our project, OAKOC should not be copied as labels. It should become a
feature taxonomy:

| OAKOC axis | Caldera interpretation | Feature role |
| --- | --- | --- |
| Observation / fields of fire | What can be seen from the player pose, and how far open space extends | exposure / prospect |
| Avenues of approach | Roads, runways, low-slope corridors, repeated route channels | movement affordance |
| Key terrain | Locally high ground, ridge control, roof/platform control | positional advantage |
| Obstacles | Steep slope, rough relief, walls/building masses, terrain occlusion | mobility cost |
| Cover / concealment | Nearby vertical mass, terrain defilade, blocked line of sight | protection / refuge |

In implementation terms, this is an MCOO-like multi-layer function over the
mesh and trajectory:

```text
f_tactical(x, t) = g(visibility, cover, slope, relief, relative_elevation, corridor, built_mass)
```

### Terrain Morphometry

Geomorphometry provides the low-level measurements needed to characterize
surface form: slope, aspect, relief, curvature, roughness, local extrema, and
surface-normal structure. For our mesh setting, these are derived directly from
triangles rather than from a raster DEM.

Relevant concepts:

- slope from local terrain surface normal;
- local relief from elevation range within a radius;
- relative elevation from player terrain height minus neighborhood mean;
- roughness/ruggedness from neighborhood elevation dispersion or vector
  ruggedness;
- local maxima/ridges as high-ground candidates.

The important design rule from TPI/openness/ruggedness work is that scale is
part of the feature definition. High ground at 25m can be low ground at 250m,
so v0 and later implementations should keep multi-scale fields rather than
collapsing them into one "height advantage" value.

### Viewshed and Line-of-Sight

Viewshed analysis defines what terrain cells are visible from a viewpoint. For
large terrains, visibility complexity depends heavily on terrain form and the
chosen interpolation/horizon model. This matters because the player experience
we want is not just "near building" but "what could the player see or be seen
from at that point?"

In our implementation, the initial viewshed should be local and approximate:

- ray cast from eye height against terrain/building mesh;
- compute open distance in the heading direction;
- compute visible sector fraction within a fixed radius;
- compute occlusion count and nearest occluder class.
- compute pairwise line of sight for relation features when opponent/team
  positions are available.

### Isovist and Visibility Graphs

Isovist analysis describes the ego-centric visible space from a viewpoint. This
is directly aligned with our first-person snapshot plan. Visibility graph
analysis extends this idea by connecting mutually visible cells or points.

For Caldera features:

- isovist area / visible volume becomes "local visual openness";
- isovist perimeter or fragmentation becomes "visual clutter / occlusion";
- forward isovist along the movement heading becomes "approach exposure";
- visibility graph centrality can later identify key observation positions.

### Movement Cost and Avenues

Movement through terrain is shaped by slope, roughness, and corridor structure.
We do not need to claim exact game movement physics. A causal movement-pressure
proxy is enough:

- slope cost from current/forward terrain grade;
- roughness cost from local relief;
- road/runway proximity as corridor affordance;
- building/terrain channeling as corridor constriction;
- speed drop or turn-rate increase as behavioral response.

### Cover Evaluation and Exposure Paths

The cover/concealment axis should be computed as visibility-aware position
evaluation, not only as "nearest building". The first working proxy can use
nearby vertical mass. The v1/v2 version should use ray casts:

```text
concealment(x; S) = 1 - mean_{s in S} LOS(s, x_torso)
exposure_integral(pi) = sum_t observed_risk(x_t) * delta_t
```

This connects military cover/concealment, game-AI tactical position scoring,
and covert path-planning. In our model, the same terms become time-varying
hazard covariates.

## Feature Family Contract

Every feature must declare:

| Field | Requirement |
| --- | --- |
| owner | player sample, player window, team phase, or relation |
| clock | trajectory sample, trajectory window, or phase cut |
| spatial reference | Caldera z-up mesh coordinates |
| causal window | only current or past trajectory samples before the prediction cut |
| source profile | mesh cache/object profile used to compute terrain/building geometry |

No feature in this family should use future survival state, later endpoint
position, or post-hoc combat outcomes unless it is explicitly an evaluation
label or outcome-aligned analysis variable.

## v0 Computable Feature Families

These can be computed from the current local raw mesh cache and trajectory
samples without a full game-engine reconstruction.

### TE1. Ground Contact and Vertical Position

| Feature | Grain | Definition | Interpretation |
| --- | --- | --- | --- |
| `terrain_z_m` | sample | terrain surface z under/near player x/y | ground reference |
| `height_above_terrain_m` | sample | `player_z - terrain_z_m` | airborne / roof / slope mismatch |
| `relative_elevation_r25_m` | sample | terrain z minus mean terrain z within 25m | immediate high/low ground |
| `relative_elevation_r50_m` | sample | terrain z minus mean terrain z within 50m | local high/low ground |
| `relative_elevation_r100_m` | sample | terrain z minus mean terrain z within 100m | compound-scale high/low ground |
| `high_ground_local_r50` | sample | relative elevation above threshold | key-terrain proxy |
| `high_ground_local_r100` | sample | relative elevation above threshold | larger key-terrain proxy |

### TE2. Obstacles and Mobility Cost

| Feature | Grain | Definition | Interpretation |
| --- | --- | --- | --- |
| `terrain_slope_deg` | sample | slope of nearest terrain triangle | movement obstacle |
| `local_relief_r25_m` | sample | terrain z range within 25m | micro ruggedness |
| `local_relief_r50_m` | sample | terrain z range within 50m | macro ruggedness |
| `local_relief_r100_m` | sample | terrain z range within 100m | compound-scale ruggedness |
| `terrain_ruggedness_r25_m` | sample | terrain z std dev within 25m | roughness proxy |
| `terrain_ruggedness_r50_m` | sample | terrain z std dev within 50m | roughness proxy |
| `terrain_ruggedness_r100_m` | sample | terrain z std dev within 100m | roughness proxy |
| `forward_grade` | segment | current terrain z minus previous terrain z, divided by horizontal distance | climb/descent pressure |
| `speed_mps` | segment/sample | causal speed from previous sample | behavioral response |
| `turn_rate_deg_s` | sample | heading change per second | path deflection / searching |

Motion features must not bridge redeploy-like discontinuities.  The v0
extractor suppresses speed, heading-change, and forward-grade values when
`dt_s > 30s`, and marks conservative redeploy candidates when
`dt_s > 30s` and XY displacement exceeds `500m`.

### TE3. Avenues of Approach

| Feature | Grain | Definition | Interpretation |
| --- | --- | --- | --- |
| `road_dist_m` | sample | nearest road triangle centroid distance | route corridor |
| `runway_dist_m` | sample | nearest runway triangle centroid distance | open corridor |
| `road_or_runway_dist_m` | sample | min of road/runway distance | movement affordance |
| `on_corridor_proxy` | sample | near road/runway threshold | avenue use |

### TE4. Cover, Concealment, and Built Mass

| Feature | Grain | Definition | Interpretation |
| --- | --- | --- | --- |
| `building_dist_m` | sample | nearest building triangle centroid distance | nearest vertical mass |
| `building_tri_density_r25_ha` | sample | building triangle centroids per hectare within 25m | clutter / cover density |
| `building_tri_density_r50_ha` | sample | building triangle centroids per hectare within 50m | compound density |
| `cover_building_r10` | sample | building within 10m | cover proxy |
| `cover_building_r25` | sample | building within 25m | concealment / compound proxy |

### TE5. Observation and Exposure

These require ray casting and should be v1 after v0 is stable.

| Feature | Grain | Definition | Interpretation |
| --- | --- | --- | --- |
| `forward_open_distance_m` | sample | first mesh intersection along movement heading | immediate field of view |
| `visible_ray_ratio_r100` | sample | fraction of rays reaching 100m unobstructed | visual openness |
| `openness_r100` | sample | mean angular openness over directions | terrain openness |
| `nearest_occluder_dist_m` | sample | nearest ray hit in horizontal sector | cover/exposure boundary |
| `isovist_area_r100_m2` | sample | polar visible area approximation | observation potential |
| `terrain_defilade_score` | sample | local horizon shielding behind terrain | concealment by relief |
| `pairwise_los_ij` | directed edge | ray from eye/torso of i to j | contact gate / observed threat |
| `relative_height_ij_m` | directed edge | `terrain_or_eye_z_i - terrain_or_eye_z_j` | directed height advantage |

### TE6. Window Aggregates

Sample features should be aggregated into causal trajectory windows:

| Aggregate | Definition |
| --- | --- |
| `mean_terrain_slope_deg` | mean over window |
| `p90_local_relief_r50_m` | high ruggedness exposure |
| `p90_terrain_ruggedness_r100_m` | broad obstacle exposure |
| `high_ground_ratio_r50` | fraction of samples on local high ground |
| `high_ground_ratio_r100` | fraction of samples on compound-scale high ground |
| `cover_building_r10_ratio` | fraction near immediate cover |
| `corridor_ratio` | fraction near road/runway |
| `exposure_integral` | v1/v2 ray-cast observed-risk integral |
| `height_above_terrain_median` | ground contact quality |
| `terrain_experience_entropy` | entropy over coarse tactical states |

## Tactical State Buckets

The canonical tactical-state vocabulary is owned by:

```text
docs/feature_engineering/BATTLE_ROYALE_FEATURE_ONTOLOGY_2026-07-02.md
```

For figures and interpretable ablations, sample-level metrics can be bucketed
using those names:

| State | Rule sketch |
| --- | --- |
| `open_high_ground` | high relative elevation, low building density |
| `covered_high_ground` | high relative elevation, nearby building mass |
| `open_low_ground` | low relative elevation, low cover density |
| `compound_movement` | high building density or near building |
| `corridor_movement` | near road/runway and low slope |
| `rough_mobility` | high slope or high local relief |
| `neutral_or_unclassified` | no stronger tactical-state condition is observed |
| `airborne_or_unresolved` | high height above terrain or sparse terrain lookup |

These states are not labels of player intent. They are context states that
describe the terrain/built environment encountered by the observed trajectory.

## Implementation Plan

1. Build v0 sample-level features from the local mesh cache: terrain height,
   slope, multi-scale relief, multi-scale ruggedness, multi-scale relative
   elevation/TPI, road/runway distance, building distance, and building density.
2. Validate the features on `caldera:match_31:player_25` against the z-up
   Blender render.
3. Aggregate sample features into trajectory-window features.
4. Add v1 ray-cast features using Blender or a mesh-ray library: forward open
   distance, visible sector ratio, nearest occluder, pairwise LOS, and isovist
   approximation.
5. Register a rebuildable `br_repr` feature set once the local prototype is
   stable.

## Source Notes

Use `docs/feature_engineering/TERRAIN_EXPERIENCE_RESEARCH_LINEAGE_2026-07-02.md` as the working
source map. The manuscript bibliography should verify exact metadata, but the
active lineage is:

- doctrine root: FM 5-33, FM 34-130, ATP 2-01.3;
- computational IPB: Grindle et al. 2004 and related HiLIFE work;
- viewshed / visual fields: Franklin and Ray 1994, Wheatley 1995, Llobera
  2003;
- openness: Yokoyama, Shirasawa, and Pike 2002;
- isovist / VGA: Benedikt 1979, Turner et al. 2001;
- cover and tactical position evaluation: van der Sterren 2001, Straatman et
  al. 2005;
- terrain form: Evans 1972, Zevenbergen and Thorne 1987, Weiss 2001, Riley et
  al. 1999, Sappington et al. 2007;
- movement-cost surfaces: Dijkstra 1959, Tobler 1993, GIS cost-distance;
- nearest game analytics: Warzone/Caldera trajectory analytics, phase-aware BR
  survival strategy papers, and spatial game analytics/visualization.
