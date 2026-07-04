# Terrain Experience Research Lineage

Date: 2026-07-02

This note records the literature lineage for Caldera terrain-experience
features. The working goal is not to invent a feature taxonomy from scratch,
but to translate established military terrain analysis and spatial-visibility
methods into computable Battle Royale covariates.

## 1. Root Taxonomy: OCOKA / KOCOA and IPB

Military terrain analysis decomposes tactical terrain into five components:

```text
Observation and fields of fire
Cover and concealment
Obstacles
Key terrain
Avenues of approach
```

The acronym appears as `OCOKA`, `OAKOC`, or `KOCOA` depending on ordering and
doctrinal context. The definition lineage is:

- FM 5-33, Terrain Analysis (1990)
- FM 34-130, Intelligence Preparation of the Battlefield (1994)
- ATP 2-01.3, Intelligence Preparation of the Battlefield (2019)

For the paper, these field manuals should be used as definition sources, not
as peer-reviewed methodological claims. The safer methodological bridge is the
automated-IPB literature:

- Grindle, Lewis, Glinton, Giampapa, Owens, and Sycara (2004), "Automating
  Terrain Analysis: Algorithms for Intelligence Preparation of the Battlefield"
- related CMU HiLIFE / automated IPB work

The important formal idea is that IPB/MCOO products combine multiple terrain
layers into an operational overlay. In our representation, this becomes:

```text
mesh/trajectory observations -> feature functions -> tactical state / hazard covariates
```

Mapping to our feature families:

| OCOKA component | Caldera feature translation |
| --- | --- |
| Observation and fields of fire | `visible_ray_ratio`, `forward_open_distance`, `isovist_area`, pairwise LOS |
| Cover and concealment | `cover_building_r*`, `nearest_occluder_dist`, `concealment_score` |
| Obstacles | `terrain_slope_deg`, `local_relief_r*`, `terrain_ruggedness_r*`, movement cost |
| Key terrain | `relative_elevation_r*`, `high_ground_local_r*`, directed `delta_height_ij` |
| Avenues of approach | `road_or_runway_dist`, `on_corridor_proxy`, low-cost movement corridors |

## 2. Visibility: Viewshed, Openness, and Isovist

### Viewshed

For terrain point `p`, a target point `q` is visible when segment `pq` does not
intersect terrain or built geometry. The viewshed of `p` is the set of all
visible points.

Useful lineage:

- Franklin and Ray (1994), "Higher Isn't Necessarily Better: Visibility
  Algorithms and Experiments"
- Wheatley (1995), "Cumulative Viewshed Analysis"
- Llobera (2003), "Extending GIS-based visual analysis: the concept of
  visualscapes"

Computable Caldera forms:

```text
V(p, q) in {0, 1}
visible_ray_ratio_rL(p) = mean_d 1[ray(p, d, L) is unobstructed]
cumulative_observed_risk(x; S) = sum_{s in S} V(s, x)
```

The dyadic value `V(i, j)` should become an edge feature or graph gate, not a
node feature. The cumulative field can become a node exposure feature.

### Openness

Yokoyama, Shirasawa, and Pike (2002), "Visualizing Topography by Openness",
define positive and negative openness using radial terrain profiles. In words,
openness measures the angular openness of the surrounding terrain at an
explicit scale `L`.

Working feature:

```text
openness_L(x) = mean over directions of max/min vertical angle within radius L
```

This is a good bridge between terrain morphometry and combat-distance scale.
For Battle Royale, `L` should be treated as a parameter such as 50m, 100m,
250m, or a weapon/engagement-scale proxy.

### Isovist / Visibility Graph

For buildings, compounds, and close-range mesh geometry, DEM-style 2.5D
viewshed is not enough. Benedikt (1979) defines an isovist as the visible space
from a point:

```text
I_x = { y in E : segment(x, y) is contained in free space E }
```

Turner, Doxa, O'Sullivan, and Penn (2001) extend this into visibility graph
analysis. For Caldera, isovist-style metrics provide the player-perspective
features we need:

| Isovist metric | Player-experience interpretation |
| --- | --- |
| area / volume | openness, prospect |
| radial variance | uneven visibility, occlusion clutter |
| occlusivity | nearby occluding edges / concealment opportunities |
| forward isovist area | what the player is moving into |

Game-level link:

- Hullett and Whitehead (2010), FPS level design patterns such as sniper
  locations and chokepoints
- follow-up work on using isovists to evolve terrains with gameplay elements

These justify using isovist metrics as gameplay-space descriptors, not just
architectural metrics.

## 3. Cover and Concealment

Doctrine separates:

- `cover`: physical protection from fire;
- `concealment`: protection from observation.

In our current data, full ballistic cover is too strong a claim. The v1
definition should start with visibility-based concealment and local occluder
proxies:

```text
concealment(x; S) = 1 - mean_{s in S} V(s, x_torso)
nearest_occluder_dist(x) = min ray-hit distance around x
cover_proxy(x) = nearby vertical mass plus blocked lines of sight
```

Game-AI lineage:

- van der Sterren (2001), "Terrain Reasoning for 3D Action Games"
- Straatman, van der Sterren, and Beij (2005), "Killzone's AI: Dynamic
  Procedural Combat Tactics"

The game-AI position evaluation form is directly useful:

```text
E(x) = sum_k w_k phi_k(x, threats)
```

AI used this to select positions. We can reverse it: observed human positions
are explained by the same `phi_k` terrain and threat-context functions.

Path-level cover lineage:

- Marzouqi and Jarvis, visibility-based covert robotic navigation
- Lebeck, Molhave, and Agarwal (2013), highly occluded paths on terrain
- later cover-following and visibility-aware navigation systems

For our trajectory:

```text
exposure_integral(pi) = sum_t observed_risk(x_t) * delta_t
cover_use_residual = exposure_integral(pi_observed) - exposure_integral(pi_min_risk)
```

The residual is a later contribution candidate: it compares the chosen route to
an alternative route under the same start/end conditions.

## 4. Terrain Form and Movement Cost

Geomorphometry supplies the mathematical definitions behind elevation, slope,
curvature, ruggedness, and relative position.

Useful lineage:

- Evans (1972), general geomorphometry
- Zevenbergen and Thorne (1987), land surface topography from a quadratic local
  surface
- Weiss (2001), Topographic Position Index
- Riley, DeGloria, and Elliot (1999), Terrain Ruggedness Index
- Sappington, Longshore, and Thompson (2007), vector ruggedness measure

Current mesh-derived features:

```text
slope(x) = angle between local terrain normal and vertical
TPI_r(x) = z(x) - mean_{||y-x|| <= r} z(y)
relief_r(x) = max_{||y-x|| <= r} z(y) - min_{||y-x|| <= r} z(y)
ruggedness_r(x) = std_{||y-x|| <= r} z(y)
```

Key warning: all scales are semantic. A player may be high ground at `r=25m`
but low ground at `r=250m`. Therefore key-terrain and high-ground features
should be multi-scale.

Movement-cost lineage:

- Dijkstra (1959), shortest path
- GIS cost-distance analysis
- Tobler (1993), hiking function as a slope-speed model

PUBG/Warzone movement physics are not Tobler walking, but the structure is
still useful:

```text
movement_cost(x) = f(slope, roughness, corridor, obstacle, exposure)
```

For our data, observed speed drop, stop ratio, and turn rate can be modeled as
responses to this cost surface.

## 5. Direct Game Analytics Neighbors

Closest neighbors to verify and cite carefully before final manuscript:

- Activision/Warzone Caldera telemetry work that compares player trajectories
  by kinematic measures such as distance, entropy, acceleration, and jerk.
- Battle Royale strategy work that uses phase-aware survival modeling and
  hunter/camper style categories via scores such as FightScore and ActiveScore.
- Melhart-style PUBG telemetry work using conventional telemetry features for
  engagement prediction.
- Drachen and Schubert (2013), spatial game analytics and visualization,
  especially the critique that many game spatial analyses remain heatmap-like
  aggregate visualizations.

Our planned distinction:

```text
prior game telemetry: movement/combat/resource summaries
this project: trajectory interpreted through terrain visibility, cover, key terrain,
and avenue/obstacle structure
```

Sports-space value models are a secondary analogy:

- Fernandez and Bornn (2018), "Wide Open Spaces"
- Spearman (2018), pitch control

These show how trajectories can induce continuous spatial value fields. The
Battle Royale analogue is a zone- and visibility-constrained arena value field.

## 6. Implementation Rule From The Literature

Most viewshed/openness/TPI literature assumes a 2.5D terrain surface. Caldera
has multi-level buildings and detailed mesh structures. We should therefore
split the implementation:

| Scale | Geometry model | Feature families |
| --- | --- | --- |
| terrain scale | rasterized terrain or terrain mesh | slope, TPI, relief, ruggedness, openness |
| compound/building scale | z-up mesh ray casting | isovist, occlusion, cover/concealment |
| relation scale | agent/team pairs | LOS, relative height, mutual visibility |
| path scale | trajectory windows | exposure integral, corridor use, movement-cost response |

This split should be explicit in the paper. It avoids overclaiming exact
game-engine reconstruction while making full use of the OpenUSD mesh.

## 7. Hazard-Model Placement

All terrain-experience variables are time-varying covariates:

```text
lambda_i(t) = sigmoid(theta^T phi_i(t))
```

Node features:

- openness;
- TPI / relative elevation;
- slope / ruggedness;
- cover proxy;
- exposure integral to date;
- corridor/avenue use.

Edge features:

- pairwise line of sight `V(i,j)`;
- directed relative height `delta_h_ij = z_i - z_j`;
- distance and shared corridor relation;
- mutual exposure / contact gate.

The design is complete only when each feature is assigned to node, edge, or
global phase context with an explicit causal clock.

## Reading Order

Recommended reading order for implementation:

1. Grindle et al. 2004, automated IPB.
2. Yokoyama et al. 2002, openness.
3. Wheatley 1995 and Llobera 2003, cumulative viewshed / visualscapes.
4. Straatman et al. 2005, game-AI cover position evaluation.
5. Marzouqi and Jarvis 2006, covert visibility-based path planning.
6. Caldera/BR game-analytics neighbors for positioning the contribution.

## Bibliographic Caution

This note is a lineage map. Before manuscript submission, verify exact titles,
venues, page numbers, and availability for every non-doctrine citation,
especially the newest Warzone/Entertainment Computing neighbors.
