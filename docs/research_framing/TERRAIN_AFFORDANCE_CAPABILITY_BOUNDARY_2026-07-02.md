# Terrain Affordance Capability Boundary

Date: 2026-07-02

## Purpose

This document separates what the current PUBG/Caldera research pipeline can
support, what it can support after additional implementation, and what should
not be claimed from the available data.

The framing goal is:

```text
not: arbitrary feature engineering
but: computational terrain-affordance representation for battle royale survival
```

## Core Claim We Can Make

We can claim that Battle Royale survival is shaped not only by player/team
statistics and movement kinematics, but also by the affordances of the shrinking
arena: visibility, concealment, traversability, elevation advantage, and
safe-zone-constrained routes.  This full claim is a PUBG-axis claim, because
PUBG has the phase clocks, outcomes, and richer telemetry needed for survival
and rank modeling.

Safe phrasing:

```text
For PUBG, we compute mechanic-aligned spatiotemporal terrain affordances and
use them as time-varying covariates for phase-wise survival and rank prediction.

For Caldera, we compute trajectory-grounded terrain-affordance measurements
and visualizations under low-observability Battle Royale telemetry.
```

Avoid phrasing:

```text
We directly apply military doctrine to PUBG.
We reconstruct the real player screen.
We measure true ballistic cover.
```

## Already Possible Now

These are supported by the current local Caldera world build, trajectory JSON,
mesh cache, and v0 feature extractor.

| Capability | Status | Evidence / artifact |
| --- | --- | --- |
| Align player trajectory to a z-up 3D Caldera world | possible | `match_31_player_25_local_world_object_core_zup_trajectory_grounded_workbench.png` |
| Build an object-preserved local world tile from OpenUSD mesh | possible | `match_31_player_25_local_world_object_core.obj` |
| Render analytic snapshots of world + trajectory | possible | Blender z-up renderer |
| Compute terrain ground reference | possible | `terrain_z_m`, `height_above_terrain_m` |
| Compute terrain obstacles / traversability proxies | possible | slope, relief, ruggedness |
| Compute key-terrain proxies | possible | multi-scale TPI / relative elevation |
| Compute avenue/corridor proxies | possible | road/runway distance |
| Compute built-mass proximity / coarse cover proxy | possible | building distance, building density |
| Aggregate sample features into a trajectory-window summary | possible | terrain-experience summary JSON |

Interpretive level currently supported:

```text
This player moved through compound-dominant terrain with measurable local
relief, terrain contact, and nearby built mass.
```

## Possible With v1 Implementation

These require ray casting or mesh visibility routines, but the object-preserved
z-up world makes them technically feasible.

| Capability | Required implementation | Claim level |
| --- | --- | --- |
| Forward open distance | ray cast along movement heading | immediate field-of-fire / openness proxy |
| Visible ray ratio | radial ray fan from eye height | local visual openness |
| Isovist area / radial profile | mesh ray fan + polar integration | player-perspective visible space |
| Nearest occluder distance | first ray hits around player | occlusion / concealment proxy |
| Pairwise LOS | ray cast between two player eye/torso points | edge feature / contact gate |
| Visibility-based exposure risk | pairwise LOS over enemy/opponent samples | time-varying exposure covariate |
| Player-perspective animation | camera frames sampled along trajectory | interpretive visualization, not real replay |

Safe claim after v1:

```text
We approximate local visibility and concealment from mesh-aware ray casting.
```

## Possible With v2 / Broader Data Integration

These are feasible but require either more data, a larger compute pass, or a
more careful causal design.

| Capability | Requirement | Notes |
| --- | --- | --- |
| Safe-zone route cost | zone timeline + cost surface | stronger for PUBG telemetry than current Caldera-only public layers |
| Minimum-exposure route residual | path optimization under same start/end and zone constraints | high-value contribution, heavier computation |
| Dyadic terrain advantage graph | same-match opponent/team positions + ray casting | edge feature, not homophily smoothing |
| Zone-aware terrain value field | terrain affordance field crossed with shrinking safe zone | Battle Royale-specific novelty |
| Full `br_repr` feature set | DB schema/migration + feature registry | needed for production model runs |
| Model ablation by affordance group | stable feature table + shared split | supports paper claims |

## Not Currently Possible Or Not Claimable

These should be explicitly avoided unless additional source data appears.

| Claim / capability | Why not |
| --- | --- |
| Actual player camera reconstruction | trajectory gives position, not mouse aim, FOV changes, recoil, or view direction |
| Exact game-engine replay | OpenUSD mesh is not the full live engine with all rendering, gameplay, and collision state |
| True ballistic cover | needs weapon ballistics, hitboxes, material penetration, stance, destructibility, and collision semantics |
| Exact concealment through vegetation/materials | mesh category and material semantics are incomplete for visual/thermal/foliage behavior |
| Exact building interior tactical topology | requires navigable interiors, doors, stairs, windows, collision, and access semantics |
| Exact player intent | trajectory and features describe context, not intent |
| Causal terrain effect without design | observational features support predictive/associational claims unless causal identification is added |
| Caldera-to-PUBG geometry equivalence | Caldera OpenUSD geometry and PUBG map geometry are different observability regimes |
| Caldera phase-wise survival hazard | current Caldera `br_ground.outcomes` has no `survived` labels and death is not an absorbing state |
| Caldera final-rank prediction | current Caldera `br_ground.outcomes.final_rank` is null for all ingested agents |
| Caldera safe-zone-constrained routes | current Caldera `br_ground.zone_states` has 0 rows |

## Label And Outcome Boundary

Local DB audit on 2026-07-02 confirms that the current Caldera ingest has:

| Fact | Value |
| --- | ---: |
| matches | 36 |
| agents | 4,187 |
| trajectory samples | 598,175 |
| outcomes rows | 4,187 |
| non-null `final_rank` | 0 |
| non-null `survived` | 0 |
| `zone_states` rows | 0 |
| agents with in-play `dt > 30s` gaps | 2,460 |
| in-play gaps with `dt > 30s` and XY displacement `> 500m` | 1,272 |

The Caldera `elimination_t` currently comes from the last observed breadcrumb
position and is not an official death time.  Many trajectories contain large
time/space discontinuities consistent with multi-life or redeploy-like play.
Therefore death cannot be treated as a single absorbing event without an
explicit life-segmentation layer.

Consequence:

```text
Caldera can support geometry and trajectory-experience claims now.
Caldera cannot support phase-wise survival or rank claims now.
```

## Conditional Boundaries

### Enemy / Team Exposure

Pairwise LOS and exposure are possible only when same-match opponent or team
position samples are available at the same clock. If not, use aggregate route
density or endpoint density as an exposure proxy rather than claiming true
enemy visibility.

### Safe-Zone Pressure

Safe-zone contraction is central to Battle Royale and must be a separate axis.
However, it can only be computed where zone timeline data is available.

For PUBG telemetry:

```text
distance to zone, time to shrink, cost-to-safe, zone pressure
```

For current Caldera public layers:

```text
trajectory-grounded terrain affordance first; zone-aware claims only if zone
state is ingested or reconstructed.
```

### 2.5D Terrain vs 3D Built Environment

Large-scale terrain features should be computed on terrain mesh or rasterized
terrain:

```text
slope, TPI, relief, ruggedness, openness
```

Local building/compound features should use mesh or obstacle-aware ray casting:

```text
isovist, occlusion, LOS, cover/concealment proxy
```

Do not collapse both into a single DEM and then claim building-level accuracy.

## Paper-Level Positioning

Recommended contribution statement:

```text
This study proposes a mechanic-aligned spatiotemporal terrain representation
for battle royale survival prediction. Instead of relying only on player
statistics or movement kinematics, we model the game map as a time-varying
terrain-affordance field consisting of visibility, concealment, traversability,
elevation advantage, and safe-zone-constrained routes. We then incorporate
these node-level and dyadic edge-level affordances into phase-wise survival and
final-rank prediction models on the PUBG high-observability axis. Caldera is
used as a low-observability spatial case study for terrain-affordance
measurement and visualization, not as an equivalent supervised survival label
source.
```

## Modeling Boundary

The affordance features should enter the model as time-varying covariates:

```text
lambda_i(t) = sigmoid(theta^T phi_i(t))
```

Node features:

- openness;
- TPI / relative elevation;
- slope / ruggedness;
- cover and concealment proxy;
- exposure integral;
- corridor and route-cost features;
- phase and safe-zone pressure.

Edge features:

- pairwise LOS;
- distance;
- directed relative height;
- visibility-weighted threat;
- recent combat where available;
- ally support where team data is available.

The graph should represent threat, advantage, co-pressure, combat, and support
relations. It should not be interpreted as a simple homophily graph where nearby
teams should be averaged together.

## Minimal Strong Set

For the first robust implementation, do not attempt everything at once. A strong
initial feature block is:

```text
terrain_z
slope
multi-scale TPI
multi-scale ruggedness
building/object distance
openness at 2-3 scales
safe-zone distance and time pressure where available
pairwise LOS within radius
relative height
recent combat edge where available
```

This set is enough to separate the work from kinematic-only Battle Royale
analytics while keeping implementation tractable.
