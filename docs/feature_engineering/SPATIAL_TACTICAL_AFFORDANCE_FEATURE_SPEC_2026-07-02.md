# Spatial-Tactical Affordance Feature Spec

Date: 2026-07-02

## Purpose

This document turns the terrain-analysis and tactical-affordance discussion
into an implementation-facing feature design for the PUBG / Caldera project.

It is a subdesign of the genre-level ontology:

```text
docs/feature_engineering/BATTLE_ROYALE_FEATURE_ONTOLOGY_2026-07-02.md
-> O6. Spatial / Terrain / Environment Affordance
```

The core constraint is asymmetric observability:

```text
PUBG    = label-rich, telemetry-rich, geometry-poor
Caldera = geometry-rich, trajectory-rich, label-poor
```

The broader ontology is not 3D-only.  This document covers the PUBG/Caldera
instantiation, while preserving a mask for 2D or grid-based Battle Royale games.

Therefore the design separates:

- `geometry-native features`: computable from Caldera mesh and trajectories;
- `PUBG-observable proxies`: computable from PUBG telemetry, zone state,
  player/team positions, and combat events before the prediction cut;
- `bridge constructs`: the shared tactical concept that makes the two sides
  comparable without pretending they expose the same data.

## Design Thesis

We do not directly import military doctrine as labels.  We use OCOKA / OAKOC
as a feature ontology:

```text
Observation / fields of fire
Cover and concealment
Obstacles
Key terrain
Avenues of approach
```

For Battle Royale, this ontology must be extended with:

```text
Shrinking-zone pressure
Opponent contact / threat field
Team survival capital
```

The final model should learn survival risk from telemetry-observable tactical
affordances, not from unsupported claims about true terrain, true cover, or
exact player intent.

## Observability Classes

| Class | Meaning | Current source |
| --- | --- | --- |
| `GEO_NATIVE` | requires terrain/building geometry | Caldera only |
| `PUBG_NATIVE` | directly available in PUBG telemetry/DB | PUBG only |
| `SHARED_PROXY` | same construct, different measurement on each side | PUBG + Caldera |
| `FUTURE_ASSET` | requires new PUBG map/height/mesh asset | not current |

Every feature must declare one of these classes.  Unsupported PUBG terrain
features must be named as proxies, not as direct terrain measurements.

## Spatial Profile Classes

Every spatial-tactical feature also declares which dimensionality profile can
support it.

| Profile | Meaning | Valid feature examples |
| --- | --- | --- |
| `PLANAR_2D` | `x/y` arena without meaningful height | boundary distance, local density, route conflict, chokepoint pressure |
| `GRID_2D` | tile/cell/sector map | safe cell, adjacent threat, obstacle cell, sector control |
| `ELEVATION_2_5D` | `x/y` plus height/layer but limited geometry | relative height, vertical movement, layer advantage |
| `FULL_3D` | full position and view/physics geometry | line of sight, terrain slope, cover occlusion, isovist |
| `GEOMETRY_RICH` | external mesh/map asset exists | building distance, ruggedness, occluder density |
| `GEOMETRY_POOR` | telemetry only | behavior, contact, density, and movement proxies |

Dimensionality guard:

```text
Do not require vertical, terrain, LOS, or mesh features for a general Battle
Royale claim.  Those are optional projections when the game/data profile
supports them.
```

## Feature Families

### STA1. Observation And Fields Of Fire

Tactical meaning:

```text
How much space can an agent observe, and how exposed is the agent to others?
```

Caldera geometry-native features:

| Feature | Grain | Definition | Class |
| --- | --- | --- | --- |
| `forward_open_distance_m` | player sample | first mesh hit along movement heading | `GEO_NATIVE` |
| `visible_ray_ratio_r100` | player sample | fraction of radial rays unobstructed to 100m | `GEO_NATIVE` |
| `isovist_area_r100_m2` | player sample | polar visible area from eye-height ray fan | `GEO_NATIVE` |
| `nearest_occluder_dist_m` | player sample | nearest ray-hit distance around player | `GEO_NATIVE` |
| `pairwise_los_ij` | directed pair | mesh ray from i eye/torso to j eye/torso | `GEO_NATIVE` |

PUBG-observable proxies:

| Feature | Grain | Definition | Class |
| --- | --- | --- | --- |
| `contact_gate_ij` | team pair / snapshot | 1 if proximity or recent combat implies plausible mutual contact | `PUBG_NATIVE` |
| `enemy_density_r100` | team snapshot/window | opponents within 100m or local radius | `PUBG_NATIVE` |
| `nearest_enemy_dist_m` | team snapshot/window | nearest alive opponent distance | `PUBG_NATIVE` |
| `contact_exposure_time_s` | team phase | time under contact gate during phase/window | `PUBG_NATIVE` |
| `observed_threat_proxy` | team phase | weighted sum of nearby/contacted enemy presence | `SHARED_PROXY` |

2D-compatible projections:

| Feature | Grain | Definition | Profile |
| --- | --- | --- | --- |
| `planar_open_exposure_proxy` | team window | low obstacle/sector shielding or high enemy approach count if map labels exist | `PLANAR_2D` / `GRID_2D` |
| `local_enemy_density_r` | team window | alive opponents within radius/adjacent cells | `PLANAR_2D` / `GRID_2D` |

Model role:

- node exposure covariate;
- edge gate for relation construction;
- ablation family `observation_proxy`.

Claim boundary:

```text
In PUBG these are opponent-contact exposure proxies, not true line of sight.
```

### STA2. Cover And Concealment

Tactical meaning:

```text
How much protection or visual shielding does the local environment provide?
```

Caldera geometry-native features:

| Feature | Grain | Definition | Class |
| --- | --- | --- | --- |
| `building_dist_m` | player sample | nearest building triangle centroid distance | `GEO_NATIVE` |
| `building_tri_density_r25_ha` | player sample | building triangle centroids per hectare within 25m | `GEO_NATIVE` |
| `cover_building_r10` | player sample | building mass within 10m | `GEO_NATIVE` |
| `concealment_ray_block_ratio` | player sample | fraction of threat rays blocked by mesh | `GEO_NATIVE` |
| `occluder_ring_density` | player sample | nearby occluder hits around player | `GEO_NATIVE` |

PUBG-observable proxies:

| Feature | Grain | Definition | Class |
| --- | --- | --- | --- |
| `damage_while_contacted` | team window | damage taken during contact-gated interval | `PUBG_NATIVE` |
| `contact_time_without_damage_s` | team window | pre-cut time in contact gate without damage taken | `PUBG_NATIVE` |
| `stationary_under_contact_ratio` | team window | low-speed ratio while near/contacted by enemies | `PUBG_NATIVE` |
| `cover_behavior_proxy` | team window | low speed + contact + low damage composite | `SHARED_PROXY` |

2D-compatible projections:

| Feature | Grain | Definition | Profile |
| --- | --- | --- | --- |
| `sector_cover_proxy` | team/window | map-labeled cover/obstacle/indoor sector if available | `GRID_2D` |
| `contact_without_damage_ratio` | team/window | contact-gated time without damage | `PLANAR_2D` / `GEOMETRY_POOR` |

Model role:

- behavioral proxy for defensive positioning;
- interaction with zone pressure and enemy density;
- not a direct environmental feature in PUBG.

Claim boundary:

```text
PUBG cannot identify actual cover objects.  It can only observe behavior and
damage outcomes consistent with possible cover use.
```

### STA3. Obstacles And Mobility Cost

Tactical meaning:

```text
How hard is the terrain/path to traverse?
```

Caldera geometry-native features:

| Feature | Grain | Definition | Class |
| --- | --- | --- | --- |
| `terrain_slope_deg` | player sample | slope of nearest terrain triangle | `GEO_NATIVE` |
| `local_relief_r50_m` | player sample | z range within 50m | `GEO_NATIVE` |
| `terrain_ruggedness_r100_m` | player sample | local z dispersion within 100m | `GEO_NATIVE` |
| `forward_grade` | segment | current terrain z minus previous terrain z over XY step | `GEO_NATIVE` |
| `movement_cost_proxy_mesh` | sample/window | weighted slope + ruggedness + corridor absence | `GEO_NATIVE` |

PUBG-observable proxies:

| Feature | Grain | Definition | Class |
| --- | --- | --- | --- |
| `vertical_speed_abs_mean` | team window | mean absolute `dz/dt` | `PUBG_NATIVE` |
| `z_range_m` | team window | max `pos_z` minus min `pos_z` | `PUBG_NATIVE` |
| `speed_drop_after_climb` | team window | speed reduction after positive z change | `SHARED_PROXY` |
| `turn_rate_under_rotation` | team window | heading instability while rotating to zone | `PUBG_NATIVE` |
| `movement_friction_proxy` | team window | low speed + high vertical change + zone urgency | `SHARED_PROXY` |

2D-compatible projections:

| Feature | Grain | Definition | Profile |
| --- | --- | --- | --- |
| `planar_detour_ratio` | team window | path length over displacement under zone pressure | `PLANAR_2D` |
| `grid_path_cost_proxy` | team window | traversed obstacle/slow/unsafe cells if map labels exist | `GRID_2D` |

Model role:

- node movement-pressure block;
- phase-window aggregate;
- interaction with required speed to safe zone.

Claim boundary:

```text
PUBG obstacle features are observed movement responses, not measured terrain
slope or ruggedness.
```

### STA4. Key Terrain And Relative Advantage

Tactical meaning:

```text
Which positions provide local advantage, especially elevation advantage?
```

Caldera geometry-native features:

| Feature | Grain | Definition | Class |
| --- | --- | --- | --- |
| `relative_elevation_r50_m` | player sample | local terrain z minus 50m neighborhood mean | `GEO_NATIVE` |
| `relative_elevation_r100_m` | player sample | local terrain z minus 100m neighborhood mean | `GEO_NATIVE` |
| `high_ground_local_r50` | player sample | relative elevation above calibrated threshold | `GEO_NATIVE` |
| `delta_height_ij_m` | directed pair | i terrain/eye z minus j terrain/eye z | `GEO_NATIVE` |

PUBG-observable proxies:

| Feature | Grain | Definition | Class |
| --- | --- | --- | --- |
| `team_z_mean` | team snapshot/window | team mean `pos_z` | `PUBG_NATIVE` |
| `team_z_percentile_alive` | team snapshot/window | z percentile among alive teams in match/phase | `PUBG_NATIVE` |
| `z_advantage_vs_nearest_enemy_m` | team snapshot/window | team z minus nearest/contacted enemy z | `PUBG_NATIVE` |
| `contact_delta_z_ij_m` | directed pair | z_i minus z_j under contact gate | `PUBG_NATIVE` |
| `high_ground_contact_advantage` | directed pair/window | positive contact delta-z with enemy proximity/combat | `SHARED_PROXY` |

2D-compatible projections:

| Feature | Grain | Definition | Profile |
| --- | --- | --- | --- |
| `sector_positional_advantage` | team/window | favorable sector relative to safe area, cover, route convergence, or enemy approach vectors | `PLANAR_2D` / `GRID_2D` |
| `zone_relative_advantage_2d` | team/window | distance/angle advantage to safe center or boundary under pressure | `PLANAR_2D` |

Model role:

- primary PUBG-ready terrain-inspired feature family;
- edge feature for relation-aware GNN;
- already supported by existing `pos_z` / advantage-edge work.

Claim boundary:

```text
In PUBG, "high ground" means observed relative player/team altitude, not
verified terrain high ground.
```

### STA5. Avenues Of Approach And Rotation Corridors

Tactical meaning:

```text
Which movement routes are naturally used, forced, or contested?
```

Caldera geometry-native features:

| Feature | Grain | Definition | Class |
| --- | --- | --- | --- |
| `road_or_runway_dist_m` | player sample | nearest road/runway geometry distance | `GEO_NATIVE` |
| `on_corridor_proxy` | player sample | near road/runway and low obstacle cost | `GEO_NATIVE` |
| `corridor_ratio` | trajectory window | fraction of samples on corridor proxy | `GEO_NATIVE` |
| `channel_constriction_score` | sample/window | low openness + high built mass near path | `GEO_NATIVE` |

PUBG-observable proxies:

| Feature | Grain | Definition | Class |
| --- | --- | --- | --- |
| `zone_entry_time` | team phase | first entry to next/safe zone | `PUBG_NATIVE` |
| `required_speed_to_safe_mps` | team phase | distance to safe area over time remaining | `PUBG_NATIVE` |
| `rotation_path_density_r100` | team window | count of other team paths near current route | `PUBG_NATIVE` |
| `route_conflict_count` | team phase | number of opponent trajectories intersecting same corridor/time band | `PUBG_NATIVE` |
| `rotation_corridor_pressure` | team phase | path density + zone urgency + enemy proximity | `SHARED_PROXY` |

2D-compatible projections:

| Feature | Grain | Definition | Profile |
| --- | --- | --- | --- |
| `chokepoint_pressure` | team/window | path or cell convergence under shrinking zone | `PLANAR_2D` / `GRID_2D` |
| `sector_route_conflict` | team/window | other teams crossing same sector/cell band before cut | `PLANAR_2D` / `GRID_2D` |

Model role:

- node route-pressure feature;
- optional edge co-pressure feature;
- phase-specific effect expected in early/mid rotations.

Claim boundary:

```text
PUBG route corridors are empirical movement corridors, not road/terrain
corridors unless map geometry is added later.
```

Perspective boundary:

```text
rotation_path_density_r100 and route_conflict_count are analyst-view telemetry
features. They use pre-cut all-team trajectories, so they are causal for
supervised analysis, but they are not player-available perception features.
```

### STA6. Shrinking-Zone Tactical Pressure

Tactical meaning:

```text
How does the shrinking arena force movement, exposure, and route conflict?
```

Caldera status:

Current Caldera has no ingested zone timeline.  This family is masked for
Caldera until zone state is reconstructed or externally added.

PUBG-native features:

| Feature | Grain | Definition | Class |
| --- | --- | --- | --- |
| `dist_blue_boundary_m` | snapshot/window | signed distance to current damaging boundary | `PUBG_NATIVE` |
| `inside_blue` | snapshot/window | whether team is inside playable/safe relation frame | `PUBG_NATIVE` |
| `dist_target_center_m` | snapshot/window | distance to target safe-zone center | `PUBG_NATIVE` |
| `outside_safe_duration_s` | phase/window | time outside safe area | `PUBG_NATIVE` |
| `hab_required_speed_to_safe_mps` | phase/window | required migration speed | `PUBG_NATIVE` |
| `hab_zone_gap_pressure` | phase/window | safe-zone displacement pressure | `PUBG_NATIVE` |

Model role:

- global context and node feature;
- interaction term for all terrain-inspired proxies;
- essential confound/control for movement and contact features.

Claim boundary:

```text
Zone pressure is a core PUBG survival mechanism.  It is not currently a
Caldera claim.
```

### STA7. Tactical State Summary

The canonical tactical-state vocabulary is owned by:

```text
docs/feature_engineering/BATTLE_ROYALE_FEATURE_ONTOLOGY_2026-07-02.md
```

The model should not only ingest raw scalar features.  It should also support
interpretable, coarse states for figures and ablations using that vocabulary.

| State | Caldera rule | PUBG proxy rule |
| --- | --- | --- |
| `open_high_ground` | high relative elevation + high openness | high z percentile + low enemy density |
| `covered_high_ground` | high relative elevation + nearby occluder/building | high z advantage + contact time without damage |
| `open_low_ground` | low relative elevation + high openness | low z percentile + enemy density/contact |
| `compound_movement` | high built-mass density | not directly available; mask or behavior proxy only |
| `corridor_movement` | road/runway/corridor use | high route density + zone urgency |
| `rough_mobility` | high slope/relief/ruggedness | high vertical movement + speed drop |
| `contact_under_pressure` | LOS/contact + route/zone pressure | combat/proximity + zone pressure |
| `neutral_or_unclassified` | no stronger condition is observed | no stronger proxy condition is observed |
| `airborne_or_unresolved` | height above terrain or lookup failure | not generally available in PUBG |

These states are not labels of intention.  They are interpretable bins over
measured or proxied tactical context.

## Minimal PUBG-Ready Feature Block

This is the first block that can be used in supervised PUBG survival/rank
experiments without adding new map assets.

| Block | Features |
| --- | --- |
| vertical position | `team_z_mean`, `team_z_percentile_alive`, `z_range_m`, `vertical_speed_abs_mean` |
| relative advantage | `contact_delta_z_ij_m`, `z_advantage_vs_nearest_enemy_m`, `high_ground_contact_advantage` |
| contact exposure | `nearest_enemy_dist_m`, `enemy_density_r100`, `contact_exposure_time_s`, `observed_threat_proxy` |
| zone pressure | `dist_blue_boundary_m`, `outside_safe_duration_s`, `required_speed_to_safe_mps`, `hab_zone_gap_pressure` |
| rotation corridor proxy | `rotation_path_density_r100`, `route_conflict_count`, `rotation_corridor_pressure` |
| behavior under pressure | `stationary_under_contact_ratio`, `damage_while_contacted`, `movement_friction_proxy` |

Suggested ablation order:

1. `Z0`: existing baseline without this block.
2. `Z1`: vertical position only.
3. `Z2`: vertical + contact-gated relative height.
4. `Z3`: `Z2` + zone-pressure interactions.
5. `Z4`: `Z3` + rotation corridor proxies.
6. `Z5`: `Z4` + behavior-under-pressure proxies.

The headline test should be whether contact-gated relative-height and
zone-conditioned rotation proxies add information beyond absolute `pos_z` and
standard movement features.

## Causal And Leakage Guard

For supervised PUBG runs, every feature must be computed only from information
available before the prediction cut.

Allowed before a phase cut:

- current/past positions;
- current/past zone states;
- current/past proximity and combat events;
- current/past health/alive state if the experiment explicitly allows roster
  survival capital.

Not allowed as model input:

- final rank;
- future survival state;
- future damage/combat after the prediction cut;
- phase-end outcomes used before that phase has ended;
- Caldera endpoint or last-observed position as a death label.

Behavior-under-pressure proxies are high-risk and should be ablated separately.
If they dominate results, report them as combat/roster mechanisms rather than
as terrain mechanisms.

## Caldera Construct-Validation Block

Caldera should validate the construct vocabulary, not the survival label.

Recommended Caldera outputs:

| Output | Purpose |
| --- | --- |
| 1-person mesh snapshot | qualitative validation that features align with visible geometry |
| 36-match descriptive table | distribution of terrain/contact/corridor states after life segmentation |
| v0 feature QC | terrain lookup ratio, gap counts, state counts, density calibration |
| v1 LOS demo | pairwise LOS and local visibility on same-clock player samples |
| self-supervised appendix | optional movement/terrain-state prediction without survival labels |

Caldera should answer:

```text
Are these terrain-affordance constructs measurable in a geometry-rich BR map?
```

PUBG should answer:

```text
Do telemetry-observable projections of these constructs improve phase-wise
survival / rank prediction?
```

## Model Interface

Use a three-stream representation:

```text
X_i(t) = team node state
E_ij(t) = directed or typed relation state
U(t) = match/phase/global context
```

Node stream examples:

- vertical position;
- movement friction proxy;
- contact exposure time;
- zone pressure;
- route corridor pressure.

Edge stream examples:

- contact gate;
- distance;
- relative height;
- combat direction;
- zone-pressure difference;
- route conflict.

Global stream examples:

- phase index;
- safe-zone radius/shift;
- alive team count;
- map/mode/perspective.

Observability masks:

| Source | Geometry features | Zone features | Label features |
| --- | --- | --- | --- |
| PUBG | masked except z/proxies | available | available |
| Caldera | available | masked now | masked now |

## Naming Rules

Use direct names only for measured quantities:

```text
pos_z
contact_delta_z_ij_m
dist_blue_boundary_m
building_dist_m
pairwise_los_ij
```

Use `_proxy` when the feature stands in for an unobserved construct:

```text
cover_behavior_proxy
movement_friction_proxy
rotation_corridor_pressure
observed_threat_proxy
```

Never name a PUBG feature as:

```text
terrain_slope
true_cover
line_of_sight
building_distance
openness
```

unless the required geometry asset has been added and validated.

## Design Decisions

Locked after 2026-07-02 design review:

1. The first PUBG implementation stops at `Z3`: vertical position,
   contact-gated relative height, and zone-pressure interactions.
2. Behavior-under-pressure proxies are excluded from the headline block and
   kept as `Z5` sensitivity analysis because they may partly encode combat or
   roster mechanisms.
3. Caldera v1 prioritizes pairwise LOS over full local isovist/openness because
   LOS is dyadic and connects directly to the relation/edge stream.
4. Manuscript terminology should use `spatial-tactical affordance` on first
   definition and `affordance features` afterward.  Military doctrine remains
   a related-work taxonomy source, not the name of the feature family.
5. Self-supervised Caldera learning is future work.  The main Caldera role is
   construct validation through geometry-rich measurement and figures.
