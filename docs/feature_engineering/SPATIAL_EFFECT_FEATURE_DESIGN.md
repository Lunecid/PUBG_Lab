# Spatial Effect Feature Design

## Purpose

This document defines candidate features for showing real spatial effects in the
shared PUBG and Activision Caldera Battle Royale data layer.

Research axis:

```text
Environment Constraint -> Movement Response -> Competitive Exposure -> Survival Outcome
```

The goal is not to force PUBG and Caldera into identical observability. PUBG is
a high-observability normalized telemetry L1. Caldera is a low-observability
spatial dataset with player breadcrumbs, endpoint point clouds, and an OpenUSD
environment catalog. The common representation should therefore expose the same
Battle Royale spatial primitive vocabulary while masking unavailable channels.

## Layer Principle

Spatial effect features must be derived above ground, never by mutating L1.

| Layer | Role in this design | Rule |
| --- | --- | --- |
| `br_ground.trajectory_samples` | source-faithful sampled player/team movement | read only |
| `br_ground.endpoint_samples` | source-faithful anonymous Caldera endpoint point cloud | read only |
| `br_ground.environment_prims` | source-faithful OpenUSD prim metadata and coarse bbox/centroid catalog | read only |
| `pubg.*` / `br_ground.*` PUBG views | normalized L1 telemetry ground | read only |
| `br_repr.*` | feature views, materialized tables, density surfaces, manifests | rebuildable derived layer |

Any feature in this document should be registered through `br_repr.feature_sets`
and built into `br_repr` views/materialized tables. If a feature definition
changes, rebuild the representation. Do not update L1 positions, endpoint rows,
environment prim metadata, or PUBG normalized facts in place.

## Core Join Design

The required Caldera bridge is:

```text
trajectory_samples -> local movement path
endpoint_samples -> anonymous endpoint / terminal-location density surface
environment_prims -> environment constraint catalog
```

Because `endpoint_samples` is anonymous and may not expose match/player
identity, it should be used first as an aggregate spatial outcome surface, not
as a per-player label. The safe first feature is therefore:

```text
For each trajectory or trajectory window, measure movement behavior near
environment prims and integrate the endpoint-density field encountered by that
path.
```

This creates an honest spatial-effect chain:

| Axis step | Caldera observable | Example derived value |
| --- | --- | --- |
| Environment Constraint | prim centroid, bbox, type, asset reference, height span | nearest prim distance, prim buffer occupancy |
| Movement Response | breadcrumb path segment | speed change, stop ratio, sinuosity near prim |
| Competitive Exposure | endpoint density and route density around path/prim | endpoint density integral, hotspot crossing ratio |
| Survival Outcome | endpoint point cloud or linked outcome where available | endpoint lift near prim, path-to-endpoint risk score |

## Feature Families

### A. Common PUBG + Caldera Features

These features use observations that can be represented in both titles. PUBG may
derive endpoint surfaces from `br_ground.outcomes.endpoint_pos`, elimination
events, or final observed positions. Caldera uses `br_ground.endpoint_samples`
and breadcrumb-derived outcomes where linked.

| Feature family | Candidate feature names | L1 inputs | Interpretation | Suggested `br_repr` surface |
| --- | --- | --- | --- | --- |
| Trajectory shape | `path_length_m`, `displacement_m`, `sinuosity`, `avg_speed_mps`, `max_speed_mps`, `stop_ratio` | `trajectory_samples` | Movement response independent of title-specific combat logs | `br_repr.v_br_trajectory_window_features` |
| Endpoint proximity | `dist_to_nearest_endpoint_m`, `endpoint_density_r50`, `endpoint_density_r100`, `endpoint_density_r250` | `trajectory_samples`, endpoint/outcome points | Exposure to terminal-location hotspot fields | `br_repr.v_br_endpoint_density_at_trajectory` |
| Path-integrated endpoint exposure | `endpoint_density_path_integral`, `hotspot_path_ratio`, `max_endpoint_density_on_path` | trajectory segments, endpoint density grid | Whether a path traverses areas where players commonly end or die | `br_repr.agent_spatial_effect_windows` |
| Local route congestion | `trajectory_density_r100`, `nearest_agent_path_m`, `co_route_overlap_ratio` | same-match `trajectory_samples` | Competitive exposure through shared movement corridors | `br_repr.v_br_route_density_features` |
| Vertical movement response | `z_delta_m`, `z_range_m`, `vertical_speed_abs_mean`, `high_ground_ratio` | `trajectory_samples.pos_z_m` | Elevation use as survival positioning | `br_repr.v_br_vertical_movement_features` |
| Outcome-aligned route summary | `final_endpoint_distance_from_path_m`, `last_window_endpoint_density`, `late_path_sinuosity` | trajectory windows, outcomes/endpoints | Links movement response to survival endpoint | `br_repr.v_br_spatial_outcome_features` |

Common features should carry a `source_id`, `feature_set_id`, `clock`, and
`observability_mask`. For Caldera, combat/resource/zone channels are masked.
For PUBG, endpoint surfaces can be player/team linked when telemetry supports
it.

### B. Caldera-Only Geometry Features

These features use `environment_prims`. They are the main path for showing
Warzone/Caldera spatial effects from map geometry.

| Feature family | Candidate feature names | Required inputs | Spatial effect claim | PostGIS only? |
| --- | --- | --- | --- | --- |
| Nearest environment primitive | `nearest_prim_dist_m`, `nearest_prim_type`, `nearest_prim_height_m`, `nearest_prim_bbox_area_m2` | `trajectory_samples`, `environment_prims` | Movement and endpoints vary with proximity to authored map objects | yes, if centroid/bbox exists |
| Prim-buffer occupancy | `time_within_prim_r25`, `time_within_prim_r50`, `path_ratio_near_prim`, `endpoint_density_near_prim` | trajectory points, endpoint points, prim centroid/bbox | Players slow, cluster, or terminate near specific geometry | yes |
| Prim endpoint lift | `endpoint_lift_r50`, `endpoint_lift_r100`, `endpoint_lift_vs_background` | endpoint points, prim buffers, map/background area | A prim class is associated with higher terminal density than background | yes |
| Geometry-conditioned movement | `speed_delta_near_prim`, `stop_ratio_near_prim`, `sinuosity_near_prim`, `turn_angle_rate_near_prim` | trajectory windows, prim buffers | Environment changes route behavior, not only endpoint distribution | yes |
| Corridor or choke proxy | `bbox_gap_density`, `route_density_between_prims`, `endpoint_density_between_prims` | prim bboxes, trajectories, endpoints | Coarse bboxes reveal constrained movement corridors | partial PostGIS |
| Height and roof proxy | `prim_z_span_m`, `trajectory_above_prim_ratio`, `endpoint_high_ground_lift` | PointZ trajectories/endpoints, prim bbox z span | Vertical geometry shapes exposure and endpoints | partial PostGIS |
| Asset-class proximity | `dist_to_asset_class_*`, `time_near_asset_class_*`, `endpoint_lift_asset_class_*` | authored reference asset names, prim metadata | Specific Caldera asset classes drive path/end states | PostGIS plus catalog parsing |

Recommended first implementation:

```text
Use centroid and bbox-derived footprints first. Treat prim names, asset
references, and prim_type as coarse classes. Defer full mesh loading until a
figure or model feature requires exact geometry.
```

### C. PUBG-Only Telemetry Features

These features are not first-class Caldera features unless equivalent labels are
constructed later.

| Feature family | Candidate feature names | PUBG L1 inputs | Why PUBG-only now |
| --- | --- | --- | --- |
| Dynamic safe-zone constraint | `dist_blue_boundary_m`, `inside_blue`, `dist_target_center_m`, `required_speed_to_safe_mps`, `outside_safe_duration_s` | zone states, team positions | Caldera public player layers do not expose equivalent safe-zone timeline in the current ingest |
| Combat exposure | `nearest_enemy_m`, `edge_snapshot_combat`, `damage_taken_recent`, `shots_received_recent`, `third_party_pressure` | combat/damage/kill events, team identity | Caldera player layers do not expose combat telemetry |
| Resource and mobility state | `vehicle_ratio`, `heal_count`, `boost_count`, `ammo_or_equip_proxy`, `loot_density_recent` | inventory/use/pickup/vehicle events | Caldera layers do not expose resource event logs |
| Team survival capital | `alive_n`, `mean_health`, `min_health`, `knock_count_recent`, `revive_count_recent` | health/roster events | Caldera breadcrumb layer is player-centered and lower observability |
| Phase-cut hazard labels | `y_next_phase_elim`, `at_risk_mask`, `phase_survival_time` | phase cuts, outcomes | Current PUBG DB supports causal phase survival modeling |

PUBG-only features should stay in a separate feature block so Caldera
experiments can run with explicit observation masks rather than silent nulls.

## PostGIS vs OpenUSD / Full Asset Boundary

### Computable with Current PostgreSQL + PostGIS

These features only need point geometry, bbox metadata, centroids, and derived
buffers or grids:

- nearest trajectory point to endpoint or prim centroid;
- endpoint counts inside radius buffers;
- endpoint density by square/hex bin;
- path length, displacement, speed, stop ratio, and sinuosity;
- time/path ratio within a prim buffer;
- nearest prim type and nearest prim distance;
- coarse bbox footprint area and z-span;
- path-integrated endpoint density;
- route density and co-route overlap;
- endpoint lift around prim buffers versus background;
- simple corridor proxy from bbox gaps and high route density.

Possible derived geometry helpers:

| Helper | Type | Notes |
| --- | --- | --- |
| `br_repr.v_environment_prim_footprints` | view | bbox envelope polygons from `environment_prims` min/max x/y |
| `br_repr.spatial_bins_50m` | materialized table | source/environment grid for density joins |
| `br_repr.endpoint_density_bins` | materialized table | endpoint count and density per bin/radius |
| `br_repr.trajectory_segments` | materialized table | point-to-point LineString segments with speed and duration |

### Requires OpenUSD or Full Asset Loading

These should not be claimed from the current prim catalog alone:

- exact building interior versus exterior classification;
- wall/rock/tree occlusion and line-of-sight cover;
- collision-aware navigable routes around objects;
- doorway, stair, ladder, zipline, and roof-access topology;
- terrain mesh slope, cliffs, trenches, and micro-cover;
- material semantics such as vegetation, concrete, water, glass, or destructible
  props unless asset metadata explicitly exposes them;
- exact visibility graph between players and endpoints;
- physically accurate chokepoints from mesh surfaces rather than bbox proxies.

When one of these is needed, create a separate `br_repr` feature set whose
metadata records the OpenUSD asset profile, loaded layers, mesh extraction
script, and local artifact path. Do not store full OpenUSD mesh payloads in
PostgreSQL.

## Proposed DB Objects

These are candidate names for future migrations or ETL outputs. This document
does not create them.

| Object | Kind | Grain | Purpose |
| --- | --- | --- | --- |
| `br_repr.v_environment_prim_footprints` | view | prim | Converts bbox min/max metadata into coarse 2D footprint geometry |
| `br_repr.v_caldera_traj_sample_env_nearest` | view | trajectory sample | Nearest prim and distance for each breadcrumb sample |
| `br_repr.v_caldera_endpoint_env_nearest` | view | endpoint sample | Nearest prim and distance for each endpoint |
| `br_repr.trajectory_segments` | materialized table | agent segment | Segment geometry, duration, speed, turn angle, z delta |
| `br_repr.endpoint_density_bins` | materialized table | source/bin | Endpoint density field at one or more grid scales |
| `br_repr.route_density_bins` | materialized table | match/bin/time window | Breadcrumb density field for route congestion |
| `br_repr.prim_spatial_effect_stats` | materialized table | prim/radius | Endpoint lift, route density, speed delta, stop ratio near each prim |
| `br_repr.agent_spatial_effect_windows` | materialized table | match/agent/window | Path-integrated environment and endpoint exposure features |
| `br_repr.spatial_effect_feature_sets` | view | feature set | Human-readable registry over `br_repr.feature_sets` metadata |

Example feature set ids:

```text
spatial_effect_common_v0
spatial_effect_caldera_postgis_v0
spatial_effect_caldera_openusd_mesh_v0
spatial_effect_pubg_telemetry_v0
```

## Minimal PostGIS Build Plan

1. Build coarse prim footprints in `br_repr.v_environment_prim_footprints`.
2. Build trajectory segments from `br_ground.trajectory_samples`.
3. Build endpoint density bins from `br_ground.endpoint_samples` and PUBG
   outcome/endpoints where available.
4. Join trajectory segments to nearby prim footprints and endpoint density bins.
5. Aggregate into `br_repr.agent_spatial_effect_windows`.
6. Aggregate prim-level lift and movement response into
   `br_repr.prim_spatial_effect_stats`.
7. Register the run in `br_repr.feature_sets` with source, clock, radius, grid
   size, and observability metadata.

Suggested metadata fields:

```json
{
  "axis": "Environment Constraint -> Movement Response -> Competitive Exposure -> Survival Outcome",
  "clock": "trajectory_window",
  "radii_m": [25, 50, 100, 250],
  "grid_m": 50,
  "uses_endpoint_samples": true,
  "uses_environment_prims": true,
  "requires_full_openusd_assets": false,
  "observability_mask": {
    "combat": false,
    "resource": false,
    "zone": false,
    "geometry_catalog": true
  }
}
```

## Paper Figure Candidates

### Figure 1. Geometry-Conditioned Endpoint Hotspots

Plot Caldera `environment_prims` footprints/centroids and `endpoint_samples`
density. Highlight prim buffers where endpoint density is higher than the
background map rate.

Claim shown:

```text
Some authored environment regions are not neutral background. They coincide
with elevated terminal-location density.
```

Feature examples:

- `endpoint_lift_r50`;
- `endpoint_density_near_prim`;
- `nearest_prim_type`.

### Figure 2. Movement Deflection Near Environmental Constraints

Overlay Caldera breadcrumb trajectories around a selected compound, ridge, or
large bbox cluster. Show speed, stop ratio, or sinuosity before, during, and
after entering the prim buffer.

Claim shown:

```text
Environment constraints produce measurable movement response, such as slowing,
detouring, clustering, or route bending.
```

Feature examples:

- `speed_delta_near_prim`;
- `sinuosity_near_prim`;
- `path_ratio_near_prim`;
- `route_density_between_prims`.

### Figure 3. Path-Integrated Exposure to Endpoint Risk

Draw two or more trajectories over the endpoint-density surface: one route that
crosses high-density endpoint bins and one route that avoids them. Annotate the
path-integrated endpoint exposure and terminal outcome/end position where
available.

Claim shown:

```text
Survival-relevant exposure can be represented as the accumulated spatial risk
field encountered by a moving player or team.
```

Feature examples:

- `endpoint_density_path_integral`;
- `hotspot_path_ratio`;
- `last_window_endpoint_density`;
- `final_endpoint_distance_from_path_m`.

## Interpretation Rules

- Use Caldera to demonstrate cross-title spatial structure, especially
  trajectory, endpoint, and geometry interactions.
- Use PUBG to validate high-observability survival mechanisms such as phase
  hazard, combat pressure, team viability, and resource state.
- Do not interpret anonymous Caldera endpoint density as a verified individual
  death label unless the source layer explicitly links that endpoint to a
  trajectory or outcome.
- Do not treat coarse prim bboxes as exact walls, cover, interiors, or
  navigable surfaces.
- Report PostGIS-only features separately from OpenUSD/full-asset features in
  manifests and paper tables.
- Keep every spatial-effect feature rebuildable from L1 plus ETL code.

## Recommended First Feature Set

The first paper-safe spatial-effect feature set should be:

```text
spatial_effect_caldera_postgis_v0
```

Scope:

- Caldera only;
- `trajectory_samples`, `endpoint_samples`, and `environment_prims` all used;
- PostGIS centroid/bbox/buffer features only;
- no full mesh or line-of-sight claims;
- aggregate endpoint density used as an exposure/outcome surface;
- outputs written to `br_repr.agent_spatial_effect_windows` and
  `br_repr.prim_spatial_effect_stats`.

This feature set is sufficient for a paper figure that shows the full axis:

```text
environment prim proximity
-> local trajectory speed/sinuosity/stop response
-> endpoint-density exposure around the path
-> terminal-location density or linked outcome
```
