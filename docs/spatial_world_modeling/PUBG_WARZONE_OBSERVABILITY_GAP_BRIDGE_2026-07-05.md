# PUBG-Warzone Observability Gap Bridge

Date: 2026-07-05
Class: `WorldModel`
Lifecycle: `active`

## Purpose

This document fixes the RQ3 role of Warzone / Caldera.

Warzone is not included to add a second supervised prediction dataset. It is
included to show which Battle Royale situations are not directly expressible
from official PUBG telemetry alone, and how a geometry-rich world model can
represent those missing constructs.

The expected paper logic is:

```text
PUBG official telemetry gives label-rich survival/rank supervision.
PUBG official telemetry does not directly expose terrain mesh, building mesh,
surface semantics, route topology, or occlusion geometry.
Warzone / Caldera provides geometry-rich construct evidence for those missing
spatial-affordance representations.
```

## Bridge Matrix

| Construct / situation | PUBG official telemetry gap | Warzone / Caldera representation | Current evidence | Claim status |
| --- | --- | --- | --- | --- |
| Terrain surface and vertical context | Player or team `z` exists, but the official telemetry stream does not provide the terrain mesh below the actor; `z` alone cannot separate altitude from terrain surface, slope, or local relief. | OpenUSD terrain triangles plus trajectory samples produce `terrain_z_m`, `height_above_terrain_m`, `relative_elevation_r50_m`, and local relief. | `height_above_terrain_m` median `0.18m`, p75 `0.33m`, max `4.13m`. | supported descriptive construct |
| Rough terrain mobility | Official PUBG telemetry can show movement over coordinates, but not whether the route crosses a steep or rugged local surface without external terrain geometry. | Mesh-derived `terrain_slope_deg`, `local_relief_r50_m`, and ruggedness features around each trajectory sample. | max slope `39.19deg`; max `relief50=37.85m`; `rough_mobility=18` rows. | supported descriptive construct |
| Compound and building affordance | Official telemetry alone does not expose building meshes, wall mass, doorway topology, or compound geometry needed to interpret cover-like spatial affordance. | Semantic building triangles support nearest-building distance, building mesh density, and `cover_building_r25` proxies. | `building_dist_m` median `0.91m`; closest sample `0.07m`; `cover_building_r25=85.00%`. | supported descriptive construct |
| Route corridor and road/runway context | Coordinates reveal where a team moved, but official telemetry does not label whether the path is on a road, runway, bridge-like corridor, or open terrain. | Road and runway mesh categories support `road_or_runway_dist_m` and `on_corridor_proxy`. | `road_or_runway_dist_m` median `50.13m`; nearest corridor sample `0.78m`; `on_corridor_proxy=3.33%`. | supported descriptive construct |
| Local tactical-state labeling | Telemetry events can indicate combat and movement, but they do not directly assign geometry-grounded states such as `compound_movement` or `rough_mobility`. | Rules over mesh-derived terrain, building, and corridor features produce descriptive `tactical_state_v0` labels. | `compound_movement=102`, `rough_mobility=18`. | supported descriptive construct, not outcome supervision |
| Visibility, occlusion, and line of sight | Official telemetry lacks occluding terrain/building mesh needed to ray-cast whether two agents are mutually visible. | The same geometry cache and same-clock trajectories can support future mesh ray-cast LOS or occlusion edges. | Not computed in the current package. | pending extension, not claimed |

## Situation Cards

These are the examples that should be used when explaining the figure or
writing the RQ3 paragraph.

| Situation | Example | Why PUBG telemetry alone is insufficient | What Warzone / Caldera shows |
| --- | --- | --- | --- |
| rough terrain transition | `t=1187.232s`, slope `39.19deg`, `relief50=37.74m`, state `rough_mobility` | The route may look like ordinary coordinate movement unless local mesh slope and relief are available. | The trajectory point can be grounded in a steep, rugged local terrain surface. |
| compound movement beside built mass | `t=1482.048s`, building distance `0.07m`, `density25=26794.1 tri/ha`, `cover_r25=True` | Actor position alone does not expose nearby building mesh density or compound geometry. | The path can be described as moving through or beside dense built geometry. |
| road/runway corridor contact | `t=1625.472s`, road/runway distance `0.78m`, `on_corridor_proxy=True` | Official telemetry does not directly attach road/runway semantics to coordinates. | The path can be tied to explicit route-surface categories. |
| dense building-mesh pocket | `t=1541.808s`, building density `385435.1 tri/ha`, building distance `0.66m` | A coordinate sample cannot quantify nearby built mass without world geometry. | The local environment can be measured as a high-density built pocket. |
| relative local elevation | `t=1199.184s`, `relative_elevation_r50=2.95m`, `relief50=35.97m` | Raw `z` does not say whether the actor is high or low relative to nearby terrain. | The sample can be placed into a local elevation neighborhood. |

## Artifact Outputs

Generated outputs:

```text
reports/caldera_3d/construct_validation/caldera_o6_pubg_warzone_observability_bridge.md
reports/caldera_3d/construct_validation/caldera_o6_pubg_gap_situation_cards.md
reports/caldera_3d/construct_validation/caldera_o6_pubg_warzone_observability_bridge_board.png
reports/caldera_3d/construct_validation/caldera_o6_construct_validation_composite.png
```

The bridge board is a supplementary O6 observability-boundary artifact. The
composite figure is the visual example backing the bridge, not a core RQ3
claim figure under the revised manuscript spine.

## Claim Boundary

Allowed:

```text
Warzone / Caldera shows how geometry-rich data can represent terrain, route,
building, and local tactical-state constructs that official PUBG telemetry
cannot directly observe.
```

Allowed:

```text
This complements PUBG's role as the label-rich supervised survival and rank
prediction dataset.
```

Avoid:

```text
Caldera validates the PUBG survival model.
```

Avoid:

```text
Caldera currently supports supervised phase survival or final-rank prediction.
```

Avoid:

```text
The current artifact proves LOS or occlusion effects.
```

LOS is motivated by the gap bridge but remains an uncomputed extension.
