# Prior-Aligned ST Feature Design (2026-06-23)

## Purpose

This document redesigns the ST feature story around prior work first, then
states the project's novelty.

The paper should not say:

```text
We engineered PUBG features.
```

The stronger claim is:

```text
We adapt standard spatio-temporal representation principles from trajectory
analytics, multi-agent tracking, ST forecasting, temporal graph learning, game
telemetry, and survival modeling to a Battle Royale shrinking-arena survival
task.
```

Canonical local contract:

```text
docs/feature_engineering/ST_FEATURE_DEFINITION_2026-06-23.md
docs/INPUT_MODEL_CHANNEL_CONTRACT_2026-06-22.md
docs/GENRE_PRIMITIVE_FEATURE_MANIFEST.csv
```

## Reference Axes

### R1. Trajectory Data Management And Mining

Trajectory analytics treats moving objects as time-ordered spatial samples and
builds representations through preprocessing, querying, clustering, learning,
and forecasting. This supports our use of team centroids, speed, path length,
displacement, trajectory shape, and zone-entry timing.

Reference:

```text
Wang et al., "A Survey on Trajectory Data Management, Analytics, and Learning"
https://arxiv.org/abs/2003.11547
```

Design implication:

```text
Movement is not a single coordinate. It is a clocked path object from which
position, velocity, route geometry, and exposure summaries can be derived.
```

### R2. Multi-Agent Team ST Tracking

Team-sports ST analysis distinguishes dense trajectories from event logs and
emphasizes that team outcomes require the movement and interactions of multiple
agents, not only individual row features.

Reference:

```text
Gudmundsson and Horton, "Spatio-Temporal Analysis of Team Sports -- A Survey"
https://arxiv.org/abs/1602.06994
```

Design implication:

```text
For a team survival game, a team state feature is incomplete unless local
opponent field, contact, and interaction context are represented separately.
```

### R3. ST Forecasting With Temporal Components And External Context

Urban ST forecasting work often separates spatial dependence, temporal
dependence, and external conditions. ST-ResNet explicitly separates temporal
closeness, period, trend, and external factors; traffic STGCN models spatial
and temporal dependencies jointly with graph structure.

References:

```text
Zhang et al., "Predicting Citywide Crowd Flows Using Deep Spatio-Temporal
Residual Networks"
https://arxiv.org/abs/1701.02543

Yu et al., "Spatio-Temporal Graph Convolutional Networks: A Deep Learning
Framework for Traffic Forecasting"
https://arxiv.org/abs/1709.04875
```

Design implication:

```text
The feature design should separate current state, short temporal dynamics,
phase/horizon context, and relation structure instead of mixing them in one
flat table.
```

### R4. Game Telemetry And Sequential Player Modeling

Game telemetry literature supports the use of behavioral logs as quantitative
data, but also warns that interpretable behavior profiles matter. Sequential
game behavior work argues that player behavior should not be reduced only to
aggregate counts.

References:

```text
Drachen et al., "A Comparison of Methods for Player Clustering via Behavioral
Telemetry"
https://arxiv.org/abs/1407.3950

Bunian et al., "Modeling Individual Differences in Game Behavior using HMM"
https://arxiv.org/abs/1804.00245
```

Design implication:

```text
PUBG features should be grouped by behavior mechanisms such as rotation,
combat, resource readiness, and zone exposure, but those mechanisms still need
explicit clocks and causal windows.
```

### R5. Battle Royale / PUBG Prediction Prior Work

Kaggle-style PUBG studies usually use post-game aggregate placement features.
That supports a baseline family, but it is not a phase-causal ST survival
representation.

Reference:

```text
Rokad et al., "Survival of the Fittest in PlayerUnknown BattleGround"
https://arxiv.org/abs/1905.06052
```

Direct local prior:

```text
Sagong et al., "Are you a Hunter or Camper?", Entertainment Computing 2025
local file: paper/battleground_paper.pdf
```

Design implication:

```text
Prior PUBG prediction motivates combat/activity/strategy features. Our
extension is to make them team-level, clock-aware, relational, and evaluated as
phase-cut hazard rather than post-game placement only.
```

### R6. Time-Varying Survival / Hazard Modeling

Dynamic survival work motivates representing covariates as values observed
over time and predicting risk at an at-risk interval, rather than treating the
problem as one static classification row.

References:

```text
Jarrett et al., "MATCH-Net: Dynamic Prediction in Survival Analysis using
Convolutional Neural Networks"
https://arxiv.org/abs/1811.10746

Rava and Bradic, "DeepHazard: neural network for time-varying risks"
https://arxiv.org/abs/2007.13218
```

Design implication:

```text
The prediction unit should be an at-risk team x phase-cut cell. Features must
be observable no later than the cut being used for prediction.
```

### R7. Official Telemetry Substrate

The PUBG API telemetry schema contains timestamped event records for positions,
phase changes, attacks, damage, kills, item use, vehicle events, and other
match events. This supports reconstructing trajectories, event windows, and
typed interactions from raw logs.

Reference:

```text
PUBG official telemetry event documentation
https://documentation.pubg.com/en/telemetry-events.html
```

Design implication:

```text
Our ST features should be traceable back to timestamped telemetry events and
must name whether they come from dense position snapshots, sparse event logs,
or phase/global state.
```

## Prior-Aligned Feature Redesign

### ST-A. Domain And Episode Context

Prior-work role:

```text
external context / cohort definition
```

Current channels:

```text
C0 / U / split-only metadata
```

Examples:

- map;
- mode;
- perspective;
- team size;
- phase index;
- phase duration.

Paper treatment:

```text
Context, stratification, or optional conditioning. Not the main ST feature
claim by itself.
```

Novelty:

```text
Battle Royale phase horizon and map/mode rule-set define the survival risk
context; they are not generic row metadata.
```

### ST-B. Time-Space Trajectory State

Prior-work role:

```text
trajectory analytics / multi-agent tracking
```

Current channels:

```text
X_s snapshot node
X_phase trajectory aggregates
```

Current features:

- `pos_x`, `pos_y`, `pos_z`;
- `speed_mps`;
- `path_length_m`;
- `displacement_m`;
- `sinuosity`;
- `avg_speed_mps`;
- `rotation_timing`.

Paper treatment:

```text
This is the base ST state: where the team is, how it is moving, and how that
movement evolves before the prediction cut.
```

Novelty:

```text
We aggregate raw player motion into team-level survival actors, preserving a
10-second clock and a phase-cut target rather than collapsing the match into a
post-game aggregate row.
```

### ST-C. Shrinking-Arena / Zone Constraint Features

Prior-work role:

```text
external context + dynamic spatial constraint
```

Current channels:

```text
X_s zone-relative node features
U phase/global context
X_phase habitat/zone aggregates
```

Current features:

- `rel_zone_x`, `rel_zone_y`;
- `dist_blue_boundary_m`;
- `inside_blue`;
- `dist_target_center_m`;
- `outside_safe_duration_s`;
- `boundary_distance_slope`;
- `mean_dist_to_boundary_m`;
- `hab_required_speed_to_safe_mps`;
- `hab_migration_deficit_mps`;
- `hab_safe_alignment_to_target`.

Paper treatment:

```text
This is the paper's key domain-specific ST block. The spatial coordinate
system is not fixed only by the map; it is redefined over time by the shrinking
safe zone and target zone.
```

Novelty:

```text
Unlike road-network or court-based ST tasks with a mostly fixed spatial frame,
Battle Royale survival requires a moving hazard field. We encode teams relative
to the current and target safe zones at causal cuts.
```

### ST-D. Team Viability And Resource State

Prior-work role:

```text
time-varying covariates / game telemetry behavioral state
```

Current channels:

```text
X_s snapshot node
X_phase phase node
```

Current features:

- `mean_health`;
- `min_health`;
- `alive_n`;
- `alive_players_*`;
- `team_spread_*`;
- `vehicle_ratio`;
- pickup/equip/use/heal/boost/resource counts.

Paper treatment:

```text
These are survival-capital covariates. They are ST when they are indexed by
snapshot or phase window and evaluated before the prediction cut.
```

Novelty:

```text
The unit is team survival, so player telemetry is projected into roster-level
survival capital rather than player-only strategy or post-game totals.
```

### ST-E. Interaction Event Field

Prior-work role:

```text
event-log analytics / sequential behavior / survival pressure
```

Current channels:

```text
A_s combat relation
X_phase combat aggregates
```

Current features:

- `edge_snapshot_combat`;
- damage dealt/taken;
- kills/groggy dealt/taken;
- combat initiated;
- recent damage taken;
- third-party exposure;
- combat-zone pass count.

Paper treatment:

```text
Combat events are sparse but semantically rich ST events. They should be
windowed by time and direction, not treated as static match totals.
```

Novelty:

```text
We distinguish outgoing and incoming pressure and make combat either a
directed relation channel or a phase-local time-varying covariate.
```

### ST-F. Local Opponent Field And Density

Prior-work role:

```text
multi-agent local interaction / neighborhood exposure
```

Current channels:

```text
A_s proximity relation
X_phase local-density aggregates
```

Current features:

- `edge_snapshot_proximity`;
- `avg_enemy_density_100m`;
- `min_nearest_enemy_m`;
- `path_near_enemy_ratio`;
- `net_prox_*`;
- `hab_local_enemy_count_*`;
- `hab_safe_entry_congestion`.

Paper treatment:

```text
This block tests whether the team's risk is explained only by its own motion,
or also by the local opponent field it moves through.
```

Novelty:

```text
Nearby teams are not "similar" teams. They are survival dependencies: possible
contact, route conflict, or safe-entry competition.
```

### ST-G. Typed Relation Topology

Prior-work role:

```text
spatio-temporal graph learning
```

Current channels:

```text
A_s, A_phase, optional E_s
```

Current relations:

- proximity;
- combat;
- co-pressure;
- habitat/forced convergence;
- contact-gated relative advantage.

Paper treatment:

```text
Topology is tested only when it beats no-graph and graph-null controls.
```

Novelty:

```text
The graph is not a social/homophily graph. It is a dynamic survival-dependency
graph where edges mean exposure, pressure, combat, or competition for safe
space.
```

### ST-H. Survival-Cut Label Design

Prior-work role:

```text
time-varying survival / dynamic risk prediction
```

Current channels:

```text
C8 labels/masks, not model input
```

Current target:

```text
given observations through phase k, predict elimination before phase k+1
```

Paper treatment:

```text
This is not an input feature block. It is the evaluation unit that makes the
ST features causally interpretable.
```

Novelty:

```text
We evaluate team survival as at-risk team x phase-cut hazard rather than
post-game placement over all rows.
```

## Recommended Paper Feature Table

Use this compressed table in the 4-8 page paper.

| Prior-work concept | Our feature block | Tensor stream | Examples | Novelty in this task |
| --- | --- | --- | --- | --- |
| trajectory state | time-space trajectory | `X_s`, `X_phase` | position, speed, path length, sinuosity | team-level trajectory under survival cuts |
| external/spatial context | shrinking-arena constraint | `X_s`, `U`, `X_phase` | zone-relative position, boundary distance, required speed | moving hazard field, not fixed court/road |
| time-varying covariates | team viability/resource state | `X_s`, `X_phase` | health, alive count, vehicles, resources | roster survival capital from player telemetry |
| event logs | interaction event field | `A_s`, `X_phase` | combat edge, damage, kills, third-party exposure | directed combat pressure before cut |
| multi-agent neighborhood | local opponent field | `A_s`, `X_phase` | proximity, density, nearest enemy | survival dependence via opponent field |
| ST graph learning | typed relation topology | `A_s`, `A_phase`, `E_s` | proximity/combat/co-pressure/relative edge | null-controlled survival graph |
| survival modeling | phase-cut hazard labels | `Y`, mask only | at-risk team x phase cut | dynamic risk, not post-game placement only |

## Revised Definition

Prior-aligned definition:

```text
An ST feature is a causally cut-aligned telemetry variable whose value is
indexed by a game-time clock and whose interpretation depends on a spatial
reference frame: team trajectory, shrinking-zone geometry, local opponent
field, interaction event location, or typed team-to-team relation.
```

Project-specific extension:

```text
For Battle Royale, the spatial reference frame is dynamic: the arena itself
changes through safe-zone compression. Therefore ST features must encode not
only where a team is, but where it is relative to the future survivable space
and to other teams competing for that space.
```

## What We Should Change From The Previous Definition

Previous definition was correct but too implementation-first:

```text
entity owner + clock + spatial reference + causal window + model stream
```

Keep it as the engineering contract, but present the paper definition through
prior-work feature families:

1. trajectory state;
2. shrinking-arena context;
3. viability/resource time-varying covariates;
4. interaction event field;
5. local opponent field;
6. typed relation topology;
7. phase-cut survival labels.

That structure lets the paper say:

```text
Our features are not arbitrary PUBG columns. They are a prior-aligned ST
representation adapted to the Battle Royale survival mechanism.
```
