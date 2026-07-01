# Spatio-Temporal Feature Definition (2026-06-23)

## Purpose

This document defines what the paper means by a spatio-temporal, or ST,
feature.

Companion paper-framing document:

```text
docs/ST_FEATURE_PRIOR_ALIGNED_DESIGN_2026-06-23.md
```

Use this file as the engineering contract. Use the companion document to frame
the same contract through prior-work feature families and the project's
novelty.

The paper should not use ST as a vague adjective. In this project, an ST
feature is a telemetry-derived variable whose meaning depends on both:

```text
where the team/relation is in the arena
and
when the observation is made relative to the survival prediction cut
```

The definition must stay aligned with:

- `docs/INPUT_MODEL_CHANNEL_CONTRACT_2026-06-22.md`;
- `docs/FEATURE_CLOCK_CHANNEL_REDESIGN_2026-06-22.md`;
- `docs/GENRE_PRIMITIVE_FEATURE_MANIFEST.csv`;
- `stdb/snapshot_queries.py`;
- `stdb/snapshot_feature_etl.py`.

## Working Definition

A feature is an ST feature if it declares all five fields below:

| Requirement | Meaning | Examples |
| --- | --- | --- |
| entity owner | what object owns the value | team, team pair, global phase |
| feature clock | when the value updates | 10-second snapshot, phase window, phase cut |
| spatial reference | what spatial frame gives it meaning | map coordinates, zone-relative coordinates, pairwise distance, route/path |
| causal window | what observations are allowed | instant at `t`, trailing 10 seconds, phase-to-date/window before cut |
| model stream | where the model consumes it | `X_s`, `A_s`, `E_s`, `X_phase`, `A_phase`, `U` |

Short definition for the paper:

```text
We define an ST feature as a causally valid team-, pair-, or global-level
telemetry variable indexed by an explicit game-time clock and grounded in an
arena, zone, route, or team-to-team spatial reference frame.
```

## Not Every Time Feature Is ST

Not ST by itself:

- final rank;
- match id;
- team id;
- post-game aggregate counts;
- labels and masks;
- unclocked feature families;
- phase-end outcomes used before their prediction cut.

These may be metadata, labels, or baselines, but they are not ST input
features unless they have a declared clock, causal window, and spatial/domain
meaning.

## ST Feature Families In This Project

### F-ST1. Snapshot Team-State Features

Tensor:

```text
X_s: (B, T_s, N, F_s)
```

Clock:

```text
10-second snapshot
```

Current implementation:

```text
stdb/snapshot_queries.py::SNAP_FEATURE_SPECS
stdb.team_snapshot_features
```

Current features:

| Feature | Spatial/temporal meaning | Domain mechanism |
| --- | --- | --- |
| `pos_x`, `pos_y`, `pos_z` | absolute team centroid at snapshot `t` | space |
| `rel_zone_x`, `rel_zone_y` | team position relative to current safe-zone center/radius | zone |
| `speed_mps` | past-only movement from previous snapshot to `t` | rotation |
| `dist_blue_boundary_m` | signed distance to current blue-zone boundary | zone |
| `inside_blue` | current containment/exposure state | zone |
| `dist_target_center_m` | distance to known target/white-zone center | zone/rotation pressure |
| `mean_health`, `min_health` | team health capital at `t` | roster |
| `alive_n` | surviving roster count at `t` | roster |
| `vehicle_ratio` | vehicle use/mobility resource at `t` | resource/rotation |

These are ST features because the same value has no stable meaning without the
snapshot time and the arena/zone frame. For example, `pos_x` is not meaningful
alone; `pos_x` at phase 1 versus phase 6 implies different survival pressure.

### F-ST2. Snapshot Team-Relation Features

Tensor:

```text
A_s: (B, T_s, R_s, N, N)
```

Clock:

```text
10-second snapshot
```

Current relations:

| Relation | Spatial/temporal meaning | Domain mechanism |
| --- | --- | --- |
| `edge_snapshot_proximity` | team-to-team distance at snapshot `t` | local opponent field / relation |
| `edge_snapshot_combat` | directed combat intensity in the past window `(t-10, t]` | combat exchange |

These are ST features because they are pairwise, time-indexed, and spatial or
tactical relations. They are not node state. They must enter through a relation
stream or a graph-null control, not through silent concatenation to `X_s`.

### F-ST3. Derived Pairwise Relative Features

Tensor:

```text
E_s: (B, T_s, N, N, F_e)
```

Current or planned role:

```text
contact-gated relative advantage
```

Examples:

- relative altitude;
- relative alive count;
- relative health;
- relative zone pressure;
- relative distance to target/safe center;
- contact gate from proximity or combat.

These are ST features, but they are not part of the default paper-facing
runner. Current all-match one-epoch evidence does not make them the primary
paper claim.

### F-ST4. Phase-Window ST Aggregates

Tensor:

```text
X_phase: (B, K, N, F_phase)
```

Clock:

```text
phase_cut, phase_window, or phase_to_date
```

Examples from the manifest/code:

| Family | Examples | Meaning |
| --- | --- | --- |
| trajectory | `path_length_m`, `displacement_m`, `sinuosity`, `avg_speed_mps` | route dynamics during a phase |
| zone pressure | `outside_safe_duration_s`, `boundary_distance_slope`, `mean_dist_to_boundary_m` | exposure to shrinking arena |
| combat | `damage_dealt`, `damage_taken`, `kills_dealt`, `combat_initiated` | phase-local combat behavior |
| local opponent field | `avg_enemy_density_100m`, `min_nearest_enemy_m`, `path_near_enemy_ratio` | local spatial pressure |
| resource readiness | pickup/equip/use/heal/boost counts | phase resource state |
| network/habitat V2 | `net_*`, `cum_*`, `hab_*` features | phase relation and habitat pressure summaries |

These are also ST features, but they are a different clock stream from
snapshot features. They should not be concatenated into the snapshot-first
paper runner unless the experiment explicitly tests phase aggregate fusion.

### F-ST5. Global ST Context

Tensor owner:

```text
U
```

Examples:

- phase index;
- phase duration;
- safe-zone radius and center shift;
- alive population;
- map/mode if used as context.

These are ST context variables. They condition the risk horizon and arena
geometry, but they are not team node features.

## Current Paper Runner Scope

The paper-facing runner is:

```text
scripts/run_snapshot_paper_st_feature_matrix.sh
```

Default core variants:

```text
snapshot_last_only
snapshot_tcn_no_graph
snapshot_rgcn
snapshot_rgcn_edge_shuffle
snapshot_gate_only
```

This runner tests snapshot-clock ST features and relation channels:

| Comparison | What it isolates |
| --- | --- |
| `snapshot_last_only` vs `snapshot_tcn_no_graph` | value of temporal sequence modeling over the same snapshot ST node channel |
| `snapshot_tcn_no_graph` vs `snapshot_rgcn` | added value of ST relation channels |
| `snapshot_rgcn` vs `snapshot_rgcn_edge_shuffle` | observed topology beyond graph smoothing/null relation |
| `snapshot_rgcn` vs `snapshot_gate_only` | richer relation channels beyond binary contact/capacity |

Important limitation:

```text
The current paper runner does not test all phase-window ST aggregates. It tests
the snapshot-first ST representation: X_s and, for relation models, A_s.
```

## What Would Count As ST Feature Significance?

The paper should define significance as a ladder, not a single number.

### Level 1. Snapshot State Is Useful

Evidence:

```text
snapshot_last_only improves over a non-ST/static baseline or phase-only
aggregate baseline.
```

Current status:

```text
not yet isolated in the new paper runner unless an explicit static/phase-only
baseline is included or cited from legacy controls.
```

### Level 2. Temporal ST Dynamics Are Useful

Evidence:

```text
snapshot_tcn_no_graph improves over snapshot_last_only.
```

Interpretation:

```text
The sequence of ST node states carries information beyond the latest observed
state at the cut.
```

### Level 3. ST Relations Are Useful

Evidence:

```text
snapshot_rgcn improves over snapshot_tcn_no_graph.
```

Interpretation:

```text
Team-to-team proximity/combat channels add predictive signal beyond team-only
temporal dynamics.
```

### Level 4. Observed Topology Is Useful

Evidence:

```text
snapshot_rgcn improves over snapshot_rgcn_edge_shuffle and snapshot_gate_only.
```

Interpretation:

```text
The observed relation structure carries signal beyond graph capacity, smoothing,
or binary contact information.
```

Current all-match one-epoch evidence favors `snapshot_rgcn`, but this remains
hypothesis-generating until repeats or longer training confirm the gap.

## Writing Contract For The 4-8 Page Paper

Use:

```text
ST features are causally cut-aligned telemetry variables with explicit update
clocks and spatial reference frames. We separate snapshot team-state features,
snapshot team-relation features, phase-window aggregates, and global phase
context, and evaluate whether the snapshot-first ST representation improves
next-phase survival prediction under no-graph and graph-null controls.
```

Avoid:

```text
We use spatial and temporal features.
```

That wording hides the key contribution: a clock-aware ST database
representation, not just a larger feature list.
