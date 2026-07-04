# PUBG Feature Operational Manifest

Date: 2026-07-02

Class: `FeatureContract`

Status: design lock candidate

## Purpose

This document converts the Battle Royale ontology into a PUBG-operational
feature contract.  It defines which observed PUBG telemetry features may stand
for each conceptual family, which clock they live on, and which leakage guard
must hold before any long-running model experiment is started.

The goal is not to maximize feature count.  The goal is to make the feature
root stable enough that every model, ablation, and paper claim can point back
to the same contract.

Upstream concepts:

- `docs/feature_engineering/BATTLE_ROYALE_FEATURE_ONTOLOGY_2026-07-02.md`
- `docs/feature_engineering/SPATIAL_TACTICAL_AFFORDANCE_FEATURE_SPEC_2026-07-02.md`
- `docs/feature_engineering/PUBG_NETWORK_STRUCTURE_FEATURE_SPEC_2026-07-02.md`
- `docs/experiment_design/BATTLE_ROYALE_FRAMEWORK_VALIDATION_PLAN_2026-07-02.md`

## Core Mechanism

The target framework is:

```text
phase contraction
-> survival-environment change
-> agent/team behavioral response
-> interaction/exposure change
-> survival / rank outcome
```

Therefore PUBG feature design must expose three things:

1. how the survival environment changes by phase;
2. how teams respond to that environment;
3. whether those phase-conditioned signals explain next-phase survival or
   rank-related outcomes.

## Prediction Unit

The primary supervised unit is a team at a completed phase cut:

```text
row key      = (match_id, team_id, cut_phase = k)
input        = all allowed telemetry observed through phase k
target       = elimination during the next phase interval
primary y    = hazard_next
auxiliary y  = final_rank_norm / final_topK, used only as supervision
mask         = target_defined and team at risk at cut k
```

The intended model interface is:

```text
U(k)      = global match/phase context
X_i(k)    = team/node features through phase cut k
E_ij(k)   = relation/edge features through phase cut k
Y_i(k+1)  = next-phase hazard or valid rank-related supervision
```

Snapshot models may consume `X_i(t)` and `E_ij(t)` at 10s resolution, but the
same cut rule applies: a prediction at phase cut `k` must not see snapshots or
events from phase `k+1`.

## Clock Contract

| Clock | Meaning | Allowed for headline model? | Notes |
| --- | --- | --- | --- |
| `match_static` | map/mode/settings known before play | yes | split/domain metadata must not leak across match splits |
| `snapshot_10s` | team state at snapshot `t` | yes | only snapshots with `snapshot_t <= phase_end_s(k)` for cut `k` |
| `trailing_10s` | event intensity in `(t-10s, t]` | yes | use ceil-to-snapshot attribution; no future window |
| `phase_window` | aggregate over completed phase `k` | yes | valid only because the prediction cut is after phase `k` |
| `phase_to_date` | cumulative aggregate over phases `1..k` | yes | cumulative graph/role summaries are allowed |
| `train_split_only` | learned transform fit only on training split | sensitivity only | never fit on validation/test |
| `match_end` | final result after match end | labels only | forbidden as input |

If the prediction cut is later changed to phase-start or mid-phase inference,
all `phase_window` and phase-end columns must be recomputed or masked.

## Perspective Contract

The paper models PUBG as an analyst-view telemetry system, not as a strict
player-perception model.

| Perspective | Allowed use | Claim boundary |
| --- | --- | --- |
| player-available proxy | public zone, own state, directly encountered combat | decision-support interpretation may be discussed carefully |
| analyst-view telemetry | all pre-cut teams, paths, contacts, distances, graph summaries | survival environment and relation analysis |
| forbidden future view | future phase events, final outcomes as inputs | never allowed |

Examples such as enemy density, route conflict, proximity graph features, and
network-role summaries are analyst-view telemetry.  They are causal if computed
only from pre-cut data, but they should not be described as what a player
literally observed.

## Portability Contract

This manifest is a PUBG operationalization of a general Battle Royale ontology.
It must keep conceptual portability separate from measurement observability.

Conceptual portability axis:

| Layer | Meaning | Claim status |
| --- | --- | --- |
| `BR_GENERAL` | genre-level mechanism that should transfer across Battle Royale games | allowed in framework claims |
| `PUBG_NATIVE` | PUBG-specific rule, telemetry term, item category, or naming convention | allowed only as PUBG instantiation |
| `STDB_IMPLEMENTATION` | current database/view/column/table name | engineering contract only |

Measurement observability axis:

| Observability | Meaning | Examples |
| --- | --- | --- |
| `TELEMETRY_OBSERVED` | directly measured by event or state telemetry | position, health, damage, phase table |
| `GEOMETRY_OBSERVED` | measured only when map/mesh/visibility assets exist | slope, LOS, building distance |
| `LABEL_OBSERVED` | valid only as target, mask, or evaluation label | final rank, next-phase hazard |
| `PROXY_OBSERVED` | inferred from behavior because the direct construct is absent | cover proxy, openness proxy |
| `MASKED` | not available in the current dataset | Caldera rank labels, PUBG mesh cover |

Mapping rule:

```text
The paper should name the BR_GENERAL construct first, then mention the
PUBG_NATIVE or STDB_IMPLEMENTATION projection only when describing data.
```

Portability map:

| Ontology family | BR_GENERAL construct | PUBG_NATIVE projection | STDB_IMPLEMENTATION examples |
| --- | --- | --- | --- |
| O0 rule/context | match rules, map, mode, phase, team size | PUBG map names, FPP/TPP camera mode, squad/duo/solo conventions | `map_name`, `game_mode`, `match_team_size`, `camera_view_type` |
| O1 survival capital | alive roster, health/capacity, team cohesion | DBNO/groggy state where available, PUBG vehicle occupancy | `alive_players_*`, `mean_health_avg`, `vehicle_ratio`, `team_spread_*` |
| O2 contraction/elimination pressure | shrinking playable area, boundary pressure, habitat scarcity | blue zone, poison/target safe zone, red/black zone metadata | `safe_radius_*`, `poison_radius_*`, `zone_gap_*`, `hab_*` |
| O3 movement/rotation | movement response to pressure, route timing, path efficiency | PUBG vehicle movement and safe-zone entry behavior | `path_length_m`, `sinuosity`, `zone_entry_time`, `vehicle_speed_*` |
| O4 interaction/threat | opponent exposure, combat pressure, third-party pressure | PUBG damage/groggy/kill event taxonomy | `damage_*`, `groggy_*`, `kills_*`, `third_party_exposure_count` |
| O5 resource/support | equipment, consumables, mobility resources | boost, heal, care package, lootbox categories | `heal_use_count`, `boost_use_count`, `carepackage_pickup_count`, `lootbox_pickup_count` |
| O6 spatial affordance | position, relative height, cover/openness/mobility affordance | PUBG `pos_z` proxy only; no native mesh terrain | `pos_z`, `rel_zone_*`, `contact_delta_z_ij_m` |
| O7 relation/interaction | proximity, combat relation, co-pressure, relative advantage | PUBG telemetry-derived team graph | `team_phase_edge_cache`, `net_*`, `cum_*`, `d_over_ell` |
| O8 outcome/win condition | next-phase survival, rank, at-risk mask | PUBG final placement / top-k labels as supervision | `hazard_next`, `final_rank_norm`, `target_defined` |

Non-portable terms such as `groggy`, `blue zone`, `boost`, `carepackage`, and
`lootbox` are valid PUBG feature names, but they should be translated to the
general construct before making cross-game claims.

## Dimensionality Contract

PUBG exposes 3D-ish telemetry (`x/y/z`) but the current project lacks validated
PUBG terrain/building/LOS geometry.  Therefore PUBG is treated as:

```text
ELEVATION_2_5D + GEOMETRY_POOR
```

This is different from both a full 3D geometry-rich game and a planar 2D game.

| Feature concept | General Battle Royale status | PUBG current status | Non-3D fallback |
| --- | --- | --- | --- |
| zone-relative position | portable | available | same concept with `x/y` or grid cells |
| local density / proximity | portable | available | same concept with radius/cell adjacency |
| route conflict / corridor pressure | portable | analyst-view telemetry | path/cell/sector conflict |
| vertical position / relative height | conditional | available as `pos_z` proxy | masked or replaced by sector/layer advantage |
| terrain slope / ruggedness | geometry-specific | unavailable for PUBG | masked unless map costs exist |
| LOS / occlusion / cover object | geometry-specific | unavailable for PUBG | masked or replaced by contact/damage behavior proxy |
| building/compound distance | asset-specific | unavailable for PUBG | masked unless map labels exist |

Rule:

```text
OM6 must be evaluated under an observability mask.  A non-3D Battle Royale game
can instantiate O6 through planar position, density, path, sector, and zone
features without any height or terrain terms.
```

## Operational Feature Families

### O0. Rule And Match Context

Contract role:

```text
Control the rule-set under which survival should be interpreted.
```

| Field group | Level | Current source | Example features | Inclusion |
| --- | --- | --- | --- | --- |
| match identity | key | `stdb.matches`, `stdb.teams` | `match_id`, `team_id` | split/key only |
| rule context | `U(k)` | `stdb.matches` | `map_name`, `game_mode`, `match_team_size`, `camera_view_type`, `weather_id` | control / stratification |
| phase context | `U(k)` | `stdb.v_match_phase_bounds`, `match_phase_global` | `phase_idx`, `phase_duration_s`, `phase_start_s`, `phase_end_s` | required |

Feature claim:

```text
O0 is not a mechanism by itself; it defines the comparison frame.
```

Guard:

- never row-split within a match;
- use match-level splits for all supervised experiments;
- `match_id` and `team_id` are identifiers, not model signal.

### O1. Agent / Team Survival Capital

Contract role:

```text
Represent the team's remaining capacity to survive before the next phase.
```

| Field group | Level | Current source | Example features | Inclusion |
| --- | --- | --- | --- | --- |
| roster alive state | `X_i(k)` | `team_phase_state`, `team_snapshot_features` | `alive_players_start`, `alive_players_end`, `alive_players_min`, `alive_players_avg`, `alive_n` | headline |
| health state | `X_i(k)` | `team_phase_state`, `team_snapshot_features` | `mean_health_avg`, `min_health_min`, `health_slope`, `mean_health`, `min_health` | headline |
| cohesion / support | `X_i(k)` | `team_phase_state` | `team_spread_avg_m`, `team_spread_max_m` | headline |
| prior skill proxy | `X_i(k)` | derived historical feature only | `team_pre_rating`, split-fit scores | sensitivity only |

Feature claim:

```text
O1 measures survival capital, not player skill unless the feature is explicitly
fit from prior matches in a train-split-only protocol.
```

Guard:

- same-match final rank, win/topK, or future survival must not enter O1;
- `alive_players_end` is allowed only for completed-phase cut prediction;
- prior-rating features must be excluded from the headline unless their split
  protocol is audited.

### O2. Contraction / Elimination Pressure

Contract role:

```text
Measure how Battle Royale contraction pressure changes the survival
environment.
```

In PUBG, this genre-level O2 mechanism is observed mainly through phase and
zone telemetry.  Shrinking safe/poison zones are a PUBG-native projection of
the broader contraction / elimination-pressure construct.

| Field group | Level | Current source | Example features | Inclusion |
| --- | --- | --- | --- | --- |
| alive population pressure | `U(k)` | `match_phase_global` | `avg_alive_teams`, `start_alive_teams`, `end_alive_teams`, `avg_alive_players` | headline |
| zone geometry | `U(k)` | `dynamic_regions`, `match_phase_global` | `safe_radius_start_m`, `safe_radius_end_m`, `safe_radius_delta_m`, `safe_center_shift_m` | headline |
| target / poison relation | `U(k)` | `dynamic_regions`, `match_phase_global` | `poison_radius_start_m`, `poison_radius_end_m`, `zone_gap_start_m`, `zone_gap_end_m` | headline |
| habitat pressure | `U(k)` / `X_i(k)` | `team_phase_habitat_features` | `hab_loss_ratio`, `hab_crowding_pressure`, `hab_zone_gap_pressure`, `hab_required_speed_to_safe_mps` | headline / ablation |
| local zone exposure | `X_i(k)` | `team_phase_trajectory`, `team_phase_exposure`, `team_snapshot_features` | `outside_safe_duration_s`, `outside_safe_ratio`, `dist_blue_boundary_m`, `inside_blue` | headline |

Habitat ownership rule:

```text
Habitat features are O2 x O6 interaction features: contraction pressure applied
to the spatial affordance field.  For supervised ablation attribution they
belong to OM3, the O2 environmental-pressure block.  OM6 should not duplicate
habitat signals already included in OM3.
```

Feature claim:

```text
O2 is the core Battle Royale environmental force.  Every movement, interaction,
contact, and terrain-proxy feature should be interpreted conditional on
contraction pressure.
```

Guard:

- zone features are PUBG-native in the current project;
- Caldera currently has no usable zone-state table and cannot support O2
  supervised claims;
- phase-end population counts are valid only at completed phase cuts.

### O3. Movement / Rotation Behavior

Contract role:

```text
Represent behavioral response to contraction pressure.
```

| Field group | Level | Current source | Example features | Inclusion |
| --- | --- | --- | --- | --- |
| displacement and speed | `X_i(k)` | `team_phase_trajectory`, `team_snapshot_features` | `path_length_m`, `displacement_m`, `avg_speed_mps`, `speed_mps` | headline |
| path shape | `X_i(k)` | `team_phase_trajectory` | `sinuosity`, `rotation_timing`, `safe_alignment` | headline |
| entry timing | `X_i(k)` | `team_phase_trajectory`, `team_phase_habitat_features` | `zone_entry_time`, `hab_rotation_delay_ratio` | headline |
| vehicle mobility | `X_i(k)` | `team_phase_state`, `team_snapshot_features` | `vehicle_ratio`, `vehicle_speed_mean`, `vehicle_speed_max` | headline / resource interaction |

Feature claim:

```text
O3 does not by itself prove strategic intention.  It measures realized movement
response under a phase-conditioned environment.
```

Guard:

- path summaries must be computed only through cut `k`;
- route-density variants using other teams are analyst-view telemetry and
  should be grouped with O7 or O6 sensitivity blocks.

### O4. Interaction / Threat Exposure

Contract role:

```text
Measure exposure to hostile interaction and realized combat pressure.
```

| Field group | Level | Current source | Example features | Inclusion |
| --- | --- | --- | --- | --- |
| damage exchange | `X_i(k)` | `team_phase_combat` | `damage_dealt`, `damage_taken`, `damage_events_dealt`, `damage_events_taken` | headline |
| knock / kill pressure | `X_i(k)` | `team_phase_combat` | `groggy_dealt`, `groggy_taken`, `kills_dealt`, `kills_taken` | headline |
| encounter breadth | `X_i(k)` | `team_phase_combat` | `distinct_opponents_damaged`, `distinct_opponents_damaged_by`, `combat_partner_count` | headline |
| timing and nearby combat | `X_i(k)` | `team_phase_combat`, `team_phase_exposure` | `combat_initiated`, `nearby_enemy_combat_events_200m`, `third_party_exposure_count`, `recent_damage_taken` | headline / sensitivity |
| contact edges | `E_ij(k)` | `team_phase_edge_cache`, snapshot graph | `combat`, `proximity_combat`, `combat_dmg_log`, `combat_kills` | graph headline |

Feature claim:

```text
O4 is interaction exposure and combat involvement.  It should be separated
from O1 survival capital even though damage also depletes capital.
```

Boundary with O7:

```text
O4 can contain realized hostile events and simple scalar ego exposure summaries
such as nearest enemy, local enemy density, or recent combat nearby.  Each
exposure summary must declare the source relation channel, radius/window, or
event family from which it is derived.  Edge streams, centralities, bridge
roles, and global graph state remain O7.
```

Guard:

- do not use victim final rank or post-cut combat;
- combat events attributed to snapshot `t` must belong to the trailing window,
  not a future window;
- self-elimination and suicide counts are descriptive or sensitivity-only.

### O5. Resource / Equipment / Mobility Support

Contract role:

```text
Represent resources that enable survival, combat readiness, healing, and
rotation.
```

| Field group | Level | Current source | Example features | Inclusion |
| --- | --- | --- | --- | --- |
| general loot activity | `X_i(k)` | `team_phase_loot` | `pickup_count`, `equip_count`, `use_count`, `unique_item_count` | secondary |
| combat readiness | `X_i(k)` | `team_phase_loot` | `weapon_pickup_count`, `weapon_equip_count`, `ammo_pickup_count`, `armor_pickup_count` | secondary |
| recovery resources | `X_i(k)` | `team_phase_loot` | `heal_use_count`, `boost_use_count` | secondary |
| high-value resources | `X_i(k)` | `team_phase_loot` | `carepackage_pickup_count`, `lootbox_pickup_count` | sensitivity |
| mobility support | `X_i(k)` | `team_phase_state`, `team_snapshot_features` | `vehicle_ratio`, `vehicle_speed_mean`, `vehicle_speed_max` | interaction with O3 |

Feature claim:

```text
O5 is support capacity.  It should not dominate the headline unless the paper
explicitly shifts from spatial survival mechanisms to equipment mechanisms.
```

Guard:

- item-use features can be consequences of damage; interpret them as support
  behavior, not pure exogenous capacity;
- lootbox pickup is partly post-elimination interaction and should be checked
  for phase-cut validity.

### O6. Spatial / Terrain / Environment Affordance

Contract role:

```text
Represent spatial-tactical affordances using PUBG-native proxies, not direct
terrain claims.
```

The spatial-tactical sub-vocabulary is owned by the `STA*` family in:

```text
docs/feature_engineering/SPATIAL_TACTICAL_AFFORDANCE_FEATURE_SPEC_2026-07-02.md
```

| Affordance construct | PUBG operational proxy | Level | Inclusion |
| --- | --- | --- | --- |
| vertical position | `pos_z`, `team_z_mean`, `team_z_percentile_alive`, `z_range_m` | `X_i(k)` / snapshot | headline |
| relative elevation advantage | `contact_delta_z_ij_m`, `z_advantage_vs_nearest_enemy_m`, `high_ground_contact_advantage` | `E_ij(k)` / window | headline target |
| zone-relative location | `rel_zone_x`, `rel_zone_y`, `dist_blue_boundary_m`, `inside_blue` | snapshot / node | headline |
| exposure/openness proxy | `avg_enemy_density_100m`, `min_nearest_enemy_m`, `path_near_enemy_ratio` | node / window | headline |
| empirical corridor pressure | `rotation_path_density_r100`, `route_conflict_count`, `rotation_corridor_pressure` | node / edge | after Z3 only |
| cover/concealment proxy | `contact_time_without_damage_s`, `stationary_under_contact_ratio`, `damage_while_contacted` | node / window | sensitivity only |
| mobility friction proxy | `vertical_speed_abs_mean`, `speed_drop_after_climb`, `movement_friction_proxy` | node / window | sensitivity only |

Feature claim:

```text
In PUBG, O6 is spatial-tactical proxying.  It does not measure true terrain
slope, building cover, line of sight, or mesh occlusion unless external map
geometry is added.
```

Guard:

- do not name PUBG features `terrain_slope`, `cover_object`, `building_dist`,
  or `line_of_sight` without geometry;
- do not require `pos_z`, relative height, terrain, or LOS for cross-game
  Battle Royale claims;
- the headline PUBG implementation stops at `Z3` from the spatial-tactical
  spec: vertical position, contact-gated relative height, and zone-pressure
  interactions;
- route conflict and corridor pressure are analyst-view telemetry, not
  player-available perception.

Caldera counterpart:

| Construct | Caldera role |
| --- | --- |
| terrain slope / relief / ruggedness | construct validation only |
| building distance / built mass | construct validation only |
| pairwise LOS | v1 geometry validation priority |
| tactical-state snapshots | figure / interpretation aid |
| survival/rank prediction | not supported by current Caldera labels |

### O7. Relation / Interaction Structure

Contract role:

```text
Represent the fact that Battle Royale survival is relational.
```

The network-structure subdesign is owned by:

```text
docs/feature_engineering/PUBG_NETWORK_STRUCTURE_FEATURE_SPEC_2026-07-02.md
```

| Field group | Level | Current source | Example features | Inclusion |
| --- | --- | --- | --- | --- |
| snapshot proximity edges | `E_ij(t)` | `team_proximity`, snapshot graph | `edge_snapshot_proximity`, distance gates | graph headline |
| snapshot combat edges | `E_ij(t)` | spatial/combat events | `edge_snapshot_combat`, trailing combat intensity | graph headline |
| advantage edges | `E_ij(k)` | `model_stgcn/v3/advantage_edges.py` | `d_over_ell`, `dz`, `dalive`, `dhealth`, `dzone`, `combat_dmg_log` | graph headline |
| phase network role | `X_i(k)` | `team_phase_network_features` | `net_combat_*`, `net_prox_*`, `net_copress_*` | ablation / interpretive |
| cumulative relation role | `X_i(k)` | `team_phase_network_features` | `cum_combat_*`, `cum_unique_*`, `cum_prox_*`, `cum_copress_*` | ablation / interpretive |

Feature claim:

```text
O7 is the relation field itself, not generic homophily smoothing.  If graph
models improve prediction, the interpretation must be tied back to specific
relation channels and edge ablations.
```

Guard:

- graph edges must be pre-cut and inside the alive/at-risk mask;
- full-telemetry graph summaries are analyst-view;
- graph model gains require edge controls or relation ablations before making
  a mechanism claim.

Overlap ownership rule:

```text
Graph edges, centralities, bridge roles, community/global metrics, and message
passing inputs are owned by O7.  Simple scalar ego exposure summaries may appear
in O4 only when they declare their O7 source graph/radius/window.
```

Examples:

- `net_prox_degree_100m`, `net_prox_degree_200m`, and related graph-degree
  features are O7 network-role features;
- O6 may reference them as spatial-exposure context but must not add duplicate
  copies in the OM6 spatial-tactical block;
- O4 owns realized combat/contact events and simple scalar ego exposure
  summaries, not graph role or global-structure features.

This keeps the OM4 -> OM5 -> OM6 ablation ladder from assigning the same signal
to multiple families.

### O8. Outcome / Win Condition

Contract role:

```text
Define the match or phase result.  In supervised PUBG experiments, this becomes
the target, mask, or evaluation label, never an input feature.
```

| Label | Current source | Use |
| --- | --- | --- |
| `target_defined` | `team_phase_labels` | at-risk mask |
| `hazard_next` | `team_phase_labels` | primary next-phase elimination target |
| `survives_next_phase` / `eliminated_next_phase` | `team_phase_labels` | alternative binary phrasing |
| `final_rank_norm` | `team_phase_labels` | auxiliary rank supervision / evaluation |
| `final_top3`, `final_top5`, `final_top10`, `final_winner` | `team_phase_labels` | auxiliary or descriptive labels |
| `last_phase_alive` | `team_phase_labels` | censoring / descriptive analysis |

Feature claim:

```text
O8 anchors the genre-level outcome and the supervised problem.  It must be
outside every input feature set.
```

Guard:

- all O8 fields are forbidden input features;
- final rank can supervise/evaluate PUBG, but not Caldera in the current data;
- survival claims are PUBG-only unless another label-rich Battle Royale dataset
  is added.

## Operational Ablation Ladder

Use `OM*` for operational-manifest model blocks.  Do not reuse `F*`, `A*`, or
`STA*` for supervised model blocks.

| Block | Added family | Purpose | Expected model preset link |
| --- | --- | --- | --- |
| `OM0` | conventional baseline / existing core | sanity baseline | existing `core` or legacy baseline |
| `OM1` | O1 survival capital | roster/health/capacity | `core` node features |
| `OM2` | O3 movement / rotation | behavioral response | `core` + STDB movement |
| `OM3` | O2 contraction / zone pressure | survival-environment change | `core_stdb` / `core_habitat` |
| `OM4` | O4 interaction / threat | interaction pressure | `core` combat features |
| `OM5` | O7 relation / interaction structure | dyadic survival structure | graph relation stream / `network` |
| `OM6` | O6 spatial-tactical proxy | vertical/contact/zone affordance | Z0-Z3 subset first |
| `OM7` | O5 resource / mobility support | secondary support mechanisms | resource sensitivity |

Recommended headline path:

```text
OM0 -> OM1 -> OM2 -> OM3 -> OM4 -> OM5 -> OM6
```

`OM7` should be reported as sensitivity unless resource mechanisms become a
primary paper claim.

## Spatial-Tactical Z Ladder For PUBG

The first PUBG O6 implementation should follow this restricted ladder.  This
ladder is not the universal Battle Royale O6 ladder because it relies on
PUBG's available `pos_z` proxy.

| Block | Meaning | Headline status |
| --- | --- | --- |
| `Z0` | no spatial-tactical proxy beyond baseline | baseline |
| `Z1` | absolute / percentile vertical position | allowed |
| `Z2` | contact-gated relative height advantage | allowed |
| `Z3` | Z2 plus zone-pressure interaction | headline stopping point |
| `Z4` | route conflict / corridor pressure | defer until Z3 signal |
| `Z5` | behavior-under-pressure cover/friction proxies | sensitivity only |

Reason:

```text
Z1-Z3 are cheap, defensible with current PUBG telemetry, and directly test the
paper's spatial-tactical bridge.  Z4-Z5 are useful but more analyst-view and
more confounded with combat/roster mechanisms.
```

For a `PLANAR_2D` or `GRID_2D` game, replace `Z1-Z3` with a planar O6 ladder:

| Block | Meaning |
| --- | --- |
| `P0` | no spatial-affordance proxy beyond baseline |
| `P1` | zone-relative position / boundary distance |
| `P2` | local density / proximity exposure |
| `P3` | route, chokepoint, or sector conflict under zone pressure |

Do not import PUBG relative-height features into a game profile that does not
observe height or layer advantage.

## Current Code Mapping

| Code surface | Contract role |
| --- | --- |
| `stdb.v_team_phase_feature_long` | base phase-window table with O0-O5/O8 fields |
| `stdb.v_team_phase_feature_v2` | joined phase table with legacy F6/F7 implementation columns |
| `stdb.team_snapshot_features` | 10s snapshot node stream |
| `stdb.team_phase_edge_cache` | phase relation cache for proximity/combat/co-pressure |
| `stdb.team_phase_network_features` | O7 node summaries; legacy code name F6 |
| `stdb.team_phase_habitat_features` | O2 x O6 habitat interaction features; ablation owner OM3; legacy code name F7 |
| `model_stgcn/v3/spec.py` `core` | O0/O1/O3/O4/O5 conventional block |
| `model_stgcn/v3/spec.py` `stdb` | STDB spatial/zone/movement block |
| `model_stgcn/v3/spec.py` `network` | O7 network-role block |
| `model_stgcn/v3/spec.py` `habitat` | O2 habitat-pressure block |
| `model_stgcn/v3/advantage_edges.py` | O7 edge stream with O6 relative-height bridge |

Legacy implementation names such as F6 network-role and F7 habitat-pressure may
remain in code for reproducibility.  Manuscript and design documents should map
them to O7 relation structure and O2 x O6 habitat interaction rather than
introducing new `F*` meanings.

## Forbidden Inputs

The following are label/outcome fields and must never appear in a model input
feature list:

```text
target_defined
hazard_next
survives_next_phase
eliminated_next_phase
final_rank
final_rank_norm
final_top3
final_top5
final_top10
final_winner
last_phase_alive
won
rank
victim_rank
match_end_rank
match_end_result
```

This list is synchronized with `model_stgcn/v3/spec.py` by
`tests/test_feature_contract_docs.py`.

## Feature Admission Checklist

A new PUBG feature may enter the manifest only when all questions are answered:

1. Which ontology family owns it: O0-O8?
2. Is it `U(k)`, `X_i(k)`, `E_ij(k)`, snapshot-level, or label-only?
3. What is the exact source table or extraction script?
4. What is the cut rule and clock?
5. Is it player-available, analyst-view, or forbidden future information?
6. What is its conceptual portability layer and measurement observability class?
7. Is it a direct measurement or a proxy?
8. If it is habitat, is it assigned to OM3 as O2 x O6?
9. If it is exposure, which O7 graph/radius/window does it marginalize?
10. If it is information/intel, which part is O5, O4, O6, or O7?
11. Which ablation block first uses it?
12. What claim is invalid if this feature fails or is masked?

Features that cannot answer these questions belong in exploratory notebooks,
not in the supervised paper run.

## Claim Boundary

Allowed headline claim:

```text
PUBG telemetry can instantiate a Battle Royale feature framework in which
phase-conditioned contraction pressure, movement response, interaction
exposure, relation structure, and spatial-tactical proxies jointly explain
next-phase survival and rank-related outcomes.
```

Allowed Caldera bridge:

```text
Caldera validates that the spatial-tactical affordance vocabulary can be
computed from richer 3D geometry, while PUBG supplies the supervised survival
axis.
```

Disallowed claim:

```text
PUBG currently measures true terrain slope, building cover, exact line of
sight, or mesh-based field of fire.
```

## Immediate Design Lock

For the next implementation pass:

1. keep `OM0-OM6` as the main supervised ladder;
2. implement O6 headline only through `Z3`;
3. keep O5 and `Z5` as sensitivity blocks;
4. keep Caldera self-supervision as future work;
5. use this manifest as the source of truth before launching long model runs.
