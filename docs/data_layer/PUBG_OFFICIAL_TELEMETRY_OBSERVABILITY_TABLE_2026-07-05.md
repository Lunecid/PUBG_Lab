# PUBG Official Telemetry Observability Table

Date: 2026-07-05
Class: `DataContract`
Lifecycle: `active`

## Purpose

This document is the manuscript-facing RQ2 table for exactly what PUBG official
telemetry provides, what the current project derives from it, what is usable
only as a label, and what PUBG official telemetry does not directly provide.

It supports the argument:

```text
Battle Royale genre formalization
-> PUBG official-telemetry representation and mapping
-> PUBG terrain / geometry observability gap
-> phase-wise survival and final outcome prediction
```

## Source Boundary

The local PUBG database should be described precisely:

- `pubg.*` is the current official telemetry-derived normalized L1 layer.
- `pubg.*` is not byte-identical official JSON telemetry.
- `stdb.*` is the derived research representation layer.
- The primary prediction unit is `(match_id, team_id, cut_phase = k)`.
- Inputs must use telemetry observed through phase cut `k`.
- Final outcomes and next-phase hazards are labels only, never input features.
- The current paper uses analyst-view telemetry, not literal player perception.

## Observability Classes

| Class | Meaning | Manuscript use |
| --- | --- | --- |
| `TELEMETRY_OBSERVED` | directly present as official telemetry-derived state, event, or metadata in L1 | can be described as observed PUBG telemetry |
| `DERIVED_PRE_CUT` | computed from official telemetry before the prediction cut | can be described as leakage-guarded representation |
| `PROXY_OBSERVED` | indirect proxy because the direct construct is absent | must be described as proxy evidence |
| `LABEL_OBSERVED` | outcome, target, mask, or evaluation label | target/evaluation only, forbidden as input |
| `NOT_PROVIDED` | not directly available from official PUBG telemetry in the current project | motivates RQ3 and Warzone / Caldera bridge |

## Official Telemetry To Formalization Map

| BR family / formal role | What official PUBG telemetry provides | Current normalized or derived representation | Observability class | Allowed manuscript use | Boundary |
| --- | --- | --- | --- | --- | --- |
| `O0` rule / match context, `U_m(k)` | match metadata, map, mode, perspective/team-size conventions, phase index, elapsed or phase time | `map_name`, `game_mode`, `match_team_size`, `camera_view_type`, cut phase and phase timing fields | `TELEMETRY_OBSERVED` plus `DERIVED_PRE_CUT` | define the rule context for each team-phase row and stratify cohorts | PUBG rule terms are the PUBG instantiation of the genre, not the genre definition itself |
| `O1` survival capital, `C_i(k)` | player/team identity, roster state, alive state, health-like state, damage/groggy traces where available | `alive_players_*`, `mean_health_avg`, `team_spread_*`, `vehicle_ratio`, phase-to-date survival-capital summaries | `TELEMETRY_OBSERVED` plus `DERIVED_PRE_CUT` | represent current team viability before the next phase | do not use final rank or future elimination as survival-capital input |
| `O2` contraction / elimination pressure, `Z_m(k)` | safe-zone and poison/blue-zone phase information, zone center/radius/timing, alive population context | `safe_radius_*`, `poison_radius_*`, `zone_gap_*`, edge pressure, alive-population and habitat-pressure summaries | `TELEMETRY_OBSERVED` plus `DERIVED_PRE_CUT` | represent the phase-conditioned survival environment | `blue zone` is PUBG-native; the portable construct is shrinking playable-area pressure |
| `O3` movement / rotation response, `B_i(k)` | timestamped actor/team positions and movement traces including `x/y/z` coordinates | `path_length_m`, displacement, speed, sinuosity, zone entry timing, safe-zone alignment, trajectory summaries | `TELEMETRY_OBSERVED` plus `DERIVED_PRE_CUT` | show how teams respond to phase pressure through rotation and movement | analyst-view movement summaries are not literal player perception |
| `O4` interaction / threat exposure, `E_i(k)` | combat, damage, groggy, kill, attacker/victim, and timing events | `damage_*`, `groggy_*`, `kills_*`, distinct-opponent counts, recent combat and third-party exposure summaries | `TELEMETRY_OBSERVED` plus `DERIVED_PRE_CUT` | represent observed interaction and threat exposure before the cut | telemetry does not directly encode intent, visual contact, or every unobserved threat |
| `O5` resources / support / mobility, `R_i(k)` | loot, equipment, item-use, healing/boosting, vehicle, care-package, and lootbox-like PUBG events where present | `heal_use_count`, `boost_use_count`, `carepackage_pickup_count`, `lootbox_pickup_count`, vehicle and support summaries | `TELEMETRY_OBSERVED` plus `DERIVED_PRE_CUT` | represent resource and mobility support as PUBG-specific instantiations | PUBG item names must be translated to portable resource/support constructs before cross-game claims |
| `O6` spatial / terrain affordance, position part | actor/team coordinates, zone-relative position, `pos_z`-like vertical coordinate proxies | `pos_z`, relative zone position, proximity/contact vertical deltas, spatial-exposure proxies | `PROXY_OBSERVED` | use as PUBG spatial proxy evidence and as a bounded O6 sensitivity | does not validate terrain mesh, cover, line-of-sight, building, or high-ground affordance |
| `O7` relation / interaction structure, `Q_i(k)` | no raw graph object, but positions and combat events allow pairwise relations to be derived before the cut | `team_phase_edge_cache`, `net_*`, `cum_*`, `d_over_ell`, contact-gated edge streams | `DERIVED_PRE_CUT` | use as analyst-view relation structure or supplementary edge-stream evidence | node summaries and graph-edge experiments are distinct tests; do not collapse them |
| `O8` outcome / win condition, `Y_i(k+1)` | final placement, win/top-k outcome, elimination timing, at-risk status | `hazard_next`, `final_rank_norm`, `target_defined`, top-k / win labels | `LABEL_OBSERVED` | define targets, masks, and evaluation labels for phase survival and final outcome | labels are forbidden as input features |

## What PUBG Official Telemetry Does Not Directly Provide

These are not failures of the PUBG supervised case.  They are the reason RQ3
needs a geometry-rich Warzone / Caldera bridge.

| Missing or weak construct | PUBG official telemetry status | Correct paper treatment |
| --- | --- | --- |
| terrain mesh below the actor | `NOT_PROVIDED` in current PUBG official telemetry layer | use Warzone / Caldera OpenUSD terrain as geometry-rich construct representation |
| slope, ruggedness, local relief | `NOT_PROVIDED`; `pos_z` alone is only a vertical proxy | do not claim PUBG validates fine terrain affordance |
| building / compound geometry | `NOT_PROVIDED` without external map assets | use Caldera semantic building geometry for RQ3 representation |
| road, runway, surface, and corridor semantics | `NOT_PROVIDED` without external map labels | treat as geometry/topology constructs, not PUBG official telemetry facts |
| line-of-sight, occlusion, cover objects | `NOT_PROVIDED` in the current official telemetry representation | leave LOS as future mesh ray-casting work unless implemented |
| literal player perception / fog-of-war state | `NOT_PROVIDED`; current system is analyst-view telemetry | avoid claiming player-observed information unless a player-view proxy is explicitly defined |

## Current Executable Availability Snapshot

From the current feature availability audit:

| Surface | Status |
| --- | --- |
| `stdb.v_team_phase_feature_v2` | 187 columns available |
| `f0_official_raw_phase` | 78 of 85 requested columns present; 7 missing; 0 forbidden hits |
| missing `f0_official_raw_phase` columns | `assists_count`, `suicide_count`, `landing_t`, `landing_x`, `landing_y`, `landing_distance`, `landing_team_spread_m` |
| `f4_current_conventional` | 57 of 57 present |
| `f5_stdb_spatial` | 18 of 18 present |
| `f6_network_phase_role_tier1` | 43 of 43 present |
| `f7_habitat_pressure_tier1` | 32 of 32 present |

Clean OM preset surface:

| Preset | Role | Total features | Status |
| --- | --- | ---: | --- |
| `om0_context` | O0 context | 3 | ready |
| `om1_survival_capital` | OM0 + O1 | 12 | ready |
| `om2_movement` | OM1 + O3 | 19 | ready |
| `om3_pressure_habitat` | OM2 + O2 and O2 x O6 habitat pressure | 66 | ready |
| `om4_threat_exposure` | OM3 + O4 | 79 | ready |
| `om5_relation_structure` | OM4 + O7 node-summary relation structure | 122 | ready |
| `om6_spatial_proxy` | OM5 + remaining PUBG spatial-exposure proxies | 125 | ready as proxy-only O6 |

## RQ2 Claim Boundary

Allowed:

```text
PUBG official telemetry can instantiate the Battle Royale formalization as
leakage-guarded team x phase-cut feature blocks across exact map, perspective,
and team-size cohorts.
```

Allowed:

```text
PUBG is the label-rich supervised case for phase survival and final
win/rank/top-k prediction.
```

Not allowed:

```text
Every Battle Royale formal construct is fully observed in PUBG official
telemetry.
```

Not allowed:

```text
PUBG official telemetry directly contains validated terrain mesh, building
mesh, route topology, line-of-sight, occlusion, cover, or surface semantics.
```

Not allowed:

```text
The Warzone / Caldera geometry bridge is a supervised survival or final-rank
prediction dataset under the current labels.
```

## Upstream Sources

- `docs/data_layer/BR_DATA_LAYER_CONTRACT.md`
- `docs/feature_engineering/PUBG_FEATURE_OPERATIONAL_MANIFEST_2026-07-02.md`
- `docs/experiment_design/BATTLE_ROYALE_FEATURE_AVAILABILITY_AUDIT_2026-07-03.md`
- `docs/research_framing/BATTLE_ROYALE_MANUSCRIPT_CANONICAL_FLOW_2026-07-05.md`
- `docs/spatial_world_modeling/PUBG_WARZONE_OBSERVABILITY_GAP_BRIDGE_2026-07-05.md`
