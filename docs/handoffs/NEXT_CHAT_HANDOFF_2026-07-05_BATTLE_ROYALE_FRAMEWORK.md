# Next Chat Handoff: Battle Royale Framework

Date: 2026-07-05
Class: `ContinuationState`
Lifecycle: `handoff`

## One-Line State

The current paper path is no longer ontology construction.  It is a canonical
four-step Battle Royale manuscript: genre formalization, PUBG official
telemetry representation/mapping, Warzone / Caldera geometry bridge for
constructs official PUBG telemetry cannot directly observe, and PUBG
phase-survival accumulation plus final win/rank prediction.  The Caldera RQ3
bridge artifact is complete.  O7 advantage-edge is supplementary PUBG
null-control evidence, not the paper spine.

## Current Claim Level

Use:

```text
We propose an ontology-guided analytic framework for Battle Royale survival and
validate its central pressure-habitat mechanism through cohort-separated PUBG
prediction experiments.
```

Do not use:

```text
We prove or establish the universal ontology of the Battle Royale genre.
```

The preferred vocabulary is:

```text
controlled mechanism vocabulary
ontology-guided analytic framework
lightweight domain vocabulary
```

## What Is Supported Now

### 0. Paper Research Question Spine

Canonical flow:

```text
1. Battle Royale genre formalization.
2. PUBG official-telemetry representation and mapping to the formalization.
3. Terrain / geometry constructs missing from PUBG official telemetry and
   represented through Warzone / Caldera.
4. PUBG phase-wise survival accumulation and final win/rank prediction.
```

Source of truth:

```text
docs/research_framing/BATTLE_ROYALE_MANUSCRIPT_CANONICAL_FLOW_2026-07-05.md
docs/research_framing/BATTLE_ROYALE_RESEARCH_QUESTION_CHAIN_2026-07-05.md
```

This is the manuscript order.  Auxiliary experiments should not displace it.

### 1. Problem Formulation

Battle Royale is formulated as:

```text
phase-conditioned multi-agent survival
```

The primary supervised unit is:

```text
at-risk team x cut phase
```

The primary target is:

```text
next-phase elimination hazard
```

Final rank and top-k are evaluated as phase-cut forecasts and cumulative
survival/rank bridge outputs, not as one-shot post-game aggregate prediction.

The manuscript should use this prediction chain:

```text
next-phase hazard lambda_i(k)
-> cumulative survival S_i(k+1) = S_i(k) * (1 - lambda_i(k))
-> phase-cut final rank / top-k / win forecast
```

### 2. Framework Vocabulary

O0-O8 are locked for the current paper path as a lightweight controlled
vocabulary:

| Family | Meaning |
| --- | --- |
| `O0` | rule / match context |
| `O1` | survival capital |
| `O2` | contraction / elimination pressure |
| `O3` | movement / rotation response |
| `O4` | interaction / threat exposure |
| `O5` | resources / equipment / mobility support |
| `O6` | spatial / terrain / environment affordance |
| `O7` | relation / interaction structure |
| `O8` | outcome / win condition |

### 3. PUBG Official-Telemetry Mapping And Central Evidence

The manuscript is now assembled by problem -> method -> experiment/result ->
boundary, not by experiment chronology:

```text
docs/research_framing/BATTLE_ROYALE_PROBLEM_METHOD_EXPERIMENT_STRUCTURE_2026-07-05.md
```

The next-action plan is:

```text
docs/paper_writing/BATTLE_ROYALE_MANUSCRIPT_ASSEMBLY_PLAN_2026-07-05.md
```

The PUBG section must first show how official telemetry instantiates the
formalization:

Source table:

```text
docs/data_layer/PUBG_OFFICIAL_TELEMETRY_OBSERVABILITY_TABLE_2026-07-05.md
```

| Formal object | PUBG representation |
| --- | --- |
| `(m, i, k)` | match, team, phase cut |
| `U_m(k)` | map/mode/team size/phase context |
| `C_i(k)` | survival capital and team state |
| `Z_m(k)` | contraction / safe-zone pressure |
| `B_i(k)` | movement and rotation response |
| `H_i(k)` | pressure-conditioned habitat |
| `E_i(k)` | threat / exposure summaries |
| `Q_i(k)` | relation summaries or supplementary edge stream |
| `Y_i(k+1)` | next-phase hazard and final-outcome labels |

The central mechanism is:

```text
O2 contraction / elimination pressure x O6 affordance -> H, survival habitat
```

In the OM ladder this is tested as:

```text
OM2 movement -> OM3 pressure / habitat
```

The completed required-cohort OM ladder found:

| Result | Value |
| --- | ---: |
| Required cohorts | 13 |
| Positive `OM2 -> OM3` hazard AUC deltas | `13/13` |
| Positive map groups | `8/8` |
| Median `OM2 -> OM3` hazard AUC delta | about `+0.010` |

Phase-stratified and bootstrap analyses strengthened this:

- same-phase early hazard support remains after removing pooled phase-base-rate
  concerns;
- early/mid rank and top-10 patterns support the hazard-to-rank bridge;
- paired match-cluster bootstrap supports the pressure-habitat block in the
  pre-specified cells.

### 4. O1 Survival Capital Boundary

O1 has direct univariate association with next-phase hazard, but adds little
incrementally in the cumulative OM ladder after phase/team-size/context.

Allowed:

```text
O1 survival capital is predictive in isolation but largely redundant after the
current context and ladder ordering.
```

Avoid:

```text
Survival capital has no relation to survival.
```

### 5. O6Z Spatial-Affordance Sensitivity

The corrected O6 Z-ladder was implemented and run.

Implemented Z features:

```text
team_z_mean_norm
team_z_percentile_alive
team_z_range_norm
contact_delta_z_mean_norm
contact_delta_z_positive_ratio
high_ground_contact_advantage
high_ground_x_boundary_pressure
z_percentile_x_hab_crowding_pressure
contact_delta_z_x_outside_safe_ratio
```

The OM6Z run completed all 13 exact map / perspective / team-size cohorts.

Result:

```text
WEAK_OR_NULL_O6Z_SUPPORT_IN_PUBG
```

Key deltas:

| Comparison | Metric | Median delta | Positive cohorts |
| --- | --- | ---: | ---: |
| `OM5 -> OM6Z` | hazard AUC | `-0.0012` | `4/13` |
| `OM5 -> OM6Z` | rank Spearman | `-0.0034` | `2/13` |
| `OM5 -> OM6Z` early `1-2` | hazard AUC | `+0.0004` | `7/13` |
| `OM5 -> OM6Z` mid `3-4` | hazard AUC | `-0.0036` | `1/13` |

Interpretation:

```text
PUBG `pos_z`-only vertical proxies do not validate fine-grained terrain or
high-ground affordance after OM5.
```

This supports the observability framing:

| Dataset | Role |
| --- | --- |
| PUBG | label-rich, geometry-poor supervised validation of pressure-habitat |
| Caldera / Warzone | geometry-rich, label-poor construct validation for richer O6 |

### 6. Caldera / Warzone RQ3 Observability-Gap Bridge

The bounded Caldera artifact package is complete, but the correct RQ3 claim
surface is the bridge:

```text
PUBG official telemetry gap
-> Warzone / Caldera geometry representation
-> current descriptive evidence
-> claim status
```

Generated outputs:

```text
docs/spatial_world_modeling/PUBG_WARZONE_OBSERVABILITY_GAP_BRIDGE_2026-07-05.md
reports/caldera_3d/construct_validation/caldera_o6_pubg_warzone_observability_bridge.md
reports/caldera_3d/construct_validation/caldera_o6_pubg_warzone_observability_bridge_board.png
reports/caldera_3d/construct_validation/caldera_o6_pubg_gap_situation_cards.md
reports/caldera_3d/construct_validation/caldera_o6_construct_validation_composite.png
reports/caldera_3d/construct_validation/caldera_o6_construct_validation_descriptive_table.md
reports/caldera_3d/construct_validation/caldera_o6_construct_validation_manifest.json
docs/spatial_world_modeling/CALDERA_O6_CONSTRUCT_VALIDATION_ARTIFACT_RESULT_2026-07-05.md
```

Bridge rows:

| PUBG official telemetry gap family | Warzone / Caldera representation | Status |
| --- | --- |
| terrain surface and vertical context | mesh-grounded `terrain_z_m`, height above terrain, relative elevation, relief | supported descriptive construct |
| rough terrain mobility | slope, relief, ruggedness around path samples | supported descriptive construct |
| compound and building affordance | building distance, building density, cover proxy from semantic building triangles | supported descriptive construct |
| route corridor context | road/runway semantic mesh proximity | supported descriptive construct |
| local tactical-state labeling | rules over terrain/building/corridor features | supported descriptive construct, not outcome supervision |
| visibility / occlusion / LOS | future mesh ray-cast over occluding geometry | pending extension, not claimed |

Key descriptive values:

| Quantity | Value |
| --- | ---: |
| feature rows | 120 |
| life segments | 1 |
| motion gap rows | 0 |
| redeploy-gap candidate rows | 0 |
| `height_above_terrain_m` median | `0.18m` |
| `building_dist_m` median | `0.91m` |
| `cover_building_r25` | `85.00%` |

Allowed:

```text
Warzone / Caldera shows how geometry-rich data can represent terrain, route,
building, and local tactical-state constructs that official PUBG telemetry
cannot directly observe.
```

Avoid:

```text
Caldera validates the supervised PUBG survival/rank model.
```

### 7. Supplementary O7 Advantage-Edge Null-Control

The gen-1 / gen-2 boundary was resolved, the edge-shuffle null was audited, and
the gen-2 A/B/C run completed all 13 required cohorts.

Manuscript placement:

```text
supplementary PUBG relation-topology null-control
```

Non-role:

```text
not the core answer to RQ1, RQ3, or RQ4
```

Locked contrast:

```text
B = observed contact-gated relative-advantage edges
C = same graph pathway with edge_shuffle null
primary topology-specific contrast = B - C
```

Hazard-AUC macro result:

| Phase group | `B-C` positive cohorts | CI-positive cohorts | Median `B-C` |
| --- | ---: | ---: | ---: |
| all | `13/13` | `6/13` | `+0.0023` |
| early `1-2` | `12/13` | `4/13` | `+0.0027` |
| mid `3-4` | `8/13` | `3/13` | `+0.0007` |
| late `5-6` | `9/13` | `3/13` | `+0.0038` |
| final `7` | `8/13` | `0/13` | `+0.0071` |

Hazard-implied rank bridge:

```text
Spearman(final_rank_norm, hazard_prob)
```

| Phase group | `B-C` positive cohorts | CI-positive cohorts | Median `B-C` |
| --- | ---: | ---: | ---: |
| all | `9/13` | `7/13` | `+0.0081` |
| early `1-2` | `9/13` | `5/13` | `+0.0062` |
| mid `3-4` | `11/13` | `8/13` | `+0.0168` |
| late `5-6` | `11/13` | `5/13` | `+0.0221` |
| final `7` | `8/13` | `2/13` | `+0.0164` |

Allowed:

```text
As supplementary evidence, observed contact-gated relative-advantage edges
carry topology-specific signal over an edge-shuffled null in PUBG.
```

Avoid:

```text
The O7 model has an independent rank head.
```

Also avoid:

```text
O7 is a core mechanism result of the paper.
```

## Required Cohorts

All headline training remains separated by exact cohort.  Do not pool these for
headline ablations:

- Erangel TPP squad;
- Taego TPP squad;
- Miramar / Desert TPP squad;
- Rondo TPP squad;
- Sanhok TPP squad;
- Vikendi TPP squad;
- Paramo TPP squad;
- Karakin TPP squad;
- Erangel TPP solo;
- Erangel TPP duo;
- Taego TPP duo;
- Taego FPP squad;
- Erangel FPP squad.

## Key Documents

### Operational Board

```text
docs/experiment_design/BATTLE_ROYALE_RESEARCH_EXECUTION_CHECKLIST_2026-07-03.md
```

### Framework Claim Boundary

```text
docs/research_framing/BATTLE_ROYALE_ANALYTIC_FRAMEWORK_STATUS_2026-07-04.md
docs/research_framing/BATTLE_ROYALE_ANALYTIC_FRAMEWORK_NARRATIVE_FOR_MANUSCRIPT_2026-07-04.md
```

### Core Results

```text
docs/experiment_design/BATTLE_ROYALE_EXPERIMENT_EVIDENCE_LEDGER_2026-07-05.md
docs/experiment_design/BATTLE_ROYALE_OM_LADDER_RESULT_FEATURE_AUDIT_2026-07-04.md
docs/experiment_design/BATTLE_ROYALE_PHASE_STRATIFIED_OM_RESULT_2026-07-04.md
docs/experiment_design/BATTLE_ROYALE_PHASE_GROUP_BOOTSTRAP_RESULT_2026-07-04.md
docs/experiment_design/BATTLE_ROYALE_O6_Z_LADDER_RESULT_2026-07-05.md
docs/spatial_world_modeling/PUBG_WARZONE_OBSERVABILITY_GAP_BRIDGE_2026-07-05.md
docs/spatial_world_modeling/CALDERA_O6_CONSTRUCT_VALIDATION_ARTIFACT_RESULT_2026-07-05.md
docs/experiment_design/BATTLE_ROYALE_O7_REQUIRED_COHORT_HAZARD_RESULT_2026-07-05.md
docs/experiment_design/BATTLE_ROYALE_O7_REQUIRED_COHORT_RANK_BRIDGE_RESULT_2026-07-05.md
```

### Output Roots

```text
model_stgcn/checkpoints/br_om_ladder_required_v1/
model_stgcn/checkpoints/br_om6z_required_v1/
model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/
reports/caldera_3d/construct_validation/
```

Important result files:

```text
model_stgcn/checkpoints/br_om_ladder_required_v1/om_ladder_metrics_all_cohorts.csv
model_stgcn/checkpoints/br_om_ladder_required_v1/phase_stratified_deltas.csv
model_stgcn/checkpoints/br_om_ladder_required_v1/bootstrap_phase_group_core_b2000/om_paired_bootstrap_phase_groups.csv
model_stgcn/checkpoints/br_om6z_required_v1/om6z_metrics_all_cohorts.csv
model_stgcn/checkpoints/br_om6z_required_v1/om6z_vs_om5_om6_comparison.csv
model_stgcn/checkpoints/br_om6z_required_v1/om6z_phase_deltas_vs_om5_om6.csv
model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/o7_required_13cohort_phase_group_deltas.csv
model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/o7_required_13cohort_summary.json
model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/o7_required_13cohort_rank_bridge_phase_group_deltas.csv
model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/o7_required_13cohort_rank_bridge_summary.json
reports/caldera_3d/construct_validation/caldera_o6_construct_validation_composite.png
reports/caldera_3d/construct_validation/caldera_o6_construct_validation_descriptive_table.md
reports/caldera_3d/construct_validation/caldera_o6_pubg_warzone_observability_bridge.md
reports/caldera_3d/construct_validation/caldera_o6_pubg_warzone_observability_bridge_board.png
reports/caldera_3d/construct_validation/caldera_o6_pubg_gap_situation_cards.md
```

## Code Changes To Remember

The corrected O6Z work added or touched:

```text
stdb/z_affordance_feature_etl.py
tools/run_o6z_required_cohort_etl.py
tools/audit_o6_z_ladder_availability.py
scripts/run_br_om6z_required_cohorts.sh
stdb/queries.py
stdb/v2_view.py
model_stgcn/v3/spec.py
model_stgcn/v3/experiment_matrix.py
model_stgcn/v3/run_matrix.py
tests/test_v2_node_features.py
tests/test_v3_factorized.py
scripts/assemble_caldera_o6_construct_validation_artifacts.py
```

Verification passed:

```text
python3 -m py_compile ...
python3 -m pytest tests/test_v2_node_features.py tests/test_v3_factorized.py -q
python3 -m py_compile scripts/assemble_caldera_o6_construct_validation_artifacts.py
python3 scripts/assemble_caldera_o6_construct_validation_artifacts.py
```

Observed result:

```text
10 passed, 2 skipped
```

The O7 hazard readout added or touched:

```text
tools/audit_o7_advantage_edge_shuffle.py
tools/analyze_o7_advantage_readout.py
tools/analyze_o7_advantage_rank_bridge.py
tools/aggregate_o7_advantage_readout.py
scripts/adv_precompute.py
scripts/run_br_o7_advantage_edge_early_readout.sh
scripts/run_br_o7_rank_bridge_analysis.sh
tests/test_v3_edge_controls.py
```

## Next Critical Path

The next experiment should not be another O6Z rerun unless the feature
definition changes.

The supplementary O7 null-control is complete under the gen-2 required-cohort
A/B/C contract:

```text
docs/experiment_design/BATTLE_ROYALE_O7_REQUIRED_COHORT_HAZARD_RESULT_2026-07-05.md
docs/experiment_design/BATTLE_ROYALE_O7_REQUIRED_COHORT_RANK_BRIDGE_RESULT_2026-07-05.md
```

Current O7 boundary:

```text
supplementary / defensive evidence only
hazard AUC B-C positive 13/13 overall and 12/13 early
hazard-implied rank bridge B-C positive 9/13 early and 11/13 mid
rank bridge is not an independent rank-head metric
late 5-6 remains secondary / exploratory
```

Next work:

```text
Move to manuscript figure/prose drafting around RQ1-RQ4.
```

Use the completed RQ3 artifact:

```text
reports/caldera_3d/construct_validation/caldera_o6_pubg_warzone_observability_bridge.md
reports/caldera_3d/construct_validation/caldera_o6_pubg_warzone_observability_bridge_board.png
reports/caldera_3d/construct_validation/caldera_o6_construct_validation_composite.png
reports/caldera_3d/construct_validation/caldera_o6_construct_validation_descriptive_table.md
```

Its result and scope are documented in:

```text
docs/spatial_world_modeling/CALDERA_O6_CONSTRUCT_VALIDATION_ARTIFACT_RESULT_2026-07-05.md
docs/spatial_world_modeling/CALDERA_O6_CONSTRUCT_VALIDATION_ARTIFACT_SCOPE_2026-07-05.md
```

Pairwise LOS remains a stretch goal, not part of the completed first artifact.
