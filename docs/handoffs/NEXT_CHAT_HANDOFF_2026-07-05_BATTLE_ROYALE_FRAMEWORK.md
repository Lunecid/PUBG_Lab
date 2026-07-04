# Next Chat Handoff: Battle Royale Framework

Date: 2026-07-05
Class: `ContinuationState`
Lifecycle: `handoff`

## One-Line State

The current paper path is no longer ontology construction.  It is an
ontology-guided Battle Royale survival analysis framework with a robust central
PUBG result for pressure-habitat and a bounded negative O6Z sensitivity result
for PUBG `pos_z`-only spatial affordance.

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

### 3. Central PUBG Evidence

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
```

### Output Roots

```text
model_stgcn/checkpoints/br_om_ladder_required_v1/
model_stgcn/checkpoints/br_om6z_required_v1/
```

Important result files:

```text
model_stgcn/checkpoints/br_om_ladder_required_v1/om_ladder_metrics_all_cohorts.csv
model_stgcn/checkpoints/br_om_ladder_required_v1/phase_stratified_deltas.csv
model_stgcn/checkpoints/br_om_ladder_required_v1/bootstrap_phase_group_core_b2000/om_paired_bootstrap_phase_groups.csv
model_stgcn/checkpoints/br_om6z_required_v1/om6z_metrics_all_cohorts.csv
model_stgcn/checkpoints/br_om6z_required_v1/om6z_vs_om5_om6_comparison.csv
model_stgcn/checkpoints/br_om6z_required_v1/om6z_phase_deltas_vs_om5_om6.csv
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
```

Verification passed:

```text
python3 -m py_compile ...
python3 -m pytest tests/test_v2_node_features.py tests/test_v3_factorized.py -q
```

Observed result:

```text
10 passed, 2 skipped
```

## Next Critical Path

The next experiment should not be another O6Z rerun unless the feature
definition changes.

O7 protocol is now locked in:

```text
docs/experiment_design/BATTLE_ROYALE_O7_EDGE_STREAM_NULL_GUARD_PROTOCOL_2026-07-05.md
```

Locked O7 decision:

```text
headline stream = contact-gated relative-advantage edges
primary contrast = observed advantage edge graph - edge_shuffle null
three arms = A no graph, B observed advantage edge graph, C edge_shuffle null
U-layer global graph summaries excluded from this run
```

Do not launch the long O7 run yet.  First resolve:

```text
gen-1 / gen-2 result boundary
```

This needs a short decision note that says which legacy/preliminary results are
usable as manuscript evidence and which claims require reruns under the current
cohort / feature / split contract.

Parallel next work:

```text
Produce a Caldera O6 construct-validation artifact showing what geometry-rich
spatial affordance can measure even without supervised outcome labels.
```

Its scope is locked in:

```text
docs/spatial_world_modeling/CALDERA_O6_CONSTRUCT_VALIDATION_ARTIFACT_SCOPE_2026-07-05.md
```

Keep it to one composite figure and one descriptive table.  Pairwise LOS is a
stretch goal, not the first artifact requirement.
