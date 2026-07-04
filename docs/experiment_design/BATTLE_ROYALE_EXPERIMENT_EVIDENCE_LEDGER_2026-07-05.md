# Battle Royale Experiment Evidence Ledger

Date: 2026-07-05
Class: `ExperimentResult`
Lifecycle: `active`

## Purpose

This document records what the project has tested so far, how each test was
run, what result it produced, and what claim can or cannot be made from it.

It is the bridge from internal experiment artifacts to manuscript claims.

The project should use evidence-status language:

| Term | Meaning |
| --- | --- |
| `supported` | backed by current artifacts and safe for manuscript use with stated scope |
| `bounded` | tested and found weak/null, clarifying the limit of the framework or dataset |
| `pending` | protocol exists or code exists, but the current evidence is not enough for a claim |
| `not claimed` | outside the current evidence or explicitly contradicted by audits |

Avoid using "proved" for causal, universal, or ontology-exhaustiveness claims.
The strongest current language is:

```text
The framework's central pressure-habitat mechanism is supported by
cohort-separated PUBG prediction experiments.
```

## Current Bottom Line

The core paper result is supported:

```text
Battle Royale can be formulated as a phase-conditioned multi-agent survival
process, and the central pressure-habitat block improves next-phase survival
prediction and phase-cut rank/top-k forecasts across independently trained PUBG
cohorts.
```

The current bounded results are also important:

```text
PUBG basic telemetry does not validate fine-grained terrain affordance through
`pos_z`-only O6Z features, and Caldera cannot be used as a supervised survival
or rank-prediction dataset under current labels.
```

## Evidence Status Summary

| Claim / question | Evidence status | Main evidence |
| --- | --- | --- |
| O0-O8 is not merely a PUBG feature list | `supported as framework vocabulary` | mini corpus: 15/15 source-rich games/modes map to O0-O8 without requiring O9 |
| OM0-OM6 can be executed without forbidden inputs on required PUBG cohorts | `supported` | feature availability audit, forbidden-list tests, cohort adapter, smoke runs |
| Pressure-habitat is the central supervised mechanism | `supported` | OM2->OM3 positive in 13/13 cohorts; median hazard AUC delta about `+0.010` |
| OM3 is not just pooled phase-base-rate encoding | `supported` | phase-stratified within-cut-phase gains and paired bootstrap |
| Hazard-to-rank bridge is meaningful | `supported with phase boundary` | early/mid rank Spearman and top-10 gains after OM3 |
| O1 survival capital is broken or irrelevant | `not supported` | O1 tensor integrity audit shows valid direct univariate hazard signal |
| O1 adds large independent value in the cumulative ladder | `not supported` | OM0->OM1 deltas near zero and bootstrap unsupported |
| O4 threat exposure is universally dominant | `not supported` | conditional, smaller gains; stronger after pressure creates denser environments |
| O7 node-summary relation features are a complete graph test | `not supported` | OM5 used `graph_encoder=none` and no edge stream |
| PUBG `pos_z` Z-ladder validates fine terrain/high-ground affordance | `bounded negative` | OM5->OM6Z fails pre-registered 9/13 sign threshold |
| Caldera can be used for supervised survival/rank prediction | `not supported` | no rank/survival labels, no zone states, redeploy-like discontinuities |
| Caldera can support geometry-rich O6 construct validation | `supported as construct role` | OpenUSD geometry + trajectories + semantic world modeling capability |
| O7 advantage-edge topology carries signal | `pending` | protocol locked; long run not yet launched under current required-cohort contract |

Scope note: the supervised PUBG ladder runs currently use a single training
seed (`42`).  The paired bootstrap covers test-match uncertainty, not training
seed variance.  The main OM3 pattern is still treated as strong because it
repeats across 13 independently trained, map / perspective / team-size separated
cohorts, but seed-repeat robustness remains a possible follow-up rather than a
completed claim.

## Experiment 1. Mini Corpus Coverage

### Question

```text
Is O0-O8 only a PUBG-specific feature list, or can it cover representative
Battle Royale variants?
```

### Method

Manual confirmatory corpus coding of 15 source-rich Battle Royale and
Battle-Royale-like games/modes using official, publisher-controlled, or store
sources.

The coding asked whether each game required a new top-level family beyond:

```text
O0 rule/context
O1 survival capital
O2 contraction / elimination pressure
O3 movement / rotation
O4 interaction / threat exposure
O5 resource / equipment / mobility support
O6 spatial / environment affordance
O7 relation / interaction structure
O8 outcome / win condition
```

### Artifact

```text
docs/research_framing/BATTLE_ROYALE_MINI_CORPUS_COVERAGE_TABLE_2026-07-04.md
```

### Result

Across 15 games/modes:

| Family | Coverage |
| --- | ---: |
| `O0` | 15/15 |
| `O1` | 15/15 |
| `O2` | 15/15 |
| `O3` | 15/15 |
| `O4` | 15/15 |
| `O5` | 15/15 |
| `O6` | 15/15 |
| `O7` | 15/15 |
| `O8` | 15/15 |

No new `O9` family was required.

Important refinements:

- `O2` must be contraction / elimination pressure, not only PUBG-style zone
  pressure.
- `O4` must be interaction / threat exposure, not only gun combat.
- `O6` must not require full 3D terrain; it generalizes to 2D maps, tracks,
  obstacle spaces, and puzzle grids through the observability axis.

### Supported Claim

```text
O0-O8 is sufficiently broad to serve as a genre-level analytic vocabulary for
the current Battle Royale prediction paper.
```

### Not Claimed

```text
The mini corpus proves an exhaustive universal ontology of every Battle Royale
game.
```

## Experiment 2. Feature Availability And Leakage Audit

### Question

```text
Can the ontology-derived OM0-OM6 feature ladder be computed for every required
PUBG cohort without forbidden outcome leakage?
```

### Method

The audit combined:

- DB/view inspection of `stdb.v_team_phase_feature_v2`;
- cohort registry checks by map / perspective / team-size;
- forbidden-input synchronization tests;
- feature-manifest validation;
- clean OM preset unit tests;
- one-match DB smoke;
- 30-match smoke runs on representative TPP squad, TPP duo, and FPP squad
  cohorts.

### Artifact

```text
docs/experiment_design/BATTLE_ROYALE_FEATURE_AVAILABILITY_AUDIT_2026-07-03.md
```

### Result

All 13 required cohorts had the necessary rows for:

- phase features;
- target-defined next-phase hazard labels;
- O7 network rows;
- O2 x O6 habitat rows;
- phase edge rows.

Representative data coverage examples:

| Cohort | Feature eligible | Graph eligible |
| --- | ---: | ---: |
| Erangel TPP squad | 6,205 | 6,204 |
| Taego TPP squad | 4,160 | 4,159 |
| Miramar / Desert TPP squad | 1,545 | 1,543 |
| Erangel TPP solo | 800 | 800 |
| Taego FPP squad | 435 | 435 |

The audit also added clean executable presets:

```text
om0_context
om1_survival_capital
om2_movement
om3_pressure_habitat
om4_threat_exposure
om5_relation_structure
om6_spatial_proxy
```

### Supported Claim

```text
The framework vocabulary is operationalizable as leakage-guarded PUBG feature
blocks on the required cohorts.
```

### Not Claimed

```text
Feature availability itself proves predictive utility.
```

Feature availability only made the supervised ladder possible.

## Experiment 3. Required-Cohort OM0-OM6 Ladder

### Question

```text
Which ontology-guided feature blocks add predictive signal when introduced in
a controlled cumulative ladder?
```

### Method

The OM ladder was trained independently for each exact map / perspective /
team-size cohort.

Run contract:

| Field | Value |
| --- | --- |
| Cohorts | 13 |
| Presets per cohort | 7 |
| Total result rows | 91 |
| Model | `mr` |
| Graph encoder | `none` |
| Edge mode | `none` |
| Seed | `42` |
| Epochs | 12 |
| Eval split | `test` |
| Metric unit | at-risk team-phase rows |

The required set covers 8 PUBG maps.  Deston is not included in the required
cohort contract because it did not meet the same evidence-ready cohort
threshold used for the locked map / perspective / team-size ladder.

The ladder was:

```text
OM0 context
-> OM1 survival capital
-> OM2 movement
-> OM3 pressure / habitat
-> OM4 threat exposure
-> OM5 relation node summaries
-> OM6 old spatial proxies
```

### Artifacts

```text
docs/experiment_design/BATTLE_ROYALE_OM_LADDER_RESULT_FEATURE_AUDIT_2026-07-04.md
model_stgcn/checkpoints/br_om_ladder_required_v1/om_ladder_metrics_all_cohorts.csv
model_stgcn/checkpoints/br_om_ladder_required_v1/om_ladder_metrics_summary.json
```

### Result

Sequential hazard AUC deltas:

| Step | Added construct | Median delta | Positive cohorts | Interpretation |
| --- | --- | ---: | ---: | --- |
| `OM0 -> OM1` | O1 survival capital | `-0.0007` | 5/13 | no stable incremental gain |
| `OM1 -> OM2` | O3 movement | `+0.0028` | 10/13 | small gain |
| `OM2 -> OM3` | O2 pressure + habitat | `+0.0103` | 13/13 | headline gain |
| `OM3 -> OM4` | O4 threat exposure | `+0.0020` | 10/13 | conditional small gain |
| `OM4 -> OM5` | O7 node summaries | `+0.0003` | 7/13 | near zero |
| `OM5 -> OM6` | old O6 spatial proxies | `+0.0010` | 10/13 | tiny, not planned O6 test |

Macro median-like performance pattern:

| Preset | AUC | Rank Spearman | Top10 AUC |
| --- | ---: | ---: | ---: |
| `om0_context` | `0.7375` | `0.5891` | `0.8474` |
| `om2_movement` | `0.7405` | `0.5912` | `0.8493` |
| `om3_pressure_habitat` | `0.7523` | `0.6173` | `0.8805` |
| `om6_spatial_proxy` | `0.7569` | `0.6237` | `0.8801` |

### Supported Claim

```text
The pressure-habitat block is the only universally positive block in the
required-cohort ladder and is the current supervised headline mechanism.
```

### Not Claimed

```text
OM5 proves O7 graph structure.
```

OM5 used only node-summary `net_*` and `cum_*` features with no graph message
passing or edge stream.

```text
OM6 proves O6 terrain affordance.
```

Old OM6 contained only three exposure-like proxies and overlapped with O4/O7.

## Experiment 4. Phase-Stratified OM Analysis

### Question

```text
Is the OM3 gain only a pooled phase-base-rate artifact, or does it remain
visible within the same cut phase?
```

### Method

No retraining was performed.  The analysis flattened `phase_metrics` from the
existing cohort-specific OM0-OM6 `eval_test.json` files and computed same-
cohort, same-phase sequential deltas.

### Artifacts

```text
docs/experiment_design/BATTLE_ROYALE_PHASE_STRATIFIED_OM_RESULT_2026-07-04.md
tools/collect_om_phase_metrics.py
model_stgcn/checkpoints/br_om_ladder_required_v1/phase_stratified_metrics.csv
model_stgcn/checkpoints/br_om_ladder_required_v1/phase_stratified_deltas.csv
```

### Result

For `OM2 -> OM3` hazard AUC:

| Phase group | Positive cells | Median delta | Interpretation |
| --- | ---: | ---: | --- |
| early `1-2` | 24/26 | `+0.0161` | strongest pressure-habitat support |
| mid `3-4` | 20/26 | `+0.0070` | within-phase signal persists |
| late `5-7` | 23/39 | `+0.0078` | phase 5/6 signal; phase 7 unstable |

By cut phase:

| Cut phase | Positive cohorts | Median hazard AUC delta |
| ---: | ---: | ---: |
| 1 | 13/13 | `+0.0154` |
| 2 | 11/13 | `+0.0168` |
| 3 | 10/13 | `+0.0066` |
| 4 | 10/13 | `+0.0075` |
| 5 | 12/13 | `+0.0116` |
| 6 | 7/13 | `+0.0068` |
| 7 | 4/13 | `-0.0345` |

Rank and top-10 also improved strongly in early/mid phases:

| Metric | Strong pattern |
| --- | --- |
| rank Spearman | `OM2 -> OM3` positive in 13/13 cohorts for phases 1-3 |
| top-10 AUC | `OM2 -> OM3` positive in all valid cohorts for phases 1-3 |

### Supported Claim

```text
The pressure-habitat mechanism is not merely a pooled phase-clock proxy.  It
adds within-phase predictive signal, especially before the sparse final tail.
```

### Not Claimed

```text
OM3 is uniformly beneficial in every late-phase cell.
```

Phase 7 is too sparse and unstable.

## Experiment 5. O1 Survival-Capital Integrity Audit

### Question

```text
Is the weak OM0->OM1 result caused by broken O1 feature construction?
```

### Method

No model was retrained.  The audit inspected O1 tensor values on the same
cohort-separated test splits used in the OM ladder.

Audited features:

```text
alive_players_start_norm
alive_players_end_norm
alive_players_min_norm
alive_players_avg_norm
mean_health_avg_norm
min_health_min_norm
health_slope_norm
team_spread_avg_norm
team_spread_max_norm
```

### Artifacts

```text
docs/experiment_design/BATTLE_ROYALE_O1_SURVIVAL_CAPITAL_INTEGRITY_AUDIT_2026-07-04.md
tools/audit_o1_survival_capital.py
model_stgcn/checkpoints/br_om_ladder_required_v1/o1_survival_capital_integrity_summary.json
```

### Result

Audit sample:

```text
103,860 test team-phase rows from 1,789 test matches
```

O1 features were finite, non-empty, and non-degenerate except for structurally
expected solo properties.

Direct univariate hazard association was directionally correct:

| Feature | Median Pearson with hazard |
| --- | ---: |
| `mean_health_avg_norm` | `-0.1527` |
| `alive_players_avg_norm` | `-0.1462` |
| `alive_players_min_norm` | `-0.1246` |
| `health_slope_norm` | `-0.1088` |

Negative values mean higher survival capital is associated with lower
next-phase elimination risk.

### Supported Claim

```text
O1 is valid and directly associated with survival, but its incremental value is
small in the current cumulative ladder.
```

### Not Claimed

```text
Health, alive player count, or survival capital are unimportant.
```

## Experiment 6. Phase-Group Paired Bootstrap

### Question

```text
Do the central OM deltas survive match-cluster paired uncertainty checks?
```

### Method

No model was retrained.

The analysis:

1. generated row-level predictions for all 91 OM cohort/preset models;
2. grouped rows by test match;
3. resampled test matches with replacement inside each cohort;
4. computed paired deltas between OM steps.

Bootstrap settings:

| Setting | Value |
| --- | --- |
| Repetitions | `B=2000` |
| Seed | `42` |
| Phase groups | early `1-2`, mid `3-4`, late `5-6` |
| Excluded | phase `7` |
| Metrics | hazard AUC, rank Spearman, top-10 AUC |
| Tested steps | `OM0->OM1`, `OM2->OM3`, `OM3->OM4` |

### Artifacts

```text
docs/experiment_design/BATTLE_ROYALE_PHASE_GROUP_BOOTSTRAP_RESULT_2026-07-04.md
tools/generate_om_prediction_csvs.py
tools/bootstrap_om_paired_deltas.py
model_stgcn/checkpoints/br_om_ladder_required_v1/predictions/
model_stgcn/checkpoints/br_om_ladder_required_v1/bootstrap_phase_group_core_b2000/om_paired_bootstrap_phase_groups.csv
```

### Result

For `OM2 -> OM3`:

| Phase group | Metric | Median delta | Positive cohorts | Supported CIs |
| --- | --- | ---: | ---: | ---: |
| early `1-2` | hazard AUC | `+0.0154` | 13/13 | 10/13 |
| early `1-2` | rank Spearman | `+0.0386` | 13/13 | 12/13 |
| early `1-2` | top-10 AUC | `+0.0317` | 13/13 | 11/13 |
| mid `3-4` | hazard AUC | `+0.0080` | 11/13 | 4/13 |
| mid `3-4` | rank Spearman | `+0.0531` | 13/13 | 11/13 |
| mid `3-4` | top-10 AUC | `+0.0537` | 11/11 | 10/11 |

### Supported Claim

```text
Paired match-cluster bootstrap supports the central pressure-habitat mechanism:
after movement is included, OM3 gives robust early hazard gains and early/mid
rank or top-10 gains.
```

### Not Claimed

```text
Every OM3 late-phase hazard cell is statistically supported.
```

Late hazard has weaker bootstrap support and should be secondary.

## Experiment 7. Corrected O6 Z-Ladder

### Question

```text
Does a corrected PUBG vertical / relative-height O6 block add predictive signal
after OM5?
```

### Method

This was an ETL-first sensitivity experiment.

First, the project audited whether Z1-Z3 features existed.  They did not.
Then it implemented:

```text
stdb.team_phase_z_affordance_features
```

and exposed 9 normalized Z features into the V3 phase tensor.

The corrected preset was:

```text
om6z_spatial_affordance = OM5 relation structure + Z1-Z3
```

The model was trained separately for all 13 required cohorts using locked OM
ladder splits.

### Artifacts

```text
docs/experiment_design/BATTLE_ROYALE_O6_Z_LADDER_FEATURE_AVAILABILITY_AUDIT_2026-07-04.md
docs/experiment_design/BATTLE_ROYALE_O6_Z_LADDER_RESULT_2026-07-05.md
stdb/z_affordance_feature_etl.py
tools/run_o6z_required_cohort_etl.py
scripts/run_br_om6z_required_cohorts.sh
model_stgcn/checkpoints/br_om6z_required_v1/om6z_metrics_all_cohorts.csv
model_stgcn/checkpoints/br_om6z_required_v1/om6z_vs_om5_om6_comparison.csv
model_stgcn/checkpoints/br_om6z_required_v1/om6z_phase_deltas_vs_om5_om6.csv
```

### Result

The corrected Z block did not pass the pre-registered support threshold.

Overall:

| Comparison | Metric | Median delta | Positive cohorts |
| --- | --- | ---: | ---: |
| `OM5 -> old OM6` | hazard AUC | `+0.0010` | 10/13 |
| `OM5 -> OM6Z` | hazard AUC | `-0.0012` | 4/13 |
| `old OM6 -> OM6Z` | hazard AUC | `-0.0020` | 2/13 |
| `OM5 -> OM6Z` | rank Spearman | `-0.0034` | 2/13 |

Confirmatory phase-group cells:

| Metric | Phase group | Median delta | Positive cohorts |
| --- | --- | ---: | ---: |
| hazard AUC | early `1-2` | `+0.0004` | 7/13 |
| hazard AUC | mid `3-4` | `-0.0036` | 1/13 |
| rank Spearman | early `1-2` | `-0.0039` | 2/13 |
| rank Spearman | mid `3-4` | `-0.0132` | 2/13 |

The best primary cell reached only 7/13 positive cohorts, below the
pre-registered 9/13 threshold.

### Supported Boundary

```text
PUBG `pos_z`-only vertical proxies do not robustly validate fine-grained O6
terrain or high-ground affordance after OM5.
```

### Important Interpretation

The old OM6 tiny gain likely came from O4/O7 exposure-style overlap rather than
clean spatial-affordance signal:

```text
old OM6: +0.0010, 10/13
corrected OM6Z: -0.0012, 4/13
```

That reversal is useful because it shows the correction sharpened the claim
boundary.

### Not Claimed

```text
O6 spatial affordance is useless.
```

Only the PUBG `pos_z`-only projection is bounded negative.

## Experiment 8. Caldera Dataset Boundary And O6 Construct Role

### Question

```text
Can Caldera / Warzone serve as a second supervised survival/rank dataset, or
only as geometry-rich construct validation?
```

### Method

Read-only DB audit of Caldera matches, agents, trajectory samples, outcome
fields, zone-state availability, and trajectory discontinuities.

### Artifacts

```text
docs/research_framing/CALDERA_DATASET_STATUS_REDECISION_2026-07-02.md
docs/research_framing/TERRAIN_AFFORDANCE_CAPABILITY_BOUNDARY_2026-07-02.md
docs/spatial_world_modeling/CALDERA_O6_CONSTRUCT_VALIDATION_ARTIFACT_SCOPE_2026-07-05.md
```

### Result

Caldera audit:

| Object / metric | Value |
| --- | ---: |
| matches | 36 |
| agents | 4,187 |
| trajectory samples | 598,175 |
| non-null `final_rank` | 0 |
| non-null `survived` | 0 |
| `zone_states` | 0 |
| agents with `dt > 30s` gap | 2,460 |
| `dt > 30s` gaps | 3,381 |
| gaps with `dt > 30s` and displacement `> 500m` | 1,272 |

Interpretation:

```text
The public Caldera breadcrumb layer includes redeploy-like discontinuities.
Death is not an absorbing state in this source, and phase/zone/rank labels are
not available for supervised survival or rank prediction.
```

### Supported Claim

```text
Caldera is geometry-rich but label-poor.  It should be used for O6
construct-validation artifacts, not supervised survival/rank prediction.
```

### Not Claimed

```text
Caldera validates the PUBG supervised model.
```

### Current Artifact Scope

Caldera artifact scope is locked to:

```text
1 composite figure + 1 descriptive statistics table
```

Pairwise LOS is a useful stretch goal, not the first required artifact.

## Experiment 9. O7 Edge-Stream Protocol

### Question

```text
Can observed relation topology carry signal beyond node summaries and graph
encoder smoothing?
```

### Method Status

This is not yet a completed experiment.  The protocol is locked before the run.

### Artifact

```text
docs/experiment_design/BATTLE_ROYALE_O7_EDGE_STREAM_NULL_GUARD_PROTOCOL_2026-07-05.md
```

### Locked Design

Headline stream:

```text
contact-gated relative-advantage edges
```

Three-arm contract:

| Arm | Meaning |
| --- | --- |
| `A` | OM5-equivalent node/global features, no edge stream |
| `B` | same nodes + graph encoder + observed advantage edges |
| `C` | same as B + `edge_shuffle` |

Primary contrast:

```text
B - C
```

Secondary contrast:

```text
B - A
```

Interpretation rule:

| Pattern | Interpretation |
| --- | --- |
| `B-C` positive in at least 9/13 cohorts for a primary cell | observed relation topology support |
| `B-A > 0` but `B-C ~= 0` | capacity / smoothing gain, not topology-specific signal |
| `B-C` below threshold | bounded negative for current O7 edge stream |

### Pending Work

Do not launch the long O7 run before:

```text
gen-1 / gen-2 result boundary note
edge-shuffle implementation audit
```

## Overall Manuscript Interpretation

### Strongest Current Contribution

```text
We formulate Battle Royale as phase-conditioned multi-agent survival and show
that a leakage-guarded pressure-habitat feature block improves next-phase
survival and phase-cut rank/top-k prediction across independently trained PUBG
cohorts.
```

### Evidence Chain

1. Mini corpus shows O0-O8 is not PUBG-only.
2. Feature audit shows the vocabulary can be operationalized without forbidden
   inputs.
3. OM ladder shows the central pressure-habitat block is the only universally
   positive block.
4. Phase stratification shows the result is not only a pooled phase artifact.
5. Paired bootstrap supports the early hazard and early/mid rank/top-k gains.
6. O1 audit rules out a cheap feature-construction bug.
7. O6Z bounds PUBG as geometry-poor for fine terrain affordance.
8. Caldera provides the geometry-rich construct-validation path.
9. O7 advantage-edge remains a pre-registered next test, not a completed claim.

### Claims Allowed Now

- Battle Royale survival is well-modeled as a phase-conditioned survival
  process.
- O0-O8 is a useful lightweight mechanism vocabulary for the current analytic
  framework.
- The central pressure-habitat mechanism is supported in PUBG.
- Phase-level habitat signal contributes beyond movement and phase context.
- O1 survival capital is valid but not a headline incremental block in this
  ladder.
- PUBG `pos_z`-only features do not validate fine-grained O6 terrain
  affordance.
- Caldera should be used for O6 construct validation, not supervised outcome
  prediction.

### Claims Not Allowed Now

- A universal formal Battle Royale ontology has been proven.
- Causal terrain or habitat effects have been identified.
- O7 graph topology has been proven useful.
- PUBG validates true terrain, cover, LOS, or building affordance.
- Caldera supports phase-wise survival or final-rank prediction.

## Immediate Next Step

Before launching new long experiments:

```text
Resolve the gen-1 / gen-2 result boundary.
```

The boundary note must decide which legacy / preliminary results are reusable
as manuscript evidence and which must be rerun under the current required-
cohort, feature, split, and claim-boundary contract.
