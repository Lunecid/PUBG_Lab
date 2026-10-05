# Battle Royale Experiment Evidence Ledger

Date: 2026-07-05
Class: `ExperimentResult`
Lifecycle: `active`

## Purpose

This document records what the project has tested so far, how each test was
run, what result it produced, and what claim can or cannot be made from it.

It is the bridge from internal experiment artifacts to manuscript claims.
It is not the manuscript order.  The manuscript order is problem
formalization -> required methodology -> experiments/results -> claim
boundary.

The paper-writing rule is fixed in:

```text
docs/research_framing/BATTLE_ROYALE_PROBLEM_METHOD_EXPERIMENT_STRUCTURE_2026-07-05.md
```

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
The Battle Royale research program is organized as a four-step chain:
genre formalization -> PUBG official-telemetry representation and mapping ->
prior tabular/phase-aware prediction plus 10-second realtime elimination-risk
model -> phase-transition most-at-risk / most-advantaged team prediction.
```

The manuscript spine is fixed in:

```text
docs/research_framing/BATTLE_ROYALE_RESEARCH_QUESTION_CHAIN_2026-07-05.md
docs/research_framing/BATTLE_ROYALE_MANUSCRIPT_CANONICAL_FLOW_2026-07-05.md
```

## Current Bottom Line

The core paper path is:

```text
1. establish Battle Royale as a phase-conditioned multi-agent survival
   framework;
2. represent that formalization exactly in PUBG official telemetry through
   an explicit observability table and leakage-guarded team x phase-cut
   and 10-second snapshot features;
3. extend prior tabular and phase-aware PUBG prediction into a causal
   realtime elimination-risk model using 10-second telemetry;
4. use the realtime survival readout to identify the most unfavorable and most
   favorable teams at phase transitions.
```

The strongest completed supervised result remains:

```text
In PUBG, the pressure-habitat block improves next-phase survival prediction,
and phase-wise survival scores support phase-cut final-rank/top-k forecasts
across independently trained cohorts.
```

The newest RQ3 readout-form result adds:

```text
Phase-wise hazards can be accumulated into dynamic match-local survival-share
trajectories, making the output a changing survival probability process rather
than only a static phase score or post-game rank label.
```

The newest board-level result adds:

```text
Those trajectories can be converted into phase-transition risk and advantage
boards that identify high next-phase elimination risk and strong match-local
survival advantage.
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
| PUBG official telemetry observability is explicitly bounded | `supported as data contract` | official telemetry observability table separates direct telemetry, derived pre-cut features, proxy-only O6 signals, label-only outcomes, and not-provided geometry |
| OM0-OM6 can be executed without forbidden inputs on required PUBG cohorts | `supported` | feature availability audit, forbidden-list tests, cohort adapter, smoke runs |
| Pressure-habitat is the central supervised mechanism | `supported` | OM2->OM3 positive in 13/13 cohorts; median hazard AUC delta about `+0.010` |
| OM3 is not just pooled phase-base-rate encoding | `supported` | phase-stratified within-cut-phase gains and paired bootstrap |
| RQ3 comparison against prior-work/basic tabular baselines | `supported` | 13-cohort prior baseline result: representation-rich habitat/full features beat Sagong-style and basic official-telemetry baselines |
| RQ3 realtime survival readout | `supported with phase-cut boundary` | phase-wise LSTM hazards converted to cumulative survival and match-local survival share over 13 cohorts; one-match 10-second prefix demo exists |
| RQ4 phase-transition risk/advantage board | `supported with HGB-scoring boundary` | LSTM board: 10,616 phase boards, median risk top-1 0.7004, median risk top-3 0.9172, median advantage top-team final top-10 0.9490; seven-board comparison beats simple/prior boards on the intersected board universe, while pressure-habitat HGB is stronger on median phase-cut risk scoring |
| Graph architecture as core RQ4 explanation | `not retained` | bounded Erangel early readout: observed graph did not beat no-graph hazard AUC and shuffle controls were comparable |
| Hazard-to-rank bridge is meaningful | `supported with phase boundary` | early/mid rank Spearman and top-10 gains after OM3 |
| O1 survival capital is broken or irrelevant | `not supported` | O1 tensor integrity audit shows valid direct univariate hazard signal |
| O1 adds large independent value in the cumulative ladder | `not supported` | OM0->OM1 deltas near zero and bootstrap unsupported |
| O4 threat exposure is universally dominant | `not supported` | conditional, smaller gains; stronger after pressure creates denser environments |
| O7 node-summary relation features are a complete graph test | `not supported` | OM5 used `graph_encoder=none` and no edge stream |
| PUBG `pos_z` Z-ladder validates fine terrain/high-ground affordance | `bounded negative` | OM5->OM6Z fails pre-registered 9/13 sign threshold |
| Caldera can be used for supervised survival/rank prediction | `not supported` | no rank/survival labels, no zone states, redeploy-like discontinuities |
| Caldera can support geometry-rich O6 construct validation | `supported as supplementary construct role` | PUBG-Warzone observability gap bridge, completed composite figure, and descriptive O6 construct table from OpenUSD geometry and trajectories |
| Supplementary O7 advantage-edge null-control | `supported as auxiliary / defensive evidence` | one PUBG relation stream passes edge-shuffle null; useful for reviewer defense, not the paper spine |

Scope note: the supervised PUBG ladder runs currently use a single training
seed (`42`).  The paired bootstrap covers test-match uncertainty, not training
seed variance.  The main OM3 pattern is still treated as strong because it
repeats across 13 independently trained, map / perspective / team-size separated
cohorts, but seed-repeat robustness remains a possible follow-up rather than a
completed claim.

## RQ4 Addendum. Dynamic Probability Fluctuation

### Question

```text
Can the phase-wise Battle Royale survival formulation produce a dynamic
probability trajectory rather than only a static score or final-label
prediction?
```

### Method

The completed `snapshot_lstm_no_graph` phase-cut hazard predictions were
converted into:

```text
lambda_i(k) = predicted next-phase elimination hazard
S_i(k)      = product_{j <= k} (1 - lambda_i(j))
pi_i(k)     = S_i(k) / sum_j S_j(k)
```

The script carries cumulative survival forward through missing phase rows so
that the match-local denominator does not change merely because one team lacks
an explicit later readout.

### Artifact

```text
docs/experiment_design/BATTLE_ROYALE_RQ4_DYNAMIC_PROBABILITY_FLUCTUATION_RESULT_2026-07-06.md
tools/build_dynamic_probability_fluctuation.py
tools/build_realtime_prefix_probability_demo.py
reports/rq4_dynamic_probability_fluctuation/dynamic_probability_trajectories.csv
reports/rq4_dynamic_probability_fluctuation/dynamic_probability_example_match.png
reports/rq4_dynamic_probability_fluctuation/realtime_prefix_probability_trajectory.csv
reports/rq4_dynamic_probability_fluctuation/realtime_prefix_probability_example_match.png
```

### Result

The artifact covers:

| Field | Value |
| --- | ---: |
| cohorts | 13 |
| matches | 1,789 |
| teams | 33,584 |
| observed prediction rows | 103,860 |
| trajectory rows | 105,491 |
| median hazard AUC | 0.7441 |
| median latest top-5 AUC from `pi` | 0.9290 |
| median latest top-10 AUC from `pi` | 0.9033 |
| median latest rank Spearman | 0.7914 |

The descriptive 10-second prefix demo covers one Erangel TPP squad test match:

| Field | Value |
| --- | ---: |
| snapshots in match tensor | 200 |
| prefix probability rows | 2,234 |
| unique teams | 28 |
| represented prefix snapshots | 146 |

### Supported Claim

```text
The proposed Battle Royale survival formulation supports a dynamic phase-cut
probability readout: hazards update cumulative survival and match-local
survival share as a match progresses.
```

### Not Claimed

```text
The LSTM sequence model is the strongest overall predictor.
```

The strongest current hazard-AUC models remain the mechanism-aligned tabular
representation models.

Also do not claim:

```text
This is a separately validated 10-second real-time deployment model.
```

The aggregate result is dynamic at phase-cut resolution.  A descriptive
10-second prefix-readout demo exists, but strict streaming validation requires a
dedicated prefix-label evaluation protocol.

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
fields, zone-state availability, and trajectory discontinuities, followed by a
bounded artifact assembly from existing semantic basemap, trajectory,
terrain-experience, and feature-summary outputs.

### Artifacts

```text
docs/research_framing/CALDERA_DATASET_STATUS_REDECISION_2026-07-02.md
docs/research_framing/TERRAIN_AFFORDANCE_CAPABILITY_BOUNDARY_2026-07-02.md
docs/spatial_world_modeling/CALDERA_O6_CONSTRUCT_VALIDATION_ARTIFACT_SCOPE_2026-07-05.md
docs/spatial_world_modeling/CALDERA_O6_CONSTRUCT_VALIDATION_ARTIFACT_RESULT_2026-07-05.md
docs/spatial_world_modeling/PUBG_WARZONE_OBSERVABILITY_GAP_BRIDGE_2026-07-05.md
reports/caldera_3d/construct_validation/caldera_o6_pubg_warzone_observability_bridge.md
reports/caldera_3d/construct_validation/caldera_o6_pubg_warzone_observability_bridge_board.png
reports/caldera_3d/construct_validation/caldera_o6_pubg_gap_situation_cards.md
reports/caldera_3d/construct_validation/caldera_o6_construct_validation_composite.png
reports/caldera_3d/construct_validation/caldera_o6_construct_validation_descriptive_table.md
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

Completed construct artifact:

| Artifact quantity | Value |
| --- | --- |
| bridge artifact | PUBG official telemetry gap -> Warzone / Caldera geometry representation -> current evidence -> claim status |
| figure panels | semantic basemap, trajectory overlay, terrain-experience zoom, feature callout |
| descriptive feature rows | 120 |
| time window | `1155.360-1629.456s` |
| life segments | 1 |
| motion gap rows | 0 |
| redeploy-gap candidate rows | 0 |
| `height_above_terrain_m` median / p75 / max | `0.18` / `0.33` / `4.13` |
| `building_dist_m` median | `0.91` |
| `cover_building_r25` | `85.00%` |
| tactical states | `compound_movement=102`, `rough_mobility=18` |

Bridge rows supported now:

| PUBG official telemetry gap family | Warzone / Caldera representation | Status |
| --- | --- | --- |
| terrain surface and vertical context | terrain mesh + trajectory -> `terrain_z_m`, `height_above_terrain_m`, relative elevation, relief | supported descriptive construct |
| rough terrain mobility | mesh slope / relief / ruggedness around path samples | supported descriptive construct |
| compound and building affordance | semantic building triangles -> building distance, density, cover proxy | supported descriptive construct |
| route corridor context | road / runway mesh categories -> corridor proximity | supported descriptive construct |
| local tactical-state labeling | rules over geometry-derived terrain / building / corridor features | supported descriptive construct, not outcome supervision |
| visibility / occlusion / LOS | future mesh ray casting over occluding geometry | pending extension, not claimed |

### Supported Claim

```text
Caldera is geometry-rich but label-poor.  The completed bridge artifact
demonstrates how Warzone / Caldera can represent terrain, route, building, and
local tactical-state constructs that official PUBG telemetry cannot directly
observe.  This is construct evidence, not supervised survival/rank prediction.
```

### Not Claimed

```text
Caldera validates the PUBG supervised model.
```

### Completed Artifact Package

Caldera artifact scope was locked to and completed as:

```text
1 composite figure + 1 descriptive statistics table
```

Pairwise LOS is a useful stretch goal, not part of the completed first artifact
package.

## Experiment 9. Supplementary O7 Required-Cohort Null-Control

### Question

```text
Can observed relation topology carry signal beyond node summaries and graph
encoder smoothing?
```

### Manuscript Placement

This is not a core research-question experiment.

Correct placement:

```text
Supplementary PUBG relation-topology null-control.
```

It supports a boundary claim inside RQ2/RQ4:

```text
One specific PUBG relation stream is not reducible to graph capacity/smoothing
under an edge-shuffled null.
```

It does not answer RQ1, RQ3, or the main RQ4 pressure-habitat prediction claim.

### Method

The experiment used the pre-registered O7 advantage-edge protocol after the
gen-1 / gen-2 boundary decision and edge-shuffle audit.

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

Primary topology-specific contrast:

```text
B - C
```

Secondary graph-capacity / smoothing contrast:

```text
B - A
```

The first analyzer covers hazard AUC with match-cluster bootstrap over phase
groups.  A second bridge analyzer evaluates `Spearman(final_rank_norm,
hazard_prob)`.  The O7 advantage-edge model has no independent rank head, so
this is a hazard-implied rank bridge rather than a rank-head metric.

### Artifacts

```text
docs/experiment_design/BATTLE_ROYALE_O7_EDGE_STREAM_NULL_GUARD_PROTOCOL_2026-07-05.md
docs/experiment_design/BATTLE_ROYALE_O7_EDGE_SHUFFLE_AUDIT_2026-07-05.md
docs/experiment_design/BATTLE_ROYALE_O7_EARLY_READOUT_ERANGEL_2026-07-05.md
docs/experiment_design/BATTLE_ROYALE_O7_EARLY_READOUT_4COHORT_RESULT_2026-07-05.md
docs/experiment_design/BATTLE_ROYALE_O7_REQUIRED_COHORT_HAZARD_RESULT_2026-07-05.md
docs/experiment_design/BATTLE_ROYALE_O7_REQUIRED_COHORT_RANK_BRIDGE_RESULT_2026-07-05.md
model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/o7_required_13cohort_phase_group_deltas.csv
model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/o7_required_13cohort_summary.json
model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/o7_required_13cohort_rank_bridge_phase_group_deltas.csv
model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/o7_required_13cohort_rank_bridge_summary.json
tools/analyze_o7_advantage_readout.py
tools/analyze_o7_advantage_rank_bridge.py
tools/aggregate_o7_advantage_readout.py
```

### Result

All 13 required cohorts completed.

Macro hazard-AUC result:

| Phase group | `B-C` positive cohorts | `B-C` CI-positive cohorts | Median `B-C` | Weighted mean `B-C` | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| all | 13/13 | 6/13 | `+0.0023` | `+0.0028` | positive overall |
| early `1-2` | 12/13 | 4/13 | `+0.0027` | `+0.0033` | clears hazard sign threshold |
| mid `3-4` | 8/13 | 3/13 | `+0.0007` | `+0.0018` | below sign threshold |
| late `5-6` | 9/13 | 3/13 | `+0.0038` | `+0.0069` | secondary / exploratory |
| final `7` | 8/13 | 0/13 | `+0.0071` | `+0.0054` | sparse and unstable |

The pre-registered 9/13 sign threshold is met for the hazard-AUC early
phase-group cell, but not for the mid hazard-AUC cell.  The overall cell is
directionally unanimous, and late `5-6` also reaches 9/13, but late effects were
not registered as headline evidence.

Hazard-implied rank bridge result:

| Phase group | `B-C` positive cohorts | `B-C` CI-positive cohorts | Median `B-C` | Weighted mean `B-C` | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| all | 9/13 | 7/13 | `+0.0081` | `+0.0149` | positive overall |
| early `1-2` | 9/13 | 5/13 | `+0.0062` | `+0.0064` | clears bridge sign threshold |
| mid `3-4` | 11/13 | 8/13 | `+0.0168` | `+0.0136` | strongest bridge cell |
| late `5-6` | 11/13 | 5/13 | `+0.0221` | `+0.0214` | secondary / exploratory |
| final `7` | 8/13 | 2/13 | `+0.0164` | `+0.0351` | sparse and unstable |

The bridge result meets the 9/13 sign threshold in both primary phase groups,
but it must be described as hazard-implied rank ordering because no O7 rank head
was trained.

### Supported Claim

```text
As a supplementary null-control, observed contact-gated relative-advantage
edges carry topology-specific signal over an edge-shuffled null in hazard AUC
and in a hazard-implied final-rank ordering bridge.
```

### Not Claimed

```text
The O7 model's independent rank head validates relation topology.
```

The O7 advantage-edge model has no independent rank head.

```text
All graph relation structure is predictive.
```

The result covers one edge stream: contact-gated relative-advantage edges.

```text
O7 validates terrain, high-ground, cover, or LOS affordance.
```

That remains bounded by the corrected O6Z negative result and the Caldera
construct-validation boundary.

### Remaining Boundary

```text
Do not describe the rank result as a separate rank-head metric.  Describe it as
a hazard-implied rank bridge unless a future O7 model with an explicit rank head
is trained.
```

## Experiment 10. Revised-RQ3 Prior Baseline Comparison

### Question

```text
Does the realtime phase-survival representation add evidence beyond Sagong-style
strategy scores and basic official-telemetry tabular baselines?
```

### Artifact

```text
docs/experiment_design/BATTLE_ROYALE_RQ4_PRIOR_BASELINE_RESULT_2026-07-05.md
```

### Runnable Artifacts

```text
scripts/run_br_rq4_prior_baseline_required_cohorts.sh
tools/collect_rq4_baseline_metrics.py
```

### Status

```text
supported
```

The 13-cohort run completed.  The full feature-only representation improves
hazard AUC over Sagong-style raw features, Sagong-style scores, current
conventional features, and official raw telemetry on the same team-phase risk
set.

Key completed deltas:

| Comparison | Hazard AUC wins | Mean AUC delta | Median AUC delta |
| --- | ---: | ---: | ---: |
| `a7_raw_plus_habitat - a0_official_raw` | 13/13 | +0.021690 | +0.019163 |
| `a11_full_representation - a0_official_raw` | 13/13 | +0.021434 | +0.021297 |
| `a11_full_representation - a1_prior_raw` | 13/13 | +0.059714 | +0.061459 |
| `a11_full_representation - a3_current_conventional` | 13/13 | +0.028539 | +0.027681 |

### Graph Exclusion Decision

Graph model design was tested as a possible core comparison and removed
from the core claim.  In the bounded Erangel TPP squad early readout, observed
graph did not improve hazard AUC over no-graph and shuffle controls were
comparable:

```text
no_graph AUC        0.776948
observed graph AUC  0.774473
node_shuffle AUC    0.774039
edge_shuffle AUC    0.774102
```

Therefore the manuscript should not frame RQ3 or RQ4 as a graph-model
contribution.
Graph runs may be cited only as an internal/supplementary design audit showing
why the thesis relies on phase-survival representation and pressure-habitat
features rather than graph architecture.

### Boundary

The Sagong-style baseline is a team-phase adaptation on this project's data,
not a reproduction of Sagong et al.'s confidential solo-player dataset.

Do not claim that graph architecture is necessary for the realtime survival
or risk/advantage-board result.

## Experiment 11. Revised-RQ3 Hierarchical Survival Model

### Question

```text
Can the phase-survival claim be modeled directly as within-phase state
accumulation, phase-to-phase hazard recurrence, and cumulative survival to
final top-k/rank outcomes?
```

### Artifact

```text
docs/experiment_design/BATTLE_ROYALE_RQ4_HIERARCHICAL_SURVIVAL_PROTOCOL_2026-07-05.md
```

### Runnable Artifacts

```text
model_stgcn/snapshot_tabular_baseline.py
model_stgcn/snapshot_baselines.py::SnapshotGRUNoGraphHazard
model_stgcn/snapshot_baselines.py::SnapshotHierGRUNoGraphHazard
tools/eval_snapshot_survival_bridge.py
scripts/run_br_rq4_hier_survival_required_cohorts.sh
tools/collect_snapshot_survival_bridge_metrics.py
docs/experiment_design/BATTLE_ROYALE_RQ4_SAME_CONDITION_SEQUENCE_RESULT_2026-07-05.md
```

### Status

```text
protocol implemented; smoke passed; Erangel TPP squad anchor complete; full
required-cohort result complete with bounded model-form claim
```

The implemented model consumes official PUBG 10s snapshot node features and no
graph relation channel.  The default recurrent arm now uses a causal
snapshot-level GRU over the full 10s sequence, then reads hidden states at phase
cuts before predicting next-phase elimination hazard.  The slower
phase-explicit hierarchical GRU remains implemented but is not the default
required-cohort arm.

The same-condition tabular baselines consume the same official 10s snapshot
feature surface:

| Baseline | Input |
| --- | --- |
| `snapshot_last_hgb` | last official snapshot at the phase cut |
| `snapshot_summary_hgb` | summary statistics over official snapshots up to the cut |

These are not the same as the stronger phase-aggregate representation
baselines.  They exist specifically to test whether DL adds value over tabular
reductions under the same snapshot feature surface.

The evaluator converts hazard outputs to cumulative survival scores:

```text
S_i(k) = product_{j <= k} (1 - hazard_i(j))
rank_pred_norm_i(k) = 1 - S_i(k)
```

Those derived scores are evaluated against final top5/top10 and rank labels as
hazard-implied bridge metrics.  They are not independent rank or top-k heads.
The evaluator also reports a `latest_cut_bridge` block that uses each team's
latest observed cut as the final-outcome bridge unit, avoiding a pooled-phase
comparison that can be confounded by phase depth.

Smoke checks passed:

| Check | Result |
| --- | --- |
| compile | passed |
| manual pytest-style unit functions | passed; `pytest` package unavailable |
| causal no-future-snapshot-leak regression | passed |
| 30-match / 1-epoch Erangel TPP squad training smoke | passed |
| survival bridge JSON/CSV export | passed |
| 30-match / 1-epoch three-variant runner smoke | passed |
| 30-match / 12-epoch same-condition tabular + DL runner smoke | passed; tabular `snapshot_summary_hgb` was stronger in this smoke |
| Erangel TPP squad full anchor | passed for `snapshot_last_hgb`, `snapshot_summary_hgb`, `snapshot_last_only`, `snapshot_tcn_no_graph`, `snapshot_gru_no_graph` |
| 13-cohort required same-condition run | passed; `snapshot_summary_hgb` remains strongest aggregate arm |

Smoke artifacts:

```text
/private/tmp/pubg_hier_gru_smoke.pt
/private/tmp/pubg_hier_gru_smoke_survival_bridge.json
/private/tmp/pubg_hier_gru_smoke_survival_bridge.pred.csv
/private/tmp/pubg_hier_runner_smoke/snapshot_survival_bridge_metrics_summary.csv
/private/tmp/pubg_hier_same_condition_smoke12/snapshot_survival_bridge_metrics_summary.csv
```

### Boundary

The smoke run is not performance evidence.  It only shows that the modeling and
evaluation pipeline is executable.

The same-condition smoke does not support a DL advantage yet.  It shows why the
required-cohort comparison must include `snapshot_summary_hgb` before making a
deep-learning claim.

Erangel TPP squad full anchor result:

| Variant | Hazard AUC | Hazard AP | Boundary |
| --- | ---: | ---: | --- |
| `snapshot_summary_hgb` | `0.7723` | `0.6627` | strongest same-condition arm |
| `snapshot_gru_no_graph` | `0.7658` | `0.6541` | beats last-only, TCN, and last-snapshot HGB, but not summary HGB |
| `snapshot_last_hgb` | `0.7491` | `0.6346` | last snapshot tabular baseline |
| `snapshot_tcn_no_graph` | `0.7436` | `0.6269` | causal temporal conv baseline |
| `snapshot_last_only` | `0.7281` | `0.6113` | neural lower bound |

This anchor supports a limited sequence-accumulation claim but does not support
a general deep-learning advantage over strong tabular summary reductions of
the same official snapshot feature surface.  The hazard-only cumulative
survival bridge also does not yet support a final-rank headline claim because
the recurrent models have negative rank Spearman on this anchor.

Full required-cohort result:

| Variant | Cohorts | Median Hazard AUC | Mean Hazard AUC | Boundary |
| --- | ---: | ---: | ---: | --- |
| `snapshot_summary_hgb` | 13 | `0.7534` | `0.7487` | strongest overall same-condition arm |
| `snapshot_last_hgb` | 13 | `0.7411` | `0.7407` | stronger than current sequence models in aggregate |
| `snapshot_gru_no_graph` | 13 | `0.7355` | `0.7345` | best sequence arm by median AUC |
| `snapshot_tcn_no_graph` | 13 | `0.7325` | `0.7316` | best overall on `paramo_tpp_squad` only |
| `snapshot_last_only` | 13 | `0.7213` | `0.7233` | neural lower bound |

Derived comparison:

| Comparison | Result |
| --- | --- |
| best sequence beats `snapshot_last_only` | 12/13 cohorts |
| best sequence beats `snapshot_last_hgb` | 4/13 cohorts |
| best sequence beats `snapshot_summary_hgb` | 2/13 cohorts |
| overall winner count | `snapshot_summary_hgb`: 11/13; `snapshot_tcn_no_graph`: 1/13; `snapshot_last_only`: 1/13 |
| median AUC delta, best sequence minus last-only | `+0.0123` |
| median AUC delta, best sequence minus summary-HGB | `-0.0156` |

Supported:

```text
Official PUBG snapshot telemetry can support next-phase hazard prediction, and
temporal accumulation generally improves over a neural phase-cut-only lower
bound.
```

Do not claim:

```text
The recurrent snapshot model improves over same-condition tabular summary
baselines across required cohorts.
```

The hazard-only cumulative survival bridge also remains bounded: all neural
arms have negative median rank Spearman, so this experiment should not be used
as a final-rank headline without calibration or an explicit final-outcome head.

## Overall Manuscript Interpretation

### Research Question Chain

| RQ | Core question | Current evidence |
| --- | --- | --- |
| `RQ1` | How can the Battle Royale genre be established as an analyzable framework? | O0-O8 lock, mini corpus, survival formalization |
| `RQ2` | How can that framework be represented in PUBG? | official telemetry observability table, feature audit, required cohorts, OM ladder |
| `RQ3` | How can prior tabular and phase-aware PUBG prediction be extended into a 10-second realtime elimination-risk model? | prior/basic baseline comparison, same-condition sequence result, dynamic probability trajectory, 10-second prefix demo |
| `RQ4` | Can the realtime survival readout identify the most unfavorable and most favorable teams at phase transitions? | phase-transition risk/advantage board, risk top-k hit, advantage top-k/final-rank readout, completed board-baseline comparison with HGB-scoring boundary |

### Evidence Chain

1. Mini corpus and ontology lock address RQ1.
2. Official telemetry observability table and feature audit show the vocabulary
   can be operationalized in PUBG without forbidden inputs.
3. The OM ladder and cohort contract address RQ2.
4. Prior/basic baseline comparison positions the work against existing
   tabular and phase-aware PUBG prediction.
5. Same-condition sequence and dynamic-probability results address RQ3 by
   turning phase hazards into realtime survival readouts.
6. The phase-transition risk/advantage board addresses RQ4; the board-level
   baseline comparison is complete and supports the board claim against
   simple/prior baselines while preserving the pressure-habitat HGB scoring
   boundary.
7. O6Z and Caldera / Warzone are supplementary boundary evidence: PUBG is
   geometry-poor for fine terrain affordance, while Caldera / Warzone is
   geometry-rich but not supervised-label ready.
8. The same-condition hierarchical survival run is complete and bounds the
   model-form claim: temporal sequence models generally beat the neural
   phase-cut-only lower bound, but not strong tabular snapshot summaries.
9. O1 and O7 are supplementary checks, not the main chain.

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
  prediction; the completed artifact shows geometry-derived terrain, route,
  building, and tactical-state measurements.
- As supplementary evidence, observed contact-gated relative-advantage edges
  carry topology-specific signal over an edge-shuffled null in the gen-2 PUBG
  O7 run.
- The same-condition hierarchical survival pipeline is implemented,
  required-cohort tested, and supports a bounded model-form claim: sequence
  accumulation helps over last-only, but current compact DL sequence models do
  not generally beat strong tabular snapshot-summary baselines.

### Claims Not Allowed Now

- A universal formal Battle Royale ontology has been proven.
- Causal terrain or habitat effects have been identified.
- O7 graph topology has been fully validated across all relation streams and
  primary metrics.
- O7 is one of the paper's core research-question answers.
- The O7 advantage-edge model has an independent rank head.
- The hierarchical GRU or compact snapshot sequence models are generally
  better than same-surface tabular summary baselines.
- PUBG validates true terrain, cover, LOS, or building affordance.
- Caldera supports phase-wise survival or final-rank prediction.

## Immediate Next Step

Move to manuscript figure and prose drafting around the RQ1-RQ4 chain.  The
revised-RQ3 same-condition sequence result should be presented after the prior/basic
baseline comparison and the OM2->OM3 pressure-habitat result, as a bounded
model-form test rather than as the main positive supervised evidence.
The O7 wording is usable if it preserves the metric boundary:

```text
positive for contact-gated relative-advantage edges, hazard AUC, and
hazard-implied final-rank ordering; not all O7 relation structure, and not an
independent rank-head result.
```
