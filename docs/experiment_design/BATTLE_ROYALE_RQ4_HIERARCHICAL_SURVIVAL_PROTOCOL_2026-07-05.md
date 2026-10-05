# Battle Royale RQ4 Hierarchical Survival Protocol

Date: 2026-07-05
Class: `ExperimentContract`
Lifecycle: `active`

## Purpose

This document registers the RQ4 model experiment that matches the thesis
claim:

```text
Battle Royale survival is a phase-wise accumulation process.  Actions and
states within a phase change the next-phase survival hazard, and phase hazards
accumulate into final survival, rank, and victory-related outcomes.
```

This experiment is not a graph-model claim.  Graph design has been removed from
the manuscript core.  The model here is a no-graph recurrent survival model
that consumes official PUBG snapshot telemetry and evaluates whether phase-wise
hazard forecasts can be bridged to final top-k/rank labels.

## Research Question Link

This sits in RQ4:

```text
Given the Battle Royale formalization and the PUBG telemetry mapping, can
10-second PUBG telemetry be converted into causal phase-wise survival readouts?
```

It follows the completed revised-RQ3 prior-baseline layer:

```text
Sagong-style / basic official tabular baselines
-> Battle Royale representation feature ablations
-> hierarchical phase-survival model
-> cumulative survival bridge to final top-k/rank
```

## Model Contract

Executable implementation:

```text
model_stgcn/snapshot_tabular_baseline.py
model_stgcn/snapshot_baselines.py::SnapshotGRUNoGraphHazard
model_stgcn/snapshot_baselines.py::SnapshotHierGRUNoGraphHazard
model_stgcn/train_snapshot_phase.py
tools/eval_snapshot_survival_bridge.py
```

Input:

```text
X_s: 10s team snapshot node features
phase_of_snap: phase index for each snapshot
node_mask_s: valid team/snapshot mask
```

The current snapshot node features are:

```text
pos_x, pos_y, pos_z,
rel_zone_x, rel_zone_y,
speed_mps,
dist_blue_boundary_m, inside_blue, dist_target_center_m,
mean_health, min_health, alive_n,
vehicle_ratio
```

The model deliberately consumes no relation or graph channel:

```text
input_channels_consumed = [C1_snapshot_team_state]
relation_handling = none
graph_resolution = none
```

Current default architecture:

```text
10s snapshot sequence
  -> causal unidirectional GRU accumulates team state through time
  -> phase-cut hidden states are gathered at each cut k
  -> hazard head predicts h_i(k) = P(eliminated before next phase | observed up to k)
```

`snapshot_hier_gru_no_graph` remains available as a slower phase-explicit
prototype.  The default recurrent arm is `snapshot_gru_no_graph` because it is
the executable version of the same accumulation claim for required-cohort runs.

Evaluation bridge:

```text
S_i(k) = product_{j <= k} (1 - h_i(j))
rank_pred_norm_i(k) = 1 - S_i(k)
```

`S_i(k)` is used as a cumulative survival score for final top5/top10 AUC/AP.
`rank_pred_norm_i(k)` is compared with `final_rank_norm` using Spearman and
pairwise rank accuracy.  These are hazard-implied final-outcome metrics, not
separately trained rank/top-k heads.

The evaluator reports both:

| Unit | Meaning |
| --- | --- |
| `at_risk_team_phase` | every target-defined team x cut-phase row, mainly for phase-wise diagnostics |
| `latest_at_risk_team` | each team's latest observed cut, used as the cleaner final-outcome bridge because cumulative survival scores naturally change with phase depth |

## Required Comparison

Run all models on the same cohort split:

| Variant | Role |
| --- | --- |
| `snapshot_last_hgb` | same-condition tabular baseline using the last phase-cut snapshot |
| `snapshot_summary_hgb` | same-condition tabular baseline using summary statistics of official 10s snapshot features up to the cut |
| `snapshot_last_only` | phase-cut state baseline; no within-phase accumulation |
| `snapshot_tcn_no_graph` | causal temporal baseline over snapshots; no explicit phase hierarchy |
| `snapshot_gru_no_graph` | causal recurrent survival model over snapshots |

The same-condition tabular baselines are distinct from the stronger completed
revised-RQ3 prior-baseline / representation ablation.  They consume only the official
10s snapshot feature tensor surface used by the no-graph DL models, not the
phase-aggregate STDB/habitat feature view.

Primary metric:

```text
next-phase hazard AUC/AP/BCE on at-risk team-phase rows
```

Bridge metrics:

```text
cumulative-survival top5/top10 AUC/AP
rank_from_survival Spearman and pairwise rank accuracy
latest_cut_bridge top5/top10 AUC/AP and rank Spearman
```

Required-cohort runner:

```bash
PG_PASSWORD=... scripts/run_br_rq4_hier_survival_required_cohorts.sh
```

Aggregate outputs:

```text
model_stgcn/checkpoints/br_rq4_hier_survival_required_v1/snapshot_survival_bridge_metrics_all_cohorts.csv
model_stgcn/checkpoints/br_rq4_hier_survival_required_v1/snapshot_survival_bridge_metrics_summary.csv
model_stgcn/checkpoints/br_rq4_hier_survival_required_v1/snapshot_survival_bridge_summary.json
```

Required-cohort result outputs:

```text
model_stgcn/checkpoints/br_rq4_same_condition_snapshot_required_v2/snapshot_survival_bridge_metrics_all_cohorts.csv
model_stgcn/checkpoints/br_rq4_same_condition_snapshot_required_v2/snapshot_survival_bridge_metrics_summary.csv
```

## Execution Status

Implemented and smoke-tested:

| Check | Status |
| --- | --- |
| Python compile | passed |
| Manual pytest-style unit functions | passed; `pytest` package is not installed in the current Python |
| Hierarchical GRU forward shape | passed |
| No-future-snapshot-leak regression | passed |
| 30-match Erangel TPP squad 1-epoch training smoke | passed |
| Survival bridge JSON/CSV export smoke | passed |
| 30-match Erangel TPP squad runner smoke for all three variants | passed |
| 30-match Erangel TPP squad same-condition tabular + DL runner smoke | passed |
| Erangel TPP squad full anchor run | passed for `snapshot_last_hgb`, `snapshot_summary_hgb`, `snapshot_last_only`, `snapshot_tcn_no_graph`, `snapshot_gru_no_graph` |
| 13-cohort required same-condition run | passed; bounded model-form result |

Smoke checkpoint:

```text
/private/tmp/pubg_hier_gru_smoke.pt
```

Smoke bridge outputs:

```text
/private/tmp/pubg_hier_gru_smoke_survival_bridge.json
/private/tmp/pubg_hier_gru_smoke_survival_bridge.pred.csv
/private/tmp/pubg_hier_runner_smoke/snapshot_survival_bridge_metrics_summary.csv
/private/tmp/pubg_hier_same_condition_smoke12/snapshot_survival_bridge_metrics_summary.csv
```

Smoke result boundary:

```text
The 30-match / 1-epoch smoke confirms executable plumbing only.  It is not
performance evidence and must not be cited as a model result.
```

The 30-match / 12-epoch same-condition smoke is also not performance evidence,
but it was a useful pre-run warning: `snapshot_summary_hgb` outperformed the
current DL variants on this tiny smoke.

Anchor result:

```text
docs/experiment_design/BATTLE_ROYALE_RQ4_SAME_CONDITION_SEQUENCE_RESULT_2026-07-05.md
```

The Erangel TPP squad anchor shows that `snapshot_gru_no_graph` improves over
`snapshot_last_only`, `snapshot_tcn_no_graph`, and `snapshot_last_hgb`, but does
not beat `snapshot_summary_hgb`.  Therefore it is not yet valid to claim that
deep sequence modeling beats strong tabular reductions of the same official
snapshot feature surface.

Full required-cohort result:

```text
docs/experiment_design/BATTLE_ROYALE_RQ4_SAME_CONDITION_SEQUENCE_RESULT_2026-07-05.md
```

Aggregate hazard AUC:

| Variant | Median Hazard AUC | Mean Hazard AUC | Boundary |
| --- | ---: | ---: | --- |
| `snapshot_summary_hgb` | 0.7534 | 0.7487 | strongest overall same-condition arm |
| `snapshot_last_hgb` | 0.7411 | 0.7407 | stronger than current sequence models in aggregate |
| `snapshot_gru_no_graph` | 0.7355 | 0.7345 | best sequence arm by median AUC |
| `snapshot_tcn_no_graph` | 0.7325 | 0.7316 | best overall on `paramo_tpp_squad` only |
| `snapshot_last_only` | 0.7213 | 0.7233 | neural lower bound |

## Claim After Full Run

The required-cohort run supports this bounded claim:

```text
The phase-wise survival formulation is executable under official PUBG snapshot
telemetry, and temporal sequence models generally improve over a neural
phase-cut-only lower bound.
```

It does not support this stronger claim:

```text
Compact TCN/GRU sequence models generally outperform strong same-snapshot
tabular summary baselines.
```

The safe interpretation is:

```text
The survival problem is correctly formulated as phase-wise hazard accumulation,
but the current compact recurrent/convolutional architectures are not superior
to strong tabular snapshot-summary baselines under official PUBG snapshot
features.
```

In either case, the already completed OM and revised-RQ3 prior-baseline results
remain the central feature-representation evidence.

## Boundary

Do not claim from this experiment:

```text
Graph modeling is necessary.
```

Do not claim:

```text
PUBG official telemetry fully captures terrain/topology/high-ground affordance.
```

Do not claim from the smoke:

```text
The hierarchical GRU is better than the baselines.
```

Do not claim from the required-cohort run:

```text
Deep sequence modeling generally beats same-snapshot tabular summaries.
```

Do claim:

```text
Sequence accumulation improves over the neural phase-cut-only lower bound in
12/13 cohorts, but strong tabular snapshot summaries remain the aggregate
same-condition winner.
```
