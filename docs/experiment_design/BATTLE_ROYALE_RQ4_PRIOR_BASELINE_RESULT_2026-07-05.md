# Battle Royale RQ4 Prior Baseline Result

Date: 2026-07-05
Class: `ExperimentResult`
Lifecycle: `active`

## Purpose

This document records the prior-work / basic-tabular comparison layer.  Under
the revised manuscript spine, this is part of RQ3: extending prior tabular and
phase-aware PUBG prediction into a realtime elimination-risk model.

The paper should not jump directly from Battle Royale formalization to realtime
risk/advantage boards.  It must first locate the proposed representation
against:

1. prior PUBG phase-aware survival work, especially the Sagong-style
   FightScore / ActiveScore line;
2. basic official-telemetry tabular prediction baselines;
3. feature-only Battle Royale representation ablations under the same
   team-phase risk set.

Graph model design is no longer a core thesis comparison.  It can remain as an
internal or supplementary negative design audit, but the manuscript's central
prediction claim is feature-representation and realtime phase-survival based.

## Manuscript Role

This is a core revised-RQ3 comparison layer, not a new independent paper spine.

It sits before the current OM3 headline:

```text
Sagong-style / official raw tabular baselines
-> feature-only Battle Royale representation ablation
-> OM2 -> OM3 pressure-habitat headline
-> dynamic probability trajectory
-> phase-transition risk/advantage board
```

The current OM ladder remains the mechanism result.  This comparison positions
that result against prior-work style scores and basic official-telemetry
tabular baselines.

## Prior-Work Boundary

Sagong et al. is the closest direct prior because it uses phase-aware PUBG
survival prediction with FightScore, ActiveScore, and strategy types.

However:

- the original data are confidential;
- the original setting is solo-player;
- the current thesis setting is team-level, cohort-separated, and
  phase-hazard based.

Therefore the correct experiment name is:

```text
Sagong-style team-phase adapted baseline
```

Do not call it:

```text
Sagong reproduction
```

Kaggle-style PUBG placement predictors and basic tabular papers are treated as
post-game or aggregate prediction prior work.  Their numbers should not be
directly compared to the current phase-causal hazard metrics unless an explicit
external benchmark is run.

## Feature-Only Prior And Tabular Baselines

### Question

```text
Under the same PUBG team-phase risk set, how far do prior-work strategy scores
and basic official-telemetry tabular features go before Battle Royale
representation blocks are added?
```

### Unit And Split

| Field | Contract |
| --- | --- |
| Unit | `(match_id, team_id, phase_idx)` |
| Label | `hazard_next` |
| Split | reuse each required cohort's OM ladder `split.json` |
| Rows | target-defined at-risk team-phase rows |
| Model family | feature-only tabular learner |
| Primary learner | `HistGradientBoostingClassifier` |

### Feature Configs

The executable source is:

```text
model_stgcn/feature_ablation.py
model_stgcn/feature_manifests/pubg_representation_v2.json
```

Main manuscript-facing configs:

| Config | Meaning | Paper role |
| --- | --- | --- |
| `a0_official_raw` | official telemetry phase aggregate baseline | basic tabular baseline |
| `a1_prior_raw` | raw ingredients corresponding to Sagong-style inputs | prior-work raw baseline |
| `a2_prior_scores` | FightScore, ActiveScore, strategy type, type survival rate | Sagong-style adapted baseline |
| `a3_current_conventional` | current conventional telemetry aggregate | conventional feature baseline |
| `a5_raw_plus_stdb` | official raw plus STDB spatial/exposure | representation increment |
| `a7_raw_plus_habitat` | official raw plus habitat pressure | pressure/habitat increment |
| `a9_raw_stdb_habitat` | official raw plus STDB plus habitat | strong feature-only RQ4 baseline |
| `a11_full_representation` | official raw plus STDB plus network plus habitat | full feature-only representation |

### Execution

Required-cohort runner:

```bash
PG_PASSWORD=... bash scripts/run_br_rq4_prior_baseline_required_cohorts.sh
```

Aggregate outputs:

```text
model_stgcn/checkpoints/br_rq4_prior_baseline_required_v1/rq4_prior_baseline_metrics_all_cohorts.csv
model_stgcn/checkpoints/br_rq4_prior_baseline_required_v1/rq4_prior_baseline_metrics_summary.csv
model_stgcn/checkpoints/br_rq4_prior_baseline_required_v1/rq4_prior_baseline_paired_tests_all_cohorts.csv
```

## Completed Result

The required 13-cohort prior/basic tabular baseline run completed.

Key summary:

| Model | Cohorts | Median AUC | Mean AUC | Median AP | Mean AP | Median BCE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `a2_prior_scores` | 13 | 0.652485 | 0.646172 | 0.448034 | 0.452329 | 0.600620 |
| `a1_prior_raw` | 13 | 0.717188 | 0.709667 | 0.550470 | 0.550204 | 0.554581 |
| `a3_current_conventional` | 13 | 0.745778 | 0.740842 | 0.617431 | 0.624600 | 0.530996 |
| `a0_official_raw` | 13 | 0.752092 | 0.747946 | 0.630731 | 0.635428 | 0.524477 |
| `a7_raw_plus_habitat` | 13 | 0.774240 | 0.769637 | 0.656621 | 0.661747 | 0.506304 |
| `a11_full_representation` | 13 | 0.774030 | 0.769381 | 0.658584 | 0.662173 | 0.506481 |
| `conventional_stdb_network_habitat` | 13 | 0.775253 | 0.769325 | 0.657679 | 0.660987 | 0.506274 |

Pairwise cohort-level AUC deltas:

| Comparison | Wins | Mean delta | Median delta |
| --- | ---: | ---: | ---: |
| `a7_raw_plus_habitat - a0_official_raw` | 13/13 | +0.021690 | +0.019163 |
| `a11_full_representation - a0_official_raw` | 13/13 | +0.021434 | +0.021297 |
| `a11_full_representation - a1_prior_raw` | 13/13 | +0.059714 | +0.061459 |
| `a11_full_representation - a3_current_conventional` | 13/13 | +0.028539 | +0.027681 |

## Graph Exclusion Note

A bounded graph-design early readout was run on Erangel TPP squad to decide
whether graph model design should remain in the RQ4 core.  It should not.

Observed graph did not improve hazard AUC over the no-graph baseline, and
shuffle controls were comparable:

| Arm | AUC | AP | Rank Spearman |
| --- | ---: | ---: | ---: |
| `graph_core_habitat_no_graph` | 0.776948 | 0.652919 | 0.629964 |
| `graph_core_habitat_full_observed` | 0.774473 | 0.648425 | 0.633193 |
| `graph_core_habitat_full_node_shuffle` | 0.774039 | 0.646883 | 0.634291 |
| `graph_core_habitat_full_edge_shuffle` | 0.774102 | 0.646519 | 0.643789 |

Therefore graph design is removed from the manuscript core.  It can be
mentioned, if needed, only as a bounded design audit showing why the thesis
does not rely on a graph architecture claim.

## Supported Claim

```text
Under the same phase-cut risk set, Battle Royale representation blocks add
predictive value beyond Sagong-style strategy scores and basic official
telemetry tabular aggregates.
```

The manuscript-safe RQ4 interpretation is:

```text
The strongest supervised PUBG evidence is not a graph architecture effect.  It
is the phase-wise pressure-habitat representation: official telemetry becomes
more predictive when organized around contraction pressure and survival
environment rather than only prior-work style scores or basic tabular
aggregates.
```

## Boundary

Do not claim:

```text
The experiment reproduces Sagong et al.'s private solo-player study.
```

Do not claim:

```text
Graph architecture is necessary for the RQ4 result.
```

Do not compare directly against Kaggle leaderboard numbers because those use a
post-game placement dataset and a different target.
