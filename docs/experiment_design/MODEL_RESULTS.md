# PUBG Phase-Level Dynamic GNN Results

## Scope

This document summarizes the first full model evaluation for the PUBG STDB
project.

- Dataset: 5,950 phase-ready matches
- Split: 4,760 train / 595 validation / 595 test
- Test hazard rows after common alignment: 30,166 `(match, team, phase)` cells
- Primary task: predict next-phase team elimination hazard
- Node: team
- Time step: phase
- Edge channels: proximity / combat / copressure

All model comparisons below use the same match-level split and the same aligned
test target cells.

## Overall Hazard Metrics

| Model | Hazard AUC | AP | Accuracy | BCE |
|---|---:|---:|---:|---:|
| Tabular baseline, HistGradientBoosting | 0.7680 | 0.6458 | 0.7513 | 0.5141 |
| DynamicSTGNN, no graph | 0.7593 | 0.6293 | 0.7460 | 0.5227 |
| DynamicSTGNN, proximity only | 0.7665 | 0.6393 | 0.7496 | 0.5167 |
| DynamicSTGNN, proximity + combat | 0.7665 | 0.6378 | 0.7481 | 0.5171 |
| DynamicSTGNN, full 3-edge | 0.7693 | 0.6423 | 0.7515 | 0.5147 |
| DynamicSTGNN, tuned full 3-edge | 0.7708 | 0.6469 | 0.7166 | 0.5544 |

Note: the table above uses the common aligned 30,166 test cells from
`phase_result_analysis/hazard_overall_metrics.csv`. The standalone baseline
artifact `baseline_phase_5951_shared.json` reports slightly different test
hazard metrics on its raw 30,243 rows: AUC 0.7678, AP 0.6471, and accuracy
0.7504. The AP reversal between the full graph model and the tabular baseline
holds under both views.

Interpretation:

- The full 3-edge graph improves over the no-graph neural baseline across AUC,
  AP, accuracy, and BCE.
- The tuned full model gives the best ranking-oriented hazard metrics
  (AUC/AP), but its class-balanced loss lowers threshold accuracy and worsens
  BCE calibration.
- The untuned full 3-edge model is the best balanced default checkpoint.

## Feature Ablation vs Prior-Paper Style Baselines

This section asks whether our STDB feature design adds information beyond the
prior study's FightScore/ActiveScore style representation. All rows use the
same 30,166 aligned test cells.

| Feature set | Description | AUC | AP | Accuracy | BCE |
|---|---|---:|---:|---:|---:|
| Prior scores | FightScore + ActiveScore + 4 strategy type + train-only type survival rate | 0.6680 | 0.4657 | 0.6902 | 0.5862 |
| Prior raw | Aggressiveness/Activeness raw proxy columns | 0.7287 | 0.5649 | 0.7219 | 0.5500 |
| Global + state | Phase context + team state | 0.7321 | 0.5893 | 0.7302 | 0.5441 |
| Conventional telemetry | Global/state + combat + loot | 0.7581 | 0.6338 | 0.7458 | 0.5221 |
| STDB spatial only | Trajectory + exposure only | 0.7101 | 0.5672 | 0.7216 | 0.5555 |
| Conventional + trajectory | Conventional + trajectory morphology | 0.7615 | 0.6381 | 0.7475 | 0.5192 |
| Conventional + exposure | Conventional + zone/enemy exposure | 0.7667 | 0.6454 | 0.7516 | 0.5150 |
| Conventional + STDB spatial | Conventional + trajectory + exposure | 0.7674 | 0.6459 | 0.7510 | 0.5144 |
| Full node features | All node feature columns | 0.7682 | 0.6465 | 0.7517 | 0.5140 |

Paired tests use one match as one paired unit:

| Comparison | Metric | Mean improvement | Median improvement | p-value | Rank-biserial |
|---|---:|---:|---:|---:|---:|
| Full features vs prior scores | AUC up | 0.0989 | 0.0929 | 2.31e-87 | 0.937 |
| Full features vs prior raw | AUC up | 0.0369 | 0.0326 | 1.44e-50 | 0.712 |
| Conventional + STDB spatial vs conventional | BCE down | 0.0081 | 0.0068 | 7.03e-22 | 0.451 |
| Conventional + STDB spatial vs conventional | AUC up | 0.0085 | 0.0070 | 4.81e-13 | 0.344 |
| Conventional + STDB spatial vs conventional | AP up | 0.0091 | 0.0076 | 4.05e-08 | 0.254 |
| STDB spatial only vs prior scores | AUC up | 0.0388 | 0.0362 | 2.58e-26 | 0.500 |

Conclusion:

The prior-paper style representation is a useful baseline, but it is not enough
for our team-phase hazard task. The STDB spatial layer adds statistically
significant predictive signal on top of conventional combat/loot/state
telemetry, and even the STDB spatial-only feature set beats the prior score
baseline. This is the strongest current evidence for the project's novelty:
the value is not just a different model, but a richer spatio-temporal
representation of PUBG team behavior.

## Novelty Diagnostics

The next diagnostic step decomposes the feature gain by phase and asks which
feature families carry the signal in the full node-feature model.

### Phase-Wise Uplift

`Conventional + STDB spatial` beats `Conventional telemetry` in every phase.

| Phase | n | AUC uplift | AP uplift | BCE improvement |
|---:|---:|---:|---:|---:|
| 1 | 9,472 | 0.0123 | 0.0183 | 0.0114 |
| 2 | 7,345 | 0.0144 | 0.0133 | 0.0070 |
| 3 | 5,790 | 0.0047 | 0.0056 | 0.0030 |
| 4 | 4,011 | 0.0066 | 0.0078 | 0.0060 |
| 5 | 2,110 | 0.0158 | 0.0170 | 0.0104 |
| 6 | 1,080 | 0.0080 | 0.0046 | 0.0033 |
| 7 | 358 | 0.0159 | 0.0161 | 0.0120 |

Interpretation:

- STDB spatial features help across the full match, not only in one isolated
  phase.
- The largest AUC uplifts appear in phases 1, 2, 5, and 7. This supports the
  phase-aware framing: early landing/looting, first rotations, and late-zone
  pressure have different spatial meanings.
- Phase 7 remains small-sample evidence and should be described carefully.

Dynamic heterogeneous graph vs no-graph neural model also improves phases 1-6
but is weaker in phase 7:

| Phase | AUC uplift | AP uplift | BCE improvement |
|---:|---:|---:|---:|
| 1 | 0.0136 | 0.0198 | 0.0107 |
| 2 | 0.0135 | 0.0196 | 0.0078 |
| 3 | 0.0069 | 0.0129 | 0.0053 |
| 4 | 0.0083 | 0.0121 | 0.0077 |
| 5 | 0.0107 | 0.0027 | 0.0065 |
| 6 | 0.0128 | 0.0159 | 0.0077 |
| 7 | -0.0081 | -0.0031 | -0.0035 |

### Feature Group Importance

Permutation importance on the full node-feature HGB model gives this ordering:

| Feature group | AUC drop | AP drop | BCE increase |
|---|---:|---:|---:|
| Loot/readiness | 0.0572 | 0.0784 | 0.0420 |
| Combat | 0.0364 | 0.0688 | 0.0357 |
| Team state | 0.0358 | 0.0706 | 0.0320 |
| STDB exposure | 0.0309 | 0.0757 | 0.0355 |
| Global phase context | 0.0234 | 0.0357 | 0.0189 |
| Prior scores | 0.0081 | 0.0131 | 0.0078 |
| Trajectory morphology | 0.0070 | 0.0150 | 0.0072 |

Important correction for the paper:

- The strongest STDB signal is currently **exposure**: distance to safe-zone
  boundary, enemy proximity, enemy path exposure, and outside-safe behavior.
- Trajectory morphology is useful, but weaker on its own. Do not overclaim that
  trajectory shape alone drives the result.
- A fair wording is: "phase-wise STDB modeling captures both trajectory
  morphology and spatial exposure; exposure is the dominant predictive
  spatio-temporal signal in the current dataset."

Selected feature-level importance supports the same reading:

| Feature | AUC drop |
|---|---:|
| `pickup_count` | 0.0160 |
| `vehicle_ratio` | 0.0108 |
| `min_nearest_enemy_m` | 0.0098 |
| `damage_taken` | 0.0077 |
| `outside_safe_ratio` | 0.0029 |
| `combat_partner_count` | 0.0019 |
| `path_near_enemy_ratio` | 0.0013 |

Artifacts:

- `model_stgcn/checkpoints/novelty_analysis/phase_novelty_deltas.csv`
- `model_stgcn/checkpoints/novelty_analysis/group_permutation_importance.csv`
- `model_stgcn/checkpoints/novelty_analysis/selected_feature_permutation_importance.csv`
- `model_stgcn/checkpoints/novelty_analysis/phase_auc_uplift.png`
- `model_stgcn/checkpoints/novelty_analysis/group_importance_auc.png`
- `model_stgcn/checkpoints/novelty_analysis/selected_feature_importance_auc.png`

## Edge Ablation

Paired Wilcoxon signed-rank tests use one match as one paired unit. Positive
improvement means the left model is better.

| Comparison | Metric | Mean improvement | Median improvement | p-value | Rank-biserial |
|---|---:|---:|---:|---:|---:|
| Full 3-edge vs no graph | BCE ↓ | 0.0080 | 0.0071 | 1.84e-17 | 0.399 |
| Full 3-edge vs no graph | AUC ↑ | 0.0075 | 0.0079 | 5.86e-10 | 0.291 |
| Full 3-edge vs no graph | AP ↑ | 0.0087 | 0.0100 | 2.16e-07 | 0.240 |
| Full 3-edge vs proximity only | AUC ↑ | 0.0026 | 0.0029 | 0.0044 | 0.126 |
| Full 3-edge vs proximity + combat | AUC ↑ | 0.0016 | 0.0024 | 0.0114 | 0.110 |

Conclusion:

The graph component is not cosmetic. The full heterogeneous graph is
statistically better than the no-graph sequence model, and adding all three
edge channels is better than proximity-only or proximity+combat variants.

## Baseline Comparison

| Comparison | Metric | Mean improvement | Median improvement | p-value | Rank-biserial |
|---|---:|---:|---:|---:|---:|
| Full 3-edge vs tabular baseline | AUC ↑ | 0.0028 | 0.0044 | 0.0205 | 0.097 |
| Full 3-edge vs tabular baseline | AP ↑ | 0.0013 | 0.0029 | 0.3261 | 0.021 |
| Full 3-edge vs tabular baseline | BCE ↓ | 0.0000 | 0.0007 | 0.4798 | 0.002 |
| Tuned full vs tabular baseline | AUC ↑ | 0.0044 | 0.0038 | 0.0046 | 0.124 |
| Tuned full vs tabular baseline | AP ↑ | 0.0051 | 0.0042 | 0.0318 | 0.088 |

Conclusion:

The tuned full model provides a statistically significant AUC/AP advantage over
the tabular baseline. The untuned full model is better calibrated and more
balanced, but its AP/BCE improvement over the baseline is not significant.

## Phase-Wise AUC

| Phase | n | pos rate | Baseline | No graph | Full 3-edge | Tuned full |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9,472 | 0.255 | 0.7810 | 0.7630 | 0.7766 | 0.7807 |
| 2 | 7,345 | 0.242 | 0.7238 | 0.7197 | 0.7332 | 0.7344 |
| 3 | 5,790 | 0.333 | 0.7202 | 0.7224 | 0.7293 | 0.7291 |
| 4 | 4,011 | 0.473 | 0.7690 | 0.7594 | 0.7677 | 0.7699 |
| 5 | 2,110 | 0.437 | 0.7386 | 0.7169 | 0.7276 | 0.7256 |
| 6 | 1,080 | 0.445 | 0.6983 | 0.6952 | 0.7080 | 0.7008 |
| 7 | 358 | 0.444 | 0.6087 | 0.6545 | 0.6464 | 0.6470 |

Interpretation:

- Graph models help most clearly in phases 2, 3, 6, and 7.
- The tabular baseline remains strong in phase 1 and phase 5.
- Late phases have much smaller sample sizes, so phase 7 should be treated as
  directional rather than conclusive.

## Recommended Checkpoints

| Use | Checkpoint |
|---|---|
| Balanced final model | `model_stgcn/checkpoints/phase_full_5951_e12.pt` |
| Best AUC/AP model | `model_stgcn/checkpoints/phase_full_tuned_h96_e30.pt` |
| No-graph ablation | `model_stgcn/checkpoints/phase_ablation_none_e12.pt` |
| Proximity-only ablation | `model_stgcn/checkpoints/phase_ablation_proximity_masked_e12.pt` |
| Proximity+combat ablation | `model_stgcn/checkpoints/phase_ablation_proximity_combat_masked_e12.pt` |

## Output Artifacts

- `model_stgcn/checkpoints/phase_result_analysis/phase_result_analysis_summary.json`
- `model_stgcn/checkpoints/phase_result_analysis/hazard_overall_metrics.csv`
- `model_stgcn/checkpoints/phase_result_analysis/hazard_phase_metrics.csv`
- `model_stgcn/checkpoints/phase_result_analysis/hazard_per_match_metrics.csv`
- `model_stgcn/checkpoints/phase_result_analysis/hazard_paired_tests.csv`
- `model_stgcn/checkpoints/phase_result_analysis/hazard_predictions_test_aligned.csv`
- `model_stgcn/checkpoints/feature_ablation/feature_ablation_summary.json`
- `model_stgcn/checkpoints/feature_ablation/feature_ablation_metrics.csv`
- `model_stgcn/checkpoints/feature_ablation/feature_ablation_phase_metrics.csv`
- `model_stgcn/checkpoints/feature_ablation/feature_ablation_paired_tests.csv`
- `model_stgcn/checkpoints/novelty_analysis/novelty_analysis_summary.json`
- `model_stgcn/checkpoints/novelty_analysis/phase_novelty_deltas.csv`
- `model_stgcn/checkpoints/novelty_analysis/group_permutation_importance.csv`
- `model_stgcn/checkpoints/novelty_analysis/selected_feature_permutation_importance.csv`
