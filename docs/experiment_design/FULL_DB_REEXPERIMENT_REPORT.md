# Full-DB Reexperiment Report

Last updated: 2026-06-11 KST

## 1. Data Population

The current raw-to-usable conversion separates the raw telemetry population from
the model-ready graph population.

| Stage | Matches |
| --- | ---: |
| Raw fetched telemetry | 27,491 |
| Base STDB converted | 21,833 |
| Base STDB with team positions | 21,830 |
| Phase features built | 21,830 |
| Target labels built | 21,824 |
| Edge cache built | 21,830 |
| Graph eligible, `N_max=128` | 21,824 |

The full-data split is match-level, not row-level:

| Split | Matches |
| --- | ---: |
| Train | 17,460 |
| Validation | 2,182 |
| Test | 2,182 |

## 2. Aligned Hazard Results

The aligned comparison uses common target-defined test cells across all models:
`134,245` team-phase cells from `2,182` test matches.

| Model | AUC | AP | Accuracy | BCE |
| --- | ---: | ---: | ---: | ---: |
| Tabular HGB baseline | 0.7651 | 0.6626 | 0.7483 | 0.5140 |
| DynamicSTGNN no graph | 0.7681 | 0.6695 | 0.7518 | 0.5090 |
| DynamicSTGNN proximity | 0.7741 | 0.6762 | 0.7545 | 0.5046 |
| DynamicSTGNN proximity + combat | 0.7755 | 0.6779 | 0.7556 | 0.5036 |
| DynamicSTGNN full graph | 0.7782 | 0.6808 | 0.7568 | 0.5016 |

Full graph improvements:

| Comparison | AUC delta | AP delta | Accuracy delta | BCE improvement |
| --- | ---: | ---: | ---: | ---: |
| Full graph vs no graph | +0.0100 | +0.0112 | +0.0051 | 0.0074 |
| Full graph vs proximity | +0.0041 | +0.0045 | +0.0023 | 0.0030 |
| Full graph vs proximity + combat | +0.0027 | +0.0029 | +0.0013 | 0.0020 |
| Full graph vs tabular baseline | +0.0130 | +0.0181 | +0.0085 | 0.0124 |

## 3. Match-Paired Tests

Wilcoxon signed-rank tests use one match as one paired unit. Positive
improvement means the full graph model is better.

| Comparison | Metric | Mean improvement | p-value |
| --- | --- | ---: | ---: |
| Full vs no graph | BCE | 0.0071 | 1.63e-47 |
| Full vs no graph | AUC | 0.0080 | 1.17e-30 |
| Full vs no graph | AP | 0.0090 | 3.12e-23 |
| Full vs proximity | BCE | 0.0030 | 2.07e-15 |
| Full vs proximity | AUC | 0.0039 | 4.61e-10 |
| Full vs proximity | AP | 0.0051 | 1.89e-10 |
| Full vs proximity + combat | BCE | 0.0019 | 1.89e-09 |
| Full vs proximity + combat | AUC | 0.0022 | 1.69e-05 |
| Full vs proximity + combat | AP | 0.0028 | 7.71e-06 |
| Full vs tabular | BCE | 0.0120 | 3.30e-74 |
| Full vs tabular | AUC | 0.0113 | 2.60e-37 |
| Full vs tabular | AP | 0.0154 | 9.28e-36 |

## 4. Calibration

ECE uses 10 equal-width probability bins.

| Model | Brier | BCE | ECE | Mean probability |
| --- | ---: | ---: | ---: | ---: |
| Tabular HGB baseline | 0.1708 | 0.5140 | 0.0075 | 0.3223 |
| DynamicSTGNN no graph | 0.1692 | 0.5090 | 0.0039 | 0.3264 |
| DynamicSTGNN proximity | 0.1674 | 0.5046 | 0.0023 | 0.3253 |
| DynamicSTGNN proximity + combat | 0.1670 | 0.5036 | 0.0028 | 0.3266 |
| DynamicSTGNN full graph | 0.1662 | 0.5016 | 0.0015 | 0.3260 |

The empirical positive rate on aligned cells is `0.3245`.

## 5. Current Interpretation

The full-data result strengthens the paper narrative. The tabular baseline is
still strong, but the full DynamicSTGNN improves over the tabular baseline,
the no-graph neural control, and the partial edge variants on AUC, AP, BCE,
Brier score, and ECE. The edge ablation follows the expected order:
`no graph < proximity < proximity + combat < full graph`. This supports the
claim that dynamic team relations provide additional predictive signal for
phase-level elimination hazard.

The claim should remain predictive rather than causal. The model shows that
observed proximity, combat, and co-pressure relations help predict risk; it does
not prove that changing a team strategy would causally change survival.

## 6. Snapshot Consistency

The main paper tables should use only the full-DB results above. Earlier
seed-repeat, phase-wise, and random-edge sanity experiments were run on older
17,977/5,950-match snapshots. They can be cited only as auxiliary evidence and
must be labeled with their data source if included.

The full-DB edge ablation and the full-DB feature (representation) ablation
have been rerun and are no longer legacy-snapshot results.

## 7. Final Full-DB Outputs

```text
model_stgcn/checkpoints/full_db_reexperiment/full_db_edge_ablation.log
model_stgcn/checkpoints/full_db_reexperiment/phase_result_analysis_edges/
model_stgcn/checkpoints/full_db_reexperiment/calibration_edges/
model_stgcn/checkpoints/full_db_reexperiment/calibration_edges_by_phase/
model_stgcn/checkpoints/full_db_reexperiment/phase_result_analysis/
model_stgcn/checkpoints/full_db_reexperiment/calibration/
model_stgcn/checkpoints/full_db_reexperiment/calibration_by_phase/
model_stgcn/checkpoints/feature_ablation_full_db_reexperiment/
```

## 8. Representation (Feature) Ablation — added 2026-06-11

Same gradient-boosting pipeline (HistGradientBoosting, max_iter=200, lr=0.06,
l2=0.01, seed 42), same full-DB match split, same `134,245` aligned test
cells as Section 2. Output:
`model_stgcn/checkpoints/feature_ablation_full_db_reexperiment/`.

| Feature set | d | AUC | AP | BCE | Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| prior_scores (FightScore/ActiveScore style) | 5 | 0.6490 | 0.4547 | 0.5993 | 0.6821 |
| prior_raw (score ingredients) | 12 | 0.7184 | 0.5703 | 0.5584 | 0.7140 |
| stdb_spatial only (trajectory+exposure) | 16 | 0.7124 | 0.5895 | 0.5529 | 0.7214 |
| global_state | 30 | 0.7169 | 0.5849 | 0.5561 | 0.7216 |
| conventional | 57 | 0.7522 | 0.6476 | 0.5242 | 0.7443 |
| conventional+trajectory | 66 | 0.7580 | 0.6550 | 0.5191 | 0.7466 |
| conventional+exposure | 63 | 0.7632 | 0.6608 | 0.5156 | 0.7487 |
| **conventional+stdb_spatial** | **72** | **0.7652** | **0.6626** | **0.5141** | 0.7484 |
| full_node_features (+ scores) | 76 | 0.7649 | 0.6624 | 0.5141 | 0.7489 |

Match-paired Wilcoxon (n=2,182 test matches):

| Comparison | ΔAUC | ΔAP | p (AUC) |
| --- | ---: | ---: | ---: |
| full vs prior_scores | +0.1120 | +0.1700 | 0.0 (underflow; rank-biserial 0.957/0.967) |
| full vs prior_raw | +0.0427 | +0.0731 | 4.33e-218 |
| conv+stdb_spatial vs conventional | +0.0123 | +0.0140 | 8.79e-79 |
| conv+exposure vs conventional | +0.0107 | +0.0125 | 1.93e-70 |
| conv+trajectory vs conventional | +0.0049 | +0.0055 | 7.50e-31 |
| stdb_spatial(16) vs prior_scores(5) | +0.0573 | +0.1028 | 1.40e-157 |
| full vs conv+stdb_spatial | −0.0003 | −0.0002 | 0.754 (n.s.) |

Interpretation: the largest gain comes from the representation itself
(+0.112 AUC over prior-style scores, far exceeding the old-snapshot +0.088).
The STDB-specific spatial features now add +0.0123 AUC over conventional
aggregates (vs +0.0035 on the old squad-only snapshot), with exposure
contributing more than trajectory. Derived prior scores add nothing on top of
the 72-feature set, which also reproduces the Section-2 tabular baseline to
within 0.0001 AUC. Note: the 72-feature ablation refit and the Section-2
baseline are separate fits of the same configuration.

## 9. Phase-Wise Evaluation on the Full DB — verified 2026-06-11

Source: `phase_result_analysis_edges/hazard_phase_metrics.csv` (same aligned
134,245 cells; pooled per cut phase, full vs no-graph):

| cut | n | pos rate | AUC none | AUC full | ΔAUC | ΔAP | ΔBCE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 43,957 | 0.276 | 0.7885 | 0.8010 | +0.0125 | +0.0153 | −0.0090 |
| 2 | 32,553 | 0.251 | 0.7326 | 0.7437 | +0.0111 | +0.0167 | −0.0065 |
| 3 | 24,978 | 0.341 | 0.7247 | 0.7318 | +0.0070 | +0.0074 | −0.0044 |
| 4 | 16,896 | 0.459 | 0.7591 | 0.7694 | +0.0103 | +0.0070 | −0.0074 |
| 5 | 9,297 | 0.446 | 0.7308 | 0.7418 | +0.0111 | +0.0128 | −0.0093 |
| 6 | 4,863 | 0.445 | 0.6808 | 0.7000 | +0.0192 | +0.0151 | −0.0104 |
| 7 | 1,701 | 0.422 | 0.6292 | 0.6538 | +0.0245 | +0.0223 | −0.0078 |

**Narrative update vs the old snapshot:** on the old TPP-squad snapshot the
graph gain concentrated in early-mid phases and faded late (p6 +0.0024). On
the full population the gain is positive at every cut phase and is largest in
the late, sparse phases. The old phase-wise claim must not be carried into
the paper.

## 10. Robustness Reruns — started 2026-06-11

- **Edge sanity DONE** (inference-only, node_shuffle x5, full-graph
  checkpoint, same 134,245 aligned cells; observed reproduces the Section-2
  full-graph row exactly):
  observed AUC 0.7782 / AP 0.6808 / BCE 0.5016 vs node-shuffled
  AUC 0.7013±0.0006 / AP 0.6024±0.0006 / BCE 0.5770±0.0005
  → ΔAUC −0.0769, ΔAP −0.0784, ΔBCE +0.0754. Shuffled is far below even the
  no-graph control (0.7681), so the relational gain is not regularization.
  (Old snapshot: ΔAUC ≈ −0.065 — consistent, slightly stronger here.)
  Output: `full_db_reexperiment/edge_sanity/`.
- Seed repeats IN PROGRESS (training seeds 13, 2026 on the fixed canonical
  split, edge modes full and none; seed 42 exists from the main run):
  `full_db_reexperiment/seed_repeats/`.
- Orchestration: `scripts/run_full_db_robustness.sh`, log:
  `full_db_reexperiment/full_db_robustness.log`.
- **Aux-head ablation QUEUED** (hazard-only: `--beta 0 --gamma 0`, full and
  none, seed 42) — starts automatically after the robustness chain:
  `scripts/run_full_db_aux_ablation.sh` → `full_db_reexperiment/aux_ablation/`.
  Purpose: verify the relational gain does not depend on final-outcome
  auxiliary supervision (reviewer-critical).

## 11. Mode-Wise Discrimination — verified 2026-06-11

Aligned test cells joined with `stdb.matches.game_mode` (AUC/AP):

| mode | n cells | HGB | no-graph | full | ΔAUC full−none |
| --- | ---: | ---: | ---: | ---: | ---: |
| squad | 77,742 | 0.767/0.641 | 0.770/0.649 | 0.778/0.659 | +0.008 |
| squad-fpp | 7,724 | 0.765/0.667 | 0.763/0.671 | 0.777/0.686 | +0.014 |
| duo | 16,312 | 0.750/0.663 | 0.752/0.666 | 0.761/0.676 | +0.009 |
| duo-fpp | 5,885 | 0.738/0.648 | 0.740/0.655 | 0.760/0.674 | +0.020 |
| solo | 19,269 | 0.767/0.714 | 0.774/0.722 | 0.785/0.733 | +0.011 |
| solo-fpp | 7,313 | 0.774/0.708 | 0.777/0.716 | 0.790/0.727 | +0.013 |

**Full graph beats both references in all six modes** — the relational gain
is not a mode shortcut. Split mode proportions are balanced
(squad 69.7/68.7/70.9% across train/val/test).
