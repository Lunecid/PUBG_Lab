# Representation Ablation Status — 2026-06-14

## Current State

The project now has an executable representation-family ablation path:

- `stdb.v_team_phase_official_raw_features`: official raw phase baseline view, 85-column contract with 78 currently available columns.
- `stdb.v_team_phase_feature_v2`: unified 187-column view joining current STDB, F0 extras, F6 network role, and F7 habitat pressure.
- `model_stgcn.feature_ablation --config-style manifest`: runs manifest-defined A0-A11 feature-only ablations.
- `model_stgcn.eval_mr --shuffle-time`: adversarial temporal-order test for MR-STGNN.

`combat_zone_pass_count` has been patched to use only phase-local combat events, preventing cross-cut leakage.

## Feature-Only A0-A11 Results

Protocol:

- Split: `model_stgcn/checkpoints/full_db_reexperiment/split_phase_all_current_seed42.json`
- Test alignment: `model_stgcn/checkpoints/full_db_reexperiment/v2_node_features/phase_result_analysis/hazard_predictions_test_aligned.csv`
- Test rows: 134,245 team-phase risk cells, 2,182 matches
- Feature view: `stdb.v_team_phase_feature_v2`
- Output: `model_stgcn/checkpoints/feature_ablation_full_db_manifest_v2/`

| Config | Representation | Features | AUC | AP | BCE |
| --- | --- | ---: | ---: | ---: | ---: |
| A0 | official raw phase | 78 | 0.7589 | 0.6564 | 0.5183 |
| A1 | prior raw inputs | 12 | 0.7185 | 0.5704 | 0.5583 |
| A2 | prior score summary | 5 | 0.6488 | 0.4547 | 0.5994 |
| A3 | current conventional | 57 | 0.7523 | 0.6477 | 0.5241 |
| A4 | STDB spatial only | 18 | 0.7212 | 0.6007 | 0.5476 |
| A5 | raw + STDB | 95 | 0.7676 | 0.6667 | 0.5115 |
| A6 | raw + network | 121 | 0.7644 | 0.6621 | 0.5147 |
| A7 | raw + habitat | 110 | 0.7828 | 0.6833 | 0.4996 |
| A8 | raw + STDB + network | 138 | 0.7692 | 0.6685 | 0.5103 |
| A9 | raw + STDB + habitat | 127 | 0.7834 | 0.6843 | 0.4991 |
| A10 | raw + network + habitat | 153 | 0.7832 | 0.6841 | 0.4993 |
| A11 | full representation | 170 | 0.7838 | 0.6846 | 0.4989 |

Main reading:

- A0 official raw is already strong: AUC 0.7589.
- Prior score compression remains weak: A2 AUC 0.6488.
- STDB adds a smaller but significant gain over raw: A5 vs A0, +0.0087 mean per-match AUC, p=1.58e-54.
- Habitat is the strongest feature family in this design: A7 vs A0, +0.0247 mean per-match AUC, p=1.11e-137.
- Network feature-only gains are modest: A6 vs A0, +0.0057 mean per-match AUC, p=1.58e-28.
- Full A11 is only marginally above A9; the extra network block over raw+STDB+habitat is not clearly decisive in feature-only HGB.

## Graph Ablation Result

B3/B4 is now complete under the 141d V2 node-feature setting:

| Model | Node Features | Graph | AUC | AP |
| --- | --- | --- | ---: | ---: |
| B3 | F4/F5/F6/F7 phase node features | off | 0.7862 | 0.6889 |
| B4 | F4/F5/F6/F7 phase node features | on | 0.7924 | 0.6980 |

B4 beats B3 by +0.0062 overall AUC. Per-match paired AUC improvement is +0.0037 with p=4.89e-14.

Interpretation: graph structure was not useless; it needed stronger node-side representation. With F6/F7 role and pressure features available, graph message passing adds measurable signal.

## MR Time-Shuffle Adversarial Test

Full squad MR checkpoint:

| Evaluation | AUC | AP | Acc |
| --- | ---: | ---: | ---: |
| MR normal | 0.8422 | 0.7319 | 0.7944 |
| MR snapshot time-shuffled | 0.5124 | 0.3485 | 0.6476 |

Output: `model_stgcn/checkpoints/mr_full_squad/mr_shuffle_time_seed0_test.json`

Interpretation: the MR gain is not just from adding more snapshot observations. When snapshot order is destroyed while retaining the same snapshot set, predictive performance collapses close to random AUC. The snapshot tier is using temporal structure.

Robustness runs:

| Dataset / checkpoint | Normal AUC | Time-shuffle AUC | ΔAUC |
| --- | ---: | ---: | ---: |
| full squad, shuffle seed 0 | 0.8422 | 0.5124 | -0.3298 |
| full squad, shuffle seed 1 | 0.8422 | 0.5115 | -0.3307 |
| full squad, shuffle seed 2 | 0.8422 | 0.5110 | -0.3313 |
| Tiger squad MR seed 42 | 0.8374 | 0.5376 | -0.2998 |
| Tiger squad MR seed 13 | 0.8375 | 0.5018 | -0.3357 |
| Tiger squad MR seed 2026 | 0.8340 | 0.6000 | -0.2340 |
| solo/duo | 0.8141 | 0.6336 | -0.1805 |

The collapse is strongest and most stable on full squad. Tiger and solo/duo also lose large margins, although the residual AUC is higher for Tiger seed 2026 and solo/duo.

## Full-Squad MR Seed Repeat

Full-squad MR and phase-only models were retrained on the same fixed `mr_full_squad/split.json` split with three initialization seeds. The split manifest remains seed 42; the repeated seeds affect model initialization and training stochasticity, not the train/validation/test membership.

Output: `model_stgcn/checkpoints/mr_full_squad/`

| Seed | MR AUC | MR AP | Phase-only AUC | Phase-only AP | MR - Phase AUC | Time-shuffle AUC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.8422 | 0.7317 | 0.7905 | 0.6734 | +0.0517 | 0.5124 |
| 13 | 0.8428 | 0.7325 | 0.7910 | 0.6742 | +0.0518 | 0.5693 |
| 2026 | 0.8425 | 0.7317 | 0.7915 | 0.6752 | +0.0510 | 0.5187 |
| Mean | 0.8425 | 0.7320 | 0.7910 | 0.6743 | +0.0515 | 0.5335 |
| Std | 0.0003 | 0.0004 | 0.0005 | 0.0009 | 0.0004 | 0.0312 |

Interpretation: the full-squad MR result is not seed-fragile. Across three independently initialized training runs, MR remains about +0.0515 AUC above the phase-only model on the same risk cells. Time-shuffling the snapshot order still causes a large AUC collapse, confirming that the MR tier is using temporal ordering rather than simply benefiting from repeated observations.

## V2 Graph Sanity Control

V2 node-feature full graph checkpoint was evaluated with inference-time graph randomization:

| Control | Repeats | AUC | AP | BCE |
| --- | ---: | ---: | ---: | ---: |
| observed graph | 1 | 0.7924 | 0.6980 | 0.4897 |
| node-shuffled graph | 5 | 0.7753 | 0.6724 | 0.5097 |
| edge-shuffled graph | 5 | 0.7666 | 0.6575 | 0.5305 |

Output: `model_stgcn/checkpoints/full_db_reexperiment/v2_node_features/edge_sanity/`

Interpretation: observed graph relations outperform both node-shuffled and density-preserving edge-shuffled controls. The graph gain is not explained by generic edge density or regularization.

Implementation note: `model_stgcn.edge_sanity` now row-normalizes shuffled control adjacency before inference, preserving the model input contract.

## Phase-Band Summary

Output: `model_stgcn/checkpoints/manifest_importance_v2/manifest_phase_band_metrics.csv`

| Model | Early AUC | Mid AUC | Late AUC |
| --- | ---: | ---: | ---: |
| A0 official raw | 0.7570 | 0.7309 | 0.6715 |
| A5 raw + STDB | 0.7686 | 0.7387 | 0.6796 |
| A7 raw + habitat | 0.7885 | 0.7521 | 0.6923 |
| A11 full representation | 0.7892 | 0.7538 | 0.6969 |

Habitat pressure contributes across all phase bands, with the largest absolute shift over A0 in early and mid phases. Late phase remains the hardest segment because the risk set is small and highly selective.

## A11 Permutation Importance

Output: `model_stgcn/checkpoints/manifest_importance_v2/`

Family-level permutation importance for A11:

| Family | AUC drop | AP drop | BCE increase |
| --- | ---: | ---: | ---: |
| F0 official raw | 0.1441 | 0.2247 | 0.1440 |
| F7 habitat pressure | 0.0873 | 0.1416 | 0.0915 |
| F6 network role | 0.0068 | 0.0100 | 0.0064 |
| F5 STDB spatial | 0.0059 | 0.0084 | 0.0054 |

Top individual features by AUC drop:

| Feature | AUC drop |
| --- | ---: |
| `position_rows` | 0.0199 |
| `hab_local_enemy_count_200m` | 0.0175 |
| `kills_taken` | 0.0144 |
| `hab_signed_boundary_distance_end_m` | 0.0123 |
| `hab_distance_to_target_safe_m` | 0.0089 |
| `hab_local_enemy_count_100m` | 0.0081 |
| `hab_local_enemy_count_500m` | 0.0070 |
| `pickup_count` | 0.0052 |
| `team_size` | 0.0052 |
| `vehicle_ratio` | 0.0044 |

Interpretation: the strongest non-raw family is F7 habitat pressure. Network and STDB spatial remain useful but are smaller at the feature-only HGB level; network relation strength is clearer in the graph sanity and B3/B4 GNN experiments.

## Implementation Notes

- `tools/check_feature_manifest.py` now defaults to the manifest target view (`source_view_target`) instead of the old long view.
- `model_stgcn.feature_ablation` now supports:
  - `--feature-view stdb.v_team_phase_feature_v2`
  - `--config-style manifest`
  - manifest-defined A0-A11 configs
- `model_stgcn.eval_mr --shuffle-time` now handles device-safe permutation for MPS/CUDA/CPU.
- `model_stgcn.manifest_importance` adds A11 family/feature permutation importance and early/mid/late phase-band metrics.
- `scripts/run_mr_shuffle_robustness.sh` runs repeatable MR time-shuffle evaluations across existing checkpoints.
- `scripts/run_mr_full_squad_seed_repeats.sh` retrains full-squad MR and phase-only checkpoints across initialization seeds and evaluates time-shuffle controls.

## Remaining Work

1. Decide whether to implement ST2. Current evidence suggests ST2 is less urgent than documenting A0-A11, B3/B4, and MR.
2. Add a compact statistical comparison table for full-squad MR vs phase-only across the three seed repeats.
3. Add a small result table generator for A0-A11/B3-B4/MR so paper tables are reproducible from checkpoint CSV/JSON files.
4. Update manuscript framing: representation family comparison first, architecture second.
