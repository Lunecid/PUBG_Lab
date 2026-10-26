# pubg_lab.survival

Responsibility: BR/PTR/survival representation and model components.

Dependency boundary: Data contracts; private split access remains outside the public package.

| Component | Responsibility |
|---|---|
| `baselines.py` | Sklearn baselines for phase-level PUBG survival/rank prediction. |
| `bootstrap_ci.py` | Match-level (cluster) bootstrap confidence intervals for evaluation metrics. |
| `calibration.py` | Calibration diagnostics for hazard prediction CSVs. |
| `config.py` | Central path/template constants for the model_stgcn experiment suite. |
| `dataset.py` | model_stgcn/dataset.py |
| `domain_ablation.py` | Domain-stratified ablation summaries. |
| `dump_phase_tier.py` | Per-cohort tensor dumps of the phase tier (design v2.3: one clock + Fix B cap) for the PTR study. |
| `dump_phase_tier_v2.py` | Leak-free phase-tier dump v2 for the BigComp 2027 PTR study (writes ``bigcomp2027_phase_v2``; v1 is untouched). |
| `dump_player_tier.py` | Per-cohort PLAYER-level tensor dumps (design stage 2, section 3) next to the phase dumps. |
| `dyn_stgnn.py` | Dynamic Spatio-Temporal GNN for phase-level PUBG team survival modeling. |
| `edge_modes.py` | Edge-channel ablation helpers for phase-level DynamicSTGNN experiments. |
| `edge_sanity.py` | Random-edge sanity checks for phase-level DynamicSTGNN checkpoints. |
| `eval.py` | model_stgcn/eval.py |
| `eval_mr.py` | Test-set evaluation for the MR-STGNN (hazard AUC/AP/acc), the snapshot-tier |
| `eval_phase.py` | Evaluate DynamicSTGNN phase checkpoints with baseline-compatible metrics. |
| `feature_ablation.py` | Feature-group ablation for phase-level PUBG hazard prediction. |
| `infer.py` | model_stgcn/infer.py |
| `manifest_importance.py` | Manifest-V2 phase-band and permutation-importance diagnostics. |
| `mr_dataset.py` | Dataset + collate for the MR-STGNN: each item carries BOTH the phase tier |
| `mr_stgnn.py` | Multi-Resolution hierarchical ST-GNN (MR-STGNN). |
| `novelty_analysis.py` | Novelty diagnostics for the PUBG STDB model. |
| `phase_result_analysis.py` | Phase-wise hazard analysis and paired statistical tests. |
| `phase_wise_eval.py` | Phase-wise evaluation for domain edge ablation checkpoints. |
| `provenance.py` | Small provenance helpers for experiment artifacts. |
| `run_domain_training.py` | Run map/mode-specific DynamicSTGNN training experiments. |
| `seed_repeat_experiments.py` | Run and aggregate required seed-repeat DynamicSTGNN experiments. |
| `snapshot_baselines.py` | Minimal no-graph snapshot baselines for pre-experiment validation. |
| `snapshot_edge_controls.py` | Snapshot graph null controls. |
| `snapshot_encoder.py` | Snapshot (10s) encoder + phase pooling for the MR-STGNN Tier-S. |
| `snapshot_phase_dataset.py` | Dataset for snapshot-resolution phase survival training. |
| `snapshot_phase_survival.py` | Snapshot-resolution ST survival model with phase-cut outputs. |
| `snapshot_relative_edge.py` | Snapshot relative-edge hazard model. |
| `snapshot_tabular_baseline.py` | Same-condition tabular baselines for snapshot phase-survival models. |
| `stgcn.py` | model_stgcn/stgcn.py |
| `train.py` | model_stgcn/train.py |
| `train_mr.py` | Train the Multi-Resolution ST-GNN (snapshot 10s + phase). |
| `train_phase.py` | Train DynamicSTGNN on phase-level PUBG tensors. |
| `train_snapshot_phase.py` | Train snapshot-resolution STGNN with phase-cut survival outputs. |

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.
