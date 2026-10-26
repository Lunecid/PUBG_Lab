# projects.battle_royale.analysis.robustness

robustness package.

| Component | Purpose |
|---|---|
| `analyze_graph_signal.py` | Re-verify the V3 dynamic-graph signal WITHOUT retraining. |
| `audit_spatial_feasibility.py` | Bounded read-only spatial support audit, not a walkability or learning test. |
| `audit_spatial_features.py` | Audit whether elevation(z)/terrain/high-ground/LOS spatial variables exist |
| `build_dynamic_probability_fluctuation.py` | Build dynamic survival / win-probability trajectories from hazard predictions. |
| `build_realtime_prefix_probability_demo.py` | Build a 10-second prefix probability demo from a snapshot LSTM checkpoint. |
| `highground_signal_test.py` | NOW-WORK step 1: does high-ground actually predict next-phase elimination? |
| `mr_paired_significance.py` | Match-paired significance: MR vs phase-only vs HGB on the full-squad test set. |
| `phase_significance.py` | Per-phase match-level (cluster) bootstrap significance for hazard AUC. |
| `phase_time_profile.py` | Ground-truth phase timing/zone profile for squad matches from the ST-DB. |
| `v3_phase_significance.py` | V3 per-cut-phase significance via match-level (cluster) bootstrap. |
| `v3_resolution_significance.py` | Resolution-group significance: does the multi-resolution snapshot model beat |
| `v3_tabular_baseline.py` | V3-comparable tabular hazard baselines (LR, HGB). |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
