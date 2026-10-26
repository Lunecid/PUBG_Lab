# pubg_lab.survival.models

Model architectures and reusable components.

| Component | Purpose |
|---|---|
| `dyn_stgnn.py` | Dynamic Spatio-Temporal GNN for phase-level PUBG team survival modeling. |
| `mr_stgnn.py` | Multi-Resolution hierarchical ST-GNN (MR-STGNN). |
| `snapshot_baselines.py` | Minimal no-graph snapshot baselines for pre-experiment validation. |
| `snapshot_encoder.py` | Snapshot (10s) encoder + phase pooling for the MR-STGNN Tier-S. |
| `snapshot_phase_survival.py` | Snapshot-resolution ST survival model with phase-cut outputs. |
| `snapshot_relative_edge.py` | Snapshot relative-edge hazard model. |
| `stgcn.py` | model_stgcn/stgcn.py |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
