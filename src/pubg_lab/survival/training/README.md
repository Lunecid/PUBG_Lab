# pubg_lab.survival.training

Explicit training entry points and objective components.

| Component | Purpose |
|---|---|
| `train.py` | model_stgcn/train.py |
| `train_mr.py` | Train the Multi-Resolution ST-GNN (snapshot 10s + phase). |
| `train_phase.py` | Train DynamicSTGNN on phase-level PUBG tensors. |
| `train_snapshot_phase.py` | Train snapshot-resolution STGNN with phase-cut survival outputs. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
