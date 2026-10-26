# pubg_lab.survival.ptr.models

Model architectures and reusable components.

| Component | Purpose |
|---|---|
| `arms.py` | Experiment arms: design doc section 4 and section 4.1 (v2.1). |
| `gatv2.py` | Snapshot GATv2 baseline using the pinned PyG 2.7.0 operator directly. |
| `hgb.py` | HGB reference baselines built from the same dump (identical rows to the neural arms). |
| `model.py` | PTR-Former: phase-axis, team-axis and relation-biased attention over team tokens. |
| `player.py` | Stage-2 player layer: the player dump, its join to the phase dump, and the player relation tensors. |
| `relations.py` | Relation matrices for the team-axis attention bias, and the within-match edge shuffle. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
