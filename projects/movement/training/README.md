# projects.movement.training

Explicit training entry points and objective components.

| Component | Purpose |
|---|---|
| `experiment_v2.py` | Fixed-budget paired experiment. Validation is evaluated only after all training. |
| `experiment_v3.py` | Fixed-budget V3 paired experiment. Test access is explicitly forbidden. |
| `experiment_v4.py` | Frozen V4 protocol: fit-only balanced sampling, paired regression and diffusion. |
| `parallel_replica_v2.py` | Run one independent preplanned seed; no changes to the frozen experiment. |
| `prepare_v4.py` | Fit-only sampling metadata; evaluation plans are byte-identical V3 copies. |
| `train.py` | Train paired information views using identical seeds/anchors; optional bounded runs. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
