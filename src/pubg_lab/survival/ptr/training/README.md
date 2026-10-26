# pubg_lab.survival.ptr.training

Explicit training entry points and objective components.

| Component | Purpose |
|---|---|
| `loss.py` | PTR loss: hazard BCE over ``target_mask`` + ``rank_weight`` * pairwise logistic rank loss. |
| `train.py` | Train one PTR arm on one cohort dump (DB-free) and write predictions + metrics. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
