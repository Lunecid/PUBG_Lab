# projects.movement.evaluation.verification

verification package.

| Component | Purpose |
|---|---|
| `audit.py` | Full-bank/source integrity and sample-index audit without re-querying the DB. |
| `audit_experiment_v2.py` | Leakage audit independent of training; all outputs local, no DB mutation. |
| `audit_v3.py` | Read-only rebuilt-bank audit; never loads held-out test raw or banks. |
| `audit_v4.py` | V4 leakage/exposure audit: only train/validation banks are opened. |
| `smoke.py` | Real paired batches: optimizer steps, generation, masks, and runtime tensor shapes. |
| `verify_model_v2.py` | Runtime shape / gradient / seed reproducibility probe on training samples only. |
| `verify_results_v2.py` | Verify saved checkpoints, paired sample coverage, and exact inference replay. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
