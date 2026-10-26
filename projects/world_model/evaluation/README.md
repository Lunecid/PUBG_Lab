# projects.world_model.evaluation

Evaluation primitives and explicit verification commands.

| Component | Purpose |
|---|---|
| `audit_world_model_g4.py` | Post-run support and parameter audit; no fitting and no protected data loader. |
| `audit_world_model_inputs.py` | Full allowed-bank audit; never opens protected match banks. |
| `verify_world_model_encoder.py` | Bounded CPU G1 integration check. No optimizer, training or protected loader. |
| `verify_world_model_g3.py` | All primary selected labels + per-match/kind two-view adapter verification. |
| `verify_world_model_historical_prior.py` | Read-only historical side-channel demonstration on fixed current-state crops. |
| `verify_world_model_preprocess.py` | Offline audit of the completed preprocessing; never opens protected matches. |
| `world_model_toy_checks.py` | Hand-checkable unit/time/label examples; no DB and no training. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
