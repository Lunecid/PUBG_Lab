# projects.movement.evaluation.diagnostics

diagnostics package.

| Component | Purpose |
|---|---|
| `condition_diagnostics_v2.py` | Post-evaluation sensitivity diagnostic. NOT a retrained feature ablation. |
| `diagnose_v3.py` | Locked-plan no-zone retraining comparison, sensitivity, and exact replay. |
| `diagnose_v4.py` | Diagnose frozen V3 predictions, without test access or fitting. |
| `generation_diagnostics_v2.py` | Distribution and simple numerical feasibility diagnostics; no collision claims. |
| `mobility_diagnostics_v2.py` | Post-hoc raw-event check of aerial transport in the fixed evaluation cohort. |
| `pre_evaluation_diagnostics.py` | Dataset-only validation diagnostics; no model scores or checkpoint selection. |
| `residual_drift_v4.py` | Post-evaluation residual mean drift; no refitting or changed primary estimates. |
| `transport_audit_v3.py` | Pre-prediction, mode-stratified secondary evaluation; no tuning or test access. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
