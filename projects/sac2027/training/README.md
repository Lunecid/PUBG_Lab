# projects.sac2027.training

Explicit training entry points and objective components.

| Component | Purpose |
|---|---|
| `sac2027_fair_tune.py` | Frozen equal-trial-budget search. No test predictions until all trials finish. |
| `sac2027_followup_run.py` | Dependency-aware local queues for pre-specified controls (no external scheduling). |
| `sac2027_graph_run.py` | One graph architecture, frozen equal-budget tuning and an edge-shuffle control. |
| `sac2027_logistic_baseline.py` | Full-feature regularized logistic baseline on the patch split (phase intercepts + the reference GBDT's 151 inputs). |
| `sac2027_patch_rebuild.py` | Rebuild causal features from raw, read-only telemetry under a frozen patch split. |
| `sac2027_patch_run.py` | Launch isolated patch experiments after their data-validation gate completes. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
