# projects.sac2027.evaluation.metrics

metrics package.

| Component | Purpose |
|---|---|
| `sac2027_baseline_audit.py` | Reproducible source and numerical fidelity checks against the pinned archive. |
| `sac2027_count_error.py` | Post hoc checks of the RQ1 explanation on stored objective-control predictions (nothing is retrained). |
| `sac2027_did_rq2.py` | Phase-minus-match differences for the RQ2 contrasts (review item S2), on the resampled matches of the RQ2 offset analysis. |
| `sac2027_evaluate_controls.py` | Matched test contrasts of frozen controls; average seed metrics, not ensembles. |
| `sac2027_extra_evaluation.py` | Review follow-ups on the patch-split test set (stored seed-42 predictions; only the logistic baseline is new). |
| `sac2027_native_evaluation.py` | Original-task discrimination of the new patch predictions, on matched rows. |
| `sac2027_recall_norm.py` | Ceiling-aware review metrics for the SAC 2027 manuscript (stored seed-42 test predictions; nothing is retrained). |
| `sac2027_rq1_phase_equality.py` | AI-adjusted phase-band equality tests using full clustered covariance. |
| `sac2027_sesoi_top3.py` | Review-scale meaning of the SESOI: top-3 capture differences for the primary and design contrasts (stored seed-42 test |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
