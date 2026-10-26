# projects.sac2027.evaluation.risk_models

risk models package.

| Component | Purpose |
|---|---|
| `sac2027_condlogit_eval.py` | Conditional-logit control and BCE-weight sweep: contrasts, closure shares, phase-minus-match differences, predictions. |
| `sac2027_match_offset.py` | Match-offset decomposition of within-phase AUC for the matched objective controls (stored test predictions only). |
| `sac2027_match_offset_rq2.py` | Match-offset decomposition of the RQ2 within-phase gaps and a spread check for the team axis (stored seed-42 test |
| `sac2027_match_offset_structure.py` | Match-offset decomposition for the structural contrasts of the main matrix (stored seed-42 test predictions only). |
| `sac2027_offset_only.py` | Offset-only control: frozen next-phase Cox scores plus a match-phase term learned with cross-entropy. |
| `sac2027_offset_only_cutoff.py` | Post hoc sensitivity of the offset-only control to its information set (not part of the recorded plan). |
| `sac2027_pair_share.py` | Share of same-match pairs among the (eliminated, surviving) pairs of within-phase AUC, per primary cohort. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
