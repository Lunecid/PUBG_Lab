# projects.battle_royale.evaluation.contracts

contracts package.

| Component | Purpose |
|---|---|
| `check_feature_manifest.py` | Check a feature-family manifest against an STDB feature view. |
| `collect_snapshot_survival_bridge_metrics.py` | Aggregate snapshot cumulative-survival bridge metrics across cohorts. |
| `compare_phase_dumps_v1_v2.py` | Compare the phase-tier dumps v1 and v2 row by row (read-only; both dump trees are only loaded). |
| `diagnose_snapshot_relative_edge.py` | Diagnose the minimal snapshot relative-edge pilot. |
| `eval_snapshot_phase_checkpoint.py` | Evaluate a snapshot-phase checkpoint and write a train-style summary. |
| `eval_snapshot_survival_bridge.py` | Evaluate snapshot-phase hazard checkpoints as cumulative survival forecasts. |
| `smoke_snapshot_phase_contract.py` | Smoke-test the snapshot-first phase-survival tensor contract. |
| `validate_genre_feature_manifest.py` | Validate the DSACG genre-primitive feature manifest. |
| `verify_dump_match.py` | Hand check of ONE dumped match against the database (independent re-derivation, design v2.3). |
| `verify_dump_match_v2.py` | Independent re-derivation of one v2 phase-dump record straight from the raw ``pubg.*`` tables (hand-check tool). |
| `verify_player_dump_match.py` | Hand check of dumped PLAYER-level matches against the database (independent re-derivation). |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
