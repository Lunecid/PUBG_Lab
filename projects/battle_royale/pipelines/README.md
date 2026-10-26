# projects.battle_royale.pipelines

Explicit orchestration.

| Component | Purpose |
|---|---|
| `fixb_backup_tables.py` | Back up the derived stdb tables that the Fix B ETL rerun overwrites. |
| `generate_dynamic_graph_example.py` | Generate a paper figure showing a phase snapshot as a dynamic graph. |
| `generate_om_prediction_csvs.py` | Generate per-team-phase prediction CSVs for a completed OM ladder run. |
| `generate_team_phase_feature_example.py` | Generate a paper figure showing PUBG telemetry as team-phase features. |
| `rerun_v2.py` | Orchestrate the whole BigComp 2027 re-run on the leak-free v2 dump: gate, CPU lane, MPS lane(s), aggregation, tables, paper numbers. |
| `run_domain_training.py` | Run map/mode-specific DynamicSTGNN training experiments. |
| `run_fixb_phase_etl.py` | Fix B rerun: rebuild the phase-tier derived tables with capped grid joins. |
| `run_o6z_required_cohort_etl.py` | Build O6Z Z-affordance features for the required PUBG cohorts. |
| `seed_repeat_experiments.py` | Run and aggregate required seed-repeat DynamicSTGNN experiments. |
| `stamp_artifact_provenance.py` | Stamp existing experiment artifact roots with code and file hashes. |

Import the canonical module path. `python -m pubg_lab list` lists explicit commands without executing them.
Research inputs, completed evidence, and access restrictions follow the repository guidelines.
