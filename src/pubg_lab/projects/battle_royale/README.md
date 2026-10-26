# pubg_lab.projects.battle_royale

Responsibility: SAC, BR and figure applications composing reusable components.

Dependency boundary: Domain libraries; do not become dependencies of common infrastructure.

| Component | Responsibility |
|---|---|
| `adv_analysis.py` | Analyze the learned-advantage edge experiment: per-cut-phase hazard AUC per |
| `adv_precompute.py` | Precompute & cache phase-tier tensors + fused-gate advantage edges to disk. |
| `advantage_edge_feasibility.py` | Feasibility for the contact-gated, learned-advantage edge design. |
| `aggregate_o7_advantage_readout.py` | Aggregate O7 advantage-edge A/B/C phase-group readouts across cohorts. |
| `analyze_graph_signal.py` | Re-verify the V3 dynamic-graph signal WITHOUT retraining. |
| `analyze_o7_advantage_rank_bridge.py` | Hazard-implied rank bridge for O7 advantage-edge A/B/C predictions. |
| `analyze_o7_advantage_readout.py` | Analyze O7 advantage-edge A/B/C early-readout predictions. |
| `audit_o1_survival_capital.py` | Audit O1 survival-capital features on the OM ladder test splits. |
| `audit_o6_z_ladder_availability.py` | Audit availability of the planned PUBG O6 Z-ladder features. |
| `audit_o7_advantage_edge_shuffle.py` | Audit the O7 advantage-edge edge-shuffle null control. |
| `audit_spatial_feasibility.py` | Bounded read-only spatial support audit, not a walkability or learning test. |
| `audit_spatial_features.py` | Audit whether elevation(z)/terrain/high-ground/LOS spatial variables exist |
| `bootstrap_om_paired_deltas.py` | Paired match-cluster bootstrap for OM ladder prediction CSVs. |
| `build_dynamic_probability_fluctuation.py` | Build dynamic survival / win-probability trajectories from hazard predictions. |
| `build_realtime_prefix_probability_demo.py` | Build a 10-second prefix probability demo from a snapshot LSTM checkpoint. |
| `check_feature_manifest.py` | Check a feature-family manifest against an STDB feature view. |
| `collect_om_ladder_metrics.py` | Collect per-cohort OM ladder matrix metrics into one run-level table. |
| `collect_om_phase_metrics.py` | Collect phase-stratified metrics for the required-cohort OM ladder. |
| `collect_rq4_baseline_metrics.py` | Aggregate RQ4 prior-work/basic-tabular baseline results across cohorts. |
| `collect_rq4_graph_design_metrics.py` | Aggregate RQ4 graph-design matrix metrics across required cohorts. |
| `collect_snapshot_survival_bridge_metrics.py` | Aggregate snapshot cumulative-survival bridge metrics across cohorts. |
| `compare_phase_dumps_v1_v2.py` | Compare the phase-tier dumps v1 and v2 row by row (read-only; both dump trees are only loaded). |
| `compare_rq4_board_baselines.py` | Compare RQ4 phase-transition risk/advantage boards against baselines. |
| `diagnose_snapshot_relative_edge.py` | Diagnose the minimal snapshot relative-edge pilot. |
| `eval_phase_transition_risk_board.py` | Evaluate phase-transition risk and advantage boards from trajectory rows. |
| `eval_snapshot_phase_checkpoint.py` | Evaluate a snapshot-phase checkpoint and write a train-style summary. |
| `eval_snapshot_survival_bridge.py` | Evaluate snapshot-phase hazard checkpoints as cumulative survival forecasts. |
| `fixb_backup_tables.py` | Back up the derived stdb tables that the Fix B ETL rerun overwrites. |
| `generate_dynamic_graph_example.py` | Generate a paper figure showing a phase snapshot as a dynamic graph. |
| `generate_om_prediction_csvs.py` | Generate per-team-phase prediction CSVs for a completed OM ladder run. |
| `generate_team_phase_feature_example.py` | Generate a paper figure showing PUBG telemetry as team-phase features. |
| `highground_signal_test.py` | NOW-WORK step 1: does high-ground actually predict next-phase elimination? |
| `make_result_figures_v2.py` | Result figures and the main results table of the BigComp 2027 paper, rebuilt from the finished v2 re-run. |
| `mr_paired_significance.py` | Match-paired significance: MR vs phase-only vs HGB on the full-squad test set. |
| `paper_numbers_v2.py` | Recompute every dump-dependent number of the BigComp 2027 paper drafts from the v2 dump and the database (manifest section E). |
| `phase_significance.py` | Per-phase match-level (cluster) bootstrap significance for hazard AUC. |
| `phase_time_profile.py` | Ground-truth phase timing/zone profile for squad matches from the ST-DB. |
| `rerun_v2.py` | Orchestrate the whole BigComp 2027 re-run on the leak-free v2 dump: gate, CPU lane, MPS lane(s), aggregation, tables, paper numbers. |
| `run_fixb_phase_etl.py` | Fix B rerun: rebuild the phase-tier derived tables with capped grid joins. |
| `run_o6z_required_cohort_etl.py` | Build O6Z Z-affordance features for the required PUBG cohorts. |
| `smoke_snapshot_phase_contract.py` | Smoke-test the snapshot-first phase-survival tensor contract. |
| `stamp_artifact_provenance.py` | Stamp existing experiment artifact roots with code and file hashes. |
| `v3_phase_significance.py` | V3 per-cut-phase significance via match-level (cluster) bootstrap. |
| `v3_resolution_significance.py` | Resolution-group significance: does the multi-resolution snapshot model beat |
| `v3_tabular_baseline.py` | V3-comparable tabular hazard baselines (LR, HGB). |
| `validate_genre_feature_manifest.py` | Validate the DSACG genre-primitive feature manifest. |
| `verify_dump_match.py` | Hand check of ONE dumped match against the database (independent re-derivation, design v2.3). |
| `verify_dump_match_v2.py` | Independent re-derivation of one v2 phase-dump record straight from the raw ``pubg.*`` tables (hand-check tool). |
| `verify_player_dump_match.py` | Hand check of dumped PLAYER-level matches against the database (independent re-derivation). |

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.
