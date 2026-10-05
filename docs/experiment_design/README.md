# Experiment Design Docs

Package class: `ExperimentContract`

## Responsibility

Own split design, run matrices, result summaries, evaluation metrics,
statistical tests, and ablation plans.

## Public Interface

| Object | Role |
| --- | --- |
| `BATTLE_ROYALE_RESEARCH_EXECUTION_CHECKLIST_2026-07-03.md` | operational checklist for current Battle Royale framework work |
| `BATTLE_ROYALE_EXPERIMENT_EVIDENCE_LEDGER_2026-07-05.md` | detailed evidence ledger mapping experiments to supported and bounded claims |
| `BATTLE_ROYALE_RESULT_SYNTHESIS_2026-07-10.md` | manuscript-facing synthesis of current RQ2-RQ4 experiment results, claim boundaries, and result-table plan |
| `BATTLE_ROYALE_CLAIM_EXPERIMENT_MAPPING_2026-07-06.md` | manuscript-facing RQ1-RQ4 claim-to-experiment/result mapping and remaining gap summary |
| `BATTLE_ROYALE_EXPERIMENT_STATUS_AND_NEXT_DECISION_2026-07-06.md` | current revised RQ1-RQ4 experiment status, model comparison conclusion, and board-baseline decision |
| `BATTLE_ROYALE_RQ4_PRIOR_BASELINE_RESULT_2026-07-05.md` | revised-RQ3 prior-work positioning result for Sagong-style/basic tabular baselines; graph design removed from core |
| `BATTLE_ROYALE_RQ4_DYNAMIC_PROBABILITY_FLUCTUATION_RESULT_2026-07-06.md` | revised-RQ3 dynamic probability trajectory result that distinguishes realtime survival readouts from static score/final-label baselines |
| `BATTLE_ROYALE_RQ4_PHASE_TRANSITION_RISK_ADVANTAGE_BOARD_RESULT_2026-07-06.md` | RQ4 phase-transition board result for most-at-risk and most-advantaged team identification |
| `BATTLE_ROYALE_RQ4_BOARD_BASELINE_COMPARISON_PROTOCOL_2026-07-06.md` | frozen RQ4 board-baseline comparison protocol |
| `BATTLE_ROYALE_RQ4_BOARD_BASELINE_COMPARISON_RESULT_2026-07-06.md` | RQ4 board-baseline comparison result and claim boundary |
| `BATTLE_ROYALE_RQ4_HIERARCHICAL_SURVIVAL_PROTOCOL_2026-07-05.md` | revised-RQ3 hierarchical / 10-second phase-survival model protocol and smoke status |
| `BATTLE_ROYALE_FRAMEWORK_VALIDATION_PLAN_2026-07-02.md` | validation plan for the general Battle Royale framework |
| `BATTLE_ROYALE_REQUIRED_COHORT_SET_2026-07-03.md` | required cohort set for main training and cohort ablations |
| `BATTLE_ROYALE_ONTOLOGY_VALIDATION_PROTOCOL_2026-07-03.md` | staged ontology validation protocol before predictive claims |
| `BATTLE_ROYALE_FEATURE_AVAILABILITY_AUDIT_2026-07-03.md` | OM0-OM6 feature availability audit for required PUBG cohorts |
| `BATTLE_ROYALE_OM_LADDER_RUNBOOK_2026-07-03.md` | runbook and artifact contract for cohort-separated OM0-OM6 runs |
| `BATTLE_ROYALE_OM_LADDER_RESULT_FEATURE_AUDIT_2026-07-04.md` | completed required-cohort OM0-OM6 result, cohort-size, and feature-block audit |
| `BATTLE_ROYALE_PHASE_STRATIFIED_OM_RESULT_2026-07-04.md` | phase-stratified required-cohort OM result and within-phase mechanism check |
| `BATTLE_ROYALE_O1_SURVIVAL_CAPITAL_INTEGRITY_AUDIT_2026-07-04.md` | O1 survival-capital tensor integrity and univariate hazard-signal audit |
| `BATTLE_ROYALE_PHASE_GROUP_BOOTSTRAP_RESULT_2026-07-04.md` | paired match-cluster bootstrap result for phase-group OM deltas |
| `BATTLE_ROYALE_O6_Z_LADDER_SENSITIVITY_PROTOCOL_2026-07-04.md` | corrected O6 Z-ladder sensitivity protocol and pre-run gates |
| `BATTLE_ROYALE_O6_Z_LADDER_FEATURE_AVAILABILITY_AUDIT_2026-07-04.md` | Z1-Z3 source/tensor availability audit before OM6Z training |
| `BATTLE_ROYALE_O6_Z_LADDER_RESULT_2026-07-05.md` | corrected O6Z required-cohort sensitivity result and claim boundary |
| `BATTLE_ROYALE_O7_EDGE_STREAM_NULL_GUARD_PROTOCOL_2026-07-05.md` | O7 advantage-edge 3-arm protocol and null-guard contract |
| `BATTLE_ROYALE_GEN1_GEN2_RESULT_BOUNDARY_2026-07-05.md` | decision note separating legacy gen-1 pilots from current gen-2 manuscript evidence |
| `BATTLE_ROYALE_O7_EDGE_SHUFFLE_AUDIT_2026-07-05.md` | O7 advantage-edge edge-shuffle audit and cohort-aware runner smoke result |
| `BATTLE_ROYALE_O7_EARLY_READOUT_ERANGEL_2026-07-05.md` | first gen-2 O7 advantage-edge A/B/C early readout on Erangel TPP squad |
| `BATTLE_ROYALE_O7_EARLY_READOUT_4COHORT_RESULT_2026-07-05.md` | four-cohort O7 advantage-edge early readout and expansion decision |
| `BATTLE_ROYALE_O7_REQUIRED_COHORT_HAZARD_RESULT_2026-07-05.md` | 13-cohort O7 advantage-edge hazard-AUC result and claim boundary |
| `BATTLE_ROYALE_O7_REQUIRED_COHORT_RANK_BRIDGE_RESULT_2026-07-05.md` | 13-cohort O7 hazard-implied rank bridge result and claim boundary |
| `REQUIRED_EXPERIMENTS.md` | required experiment checklist |
| `PRE_EXPERIMENT_DESIGN_LOCK_2026-06-22.md` | locked pre-experiment design |
| `PAPER_ST_FEATURE_MATRIX_RUNBOOK_2026-06-23.md` | ST feature matrix runbook |
| `FULL_DB_REEXPERIMENT_REPORT.md` | full DB re-experiment report |
| `MODEL_RESULTS.md` | model result summary |
| `V3_FINAL_DESIGN_STATUS_2026-06-15.md` | V3 final design status |

## Does Not Own

- model architecture: use `../model_design/`;
- feature definitions: use `../feature_engineering/`;
- manuscript prose: use `../paper_writing/`.

See `../DESIGN_DOC_STRUCTURE_2026-07-02.md` and
`../DOC_OBJECT_MODEL_2026-07-02.md` for routing rules.
