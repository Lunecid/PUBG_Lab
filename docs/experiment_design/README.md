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
