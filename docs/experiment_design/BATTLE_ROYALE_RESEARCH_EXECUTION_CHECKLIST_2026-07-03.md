# Battle Royale Research Execution Checklist

Date: 2026-07-03
Class: `ExperimentContract`
Lifecycle: `active`

## Purpose

This document is the working checklist for the Battle Royale framework paper.
It tracks what has been completed, what remains, which methodology was applied,
and what concrete artifact should exist after each step.

The current state is post-ontology-lock, post-formalization,
post-ontology-validation-protocol, post analytic-framework claim-boundary
decision, post required-cohort OM ladder first run, post phase-stratified
OM mechanism analysis, post O1 survival-capital integrity audit, and post
phase-group paired bootstrap uncertainty check, corrected O6 Z-ladder
sensitivity protocol, Z1-Z3 feature availability audit, O6 Z-affordance
ETL/tensor mapping, and corrected OM6Z required-cohort run. The next major
work is the gen-1/gen-2 result boundary audit before any long O7 run,
followed by the O7 edge-stream/null-guard execution and, separately, the
Caldera O6 construct-validation artifact.

## Update Rule

Before starting a major new work item, add or confirm the checklist row for that
item. After completing it, update:

1. `Status`
2. `Progress / Finding`
3. `Method Applied`
4. `Artifact / Evidence`
5. `Next Action`

This file is the operational progress board. Design details stay in their owning
package docs.

## Status Legend

| Status | Meaning |
| --- | --- |
| `[x]` | complete for the current paper path |
| `[~]` | in progress or partially complete |
| `[ ]` | pending |
| `[!]` | blocked or requires an explicit decision |

## Current Critical Path

| Order | Status | Work Item | Method Applied | Artifact / Evidence | Next Action |
| --- | --- | --- | --- | --- | --- |
| 1 | `[x]` | Documentation object model cleanup | Documentation object model, package ownership | `docs/README.md`, package READMEs, `DOC_OBJECT_MANIFEST_2026-07-02.md` | Keep new docs registered through package interfaces |
| 2 | `[x]` | Caldera capability boundary | Data audit, observability mask, label boundary analysis | `research_framing/CALDERA_DATASET_STATUS_REDECISION_2026-07-02.md`, `TERRAIN_AFFORDANCE_CAPABILITY_BOUNDARY_2026-07-02.md` | Use Caldera for O6 construct validation, not supervised survival prediction |
| 3 | `[x]` | Network-analysis concept cleanup | Conceptual decomposition of graph measures | `feature_engineering/NETWORK_ANALYSIS_FOUNDATION_CONCEPTS_2026-07-01.md` | Keep graph measures as O7 projections, not genre definitions |
| 4 | `[x]` | Battle Royale ontology v1 lock | Ontology engineering, triangulation, claim boundary setting | `research_framing/BATTLE_ROYALE_ONTOLOGY_LOCK_2026-07-03.md` | Treat O0-O8 families as frozen for OM experiments |
| 5 | `[x]` | Ontology naming and ownership fixes | Corpus edge-case audit, ablation ownership rule | O2/O4/O7/O8 wording fixed in ontology and manifest docs | Prevent family-number drift and duplicate feature ownership |
| 6 | `[x]` | Battle Royale problem definition | Problem formulation, motivation, contribution boundary | `research_framing/BATTLE_ROYALE_PROBLEM_DEFINITION_2026-07-03.md` | Treat the project as phase-conditioned multi-agent survival, not just final-rank feature expansion |
| 7 | `[x]` | Battle Royale survival formalization | Phase-cut hazard formulation | `model_design/BATTLE_ROYALE_SURVIVAL_FORMALIZATION_2026-07-03.md` | Use the formula as the bridge from ontology to experiments |
| 8 | `[x]` | Paper argument chain premise | Related-work positioning, prior-work gap analysis, claim-boundary control | `research_framing/BATTLE_ROYALE_PAPER_ARGUMENT_CHAIN_2026-07-03.md` | Final-verify non-Sagong aggregate/rank prediction citations before manuscript submission |
| 9 | `[x]` | PUBG cohort registry lock | Cohort stratification by map, perspective, and team size | `data_layer/BATTLE_ROYALE_COHORT_REGISTRY_2026-07-03.md` | Use registered atomic cohorts for OM0-OM6 reporting |
| 10 | `[x]` | Required experiment cohort set lock | Independent training cohorts by map, perspective, and team size | `experiment_design/BATTLE_ROYALE_REQUIRED_COHORT_SET_2026-07-03.md` | Train OM0-OM6 separately for each required cohort |
| 11 | `[x]` | Ontology validation protocol lock | Validation ladder, claim boundary, construct-validation split | `experiment_design/BATTLE_ROYALE_ONTOLOGY_VALIDATION_PROTOCOL_2026-07-03.md` | Use OV0-OV6 as the gate for interpreting OM results |
| 12 | `[x]` | Feature availability audit for OM0-OM6 on required cohorts | Code-grounded manifest check, forbidden-input guard, DB/view inspection | `experiment_design/BATTLE_ROYALE_FEATURE_AVAILABILITY_AUDIT_2026-07-03.md` | Use named cohorts for smoke and required runs |
| 13 | `[!]` | Resolve gen-1/gen-2 experiment boundary | Result inventory, naming lock, legacy/current split | Pending decision note or results registry | Timebox before O7 execution: identify which old runs are reusable and which claims need reruns |
| 14 | `[x]` | OM preset/code readiness | Implementation audit against the survival formula and required cohorts | Clean `om0`-`om6` presets and named cohort filter mapping are implemented | Keep full preset as later stress test, not headline ladder |
| 15 | `[x]` | OM smoke runs | Small matched required-cohort splits, leakage check, cached/CPU sanity run | Erangel TPP squad, Erangel TPP duo, and Taego FPP squad smoke runs passed for OM0-OM6 with 30-match limit | Expand to required-cohort OM ladder runs |
| 16 | `[x]` | OM ladder artifact management | Runbook, cohort run root, run index, aggregation contract, completion-marker guard | `experiment_design/BATTLE_ROYALE_OM_LADDER_RUNBOOK_2026-07-03.md`, `scripts/run_br_om_ladder_required_cohorts.sh`, `tools/collect_om_ladder_metrics.py`, `model_stgcn/v3/run_matrix.py` | Use guarded runbook for required-cohort runs |
| 17 | `[x]` | OM0-OM6 pilot or full run | Matched required-cohort ablation ladder, same seeds, same target | 13 required cohorts completed with 91 OM rows under `model_stgcn/checkpoints/br_om_ladder_required_v1`; aggregate CSV and summary JSON exist | Treat completed OM ladder as first-run evidence; do not rerun unless feature definition changes |
| 18 | `[x]` | OM delta analysis | Cohort-separated macro deltas, feature-block audit, phase-stratified deltas, O1 integrity audit, paired match-cluster bootstrap, claim-boundary control | `experiment_design/BATTLE_ROYALE_OM_LADDER_RESULT_FEATURE_AUDIT_2026-07-04.md`, `experiment_design/BATTLE_ROYALE_PHASE_STRATIFIED_OM_RESULT_2026-07-04.md`, `experiment_design/BATTLE_ROYALE_O1_SURVIVAL_CAPITAL_INTEGRITY_AUDIT_2026-07-04.md`, `experiment_design/BATTLE_ROYALE_PHASE_GROUP_BOOTSTRAP_RESULT_2026-07-04.md`, `research_framing/BATTLE_ROYALE_ANALYTIC_FRAMEWORK_STATUS_2026-07-04.md` | Use OM3 as the current robust central mechanism result; add seed repeats only for small-cohort sensitivity if manuscript space allows |
| 19 | `[x]` | Phase mechanism analysis | Cut-phase stratification and phase-group paired bootstrap over the completed cohort-separated OM ladder | `experiment_design/BATTLE_ROYALE_PHASE_STRATIFIED_OM_RESULT_2026-07-04.md`, `experiment_design/BATTLE_ROYALE_PHASE_GROUP_BOOTSTRAP_RESULT_2026-07-04.md`, `phase_stratified_metrics.csv`, `phase_stratified_deltas.csv`, `bootstrap_phase_group_core_b2000/om_paired_bootstrap_phase_groups.csv` | Produce manuscript figure for phase-group deltas |
| 20 | `[x]` | Mini corpus coverage table | Confirmatory corpus coding, source grading, projection audit | `research_framing/BATTLE_ROYALE_MINI_CORPUS_COVERAGE_TABLE_2026-07-04.md` | Use as OV1 support that O0-O8 is not PUBG-only; keep full corpus and negative controls as future/appendix work |
| 21 | `[x]` | Corrected O6 spatial-affordance sensitivity | Z1-Z3 ETL/tensor mapping and 13 required-cohort OM6Z run completed; result is weak/null O6Z support in PUBG telemetry | `experiment_design/BATTLE_ROYALE_O6_Z_LADDER_SENSITIVITY_PROTOCOL_2026-07-04.md`, `experiment_design/BATTLE_ROYALE_O6_Z_LADDER_FEATURE_AVAILABILITY_AUDIT_2026-07-04.md`, `experiment_design/BATTLE_ROYALE_O6_Z_LADDER_RESULT_2026-07-05.md`, `model_stgcn/checkpoints/br_om6z_required_v1/om6z_metrics_all_cohorts.csv` | Use as a bounded negative sensitivity: PUBG is label-rich but geometry-poor for fine O6 |
| 22 | `[~]` | Caldera O6 construct-validation artifact | Geometry-rich projection, terrain/LOS/trajectory example; scope is now limited to one composite figure plus one descriptive table | `spatial_world_modeling/CALDERA_O6_CONSTRUCT_VALIDATION_ARTIFACT_SCOPE_2026-07-05.md` | Assemble existing semantic basemap, trajectory overlay, terrain-experience panel, and descriptive feature table |
| 23 | `[~]` | Paper claim assembly | Claim-boundary alignment, result-backed wording, reviewer-attack mapping; central pressure-habitat result is supported, O6Z is bounded negative evidence | `research_framing/BATTLE_ROYALE_ANALYTIC_FRAMEWORK_NARRATIVE_FOR_MANUSCRIPT_2026-07-04.md`, `experiment_design/BATTLE_ROYALE_O6_Z_LADDER_RESULT_2026-07-05.md` | Convert this narrative into manuscript section drafts and keep O6Z as limitation/observability evidence |
| 24 | `[x]` | Experiment evidence ledger | Detailed mapping from each experiment to question, method, artifact, result, supported claim, and forbidden claim | `experiment_design/BATTLE_ROYALE_EXPERIMENT_EVIDENCE_LEDGER_2026-07-05.md` | Use as source of truth for Results, Validation, Limitations, and reviewer-response drafting |

## Detailed Work Queue

### A. Ontology And Formula Foundation

| Status | Task | Progress / Finding | Method Applied | Artifact / Evidence | Next Action |
| --- | --- | --- | --- | --- | --- |
| `[x]` | Lock O0-O8 families | O0-O8 are stable enough for current prediction cycle | Ontology engineering, frozen/changeable boundary | `BATTLE_ROYALE_ONTOLOGY_LOCK_2026-07-03.md` | No family renaming during OM experiments |
| `[x]` | Define the core problem | The paper is about phase-conditioned multi-agent survival, not merely bigger feature sets for final-rank prediction | Problem formulation, contribution boundary | `BATTLE_ROYALE_PROBLEM_DEFINITION_2026-07-03.md` | Keep all experiment claims tied to this problem statement |
| `[x]` | Add feedback dynamics | O4 to O1, O8 to O2/O7, and other-agent behavior to habitat are represented | Dynamic system formalization | Ontology lock and survival formalization docs | Use feedback language in paper diagrams |
| `[x]` | Separate concept portability from measurement observability | 3-layer conceptual portability no longer confuses 2D/3D measurement availability | Two-axis schema design | Ontology and manifest docs | Apply both axes when admitting features |
| `[x]` | Define survival formula | Prediction target is next-phase elimination hazard | Phase-cut hazard modeling | `BATTLE_ROYALE_SURVIVAL_FORMALIZATION_2026-07-03.md` | Map every term to executable feature blocks |
| `[x]` | Fix paper argument chain | Prior result-level prediction and Sagong phase-aware PUBG work now lead into genre-level contraction pressure, team/relation hazard, cumulative survival, and rank evaluation | Related-work positioning, prior-work gap analysis, claim-boundary control | `BATTLE_ROYALE_PAPER_ARGUMENT_CHAIN_2026-07-03.md` | Final-verify non-Sagong aggregate/rank prediction citations |
| `[x]` | Define ontology validation protocol | Ontology evidence is staged as coverage, operationalization, mechanism sanity, prediction, cross-cohort robustness, and Caldera construct projection | Validation ladder, construct-validation boundary | `BATTLE_ROYALE_ONTOLOGY_VALIDATION_PROTOCOL_2026-07-03.md` | Use OV0-OV6 to interpret feature audits and OM results |

### B. Feature And Code Grounding

| Status | Task | Progress / Finding | Method Applied | Artifact / Evidence | Next Action |
| --- | --- | --- | --- | --- | --- |
| `[x]` | Lock registered cohorts | Atomic cohorts are map x perspective x team_size; 13 main cohorts and 41 auxiliary cohorts are fixed | Cohort stratification, confounding control | `BATTLE_ROYALE_COHORT_REGISTRY_2026-07-03.md` | Audit OM feature availability against registered main cohorts first |
| `[x]` | Lock required experiment cohort set | Required runs are 13 independently trained map / perspective / team-size cohorts | Experiment cohort gating | `BATTLE_ROYALE_REQUIRED_COHORT_SET_2026-07-03.md` | Audit OM feature availability against required cohorts |
| `[x]` | Map OM blocks to current feature presets | Data/view availability is ready, but current V3 presets are mixed-family and not a clean OM ladder | Code-grounded manifest audit, DB/view inspection | `BATTLE_ROYALE_FEATURE_AVAILABILITY_AUDIT_2026-07-03.md` | Implement clean `om0`-`om6` presets |
| `[x]` | Validate forbidden inputs | Forbidden-list sync passed after feature mapping and cohort adapter changes | Leakage-safety invariant | `tests/test_feature_contract_docs.py`, `tools/validate_genre_feature_manifest.py` | Re-run when feature manifests change |
| `[ ]` | Audit U-layer network controls | U-layer metrics may encode alive-team count unless normalized or residualized | Null-control design, edge-preserving or N-preserving baselines | Pending | Decide U-layer residualization rule before OM5 claims |
| `[ ]` | Resolve O4/O7 duplicate projections in code | Conceptual owner is fixed, but implementation names may still overlap | Feature ownership audit | Pending | Mark O4 exposure marginals with their source O7 graph/radius/window |
| `[ ]` | Decide combat betweenness status | Combat betweenness has weak shortest-path interpretation | Interpretability boundary | Pending | Keep as interpretive/sensitivity unless proximity graph version supports headline use |

### C. Experiment Execution

| Status | Task | Progress / Finding | Method Applied | Artifact / Evidence | Next Action |
| --- | --- | --- | --- | --- | --- |
| `[!]` | Gen-1/gen-2 cleanup | Naming/result boundary remains unresolved and now blocks manuscript claims that reference core_habitat null-control, rank lock-in, or preliminary advantage-edge evidence | Result registry audit | Pending | Produce a half-day decision note before launching long O7 runs |
| `[x]` | OM preset/code readiness | Clean OM0-OM6 presets, `om_ladder` experiment group, and named cohort filter mapping are implemented | Implementation readiness audit, unit tests, one-match DB smoke | `BATTLE_ROYALE_FEATURE_AVAILABILITY_AUDIT_2026-07-03.md`, `model_stgcn/v3/spec.py`, `model_stgcn/v3/experiment_matrix.py`, `model_stgcn/v3/cohorts.py` | Execute required-cohort OM ladder |
| `[x]` | OM smoke test | TPP squad, TPP duo, and FPP squad representative smoke tests passed | Minimal executable validation | `model_stgcn/checkpoints/om_ladder_erangel_tpp_squad_smoke30/matrix_metrics.csv`, `model_stgcn/checkpoints/om_ladder_erangel_tpp_duo_smoke30/matrix_metrics.csv`, `model_stgcn/checkpoints/om_ladder_taego_fpp_squad_smoke30/matrix_metrics.csv` | Execute required-cohort OM ladder |
| `[x]` | OM ladder artifact management | Required-cohort runs now have a runbook, cohort-specific output roots, append-only run index, aggregate metrics collector, and a guard against treating partial checkpoints as completed runs | Artifact governance, run reproducibility, result collation, completion-marker invariant | `BATTLE_ROYALE_OM_LADDER_RUNBOOK_2026-07-03.md`, `scripts/run_br_om_ladder_required_cohorts.sh`, `tools/collect_om_ladder_metrics.py`, `model_stgcn/v3/run_matrix.py` | Execute required-cohort OM ladder via guarded runbook |
| `[x]` | OM ladder experiment | Required-cohort OM0-OM6 run completed for 13 independently trained cohorts, producing 91 result rows | Matched ablation design, detached execution, completion-marker guard, aggregate metrics collection | `model_stgcn/checkpoints/br_om_ladder_required_v1/om_ladder_metrics_all_cohorts.csv`, `model_stgcn/checkpoints/br_om_ladder_required_v1/om_ladder_metrics_summary.json`, `experiment_design/BATTLE_ROYALE_OM_LADDER_RESULT_FEATURE_AUDIT_2026-07-04.md` | Use as first-run evidence for the ontology-guided analytic framework |
| `[x]` | Phase-specific effect analysis | Phase metrics were flattened for all 13 required cohorts, 7 OM presets, and 7 cut phases; phase-group paired bootstrap supports OM2->OM3 early hazard and early/mid rank/top-10 gains | Phase stratification, same-cohort same-phase delta analysis, paired match-cluster bootstrap | `BATTLE_ROYALE_PHASE_STRATIFIED_OM_RESULT_2026-07-04.md`, `BATTLE_ROYALE_PHASE_GROUP_BOOTSTRAP_RESULT_2026-07-04.md`, `model_stgcn/checkpoints/br_om_ladder_required_v1/phase_stratified_delta_summary.json`, `model_stgcn/checkpoints/br_om_ladder_required_v1/bootstrap_phase_group_core_b2000/om_paired_bootstrap_summary.json` | Build manuscript figure for phase-group deltas |
| `[x]` | O1 survival-capital integrity audit | O1 tensor columns are not globally degenerate and have consistent direct univariate association with next-phase hazard; weak OM1 delta is not explained by a construction failure | Cohort-separated test-split tensor audit, variance check, point-biserial/Pearson hazard association | `BATTLE_ROYALE_O1_SURVIVAL_CAPITAL_INTEGRITY_AUDIT_2026-07-04.md`, `tools/audit_o1_survival_capital.py`, `model_stgcn/checkpoints/br_om_ladder_required_v1/o1_survival_capital_integrity_summary.json` | Treat O1 as a non-headline conditional result and proceed to paired bootstrap |
| `[x]` | Robustness and sensitivity through O6Z | Phase-group paired bootstrap is complete for OM0->OM1, OM2->OM3, and OM3->OM4; corrected O6 Z-ladder ETL, tensor mapping, and OM6Z run are complete; O6Z is a bounded negative sensitivity | Paired bootstrap, source/tensor availability audit, ETL-first sensitivity, cohort-separated OM6Z training, phase-group delta analysis | `BATTLE_ROYALE_PHASE_GROUP_BOOTSTRAP_RESULT_2026-07-04.md`, `BATTLE_ROYALE_O6_Z_LADDER_SENSITIVITY_PROTOCOL_2026-07-04.md`, `BATTLE_ROYALE_O6_Z_LADDER_FEATURE_AVAILABILITY_AUDIT_2026-07-04.md`, `BATTLE_ROYALE_O6_Z_LADDER_RESULT_2026-07-05.md` | Move to O7 edge-stream/null-guard protocol before graph-relation claims |

### D. Generality And Construct Validation

| Status | Task | Progress / Finding | Method Applied | Artifact / Evidence | Next Action |
| --- | --- | --- | --- | --- | --- |
| `[x]` | Reposition corpus validation | Corpus is confirmatory, not a hard gate before prediction | Claim-boundary control | `BATTLE_ROYALE_ONTOLOGY_CORPUS_DESIGN_2026-07-02.md` | Build a small coverage table when paper path needs it |
| `[x]` | Mini corpus coverage table | 15 source-rich Battle Royale and Battle-Royale-like games/modes map to O0-O8 without requiring a new top-level family | Source-grade coding, projection audit, claim-boundary control | `research_framing/BATTLE_ROYALE_MINI_CORPUS_COVERAGE_TABLE_2026-07-04.md` | Use this as the current-paper OV1 defense; do not let extended corpus work block supervised analyses |
| `[ ]` | Negative controls | Protocol exists, table not built | Boundary validation | Pending | Select 4-5 non-BR or partial-BR controls |
| `[~]` | Caldera spatial construct role | Caldera is geometry-rich but label-poor | Observability mask, construct validation | Capability boundary docs, Caldera scripts | Produce one final O6 demonstration artifact if needed |

## Method Registry

| Method | Use In This Project |
| --- | --- |
| Ontology engineering | Defines O0-O8 as reusable Battle Royale constructs rather than ad hoc feature names |
| Phase-cut hazard formalization | Turns the ontology into a prediction target: next-phase elimination hazard |
| Observability mask | Separates concept existence from whether a dataset can measure it directly |
| Feature ownership rule | Prevents the same signal from being credited to multiple ontology families |
| Ablation ladder | Tests incremental predictive value from general to game-specific feature blocks |
| Leakage and causal-clock guard | Prevents future labels, final rank, or post-cut information from entering inputs |
| Graph/null controls | Distinguishes relation structure from trivial alive-team count or edge density |
| Construct validation | Uses Caldera to show richer O6 spatial affordance measurement even without labels |
| Confirmatory corpus coding | Defends that O0-O8 are not PUBG-only by checking official game descriptions |
| Argument-chain framing | Connects prior PUBG prediction, Sagong phase decomposition, genre-level contraction pressure, team/relation hazard, cumulative survival, final-rank evaluation, and Caldera's construct role |
| Problem formulation | Defines why the framework exists and what progress it makes beyond final-result prediction |
| Cohort stratification | Defines map / perspective / team-size atomic cohorts before any pooled or group-level claim |
| Required cohort gating | Separates independently trained required cohorts from report-only aggregates and optional auxiliary sensitivity |
| Ontology validation ladder | Separates conceptual coverage, operational availability, predictive utility, cross-cohort robustness, and Caldera construct projection |
| Artifact completion marker | Requires `train_complete.json` or an evaluation JSON before a checkpoint can be skipped as completed |
| Tensor integrity audit | Checks that surprising model deltas are not caused by degenerate, missing, or misjoined feature columns |

## Progress Log

| Date | Task | Action | Method Applied | Output | Next |
| --- | --- | --- | --- | --- | --- |
| 2026-07-03 | Execution checklist | Created the operational progress board | Documentation object model, experiment governance | `BATTLE_ROYALE_RESEARCH_EXECUTION_CHECKLIST_2026-07-03.md` | Feature availability audit for OM0-OM6 |
| 2026-07-03 | Paper argument premise | Fixed the manuscript order from prior PUBG prediction to phase hazard and cumulative-rank evaluation | Related-work positioning, claim-boundary control | `research_framing/BATTLE_ROYALE_PAPER_ARGUMENT_CHAIN_2026-07-03.md` | Verify exact prior-work citations |
| 2026-07-03 | Paper argument revision | Added the missing bridge from phase decomposition to genre formulation, restored team/relation novelty, and bounded hazard-to-rank evaluation | Prior-work gap analysis, formulation bridge, calibration/rank-evaluation boundary | `research_framing/BATTLE_ROYALE_PAPER_ARGUMENT_CHAIN_2026-07-03.md`, `model_design/BATTLE_ROYALE_SURVIVAL_FORMALIZATION_2026-07-03.md` | Proceed to feature availability audit for OM0-OM6 |
| 2026-07-03 | Problem definition | Fixed why the project matters: Battle Royale is modeled as phase-conditioned multi-agent survival rather than final-result feature expansion | Problem formulation, motivation, contribution boundary | `research_framing/BATTLE_ROYALE_PROBLEM_DEFINITION_2026-07-03.md` | Keep feature availability audit tied to this problem definition |
| 2026-07-03 | Cohort registry correction | Replaced single primary-cohort framing with map / perspective / team-size atomic cohort registry: 13 main, 41 auxiliary | Cohort stratification, confounding control, master-split intersection | `data_layer/BATTLE_ROYALE_COHORT_REGISTRY_2026-07-03.md` | Audit OM0-OM6 feature readiness by registered cohort |
| 2026-07-03 | Required cohort set lock | Fixed 13 independently trained required cohorts by map, perspective, and team size | Required cohort gating, ablation design | `experiment_design/BATTLE_ROYALE_REQUIRED_COHORT_SET_2026-07-03.md` | Audit OM0-OM6 feature readiness on required cohorts |
| 2026-07-03 | Ontology validation protocol | Defined OV0-OV6 evidence stages before predictive claims | Validation ladder, claim-boundary control, construct-validation split | `experiment_design/BATTLE_ROYALE_ONTOLOGY_VALIDATION_PROTOCOL_2026-07-03.md` | Run feature availability audit for OM0-OM6 on required cohorts |
| 2026-07-03 | Feature availability audit | Confirmed required cohorts are data-ready for feature/label/network/habitat/edge rows; found current V3 presets are not a clean OM0-OM6 ladder | Code-grounded manifest check, DB/view inspection, leakage guard validation | `experiment_design/BATTLE_ROYALE_FEATURE_AVAILABILITY_AUDIT_2026-07-03.md` | Implement clean OM presets and named cohort filtering |
| 2026-07-03 | Clean OM presets | Added explicit `om0_context` through `om6_spatial_proxy` presets and an `om_ladder` experiment group | Feature-family boundary implementation, cumulative ablation design, unit tests | `model_stgcn/v3/spec.py`, `model_stgcn/v3/experiment_matrix.py`, `tests/test_v3_factorized.py` | Implement cohort filter adapter |
| 2026-07-03 | Cohort filter adapter | Added named cohort resolution from human-readable map / perspective / team-size labels to current DB `map_name` and `game_mode` filters | Cohort stratification, execution guard, DB smoke | `model_stgcn/v3/cohorts.py`, `model_stgcn/v3/train.py`, `model_stgcn/v3/eval.py`, `model_stgcn/v3/run_matrix.py`, `tests/test_v3_cohorts.py` | Run Erangel TPP squad OM ladder smoke |
| 2026-07-03 | Erangel TPP squad OM smoke | Ran `om_ladder` with 30-match limit, 1 epoch, CPU, same split; OM0-OM6 all trained and evaluated | Minimal executable validation, named-cohort smoke | `model_stgcn/checkpoints/om_ladder_erangel_tpp_squad_smoke30/matrix_metrics.csv`, `matrix_summary.json` | Run representative non-squad and FPP smoke tests |
| 2026-07-03 | Representative non-squad and FPP OM smoke | Ran `om_ladder` with 30-match limit on Erangel TPP duo and Taego FPP squad; OM0-OM6 all trained and evaluated in both cohorts | Minimal executable validation, cohort-stratified smoke | `model_stgcn/checkpoints/om_ladder_erangel_tpp_duo_smoke30/matrix_metrics.csv`, `model_stgcn/checkpoints/om_ladder_taego_fpp_squad_smoke30/matrix_metrics.csv` | Execute required-cohort OM ladder runs |
| 2026-07-03 | OM ladder artifact management | Added required-cohort runbook, sequential cohort runner, append-only run index, and aggregate metrics collector | Artifact governance, cohort-separated run management | `experiment_design/BATTLE_ROYALE_OM_LADDER_RUNBOOK_2026-07-03.md`, `scripts/run_br_om_ladder_required_cohorts.sh`, `tools/collect_om_ladder_metrics.py` | Execute required-cohort OM ladder runs |
| 2026-07-03 | OM ladder completion guard | Found that a partial `model.pt` could be misread as a completed preset after interruption; changed `run_matrix.py` to require `train_complete.json` or an evaluation JSON before skipping, documented the artifact contract, and verified with a 30-match Erangel TPP squad smoke | Artifact completion marker, interrupted-run safety, integration smoke | `model_stgcn/v3/run_matrix.py`, `tests/test_v3_factorized.py`, `docs/experiment_design/BATTLE_ROYALE_OM_LADDER_RUNBOOK_2026-07-03.md`, `/private/tmp/pubg_om_artifact_guard_smoke30/om_ladder/om0_context/train_complete.json` | Restart required-cohort OM ladder runs with guarded resume semantics |
| 2026-07-03 | Erangel TPP squad paper-grade OM run start | Started the first required-cohort paper-grade OM0-OM6 run in detached `screen` session `pubg_om_erangel_tpp_squad`; fixed detached-shell Python drift by adding `BR_OM_PYTHON` to the runner | Required cohort gating, detached execution, environment pinning | `scripts/run_br_om_ladder_required_cohorts.sh`, `docs/experiment_design/BATTLE_ROYALE_OM_LADDER_RUNBOOK_2026-07-03.md`, `model_stgcn/checkpoints/br_om_ladder_required_v1/erangel_tpp_squad/run.log` | Monitor Erangel TPP squad artifacts to completion |
| 2026-07-03 | Remaining required cohort queue | Scheduled a detached queue that waits for Erangel TPP squad OM0-OM6 completion markers, then runs the remaining 12 required cohorts without overwriting the active Erangel run log | Completion-marker polling, sequential cohort queue, environment pinning | `scripts/run_br_om_ladder_remaining_after_erangel.sh`, `model_stgcn/checkpoints/br_om_ladder_required_v1/required_remaining_queue.log`, `screen` session `pubg_om_required_remaining_queue` | Let the queue start after Erangel TPP squad completes; monitor `run_index.tsv` and aggregate metrics |
| 2026-07-04 | Required-cohort OM ladder completion | Completed OM0-OM6 for all 13 required map / perspective / team-size cohorts and aggregated 91 result rows | Cohort-separated ablation ladder, guarded artifact completion, macro result aggregation | `model_stgcn/checkpoints/br_om_ladder_required_v1/om_ladder_metrics_all_cohorts.csv`, `model_stgcn/checkpoints/br_om_ladder_required_v1/om_ladder_metrics_summary.json`, `model_stgcn/checkpoints/br_om_ladder_required_v1/run_index.tsv` | Interpret deltas with feature-block and claim-boundary audit |
| 2026-07-04 | OM result and feature audit | Documented exact OM0-OM6 feature lists, cohort split sizes, test `N`, macro results, sequential deltas, and claim boundaries | Code-grounded feature audit, cohort-separated result interpretation | `experiment_design/BATTLE_ROYALE_OM_LADDER_RESULT_FEATURE_AUDIT_2026-07-04.md` | Use with phase-stratified result and run paired uncertainty checks |
| 2026-07-04 | Analytic-framework claim decision | Lowered the claim from "Battle Royale ontology established" to "ontology-guided Battle Royale survival analysis framework proposed and validated through its central mechanism" | Claim-boundary control, framework validation interpretation, terminology risk reduction | `research_framing/BATTLE_ROYALE_ANALYTIC_FRAMEWORK_STATUS_2026-07-04.md`, `research_framing/BATTLE_ROYALE_ONTOLOGY_LOCK_2026-07-03.md` | Use framework wording in manuscript and proceed to evidence-strengthening analyses |
| 2026-07-04 | Mini corpus coverage table | Coded 15 source-rich Battle Royale and Battle-Royale-like games/modes from official or publisher-controlled pages; found projection differences but no new O-family requirement | Confirmatory corpus coding, source grading, projection audit | `research_framing/BATTLE_ROYALE_MINI_CORPUS_COVERAGE_TABLE_2026-07-04.md` | Return to phase-stratified OM aggregation and uncertainty checks |
| 2026-07-04 | Analytic-framework manuscript narrative | Created the first-question paper narrative: pre-specified pressure-habitat testable implication, 13/13 cohort and 8/8 map support, feature-count rebuttal, secondary density-conditioned family importance, and claim boundaries | Manuscript narrative synthesis, reviewer-defense mapping, claim-boundary control | `research_framing/BATTLE_ROYALE_ANALYTIC_FRAMEWORK_NARRATIVE_FOR_MANUSCRIPT_2026-07-04.md` | Use as source for Introduction, Theory, and Validation draft sections |
| 2026-07-04 | Phase-stratified OM analysis | Flattened existing `phase_metrics` from all cohort-separated OM eval JSONs; produced 637 phase metric rows and 546 same-cohort same-phase OM delta rows | Phase stratification, within-phase mechanism check, cohort-preserving result aggregation | `experiment_design/BATTLE_ROYALE_PHASE_STRATIFIED_OM_RESULT_2026-07-04.md`, `tools/collect_om_phase_metrics.py`, `model_stgcn/checkpoints/br_om_ladder_required_v1/phase_stratified_metrics.csv`, `model_stgcn/checkpoints/br_om_ladder_required_v1/phase_stratified_deltas.csv` | Run paired bootstrap confidence intervals |
| 2026-07-04 | O1 survival-capital integrity audit | Audited the same test splits for all 13 required map / perspective / team-size cohorts; O1 has variance and expected negative hazard association, with only structurally expected solo degeneracy | Cohort-separated tensor integrity audit, univariate hazard association, claim-boundary check | `experiment_design/BATTLE_ROYALE_O1_SURVIVAL_CAPITAL_INTEGRITY_AUDIT_2026-07-04.md`, `tools/audit_o1_survival_capital.py`, `model_stgcn/checkpoints/br_om_ladder_required_v1/o1_survival_capital_integrity_summary.json` | Run paired bootstrap confidence intervals |
| 2026-07-04 | Phase-group paired bootstrap | Generated row-level prediction CSVs for all 13 required cohorts and 7 OM presets, then ran B=2000 match-cluster paired bootstrap for OM0->OM1, OM2->OM3, and OM3->OM4 over early, mid, and late phase groups | Row-level prediction export, paired match-cluster bootstrap, cohort-separated uncertainty check | `experiment_design/BATTLE_ROYALE_PHASE_GROUP_BOOTSTRAP_RESULT_2026-07-04.md`, `tools/generate_om_prediction_csvs.py`, `tools/bootstrap_om_paired_deltas.py`, `model_stgcn/checkpoints/br_om_ladder_required_v1/bootstrap_phase_group_core_b2000/om_paired_bootstrap_phase_groups.csv` | Run corrected O6 spatial-affordance sensitivity |
| 2026-07-04 | Corrected O6 Z-ladder protocol | Registered the O6 spatial-affordance sensitivity as an ETL-first experiment with Z1-Z3 gates, confirmatory early/mid cells, and old-vs-corrected OM6 comparison | Pre-run protocol, claim-boundary control, sensitivity design | `experiment_design/BATTLE_ROYALE_O6_Z_LADDER_SENSITIVITY_PROTOCOL_2026-07-04.md` | Audit Z1-Z3 feature availability |
| 2026-07-04 | O6 Z-ladder feature availability audit | Confirmed `team_snapshot_features.pos_z` exists and is non-degenerate in all 13 required cohorts, but 0/9 Z1-Z3 target features exist in the V3 phase tensor or phase feature views | Source/tensor availability audit, cohort-separated vertical-signal check | `experiment_design/BATTLE_ROYALE_O6_Z_LADDER_FEATURE_AVAILABILITY_AUDIT_2026-07-04.md`, `tools/audit_o6_z_ladder_availability.py`, `model_stgcn/checkpoints/br_om_ladder_required_v1/o6_z_ladder_availability/o6_z_ladder_availability_summary.json` | Implement O6 Z-affordance ETL and tensor mapping |
| 2026-07-05 | O6 Z-affordance ETL and tensor mapping | Implemented `team_phase_z_affordance_features`, exposed all 9 Z1-Z3 features to the V3 phase tensor, and completed required-cohort ETL with zero remaining matches | ETL-first feature construction, cohort-separated coverage audit, tensor-surface mapping | `stdb/z_affordance_feature_etl.py`, `tools/run_o6z_required_cohort_etl.py`, `tools/audit_o6_z_ladder_availability.py`, `stdb/queries.py`, `stdb/v2_view.py`, `model_stgcn/v3/spec.py` | Train corrected `om6z_spatial_affordance` on all required cohorts |
| 2026-07-05 | Corrected OM6Z required-cohort run | Trained `om6z_spatial_affordance` separately for all 13 map / perspective / team-size cohorts using locked OM ladder splits; OM6Z did not add robust signal after OM5 | Cohort-separated sensitivity run, old-vs-corrected OM6 comparison, phase-group delta analysis | `experiment_design/BATTLE_ROYALE_O6_Z_LADDER_RESULT_2026-07-05.md`, `model_stgcn/checkpoints/br_om6z_required_v1/om6z_metrics_all_cohorts.csv`, `model_stgcn/checkpoints/br_om6z_required_v1/om6z_vs_om5_om6_comparison.csv`, `model_stgcn/checkpoints/br_om6z_required_v1/om6z_phase_deltas_vs_om5_om6.csv` | Treat O6Z as bounded negative PUBG evidence and move to O7 edge-stream/null-guard protocol |
| 2026-07-05 | O7 edge-stream protocol lock | Locked advantage-edge as the headline O7 stream and defined the 3-arm A/B/C comparison, edge-shuffle null guard, U-layer exclusion, confirmatory cells, and bounded-negative interpretation | Pre-registered graph-relation protocol, null-control design, claim-boundary control | `experiment_design/BATTLE_ROYALE_O7_EDGE_STREAM_NULL_GUARD_PROTOCOL_2026-07-05.md` | Resolve gen-1/gen-2 result boundary before launching the long O7 run |
| 2026-07-05 | Caldera O6 artifact scope | Bounded Caldera to one composite construct-validation figure plus one descriptive table; LOS remains stretch only | Construct-validation scoping, observability boundary | `spatial_world_modeling/CALDERA_O6_CONSTRUCT_VALIDATION_ARTIFACT_SCOPE_2026-07-05.md` | Assemble existing semantic basemap, trajectory overlay, terrain-experience panel, and feature table |
| 2026-07-05 | Experiment evidence ledger | Recorded every completed or locked experiment as question -> method -> artifact -> result -> supported claim -> claim boundary | Evidence synthesis, manuscript-claim traceability | `experiment_design/BATTLE_ROYALE_EXPERIMENT_EVIDENCE_LEDGER_2026-07-05.md` | Use this as the primary Results/Validation outline |
| 2026-07-05 | Current progress handoff | Consolidated the supported claims, bounded claims, completed artifacts, and next critical path after OM6Z | Continuation-state recording, claim-boundary preservation, artifact index | `handoffs/NEXT_CHAT_HANDOFF_2026-07-05_BATTLE_ROYALE_FRAMEWORK.md` | Resume from gen-1/gen-2 result boundary, then O7 edge-stream/null guard execution |
| 2026-07-03 | Survival formalization | Defined Battle Royale survival as phase-cut hazard | Phase-cut hazard formalization | `model_design/BATTLE_ROYALE_SURVIVAL_FORMALIZATION_2026-07-03.md` | Map formula variables to feature presets |
| 2026-07-03 | Ontology lock | Froze O0-O8 for the current paper path | Ontology lock, claim boundary | `research_framing/BATTLE_ROYALE_ONTOLOGY_LOCK_2026-07-03.md` | Use locked families in all OM rows |

## Immediate Next Step

The next work item is:

```text
Resolve the gen-1 / gen-2 result boundary before launching long O7 runs
```

Expected output:

```text
A short decision note that shows:
- which legacy / preliminary results are reusable as manuscript evidence
- which results must be rerun under the current required-cohort contract
- whether core_habitat null-control CLEAN, rank lock-in, and preliminary
  advantage-edge evidence can be cited
- which checkpoint/script families belong to gen-1 versus gen-2
- what exact O7 run should start after the boundary is settled
```
