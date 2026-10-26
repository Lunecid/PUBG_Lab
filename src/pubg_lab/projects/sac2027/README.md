# pubg_lab.projects.sac2027

Responsibility: SAC, BR and figure applications composing reusable components.

Dependency boundary: Domain libraries; do not become dependencies of common infrastructure.

| Component | Responsibility |
|---|---|
| `sac2027_analysis_queue.py` | Run analyses only when every prespecified producer has finished. |
| `sac2027_auc_population.py` | Exact descriptive decomposition of mixed-population AUC into four pair types. |
| `sac2027_baseline_audit.py` | Reproducible source and numerical fidelity checks against the pinned archive. |
| `sac2027_condlogit_eval.py` | Conditional-logit control and BCE-weight sweep: contrasts, closure shares, phase-minus-match differences, predictions. |
| `sac2027_count_error.py` | Post hoc checks of the RQ1 explanation on stored objective-control predictions (nothing is retrained). |
| `sac2027_did_rq2.py` | Phase-minus-match differences for the RQ2 contrasts (review item S2), on the resampled matches of the RQ2 offset analysis. |
| `sac2027_evaluate_controls.py` | Matched test contrasts of frozen controls; average seed metrics, not ensembles. |
| `sac2027_extra_evaluation.py` | Review follow-ups on the patch-split test set (stored seed-42 predictions; only the logistic baseline is new). |
| `sac2027_fair_tune.py` | Frozen equal-trial-budget search. No test predictions until all trials finish. |
| `sac2027_figure3.py` | Figure 3 of the SAC 2027 manuscript (v16): per-phase calibration, horizon alignment, and match-phase centering. |
| `sac2027_figure_concept.py` | Concept figure of the SAC 2027 manuscript: what each evaluation unit compares and what each loss can learn. |
| `sac2027_figure_eval_units.py` | Figure 2 of the SAC 2027 manuscript (v17e): AUC differences on four evaluation units. |
| `sac2027_followup_run.py` | Dependency-aware local queues for pre-specified controls (no external scheduling). |
| `sac2027_graph_run.py` | One graph architecture, frozen equal-budget tuning and an edge-shuffle control. |
| `sac2027_history_controls.py` | Causal prefix-order and summary controls. Never permute across a forecast cutoff. |
| `sac2027_history_queue.py` | Independent history conditions run concurrently after their frozen selections. |
| `sac2027_logistic_baseline.py` | Full-feature regularized logistic baseline on the patch split (phase intercepts + the reference GBDT's 151 inputs). |
| `sac2027_logistic_summary.py` | Table-4 metrics of the full-feature logistic baseline, computed with the code that produced the other rows. |
| `sac2027_mapping_factorial.py` | Explicit 2x2 objective x validation probability-mapping analysis, no rounded pooled arithmetic. |
| `sac2027_match_offset.py` | Match-offset decomposition of within-phase AUC for the matched objective controls (stored test predictions only). |
| `sac2027_match_offset_rq2.py` | Match-offset decomposition of the RQ2 within-phase gaps and a spread check for the team axis (stored seed-42 test |
| `sac2027_match_offset_structure.py` | Match-offset decomposition for the structural contrasts of the main matrix (stored seed-42 test predictions only). |
| `sac2027_native_evaluation.py` | Original-task discrimination of the new patch predictions, on matched rows. |
| `sac2027_objective_all_cohorts.py` | Objective controls in all 11 primary cohorts: per-cohort contrasts, random-effects pooling, and the recorded predictions. |
| `sac2027_objective_controls.py` | Matched objective/horizon controls; validation-only stopping and calibration. |
| `sac2027_offset_only.py` | Offset-only control: frozen next-phase Cox scores plus a match-phase term learned with cross-entropy. |
| `sac2027_offset_only_cutoff.py` | Post hoc sensitivity of the offset-only control to its information set (not part of the recorded plan). |
| `sac2027_package.py` | Package only a verified paper; rebuild the standalone source before delivery. |
| `sac2027_pair_share.py` | Share of same-match pairs among the (eliminated, surviving) pairs of within-phase AUC, per primary cohort. |
| `sac2027_paper_tables.py` | Source-linked manuscript tables; all values come from completed patch analyses. |
| `sac2027_patch_rebuild.py` | Rebuild causal features from raw, read-only telemetry under a frozen patch split. |
| `sac2027_patch_run.py` | Launch isolated patch experiments after their data-validation gate completes. |
| `sac2027_patch_validate.py` | Validate the new split, all rebuilt labels, and raw-prefix replay with train-only schedules. |
| `sac2027_plot_rq1.py` | Owned implementation component |
| `sac2027_population_controls.py` | Train-only population/AI-input controls with frozen full-population settings. |
| `sac2027_recall_norm.py` | Ceiling-aware review metrics for the SAC 2027 manuscript (stored seed-42 test predictions; nothing is retrained). |
| `sac2027_release_summary.py` | Summarize completed, immutable v14 analysis outputs for the paper handoff. |
| `sac2027_result_figures.py` | Rebuild the v14 contrast figure from only the eleven primary patch cohorts. |
| `sac2027_rq1_controls.py` | AI-adjusted RQ1 and movement-component diagnostics, estimated only on patch 40.2. |
| `sac2027_rq1_phase_equality.py` | AI-adjusted phase-band equality tests using full clustered covariance. |
| `sac2027_rq1_table.py` | RQ1 contrast table of the SAC 2027 manuscript (v20h): matched objective controls in the two largest cohorts. |
| `sac2027_sesoi_top3.py` | Review-scale meaning of the SESOI: top-3 capture differences for the primary and design contrasts (stored seed-42 test |
| `sac2027_verify_paper.py` | Verify the completed v14 source/PDF and trace its experimental inputs. |
| `sac2027_verify_results.py` | Completion and selection audit for the fixed v14 experiment plan. |

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.
