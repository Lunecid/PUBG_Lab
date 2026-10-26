# pubg_lab.movement.cli

Responsibility: Joint player/team movement models, datasets, metrics and owned CLI.

Dependency boundary: Shared infrastructure and spatial/data components.

| Component | Responsibility |
|---|---|
| `analyze_experiment_v2.py` | Match-clustered paired summaries, reproducible scientific plots, and viewer data. |
| `analyze_v3.py` | All planned metrics, same-match contrasts and illustrative validation figures. |
| `analyze_v4.py` | All fixed V4 comparisons; descriptive match bootstrap, frozen development cases. |
| `audit.py` | Full-bank/source integrity and sample-index audit without re-querying the DB. |
| `audit_experiment_v2.py` | Leakage audit independent of training; all outputs local, no DB mutation. |
| `audit_v3.py` | Read-only rebuilt-bank audit; never loads held-out test raw or banks. |
| `audit_v4.py` | V4 leakage/exposure audit: only train/validation banks are opened. |
| `build.py` | Materialize match banks, train-only vocab/spatial prior, and paired sample index. |
| `build_v3.py` | Rebuild fit-only priors and multi-life banks without opening locked-test raw. |
| `condition_diagnostics_v2.py` | Post-evaluation sensitivity diagnostic. NOT a retrained feature ablation. |
| `coverage_v3.py` | Count actual fixed-plan current-mode exposure without accessing test banks. |
| `diagnose_v3.py` | Locked-plan no-zone retraining comparison, sensitivity, and exact replay. |
| `diagnose_v4.py` | Diagnose frozen V3 predictions, without test access or fitting. |
| `experiment_v2.py` | Fixed-budget paired experiment. Validation is evaluated only after all training. |
| `experiment_v3.py` | Fixed-budget V3 paired experiment. Test access is explicitly forbidden. |
| `experiment_v4.py` | Frozen V4 protocol: fit-only balanced sampling, paired regression and diffusion. |
| `export_example.py` | Write two inspectable paired batches, with condition files separate from labels. |
| `extract.py` | Export complete official TPP squad cohort, readonly and resumable. |
| `finalize_v3.py` | Verify delivered development artifacts without opening locked test data. |
| `finalize_v4.py` | Verify model lineage, frozen plans, exact inference replay, and locked tests. |
| `finish_v3.py` | Wait for authorized fixed runs, then evaluate development data only. |
| `finish_v4.py` | Wait for fixed jobs then run frozen evaluations and presentation checks once. |
| `generation_diagnostics_v2.py` | Distribution and simple numerical feasibility diagnostics; no collision claims. |
| `horizons_v4.py` | Matched horizon diagnostics, retaining actual per-horizon availability masks. |
| `mobility_diagnostics_v2.py` | Post-hoc raw-event check of aerial transport in the fixed evaluation cohort. |
| `parallel_replica_v2.py` | Run one independent preplanned seed; no changes to the frozen experiment. |
| `pre_evaluation_diagnostics.py` | Dataset-only validation diagnostics; no model scores or checkpoint selection. |
| `prepare_v4.py` | Fit-only sampling metadata; evaluation plans are byte-identical V3 copies. |
| `report_experiment_v2.py` | Render the Korean experiment record from measured artifacts, never placeholders. |
| `report_v3.py` | Generate measured V3 report with explicit retrospective-test limitation. |
| `report_v4.py` | Evidence-linked Korean experiment record, including unsuccessful outcomes. |
| `residual_drift_v4.py` | Post-evaluation residual mean drift; no refitting or changed primary estimates. |
| `smoke.py` | Real paired batches: optimizer steps, generation, masks, and runtime tensor shapes. |
| `state_figures_v3.py` | Evidence-led examples of life reset and transport restoration, fit matches only. |
| `train.py` | Train paired information views using identical seeds/anchors; optional bounded runs. |
| `transport_audit_v3.py` | Pre-prediction, mode-stratified secondary evaluation; no tuning or test access. |
| `verify_model_v2.py` | Runtime shape / gradient / seed reproducibility probe on training samples only. |
| `verify_results_v2.py` | Verify saved checkpoints, paired sample coverage, and exact inference replay. |
| `visualize_v4.py` | Frozen V3 cases plus error-independent direct-transport cases; no case cherry picking. |

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.
