# Artifact notes for the SAC 2027 MLA submission

This file collects engineering and verification records that support the manuscript but are not needed to judge it.
The anonymized code release carries the same notes. Raw and processed telemetry are not redistributed.

## Cutoff and sample checks

- Prefix replay: inputs were rebuilt from telemetry truncated at the feature cutoff for 453 cutoffs from 72 matches,
  stratified by cohort and split. No rebuilt input differed from the original. With truncation disabled, 450 of the 453
  cutoffs changed. This is an empirical negative control, not proof that every possible leak would be detected.
- Rebuilding the full dataset for the patch split changed features in all 13,948 matches, because scheduled phase lengths
  are re-estimated on the training patch. Every label and risk-set row was preserved.
- Risk-set check: two of the 817,766 retained rows (both in training) lack an observed living member at the cutoff. The
  check covers retained rows only, not every excluded candidate. It does not establish that the retrospective risk set
  equals eligibility reconstructed from events available at the boundary, because future returns can change the final
  elimination time. Online risk-set validity remains unverified.
- Outcome-window filter: a boundary is excluded if match end is missing or a gap over 60 s in the pooled position stream
  overlaps the outcome window. The filter does not establish that every player's log is complete.
- Competition and centroids: centroids average sampled members with positive or missing health. Grid points without a
  living member are excluded from movement and competition distances.

## Split notes

- Five validation matches fall within an announced maintenance window. They keep their patch assignment.

## Baseline integration

- KRAFTON: public commit 99ed9d1. Our integration fixes score direction, padding, squad truncation, and the 55/57-dimensional
  inference path. Its 55 features comprise 32 definition-matched reconstructions, 22 proxies, and one zero-filled feature.
  The upstream "IBS" metric is a snapshot winner Brier score, not a next-phase Brier score.
- Sagong: paper-based reimplementation with attack and assist proxies, aggregated from its solo, half-phase task to teams
  and full phases.
- Source and integration tests: 113 baseline tests pass, including corrected assist reconstruction and stale-cache
  rejection. Two history tests cover causal masking and cutoff-preserving shuffles. Six graph tests cover edge direction,
  padding, permutation equivariance, and snapshot isolation.

## Controls and compute

- Population controls: neural controls include a mixed-population reference retrained on the same CPU device. Device
  changes are therefore not attributed to population restrictions.
- History controls: the matched-width token adds summaries of earlier tokens. Deterministic prefix shuffles never cross the
  forecast cutoff. Phase-related covariates remain, so these controls test presentation order rather than removing every
  source of temporal information.
- GATv2: official PyG 2.7.0 operator. Training graphs are nearly complete (mean density 99.4% and 99.5% in the two
  cohorts), so edge features are used directly instead of shortest-path distinctions. Each match-phase is a separate graph,
  with no pruning or temporal module. Neural references for this comparison are retrained on CPU at their selected
  settings.
- Equal trials do not equalize computation. Parameter counts and runtimes are recorded. Mixed-device and concurrent
  runtimes are not an efficiency benchmark.

## Match-offset decomposition (v16)

- `tools/sac2027_match_offset.py`: objective controls (five losses x three seeds x two cohorts). It splits within-phase AUC
  into same-match and different-match pairs and recomputes it after match-phase centring, plus the AUC of match-phase mean
  logits alone. Outputs go to `reports/sac2027_patch_v14/control_evaluation/objective_match_offset/`.
- `tools/sac2027_match_offset_structure.py`: the same decomposition for the team-axis main effect and PTR-Former minus
  GBDT in the 11 main cohorts, pooled with DerSimonian-Laird. Outputs go to
  `reports/sac2027_patch_v14/control_evaluation/structure_match_offset/`.
- Bootstrap draws reuse the objective suite's seeds, so the intervals share resampled matches with the objective table.

## Review follow-ups (v16)

- `tools/sac2027_logistic_baseline.py`: full-feature L2 logistic baseline. Inputs are phase indicators plus the reference
  GBDT's 151 inputs, standardized with training means and SDs (constant columns dropped). The penalty C is chosen from
  {0.001, 0.01, 0.1, 1, 10} by validation (patch 41.1) log loss. Predictions use the common format and go to
  `reports/sac2027_patch_v14/baseline_logistic/<cohort>/`.
- `tools/sac2027_logistic_summary.py`: main-table metrics of the logistic baseline, computed with the code behind the other
  rows. It recomputes the reference GBDT and asserts agreement with the stored values.
- `tools/sac2027_extra_evaluation.py`: paired match bootstrap and DerSimonian-Laird pooling for the logistic contrasts;
  top-3 capture (expected value under random tie-breaking, boundaries without eliminations excluded) with random and
  fewest-remaining-members references; calibration slope and calibration-in-the-large; normal and ranked strata with at
  least 20 test matches. Outputs go to `reports/sac2027_patch_v14/control_evaluation/extra_v16/`.

## Training settings (moved from the paper, v16)

- Reference GBDT: scikit-learn `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, l2_regularization=0.01,
  early_stopping=False)` (`model_stgcn/ptr/hgb.py`).
- PTR-Former: width 128, four heads, two blocks, dropout 0.1, AdamW at 1e-3 with cosine decay, at most 12 epochs, stopping on
  validation NLL; placement-ranking loss weight 0.1.
- KRAFTON backbone: width 64, eight heads/layers, batch 512, AdamW at 1e-4, stopping on validation loss; each epoch samples
  many in-match time points.
- Equal-trial tuning: 16 configurations per architecture in Erangel TPP squad and Taego TPP squad, 153 columns, BCE,
  selection by validation NLL; neural models share the search list, a 12-epoch cap, three-epoch patience, and eight-match
  batches; selected settings are rerun with seeds 42/13/2026. GATv2 uses the same 16 configurations and seeds.

## v16 review follow-ups (second round)

- `tools/sac2027_sesoi_top3.py`: top-3 capture, within-match AUC, and AUC for the arms of the primary structure decisions
  and the RQ2 design contrasts (seed 42, 11 cohorts); DerSimonian-Laird pooling; slope of top-3 capture on within-match AUC
  through the origin with a cohort-cluster bootstrap, converting the SESOI into percentage points and boundaries per extra
  captured elimination. Outputs: `reports/sac2027_patch_v14/control_evaluation/sesoi_top3/`.
- `tools/sac2027_match_offset_rq2.py`: match-offset decomposition of PTR-Former and GBDT minus phase-adjusted KRAFTON, and
  the opt-in `scaled` score of `Decomposition` (logits divided by their match-phase SD after centering) for the team axis.
  Post hoc diagnostic. Outputs: `reports/sac2027_patch_v14/control_evaluation/rq2_match_offset/`.

