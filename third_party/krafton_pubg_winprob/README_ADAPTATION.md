# Adaptation of krafton-ai/pubg-winprob for the BigComp 2027 PTR study

Status: exploratory external baseline, added after the primary analysis (study design v2.4, section 4.3).
Origin: `https://github.com/krafton-ai/pubg-winprob`, commit `99ed9d1fa5f1df0cb242bb42ed8a511e6c972896`
(archive SHA-256 `7cdfc8fade1e38538f264ab0f096de65a76420dfeab43ecb2798916522df7b30`, copied from
`reports/krafton_pubg_winprob_audit_2026-10-02/upstream_99ed9d1.zip`).  License: Apache-2.0 (`LICENSE` unchanged,
`NOTICE` added).  The adapted code is not KRAFTON's code and results from it are not KRAFTON's results.

The upstream archive contains no data, no split file and no checkpoint.  Everything below the model code (input
reconstruction from this project's `pubg.*` tables, sampling, training driver, evaluation at the project's phase
cuts) is new.

## 1. Changes to upstream files (exact diff: `ADAPTATION.patch`)

Every changed file starts with the comment `# Modified from krafton-ai/pubg-winprob (99ed9d1) - see README_ADAPTATION.md`.

| File | Change | Why (audit 2026-10-02) |
| --- | --- | --- |
| `src/models/backbone.py` | `forward(..., valid_mask=None)`: `src_key_padding_mask = ~valid_mask` for the Transformer encoder; `num_maps` constructor argument (default `NUM_MAPS`); `enable_nested_tensor=False` | Padding tokens and dead squads were attention keys. `valid_mask=None` reproduces the old behaviour |
| `src/data/dataset.py`, `src/data/dataset_v2.py` | `collate_fn(batch, max_squads=None)` pads to the largest squad count of the batch and never truncates (upstream copied `min(n, 16)` squads); `MAP_TYPES` gets `sanhok`, `karakin`, `paramo` after the four upstream maps (same indices for those) | 16-squad cap silently dropped squads; the cohorts need 7 maps |
| `src/training/trainer.py` | `_forward` passes `valid_mask = squad_alive_mask & padding_mask` to the backbone; tqdm bar disabled when stderr is not a tty | mask fix; clean batch logs |
| `src/training/metrics.py` | new `COX_LOSS_TYPES`, `survival_score(pred, loss_type)` = `-pred` for the Cox-family losses; the `compute_*` functions are unchanged and keep reading "higher = survives longer" | Cox score direction: the loss trains a hazard, the metrics and the softmax read a survival score |
| `src/training/evaluation.py` | `loss_type` argument; scores go through `survival_score`; backbone gets the key padding mask; `evaluate_test_set` masks invalid squads before `argmax` | sign, mask, arg-max over dead / padded squads |
| `src/training/calibration.py` | `collect_logits_and_targets(..., loss_type)` applies `survival_score` and the mask; callers pass the config's `loss_type` | sign, mask |
| `src/training/inference.py` | uses `PGCDatasetV2` (57-d) when the config says `use_dataset_v2` (the trainer default), default `input_dim` 57; scores through `survival_score`; mask | 57-d training input vs 55-d CSV inference (`mat1 and mat2 shapes cannot be multiplied`) |
| `src/training/inference_with_json.py` | scores through `survival_score`; dead squads masked as attention keys | sign, mask |

Not changed: `losses.py` (`weighted_cox` as published), `heads.py`, `modules.py`, `trainer.py` optimiser / early
stopping logic, `baselines.py`, `data_generation/`, `scripts/`, `notebooks/`.  Audit items that are not fixed in the vendored
files because nothing here uses them: the V1 `PGCDataset` alive mask that depends on standardised health, the metric
functions' return values for a sample without valid squads (accuracy 1, Brier 0), and the zone fallback to the first
future zone row in `extracting_pgc_features.py` (this copy never runs that script; see section 3).

Tests (`tests/test_krafton_adaptation.py`, no database): feature reconstruction against a naive per-(squad, time)
re-implementation of the upstream definitions, prefix-replay causality, padded / dead squads leave the alive squads'
embeddings unchanged (and leak without the mask, as in the audit), collate does not truncate 20 squads, the Cox sign
convention, 57-d input.

## 2. Added files (`adaptation/`, not part of upstream)

| File | Purpose |
| --- | --- |
| `schema.py` | feature list (55 + 2), item / weapon dictionaries read from the vendored upstream files, sampling schedule, parity table definitions |
| `labels.py` | compact label tables of the PTR dumps (`select_labels` of `model_stgcn.ptr.data` is called, not re-implemented) |
| `features.py` | causal reconstruction of the per-squad inputs from raw event rows (numpy, one match, all times at once) |
| `extract.py` | read-only SQL fetch, sampling, shard writer, 3 worker processes |
| `data.py` | in-memory sample sets, standardisation, padded batches (layout of upstream `collate_fn`) |
| `train.py` | training / evaluation driver: upstream `TransformerBackbone`, `get_head`, `Trainer`; Platt scaling; PTR-format files; native metrics |
| `ourfeat.py` | K2 = KRAFTON-tf-ourfeat: upstream's transformer modules on the PTR team tokens at the phase cuts (section 3b) |
| `kftable.py` | persisted KRAFTON feature table at the PTR phase cuts, `model_stgcn/dumps/bigcomp2027_kraftonfeat_v1` (section 3d) |
| `ourfeat_hazard.py` | K3a / K3: K2 with a per-team BCE objective on the next-phase hazard, K3 with a learned per-cut-phase logit bias (section 3e) |
| `hgb_kraftonfeat.py` | HGB-kraftonfeat: the PTR HGB classifier on the reconstructed KRAFTON squad features at the phase cuts (section 3c) |
| `run_all.py`, `summarize.py`, `parity.py`, `validate_causal.py`, `diagnostics.py` | matrix runner (stages main / seeds / ourfeat / hgbkf / k3), result table, parity statistics, real-data causality check, phase-offset AUC diagnostic |
| `phase_auc.py`, `k3_contrasts.py`, `k3_facts.py` | pooled / within-phase AUC tables, paired match-level bootstrap contrasts with random-effects pooling, run facts of the K3 test |
| `diag_phase_init.py` | diagnostic: K3 with zero-initialised phase offsets (section 3e), separate output root |
| `PARITY_TABLE.csv` | status, source and descriptive statistics of every feature |

## 3. How the experiment differs from the upstream recipe (not code patches)

Inputs are rebuilt from the raw `pubg.*` tables (SELECT only, read-only session) by `adaptation/features.py`; the
dumps' cohorts, splits, event-based death times, winner flags and phase boundaries are used as they are
(`model_stgcn/dumps/bigcomp2027_phase_v1`, 12 cohorts, airoyale and Vikendi already excluded).

* **Clock and causality.** Every time is on the unified clock W = `event_time - stdb.matches.started_at`
  (the clock of `team_death_s` and `phase_end_s`).  Position rows are placed on W by their own `event_time`; the
  per-match game-clock delta of `stdb/causal_cut.py` is only needed for the `elapsed_time` (G) columns, which are not
  used.  (Example of why: in match 6f6cd764 the offset W - G grows from 0.3 s at 10 s to 28 s at 1600 s.)  Features at
  time t read events with time < t (cumulative windows [0, t) / (0, t) as upstream) or <= t (snapshots as upstream).
  Verified by prefix replay on real matches (section 5).
* **Who is a valid squad.** Upstream: squads with at least one valid position.  Here: squads alive at t on the event
  clock (`team_death_s > t`, winners censored as `+inf`, exactly the dump's alive rule); dead squads are not stored and
  padding is per batch.  Members are alive while their last death time (dump rule) is later than t; revivals are not
  modelled (the database has no revive events), which is the dump's own convention.
* **Sampling.** Train: upstream's schedule (178 uniform-random points over the fixed phase schedule 0-2000 s + the 14
  anchors 599, 601, ..., 1679, 1681; `np.random.RandomState` seeded per match by crc32(match_id) xor 42).  Points before
  the first game state (no zone observed; upstream falls back to the first future zone row) and points with fewer than
  two alive squads (the loss ignores them) are dropped, so matches that end before 2000 s contribute fewer than 192
  samples (mean 40 on Karakin, 60 on Paramo, 77 on Sanhok, 95-112 elsewhere).  Val / test: the dump's phase cuts,
  t = `phase_end_s - 10 s` on W, for every cut with at least one v2 risk-set row.  Every training sample is a point in
  time with all alive squads; no val / test match is used in training.  Extraction took 6.5 min for 13,948 matches
  (3 workers, final run), so the every-30-s fallback was not needed.
* **Target.** Squad death time = the dump's event-based `team_death_s`; winners (and any survivor) are censored at
  max(2000 s, match end, latest death + 1 s), upstream's "survivors at 2000" convention.  Loss `weighted_cox`
  through `get_head` / `get_loss_fn` (their paper loss), target / 2000 as upstream.
* **Zones.** `bluezone` = `safe_zone_*` (x, y, radius), `whitezone` = `poison_zone_*`, replaced by the blue zone when it
  is (0, 0, 0), as upstream's `parse_zone_info`; the latest game state <= t is used (never a later one).  A radius below
  1 m is treated as "no zone", like upstream's `radius <= 0`: the game state ends with a degenerate final circle of
  radius 2.3 / 7.8 cm, where distance / radius reached 2.6e5 (pooled train rows: std of `dist_from_whitezone` 355 before the rule, 1.1 after; max 152 after).
* **Standardisation.** Train mean / std (ddof = 1, zero std -> 1) over the alive-squad rows of the train samples
  (upstream: all CSV rows including dead squads, which are not materialised here); the two v2 distance ratios are
  appended unscaled (57-d), as `dataset_v2.py`.  Positions / 80000, zones / 800000.
* **Training.** Paper hyperparameters: embed 64, 8 heads, 8 layers, FFN 256, dropout 0.1, batch 512, AdamW lr 1e-4,
  weight decay 1e-4, 8 Fourier bands, `weighted_cox`, early stopping on the val loss (upstream trainer already has it),
  patience 4.  Val loss = `weighted_cox` over the val phase-cut samples.  Epoch cap = max(30, ceil(5000 / steps per
  epoch)): on small cohorts (21-32 optimizer steps per epoch) 30 epochs would end before convergence; early stopping
  decides.  Seed 42 (seeds 13 and 2026 for the two large cohorts were planned and not run).  Device: Apple MPS, 3 torch
  threads (a repeated Paramo run reproduced the validation-loss history exactly; bitwise reproducibility is not guaranteed).  `--encoder mlp` = upstream's per-squad MLP ablation (8 residual-free MLP blocks, no attention).
* **Outputs at the phase cuts.** `pred_val.csv`, `pred_test.csv` (PTR columns + `raw_score`) in
  `model_stgcn/checkpoints/bigcomp2027_ptr_v1/<cohort>/KRAFTON-tf|KRAFTON-mlp/seed<seed>/`, with `metrics.json`
  (`model_stgcn.ptr.metrics.all_metrics`), `best.pt`, `history.json`.  `raw_score` is the Cox hazard score (higher =
  eliminated earlier), `rank_score = -raw_score`, `p_hazard = sigmoid(a * raw_score + b)` with (a, b) fitted by logistic
  regression on the val risk-set rows only (Platt; the temperature scaling of upstream is not used).  Platt scaling is a
  monotone map for a > 0, so AUC / ranking metrics do not depend on it.  Every run asserts that its val and test rows,
  `y`, `top5`, `final_rank_norm` and `match_type` equal those of the PTR arms (`M00`, `HGB-cut`, `M11R`,
  `metrics.align_predictions`) and stores the check in `metrics.json` (`row_assertion`).
* **KRAFTON-native metrics at the cuts** (`metrics.json -> krafton_native`): winner accuracy (arg-max of the survival
  score = -hazard vs arg-max of the death-time target among the alive squads) and the C-index of upstream `metrics.py`
  (pooled pairs, ties 1/2; `c_index_mean` = mean over samples), overall and per cut phase, with the chance level of
  the accuracy and, for reference, the accuracy that the unfixed upstream direction would report
  (`winner_accuracy_unfixed_direction`).

## 3b. K2 = KRAFTON-tf-ourfeat (input vs model)

Upstream's transformer, bug-fixed as in section 1, on the PTR team features instead of the reconstructed KRAFTON
inputs; it separates the effect of the input from the effect of the model (compare with KRAFTON-tf: same model and loss,
other input; with M01: same input, other model and loss).  `adaptation/ourfeat.py`, no further change to upstream files.

* **Token.** The dump's team token at the cut: `X` (150) + team centre (2) + `team_ai_frac` (1) = 153 values, z-scored
  with train statistics over the train alive tokens (`model_stgcn.ptr.data.Standardizer` / `PTRDataset`, read-only use;
  NaN -> 0 after scaling).  Upstream's 55 features, the per-player Fourier xyz, the zone embeddings and the map embedding
  are not used: `TokenOnlyBackbone` (subclass of the patched `TransformerBackbone`) removes them and keeps the token MLP,
  the 8-layer encoder with the key padding mask and the final LayerNorm; our features already carry the zone information.
* **Samples.** Phase cuts only, for train, val and test: one sample = (match, cut phase k) with at least one risk-set
  row; its squads are those alive at t = `phase_end_s[k-1] - 10 s` (`team_death_s > t`, winners censored) -- the same squad
  set as KRAFTON-tf at that cut (checked for every val / test sample against the KRAFTON-tf shards:
  `token_check_vs_krafton_tf` in `metrics.json`).  The target is the same censored death time (same `t_win` per match as
  KRAFTON-tf).  192-point sampling is not used (the dump's features exist only at the cuts).
* **Recipe.** Same hyperparameters, `weighted_cox`, early stopping on the val loss (patience 4), Platt scaling on val, same
  files and rows as KRAFTON-tf (`<cohort>/KRAFTON-tf-ourfeat/seed42/`), same row assertion against the PTR arms.
* **One deviation: the epoch.** ~6 cut samples per match give only 30-40 optimizer steps per pass over the train matches,
  and patience 4 epochs would stop after ~150 steps.  An epoch has the same number of optimizer steps as an epoch of KRAFTON-tf
  on the same cohort (the train cut samples are re-shuffled and repeated to fill it; `train_passes_per_epoch` in
  `metrics.json`), so patience and the epoch cap act on the same step scale in both models.

## 3c. HGB-kraftonfeat (GBDT on the reconstructed KRAFTON inputs)

`adaptation/hgb_kraftonfeat.py`: the classifier of `model_stgcn/ptr/hgb.py` (`make_classifier`: `HistGradientBoostingClassifier(max_iter=200,
learning_rate=0.06, l2_regularization=0.01, early_stopping=False, random_state=42)`) on the 55 reconstructed upstream squad
features at the phase cuts, so that the input effect can be read without a neural model.

* **Rows.** Every (match, team, cut phase) of the dump's v2 risk set with winner censoring, for train, val and test, ordered by
  (match_id, team_id, phase); label `y_hazard_v2`.  The features come from the persisted table of section 3d (rows with
  `in_risk_set = 1`; train, val and test at t = `phase_end_s - 10 s` on the unified clock).  The train rows (keys and labels) are asserted equal to `model_stgcn.ptr.hgb.build_rows(train, 'HGB-cut')`,
  the val / test rows to the PTR arms (`row_assertion`, `train_rows_check` in `metrics.json`).
* **Features.** The 55 features of `schema.FEATURES` (raw, un-standardised; trees and NaN-free).  No other squad-level aggregate
  was built.  Left out: per-player xyz, zone values, map, the PTR team features (`team_ai_frac` too) and the cut phase.
* **Output.** `<cohort>/HGB-kraftonfeat/seed42/{pred_val.csv, pred_test.csv, metrics.json}`; `p_hazard` = predicted probability,
  `rank_score = -p_hazard` (as the PTR HGB arms).

## 3d. Persisted feature table (`model_stgcn/dumps/bigcomp2027_kraftonfeat_v1`)

`adaptation/kftable.py` writes the reconstructed KRAFTON squad features at the PTR phase cuts as a reusable table
(`<cohort>/{train,val,test}.csv.gz`, gzip CSV with 9 significant digits = exact float32 values; read with
`pd.read_csv(..., float_precision='round_trip')`; pyarrow is not installed in the interpreter) and `meta.json` (column
list with upstream name, block, status exact / proxy / zero, source and note; zero-filled and approximate columns;
conventions; row counts).

* **Rows.** One row per (`match_id`, `team_id`, `phase`) for the cut phases k = 1..7 (PTR CSV convention, k = dump slot + 1)
  and for every team alive at the start of phase k (the PTR token set `node_mask_v2` = `model_stgcn.ptr.data.event_alive_mask`,
  winners always alive).  Every row of the dump's v2 risk set (winner censoring) is in it (`in_risk_set = 1`,
  `y_hazard_v2`); `alive_at_cut = 0` marks teams that are alive at phase start but dead at the cut.
* **Columns.** `cut_t` (= `phase_end_s - 10 s` on the unified clock W), `in_risk_set`, `y_hazard_v2`, `alive_at_cut`, and the 55
  continuous features as `kf_<upstream name>` (cumulative actions, location counts, zone distances, equipment levels, holdings,
  team status; no per-player xyz).  Dead-squad convention (`alive_at_cut = 0`): cumulative actions as upstream, mean snapshot
  features -1, sums / levels / counts 0.
* **Causality.** `cut_t = phase_end_s - 10 s`; every feature reads events / samples before (cumulative) or at (snapshot) `cut_t`
  only (prefix replay, section 5).
* **Checks (`kftable --verify`).** The risk-set rows of every split equal the dump's v2 risk set (count per cohort and split);
  the alive-at-cut rows of val / test equal the KRAFTON-tf cut shards (same squads, features equal up to float32 rounding).

## 3e. K3a / K3: the objective test (`adaptation/ourfeat_hazard.py`)

Two arms identical to K2 except for the objective, to separate the effect of the objective from the effect of the input and the
recipe.  No upstream file is touched (`ADAPTATION.patch` is unchanged).

* **Identical to K2:** the official bug-fixed `TransformerBackbone` (`TokenOnlyBackbone`: key padding mask, no position / zone / map
  embeddings), embed 64 / 8 heads / 8 layers / FFN 256 / dropout 0.1, batch 512, AdamW lr 1e-4 / wd 1e-4, the input (the dump's team
  token at the cut: X(150) + team centre(2) + `team_ai_frac`, train-token z-score), the cut-only training samples (the K2 builder
  itself; `same_samples_as_k2` in `metrics.json` compares matches / samples / rows of all three splits and holds in all 24 runs), the
  steps per epoch (those of a KRAFTON-tf epoch), early stopping with patience 4 and the epoch cap max(30, 5000 steps), the squads that
  are attention keys (every squad alive at the cut, risk-set or not), and the initial weights (same seed, same construction order).
* **K3a `KRAFTON-tf-ourfeat-bce`.** Per-team BCE on `y_hazard_v2` over the v2 risk-set rows (winner censoring: winners are at risk
  at every existing cut and never positive), mean over the rows of a batch; the head is upstream's scalar `HazardHead` read as a
  logit.  Early stopping on the val NLL (pooled BCE over the val risk-set rows; `Trainer.validate` is overridden so that it is a pooled
  value, not a mean of batch means).  Alive squads outside the risk set (they die in the last 10 s before the phase end) carry no loss.
* **K3 `KRAFTON-tf-ourfeat-hazard`.** K3a + a learned per-cut-phase bias on the logit, `logit = head(h) + b[k]`, `b` =
  `nn.Embedding(7, 1)` over k = 1..7 (the discrete-time hazard with a phase-specific baseline).
* **Intercept initialisation (both arms, train labels only).** The last bias of the head starts at the pooled train log-odds of the
  hazard; in K3 `b` starts at `logit(rate_k) - pooled log-odds`, i.e. the offsets of the phases start at their train base rates.
  Reason: AdamW with lr 1e-4 moves a scalar by at most ~1e-4 per step, so a zero-initialised embedding could not reach logits between
  -1.3 and +0.2 within the few thousand steps of a run.  `b` is in the optimiser, but it stays close to its start: the largest move of
  any phase offset in any cohort is 0.119 (mean over cohorts 0.031) against an initial spread of
  1.50 between the phases.
* **Zero-initialisation check** (`adaptation/diag_phase_init.py`, not part of the matrix; Paramo, Karakin, Erangel FPP squad, Taego FPP
  squad; written to the separate root `model_stgcn/checkpoints/bigcomp2027_ptr_v1_diag`): with the phase offsets initialised at 0 they
  move by at most 0.017 during training (the base-rate offsets are 0.1-1.0 in size), and the test AUC equals that of K3a (differences
  0.0000, +0.0001, -0.0001, 0.0000).  Without the base-rate initialisation K3 would therefore be K3a.
* **Platt scaling and the raw logit.** As in K1 / K2: `p_hazard = sigmoid(a * raw_score + b)` fitted on the val risk-set rows,
  `raw_score` = the raw logit, `rank_score = -raw_score`.  For the BCE logits it is close to the identity: a = 0.916
  (0.817-0.991) and b = -0.032 (-0.122 to +0.055) for K3a, a = 0.931 (0.802-1.003), b = -0.031
  for K3; test NLL of sigmoid(raw logit) -> Platt: 0.4790 -> 0.4778 (K3a), 0.4796 -> 0.4790 (K3)
  (`uncalibrated` in `metrics.json`).
* **The input already contains the phase.** Column 0 of X is `phase_idx_norm` (group `context` with `phase_duration_10min` and
  `team_size_norm`); X also holds the event-based alive-team / alive-player counts and zone radii.  "No phase information enters
  the head" therefore means no phase term at the logit: K2, K3a and K3 all see the phase index at their input.

## 4. Feature parity (55 upstream continuous features + the 2 appended v2 ratios; `PARITY_TABLE.csv`)

| status | n | features |
| --- | ---: | --- |
| exact (same definition, same event type) | 32 | n_pickup, n_drop, n_kill, n_finish, n_groggy, n_dmg, n_dmg_by_AR, n_dmg_by_SR, n_dmg_by_DMR, n_dmg_by_SMG, n_dmg_by_SHOTGUN, sum_dmg, sum_dmg_by_AR, sum_dmg_by_SR, sum_dmg_by_DMR, sum_dmg_by_SMG, sum_dmg_by_SHOTGUN, n_take_dmg, sum_take_dmg, n_heal, n_heal_bandage, n_heal_firstaid, n_heal_medkit, n_heal_painkiller, n_heal_energydrink, n_heal_adrenaline, dist_from_bluezone_v2, dist_from_whitezone_v2, item_helmet_level, item_vest_level, item_backpack_level, squad_alive_count |
| proxy (coarser / different signal) | 22 | n_vehicle_ride, n_teammates, n_enemies, dist_from_bluezone, dist_from_whitezone, is_vehicle_count, item_throwables_count, item_grenade_count, item_smoke_count, item_flashbang_count, item_molotov_count, item_bluezone_grenade_count, item_aids_count, item_bandage_count, item_firstaid_count, item_medkit_count, item_painkiller_count, item_energydrink_count, item_adrenaline_count, squad_total_health, squad_non_groggy_count, squad_out_bluezone_count |
| zero-filled (not buildable) | 1 | sum_heal_amount |

`n_drop` is built: the collector stores `LogItemDrop` in `pubg.telem_item_pickup` (`event_type = 'drop'`).  The only column that cannot be built is `sum_heal_amount` (needs `LogHeal.healAmount`).  Reasons for the proxies (details per feature in `PARITY_TABLE.csv`, which also holds mean / std / p50 / p99 / non-zero share of the train rows):

* positions only every 10 s, no event-sourced positions: `n_teammates`, `n_enemies`, `dist_from_bluezone`, `dist_from_whitezone`
* no vehicle ride / leave events: vehicle flag and rides from `telem_positions.vehicle_type`: `n_vehicle_ride`, `is_vehicle_count`
* no `LogPlayerUseThrowable` (thrown items never subtracted; counts are upper bounds): `item_throwables_count`, `item_grenade_count`, `item_smoke_count`, `item_flashbang_count`, `item_molotov_count`, `item_bluezone_grenade_count`
* no `LogHeal` (aids = picked up - dropped - used, from `telem_item_use`): `item_aids_count`, `item_bandage_count`, `item_firstaid_count`, `item_medkit_count`, `item_painkiller_count`, `item_energydrink_count`, `item_adrenaline_count`
* health / knock / zone state from position samples and game states instead of every event type: `squad_total_health`, `squad_non_groggy_count`, `squad_out_bluezone_count`

## 5. Verification

* `tests/test_krafton_adaptation.py` (16 tests, no database): see section 1; plus tests for the K2 backbone / epoch loader, the feature table rows and K3a / K3 (label attachment and risk-row masking, phase head bias and gradient, intercept initialisation and shared initial weights with K2, pooled val NLL).
* `python -m adaptation.validate_causal`: prefix replay on real matches (features at the evaluation cuts and at
  KRAFTON-style random times, computed from the full log and from the rows with event time <= t only).
  Result on 50 test matches of 5 cohorts (Karakin, Erangel TPP squad, Taego TPP duo, Erangel TPP solo, Sanhok): 898 (match, time) checks, 0 differences
  (max abs difference 0.0 in all 55 features and in the positions); 4 of 8,749 alive-squad rows had no position at all.
* Reconstructed team death times equal the dump's `team_death_s` for all 13,948 matches (0 mismatches above 5 ms;
  422 winners whose members all have finite death events keep their latest member alive, as the winner censoring of
  `model_stgcn.ptr.data` implies).
* MPS and CPU give the same loss and outputs on a random batch (1.8e-7), gradients finite.

## 6. Results (seed 42, test split, hazard AUC pooled over the cuts; `reports/krafton_baseline_2026-10-06/`)

| cohort | test rows | KRAFTON-tf | KRAFTON-mlp | KRAFTON-tf-ourfeat | HGB-kraftonfeat | M01 | HGB-cut | M11R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| erangel_tpp_squad | 20010 | 0.765 | 0.765 | 0.778 | 0.808 | 0.826 | 0.825 | 0.835 |
| taego_tpp_squad | 22848 | 0.751 | 0.769 | 0.781 | 0.825 | 0.838 | 0.836 | 0.846 |
| miramar_desert_tpp_squad | 9685 | 0.768 | 0.783 | 0.768 | 0.820 | 0.827 | 0.833 | 0.834 |
| rondo_tpp_squad | 5464 | 0.755 | 0.761 | 0.753 | 0.805 | 0.817 | 0.817 | 0.822 |
| sanhok_tpp_squad | 4880 | 0.766 | 0.777 | 0.789 | 0.811 | 0.827 | 0.827 | 0.829 |
| erangel_tpp_solo | 3112 | 0.656 | 0.627 | 0.662 | 0.712 | 0.724 | 0.726 | 0.726 |
| erangel_tpp_duo | 5912 | 0.698 | 0.667 | 0.705 | 0.734 | 0.743 | 0.747 | 0.745 |
| taego_tpp_duo | 4616 | 0.676 | 0.670 | 0.668 | 0.735 | 0.742 | 0.748 | 0.743 |
| taego_fpp_squad | 2103 | 0.742 | 0.726 | 0.729 | 0.796 | 0.797 | 0.810 | 0.795 |
| erangel_fpp_squad | 2341 | 0.742 | 0.702 | 0.739 | 0.784 | 0.801 | 0.804 | 0.789 |
| karakin_tpp_squad | 1232 | 0.734 | 0.734 | 0.740 | 0.770 | 0.790 | 0.789 | 0.781 |
| paramo_tpp_squad | 993 | 0.741 | 0.757 | 0.757 | 0.762 | 0.781 | 0.773 | 0.778 |

Mean over the 12 cohorts: KRAFTON-tf 0.733, KRAFTON-mlp 0.728, KRAFTON-tf-ourfeat 0.739, HGB-kraftonfeat 0.780, M01 0.793,
HGB-cut 0.794, M11R 0.794.  HGB-cut minus KRAFTON-tf: mean 0.062 (0.031-0.085); minus KRAFTON-tf-ourfeat: 0.055 (0.015-0.080);
minus HGB-kraftonfeat: 0.014 (0.010-0.020).  KRAFTON-tf-ourfeat minus KRAFTON-tf: +0.006 (-0.012 ... +0.030; higher on 8 of 12
cohorts); KRAFTON-mlp minus KRAFTON-tf: -0.005 (higher on 5 of 12).  Rows and labels are those of the PTR arms in every run
(`row_assertion`); no run failed.  All KRAFTON-tf and KRAFTON-tf-ourfeat runs stopped early (best epoch 6-13 and 4-13); of the KRAFTON-mlp
runs one, Taego TPP squad, ran to the 30-epoch cap (best epoch 29).

* **KRAFTON-native metrics at the cuts** (KRAFTON-tf, test, `training_and_native_metrics.csv`): winner accuracy 0.26-0.52 against a
  chance level of 0.11-0.23 (Erangel TPP squad 0.328 vs 0.148); the same score read in the unfixed upstream direction gives
  0.02-0.09; pooled C-index 0.63-0.70.
* **Phase base rate** (`phase_offset_diagnostic_*.csv`, diagnostic only, not part of the protocol): the Cox score is identified only up
  to a constant per sample, so the pooled AUC mixes the phases' base hazards (0.21-0.56).  A phase-specific intercept fitted on val raises
  the pooled AUC of KRAFTON-tf by 0.028 on average (0.001-0.050) and of KRAFTON-tf-ourfeat by 0.029 (0.005-0.047); the mean within-phase
  AUC is 0.735 (tf) and 0.743 (ourfeat) against 0.772 (HGB-cut) and 0.773 (M11R).
* **Run times** (`krafton_run_status.csv`; 48 runs, 0 failed): KRAFTON-tf 68.4 min in total (0.4-22.5 min per cohort), KRAFTON-mlp 39.4 min
  (0.4-15.2), KRAFTON-tf-ourfeat 40.2 min (0.4-12.7), HGB-kraftonfeat 1.3 min (3-17 s); MPS, 3 torch threads.  Feature extraction
  (192-point train samples + cuts, 13,948 matches, 3 workers) 6.5 min; feature table (`kftable`, 1 worker) 11 min.  Seeds 13 and 2026
  were not run.  Two restarts happened before the final runs and their outputs were deleted: the degenerate-zone rule (section 3) was
  added after the first full extraction, and a duplicated runner was stopped.


## 6b. K3 test: Cox vs BCE objective, phase bias (seed 42, test split; `k3_phase_auc.csv`, `k3_contrasts.csv`, `k3_run_facts.csv`)

Pooled test AUC (hazard AUC over all risk-set rows):

| cohort | K2 | K3a | K3 | M01 |
| --- | ---: | ---: | ---: | ---: |
| erangel_tpp_squad | 0.778 | 0.824 | 0.824 | 0.826 |
| taego_tpp_squad | 0.781 | 0.839 | 0.838 | 0.838 |
| miramar_desert_tpp_squad | 0.768 | 0.826 | 0.824 | 0.827 |
| rondo_tpp_squad | 0.753 | 0.814 | 0.813 | 0.817 |
| sanhok_tpp_squad | 0.789 | 0.825 | 0.825 | 0.827 |
| erangel_tpp_solo | 0.662 | 0.726 | 0.728 | 0.724 |
| erangel_tpp_duo | 0.705 | 0.744 | 0.743 | 0.743 |
| taego_tpp_duo | 0.668 | 0.742 | 0.740 | 0.742 |
| taego_fpp_squad | 0.729 | 0.798 | 0.795 | 0.797 |
| erangel_fpp_squad | 0.739 | 0.800 | 0.796 | 0.801 |
| karakin_tpp_squad | 0.740 | 0.779 | 0.779 | 0.790 |
| paramo_tpp_squad | 0.757 | 0.776 | 0.773 | 0.781 |
| **mean (12 cohorts)** | 0.739 | 0.791 | 0.790 | 0.793 |

Within-phase test AUC (unweighted mean over the cut phases of the per-phase AUC; phases with both classes and >= 30 rows, 6-7 per cohort):

| cohort | K2 | K3a | K3 | M01 |
| --- | ---: | ---: | ---: | ---: |
| erangel_tpp_squad | 0.754 | 0.780 | 0.780 | 0.783 |
| taego_tpp_squad | 0.779 | 0.804 | 0.803 | 0.807 |
| miramar_desert_tpp_squad | 0.754 | 0.787 | 0.785 | 0.794 |
| rondo_tpp_squad | 0.728 | 0.786 | 0.787 | 0.790 |
| sanhok_tpp_squad | 0.784 | 0.794 | 0.795 | 0.800 |
| erangel_tpp_solo | 0.679 | 0.687 | 0.701 | 0.695 |
| erangel_tpp_duo | 0.689 | 0.711 | 0.710 | 0.701 |
| taego_tpp_duo | 0.674 | 0.719 | 0.718 | 0.716 |
| taego_fpp_squad | 0.724 | 0.765 | 0.764 | 0.767 |
| erangel_fpp_squad | 0.738 | 0.759 | 0.753 | 0.764 |
| karakin_tpp_squad | 0.753 | 0.771 | 0.773 | 0.787 |
| paramo_tpp_squad | 0.749 | 0.763 | 0.763 | 0.773 |
| **mean (12 cohorts)** | 0.734 | 0.760 | 0.761 | 0.765 |

Paired differences A - B on the same test matches (random-effects pooled estimate over the 12 cohorts, 95% CI from 2,000 paired match
bootstraps per cohort, number of cohorts with a positive difference):

| A - B | pooled AUC | within-phase AUC |
| --- | --- | --- |
| K3a - K2 | +0.0509 (+0.0438, +0.0580); 12/12 | +0.0268 (+0.0199, +0.0338); 12/12 |
| K3 - K3a | -0.0005 (-0.0013, +0.0002); 3/12 | -0.0001 (-0.0013, +0.0011); 5/12 |
| K3 - K2 | +0.0497 (+0.0427, +0.0568); 12/12 | +0.0273 (+0.0204, +0.0342); 12/12 |
| M01 - K3a | +0.0010 (-0.0002, +0.0023); 8/12 | +0.0033 (+0.0010, +0.0057); 10/12 |
| M01 - K3 | +0.0016 (+0.0003, +0.0029); 11/12 | +0.0036 (+0.0012, +0.0060); 9/12 |

* Calibration in the large per phase (row-weighted mean over the phases of |mean p - observed rate|, mean over the cohorts): K2 0.074,
  K3a 0.016, K3 0.016, M01 0.018.
* Runs: 24 (12 cohorts x K3a, K3), none failed; every run stopped early (epochs 7-16, best epoch 3-12; K2: 8-17, 4-13);
  total run time K3a 38 min, K3 37 min (K2 40 min), MPS, 3 torch threads.


## 7. Reproduce

```sh
cd third_party/krafton_pubg_winprob
set -a; source <env file with PG_* variables>; set +a            # database access is read-only
python -m adaptation.labels                                       # label tables from the PTR dumps
python -m adaptation.extract --cohorts <c1,c2,...> --splits val,test,train --workers 3
python -m adaptation.run_all --stage main --threads 3             # 12 cohorts x {transformer, mlp}, seed 42
python -m adaptation.run_all --stage seeds --threads 3            # seeds 13 / 2026 on two cohorts
python -m adaptation.run_all --stage ourfeat --threads 3          # K2 = KRAFTON-tf-ourfeat, 12 cohorts
python -m adaptation.kftable --cohorts <c1,c2,...> --workers 1 --verify    # persisted feature table (needs the K1 shards for --verify)
python -m adaptation.run_all --stage hgbkf                        # HGB-kraftonfeat, 12 cohorts (reads the table)
python -m adaptation.run_all --stage k3 --threads 3               # K3a (bce) and K3 (hazard), 12 cohorts (needs the K1 and K2 runs)
python -m adaptation.phase_auc; python -m adaptation.k3_contrasts; python -m adaptation.k3_facts
python -m adaptation.summarize --out reports/krafton_baseline_2026-10-06/summary_seed42.csv
python -m pytest tests/test_krafton_adaptation.py
```


## 8. Limitations

* Reconstructed inputs, not KRAFTON's: 22 proxies and one zero column (section 4); positions every 10 s (upstream
  forward-fills a grid built from every event type); no revive events.
* The fixed 2000 s sampling schedule is dense in the late game, so the early cuts (k = 1-3, 20+ alive squads) are covered
  by few training samples per match, and short maps (Karakin, Paramo, Sanhok) give 40-77 samples per match.
* Early stopping on a small val set (one batch on Karakin / Paramo) is noisy; seeds differ only on the two large cohorts.
* Hazard probabilities come from a Platt map of a score that the loss only identifies up to a per-sample constant;
  AUC / ranking metrics are unaffected, NLL / Brier depend on this calibration.
* KRAFTON-tf-ourfeat trains on ~6 cut samples per match (1.2k-21k samples per cohort); its epoch repeats them 7-18 times to match the
  step count of a KRAFTON-tf epoch (section 3b), so it sees few distinct training samples.  HGB-kraftonfeat uses the 55 features only
  (no zone values, map, cut phase, `team_ai_frac`), so it has less phase information than HGB-cut.
* K3 / K3a: the input contains the phase index (X column 0), so K3a is not phase-blind and K3 differs from it only by the logit offset;
  the offsets start at the train base rates and move little (section 3e), so K3 tests "an explicit phase baseline on top of an input that
  already carries the phase", not a model without any phase information.  Seed 42 only.
* Exploratory; no claim that the published performance of the method is reproduced.

