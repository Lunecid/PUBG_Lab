# BigComp 2027 full re-run manifest v2 (read-only inventory)

Date: 2026-10-06 (written about 20:00 KST). Branch `paper/bigcomp2027`, HEAD `024a317`, working tree dirty (A.0).
Scope: everything needed to re-run all results of the paper plan on the leak-free v2 dump `model_stgcn/dumps/bigcomp2027_phase_v2`.

This file is an audit only. No code, database row, dump, checkpoint or test was changed, and nothing was trained.
The only commands run were `--dry-run` planners (nothing written; importing modules may refresh `__pycache__`), `sysctl`, `vm_stat`, import checks and file reads.

State of the v2 dump when this was written:

- `model_stgcn/dumps/bigcomp2027_phase_v2` does not exist yet, and `model_stgcn/dump_phase_tier.py` is unchanged (mtime Oct 5 20:14).
- The other agent's in-progress package `model_stgcn/phase_v2/` (`source`, `phases`, `alive`, `grid`, `labels`; files still changing) was read to infer the v2 contract.
- Every statement about the v2 dump is therefore marked (inferred) and must be confirmed against the finished dump script (B.0, B.5).

Conventions:

- `[R]` needed by the paper plan (RQ blueprint 2026-10-06, SEC3/SEC4 drafts, PAPER_FOCUS v5). `[O]` supporting or robustness, cheap to skip. `[X]` not in the plan (dropped in PAPER_FOCUS or never cited).
- Status CSVs are UTC (KST = UTC + 9). "CPU-min" = sum of per-run elapsed seconds. "wall" = first start to last end.
- v1 = existing outputs under `*_v1`. v2 = to be produced.
- Interpreters: default `python` = pyenv 3.11.5 (torch 2.11.0 with MPS, sklearn 1.7.0, no xgboost, no pyarrow). `/usr/local/bin/python3` = 3.12.6 (xgboost 3.0.2, pyarrow 21.0.0, torch 2.7.1, sklearn 1.7.0), required by the Sagong stages.

---

## 0. Read this first

### 0.1 Blockers: code that must change, or a decision that must be made, before the first v2 run

| # | Where | Problem | Needed |
|---|---|---|---|
| 1 | `model_stgcn/ptr/data.py:222-253` (`event_alive_mask`, `effective_alive`), `model.py:236-262,404-420` | The v2 record (`phase_v2/assemble.py`) has `node_mask` = team causally alive at `a_k` (key mask) and `query_mask = node_mask \| target_mask`; a "comeback" team (target and not node: dead at the cut, alive at `b_k`) keeps a query token but must not be a key. The loader knows one mask: with no event marker it uses `node_mask \| target` (= `query_mask`) for keys and queries, with the marker it recomputes the v1 rule `team_death_s > phase_start_s[k]` (neither is the v2 key mask; T uses the future). The model uses one `alive` mask for keys, queries and outputs (dead tokens are zeroed, so a comeback row would get a constant hazard). | Loader passes both masks (`node_mask` for keys, `query_mask` for tokens, outputs, standardiser and shuffle); model separates key and query masks. Copies of the rule follow: `kftable.py:61-65` (token set), `adaptation/labels.py:36-52` key list, `extra.py:294-307` coverage, Sagong `alive_start` (`sagong2025/plan.py:38`). Count comeback rows per cohort first (`_stats.comeback_rows`). |
| 2 | `data.py:138-180` (`_cut_exists`, `select_labels`) | The winner override (L173-178) adds winners' rows at every "existing" cut. v2 excludes (match, cut) pairs (`no_match_end`, `position_gap`), so winners would reappear at excluded cuts. | Dump pre-applies winner censoring and the loader skips the override (flag in meta), or the exclusion is encoded in `phase_mask`. Confirm label keys (`y_hazard_v2`, `target_mask_v2`; v2 also writes the same arrays as `y_hazard`, `target_mask`) and the event marker: v2 records have no `node_mask_v2`, so unless `meta['team_death_source']` is set the loader takes the non-event path (`data.py:79-81,204-212`). |
| 3 | `third_party/krafton_pubg_winprob/adaptation/features.py:77-92`, `extract.py:162-205`, `kftable.py:126-127` | The KRAFTON chain re-derives `team_death_s` with the v1 rule (`alive` -> inf; last kill; `time_survived` + offset) and compares with the dump (5 ms). `kftable` raises on any mismatch. v2 T differs (revive evidence, members without kill events are alive, `phase_v2/alive.py`). | Update the rule (or the assertion), and the K1/K2 squad-at-cut definition (`extract.py:277-308`, `ourfeat.py:183-190`). |
| 4 | `adaptation/run_all.py:28-29,40-46`, `ourfeat.py:54-56`, `ourfeat_hazard.py:370-372`, `hgb_kraftonfeat.py:49-52`, `h0_baseline.py:72`, `h0_report.py:35`, `ptr/run_matrix.py:687`, `paper/bigcomp2027/tables/make_paper_tables.py:19` | Defaults point to v1. `run_all` forwards only `--out-dir` and `--data-dir`, so K2, K3 and HGB-kraftonfeat read the v1 dump and v1 table. `h0_baseline` and `h0_report` have no path option. `run_matrix --union` defaults to the v1 kf table. | Add pass-through options or edit the constants (list in C). |
| 5 | `ptr/aggregate.py:326-334` vs SEC4 `tab:ladder` (e), (f) | SEC4 says the architecture and structure steps use no rank loss. The aggregator step table uses M01 and M11R (with rank loss). The equal-loss steps exist only in `supplement.py` E1 (`k3_to_m01norank`, `m01norank_to_m11rnorank`). Step (c) "KR-Cox (+H0)" exists only in `adaptation/h0_report.py`. | Decide which output feeds Table III; run the `norank` and `h0` arms either way. |
| 6 | Feature semantics (dump side) | Names and scales are inherited (`phase_v2/features.py:25-31` builds its 150 columns from `stdb.queries.PHASE_NODE_FEATURE_SPECS(_V2)`), so EHM column names stay valid and `stdb/queries.py` becomes a dependency of the v2 builder. But `DURATION_FEATURES` (`features.py:46-51`: `phase_duration_10min`, `hab_required_speed_to_safe_mps`, `hab_observed_speed_mps`, `hab_migration_deficit_mps`, ...) now use scheduled durations, and `alive_players_*`, `hab_alive_teams` count causal alive at `a_k`. `ecomodel.py:94-97,1419-1452` duplicates scale constants (8160 m, 40 m/s, 20 enemies, 4 players, /100 teams); `check_units` only verifies the integer-valued columns. | Do not edit `stdb/queries.py`; re-run EHM and RQ1 (A.5); rewrite the two EHM docstring claims (B.1). |
| 7 | Provenance | Untracked pipeline modules: `ptr/ecomodel.py`, `ptr/supplement.py`, `sagong2025/hab.py`, `adaptation/h0_{baseline,fit,report}.py`, `paper/bigcomp2027/tables/`. Modified: `ptr/{aggregate,arms,run_matrix,train}.py`, `sagong2025/{__init__,__main__,fit}.py`. v1 runs were stamped with dirty trees. | Commit everything before the first run (design doc 6.6), then freeze `RESULT_MODULES` (any edit marks finished runs stale, D.6). |
| 8 | Missing producers | See E.4 (Sagong summary CSVs, 400-match no-look-ahead check, 0.79 / 0.41 correlations, audit percentages). | Write scripts after the run; no number may be copied from v1. |

### 0.2 Wall-clock (details in D.2)

- From a passing dump gate to the last table: about 4.3 h with one MPS stream (the KRAFTON chain, 3.9 h, is the critical path; v1 took 4.0 h wall for the same chain). About 3 h if the chain is split over two MPS streams (estimate). The CPU-only part (PTR, HGB, Sagong, EHM) is about 2 h and overlaps with the MPS lane.
- Add about 30 min for the gate, patches and the commit. The dump build itself is not included.

### 0.3 Recommended fresh roots (D.4)

`model_stgcn/checkpoints/bigcomp2027_ptr_v2` (+ `_lc`, `_lc_steps`), `model_stgcn/dumps/{krafton_winprob_v2, bigcomp2027_kraftonfeat_v2, bigcomp2027_sagong_v2}`, `reports/bigcomp2027_results_v2`, `reports/krafton_baseline_v2`.

---

## A. Experiment inventory

### A.0 Repository state at audit

- Tracked but modified: `model_stgcn/ptr/{aggregate,arms,run_matrix,train}.py`, `model_stgcn/sagong2025/{__init__,__main__,fit}.py`, `tests/test_ptr_{aggregate,run_matrix,train}.py`, `tests/test_sagong2025.py`.
- Untracked and required by the pipeline: `model_stgcn/ptr/ecomodel.py`, `model_stgcn/ptr/supplement.py`, `model_stgcn/sagong2025/hab.py`, `third_party/krafton_pubg_winprob/adaptation/h0_{baseline,fit,report}.py`, `paper/bigcomp2027/tables/`, `tests/test_{h0_fit,ptr_ecomodel,ptr_supplement}.py`.
- Git-ignored (exist only on this disk, never delete): `model_stgcn/dumps/`, `model_stgcn/checkpoints/*`. Disk: 292 GB free; v1 uses about 9.3 GB (dumps 6.9 GB, checkpoints 2.4 GB).
- Confirmed read-only: `python -m model_stgcn.ptr.run_matrix --dump-dir ..._phase_v1 --out-dir ..._ptr_v1 --dry-run` reports `170 runs ... already done with the same config hash: 0` (the v1 stamps are stale because `arms.py` and `train.py` changed after the runs), and the v1 status CSV was not touched.

### A.1 Stage map

| Stage | Experiment / arms | Module | Need | v1 cost (CPU-min unless noted) |
|---|---|---|---|---|
| S1 | PTR factorial, 9 arms x 12 cohorts, seed 42 (108 runs) | `ptr.run_matrix` -> `ptr.train` | R | 180.6 |
| S2 | PTR seeds 13 / 2026 on the two large cohorts (20 runs) | same | R | 108.6 |
| S3 | M11R-noProx / -noCombat / -noCopress (6 runs) | same | X | 34.4 |
| S4 | HGB-cut, HGB-lag, HGB-noHab (36 runs) | `ptr.hgb` | R (cut, noHab), O (lag) | 7.1 |
| S5 | M01-norank, M11R-norank (24 runs) | `run_matrix --norank` | R (ladder e, f) | 28.8 |
| S6 | HGB-union (12), M11R-union (16 runs) | `run_matrix --union` | R (HGB-union), O (M11R-union) | 3.2 + 45.6 |
| S7 | Stage 2 player layer M11RZ-P, M11RZ-P-tm (24 runs) | `run_matrix --stage2` | X | 87.1 |
| S8 | Learning curve, fixed epochs (102 runs) and steps-matched (90 runs) | `run_matrix --lc` | O | 122.8 + 98.9 |
| S9 | KRAFTON K1: KRAFTON-tf, KRAFTON-mlp (24 runs) | `adaptation.run_all --stage main` | R (tf), O (mlp) | MPS 68.4 + 39.4 min |
| S10 | K2 = KR-Cox, K3a = KR-Hazard, K3 = +alpha (36 runs) | `adaptation.ourfeat`, `ourfeat_hazard` | R | MPS 40.2 + 38.8 + 37.2 min |
| S11 | +H0 variants (72 runs) | `adaptation.h0_baseline` | R (-h0 of K2, KRAFTON-tf-h0), O (-slope, -h0val) | 1.3 min total |
| S12 | kf feature table, HGB-kraftonfeat (12 runs) | `adaptation.kftable`, `hgb_kraftonfeat` | R | 11 min + 1.3 min |
| S13 | Sagong re-implementation (xgb, rf, lr, hgb, hgb-hab) | `model_stgcn.sagong2025` | R (xgb, hgb, hgb-hab), O (rf, lr, sensitivities) | about 8 min total |
| S14 | EHM family (PHASE-only, EHM, EHM-int, EHM-lite), RQ1, RQ1 physical units + AME | `ptr.ecomodel` | R | about 2 min fit + report (est. 5-10 min) |
| S15 | Aggregation: P1-P5 Holm, DL pooling, tab_decomp, tab_grid, kappa, metareg, seed stability | `ptr.aggregate` | R | at most 7 min |
| S16 | Supplement E1 (norank), E2 (within match-phase AUC), E3 (REML + HKSJ) | `ptr.supplement` | R | at most 8 min |
| S17 | Reports: learning curve, h0_report, k3_contrasts, phase_auc, k3_facts, summarize, derived, interpretability | various | mixed (A.6) | minutes |
| S18 | Paper tables | `paper/bigcomp2027/tables/make_paper_tables.py` | R | under 1 min |
| S19 | K3 zero-init diagnostic (4 cohorts) | `adaptation.diag_phase_init` | O | few min |

### A.2 PTR / HGB matrix (`model_stgcn.ptr.run_matrix`)

Common facts (S1-S8):

- Every run is a subprocess `python -m model_stgcn.ptr.train|hgb ...` with `--labels v2 --match-types official,competitive --threads N` (`run_matrix.py:327-347`), log `<out>/logs/<cohort>__<arm>__seed<s>.log`, status CSV `<out>/<status-name>`.
- Inputs: `<dump>/<cohort>/{train,val,test}.pt` only (plus `population.json` and `dump_manifest.json` for the planner and aggregator). No DB. No module here computes phase bounds or cuts itself; they come from the dump keys listed in B.1.
- Outputs per run: `<out>/<cohort>/<arm>/seed<s>/{metrics.json, pred_val.csv, pred_test.csv, best.pt}` (HGB: no `best.pt`). Subsampled runs: `sub<N>_ss<s>_seed<s>/`.
- Prior CLI (reconstructed: the status CSV shows max concurrency 7 and relative log paths; flags per the module docstring):

```
python -m model_stgcn.ptr.run_matrix --dump-dir model_stgcn/dumps/bigcomp2027_phase_v1 \
    --out-dir model_stgcn/checkpoints/bigcomp2027_ptr_v1 --workers 7 --threads 2        # device cpu, max-epochs 12
python -m model_stgcn.ptr.run_matrix ... --stage2 --status-name run_matrix_status_stage2.csv --workers 3
python -m model_stgcn.ptr.run_matrix ... --union  --status-name run_matrix_status_union.csv  --workers 3
python -m model_stgcn.ptr.run_matrix ... --norank --status-name run_matrix_status_norank.csv --workers 3
python -m model_stgcn.ptr.run_matrix ... --out-dir ..._ptr_v1_lc       --lc --workers 3
python -m model_stgcn.ptr.run_matrix ... --out-dir ..._ptr_v1_lc_steps --lc --lc-steps-matched --workers 3
```

| Stage | Plan | Prior runtime and parallelism (v1) | Paper use |
|---|---|---|---|
| S1 main | 9 `MAIN_ARMS` x 12 cohorts: M00, M10, M01, M11, M11R, M11R-shuf, M11RZ, M11R-noHab, M11RZ-noHab | Per arm over 12 cohorts: M00 14.5, M10 20.0, M01 14.0, M11 20.4, M11R 22.3, M11R-shuf 22.6, M11R-noHab 21.2, M11RZ 22.3, M11RZ-noHab 23.3. Largest run 7.1 min (12 epochs, 24.6 s/epoch at 2 threads, no early stop). Whole 170-run matrix: wall 47 min, 7 workers x 2 threads, CPU, 2026-10-05 12:49-13:36 UTC (sum 5.51 h) | R: P1 (phase main effect from M10, M00, M11, M01), P2/P3 (team main effect), P4 (M11R vs M11R-shuf), P5 (M11RZ-noHab vs M11R-noHab), kappa, Table III steps (f), RQ3 3c |
| S2 seeds | M00, M11, M11R, M11R-shuf, M11RZ x {erangel_tpp_squad, taego_tpp_squad} x {13, 2026} | 20 runs, 108.6 CPU-min, same matrix | R: SEC4 L166-167, `seed_stability.csv` (never pooled) |
| S3 exploratory | 3 relation-removal arms x 2 large cohorts, seed 42 | 6 runs, 34.4 CPU-min | X (not in blueprint; PAPER_FOCUS "delete") |
| S4 HGB | HGB-cut, HGB-lag, HGB-noHab x 12 | 36 runs: cut 1.9, lag 3.6, noHab 1.6 CPU-min (largest run 0.8 min); scheduled last by the planner (lightest) | R: HGB-cut (row reference of every external arm, EHM denominator, ladder a, b, g), HGB-noHab (`sagong2025 habcols` asserts against it); O: HGB-lag (formula `hgb_lag_cut`) |
| S5 norank | `--norank`: M01 and M11R trained with BCE only (`rank_weight` 0) x 12 | 24 runs: M01-norank 11.3, M11R-norank 17.5 CPU-min; wall 9.6 min, 3 workers (2026-10-06 09:59-10:09 UTC) | R: SEC4 L118 (architecture and structure steps without rank loss), supplement E1 |
| S6 union | `--union --extra-features <kf table>`: HGB-union x 12, M11R-union x 12 + seeds 13/2026 on 2 large | HGB-union 3.2, M11R-union 45.6 CPU-min; wall 16 min, 3 workers (2026-10-05 17:10-17:27 UTC). Needs the kf tables (S12) | R: HGB-union (ladder g, "extended representation"); O: M11R-union (`tab_grid` only) |
| S7 stage 2 | M11RZ-P, M11RZ-P-tm x 12, needs `bigcomp2027_player_v1` | 24 runs, 87.1 CPU-min, wall 29 min, 3 workers | X (PAPER_FOCUS "delete"; player dump not rebuilt for v2) |
| S8 learning curve | Two large cohorts, train sizes 300/600/1200, seeds 13/42/2026, models HGB-cut, M00, M10, M01, M11R; fixed 12 epochs (+12 full-size extra seeds) and steps-matched (`ceil(12 n_full / N)` epochs, cap 150) | Fixed: 102 runs, 122.8 CPU-min, wall 41 min, 3 workers. Steps: 90 runs, 98.9 CPU-min, wall 33 min | O: optional Fig. 3 (RQ3 3c "data scale"); the meta-regression (`metareg.csv`) does not use these runs |

Notes:

- Planner order: heaviest first (`run_matrix.py:578`), so HGB runs finish last. The KRAFTON and Sagong lanes need HGB-cut early (D.1), so run `--only group=hgb` first.
- Built-in cost model for the 12-cohort registry (`--projected`, nothing read): main matrix 170 runs, about 0.6 h wall at 7 workers (upper bound; actual 0.79 h), 4.0 h summed; LC 102 runs about 0.2 h at 7 workers.

### A.3 KRAFTON adapted arms (`third_party/krafton_pubg_winprob`, cwd required for `import src`)

Prior CLI (README_ADAPTATION.md section 7, plus the h0 module docstring; environment `set -a; source /Users/lunecid/project/pubg_collector/.env; set +a`, never printed):

```
python -m adaptation.labels                                            # label tables from the PTR dump
python -m adaptation.extract --cohorts <12 names> --splits val,test,train --workers 3
python -m adaptation.run_all --stage main --threads 3                  # K1: 12 cohorts x {transformer, mlp}, seed 42, MPS
python -m adaptation.run_all --stage ourfeat --threads 3               # K2
python -m adaptation.kftable --cohorts <12 names> --workers 1 --verify
python -m adaptation.run_all --stage hgbkf                             # HGB-kraftonfeat
python -m adaptation.run_all --stage k3 --threads 3                    # K3a (bce) and K3 (hazard)
python -m adaptation.h0_baseline                                       # --cohorts all, --device cpu, --threads 3
python -m adaptation.phase_auc; python -m adaptation.k3_contrasts; python -m adaptation.k3_facts
python -m adaptation.h0_report; python -m adaptation.summarize --out reports/krafton_baseline_2026-10-06/summary_seed42.csv
```

| Step | Reads | Writes | Prior runtime | Paper use |
|---|---|---|---|---|
| `labels` | dump `.pt` (keys in B.3) | `model_stgcn/dumps/krafton_winprob_v1/<cohort>/labels_{train,val,test}.pkl` | about 1 min (2.1 s for the largest split) | R (prerequisite) |
| `extract` | DB read-only: `stdb.matches`, `pubg.telem_match_end`, `participants`, `rosters`, `telem_kills`, `telem_groggy`, `telem_damage`, `telem_positions`, `telem_game_states`, `telem_item_pickup`, `telem_item_equip`, `telem_item_use`; labels pickles | shards `<data>/<cohort>/shards/<split>/*.npz`, `<data>/logs/extract_<t>.jsonl` (per-match QA incl. `t_win`; the directory `logs/` must exist) | 391 s, 3 workers, 1,763 jobs, 13,948 matches, 0 failed | R (K1 samples; K2/K3 read `t_win` and steps from the logs) |
| K1 `KRAFTON-tf`, `KRAFTON-mlp` (`adaptation.train`) | shards, labels, PTR prediction files of M00 / HGB-cut / M11R (row assertion, `train.py:150-160`) | `<ptr_out>/<cohort>/KRAFTON-tf/seed42/...` and `.../KRAFTON-mlp/seed42/...` (`metrics.json`, `pred_*.csv`, `best.pt`, `history.json`), `krafton_run_status.csv` (appended), logs | tf 68.4 min, mlp 39.4 min, MPS, 3 torch threads, one process at a time (final runs 14:34-16:21 UTC; two earlier restarts were deleted, README_ADAPTATION section 6) | R: tf (RQ2: M11R - KRAFTON-tf, Table III first rows). O: mlp (design 4.3 only; `krafton_tf_mlp`) |
| K2 `KRAFTON-tf-ourfeat` (KR-Cox) | dump (tokens), labels, extract logs, K1 shards (token check), K1 `metrics.json` (`steps_per_epoch`, else the extract log), PTR reference arms | `<ptr_out>/<cohort>/KRAFTON-tf-ourfeat/seed42/...` | 40.2 min | R: ladder (c) start, grid, `k3a_vs_k2` |
| K3a `-ourfeat-bce` (KR-Hazard), K3 `-ourfeat-hazard` (+alpha) | same as K2 and K2 `metrics.json` (`same_as_k2`, optional) | `.../KRAFTON-tf-ourfeat-bce/seed42/...` and `.../KRAFTON-tf-ourfeat-hazard/seed42/...` | 38.8 + 37.2 min | R: ladder (c), (d), (e) |
| `kftable` | DB (same tables as `extract`), labels pickles, K1 shards for `--verify` | `model_stgcn/dumps/bigcomp2027_kraftonfeat_v1/<cohort>/{train,val,test}.csv.gz`, `meta.json`, `counts_partial.json`, `logs/kftable_build.log` | about 11 min, 1 worker | R: feeds HGB-kraftonfeat and the union arms |
| `hgbkf` `HGB-kraftonfeat` | kf tables, labels, dump (asserts train rows equal `ptr.hgb.build_rows(train, 'HGB-cut')`, `hgb_kraftonfeat.py:100-105`) | `.../HGB-kraftonfeat/seed42/...` | 1.3 min total (3-17 s per cohort), 1 thread | R: ladder (b), grid |
| `h0_baseline` | K1 and K2 `best.pt` + `pred_*.csv`, K2 sample builder, K1 val/test shards; no DB | `.../KRAFTON-tf-ourfeat-h0`, `-h0-slope`, `-h0val`, `-h0val-slope`, `KRAFTON-tf-h0`, `-h0-slope` (+ `train_scores.csv.gz`) | 19:01:35-19:02:51 KST (72 arm dirs, 3 CPU threads) | R: `KRAFTON-tf-ourfeat-h0` = KR-Cox+H0 (ladder c), `KRAFTON-tf-h0` = KRAFTON+H0 (K1 fit on val, documented deviation). O: -slope, -h0val |
| `phase_auc`, `k3_contrasts`, `k3_facts`, `h0_report`, `summarize`, `diagnostics` | PTR prediction files | `reports/krafton_baseline_2026-10-06/{k3_phase_auc*.csv, k3_contrasts.csv, k3_run_facts.csv, h0_arm_metrics.csv, h0_contrasts.csv, h0_fit_summary.csv, h0_report.txt, summary_seed42.csv, phase_offset_diagnostic_*.csv}` | minutes (bootstrap 2,000) | R: `h0_report` (ladder c), `k3_contrasts` duplicates aggregator K3 contrasts; O: others |
| `diag_phase_init` | K3 code, K1/K2 data | `model_stgcn/checkpoints/bigcomp2027_ptr_v1_diag/<4 cohorts>/KRAFTON-tf-ourfeat-hazard` | few min | O: README 3e zero-init check |
| K1 seeds 13 / 2026 (`run_all --stage seeds`) | | | planned, never run | X |

### A.4 Sagong re-implementation (`python -m model_stgcn.sagong2025`, interpreter `/usr/local/bin/python3`)

Prior CLI (module docstring, sensitivity text in `BIGCOMP2027_BASELINES_AND_CONTRAST_2026-10-06.md`:199, reconstructed from file times):

```
python3 -m model_stgcn.sagong2025 plan    --dump-dir D --plan-dir model_stgcn/dumps/bigcomp2027_sagong_v1 --runs-dir R
python3 -m model_stgcn.sagong2025 extract --dump-dir D --plan-dir ... --workers 2           # DB, read-only
python3 -m model_stgcn.sagong2025 fit     --dump-dir D --plan-dir ... --runs-dir R          # --models xgb,rf,lr,hgb
python3 -m model_stgcn.sagong2025 habcols --dump-dir D --plan-dir ... --runs-dir R
python3 -m model_stgcn.sagong2025 fit     ... --models hgb-hab
python3 -m model_stgcn.sagong2025 fit --scope per_phase --models xgb --out-dir model_stgcn/checkpoints/bigcomp2027_sagong_per_phase_v1
python3 -m model_stgcn.sagong2025 fit --sr-mode label   --models xgb --out-dir model_stgcn/checkpoints/bigcomp2027_sagong_sr_label_v1
```

- Inputs: `plan` and `habcols` read the dump (`phase_start_s`, `phase_end_s`, `phase_mask`, labels, `meta.feature_groups`), no DB. `extract` reads `stdb.matches.started_at`, `pubg.participants`, `rosters`, `telem_positions`, `telem_damage`, `telem_kills`, `telem_item_pickup` and the windows `[phase_start_s, phase_end_s - 10]` from `windows.parquet` (cut subtraction at `features.py:186`). `fit` needs `units.parquet`, `features.parquet`, `hab44.parquet` and, in `--runs-dir`, the predictions and `metrics.json` of HGB-cut (required, `fit.py:64-65,386-406`), M11R (checked if present), HGB-noHab (`habcols` requires it).
- Outputs: cache `model_stgcn/dumps/bigcomp2027_sagong_v1/<cohort>/{units,windows,features,qc,hab44}.parquet`, `plan_audit.json`, `hab44_audit.json`, `event_types.json`; arms `<runs>/<cohort>/SAGONG-{xgb,rf,lr,hgb,hgb-hab}/seed42/...`.
- Prior runtime: `plan` about 0.5 min (2026-10-05 23:08:00-23:08:28 KST), `extract` 4.3 min (23:08:46-23:13:03, 2 workers, enforced cap), `habcols` 16 s (2026-10-06 14:05), `fit` sums: xgb 2.6 s, rf 49.5 s, lr 2.0 s, hgb 6.3 s, hgb-hab 15.9 s.
- Paper use: R: `SAGONG-xgb` (RQ2: M11R - SAGONG-xgb), `SAGONG-hgb` (ladder a: same HGB, Sagong inputs), `SAGONG-hgb-hab` (RQ3 3b: habitat share 0.053 of 0.091, via the summary CSV of E.4). O: `-rf`, `-lr` (`tab_external`), the two sensitivity runs (`BASELINES` A.3, XGBoost only).

### A.5 Ecological hazard model (`model_stgcn.ptr.ecomodel`)

Prior CLI (module docstring):

```
python -m model_stgcn.ptr.ecomodel --dump-dir D --runs-dir R --out-dir reports/bigcomp2027_results/ecological_model            # stage all = fit + report
python -m model_stgcn.ptr.ecomodel --dump-dir D --runs-dir R --out-dir ... --stage rq1                                         # -> <out>/rq1/
python -m model_stgcn.ptr.ecomodel --dump-dir D --runs-dir R --out-dir ... --stage rq1 --rq1-units physical                  # -> <out>/rq1_physical/ (E4: units + AME)
```

- Inputs: `fit` and `rq1`: the dump only (`hgb.build_rows(dump, 'HGB-cut', 'v2')`; columns `hab_signed_boundary_distance_end_m`, `hab_required_speed_to_safe_mps`, `hab_migration_deficit_mps`, `hab_local_enemy_count_200m`, `hab_alive_teams`, `alive_players_end_norm`). `report` also reads prediction files of HGB-cut, M11R, KRAFTON-tf-ourfeat-bce, KRAFTON-tf, SAGONG-hgb from `--runs-dir` (missing arms are skipped). `rq1` reads the `EHM-lite` / `EHM` prediction files that `--stage fit` wrote into `--runs-dir` (so `fit` must come first).
- Outputs: arms `PHASE-only`, `EHM`, `EHM-int`, `EHM-lite` under `<runs>/<cohort>/<arm>/seed42/`; tables `auc_comparison.csv`, `share_of_discrimination.csv`, `coefficients_{per_cohort,pooled}.csv`, `collinearity.csv`, `vif.csv`, `single_term_*.csv`, `summary.json`, `rq1/*`, `rq1_physical/*`.
- Prior runtime: fit sums 21 / 25 / 29 / 20 s for the four arms over 12 cohorts (about 2 min); report with 2,000 resamples and the two `rq1` stages not logged (estimate 5-10 min); latest outputs 14:32 (report), 17:22 (rq1), 19:14 KST (physical).
- Paper use: R, RQ1 (claim 1b): EHM-lite 79.4% of HGB-cut discrimination, EHM 80.3%, EHM-int, coefficient forest (Fig. 2), physical units and AME for the text.

### A.6 Aggregation and downstream reports

| Step | CLI used before (reconstructed unless quoted) | Reads | Writes | Runtime | Paper use |
|---|---|---|---|---|---|
| `ptr.aggregate` | `python -m model_stgcn.ptr.aggregate --runs-dir model_stgcn/checkpoints/bigcomp2027_ptr_v1 --out-dir reports/bigcomp2027_results --dump-dir D` (`summary.json`: 342 runs, `n_boot` 2000, seed 0, generated 2026-10-05 18:42:47 UTC) | every `<runs>/<cohort>/<arm>/seed<s>/{pred_test.csv, metrics.json}` (unknown arm names accepted; one label choice and identical rows enforced, `aggregate.py:484-560`) | `contrasts_per_cohort.csv`, `contrasts_pooled.csv` (P1-P5 Holm, DL), `arm_metrics_*.csv`, `kappa.csv`, `metareg*.csv`, `seed_stability.csv`, `run_inventory.csv`, `issues.csv`, `summary.json`, `tables/tab_{primary,primary_strata,arms_metrics,exploratory,external,external_contrasts,grid,decomp}.tex`, `figures/forest_primary.*`, `figures/metareg_bubble.*` | at most 7 min (last K3 run ended 18:35:37 UTC) | R: P1-P5, Table III (`tab_decomp`: 7 steps, `aggregate.py:326-334`), `tab_grid`, kappa, meta-regression, external contrasts |
| `ptr.supplement` | `python -m model_stgcn.ptr.supplement --runs-dir model_stgcn/checkpoints/bigcomp2027_ptr_v1 --results-dir reports/bigcomp2027_results --out-dir reports/bigcomp2027_results/supplement` (docstring) | prediction files (incl. norank arms) and `contrasts_per_cohort.csv` | `supplement/e1_*`, `e2_*`, `e3_pooling_sensitivity.csv`, `supplement_summary.json` | at most 8 min (norank finished 19:09 KST, outputs 19:16 KST) | R: E1 (ladder e, f at equal loss), E2 within match-phase AUC (SEC4 evaluation protocol), E3 REML + HKSJ (SEC4 L159) |
| `ptr.learning_curve` | `python -m model_stgcn.ptr.learning_curve --lc-dir ..._lc --main-dir ..._ptr_v1 --steps-dir ..._lc_steps --out-dir reports/bigcomp2027_results` (docstring) | LC and main `metrics.json` | `learning_curve.csv`, `learning_curve_summary.csv`, `figures/learning_curve*.{pdf,png}` | about 1 min | O |
| `ptr.derived` (phase strata, team size, calibration) | docstring | predictions, `contrasts_per_cohort.csv` | `contrasts_by_phase.csv`, ... (no output found under `reports/`) | | X |
| `interpretability/interp.py a1 / a23 / make_figures.py` | README: `python interp.py a1 && python interp.py a23 && python make_figures.py` (pyenv, 2 threads) | M11R, M11, M11RZ, M11R-shuf `best.pt` + dump (constants `interp.py:45-47`) | `reports/bigcomp2027_results/interpretability/*` | not logged | X (PAPER_FOCUS keeps one sentence; absent from the blueprint) |
| `make_paper_tables.py` | `python paper/bigcomp2027/tables/make_paper_tables.py` | `reports/bigcomp2027_results/{contrasts_pooled,arm_metrics_mean,metareg}.csv`, `tables/*.tex`, `figures/*.pdf` (constant `RES`, line 19) | `paper/bigcomp2027/tables/tab_*.tex` | under 1 min | R |

### A.7 Arm directory inventory (42 directories in each v1 cohort folder)

| Arm | Stage | Need | Used for |
|---|---|---|---|
| M00, M10, M01, M11, M11R, M11R-shuf, M11R-noHab, M11RZ-noHab | S1 | R | P1-P5, kappa, Table III (f) |
| M11RZ | S1, S2 | R | `m11rz_m11r`, seeds |
| M11R-noProx, M11R-noCombat, M11R-noCopress | S3 | X | |
| M11RZ-P, M11RZ-P-tm | S7 | X | |
| M01-norank, M11R-norank | S5 | R | ladder (e), (f), E1 |
| M11R-union | S6 | O | `tab_grid` |
| HGB-cut, HGB-noHab | S4 | R | rows reference, EHM denominator, ladder a, b, g |
| HGB-lag | S4 | O | `hgb_lag_cut` |
| HGB-union | S6 | R | ladder (g) |
| HGB-kraftonfeat | S12 | R | ladder (b), grid |
| KRAFTON-tf | S9 | R | RQ2 |
| KRAFTON-mlp | S9 | O | design 4.3 |
| KRAFTON-tf-ourfeat (K2), -bce (K3a), -hazard (K3) | S10 | R | ladder (c), (d), (e) |
| KRAFTON-tf-ourfeat-h0, KRAFTON-tf-h0 | S11 | R | ladder (c) with +H0 |
| KRAFTON-tf-ourfeat-h0-slope, KRAFTON-tf-h0-slope, KRAFTON-tf-ourfeat-h0val, KRAFTON-tf-ourfeat-h0val-slope | S11 | O | sensitivities |
| SAGONG-xgb, SAGONG-hgb, SAGONG-hgb-hab | S13 | R | RQ2, ladder (a), RQ3 3b |
| SAGONG-rf, SAGONG-lr | S13 | O | `tab_external` |
| PHASE-only, EHM, EHM-int, EHM-lite | S14 | R | RQ1 |

### A.8 What can be skipped to save time

Dropping every `[X]` item and the heavier `[O]` items saves about 4.4 CPU-hours (stage 2 1.45 h, exploratory 0.57 h, LC steps 1.65 h, M11R-union 0.76 h) and 39 MPS-minutes (K1-mlp, via `--encoders transformer`); the fixed-epoch learning curve (2.05 CPU-h) is a further optional saving. The paper-critical set is 164 main/seeds/hgb runs + 24 norank + 12 HGB-union (+ 12 M11R-union if the grid cell is kept).

---

## B. Phase-boundary, cut, duration, alive-status and 10-s-grid consumers outside the dump

### B.0 v2 contract assumed (inferred from `model_stgcn/phase_v2/*`, read 19:47-20:05 KST; the files were still changing)

1. `b_k` = W-clock time of the first LogPhaseChange with `ceil(isGame) = k + 1`; `a_k = b_k - 10 s`; cut k exists iff `b_k` exists; cuts 1..7 on an 8-slot axis (`phases.py`, `labels.py: MAX_CUT = 7, N_PHASES = 8`).
2. Risk set `T > b_k`, label `b_k < T <= b_{k+1}` (terminal: `(b_k, C_m]`); winners and teams alive at the end are censored at `C_m` (`labels.py`).
3. A (match, cut) pair is excluded when the log is truncated: no LogMatchEnd, or a position gap above 60 s overlapping the window (`labels.py`).
4. T is the final elimination time: last kill of the last-dying member unless a later healthy position sample shows a revive; members without kills are alive; leavers die at `time_survived` + median offset (`alive.py`). Causal alive status at t uses kills and healthy positions up to t only.
5. The 10-s team grid lives on W, a grid point is usable for a cut only if `t + 5 < a_k`, and there is no grid cap (`grid.py`).
6. Phase durations come from a per-match schedule (`schedule_from_options`, else the cohort median) (`source.py`).
7. The builder is DB-free and re-derives inputs from raw `pubg.*` tables, so the `stdb.*` derived tables and `stdb.v_match_phase_bounds` are probably no longer inputs of the v2 dump (to confirm). It still imports `stdb.queries` for the 150 feature specs.
8. Record schema (`phase_v2/assemble.py`, 19:57; the dump script and `meta` do not exist yet): the v1 keys plus `query_mask`.
   - `node_mask` = causal alive at `a_k` (token and key mask); `query_mask = node_mask | target_mask`; "comeback" rows = target and not node (`_stats.comeback_rows`, by cut).
   - `target_mask` / `target_mask_v2` and `y_hazard` / `y_hazard_v2` are the same v2 arrays (the pipeline labels are not recomputed); there is no `node_mask_v2`.
   - `phase_mask[k-1]` = phase k exists, `phase_start_s[k-1]` = `s_k`, `phase_end_s[k-1]` = `b_k` (NaN for the final phase), `cut_t = a_k`, `last_phase_idx`, `clock_delta_s` = 0.
   - `team_death_s` = T with winners +inf; winners are in the risk set at every non-excluded cut; an excluded cut has an all-False risk row and is listed in `_stats.excluded` / `excluded_rows`.

### B.1 PTR loader and its consumers (`model_stgcn/ptr`)

| file:line | Reads or computes | Follows v2? | Change |
|---|---|---|---|
| `data.py:62-75,99-102` | Label keys `y_hazard_v2`, `target_mask_v2`; required keys `_MATCH_KEYS` | Yes if the names are kept | Confirm names. A rename must also touch `extra.py:46`, `kftable` column `y_hazard_v2`, `adaptation/labels.py:36`, `sagong2025/fit.py:360` (hard-coded) |
| `data.py:79-81,204-219` | Event-dump marker (`meta['team_death_source'] == 'last_kill_event'` or `node_mask_v2` in the first match) | No (v2 records have no `node_mask_v2`) | Without a marker the loader uses `node_mask` raised over the targets (= `query_mask`) as key and query mask; with one it recomputes the v1 rule. Replace both by an explicit key mask / query mask pair (blocker 1) |
| `data.py:138-147` `_cut_exists` | Cut exists iff `phase_mask[p]` and `phase_mask[p+1]` | Conditional | Must equal v2 "cut k exists iff b_k exists" and the exclusions (blocker 2) |
| `data.py:150-180` `select_labels` | Winner censoring: rows added for winners at every existing cut (L173-178) | No | Skip when the dump pre-censors (blocker 2) |
| `data.py:222-253` | Tokens `team_death_s > phase_start_s[k]` (fallback `phase_end_s[k-1]`) | No | Read `node_mask` (keys) and `query_mask` (tokens) from the dump (blocker 1) |
| `data.py:256-292` | Audit counters loader mask vs dump `node_mask_v2` (v1 Erangel TPP squad train: agree 225,666, only loader 101, only dump 0, in `metrics.json` `data.alive_mask`) | Gate | Trivial once the loader reads the dump mask; until then it shows how far the loader rule is from the dump's |
| `data.py:372-414` | Validation: `team_death_s` and `phase_end_s` shapes, `target` inside alive, no target in the last slot | Partly | Last-slot rule matches 7 cuts on 8 slots; inside-alive rule needs the dump to guarantee it |
| `data.py:629-631` | Standardiser statistics over alive tokens and `zone_valid` | Follows | None |
| `hgb.py:123,141,153-155` | HGB-lag lag-block presence uses `effective_alive` at slot p-1 | Follows | None (a risk-set row is alive at p-1 under either rule) |
| `extra.py:294-307,413-417` | kf table must cover every alive token of cut slots 0..P-2; last slot has no cut (v2: the final phase of a match has an empty slot, `assemble.py`) | Follows if the kf table uses the same tokens | Coverage set = query tokens; rebuild the kf table after the loader fix |
| `model.py:22,236-262,404-420` | One `alive` mask: zeroes dead tokens, masks keys on both axes, feeds the hazard head | No | Separate key and query masks so that a comeback row keeps a query token (blocker 1); docstring "alive at phase start" |
| `relations.py:123-134` | M11R-shuf permutes alive tokens per (match, phase) | Follows the mask it is given | Pass the query mask |
| `ecomodel.py:94-97,146-163,786-818,1417-1452` | Habitat columns by name, scale constants, `log_alive_teams = log(100 hab_alive_teams)`, `check_units` integer test; docstring claims that `hab_alive_teams` equals the risk-set size in 96% of cells (L24-26) and that `alive_players_end_norm` is event-based | Conditional | Confirm names and scales (blocker 6); rewrite the two docstring claims |
| `aggregate.py:118,326-334`, `metrics.py:65,75` | Phase groups 1-2 / 3-4 / 5-7, `PHASE_OFFSET`, `MIN_PHASE_ROWS = 30` | Follows (7 cuts kept) | None |
| `learning_curve.py`, `supplement.py`, `derived.py` | Prediction CSVs and `metrics.json` only | Follows | None |
| `player.py`, `dump_player_tier.py` | Stage 2 (`player_alive_cut`, own phase dump) | X | Not rebuilt |
| `synthetic.py` (+ tests) | Fixture dumps with v1 keys | n/a | Update with the loader contract (tests not touched here) |

### B.2 Sagong (`model_stgcn/sagong2025`)

| file:line | Reads or computes | Follows v2? | Change |
|---|---|---|---|
| `plan.py:30,41-47,53-60` | `phase_start_s`, `phase_end_s`, `phase_mask`; `alive_start = effective_alive`; `N_CUTS = 7` | Windows follow; `alive_start` follows the loader | Re-run `plan` (cache). Decide whether type means and the Eq. 4 table use phase-start or cut-time alive status (next row) |
| `fit.py:109-132,263-266` | `alive_start` defines the units for phase means (types) and the survival-rate table | Follows the loader | Semantics shift if the v2 token mask is "alive at the cut" |
| `features.py:6-7,47,186` | Window `[phase_start_s, phase_end_s - 10]` on W, `GUARD_S = 10` | Yes if the dump's `phase_end_s` is `b_k` on W | None (only `tests/test_sagong2025.py:68-70` compares `GUARD_S` with `stdb.causal_cut.GUARD_S`; keep that module's constant) |
| `fit.py:62,341,360` | `N_CUTS = 7`, label fields hard-coded "v2" | Yes while 7 cuts and the v2 key names hold | None |
| `fit.py:64-65,386-406` | Row equality against HGB-cut (required) and M11R | Order | HGB group first |
| `hab.py:24,38-39,45-55` | 44 columns = zone 13 + habitat 31 groups of `meta.feature_groups`, asserted equal to HGB-noHab `feature_removal` | Conditional | Fails loudly if group sizes change |
| `db.py:SQL_*` | `stdb.matches.started_at`, participants, rosters, positions, damage, kills, item pickups on W | Independent of the dump | None |
| `__main__.py:11-13` | Default dirs are v1 | No | Pass `--dump-dir --plan-dir --runs-dir` |

### B.3 KRAFTON adaptation (`third_party/krafton_pubg_winprob/adaptation`)

| file:line | Reads or computes | Follows v2? | Change |
|---|---|---|---|
| `labels.py:36-52` | Hard keys `team_death_s`, `team_won`, `phase_end_s`, `phase_start_s`, `phase_mask`; optional `last_phase_idx`, `clock_delta_s`; `select_labels(m, "v2")` | Yes if the keys are kept | Add `node_mask` and `query_mask` (kftable calls `event_alive_mask` on this compact dict; its rows must cover the query tokens) |
| `extract.py:162-205`, `features.py:77-92` | Own death-time reconstruction (v1 rule) compared with the dump at 5 ms; flag `n_death_mismatch` | No | Blocker 3 |
| `extract.py:277-308` | Val / test sample times `phase_end_s[p] - 10`, slots with any target; squads `T_eff > t` | `b_k` follows | Squad rule `T_eff > t` is "alive at the cut by T", not causal alive. K1 squads, K2 tokens (`ourfeat.py:188`) and the kf-table rows must be one set (`check_against_krafton_tf`, `kftable --verify` compare them); a revived team can be in one and not the other, so fix a common rule (the v2 `query_mask`) |
| `extract.py:287,304`, `schema.py:67,78`, `ourfeat.py:180` | KRAFTON's own 2000-s schedule and Cox censoring `max(2000, match end, last death + 1)` | n/a | Upstream convention, not the dump's old grid cap; keep |
| `kftable.py:61-65,70-77,88-92,126-127,294-296` | Tokens by `event_alive_mask(lab)`, cuts `phase_end_s - 10`, asserts risk set inside alive-at-start and inside `T_eff > cut`, raises on death mismatch | Conditional | Blocker 1 and 3; rebuild the table |
| `ourfeat.py:161-190` | `tok = (death_eff > phase_end_s[p] - 10) & node[p]`, ValueError if tokens do not cover the risk set | Follows | None if the invariant holds |
| `hgb_kraftonfeat.py:100-105` | Table rows equal `ptr.hgb.build_rows(train, 'HGB-cut')` | Follows | Rebuild the table first |
| `validate_causal.py:58-72` | Prefix replay at the cuts `phase_end_s - 10` and KRAFTON times | Follows | Re-run as a gate |
| `h0_baseline.py:72,115-116,233-236`, `phase_auc.py:37` | `CKPT` constant, K2 sample builder, hard-coded 12-cohort list | Path | Section C |

### B.4 Dump-side code (FYI, owned by the dump agent)

These compute phase bounds, windows, durations and alive counts for the v1 dump. They do not feed any downstream module directly; v2 either re-runs them on a redefined bounds view or bypasses them.

- `stdb/schema_rq1.sql:24-35` view `stdb.v_match_phase_bounds` (min / max grid `snapshot_t` per phase: the grid-based `phase_start_s`, `phase_end_s`, `phase_duration_s`).
- `stdb/etl.py:48-49,174,187,312-345` (10-s grid, +-5 s resample, phase tagging from `telem_phase_changes`), `stdb/causal_cut.py` (`GUARD_S`, `PHASE_BOUNDS_SQL`, `PHASE_START_W_SQL`, clock delta), `stdb/queries.py:39,59,74-75,500-535,673-685` (`PHASE_T_MAX = 8`, observed `phase_duration_s` as feature column, node mask from `team_phase_labels.alive_at_phase_start`).
- ETLs reading `v_match_phase_bounds` and windows `[phase_start_s - delta, phase_end_s - delta - 10]`: `phase_spatial_feature_etl.py:117-128,276-278,337,366`, `habitat_feature_etl.py:140-147,175,192,284-311`, `phase_edge_etl.py:90-92,141-142,158,198-200`, `network_feature_etl.py:135-138`, `z_affordance_feature_etl.py:94-96`, `f0_feature_etl.py:81-83,113,126,144,196`, `rq1_feature_etl.py:76-86,119,140,157,170,232,394,415,433,513,567,673,703-713`, `snapshot_queries.py:98-110`, `patch_combat_zone_pass.py:38-40,72`.
- `tools/fixb_backup_tables.py`, `tools/run_fixb_phase_etl.py`, `tools/verify_dump_match.py` (L595 marker `fixb_2026-10-05`), `tools/verify_player_dump_match.py`.
- Audits with their own boundaries from `stdb_bak_20261005.match_phase_global` or `stdb.match_phase_global`: `reports/bigcomp2027_data_profile/scripts/02_sample_player_level.py:17,116` (`tc = te - 10.0`), `09_clock_probe.py:18-20`, `11_label_audit_tier2.py:39`, `label_audit_lib.py:19`. These produce the SEC3 audit numbers (E.1) and must be re-run against v2 bounds.

### B.5 Contract questions for the dump agent (what `assemble.py` already shows is noted)

1. Token masks: `node_mask` = causal alive at `a_k` and `query_mask = node | target` are in the record (answered). Is the loader / model change in blocker 1 agreed (separate key and query masks)? Is `meta['team_death_source']` set, or is there another event marker?
2. Winner censoring is baked in (`T[won] = inf`, winners in every non-excluded risk row) and an excluded cut has an all-False risk row (answered by `labels.py` / `assemble.py`). Are the exclusions summarised in `population.json` / `dump_manifest.json` (`_stats.excluded`, `excluded_rows`) so that blocker 2 can be checked?
3. `phase_start_s[p]` = `s_k`, `phase_end_s[p]` = `b_k` (NaN for the final phase), `cut_t = a_k` (answered). Are `cut_wall_s`, `clock_delta_s` = 0 still written for the consumers that read `clock_delta_s` with `.get` (`adaptation/labels.py`)?
4. `team_death_s` carries T (revive-aware, winners +inf) (answered); the KRAFTON chain must use the same rule.
5. Are `meta['feature_names']` (150, from `stdb.queries` specs), `meta['feature_groups']` ('zone' 13, 'habitat' 31, 'alive_counts' 7), `zone_columns` and `population` still written?
6. Does the v2 dump keep the 12 cohorts, the stored 80/10/10 split (`split_ids_sha1`) and `population.json` with the same schema?
7. Does every match keep the keys of `data._MATCH_KEYS` with the same shapes (`data.py:99-102,384-394`)? `assemble.py` writes all of them.

### B.6 Change list distilled from B and C (for dispatch; none done here)

1. `data.py` and `model.py`: key mask (`node_mask`) and query mask (`query_mask`) from the dump; winner override switch; marker or explicit flag; update `labels.compact_match`, `kftable._event_alive` and `extra.py` coverage in lock-step.
2. `adaptation/features.py`, `extract.py`: v2 death-time rule and squad-at-cut definition; `kftable.py:126-127` assertion stays.
3. `adaptation/run_all.py`: forward `--dump-dir` and `--table-dir` to `ourfeat`, `ourfeat_hazard`, `hgb_kraftonfeat`; `h0_baseline.py`, `h0_report.py`, `diag_phase_init.py`: a `--ckpt-dir` / `--dump-dir` option (or edit the constants).
4. `make_paper_tables.py:19`: `RES` to the v2 results root.
5. Optional: `aggregate.DECOMP_STEPS` or a new table reading the E1 equal-loss contrasts and the h0 contrasts.
6. New scripts for the E.4 numbers.

---

## C. Hard-coded v1 paths

### C.1 Functional defaults (a run reads or writes v1 if the option is not passed)

| file:line | Value | CLI override |
|---|---|---|
| `model_stgcn/dump_phase_tier.py:122` | `DEFAULT_OUT_DIR = .../bigcomp2027_phase_v1` | `--out-dir` (dump agent) |
| `model_stgcn/dump_player_tier.py:107-108` | phase dump v1 -> player dump v1 | yes (X) |
| `model_stgcn/ptr/player.py:46`, `train.py:249,279-280` | `bigcomp2027_player_v1` next to `--dump-dir` | `--player-dump-dir` (X) |
| `model_stgcn/ptr/extra.py:72-76`, `run_matrix.py:662,687` | kf table `<dump parent>/bigcomp2027_kraftonfeat_v1` (a v2 dump would silently get the v1 table) | `--extra-features` |
| `model_stgcn/ptr/aggregate.py:1897`, `learning_curve.py:273`, `derived.py:510`, `supplement.py:547-548` | `reports/bigcomp2027_results` | `--out-dir`, `--results-dir` |
| `model_stgcn/ptr/supplement.py:546` | `--runs-dir` default `.../bigcomp2027_ptr_v1` | `--runs-dir` |
| `model_stgcn/sagong2025/__main__.py:11-13` | dump v1, plan dir `bigcomp2027_sagong_v1`, runs `bigcomp2027_ptr_v1` | `--dump-dir --plan-dir --runs-dir` |
| `adaptation/labels.py:88-89` | dump v1 -> `krafton_winprob_v1` | `--dump-dir --out-dir` |
| `adaptation/extract.py:48`, `train.py:49-50` | `krafton_winprob_v1`, ptr v1 | `--out-dir --data-dir` |
| `adaptation/run_all.py:28-29` | `DEFAULT_OUT` ptr v1, `DEFAULT_DATA` `krafton_winprob_v1` | `--out-dir --data-dir` (but no dump / table option, see next rows) |
| `adaptation/ourfeat.py:54-56`, `ourfeat_hazard.py:370-372` | `DEFAULT_DUMP` phase v1, data v1, out v1 | `--dump-dir` exists but `run_all` does not pass it |
| `adaptation/hgb_kraftonfeat.py:49-52` | data v1, table `bigcomp2027_kraftonfeat_v1`, dump v1, out v1 | `--table-dir --dump-dir` exist but `run_all` does not pass them |
| `adaptation/kftable.py:54,257-261` | root `bigcomp2027_kraftonfeat_v1`; `--label-dir`, `--data-dir` default `krafton_winprob_v1` | yes |
| `adaptation/h0_baseline.py:72` | `CKPT = .../bigcomp2027_ptr_v1` (reads K1, K2 checkpoints and writes the h0 arms there) | none |
| `adaptation/h0_report.py:35,121` | `CKPT` v1; out `reports/krafton_baseline_2026-10-06` | out only |
| `adaptation/phase_auc.py:55,59`, `k3_contrasts.py:75,79`, `k3_facts.py:32,34`, `summarize.py:26-27`, `diagnostics.py:55` | `--ckpt-dir` default ptr v1; outputs `reports/krafton_baseline_2026-10-06` | yes |
| `adaptation/diag_phase_init.py:32-33` | `MAIN` ptr v1, `DIAG` `..._ptr_v1_diag` | `--out-dir` only for DIAG |
| `paper/bigcomp2027/tables/make_paper_tables.py:19` | `RES = reports/bigcomp2027_results` | none |
| `reports/bigcomp2027_results/interpretability/interp.py:45-47`, `make_figures.py:21` | `CK`, `DUMPS`, `OUT` (absolute path) | none (X) |
| `model_stgcn/ptr/run_matrix.py:84,108-121` | `LARGE_COHORTS`, `_REGISTRY_SIZES` (12-cohort registry) | `--large-cohorts`; registry is only the `--projected` fallback |
| `adaptation/phase_auc.py:37`, `diagnostics.py:34`, `summarize.py:20`, `run_all.py:33,50` | Hard-coded 12-cohort lists, two large cohorts | `--only` for `run_all`; none for the others (also imported by `h0_*`) |
| `model_stgcn/ptr/derived.py:246` | `RELIABILITY_COHORTS` | none (X) |

### C.2 Docstring or usage text only (no behaviour)

`model_stgcn/ptr/{train.py:3-5, run_matrix.py:3-4,23, cohorts.py:7, learning_curve.py:3-5, ecomodel.py:56-57, derived.py:4-5, supplement.py:40-41, arms.py:26, aggregate.py:75}`, `model_stgcn/sagong2025/__init__.py:12-13`, `model_stgcn/dump_phase_tier.py:7`, `model_stgcn/dump_player_tier.py:4`, `adaptation/{labels.py:1,8, extract.py:4, kftable.py:6,185, hgb_kraftonfeat.py:9,147, h0_baseline.py:23, parity.py:3, phase_auc.py:3, k3_contrasts.py:3, k3_facts.py:3, h0_report.py:3, diagnostics.py:12, summarize.py:3}`, `tools/verify_player_dump_match.py:3`, `paper/bigcomp2027/tables/make_paper_tables.py:6,63,99,147` (the string `kftable.py:185` also writes `"name": "bigcomp2027_kraftonfeat_v1"` into `meta.json`).

### C.3 Tests that name v1 paths (not touched)

`tests/test_ptr_player.py:161,544-549`, `tests/test_ptr_run_matrix.py:427-548`, `tests/test_dump_player_tier.py:376` (reads the real v1 dump), `tests/test_dump_phase_tier.py:680-684`, `tests/test_causal_cut.py:769`. All but the first-named real-file reads use temporary roots.

### C.4 Other v1 identifiers

- `model_stgcn/checkpoints/br_om_ladder_required_v1/<cohort>/split.json`: the frozen 80/10/10 split used by the dump scripts (`tools/run_fixb_phase_etl.py:52`). Keep; do not re-split.
- Backup schema `stdb_bak_20261005` (`tools/fixb_backup_tables.py:28`, audit scripts in B.4) and `model_stgcn/etl_runs/fixb_2026-10-05/` (marker read at `tools/verify_dump_match.py:595`).
- Dated report root `reports/krafton_baseline_2026-10-06` (C.1); keep v1, write v2 elsewhere.

---

## D. Execution plan

### D.1 Dependency DAG and order

```
G0  v2 dump (other agent) + population.json  -- gate: cohorts --strict, dry-run, 1-cohort smoke, risk set inside tokens (D.7)
 |
 +-> P1  run_matrix --only group=hgb                      (3 min)   unlocks every row-equality assertion
 |    |
 |    +-> S13a Sagong plan -> extract(DB) -> [needs HGB-cut, HGB-noHab] habcols -> fit      (8 min)
 |    +-> S14a ecomodel --stage fit  (also needs only the dump)                             (2 min)
 |    +-> K0   adaptation.labels -> extract(DB)              (8 min)
 |            +-> K1 tf [+ mlp] ----------------------------+
 |            +-> K2 -> K3a, K3 (K1 metrics.json optional) --+-> h0_baseline (needs K1, K2 best.pt)
 |            +-> kftable(DB) -> hgbkf ; -> S6 run_matrix --union
 |
 +-> P2  run_matrix --only group=main,seeds,hgb  (164 runs)  ; S5 --norank ; S8 --lc, --lc --lc-steps-matched
 |
 +-> (all arms done) -> S15 aggregate -> S16 supplement
                      -> S14b ecomodel --stage report, rq1, rq1 physical   (needs M11R, K3a, KRAFTON-tf, SAGONG-hgb)
                      -> reports: learning_curve, h0_report, k3_contrasts, phase_auc, k3_facts
                      -> make_paper_tables
```

Recommended order:

1. Gate and patches (B.6), commit, freeze `RESULT_MODULES`.
2. `run_matrix --only group=hgb` first (HGB-cut and HGB-noHab are required by Sagong, K1/K2/K3 and EHM; the planner would otherwise run them last).
3. In parallel: the CPU lane (P2, S5, then S6 and S8 after kftable) and the MPS lane (K chain). Sagong and `extract` / `kftable` DB jobs fit into the first hour (they are short and need only 2-3 cores).
4. When both lanes are done: aggregate, supplement, EHM report / rq1, reports, paper tables.

### D.2 Wall-clock estimate

| Block | Basis (v1) | v2 estimate |
|---|---|---|
| CPU lane, `run_matrix`, 6 workers x 2 threads | main + seeds + hgb 164 runs about 296 CPU-min; norank 28.8; union 48.8; LC 122.8 + 98.9 | HGB 3 min; main/seeds/hgb about 50 min; norank 6 min; union 9 min; LC fixed 21 min; LC steps 17 min. Total about 1.8 h (about 1.4 h without LC steps and M11R-union) |
| MPS lane, one stream | labels 1 + extract 6.5 + K1 tf 68.4 + K1 mlp 39.4 + K2 40.2 + K3a 38.8 + K3 37.2 + h0 1.3 min | 3.9 h (3.2 h without K1-mlp). kftable 11 min and hgbkf 1.3 min run on the side |
| MPS lane, two streams (est.) | stream A: K1 tf + mlp 108 min; stream B: K2 + K3a + K3 116 min (K2/K3 need only the extract logs, K1 `metrics.json` optional) | about 2.1 h plus contention (est. 2.5-2.8 h) |
| DB / Sagong | plan 0.5, extract 4.3, habcols 0.3, fit 1.3 min | about 8 min |
| EHM | fit about 2 min, report / rq1 / physical (not logged) | about 10 min |
| Analysis tail | aggregate at most 7, supplement at most 8, KRAFTON reports about 5, LC report 1, tables 1 min | about 25 min |
| Critical path | MPS lane + tail | about 4.3 h (one stream) or about 3 h (two streams), plus 30 min gate and patches |

All v1 timings come from `run_matrix_status*.csv`, `krafton_run_status.csv`, `metrics.json` and file times; "at most" figures assume the next step started immediately. v2 row counts may differ slightly (different cuts and exclusions), so allow 20% contingency.

### D.3 Worker and thread budget (this Mac)

- Hardware: Apple M5 Pro, `hw.ncpu` 15 (5 + 10 cores in two performance levels), `hw.memsize` 51,539,607,552 B (48 GiB). Python check without training: `torch 2.11.0, mps_built True, mps_available True` (pyenv 3.11.5). At 19:32 the load average was 3.2 and `vm_stat` showed about 19 GiB free plus inactive, so check `uptime` and `vm_stat` (keep at least 15 GiB headroom) before launching.
- Recommended:

| Job | Workers x threads | Cores |
|---|---|---|
| `run_matrix` (PTR, HGB, norank, union, LC), CPU, bitwise reproducible | 6 x 2 while an MPS job runs; 7 x 2 as in v1 when alone; 4 x 2 while two MPS streams run | 12 / 14 / 8 |
| KRAFTON K1, K2, K3 (`run_all --device mps --threads 3`) | 1 stream x 3 threads (2 streams only when the CPU lane is light) | 3 (6) |
| KRAFTON `extract`, `kftable` | `--workers 3`, `--workers 2` (README used 1) | 3, 2 |
| Sagong `extract`, `fit` | `--workers 2`, `--n-jobs 2` (both capped at 2 in code) | 2 |
| `h0_baseline` | `--threads 3` (capped at 3 in code) | 3 |
| `ecomodel` | 1 process, `OMP_NUM_THREADS=2` | 2 |
| `aggregate` | `--workers 6` (default 7) | 6 |
| `supplement` | `--workers 3` (children set 2 threads) | 6 |

- Memory: the largest v1 dump file is 484 MB (`erangel_tpp_squad` train); a worker holds one cohort, so 6-7 workers fit in 48 GiB. DB jobs are read-only; do not overlap `extract` / `kftable` with the dump agent's heavy DB reads.
- `run_matrix` exports `OMP/MKL/OPENBLAS/VECLIB/NUMEXPR_NUM_THREADS` to its children (`run_matrix.py:520-526`).

### D.4 Fresh v2 output roots

| Purpose | v1 | v2 (recommended) |
|---|---|---|
| Phase dump | `model_stgcn/dumps/bigcomp2027_phase_v1` | `model_stgcn/dumps/bigcomp2027_phase_v2` (given) |
| PTR / HGB / KRAFTON / Sagong / EHM runs | `model_stgcn/checkpoints/bigcomp2027_ptr_v1` | `model_stgcn/checkpoints/bigcomp2027_ptr_v2` |
| Learning curve (fixed, steps-matched) | `..._ptr_v1_lc`, `..._ptr_v1_lc_steps` | `..._ptr_v2_lc`, `..._ptr_v2_lc_steps` |
| K3 zero-init diagnostic | `..._ptr_v1_diag` | `..._ptr_v2_diag` (optional) |
| KRAFTON labels, shards, QA logs | `model_stgcn/dumps/krafton_winprob_v1` | `model_stgcn/dumps/krafton_winprob_v2` |
| KRAFTON feature table | `model_stgcn/dumps/bigcomp2027_kraftonfeat_v1` | `model_stgcn/dumps/bigcomp2027_kraftonfeat_v2` |
| Sagong cache | `model_stgcn/dumps/bigcomp2027_sagong_v1` | `model_stgcn/dumps/bigcomp2027_sagong_v2` |
| Sagong sensitivities | `model_stgcn/checkpoints/bigcomp2027_sagong_{per_phase,sr_label}_v1` | `..._v2` (optional) |
| Player dump (stage 2) | `model_stgcn/dumps/bigcomp2027_player_v1` | not rebuilt |
| Results | `reports/bigcomp2027_results` | `reports/bigcomp2027_results_v2` (`ecological_model`, `supplement`, `figures`, `tables` below it) |
| KRAFTON reports | `reports/krafton_baseline_2026-10-06` | `reports/krafton_baseline_v2` |
| Paper tables | `paper/bigcomp2027/tables/*.tex` | regenerated from the v2 results root |

Keep every v1 root as it is (git-ignored, not recoverable).

### D.5 Run-book (placeholders: `V2`, `R`; commands use the flags that exist today)

```
V2=model_stgcn/dumps/bigcomp2027_phase_v2 ; R=model_stgcn/checkpoints/bigcomp2027_ptr_v2
KF=model_stgcn/dumps/bigcomp2027_kraftonfeat_v2 ; KW=model_stgcn/dumps/krafton_winprob_v2 ; SG=model_stgcn/dumps/bigcomp2027_sagong_v2

# gate (no training)
python -m model_stgcn.ptr.cohorts --dump-dir $V2 --json --strict
python -m model_stgcn.ptr.run_matrix --dump-dir $V2 --out-dir $R --only group=main,seeds,hgb --dry-run

# CPU lane
python -m model_stgcn.ptr.run_matrix --dump-dir $V2 --out-dir $R --only group=hgb --status-name run_matrix_status_hgb.csv --workers 6 --threads 2
python -m model_stgcn.ptr.run_matrix --dump-dir $V2 --out-dir $R --only group=main,seeds,hgb --workers 6 --threads 2      # 164 runs; HGB rows show skipped_done
python -m model_stgcn.ptr.run_matrix --dump-dir $V2 --out-dir $R --norank --status-name run_matrix_status_norank.csv --workers 6 --threads 2
python -m model_stgcn.ptr.run_matrix --dump-dir $V2 --out-dir $R --union --extra-features $KF --status-name run_matrix_status_union.csv --workers 6 --threads 2   # after kftable
python -m model_stgcn.ptr.run_matrix --dump-dir $V2 --out-dir ${R}_lc --lc --workers 6 --threads 2                          # optional
python -m model_stgcn.ptr.run_matrix --dump-dir $V2 --out-dir ${R}_lc_steps --lc --lc-steps-matched --workers 6 --threads 2  # optional

# MPS lane (cwd third_party/krafton_pubg_winprob; paths relative to it; DB env sourced, never printed)
python -m adaptation.labels --dump-dir ../../$V2 --out-dir ../../$KW
mkdir -p ../../$KW/logs        # extract opens <out>/logs/extract_<t>.jsonl without creating the directory; --out-dir is also the label-table dir
python -m adaptation.extract --cohorts <12 names> --splits val,test,train --workers 3 --out-dir ../../$KW
python -m adaptation.run_all --stage main --encoders transformer --out-dir ../../$R --data-dir ../../$KW --device mps --threads 3
python -m adaptation.kftable --cohorts <12 names> --workers 2 --verify --label-dir ../../$KW --data-dir ../../$KW --out-dir ../../$KF
python -m adaptation.run_all --stage ourfeat|hgbkf|k3 ...      # after the pass-through patch (B.6 item 3)
python -m adaptation.h0_baseline --device cpu --threads 3      # after the CKPT patch

# Sagong lane (repo root, interpreter with xgboost + pyarrow)
P=/usr/local/bin/python3
$P -m model_stgcn.sagong2025 plan    --dump-dir $V2 --plan-dir $SG --runs-dir $R
$P -m model_stgcn.sagong2025 extract --dump-dir $V2 --plan-dir $SG --workers 2
$P -m model_stgcn.sagong2025 habcols --dump-dir $V2 --plan-dir $SG --runs-dir $R
$P -m model_stgcn.sagong2025 fit     --dump-dir $V2 --plan-dir $SG --runs-dir $R --models xgb,rf,lr,hgb,hgb-hab

# analysis (after all arms exist)
python -m model_stgcn.ptr.ecomodel --dump-dir $V2 --runs-dir $R --out-dir reports/bigcomp2027_results_v2/ecological_model --stage fit   # may run early
python -m model_stgcn.ptr.aggregate --runs-dir $R --out-dir reports/bigcomp2027_results_v2 --dump-dir $V2 --workers 6
python -m model_stgcn.ptr.supplement --runs-dir $R --results-dir reports/bigcomp2027_results_v2 --out-dir reports/bigcomp2027_results_v2/supplement --workers 3
python -m model_stgcn.ptr.ecomodel ... --stage report ; --stage rq1 ; --stage rq1 --rq1-units physical
python -m model_stgcn.ptr.learning_curve --lc-dir ${R}_lc --main-dir $R --steps-dir ${R}_lc_steps --out-dir reports/bigcomp2027_results_v2
python -m adaptation.h0_report --out-dir ../../reports/krafton_baseline_v2 ; python -m adaptation.k3_contrasts --ckpt-dir ../../$R --out ../../reports/krafton_baseline_v2/k3_contrasts.csv
```

### D.6 Keeping `run_matrix` from reusing v1 status files or config fingerprints

- The status CSV is write-only. `StatusTable` builds its rows from the current plan and rewrites the file (`run_matrix.py:533-562`); it is never read to decide anything. A stale CSV in the same directory is overwritten, not reused. Use `--status-name` per plan type so that the main CSV is not replaced by a later `--union` or `--norank` call (as in v1).
- Reuse is decided only by `is_done` (`run_matrix.py:358-365`): `<out>/<cohort>/<arm>/seed<s>/metrics.json` must exist and carry `config_hash == expected_hash`. The hash (`train.py:309-327`, `hgb.py:201-220`, `util.py:19-66`) covers the arguments, the dump files' size and mtime per cohort (`dump_fingerprint`), and a SHA-1 of the sources `arms, data, extra, hgb, loss, metrics, model, player, relations, train, util`. A new `--dump-dir` or any edit of those modules therefore invalidates every run: this is why the v1 matrix reads "0 of 170 done" today (`arms.py` and `train.py` changed after the v1 stamps).
- Practical rules:
  1. A fresh `--out-dir` (D.4): no v1 `metrics.json` exists, so nothing can be skipped by mistake.
  2. Commit, then freeze the ten `RESULT_MODULES` (and do not rewrite dump files) once the first run starts: a mid-run edit (for example `data.py`) silently marks finished runs stale and a resume would re-run them. The status CSV then shows `planned` for runs that were already done.
  3. Pass `--extra-features $KF` explicitly for `--union`; the default is the v1 table next to the dump (`extra.py:72-76`). `--force` is not needed.
  4. Do not use `--skip-misaligned` in the final aggregation; an alignment failure is the guard against mixing dumps (`aggregate.py` also refuses runs with different label choices).
- The hash-less skips live outside `run_matrix` and trust only file existence: `adaptation/run_all.py` (skips when `metrics.json` exists), `ourfeat.py`, `ourfeat_hazard.py`, `hgb_kraftonfeat.py` (`--redo`), `h0_baseline.py` (all six arm files, `--redo`), `kftable.py` (`counts_partial.json` plus file), `extract.py` (shards, `--redo`), Sagong `extract` (`features.parquet` exists, `--force`). With fresh roots none of them can see a v1 file.

### D.7 Acceptance gates (no training needed unless stated)

1. `python -m model_stgcn.ptr.cohorts --dump-dir $V2 --json --strict` lists 12 complete cohorts, or the changed set is explained (hard-coded lists in C.1).
2. One tiny real run per loader path, e.g. `python -m model_stgcn.ptr.train --dump-dir $V2 --cohort paramo_tpp_squad --arm M11R --max-epochs 1 --out-dir <scratch>` and `hgb --variant HGB-cut`: `validate_dump` must pass without "target rows are on dead tokens"; the `alive tokens:` log line (ghost tokens removed, tokens added) and the `winner censoring` line (`target_rows_added`) must show no rows re-added at excluded cuts. The loader-vs-dump audit (v1 Erangel TPP squad train: agree 225,666, only loader 101, only dump 0) becomes trivially 0 / 0 once the loader reads the dump mask, so it is no longer a check. Also log the comeback rows (`target & ~node`) per cohort.
3. Plausibility against v1 after the HGB group: 12-cohort mean test AUC v1 = HGB-cut 0.794, M00 0.790, M01 0.793, M11R 0.794; v1 hazard by cut 1..7 = 0.208, 0.211, 0.280, 0.393, 0.392, 0.506, 0.557; v1 rows 829,640 with 26.5% positives. A large shift needs an explanation before the neural lane starts.
4. No-look-ahead check on 400 random v2 matches (E.4) and `adaptation.validate_causal` after the KRAFTON patches.
5. Every arm of a cohort carries the same `row_assertion` / row fingerprint (the aggregator fails otherwise).
6. Tests before the run (do not edit them here): `pytest -q tests/test_ptr_*.py tests/test_sagong2025.py tests/test_krafton_adaptation.py tests/test_h0_fit.py`.

---

## E. Dump-dependent numbers in the paper drafts

### E.1 `BIGCOMP2027_SEC3_DATA_KO_v1_2026-10-06.tex`

| Line | Statement | Changes with v2? | Producer (v1) | v2 procedure |
|---|---|---|---|---|
| 9 | About 32.5 M consecutive position intervals, median 10 s, 99.94% within 11 s | No (raw telemetry; stratified sample of 3,250 matches, 250 per cohort, 13 cohorts) | `reports/bigcomp2027_data_profile/scripts/02_sample_player_level.py`, `03_summarize_sample.py` -> `csv/position_interval_stats_by_cohort.csv`, `csv/sample_position_interval_hist.csv`; text `docs/data_layer/BIGCOMP2027_DATA_PROFILE_2026-10-05.md:381` (32,485,309 intervals, 87.66% exactly 10 s) | None unless the sample is redrawn |
| 14 | 13,948 matches | Yes if v2 drops matches (no LogPhaseChange / LogMatchEnd, all cuts excluded) | `<dump>/population.json` (sum of `cohorts.*.after.total` for status kept; v1 = 11,188 + 1,376 + 1,384) written by the dump script | Sum the v2 `population.json` |
| 15 | Matches 2026-03-26 to 06-02; 96.43 M positions, 225.97 M events, 322.40 M records | Yes (match set) | No saved script; inline SQL over the benchmark `match_id`s (TeX comment L40: "DB check 2026-10-06") | `SELECT count(*) FROM pubg.telem_<t> WHERE match_id = ANY(:ids)` per table below, `min/max(created_at)` of `pubg.matches` for the benchmark ids (the profile's `csv/matches_per_day.csv`, `01_stable_profile.py:89`, counts the 13 cohorts before the exclusions) |
| 26-37 | Table I: positions 96,426,398 (`telem_positions`), damage 90,178,937 (`telem_damage`), item pick-up and drop 67,865,634 (`telem_item_pickup`), equip 46,218,920 (`telem_item_equip`), use 15,229,394 (`telem_item_use`), zone state 2,134,448 (`telem_game_states`), landing 1,673,902 (`telem_parachute_landing`), kills 1,267,179 (`telem_kills`), knocks 1,199,290 (`telem_groggy`), phase changes 200,979 (`telem_phase_changes`), total 322,395,081 | Yes (match set only) | Same inline SQL | Same |
| 49 | Clock offset median 5.7 s, max 41.9 s (before alignment) | The offset is a raw-data property (no); the pipeline sentence must match v2 (events on W; game-clock offset only where a G grid is used) | `scripts/09_clock_probe.py` -> `csv/label_audit_clock_probe_matches.csv` (`delta`); `docs/data_layer/BIGCOMP2027_LABEL_AUDIT_2026-10-05.md:11` (delta_pos median 5.7, p95 15.6, max 41.9; 3,250-match sample) | Reuse the number; reword to the v2 method |
| 50 | 400 random matches checked, no input after the next phase transition | Yes | No saved script (inline) | New check (E.4) on v2 inputs: every record behind every input and edge of cut k has `w < a_k` and `a_k = b_k - 10` |
| 51 | `a_k = b_k - 10 s` | Yes (definition, b_k now event-based) | design | Verify wording against B.0 |
| 52 | 99.4% of alive players have a position sample within 15 s before the cut | Yes (cut and alive set change) | `02_sample_player_level.py:116` (`tc = te - 10.0`, `te` = `stdb.match_phase_global.phase_end_s`, grid-based v1 bounds) -> `csv/sample_cut_level.csv` (`C_miss15`, `C_players`) -> `03_summarize_sample.py:71` `csv/missing_positions_near_cut_by_cohort_phase.csv`. v1 reproduction: 1 - 3,505 / 633,733 = 99.45% (12 benchmark cohorts), 99.42% (13 cohorts) | Re-run 02 / 03 with v2 `b_k`, or compute from the v2 builder's per-cut position coverage |
| 56-60 | Event-based label definition, risk set, TODO(Q1) terminal interval and log gaps | Yes | `phase_v2/labels.py` (terminal `(b_k, C_m]`; exclusions `no_match_end`, `position_gap` above 60 s) | Fill TODO(Q1) from the final v2 rule |
| 60 | Cut phases 1..7 | No (`MAX_CUT = 7`) | | |
| 67-69 | airoyale about 92% AI accounts; Vikendi about 30 non-player combat accounts | No | `csv/bots_by_cohort_matchtype.csv` (Erangel TPP squad airoyale 92.1%) from `03_summarize_sample.py:88`; `scripts/08_nonparticipant_accounts.py` -> `csv/nonparticipant_accounts_check.csv` | None |
| 70 | "300 matches or more before the airoyale exclusion" | Wording only | `population.json` rules: `min_matches_per_cohort` 200 after the exclusions (smallest cohort 224) | Reconcile the sentence with the rule actually used |
| 71 | 11,285 official, 2,663 competitive | Yes | `population.json` `match_type_after` summed | Same |
| 77 | Train / val / test 11,188 / 1,376 / 1,384 matches; cohort sizes 224 to 4,159 | Yes | `population.json` `after` | Same |
| 78 | 829,640 risk-set samples | Yes (v2 risk set, exclusions) | Sum of `n_target` over all splits and cohorts of `bigcomp2027_sagong_v1/<cohort>/plan_audit.json` = 829,640 (positives 220,171 = 26.54%); equivalently `data.n_target_rows` in the neural `metrics.json` | Sum `data.n_target_rows` (train, val, test) of M00 over the 12 cohorts of v2 |

### E.2 `BIGCOMP2027_SEC4_METHOD_KO_v1_2026-10-06.tex`

| Line | Statement | Changes with v2? | Producer (v1) | v2 procedure |
|---|---|---|---|---|
| 8-17 | `Delta = 10 s`, `b_{m,k}`, `a_{m,k}`, `T`, `C_m`, risk set, label, hazard definition | Yes (v2 spec) | design doc 4.2 | Check against B.0 |
| 18 | TODO(Q1): terminal interval and log gaps | Yes | `phase_v2/labels.py` | Fill from the final rule |
| 42 | Rank correlation of required speed with outside depth 0.79, of migration deficit 0.41 (training set) | Yes | No saved producer. Pearson r of required speed vs outside depth is 0.90 per cohort in `reports/bigcomp2027_results/ecological_model/collinearity.csv`; `ecomodel.py:786-792` states r about 0.4 for the deficit | Spearman over the pooled train rows of `hgb.build_rows(train, 'HGB-cut')`: `hab_required_speed_to_safe_mps`, `max(0, -hab_signed_boundary_distance_end_m)`, `hab_migration_deficit_mps` |
| 44 | TODO(Q2): exact formula of the movement deficit | Yes (scheduled durations) | `stdb/habitat_feature_etl.py:29-32,284-311` (v1: distance to target at the cut / phase duration; deficit = required - observed trajectory speed) | Fill from the v2 builder |
| 47-48 | 150-dimensional features, nine families, plus team centre and AI fraction | The count probably stays 150 (v2 builds the same columns from `stdb.queries` specs, `phase_v2/features.py:25-32`); the values change | Dump `meta['feature_names']`, `meta['feature_groups']` (v1: F = 150 = 66 + 84; groups include zone 13, habitat 31, network 43, z_affordance 9, alive_counts 7, context 3, survival_capital 9, movement 7, threat_exposure 13, spatial_proxy 3, stdb_spatial 17) | Recount from the v2 meta; the "nine families" mapping is not stored in the dump |
| 75-77 | HGB 200 trees, learning rate 0.06; PTR-Former d = 128, 4 heads, 2 blocks, dropout 0.1, AdamW 1e-3 / 1e-4, cosine, 12 epochs, batch 8, patience 3 | No | `ptr/hgb.py:101-108`, `ptr/train.py:175-262` | None |
| 96 | `tab:adapt`: 55 KRAFTON metrics = 32 exact + 22 approximate + 1 zero; 12 Sagong inputs | No | `third_party/krafton_pubg_winprob/PARITY_TABLE.csv`, README_ADAPTATION section 4 | None |
| 99 | 60-s assist look-back, team aggregation rules | No | `sagong2025/features.py:48`, `fit.py` | None |
| 151 | Within-phase AUC needs at least 30 samples | No | `ptr/metrics.py:75` | None |
| 158-167 | 2,000 bootstrap resamples, SESOI 0.005, DL / REML + HKSJ, Holm over five, seeds 42 and 13 / 2026 on the two largest cohorts | No (the two largest cohorts may change if v2 sizes change) | `ptr/aggregate.py:110,122`, `ptr/supplement.py`, `ptr/run_matrix.py:81-84` | Check the two largest cohorts in v2 `population.json` |
| 164 | TODO(Q9): commit hash and date of the pre-specified analysis | n/a | design doc 6.6 (`e536ecb`) | Fill |

### E.3 Headline numbers of the RQ blueprint that the re-run regenerates (no manual edit)

| Number | Output that holds it |
|---|---|
| M11R - KRAFTON-tf +0.062, M11R - SAGONG-xgb +0.103 (and within-phase) | `contrasts_pooled.csv` (`m11r_vs_krafton_tf`, `m11r_vs_sagong_xgb`) |
| K3a - K2 +0.051; K3a 0.791 vs M01 0.793; explicit baseline effect 0 | `contrasts_pooled.csv` (`k3a_vs_k2`, `k3_vs_k3a`), `arm_metrics_mean.csv`, `tables/tab_decomp.tex` |
| HGB-cut - SAGONG-hgb +0.091; habitat 0.053 and rest 0.039 | `contrasts_pooled.csv` (`hgbcut_vs_sagong_hgb`); the 0.053 / 0.039 split needs the missing producer in E.4 |
| Architecture +0.002, structure +0.002, input +0.014, extension +0.008 | `tab_decomp.tex` (`m01_vs_k3`, `m11r_vs_m01`, `hgbcut_vs_hgb_kraftonfeat`, `hgb_union_vs_hgbcut`) |
| EHM 80% [0.79, 0.82], EHM-lite 79.4% [0.780, 0.808], coefficient signs 12/12, 11/12, 11/11 | `ecological_model/share_of_discrimination.csv`, `coefficients_pooled.csv`, `rq1/*` |
| Structure helps only at about 700 or more training matches | `metareg.csv` (`crossover_n`) |
| P1-P5 verdicts, Holm | `contrasts_pooled.csv`, `summary.json` |
| "13,948 matches, 12 cohorts, 829,640 samples" | E.1 |

### E.4 Numbers or checks without a saved producer (must be written, then run on v2)

1. `model_stgcn/dumps/bigcomp2027_sagong_v1/{summary_hab_increment, summary_hgb_control, summary_hgb_control_by_phase, summary_test_auc}.csv`: no script in the repository. Verified recipe for `summary_hab_increment.csv` from the prediction files of SAGONG-hgb, SAGONG-hgb-hab, HGB-cut, HGB-noHab (pooled AUC and `metrics.within_phase_auc`): `step_hab = hab - hgb`, `step_rest = cut - hab`, `step_total = cut - hgb`, `cut_minus_nohab = cut - nohab` (checked on the first two rows).
2. The 400-match no-look-ahead check (SEC3 L50).
3. The rank correlations 0.79 and 0.41 (SEC4 L42).
4. Label-audit percentages used by the blueprint (claim D) and design doc 4.2: mislabel 11.4%, positive precision 70.4%, recall 75.2%, hazard by cut, and the "about 27% of matches would leak without alignment". Source: `docs/data_layer/BIGCOMP2027_LABEL_AUDIT_2026-10-05.md` with `10_label_audit_tier1.py`, `11_label_audit_tier2.py`, `label_audit_lib.py` (v1 bounds from `stdb_bak_20261005.match_phase_global`). Against v2 labels they must be recomputed or dropped.
5. Edge-by-phase elimination rates 0.589 (outside, phase 4) and 0.243 (deep inside) in the blueprint: no producer found.

---

## Appendix: v1 reference values for the gates

- v1 12-cohort mean test AUC (`arm_metrics_mean.csv`): M00 0.7897, M01 0.7927, M11R 0.7936, HGB-cut 0.7944.
- v1 label v2 hazard by cut k = 1..7: 0.208, 0.211, 0.280, 0.393, 0.392, 0.506, 0.557 (design doc 4.2).
- v1 dump: 12 cohorts, 13,948 matches, 829,640 risk-set rows, 220,171 positives; F = 150; 8 phase slots, cuts 1..7; `population.json` rules: airoyale and Vikendi excluded, at least 200 matches per cohort.
- v1 run counts: main matrix 170 (108 + 6 + 20 + 36), stage 2 24, union 28, norank 24, LC 102 + 90, KRAFTON 72 (+ 72 h0), aggregate 342 runs.
