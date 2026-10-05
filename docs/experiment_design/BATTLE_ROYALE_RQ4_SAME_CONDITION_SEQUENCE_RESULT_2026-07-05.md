# Battle Royale RQ4 Same-Condition Sequence Result

Date: 2026-07-05
Class: `ExperimentResult`
Lifecycle: `required_complete_bounded`

## Question

```text
Under the same official PUBG 10s snapshot feature surface, do sequence models
outperform tabular reductions for next-phase survival hazard prediction?
```

## Scope

This is the full required-cohort same-condition comparison for the revised RQ3
model-form claim: prior tabular/phase-aware prediction is extended with
10-second telemetry sequence models, but the thesis must still respect strong
tabular snapshot baselines.

Run root:

```text
model_stgcn/checkpoints/br_rq4_same_condition_snapshot_required_v2
```

Required cohorts:

```text
Erangel TPP squad
Taego TPP squad
Miramar / Desert TPP squad
Rondo TPP squad
Sanhok TPP squad
Erangel TPP solo
Erangel TPP duo
Taego TPP duo
Vikendi TPP squad
Taego FPP squad
Erangel FPP squad
Paramo TPP squad
Karakin TPP squad
```

Aggregate files:

```text
model_stgcn/checkpoints/br_rq4_same_condition_snapshot_required_v2/snapshot_survival_bridge_metrics_all_cohorts.csv
model_stgcn/checkpoints/br_rq4_same_condition_snapshot_required_v2/snapshot_survival_bridge_metrics_summary.csv
```

## Implemented Arms

| Variant | Role |
| --- | --- |
| `snapshot_last_hgb` | HGB on the last official snapshot at each phase cut |
| `snapshot_summary_hgb` | HGB on summary statistics over official snapshots up to the cut |
| `snapshot_last_only` | neural lower-bound using only phase-cut state |
| `snapshot_tcn_no_graph` | causal temporal convolution over 10s snapshots |
| `snapshot_gru_no_graph` | causal snapshot-level GRU over 10s snapshots |

`snapshot_hier_gru_no_graph` remains implemented but is not the default arm
because its Python-loop phase encoder is too slow for the required-cohort
queue.  The GRU arm better matches the current thesis claim: the team state is
updated by the full observed snapshot sequence and read out at phase cuts.

## Aggregate Result

| Variant | Cohorts | Median Hazard AUC | Mean Hazard AUC | Median Hazard AP | Mean Hazard AP | Median Rank Spearman from Survival |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `snapshot_summary_hgb` | 13 | 0.7534 | 0.7487 | 0.6450 | 0.6474 | n/a |
| `snapshot_last_hgb` | 13 | 0.7411 | 0.7407 | 0.6313 | 0.6327 | n/a |
| `snapshot_gru_no_graph` | 13 | 0.7355 | 0.7345 | 0.6148 | 0.6162 | -0.1820 |
| `snapshot_tcn_no_graph` | 13 | 0.7325 | 0.7316 | 0.6238 | 0.6282 | -0.1325 |
| `snapshot_last_only` | 13 | 0.7213 | 0.7233 | 0.6091 | 0.6194 | -0.1628 |

## Cohort-Level Result

| Cohort | Best Overall | Best AUC | Best Sequence | Sequence AUC | Last-Only AUC | Last-HGB AUC | Summary-HGB AUC |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `erangel_fpp_squad` | `snapshot_summary_hgb` | 0.7632 | `snapshot_tcn_no_graph` | 0.7325 | 0.7202 | 0.7526 | 0.7632 |
| `erangel_tpp_duo` | `snapshot_summary_hgb` | 0.7304 | `snapshot_tcn_no_graph` | 0.7148 | 0.7016 | 0.7247 | 0.7304 |
| `erangel_tpp_solo` | `snapshot_summary_hgb` | 0.7583 | `snapshot_gru_no_graph` | 0.7355 | 0.7138 | 0.7361 | 0.7583 |
| `erangel_tpp_squad` | `snapshot_summary_hgb` | 0.7723 | `snapshot_gru_no_graph` | 0.7658 | 0.7281 | 0.7491 | 0.7723 |
| `karakin_tpp_squad` | `snapshot_last_only` | 0.7107 | `snapshot_gru_no_graph` | 0.7042 | 0.7107 | 0.6903 | 0.6991 |
| `miramar_desert_tpp_squad` | `snapshot_summary_hgb` | 0.7711 | `snapshot_gru_no_graph` | 0.7524 | 0.7386 | 0.7654 | 0.7711 |
| `paramo_tpp_squad` | `snapshot_tcn_no_graph` | 0.7140 | `snapshot_tcn_no_graph` | 0.7140 | 0.7091 | 0.7099 | 0.7034 |
| `rondo_tpp_squad` | `snapshot_summary_hgb` | 0.7647 | `snapshot_gru_no_graph` | 0.7468 | 0.7377 | 0.7597 | 0.7647 |
| `sanhok_tpp_squad` | `snapshot_summary_hgb` | 0.7534 | `snapshot_tcn_no_graph` | 0.7425 | 0.7407 | 0.7532 | 0.7534 |
| `taego_fpp_squad` | `snapshot_summary_hgb` | 0.7435 | `snapshot_tcn_no_graph` | 0.7160 | 0.7096 | 0.7360 | 0.7435 |
| `taego_tpp_duo` | `snapshot_summary_hgb` | 0.7427 | `snapshot_gru_no_graph` | 0.7309 | 0.7213 | 0.7392 | 0.7427 |
| `taego_tpp_squad` | `snapshot_summary_hgb` | 0.7805 | `snapshot_gru_no_graph` | 0.7623 | 0.7465 | 0.7718 | 0.7805 |
| `vikendi_tpp_squad` | `snapshot_summary_hgb` | 0.7507 | `snapshot_gru_no_graph` | 0.7423 | 0.7255 | 0.7411 | 0.7507 |

## Derived Comparisons

| Comparison | Result |
| --- | --- |
| Overall winner count | `snapshot_summary_hgb`: 11/13; `snapshot_tcn_no_graph`: 1/13; `snapshot_last_only`: 1/13 |
| Best sequence arm | GRU in 8/13 cohorts; TCN in 5/13 cohorts |
| Best sequence beats `snapshot_last_only` | 12/13 cohorts |
| Best sequence beats `snapshot_last_hgb` | 4/13 cohorts |
| Best sequence beats `snapshot_summary_hgb` | 2/13 cohorts |
| Median AUC delta, best sequence minus last-only | +0.0123 |
| Median AUC delta, best sequence minus last-HGB | -0.0095 |
| Median AUC delta, best sequence minus summary-HGB | -0.0156 |

## Supported

The required-cohort result supports the following limited statements:

1. Official PUBG 10s snapshot telemetry contains useful phase-hazard signal
   across all required cohorts.
2. Sequence accumulation is usually useful relative to a neural phase-cut-only
   lower bound: the best sequence arm improves over `snapshot_last_only` in
   12/13 cohorts.
3. The choice of temporal sequence model is cohort-dependent.  GRU is the best
   sequence arm in 8 cohorts and TCN is the best sequence arm in 5 cohorts.
4. A sequence model can occasionally become the best same-condition arm:
   `snapshot_tcn_no_graph` is the best overall arm on `paramo_tpp_squad`.

## Not Supported

Do not claim a general deep-learning advantage over tabular reductions of the
same official snapshot feature surface.  `snapshot_summary_hgb` is the best
overall arm in 11/13 cohorts and has the strongest aggregate median hazard AUC:

```text
median(snapshot_summary_hgb) - median(snapshot_gru_no_graph) = +0.0179 AUC
median(snapshot_summary_hgb) - median(snapshot_tcn_no_graph) = +0.0209 AUC
```

Do not claim that cumulative survival scores already predict final rank well.
All three neural hazard-derived rank bridges have negative median Spearman:

| Variant | Median Rank Spearman | Median Latest-Cut Rank Spearman |
| --- | ---: | ---: |
| `snapshot_last_only` | -0.1628 | -0.4030 |
| `snapshot_tcn_no_graph` | -0.1325 | -0.3082 |
| `snapshot_gru_no_graph` | -0.1820 | -0.5798 |

This means the next-phase hazard formulation is useful for phase survival, but
the final-rank bridge likely needs calibration, a latest-cut reporting
discipline plus model redesign, or an explicit auxiliary final-outcome head
before becoming a headline final-rank result.

## Manuscript-Safe Interpretation

This experiment should be written as a bounded model-form result:

```text
The phase-wise survival formulation is executable under official PUBG snapshot
telemetry, and temporal sequence models generally improve over a neural
phase-cut-only lower bound.  However, strong tabular summary reductions over
the same official snapshot history remain more competitive than the current
compact TCN/GRU architectures.  Therefore the thesis should not claim a general
deep-learning advantage; it should claim that phase survival is predictable and
that representation/feature design, especially the previously supported
pressure-habitat block, remains the central source of supervised evidence.
```

## Engineering Notes

Two fixes were required before the required-cohort run was feasible:

1. Same-condition tabular SQL now uses `stdb.match_phase_global` instead of
   recomputing `stdb.v_match_phase_bounds` from `dynamic_regions`.
2. No-graph snapshot models now slice to active team slots inside each batch
   and scatter hazards back to the padded 128-slot contract.

These changes do not alter the input feature surface or labels; they remove
unnecessary recomputation over phase bounds and padded team slots.

## Next

The next manuscript-facing step is not another architecture run.  Integrate
this result into the RQ4 chapter after:

```text
Sagong-style/basic tabular baseline comparison
OM2 -> OM3 pressure-habitat result
phase-stratified and paired-bootstrap robustness
```

Then present this same-condition sequence experiment as a model-form boundary:
sequence accumulation helps over last-only, but current compact DL sequence
models do not replace strong tabular summary baselines.

## Addendum: LSTM and Learner-Zoo Same-Condition Run

Date: 2026-07-06
Class: `ExperimentResultAddendum`
Lifecycle: `required_complete_bounded`

This addendum extends the same-condition RQ4 comparison with:

1. a full required-cohort `snapshot_lstm_no_graph` arm; and
2. a wider tabular learner zoo for the same official snapshot surfaces:
   `lr`, `rf`, `hgb`, `xgb`, and `mlp` over `snapshot_last` and
   `snapshot_summary` reductions.

Run root:

```text
model_stgcn/checkpoints/br_rq4_sequence_model_zoo_required_v1
```

Aggregate files:

```text
model_stgcn/checkpoints/br_rq4_sequence_model_zoo_required_v1/snapshot_survival_bridge_metrics_all_cohorts.csv
model_stgcn/checkpoints/br_rq4_sequence_model_zoo_required_v1/snapshot_survival_bridge_metrics_summary.csv
```

### Addendum Aggregate Result

| Variant | Cohorts | Median Hazard AUC | Mean Hazard AUC | Median Hazard AP |
| --- | ---: | ---: | ---: | ---: |
| `snapshot_summary_hgb` | 13 | 0.7550 | 0.7490 | 0.6457 |
| `snapshot_summary_xgb` | 13 | 0.7530 | 0.7483 | 0.6427 |
| `snapshot_summary_rf` | 13 | 0.7474 | 0.7403 | 0.6367 |
| `snapshot_last_mlp` | 13 | 0.7444 | 0.7377 | 0.6240 |
| `snapshot_lstm_no_graph` | 13 | 0.7441 | 0.7379 | 0.6209 |
| `snapshot_summary_mlp` | 13 | 0.7439 | 0.7398 | 0.6215 |
| `snapshot_last_hgb` | 13 | 0.7411 | 0.7411 | 0.6315 |
| `snapshot_summary_lr` | 13 | 0.7408 | 0.7350 | 0.6120 |
| `snapshot_last_xgb` | 13 | 0.7408 | 0.7393 | 0.6211 |
| `snapshot_last_rf` | 13 | 0.7251 | 0.7278 | 0.6194 |
| `snapshot_last_lr` | 13 | 0.7236 | 0.7232 | 0.5971 |

### LSTM Cohort-Level Hazard AUC

| Cohort | Hazard AUC | Hazard BCE | Hazard AP |
| --- | ---: | ---: | ---: |
| `erangel_tpp_squad` | 0.7664 | 0.5085 | 0.6560 |
| `taego_tpp_squad` | 0.7645 | 0.4908 | 0.6502 |
| `miramar_desert_tpp_squad` | 0.7549 | 0.4981 | 0.6209 |
| `rondo_tpp_squad` | 0.7462 | 0.5064 | 0.6130 |
| `erangel_fpp_squad` | 0.7447 | 0.5374 | 0.6017 |
| `vikendi_tpp_squad` | 0.7445 | 0.5272 | 0.6149 |
| `sanhok_tpp_squad` | 0.7441 | 0.5223 | 0.6561 |
| `erangel_tpp_solo` | 0.7428 | 0.5532 | 0.6649 |
| `taego_tpp_duo` | 0.7344 | 0.5355 | 0.6592 |
| `taego_fpp_squad` | 0.7253 | 0.5635 | 0.5932 |
| `erangel_tpp_duo` | 0.7145 | 0.5532 | 0.6215 |
| `paramo_tpp_squad` | 0.7093 | 0.5717 | 0.6023 |
| `karakin_tpp_squad` | 0.7017 | 0.5999 | 0.6007 |

Summary:

```text
snapshot_lstm_no_graph median hazard AUC = 0.7441
snapshot_lstm_no_graph mean hazard AUC   = 0.7379
snapshot_lstm_no_graph min/max AUC       = 0.7017 / 0.7664
```

### Addendum Interpretation

The LSTM arm improves the sequence-model family relative to the earlier compact
TCN/GRU aggregate, but it still does not overturn the main boundary:
strong tabular snapshot summaries remain the best same-condition supervised
hazard baseline.

Safe statements:

1. A causal LSTM over the full official 10s snapshot stream is a viable
   required-cohort phase-hazard model.
2. The LSTM is competitive with weaker last-only and neural baselines, but it is
   not better than `snapshot_summary_hgb` or `snapshot_summary_xgb`.
3. The best current evidence still points to representation design and
   pressure/habitat feature construction as the stronger supervised source of
   hazard prediction performance.

Not supported:

1. Do not claim a general deep-learning advantage over same-condition tabular
   snapshot summaries.
2. Do not claim that cumulative hazard-derived survival scores already solve
   final placement or victory prediction.  The LSTM bridge remains weak:

```text
median latest-cut top5 AUC         = 0.2147
median latest-cut top10 AUC        = 0.2869
median latest-cut rank Spearman    = -0.5349
```

Transformer note:

`snapshot_transformer_no_graph` was implemented and unit-tested, but the full
required-cohort CPU run is not included as evidence.  On the full Erangel TPP
squad snapshot stream, the first epoch did not complete within the initial
feasibility window, and disk caching the full no-edge snapshot tensors required
approximately 6.3GB for the train split alone.  Under the current local CPU and
disk constraints, Transformer should be treated as implemented/smoke-tested but
not a completed required-cohort result.
