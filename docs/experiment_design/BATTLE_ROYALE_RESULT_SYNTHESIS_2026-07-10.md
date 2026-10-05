# Battle Royale Experiment Result Synthesis

Date: 2026-07-10
Class: `ExperimentResult`
Lifecycle: `active`

## Purpose

This document consolidates the current manuscript-facing experiment results.
It is not a raw run log.  It is the result map for writing the results chapter:

```text
genre formalization
-> PUBG telemetry representation
-> phase and realtime survival prediction
-> phase-transition risk and advantage ranking
```

The main evidence should be written with explicit claim boundaries.  The paper
does not need to argue that realtime or in-match prediction did not exist
before.  It should argue that this work reorganizes Battle Royale prediction as
a phase-conditioned multi-agent survival problem, instantiates it in PUBG
official telemetry, and evaluates whether the resulting survival readout can be
used for phase-level risk and advantage ranking.

## Result Bottom Line

| Question | Current result | Manuscript-safe conclusion |
| --- | --- | --- |
| Data and cohort design | Locked paper cohort: 27,491 raw-fetched matches, 21,824 graph-eligible matches, 13 main cohorts with 17,943 matches. Match dates in the locked cohort span 2026-03-26 to 2026-06-02. | Main claims should be attached to the locked cohort and to independently trained map / perspective / team-size cohorts, not to an uncontrolled pooled population. |
| Representation blocks | In the 13-cohort OM ladder, the largest consistent gain appears when pressure and position-condition features are added. Median hazard AUC rises from OM2 `0.7397` to OM3 `0.7499`. | The formal survival factors are not only terminology; a subset of them yields measurable predictive gains in PUBG telemetry. |
| Prior and basic baselines | Representation-rich HGB baselines outperform official-raw and Sagong-style prior-score baselines across 13 cohorts. | The contribution is not simply using PUBG telemetry, but organizing telemetry into phase-aware survival factors. |
| Sequence and realtime readout | Sequence models support phase-cut hazard trajectories, but tabular summary models remain the strongest same-condition hazard baselines. | Claim a useful realtime-style survival readout, not general deep-learning superiority. |
| Representative model comparison | On Erangel TPP squad, ST-GNN-Mamba has the highest AUC/AP among the repeated main neural variants; FARNET is slightly stronger on rank Spearman. | Use this as a representative model comparison, supported by seed repeats, not as the sole proof of the framework. |
| Risk / advantage board | The LSTM realtime board covers 13 cohorts and 10,616 phase boards. Median risk top-1 is `0.7004`, risk top-3 is `0.9172`, and advantage top-team final top-10 is `0.9490`. | Phase-cut survival outputs can be turned into interpretable risk and advantage rankings. |
| Board boundary | Pressure-habitat HGB is the strongest board scorer in the audited comparison: median risk top-1 `0.7188`. | The LSTM board is the realtime readout form; do not claim it is the strongest possible scorer under every comparison. |

## Data Scope

The paper dataset is locked by:

```text
model_stgcn/experiment_datasets/research_cohort_flat/match_cohort_flat.csv
model_stgcn/experiment_datasets/research_cohort_flat/paper_graph_n128_v1_split_seed42.json
```

| Scope | Matches | Date range | Role |
| --- | ---: | --- | --- |
| Raw-fetched manifest rows | 27,491 | 2026-03-26 to 2026-06-02 | Source population recorded in the research cohort manifest |
| Graph-eligible paper population | 21,824 | 2026-03-26 to 2026-06-02 | Common-denominator paper cohort |
| Main independently trained cohorts | 17,943 | 2026-03-26 to 2026-06-02 | Main manuscript result set |
| Train split | 17,460 | 2026-03-26 to 2026-06-02 | Master split train partition |
| Validation split | 2,182 | 2026-03-26 to 2026-06-02 | Master split validation partition |
| Test split | 2,182 | 2026-03-26 to 2026-06-02 | Master split test partition |

The 13 main cohorts are trained and evaluated separately:

| Cohort | Matches |
| --- | ---: |
| Erangel TPP squad | 6,204 |
| Taego TPP squad | 4,159 |
| Miramar / Desert TPP squad | 1,543 |
| Rondo TPP squad | 1,039 |
| Sanhok TPP squad | 940 |
| Erangel TPP solo | 800 |
| Erangel TPP duo | 656 |
| Taego TPP duo | 573 |
| Vikendi TPP squad | 498 |
| Taego FPP squad | 435 |
| Erangel FPP squad | 430 |
| Paramo TPP squad | 333 |
| Karakin TPP squad | 333 |

Use this wording in the manuscript:

```text
Experiments were conducted on a locked PUBG official-telemetry cohort.  The
main results use 13 independently trained cohorts separated by map, camera
perspective, and team size.
```

Avoid saying that all results are from a single pooled PUBG dataset.

## RQ2. Representation Results

### OM ladder across 13 cohorts

Main artifact:

```text
model_stgcn/checkpoints/br_om_ladder_required_v1/
```

Median performance across the 13 main cohorts:

| Block | Meaning in prose | Median AUC | Median AP | Median rank Spearman |
| --- | --- | ---: | ---: | ---: |
| OM0 | match and phase context | 0.7353 | 0.6103 | 0.5883 |
| OM1 | team survival state | 0.7336 | 0.6186 | 0.5870 |
| OM2 | movement and rotation | 0.7397 | 0.6200 | 0.5914 |
| OM3 | pressure and position condition | 0.7499 | 0.6338 | 0.6147 |
| OM4 | combat and threat exposure | 0.7527 | 0.6364 | 0.6268 |
| OM5 | relation structure | 0.7530 | 0.6332 | 0.6235 |
| OM6 | spatial proxy | 0.7540 | 0.6414 | 0.6178 |

Representative Erangel TPP squad values:

| Block | AUC | AP | Rank Spearman |
| --- | ---: | ---: | ---: |
| OM0 | 0.7832 | 0.6658 | 0.6200 |
| OM1 | 0.7836 | 0.6652 | 0.6225 |
| OM2 | 0.7836 | 0.6673 | 0.6199 |
| OM3 | 0.7937 | 0.6801 | 0.6525 |
| OM4 | 0.7951 | 0.6838 | 0.6546 |
| OM5 | 0.7944 | 0.6821 | 0.6531 |
| OM6 | 0.7948 | 0.6838 | 0.6524 |

Interpretation:

```text
The major step is not the addition of every possible PUBG-specific signal.
The largest consistent change occurs when the model receives information about
phase pressure and position conditions.  Later relation and spatial-proxy
features add smaller and more conditional gains.
```

### Representative feature-preset comparison

Main artifact:

```text
model_stgcn/checkpoints/followup_erangel_required_v1/feature_preset/matrix_metrics.csv
```

Erangel TPP squad results:

| Preset | AUC | AP | Rank Spearman |
| --- | ---: | ---: | ---: |
| Core | 0.7869 | 0.6722 | 0.6519 |
| STDB only | 0.7842 | 0.6694 | 0.6215 |
| Network only | 0.7831 | 0.6670 | 0.6182 |
| Pressure / position condition | 0.7930 | 0.6784 | 0.6481 |
| Core + pressure / position condition | 0.7948 | 0.6829 | 0.6565 |
| Base 66 | 0.7878 | 0.6725 | 0.6515 |
| Full | 0.7951 | 0.6826 | 0.6559 |

This supports a careful feature claim:

```text
The genre-level survival representation should not be collapsed into a raw
PUBG telemetry dump.  Pressure and position-condition features carry most of
the additional predictive signal in the representative cohort, while network
features are better treated as a supplementary relation channel.
```

## RQ3. Prediction And Model Results

### Prior-style and basic telemetry baselines

Main artifacts:

```text
model_stgcn/checkpoints/br_rq4_prior_baseline_required_v1/
model_stgcn/checkpoints/br_rq4_prior_model_zoo_required_v1/
```

Selected 13-cohort HGB baseline results:

| Model family | Median AUC | Median AP | Meaning |
| --- | ---: | ---: | --- |
| Prior score only | 0.6525 | 0.4480 | Sagong-style score-only proxy |
| Prior raw | 0.7172 | 0.5505 | prior-style raw summary |
| Official raw | 0.7521 | 0.6307 | official telemetry without the full survival-factor organization |
| Current conventional | 0.7458 | 0.6174 | conventional current-state features |
| Raw + pressure / position condition | 0.7742 | 0.6566 | official raw plus central survival-factor block |
| Full representation | 0.7740 | 0.6586 | broad representation |
| Conventional + STDB + network + pressure / position condition | 0.7753 | 0.6577 | strongest selected HGB baseline |

Model-zoo check:

| Model | Median AUC | Median AP |
| --- | ---: | ---: |
| Conventional + STDB + network + pressure / position condition, HGB | 0.7753 | 0.6577 |
| Same feature set, XGB | 0.7717 | 0.6517 |
| Same feature set, RF | 0.7631 | 0.6566 |
| Same feature set, MLP | 0.7619 | 0.6514 |
| Same feature set, logistic regression | 0.7523 | 0.6227 |

Safe claim:

```text
The proposed representation improves over prior-style score baselines and
basic official-raw baselines under the same cohort contract.
```

Unsafe claim:

```text
Prior work did not attempt in-match or phase-aware prediction.
```

The paper should instead say that prior work motivates the baseline family,
while this study changes the object from score/rank prediction to a
phase-conditioned survival process.

### Same-condition sequence comparison

Main artifacts:

```text
model_stgcn/checkpoints/br_rq4_same_condition_snapshot_required_v2/
model_stgcn/checkpoints/br_rq4_sequence_model_zoo_required_v1/
```

Selected 13-cohort results:

| Model variant | Median hazard AUC | Median hazard AP | Boundary |
| --- | ---: | ---: | --- |
| Snapshot summary HGB | 0.7550 | 0.6457 | strongest same-condition tabular summary |
| Snapshot summary XGB | 0.7530 | 0.6427 | close tree baseline |
| Snapshot last HGB | 0.7411 | 0.6315 | last-state tabular baseline |
| Snapshot LSTM | 0.7441 | 0.6209 | realtime-style sequence readout |
| Snapshot GRU | 0.7355 | 0.6148 | neural sequence baseline |
| Snapshot TCN | 0.7325 | 0.6238 | neural temporal convolution baseline |
| Snapshot last only | 0.7213 | 0.6091 | weak last-observation baseline |

Interpretation:

```text
Temporal histories are useful for constructing a changing survival readout, but
the strongest same-condition hazard scorer remains the tabular summary HGB.
The manuscript should therefore separate model form from prediction object:
the main contribution is the survival readout and board construction, not a
claim that sequence neural models dominate all tabular alternatives.
```

### Representative neural architecture comparison

Main artifacts:

```text
model_stgcn/checkpoints/full_required_core13_v1/erangel_tpp_squad/
model_stgcn/checkpoints/followup_erangel_required_v1/seed_repeats/
```

Erangel TPP squad, seed 42:

| Model | AUC | AP | Rank Spearman |
| --- | ---: | ---: | ---: |
| TCN | 0.7962 | 0.6845 | 0.6587 |
| GraphSAGE | 0.7973 | 0.6873 | 0.6573 |
| R-GCN | 0.7960 | 0.6836 | 0.6605 |
| FARNET | 0.7967 | 0.6862 | 0.6889 |
| ST-GNN-Mamba | 0.7972 | 0.6884 | 0.6886 |

Seed-repeat summary for the repeated main neural variants:

| Model | AUC mean | AUC sd | AP mean | AP sd | Rank Spearman mean | Rank Spearman sd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TCN | 0.7955 | 0.0007 | 0.6839 | 0.0008 | 0.6578 | 0.0012 |
| FARNET | 0.7963 | 0.0004 | 0.6862 | 0.0008 | 0.6892 | 0.0012 |
| ST-GNN-Mamba | 0.7972 | 0.0002 | 0.6884 | 0.0001 | 0.6884 | 0.0009 |

Safe claim:

```text
In the representative Erangel TPP squad cohort, ST-GNN-Mamba is the most
stable high-performing neural variant for hazard AUC/AP, while FARNET is
slightly stronger on the rank-correlation readout.
```

## RQ4. Risk And Advantage Ranking

### Dynamic survival readout

Main artifacts:

```text
reports/rq4_dynamic_probability_fluctuation/
tools/build_dynamic_probability_fluctuation.py
```

Audited aggregate:

| Metric | Value |
| --- | ---: |
| Cohorts | 13 |
| Phase boards | 10,616 |
| Median risk top-1 hit | 0.7004 |
| Median risk top-3 hit | 0.9172 |
| Median risk MRR | 0.8121 |
| Median advantage top-team final top-5 | 0.7768 |
| Median advantage top-team final top-10 | 0.9490 |
| Median advantage top-5 contains final top-10 | 1.0000 |
| Median advantage NDCG@5 over final top-10 | 0.9381 |
| Mean leader-change rate | 0.4214 |

Safe claim:

```text
Phase-wise hazards can be accumulated into a match-local survival-share
trajectory, and that trajectory can be evaluated as a changing risk and
advantage ranking at phase transitions.
```

### Board-baseline comparison

Main artifacts:

```text
reports/rq4_board_baseline_comparison/
tools/compare_rq4_board_baselines.py
```

All rows below use the audited intersected board universe:

| Board | Phase boards | Median risk top-1 | Median risk top-3 | Median risk MRR | Median advantage final top-10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pressure-habitat HGB | 10,616 | 0.7188 | 0.9359 | 0.8296 | 0.9514 |
| LSTM realtime board | 10,616 | 0.7004 | 0.9172 | 0.8121 | 0.9490 |
| Official-raw HGB | 10,616 | 0.6952 | 0.9252 | 0.8113 | 0.9473 |
| Health / alive-count | 10,616 | 0.5094 | 0.8349 | 0.6853 | 0.9184 |
| Sagong-style adapted HGB | 10,616 | 0.4897 | 0.8385 | 0.6756 | 0.9042 |
| Zone-distance | 10,616 | 0.4240 | 0.7599 | 0.6117 | 0.8926 |
| Random alive | 10,616 | 0.3868 | 0.7564 | 0.6008 | 0.8742 |

Interpretation:

```text
The realtime LSTM board strongly outperforms random, alive-count, zone-distance,
and Sagong-style boards.  The pressure-habitat HGB board is the strongest
phase-cut scorer, which should be reported as a boundary rather than hidden.
```

Manuscript-safe phrasing:

```text
The proposed survival readout can be operationalized as a phase-transition
risk and advantage ranking.  The LSTM readout provides the realtime trajectory
form, while the pressure-habitat HGB result shows that the same survival
factors also form a strong phase-cut scoring baseline.
```

## Supplementary Relation Evidence

Main artifact:

```text
model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/
```

The O7 advantage-edge experiment is useful as supplementary reviewer defense,
not as the main thesis spine.

Hazard readout, B minus C edge control:

| Phase group | Positive cohorts | CI-positive cohorts | Median delta | Weighted mean delta |
| --- | ---: | ---: | ---: | ---: |
| All phases | 13 / 13 | 6 / 13 | 0.0023 | 0.0028 |
| Early phases 1-2 | 12 / 13 | 4 / 13 | 0.0027 | 0.0033 |
| Mid phases 3-4 | 8 / 13 | 3 / 13 | 0.0007 | 0.0018 |

Hazard-implied rank bridge, B minus C edge control:

| Phase group | Positive cohorts | CI-positive cohorts | Median delta | Weighted mean delta |
| --- | ---: | ---: | ---: | ---: |
| All phases | 9 / 13 | 7 / 13 | 0.0081 | 0.0149 |
| Early phases 1-2 | 9 / 13 | 5 / 13 | 0.0062 | 0.0064 |
| Mid phases 3-4 | 11 / 13 | 8 / 13 | 0.0168 | 0.0136 |

Use this only as:

```text
Supplementary edge-control evidence suggests that one relation stream contains
non-random signal under the current telemetry representation.
```

Do not use it as:

```text
Graph architecture is the core contribution, or relation topology is fully
validated.
```

## Claims To Use

Use these sentences as result-section anchors:

1. PUBG official telemetry can instantiate the proposed Battle Royale survival
   formulation under an explicit observability boundary.
2. Across independently trained cohorts, pressure and position-condition
   features provide the clearest incremental signal for next-phase survival.
3. Representation-rich survival features outperform prior-style score
   baselines and basic official-raw baselines under the same cohort contract.
4. Sequence models support a changing survival readout, but the strongest
   same-condition hazard scorer remains a tabular summary HGB baseline.
5. Phase-cut survival predictions can be converted into risk and advantage
   rankings that identify likely next-phase eliminations and strong final-rank
   candidates.
6. The realtime LSTM board is useful as a trajectory-producing readout, while
   the pressure-habitat HGB board marks the current upper baseline for phase-cut
   scoring.

## Claims To Avoid

Do not write:

| Avoided statement | Reason |
| --- | --- |
| Prior work did not study in-match or realtime prediction. | Too broad and unnecessary.  The paper should position prior work as using different prediction objects and feature organizations. |
| LSTM or ST-GNN-Mamba is the best model overall. | Same-condition tabular baselines are stronger on several hazard metrics. |
| The graph model validates the whole Battle Royale framework. | Current graph evidence is supplementary and bounded. |
| PUBG validates fine terrain, cover, or line-of-sight affordance. | PUBG official telemetry provides only proxies such as position and vertical coordinate signals. |
| The 10-second streaming deployment is fully validated. | The main aggregate result is phase-cut.  The 10-second prefix output is a technical demonstration unless separately evaluated. |
| All results use the whole raw-fetched DB. | The manuscript evidence uses the locked graph-eligible paper cohort and the 13 main independently trained cohorts. |

## Manuscript Table Plan

| Manuscript table / figure | Content |
| --- | --- |
| Table 4.1 | Locked data scope, date range, graph-eligible population, train/validation/test split |
| Table 4.2 | 13 main cohorts by map, perspective, team size, and match count |
| Table 5.1 | OM0-OM6 representation ladder across 13 cohorts, plus Erangel TPP squad representative row |
| Table 5.2 | Prior-style, official-raw, and representation-rich HGB/XGB baselines |
| Table 5.3 | Same-condition sequence and tabular model comparison |
| Table 5.4 | Representative Erangel neural model comparison with seed repeats |
| Table 5.5 | Risk / advantage board comparison on the audited intersected board universe |
| Figure 5.1 | Dynamic survival-share trajectory example |
| Supplementary table | O7 advantage-edge control results and O6 spatial-proxy boundary |

## Remaining Evidence Lock

Before final manuscript submission, do these as provenance work rather than new
scientific redesign:

1. Commit or archive the exact code and artifact manifests used for the current
   result tables.
2. Keep board metrics on the audited intersected board universe.
3. Preserve split manifests with provenance and `--require-existing-split` for
   all reruns.
4. If time allows, add a short sensitivity note for the telemetry resampling
   uncertainty around the 10-second grid.
5. Treat any new full experiment as an extension, not as a replacement for the
   current result chain unless it changes one of the table-level conclusions.
