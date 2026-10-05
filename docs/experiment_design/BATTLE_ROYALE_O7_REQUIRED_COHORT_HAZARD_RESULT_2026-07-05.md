# Battle Royale O7 Required-Cohort Hazard Result

Date: 2026-07-05
Class: `ExperimentResult`
Lifecycle: `active`

## Purpose

This document records the full 13-cohort hazard-AUC readout for the locked O7
advantage-edge A/B/C protocol.

Manuscript placement:

```text
Supplementary PUBG relation-topology null-control.
```

This is not the core paper spine.  The core spine is RQ1 genre framework, RQ2
PUBG operationalization, RQ3 prior tabular/phase-aware prediction plus
10-second realtime elimination-risk modeling, and RQ4 phase-transition
most-at-risk / most-advantaged team prediction.

It answers a narrow question:

```text
Do observed contact-gated relative-advantage edges improve hazard prediction
over an edge-shuffled null under the same node/global features, graph encoder,
edge gate, and training protocol?
```

This document is the hazard-AUC result.  The companion hazard-implied rank
bridge is now recorded in:

```text
docs/experiment_design/BATTLE_ROYALE_O7_REQUIRED_COHORT_RANK_BRIDGE_RESULT_2026-07-05.md
```

## Run Contract

| Arm | Condition | Interpretation |
| --- | --- | --- |
| `A` | `gate=none`, `control=observed` | node/global baseline |
| `B` | `gate=knn`, `control=observed` | observed advantage-edge graph |
| `C` | `gate=knn`, `control=edge_shuffle` | edge-shuffled null |

Primary topology-specific contrast:

```text
B - C
```

Secondary graph-capacity / smoothing contrast:

```text
B - A
```

Runner:

```text
scripts/run_br_o7_advantage_edge_early_readout.sh
```

Analyzers:

```text
tools/analyze_o7_advantage_readout.py
tools/analyze_o7_advantage_rank_bridge.py
tools/aggregate_o7_advantage_readout.py
```

## Completed Cohorts

All 13 required cohorts passed.

| Cohort | Slug | Status |
| --- | --- | --- |
| Erangel TPP squad | `erangel_tpp_squad` | passed |
| Taego TPP squad | `taego_tpp_squad` | passed |
| Erangel TPP solo | `erangel_tpp_solo` | passed |
| Miramar / Desert TPP squad | `miramar_desert_tpp_squad` | passed |
| Rondo TPP squad | `rondo_tpp_squad` | passed |
| Sanhok TPP squad | `sanhok_tpp_squad` | passed |
| Vikendi TPP squad | `vikendi_tpp_squad` | passed |
| Paramo TPP squad | `paramo_tpp_squad` | passed |
| Karakin TPP squad | `karakin_tpp_squad` | passed |
| Erangel TPP duo | `erangel_tpp_duo` | passed |
| Taego TPP duo | `taego_tpp_duo` | passed |
| Taego FPP squad | `taego_fpp_squad` | passed |
| Erangel FPP squad | `erangel_fpp_squad` | passed |

Run index:

```text
model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/early_readout_run_index.tsv
```

Aggregate artifacts:

```text
model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/o7_required_13cohort_phase_group_deltas.csv
model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/o7_required_13cohort_summary.json
```

Each cohort root contains:

```text
A_none_observed/pred_test.csv
B_knn_observed/pred_test.csv
C_knn_edge_shuffle/pred_test.csv
o7_early_readout_phase_group_deltas.csv
o7_early_readout_summary.json
o7_early_readout.log
```

## Macro Result

| Phase group | `B-C` positive cohorts | `B-C` CI-positive cohorts | Median `B-C` | Weighted mean `B-C` | `B-A` positive cohorts | `B-A` CI-positive cohorts | Median `B-A` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 13/13 | 6/13 | `+0.0023` | `+0.0028` | 11/13 | 7/13 | `+0.0070` |
| early `1-2` | 12/13 | 4/13 | `+0.0027` | `+0.0033` | 11/13 | 7/13 | `+0.0079` |
| mid `3-4` | 8/13 | 3/13 | `+0.0007` | `+0.0018` | 9/13 | 4/13 | `+0.0028` |
| late `5-6` | 9/13 | 3/13 | `+0.0038` | `+0.0069` | 9/13 | 3/13 | `+0.0070` |
| final `7` | 8/13 | 0/13 | `+0.0071` | `+0.0054` | 4/13 | 0/13 | `-0.0096` |

The pre-registered 9/13 sign threshold is met for hazard AUC in:

- overall phases: `13/13`;
- early phases `1-2`: `12/13`.

It is not met for the pre-registered mid phase group:

- mid phases `3-4`: `8/13`.

Late phases reach `9/13`, but late `5-6` was explicitly secondary /
exploratory in the O7 protocol.

## Primary Phase Detail

Early `1-2` hazard AUC `B-C`:

| Cohort | `B-C` | 95% match-bootstrap CI | `B-A` |
| --- | ---: | ---: | ---: |
| Erangel TPP squad | `+0.0025` | `[-0.0002, +0.0052]` | `+0.0089` |
| Taego TPP squad | `+0.0037` | `[+0.0006, +0.0068]` | `+0.0033` |
| Erangel TPP solo | `-0.0007` | `[-0.0064, +0.0048]` | `+0.0079` |
| Miramar / Desert TPP squad | `+0.0035` | `[-0.0002, +0.0071]` | `+0.0071` |
| Rondo TPP squad | `+0.0027` | `[-0.0015, +0.0067]` | `-0.0014` |
| Sanhok TPP squad | `+0.0090` | `[+0.0015, +0.0161]` | `+0.0146` |
| Vikendi TPP squad | `+0.0110` | `[+0.0024, +0.0188]` | `+0.0163` |
| Paramo TPP squad | `+0.0085` | `[-0.0022, +0.0198]` | `+0.0125` |
| Karakin TPP squad | `+0.0021` | `[-0.0057, +0.0093]` | `+0.0115` |
| Erangel TPP duo | `+0.0012` | `[-0.0063, +0.0092]` | `-0.0010` |
| Taego TPP duo | `+0.0022` | `[-0.0049, +0.0093]` | `+0.0207` |
| Taego FPP squad | `+0.0069` | `[+0.0003, +0.0142]` | `+0.0062` |
| Erangel FPP squad | `+0.0006` | `[-0.0095, +0.0094]` | `+0.0003` |

Mid `3-4` hazard AUC `B-C`:

| Cohort | `B-C` | 95% match-bootstrap CI | `B-A` |
| --- | ---: | ---: | ---: |
| Erangel TPP squad | `+0.0013` | `[-0.0017, +0.0042]` | `+0.0057` |
| Taego TPP squad | `+0.0007` | `[-0.0024, +0.0038]` | `+0.0017` |
| Erangel TPP solo | `+0.0056` | `[-0.0030, +0.0142]` | `+0.0153` |
| Miramar / Desert TPP squad | `+0.0016` | `[-0.0028, +0.0057]` | `+0.0030` |
| Rondo TPP squad | `-0.0004` | `[-0.0059, +0.0057]` | `-0.0032` |
| Sanhok TPP squad | `+0.0117` | `[+0.0044, +0.0190]` | `+0.0107` |
| Vikendi TPP squad | `+0.0123` | `[+0.0029, +0.0212]` | `+0.0176` |
| Paramo TPP squad | `+0.0305` | `[+0.0160, +0.0471]` | `-0.0045` |
| Karakin TPP squad | `-0.0002` | `[-0.0086, +0.0069]` | `+0.0105` |
| Erangel TPP duo | `+0.0001` | `[-0.0055, +0.0050]` | `-0.0005` |
| Taego TPP duo | `-0.0081` | `[-0.0149, -0.0011]` | `+0.0026` |
| Taego FPP squad | `-0.0022` | `[-0.0095, +0.0051]` | `+0.0028` |
| Erangel FPP squad | `-0.0039` | `[-0.0119, +0.0039]` | `-0.0078` |

## Interpretation

The hazard-AUC evidence supports an O7 relation-structure effect for this
specific edge stream:

```text
Observed contact-gated relative-advantage edge topology improves hazard AUC
over an edge-shuffled null in the required-cohort set, especially in early
phases.
```

Why this is stronger than a pure capacity/smoothing result:

- `B-C`, the topology-specific contrast, is positive in 13/13 cohorts overall;
- `B-C` is positive in 12/13 cohorts in early phases `1-2`, crossing the
  pre-registered 9/13 hazard-AUC sign threshold;
- `B-A` is also mostly positive, but the primary result is not reducible to
  adding a graph encoder because `C` has the same graph pathway and edge gate.

Important limits:

- this document's main table is hazard AUC only;
- the companion rank bridge is hazard-implied, not an independent rank-head
  metric;
- bootstrap support is concentrated rather than universal: early `B-C` has
  CI-positive support in 4/13 cohorts;
- mid `3-4` does not reach the 9/13 sign threshold;
- late `5-6` reaches 9/13 but was secondary / exploratory;
- phase `7` remains too sparse and unstable for a headline claim.

## Claim Boundary

Safe wording:

```text
As supplementary PUBG evidence, observed contact-gated relative-advantage
edges show topology-specific hazard-AUC signal over an edge-shuffled null, with
the clearest confirmatory support in early phases.
```

Avoid:

```text
O7 is fully validated.
```

Avoid:

```text
O7 is a core research-question answer of the paper.
```

Avoid:

```text
All graph relation structure is predictive.
```

Avoid:

```text
High-ground / terrain affordance is validated by O7.
```

The result supports one relation-structure stream in PUBG under this model and
null-control design.  It does not rescue the corrected O6Z vertical-proxy
negative result, and it does not generalize by itself to all relation families.

## Companion Rank Bridge

The hazard-implied rank bridge aligns with the hazard-AUC result:

| Phase group | `B-C` positive cohorts | CI-positive cohorts | Median `B-C` |
| --- | ---: | ---: | ---: |
| early `1-2` | 9/13 | 5/13 | `+0.0062` |
| mid `3-4` | 11/13 | 8/13 | `+0.0168` |

The correct combined interpretation is:

```text
O7 advantage-edge topology is positive for hazard AUC and for a
hazard-implied rank bridge, but not for an independent rank-head metric.
```
