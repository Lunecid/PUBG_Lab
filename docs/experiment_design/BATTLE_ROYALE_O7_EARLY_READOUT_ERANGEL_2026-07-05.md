# Battle Royale O7 Early Readout: Erangel TPP Squad

Date: 2026-07-05
Class: `ExperimentResult`
Lifecycle: `active`

## Purpose

This is the first high-power gen-2 O7 advantage-edge readout after the
gen-1/gen-2 boundary decision and the edge-shuffle audit.

It tests the locked A/B/C contract on one required cohort:

| Arm | Condition | Interpretation |
| --- | --- | --- |
| `A` | `gate=none`, `control=observed` | node/global baseline |
| `B` | `gate=knn`, `control=observed` | observed advantage-edge graph |
| `C` | `gate=knn`, `control=edge_shuffle` | edge-shuffled null |

Primary O7 evidence is:

```text
B - C
```

This document is an early readout only.  It is not yet a 13-cohort O7 result.

## Run Contract

| Field | Value |
| --- | --- |
| Cohort | Erangel TPP squad |
| Root | `model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/erangel_tpp_squad/` |
| Split JSON | `model_stgcn/checkpoints/br_om_ladder_required_v1/erangel_tpp_squad/split.json` |
| Train / val / test matches | `4964 / 620 / 620` |
| Feature preset | `om5_relation_structure` |
| Node absolute control features | enabled |
| Device | `mps` |
| Epochs | 12 |
| Seed | 42 |
| Test cells | 31,404 |

The pre-run edge-shuffle audit also passed on this cohort:

| Field | Value |
| --- | ---: |
| Matches audited | 8 |
| Phase blocks | 53 |
| Edge-containing blocks | 53 |
| Rewired blocks | 52 |
| Active edges checked | 3,009 |
| Failures | 0 |

## Artifacts

| Artifact | Role |
| --- | --- |
| `A_none_observed/pred_test.csv` | node/global baseline predictions |
| `B_knn_observed/pred_test.csv` | observed advantage-edge predictions |
| `C_knn_edge_shuffle/pred_test.csv` | edge-shuffled null predictions |
| `o7_early_readout_phase_group_deltas.csv` | phase-group AUC and delta table |
| `o7_early_readout_summary.json` | machine-readable summary |
| `o7_early_readout.log` | run log |

Analyzer:

```text
tools/analyze_o7_advantage_readout.py
```

## Result

Overall test hazard AUC:

| Arm | AUC |
| --- | ---: |
| `A_none_observed` | `0.7690` |
| `B_knn_observed` | `0.7761` |
| `C_knn_edge_shuffle` | `0.7734` |

Phase-group deltas:

| Phase group | `B - C` | 95% match-bootstrap CI | `B - A` | 95% match-bootstrap CI |
| --- | ---: | ---: | ---: | ---: |
| all | `+0.0027` | `[+0.0009, +0.0043]` | `+0.0071` | `[+0.0052, +0.0090]` |
| early `1-2` | `+0.0025` | `[-0.0002, +0.0052]` | `+0.0089` | `[+0.0060, +0.0120]` |
| mid `3-4` | `+0.0013` | `[-0.0017, +0.0042]` | `+0.0057` | `[+0.0022, +0.0093]` |
| late `5-6` | `+0.0038` | `[-0.0026, +0.0113]` | `+0.0071` | `[+0.0001, +0.0145]` |
| final `7` | `+0.0185` | `[-0.0123, +0.0505]` | `-0.0037` | `[-0.0429, +0.0349]` |

## Interpretation

The first required-cohort O7 readout is directionally positive:

- observed advantage edges beat the edge-shuffled null overall;
- observed advantage edges also beat the no-graph baseline overall and in the
  early/mid phase groups;
- the pre-registered early and mid `B - C` cells have positive point estimates
  but their CIs still cross zero.

Safe wording:

```text
Erangel TPP squad gives a positive first readout for observed advantage-edge
structure, but it is not sufficient for an O7 claim.
```

Avoid:

```text
O7 advantage-edge topology is validated.
```

The locked O7 support rule still requires the broader required-cohort result,
including the sign threshold and phase-group interpretation.

## Next Action

Run the remaining high-power early readout cohorts:

```text
Taego TPP squad
Erangel TPP solo
Miramar / Desert TPP squad
```
