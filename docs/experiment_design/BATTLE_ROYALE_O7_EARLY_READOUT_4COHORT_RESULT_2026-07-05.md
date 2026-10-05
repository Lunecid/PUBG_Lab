# Battle Royale O7 Early Readout: Four-Cohort Result

Date: 2026-07-05
Class: `ExperimentResult`
Lifecycle: `active`

## Purpose

This document summarizes the high-power early readout for the locked O7
advantage-edge A/B/C protocol.

The goal is to decide whether the O7 run should expand to all 13 required
cohorts.  This is not yet the final O7 result.

## Completed Cohorts

| Cohort | Status | Root |
| --- | --- | --- |
| Erangel TPP squad | passed | `model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/erangel_tpp_squad/` |
| Taego TPP squad | passed | `model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/taego_tpp_squad/` |
| Erangel TPP solo | passed | `model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/erangel_tpp_solo/` |
| Miramar / Desert TPP squad | passed | `model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/miramar_desert_tpp_squad/` |

Run index:

```text
model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/early_readout_run_index.tsv
```

Aggregates:

```text
model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/o7_early_readout_4cohort_phase_group_deltas.csv
model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/o7_early_readout_4cohort_summary.json
```

## A/B/C Contract

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

## Macro Result

| Phase group | `B-C` positive cohorts | Median `B-C` | Range `B-C` | `B-A` positive cohorts | Median `B-A` |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 4/4 | `+0.0023` | `+0.0008` to `+0.0027` | 4/4 | `+0.0062` |
| early `1-2` | 3/4 | `+0.0030` | `-0.0007` to `+0.0037` | 4/4 | `+0.0075` |
| mid `3-4` | 4/4 | `+0.0014` | `+0.0007` to `+0.0056` | 4/4 | `+0.0044` |
| late `5-6` | 3/4 | `+0.0080` | `-0.0304` to `+0.0132` | 3/4 | `+0.0093` |
| final `7` | 3/4 | `+0.0132` | `-0.0138` to `+0.3214` | 1/4 | `-0.0066` |

## Interpretation

The four-cohort early readout is positive enough to justify expanding to all
13 required cohorts:

- the topology-specific contrast `B-C` is positive in 4/4 cohorts overall;
- the pre-registered mid phase group is positive in 4/4 cohorts;
- the pre-registered early phase group is positive in 3/4 cohorts;
- the broader graph-stream contrast `B-A` is positive in 4/4 cohorts overall,
  early, and mid.

But this is still not a final O7 claim:

- the pre-registered support threshold is defined over 13 cohorts, not 4;
- several cohort-level early/mid bootstrap intervals still cross zero;
- the current analyzer covers hazard AUC only, while the protocol also names
  rank Spearman as a primary metric.

Safe wording:

```text
The high-power early readout supports continuing the O7 advantage-edge run to
the full required-cohort set.
```

Avoid:

```text
O7 advantage-edge topology is validated.
```

## Decision

Proceed to the remaining 9 required cohorts.

If the 13-cohort result preserves a positive `B-C` sign pattern in early or mid
phase groups, run the full paired bootstrap / rank bridge analysis before
making any manuscript claim.
