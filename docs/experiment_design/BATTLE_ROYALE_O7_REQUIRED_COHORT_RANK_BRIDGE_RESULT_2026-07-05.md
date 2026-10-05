# Battle Royale O7 Required-Cohort Rank Bridge Result

Date: 2026-07-05
Class: `ExperimentResult`
Lifecycle: `active`

## Purpose

This document records the rank bridge for the full 13-cohort O7
advantage-edge A/B/C run.

Manuscript placement:

```text
Supplementary PUBG relation-topology null-control.
```

This is not the core paper spine.  It is reviewer-defense material for the
question of whether a specific PUBG relation stream survives an edge-shuffled
null.

Important scope boundary:

```text
The O7 advantage-edge model is hazard-only.  It has no rank head.
```

Therefore this analysis is not an independent rank-head Spearman metric.  It is
a hazard-implied rank bridge:

```text
Spearman(final_rank_norm, hazard_prob)
```

Because `final_rank_norm` increases with worse final placement, a larger
positive Spearman means that the phase-cut hazard probability also orders teams
toward worse final rank.

## Artifacts

Analyzer:

```text
tools/analyze_o7_advantage_rank_bridge.py
scripts/run_br_o7_rank_bridge_analysis.sh
```

Aggregate outputs:

```text
model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/o7_required_13cohort_rank_bridge_phase_group_deltas.csv
model_stgcn/checkpoints/br_o7_advantage_edge_required_v1/o7_required_13cohort_rank_bridge_summary.json
```

Per-cohort outputs:

```text
o7_rank_bridge_phase_group_deltas.csv
o7_rank_bridge_summary.json
```

Bootstrap:

```text
B = 200 match-cluster bootstrap resamples
```

This bridge reuses the existing A/B/C `pred_test.csv` files and attaches final
rank labels from the factorized phase data.

## Macro Result

| Phase group | `B-C` positive cohorts | `B-C` CI-positive cohorts | Median `B-C` | Weighted mean `B-C` | `B-A` positive cohorts | Median `B-A` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 9/13 | 7/13 | `+0.0081` | `+0.0149` | 8/13 | `+0.0088` |
| early `1-2` | 9/13 | 5/13 | `+0.0062` | `+0.0064` | 8/13 | `+0.0039` |
| mid `3-4` | 11/13 | 8/13 | `+0.0168` | `+0.0136` | 7/13 | `+0.0009` |
| late `5-6` | 11/13 | 5/13 | `+0.0221` | `+0.0214` | 8/13 | `+0.0124` |
| final `7` | 8/13 | 2/13 | `+0.0164` | `+0.0351` | 5/13 | `-0.0201` |

The pre-registered 9/13 sign threshold is met for the hazard-implied rank
bridge in both primary phase groups:

- early `1-2`: `9/13`;
- mid `3-4`: `11/13`.

Mid `3-4` is the stronger bridge cell, with CI-positive support in `8/13`
cohorts.

## Primary Phase Detail

Early `1-2` hazard-implied rank `B-C`:

| Cohort | `B-C` | 95% match-bootstrap CI |
| --- | ---: | ---: |
| Erangel TPP squad | `+0.0062` | `[+0.0014, +0.0107]` |
| Taego TPP squad | `+0.0186` | `[+0.0133, +0.0225]` |
| Erangel TPP solo | `-0.0190` | `[-0.0298, -0.0100]` |
| Miramar / Desert TPP squad | `+0.0042` | `[-0.0036, +0.0099]` |
| Rondo TPP squad | `-0.0020` | `[-0.0087, +0.0040]` |
| Sanhok TPP squad | `+0.0151` | `[+0.0077, +0.0249]` |
| Vikendi TPP squad | `+0.0113` | `[-0.0031, +0.0274]` |
| Paramo TPP squad | `+0.0183` | `[+0.0056, +0.0349]` |
| Karakin TPP squad | `+0.0062` | `[-0.0100, +0.0173]` |
| Erangel TPP duo | `-0.0291` | `[-0.0385, -0.0189]` |
| Taego TPP duo | `+0.0104` | `[-0.0037, +0.0267]` |
| Taego FPP squad | `-0.0067` | `[-0.0213, +0.0071]` |
| Erangel FPP squad | `+0.0184` | `[+0.0045, +0.0311]` |

Mid `3-4` hazard-implied rank `B-C`:

| Cohort | `B-C` | 95% match-bootstrap CI |
| --- | ---: | ---: |
| Erangel TPP squad | `+0.0191` | `[+0.0140, +0.0244]` |
| Taego TPP squad | `+0.0190` | `[+0.0146, +0.0246]` |
| Erangel TPP solo | `+0.0230` | `[+0.0081, +0.0387]` |
| Miramar / Desert TPP squad | `+0.0022` | `[-0.0047, +0.0095]` |
| Rondo TPP squad | `+0.0198` | `[+0.0110, +0.0297]` |
| Sanhok TPP squad | `+0.0121` | `[+0.0008, +0.0230]` |
| Vikendi TPP squad | `+0.0188` | `[+0.0001, +0.0304]` |
| Paramo TPP squad | `+0.0341` | `[+0.0168, +0.0554]` |
| Karakin TPP squad | `+0.0088` | `[-0.0052, +0.0218]` |
| Erangel TPP duo | `-0.0262` | `[-0.0367, -0.0166]` |
| Taego TPP duo | `+0.0059` | `[-0.0064, +0.0165]` |
| Taego FPP squad | `+0.0168` | `[+0.0018, +0.0303]` |
| Erangel FPP squad | `-0.0228` | `[-0.0387, -0.0091]` |

## Interpretation

Together with the hazard-AUC result, this bridge strengthens the O7
advantage-edge finding:

```text
Observed contact-gated relative-advantage edges improve next-phase hazard
prediction over an edge-shuffled null, and those hazard probabilities also
carry final-rank ordering signal in the primary early/mid phase groups.
```

Why this is not merely graph capacity:

- the primary contrast is `B-C`, where `B` and `C` share the graph pathway and
  differ by observed versus edge-shuffled topology;
- early `B-C` reaches `9/13`;
- mid `B-C` reaches `11/13` and has CI-positive support in `8/13`.

Important limits:

- this is a hazard-implied rank bridge, not a separate rank-head model;
- Erangel TPP duo is negative in both primary bridge cells;
- Erangel FPP squad is negative in mid `3-4`;
- Erangel TPP solo is negative in early `1-2`;
- final phase `7` remains sparse and unstable.

## Claim Boundary

Safe wording:

```text
As supplementary PUBG evidence, the gen-2 O7 advantage-edge run supports
topology-specific relation signal in hazard prediction and in a hazard-implied
final-rank ordering bridge.
```

Avoid:

```text
The O7 model's independent rank head validates relation topology.
```

There is no independent rank head in the O7 advantage-edge model.

Avoid:

```text
All O7 relation structures are validated.
```

Avoid:

```text
O7 is a core research-question answer of the paper.
```

The result is specific to observed contact-gated relative-advantage edges under
the current null-control design.
