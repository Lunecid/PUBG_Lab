# Battle Royale Required Cohort Set

Date: 2026-07-03
Class: `ExperimentContract`
Lifecycle: `active`

## Purpose

This document defines the required cohort set for OM0-OM6 training,
ablation, and robustness experiments.

The key rule is:

```text
Do not merge cohorts for training.
Train and evaluate each required map / perspective / team-size cohort
separately.
```

Aggregation is allowed only after cohort-level results exist.

## Naming Rule

When discussing cohorts in prose, use the explicit cohort description:

```text
<map label> <perspective> <team size>
```

Examples:

- Erangel TPP squad
- Taego FPP squad
- Erangel TPP solo
- Taego TPP duo

Internal cohort IDs from the registry may be used in scripts or machine-readable
tables, but paper planning and discussion should name the map, perspective, and
team size directly.

## Decision Summary

| Role | Required set | Training rule |
| --- | --- | --- |
| required main training cohorts | 13 map / perspective / team-size cohorts | train OM0-OM6 separately for each cohort |
| required squad map analysis | 8 TPP squad map cohorts | train separately, then summarize map variation |
| required perspective analysis | Erangel TPP squad vs Erangel FPP squad; Taego TPP squad vs Taego FPP squad | train each cohort separately, compare same-map results |
| required team-size analysis | Erangel TPP squad / duo / solo; Taego TPP squad / duo | train each cohort separately, compare same-map results |
| auxiliary cohorts | all below-threshold cohorts | optional sensitivity only |

No required experiment should train one model on a pooled mixture of maps,
perspectives, or team sizes.

## Required Independent Training Cohorts

Each row below is a separate OM0-OM6 training and evaluation job.

| Map | Perspective | Team size | Train | Val | Test | Total | Required purpose |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Erangel | TPP | squad | 4,998 | 568 | 638 | 6,204 | largest team-relation anchor |
| Taego | TPP | squad | 3,324 | 431 | 404 | 4,159 | second largest team-relation anchor |
| Miramar / Desert | TPP | squad | 1,210 | 158 | 175 | 1,543 | squad map variation |
| Rondo | TPP | squad | 838 | 102 | 99 | 1,039 | squad map variation |
| Sanhok | TPP | squad | 743 | 109 | 88 | 940 | compact-map squad variation |
| Vikendi | TPP | squad | 398 | 51 | 49 | 498 | squad map variation |
| Paramo | TPP | squad | 265 | 32 | 36 | 333 | small-map squad variation above threshold |
| Karakin | TPP | squad | 270 | 31 | 32 | 333 | small-map squad variation above threshold |
| Erangel | TPP | solo | 643 | 80 | 77 | 800 | solo boundary and Sagong-adjacent comparison |
| Erangel | TPP | duo | 529 | 67 | 60 | 656 | same-map team-size contrast |
| Taego | TPP | duo | 456 | 64 | 53 | 573 | second-map team-size contrast |
| Taego | FPP | squad | 335 | 58 | 42 | 435 | same-map perspective contrast |
| Erangel | FPP | squad | 346 | 36 | 48 | 430 | same-map perspective contrast |

Total required cohort coverage:

```text
13 independently trained cohorts
17,943 graph-eligible matches
```

## Required Protocol

### P0. Per-Cohort OM Ladder

For every required cohort, train and evaluate:

```text
OM0, OM1, OM2, OM3, OM4, OM5, OM6
```

Each cohort keeps its own train/val/test split from the master split
intersection.  No model should be trained on a pooled multi-cohort dataset for
the main ablation.

Required output:

```text
map, perspective, team_size, OM block, metric, delta_vs_previous, delta_vs_OM0
```

### P1. Squad Map Analysis

Use the eight TPP squad cohorts:

| Map | Perspective | Team size |
| --- | --- | --- |
| Erangel | TPP | squad |
| Taego | TPP | squad |
| Miramar / Desert | TPP | squad |
| Rondo | TPP | squad |
| Sanhok | TPP | squad |
| Vikendi | TPP | squad |
| Paramo | TPP | squad |
| Karakin | TPP | squad |

Purpose:

```text
Check whether OM deltas are consistent across maps in the main team setting.
```

### P2. Perspective Analysis

Use same-map squad comparisons:

| Map | TPP cohort | FPP cohort |
| --- | --- | --- |
| Erangel | Erangel TPP squad | Erangel FPP squad |
| Taego | Taego TPP squad | Taego FPP squad |

Purpose:

```text
Check whether the ontology-derived feature gains survive camera perspective
shift without pooling FPP and TPP in training.
```

### P3. Team-Size Analysis

Use same-map comparisons:

| Map | Cohorts |
| --- | --- |
| Erangel | Erangel TPP squad, Erangel TPP duo, Erangel TPP solo |
| Taego | Taego TPP squad, Taego TPP duo |

Purpose:

```text
Check how team-level relation features behave when the number of agents per
team changes.
```

### P4. Summary Reporting

After all required cohort-level runs are complete, summary reporting may include:

- macro average across the 13 required cohorts;
- match-count weighted average across the 13 required cohorts;
- macro / weighted summary across the eight TPP squad map cohorts;
- same-map contrast tables for perspective and team-size effects.

These summaries are reports over separately trained cohorts.  They are not
pooled-training results.

## Optional Sensitivity

The following are not required before the first paper-ready OM table:

- below-threshold auxiliary cohorts;
- held-out-map transfer;
- train-on-squad and test-on-solo/duo transfer;
- Caldera supervised prediction.

They may be added only after the required independent cohort runs are complete.

## Immediate Consequence

The next checklist item remains:

```text
Feature availability audit for OM0-OM6 on required cohorts
```

The audit must answer, for each explicit cohort such as Erangel TPP squad or
Taego FPP squad, whether every OM block can be computed without leakage.
