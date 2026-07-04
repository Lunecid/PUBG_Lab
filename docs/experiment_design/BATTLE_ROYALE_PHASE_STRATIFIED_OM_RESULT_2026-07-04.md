# Battle Royale Phase-Stratified OM Result

Date: 2026-07-04
Class: `ExperimentResult`
Lifecycle: `active`

## Purpose

This document records the phase-stratified analysis of the completed
required-cohort OM0-OM6 ladder.

The goal is to answer:

```text
Does the OM3 pressure-habitat gain remain visible within cut phases, or is the
pooled OM3 gain mostly a between-phase base-rate effect?
```

## Analysis Contract

This is not a pooled retraining run.

Training remains separated by exact map / perspective / team-size cohort:

- Erangel TPP squad;
- Taego TPP squad;
- Miramar / Desert TPP squad;
- Rondo TPP squad;
- Sanhok TPP squad;
- Vikendi TPP squad;
- Paramo TPP squad;
- Karakin TPP squad;
- Erangel TPP solo;
- Erangel TPP duo;
- Taego TPP duo;
- Taego FPP squad;
- Erangel FPP squad.

This analysis reuses the already trained cohort-specific OM0-OM6 checkpoints
and the `phase_metrics` stored in each cohort-specific `eval_test.json`.

No model was retrained and no map / perspective / team-size cohorts were mixed
during learning.

## Artifacts

Source run root:

```text
model_stgcn/checkpoints/br_om_ladder_required_v1
```

Generated artifacts:

| Artifact | Role |
| --- | --- |
| `tools/collect_om_phase_metrics.py` | collector that flattens `phase_metrics` and computes same-cohort, same-phase OM deltas |
| `model_stgcn/checkpoints/br_om_ladder_required_v1/phase_stratified_metrics.csv` | flattened phase metrics for every cohort / OM preset / cut phase |
| `model_stgcn/checkpoints/br_om_ladder_required_v1/phase_stratified_deltas.csv` | sequential OM deltas by cohort and cut phase |
| `model_stgcn/checkpoints/br_om_ladder_required_v1/phase_stratified_delta_summary.json` | summary of phase-level and phase-group deltas |

Row counts:

| Artifact | Rows | Expected structure |
| --- | ---: | --- |
| phase metrics | 637 | 13 cohorts x 7 OM presets x 7 cut phases |
| phase deltas | 546 | 13 cohorts x 7 cut phases x 6 sequential OM steps |

## Headline Finding

The OM3 pressure-habitat gain is not only a pooled phase-base-rate effect.

Within cut phases, `OM2 movement -> OM3 pressure / habitat` is positive in
most cohort-phase cells:

```text
67/91 cohort-phase cells have positive hazard AUC gain.
```

The gain is strongest and most stable in early phases, remains positive in
mid phases and phase 5, is mixed but still positive at the median in phase 6,
and becomes unreliable in phase 7 because the at-risk sample is small.

This supports the analytic-framework claim:

```text
Contraction / elimination pressure changes the survival environment in a way
that provides predictive information beyond context, survival capital, and
movement.
```

## OM2 To OM3 Hazard AUC Delta By Cut Phase

Sequential delta:

```text
OM2 movement -> OM3 pressure / habitat
```

| Cut phase | Total at-risk team-phase N | Median cohort N | Positive cohorts | Median Delta AUC | Mean Delta AUC | Interpretation |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 32,965 | 1,526 | 13/13 | `+0.0154` | `+0.0202` | strongest early evidence |
| 2 | 25,045 | 1,051 | 11/13 | `+0.0168` | `+0.0147` | strong early evidence |
| 3 | 19,441 | 808 | 10/13 | `+0.0066` | `+0.0073` | smaller but positive mid-phase signal |
| 4 | 13,551 | 627 | 10/13 | `+0.0075` | `+0.0074` | smaller but positive mid-phase signal |
| 5 | 7,531 | 385 | 12/13 | `+0.0116` | `+0.0097` | late survival pressure remains informative |
| 6 | 3,957 | 220 | 7/13 | `+0.0068` | `+0.0165` | mixed; compact cohorts can show large gains |
| 7 | 1,370 | 88 | 4/13 | `-0.0345` | `-0.0161` | sparse tail; not a stable claim target |

Phase-group summary:

| Phase group | Cohort-phase cells | Positive cells | Median Delta AUC | Mean Delta AUC | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| early phases 1-2 | 26 | 24/26 | `+0.0161` | `+0.0175` | strongest framework support |
| mid phases 3-4 | 26 | 20/26 | `+0.0070` | `+0.0074` | within-phase signal persists |
| late phases 5-7 | 39 | 23/39 | `+0.0078` | `+0.0034` | phase 5/6 signal, phase 7 instability |

## Rank And Top-10 Link

The OM3 block also improves final-rank related signals in earlier and mid cut
phases.

`OM2 -> OM3` rank Spearman delta:

| Cut phase | Positive cohorts | Median Delta Rank Spearman | Mean Delta Rank Spearman |
| ---: | ---: | ---: | ---: |
| 1 | 13/13 | `+0.0344` | `+0.0451` |
| 2 | 13/13 | `+0.0501` | `+0.0577` |
| 3 | 13/13 | `+0.0708` | `+0.0675` |
| 4 | 11/13 | `+0.0544` | `+0.0550` |
| 5 | 10/13 | `+0.0359` | `+0.0094` |
| 6 | 5/13 | `-0.0494` | `-0.0722` |
| 7 | 6/13 | `-0.0068` | `-0.1059` |

`OM2 -> OM3` top-10 AUC delta:

| Cut phase | Valid cohorts | Positive cohorts | Median Delta Top10 AUC | Mean Delta Top10 AUC |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 13 | 13/13 | `+0.0452` | `+0.0469` |
| 2 | 13 | 13/13 | `+0.0592` | `+0.0562` |
| 3 | 11 | 11/11 | `+0.0797` | `+0.0720` |
| 4 | 10 | 9/10 | `+0.0738` | `+0.0647` |
| 5 | 7 | 7/7 | `+0.0352` | `+0.0584` |
| 6 | 1 | 1/1 | `+0.0550` | `+0.0550` |

Top-10 AUC becomes undefined in many late-phase cells because almost all
remaining at-risk teams are already top-10.  Therefore late top-10 deltas
should be treated as descriptive only.

## Sequential Phase-Level Block Pattern

Across all 91 cohort-phase cells:

| Step | Positive cells | Median Delta Hazard AUC | Mean Delta Hazard AUC | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `OM0 -> OM1` | 41/91 | `-0.0002` | `+0.0010` | no stable pooled phase-level contribution |
| `OM1 -> OM2` | 63/91 | `+0.0030` | `+0.0045` | small movement gain |
| `OM2 -> OM3` | 67/91 | `+0.0114` | `+0.0085` | strongest phase-level mechanism block |
| `OM3 -> OM4` | 63/91 | `+0.0032` | `+0.0013` | threat exposure becomes more useful after early phase |
| `OM4 -> OM5` | 43/91 | `-0.0007` | `+0.0041` | relation node-summary signal is unstable |
| `OM5 -> OM6` | 54/91 | `+0.0011` | `+0.0001` | current spatial proxy adds little |

## Environment-Conditioned Interaction Pattern

The phase-stratified result strengthens the earlier interpretation that
interaction and relation features are conditionally useful.

`OM3 -> OM4` threat exposure is weak in phase 1 but grows in phases 3-6:

| Cut phase | Positive cohorts | Median Delta AUC | Mean Delta AUC |
| ---: | ---: | ---: | ---: |
| 1 | 6/13 | `-0.0003` | `+0.0028` |
| 2 | 8/13 | `+0.0026` | `+0.0035` |
| 3 | 11/13 | `+0.0016` | `+0.0073` |
| 4 | 12/13 | `+0.0060` | `+0.0067` |
| 5 | 12/13 | `+0.0064` | `+0.0115` |
| 6 | 8/13 | `+0.0031` | `+0.0059` |
| 7 | 6/13 | `-0.0004` | `-0.0285` |

This should be framed as a secondary pattern:

```text
Threat exposure appears more useful once contraction has produced denser and
more constrained survival environments, but this requires paired uncertainty
checks before becoming a strong manuscript claim.
```

The current OM5 relation node-summary block remains unstable.  Its best gains
are concentrated in specific late or compact settings, but this is not yet a
full test of O7 relation modeling because the run used `graph_encoder=none` and
no edge stream.

## Interpretation

The phase-stratified result changes the OM3 interpretation in an important way.

Before this analysis, the main concern was:

```text
OM3 may look strong only because pooled evaluation rewards models that encode
phase-level base-rate differences.
```

The phase-specific deltas reduce that concern.  OM3 improves hazard AUC within
the same cut phase, especially phases 1-5, after movement has already been
included.  Therefore, the pressure-habitat block is not merely a pooled phase
clock proxy.

The strongest supported claim is:

```text
The pressure-habitat mechanism contributes within-phase predictive signal,
especially before the sparse final tail of the match.
```

The claim should not be:

```text
OM3 is uniformly beneficial in every phase of every cohort.
```

Phase 7 is too sparse and unstable for that claim.

## Claim Boundary

Allowed:

```text
In phase-stratified evaluation, the pressure-habitat block remains positive in
most cohort-phase cells and is strongest in early-to-mid phases.  This supports
the framework view that contraction pressure changes the survival environment
in a predictively useful way beyond movement alone.
```

Avoid:

```text
The phase-stratified analysis proves the causal effect of habitat.
```

Avoid:

```text
Late-phase results are stable across all phases.
```

## Next Work

The next experiment is paired uncertainty:

```text
match-level paired bootstrap confidence intervals for OM deltas
```

Priority targets:

1. `OM2 -> OM3` hazard AUC by phase group;
2. `OM2 -> OM3` rank Spearman and top-10 AUC by phase where defined;
3. `OM3 -> OM4` in phases 3-6;
4. small-cohort sensitivity for Karakin TPP squad, Paramo TPP squad, Taego FPP
   squad, and Erangel FPP squad.

