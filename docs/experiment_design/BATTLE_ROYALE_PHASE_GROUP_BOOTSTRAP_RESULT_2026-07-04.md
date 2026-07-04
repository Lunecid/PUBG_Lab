# Battle Royale Phase-Group Bootstrap Result

Date: 2026-07-04
Class: `ExperimentResult`
Lifecycle: `active`

## Purpose

This document records the paired uncertainty check for the completed
required-cohort OM0-OM6 ladder and the phase-stratified mechanism result.

The analysis asks:

```text
Do the observed OM deltas remain positive after match-cluster paired bootstrap
within exact map / perspective / team-size cohorts and phase groups?
```

## Analysis Contract

No model was retrained for this analysis.

The analysis reuses the cohort-specific checkpoints from:

```text
model_stgcn/checkpoints/br_om_ladder_required_v1
```

Training and evaluation remain separated by exact cohort:

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

The bootstrap unit is the test match.  Each resample draws test matches with
replacement inside a cohort, preserving all team-phase rows belonging to the
sampled match.  Deltas are paired because the compared OM presets are evaluated
on the same test matches.

## Artifacts

Generated prediction artifacts:

| Artifact | Role |
| --- | --- |
| `tools/generate_om_prediction_csvs.py` | re-evaluates saved OM checkpoints with row-level prediction output |
| `model_stgcn/checkpoints/br_om_ladder_required_v1/predictions/` | 91 prediction CSVs, one per cohort and OM preset |

Generated bootstrap artifacts:

| Artifact | Role |
| --- | --- |
| `tools/bootstrap_om_paired_deltas.py` | paired match-cluster bootstrap over prediction CSVs |
| `model_stgcn/checkpoints/br_om_ladder_required_v1/bootstrap_phase_group_core_b2000/om_paired_bootstrap_phase_groups.csv` | phase-group paired bootstrap results |
| `model_stgcn/checkpoints/br_om_ladder_required_v1/bootstrap_phase_group_core_b2000/om_paired_bootstrap_all.csv` | all-scope paired bootstrap results |
| `model_stgcn/checkpoints/br_om_ladder_required_v1/bootstrap_phase_group_core_b2000/om_paired_bootstrap_summary.json` | compact summary |

Run settings:

| Setting | Value |
| --- | --- |
| Bootstrap repetitions | `B=2000` |
| Seed | `42` |
| Phase groups | early `1-2`, mid `3-4`, late `5-6` |
| Excluded phase | phase `7` |
| Metrics | `hazard_auc`, `rank_spearman`, `top10_auc` |
| Tested steps | `OM0 -> OM1`, `OM2 -> OM3`, `OM3 -> OM4` |
| Prediction CSV count | `91` |

Phase 7 is excluded from the bootstrap claim target because the phase-stratified
audit showed sparse and unstable at-risk samples in the final tail.

## Confirmatory Target Cells

The confirmatory target cells for manuscript claims are the cells motivated by
the prior phase-stratified analysis:

- `OM2 -> OM3` early phases `1-2`, `hazard_auc`;
- `OM2 -> OM3` early phases `1-2`, `rank_spearman`;
- `OM2 -> OM3` early phases `1-2`, `top10_auc`;
- `OM2 -> OM3` mid phases `3-4`, `rank_spearman`;
- `OM2 -> OM3` mid phases `3-4`, `top10_auc`.

The full 27-cell table is retained for transparency, but non-target cells
should be treated as exploratory or diagnostic.  This matters because the table
contains multiple steps, phase groups, and metrics; the paper should not imply
that every supported cell was selected before inspection.

## Headline Finding

The paired bootstrap supports the central pressure-habitat result.

The strongest evidence is:

```text
OM2 movement -> OM3 pressure / habitat
```

After movement has already been included, adding contraction-pressure and
habitat features improves:

- early next-phase hazard AUC in `13/13` cohorts, with `10/13` cohort CIs
  supporting a positive gain;
- early rank Spearman in `13/13` cohorts, with `12/13` cohort CIs supporting a
  positive gain;
- early top-10 AUC in `13/13` cohorts, with `11/13` cohort CIs supporting a
  positive gain;
- mid-phase rank and top-10 evaluation with large median deltas and broad CI
  support.

This supports the manuscript-safe statement:

```text
Match-cluster paired bootstrap confirms that adding pressure-habitat features
after movement yields robust early-phase hazard gains and mid-phase rank/top-10
gains across cohort-separated PUBG settings.
```

## Summary Table

`Supported cohorts` means the paired bootstrap interval excludes zero in the
direction where the later OM preset is better.

| Step | Phase group | Metric | Median delta | Positive cohorts | Supported cohorts |
| --- | --- | --- | ---: | ---: | ---: |
| `OM0 -> OM1` | early `1-2` | hazard AUC | `+0.0002` | 9/13 | 0/13 |
| `OM0 -> OM1` | early `1-2` | rank Spearman | `-0.0027` | 3/13 | 1/13 |
| `OM0 -> OM1` | early `1-2` | top-10 AUC | `-0.0038` | 3/13 | 1/13 |
| `OM0 -> OM1` | mid `3-4` | hazard AUC | `-0.0016` | 6/13 | 1/13 |
| `OM0 -> OM1` | mid `3-4` | rank Spearman | `-0.0010` | 5/13 | 0/13 |
| `OM0 -> OM1` | mid `3-4` | top-10 AUC | `-0.0003` | 5/11 | 1/11 |
| `OM0 -> OM1` | late `5-6` | hazard AUC | `-0.0010` | 6/13 | 0/13 |
| `OM0 -> OM1` | late `5-6` | rank Spearman | `-0.0005` | 6/13 | 2/13 |
| `OM0 -> OM1` | late `5-6` | top-10 AUC | `-0.0008` | 3/7 | 0/7 |
| `OM2 -> OM3` | early `1-2` | hazard AUC | `+0.0154` | 13/13 | 10/13 |
| `OM2 -> OM3` | early `1-2` | rank Spearman | `+0.0386` | 13/13 | 12/13 |
| `OM2 -> OM3` | early `1-2` | top-10 AUC | `+0.0317` | 13/13 | 11/13 |
| `OM2 -> OM3` | mid `3-4` | hazard AUC | `+0.0080` | 11/13 | 4/13 |
| `OM2 -> OM3` | mid `3-4` | rank Spearman | `+0.0531` | 13/13 | 11/13 |
| `OM2 -> OM3` | mid `3-4` | top-10 AUC | `+0.0537` | 11/11 | 10/11 |
| `OM2 -> OM3` | late `5-6` | hazard AUC | `+0.0107` | 9/13 | 0/13 |
| `OM2 -> OM3` | late `5-6` | rank Spearman | `+0.0277` | 7/13 | 5/13 |
| `OM2 -> OM3` | late `5-6` | top-10 AUC | `+0.0240` | 7/7 | 3/7 |
| `OM3 -> OM4` | early `1-2` | hazard AUC | `-0.0005` | 6/13 | 2/13 |
| `OM3 -> OM4` | early `1-2` | rank Spearman | `+0.0094` | 12/13 | 6/13 |
| `OM3 -> OM4` | early `1-2` | top-10 AUC | `+0.0002` | 8/13 | 0/13 |
| `OM3 -> OM4` | mid `3-4` | hazard AUC | `+0.0037` | 12/13 | 3/13 |
| `OM3 -> OM4` | mid `3-4` | rank Spearman | `+0.0103` | 12/13 | 6/13 |
| `OM3 -> OM4` | mid `3-4` | top-10 AUC | `+0.0024` | 9/11 | 3/11 |
| `OM3 -> OM4` | late `5-6` | hazard AUC | `+0.0044` | 11/13 | 1/13 |
| `OM3 -> OM4` | late `5-6` | rank Spearman | `+0.0177` | 10/13 | 3/13 |
| `OM3 -> OM4` | late `5-6` | top-10 AUC | `+0.0026` | 6/7 | 0/7 |

## Central Step Detail

The following table reports `OM2 -> OM3` hazard AUC deltas.  A star marks a
cohort-phase-group where the paired bootstrap interval supports a positive
gain.

| Cohort | Early phases 1-2 | Mid phases 3-4 | Late phases 5-6 |
| --- | ---: | ---: | ---: |
| Erangel FPP squad | `+0.0235`* `[+0.0048, +0.0414]` | `+0.0031` `[-0.0163, +0.0229]` | `-0.0057` `[-0.0570, +0.0424]` |
| Erangel TPP duo | `+0.0241`* `[+0.0137, +0.0345]` | `+0.0116` `[-0.0019, +0.0247]` | `+0.0176` `[-0.0089, +0.0451]` |
| Erangel TPP solo | `+0.0309`* `[+0.0230, +0.0393]` | `+0.0150`* `[+0.0074, +0.0232]` | `+0.0153` `[-0.0019, +0.0341]` |
| Erangel TPP squad | `+0.0142`* `[+0.0106, +0.0180]` | `+0.0109`* `[+0.0073, +0.0144]` | `+0.0066` `[-0.0005, +0.0139]` |
| Karakin TPP squad | `+0.0112` `[-0.0024, +0.0254]` | `-0.0023` `[-0.0223, +0.0186]` | `+0.0269` `[-0.0204, +0.0771]` |
| Miramar / Desert TPP squad | `+0.0062` `[-0.0037, +0.0160]` | `+0.0061` `[-0.0028, +0.0150]` | `+0.0155` `[-0.0034, +0.0345]` |
| Paramo TPP squad | `+0.0014` `[-0.0120, +0.0166]` | `+0.0080` `[-0.0102, +0.0260]` | `+0.0538` `[-0.0559, +0.1531]` |
| Rondo TPP squad | `+0.0149`* `[+0.0044, +0.0255]` | `-0.0057` `[-0.0173, +0.0061]` | `+0.0107` `[-0.0140, +0.0325]` |
| Sanhok TPP squad | `+0.0176`* `[+0.0088, +0.0263]` | `+0.0101` `[-0.0004, +0.0212]` | `-0.0031` `[-0.0233, +0.0172]` |
| Taego FPP squad | `+0.0248`* `[+0.0029, +0.0466]` | `+0.0015` `[-0.0109, +0.0138]` | `+0.0195` `[-0.0296, +0.0724]` |
| Taego TPP duo | `+0.0319`* `[+0.0179, +0.0455]` | `+0.0231`* `[+0.0083, +0.0372]` | `-0.0235` `[-0.0598, +0.0116]` |
| Taego TPP squad | `+0.0125`* `[+0.0071, +0.0182]` | `+0.0079`* `[+0.0026, +0.0134]` | `+0.0081` `[-0.0005, +0.0168]` |
| Vikendi TPP squad | `+0.0154`* `[+0.0025, +0.0297]` | `+0.0080` `[-0.0050, +0.0215]` | `-0.0026` `[-0.0293, +0.0198]` |

## Interpretation

The bootstrap changes the result status in three ways.

First, the pressure-habitat mechanism is no longer just a descriptive
phase-stratified pattern.  Early hazard, early rank, early top-10, and mid
rank/top-10 deltas have broad paired-CI support across independently trained
cohorts.

Second, O1 survival capital should remain a non-headline result.  The separate
O1 integrity audit showed that the O1 tensor is not broken and has direct
univariate hazard signal.  However, this bootstrap shows little robust
incremental value after OM0 context in the current cumulative ladder.

Third, O4 threat exposure is a secondary and conditional result.  Its rank
signal is more consistent than its hazard signal, especially in mid and late
phase groups, but the paired support is not strong enough to make it a universal
headline block.

The unsupported early-hazard cells are informative but not headline claims.
Karakin TPP squad and Paramo TPP squad have positive point estimates with wide
intervals, consistent with low test-match counts.  Miramar / Desert TPP squad
has a smaller positive point estimate despite a larger test set, so it should be
reported only as a possible map-density or map-geometry moderator unless a
dedicated moderator analysis is run.

The mid-phase split between hazard and rank is a useful interpretation target:
mid-phase hazard support is weaker, while rank and top-10 support remain broad.
This is consistent with a survival-process reading in which early phases are
better for next-phase elimination discrimination, while mid phases still sort
eventual rank and deep-survival ordering.

## Claim Boundary

Allowed:

```text
Across cohort-separated PUBG settings, paired match-cluster bootstrap supports
the central pressure-habitat mechanism: after movement features are included,
OM3 produces robust early-phase hazard gains and early/mid rank or top-10 gains.
```

Allowed, with caution:

```text
Threat exposure appears to become more useful after pressure and habitat have
constrained the survival environment, but this is a secondary and conditional
pattern.
```

Avoid:

```text
The bootstrap proves causal habitat effects.
```

Avoid:

```text
O1 survival capital has no relationship with survival.
```

Avoid:

```text
O7 relation modeling or O6 spatial affordance has been fully tested.
```

## Remaining Experiment Implications

This result clears the uncertainty check for the core OM3 claim.  The next
experiment should not be another broad rerun of the same ladder.

The next required experiment is:

```text
Run the corrected O6 Z-ladder / spatial-affordance sensitivity after adding the
missing vertical-position and contact-gated relative-height projections.
```

Protocol owner:

```text
docs/experiment_design/BATTLE_ROYALE_O6_Z_LADDER_SENSITIVITY_PROTOCOL_2026-07-04.md
```

Reason:

- the current OM6 block contains only weak spatial proxies and overlaps with
  O4/O7 exposure signals;
- the intended O6 headline design requires Z-style spatial affordance
  projections;
- the central OM3 claim is now strong enough that the next uncertainty belongs
  to the unfinished spatial-affordance claim, not to another OM3 rerun.
