# O1 Survival-Capital Integrity Audit

Date: 2026-07-04
Class: `ExperimentResult`
Lifecycle: `active`

## Purpose

The phase-stratified OM analysis found that `OM0 -> OM1` contributes almost no
incremental predictive value after adding O1 survival-capital features. Because
this is counterintuitive, this audit checks whether O1 is actually present and
informative in the tensors, or whether the weak OM1 delta is caused by a
feature extraction, join, or normalization failure.

This is not a new model run. It reads the same cohort-separated test splits
from the completed required-cohort OM ladder.

## Inputs

Run root:

```text
model_stgcn/checkpoints/br_om_ladder_required_v1
```

Audit command:

```bash
PG_PASSWORD=104204 python3 tools/audit_o1_survival_capital.py \
  --root model_stgcn/checkpoints/br_om_ladder_required_v1 \
  --split test
```

Generated artifacts:

```text
model_stgcn/checkpoints/br_om_ladder_required_v1/o1_survival_capital_integrity_by_feature.csv
model_stgcn/checkpoints/br_om_ladder_required_v1/o1_survival_capital_integrity_by_cohort.csv
model_stgcn/checkpoints/br_om_ladder_required_v1/o1_survival_capital_integrity_summary.json
```

Script:

```text
tools/audit_o1_survival_capital.py
```

## O1 Features Audited

| Feature | Intended Meaning |
| --- | --- |
| `alive_players_start_norm` | normalized alive player count at phase start |
| `alive_players_end_norm` | normalized alive player count at phase end |
| `alive_players_min_norm` | normalized minimum alive player count within phase |
| `alive_players_avg_norm` | normalized average alive player count within phase |
| `mean_health_avg_norm` | normalized average health state |
| `min_health_min_norm` | normalized minimum health state |
| `health_slope_norm` | normalized health trend through the phase |
| `team_spread_avg_norm` | normalized average team spatial spread |
| `team_spread_max_norm` | normalized maximum team spatial spread |

## Cohort-Separated Sample Counts

Each row uses the test split for the named map / perspective / team-size cohort.

| Cohort | Test Matches | Team-Phase Rows | Hazard Rate | Degenerate O1 Features | Strongest O1 Feature |
| --- | ---: | ---: | ---: | ---: | --- |
| Erangel TPP squad | 620 | 31,628 | 0.317 | 0 | `mean_health_avg_norm` |
| Taego TPP squad | 415 | 21,805 | 0.297 | 0 | `alive_players_avg_norm` |
| Miramar / Desert TPP squad | 154 | 9,068 | 0.289 | 0 | `alive_players_avg_norm` |
| Rondo TPP squad | 103 | 5,200 | 0.289 | 0 | `mean_health_avg_norm` |
| Sanhok TPP squad | 94 | 4,637 | 0.324 | 0 | `alive_players_avg_norm` |
| Erangel TPP solo | 80 | 12,065 | 0.358 | 3 | `mean_health_avg_norm` |
| Erangel TPP duo | 65 | 5,763 | 0.330 | 0 | `alive_players_min_norm` |
| Taego TPP duo | 57 | 4,610 | 0.328 | 0 | `alive_players_min_norm` |
| Vikendi TPP squad | 49 | 2,778 | 0.302 | 0 | `alive_players_avg_norm` |
| Taego FPP squad | 43 | 1,995 | 0.333 | 0 | `mean_health_avg_norm` |
| Erangel FPP squad | 43 | 2,204 | 0.313 | 0 | `mean_health_avg_norm` |
| Paramo TPP squad | 33 | 948 | 0.330 | 0 | `alive_players_min_norm` |
| Karakin TPP squad | 33 | 1,159 | 0.368 | 0 | `alive_players_avg_norm` |

Total audited test rows: 103,860 team-phase rows from 1,789 test matches.

## Feature Integrity Finding

O1 features are not empty, misjoined, or globally degenerate.

Across all required-cohort test rows, every O1 feature has finite values and
non-trivial variation. The only cohort-level degeneracy is structurally
expected in Erangel TPP solo:

- `alive_players_end_norm` is constant at 0.25 because each solo team has one
  player.
- `team_spread_avg_norm` and `team_spread_max_norm` are constant at zero because
  solo teams have no intra-team spread.

This means the weak `OM0 -> OM1` model delta should not be explained as an O1
data construction failure.

## Univariate Hazard Signal

The univariate direction is consistent with survival-capital semantics. Since
`hazard = 1` means eliminated before the next phase, negative correlations mean
that higher O1 capital is associated with lower next-phase elimination risk.

| Feature | Degenerate Cohorts | Median Std. | Median Pearson With Hazard | Positive / Negative Cohorts | Median Eliminated Minus Survived |
| --- | ---: | ---: | ---: | ---: | ---: |
| `alive_players_start_norm` | 0 | 0.1098 | -0.0605 | 0 / 13 | -0.0144 |
| `alive_players_end_norm` | 1 | 0.0881 | -0.0869 | 0 / 12 | -0.0154 |
| `alive_players_min_norm` | 0 | 0.1212 | -0.1246 | 0 / 13 | -0.0308 |
| `alive_players_avg_norm` | 0 | 0.0534 | -0.1462 | 0 / 13 | -0.0181 |
| `mean_health_avg_norm` | 0 | 0.1262 | -0.1527 | 0 / 13 | -0.0424 |
| `min_health_min_norm` | 0 | 0.3499 | -0.0979 | 0 / 13 | -0.0737 |
| `health_slope_norm` | 0 | 0.0201 | -0.1088 | 0 / 13 | -0.0052 |
| `team_spread_avg_norm` | 1 | 0.0386 | -0.0598 | 1 / 11 | -0.0036 |
| `team_spread_max_norm` | 1 | 0.2708 | -0.0849 | 0 / 12 | -0.0478 |

Pooled across all required-cohort test rows, the strongest O1 univariate signals
are health and average alive-player state:

| Feature | Pooled Pearson With Hazard | Pooled Eliminated Minus Survived |
| --- | ---: | ---: |
| `mean_health_avg_norm` | -0.162 | -0.0467 |
| `alive_players_avg_norm` | -0.141 | -0.0158 |
| `alive_players_min_norm` | -0.099 | -0.0250 |
| `min_health_min_norm` | -0.099 | -0.0755 |
| `team_spread_max_norm` | -0.085 | -0.0482 |
| `alive_players_end_norm` | -0.081 | -0.0142 |

## Interpretation

The integrity audit closes the cheap bug hypothesis. O1 is present, has
variance, and points in the expected direction: teams with weaker alive-player
and health states are more likely to be eliminated before the next phase.

The weak `OM0 -> OM1` ladder result therefore means something narrower:

```text
O1 has direct univariate hazard signal, but in the current cumulative neural
OM ladder it adds little independent predictive value beyond the existing
context and subsequent movement / pressure-habitat blocks.
```

For the manuscript, O1 should be treated as a non-headline negative or
conditional result, not as evidence that survival capital is irrelevant to
Battle Royale play. The safe wording is:

```text
Survival-capital variables are well-formed and directionally associated with
next-phase elimination, but their incremental value in our cumulative ladder is
small relative to the pressure-habitat mechanism.
```

## Claim Boundary

Allowed:

- O1 feature construction passed a cohort-separated integrity audit.
- O1 has consistent direct univariate association with next-phase hazard.
- The completed OM ladder does not show strong incremental value for O1 as an
  early cumulative block.

Not allowed:

- Claim that health or alive-player state is unimportant in PUBG.
- Claim that O1 would be weak under all model classes or all feature orderings.
- Treat solo O1 degeneracy as a data error; it is a structural property of solo
  cohorts.

## Next

Run paired bootstrap confidence intervals for the phase-stratified OM deltas.
The O1 integrity concern no longer blocks the bootstrap step.
