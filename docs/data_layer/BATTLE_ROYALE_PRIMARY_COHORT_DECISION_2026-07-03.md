# Battle Royale Primary Cohort Decision

Date: 2026-07-03
Class: `DataContract`
Lifecycle: `supporting`

## Purpose

This document records the initial all-map TPP squad cohort decision.  It has
now been narrowed into a supporting group-level policy because the canonical
cohort definition is map / perspective / team-size specific.

Canonical cohort registry:

```text
docs/data_layer/BATTLE_ROYALE_COHORT_REGISTRY_2026-07-03.md
```

## Corrected Decision

The atomic supervised PUBG cohort is not the pooled all-map TPP squad group.
The atomic unit is:

```text
map_name x perspective x team_size
```

The all-map TPP squad set remains a useful group summary:

| Field | Value |
| --- | --- |
| Group ID | `G_TPP_SQUAD_ALL_MAPS` |
| Source master cohort | `paper_graph_n128_v1` |
| Source match list | `model_stgcn/experiment_datasets/research_cohort_flat/match_cohort_flat.csv` |
| Source split | `model_stgcn/experiment_datasets/research_cohort_flat/paper_graph_n128_v1_split_seed42.json` |
| Eligibility | `graph_eligible_n128 = True` |
| Perspective | `TPP` |
| Team size | `squad` |
| Maps | all graph-eligible standard PUBG maps in the current DB |
| Excluded maps | `Tutorial_Main`, `Range_Main` |
| Match count | 15,213 |
| Train / val / test | 12,174 / 1,496 / 1,543 |
| Subset match-id SHA1 | `172d5dea4d48a9fb0c0c5beb79f76c0c87701260` |

Use this group only after its map-level atomic cohorts have been evaluated.

## Why This Group Still Matters

The paper's current contribution is:

```text
team-level, relation-aware phase survival prediction
```

Therefore the cleanest primary cohort should keep the unit semantics stable.
The TPP squad group is important because:

- it is the largest coherent team-mode stratum in the DB;
- it preserves team-level survival capital meaning;
- relation features such as proximity, combat, co-pressure, and network role
  have stable multi-agent team interpretation;
- it avoids mixing solo, duo, and squad node semantics in the headline ablation;
- it avoids mixing FPP and TPP player-behavior differences in the headline
  effect estimate.

However, this does not replace map-level cohort reporting.  It is a group
summary over registered cohorts.

## TPP Squad Group Map Distribution

These counts use the master split intersection.

| Map label | `map_name` | Train | Val | Test | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| Erangel | `Baltic_Main` | 4,998 | 568 | 638 | 6,204 |
| Taego | `Tiger_Main` | 3,324 | 431 | 404 | 4,159 |
| Miramar / Desert | `Desert_Main` | 1,210 | 158 | 175 | 1,543 |
| Rondo | `Neon_Main` | 838 | 102 | 99 | 1,039 |
| Sanhok | `Savage_Main` | 743 | 109 | 88 | 940 |
| Vikendi | `DihorOtok_Main` | 398 | 51 | 49 | 498 |
| Paramo | `Chimera_Main` | 265 | 32 | 36 | 333 |
| Karakin | `Summerland_Main` | 270 | 31 | 32 | 333 |
| Deston | `Kiki_Main` | 128 | 14 | 22 | 164 |
| Total | all TPP squad maps | 12,174 | 1,496 | 1,543 | 15,213 |

## Rules For OM Experiments

1. OM0-OM6 must be reportable at the atomic cohort level defined in the
   registry.
2. Group summaries such as `G_TPP_SQUAD_ALL_MAPS` must reuse the master split
   intersection and name the included atomic cohorts.
3. If a feature is missing for part of a registered cohort, either:
   - mask it with an explicit observability flag; or
   - exclude that feature from the headline OM block and move it to sensitivity.
4. Do not compare a group summary against an atomic cohort as if the cohort were
   the same.

## Relation To Existing Cohort Artifacts

Existing master cohort:

```text
paper_graph_n128_v1
```

Master size:

```text
21,824 graph-eligible matches
```

Master distribution by perspective and team size:

| Perspective | Team size | Matches |
| --- | --- | ---: |
| TPP | squad | 15,213 |
| FPP | squad | 1,712 |
| TPP | duo | 2,272 |
| FPP | duo | 783 |
| TPP | solo | 1,432 |
| FPP | solo | 412 |

The master split remains the source of truth.  `G_TPP_SQUAD_ALL_MAPS` is a
locked group summary, not a replacement for the atomic cohort registry.

## Immediate Consequence

The next execution step is still the OM0-OM6 feature availability audit, but it
must now answer availability by registered cohort:

```text
For each OM block, can the feature family be computed for every main atomic
cohort under the master split without leakage?
```

Only after main cohorts are feature-ready should auxiliary cohorts be audited.
