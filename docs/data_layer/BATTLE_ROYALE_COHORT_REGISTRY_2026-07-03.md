# Battle Royale Cohort Registry

Date: 2026-07-03
Class: `DataContract`
Lifecycle: `active`

## Purpose

This document locks the PUBG supervised experiment cohorts by:

```text
map_name x perspective x team_size
```

The project should not treat map, camera perspective, and team size as one
pooled population when making headline claims.  They change the semantics of
movement, survival capital, relation structure, and final-rank difficulty.

## Atomic Cohort Rule

The atomic cohort is:

```text
(map_name, perspective, team_size)
```

Every OM0-OM6 experiment should be able to report results at this level before
any aggregate is interpreted.

Source artifacts:

| Field | Value |
| --- | --- |
| Master cohort | `paper_graph_n128_v1` |
| Master match list | `model_stgcn/experiment_datasets/research_cohort_flat/match_cohort_flat.csv` |
| Master split | `model_stgcn/experiment_datasets/research_cohort_flat/paper_graph_n128_v1_split_seed42.json` |
| Domain split summary | `model_stgcn/experiment_datasets/research_cohort_flat/paper_graph_n128_v1_domain_split_summary.csv` |
| Eligibility | `graph_eligible_n128 = True` |
| Excluded maps | `Tutorial_Main`, `Range_Main` |
| Total graph-eligible matches | 21,824 |
| Registry threshold | `main` if total >= 300 matches; otherwise `aux` |

## Reporting Policy

| Level | Meaning | Use |
| --- | --- | --- |
| atomic cohort | one map, one perspective, one team size | primary unit of evaluation |
| main stratum | atomic cohort with >= 300 matches | headline cohort-level table |
| auxiliary stratum | atomic cohort with < 300 matches | sensitivity, boundary, descriptive evidence |
| group summary | macro or weighted aggregate over registered cohorts | secondary summary only |

Rules:

1. Do not use a pooled result as the only headline result.
2. Report OM deltas per registered cohort first.
3. If an aggregate is shown, report both macro average and match-count weighted
   average, and name the included cohort set.
4. Do not compare results across cohorts as if the train/test populations were
   identical.
5. The all-map TPP squad group remains important, but it is a group summary
   over registered map cohorts, not the only cohort definition.
6. In prose, describe cohorts by map, perspective, and team size.  Registry IDs
   are internal handles, not the preferred explanation format.

## Main Cohorts

These cohorts have at least 300 graph-eligible matches.

| Cohort ID | Domain slug | Tier | Map label | `map_name` | Perspective | Team size | `game_mode` | Train | Val | Test | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `C_MAIN_01` | `Baltic_Main__TPP__squad` | main | Erangel | `Baltic_Main` | `TPP` | `squad` | `squad` | 4,998 | 568 | 638 | 6,204 |
| `C_MAIN_02` | `Tiger_Main__TPP__squad` | main | Taego | `Tiger_Main` | `TPP` | `squad` | `squad` | 3,324 | 431 | 404 | 4,159 |
| `C_MAIN_03` | `Desert_Main__TPP__squad` | main | Miramar / Desert | `Desert_Main` | `TPP` | `squad` | `squad` | 1,210 | 158 | 175 | 1,543 |
| `C_MAIN_04` | `Neon_Main__TPP__squad` | main | Rondo | `Neon_Main` | `TPP` | `squad` | `squad` | 838 | 102 | 99 | 1,039 |
| `C_MAIN_05` | `Savage_Main__TPP__squad` | main | Sanhok | `Savage_Main` | `TPP` | `squad` | `squad` | 743 | 109 | 88 | 940 |
| `C_MAIN_06` | `Baltic_Main__TPP__solo` | main | Erangel | `Baltic_Main` | `TPP` | `solo` | `solo` | 643 | 80 | 77 | 800 |
| `C_MAIN_07` | `Baltic_Main__TPP__duo` | main | Erangel | `Baltic_Main` | `TPP` | `duo` | `duo` | 529 | 67 | 60 | 656 |
| `C_MAIN_08` | `Tiger_Main__TPP__duo` | main | Taego | `Tiger_Main` | `TPP` | `duo` | `duo` | 456 | 64 | 53 | 573 |
| `C_MAIN_09` | `DihorOtok_Main__TPP__squad` | main | Vikendi | `DihorOtok_Main` | `TPP` | `squad` | `squad` | 398 | 51 | 49 | 498 |
| `C_MAIN_10` | `Tiger_Main__FPP__squad` | main | Taego | `Tiger_Main` | `FPP` | `squad` | `squad-fpp` | 335 | 58 | 42 | 435 |
| `C_MAIN_11` | `Baltic_Main__FPP__squad` | main | Erangel | `Baltic_Main` | `FPP` | `squad` | `squad-fpp` | 346 | 36 | 48 | 430 |
| `C_MAIN_12` | `Chimera_Main__TPP__squad` | main | Paramo | `Chimera_Main` | `TPP` | `squad` | `squad` | 265 | 32 | 36 | 333 |
| `C_MAIN_13` | `Summerland_Main__TPP__squad` | main | Karakin | `Summerland_Main` | `TPP` | `squad` | `squad` | 270 | 31 | 32 | 333 |

## Auxiliary Cohorts

These cohorts are locked and may be used for sensitivity, descriptive
coverage, or boundary checks.  They are not large enough to carry the main
headline claim alone.

| Cohort ID | Domain slug | Tier | Map label | `map_name` | Perspective | Team size | `game_mode` | Train | Val | Test | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `C_AUX_01` | `Desert_Main__FPP__squad` | aux | Miramar / Desert | `Desert_Main` | `FPP` | `squad` | `squad-fpp` | 231 | 30 | 27 | 288 |
| `C_AUX_02` | `Neon_Main__FPP__squad` | aux | Rondo | `Neon_Main` | `FPP` | `squad` | `squad-fpp` | 211 | 25 | 28 | 264 |
| `C_AUX_03` | `Savage_Main__TPP__duo` | aux | Sanhok | `Savage_Main` | `TPP` | `duo` | `duo` | 189 | 34 | 20 | 243 |
| `C_AUX_04` | `Desert_Main__TPP__duo` | aux | Miramar / Desert | `Desert_Main` | `TPP` | `duo` | `duo` | 193 | 23 | 25 | 241 |
| `C_AUX_05` | `Tiger_Main__FPP__duo` | aux | Taego | `Tiger_Main` | `FPP` | `duo` | `duo-fpp` | 158 | 17 | 24 | 199 |
| `C_AUX_06` | `Baltic_Main__FPP__duo` | aux | Erangel | `Baltic_Main` | `FPP` | `duo` | `duo-fpp` | 147 | 17 | 22 | 186 |
| `C_AUX_07` | `Kiki_Main__TPP__squad` | aux | Deston | `Kiki_Main` | `TPP` | `squad` | `squad` | 128 | 14 | 22 | 164 |
| `C_AUX_08` | `Tiger_Main__TPP__solo` | aux | Taego | `Tiger_Main` | `TPP` | `solo` | `solo` | 128 | 17 | 15 | 160 |
| `C_AUX_09` | `Neon_Main__TPP__duo` | aux | Rondo | `Neon_Main` | `TPP` | `duo` | `duo` | 125 | 16 | 14 | 155 |
| `C_AUX_10` | `Summerland_Main__TPP__duo` | aux | Karakin | `Summerland_Main` | `TPP` | `duo` | `duo` | 110 | 10 | 14 | 134 |
| `C_AUX_11` | `DihorOtok_Main__TPP__duo` | aux | Vikendi | `DihorOtok_Main` | `TPP` | `duo` | `duo` | 103 | 11 | 8 | 122 |
| `C_AUX_12` | `DihorOtok_Main__FPP__squad` | aux | Vikendi | `DihorOtok_Main` | `FPP` | `squad` | `squad-fpp` | 89 | 14 | 12 | 115 |
| `C_AUX_13` | `Chimera_Main__TPP__duo` | aux | Paramo | `Chimera_Main` | `TPP` | `duo` | `duo` | 97 | 2 | 7 | 106 |
| `C_AUX_14` | `Neon_Main__FPP__duo` | aux | Rondo | `Neon_Main` | `FPP` | `duo` | `duo-fpp` | 77 | 16 | 12 | 105 |
| `C_AUX_15` | `Baltic_Main__FPP__solo` | aux | Erangel | `Baltic_Main` | `FPP` | `solo` | `solo-fpp` | 83 | 8 | 11 | 102 |
| `C_AUX_16` | `Savage_Main__TPP__solo` | aux | Sanhok | `Savage_Main` | `TPP` | `solo` | `solo` | 78 | 12 | 10 | 100 |
| `C_AUX_17` | `Desert_Main__FPP__duo` | aux | Miramar / Desert | `Desert_Main` | `FPP` | `duo` | `duo-fpp` | 80 | 9 | 7 | 96 |
| `C_AUX_18` | `Tiger_Main__FPP__solo` | aux | Taego | `Tiger_Main` | `FPP` | `solo` | `solo-fpp` | 67 | 18 | 8 | 93 |
| `C_AUX_19` | `Desert_Main__TPP__solo` | aux | Miramar / Desert | `Desert_Main` | `TPP` | `solo` | `solo` | 77 | 5 | 9 | 91 |
| `C_AUX_20` | `Neon_Main__TPP__solo` | aux | Rondo | `Neon_Main` | `TPP` | `solo` | `solo` | 66 | 7 | 8 | 81 |
| `C_AUX_21` | `DihorOtok_Main__FPP__duo` | aux | Vikendi | `DihorOtok_Main` | `FPP` | `duo` | `duo-fpp` | 57 | 5 | 8 | 70 |
| `C_AUX_22` | `DihorOtok_Main__TPP__solo` | aux | Vikendi | `DihorOtok_Main` | `TPP` | `solo` | `solo` | 54 | 7 | 9 | 70 |
| `C_AUX_23` | `Summerland_Main__TPP__solo` | aux | Karakin | `Summerland_Main` | `TPP` | `solo` | `solo` | 49 | 6 | 7 | 62 |
| `C_AUX_24` | `Desert_Main__FPP__solo` | aux | Miramar / Desert | `Desert_Main` | `FPP` | `solo` | `solo-fpp` | 45 | 6 | 5 | 56 |
| `C_AUX_25` | `Savage_Main__FPP__squad` | aux | Sanhok | `Savage_Main` | `FPP` | `squad` | `squad-fpp` | 51 | 3 | 2 | 56 |
| `C_AUX_26` | `Neon_Main__FPP__solo` | aux | Rondo | `Neon_Main` | `FPP` | `solo` | `solo-fpp` | 42 | 6 | 4 | 52 |
| `C_AUX_27` | `Kiki_Main__FPP__squad` | aux | Deston | `Kiki_Main` | `FPP` | `squad` | `squad-fpp` | 40 | 5 | 3 | 48 |
| `C_AUX_28` | `Kiki_Main__TPP__duo` | aux | Deston | `Kiki_Main` | `TPP` | `duo` | `duo` | 36 | 4 | 2 | 42 |
| `C_AUX_29` | `Savage_Main__FPP__duo` | aux | Sanhok | `Savage_Main` | `FPP` | `duo` | `duo-fpp` | 32 | 4 | 4 | 40 |
| `C_AUX_30` | `Chimera_Main__TPP__solo` | aux | Paramo | `Chimera_Main` | `TPP` | `solo` | `solo` | 29 | 5 | 5 | 39 |
| `C_AUX_31` | `Summerland_Main__FPP__squad` | aux | Karakin | `Summerland_Main` | `FPP` | `squad` | `squad-fpp` | 28 | 8 | 3 | 39 |
| `C_AUX_32` | `Chimera_Main__FPP__squad` | aux | Paramo | `Chimera_Main` | `FPP` | `squad` | `squad-fpp` | 30 | 3 | 4 | 37 |
| `C_AUX_33` | `Chimera_Main__FPP__duo` | aux | Paramo | `Chimera_Main` | `FPP` | `duo` | `duo-fpp` | 28 | 2 | 3 | 33 |
| `C_AUX_34` | `DihorOtok_Main__FPP__solo` | aux | Vikendi | `DihorOtok_Main` | `FPP` | `solo` | `solo-fpp` | 25 | 5 | 1 | 31 |
| `C_AUX_35` | `Kiki_Main__FPP__duo` | aux | Deston | `Kiki_Main` | `FPP` | `duo` | `duo-fpp` | 25 | 3 | 3 | 31 |
| `C_AUX_36` | `Kiki_Main__TPP__solo` | aux | Deston | `Kiki_Main` | `TPP` | `solo` | `solo` | 22 | 5 | 2 | 29 |
| `C_AUX_37` | `Savage_Main__FPP__solo` | aux | Sanhok | `Savage_Main` | `FPP` | `solo` | `solo-fpp` | 20 | 5 | 3 | 28 |
| `C_AUX_38` | `Summerland_Main__FPP__duo` | aux | Karakin | `Summerland_Main` | `FPP` | `duo` | `duo-fpp` | 18 | 0 | 5 | 23 |
| `C_AUX_39` | `Chimera_Main__FPP__solo` | aux | Paramo | `Chimera_Main` | `FPP` | `solo` | `solo-fpp` | 17 | 4 | 0 | 21 |
| `C_AUX_40` | `Kiki_Main__FPP__solo` | aux | Deston | `Kiki_Main` | `FPP` | `solo` | `solo-fpp` | 8 | 4 | 3 | 15 |
| `C_AUX_41` | `Summerland_Main__FPP__solo` | aux | Karakin | `Summerland_Main` | `FPP` | `solo` | `solo-fpp` | 12 | 0 | 2 | 14 |

## Group Summaries

Group summaries are allowed only after atomic cohort results are available.

| Group ID | Included cohorts | Matches | Role |
| --- | --- | ---: | --- |
| `G_TPP_SQUAD_MAIN_MAPS` | Erangel TPP squad, Taego TPP squad, Miramar/Desert TPP squad, Rondo TPP squad, Sanhok TPP squad, Vikendi TPP squad, Paramo TPP squad, Karakin TPP squad | 15,049 | report-only summary over separately trained TPP squad main-map cohorts |
| `G_MAIN_ALL` | all 13 main cohorts listed above | 17,943 | headline macro/weighted aggregate over experiment-ready strata |
| `G_TPP_SQUAD_ALL_MAPS` | all TPP squad cohorts | 15,213 | team-level relation summary; report as group, not atomic cohort |
| `G_FPP_SQUAD_ALL_MAPS` | all FPP squad cohorts | 1,712 | perspective robustness |
| `G_TPP_DUO_ALL_MAPS` | all TPP duo cohorts | 2,272 | team-size robustness |
| `G_FPP_DUO_ALL_MAPS` | all FPP duo cohorts | 783 | auxiliary robustness |
| `G_TPP_SOLO_ALL_MAPS` | all TPP solo cohorts | 1,432 | solo boundary / Sagong-adjacent comparison |
| `G_FPP_SOLO_ALL_MAPS` | all FPP solo cohorts | 412 | auxiliary solo boundary |
| `G_ALL_GRAPH_ELIGIBLE` | all registered cohorts | 21,824 | broad descriptive population only unless stratified reporting is preserved |

## Experiment Order

1. Audit OM0-OM6 feature availability for each required map / perspective /
   team-size cohort.
2. Run smoke tests on a small representative set:
   - one large TPP squad map;
   - one non-squad main cohort;
   - one FPP main cohort.
3. Run OM0-OM6 separately on every required main cohort.
4. Report cohort-level deltas plus macro and weighted summaries.
5. Add `C_AUX_*` only as sensitivity or descriptive coverage.

## Immediate Consequence

The next checklist item is:

```text
Feature availability audit for OM0-OM6 on required cohorts
```

The audit must answer whether every OM block is available for each required
cohort first, then mark auxiliary coverage separately.
