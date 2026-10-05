# Battle Royale RQ1 Genre Formalization Object

Date: 2026-07-06
Package: research_framing
Lifecycle: active
Responsibility: Define the RQ1 claim, evidence objects, experiments, results, and boundaries.
Depends on: docs/research_framing/BATTLE_ROYALE_ONTOLOGY_LOCK_2026-07-03.md
Used by: docs/research_framing/BATTLE_ROYALE_RQ_OBJECT_CHAIN_2026-07-06.md

## RQObject

Object id:

```text
RQ1GenreFormalization
```

Research question:

```text
How can the Battle Royale genre be formalized?
```

Role in the paper:

```text
Define the genre-level problem before PUBG-specific representation or
prediction experiments are introduced.
```

## Narrative Lock

RQ1 must be written in this order:

```text
general Battle Royale definition
-> representative game set
-> cross-game common structure
-> prediction-oriented O0-O8 mechanism families
-> phase-conditioned survival formalization
-> PUBG as representative empirical instantiation
-> RQ2 observability bridge
```

PUBG is not the source of the genre definition.  PUBG enters after the
genre-level formulation, as the telemetry-rich case where the formal structure
can be instantiated.

## Reference-Class Source

The manuscript uses Wikipedia's "List of battle royale games" page as a
reference-class source for examples of games and modes publicly catalogued as
Battle Royale cases:

```text
https://en.wikipedia.org/wiki/List_of_battle_royale_games
Retrieved: 2026-07-06
```

Use this source as an example list only.  It is a dynamic list and must not be
used as proof of completeness or as a final taxonomy of the genre.

## Main Claim

```text
Battle Royale can be formalized as a phase-conditioned multi-agent survival
system with reusable prediction-oriented mechanism families O0-O8.
```

Manuscript-safe wording:

```text
RQ1 does not claim that PUBG defines the Battle Royale genre.  Rather, it
argues that Battle Royale games can be analytically framed as
phase-conditioned survival systems, and PUBG is used as a representative
empirical case where this formal structure can be instantiated with official
telemetry.
```

## Claim Objects

| Claim object | Claim | Evidence object | Result | Status |
| --- | --- | --- | --- | --- |
| `RQ1.C1.GeneralDefinition` | Battle Royale should be introduced as a match-based survival-elimination genre. | `ReferenceClassList`, `OntologyLock` | Definition uses match-based, bounded rule space, time-varying survival pressure, and progressive elimination/ranking. | supported with source boundary |
| `RQ1.C2.RepresentativeGameSet` | Representative examples should be grouped by role rather than treated as mechanically identical. | `ReferenceClassList`, `MiniCorpusCoverage` | Canonical combat, hybrid/ability, and non-shooter survival-elimination variants are separated. | supported with boundary |
| `RQ1.C3.CommonStructure` | Representative games differ in mechanics but share a survival-oriented elimination structure. | `OntologyLock`, `MiniCorpusCoverage` | Common structures map to O0-O8. | supported |
| `RQ1.C4.PredictionOrientedMechanismFamily` | O0-O8 is a survival-prediction mechanism family, not a complete genre ontology. | `OntologyLock` | O0-O8 are locked as lightweight analytic vocabulary for this paper. | supported |
| `RQ1.C5.CrossGenreCoverage` | The mechanism families are not PUBG-only. | `MiniCorpusCoverage` | O0-O8 each receive 15/15 mini-corpus coverage; no O9 required. | supported with boundary |
| `RQ1.C6.PhaseSurvivalObject` | Final rank alone is insufficient; the core prediction object is phase-cut survival hazard. | `SurvivalFormalization` | unit `(match m, team i, phase cut k)`, hazard `lambda_i(k)`, survival `S_i(k)`, and share `pi_i(k)` are defined. | supported |
| `RQ1.C7.PressureGeneralization` | Battle Royale pressure is broader than PUBG blue zone. | `MiniCorpusCoverage`, `OntologyLock` | O2 is locked as contraction / elimination pressure. | supported |
| `RQ1.C8.ThreatGeneralization` | Battle Royale threat is broader than gun combat. | `MiniCorpusCoverage`, `OntologyLock` | O4 is locked as interaction / threat exposure. | supported |
| `RQ1.C9.SpatialAffordanceGeneralization` | Battle Royale space is not necessarily 3D terrain. | `MiniCorpusCoverage` | O6 appears in full 3D, 2D, grid, track, and obstacle-course projections. | supported with observability boundary |
| `RQ1.C10.PUBGInstantiationBridge` | PUBG is a representative empirical instantiation, not the definition of the genre. | `RQ2` | Detailed PUBG observability is deferred to RQ2. | exported to RQ2 |

## Experiment Objects

| Experiment object | Question | Method | Evidence produced |
| --- | --- | --- | --- |
| `RQ1.E0.ReferenceClassList` | Which examples should introduce the Battle Royale genre? | Use Wikipedia's list of Battle Royale games as an example-list reference, with a dynamic-list boundary. | external URL, retrieved 2026-07-06 |
| `RQ1.E1.MechanismFamilyLock` | What reusable top-level mechanism families are relevant to survival prediction? | Lock O0-O8 names, roles, revision rules, and projection boundaries. | `BATTLE_ROYALE_ONTOLOGY_LOCK_2026-07-03.md` |
| `RQ1.E2.MiniCorpusCoverage` | Is O0-O8 merely PUBG-specific? | Manually code 15 source-rich Battle Royale and Battle-Royale-like games/modes using official, publisher, or store sources. | `BATTLE_ROYALE_MINI_CORPUS_COVERAGE_TABLE_2026-07-04.md` |
| `RQ1.E3.SurvivalFormalization` | What is the formal prediction object? | Define team-phase unit, next-phase hazard, cumulative survival, and match-local survival share. | `BATTLE_ROYALE_SURVIVAL_FORMALIZATION_2026-07-03.md` |

## Evidence Objects

| Evidence object | Path | What it supports |
| --- | --- | --- |
| `ReferenceClassList` | `https://en.wikipedia.org/wiki/List_of_battle_royale_games` | representative Battle Royale games and modes for the genre-introduction paragraph |
| `OntologyLock` | `docs/research_framing/BATTLE_ROYALE_ONTOLOGY_LOCK_2026-07-03.md` | O0-O8 family definitions, projection rules, and claim boundary |
| `MiniCorpusCoverage` | `docs/research_framing/BATTLE_ROYALE_MINI_CORPUS_COVERAGE_TABLE_2026-07-04.md` | non-PUBG coverage and O2/O4/O6 generalization |
| `SurvivalFormalization` | `docs/model_design/BATTLE_ROYALE_SURVIVAL_FORMALIZATION_2026-07-03.md` | formal phase-cut hazard, cumulative survival, and match-local survival-share variables |
| `RQChain` | `docs/research_framing/BATTLE_ROYALE_RESEARCH_QUESTION_CHAIN_2026-07-05.md` | RQ1 role in the four-question manuscript chain |

## Result Objects

### RQ1.R1 Ontology Family Lock

In manuscript prose, call these prediction-oriented mechanism families rather
than a complete ontology of the Battle Royale genre.

Locked families:

| Family | Locked name | Manuscript meaning |
| --- | --- | --- |
| `O0` | Rule And Match Context | game rules, match mode, and phase context |
| `O1` | Agent / Team Survival Capital | capacity to remain alive or qualified |
| `O2` | Contraction / Elimination Pressure | pressure schedule that changes viable survival conditions |
| `O3` | Movement / Rotation Behavior | movement response to pressure and opportunity |
| `O4` | Interaction / Threat Exposure | hostile, contest, or obstacle exposure |
| `O5` | Resource / Equipment / Mobility Support | survival-enabling resources and support |
| `O6` | Spatial / Terrain / Environment Affordance | spatial opportunities and constraints under an observability mask |
| `O7` | Relation / Interaction Structure | opponent, ally, proximity, contest, and pressure relations |
| `O8` | Outcome / Win Condition | survival, rank, qualification, or winner condition |

### RQ1.R2 Mini Corpus Coverage

| Result | Value |
| --- | ---: |
| coded games / modes | 15 |
| O0 coverage | 15/15 |
| O1 coverage | 15/15 |
| O2 coverage | 15/15 |
| O3 coverage | 15/15 |
| O4 coverage | 15/15 |
| O5 coverage | 15/15 |
| O6 coverage | 15/15 |
| O7 coverage | 15/15 |
| O8 coverage | 15/15 |
| required new top-level family | 0 |

### RQ1.R2a Representative Example Strata

| Stratum | Examples | Manuscript role |
| --- | --- | --- |
| canonical combat-centered cases | PUBG, Fortnite, Apex Legends, Call of Duty: Warzone, Garena Free Fire | central Battle Royale cases |
| hybrid or ability-based cases | NARAKA: BLADEPOINT, Eternal Return, Super Animal Royale | combat, camera, and system variation cases |
| non-shooter survival-elimination variants | Fall Guys, Tetris 99, F-ZERO 99 | boundary / extension cases showing survival-elimination structure beyond shooters |

### RQ1.R3 Formal Prediction Object

Primary unit:

```text
(match m, team i, phase cut k)
```

Primary target:

```text
Y_i(k+1) = 1 if team i is eliminated before or during the next phase
```

Primary prediction:

```text
lambda_i(k) = Pr(Y_i(k+1) = 1 | history through k)
```

Secondary readout path:

```text
phase-wise hazard
-> cumulative survival probability
-> match-local dynamic survival share
-> final-rank / top-k evaluation
```

### RQ1.R4 PUBG Instantiation Bridge

PUBG: Battlegrounds is used as the empirical instantiation because it provides:

| Formal requirement | PUBG evidence path |
| --- | --- |
| match and team structure | official match, mode, map, roster, team, and participant metadata |
| time-varying pressure | phase / zone progression and alive-population state |
| team survival state | alive count, health, position, movement, and event histories |
| interaction / threat exposure | damage, kill, knock, proximity, and combat events |
| outcome condition | elimination, final rank, top-k, and winner labels |

Boundary:

```text
PUBG official telemetry does not fully expose all genre-level mechanisms,
especially terrain semantics, cover, visibility, and line-of-sight.  This
motivates the observability analysis in RQ2.
```

## Boundary Object

Allowed:

```text
Battle Royale games can be analyzed as phase-conditioned survival systems.
Representative games share recurring mechanism families relevant to survival
prediction.
```

Allowed:

```text
PUBG provides a telemetry-rich empirical case for instantiating this
formalization.
```

Allowed:

```text
Wikipedia's list of Battle Royale games can be used as an example-list
reference for naming representative games and modes.
```

Forbidden:

```text
This thesis proves a complete ontology or exhaustive taxonomy of every Battle
Royale variant.
```

Forbidden:

```text
The Wikipedia list proves a complete taxonomy of Battle Royale games.
```

Forbidden:

```text
PUBG defines the Battle Royale genre.
```

Forbidden:

```text
The mini corpus validates predictive model performance.
```

The mini corpus supports vocabulary coverage only.  Predictive utility is
tested later in PUBG telemetry under RQ2, RQ3, and RQ4.

Forbidden:

```text
PUBG telemetry fully observes all genre-level mechanisms.
```

## Interface To RQ2

RQ1 exports the following formal objects to RQ2:

```text
O0-O8 mechanism families
(m, i, k) team-phase unit
lambda_i(k) next-phase hazard
S_i(k) cumulative survival
pi_i(k) match-local survival share
observability mask concept
```

RQ2 must show how these objects are represented in PUBG official telemetry.
