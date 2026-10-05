# Battle Royale RQ1 Manuscript Claim

Date: 2026-07-06
Package: paper_writing
Lifecycle: active
Responsibility: Provide the manuscript-facing RQ1 build-up, claim stack,
experiment/result map, and claim boundary.
Depends on:
docs/research_framing/BATTLE_ROYALE_RQ1_GENRE_FORMALIZATION_OBJECT_2026-07-06.md
docs/research_framing/BATTLE_ROYALE_ONTOLOGY_LOCK_2026-07-03.md
docs/research_framing/BATTLE_ROYALE_MINI_CORPUS_COVERAGE_TABLE_2026-07-04.md
docs/model_design/BATTLE_ROYALE_SURVIVAL_FORMALIZATION_2026-07-03.md

## RQ Definition

```text
RQ1. How can the Battle Royale genre be formalized as a phase-conditioned
multi-agent survival system?
```

## Core Thesis

RQ1 does not claim that PUBG defines the Battle Royale genre.  Rather, it
argues that Battle Royale games can be analytically framed as
phase-conditioned survival systems, and PUBG is used as a representative
empirical case where this formal structure can be instantiated with official
telemetry.

## Required Build-Up

```text
1. General Definition of Battle Royale
2. Representative Game Set
3. Cross-Game Common Structure
4. Prediction-Oriented O0-O8 Mechanism Families
5. Mathematical Survival Formalization
6. PUBG as Representative Empirical Instantiation
7. RQ2 Observability Bridge
```

## 1. General Definition Of Battle Royale

Manuscript definition:

```text
Battle Royale games are match-based competitive games in which multiple agents
or teams enter a bounded rule space, adapt to time-varying survival pressure,
and are progressively eliminated or ranked according to survival-oriented
victory conditions.
```

Important terms:

| Term | Why it matters |
| --- | --- |
| match-based | prevents the definition from becoming a generic competition definition |
| multiple agents or teams | preserves the multi-agent survival setting |
| bounded rule space | captures map, mode, phase rules, and match constraints |
| time-varying survival pressure | captures shrinking zones, staged elimination, board pressure, racing attrition, and similar schedules |
| progressive elimination/ranking | connects the genre to survival outcomes rather than a single static score |

## 2. Representative Game Set

Reference-class source:

```text
Wikipedia, "List of battle royale games"
https://en.wikipedia.org/wiki/List_of_battle_royale_games
Retrieved: 2026-07-06
```

Use the source as an example list, not as a completeness proof.

| Stratum | Examples | Role in the RQ1 narrative |
| --- | --- | --- |
| canonical combat-centered cases | PUBG, Fortnite, Apex Legends, Call of Duty: Warzone, Garena Free Fire | central Battle Royale cases |
| hybrid or ability-based cases | NARAKA: BLADEPOINT, Eternal Return, Super Animal Royale | combat, camera, and system-variation cases |
| non-shooter survival-elimination variants | Fall Guys, Tetris 99, F-ZERO 99 | boundary / extension cases showing survival-elimination structure beyond shooters |

Safe wording:

```text
These games differ substantially in combat style, camera perspective, input
mechanics, and resource systems, but they share a survival-oriented elimination
structure.
```

## 3. Cross-Game Common Structure

Across these examples, a match can be understood as a bounded survival contest
where agents repeatedly update their state, respond to changing pressure,
interact with threats and resources, move through spatial or environmental
affordances, and eventually produce elimination or rank outcomes.

## 4. Prediction-Oriented O0-O8 Mechanism Families

Use "prediction-oriented mechanism families", not "complete genre ontology".

| Family | Manuscript label | Meaning |
| --- | --- | --- |
| `O0` | Match rule/context | map, mode, team size, phase rule, match setting |
| `O1` | Agent/team survival state | health, alive members, equipment state, current survival capital |
| `O2` | Time-varying survival pressure | zone contraction, shrinking area, external hazard, elimination pressure |
| `O3` | Movement / behavioral response | movement, avoidance, repositioning, engagement avoidance |
| `O4` | Threat exposure / interaction | combat, proximity, damage, enemy exposure, contest pressure |
| `O5` | Resource / support | items, healing, ammo, vehicle, revive/support |
| `O6` | Spatial / environmental affordance | terrain, buildings, cover, movement possibility, line-of-sight, habitat |
| `O7` | Inter-agent relation | ally, enemy, proximity, pressure-sharing, encounter relation |
| `O8` | Outcome / victory condition | elimination, survival, final rank, victory |

O6 must be handled carefully.  The genre-level structure contains spatial and
environmental affordance, but PUBG official telemetry only partially observes
terrain, cover, visibility, and line-of-sight.  This is the hand-off to RQ2.

## 5. Mathematical Survival Formalization

Let `H_i^m(k)` denote the observable history of team `i` in match `m` up to
phase or time cut `k`.

Prediction unit:

```text
(match m, team i, phase/time cut k)
```

Next-phase elimination hazard:

```text
lambda_i^m(k) = Pr(Y_i^m(k+1) = 1 | H_i^m(k))
```

Cumulative survival:

```text
S_i^m(0) = 1
S_i^m(k+1) = S_i^m(k) * (1 - lambda_i^m(k))
```

Match-local survival share:

```text
pi_i^m(k) = S_i^m(k) / sum_{j in A_m(k)} S_j^m(k)
```

where `A_m(k)` is the set of teams still alive at cut `k`.

This formalization shifts the problem from predicting a single final rank to
modeling a dynamic survival process under changing phase conditions.

## 6. PUBG As Representative Empirical Instantiation

PUBG: Battlegrounds is used as the empirical instantiation of this
formalization because it provides official match telemetry with team structure,
phase/zone progression, time-stamped positions and events, elimination
outcomes, and final-rank labels.

Safe wording:

```text
PUBG is not used to define the genre; it is used to instantiate the proposed
survival formalization in a telemetry-rich representative case.
```

Boundary wording:

```text
At the same time, PUBG telemetry does not fully expose all genre-level
mechanisms, such as terrain semantics, cover, visibility, and line-of-sight.
This motivates the observability analysis in RQ2.
```

## 7. RQ2-RQ4 Bridge

| Next RQ | RQ1 export |
| --- | --- |
| `RQ2` | Given the formal structure, which parts are observable, derivable, proxied, label-only, or unavailable in PUBG official telemetry? |
| `RQ3` | Given observable history `H_i^m(k)`, can the model predict `lambda`, `S`, and realtime survival readouts? |
| `RQ4` | Can `lambda` and `pi` be rendered as a phase-transition risk/advantage board? |

## RQ1 Experiment / Result Map

| Result object | Method | Main result | Claim strength |
| --- | --- | --- | --- |
| reference-class list | use Wikipedia list of Battle Royale games for examples | supports naming canonical, hybrid, and non-shooter Battle Royale cases | example-list support only |
| mechanism-family lock | lock O0-O8 and projection boundaries | O0-O8 fixed for the current prediction paper | analytic vocabulary, not exhaustive ontology |
| mini corpus coverage | manually code 15 source-rich Battle Royale and Battle-Royale-like games/modes | O0-O8 each appear in 15/15 rows; no O9 required | supports non-PUBG coverage |
| survival formalization | define team-phase hazard, cumulative survival, and match-local survival share | `(m, i, k)`, `lambda_i(k)`, `S_i(k)`, and `pi_i(k)` are defined | supports phase-conditioned survival formulation |

## Manuscript Paragraph Draft

```text
Battle Royale games can be broadly understood as match-based competitive games
in which multiple agents or teams enter a bounded rule space, adapt to
time-varying survival pressure, and are progressively eliminated or ranked
according to survival-oriented victory conditions.  Although representative
games such as PUBG: Battlegrounds, Fortnite, Apex Legends, Call of Duty:
Warzone, NARAKA: BLADEPOINT, Fall Guys, Tetris 99, and F-ZERO 99 differ in
combat style, viewpoint, control scheme, and resource system, they share a
recurring structure: agents maintain survival states, respond to changing
pressure, move through spatial or environmental affordances, interact with
threats and resources, and eventually produce elimination or rank outcomes.
```

```text
From this cross-game structure, we organize Battle Royale survival into nine
prediction-oriented mechanism families: match context, survival state,
time-varying pressure, movement response, threat exposure, resource support,
spatial/environmental affordance, inter-agent relations, and outcome
conditions.  These families are not intended as a complete ontology of the
genre.  Rather, they provide an analytical scaffold for deriving a
phase-conditioned survival prediction problem.
```

```text
Formally, for team i in match m at phase or time cut k, let H_i^m(k) denote
the observable history up to that cut.  The central prediction object is the
next-phase elimination hazard lambda_i^m(k) = Pr(Y_i^m(k+1)=1 | H_i^m(k)).
The cumulative survival readout is then updated as
S_i^m(k+1)=S_i^m(k)(1-lambda_i^m(k)), and the match-local survival share can
be defined over the currently alive teams.  This formulation shifts the
problem from predicting a single final rank to modeling a dynamic survival
process under changing phase conditions.
```

```text
PUBG: Battlegrounds is used as the representative empirical instantiation of
this formalization.  It does not define the Battle Royale genre; rather, it
provides official telemetry with team structure, phase/zone progression,
time-stamped events and positions, elimination outcomes, and final-rank
labels.  At the same time, some genre-level mechanisms, such as terrain
semantics, cover, visibility, and line-of-sight, are not fully observable in
the official telemetry.  This motivates the observability analysis in RQ2 and
the representation ladder evaluated in the subsequent experiments.
```

## Claim Boundary

Allowed:

```text
Battle Royale games can be analyzed as phase-conditioned survival systems.
```

Allowed:

```text
Representative Battle Royale games share recurring mechanism families relevant
to survival prediction.
```

Allowed:

```text
PUBG provides a telemetry-rich empirical case for instantiating this
formalization.
```

Forbidden:

```text
This paper proves a complete ontology of Battle Royale games.
```

Forbidden:

```text
PUBG defines the Battle Royale genre.
```

Forbidden:

```text
All Battle Royale games share identical mechanics.
```

Forbidden:

```text
PUBG telemetry fully observes all genre-level mechanisms.
```
