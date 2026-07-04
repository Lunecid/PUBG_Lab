# Battle Royale Feature Ontology

Date: 2026-07-02

## Purpose

This document defines the high-level feature ontology for a general Battle
Royale framework.

The goal is not to enumerate every possible variable.  The goal is to define
the genre-level mechanisms that can be instantiated differently across datasets
such as PUBG and Caldera.

Core research goal:

```text
contraction / elimination pressure
-> survival-environment change
with survival capital
-> agent/team behavioral response
-> interaction and exposure change
-> survival / rank outcome
```

## Central Mechanism

Battle Royale is not only a combat-prediction problem.  It is a survival
problem under time-varying contraction or elimination pressure.

The framework treats each match as:

```text
Z_t       = contraction / elimination pressure state
G_t       = spatial / terrain / environment affordance field
H_i(t)    = survivable habitat available to team/player i, induced by Z_t x G_t
A_i(t)    = team or player survival capital
B_i(t)    = movement, interaction, resource, and positioning behavior
R_ij(t)   = relations among agents under shared pressure
Y_i(k)    = survival / elimination / rank / qualification outcome at phase cut k
```

The main explanatory direction for phase-cut prediction is:

```text
Z_t x G_t changes H_i(t).
H_i(t), A_i(t), and resources constrain B_i(t).
B_i(t) changes R_ij(t) and exposure.
R_ij(t), A_i(t), H_i(t), and exposure jointly determine Y_i(k).
```

This is not a one-way causal graph.  The ontology also permits state feedback:

- interaction/threat exposure can deplete survival capital (`O4 -> O1`);
- elimination outcomes can change alive population, crowding, and relation
  structure (`O8 -> O2/O7`);
- other agents' movement and relation positions can alter a team's local
  habitat (`O3_j/O7_ij -> H_i`).

This is the object the framework must represent.

## Lock Status

Status:

```text
O0-O8 ontology v1 is locked for the OM0-OM6 PUBG prediction ladder.
```

The lock freezes the top-level families and their names.  Future corpus work
may add game-specific projections, source evidence, or observability masks, but
it should not introduce a new top-level family before the current prediction
experiments unless a genuine blocker is found.

## Portability Levels

The ontology owns genre-general constructs.  Game-specific terms and database
columns are lower-level projections of those constructs.  Conceptual
portability and measurement observability are separate axes: a construct can be
`BR_GENERAL` while requiring `GEOMETRY_SPECIFIC` or dataset-specific evidence to
measure.

| Level | Meaning | Examples |
| --- | --- | --- |
| `BR_GENERAL` | transferable Battle Royale mechanism | contraction / elimination pressure, survival capital, behavior response, interaction/threat exposure, resource support, relation field |
| `GAME_SPECIFIC` | a game's concrete rule or telemetry vocabulary | PUBG blue zone, DBNO/groggy, boost, care package, FPP/TPP |
| `DATASET_SPECIFIC` | collection schema or implementation artifact | `stdb.team_phase_state`, `hazard_next`, 10s snapshot grid |
| `GEOMETRY_SPECIFIC` | external geometry or mesh-derived construct | Caldera terrain slope, building mesh, pairwise LOS |

Rule:

```text
Ontology claims use BR_GENERAL language.
PUBG documents may map BR_GENERAL constructs to GAME_SPECIFIC and
DATASET_SPECIFIC or GEOMETRY_SPECIFIC features, but those names must not
redefine the ontology.
```

## Spatial Dimensionality

Battle Royale is a genre-level framework, not a 3D-only framework.  Some games
expose full 3D position and geometry; others expose only 2D or grid-based
position.  O6 therefore represents spatial affordance under an observability
mask, not a mandatory terrain/height feature set.

| Spatial profile | Meaning | O6 examples |
| --- | --- | --- |
| `PLANAR_2D` | position is mainly `x/y`; no meaningful altitude | zone-relative position, boundary distance, local density, route conflict, chokepoint/sector pressure |
| `GRID_2D` | tile/cell/sector space | cell safety, adjacency, obstacle tiles, area control, path bottlenecks |
| `ELEVATION_2_5D` | `x/y` plus height or layer, but limited geometry | relative height, floor/layer advantage, vertical movement proxy |
| `FULL_3D` | `x/y/z` and possibly view/physics geometry | height advantage, LOS, cover, terrain slope, building occlusion |
| `GEOMETRY_RICH` | mesh/map assets exist | slope, ruggedness, occluders, isovist, building distance |
| `GEOMETRY_POOR` | only telemetry exists | movement/contact/density proxies; no direct terrain claim |

Rule:

```text
All O6 features must declare their spatial profile.  A non-3D game can still
instantiate O6 through planar exposure, route, sector, and zone-pressure
features.  Height, terrain, LOS, and mesh claims are optional projections, not
ontology requirements.
```

## Ontology-Level Feature Families

### O0. Rule And Match Context

What it captures:

```text
The rule-set under which survival should be interpreted.
```

Examples:

- map;
- mode;
- perspective;
- team size;
- phase index;
- alive population;
- elapsed time.

Role:

- global context;
- stratification;
- cross-map / cross-mode generalization test.

### O1. Agent / Team Survival Capital

What it captures:

```text
The team's current capacity to keep surviving.
```

Examples:

- alive count;
- health;
- knocked/dead pressure where available;
- vehicle state;
- team spread;
- prior damage state;
- survival capital slope over time.

Role:

- strong baseline mechanism;
- confound/control for spatial and behavioral claims;
- survival capital stream in the model.

### O2. Contraction / Elimination Pressure

What it captures:

```text
How match rules progressively compress viable survival conditions.
```

Examples:

- phase index or round index;
- safe-zone radius;
- safe-zone center shift;
- boundary distance;
- outside-safe duration;
- required speed to safe area;
- safe area per alive team;
- zone-gap pressure;
- staged elimination or qualification pressure;
- restricted-area / unavailable-area pressure;
- phase duration.

Role:

- the framework's core environmental driver;
- global context and node pressure feature;
- interaction term for movement, contact, and terrain affordance.

Rule:

```text
Shrinking zones are the most common projection of O2, but not the definition.
Party, racing, puzzle, and mode-based Battle Royale variants can instantiate O2
through staged elimination, route compression, restricted areas, or
qualification pressure.
```

Habitat rule:

```text
Habitat is an O2 x O6 interaction construct.  It measures pressure applied to
the spatial affordance field, not O2 alone and not O6 alone.
```

### O3. Movement / Rotation Behavior

What it captures:

```text
How teams respond behaviorally to contraction or elimination pressure.
```

Examples:

- path length;
- displacement;
- speed;
- sinuosity;
- rotation timing;
- safe-zone entry time;
- stop/go ratio;
- vertical movement where available;
- route efficiency.

Role:

- behavioral response stream;
- evidence that agents adapt to changing survival environment;
- phase-specific response analysis.

### O4. Interaction / Threat Exposure

What it captures:

```text
How agents become exposed to, avoid, or engage other agents under pressure.
```

Examples:

- nearest enemy distance;
- enemy density;
- proximity/contact gate;
- interaction or contest events;
- combat events;
- damage dealt/taken;
- third-party exposure;
- contact duration;
- interaction direction.

Role:

- interaction field;
- edge or node exposure stream;
- distinguishes movement through empty space from movement through contested
  space.

Rule:

```text
Gun combat is a game-specific projection of O4, not the ontology definition.
Racing duels, obstacle contests, word/puzzle interference, melee exchanges, and
team proximity can all instantiate interaction or threat exposure.
```

### O5. Resource / Equipment / Mobility Support

What it captures:

```text
The tools that make survival and rotation possible.
```

Examples:

- weapon/equipment state where available;
- healing / temporary enhancement / pickup / use events where available;
- vehicle use;
- ammunition/resource proxies;
- loot-phase behavior;
- resupply or depletion signals.

Role:

- survival capital modifier;
- movement/rotation enabler;
- dataset-dependent family often missing in low-observability releases.

### O6. Spatial / Terrain / Environment Affordance

What it captures:

```text
How the physical environment affords visibility, cover, movement, and
positional advantage.
```

Examples:

- position, sector, layer, or zone-relative location;
- altitude / relative height where available;
- terrain slope / ruggedness where geometry exists;
- cover / concealment proxy;
- building or compound proximity where geometry exists;
- openness / LOS where geometry exists;
- planar route conflict or local density in 2D games;
- dataset-specific proxies such as contact-gated relative position or height.

Role:

- environment affordance stream;
- geometry-rich construct-validation axis where available;
- telemetry proxy feature family when geometry is poor;
- dimensionality-masked feature family for non-3D games.

Detailed subdesign:

```text
docs/feature_engineering/SPATIAL_TACTICAL_AFFORDANCE_FEATURE_SPEC_2026-07-02.md
```

### O7. Relation / Interaction Structure

What it captures:

```text
How agents are coupled by proximity, hostility, support, contest, or shared
pressure.
```

Examples:

- nearby enemy relation;
- hostile contact relation;
- co-pressure relation;
- relative height advantage;
- zone-pressure difference;
- route conflict;
- support/ally relation where available.

Role:

- relation field for interaction and exposure;
- source for edge streams and ego exposure summaries;
- bridge from game-readable relations to dataset-specific graph measures.

Rule:

```text
Game-facing O7 language should describe relations among agents.  Graph measures
such as degree, betweenness, bridge score, communities, or centralization are
dataset-specific measurement projections, not genre-level wording.
```

### O8. Outcome / Win Condition

What it captures:

```text
The match or phase result the framework explains or predicts.
```

Examples:

- next-phase elimination hazard;
- at-risk mask;
- survival curve;
- final rank;
- top-k placement;
- round qualification;
- win condition;
- censoring state.

Role:

- genre-level outcome / win condition;
- operational labels and masks in supervised datasets;
- never model input when used as a label.

## Phase-Conditioned Interpretation

The meaning of a feature changes by phase.  The framework should therefore
analyze every feature family as phase-conditioned.

| Phase regime | Expected survival environment | Expected behavioral response |
| --- | --- | --- |
| early | wide space, high loot/landing uncertainty | spread, loot, first rotations, first contacts |
| mid | zone shrink begins to force route selection | rotation timing, route conflict, vehicles, contact avoidance or engagement |
| late | small safe area, high density, high exposure | positional holding, contact pressure, micro-rotation, survival capital depletion |
| endgame | tiny arena, censoring and sample-size issues | high variance, strong selection, careful interpretation |

The framework claim should be:

```text
Phase contraction changes the meaning and importance of feature families.
```

Not:

```text
One global feature ranking explains all phases.
```

## Dataset Observability Matrix

| Family | PUBG current DB | Caldera current ingest | Interpretation |
| --- | --- | --- | --- |
| O0 rule/match context | strong | partial | PUBG anchors supervised runs |
| O1 survival capital | strong | weak / masked | Caldera cannot validate this family now |
| O2 contraction/elimination pressure | strong through zones | absent now | zone claims are PUBG-only now |
| O3 movement/rotation | strong | strong trajectory | shared movement vocabulary |
| O4 interaction/threat | strong through combat/contact | weak / no combat labels | Caldera can use density/trajectory proxies only |
| O5 resources/equipment | moderate/strong | absent | PUBG-only now |
| O6 spatial/terrain/environment | weak proxy | strong geometry | Caldera validates construct, PUBG uses proxies |
| O7 relation/interaction | strong in PUBG, possible for Caldera trajectory clock | Caldera LOS possible but no supervised outcome | shared relation vocabulary, different claims |
| O8 outcome/win condition | strong | absent for rank/survival | supervised claims are PUBG-only now |

This matrix is not a weakness of the framework.  It is the framework's
observability model:

```text
same genre ontology
different dataset projections
explicit masks
```

## Framework-Level Feature Catalogue

This is the compact feature universe that should appear in the paper-level
method section.

| Family | Primitive question | Example variables |
| --- | --- | --- |
| Rule/context | What kind of match and phase is this? | map, mode, perspective, phase, alive population |
| Survival capital | How much survivable capacity remains? | alive count, health, vehicle, spread |
| Contraction pressure | How is the match forcing survival decisions? | boundary distance, safe radius, required speed, outside duration, elimination stage |
| Movement/rotation | How is the team moving in response? | path length, speed, rotation timing, entry time |
| Interaction/threat | What opponent or contest pressure is nearby? | enemy density, nearest enemy, combat, damage |
| Resource/mobility | What tools support survival? | loot, healing, consumables, vehicle, resource use |
| Spatial affordance | What does the environment afford? | z, relative height, cover proxy, LOS where available |
| Relations/interaction | How are teams coupled? | proximity, combat, co-pressure, relative advantage |
| Outcome/win condition | What result is being predicted or explained? | next-phase hazard, rank, at-risk mask |

## Model Interface

The ontology should be instantiated as:

```text
X_i(k)   = team/node features through phase cut k
E_ij(k)  = relation/edge features through phase cut k
U(k)     = global match/phase context
Y_i(k+1) = next-phase elimination or other valid supervision
M_i(k)   = at-risk and observability masks
```

The model should be evaluated under:

- family ablations;
- phase-specific metrics;
- observability masks;
- cross-map / cross-mode splits;
- calibration by phase.

## Perspective Boundary

Many Battle Royale telemetry features are analyst-view features, not
player-available perception features.  Examples include global enemy density,
route conflict, full-match proximity fields, and all-team graph relations.

This is acceptable for the current research framing because the framework
models the match as a telemetry-observed survival system.  It should not be
described as a real-time player decision model unless a player-observability
mask is explicitly added.

Use this distinction:

| Perspective | Allowed features | Claim |
| --- | --- | --- |
| analyst-view telemetry | all pre-cut telemetry observed by the dataset | survival environment and relation analysis |
| player-available proxy | only information plausibly visible/known to the agent | decision-support or cognition claims |

## Canonical Tactical State Vocabulary

The ontology owns the tactical-state names used by dataset-specific figures and
proxy summaries.  Lower-level feature documents and code should reference this
vocabulary rather than inventing local state names.

| State | Meaning |
| --- | --- |
| `open_high_ground` | locally high or high-relative position with little cover/contact shielding |
| `covered_high_ground` | high-relative position with nearby cover/occluder or low-damage contact behavior |
| `open_low_ground` | low-relative position with little cover/contact shielding |
| `compound_movement` | movement through built-mass / compound-like environment |
| `corridor_movement` | movement through corridor-like routes, roads, runways, or empirical rotation channels |
| `rough_mobility` | high terrain/mobility cost or high vertical/turning effort |
| `contact_under_pressure` | hostile interaction, contact, or combat while zone/route pressure is high |
| `neutral_or_unclassified` | no stronger tactical-state condition is observed |
| `airborne_or_unresolved` | position is above/below resolved terrain or geometry lookup is unreliable |

## Claim Boundary

Allowed claim:

```text
We propose a Battle Royale feature ontology centered on contraction /
elimination survival pressure, and show how it can be instantiated under
different observability regimes.
```

Stronger claim only after experiments:

```text
Feature families derived from this ontology improve phase-wise survival
prediction and generalize across map/mode or observability conditions.
```

Avoid:

```text
Every feature family is available in every Battle Royale dataset.
Caldera validates PUBG survival prediction.
PUBG contains direct terrain geometry.
```
