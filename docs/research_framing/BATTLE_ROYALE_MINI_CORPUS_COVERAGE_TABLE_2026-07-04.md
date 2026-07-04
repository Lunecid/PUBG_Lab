# Battle Royale Mini Corpus Coverage Table

Date: 2026-07-04
Class: `ClaimBoundary`
Lifecycle: `active`

## Purpose

This document completes the current-paper `OV1` mini corpus check.

It does not try to prove the final taxonomy of all Battle Royale games.  Its
purpose is narrower:

```text
Show that the locked O0-O8 ontology is not merely a PUBG feature list.
```

The table uses representative official, publisher-controlled, or store pages
for source-rich Battle Royale and Battle-Royale-like games.  Each row asks
whether the game requires a new top-level ontology family or whether its
mechanics can be represented as projections under the locked O0-O8 families.

## Decision

The mini corpus supports the current paper claim.

Across the 15 coded games/modes below, no new top-level family is required.
The games differ strongly in projection:

- shooter versus party, puzzle, racing, top-down, and MOBA-like variants;
- full 3D versus 2D/top-down/grid spatial profiles;
- solo, duo, squad, and staged qualification outcomes;
- combat, obstacle, racing, puzzle-attack, and intel-mediated interaction.

Those differences are projection differences under O0-O8, not evidence for an
additional `O9` family.

Allowed claim:

```text
The O0-O8 ontology has representative cross-genre coverage sufficient to use it
as an analytic framework for the current Battle Royale prediction paper.
```

Avoid:

```text
The official game descriptions prove a universal and exhaustive Battle Royale
taxonomy.
```

## Coding Schema

| Field | Meaning |
| --- | --- |
| `Source grade` | `S0` official developer/publisher game page, guide, support page, or official mode page; `S1` publisher-controlled store page |
| `Mode status` | `live`, `discontinued`, `mode_removed`, `legacy_page`, or `unknown` |
| `Spatial profile` | `FULL_3D`, `ELEVATION_2_5D`, `PLANAR_2D`, `GRID_2D`, `TRACK_2D_3D`, or `NON_CLASSIC_SPATIAL` |
| `Coverage` | O-family labels directly or reasonably supported by the source text |
| `Projection note` | How this game maps onto the locked ontology |

Coding is manual and claim-bounded.  A row can support ontology coverage even
when it does not expose telemetry-level measurements.

## Locked Family Reminder

| Family | Locked name |
| --- | --- |
| `O0` | Rule And Match Context |
| `O1` | Agent / Team Survival Capital |
| `O2` | Contraction / Elimination Pressure |
| `O3` | Movement / Rotation Behavior |
| `O4` | Interaction / Threat Exposure |
| `O5` | Resource / Equipment / Mobility Support |
| `O6` | Spatial / Terrain / Environment Affordance |
| `O7` | Relation / Interaction Structure |
| `O8` | Outcome / Win Condition |

## Mini Corpus Coverage Table

Retrieved at: 2026-07-04.

| Game / mode | Corpus stratum | Source | Grade | Mode status | Spatial profile | Coverage | Projection note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PUBG: Battlegrounds | canonical shrinking-zone shooter | https://pubg.com/en/game-info/overview | `S0` | `live` | `FULL_3D` | O0 O1 O2 O3 O4 O5 O6 O7 O8 | Canonical land-loot-survive loop: 100-player start, parachute landing, loot, vehicles, blue-zone pressure, solo/duo/squad, lone-survivor outcome. |
| Fortnite | construction / no-build shooter variant | https://store.epicgames.com/en-US/p/fortnite | `S1` | `live` | `FULL_3D` | O0 O1 O2 O3 O4 O5 O6 O7 O8 | Battle Royale, Zero Build, building/no-build, traversal, cover, squads, and Victory Royale all fit existing families; building is an O6/O5 projection, not a new family. |
| Apex Legends | hero squad shooter | https://www.ea.com/games/apex-legends/apex-legends/features | `S0` | `live` | `FULL_3D` | O0 O1 O2 O3 O4 O5 O6 O7 O8 | 60-person Battle Royale with Legends, squad identity, maps, and ability kits; hero abilities are O5/O1 projections. |
| Call of Duty: Warzone | redeploy / contract shooter | https://www.callofduty.com/warzone | `S0` | `live` | `FULL_3D` | O0 O1 O2 O3 O4 O5 O6 O7 O8 | Battle Royale and Resurgence framing, placement challenges, contracts, movement mechanics, revive/redeploy support, and map POIs project cleanly into O0-O8. |
| Garena Free Fire / Free Fire MAX | mobile survival shooter | https://ff.garena.com/en/ | `S0` | `live` | `FULL_3D` | O0 O1 O2 O3 O4 O5 O6 O7 O8 | Mobile 10-minute survival shooter with Battle Royale modes, characters, maps, weapons, movement, and last-standing objective. |
| Super Animal Royale | 2D top-down shooter | https://animalroyale.com/ | `S0` | `live` | `PLANAR_2D` | O0 O1 O2 O3 O4 O5 O6 O7 O8 | 64 animals drop from the sky; last squad/person survives.  It supports the rule that O6 is not 3D-only: 2D island layout, routes, weapons, and cover still create spatial affordance. |
| Brawl Stars Showdown / battle royale modes | small-lobby mobile arena BR-like | https://supercell.com/en/games/brawlstars/ | `S0` | `live` | `PLANAR_2D` | O0 O1 O2 O3 O4 O5 O6 O7 O8 | Official page frames Brawl Stars as 3v3 and battle royale.  The source is coarse, so the row is medium-confidence, but it supports small-lobby, short-session BR projection. |
| NARAKA: BLADEPOINT | melee / martial-arts hero BR | https://store.steampowered.com/app/1203220/NARAKA_BLADEPOINT/ | `S1` | `live` | `FULL_3D` | O0 O1 O2 O3 O4 O5 O6 O7 O8 | Battle Royale with heroes, melee/ranged weapons, parkour, grappling, mountains/buildings, and last-fighter-standing logic.  Melee combat is an O4 projection, not a separate genre family. |
| Fall Guys | party / obstacle royale | https://store.epicgames.com/en-US/p/fall-guys | `S1` | `live` | `NON_CLASSIC_SPATIAL` | O0 O1 O2 O3 O4 O5 O6 O7 O8 | Party royale with obstacle courses and last Bean / Crown outcome.  O2 is staged elimination and qualification pressure rather than a storm circle. |
| Tetris 99 | puzzle Battle Royale | https://www.nintendo.com/us/store/products/tetris-99-switch/ | `S0` | `live` | `GRID_2D` | O0 O1 O2 O3 O4 O5 O6 O7 O8 | 99-player puzzle competition with K.O. targeting, badges, garbage-block attacks, team mode, and last-player/team-standing outcome.  Spatial affordance is grid/board-state, not terrain. |
| F-ZERO 99 | racing Battle Royale-like | https://www.nintendo.com/us/store/products/f-zero-99-switch/ | `S0` | `live` | `TRACK_2D_3D` | O0 O1 O2 O3 O4 O5 O6 O7 O8 | 99-player racing with survival under speed/power-meter pressure and scheduled qualifying modes.  O4 is racing contest/collision pressure rather than gun combat. |
| Eternal Return | 2.5D MOBA / crafting BR | https://store.steampowered.com/app/1049590/Eternal_Return/ | `S1` | `live` | `ELEVATION_2_5D` | O0 O1 O2 O3 O4 O5 O6 O7 O8 | 2.5D Battle Royale with search, craft, hunt, fight, character selection, strategy, and last-survivor outcome. |
| Battlerite Royale | top-down Action MOBA BR | https://store.steampowered.com/app/879160/Battlerite_Royale/ | `S1` | `unknown` | `PLANAR_2D` | O0 O1 O2 O3 O4 O5 O6 O7 O8 | Top-down skillshot Action MOBA with champions, loot, Death Vortex, island traversal, duos, and resurrection shrine. |
| Realm Royale Reforged | fantasy class / fog / revive variant | https://store.steampowered.com/app/813820/Realm_Royale_Reforged/ | `S1` | `discontinued` | `FULL_3D` | O0 O1 O2 O3 O4 O5 O6 O7 O8 | Class, forge, mount, fog, team support, revive/chicken mechanic, and last-champion outcome.  The store page explicitly notes service shutdown, so lifecycle is separated from source grade. |
| Counter-Strike: Danger Zone | tactical-shooter BR mode / intel boundary case | https://www.counter-strike.net/dangerzone | `S0` | `mode_removed` / `legacy_page` | `FULL_3D` | O0 O1 O2 O3 O4 O5 O6 O7 O8 | 16/18-player CS:GO battle royale mode with resource ordering, missions/contracts, tablet intel, solos/duos/trios, and short matches.  Intel is coded by rule: acquisition means O5, changed exposure O4, relation knowledge O7, visibility/pathing O6. |

## Coverage Summary

| Family | Corpus support | Interpretation |
| --- | --- | --- |
| `O0` | 15/15 | Every row has a bounded match/mode context. |
| `O1` | 15/15 | Survival capital appears as health, roster, lives, shields, vehicle/power meter, badges, character state, or qualification capacity. |
| `O2` | 15/15 | Pressure appears as shrinking zone, fog, ring, vortex, stage elimination, qualification, racing attrition, or board-pressure escalation. |
| `O3` | 15/15 | Movement generalizes to parachuting, rotating, parkour, racing, obstacle traversal, board placement, or route/path decisions. |
| `O4` | 15/15 | Threat generalizes beyond gun combat to melee, obstacle contest, racing collision, puzzle attack, and target selection. |
| `O5` | 15/15 | Resource support includes loot, weapons, abilities, vehicles, badges, power-ups, blocks, contracts, tablets, or craftable gear. |
| `O6` | 15/15 | Environment affordance exists in full 3D, 2D islands, tracks, obstacle courses, or puzzle grids; measurement observability varies by dataset. |
| `O7` | 15/15 | Relation structure exists as enemy/squad/proximity/targeting/contest/support relations, even when telemetry is not available. |
| `O8` | 15/15 | Outcomes are last survivor/team, crown, qualification, race survival, K.O. victory, or final placement. |

## Cross-Stratum Findings

### O2 Is Broader Than Zone

The corpus confirms the wording change from `zone pressure` to `contraction /
elimination pressure`.

Classic shooters use storm, fog, ring, blue zone, gas, or vortex.  Fall Guys
uses staged qualification and elimination.  Tetris 99 uses escalating board
pressure and attack relations.  F-ZERO 99 uses racing survival, power-meter
pressure, and qualification.  These are all pressure schedules that change the
viable survival condition over time.

### O4 Is Broader Than Combat

The corpus confirms the wording change from `combat` to `interaction / threat
exposure`.

PUBG, Fortnite, Apex, Warzone, Free Fire, Naraka, Battlerite, Realm, and Danger
Zone have direct hostile combat.  Fall Guys has obstacle and qualification
contest.  Tetris 99 has targeted garbage-block attacks.  F-ZERO 99 has racing
collision, boost, and outlive/outpace pressure.  The common construct is
interaction that can remove, endanger, block, or qualify agents.

### O6 Is Not Only 3D Terrain

The corpus supports the two-axis rule:

```text
concept portability != measurement observability
```

O6 is conceptually present in all rows, but its measurement form differs:

- `FULL_3D`: terrain, buildings, verticality, line of sight, routes;
- `PLANAR_2D`: cover, island layout, route bottlenecks, safe cells;
- `ELEVATION_2_5D`: top-down or isometric layout plus region affordances;
- `GRID_2D`: board occupancy, attack lanes, and pressure on playable cells;
- `TRACK_2D_3D`: racing line, collision space, boost/survival tradeoff, and
  qualification routes;
- `NON_CLASSIC_SPATIAL`: obstacle course geometry and qualification routes.

Therefore, a Battle Royale framework must not require native 3D terrain data.
It must require a spatial/environmental affordance slot whose measurable
projection depends on the game and dataset.

### Information Mechanics Do Not Require O9

Danger Zone and Warzone confirm that intel mechanics can be handled without
adding an `O9` information family:

- the item or contract that grants information is O5;
- the changed threat/exposure state is O4;
- visibility/path/occlusion affordance is O6;
- known or inferred opponent relation is O7.

Corpus rows should retain an `intel_flag` when information is central.

## Boundary Decisions

### What This Completes

This document completes the current paper's mini corpus requirement:

```text
OV1 mini coverage table: complete.
```

It allows the following argument:

```text
The ontology is not PUBG-only.  Official/source-controlled descriptions from
representative Battle Royale variants map into O0-O8 without requiring a new
top-level family.
```

### What Remains Outside This Artifact

The following remain future or appendix work:

- exhaustive enumeration of all Battle Royale games;
- multi-coder agreement over paragraph-level chunks;
- Web Archive snapshots for discontinued long-tail games;
- negative-control table over non-BR games;
- automatic NLP chunking/codebook generation.

These are useful for a larger corpus study, but they are not blocking
prerequisites for the current prediction paper.

## How To Use In The Manuscript

Use this document as a short appendix table or a compressed paper table.

Recommended prose:

```text
We did not derive the ontology from PUBG alone.  A mini corpus of 15
source-rich Battle Royale and Battle-Royale-like games, spanning canonical
shooters, mobile variants, top-down/MOBA variants, party royale, racing, and
puzzle royale, maps to O0-O8 without requiring an additional top-level family.
This supports the ontology as a genre-level analytic vocabulary, while the
predictive utility of the vocabulary is tested separately in PUBG telemetry.
```

Do not use this corpus as evidence for model performance.  That evidence comes
from the cohort-separated PUBG OM ladder and the follow-up phase/uncertainty
analyses.

## Next Action

Return to the supervised evidence path:

```text
phase-stratified OM aggregation
paired uncertainty checks
O6 Z-ladder implementation
O7 edge-stream / graph-relation tests
```
