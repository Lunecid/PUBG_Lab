# Battle Royale Ontology Corpus Design

Date: 2026-07-02
Updated: 2026-07-03

## Purpose

This document defines the text corpus and coding protocol used to validate the
Battle Royale ontology around the prediction experiments.

The purpose is not to prove predictive utility.  Predictive utility is tested
later in PUBG telemetry.  This document supports an earlier claim:

```text
The O0-O8 ontology covers recurring Battle Royale genre structure across
representative game descriptions, official pages, store descriptions, and
gameplay documents.
```

This is a coverage and construct-validation document, not a supervised-learning
result.

## Research Sequence

The paper critical path should proceed in this order:

```text
1. ontology v1 lock using the mini corpus sanity check
2. Battle Royale genre formulation
3. PUBG telemetry operationalization
4. PUBG predictive utility test
5. game-specific incremental feature test
6. Caldera / Warzone spatial construct validation
7. extended corpus confirmation and future geometry-rich prediction
```

The full corpus protocol is confirmatory, not a hard gate for PUBG prediction.
For the current paper path, the required corpus product is a compact
representative coverage table over the already sourced S0/S1 games plus a
limitations paragraph.  The larger multi-game, multi-coder saturation study can
continue in parallel or become a separate game-studies contribution.

The mini corpus sanity check still matters: it prevents the feature ontology
from being a PUBG-only feature list disguised as a genre theory.

Current mini corpus artifact:

```text
docs/research_framing/BATTLE_ROYALE_MINI_CORPUS_COVERAGE_TABLE_2026-07-04.md
```

For the current paper path, this artifact satisfies the compact OV1 coverage
table requirement.  The larger corpus protocol remains confirmatory/future
work.

## Source Universe

The initial universe is seeded from:

- Wikipedia, "List of battle royale games":
  https://en.wikipedia.org/wiki/List_of_battle_royale_games
- Steam, "Battle Royale" tag pages and publisher-controlled store pages;
- official platform or publisher examples outside the Wikipedia seed, including
  Nintendo *99-style titles and major party-royale variants.

The Wikipedia page is treated only as one candidate seed.  It is not treated as
the final source of gameplay mechanics or as an exhaustive boundary of the
genre.  This avoids circularity where Wikipedia editors' Battle Royale boundary
quietly becomes this paper's ontology boundary.

For each candidate, source quality is graded as follows.

| Grade | Meaning | Use |
| --- | --- | --- |
| `S0_OFFICIAL_GAME` | official developer/publisher game page, guide, support page, or patch page | preferred source for mechanic coding |
| `S1_OFFICIAL_STORE` | Steam, Epic Games Store, console store, app store page controlled by publisher/developer | acceptable when a live official guide is missing |
| `S2_OFFICIAL_NEWS` | official announcement, patch note, shutdown notice, or season page | useful for modes and discontinued cases |
| `S3_SECONDARY` | Wikipedia, press, fan wiki, review, guide | only for candidate discovery or unavailable historical titles |

Coding rule:

```text
An ontology claim should not rely only on S3 unless the game or mode is
historical, discontinued, or no official/source-controlled page is available.
```

Source grade measures provenance, not live status.  Therefore every game/mode
also needs a lifecycle field:

| Field | Values |
| --- | --- |
| `mode_status` | `live`, `discontinued`, `mode_removed`, `relaunched`, `legacy_page`, `unknown` |
| `retrieved_at` | date when the source text was captured |
| `archive_url` | Web Archive or local snapshot URL/path when possible |

This distinction is required because an official page can remain online after a
mode has been removed or discontinued.

## Candidate Universe

This table is the v0 universe for corpus construction.  The source status is a
working audit status, not the final coded dataset.

| Year | Game / Mode | Corpus role | Source status | Notes for ontology validation |
| --- | --- | --- | --- | --- |
| 2017 | PUBG: Battlegrounds | core BR | S0 found | canonical shrinking-zone survival shooter; also supervised telemetry anchor |
| 2017 | Fortnite Battle Royale | core BR / construction variant | S1 found | building / no-build split tests whether O6 is geometry or affordance |
| 2017 | Garena Free Fire | mobile core BR | S0 found | short mobile matches, character layer, squad/mobile projection |
| 2017 | Surviv.io | 2D shooter variant | S3 needed | top-down 2D BR; official service halted |
| 2017 | Rules of Survival | historical mobile BR | S3 plus archived official needed | high-player-count PUBG-like mobile/PC case, discontinued |
| 2017 | Knives Out | historical mobile BR | source pending | NetEase PUBG-like mobile/PC case |
| 2018 | The Culling | historical melee/survival BR | source pending | first-person survival BR, discontinued/relaunched history |
| 2018 | Path of Exile: Royale | mode-based ARPG BR | S3 plus official news needed | non-standalone top-down mode; useful for mode projection |
| 2018 | Realm Royale / Reforged | class/fantasy BR | S1 found | class, forge, mount, revive/chicken mechanics |
| 2018 | Totally Accurate Battlegrounds | parody physics BR | S1 found | mechanics are comic, but last-standing/arena logic remains |
| 2018 | Z1 Battle Royale | early core BR | S3 plus archived official needed | important historical predecessor; official live source weak |
| 2018 | Rec Royale | VR / mode BR | source pending | VR first-person mode, small lobby |
| 2018 | Creative Destruction | construction BR clone | S3 plus archived official needed | Fortnite-like building/destruction projection |
| 2018 | Ring of Elysium | weather/extraction BR | S3 plus archived official needed | rescue/extraction ending, traversal gear, storm/fog pressure |
| 2018 | Battlerite Royale | MOBA/top-down BR | S1 found | top-down skillshot MOBA projection |
| 2018 | Blackout | franchise mode BR | source pending | non-standalone Call of Duty BR mode |
| 2018 | Cyber Hunter | sci-fi movement/building BR | S3 plus archived official needed | hoverboard, parkour, gliding, defensive construction |
| 2018 | Danger Zone | tactical-shooter mode BR | S0 legacy found | small-lobby CS:GO mode; economy/intel/tablet projection; mark `mode_status=mode_removed` / `legacy_page` after CS2 status audit |
| 2018 | Brawl Stars Showdown | mobile arena BR-like mode | S0 found | 10-player short match; tests small-lobby abstraction |
| 2019 | Apex Legends | hero/squad core BR | S0 found | 60-person squad BR, hero abilities |
| 2019 | Firestorm | franchise mode BR | source pending | Battlefield mode with firestorm boundary |
| 2019 | Nuclear Winter | franchise mode BR | source pending | Fallout 76 mode; discontinued |
| 2019 | Rogue Heist | historical shooter BR-like | source pending | may be low priority if source quality weak |
| 2019 | Red Death | strategy/4X BR-like mode | source pending | Civilization VI mode; important non-shooter negative-edge case |
| 2019 | Call of Duty: Mobile BR | mobile franchise BR | S3 plus official source pending | mobile class/loadout/safe-zone projection |
| 2019 | ScarFall | mobile BR | source pending | regional mobile BR, source quality likely weaker |
| 2019 | CRSED: F.O.A.D. | supernatural shooter BR | source pending | ritual/ability projection |
| 2019 | The Eliminator, Forza Horizon 4 | racing BR-like mode | S3 found | shrinking safe zone without combat shooting; key genre boundary test |
| 2020 | Darwin Project | survival/arena BR | source pending | director/survival-crafting variant |
| 2020 | Call of Duty: Warzone | core BR / redeploy variant | S0 found | Gulag/redeploy and buy station mechanics; Caldera construct anchor |
| 2020 | Hyper Scape | vertical urban BR | source pending | fast movement, sectors collapsing; discontinued |
| 2020 | Fall Guys | party/obstacle royale | S1 found | staged elimination without classic shooting/loot |
| 2020 | Spellbreak | magic/combat BR | source pending | class/magic gauntlet projection; discontinued |
| 2020 | Farlight 84 | hero/mobile/vehicle BR | S1 found | hero, pet/buddy, respawn, zone-shift projection |
| 2020 | Eternal Return | MOBA/survival/crafting BR | S1 found | restricted zones, crafting, top-down 2.5D projection |
| 2020 | Population: One | VR BR | source pending | VR vertical/climbing/gliding case |
| 2021 | Sausage Man | mobile cartoon BR | source pending | mobile/character-humor projection |
| 2021 | Battlegrounds Mobile India | PUBG Mobile regional projection | source pending | PUBG family, but useful for regional/publisher projection |
| 2021 | Naraka: Bladepoint | melee/martial-arts BR | S0 and S1 found | parkour/grappling/melee BR; 3D but not shooter-centric |
| 2021 | Super Animal Royale | 2D top-down BR | S0 found | strong test that O6 is not 3D-only |
| 2021 | Vampire: The Masquerade - Bloodhunt | vertical urban BR | source pending | vampire powers, rooftops, urban verticality |
| 2021 | The Eliminator, Forza Horizon 5 | racing BR-like mode | S3 found | sequel mode; racing elimination pressure |
| 2021 | Babble Royale | word/puzzle BR-like | S3 found | non-combat interaction via word placement |
| 2019 | Tetris 99 | puzzle BR-like | source pending outside Wikipedia seed | canonical 99-player puzzle Battle Royale; add to avoid Wikipedia-seed blind spot |
| 2020 | Super Mario Bros. 35 | platformer BR-like | source pending outside Wikipedia seed | 35-player limited-time platformer competition; useful discontinued official-mode case |
| 2023 | F-ZERO 99 | racing BR-like | source pending outside Wikipedia seed | 99-player racing Battle Royale; stronger racing stratum than Forza-only mode evidence |
| 2021 | Stumble Guys | party / obstacle royale | source pending outside Wikipedia seed | mobile/party knockout variant; complements Fall Guys |
| 2022 | Call of Duty: Warzone 2.0 / Warzone | core BR / redeploy variant | S0 found | modern Warzone label; source merged with current Warzone page |
| 2024 | World of Warcraft: Plunderstorm | MMORPG mode BR | source pending | mode-based BR with WoW controls/classes |
| 2024 | Call of Duty: Warzone Mobile | mobile Warzone projection | S3 plus official shutdown/source pending | discontinued; useful for mobile projection and redeploy |
| 2024 | Supervive | MOBA/hero BR | S3 plus official source pending | hunter/team/storm/isometric projection |
| 2025 | Sonic Rumble / Sonic Rumble Party | party BR-like | S3 plus official source pending | staged elimination and ring objective; boundary case |

## Representative Source Notes

The following sources are already useful enough for the v1 coding pass.

| Game / Mode | Source | Grade | Mode status | Mechanic evidence useful for coding |
| --- | --- | --- | --- | --- |
| PUBG: Battlegrounds | https://pubg.com/en/game-info/overview | `S0_OFFICIAL_GAME` | `live` | 100-player land/loot/survive loop, vehicles, care packages, blue zone, solo/duo/squad |
| Fortnite | https://store.epicgames.com/en-US/p/fortnite | `S1_OFFICIAL_STORE` | `live` | Battle Royale, Zero Build, building/no-build distinction, squad modes |
| Garena Free Fire | https://ff.garena.com/en/ | `S0_OFFICIAL_GAME` | `live` | 10-minute mobile survival shooter, battle royale modes, last-standing objective |
| Apex Legends | https://www.ea.com/games/apex-legends/apex-legends/features | `S0_OFFICIAL_GAME` | `live` | 60-person battle royale, Legends, squad/team identity, maps |
| Call of Duty: Warzone | https://www.callofduty.com/warzone | `S0_OFFICIAL_GAME` | `live` | Battle Royale/Resurgence framing, placement challenges, contracts, movement mechanics |
| Counter-Strike: Danger Zone | https://www.counter-strike.net/dangerzone | `S0_OFFICIAL_GAME` | `mode_removed` / `legacy_page` | legacy CS:GO page; 16/18-player mode, resource ordering, intel tablet, solos/duos/trios |
| Brawl Stars | https://supercell.com/en/games/brawlstars/ | `S0_OFFICIAL_GAME` | `live` | official 3v3 and battle royale positioning |
| Super Animal Royale | https://animalroyale.com/ | `S0_OFFICIAL_GAME` | `live` | 64 animals drop, last squad/person survival, 2D top-down mode |
| Naraka: Bladepoint | https://www.narakathegame.com/ and https://store.steampowered.com/app/1203220/NARAKA_BLADEPOINT/ | `S0_OFFICIAL_GAME` / `S1_OFFICIAL_STORE` | `live` | martial-arts BR, heroes, parkour, grappling, melee/ranged mix |
| Realm Royale Reforged | https://store.steampowered.com/app/813820/Realm_Royale_Reforged/ | `S1_OFFICIAL_STORE` | `unknown` | class, forge, mount, fog, squad, chicken/revive mechanic |
| Battlerite Royale | https://store.steampowered.com/app/879160/Battlerite_Royale/ | `S1_OFFICIAL_STORE` | `unknown` | top-down Action MOBA BR, Death Vortex, champion abilities, resurrection shrine |
| Eternal Return | https://store.steampowered.com/app/1049590/Eternal_Return/ | `S1_OFFICIAL_STORE` | `live` | 2.5D BR, search/craft/hunt/fight loop, last survivor |
| Farlight 84 | https://store.steampowered.com/app/1928420/Farlight_84/ | `S1_OFFICIAL_STORE` | `live` | 60-player hero shooter, hero/pet skills, respawn, zone manipulation |
| Totally Accurate Battlegrounds | https://store.steampowered.com/app/823130/Totally_Accurate_Battlegrounds/ | `S1_OFFICIAL_STORE` | `unknown` | last-standing physics-based BR parody |
| Fall Guys | https://store.epicgames.com/en-US/p/fall-guys | `S1_OFFICIAL_STORE` | `live` | party royale, obstacle courses, last Bean standing, staged minigames |

## Coding Unit

Each source should be split into coding units:

- paragraph;
- bullet item;
- mode description;
- FAQ answer;
- store description block;
- official guide subsection.

Each unit receives:

| Field | Values |
| --- | --- |
| `game_id` | normalized game or mode name |
| `source_url` | URL |
| `source_grade` | `S0_OFFICIAL_GAME`, `S1_OFFICIAL_STORE`, `S2_OFFICIAL_NEWS`, `S3_SECONDARY` |
| `mode_status` | `live`, `discontinued`, `mode_removed`, `relaunched`, `legacy_page`, `unknown` |
| `retrieved_at` | ISO date |
| `archive_url` | archive URL or local snapshot path, nullable |
| `primary_ontology_family` | `O0` to `O8`, or `OUT_OF_SCOPE` |
| `secondary_ontology_families` | zero or more additional `O0` to `O8` labels |
| `projection_level` | `BR_GENERAL`, `GAME_SPECIFIC`, `DATASET_SPECIFIC`, `GEOMETRY_SPECIFIC` |
| `spatial_profile` | `PLANAR_2D`, `GRID_2D`, `ELEVATION_2_5D`, `FULL_3D`, `GEOMETRY_RICH`, `GEOMETRY_POOR`, `NON_SPATIAL` |
| `intel_flag` | `true` when the chunk concerns scan, reveal, ping, UAV, radar, contract intel, forecast, killfeed, or intel-tablet mechanics |
| `mechanic_text` | short paraphrase of the coded mechanic |
| `confidence` | `high`, `medium`, `low` |
| `coder_note` | ambiguity, missing source, or proposed ontology revision |

Multi-label rule:

```text
Chunks can have one primary family and multiple secondary families.  For
example, "loot to survive the storm" can be coded as primary O5 with secondary
O2 rather than forcing an artificial single-family disagreement.
```

The corpus can be produced manually first, then semi-automated with NLP.  NLP is
allowed as an assistive coder, but the ontology lock must depend on human
adjudication for disputed chunks.

## Ontology Coverage Matrix

Initial cross-game reading supports the O0-O8 structure and has already
informed the v1 wording of O2 and O4.

| Family | Locked meaning | Cross-game finding | Lock recommendation |
| --- | --- | --- | --- |
| `O0` rule/match context | mode, map, lobby, team size, clock | present in all candidates as rule frame | keep |
| `O1` agent/team survival capital | health, lives, team capacity, character state | present, but represented as HP, vehicle tier, class, rings, crowns, or lives depending on game | keep but avoid shooter-only terms |
| `O2` contraction/elimination pressure | viable survival conditions compressed by shrinking space, staged elimination, route compression, or qualification pressure | classic shooters use storm/fog/gas/blue zone; Fall Guys/Sonic use staged elimination; Forza uses shrinking arena plus races; Eternal Return uses restricted areas | locked as `contraction / elimination pressure` |
| `O3` movement/rotation behavior | traversal, repositioning, path choice | universal but modality varies: parachute, vehicle, parkour, grappling, racing, word placement, obstacle traversal | keep |
| `O4` interaction/threat exposure | interaction that can remove, endanger, block, or qualify agents | universal, but not always gun combat; can be race duel, obstacle contest, word connection, brawler fight | locked as interaction/threat; combat is a game-specific projection |
| `O5` resource/equipment/mobility support | loot, items, abilities, vehicles | strong in shooters and survival variants; weaker but still present as abilities, powerups, rings, crowns, car drops, spells | keep |
| `O6` spatial/terrain/environment affordance | position and environmental advantage | universal as affordance, not universal as 3D terrain; 2D and party games use lanes, sectors, routes, obstacles, safe cells | keep with mandatory spatial-profile declaration |
| `O7` relation/interaction structure | team/enemy/proximity/contact/support/contest relations | universal at conceptual level; graph observability depends on telemetry or analyst reconstruction | keep as game-readable relation language |
| `O8` outcome/win condition | survival, rank, win/loss, elimination, qualification | universal; exact labels vary from rank, crown, last team, escape, points, round qualification | keep; supervision is an operational projection |

## Common Battle Royale Minimum

The corpus suggests a safer minimal definition than "shrinking-zone shooter."

For coding, a Battle Royale or BR-like mode must satisfy all required criteria
and at least two optional criteria.

Required criteria:

1. many agents or teams enter a bounded match;
2. the match has a terminal survival, rank, qualification, or winner outcome;
3. the playable or viable field narrows over time, either spatially or through
   staged elimination;
4. the pressure schedule forces convergence, encounter, risk, or qualification.

Optional criteria:

5. agents begin from a relatively reset or constrained state;
6. agents improve survival capital during the match through loot, skills,
   upgrades, resources, position, or movement;
7. agents are exposed to other agents through contact, proximity, competition,
   or shared pressure;
8. the winning condition is defined at the match/mode level, not as permanent
   campaign progression.

This minimum supports a broad genre formulation while still excluding ordinary
team deathmatch, MOBA lane play, racing, and survival games that lack the
contraction/elimination-pressure structure.

## Feature-Layer Interpretation

The corpus supports a three-layer feature design.

This is a conceptual-portability axis.  Measurement availability is a separate
observability axis.  A concept can be `BR_GENERAL` while its measurement is
`GEOMETRY_SPECIFIC` or dataset-specific.

| Layer | Meaning | Examples |
| --- | --- | --- |
| `BR_GENERAL` | features every BR-like game can express in some projection | alive count, team size, phase/round index, safe/unsafe pressure, movement rate, interaction exposure, resource gain, outcome |
| `GAME_SPECIFIC` | game-native implementations of BR_GENERAL | PUBG blue zone, Fortnite build/no-build, Apex Legend abilities, Warzone Gulag, Realm Royale forge/chicken, Forza car tier |
| `DATASET_SPECIFIC` | only available in a particular telemetry schema | STDB phase table, PUBG `pos_z`, dataset-specific item names |
| `GEOMETRY_SPECIFIC` | only available when map/mesh/visibility assets exist | Caldera mesh slope, building distance, pairwise LOS |

The predictive paper should first test `BR_GENERAL` features in PUBG, then add
PUBG-specific features, then optionally add more dataset-specific or geometry
proxies.

Intel coding rule:

```text
Code the information-acquisition means as O5, the changed threat/exposure state
as O4, spatial visibility/occlusion as O6, and known/inferred relations as O7.
Use intel_flag=true so these chunks can be audited separately.
```

## Corpus Validation Tracks

The corpus work has two tracks.

### V0. Mini Corpus Sanity Check

Use the already sourced S0/S1 games to build one paper table:

```text
game / source grade / mode status / O0-O8 coverage / notable projection /
missing or masked families
```

Pass condition:

```text
At least 12-15 representative games cover the main genre strata and do not
force a new top-level family beyond O0-O8.
```

This is enough to lock the ontology for the PUBG prediction experiments.  The
remaining C1-C4 protocol is a confirmatory extension, not a prerequisite for
OM0-OM6.

### C1. Coverage

For each game/mode, code official/store text into O0-O8.

Pass condition:

```text
Core BR games cover O0-O8 or explain missing families as observability/source
gaps.  BR-like variants should still map to O0, O1, O2, O3, O4/O7, and O8.
```

### C2. Saturation

Add games by strata:

- core shooter BR;
- mobile BR;
- hero/class BR;
- top-down / 2D BR;
- party / obstacle royale;
- racing / puzzle / strategy BR-like modes;
- franchise modes rather than standalone games.

Pass condition:

```text
New games introduce new projections under O0-O8 rather than repeatedly forcing
new top-level families O9, O10, ...
```

Stop rule:

```text
After the mini corpus, the confirmatory corpus can stop when two consecutive
new strata add zero new top-level families and only add projections under
existing O0-O8 families.
```

### C3. Adjudication Audit

The realistic v1 setup is:

- LLM-assisted first-pass coding;
- human adjudication for top-level family labels and projection levels;
- optional second human coder if available.

Pass condition:

```text
For a two-human subset, family-level Cohen's kappa should be at least 0.70.
If only one human adjudicator is available, report this as an adjudication audit
rather than an inter-coder agreement study.
```

### C4. Negative Controls

Code several non-BR games:

- ordinary tactical shooter;
- ordinary team deathmatch arena shooter;
- conventional MOBA;
- conventional racing game;
- survival crafting game without last-standing match structure.

Pass condition:

```text
Negative controls fail at least two required BR-like criteria, or they lack the
O2 -> O3 -> O4/O7 -> O8 contraction-pressure chain.
```

## What This Can And Cannot Claim

Allowed claims:

- O0-O8 is a stable v1 analytic ontology for the current prediction framework.
- The mini corpus confirms enough coverage to defend the framework as not
  PUBG-only.
- The ontology is not PUBG-only because it covers shooter, mobile, top-down,
  party, racing, puzzle, and mode-based variants.
- Game-specific mechanics can be treated as projections of genre-general
  constructs.
- Spatial affordance is not equivalent to 3D terrain; 3D terrain is one rich
  projection of O6.

Forbidden claims at this stage:

- NLP or text coverage proves survival prediction performance.
- Caldera proves rank prediction without outcome labels.
- Every Battle Royale game has been exhaustively surveyed.
- This paper defines the final taxonomy of the Battle Royale genre.
- All Battle Royale games require 3D terrain, gun combat, loot, or a shrinking
  circular zone.

## Immediate Lock Implications

The lock update on 2026-07-03 propagates the first two items into the ontology,
PUBG operational manifest, validation plan, and feature-engineering README.
Remaining work is implementation/codebook work.

1. `O2` is stated as `contraction / elimination pressure`.
   Shrinking zones are the most common projection, not the only projection.
2. `O4` examples use `interaction / threat exposure` as the
   genre-general phrase.  Gun combat is a projection.
3. `O6` already has the correct rule: every feature must declare spatial
   profile, and terrain/LOS/mesh are optional geometry-rich projections.
4. PUBG prediction remains the main predictive validation.
5. Caldera/Warzone remains geometry-rich construct validation unless labels are
   rebuilt.

## Next Artifact

The next artifact should be a coded CSV or table:

```text
docs/research_framing/BATTLE_ROYALE_ONTOLOGY_CORPUS_CODEBOOK_2026-07-02.md
```

That codebook should store actual source chunks and O0-O8 labels.  This design
document only defines the corpus and validates the research route.
