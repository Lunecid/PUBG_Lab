# RQ1 STDB Redesign

## Goal

RQ1 asks whether PUBG telemetry can be represented as a spatio-temporal
database. The redesigned database keeps the original STDB abstraction:

- Moving objects: players and team centroids over time.
- Dynamic regions: safe / poison zone polygons over time.
- Spatial events: combat, damage, kill, groggy, landing, and item events.

The v2 layer adds phase-level PUBG domain features so the database can support
both analysis and next-phase hazard modeling.

## Raw-to-STDB Mapping

| Raw source | STDB object | Feature layer |
|---|---|---|
| `pubg.telem_positions` | `moving_objects` | `team_phase_state` |
| `pubg.telem_game_states` | `dynamic_regions` | `match_phase_global` |
| `pubg.telem_phase_changes` | phase recovery | `v_match_phase_bounds` |
| `pubg.telem_damage` | `spatial_events` | `team_phase_combat` |
| `pubg.telem_groggy` | `spatial_events` | `team_phase_combat` |
| `pubg.telem_kills` | `spatial_events` | `team_phase_combat` |
| `pubg.telem_parachute_landing` | `spatial_events` | trajectory/drop analysis |
| `pubg.telem_item_pickup` | raw item events | `team_phase_loot` |
| `pubg.telem_item_equip` | raw item events | `team_phase_loot` |
| `pubg.telem_item_use` | raw item events | `team_phase_loot` |

## New Tables

| Table | Purpose |
|---|---|
| `stdb.match_phase_global` | Phase context: alive population, zone radius/shift/gap, weather, blue-zone options |
| `stdb.team_phase_state` | Basic team state: alive members, health, squad spread, vehicle usage |
| `stdb.team_phase_combat` | Directed combat: dealt/taken damage, groggy/kill, initiation, third-party exposure |
| `stdb.team_phase_loot` | PUBG readiness: weapons, ammo, armor, throwables, heals, boosts, carepackage/lootbox pickup |
| `stdb.team_phase_labels` | Leakage-safe next-phase hazard labels and final-rank auxiliary labels |
| `stdb.v_team_phase_feature_long` | One row per `(match, team, phase)` joining global/state/trajectory/exposure/combat/loot/labels |

## Modeling Semantics

The model unit is:

```text
(match_id, team_id, cut_phase)
```

The intended supervised target is:

```text
X observed through cut_phase  ->  hazard_next = eliminated during next phase
```

This avoids same-phase leakage. Full-match final rank and Top-K labels stay
auxiliary or reporting targets.

## What This Redesign Fixes

- Adds PUBG-specific state features that were missing from centroid-only
  trajectory features.
- Separates directed combat features from undirected proximity.
- Adds loot/readiness signals from item telemetry already present in raw DB.
- Preserves global zone context as a first-class phase table.
- Materializes leakage-safe labels instead of deriving them ad hoc inside model
  code.

## Current Limitations

- Terrain, buildings, cover, line-of-sight, and true map traversal difficulty
  are not derivable from telemetry alone.
- Vehicle speed is preserved from raw telemetry as reported by the source; it is
  not recalibrated against coordinate deltas yet.
- `team_phase_combat.nearby_enemy_combat_events_200m` and
  `third_party_exposure_count` are schema placeholders in the current build.
  The first direct `ST_DWithin` implementation was too expensive on some
  matches, so this should be rebuilt later from a cached combat-point table.
- `team_phase_loot` uses item-id/category pattern rules. If a paper-grade item
  taxonomy is needed, add a curated item dimension table.
- Late phases have fewer samples because most teams have already been
  eliminated; evaluation should report cut phase explicitly.

## Build Command

```bash
python -m stdb.rq1_feature_etl --apply-schema --all-current --verify
```

Smoke test one match first:

```bash
python -m stdb.rq1_feature_etl --apply-schema --limit 1 --verify
```

## Current Build Status

As of the current 100-match STDB subset:

| Asset | Rows |
|---|---:|
| `match_phase_global` | 669 |
| `team_phase_state` | 16,881 |
| `team_phase_combat` | 16,881 |
| `team_phase_loot` | 16,881 |
| `team_phase_labels` | 16,881 |
| `v_team_phase_feature_long` | 16,881 |
| next-phase target rows | 4,308 |
| positive hazard rows | 1,350 |

Phase coverage: phase 1-5 exist for all 100 matches, phase 6 for 91 matches,
phase 7 for 54, phase 8 for 21, and phase 9 for 3. Model evaluation should
therefore report cut-phase results explicitly and treat late phases as sparse.
