# pubg_lab.survival.phase_v2

Responsibility: BR/PTR/survival representation and model components.

Dependency boundary: Data contracts; private split access remains outside the public package.

| Component | Responsibility |
|---|---|
| `alive.py` | Causal alive status of players and teams, and the final team elimination time (labels / risk set only). |
| `assemble.py` | One match -> the v2 dump record (same schema as model_stgcn/dump_phase_tier.py v1 plus the v2 additions). |
| `events.py` | Team-attributed combat events and loot counters from the raw event tables (ports of stdb.etl spatial events, |
| `features.py` | All 150 phase-tier features at ONE landmark cut, from records strictly before the cut (spec rule 2). |
| `grid.py` | The 10 s team grid on the LogMatchStart clock W (v2 replacement of stdb.moving_objects / team_proximity). |
| `labels.py` | Landmark risk set and hazard labels of the v2 dump (labels only; they may use the whole log). |
| `phases.py` | Phase boundaries from LogPhaseChange EVENTS (not from the 10 s grid) on the LogMatchStart clock W. |
| `replay.py` | Prefix-replay test of the v2 inputs (the key leak test; DB-gated). |
| `source.py` | Raw-telemetry source of ONE match on the LogMatchStart clock W (DB-free container + read-only fetch). |

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.
