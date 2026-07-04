# Battle Royale Data Layer Contract

이 문서는 현재 PUBG DB와 Activision Caldera 데이터를 함께 다루기 위한
비파괴 데이터 계층 계약이다.

핵심 원칙:

```text
원본/ground는 보존하고, 연구 표현은 언제든 재생성한다.
```

식물 비유로 말하면, `pubg.*`와 Caldera 원본 파일은 토양이다. 모델 학습에
맞게 평탄화한 cohort, tensor, feature cache는 그 토양에서 자라는 식물이며
필요하면 잘라내고 다시 키울 수 있어야 한다.

## Layer Map

| Layer | Local schema/path | Role | Deletion policy |
| --- | --- | --- | --- |
| L0 raw archive | `data/br_l0/*`, `br_raw.artifacts` | byte/file provenance, URL, size, checksum, git ref | raw files are protected; DB rows are registry |
| L1 ground | `pubg.*`, `br_ground.*` | source-faithful normalized facts and canonical views | protected |
| L2 representation | `stdb.*`, `br_repr.*` | research features, windows, tensor manifests | rebuildable |
| L3 experiments | `model_stgcn/experiment_datasets/*`, `br_exp.*` | cohorts, splits, domain membership | rebuildable if manifest exists |

Current PUBG state:

- `pubg.*` is **not** byte-identical official JSON telemetry.
- It is the current **official telemetry-derived normalized L1**.
- It must be preserved as the local ground/soil layer.
- `stdb.*` is derived and can be rebuilt from `pubg.*` plus ETL code, although
  some objects are expensive.

Caldera state:

- Caldera should not be forced into `pubg.*` or current `stdb.team_phase_*`.
- It is a separate source registered as `activision_caldera`.
- Its raw OpenUSD files stay outside PostgreSQL under `data/br_l0/caldera`.
- The DB stores manifests and thin extracted canonical samples only.

## Cross-Title Interpretation

PUBG and Caldera do not have the same observability.

| Capability | PUBG current DB | Caldera public dataset |
| --- | --- | --- |
| Match metadata | yes | partial/implicit |
| Player/team trajectory | yes | player breadcrumbs, sampled |
| Team identity | yes | not verified publicly |
| Safe-zone phases | yes | not verified publicly |
| Combat/damage/kill | yes | not exposed in README/player layers |
| Loot/resource events | yes | not exposed in README/player layers |
| Map geometry | partial/external assets | yes, OpenUSD geometry |
| Endpoint / terminal proxy | yes via outcome | anonymous endpoint point cloud and breadcrumb last observed position |
| Final rank / survived label | yes | no in current ingest |

Therefore Caldera is best used first as:

```text
low-observability spatial trajectory / terrain-affordance generalization data
```

It can support claims such as:

- Does a BR survival representation retain useful spatial/trajectory structure
  outside PUBG?
- Can trajectory, endpoint, proximity, and habitat geometry be represented by
  the same primitive vocabulary?
- Which DSACG channels remain valid when combat, loot, and zone channels are
  masked out?

It should not support claims such as:

- Full cross-game validation of PUBG combat/resource/zone mechanisms.
- Direct comparison of PUBG team phase hazard prediction with Caldera unless
  equivalent labels and observation windows are constructed.
- Caldera final-rank or survived-label prediction unless official or
  source-valid outcome labels are added.

## Database Additive Contract

The additive schema is:

```text
stdb/schema_br_layers.sql
```

It creates:

- `br_raw.sources`
- `br_raw.artifacts`
- `br_raw.l1_table_registry`
- `br_ground.matches`
- `br_ground.agents`
- `br_ground.trajectory_samples`
- `br_ground.zone_states`
- `br_ground.events`
- `br_ground.outcomes`
- `br_ground.environment_prims`
- `br_repr.feature_sets`
- `br_repr.observation_windows`
- `br_repr.tensor_artifacts`
- `br_repr.environment_catalog_runs`
- `br_exp.cohorts`
- `br_exp.cohort_members`

It also creates:

```text
br_ground.v_pubg_matches_from_existing_l1
```

This view reads `pubg.matches` directly and exposes current PUBG L1 as BR
ground without copying rows.

## Disk Policy

The current machine has limited free space, so use this order:

1. Manifest only: GitHub API metadata, no large blobs.
2. Minimal Caldera player layers only:
   - `README.md`
   - `LICENSE.md`
   - `caldera.py`
   - `caldera.usda`
   - `layers/breadcrumbs.usd`
   - `layers/endpoints.usd`
   - `layers/cameras.usd`
3. Extract thin L1/L2 tables or files.
4. Sparse checkout of `map_source` only.
5. Geometry catalog into PostgreSQL.
6. Only then consider full geometry assets.

Never load the full Caldera geometry into PostgreSQL as raw blobs.

## Commands

Fetch a small GitHub-size manifest:

```bash
python3 scripts/caldera_dataset_manager.py manifest
```

Dry-run the minimal player-layer download:

```bash
python3 scripts/caldera_dataset_manager.py fetch-minimal-players --dry-run
```

Download the minimal player layer:

```bash
python3 scripts/caldera_dataset_manager.py fetch-minimal-players --yes
```

Apply the additive schema and register Caldera manifests:

```bash
PG_PASSWORD=... python3 scripts/caldera_dataset_manager.py register-db --apply-schema
```

Install optional OpenUSD tooling in the local project environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-caldera.txt
```

Ingest Caldera player endpoints and breadcrumbs into `br_ground`/`br_repr`:

```bash
PG_PASSWORD=... .venv/bin/python scripts/caldera_ingest_players.py --apply-schema
```

Smoke-test before full ingest:

```bash
PG_PASSWORD=... .venv/bin/python scripts/caldera_ingest_players.py \
  --apply-schema --endpoint-limit 1000 --match-limit 2
```

Optional full or sparse checkout, only with enough disk headroom:

```bash
python3 scripts/caldera_dataset_manager.py fetch --profile metadata --yes
python3 scripts/caldera_dataset_manager.py fetch --profile source --yes --reuse
python3 scripts/caldera_dataset_manager.py fetch --profile full --min-free-gib 12 --yes
```

Catalog Caldera map geometry metadata without loading full payloads:

```bash
PG_PASSWORD=... .venv/bin/python scripts/caldera_ingest_geometry_catalog.py \
  --apply-schema --max-depth 5
```

Dry-run a small catalog sample:

```bash
.venv/bin/python scripts/caldera_ingest_geometry_catalog.py \
  --dry-run --max-depth 5 --limit 2000
```

## Research Handling

For paper-level generalization, use this language:

```text
We preserve PUBG as a high-observability normalized telemetry ground layer and
introduce Caldera as a low-observability cross-title spatial dataset.  The
framework is evaluated under observation masks rather than by pretending every
title exposes identical telemetry.
```

This keeps the claim honest:

- PUBG validates the rich DSACG instantiation.
- Caldera tests whether a subset of the same BR primitive vocabulary survives
  under another title and another data-release style.

## Current Caldera Ingest Status

As of 2026-07-01, the minimal Caldera player layer has been ingested and the
`source` sparse checkout has been added under `data/br_l0/caldera/repo`.

| Object | Count |
| --- | ---: |
| `br_ground.endpoint_samples` | 1,000,000 |
| `br_ground.matches` for Caldera | 36 |
| `br_ground.agents` for Caldera | 4,187 |
| `br_ground.trajectory_samples` for Caldera | 598,175 |
| `br_ground.outcomes` for Caldera | 4,187 |
| `br_repr.trajectory_summaries` for Caldera | 4,187 |

Outcome / phase audit on 2026-07-02:

| Object | Count |
| --- | ---: |
| non-null Caldera `br_ground.outcomes.final_rank` | 0 |
| non-null Caldera `br_ground.outcomes.survived` | 0 |
| Caldera `br_ground.zone_states` | 0 |
| Caldera `alive_state = in_play` trajectory samples | 594,025 |
| Caldera `alive_state = pre_spawn` trajectory samples | 4,150 |

The largest added DB tables are:

| Table | Size |
| --- | ---: |
| `br_ground.endpoint_samples` | 382 MB |
| `br_ground.trajectory_samples` | 352 MB |

Interpretation:

- `endpoint_samples` is an anonymous endpoint point cloud. It does not expose
  match/player identity.
- `trajectory_samples` is the breadcrumb layer with `caldera:match_X` and
  `caldera:match_X:player_Y` ids.
- Coordinates are converted to meters using `metersPerUnit = 0.0254`.
- Breadcrumb time is converted with Activision's sample-code rule:
  `game_seconds = 60 * timecode / fps`.
- `life_norm < 0` means pre-spawn and `0..1` means in-play in the ingested
  breadcrumb layer.  The current ingest does not expose official death,
  final-rank, survived, or safe-zone phase labels.
- Large in-play gaps indicate multi-life or redeploy-like discontinuities:
  2,460 agents have `dt > 30s` gaps, and 1,272 gaps combine `dt > 30s` with
  XY displacement above 500m.  Derived trajectory features must segment or
  suppress motion across these gaps.

The source checkout is deliberately external to PostgreSQL:

| Local object | Size |
| --- | ---: |
| `data/br_l0/caldera/repo` source sparse checkout | 943 MB |
| `data/br_l0/caldera` total | 1.0 GB |

Geometry catalog policy:

- `br_ground.environment_prims` stores OpenUSD prim metadata, not mesh vertices.
- `br_repr.environment_catalog_runs` records catalog runs and type counts.
- The catalog is the spatial DB substrate for terrain/building/topology features.
- Full `assets` ingestion should wait until a concrete feature needs it.

Current geometry catalog:

| Object | Count/Size |
| --- | ---: |
| `br_ground.environment_prims` for `caldera_map_source_v1` | 2,715 rows |
| prims with centroid/bbox metadata | 2,714 rows |
| prims with authored references | 89 rows |
| unique referenced assets in latest run | 61 |
| `br_ground.environment_prims` table size | 3.2 MB |
| `br_repr.environment_catalog_runs` table size | 48 kB |

Latest catalog run:

```text
caldera_map_source_v1_none_20260701T031852Z
```

Top prim types in this first depth-5 catalog:

| Prim type | Count |
| --- | ---: |
| `Cube` | 2,519 |
| `Xform` | 139 |
| `Mesh` | 57 |
