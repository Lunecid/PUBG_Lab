# Battle Royale Feature Availability Audit

Date: 2026-07-03
Class: `ExperimentContract`
Lifecycle: `active`

## Purpose

This audit checks whether the locked Battle Royale ontology and OM0-OM6 ladder
can be operationalized on the required PUBG cohorts without leakage.

It answers the first execution question after the problem definition:

```text
Can the ontology-derived feature blocks actually be computed for each required
map / perspective / team-size cohort before any model training starts?
```

## Short Answer

The required PUBG cohorts are feature-ready at the database/view level.

All 13 required cohorts have:

- graph-eligible split manifests;
- rows in `stdb.v_team_phase_feature_v2`;
- target-defined next-phase hazard labels;
- O7 network rows in `stdb.team_phase_network_features`;
- O2 x O6 habitat rows in `stdb.team_phase_habitat_features`;
- phase edge rows in `stdb.team_phase_edge_cache`.

Clean OM0-OM6 code presets have now been added to `model_stgcn/v3/spec.py`,
connected to the `om_ladder` experiment group in
`model_stgcn/v3/experiment_matrix.py`, and cohort-filtered through
`model_stgcn/v3/cohorts.py`.

The execution smoke gate has now passed on three representative named cohorts:
TPP squad, TPP duo, and FPP squad.

## Audit Inputs

| Source | Role |
| --- | --- |
| `docs/feature_engineering/PUBG_FEATURE_OPERATIONAL_MANIFEST_2026-07-02.md` | conceptual feature contract |
| `docs/experiment_design/BATTLE_ROYALE_REQUIRED_COHORT_SET_2026-07-03.md` | required map / perspective / team-size cohorts |
| `model_stgcn/v3/spec.py` | executable feature preset and forbidden input contract |
| `model_stgcn/v3/experiment_matrix.py` | current runnable experiment groups |
| `model_stgcn/feature_manifests/pubg_representation_v2.json` | JSON feature-family manifest |
| `docs/feature_engineering/GENRE_PRIMITIVE_FEATURE_MANIFEST.csv` | primitive feature contract |
| `stdb.v_team_phase_feature_v2` | main phase feature view |
| `stdb.team_phase_network_features` | O7 node network-role features |
| `stdb.team_phase_habitat_features` | O2 x O6 habitat-pressure features |
| `stdb.team_phase_edge_cache` | phase graph relation edges |

## Validation Commands Run

```text
python3 -m pytest tests/test_feature_contract_docs.py -q
python3 -m pytest tests/test_v3_cohorts.py tests/test_v3_factorized.py tests/test_feature_contract_docs.py -q
python3 tools/validate_genre_feature_manifest.py
PG_PASSWORD=... python3 tools/check_feature_manifest.py --pretty
```

Results:

| Check | Result |
| --- | --- |
| operational manifest forbidden list vs `model_stgcn/v3/spec.py` | passed |
| genre primitive manifest validation | passed after path and label sync fixes |
| JSON feature manifest vs `stdb.v_team_phase_feature_v2` | passed for implemented families; no forbidden hits |
| clean OM preset unit tests | passed |
| named cohort adapter unit tests | passed |
| one-match Erangel TPP squad preset load smoke | passed |
| Erangel TPP squad `om_ladder` run-matrix smoke | passed with 30-match limit |
| Erangel TPP duo `om_ladder` run-matrix smoke | passed with 30-match limit |
| Taego FPP squad `om_ladder` run-matrix smoke | passed with 30-match limit |

Fixes made during audit:

- added `eliminated_next_phase` to
  `model_stgcn/feature_manifests/pubg_representation_v2.json`;
- added `eliminated_next_phase` to
  `docs/feature_engineering/GENRE_PRIMITIVE_FEATURE_MANIFEST.csv`;
- updated `tools/validate_genre_feature_manifest.py` default manifest path to
  `docs/feature_engineering/GENRE_PRIMITIVE_FEATURE_MANIFEST.csv`.

## Required Cohort Coverage

Local split summary:

| Cohort | Feature eligible | Graph eligible | Raw fetched | Excluded |
| --- | ---: | ---: | ---: | ---: |
| Erangel TPP squad | 6,205 | 6,204 | 6,206 | 2 |
| Taego TPP squad | 4,160 | 4,159 | 4,160 | 1 |
| Miramar / Desert TPP squad | 1,545 | 1,543 | 1,545 | 2 |
| Rondo TPP squad | 1,041 | 1,039 | 1,041 | 2 |
| Sanhok TPP squad | 940 | 940 | 940 | 0 |
| Vikendi TPP squad | 498 | 498 | 498 | 0 |
| Paramo TPP squad | 333 | 333 | 333 | 0 |
| Karakin TPP squad | 333 | 333 | 333 | 0 |
| Erangel TPP solo | 800 | 800 | 801 | 1 |
| Erangel TPP duo | 656 | 656 | 656 | 0 |
| Taego TPP duo | 573 | 573 | 574 | 1 |
| Taego FPP squad | 435 | 435 | 435 | 0 |
| Erangel FPP squad | 430 | 430 | 430 | 0 |

Direct DB view check:

| Cohort | Matches in v2 | Feature rows | Target-defined rows | Hazard positive rows | Network rows | Habitat rows | Edge rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Erangel TPP squad | 6,205 | 1,098,292 | 315,108 | 100,056 | 1,098,292 | 1,098,292 | 8,661,628 |
| Taego TPP squad | 4,160 | 812,648 | 213,401 | 62,094 | 812,648 | 812,648 | 5,593,952 |
| Miramar / Desert TPP squad | 1,545 | 270,969 | 87,810 | 25,070 | 270,969 | 270,969 | 2,111,316 |
| Rondo TPP squad | 1,041 | 170,167 | 51,984 | 14,947 | 170,167 | 170,167 | 1,624,056 |
| Sanhok TPP squad | 940 | 190,831 | 46,793 | 15,128 | 190,831 | 190,831 | 1,273,982 |
| Vikendi TPP squad | 498 | 96,072 | 28,237 | 8,556 | 96,072 | 96,072 | 837,083 |
| Paramo TPP squad | 333 | 39,720 | 12,243 | 4,359 | 39,720 | 39,720 | 237,754 |
| Karakin TPP squad | 333 | 36,749 | 9,672 | 3,306 | 36,749 | 36,749 | 185,219 |
| Erangel TPP solo | 800 | 584,223 | 117,031 | 42,523 | 584,223 | 584,223 | 11,859,127 |
| Erangel TPP duo | 656 | 245,295 | 59,480 | 19,239 | 245,295 | 245,295 | 2,769,751 |
| Taego TPP duo | 573 | 214,290 | 48,036 | 15,683 | 214,290 | 214,290 | 2,241,156 |
| Taego FPP squad | 435 | 86,110 | 22,702 | 7,556 | 86,110 | 86,110 | 771,590 |
| Erangel FPP squad | 430 | 78,217 | 20,974 | 6,577 | 78,217 | 78,217 | 579,815 |

Implementation note:

```text
`stdb.v_team_phase_feature_v2.camera_view_type` is currently NULL.
Perspective is encoded in `game_mode` values such as `squad`, `duo`, `solo`,
`squad-fpp`, and `duo-fpp`.
```

Therefore cohort filtering code translates the registry perspective into the
current view's `game_mode` representation before training or evaluation.

## View-Level Family Availability

`tools/check_feature_manifest.py` reports `stdb.v_team_phase_feature_v2` has
187 columns.

| Manifest family | Ontology meaning | Wanted | Present | Missing | Forbidden hits | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `f0_official_raw_phase` | official raw phase baseline | 85 | 78 | 7 | 0 | partial |
| `f1_prior_paper_raw_inputs` | prior raw ingredients | 12 | 12 | 0 | 0 | ready |
| `f2_prior_paper_score_summary` | train-split-fitted scores | 5 | 1 | 4 | 0 | missing as physical columns |
| `f4_current_conventional` | current conventional feature set | 57 | 57 | 0 | 0 | ready |
| `f5_stdb_spatial` | movement / exposure / spatial proxies | 18 | 18 | 0 | 0 | ready |
| `f6_network_phase_role_tier1` | O7 network role | 43 | 43 | 0 | 0 | ready |
| `f7_habitat_pressure_tier1` | O2 x O6 habitat pressure | 32 | 32 | 0 | 0 | ready |
| `f8_temporal_patch_proxy` | temporal drift controls | 5 | 0 | 5 | 0 | not implemented |

Missing `f0_official_raw_phase` columns:

```text
assists_count
suicide_count
landing_t
landing_x
landing_y
landing_distance
landing_team_spread_m
```

These do not block OM0-OM6 because `f4_current_conventional`,
`f5_stdb_spatial`, `f6_network_phase_role_tier1`, and
`f7_habitat_pressure_tier1` are available.

## Current V3 Preset Surface

Current executable presets in `model_stgcn/v3/spec.py`:

| Preset | Current role | Total | Global | Node |
| --- | --- | ---: | ---: | ---: |
| `core` | current conventional O0/O1/O2/O3/O4/O5 mixed baseline | 48 | 12 | 36 |
| `stdb` | movement/spatial/exposure proxy block | 19 | 2 | 17 |
| `network` | O7 node network-role block | 45 | 2 | 43 |
| `habitat` | O2 x O6 habitat-pressure block | 34 | 15 | 19 |
| `core_stdb` | core + movement/spatial/exposure | 65 | 12 | 53 |
| `core_network` | core + O7 node network-role | 91 | 12 | 79 |
| `core_habitat` | core + O2 x O6 habitat | 80 | 25 | 55 |
| `base66` | legacy 66 phase features | 66 | 12 | 54 |
| `full` | all current phase features | 141 | 25 | 116 |

The legacy/broad presets above remain useful for reproducibility and smoke
tests, but they should not be used as the final OM0-OM6 paper ladder because
`core` mixes several ontology families.

Clean OM presets added after this audit:

| Preset | Role | Total | Global | Node |
| --- | --- | ---: | ---: | ---: |
| `om0_context` | O0 minimal phase context plus team-size anchor | 3 | 2 | 1 |
| `om1_survival_capital` | OM0 + O1 roster, health, and team cohesion | 12 | 2 | 10 |
| `om2_movement` | OM1 + O3 movement and rotation response | 19 | 2 | 17 |
| `om3_pressure_habitat` | OM2 + O2 contraction and O2 x O6 habitat pressure | 66 | 25 | 41 |
| `om4_threat_exposure` | OM3 + O4 combat and threat exposure | 79 | 25 | 54 |
| `om5_relation_structure` | OM4 + O7 precomputed network-role summaries | 122 | 25 | 97 |
| `om6_spatial_proxy` | OM5 + remaining PUBG spatial-exposure proxies | 125 | 25 | 100 |

The `om_ladder` experiment group runs these presets with `graph_encoder=none`
and `edge_mode=none`.  This keeps the first OM ladder as a feature-family
ablation.  Graph message passing and edge controls remain a separate O7
relation-structure test.

## OM0-OM6 Readiness

| OM block | Conceptual owner | Feature availability | Current executable surface | Audit status |
| --- | --- | --- | --- | --- |
| OM0 | conventional baseline / minimal context | phase context and current conventional columns exist | `om0_context` | ready |
| OM1 | O1 survival capital | alive roster, health, spread exist | `om1_survival_capital` | ready |
| OM2 | O3 movement / rotation | path length, displacement, speed, sinuosity, rotation timing, safe alignment exist | `om2_movement` | ready |
| OM3 | O2 contraction / habitat pressure | zone, alive population, habitat pressure exist | `om3_pressure_habitat` | ready |
| OM4 | O4 interaction / threat exposure | damage, groggy, kills, distinct opponents, recent combat exposure exist | `om4_threat_exposure` | ready |
| OM5 | O7 relation / interaction structure | network-role rows and edge cache exist for all required cohorts | `om5_relation_structure`; graph-edge tests remain separate | ready for node-summary ladder |
| OM6 | O6 spatial / terrain proxy | PUBG spatial-exposure proxies exist; true terrain/LOS/buildings do not | `om6_spatial_proxy`; Caldera construct artifacts for richer geometry | ready as PUBG proxy |

Interpretation:

```text
The data are ready.
The clean OM0-OM6 feature presets are implemented.
Named cohort filtering is implemented, so each map / perspective / team-size
cohort can be selected exactly.
```

## Leakage Status

Current guards:

- `model_stgcn/v3/spec.py` rejects exact forbidden input features;
- `tests/test_feature_contract_docs.py` confirms the operational manifest and
  V3 spec forbidden list match;
- `tools/check_feature_manifest.py` found no forbidden hits inside declared
  feature families;
- `GENRE_PRIMITIVE_FEATURE_MANIFEST.csv` validates with zero warnings and zero
  errors.

Forbidden input list is synchronized at 16 fields:

```text
target_defined
hazard_next
survives_next_phase
eliminated_next_phase
final_rank
final_rank_norm
final_top3
final_top5
final_top10
final_winner
last_phase_alive
won
rank
victim_rank
match_end_rank
match_end_result
```

Residual leakage risks to handle before full runs:

- `core` includes O2/O3/O4/O5 together, so OM interpretation can be muddied;
- `alive_players_end` is allowed only at completed phase cuts and must remain
  tied to the current phase-cut target definition;
- network summaries require N/edge-density controls before strong O7 mechanism
  claims;
- `camera_view_type` is NULL in `v_team_phase_feature_v2`, so cohort filtering
  must not rely on that column.

## Remaining Gate Before OM Full Runs

### I1. Add Clean OM Presets

Status: complete.

Implemented presets:

| Target preset | Contents |
| --- | --- |
| `om0_context` | O0 context only or minimal conventional baseline |
| `om1_survival_capital` | OM0 + O1 roster/health/spread/capacity |
| `om2_movement` | OM1 + O3 movement/rotation |
| `om3_pressure_habitat` | OM2 + O2 contraction and O2 x O6 habitat |
| `om4_threat_exposure` | OM3 + O4 combat/threat exposure |
| `om5_relation_structure` | OM4 + O7 network/edge structure |
| `om6_spatial_proxy` | OM5 + O6 spatial-tactical proxies that are not duplicated in OM3 |

### I2. Add Cohort Filter Adapter

Status: complete.

The current view represents perspective through `game_mode`, not
`camera_view_type`.  Required mapping:

| User-facing cohort value | Current view filter |
| --- | --- |
| TPP squad | `game_mode = 'squad'` |
| TPP duo | `game_mode = 'duo'` |
| TPP solo | `game_mode = 'solo'` |
| FPP squad | `game_mode = 'squad-fpp'` |

Implemented surface:

```text
python3 -m model_stgcn.v3.run_matrix \
  --groups om_ladder \
  --cohort "Erangel TPP squad" \
  --out-dir model_stgcn/checkpoints/om_ladder_erangel_tpp_squad
```

The named cohort is resolved to the current DB filters by
`model_stgcn/v3/cohorts.py`.

### I3. Keep Current Broad Presets For Smoke Only

Use existing presets to smoke-test data loading:

| Smoke purpose | Existing preset |
| --- | --- |
| baseline loader | `core` |
| STDB block loader | `core_stdb` |
| O7 node block loader | `core_network` |
| habitat block loader | `core_habitat` |
| full view stress test | `full` |

Do not use the current broad presets as the final OM0-OM6 ablation table
without documenting their mixed-family contents.

## Audit Conclusion

The feature availability gate is passed at the data level, the clean preset
gate is passed at the code level, and the named-cohort filter adapter is in
place.

Passed smoke commands:

```text
PG_PASSWORD=... python3 -m model_stgcn.v3.run_matrix \
  --groups om_ladder \
  --cohort "Erangel TPP squad" \
  --limit 30 \
  --epochs 1 \
  --batch 2 \
  --workers 0 \
  --device cpu \
  --out-dir model_stgcn/checkpoints/om_ladder_erangel_tpp_squad_smoke30 \
  --force \
  --force-eval
```

```text
PG_PASSWORD=... python3 -m model_stgcn.v3.run_matrix \
  --groups om_ladder \
  --cohort "Erangel TPP duo" \
  --limit 30 \
  --epochs 1 \
  --batch 2 \
  --workers 0 \
  --device cpu \
  --out-dir model_stgcn/checkpoints/om_ladder_erangel_tpp_duo_smoke30 \
  --force \
  --force-eval
```

```text
PG_PASSWORD=... python3 -m model_stgcn.v3.run_matrix \
  --groups om_ladder \
  --cohort "Taego FPP squad" \
  --limit 30 \
  --epochs 1 \
  --batch 2 \
  --workers 0 \
  --device cpu \
  --out-dir model_stgcn/checkpoints/om_ladder_taego_fpp_squad_smoke30 \
  --force \
  --force-eval
```

Output:

```text
model_stgcn/checkpoints/om_ladder_erangel_tpp_squad_smoke30/matrix_metrics.csv
model_stgcn/checkpoints/om_ladder_erangel_tpp_squad_smoke30/matrix_summary.json
model_stgcn/checkpoints/om_ladder_erangel_tpp_duo_smoke30/matrix_metrics.csv
model_stgcn/checkpoints/om_ladder_erangel_tpp_duo_smoke30/matrix_summary.json
model_stgcn/checkpoints/om_ladder_taego_fpp_squad_smoke30/matrix_metrics.csv
model_stgcn/checkpoints/om_ladder_taego_fpp_squad_smoke30/matrix_summary.json
```

Allowed next step:

```text
Expand to required-cohort OM0-OM6 runs, starting with the required main cohorts
and preserving separate map / perspective / team-size training.
```

Not allowed yet:

```text
Treat the broad `full` preset as the headline result before the clean OM ladder
has been run and interpreted per cohort.
```

The broad `full` preset remains a later upper-bound or stress-test condition.
It is not the first headline run, because it mixes ontology families and would
obscure the OM0-OM6 interpretation.
