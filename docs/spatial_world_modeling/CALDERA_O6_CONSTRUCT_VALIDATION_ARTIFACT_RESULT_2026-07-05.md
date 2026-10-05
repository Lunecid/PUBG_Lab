# Caldera O6 Construct-Validation Artifact Result

Date: 2026-07-05
Class: `WorldModel`
Lifecycle: `active`

## Purpose

This document records the completed Caldera / Warzone artifact package for
`RQ3`: what PUBG cannot cleanly express about geometry, topology, terrain, and
spatial affordance, and how a geometry-rich game world can represent those
constructs.

This is construct-validation evidence, not supervised prediction evidence.

## Source Scope

The package uses existing local Caldera outputs:

| Input | Role |
| --- | --- |
| `reports/caldera_3d/single_agent/match_31_player_25_semantic_triangles.pkl` | OpenUSD-derived semantic triangle cache |
| `reports/caldera_3d/single_agent/match_31_player_25_topdown_snapshot.png` | selected trajectory over the semantic spatial layer |
| `reports/caldera_3d/single_agent/match_31_player_25_experience_snapshot.png` | local terrain-experience render |
| `reports/caldera_3d/single_agent/world_build/match_31_player_25_terrain_experience_features.csv` | sample-level O6 construct features |
| `reports/caldera_3d/single_agent/world_build/match_31_player_25_terrain_experience_features.summary.json` | feature and guard summary |

No model retraining, Caldera survival fitting, or full-world reconstruction was
run for this package.

## Generated Artifacts

| Artifact | Path |
| --- | --- |
| composite figure | `reports/caldera_3d/construct_validation/caldera_o6_construct_validation_composite.png` |
| descriptive table CSV | `reports/caldera_3d/construct_validation/caldera_o6_construct_validation_descriptive_table.csv` |
| descriptive table Markdown | `reports/caldera_3d/construct_validation/caldera_o6_construct_validation_descriptive_table.md` |
| PUBG-Warzone observability bridge | `reports/caldera_3d/construct_validation/caldera_o6_pubg_warzone_observability_bridge.md` |
| PUBG gap situation cards | `reports/caldera_3d/construct_validation/caldera_o6_pubg_gap_situation_cards.md` |
| observability bridge board | `reports/caldera_3d/construct_validation/caldera_o6_pubg_warzone_observability_bridge_board.png` |
| manifest | `reports/caldera_3d/construct_validation/caldera_o6_construct_validation_manifest.json` |
| reproducible assembler | `scripts/assemble_caldera_o6_construct_validation_artifacts.py` |

The composite figure has four panels:

| Panel | Content | Claim role |
| --- | --- | --- |
| A | semantic Caldera basemap | geometry-rich spatial substrate from OpenUSD mesh categories |
| B | selected player trajectory overlay | trajectory can be projected into the reconstructed terrain/building/route layer |
| C | local terrain-experience zoom | agent path can be described with altitude, terrain, semantic context, and proximity diagnostics |
| D | feature callout | O6 construct features and life-segmentation guard are explicit |

## Descriptive Feature Result

The descriptive feature table reports one guarded trajectory segment:

| Quantity | Value |
| --- | ---: |
| feature rows | 120 |
| time window | `1155.360-1629.456s` |
| life segments | 1 |
| motion gap rows | 0 |
| redeploy-gap candidate rows | 0 |

Life segmentation uses the documented guard:

```text
new segment if dt > 30s and XY displacement > 500m
```

Selected construct features:

| Feature family | Descriptive result |
| --- | --- |
| vertical consistency | `height_above_terrain_m` median `0.18m`, p75 `0.33m`, max `4.13m` |
| local terrain shape | `local_relief_r50_m` median `0.78m`, p75 `7.87m`, max `37.85m` |
| built environment | `building_dist_m` median `0.91m`; `cover_building_r25=85.00%` |
| route context | `road_or_runway_dist_m` median `50.13m`; `on_corridor_proxy=3.33%` |
| tactical state | `compound_movement=102`, `rough_mobility=18` |

These are descriptive construct measurements. They are not label-performance
metrics.

## PUBG-Warzone Bridge

The important RQ3 artifact is not the image alone. The bridge artifact maps:

```text
PUBG official telemetry gap
-> Warzone / Caldera geometry representation
-> current descriptive evidence
-> claim status
```

The canonical bridge document is:

```text
docs/spatial_world_modeling/PUBG_WARZONE_OBSERVABILITY_GAP_BRIDGE_2026-07-05.md
```

The generated bridge outputs are:

```text
reports/caldera_3d/construct_validation/caldera_o6_pubg_warzone_observability_bridge.md
reports/caldera_3d/construct_validation/caldera_o6_pubg_gap_situation_cards.md
reports/caldera_3d/construct_validation/caldera_o6_pubg_warzone_observability_bridge_board.png
```

Covered now:

| Gap family | Current status |
| --- | --- |
| terrain surface and vertical context | supported descriptive construct |
| rough terrain mobility | supported descriptive construct |
| compound and building affordance | supported descriptive construct |
| route corridor and road/runway context | supported descriptive construct |
| local tactical-state labeling | supported descriptive construct, not outcome supervision |
| visibility / occlusion / LOS | pending extension, not claimed |

## Supported Claim

Safe manuscript wording:

```text
Warzone / Caldera shows how geometry-rich data can represent terrain, route,
building, and local tactical-state constructs that official PUBG telemetry
cannot directly observe.
```

Also safe:

```text
This complements PUBG: PUBG is label-rich and supports supervised survival /
rank prediction, while Caldera is geometry-rich and supports construct
validation for spatial affordance measurements that PUBG telemetry cannot
directly observe.
```

## Claim Boundary

Do not claim:

```text
Caldera validates the supervised PUBG survival model.
```

Do not claim:

```text
Caldera supports supervised phase survival or final-rank prediction under the
current public labels.
```

Do not claim:

```text
The rendered map is a full game-engine reconstruction.
```

Pairwise LOS remains a useful future extension, but it is not part of the
current completed artifact package.

## Reproduction

Run:

```bash
python3 scripts/assemble_caldera_o6_construct_validation_artifacts.py
```

Verification performed:

```bash
python3 -m py_compile scripts/assemble_caldera_o6_construct_validation_artifacts.py
python3 scripts/assemble_caldera_o6_construct_validation_artifacts.py
```
