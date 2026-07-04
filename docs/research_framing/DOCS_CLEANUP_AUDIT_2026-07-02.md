# Docs Cleanup Audit

Date: 2026-07-02

## Purpose

This cleanup reorganized `docs/` around the current project direction:

```text
Battle Royale survival prediction via mechanic-aligned terrain-affordance
features, spatial world modeling, and graph/hazard model design.
```

The cleanup intentionally avoided deleting tracked research material. Generated
presentation artifacts and legacy audit/status files were moved into archive
folders instead.

## New Active Structure

| Folder | Role |
| --- | --- |
| `docs/feature_engineering/` | feature definitions, feature lineage, extraction contracts |
| `docs/model_design/` | model architecture, graph design, input channels |
| `docs/spatial_world_modeling/` | Caldera/OpenUSD world reconstruction and rendering concepts |
| `docs/data_layer/` | DB/layer contracts and cohort/data availability |
| `docs/experiment_design/` | experiment matrices, results, ablations, runbooks |
| `docs/research_framing/` | contribution framing, claim boundaries, cleanup audits |
| `docs/paper_writing/` | manuscript-facing prose and figure narratives |
| `docs/handoffs/` | continuation notes |
| `docs/archive/` | old generated artifacts and legacy inventories |

## High-Confidence Moves Performed

### Feature Engineering

Moved feature design, ST feature contract, proxy, and feature inventory docs into
`docs/feature_engineering/`.

This includes the current terrain-affordance feature docs:

- `TERRAIN_EXPERIENCE_FEATURE_DESIGN_2026-07-02.md`
- `TERRAIN_EXPERIENCE_RESEARCH_LINEAGE_2026-07-02.md`

### Spatial World Modeling

Moved Caldera 3D visualization and ST-aware world concepts into
`docs/spatial_world_modeling/`.

### Model Design

Moved graph formalism, model specs, input-channel contracts, and dynamic graph
design docs into `docs/model_design/`.

### Data Layer

Moved Battle Royale ground/representation contracts, cohort definitions, and
data availability docs into `docs/data_layer/`.

### Experiment Design

Moved runbooks, result summaries, leakage audits, and ablation status docs into
`docs/experiment_design/`.

### Archive

Moved generated presentation outputs and old inventory/audit/status files into:

- `docs/archive/generated_presentation_2026-06/`
- `docs/archive/legacy_inventory_2026-06/`
- `docs/archive/legacy_status_reports_2026-06/`

## Not Deleted Yet

No tracked docs were deleted in this pass.

Deletion candidates for a later explicit cleanup:

| Path | Reason |
| --- | --- |
| `docs/.DS_Store` | local macOS metadata, not a research artifact |
| `docs/archive/generated_presentation_2026-06/` | generated deck/figure outputs; keep only if still needed for provenance |
| old archive status reports | can be deleted after confirming no paper/source references remain |

## Current Rule

New design documents should not be written directly into the root of `docs/`
unless they define repo-wide documentation policy.

Use:

- `feature_engineering` for feature definitions;
- `model_design` for model architecture;
- `spatial_world_modeling` for world/mesh/rendering;
- `data_layer` for schema and ETL contracts;
- `experiment_design` for run/evaluation design;
- `research_framing` for contribution and claim boundaries;
- `paper_writing` for manuscript prose;
- `handoffs` for continuation notes.

## Next Cleanup Step

Before deleting archived artifacts, run a reference check:

```text
rg "archive/generated_presentation_2026-06|presentation_figures|deliverables|graph_deck" .
```

Then decide whether to:

1. keep archive files for provenance;
2. move large generated artifacts to `reports/`;
3. delete obsolete generated artifacts.
