# Design Document Structure

Date: 2026-07-02

## Purpose

New design documents should be organized by what kind of design expression they
define. The root `docs/` directory currently contains legacy planning, reports,
handoffs, and design notes. We will not move all legacy documents at once.
Instead, new documents should use the folder taxonomy below, and older documents
can be migrated when they are actively touched.

Companion object-model documents:

- `DOC_OBJECT_MODEL_2026-07-02.md` defines the OOP-style documentation model.
- `DOC_OBJECT_MANIFEST_2026-07-02.md` lists the current Markdown objects.

## Folder Taxonomy

| Folder | Use for | Examples |
| --- | --- | --- |
| `docs/feature_engineering/` | feature taxonomy, feature math, feature extraction contracts, feature lineage | terrain affordance features, ST feature contracts |
| `docs/model_design/` | neural/tabular model architecture, graph design, heads, loss, ablations | RGCN threat edges, hazard heads |
| `docs/spatial_world_modeling/` | map/world reconstruction, mesh extraction, rendering, coordinate alignment | Caldera z-up world, OpenUSD profiles |
| `docs/data_layer/` | DB schema, ground/derived layer contracts, ETL ownership | `br_ground`, `br_repr`, feature set registry |
| `docs/experiment_design/` | experiment matrix, splits, evaluation, ablation plans | shared split, C-index/Brier/rank metrics |
| `docs/research_framing/` | contribution framing, capability boundaries, related-work positioning | terrain affordance boundary |
| `docs/paper_writing/` | manuscript sections, figure plan, submission-facing text | related work draft, method prose |
| `docs/handoffs/` | next-chat handoffs and continuation notes | thread handoff docs |

## Routing Rules

1. If a document defines variables, feature formulas, feature clocks, or feature
   extraction code contracts, put it in `docs/feature_engineering/`.
2. If a document defines model inputs, architecture, edge semantics, loss,
   training, or ablation variants, put it in `docs/model_design/`.
3. If a document defines map reconstruction, coordinate systems, mesh profiles,
   rendering, or ray-casting geometry, put it in `docs/spatial_world_modeling/`.
4. If a document defines table ownership, schema boundaries, ETL, or feature set
   registration, put it in `docs/data_layer/`.
5. If a document defines comparison matrices, evaluation metrics, statistical
   tests, or run plans, put it in `docs/experiment_design/`.
6. If a document defines what we can claim, what we cannot claim, or how the
   paper is positioned, put it in `docs/research_framing/`.
7. If a document is prose intended for the manuscript, put it in
   `docs/paper_writing/`.
8. If a document exists only to resume work in a future chat, put it in
   `docs/handoffs/`.

## Migration Policy

Do not mass-move the legacy `docs/` directory without a dedicated cleanup pass.
The current root contains many historical references. Moving all of them at once
would create avoidable broken links and review noise.

When a legacy document is actively edited:

1. Move it into the correct folder.
2. Update references in documents touched by the same change.
3. Leave a short note in the commit or final summary describing the migration.

## Current New Documents

| Document | Folder | Reason |
| --- | --- | --- |
| `docs/feature_engineering/TERRAIN_EXPERIENCE_FEATURE_DESIGN_2026-07-02.md` | feature engineering | feature taxonomy and extraction contract |
| `docs/feature_engineering/TERRAIN_EXPERIENCE_RESEARCH_LINEAGE_2026-07-02.md` | feature engineering | lineage for terrain-affordance features |
| `docs/research_framing/TERRAIN_AFFORDANCE_CAPABILITY_BOUNDARY_2026-07-02.md` | research framing | possible / not possible / claim boundary |
| `docs/DOC_OBJECT_MODEL_2026-07-02.md` | root policy | OOP-style documentation object model |
| `docs/DOC_OBJECT_MANIFEST_2026-07-02.md` | root registry | current Markdown object registry |

## Naming Convention

Use:

```text
TOPIC_PURPOSE_YYYY-MM-DD.md
```

Examples:

```text
TERRAIN_AFFORDANCE_FEATURE_DESIGN_2026-07-02.md
THREAT_EDGE_MODEL_DESIGN_2026-07-02.md
CALDERA_ZUP_WORLD_MODELING_2026-07-02.md
```

For handoffs, keep the existing pattern:

```text
NEXT_CHAT_HANDOFF_YYYY-MM-DD_TOPIC.md
```
