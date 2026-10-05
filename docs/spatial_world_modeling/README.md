# Spatial World Modeling Docs

Package class: `WorldModel`

## Responsibility

Own map reconstruction, coordinate systems, OpenUSD/mesh extraction, z-up
alignment, ray casting, rendering, and world-model profiles.

## Public Interface

| Object | Role |
| --- | --- |
| `WARZONE_CALDERA_RQ3_3D_RENDER_EVIDENCE_PACKAGE_2026-07-06.md` | supplementary 3D render evidence package, figure selection, captions, and claim boundary |
| `CALDERA_O6_CONSTRUCT_VALIDATION_ARTIFACT_SCOPE_2026-07-05.md` | bounded Caldera O6 construct-validation artifact scope |
| `CALDERA_O6_CONSTRUCT_VALIDATION_ARTIFACT_RESULT_2026-07-05.md` | completed Caldera O6 construct-validation artifact package and claim boundary |
| `PUBG_WARZONE_OBSERVABILITY_GAP_BRIDGE_2026-07-05.md` | supplementary bridge from PUBG official telemetry gaps to Warzone / Caldera geometry representations |
| `CALDERA_ST_3D_DESIGN_CONCEPTS_2026-07-01.md` | Caldera ST-aware 3D concepts |
| `CALDERA_3D_VISUALIZATION_PLAN.md` | 3D visualization and reconstruction plan |

## Does Not Own

- terrain feature formulas after extraction: use `../feature_engineering/`;
- model use of features: use `../model_design/`;
- paper contribution claims: use `../research_framing/`.

See `../DESIGN_DOC_STRUCTURE_2026-07-02.md` and
`../DOC_OBJECT_MODEL_2026-07-02.md` for routing rules.
