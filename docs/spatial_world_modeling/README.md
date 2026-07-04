# Spatial World Modeling Docs

Package class: `WorldModel`

## Responsibility

Own map reconstruction, coordinate systems, OpenUSD/mesh extraction, z-up
alignment, ray casting, rendering, and world-model profiles.

## Public Interface

| Object | Role |
| --- | --- |
| `CALDERA_O6_CONSTRUCT_VALIDATION_ARTIFACT_SCOPE_2026-07-05.md` | bounded Caldera O6 construct-validation artifact scope |
| `CALDERA_ST_3D_DESIGN_CONCEPTS_2026-07-01.md` | Caldera ST-aware 3D concepts |
| `CALDERA_3D_VISUALIZATION_PLAN.md` | 3D visualization and reconstruction plan |

## Does Not Own

- terrain feature formulas after extraction: use `../feature_engineering/`;
- model use of features: use `../model_design/`;
- paper contribution claims: use `../research_framing/`.

See `../DESIGN_DOC_STRUCTURE_2026-07-02.md` and
`../DOC_OBJECT_MODEL_2026-07-02.md` for routing rules.
