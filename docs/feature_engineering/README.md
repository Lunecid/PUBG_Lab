# Feature Engineering Docs

Package class: `FeatureContract`

## Responsibility

Own feature taxonomies, mathematical definitions, feature clocks, extraction
contracts, proxy definitions, and feature-lineage notes.

## Public Interface

| Object | Role |
| --- | --- |
| `BATTLE_ROYALE_FEATURE_ONTOLOGY_2026-07-02.md` | genre-level Battle Royale feature ontology; O2 is contraction / elimination pressure |
| `PUBG_FEATURE_OPERATIONAL_MANIFEST_2026-07-02.md` | PUBG-operational feature contract and ablation ladder |
| `PUBG_NETWORK_STRUCTURE_FEATURE_SPEC_2026-07-02.md` | PUBG network-structure feature design for O7 relation features |
| `SPATIAL_TACTICAL_AFFORDANCE_FEATURE_SPEC_2026-07-02.md` | spatial-tactical affordance feature spec for PUBG/Caldera |
| `TERRAIN_EXPERIENCE_FEATURE_DESIGN_2026-07-02.md` | current terrain-affordance feature contract |
| `TERRAIN_EXPERIENCE_RESEARCH_LINEAGE_2026-07-02.md` | terrain-affordance literature lineage |
| `ST_FEATURE_DEFINITION_2026-06-23.md` | canonical ST feature definition |
| `ST_FEATURE_PRIOR_ALIGNED_DESIGN_2026-06-23.md` | prior-aligned ST framing |
| `FEATURE_CLOCK_CHANNEL_REDESIGN_2026-06-22.md` | feature clock/channel redesign |
| `SPATIAL_EFFECT_FEATURE_DESIGN.md` | earlier spatial effect design |
| `GENRE_PRIMITIVE_FEATURE_MANIFEST.csv` | feature manifest |

## Does Not Own

- model message passing or heads: use `../model_design/`;
- DB schema ownership: use `../data_layer/`;
- paper claims and limitations: use `../research_framing/`.

See `../DESIGN_DOC_STRUCTURE_2026-07-02.md` and
`../DOC_OBJECT_MODEL_2026-07-02.md` for routing rules.
