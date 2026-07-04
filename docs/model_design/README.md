# Model Design Docs

Package class: `ModelContract`

## Responsibility

Own model architecture, graph edge semantics, message passing, heads, losses,
input channels, and model-facing ablation design.

## Public Interface

| Object | Role |
| --- | --- |
| `BATTLE_ROYALE_SURVIVAL_FORMALIZATION_2026-07-03.md` | model-facing survival formula for the locked ontology |
| `INPUT_MODEL_CHANNEL_CONTRACT_2026-06-22.md` | model input channel contract |
| `MODEL_SPEC_CODE_GROUNDED_2026-06-21.md` | code-grounded model spec |
| `BATTLE_ROYALE_GRAPH_FORMALISM_2026-06-22.md` | graph formalism |
| `GRAPH_DATA_MODELING_FOUNDATION_2026-06-22.md` | graph/data modeling foundation |
| `PUBG_GRAPH_MODELING_RESEARCH_FRAME_2026-06-22.md` | graph research frame |
| `FUTURE_DESIGN_3D_RELATIONAL_2026-06-21.md` | 3D relational future design |

## Does Not Own

- feature extraction math: use `../feature_engineering/`;
- experiment results: use `../experiment_design/`;
- claim boundaries: use `../research_framing/`.

See `../DESIGN_DOC_STRUCTURE_2026-07-02.md` and
`../DOC_OBJECT_MODEL_2026-07-02.md` for routing rules.
