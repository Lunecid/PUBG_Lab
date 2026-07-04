# Data Layer Docs

Package class: `DataContract`

## Responsibility

Own schema contracts, ground/derived layer boundaries, ETL ownership, cohort
definitions, and data availability.

## Public Interface

| Object | Role |
| --- | --- |
| `BATTLE_ROYALE_COHORT_REGISTRY_2026-07-03.md` | locked map / perspective / team-size cohort registry |
| `BATTLE_ROYALE_PRIMARY_COHORT_DECISION_2026-07-03.md` | supporting group-level TPP squad cohort note |
| `BR_DATA_LAYER_CONTRACT.md` | canonical ground/representation layer contract |
| `RQ1_STDB_REDESIGN.md` | STDB redesign notes |
| `RESEARCH_COHORT_FLAT.md` | cohort definition and exclusions |
| `DATA_AVAILABILITY_BY_DOMAIN.md` | domain data availability |
| `EXPERIMENT_DATASETS_BY_DOMAIN.md` | dataset inventory by domain |

## Does Not Own

- feature formulas: use `../feature_engineering/`;
- model architecture: use `../model_design/`;
- run matrices and metrics: use `../experiment_design/`.

See `../DESIGN_DOC_STRUCTURE_2026-07-02.md` and
`../DOC_OBJECT_MODEL_2026-07-02.md` for routing rules.
