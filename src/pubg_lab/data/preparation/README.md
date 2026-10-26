# pubg_lab.data.preparation

Responsibility: STDB access, quality rules and data preparation.

Dependency boundary: Shared infrastructure and scientific source contracts.

| Component | Responsibility |
|---|---|
| `build_data_availability_domains.py` | Build availability tables by perspective, map, and team size. |
| `build_experiment_domain_datasets.py` | Create experiment-ready domain datasets from the raw PUBG population. |
| `build_research_cohort_flat.py` | Build a flat, non-destructive research cohort manifest. |
| `convert_raw_to_usable.py` | Convert raw PUBG matches into experiment-usable STDB/phase/graph data. |
| `import_research_cohort_to_br_exp.py` | Import the existing paper cohort manifest into br_exp. |
| `register_pubg_l1_ground.py` | Register current pubg.* as the preserved PUBG L1 ground layer. |
| `update_data_availability_summary.py` | Update map/mode data availability summary for phase-level GNN runs. |

The implementation lives here. Old root paths are compatibility entry points.
Import reusable components by their canonical `pubg_lab` path; invoke applications through `python -m pubg_lab run <command>`.
Listing commands does not import applications, access data, or start services.
Scientific contracts and completed evidence remain under the repository docs and original artifact paths.
